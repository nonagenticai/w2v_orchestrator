"""audiobook_segment — split each chapter into speaker-attributed spans.

Runs between ``audiobook_plan`` and ``audiobook_cast`` and powers SUB-CHUNK
voice acting: instead of one voice per ~2000-char chunk, narration and each
character's dialogue become separate single-speaker units, so voices switch
mid-scene. Two phases keep LLM cost down:

* **mechanical** (``parsing.split_spans``): double-quoted runs → dialogue spans,
  the rest → narration. Deterministic, zero cost. Narration is always the
  ``Narrator`` — never sent to the model.
* **attribution** (WM worker ``audiobook-segment``): per chapter, the model
  labels each DIALOGUE span with a speaker, reusing a book-global character
  roster carried across chapters so a character introduced in chapter 1 keeps
  the same name (and therefore the same cast voice) in chapter 40. Any span the
  model leaves unattributed falls back to ``Narrator`` — a mis-attribution is at
  worst read in the narrator voice, never a crash.

Output is PLAN-SHAPED (``chapters[].chunks[]`` with an extra ``speaker``) so
``audiobook_cast``/``narrate``/``stitch`` consume it with only source-selection.
Opt-in: ``brief.acting_granularity == "span"``. Otherwise the stage is a no-op
pass-through (empty output) and the pipeline behaves exactly as before.
"""

from __future__ import annotations

import logging
from itertools import batched
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import uuid7

from w2v_common.stages import stage_call

from src.services.llm import WMClient
from src.services.stages.audiobook.parsing import (
    DEFAULT_MAX_CHUNK_CHARS,
    split_long_span,
    split_spans,
)
from src.services.stages.base import (
    SlotUse,
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
    StagePreconditionError,
)
from src.services.stages.llm_budget import (
    LLM_RETRY_ON,
    MEDIUM_ANSWER_TOKENS,
    downshift_reasoning,
    llm_budget_seconds,
    llm_client_timeout_seconds,
    llm_retries,
    output_token_budget,
)
from src.services.stages.upstream import guarded

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

logger = logging.getLogger(__name__)

_WORKER_NAME = "audiobook-segment"
_NARRATOR_SPEAKER = "Narrator"
# Bound the attribution WM call in windows of raw spans, mirroring cast.py /
# direct.py (_PERFORMANCE_BATCH_SIZE). A single unbatched call over a whole
# single-chapter book (thousands of spans) blows the WM prompt/output budget →
# the #718-class timeout → soft-degrade to ALL Narrator, silently defeating
# span-mode voice acting. A window keeps local narration↔dialogue context so
# attribution stays accurate. Each batch soft-degrades independently.
_SPAN_ATTRIBUTION_BATCH = 40

# ``_OUTPUT_SCHEMA`` answers ONE window: at most ``_SPAN_ATTRIBUTION_BATCH`` (40)
# objects of {index: int, speaker: str} and nothing else — no prose, no text
# echo. 40 x ~20 tokens is under a thousand, so the batching already makes this
# MEDIUM rather than a book-length LARGE list.
_SEGMENT_MAX_TOKENS = output_token_budget(answer_tokens=MEDIUM_ANSWER_TOKENS)
SPAN_GRANULARITY = "span"

_SYSTEM_PROMPT = (
    "You attribute audiobook dialogue to speakers. You receive a chapter's "
    "ordered spans (each marked NARRATION or DIALOGUE) and the list of "
    "characters already known in this book. For every DIALOGUE span, name the "
    "character who says it — REUSE an existing character name whenever it is the "
    "same person (use the surrounding narration, e.g. 'said Anna', for cues); "
    "only introduce a new name for a genuinely new speaker. Use short, "
    "consistent names (e.g. 'Anna', not 'Anna the manager'). Do NOT attribute "
    "NARRATION spans. Return one attribution per DIALOGUE span index."
)

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "attributions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "speaker": {"type": "string"},
                },
                "required": ["index", "speaker"],
            },
        }
    },
    "required": ["attributions"],
}


class AudiobookSegmentInput(StageInput):
    pass


class SegmentUnit(StageOutput):
    id: str
    text: str
    sequence: int
    speaker: str


class SegmentedChapter(StageOutput):
    id: str
    title: str
    sequence: int
    # ``chunks`` (not ``spans``) so narrate/stitch read it exactly like a plan.
    chunks: list[SegmentUnit]


class AudiobookSegmentOutput(StageOutputWithNotes):
    """Span-mode segmentation.

    Carries ``notes`` because attribution degrades SILENTLY: every batch soft-fails
    to ``{}`` and the whole book then reads in the narrator voice while the stage
    still reports ``completed``. ``notes`` is the run-level degrade channel
    (``api/routes/runs/_shared.py`` turns a non-empty list into a ``degraded`` run),
    so a total WM outage — a 402 "configure your LLM provider", a schema rejection —
    can no longer ship a single-voice audiobook as a clean success.

    No ``produces`` declaration: the off/chunk path returns an EMPTY output on
    purpose, and ``_enforce_produces`` cannot see that it was a legitimate no-op —
    declaring ``chapters`` would degrade every non-span run. The stage raises its
    own note instead, where the granularity is known.
    """

    chapters: list[SegmentedChapter]
    roster: list[str]
    total_spans: int


def coalesce_units(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge adjacent same-speaker spans into one unit (anti-choppiness).

    Consecutive spans by the same speaker (e.g. a character's dialogue split by
    the quote regex, or two narration runs) join into a single clip. A short
    span between DIFFERENT speakers is kept — a real one-word interjection or a
    narrator "he said" beat is legitimate voice-acting, and merging it into
    another speaker would mis-voice it.
    """
    merged: list[dict[str, Any]] = []
    for unit in units:
        if merged and merged[-1]["speaker"] == unit["speaker"]:
            merged[-1]["text"] = f"{merged[-1]['text']} {unit['text']}".strip()
        else:
            merged.append(dict(unit))
    return merged


class AudiobookSegmentStage(StageHandler[AudiobookSegmentInput, AudiobookSegmentOutput]):
    name = "audiobook_segment"
    input_model = AudiobookSegmentInput
    output_model = AudiobookSegmentOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("llm"),)

    def __init__(self, client: WMClient | None = None) -> None:
        self._llm_timeout = llm_budget_seconds()
        self._llm_retries = llm_retries()
        self._wm = client or WMClient(timeout=llm_client_timeout_seconds())
        # Attribution-degrade accounting for the notes channel. Reset per handle();
        # initialised here so `_attribute` is safe to call on a fresh instance.
        self._batches = 0
        self._batches_degraded = 0

    async def handle(
        self,
        payload: AudiobookSegmentInput,
        ctx: RunContext,
    ) -> AudiobookSegmentOutput:
        brief = ctx.config_snapshot.get("brief") or {}
        granularity = str(brief.get("acting_granularity") or "").lower()
        # No-op unless span mode — keeps the DAG shape stable while off/chunk
        # paths pay nothing (no LLM, empty output; narrate/stitch use the plan).
        if granularity != SPAN_GRANULARITY:
            return AudiobookSegmentOutput(chapters=[], roster=[], total_spans=0)

        ingest = ctx.prior_output("audiobook_ingest")
        if not ingest or not ingest.get("chapters"):
            raise StagePreconditionError("audiobook_segment: audiobook_ingest output missing")

        run_id = str(ctx.run_id)
        self._batches = 0
        self._batches_degraded = 0
        dialogue_spans = 0
        roster: list[str] = []  # book-global, accumulated in chapter order
        roster_seen: set[str] = set()
        chapters: list[SegmentedChapter] = []
        total_spans = 0

        for chapter in sorted(ingest["chapters"], key=lambda c: int(c.get("sequence") or 0)):
            raw_spans = split_spans(str(chapter.get("text") or ""))
            dialogue_spans += sum(1 for span in raw_spans if span.is_dialogue)
            speakers = await self._attribute(raw_spans, roster)

            units: list[dict[str, Any]] = []
            for span, speaker in zip(raw_spans, speakers, strict=True):
                units.append({"text": span.text, "speaker": speaker})
                if speaker != _NARRATOR_SPEAKER and speaker not in roster_seen:
                    roster_seen.add(speaker)
                    roster.append(speaker)

            units = coalesce_units(units)

            # Split any over-cap single-speaker unit (a long monologue) so no TTS
            # request exceeds the backend cap; each piece keeps the same voice.
            seq = 0
            seg_units: list[SegmentUnit] = []
            for unit in units:
                pieces = split_long_span(unit["text"], DEFAULT_MAX_CHUNK_CHARS)
                for piece in pieces:
                    seg_units.append(
                        SegmentUnit(
                            id=str(uuid7()), text=piece, sequence=seq, speaker=unit["speaker"]
                        )
                    )
                    seq += 1

            total_spans += len(seg_units)
            logger.info(
                "audiobook_segment run=%s chapter seq=%d units=%d speakers=%d",
                run_id,
                int(chapter.get("sequence") or 0),
                len(seg_units),
                len({u.speaker for u in seg_units}),
            )
            chapters.append(
                SegmentedChapter(
                    id=str(chapter["id"]),
                    title=str(chapter.get("title") or "Chapter"),
                    sequence=int(chapter.get("sequence") or 0),
                    chunks=seg_units,
                )
            )

        notes = self._degrade_notes(dialogue_spans=dialogue_spans, roster=roster)
        logger.info(
            "audiobook_segment run=%s chapters=%d total_spans=%d roster=%d %r notes=%d",
            run_id,
            len(chapters),
            total_spans,
            len(roster),
            roster[:20],
            len(notes),
        )
        return AudiobookSegmentOutput(
            chapters=chapters, roster=roster, total_spans=total_spans, notes=notes
        )

    def _degrade_notes(self, *, dialogue_spans: int, roster: list[str]) -> list[str]:
        """Say out loud that voice acting silently did not happen.

        Both conditions are DEGRADE, never failure: a single-voice reading is a
        usable audiobook, and the off/chunk path produces exactly that on purpose.
        What is not acceptable is delivering it while the run reads ``completed``.
        """
        notes: list[str] = []
        if self._batches_degraded:
            notes.append(
                f"audiobook_segment: {self._batches_degraded} of {self._batches} speaker-"
                f"attribution batches failed — those spans stay Narrator (single voice)"
            )
        if dialogue_spans and not roster:
            notes.append(
                f"audiobook_segment: {dialogue_spans} dialogue span(s) but NO speaker was "
                f"attributed — the ENTIRE audiobook will be read in the narrator voice; "
                f"span-mode voice acting did not happen"
            )
        return notes

    async def _attribute(self, raw_spans: list[Any], roster: list[str]) -> list[str]:
        """Speaker per span; narration → Narrator, dialogue → LLM (fallback Narrator).

        Returns a speaker list parallel to ``raw_spans``. If there is no dialogue,
        the model is never called. On any model failure every dialogue span falls
        back to ``Narrator`` (single-voice) rather than failing the stage.
        """
        speakers = [_NARRATOR_SPEAKER] * len(raw_spans)
        dialogue_idx = [i for i, s in enumerate(raw_spans) if s.is_dialogue]
        if not dialogue_idx:
            return speakers

        # Bound the WM call in windows of spans (mirrors cast/direct batching) so a
        # long/single-chapter book never sends one unbounded prompt (#718-class
        # timeout → all-Narrator). Global indices are preserved so batch outputs
        # merge into one attribution map; each batch soft-degrades on its own.
        indexed = list(enumerate(raw_spans))
        attributed: dict[int, str] = {}
        for batch in batched(indexed, _SPAN_ATTRIBUTION_BATCH, strict=False):
            if any(span.is_dialogue for _, span in batch):  # no dialogue ⇒ no WM call
                self._batches += 1
                attributed.update(await self._attribute_batch(list(batch), roster))

        for i in dialogue_idx:
            spk = attributed.get(i)
            if spk:
                speakers[i] = spk
        return speakers

    async def _attribute_batch(
        self, batch: list[tuple[int, Any]], roster: list[str]
    ) -> dict[int, str]:
        """Attribute one window of spans (carrying GLOBAL indices) → {index: speaker}.

        SOFT: returns ``{}`` on any WM failure so the caller leaves that window as
        Narrator while the OTHER windows still attribute (partial-survives) — a
        batch failure never fails the stage or degrades the whole book."""
        message = self._build_message(batch, roster)

        async def _attempt(enable_reasoning: bool | None) -> dict[int, str]:
            resp = await self._wm.message(
                enable_reasoning=enable_reasoning,
                worker_name=_WORKER_NAME,
                message=message,
                system_prompt=_SYSTEM_PROMPT,
                output_schema=_OUTPUT_SCHEMA,
                max_tokens=_SEGMENT_MAX_TOKENS,
            )
            data = resp.output
            if not isinstance(data, dict) or not isinstance(data.get("attributions"), list):
                raise ValueError("audiobook_segment: WM output missing attributions")
            out: dict[int, str] = {}
            for entry in data["attributions"]:
                if not isinstance(entry, dict):
                    continue
                idx = entry.get("index")
                spk = str(entry.get("speaker") or "").strip()
                if isinstance(idx, int) and spk:
                    out[idx] = spk
            return out

        try:
            # guarded() collapses a permanent WM failure (402, token limit) to a
            # single call before degrading, rather than spending the full retry ladder.
            return await stage_call(
                downshift_reasoning(lambda think: guarded(lambda: _attempt(think))),
                timeout_seconds=self._llm_timeout,
                retries=self._llm_retries,
                retry_on=LLM_RETRY_ON,
                stage_name="audiobook_segment",
            )
        except Exception as exc:  # noqa: BLE001 -- a batch failure degrades that window to single-voice
            # Counted, not just logged: a log line is invisible to the run status.
            # When EVERY batch lands here (a 402, a dead worker) the book is
            # single-voice, and `handle` turns that into a note ⇒ a `degraded` run.
            self._batches_degraded += 1
            logger.warning(
                "audiobook_segment attribution batch failed, that window stays Narrator: %s", exc
            )
            return {}

    def _build_message(self, indexed_spans: list[tuple[int, Any]], roster: list[str]) -> str:
        known = ", ".join(roster) if roster else "(none yet)"
        lines = []
        for i, span in indexed_spans:
            kind = "DIALOGUE" if span.is_dialogue else "NARRATION"
            lines.append(f"[{i}] {kind}: {span.text[:300]}")
        return (
            f"Known characters so far: {known}\n\n"
            "SPANS (attribute every DIALOGUE index to a speaker):\n" + "\n".join(lines)
        )
