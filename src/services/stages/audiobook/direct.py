"""audiobook_direct — LLM performance director for audiobook narration (spec 4.1).

One :class:`PerformanceDirective` per narration CHUNK (keyed on ``chunk_id``, the
audiobook spine): how a chunk should be *delivered* — a delivery note, an optional
per-word emphasis set, a leading pause, and an optional kokoro rate. Structurally a
sibling of ``demo_video/voice_director.py`` (same ``StageHandler`` shape, the same
WM call wrapped in ``stage_call(downshift_reasoning(lambda think: guarded(lambda: _attempt(think))), …)``, the same
model-is-advisory / stage-is-authoritative validate+clamp loop) — but scoped to the
audiobook chunk spine, not the demo_video line spine.

Authoritative clamp (the model is advisory; this stage disposes):
* ``rate_hint`` is clamped to kokoro's ``[0.5, 2.0]`` speed window, or left ``None``
  when the model proposes nothing — ``None`` means "use the cast/voice speed",
  so an un-directed chunk narrates byte-identically to today.
* ``pause_before_ms`` is clamped to ``[0, 2000]`` ms.
* ``emphasis_word_indices`` are validated against ``text.split()`` positions
  (out-of-range dropped) so :func:`primitives.ssml.build_line_ssml` — which
  tokenizes the same way — lines them up. A hallucinated 400-token directive can
  never reach synth.

The freeform ``ssml`` string the LLM may emit is intentionally NOT threaded to synth
(only the structured fields are — narrate lowers them via ``build_line_ssml``). The
SSML path itself is gated downstream on ``NARRATION_SSML_ENABLED`` + ``tts_backend``
honoring real timed ``<break>``/emphasis (TTSB-3); until that flag is on, the
directive degrades to rate-only. No ``worker_management`` change is needed: WM
auto-creates the ``audiobook-director`` worker on first call; its model/provider is
WM-side config.

Skip-clean: with no ``brief.audiobook.direction`` the stage does NO LLM work and
raises :class:`StageSkippedError`, so a no-direction run is behaviour-preserving and
never blocks the narrate stage that ``depends_on`` it.
"""

from __future__ import annotations

import json
from itertools import batched
from typing import TYPE_CHECKING, ClassVar, cast

from pydantic import BaseModel
from w2v_common.stages import stage_call

from src.services.llm import WMClient
from src.services.stages.base import (
    SlotUse,
    StageHandler,
    StageInput,
    StageOutputWithNotes,
    StagePreconditionError,
    StageSkippedError,
)
from src.services.stages.batch_ledger import BatchLedger
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

# WM worker name (regex ^[a-zA-Z0-9_-]+$). WM auto-creates it on first call and the
# per-call system_prompt steers it; the provider/model lives on the WM worker config,
# NOT here — that's the whole point of routing through WM.
_WORKER_NAME = "audiobook-director"

# Per-WM-call batch size: cap the ``directives`` output array so the small vLLM
# doesn't time out on an unbounded per-chunk schema and silently return partial
# coverage (mirrors shared/footage_critique's ``_CRITIC_BATCH_SIZE``).
_DIRECTIVE_BATCH_SIZE = 5

# ``_OUTPUT_SCHEMA`` answers ONE batch: at most ``_DIRECTIVE_BATCH_SIZE`` (5)
# directive objects, each carrying a free-text ``delivery_note`` plus an
# ``emphasis_word_indices`` array — i.e. a handful of objects WITH prose fields,
# which is exactly the MEDIUM shape.
_DIRECT_MAX_TOKENS = output_token_budget(answer_tokens=MEDIUM_ANSWER_TOKENS)

# kokoro's supported speed window; any proposed rate is forced into it.
_RATE_MIN, _RATE_MAX = 0.5, 2.0
# Leading-pause bounds. The upper bound is a safe absolute ceiling (~a long beat);
# a finer per-boundary band lives in pacing.py (L3.1) but the chunk spine does not
# yet carry per-chunk boundary_kind, so a conservative absolute clamp is used here.
_PAUSE_MIN_MS, _PAUSE_MAX_MS = 0, 2000
# How much of each chunk to show the LLM — enough to judge delivery + emphasis word
# positions on the opening of a long chunk, not the whole 2000-char body.
_PROMPT_CHUNK_CHARS = 320

_SYSTEM_PROMPT = (
    "You are a performance director for an audiobook narrator. You receive the ordered "
    "narration chunks (each with a chunk_id and its opening text) and the director's "
    "intent. For EVERY chunk, return one directive describing how it should be delivered: "
    "a short delivery_note (tone/energy), rate_hint (0.5-2.0 kokoro speed, 1.0 = normal; "
    "omit unless the line genuinely wants a different pace), pause_before_ms (0-2000, a "
    "beat before the line lands), and emphasis_word_indices (0-based indices of words in "
    "that chunk's opening to stress). Keep it restrained and literary — most chunks are "
    "normal pace with no pause. Provide a directive for EVERY chunk_id given."
)

# Object-rooted structured-output schema (WM convention). WM returns
# ``response.output`` already parsed to this shape.
_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "directives": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "delivery_note": {"type": "string"},
                    "emphasis_word_indices": {"type": "array", "items": {"type": "integer"}},
                    "pause_before_ms": {"type": "integer"},
                    "rate_hint": {"type": "number"},
                },
                "required": ["chunk_id"],
            },
        }
    },
    "required": ["directives"],
}


def _clamp_rate(value: object) -> float | None:
    """Clamp a proposed rate to kokoro's window, or ``None`` when unusable.

    ``None`` (returned for a missing/garbage rate) means "no opinion" — narrate then
    keeps the cast/voice speed, so an un-directed chunk is byte-identical to today.
    """
    if not isinstance(value, (int, float, str)):
        return None
    try:
        rate = float(value)
    except TypeError, ValueError:
        return None
    return max(_RATE_MIN, min(_RATE_MAX, rate))


def _to_int(value: object, default: int) -> int:
    if not isinstance(value, (int, float, str)):
        return default
    try:
        return int(value)
    except TypeError, ValueError:
        return default


def _int_list(value: object) -> list[int]:
    """Coerce a raw list to ints, dropping anything non-integral."""
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        if not isinstance(item, (int, float, str)):
            continue
        try:
            out.append(int(item))
        except TypeError, ValueError:
            continue
    return out


class AudiobookDirectInput(StageInput):
    pass


class PerformanceDirective(BaseModel):
    chunk_id: str
    delivery_note: str = ""
    emphasis_word_indices: list[int] = []
    pause_before_ms: int = 0
    # None ⇒ no rate opinion ⇒ narrate keeps the cast/voice speed (byte-identical).
    rate_hint: float | None = None


class AudiobookDirectOutput(StageOutputWithNotes):
    # Declared output contract — enforced once, centrally, in
    # `runs/dag_executor.py::_enforce_produces`. Empty here means the stage
    # reported success having produced nothing, which degrades the run.
    # `audiobook_narrate` keys per-chunk SSML/rate off `directives`. The stage skips
    # clean when the brief carries no direction, and coverage-fill guarantees one
    # directive per chunk — so a COMPLETED direct with none is a lost LLM result.
    produces: ClassVar[tuple[str, ...]] = ("directives",)

    directives: list[PerformanceDirective]


def _chunks_from_source(source: dict) -> list[dict]:
    """Flatten the plan/segment output to the ordered chunk list narrate iterates.

    Mirrors ``audiobook_narrate``: chapters → chunks, each carrying ``id`` + ``text``.
    """
    chunks: list[dict] = []
    for chapter in source.get("chapters") or []:
        for chunk in chapter.get("chunks") or []:
            chunks.append(chunk)
    return chunks


class AudiobookDirectStage(StageHandler[AudiobookDirectInput, AudiobookDirectOutput]):
    """LLM per-chunk performance director. Advisory model, authoritative stage —
    every returned value is clamped/coerced and every plan chunk gets a directive
    (coverage-fill) before it can reach narration."""

    name = "audiobook_direct"
    input_model = AudiobookDirectInput
    output_model = AudiobookDirectOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("llm"),)
    worker_name: str = _WORKER_NAME

    def __init__(self, client: WMClient | None = None) -> None:
        self._llm_timeout = llm_budget_seconds()
        self._llm_retries = llm_retries()
        self._wm = client or WMClient(timeout=llm_client_timeout_seconds())

    async def handle(
        self,
        payload: AudiobookDirectInput,
        ctx: RunContext,
    ) -> AudiobookDirectOutput:
        brief = ctx.config_snapshot.get("brief") or {}
        # Coerce defensively: a malformed brief may carry `audiobook` as a truthy
        # non-dict, and `.get(...)` below would AttributeError out of this stage.
        audiobook_cfg = ab if isinstance(ab := brief.get("audiobook"), dict) else {}
        direction = audiobook_cfg.get("direction")
        # Skip-clean: no director intent → no LLM work, stage marked skipped so a
        # no-direction run stays behaviour-preserving and does not block narrate.
        if not direction:
            raise StageSkippedError(
                f"{self.name}: no brief.audiobook.direction (opt-in stage)",
                missing=["brief.audiobook.direction"],
            )

        # Same source precedence narrate uses: span-mode segment, else the plan.
        source = ctx.prior_output("audiobook_segment")
        if not source or not source.get("chapters"):
            source = ctx.prior_output("audiobook_plan")
        if not source or not source.get("chapters"):
            raise StagePreconditionError(f"{self.name}: no plan/segment output")
        chunks = _chunks_from_source(source)
        if not chunks:
            raise StagePreconditionError(f"{self.name}: plan/segment has no chunks")

        # Bounded batches (≤ _DIRECTIVE_BATCH_SIZE chunks per WM call): an
        # unbounded ``directives`` output schema otherwise times out the small
        # vLLM and the stage silently degrades to neutral-default coverage.
        # Per-chunk directives are independent, so the flat merge is exact and
        # coverage-fill runs once. Retry/timeout/soft-fail semantics unchanged
        # (the whole loop stays wrapped by the stage_call ladder below).
        # The ledger and the batch list live OUTSIDE the attempt factory so a retry
        # RESUMES. When the accumulator lived inside, attempt 1 finished k of N batches,
        # timed out, and attempt 2 re-paid those k WM calls to reach the same place — a
        # stage whose N batches do not fit inside ONE attempt could never complete, no
        # matter how many retries it was given. See ``stages/batch_ledger``.
        batches = [list(batch) for batch in batched(chunks, _DIRECTIVE_BATCH_SIZE, strict=False)]
        ledger: BatchLedger[list[dict[str, str]], list[object]] = BatchLedger()

        async def _attempt(enable_reasoning: bool | None) -> AudiobookDirectOutput:
            async def _direct(batch: list[dict[str, str]]) -> list[object]:
                message = self._build_message(batch, direction)
                resp = await self._wm.message(
                    enable_reasoning=enable_reasoning,
                    worker_name=self.worker_name,
                    message=message,
                    system_prompt=_SYSTEM_PROMPT,
                    output_schema=_OUTPUT_SCHEMA,
                    max_tokens=_DIRECT_MAX_TOKENS,
                )
                data = resp.output
                if not isinstance(data, dict) or not isinstance(data.get("directives"), list):
                    # ValueError is in LLM_RETRY_ON → re-prompt for a conforming object.
                    raise ValueError(f"{self.name}: WM output missing directives")
                return list(data["directives"])

            parts = await ledger.run(batches, _direct)
            raw_directives = [item for part in parts for item in part]
            return AudiobookDirectOutput(directives=self._assemble(chunks, raw_directives))

        return await stage_call(
            downshift_reasoning(lambda think: guarded(lambda: _attempt(think))),
            timeout_seconds=self._llm_timeout,
            retries=self._llm_retries,
            retry_on=LLM_RETRY_ON,
            stage_name=self.name,
        )

    def _assemble(
        self,
        chunks: list[dict],
        raw_directives: list[object],
    ) -> list[PerformanceDirective]:
        """One directive per PLAN chunk (full coverage), matched by chunk_id. Every
        field is clamped/coerced; a chunk the model omitted gets a neutral default
        (no rate opinion, no pause, no emphasis) — i.e. today's behaviour."""
        by_id: dict[str, dict[str, object]] = {}
        for entry in raw_directives:
            if not isinstance(entry, dict):
                continue
            directive = cast("dict[str, object]", entry)
            if directive.get("chunk_id"):
                by_id[str(directive["chunk_id"])] = directive

        out: list[PerformanceDirective] = []
        for chunk in chunks:  # iterate the PLAN — guarantees full coverage
            chunk_id = str(chunk.get("id") or "")
            entry = by_id.get(chunk_id)
            if entry is None:
                out.append(PerformanceDirective(chunk_id=chunk_id))
                continue
            text = str(chunk.get("text") or "")
            word_count = len(text.split())
            emphasis = [
                i for i in _int_list(entry.get("emphasis_word_indices")) if 0 <= i < word_count
            ]
            out.append(
                PerformanceDirective(
                    chunk_id=chunk_id,
                    delivery_note=str(entry.get("delivery_note") or ""),
                    emphasis_word_indices=emphasis,
                    pause_before_ms=max(
                        _PAUSE_MIN_MS, min(_PAUSE_MAX_MS, _to_int(entry.get("pause_before_ms"), 0))
                    ),
                    rate_hint=_clamp_rate(entry.get("rate_hint")),
                )
            )
        return out

    def _build_message(self, chunks: list[dict], direction: object) -> str:
        body = "\n".join(
            f'{{"chunk_id":"{chunk.get("id")}",'
            f'"text":{json.dumps(str(chunk.get("text") or "")[:_PROMPT_CHUNK_CHARS], ensure_ascii=False)}}}'
            for chunk in chunks
        )
        intent = (
            direction if isinstance(direction, str) else json.dumps(direction, ensure_ascii=False)
        )
        return (
            f"DIRECTOR INTENT:\n{intent}\n\nNARRATION CHUNKS (one JSON object per chunk):\n{body}"
        )
