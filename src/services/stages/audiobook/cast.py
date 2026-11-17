"""audiobook_cast — LLM voice casting for multi-voice audiobooks.

Runs between ``audiobook_plan`` and ``audiobook_narrate``. An LLM (via
``worker_management`` — the ``WMClient`` gateway, NEVER inference-router / vLLM
directly, so the model behind the worker can change without touching this code)
attributes each planned chunk to a speaker (in BOUNDED batches — see below), then
casts ONE consistent kokoro voice per character from the 54 presets in a single
global call. ``narrate`` then synthesizes each chunk in its cast voice instead of
a single project-wide voice.

Two-phase, split for a reason (issue #718): a single call that returned BOTH the
global casting AND the per-chunk ``performances`` array grew unbounded with
narration length, timing out the small production vLLM — which then silently
degraded every chunk to coverage-fill Narrator. So:
  1. Attribution runs in bounded batches of ``_PERFORMANCE_BATCH_SIZE`` chunks
     (the per-batch ``performances`` array can never blow the timeout); results
     flat-merge exactly because per-chunk attribution is independent.
  2. Casting is ONE global call over the full distinct-speaker roster (reusing the
     span-mode ``_cast_roster``/``_llm_cast_roster`` path). A naive per-batch cast
     would let batch A pick voice2 for a character and batch B pick voice5 for the
     same one — colliding/inconsistent voices. Casting once makes that impossible.

Hard invariants enforced here (the model is advisory, this stage is
authoritative):
  * every emitted ``voice_id`` is validated against ``VOICE_CATALOG``; an
    invalid/hallucinated id is repaired to a valid distinct catalog voice (the
    deterministic ``_distinct_voice`` fallback) — it can never reach
    ``/api/tts/synthesize`` (which 500s on an unknown voice).
  * the ``Narrator`` speaker always keeps the project narrator voice — the model
    attributes narration but never re-casts the narrator (this is what the shipped
    span-mode ``_cast_roster`` already guarantees; the two paths now agree).
  * every plan chunk gets a performance row (coverage-fill), so narrate never
    encounters a chunk the LLM forgot.
  * voice is resolved from the book-level casting map keyed by speaker, so the
    same character keeps one voice across the whole book regardless of any
    per-chunk drift the model introduces.

Opt-in: ``brief.voice_acting`` truthy. When off, the stage does NO LLM work and
returns empty performances, so narrate falls back to today's single-voice
behaviour byte-for-byte. ``brief.voice_cast`` (speaker→voice_id) pins specific
characters; the LLM fills the rest.
"""

from __future__ import annotations

import json
from itertools import batched
from typing import TYPE_CHECKING, Any, ClassVar, cast

from w2v_common.stages import stage_call

from src.services.llm import WMClient
from src.services.stages.audiobook.voice_catalog import (
    VOICE_CATALOG,
    clamp_speed,
    voice_catalog_prompt_block,
)
from src.services.stages.base import (
    SlotUse,
    StageHandler,
    StageInput,
    StageOutput,
    StagePreconditionError,
)
from src.services.stages.batch_ledger import BatchLedger
from src.services.stages.llm_budget import (
    LLM_RETRY_ON,
    MEDIUM_ANSWER_TOKENS,
    SMALL_ANSWER_TOKENS,
    downshift_reasoning,
    llm_budget_seconds,
    llm_client_timeout_seconds,
    llm_retries,
    output_token_budget,
)
from src.services.stages.upstream import guarded

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

# WM worker name (regex ^[a-zA-Z0-9_-]+$). WM auto-creates it on first call and
# the per-call system_prompt steers it; the provider/model lives on the WM
# worker config, NOT here — that's the whole point of routing through WM.
_WORKER_NAME = "audiobook-casting"
# Concrete fallback narrator voice when the project voice is unset/invalid.
_DEFAULT_NARRATOR_VOICE = "af_heart"
_NARRATOR_SPEAKER = "Narrator"
# How much of each chunk to show the LLM for speaker attribution — dialogue
# attribution needs the line, not the whole 2000-char chunk. Bounds token cost.
_PROMPT_CHUNK_CHARS = 400
_SPAN_GRANULARITY = "span"

# Casting-only schema for span mode: attribution already happened in
# ``audiobook_segment``, so the model only assigns ONE voice per known character.
_CASTING_ONLY_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "casting": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string"},
                    "voice_id": {"type": "string"},
                },
                "required": ["speaker", "voice_id"],
            },
        }
    },
    "required": ["casting"],
}
_CASTING_ONLY_SYSTEM_PROMPT = (
    "You cast audiobook voices. You receive a list of character names and the "
    "allowed voice ids. Assign ONE voice_id (from the allowed list only) to each "
    "character, matching gender/language where the name suggests it, and keeping "
    "distinct main characters on distinct voices. Return one casting entry per "
    "character."
)

# Attribution batch size — bound the per-call ``performances`` array so the small
# production vLLM never sees an unbounded prompt (the timeout → silent
# coverage-fill degrade that #718 closes). Mirrors the batched director stages.
_PERFORMANCE_BATCH_SIZE = 5

# ``_PERFORMANCE_ONLY_SCHEMA`` answers ONE batch: at most
# ``_PERFORMANCE_BATCH_SIZE`` (5) objects of {chunk_id, speaker, speed} — three
# scalar fields, no prose. ~50 tokens each, so the answer is a few hundred
# tokens and SMALL is the honest tier; the budget below is reasoning headroom.
_PERFORMANCE_MAX_TOKENS = output_token_budget(answer_tokens=SMALL_ANSWER_TOKENS)

# ``_CASTING_ONLY_SCHEMA`` is ONE global call answering {speaker, voice_id} per
# character in the roster. Entries are tiny (two short strings) but the roster
# length is not batched — a large cast is a modest list, not a verdict — so this
# is MEDIUM rather than SMALL.
_CASTING_MAX_TOKENS = output_token_budget(answer_tokens=MEDIUM_ANSWER_TOKENS)

_PERFORMANCE_ONLY_SYSTEM_PROMPT = (
    "You attribute audiobook narration. You receive ordered chunks of a "
    "manuscript. For EACH chunk decide who speaks it — plain narration vs. a "
    f"named character — using the label '{_NARRATOR_SPEAKER}' for plain "
    "narration. Spell each character's name identically everywhere so the same "
    "person is not split into two. Do NOT assign voices; only attribute the "
    "speaker and a speaking speed (0.5-2.0, 1.0 = normal). Provide one "
    "performance for EVERY chunk_id given."
)

# Per-call attribution schema: the (bounded, per-batch) ``performances`` array
# ONLY. Voice casting is a SEPARATE single global call (``_CASTING_ONLY_SCHEMA``),
# so distinct characters can never be cast to colliding voices across batches.
_PERFORMANCE_ONLY_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "performances": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "speaker": {"type": "string"},
                    "speed": {"type": "number"},
                },
                "required": ["chunk_id", "speaker"],
            },
        },
    },
    "required": ["performances"],
}


# Stable catalog ordering for the deterministic fallback assigner.
_CATALOG_ORDERED: list[str] = sorted(VOICE_CATALOG)


def _distinct_voice(speaker: str, used: set[str]) -> str:
    """A catalog voice not already in ``used`` (distinct main characters).

    When every voice is taken (a book with >54 speakers), fall back to a stable
    per-name pick so the same character still always maps to the same voice.
    """
    for vid in _CATALOG_ORDERED:
        if vid not in used:
            return vid
    idx = int.from_bytes(speaker.encode("utf-8"), "big") % len(_CATALOG_ORDERED)
    return _CATALOG_ORDERED[idx]


class AudiobookCastInput(StageInput):
    pass


class CharacterCast(StageOutput):
    speaker: str
    voice_id: str


class ChunkPerformance(StageOutput):
    chunk_id: str
    chapter_id: str
    speaker: str
    voice_id: str
    speed: float


class AudiobookCastOutput(StageOutput):
    casting: list[CharacterCast]
    performances: list[ChunkPerformance]
    narrator_voice_id: str


# Shared with audiobook_narrate (which must not import this LLM-calling module).
_clamp_speed = clamp_speed


def _flatten_chunks(plan: dict[str, Any]) -> list[dict[str, str]]:
    """Ordered (chapter_id, chunk_id, text) across all chapters."""
    out: list[dict[str, str]] = []
    for chapter in plan.get("chapters") or []:
        chapter_id = str(chapter["id"])
        for chunk in chapter.get("chunks") or []:
            out.append(
                {
                    "chapter_id": chapter_id,
                    "chunk_id": str(chunk["id"]),
                    "text": str(chunk.get("text") or ""),
                }
            )
    return out


class AudiobookCastStage(StageHandler[AudiobookCastInput, AudiobookCastOutput]):
    name = "audiobook_cast"
    input_model = AudiobookCastInput
    output_model = AudiobookCastOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("llm"),)

    def __init__(self, client: WMClient | None = None) -> None:
        self._llm_timeout = llm_budget_seconds()
        self._llm_retries = llm_retries()
        self._wm = client or WMClient(timeout=llm_client_timeout_seconds())

    async def handle(
        self,
        payload: AudiobookCastInput,
        ctx: RunContext,
    ) -> AudiobookCastOutput:
        plan = ctx.prior_output("audiobook_plan")
        if not plan or not plan.get("chapters"):
            raise StagePreconditionError("audiobook_cast: audiobook_plan output missing")

        cfg = ctx.config_snapshot
        voice_cfg = cfg.get("voice") or {}
        project_voice = str(voice_cfg.get("voice_id") or "")
        narrator_voice = (
            project_voice if project_voice in VOICE_CATALOG else _DEFAULT_NARRATOR_VOICE
        )
        default_speed = _clamp_speed(voice_cfg.get("speed"), 1.0)

        brief = cfg.get("brief") or {}

        # User-pinned characters (validated); the LLM fills/overrides the rest.
        user_cast: dict[str, str] = {}
        # Coerce defensively: a malformed brief may carry `voice_cast` as a truthy
        # non-dict (e.g. a list), and `.items()` would AttributeError out of cast.
        voice_cast = vc if isinstance(vc := brief.get("voice_cast"), dict) else {}
        for speaker, vid in voice_cast.items():
            if str(vid) in VOICE_CATALOG:
                user_cast[str(speaker)] = str(vid)

        # Span mode: attribution already happened in audiobook_segment; here we
        # only assign one voice per character and emit a performance per span.
        if str(brief.get("acting_granularity") or "").lower() == _SPAN_GRANULARITY:
            return await self._handle_span(ctx, narrator_voice, default_speed, user_cast)

        chunks = _flatten_chunks(plan)

        # Opt-out path: no voice acting → no LLM cost, narrate uses project voice.
        if not brief.get("voice_acting"):
            return AudiobookCastOutput(
                casting=[], performances=[], narrator_voice_id=narrator_voice
            )

        # Two-phase (issue #718): (1) attribute a speaker to each chunk in BOUNDED
        # batches so the unbounded ``performances`` array can never time out the
        # small vLLM; (2) cast ONE voice per distinct speaker in a single global
        # call, so a character can never be cast to two colliding voices across
        # batches. ``_assemble`` then resolves each chunk's voice from that map.
        raw_perf = await self._attribute_performances(chunks)
        speakers = self._distinct_speakers(raw_perf)
        cast_map = await self._cast_roster(speakers, narrator_voice, user_cast)
        return self._assemble(
            chunks=chunks,
            cast_map=cast_map,
            raw_perf=raw_perf,
            default_speed=default_speed,
        )

    @staticmethod
    def _distinct_speakers(raw_perf: list[object]) -> list[str]:
        """Sorted distinct non-narrator speakers the attribution pass named — the
        roster the single global casting call then covers."""
        seen: set[str] = set()
        for entry in raw_perf:
            if not isinstance(entry, dict):
                continue
            entry_d = cast("dict[str, object]", entry)
            speaker = str(entry_d.get("speaker") or "").strip()
            if speaker and speaker != _NARRATOR_SPEAKER:
                seen.add(speaker)
        return sorted(seen)

    async def _attribute_performances(self, chunks: list[dict[str, str]]) -> list[object]:
        """Per-chunk speaker+speed attribution in bounded batches (#718). The whole
        batched loop stays wrapped by the ``stage_call`` retry/timeout ladder, so
        retry/soft-fail semantics are unchanged from the old single call; only the
        prompt size per WM call is now bounded."""

        # The ledger and the batch list live OUTSIDE the attempt factory so a retry
        # RESUMES. When the accumulator lived inside, attempt 1 finished k of N batches,
        # timed out, and attempt 2 re-paid those k WM calls to reach the same place — a
        # stage whose N batches do not fit inside ONE attempt could never complete. See
        # ``stages/batch_ledger``.
        batches = [list(batch) for batch in batched(chunks, _PERFORMANCE_BATCH_SIZE, strict=False)]
        ledger: BatchLedger[list[dict[str, str]], list[object]] = BatchLedger()

        async def _attempt(enable_reasoning: bool | None) -> list[object]:
            async def _attribute(batch: list[dict[str, str]]) -> list[object]:
                resp = await self._wm.message(
                    enable_reasoning=enable_reasoning,
                    worker_name=_WORKER_NAME,
                    message=self._build_message(batch),
                    system_prompt=_PERFORMANCE_ONLY_SYSTEM_PROMPT,
                    output_schema=_PERFORMANCE_ONLY_SCHEMA,
                    max_tokens=_PERFORMANCE_MAX_TOKENS,
                )
                data = resp.output
                if not isinstance(data, dict) or not isinstance(data.get("performances"), list):
                    # ValueError is in LLM_RETRY_ON, so this re-prompts the worker for
                    # a conforming object — unlike a permanent upstream failure, which
                    # guarded() below fails immediately.
                    raise ValueError("audiobook_cast: WM output missing performances")
                return list(data["performances"])

            parts = await ledger.run(batches, _attribute)
            return [item for part in parts for item in part]

        return await stage_call(
            downshift_reasoning(lambda think: guarded(lambda: _attempt(think))),
            timeout_seconds=self._llm_timeout,
            retries=self._llm_retries,
            retry_on=LLM_RETRY_ON,
            stage_name="audiobook_cast",
        )

    async def _handle_span(
        self,
        ctx: RunContext,
        narrator_voice: str,
        default_speed: float,
        user_cast: dict[str, str],
    ) -> AudiobookCastOutput:
        """Span mode: cast one voice per roster character; performance per span."""
        segment = ctx.prior_output("audiobook_segment")
        if not segment or not segment.get("chapters"):
            # span requested but nothing segmented → single-narrator fallback.
            return AudiobookCastOutput(
                casting=[], performances=[], narrator_voice_id=narrator_voice
            )

        roster = [str(s) for s in (segment.get("roster") or []) if str(s).strip()]
        cast_map = await self._cast_roster(roster, narrator_voice, user_cast)

        performances: list[ChunkPerformance] = []
        for chapter in segment["chapters"]:
            chapter_id = str(chapter["id"])
            for unit in chapter.get("chunks") or []:
                speaker = str(unit.get("speaker") or _NARRATOR_SPEAKER).strip() or _NARRATOR_SPEAKER
                performances.append(
                    ChunkPerformance(
                        chunk_id=str(unit["id"]),
                        chapter_id=chapter_id,
                        speaker=speaker,
                        voice_id=cast_map.get(speaker, narrator_voice),
                        speed=default_speed,
                    )
                )
        casting = [CharacterCast(speaker=s, voice_id=v) for s, v in cast_map.items()]
        return AudiobookCastOutput(
            casting=casting, performances=performances, narrator_voice_id=narrator_voice
        )

    async def _cast_roster(
        self, roster: list[str], narrator_voice: str, user_cast: dict[str, str]
    ) -> dict[str, str]:
        """Character → voice map. LLM assigns voices; a deterministic distinct-voice
        assigner fills anything the model missed; user pins win."""
        cast_map: dict[str, str] = {_NARRATOR_SPEAKER: narrator_voice}
        llm_map = await self._llm_cast_roster(roster, narrator_voice) if roster else {}
        for speaker in roster:
            vid = llm_map.get(speaker)
            if vid in VOICE_CATALOG:
                cast_map[speaker] = vid
            else:
                cast_map[speaker] = _distinct_voice(speaker, set(cast_map.values()))
        cast_map.update(user_cast)  # user-pinned characters are authoritative
        return cast_map

    async def _llm_cast_roster(self, roster: list[str], narrator_voice: str) -> dict[str, str]:
        """Ask WM for a voice per character; ``{}`` on any failure (fall back to
        the deterministic assigner rather than fail the stage)."""
        message = (
            f"The narrator's voice is '{narrator_voice}'.\n\n"
            f"Allowed voices:\n{voice_catalog_prompt_block()}\n\n"
            "CHARACTERS (assign one voice each):\n" + "\n".join(roster)
        )

        async def _attempt(enable_reasoning: bool | None) -> dict[str, str]:
            resp = await self._wm.message(
                enable_reasoning=enable_reasoning,
                worker_name=_WORKER_NAME,
                message=message,
                system_prompt=_CASTING_ONLY_SYSTEM_PROMPT,
                output_schema=_CASTING_ONLY_SCHEMA,
                max_tokens=_CASTING_MAX_TOKENS,
            )
            data = resp.output
            if not isinstance(data, dict) or not isinstance(data.get("casting"), list):
                raise ValueError("audiobook_cast: WM casting output malformed")
            out: dict[str, str] = {}
            for entry in data["casting"]:
                if isinstance(entry, dict) and entry.get("speaker"):
                    out[str(entry["speaker"]).strip()] = str(entry.get("voice_id") or "")
            return out

        try:
            # Soft-fail path: guarded() still matters here — a permanent WM failure
            # drops to the deterministic assigner on the FIRST call instead of after
            # a full retry ladder that could never have succeeded.
            return await stage_call(
                downshift_reasoning(lambda think: guarded(lambda: _attempt(think))),
                timeout_seconds=self._llm_timeout,
                retries=self._llm_retries,
                retry_on=LLM_RETRY_ON,
                stage_name="audiobook_cast",
            )
        except Exception as exc:  # noqa: BLE001 -- casting failure → deterministic fallback
            import logging

            logging.getLogger(__name__).warning(
                "audiobook_cast roster casting failed, using deterministic voices: %s", exc
            )
            return {}

    def _build_message(self, chunks: list[dict[str, str]]) -> str:
        # Attribution is voice-agnostic, so the 54-voice catalog is deliberately
        # NOT in this prompt: it would be pure token noise and could tempt the
        # model into casting (which is the separate global call's job).
        chunk_lines = "\n".join(
            f'{{"chunk_id":"{c["chunk_id"]}","text":{_short_json(c["text"])}}}' for c in chunks
        )
        return (
            "CHUNKS (one JSON object per line) — attribute the speaker of each "
            f"(use '{_NARRATOR_SPEAKER}' for plain narration):\n{chunk_lines}"
        )

    def _assemble(
        self,
        *,
        chunks: list[dict[str, str]],
        cast_map: dict[str, str],
        raw_perf: list[object],
        default_speed: float,
    ) -> AudiobookCastOutput:
        # ``cast_map`` (speaker → valid voice) is already resolved by ``_cast_roster``
        # — LLM picks validated against ``VOICE_CATALOG``, deterministic distinct-voice
        # fill for gaps, user pins authoritative, Narrator fixed to the project voice.
        narrator_voice = cast_map.get(_NARRATOR_SPEAKER, _DEFAULT_NARRATOR_VOICE)

        # Per-chunk speaker + speed from the model, resolved through cast_map.
        perf_by_chunk: dict[str, dict[str, object]] = {}
        for entry in raw_perf:
            if not isinstance(entry, dict):
                continue
            entry_d = cast("dict[str, object]", entry)
            if entry_d.get("chunk_id"):
                perf_by_chunk[str(entry_d["chunk_id"])] = entry_d

        performances: list[ChunkPerformance] = []
        for chunk in chunks:  # iterate the PLAN — guarantees full coverage
            entry = perf_by_chunk.get(chunk["chunk_id"], {})
            speaker = str(entry.get("speaker") or _NARRATOR_SPEAKER).strip() or _NARRATOR_SPEAKER
            voice_id = cast_map.get(speaker, narrator_voice)
            performances.append(
                ChunkPerformance(
                    chunk_id=chunk["chunk_id"],
                    chapter_id=chunk["chapter_id"],
                    speaker=speaker,
                    voice_id=voice_id,
                    speed=_clamp_speed(entry.get("speed"), default_speed),
                )
            )

        casting = [CharacterCast(speaker=s, voice_id=v) for s, v in cast_map.items()]
        return AudiobookCastOutput(
            casting=casting,
            performances=performances,
            narrator_voice_id=narrator_voice,
        )


def _short_json(text: str) -> str:
    """JSON-encode a chunk excerpt (truncated) for safe single-line embedding."""
    excerpt = text[:_PROMPT_CHUNK_CHARS]
    return json.dumps(excerpt, ensure_ascii=False)
