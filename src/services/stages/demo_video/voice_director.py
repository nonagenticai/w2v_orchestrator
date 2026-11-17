"""voice_director — LLM performance director for the demo_video narration spine.

One :class:`PerformanceDirective` per narration line: how a line should be
*delivered* (rate, emphasis, a leading pause, an optional SSML fragment and a
free-text delivery note). Structurally a sibling of
``shared/sound_director.py`` — same ``StageHandler`` shape, the same
``__init__`` LLM-budget wiring, the same WM call wrapped in
``stage_call(lambda: guarded(_attempt), …)``, and the same
model-is-advisory / stage-is-authoritative validate+clamp loop. Unlike the
sound director it reads the current best line spine directly (via
``resolve_plan_output`` — the same precedence every demo_video consumer uses)
rather than through a subclass hook, because voice direction is demo_video-scoped
by design.

Coverage-fill (mirrors ``audiobook_cast``): the authoritative loop iterates the
PLAN lines, matching LLM directives by ``line_id``, so a line the model forgot
still gets a directive (its neutral default) and no downstream narrate step
ever meets a line without one.

Slice-A scope: kokoro-only. Any ``provider_tier`` the model proposes (e.g.
``"expressive"``) is forced back to ``"free"`` here — provider selection is not
wired yet and must not leak a paid tier into narration.

Skip-clean: with no ``voice.direction`` in the config snapshot the stage does
NO LLM work and raises :class:`StageSkippedError`, so a no-direction run is
behaviour-preserving and never blocks a stage that ``depends_on`` it.
"""

from __future__ import annotations

import json
import logging
from itertools import batched
from typing import TYPE_CHECKING, ClassVar, cast

from pydantic import BaseModel
from w2v_common.stages import StageRetriesExhaustedError, stage_call

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
from src.services.stages.demo_video.spine import resolve_plan_output
from src.services.stages.llm_budget import (
    LLM_RETRY_ON,
    MEDIUM_ANSWER_TOKENS,
    llm_budget_seconds,
    llm_client_timeout_seconds,
    llm_retries,
    output_token_budget,
)
from src.services.stages.upstream import TRANSIENT_4XX_MARKERS, guarded

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

# WM worker name (regex ^[a-zA-Z0-9_-]+$). The worker's curated system prompt
# lives in WM's seed (SEED_MANAGED); the provider/model lives on the WM worker
# config, NOT here — that's the whole point of routing through WM.
_WORKER_NAME = "voice-director"

log = logging.getLogger(__name__)

# Per-WM-call batch size: cap the ``directives`` output array so the small vLLM
# doesn't time out on an unbounded per-line schema and silently return partial
# coverage (mirrors shared/footage_critique's ``_CRITIC_BATCH_SIZE``).
_DIRECTOR_BATCH_SIZE = 5

# Slice A is kokoro-only: provider selection is not wired, so every directive is
# pinned to the free tier no matter what the model proposes.
_FORCED_PROVIDER_TIER = "free"
# Delivery-rate multiplier bounds (1.0 = normal). A narrow window: anything wider
# makes kokoro sound sped-up/slurred rather than merely energetic/measured.
_RATE_MIN, _RATE_MAX = 0.85, 1.15
_PAUSE_MIN_MS, _PAUSE_MAX_MS = 0, 2000
# Same idiom as ``shared/sound_director._MAX_ACCENTS_PER_LINE``: restraint over a
# wall of directives. Each emphasised word becomes a SEPARATE kokoro utterance at
# 0.88x the request speed (measured mid-sentence rate drop ~12%), so an
# unbounded ``emphasis_word_indices`` list lets the model emphasise every word in
# a line and slow the whole thing down piece by piece.
_MAX_EMPHASIS_WORDS_PER_LINE = 2
# How much of a beat's OWN allotment its authored leading pause may consume before
# it is scaled back. ``pause_before_ms`` is clamped per-line into [0, 2000]ms with
# no regard for how long the beat actually is — a 2s pause on a 4s beat is half the
# beat gone to silence before a single word is spoken. 10% is a sane starting
# fraction: enough room for a genuine dramatic beat, not enough to eat the line.
_MAX_PAUSE_FRACTION_OF_BEAT = 0.10
# How much of each line to show the LLM — enough to judge delivery + emphasis
# word positions, not the whole line. Bounds token cost.
_PROMPT_LINE_CHARS = 240

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
                    "line_id": {"type": "string"},
                    "provider_tier": {"type": "string"},
                    "delivery_note": {"type": "string"},
                    "emphasis_word_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "pause_before_ms": {"type": "integer"},
                    "rate_hint": {"type": "number"},
                    "ssml": {"type": "string"},
                },
                "required": ["line_id"],
            },
        }
    },
    "required": ["directives"],
}


# MEDIUM answer: the call is already BATCHED at ``_DIRECTOR_BATCH_SIZE`` (5) lines,
# so ``directives`` is at most five objects — but each carries a prose
# ``delivery_note``, an emphasis-index array and a full ``ssml`` string roughly as
# long as the narration line itself. Five prose-bearing objects is the "one object
# with prose fields" tier, not a single verdict.
_DIRECTOR_MAX_TOKENS = output_token_budget(answer_tokens=MEDIUM_ANSWER_TOKENS)


def _append_redrive_feedback(message: str, redrive_feedback: str | None) -> str:
    """Append the critic-feedback block iff ``redrive_feedback`` is non-empty.

    Inert by default: with no feedback the message is returned byte-identical, so
    a normal DAG run is unchanged. A holistic film-critic stage sets it to
    re-drive this director from the seams it found.
    """
    if not redrive_feedback:
        return message
    return (
        f"{message}\n\nCRITIC FEEDBACK — fix these seams (regenerate to address, "
        f"keep everything else stable):\n{redrive_feedback}"
    )


def _short_json(text: str) -> str:
    """JSON-encode a line excerpt (truncated) for safe single-line embedding."""
    return json.dumps(text[:_PROMPT_LINE_CHARS], ensure_ascii=False)


def _clamp_rate(value: object, default: float = 1.0) -> float:
    if not isinstance(value, (int, float, str)):
        return default
    try:
        rate = float(value)
    except TypeError, ValueError:
        return default
    return max(_RATE_MIN, min(_RATE_MAX, rate))


def _to_int(value: object, default: int) -> int:
    if not isinstance(value, (int, float, str)):
        return default
    try:
        return int(value)
    except TypeError, ValueError:
        return default


def _to_float(value: object, default: float = 0.0) -> float:
    if not isinstance(value, (int, float, str)):
        return default
    try:
        return float(value)
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


class VoiceDirectorInput(StageInput):
    # Optional film-critic feedback threaded into the LLM message so a holistic
    # critic stage can re-drive this director. Absent on a normal DAG run ⇒ the
    # built message is byte-identical to today (inert by default).
    redrive_feedback: str | None = None


class PerformanceDirective(BaseModel):
    line_id: str
    provider_tier: str = "free"
    delivery_note: str = ""
    emphasis_word_indices: list[int] = []
    pause_before_ms: int = 0
    rate_hint: float = 1.0
    ssml: str = ""


class VoiceDirectorOutput(StageOutputWithNotes):
    # `_assemble` iterates the PLAN lines (coverage-fill), so `directives` is
    # non-empty exactly when the spine is — and the spine being empty already raises
    # StagePreconditionError here. `demo_video_narrate` looks up SSML and rate_hint
    # per line from these; an empty list silently reverts every line to undirected
    # narration, which is indistinguishable from "voice direction was never on".
    # The unrequested case raises StageSkippedError and never reaches this gate.
    produces: ClassVar[tuple[str, ...]] = ("directives",)

    directives: list[PerformanceDirective]


def _is_lost_draw(exc: BaseException) -> bool:
    """Is this exhaustion a lost SAMPLING draw rather than a real upstream fault?

    Matches the same substring ``upstream.classify`` already uses to call this 422
    transient (:data:`~src.services.stages.upstream.TRANSIENT_4XX_MARKERS`) — read from
    there, never re-spelled here, so the two cannot drift apart.
    """
    seen: set[int] = set()
    link: BaseException | None = exc
    while link is not None and id(link) not in seen:
        seen.add(id(link))
        text = str(link).lower()
        if any(marker in text for marker in TRANSIENT_4XX_MARKERS):
            return True
        # StageRetriesExhaustedError carries the real failure on `last_error` and does
        # not always set `__cause__`, so follow it explicitly rather than relying on
        # the implicit chain alone.
        nxt = getattr(link, "last_error", None)
        if not isinstance(nxt, BaseException):
            nxt = link.__cause__ or link.__context__
        link = nxt
    return False


class VoiceDirectorStage(StageHandler[VoiceDirectorInput, VoiceDirectorOutput]):
    """LLM per-line performance director. The model is advisory; this stage is
    authoritative — every returned value is clamped/coerced and every plan line
    gets a directive (coverage-fill) before it can reach narration."""

    name = "voice_director"
    input_model = VoiceDirectorInput
    output_model = VoiceDirectorOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("llm"),)
    worker_name: str = _WORKER_NAME

    def __init__(self, client: WMClient | None = None) -> None:
        self._llm_timeout = llm_budget_seconds()
        self._llm_retries = llm_retries()
        self._wm = client or WMClient(timeout=llm_client_timeout_seconds())

    async def handle(
        self,
        payload: VoiceDirectorInput,
        ctx: RunContext,
    ) -> VoiceDirectorOutput:
        voice_cfg = ctx.config_snapshot.get("voice") or {}
        direction = voice_cfg.get("direction")
        # Skip-clean: no director intent → no LLM work, stage marked skipped so a
        # no-voice-direction run stays behaviour-preserving.
        if not direction:
            raise StageSkippedError(
                f"{self.name}: no voice.direction (opt-in stage)",
                missing=["voice.direction"],
            )

        gen = resolve_plan_output(ctx)
        lines = list(gen.get("lines") or []) if gen else []
        if not lines:
            raise StagePreconditionError(f"{self.name}: no narration lines available")

        # Direct the lines in bounded batches (≤ _DIRECTOR_BATCH_SIZE per WM call) so
        # no single call carries an unbounded ``directives`` output schema — the small
        # vLLM otherwise times out and the run silently degrades to neutral-default
        # coverage. Per-line directives are independent, so the flat merge is exact.
        #
        # The ledger and the batch list live OUT here, outside the attempt factory, so
        # a retry resumes instead of restarting. Inside, the accumulator reset on every
        # attempt: attempt 1 finished k of N batches, timed out, and attempt 2 re-paid
        # those k WM calls to reach the same place and time out again — which is how
        # this stage died in production on 2026-08-20. See ``batch_ledger``.
        batches = [list(batch) for batch in batched(lines, _DIRECTOR_BATCH_SIZE, strict=False)]
        ledger: BatchLedger[list[dict[str, object]], list[object]] = BatchLedger()

        async def _attempt() -> VoiceDirectorOutput:
            async def _direct(batch: list[dict[str, object]]) -> list[object]:
                message = self._build_message(
                    batch, direction, redrive_feedback=payload.redrive_feedback
                )
                resp = await self._wm.message(
                    worker_name=self.worker_name,
                    message=message,
                    output_schema=_OUTPUT_SCHEMA,
                    max_tokens=_DIRECTOR_MAX_TOKENS,
                    # Chain-of-thought OFF, same fix and same evidence as
                    # `action_reel/segment_select.py` (#873). Measured through
                    # worker-management on the production path (Job `wmprobe-segsel`,
                    # 2026-08-20, same prompt and schema, `max_tokens=8000`): with
                    # reasoning ON the call hit the stage's own 950s deadline and
                    # returned NOTHING -- twice, on both output schemas. With reasoning
                    # OFF the same call answered in 34-67s carrying real content, every
                    # time. The model spends the whole budget thinking and never emits
                    # the answer, surfacing one layer up as `Exceeded maximum output
                    # retries (3)` or, here, `StageTimeoutError: all 2 attempts timed
                    # out (per-attempt budget 950.0s)` -- which is exactly how
                    # `voice_director` died twice in production on 2026-08-20.
                    #
                    # Only sendable because worker-management #188 added the field and
                    # `w2v_common` >= 0.61.0 carries it -- `MessageRequest` is
                    # `extra="forbid"`, so an older WM would 422 the call. Omit it and
                    # w2v's calls take WM's in-memory auto-create branch
                    # (`model_provider="vllm"`), straight to vLLM, never through
                    # inference-router -- the only component that would otherwise turn
                    # thinking off. See memory `w2v-wm-calls-bypass-inference-router`.
                    enable_reasoning=False,
                )
                data = resp.output
                if not isinstance(data, dict) or not isinstance(data.get("directives"), list):
                    # ValueError is in LLM_RETRY_ON → re-prompt for a conforming object.
                    # It propagates out of `ledger.run`, so `stage_call` retries and the
                    # retry re-issues ONLY this batch.
                    raise ValueError(f"{self.name}: WM output missing directives")
                return list(data["directives"])

            parts = await ledger.run(batches, _direct)
            raw_directives = [directive for part in parts for directive in part]
            directives, assemble_notes = self._assemble(lines, raw_directives)
            return VoiceDirectorOutput(directives=directives, notes=assemble_notes)

        try:
            return await stage_call(
                lambda: guarded(_attempt),
                timeout_seconds=self._llm_timeout,
                retries=self._llm_retries,
                retry_on=LLM_RETRY_ON,
                stage_name=self.name,
            )
        except StageRetriesExhaustedError as exc:
            # A LOST DRAW MUST NOT KILL A PAID RUN. This stage is advisory by
            # construction — "model-is-advisory / stage-is-authoritative", and
            # `_assemble` already coverage-fills every omitted line with its neutral
            # default. Losing structured-output validation on one batch is therefore a
            # reason to narrate undirected, not to discard the whole film.
            #
            # Measured: showcase run 01a089bf died here 2026-09-10 after ~64 minutes,
            # with the SAME validation error every attempt ("'directives' is a required
            # property") — 8 consecutive lost draws at temperature 0.1. `voice_director`
            # is 108 completed vs 2 failed since 2026-08-27, and the other failure took
            # demo_video on 2026-08-28. The draw is recoverable; the run was not.
            #
            # SCOPED DELIBERATELY: only the sampling marker upstream.classify already
            # treats as transient. A connection error, a 5xx, or a real contract fault
            # still propagates — swallowing those would hide an outage behind neutral
            # narration, which is precisely the failure mode the `directives` docstring
            # warns about.
            if not _is_lost_draw(exc):
                raise
            log.warning(
                "%s: exhausted retries on a lost sampling draw — narrating undirected "
                "rather than discarding the run: %s",
                self.name,
                exc,
            )
            directives, _assemble_notes = self._assemble(lines, [])
            return VoiceDirectorOutput(
                directives=directives,
                # RECORDED, not silent: without this an all-neutral result is
                # indistinguishable from "voice direction was never on" (see the
                # `directives` field docstring). `_assemble_notes` is always empty
                # here — every line took the neutral-default branch, which raises no
                # pause/emphasis note — but is still discarded explicitly rather
                # than assumed, so a future coverage-fill change cannot silently
                # drop a real note on this path.
                notes=[
                    f"{self.name}: voice direction unavailable (lost sampling draw after "
                    f"{self._llm_retries} attempts); all {len(lines)} line(s) narrated "
                    f"with neutral defaults"
                ],
            )

    def _assemble(
        self,
        lines: list[dict[str, object]],
        raw_directives: list[object],
    ) -> tuple[list[PerformanceDirective], list[str]]:
        """One directive per PLAN line (full coverage), matching LLM output by
        line_id. Every field is clamped/coerced; a line the model omitted gets a
        neutral default. Returns the directives AND any honest notes raised while
        clamping (e.g. an authored pause that would have eaten the beat)."""
        by_id: dict[str, dict[str, object]] = {}
        for entry in raw_directives:
            if not isinstance(entry, dict):
                continue
            directive = cast("dict[str, object]", entry)
            if directive.get("line_id"):
                by_id[str(directive["line_id"])] = directive

        out: list[PerformanceDirective] = []
        notes: list[str] = []
        for line in lines:  # iterate the PLAN — guarantees full coverage
            line_id = str(line.get("line_id") or line.get("id") or "")
            entry = by_id.get(line_id)
            if entry is None:
                out.append(
                    PerformanceDirective(
                        line_id=line_id,
                        rate_hint=1.0,
                        provider_tier=_FORCED_PROVIDER_TIER,
                    )
                )
                continue
            text = str(line.get("text") or "")
            word_count = len(text.split())
            # Restraint over a wall of emphasis: every emphasised word becomes its
            # own kokoro utterance at a slower request speed (see
            # `_MAX_EMPHASIS_WORDS_PER_LINE`), so the count is capped, not just the
            # index range. Order is preserved (the model's own priority ordering,
            # earliest-first) rather than re-sorted.
            emphasis = [
                i for i in _int_list(entry.get("emphasis_word_indices")) if 0 <= i < word_count
            ][:_MAX_EMPHASIS_WORDS_PER_LINE]
            pause_ms = self._clamp_pause(
                _to_int(entry.get("pause_before_ms"), 0),
                beat_seconds=_to_float(line.get("seconds")),
                line_id=line_id,
                notes=notes,
            )
            out.append(
                PerformanceDirective(
                    line_id=line_id,
                    # Slice A: force free — ignore any proposed paid/expressive tier.
                    provider_tier=_FORCED_PROVIDER_TIER,
                    delivery_note=str(entry.get("delivery_note") or ""),
                    emphasis_word_indices=emphasis,
                    pause_before_ms=pause_ms,
                    rate_hint=_clamp_rate(entry.get("rate_hint")),
                    ssml=str(entry.get("ssml") or ""),
                )
            )
        return out, notes

    @staticmethod
    def _clamp_pause(
        pause_ms: int,
        *,
        beat_seconds: float,
        line_id: str,
        notes: list[str],
    ) -> int:
        """Clamp an authored leading pause into ``[0, _PAUSE_MAX_MS]``, THEN scale
        it down further when it would eat too much of its own beat.

        A pause is clamped per-line against the API's own legal range, but nothing
        upstream of this checked it against the ONE thing that actually bounds a
        sane pause: how long the beat is. A 2000ms pause on a 4s beat is half the
        beat gone to silence before a single word plays. When the authored pause
        exceeds ``_MAX_PAUSE_FRACTION_OF_BEAT`` of the beat, it is scaled DOWN to
        exactly that fraction (never rejected to zero — a shorter dramatic pause is
        still a legitimate direction) and the clamp is recorded as an honest note,
        never silently absorbed.

        ``beat_seconds <= 0`` (unknown/unset allotment) skips the beat-relative
        check entirely — there is nothing to scale against — and only the
        pre-existing absolute clamp applies.
        """
        pause_ms = max(_PAUSE_MIN_MS, min(_PAUSE_MAX_MS, pause_ms))
        if beat_seconds <= 0 or pause_ms <= 0:
            return pause_ms
        beat_ms = beat_seconds * 1000.0
        limit_ms = beat_ms * _MAX_PAUSE_FRACTION_OF_BEAT
        if pause_ms <= limit_ms:
            return pause_ms
        scaled = int(limit_ms)
        notes.append(
            f"{_WORKER_NAME}: line {line_id!r} pause_before_ms {pause_ms} scaled to "
            f"{scaled} — it exceeded {_MAX_PAUSE_FRACTION_OF_BEAT:.0%} of its "
            f"{beat_seconds:.2f}s beat"
        )
        return scaled

    def _build_message(
        self,
        lines: list[dict[str, object]],
        direction: object,
        *,
        redrive_feedback: str | None = None,
    ) -> str:
        body = "\n".join(
            f'{{"line_id":"{line.get("line_id") or line.get("id")}",'
            f'"text":{_short_json(str(line.get("text") or ""))}}}'
            for line in lines
        )
        intent = (
            direction if isinstance(direction, str) else json.dumps(direction, ensure_ascii=False)
        )
        message = (
            f"DIRECTOR INTENT:\n{intent}\n\nNARRATION LINES (one JSON object per line):\n{body}"
        )
        return _append_redrive_feedback(message, redrive_feedback)
