"""Shared LLM sound-director base — picks per-line accent SFX cues.

Blueprint-agnostic core lifted verbatim from ``promo_sound_director`` so any
narrated blueprint (promo, showcase/demo_video) reuses the same director. A
subclass sets ``name`` + ``worker_name`` and supplies the ordered narration lines
(``line_id`` / ``text`` / ``seconds``) via ``director_lines`` from whatever prior
stage(s) it reads; the LLM call (through ``worker_management`` — the ``WMClient``
gateway, NEVER inference-router directly), catalog validation, offset/gain
clamping and the opt-in skip are all shared.

Hard invariants (the model is advisory, this stage is authoritative — mirrors
``audiobook_cast``):
  * every emitted ``sfx_id`` is validated against the loaded sound library (accent
    category); an unknown/non-accent id is dropped, never reaching a shot.
  * ``at_seconds`` is clamped into the line's duration (an out-of-range offset
    would place the cue past the shot).
  * at most ``_MAX_ACCENTS_PER_LINE`` cues per line — restraint over a wall of
    sound.

Opt-in + skip-clean: with no ``sound_effects`` config (or ``accents`` disabled)
the stage raises ``StageSkippedError`` → ``skipped`` (terminal-but-fine), so a
compile stage's ``depends_on`` never blocks a run that didn't opt in.
"""

from __future__ import annotations

import json
from itertools import batched
from typing import TYPE_CHECKING, cast

from w2v_common.stages import stage_call

from src.services.llm import WMClient
from src.services.stages.base import (
    StageHandler,
    StageInput,
    StageOutput,
    StagePreconditionError,
    StageSkippedError,
)
from src.services.stages.batch_ledger import BatchLedger
from src.services.stages.llm_budget import (
    LLM_RETRY_ON,
    SMALL_ANSWER_TOKENS,
    downshift_reasoning,
    llm_budget_seconds,
    llm_client_timeout_seconds,
    llm_retries,
    output_token_budget,
)
from src.services.stages.sound_catalog import SoundLibrary, load_sound_library
from src.services.stages.upstream import guarded

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

_MAX_ACCENTS_PER_LINE = 2
# Per-WM-call batch size: cap the ``cues`` output array so the small vLLM doesn't
# time out on an unbounded per-line schema and silently return no cues (mirrors
# shared/footage_critique's ``_CRITIC_BATCH_SIZE``).
_CUE_BATCH_SIZE = 5
# Reasoning-model output budget (see stages/llm_budget.py — a FLOOR, not a cap).
# One WM call answers ONE batch: <=_CUE_BATCH_SIZE lines x <=_MAX_ACCENTS_PER_LINE
# cues, and each cue is four scalars (line_id, sfx_id, at_seconds, gain_db) with no
# prose at all. A tiny fixed-shape reply ⇒ SMALL.
_CUE_MAX_TOKENS = output_token_budget(answer_tokens=SMALL_ANSWER_TOKENS)
_DEFAULT_ACCENT_GAIN_DB = -10.0
_GAIN_MIN, _GAIN_MAX = -24.0, 0.0
# How much of each line to show the LLM — enough to judge emphasis, not the
# whole line. Bounds token cost.
_PROMPT_LINE_CHARS = 200

_SYSTEM_PROMPT = (
    "You are a sound designer for a short promo video. You receive the ordered "
    "narration lines (each with a line_id and its duration in seconds) and a list "
    "of allowed accent sound ids. Choose AT MOST two accent cues for the lines "
    "that genuinely benefit from emphasis — a feature reveal, a confirmation, a "
    "punchline. MOST lines should get NONE; prefer restraint over a wall of "
    "sound. For each cue give the line_id, the sfx_id (ONLY from the allowed "
    "list), at_seconds (>=0 and strictly less than that line's duration), and an "
    "optional gain_db (-24..0, default -10). Do not accent every line."
)

# Object-rooted structured-output schema (WM convention). WM returns
# ``response.output`` already parsed to this shape.
_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "cues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "line_id": {"type": "string"},
                    "sfx_id": {"type": "string"},
                    "at_seconds": {"type": "number"},
                    "gain_db": {"type": "number"},
                },
                "required": ["line_id", "sfx_id", "at_seconds"],
            },
        }
    },
    "required": ["cues"],
}

# Cap the prompt text a subclass shows the LLM per line — re-exported so
# subclasses that build their own line list slice to the same bound.
DIRECTOR_LINE_CHARS = _PROMPT_LINE_CHARS


class SoundDirectorInput(StageInput):
    # Optional film-critic feedback threaded into the LLM message so a holistic
    # critic stage can re-drive this director. Absent on a normal DAG run ⇒ the
    # built message is byte-identical to today (inert by default). Lives on the
    # shared base Input so every subclass (promo, demo_video) inherits it.
    redrive_feedback: str | None = None


class LineSfxCue(StageOutput):
    line_id: str
    sfx_id: str
    at_seconds: float
    gain_db: float = _DEFAULT_ACCENT_GAIN_DB


class SoundDirectorOutput(StageOutput):
    cues: list[LineSfxCue]


class SoundDirectorStage(StageHandler[SoundDirectorInput, SoundDirectorOutput]):
    """Base LLM accent-cue director. Subclasses set ``name`` + ``worker_name`` and
    implement ``director_lines``. WM worker name must match ``^[a-zA-Z0-9_-]+$``;
    WM auto-creates it on first call (provider/model live on the WM worker config)."""

    input_model = SoundDirectorInput
    output_model = SoundDirectorOutput
    # Subclass overrides: the WM worker to route the director prompt to.
    worker_name: str = "sound-director"

    def __init__(self, client: WMClient | None = None) -> None:
        self._llm_timeout = llm_budget_seconds()
        self._llm_retries = llm_retries()
        self._wm = client or WMClient(timeout=llm_client_timeout_seconds())

    def director_lines(self, ctx: RunContext) -> list[dict[str, object]]:
        """Ordered narration lines the director scores, as
        ``[{line_id, text, seconds}]``. Subclass reads its own prior stage(s);
        raise ``StagePreconditionError`` if the required output is missing."""
        raise NotImplementedError

    async def handle(
        self,
        payload: SoundDirectorInput,
        ctx: RunContext,
    ) -> SoundDirectorOutput:
        cfg = ctx.config_snapshot.get("sound_effects") or {}
        # Opt-in + skip-clean: no sound design requested, or accents explicitly
        # off → no LLM work, stage marked skipped (never blocks the compile stage).
        if not cfg.get("enabled") or cfg.get("accents") is False:
            raise StageSkippedError(
                f"{self.name}: sound_effects accents not requested (opt-in stage)",
                missing=["sound_effects.accents"],
            )

        lines = self.director_lines(ctx)
        if not lines:
            raise StagePreconditionError(f"{self.name}: no narration lines available")

        valid_lines: dict[str, dict[str, object]] = {str(line["line_id"]): line for line in lines}
        # Tenant-scoped library: the LLM is shown only accents it's allowed to
        # pick, and every returned id is validated against the same set.
        library = await load_sound_library()

        # Bounded batches (≤ _CUE_BATCH_SIZE lines per WM call): an unbounded
        # ``cues`` output schema otherwise times out the small vLLM and the
        # stage silently degrades to no cues. Per-line cues are independent, so
        # the flat merge is exact; validation + per-line cap run once over the
        # union. Retry/timeout/soft-fail semantics are unchanged (stage_call).
        # The ledger and the batch list live OUTSIDE the attempt factory so a retry
        # RESUMES. When the accumulator lived inside, attempt 1 finished k of N batches,
        # timed out, and attempt 2 re-paid those k WM calls to reach the same place — a
        # stage whose N batches do not fit inside ONE attempt could never complete, no
        # matter how many retries it was given. See ``stages/batch_ledger``.
        batches = [list(batch) for batch in batched(lines, _CUE_BATCH_SIZE, strict=False)]
        ledger: BatchLedger[list[dict[str, object]], list[object]] = BatchLedger()

        async def _attempt(enable_reasoning: bool | None) -> SoundDirectorOutput:
            async def _cue(batch: list[dict[str, object]]) -> list[object]:
                message = self._build_message(
                    batch, library, redrive_feedback=payload.redrive_feedback
                )
                resp = await self._wm.message(
                    enable_reasoning=enable_reasoning,
                    worker_name=self.worker_name,
                    message=message,
                    system_prompt=_SYSTEM_PROMPT,
                    output_schema=_OUTPUT_SCHEMA,
                    max_tokens=_CUE_MAX_TOKENS,
                )
                data = resp.output
                if not isinstance(data, dict) or not isinstance(data.get("cues"), list):
                    # Retryable via stage_call — re-prompt for a conforming object.
                    raise ValueError(f"{self.name}: WM output missing cues")
                return list(data["cues"])

            parts = await ledger.run(batches, _cue)
            raw_cues = [item for part in parts for item in part]
            return SoundDirectorOutput(cues=self._validate(raw_cues, valid_lines, library))

        return await stage_call(
            downshift_reasoning(lambda think: guarded(lambda: _attempt(think))),
            timeout_seconds=self._llm_timeout,
            retries=self._llm_retries,
            retry_on=LLM_RETRY_ON,
            stage_name=self.name,
        )

    def _validate(
        self,
        raw: list[object],
        valid_lines: dict[str, dict[str, object]],
        library: SoundLibrary,
    ) -> list[LineSfxCue]:
        """Keep only cues with a known line_id + a valid accent id, clamp the
        offset into the line, and cap the count per line. The compiler never sees
        a bad id (the authoritative-stage invariant)."""
        per_line: dict[str, int] = {}
        out: list[LineSfxCue] = []
        for raw_cue in raw:
            if not isinstance(raw_cue, dict):
                continue
            # ``isinstance``-narrowing an ``object`` element yields
            # ``dict[Unknown, Unknown]`` (Never-typed keys); cast to the opaque
            # LLM-cue shape so the string-keyed lookups below type-check.
            cue = cast("dict[str, object]", raw_cue)
            line_id = str(cue.get("line_id") or "")
            sfx_id = str(cue.get("sfx_id") or "")
            line = valid_lines.get(line_id)
            if line is None or not library.is_valid(sfx_id, category="accent"):
                continue
            if per_line.get(line_id, 0) >= _MAX_ACCENTS_PER_LINE:
                continue
            at = _clamp_offset(cue.get("at_seconds"), cast("float", line["seconds"]))
            gain = _clamp_gain(cue.get("gain_db"))
            per_line[line_id] = per_line.get(line_id, 0) + 1
            out.append(LineSfxCue(line_id=line_id, sfx_id=sfx_id, at_seconds=at, gain_db=gain))
        return out

    def _build_message(
        self,
        lines: list[dict[str, object]],
        library: SoundLibrary,
        *,
        redrive_feedback: str | None = None,
    ) -> str:
        body = "\n".join(
            f'{{"line_id":"{line["line_id"]}","seconds":{line["seconds"]:.1f},'
            f'"text":{json.dumps(line["text"], ensure_ascii=False)}}}'
            for line in lines
        )
        message = (
            "Allowed accent sounds:\n"
            f"{library.accent_prompt_block()}\n\n"
            "NARRATION LINES (one JSON object per line):\n"
            f"{body}"
        )
        return _append_redrive_feedback(message, redrive_feedback)


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


def _clamp_offset(value: object, line_seconds: float) -> float:
    """Clamp an accent offset into ``[0, line_seconds)``; a bad/oob value lands
    at the line midpoint so the cue still plays inside the shot."""
    try:
        at = float(cast("float", value))
    except TypeError, ValueError:
        at = -1.0
    if line_seconds <= 0:
        return max(0.0, at)
    if at < 0 or at >= line_seconds:
        return line_seconds * 0.5
    return at


def _clamp_gain(value: object) -> float:
    try:
        gain = float(cast("float", value))
    except TypeError, ValueError:
        return _DEFAULT_ACCENT_GAIN_DB
    return max(_GAIN_MIN, min(_GAIN_MAX, gain))
