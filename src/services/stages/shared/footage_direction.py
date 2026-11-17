"""Shared LLM footage shot-planner base — turns per-line intent into a shot plan.

Blueprint-agnostic core mirroring ``sound_director``: a subclass sets ``name`` +
``worker_name`` and supplies the ordered lines (``line_id`` / ``text`` /
``seconds`` / ``has_click``) via ``planner_lines`` from whatever prior stage(s) it
reads; the LLM call (through ``worker_management`` — the ``WMClient`` gateway,
NEVER inference-router directly), plan validation/coercion and the opt-in skip are
all shared.

Hard invariants (the model is advisory, this stage is authoritative — mirrors
``sound_director``):
  * every emitted ``line_id`` is validated against the supplied line set; an
    unknown id is dropped, never reaching the compiler.
  * enumerated fields (``zoom`` / ``reframe`` / ``transition_signal`` / callout
    ``type``) are coerced to a known member or a safe default — the model can
    never inject a stray value.
  * COORDINATES ARE NOT THE MODEL'S TO SET. The plan carries only INTENT (zoom on
    / off, a callout label); the actual bbox comes from the captured focus track
    at compile time. A line with no captured click (``has_click`` False) has NO
    target, so its zoom is forced off and any callout dropped — a zoom easing to
    nothing / a callout on nothing is the failure mode.

Opt-in + skip-clean: with no ``direction.footage`` block the stage raises
``StageSkippedError`` → ``skipped`` (terminal-but-fine), so a compile stage's
``depends_on`` never blocks a run that didn't opt in.
"""

from __future__ import annotations

import json
from itertools import batched
from typing import TYPE_CHECKING, Any, ClassVar, TypedDict, cast

from w2v_common.stages import stage_call

from src.services.llm import WMClient
from src.services.stages.base import (
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
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

# Allowed enumerated members. Anything else the model returns is coerced to the
# first-listed safe default in ``_validate`` — the authoritative-stage invariant.
_CALLOUT_TYPES = {"badge", "label", "spotlight", "arrow"}
_ZOOM = {"auto", "off"}
_REFRAME = {"follow-cursor", "static"}
_SIGNALS = {"none", "cut", "dissolve", "push"}
_SPLIT_LAYOUTS = {"h", "v"}
_MAX_LABEL = 120
# Per-WM-call batch size: cap the ``plans`` output array so the small vLLM doesn't
# time out on an unbounded per-line schema and silently return no plans (mirrors
# shared/footage_critique's ``_CRITIC_BATCH_SIZE``).
_PLAN_BATCH_SIZE = 5
# Reasoning-model output budget (see stages/llm_budget.py — a FLOOR, not a cap).
# One WM call answers ONE batch: <=_PLAN_BATCH_SIZE ``plans`` objects, each a
# line_id + short enum fields (zoom/reframe/transition_signal/split) and at most
# one callout with a <=_MAX_LABEL-char label. A handful of small objects ⇒ MEDIUM.
_PLAN_MAX_TOKENS = output_token_budget(answer_tokens=MEDIUM_ANSWER_TOKENS)
# Run-total restraint bound (how many SHOTS carry a callout), orthogonal to the
# compiler's per-shot cap of 4. Each ShotPlan holds ≤1 callout, so this is a
# per-run count of callout-bearing shots.
_MAX_CALLOUT_SHOTS_PER_RUN = 6
# How much of each line to show the LLM — enough to judge the shot, not the whole
# line. Bounds token cost. Re-exported so subclasses slice their line text to it.
_PROMPT_LINE_CHARS = 200
DIRECTOR_LINE_CHARS = _PROMPT_LINE_CHARS

_SYSTEM_PROMPT = (
    "You are a footage director for a short screen-recording demo video. You "
    "receive the ordered lines (each with a line_id, its duration in seconds, "
    "whether the shot captured a real click/target, and the narration text). For "
    "each line, decide how its recorded footage should be framed: MOST shots "
    'should be zoom:"auto" with NO callout — prefer restraint. Add a callout '
    "ONLY for a genuine click or reveal that a viewer must not miss, giving its "
    "type (badge|label|spotlight|arrow) and a short label. Callout type: "
    "spotlight = isolate/dim around one element; arrow = point at an off-center "
    "element; badge = numbered step; label = short caption. MOST shots get NO "
    "callout. You do NOT set any coordinates — the "
    "compiler reads the real click position from the recording. "
    "Rarely — for a before/after or side-by-side comparison beat — you MAY add "
    'split:{layout:"h"|"v", second_clip:"<line_id>"} to composite a SECOND recorded '
    "clip beside this shot (h = left/right, v = top/bottom). Name ONLY another line_id "
    "from the input (lowercase, exactly as given) as second_clip; you do NOT set any "
    "URL — the compiler resolves it. MOST shots have NO split. "
    "Output exactly one "
    "plan per line_id."
)

# Object-rooted structured-output schema (WM convention). WM returns
# ``response.output`` already parsed to this shape.
_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "plans": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "line_id": {"type": "string"},
                    "zoom": {"type": "string"},
                    "callout": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "description": "one of badge|label|spotlight|arrow",
                            },
                            "label": {"type": "string"},
                        },
                    },
                    "transition_signal": {"type": "string"},
                    "split": {"type": "object"},
                    "reframe": {"type": "string"},
                },
                "required": ["line_id"],
            },
        }
    },
    "required": ["plans"],
}


class PlannerLine(TypedDict):
    """One ordered line the director plans over. ``has_click`` is True only when the
    recording captured a real target (bbox) for this line."""

    line_id: str
    text: str
    seconds: float
    has_click: bool


class FootageDirectionInput(StageInput):
    # Optional film-critic feedback threaded into the LLM message so a holistic
    # critic stage can re-drive this director. Absent on a normal DAG run ⇒ the
    # built message is byte-identical to today (inert by default). Lives on the
    # shared base Input so every subclass (promo, demo_video) inherits it.
    redrive_feedback: str | None = None


class Callout(StageOutput):
    type: str
    label: str


class ShotPlan(StageOutput):
    line_id: str
    zoom: str = "auto"
    callout: Callout | None = None
    transition_signal: str = "none"
    split: dict[str, object] | None = None
    reframe: str = "follow-cursor"


class FootageDirectionOutput(StageOutputWithNotes):
    # Both no-op paths skip cleanly (no config; no footage lines), so a COMPLETED
    # footage_direction with zero plans is the failure this module's own header
    # names: the bounded output array lets a small model time out and "silently
    # return no plans". Consumers (`promo_render.resolve_footage_plans`,
    # `demo_video/compile.py`) then reproduce the UNDIRECTED footage byte-for-byte
    # with nothing disagreeing.
    produces: ClassVar[tuple[str, ...]] = ("plans",)

    plans: list[ShotPlan]


class FootageDirectionStage(StageHandler[FootageDirectionInput, FootageDirectionOutput]):
    """Base LLM footage shot-planner. Subclasses set ``name`` + ``worker_name`` and
    implement ``planner_lines``. WM worker name must match ``^[a-zA-Z0-9_-]+$``; WM
    auto-creates it on first call (provider/model live on the WM worker config)."""

    input_model = FootageDirectionInput
    output_model = FootageDirectionOutput
    # Subclass may override: the WM worker to route the director prompt to.
    worker_name: str = "footage-director"

    def __init__(self, client: WMClient | None = None) -> None:
        self._llm_timeout = llm_budget_seconds()
        self._llm_retries = llm_retries()
        self._wm = client or WMClient(timeout=llm_client_timeout_seconds())

    def planner_lines(self, ctx: RunContext) -> list[PlannerLine]:
        """Ordered lines the director plans, as
        ``[{line_id, text, seconds, has_click}]``. Subclass reads its own prior
        stage(s); raise ``StagePreconditionError`` if the required output is
        missing. ``has_click`` is True only when the recording captured a real
        target for this line — the compiler then has a bbox to zoom/callout at."""
        raise NotImplementedError

    async def handle(
        self,
        payload: FootageDirectionInput,
        ctx: RunContext,
    ) -> FootageDirectionOutput:
        # Opt-in + skip-clean: no directed footage requested → no LLM work, stage
        # marked skipped (never blocks the compile stage).
        footage_cfg = (ctx.config_snapshot.get("direction") or {}).get("footage")
        if not footage_cfg:
            raise StageSkippedError(
                f"{self.name}: direction.footage not requested (opt-in stage)",
                missing=["direction.footage"],
            )
        # `direction.footage.director: false` turns the LLM planner OFF while leaving the
        # rest of directed footage (focus/cursor tracks, author-pinned splits) on. The
        # seeded default block cannot be emptied (a present-but-empty footage block is
        # re-seeded), so this is the author's only way to a model-free footage pass.
        if isinstance(footage_cfg, dict) and footage_cfg.get("director") is False:
            raise StageSkippedError(
                f"{self.name}: direction.footage.director is false (LLM planner off)",
                missing=["direction.footage.director"],
            )

        lines = self.planner_lines(ctx)
        if not lines:
            # Skip-clean, NOT a precondition failure: an empty list means this run
            # simply has no footage-typed shots to frame (a slides-only promo, say) —
            # nothing to direct, and the compile path renders fine without a plan.
            # A genuinely MISSING upstream output is still a hard error; the subclass
            # raises StagePreconditionError for that before returning here. This
            # matters now that `direction.footage` is seeded by default (WS-1): the
            # block's presence no longer proves an author asked for directed footage,
            # so it must not be able to fail an otherwise-good run.
            raise StageSkippedError(
                f"{self.name}: no footage lines to direct", missing=["footage_lines"]
            )

        valid_lines: dict[str, PlannerLine] = {str(line["line_id"]): line for line in lines}

        # Bounded batches (≤ _PLAN_BATCH_SIZE lines per WM call): an unbounded
        # ``plans`` output schema otherwise times out the small vLLM and the
        # stage silently degrades to no plans. Per-line plans are independent,
        # so the flat merge is exact; validation runs once over the union.
        # Retry/timeout/soft-fail semantics are unchanged (stage_call ladder).
        # The ledger and the batch list live OUTSIDE the attempt factory so a retry
        # RESUMES. When the accumulator lived inside, attempt 1 finished k of N batches,
        # timed out, and attempt 2 re-paid those k WM calls to reach the same place — a
        # stage whose N batches do not fit inside ONE attempt could never complete, no
        # matter how many retries it was given. See ``stages/batch_ledger``.
        batches = [list(batch) for batch in batched(lines, _PLAN_BATCH_SIZE, strict=False)]
        ledger: BatchLedger[list[PlannerLine], list[Any]] = BatchLedger()

        async def _attempt(enable_reasoning: bool | None) -> FootageDirectionOutput:
            async def _plan(batch: list[PlannerLine]) -> list[Any]:
                message = self._build_message(batch, redrive_feedback=payload.redrive_feedback)
                resp = await self._wm.message(
                    enable_reasoning=enable_reasoning,
                    worker_name=self.worker_name,
                    message=message,
                    system_prompt=_SYSTEM_PROMPT,
                    output_schema=_OUTPUT_SCHEMA,
                    max_tokens=_PLAN_MAX_TOKENS,
                )
                data = resp.output
                if not isinstance(data, dict) or not isinstance(data.get("plans"), list):
                    # Retryable via stage_call — re-prompt for a conforming object.
                    raise ValueError(f"{self.name}: WM output missing plans")
                return list(data["plans"])

            parts = await ledger.run(batches, _plan)
            raw_plans = [item for part in parts for item in part]
            return FootageDirectionOutput(plans=self._validate(raw_plans, valid_lines))

        return await stage_call(
            downshift_reasoning(lambda think: guarded(lambda: _attempt(think))),
            timeout_seconds=self._llm_timeout,
            retries=self._llm_retries,
            retry_on=LLM_RETRY_ON,
            stage_name=self.name,
        )

    def _validate(
        self,
        raw: list[Any],
        valid_lines: dict[str, PlannerLine],
    ) -> list[ShotPlan]:
        """Keep only plans with a known line_id, coerce every enumerated field to a
        known member, clamp the callout label, and enforce the no-target invariant:
        a line whose recording captured NO click (``has_click`` False) has no bbox
        to zoom/callout at, so its zoom is forced off and any callout dropped. The
        compiler never sees a stray value (the authoritative-stage invariant)."""
        out: list[ShotPlan] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            line_id = str(item.get("line_id") or "")
            line = valid_lines.get(line_id)
            if line is None:
                continue
            zoom = item.get("zoom")
            zoom = zoom if zoom in _ZOOM else "auto"
            reframe = item.get("reframe")
            reframe = reframe if reframe in _REFRAME else "follow-cursor"
            signal = item.get("transition_signal")
            signal = signal if signal in _SIGNALS else "none"
            callout = _parse_callout(item.get("callout"))
            split = _parse_split(item.get("split"))
            # No captured target ⇒ no bbox ⇒ zoom/callout would point at nothing.
            if not line["has_click"]:
                zoom = "off"
                callout = None
            out.append(
                ShotPlan(
                    line_id=line_id,
                    zoom=zoom,
                    callout=callout,
                    transition_signal=signal,
                    split=split,
                    reframe=reframe,
                )
            )
        # Run-total restraint cap: keep the callout on the first
        # ``_MAX_CALLOUT_SHOTS_PER_RUN`` plans that carry one, then null the
        # callout on any further callout-bearing plans. The plans themselves are
        # never dropped — only their ``.callout`` is cleared.
        kept = 0
        for plan in out:
            if plan.callout is None:
                continue
            if kept < _MAX_CALLOUT_SHOTS_PER_RUN:
                kept += 1
            else:
                plan.callout = None
        return out

    def _build_message(
        self,
        lines: list[PlannerLine],
        *,
        redrive_feedback: str | None = None,
    ) -> str:
        body = "\n".join(
            json.dumps(
                {
                    "line_id": line["line_id"],
                    "seconds": round(line["seconds"], 1),
                    "has_click": line["has_click"],
                    "text": line["text"][:DIRECTOR_LINE_CHARS],
                },
                ensure_ascii=False,
            )
            for line in lines
        )
        message = "SHOT LINES (one JSON object per line):\n" + body
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


def _parse_callout(raw: object) -> Callout | None:
    """Keep a callout only when it is a dict with a known type and a non-empty,
    length-clamped label; anything else ⇒ None (no callout rides the shot)."""
    if not isinstance(raw, dict):
        return None
    raw_d = cast(dict[str, object], raw)
    ctype = raw_d.get("type")
    if ctype not in _CALLOUT_TYPES:
        return None
    label = str(raw_d.get("label") or "")[:_MAX_LABEL].strip()
    if not label:
        return None
    return Callout(type=str(ctype), label=label)


def _parse_split(raw: object) -> dict[str, object] | None:
    """Keep a split only when it is a dict with layout in {h,v} and a non-empty
    second_clip ref; else None (authoritative-stage invariant). The ref is a
    line_id, resolved to a URL compile-side — the model names a clip, never a source."""
    if not isinstance(raw, dict):
        return None
    raw_d = cast(dict[str, object], raw)
    layout = raw_d.get("layout")
    if layout not in _SPLIT_LAYOUTS:
        return None
    second = str(raw_d.get("second_clip") or "").strip()
    if not second:
        return None
    return {"layout": str(layout), "second_clip": second}
