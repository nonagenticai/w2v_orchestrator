"""demo_video_footage_critique — the L4 footage-quality loop for demo_video/showcase.

A thin subclass of the shared :class:`FootageCritiqueStage`. It scores the video
``demo_video_compile`` already rendered, nudges the ``demo_video_footage_direction``
plan for the shots that scored badly, and re-renders through the CANONICAL compile
path (``DemoVideoCompileStage.handle`` over a context whose footage-direction output
is swapped for the nudged plan) — never duplicating compile logic. Bounded to ≤2
passes, it keeps the best-scoring artifact and NEVER fails the deliverable.

Wiring (mirrors ``demo_video/footage_direction.py``):
  * starting artifact ← ``demo_video_compile`` output (artifact_url + duration).
  * plans ← ``demo_video_footage_direction`` output (the L1 INTENT plan).
  * shot windows ← the resolved line spine (``resolve_plan_output``) with real
    per-line narration durations (``demo_video_narrate`` clips), cumulative.
  * re-render ← ``DemoVideoCompileStage`` with the nudged plan injected.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar, cast

from src.services.stages.base import SLOT_USE_OPTIONAL, SlotUse, StagePreconditionError
from src.services.stages.demo_video.compile import DemoVideoCompileInput, DemoVideoCompileStage
from src.services.stages.demo_video.spine import resolve_plan_output
from src.services.stages.shared.footage_critique import (
    FootageCritiqueStage,
    ShotWindow,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.services.runs.run_context import RunContext


#: The ``ClipIdleTrim`` fields the critic is shown — the measurement, not the
#: bookkeeping (``detail``/``floor_seconds`` stay on the record stage's output).
_IDLE_KEYS_FOR_CRITIC = (
    "status",
    "idle_fraction_before",
    "idle_fraction_after",
    "frozen_seconds_before",
    "frozen_seconds_after",
    "removed_seconds",
    "spans_shortened",
)


class DemoVideoFootageCritiqueStage(FootageCritiqueStage):
    name = "demo_video_footage_critique"
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (
        SlotUse("llm"),
        SlotUse("tts"),
        SlotUse("asr", when=SLOT_USE_OPTIONAL),
    )
    worker_name = "footage-critic"

    def starting_artifact(self, ctx: RunContext) -> tuple[str, float]:
        compile_out = ctx.prior_output("demo_video_compile") or {}
        url = compile_out.get("artifact_url")
        if not url:
            # demo_video_compile is a hard dependency that always produces a
            # validated artifact_url; a missing one means the render genuinely
            # produced nothing, so there is no deliverable to critique.
            raise StagePreconditionError(
                "demo_video_footage_critique: demo_video_compile artifact_url missing"
            )
        return str(url), float(compile_out.get("duration_seconds") or 0.0)

    def base_plans(self, ctx: RunContext) -> list[Mapping[str, object]]:
        direction = ctx.prior_output("demo_video_footage_direction") or {}
        plans = direction.get("plans")
        return cast("list[Mapping[str, object]]", list(plans)) if isinstance(plans, list) else []

    def shot_windows(self, ctx: RunContext) -> list[ShotWindow]:
        # The ordered spine (doctored/reconciled when those stages ran, else gen) and
        # the REAL narrated per-line durations define the play timeline. A running sum
        # gives each line its half-open [t0, t1) window. See the base module docstring
        # on absolute-time imprecision (title cards / cold open shift the axis).
        # Defensive throughout: a malformed spine line / narrate clip is SKIPPED,
        # never raised — this runs before the loop's soft-degrade guard, so an
        # unwrapped KeyError/TypeError here would hard-fail the deliverable, which the
        # soft gate forbids. Worst case ⇒ empty windows ⇒ handle accepts-current.
        gen = resolve_plan_output(ctx)  # always a dict (possibly empty)
        lines = gen.get("lines") or []
        narrate = ctx.prior_output("demo_video_narrate") or {}
        seconds_by_line: dict[str, float] = {}
        for c in narrate.get("clips") or []:
            if not isinstance(c, Mapping):
                continue
            clip_line = c.get("line_id")
            if clip_line:
                seconds_by_line[str(clip_line)] = float(c.get("seconds") or 0.0)
        windows: list[ShotWindow] = []
        t = 0.0
        for line in lines:
            if not isinstance(line, Mapping) or not line.get("id"):
                continue
            line_id = str(line["id"])
            dur = seconds_by_line.get(line_id, float(line.get("seconds") or 0.0))
            end = t + dur if dur > 0 else t
            windows.append({"shot_id": line_id, "t0": t, "t1": end})
            t = end
        return windows

    def idle_trim_by_shot(self, ctx: RunContext) -> dict[str, dict[str, object]]:
        # Each spine line plays over its ``clip`` (else the clip with its own id) — the
        # same join compile uses — so every line inherits its recording's idle numbers.
        record = ctx.prior_output("demo_video_record") or {}
        idle_by_clip: dict[str, dict[str, object]] = {}
        for clip in record.get("clips") or []:
            if not isinstance(clip, Mapping) or clip.get("id") is None:
                continue
            idle = clip.get("idle_trim")
            if isinstance(idle, Mapping) and idle:
                idle_by_clip[str(clip["id"])] = {
                    key: idle[key] for key in _IDLE_KEYS_FOR_CRITIC if key in idle
                }
        if not idle_by_clip:
            return {}
        out: dict[str, dict[str, object]] = {}
        for line in resolve_plan_output(ctx).get("lines") or []:
            if not isinstance(line, Mapping) or not line.get("id"):
                continue
            idle = idle_by_clip.get(str(line.get("clip") or line["id"]))
            if idle:
                out[str(line["id"])] = idle
        return out

    async def rerender(
        self, ctx: RunContext, plans: Sequence[Mapping[str, object]], pass_idx: int
    ) -> tuple[str, float]:
        # Re-render by swapping ONLY the footage-direction output for the nudged plan
        # on a shallow-copied context, then driving the canonical compile stage — so
        # title cards, mastering, captions, cold-open and every other compile-tail
        # concern ride identically to the original render. The compiler's server-side
        # per-shot cache makes an unchanged shot a no-op re-encode; this code does not
        # depend on that cache existing, it just calls compile again.
        new_ctx = replace(
            ctx,
            prior_run_outputs={
                **ctx.prior_run_outputs,
                "demo_video_footage_direction": {"plans": list(plans)},
            },
            # Distinct from the original render (empty discriminator) AND resume-stable:
            # the pass ordinal is loop-position-derived, NOT plan-hash-derived, so a
            # resumed re-render dedupes to the first paid render (first-nudge-wins) while
            # a genuinely-new pass still gets its own key.
            render_idempotency_discriminator=f"critique:pass{pass_idx}",
        )
        out = await DemoVideoCompileStage(client=self._compile_client).handle(
            DemoVideoCompileInput(), new_ctx
        )
        return str(out.artifact_url), float(out.duration_seconds)
