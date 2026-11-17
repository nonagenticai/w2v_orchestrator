"""promo_footage_critique — the L4 footage-quality loop for the promo pipeline.

A thin subclass of the shared :class:`FootageCritiqueStage`. It scores the video
``promo_render`` already compiled, nudges the ``promo_footage_direction`` plan for
the shots that scored badly, and re-renders through the CANONICAL compile path
(``PromoRenderStage.handle`` over a context whose footage-direction output is
swapped for the nudged plan) — never duplicating compile logic. Bounded to ≤2
passes, it keeps the best-scoring artifact and NEVER fails the deliverable.

Wiring (mirrors ``promo/footage_direction.py``):
  * starting artifact ← ``promo_render`` output (artifact_url + duration).
  * plans ← ``promo_footage_direction`` output (the L1 INTENT plan).
  * shot windows ← the ordered ``promo_storyboard`` shot spine with real per-line
    narration durations (``promo_narrate`` clips), cumulative.
  * re-render ← ``PromoRenderStage`` with the nudged plan injected.

``skip_when_unrequested`` is False: promo makes this stage its
``terminal_artifact_stage`` and has no ``render_manifest`` reader, so an absent
``direction.footage.critique`` block PASSES THROUGH the current promo_render
artifact unchanged rather than skipping clean.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar, cast

from src.services.stages.base import SLOT_USE_OPTIONAL, SlotUse, StagePreconditionError
from src.services.stages.promo.render import PromoRenderInput, PromoRenderStage
from src.services.stages.shared.footage_critique import (
    FootageCritiqueStage,
    ShotWindow,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.services.runs.run_context import RunContext


def _safe_seconds(value: object) -> float:
    """Coerce a duration field to float; junk (non-numeric string, list, None) ⇒
    ``0.0``. The soft-gate critique must never raise on a malformed stage output."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return 0.0
    try:
        f = float(value)
    except TypeError, ValueError:
        return 0.0
    return f if f == f and f >= 0.0 else 0.0  # NaN / negative ⇒ 0.0


class PromoFootageCritiqueStage(FootageCritiqueStage):
    name = "promo_footage_critique"
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (
        SlotUse("llm"),
        SlotUse("asr", when=SLOT_USE_OPTIONAL),
    )
    worker_name = "footage-critic"  # base default; kept explicit for clarity
    # Promo is the terminal_artifact_stage with no render_manifest reader, so an
    # absent critique block must PASS THE CURRENT ARTIFACT THROUGH, not skip clean.
    skip_when_unrequested = False

    def starting_artifact(self, ctx: RunContext) -> tuple[str, float]:
        out = ctx.prior_output("promo_render") or {}
        url = out.get("artifact_url")
        if not url:
            # promo_render is a hard dependency that always produces a validated
            # artifact_url; a missing one means the render genuinely produced
            # nothing, so there is no deliverable to critique.
            raise StagePreconditionError(
                "promo_footage_critique: promo_render artifact_url missing"
            )
        return str(url), _safe_seconds(out.get("duration_seconds"))

    def base_plans(self, ctx: RunContext) -> list[Mapping[str, object]]:
        direction = ctx.prior_output("promo_footage_direction") or {}
        plans = direction.get("plans")
        return cast("list[Mapping[str, object]]", list(plans)) if isinstance(plans, list) else []

    def shot_windows(self, ctx: RunContext) -> list[ShotWindow]:
        # The ordered storyboard spine (one shot per script line, in play order) and
        # the REAL narrated per-line durations define the play timeline. A running
        # sum gives each line its half-open [t0, t1) window. See the base module
        # docstring on absolute-time imprecision (title cards / cold open shift the
        # axis). Defensive throughout: a malformed shot / narrate clip is SKIPPED,
        # never raised — this runs before the loop's soft-degrade guard, so an
        # unwrapped KeyError/TypeError here would hard-fail the deliverable, which
        # the soft gate forbids. Worst case ⇒ empty windows ⇒ handle accepts-current.
        # KEY ON line_id (never shot["id"]) — the plan rows are keyed by line_id.
        storyboard = ctx.prior_output("promo_storyboard") or {}
        narrate = ctx.prior_output("promo_narrate") or {}
        seconds_by_line: dict[str, float] = {}
        for c in narrate.get("clips") or []:
            if not isinstance(c, Mapping):
                continue
            clip_line = c.get("line_id")
            if clip_line:
                seconds_by_line[str(clip_line)] = _safe_seconds(c.get("seconds"))
        windows: list[ShotWindow] = []
        t = 0.0
        for shot in storyboard.get("shots") or []:
            if not isinstance(shot, Mapping) or not shot.get("line_id"):
                continue
            line_id = str(shot["line_id"])
            dur = seconds_by_line.get(line_id, _safe_seconds(shot.get("seconds")))
            end = t + dur if dur > 0 else t
            windows.append({"shot_id": line_id, "t0": t, "t1": end})
            t = end
        return windows

    async def rerender(
        self, ctx: RunContext, plans: Sequence[Mapping[str, object]], pass_idx: int
    ) -> tuple[str, float]:
        # Re-render by swapping ONLY the footage-direction output for the nudged plan
        # on a shallow-copied context, then driving the canonical promo compile stage
        # — so captions, sound design, motion overlays and every other compile-tail
        # concern ride identically to the original render. The compiler's server-side
        # per-shot cache makes an unchanged shot a no-op re-encode; this code does not
        # depend on that cache existing, it just calls compile again.
        new_ctx = replace(
            ctx,
            prior_run_outputs={
                **ctx.prior_run_outputs,
                "promo_footage_direction": {"plans": list(plans)},
            },
            # Distinct from the original render (empty discriminator) AND resume-stable:
            # the pass ordinal is loop-position-derived, NOT plan-hash-derived, so a
            # resumed re-render dedupes to the first paid render (first-nudge-wins) while
            # a genuinely-new pass still gets its own key.
            render_idempotency_discriminator=f"critique:pass{pass_idx}",
        )
        out = await PromoRenderStage(client=self._compile_client).handle(
            PromoRenderInput(), new_ctx
        )
        return str(out.artifact_url), float(out.duration_seconds)
