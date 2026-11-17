"""Shared spine resolver for the demo_video / showcase line spine.

The pipeline spine is the ordered ``lines`` list first emitted by
``demo_video_gen`` (each carrying its ``clip-<n>`` join id). The optional
``script_doctor``, ``claim_finish``, ``retention_plan`` and
``reconcile_timeline`` stages each re-emit the *same shape* — the first three
rewrite only weak+unlocked line TEXT (``claim_finish`` softens ungrounded
factual claims and stamps ``claim_status``/``claim_source``; a ``"pinned"`` line
is never rewritten), the last stamps per-line timing hints (``vo_rate``) against
the real recorded footage lengths (ids / order preserved by construction).
Downstream stages that need the current best line spine must therefore prefer
the latest re-emitted output — ``reconcile_timeline`` over ``retention_plan``
over ``claim_finish`` over ``script_doctor`` over raw ``demo_video_gen`` — this
module is the single place that resolution lives, so no consumer hardcodes the
precedence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext


# Highest precedence first. Every stage here emits the ``demo_video_gen`` shape
# (same line ids + order), so a consumer reads the LAST one that actually ran.
# Also consumed by ``DemoVideoNarrateStage.source_stage_chain`` — which is how
# the re-narrate patcher resolves the read stage WITHOUT a RunContext — so this
# tuple is the single source of truth for the precedence.
PLAN_STAGE_PRECEDENCE: tuple[str, ...] = (
    "reconcile_timeline",
    "retention_plan",
    "claim_finish",
    "script_doctor",
    "demo_video_gen",
)


def resolve_plan_output(ctx: RunContext) -> dict:
    """Return the current best line-spine output dict.

    ``reconcile_timeline`` output wins when that stage ran (it is the LAST plan
    mutator), else ``retention_plan``, else ``claim_finish``, else
    ``script_doctor`` (each re-emits the ``demo_video_gen`` shape with only
    text/timing revised), else the raw ``demo_video_gen`` output; otherwise
    ``{}``. A skipped stage stores a falsy ``{}`` so the chain falls through to
    the last stage that produced lines.
    """
    for stage_name in PLAN_STAGE_PRECEDENCE:
        output = ctx.prior_output(stage_name)
        if output:
            return output
    return {}


def resolve_plan_lines(ctx: RunContext) -> list[dict]:
    """Return the ordered spine ``lines`` from the current best output."""
    return list((resolve_plan_output(ctx) or {}).get("lines") or [])
