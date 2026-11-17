"""Static invariant: every ``stage_gather``-shaped stage's outer
``default_timeout_seconds`` must cover its inner **wave** worst case.

The handler wraps each item in ``stage_call(..., timeout=per_item_timeout,
retries=per_item_retries)`` and hands the whole list to ``stage_gather``, which
keeps at most ``concurrency`` items in flight. So the wall clock is set by WAVES:

    worst case = ceil(N / concurrency) * (1 + per_item_retries) * per_item_timeout

⚠️ **This file used to compute ``(1 + retries) * per_item`` and stop there** — the
per-ITEM figure, as if every item ran in parallel. That model blessed the exact bug
it exists to catch: ``action_reel``/``shot_analysis`` carried 1400s for a fan-out
whose real ceiling at the operator's 10-clip corpus is 1800s, and this test passed.
It is right only when ``N <= concurrency``. Do not simplify it back.

The outer stage timeout is enforced PER STAGE, not per attempt — ``dag_executor``
wraps the handler in one ``TimeoutMiddleware`` — so when it is shorter than the wave
worst case the dispatcher kills the stage mid-fan-out and the configured resilience
never runs. That is what took out the 2026-05-30 06:26 UTC website_review E2E.

The table below is **hand-maintained**. Add a new entry whenever a new
``stage_gather`` handler is added. The test then enforces the budget at import time
by walking the live blueprint registry.
"""

from __future__ import annotations

import math
from typing import NamedTuple

# Import every blueprint module so the registry is fully populated. The
# blueprints self-register via ``get_registry().register(...)`` at import.
import src.services.blueprints.action_reel
import src.services.blueprints.app_review
import src.services.blueprints.audit_family
import src.services.blueprints.course
import src.services.blueprints.generic
import src.services.blueprints.lesson
import src.services.blueprints.promo
import src.services.blueprints.website_review
from src.config import get_settings
from src.services.blueprints.registry import get_registry
from src.services.stages.action_reel.narrate import MAX_NARRATION_LINES
from src.services.stages.action_reel.shot_analysis import _TIMEOUT_SOURCE_CEILING


class GatherPrimitives(NamedTuple):
    """The four numbers a ``stage_gather`` fan-out's wall clock is made of."""

    per_item_timeout_seconds: float
    #: EFFECTIVE retries — how many extra attempts an item can really cost. Zero
    #: when the per-item callable swallows ``Exception`` itself, because then
    #: ``stage_gather``'s ``retry_on=(Exception,)`` never sees one to retry.
    per_item_retries: int
    concurrency: int
    #: The corpus size the outer timeout is BUDGETED for, or ``None`` when this
    #: stage's lane has not yet decided one. ``None`` means the wave check cannot
    #: be evaluated — see ``test_every_gather_stage_declares_a_budgeted_corpus``.
    budgeted_items: int | None


# stage name → its fan-out primitives.
# Source: ``src/services/stages/**/*.py`` constants + the ``stage_gather`` call
# kwargs. Keep in lockstep with the handlers.
GATHER_STAGES: dict[str, GatherPrimitives] = {
    # TTS narration / record
    "narrate": GatherPrimitives(600.0, 1, 4, None),
    "website_review_record": GatherPrimitives(600.0, 1, 4, None),
    "lesson_narrate": GatherPrimitives(600.0, 1, 4, None),
    "seo_audit_narrate": GatherPrimitives(600.0, 1, 4, None),
    "promo_narrate": GatherPrimitives(600.0, 1, 4, None),
    "app_review_record": GatherPrimitives(600.0, 1, 4, None),
    # AI image / video generation
    "ai_image_gen": GatherPrimitives(600.0, 1, 4, None),
    "ai_video_gen": GatherPrimitives(1200.0, 1, 4, None),
    # action_reel: one video_analyzer job per uploaded source clip. Retries are 0
    # EFFECTIVE — ``_analyze_one`` returns an error-carrying ``AnalyzedSource``
    # instead of raising, so the configured retry can never fire.
    "shot_analysis": GatherPrimitives(600.0, 0, 4, _TIMEOUT_SOURCE_CEILING),
}

# The ``narrate`` name is shared: ``action_reel``'s handler caps its authored line
# count, the generic/primitive one does not. The cap is a property of the BLUEPRINT's
# handler, so it is keyed per blueprint and overrides the table above.
PER_BLUEPRINT_BUDGETED_ITEMS: dict[tuple[str, str], int] = {
    ("action_reel_v1", "narrate"): MAX_NARRATION_LINES,
}

# Gather stages whose lane has NOT declared a corpus ceiling, so their outer budget
# cannot be proven against anything. Every one of them is an UNBOUNDED fan-out with a
# single-wave-sized budget — i.e. the same defect class this file just fixed for
# ``shot_analysis``, in other lanes' code. Deliberately recorded rather than
# "corrected" by inventing numbers here: fixing one means deciding, in that handler,
# whether its N is author-controlled (cap it, like ``action_reel``/``narrate``) or
# operator-controlled (budget for a ceiling and let it resume, like ``shot_analysis``).
UNBUDGETED_GATHER_STAGES: frozenset[str] = frozenset(
    {
        "narrate",  # generic blueprint only; action_reel's is capped
        "website_review_record",
        "lesson_narrate",
        "seo_audit_narrate",
        "promo_narrate",
        "app_review_record",
        "ai_image_gen",
        "ai_video_gen",
    }
)

# Minimum headroom over the inner worst case, in seconds. Sibling stages
# (website_critique 1320→1400, video_render 2400→2500) use ~100s.
MIN_HEADROOM_SECONDS = 100.0


def _resolved_outer(spec: object, outer: float | None) -> float:
    """``None`` => the stage inherits the env-configurable base LLM ceiling."""
    if outer is None:
        return float(get_settings().pipeline.llm_stage_base_ceiling_s)
    return float(outer)


def test_every_gather_stage_outer_timeout_covers_inner_wave_worst_case() -> None:
    """Outer ``default_timeout_seconds`` >= waves x (1+retries) x per_item + headroom.

    Walks every blueprint in the registry; for any stage name we know is a
    ``stage_gather`` handler AND for which a budgeted corpus size is declared,
    asserts the inequality. Stages outside the table (single ``stage_call`` or
    non-batched) are not checked here.
    """
    registry = get_registry()
    seen: set[tuple[str, str]] = set()
    failures: list[str] = []

    for blueprint in registry.all():
        for spec in blueprint.stages:
            primitives = GATHER_STAGES.get(spec.name)
            if primitives is None:
                continue
            seen.add((blueprint.id, spec.name))
            budgeted = PER_BLUEPRINT_BUDGETED_ITEMS.get(
                (blueprint.id, spec.name), primitives.budgeted_items
            )
            if budgeted is None:
                continue
            waves = math.ceil(budgeted / primitives.concurrency)
            worst_case = (
                waves * (1 + primitives.per_item_retries) * primitives.per_item_timeout_seconds
            )
            minimum_outer = worst_case + MIN_HEADROOM_SECONDS
            outer = _resolved_outer(spec, spec.default_timeout_seconds)
            if outer < minimum_outer:
                failures.append(
                    f"{blueprint.id}/{spec.name}: "
                    f"default_timeout_seconds={outer} < "
                    f"ceil({budgeted}/{primitives.concurrency})={waves} waves "
                    f"* (1+{primitives.per_item_retries}) "
                    f"* {primitives.per_item_timeout_seconds} "
                    f"+ {MIN_HEADROOM_SECONDS} = {minimum_outer}"
                )

    assert not failures, "stage timeout vs inner WAVE budget mismatch:\n  " + "\n  ".join(failures)
    # Guardrail: the table is not empty (catches accidental deletion).
    assert seen, "no stage_gather stages found in registry — table out of date?"


def test_every_gather_stage_declares_a_budgeted_corpus() -> None:
    """A new fan-out must DECIDE its corpus ceiling, not silently opt out.

    ``budgeted_items=None`` makes the wave check above unevaluable, so the set of
    stages allowed to be un-budgeted is pinned. Adding a ``stage_gather`` handler
    without a ceiling fails here; it does not quietly skip.
    """
    unbudgeted = {
        name for name, primitives in GATHER_STAGES.items() if primitives.budgeted_items is None
    }
    assert unbudgeted == set(UNBUDGETED_GATHER_STAGES), (
        "un-budgeted stage_gather set changed. A NEW fan-out must declare "
        "GatherPrimitives.budgeted_items (a corpus ceiling); a FIXED one must be "
        "removed from UNBUDGETED_GATHER_STAGES.\n"
        f"  now un-budgeted: {sorted(unbudgeted)}\n"
        f"  pinned:          {sorted(UNBUDGETED_GATHER_STAGES)}"
    )


def test_shot_analysis_budget_covers_every_wave_of_the_ceiling_corpus() -> None:
    """THE regression this file failed to catch.

    The old budget was a hardcoded 1400s justified by PER-SOURCE arithmetic. The
    fan-out runs in waves of 4, so the ceiling corpus needs
    ``ceil(12/4) = 3`` waves x 600s = 1800s — 400s MORE than the old literal, and
    the operator's real 10-clip corpus already needs all three waves.
    """
    from src.services.stages.action_reel.shot_analysis import (
        _CONCURRENCY,
        _PER_ITEM_TIMEOUT_SECONDS,
        SHOT_ANALYSIS_TIMEOUT_SECONDS,
    )

    spec = next(
        s
        for s in src.services.blueprints.action_reel.ACTION_REEL_BLUEPRINT.stages
        if s.name == "shot_analysis"
    )
    outer = spec.default_timeout_seconds
    assert outer is not None, "shot_analysis must pin an explicit budget, not inherit one"
    assert outer == SHOT_ANALYSIS_TIMEOUT_SECONDS, (
        "the blueprint must IMPORT the derived budget, never restate it"
    )

    for corpus in (_TIMEOUT_SOURCE_CEILING, 10):
        waves = math.ceil(corpus / _CONCURRENCY)
        assert outer >= waves * _PER_ITEM_TIMEOUT_SECONDS, (
            f"{corpus} sources need {waves} waves = {waves * _PER_ITEM_TIMEOUT_SECONDS}s > {outer}s"
        )
    # And it is strictly ABOVE the old literal, which is what the bug was.
    assert outer > 1400
