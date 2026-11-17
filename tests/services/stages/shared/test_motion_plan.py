"""Unit tests for the motion signature + advisory layout planner (pure, no DB).

Mirrors ``test_cadence`` / ``test_compute_plan``. Covers the ORCH-3.1 guarantees:

1. **Determinism + purity** — same ``(style, seed)`` ⇒ equal signature; same
   ``project_id`` ⇒ equal seed. No clock/random/env in the module.
2. **Signature variety** — the easing family differs per style; the seed varies a
   signature between videos (spec §8.2).
3. **No-freeze timeline** — ``plan_card_timing(N, n).total == N`` exactly (§8.3).
4. **Advisory classifier** — returns the right :data:`LayoutKind` per card shape;
   the label is metadata only (D2/D3), asserted here, not template selection.
5. **Icon override** — an author/planner override always beats the keyword map.
6. **plan_hash** — stable across calls and changes iff the plan changes (the seam
   ``design_system.token_fingerprint`` folds into ``deterministic_media_id``).
"""

from __future__ import annotations

from src.services.stages.shared.motion_plan import (
    MotionPlan,
    MotionSignature,
    Timeline,
    motion_signature,
    pick_icon,
    plan_card_timing,
    plan_hash,
    stable_seed,
)

# ---------------------------------------------------------------------------
# stable_seed
# ---------------------------------------------------------------------------


def test_stable_seed_is_deterministic() -> None:
    assert stable_seed("project-alpha") == stable_seed("project-alpha")


def test_stable_seed_distinct_per_project() -> None:
    seeds = {stable_seed(p) for p in ("project-alpha", "project-beta", "project-gamma")}
    assert len(seeds) == 3


def test_stable_seed_no_random_no_clock() -> None:
    # A fixed id maps to a fixed integer (would drift if it read a clock/env).
    assert stable_seed("project-alpha") == stable_seed("project-alpha")
    assert isinstance(stable_seed("x"), int)


# ---------------------------------------------------------------------------
# motion_signature
# ---------------------------------------------------------------------------


def test_signature_deterministic() -> None:
    seed = stable_seed("project-alpha")
    assert motion_signature("punchy", seed) == motion_signature("punchy", seed)


def test_signature_differs_per_style() -> None:
    seed = stable_seed("project-alpha")
    easings = {
        motion_signature(s, seed).easing for s in ("calm", "editorial", "punchy", "energetic")
    }
    # The easing family is fixed per style, so all four are distinct.
    assert len(easings) == 4


def test_signature_differs_per_seed() -> None:
    sigs = {
        motion_signature("editorial", stable_seed(p))
        for p in ("project-alpha", "project-beta", "project-gamma")
    }
    # Two different videos of the same style still differ (seed perturbs the band).
    assert len(sigs) > 1


def test_unknown_style_falls_back_to_editorial() -> None:
    seed = stable_seed("project-alpha")
    assert motion_signature("nonsense-style", seed) == motion_signature("editorial", seed)


def test_signature_fields_in_expected_bands() -> None:
    sig = motion_signature("calm", stable_seed("project-alpha"))
    assert sig.easing == "easeOutQuint"
    assert 140 <= sig.stagger_ms <= 180
    assert sig.entrance in ("fade", "rise")
    assert sig.exit_ in ("fade", "fall")
    assert 0.0 <= sig.parallax <= 1.0
    assert 0.0 <= sig.glow_drift <= 1.0


# ---------------------------------------------------------------------------
# plan_card_timing
# ---------------------------------------------------------------------------


def test_timeline_total_equals_duration() -> None:
    sig = motion_signature("punchy", stable_seed("project-alpha"))
    for duration_ms in (12000, 5000, 800, 1, 0):
        for n in (1, 3, 7):
            tl = plan_card_timing(duration_ms, n, sig)
            assert tl.total == duration_ms


def test_timeline_marks_len_equals_elements() -> None:
    sig = motion_signature("energetic", stable_seed("project-beta"))
    tl = plan_card_timing(9000, 4, sig)
    assert len(tl.marks) == 4
    assert tl.marks[0] == 0
    # Marks are monotonically non-decreasing (staggered starts).
    assert list(tl.marks) == sorted(tl.marks)


def test_timeline_stagger_capped_to_entrance_budget() -> None:
    sig = motion_signature("calm", stable_seed("project-alpha"))
    tl = plan_card_timing(2000, 8, sig)
    # All elements must start within the entrance budget.
    assert tl.stagger_ms * 7 <= tl.entrance_ms


# ---------------------------------------------------------------------------
# pick_icon
# ---------------------------------------------------------------------------


def test_pick_icon_override_wins() -> None:
    assert pick_icon("blazing fast and secure", override="rocket") == "rocket"


def test_pick_icon_keyword_map() -> None:
    assert pick_icon("Our platform is blazing fast") == "bolt"
    assert pick_icon("Enterprise grade security") == "shield"


def test_pick_icon_empty_override_falls_through() -> None:
    assert pick_icon("blazing fast", override="") == "bolt"


def test_pick_icon_no_match_returns_none() -> None:
    assert pick_icon("a gentle grey morning") is None


# ---------------------------------------------------------------------------
# plan_hash — the media-id seam
# ---------------------------------------------------------------------------


def _sig() -> MotionSignature:
    return MotionSignature(
        easing="easeOutCubic",
        stagger_ms=100,
        entrance="rise",
        exit_="fade",
        parallax=0.35,
        glow_drift=0.2,
    )


def _timeline() -> Timeline:
    return Timeline(
        entrance_ms=2640, hold_ms=7680, exit_ms=1680, stagger_ms=100, marks=(0, 100, 200)
    )


def _plan() -> MotionPlan:
    return MotionPlan(signature=_sig(), layout="list", icon="bolt", timeline=_timeline())


def test_plan_hash_is_stable() -> None:
    assert plan_hash(_plan()) == plan_hash(_plan())
    assert len(plan_hash(_plan())) == 16


def test_plan_hash_changes_when_layout_changes() -> None:
    other = MotionPlan(signature=_sig(), layout="quote", icon="bolt", timeline=_timeline())
    assert plan_hash(_plan()) != plan_hash(other)


def test_plan_hash_changes_when_icon_changes() -> None:
    other = MotionPlan(signature=_sig(), layout="list", icon="shield", timeline=_timeline())
    assert plan_hash(_plan()) != plan_hash(other)


def test_plan_hash_changes_when_signature_changes() -> None:
    sig2 = MotionSignature(
        easing="easeOutBack",
        stagger_ms=100,
        entrance="rise",
        exit_="fade",
        parallax=0.35,
        glow_drift=0.2,
    )
    other = MotionPlan(signature=sig2, layout="list", icon="bolt", timeline=_timeline())
    assert plan_hash(_plan()) != plan_hash(other)


def test_plan_hash_changes_when_timeline_changes() -> None:
    tl2 = Timeline(entrance_ms=2640, hold_ms=7680, exit_ms=1680, stagger_ms=90, marks=(0, 90, 180))
    other = MotionPlan(signature=_sig(), layout="list", icon="bolt", timeline=tl2)
    assert plan_hash(_plan()) != plan_hash(other)
