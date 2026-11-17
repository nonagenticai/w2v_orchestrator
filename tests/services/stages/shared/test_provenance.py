"""Unit tests for the pure provenance resolver + deterministic seeding.

No DB — these are pure functions.
"""

from __future__ import annotations

from src.services.stages.shared.provenance import (
    ProvenanceValue,
    clamp,
    input_hash,
    master_seed,
    resolve_with_provenance,
    subseed,
)


def test_ladder_override_wins_over_smart_auto_and_default() -> None:
    pv = resolve_with_provenance("seconds", override=5, smart_auto=3, default=1)
    assert isinstance(pv, ProvenanceValue)
    assert pv.value == 5
    assert pv.source == "override"


def test_unset_sentinel_lets_none_zero_false_through() -> None:
    # An explicit None override must win the top rung (not be treated as absent).
    assert resolve_with_provenance("k", override=None, smart_auto=3, default=1).value is None
    assert resolve_with_provenance("k", override=None, smart_auto=3, default=1).source == "override"
    # 0 and False are valid smart-auto values, not "missing".
    zero = resolve_with_provenance("k", smart_auto=0, default=1)
    assert zero.value == 0
    assert zero.source == "smart-auto"
    false = resolve_with_provenance("k", smart_auto=False, default=1)
    assert false.value is False
    assert false.source == "smart-auto"


def test_clamp_sets_llm_clamped_only_when_value_changed() -> None:
    # In range → unchanged → smart-auto, no clamp_of.
    inside = clamp(5, 0, 10, computed_by="planner")
    assert inside.value == 5
    assert inside.source == "smart-auto"
    assert inside.clamp_of is None
    # Above the high bound → clamped → llm-clamped, records the original.
    high = clamp(42, 0, 10, computed_by="planner")
    assert high.value == 10
    assert high.source == "llm-clamped"
    assert high.clamp_of == 42
    # Below the low bound → clamped → llm-clamped.
    low = clamp(-3, 0, 10, computed_by="planner")
    assert low.value == 0
    assert low.source == "llm-clamped"
    assert low.clamp_of == -3


def test_input_hash_stable_and_key_order_insensitive() -> None:
    a = input_hash({"a": 1, "b": 2}, "x")
    b = input_hash({"b": 2, "a": 1}, "x")
    assert a == b  # dict key order does not change the hash
    assert a == input_hash({"a": 1, "b": 2}, "x")  # stable/repeatable
    assert isinstance(a, str) and len(a) == 64
    # Positional order IS significant.
    assert input_hash("x", "y") != input_hash("y", "x")


def test_subseed_deterministic_and_in_range() -> None:
    s1 = subseed(12345, "motion", "clip-3")
    s2 = subseed(12345, "motion", "clip-3")
    assert s1 == s2  # deterministic
    assert 0 <= s1 < 2**32  # in range
    # Different spine id → (almost surely) different subseed.
    assert subseed(12345, "motion", "clip-3") != subseed(12345, "motion", "clip-4")
    # Different namespace → different subseed.
    assert subseed(12345, "motion", "clip-3") != subseed(12345, "audio", "clip-3")


def test_master_seed_deterministic_and_in_range() -> None:
    snap = {"project_id": "proj-abc", "title": "T"}
    m1 = master_seed(snap)
    m2 = master_seed({"project_id": "proj-abc", "title": "different"})
    assert m1 == m2  # keyed on project_id, not title
    assert 0 <= m1 < 2**31
    # A different project id yields a different seed.
    assert master_seed({"project_id": "proj-xyz"}) != m1
