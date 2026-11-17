"""Unit tests for the whole-film colorist (pure, no DB).

Mirrors ``test_cold_open`` / ``test_motion_plan``. Covers the ORCH-3.3 guarantees:

1. **Table determinism** — same tokens ⇒ identical :class:`Grade` across calls,
   and a mood resolves to its finishing look (character), style scales strength.
2. **Composition rule** — the colorist drives ONLY the finishing-exclusive knobs;
   it never carries the tonal dials (per-shot ``ColorAdjust`` owns those).
3. **Brief override wins** — an authored ``finish.grade`` knob beats the table
   value and is recorded ``source="override"`` in the ledger.
4. **Neutral identity** — no mood (or an unknown one) ⇒ a fully identity grade
   with no seeded jitter, so ``is_active`` is False (byte-identity backstop).
5. **Provenance** — every resolved knob lands a :class:`ProvenanceValue` with a
   valid ``source`` in the ledger.
6. **Snapshot adapter** — ``grade_from_snapshot`` pulls the same director-intent
   tokens the rest of the pipeline reads.
"""

from __future__ import annotations

from src.services.stages.shared.colorist import Grade, grade_from_snapshot, pick_grade
from src.services.stages.shared.design_system import DEFAULT_TOKENS
from src.services.stages.shared.provenance import VALID_SOURCES

# The finishing-exclusive knobs the colorist owns — and the tonal dials it must NOT.
_FINISHING_KNOBS = frozenset(
    {
        "look",
        "shadow_tint",
        "highlight_tint",
        "vignette",
        "grain",
        "glow",
        "sharpen",
        "grade_intensity",
    }
)
_TONAL_KNOBS = frozenset({"contrast", "saturation", "gamma", "temperature", "brightness"})


def _grade(**over: object) -> Grade:
    base: dict[str, object] = {
        "tone": None,
        "mood": "cinematic",
        "style": "editorial",
        "design_tokens": None,
        "brief_grade": None,
        "seed": 123,
    }
    base.update(over)
    return pick_grade(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Determinism + table character
# ---------------------------------------------------------------------------


def test_same_tokens_are_deterministic() -> None:
    """Same inputs ⇒ equal Grade (frozen dataclass equality across calls)."""
    assert _grade() == _grade()


def test_mood_selects_a_finishing_look() -> None:
    """A named mood resolves to its look; the grade is active."""
    assert _grade(mood="cinematic").look == "cinematic"
    assert _grade(mood="warm").look == "warm"
    assert _grade(mood="bw").look == "bw"
    assert _grade(mood="cinematic").is_active is True


# nostalgic → warm look


def test_style_scales_overall_strength() -> None:
    """A calmer style wears the same mood more lightly; energetic leans in."""
    calm = _grade(mood="cinematic", style="calm")
    energetic = _grade(mood="cinematic", style="energetic")
    assert calm.grade_intensity < energetic.grade_intensity
    assert calm.vignette < energetic.vignette
    # Character (look) is style-independent.
    assert calm.look == energetic.look == "cinematic"


def test_knobs_are_finishing_exclusive_only() -> None:
    """The knob map carries the finishing knobs and NONE of the tonal dials —
    per-shot ColorAdjust owns contrast/saturation/gamma; a whole-film tonal layer
    would double-apply."""
    knobs = set(_grade(mood="vivid").knobs())
    assert knobs == _FINISHING_KNOBS
    assert not (knobs & _TONAL_KNOBS)


# ---------------------------------------------------------------------------
# Brief override wins
# ---------------------------------------------------------------------------


def test_brief_override_is_recorded_as_override_in_ledger() -> None:
    """The overridden knob's ProvenanceValue is source='override'; a table knob is
    source='default'."""
    g = _grade(mood="warm", brief_grade={"glow": 0.5})
    by_source = {pv.source for pv in g.provenance}
    assert "override" in by_source
    assert "default" in by_source
    # Every knob resolved through the colorist.
    assert all(pv.computed_by == "colorist" for pv in g.provenance)


# ---------------------------------------------------------------------------
# Neutral identity — the byte-identity backstop
# ---------------------------------------------------------------------------


def test_no_mood_is_fully_identity_no_jitter() -> None:
    """No mood (and no tone) ⇒ identity grade, NOT active, with zero jitter — a
    seeded jitter here would leak grade knobs into a run that never asked."""
    g = pick_grade(
        tone=None, mood=None, style="energetic", design_tokens=None, brief_grade=None, seed=999
    )
    assert g.look == "none"
    assert (g.vignette, g.grain, g.glow, g.sharpen) == (0.0, 0.0, 0.0, 0.0)
    assert g.is_active is False
    # A different seed changes nothing (no jitter applied on the neutral path).
    other = pick_grade(
        tone=None, mood=None, style="energetic", design_tokens=None, brief_grade=None, seed=1
    )
    assert g == other


def test_unknown_mood_degrades_to_neutral() -> None:
    g = _grade(mood="chartreuse-dreams")
    assert g.look == "none"
    assert g.is_active is False


def test_seeded_jitter_differs_between_seeds_for_active_mood() -> None:
    """Two runs of the SAME active mood differ in texture (grain/glow) by seed,
    but read as the same look."""
    a = _grade(mood="nostalgic", seed=1)
    b = _grade(mood="nostalgic", seed=2)
    assert a.look == b.look
    assert (a.grain, a.glow) != (b.grain, b.glow)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_every_knob_lands_a_valid_provenance_value() -> None:
    g = _grade(mood="cinematic", brief_grade={"look": "vivid"})
    assert len(g.provenance) == len(_FINISHING_KNOBS)
    for pv in g.provenance:
        assert pv.source in VALID_SOURCES


# ---------------------------------------------------------------------------
# Brand mood uses design tokens
# ---------------------------------------------------------------------------


def test_brand_mood_tints_from_design_tokens() -> None:
    """The `brand` mood derives its tints from the run's palette; without tokens it
    degrades to the plain cinematic look (never crashes)."""
    branded = pick_grade(
        tone=None,
        mood="brand",
        style="editorial",
        design_tokens=DEFAULT_TOKENS,
        brief_grade=None,
        seed=7,
    )
    assert branded.shadow_tint == DEFAULT_TOKENS.palette.surface
    assert branded.highlight_tint == DEFAULT_TOKENS.palette.accent_soft
    # No tokens ⇒ plain cinematic tints, still active.
    untokened = pick_grade(
        tone=None,
        mood="brand",
        style="editorial",
        design_tokens=None,
        brief_grade=None,
        seed=7,
    )
    assert untokened.look == "cinematic"
    assert untokened.is_active is True


# ---------------------------------------------------------------------------
# Snapshot adapter
# ---------------------------------------------------------------------------


def test_grade_from_snapshot_reads_finish_mood_and_motion_style() -> None:
    """The config adapter pulls mood from the finish block and style from motion."""
    g = grade_from_snapshot(
        {
            "project_id": "p-1",
            "finish": {"enabled": True, "mood": "warm"},
            "motion": {"style": "energetic"},
        }
    )
    assert g.look == "warm"
    assert g.is_active is True


def test_grade_from_snapshot_neutral_when_no_tokens() -> None:
    """A finish block with no mood/tone/design ⇒ neutral identity grade."""
    g = grade_from_snapshot({"project_id": "p-2", "finish": {"enabled": True}})
    assert g.is_active is False


def test_grade_from_snapshot_is_deterministic_per_project() -> None:
    """Same snapshot ⇒ same grade (seed derives from project_id)."""
    snap = {"project_id": "p-3", "finish": {"enabled": True, "mood": "cinematic"}}
    assert grade_from_snapshot(snap) == grade_from_snapshot(snap)
