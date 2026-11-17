"""Unit tests for the color-finish spec (pure, no DB).

Covers the two guarantees this PR ships:

1. **Presence-is-opt-in** — an absent ``finish`` block yields ``None`` from
   ``build_finish_spec`` (so the compiler stays on ``-c:v copy``, byte-identical
   output); a present block always yields a spec.
2. **Clamp-on-construct** — :class:`FinishSpec` pulls every bounded knob back to
   its C8 range at construction, so an out-of-range value can never reach the
   compiler. The frozen dataclass is deterministic (same inputs → equal specs).
"""

from __future__ import annotations

import re
from dataclasses import replace

import pytest

from src.services.stages.shared.finish import FinishSpec, build_finish_spec


def test_absent_finish_block_returns_none() -> None:
    """No `finish` key ⇒ None ⇒ compiler stays on -c:v copy (byte-identical)."""
    assert build_finish_spec({}) is None
    assert build_finish_spec({"brand": {"primary_color": "#123456"}}) is None
    # A non-dict `finish` is also treated as absent (defensive, still None).
    assert build_finish_spec({"finish": "on"}) is None
    assert build_finish_spec({"finish": None}) is None


def test_present_finish_block_returns_spec() -> None:
    """Any present dict `finish` block yields a spec (even empty)."""
    spec = build_finish_spec({"finish": {}})
    assert isinstance(spec, FinishSpec)


def test_enabled_false_carries_no_grade_layers() -> None:
    """enabled=False ⇒ deterministic layers apply but grade fields stay at C8
    defaults (this PR never populates the WM colorist knobs)."""
    spec = build_finish_spec({"finish": {"enabled": False, "broadcast_safe": True}})
    assert spec is not None
    assert spec.enabled is False
    assert spec.broadcast_safe is True
    # Grade layers untouched — C8 defaults.
    assert spec.contrast == 1.0
    assert spec.saturation == 1.0
    assert spec.gamma == 1.0
    assert spec.temperature == 0.0
    assert spec.vignette == 0.0
    assert spec.grain == 0.0
    assert spec.glow == 0.0
    assert spec.sharpen == 0.0
    assert spec.grade_intensity == 1.0
    assert spec.look == "none"
    assert spec.lut_url is None


def test_deterministic_layers_lift() -> None:
    """broadcast_safe + enabled read straight off the block."""
    spec = build_finish_spec({"finish": {"enabled": True, "broadcast_safe": True}})
    assert spec is not None
    assert spec.enabled is True
    assert spec.broadcast_safe is True


def test_letterbox_color_defaults_from_brand_primary_color() -> None:
    """Absent finish.letterbox_color ⇒ default resolves from brand.primary_color."""
    spec = build_finish_spec({"finish": {}, "brand": {"primary_color": "#ff8800"}})
    assert spec is not None
    assert spec.letterbox_color == "#ff8800"


def test_letterbox_color_falls_back_to_black() -> None:
    """No brand primary colour ⇒ fallback #000000."""
    spec_no_brand = build_finish_spec({"finish": {}})
    assert spec_no_brand is not None
    assert spec_no_brand.letterbox_color == "#000000"
    # brand present but no primary_color ⇒ still fallback.
    spec = build_finish_spec({"finish": {}, "brand": {"secondary_color": "#abcdef"}})
    assert spec is not None
    assert spec.letterbox_color == "#000000"


def test_explicit_letterbox_color_overrides_brand() -> None:
    """An explicit finish.letterbox_color wins over the brand primary colour."""
    spec = build_finish_spec(
        {"finish": {"letterbox_color": "#101010"}, "brand": {"primary_color": "#ff8800"}}
    )
    assert spec is not None
    assert spec.letterbox_color == "#101010"


def test_as_finish_payload_emits_only_letterbox_color() -> None:
    """The threaded block carries EXACTLY the compiler's FinishProfile contract —
    one field. `broadcast_safe` is NOT sent (the compiler has no such field: a
    pre-encode luma clamp cannot survive libx264 ringing), and neither are the WM
    grade knobs (not renderable yet — the compiler would ignore them)."""
    spec = build_finish_spec(
        {"finish": {"enabled": True, "broadcast_safe": True}, "brand": {"primary_color": "#ff8800"}}
    )
    assert spec is not None
    payload = spec.as_finish_payload()
    assert payload == {"letterbox_color": "#ff8800"}
    assert set(payload) == {"letterbox_color"}
    # An author asking for broadcast_safe still gets it on the spec — it just
    # never ships, so no lie reaches the compiler.
    assert spec.broadcast_safe is True
    assert "broadcast_safe" not in payload


def test_as_finish_payload_defaults() -> None:
    """An empty (but present) block still threads the resolved letterbox default."""
    spec = build_finish_spec({"finish": {}})
    assert spec is not None
    assert spec.as_finish_payload() == {"letterbox_color": "#000000"}


def test_as_finish_payload_carries_no_grade_knobs() -> None:
    """Clamping still applies to the spec, and no clamped grade knob leaks into the
    payload (an out-of-range contrast is pulled back AND withheld)."""
    spec = FinishSpec(contrast=9.9, letterbox_color="#123456")
    assert spec.contrast == 1.4
    assert spec.as_finish_payload() == {"letterbox_color": "#123456"}


# ---------------------------------------------------------------------------
# ORCH-3.3 whole-film grade emission (gated) + inertness regression
# ---------------------------------------------------------------------------


def test_grade_is_emitted_only_when_enabled_and_set() -> None:
    """An enabled spec carrying a real grade threads the finishing-exclusive knobs
    (look/tints/vignette/grain/glow/sharpen/grade_intensity) alongside the letterbox.
    The tonal dials (contrast/saturation/gamma/temperature) are NEVER sent."""
    spec = FinishSpec(
        enabled=True,
        look="cinematic",
        shadow_tint="#040404",
        highlight_tint="#f0f0f0",
        vignette=0.3,
        grain=0.1,
        glow=0.08,
        sharpen=0.12,
        grade_intensity=0.9,
        letterbox_color="#123456",
    )
    payload = spec.as_finish_payload()
    assert payload["letterbox_color"] == "#123456"
    # Grade knobs ride under the nested `grade` key (compiler FinishProfile.grade:
    # GradeSpec) — NOT flat top-level, where FinishProfile's extra=ignore drops
    # them and the grade silently never applies.
    grade = payload["grade"]
    assert "look" not in payload  # NOT a top-level key — nested under `grade`
    assert "enabled" not in payload
    assert grade["enabled"] is True
    assert grade["look"] == "cinematic"
    assert grade["shadow_tint"] == "#040404"
    assert grade["highlight_tint"] == "#f0f0f0"
    assert grade["vignette"] == 0.3
    # Tonal dials must not leak — per-shot ColorAdjust owns those.
    assert "contrast" not in grade
    assert "saturation" not in grade
    assert "gamma" not in grade
    assert "temperature" not in grade


#: The field names of the compiler's ``GradeSpec`` (video_compiler_service
#: ``src/api/schemas.py`` ``class GradeSpec``). The orchestrator is a SEPARATE
#: repo from the compiler, so the model can't be imported here — this frozen set
#: is the field-for-field contract every emitted grade knob must satisfy. If the
#: compiler adds a knob, this set is the single place to extend.
_COMPILER_GRADESPEC_FIELDS = frozenset(
    {
        "enabled",
        "look",
        "contrast",
        "saturation",
        "gamma",
        "temperature",
        "shadow_tint",
        "highlight_tint",
        "vignette",
        "grain",
        "glow",
        "sharpen",
        "grade_intensity",
        "match_sources",
        "broadcast_safe",
        "lut_url",
    }
)


def test_grade_knobs_are_nested_not_flat_and_match_gradespec() -> None:
    """BUG #1 regression: the grade knobs must ride under a nested ``grade`` key,
    NOT as top-level payload keys.

    The compiler's ``FinishProfile`` has only ``letterbox_color`` + a nested
    ``grade: GradeSpec`` and ignores unknown top-level keys (``extra=ignore``), so
    a FLAT emission silently drops every knob and the whole-film grade never
    applies. This test fails if the knobs are emitted flat again (the pre-fix
    behaviour): ``payload["grade"]`` would ``KeyError`` and the flat-key
    assertions would trip.
    """
    spec = FinishSpec(
        enabled=True,
        look="cinematic",
        shadow_tint="#040404",
        highlight_tint="#f0f0f0",
        vignette=0.3,
        grain=0.1,
        glow=0.08,
        sharpen=0.12,
        grade_intensity=0.9,
        lut_url="s3://luts/kodak.cube",
        letterbox_color="#123456",
    )
    payload = spec.as_finish_payload()

    # letterbox stays top-level; grade knobs are nested (never flat).
    assert set(payload) == {"letterbox_color", "grade"}
    for knob in (
        "enabled",
        "look",
        "shadow_tint",
        "highlight_tint",
        "vignette",
        "grain",
        "glow",
        "sharpen",
        "grade_intensity",
        "lut_url",
    ):
        assert knob not in payload, f"{knob} leaked to top level — would be dropped by extra=ignore"

    grade = payload["grade"]
    # Every emitted key is a real GradeSpec field (round-trips into the compiler
    # model field-for-field, no renames).
    assert set(grade) <= _COMPILER_GRADESPEC_FIELDS
    assert grade["enabled"] is True
    assert grade["look"] == "cinematic"
    assert grade["shadow_tint"] == "#040404"
    assert grade["highlight_tint"] == "#f0f0f0"
    assert grade["vignette"] == 0.3
    assert grade["grain"] == 0.1
    assert grade["glow"] == 0.08
    assert grade["sharpen"] == 0.12
    assert grade["grade_intensity"] == 0.9
    assert grade["lut_url"] == "s3://luts/kodak.cube"


def test_active_grade_but_disabled_emits_only_letterbox() -> None:
    """`enabled=False` is the grade gate: even a fully-authored grade stays inert
    and threads ONLY the letterbox. A naive as_finish_payload that emitted the grade
    regardless of `enabled` would fail here."""
    spec = FinishSpec(enabled=False, look="cinematic", vignette=0.4, letterbox_color="#222222")
    assert spec.grade_is_set is True
    assert spec.as_finish_payload() == {"letterbox_color": "#222222"}


def test_enabled_but_identity_grade_emits_only_letterbox() -> None:
    """An enabled spec with NO look and NO film effects is not a grade — it threads
    only the letterbox (byte-identical to the deterministic-only path). A naive
    payload that emitted grade whenever `enabled` would fail here."""
    spec = FinishSpec(enabled=True, letterbox_color="#333333")
    assert spec.grade_is_set is False
    assert spec.as_finish_payload() == {"letterbox_color": "#333333"}


def test_look_none_with_effects_still_counts_as_a_grade() -> None:
    """A grade can carry film effects without a LUT look (look stays 'none')."""
    spec = FinishSpec(enabled=True, sharpen=0.2, letterbox_color="#000000")
    assert spec.grade_is_set is True
    payload = spec.as_finish_payload()
    assert payload["grade"]["sharpen"] == 0.2
    assert payload["grade"]["look"] == "none"


def test_lut_url_rides_only_when_set() -> None:
    spec = FinishSpec(enabled=True, look="cinematic", lut_url="s3://luts/kodak.cube")
    assert spec.as_finish_payload()["grade"]["lut_url"] == "s3://luts/kodak.cube"
    spec_no_lut = FinishSpec(enabled=True, look="cinematic")
    assert "lut_url" not in spec_no_lut.as_finish_payload()["grade"]


def test_build_finish_spec_runs_colorist_when_enabled_with_mood() -> None:
    """An enabled finish block with a mood makes build_finish_spec resolve the
    deterministic colorist, so the grade knobs ride the payload."""
    spec = build_finish_spec(
        {"project_id": "p-1", "finish": {"enabled": True, "mood": "cinematic"}},
        grade_enabled=True,
    )
    assert spec is not None
    assert spec.look == "cinematic"
    payload = spec.as_finish_payload()
    assert payload["grade"]["look"] == "cinematic"
    assert payload["grade"]["enabled"] is True


def test_build_finish_spec_killswitch_off_suppresses_grade() -> None:
    """The GitOps kill-switch (grade_enabled=False, the default) suppresses the
    grade for EVERY run — even a brief that authored `finish.enabled: true` with a
    mood — so the feature is fully killable without a code deploy. Only letterbox
    rides. Reverting the grade_enabled gate makes this fail (grade would leak)."""
    spec = build_finish_spec(
        {"project_id": "p-0", "finish": {"enabled": True, "mood": "cinematic"}}
    )
    assert spec is not None
    assert spec.look == "none"
    assert spec.as_finish_payload() == {"letterbox_color": "#000000"}


def test_build_finish_spec_enabled_no_mood_stays_inert() -> None:
    """Enabled (flag on) but no mood/tone/design ⇒ neutral colorist grade ⇒ only
    letterbox (byte-identical to the deterministic-only path)."""
    spec = build_finish_spec({"project_id": "p-2", "finish": {"enabled": True}}, grade_enabled=True)
    assert spec is not None
    assert spec.look == "none"
    assert spec.as_finish_payload() == {"letterbox_color": "#000000"}


def test_match_sources_rides_the_grade_payload_when_authored() -> None:
    """REGRESSION: `match_sources` was declared on FinishSpec but OMITTED from the
    grade dict, so it could never reach the compiler's GradeSpec (which fully
    implements it: renderer `if stats and grade.match_sources: _apply_match_bias`).
    Removing the payload key makes this fail with a KeyError."""
    spec = build_finish_spec(
        {
            "project_id": "p-4",
            "finish": {"enabled": True, "mood": "cinematic", "grade": {"match_sources": True}},
        },
        grade_enabled=True,
    )
    assert spec is not None
    assert spec.match_sources is True
    grade = spec.as_finish_payload()["grade"]
    assert "match_sources" in grade
    assert grade["match_sources"] is True


def test_match_sources_is_present_and_false_by_default() -> None:
    """The key must be PRESENT (the defect was omission, not a wrong value) and
    default to False, so an un-authored grade keeps the compiler's match bias off."""
    spec = build_finish_spec(
        {"project_id": "p-5", "finish": {"enabled": True, "mood": "cinematic"}},
        grade_enabled=True,
    )
    assert spec is not None
    grade = spec.as_finish_payload()["grade"]
    assert "match_sources" in grade
    assert grade["match_sources"] is False


def test_build_finish_spec_finish_grade_override_wins() -> None:
    """A brief-authored finish.grade knob beats the colorist table value."""
    spec = build_finish_spec(
        {
            "project_id": "p-3",
            "finish": {"enabled": True, "mood": "warm", "grade": {"look": "bw"}},
        },
        grade_enabled=True,
    )
    assert spec is not None
    assert spec.look == "bw"


def test_frozen_determinism_and_equality() -> None:
    """The frozen dataclass is hashable-by-value: same inputs ⇒ equal specs, and
    a re-clamped round-trip is stable."""
    a = FinishSpec(contrast=2.0, broadcast_safe=True, letterbox_color="#222222")
    b = FinishSpec(contrast=2.0, broadcast_safe=True, letterbox_color="#222222")
    assert a == b
    # Re-constructing from an already-clamped value is a fixed point.
    assert replace(a) == a
    assert FinishSpec(contrast=a.contrast) == FinishSpec(contrast=1.4)


@pytest.mark.parametrize(
    ("brand_color", "expected"),
    [("#fff", "#ffffff")],
)
def test_brand_letterbox_colour_is_normalised_to_what_the_compiler_accepts(
    brand_color: str, expected: str
) -> None:
    """A brand colour must never be able to 422 the whole compile.

    The compiler's FinishProfile pattern is a strict ``#rrggbb[aa]`` and
    CompileRequest validation is ATOMIC — one bad colour rejects the entire
    request, not just the letterbox. design_system._hex_to_rgb deliberately
    accepts 3-digit ``#rgb``, so a brand with ``primary_color: "#fff"`` is
    legitimate in this codebase and would otherwise kill the run.
    """
    spec = build_finish_spec({"finish": {}, "brand": {"primary_color": brand_color}})
    assert spec is not None
    assert spec.letterbox_color == expected
    # The value we ship must satisfy the compiler's real pattern.
    assert re.fullmatch(r"#[0-9A-Fa-f]{6}([0-9A-Fa-f]{2})?", spec.letterbox_color)
