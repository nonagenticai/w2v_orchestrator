"""Unit tests for build_config_snapshot's WS-1 opt-in-stage default injection.

Hermetic like ``test_run_setup_grade``: the two DB touches in
build_config_snapshot (verbatim + storyboard) are stubbed to ``[]`` and
``get_settings`` is patched, so the headline guarantee — the seven stages that
had NEVER completed in prod now get their guard block seeded, each behind its own
kill-switch, without ever clobbering an authored value — runs without a live
postgres (it can never silently skip).

The precedence contract pinned here is: per-run ``config_overrides`` > brief-
authored key > injected default.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, patch
from uuid import uuid4, uuid7

import pytest

from src.config import PipelineSettings
from src.services.runs.run_setup import build_config_snapshot

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_VERBATIM = "src.services.runs.run_setup.VerbatimItemRepository"
_STORYBOARD = "src.services.runs.run_setup.StoryboardPanelRepository"
_SETTINGS = "src.services.runs.run_setup.get_settings"

#: Every WS-1 kill-switch, so a test can flip the whole set off in one call.
_FLAGS = (
    "ship_gate_default_enabled",
    "footage_direction_default_enabled",
    "footage_critique_default_enabled",
    "retention_default_enabled",
    "timing_default_enabled",
    "timing_relief_default_enabled",
    "voice_direction_default_enabled",
    "claim_finish_default_enabled",
    "script_doctor_default_enabled",
)


def test_flags_cover_every_stage_default_setting() -> None:
    """`_snapshot` sets these nine by explicit keyword, so nothing links the tuple
    to the real settings object. Pin it: a tenth flag added to `PipelineSettings`
    without being wired into `_snapshot` must fail here rather than silently go
    untested — that is exactly the "shipped but never exercised" failure this whole
    change exists to fix."""
    declared = {f for f in PipelineSettings.model_fields if f.endswith("_default_enabled")}
    assert declared == set(_FLAGS)


def _fake_session() -> AsyncSession:
    return cast("AsyncSession", object())


def _proj(
    *, brief: dict[str, Any], voice: dict[str, Any], type_: str = "demo_video"
) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid7(),
        title="A Demo",
        type=type_,
        brief=brief,
        brand={},
        voice=voice,
        theme={},
    )


def _deliv() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid7(), name="d", format="16x9", captions_mode="none", target_duration_seconds=90
    )


async def _build(
    *,
    brief: dict[str, Any] | None = None,
    voice: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
    enabled: bool = True,
    relief_enabled: bool = False,
    type_: str = "demo_video",
    ceilings: str = "",
    enforce: bool = False,
) -> dict[str, Any]:
    """Build a snapshot with the real ``PipelineSettings`` (so the new fields are
    genuinely typed/validated), with every WS-1 flag forced to ``enabled``.

    ``timing_relief_default_enabled`` is orthogonal (default OFF even when the other
    flags are on, because it arms a costly LLM loop) so it has its own
    ``relief_enabled`` knob rather than riding ``enabled``.

    The sibling showcase lifts are stubbed off so only WS-1 is under test.
    """
    settings = SimpleNamespace(
        pipeline=PipelineSettings(
            sound_default_on_showcase=False,
            grade_default_on_showcase=False,
            provenance_ledger_universal_enabled=False,
            # Spelled out rather than `**{flag: enabled for flag in _FLAGS}`: the
            # dict-unpack widens to every `PipelineSettings` field, so a bool could
            # bind to a `str` one as far as the type checker can tell. Explicit
            # keywords keep this genuinely typed — and `_FLAGS` is still asserted
            # against these names below, so adding a flag without touching this
            # helper fails loudly.
            ship_gate_default_enabled=enabled,
            ship_gate_default_ceilings=ceilings,
            ship_gate_default_enforce=enforce,
            footage_direction_default_enabled=enabled,
            footage_critique_default_enabled=enabled,
            retention_default_enabled=enabled,
            timing_default_enabled=enabled,
            timing_relief_default_enabled=relief_enabled,
            voice_direction_default_enabled=enabled,
            claim_finish_default_enabled=enabled,
            script_doctor_default_enabled=enabled,
        )
    )
    with (
        patch(_VERBATIM) as verbatim_cls,
        patch(_STORYBOARD) as storyboard_cls,
        patch(_SETTINGS, return_value=settings),
    ):
        verbatim_cls.return_value.list_for_project = AsyncMock(return_value=[])
        storyboard_cls.return_value.list_for_project = AsyncMock(return_value=[])
        return await build_config_snapshot(
            _fake_session(),
            proj=_proj(brief=brief or {}, voice=voice or {}, type_=type_),
            deliv=_deliv(),
            tenant_id=uuid4(),
            subject_id=uuid4(),
            config_overrides=overrides,
        )


# ── seeded when absent ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_all_seven_guard_keys_seeded_when_absent() -> None:
    """The whole point of WS-1: a brief that authors nothing still satisfies every
    one of the seven stage guards."""
    snap = await _build()
    assert snap["ship_gate"] == {"mode": "critique_only"}
    assert snap["direction"]["retention"] == {"mode": "critique_only"}
    assert snap["direction"]["timing"] == {"mode": "auto"}
    assert snap["direction"]["footage"] == {
        "enabled": True,
        "auto_zoom": True,
        "critique": True,
    }
    assert snap["voice"]["direction"] == {"tone": "confident"}
    assert snap["legal"]["mode"] == "critique_only"
    assert snap["script_doctor"] == {"enabled": True}


@pytest.mark.asyncio
async def test_timing_relief_not_seeded_by_default() -> None:
    """`timing_relief_default_enabled` is OFF by default: even with timing seeded, the
    `relief` sub-key is absent, so the WM relief loop stays inert (`relief_mode` → 'off')
    and a default render makes zero extra LLM calls."""
    snap = await _build()  # relief_enabled defaults False
    assert snap["direction"]["timing"] == {"mode": "auto"}
    assert "relief" not in snap["direction"]["timing"]


@pytest.mark.asyncio
async def test_timing_relief_seeded_when_flag_on() -> None:
    """The whole point of the flag: arming it folds a `relief` block INTO the same
    `direction.timing` seed, so `reconcile_timeline` reads `relief.mode == 'auto'` and
    runs the beat-shortening loop on an infeasible timeline."""
    snap = await _build(relief_enabled=True)
    assert snap["direction"]["timing"] == {
        "mode": "auto",
        "relief": {"mode": "auto", "iterations": 2, "score_threshold": 4.0},
    }


@pytest.mark.asyncio
async def test_timing_relief_cannot_seed_timing_when_timing_off() -> None:
    """Relief rides ON the timing seed, never on its own: with `timing_default_enabled`
    off, arming relief must NOT resurrect a `direction.timing` block (the reconcile stage
    would otherwise run unexpectedly). Proves the nesting, not a second `setdefault`."""
    snap = await _build(enabled=False, relief_enabled=True)
    assert "timing" not in snap.get("direction", {})


@pytest.mark.asyncio
async def test_relief_still_seeds_onto_an_authored_timing_block_lacking_it() -> None:
    """1c: `_setdefault_leaf` is a no-op the instant an author supplies ANY
    `direction.timing` block, so `direction.timing.relief` was silently lost on
    an authored-timing run even with the flag on and armed on a default run —
    the exact live defect. `relief` must still land, and the author's OTHER
    timing keys must survive untouched."""
    brief = {"direction": {"timing": {"mode": "off", "tolerance_pct": 4.0}}}
    snap = await _build(brief=brief, relief_enabled=True)
    assert snap["direction"]["timing"] == {
        "mode": "off",
        "tolerance_pct": 4.0,
        "relief": {"mode": "auto", "iterations": 2, "score_threshold": 4.0},
    }


@pytest.mark.asyncio
async def test_an_authored_relief_block_is_never_clobbered() -> None:
    """The author's own `relief` tuning wins over the seeded default, same
    leaf-precedence contract as every other WS-1 seed."""
    brief = {
        "direction": {"timing": {"mode": "off", "relief": {"mode": "off"}}},
    }
    snap = await _build(brief=brief, relief_enabled=True)
    assert snap["direction"]["timing"]["relief"] == {"mode": "off"}


@pytest.mark.asyncio
async def test_relief_not_seeded_onto_authored_timing_when_flag_off() -> None:
    """Same flag gate as the fresh-seed path: an authored `direction.timing`
    block gets no `relief` injected when `timing_relief_default_enabled` is off."""
    brief = {"direction": {"timing": {"mode": "off"}}}
    snap = await _build(brief=brief)  # relief_enabled defaults False
    assert snap["direction"]["timing"] == {"mode": "off"}
    assert "relief" not in snap["direction"]["timing"]


@pytest.mark.asyncio
async def test_seeded_script_doctor_block_satisfies_the_stage_guard() -> None:
    """WS-2's whole point. `ScriptDoctorStage.handle` raises `StageSkippedError`
    unless `config_snapshot["script_doctor"]` is a dict with a TRUTHY `enabled` — a
    presence check on the key alone is not enough. Registering the StageSpec without
    this seeding left the stage reachable and then permanently skipped (zero rows of
    any status in prod). Assert the guard's own expression, not just the key."""
    sd = (await _build()).get("script_doctor")
    assert isinstance(sd, dict) and sd.get("enabled")


@pytest.mark.asyncio
async def test_seeded_ship_gate_is_non_mutating_mode() -> None:
    """`critique_only` (never `auto`) — only `auto` lets final_critique re-drive an
    owning stage and change the delivered artifact."""
    snap = await _build()
    assert snap["ship_gate"]["mode"] != "auto"


@pytest.mark.asyncio
async def test_injected_retention_materializes_weights() -> None:
    """Ordering proof: the injector runs BEFORE the retention_weights resolve, so an
    injected `direction.retention` still produces the flat weights block."""
    snap = await _build()
    assert isinstance(snap["retention_weights"], dict)
    assert snap["retention_weights"]


# ── flag off ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_flags_off_inject_nothing() -> None:
    """All eight kill-switches off ⇒ byte-identical to before WS-1/WS-2."""
    snap = await _build(enabled=False)
    assert "ship_gate" not in snap
    assert "direction" not in snap
    assert "legal" not in snap
    assert "retention_weights" not in snap
    assert "script_doctor" not in snap
    assert snap["voice"] == {}


@pytest.mark.asyncio
async def test_script_doctor_flag_off_injects_nothing() -> None:
    """Individually killable: script_doctor is the one non-advisory default (an LLM
    critique/rewrite loop on every demo_video run), so its kill-switch is the one most
    likely to be flipped in anger from the configmap. Turning it off must leave the key
    ABSENT — the stage then skips clean, as it did before WS-2 — while the other seven
    keep seeding."""
    settings = SimpleNamespace(
        pipeline=PipelineSettings(
            sound_default_on_showcase=False,
            grade_default_on_showcase=False,
            provenance_ledger_universal_enabled=False,
            script_doctor_default_enabled=False,
        )
    )
    with (
        patch(_VERBATIM) as verbatim_cls,
        patch(_STORYBOARD) as storyboard_cls,
        patch(_SETTINGS, return_value=settings),
    ):
        verbatim_cls.return_value.list_for_project = AsyncMock(return_value=[])
        storyboard_cls.return_value.list_for_project = AsyncMock(return_value=[])
        snap = await build_config_snapshot(
            _fake_session(),
            proj=_proj(brief={}, voice={}),
            deliv=_deliv(),
            tenant_id=uuid4(),
            subject_id=uuid4(),
            config_overrides=None,
        )
    assert "script_doctor" not in snap
    assert snap["ship_gate"] == {"mode": "critique_only"}


@pytest.mark.asyncio
async def test_single_flag_is_individually_killable() -> None:
    """Each flag is separate: killing `timing` must not kill `retention`."""
    settings = SimpleNamespace(
        pipeline=PipelineSettings(
            sound_default_on_showcase=False,
            grade_default_on_showcase=False,
            provenance_ledger_universal_enabled=False,
            timing_default_enabled=False,
        )
    )
    with (
        patch(_VERBATIM) as verbatim_cls,
        patch(_STORYBOARD) as storyboard_cls,
        patch(_SETTINGS, return_value=settings),
    ):
        verbatim_cls.return_value.list_for_project = AsyncMock(return_value=[])
        storyboard_cls.return_value.list_for_project = AsyncMock(return_value=[])
        snap = await build_config_snapshot(
            _fake_session(),
            proj=_proj(brief={}, voice={}),
            deliv=_deliv(),
            tenant_id=uuid4(),
            subject_id=uuid4(),
            config_overrides=None,
        )
    assert "timing" not in snap["direction"]
    assert snap["direction"]["retention"] == {"mode": "critique_only"}


@pytest.mark.asyncio
async def test_footage_critique_flag_off_leaves_footage_block_bare() -> None:
    """The critique sub-flag is independently killable from the footage block."""
    settings = SimpleNamespace(
        pipeline=PipelineSettings(
            sound_default_on_showcase=False,
            grade_default_on_showcase=False,
            provenance_ledger_universal_enabled=False,
            footage_critique_default_enabled=False,
        )
    )
    with (
        patch(_VERBATIM) as verbatim_cls,
        patch(_STORYBOARD) as storyboard_cls,
        patch(_SETTINGS, return_value=settings),
    ):
        verbatim_cls.return_value.list_for_project = AsyncMock(return_value=[])
        storyboard_cls.return_value.list_for_project = AsyncMock(return_value=[])
        snap = await build_config_snapshot(
            _fake_session(),
            proj=_proj(brief={}, voice={}),
            deliv=_deliv(),
            tenant_id=uuid4(),
            subject_id=uuid4(),
            config_overrides=None,
        )
    assert snap["direction"]["footage"] == {"enabled": True, "auto_zoom": True}


# ── brief-authored wins ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_brief_authored_keys_are_never_clobbered() -> None:
    brief = {
        "ship_gate": {"mode": "auto", "max_passes": 3},
        "direction": {
            "retention": {"mode": "off"},
            "timing": {"mode": "off", "tolerance_pct": 4.0},
            "footage": {"auto_zoom": True, "critique": False},
        },
        "legal": {"mode": "auto", "posture": "strict"},
        "voice": {"direction": {"tone": "wry"}},
        "script_doctor": {"enabled": False},
    }
    snap = await _build(brief=brief)
    assert snap["script_doctor"] == {"enabled": False}
    assert snap["ship_gate"] == {"mode": "auto", "max_passes": 3}
    assert snap["direction"]["retention"] == {"mode": "off"}
    assert snap["direction"]["timing"] == {"mode": "off", "tolerance_pct": 4.0}
    assert snap["direction"]["footage"] == {"auto_zoom": True, "critique": False}
    assert snap["legal"] == {"mode": "auto", "posture": "strict"}
    assert snap["voice"]["direction"] == {"tone": "wry"}


@pytest.mark.asyncio
async def test_project_voice_direction_wins_over_default() -> None:
    """`voice` is seeded from the PROJECT row, not the brief — that path must win too."""
    snap = await _build(voice={"direction": {"tone": "somber"}})
    assert snap["voice"]["direction"] == {"tone": "somber"}


@pytest.mark.asyncio
async def test_authored_legal_without_mode_still_gets_mode_seeded() -> None:
    """`legal` already ships in prod as trademarks/defensibility for OTHER consumers;
    seeding is leaf-level so those survive and only the missing `mode` is added."""
    snap = await _build(brief={"legal": {"trademarks": ["Acme"], "defensibility": "high"}})
    assert snap["legal"] == {
        "trademarks": ["Acme"],
        "defensibility": "high",
        "mode": "critique_only",
    }


# ── per-run overrides win ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_config_overrides_win_over_default() -> None:
    overrides = {
        "ship_gate": {"mode": "off"},
        "direction": {"timing": {"mode": "off"}},
        "legal": {"mode": "auto"},
    }
    snap = await _build(overrides=overrides)
    assert snap["ship_gate"] == {"mode": "off"}
    assert snap["direction"]["timing"] == {"mode": "off"}
    assert snap["legal"]["mode"] == "auto"


@pytest.mark.asyncio
async def test_config_overrides_win_over_brief_and_default() -> None:
    """Full precedence ladder in one assertion: override > brief > injected."""
    snap = await _build(
        brief={"ship_gate": {"mode": "auto"}},
        overrides={"ship_gate": {"mode": "off"}},
    )
    assert snap["ship_gate"] == {"mode": "off"}


# ── leaf-level merge (the direction.* collision case) ────────────────────────


@pytest.mark.asyncio
async def test_leaf_merge_preserves_sibling_direction_subkeys() -> None:
    """The trap this design exists to avoid: injecting `direction.timing` into a brief
    that authored `direction.retention` must not wipe retention (nor `direction`'s own
    unrelated art-direction keys)."""
    brief = {"direction": {"retention": {"mode": "auto"}, "tone": "energetic"}}
    snap = await _build(brief=brief)
    assert snap["direction"]["retention"] == {"mode": "auto"}
    assert snap["direction"]["tone"] == "energetic"
    assert snap["direction"]["timing"] == {"mode": "auto"}
    assert snap["direction"]["footage"] == {
        "enabled": True,
        "auto_zoom": True,
        "critique": True,
    }


@pytest.mark.asyncio
async def test_authored_footage_block_only_gains_critique() -> None:
    """Nested leaf merge: an authored footage block keeps its knobs and only gains
    the missing `critique` sub-flag."""
    snap = await _build(brief={"direction": {"footage": {"auto_zoom": True}}})
    assert snap["direction"]["footage"] == {"auto_zoom": True, "critique": True}


@pytest.mark.asyncio
async def test_injection_does_not_mutate_the_source_brief() -> None:
    """Copy-on-write: the lifted `direction` block is the project brief's own dict
    (ORM-attached); seeding must replace it, never mutate it in place."""
    brief: dict[str, Any] = {"direction": {"retention": {"mode": "auto"}}}
    authored = brief["direction"]
    await _build(brief=brief)
    assert authored == {"retention": {"mode": "auto"}}


# ── direction.footage is TYPE-SCOPED (_FOOTAGE_DIRECTION_TYPES) ──────────────


@pytest.mark.parametrize("type_", ["promo"])
@pytest.mark.asyncio
async def test_footage_default_not_seeded_for_unscoped_types(type_: str) -> None:
    """Only demo_video/showcase wire the `*_footage_direction` stages. Seeding the block
    on promo/website_review silently un-gates VP-2 focus/cursor sidecar extraction plus
    track binding on the site_capture/render path those blueprints share — behaviour no
    author asked for. Catches a regression that drops the type gate."""
    snap = await _build(type_=type_)
    assert "footage" not in snap["direction"]


@pytest.mark.parametrize("type_", ["demo_video"])
@pytest.mark.asyncio
async def test_footage_default_seeded_for_scoped_types(type_: str) -> None:
    """Both members of `_FOOTAGE_DIRECTION_TYPES` still get the full block. `auto_zoom`
    must be truthy alongside `enabled`: site_capture only raises `capture_scale` to 2.0
    when it is, so seeding `enabled` alone ships punch-ins cropped from a 1x capture."""
    snap = await _build(type_=type_)
    footage = snap["direction"]["footage"]
    assert footage["enabled"]
    assert footage["auto_zoom"]
    assert footage["critique"] is True


@pytest.mark.asyncio
async def test_authored_footage_on_promo_does_not_gain_critique() -> None:
    """The one that bites hardest: `promo_footage_critique` is promo's TERMINAL stage
    with `skip_when_unrequested=False`, so injecting `critique` into a hand-authored
    promo footage block converts a documented pass-through into a multi-pass re-render
    loop (~2500s per pass). The author's block must come through untouched."""
    snap = await _build(brief={"direction": {"footage": {"auto_zoom": True}}}, type_="promo")
    assert snap["direction"]["footage"] == {"auto_zoom": True}
    assert "critique" not in snap["direction"]["footage"]


@pytest.mark.asyncio
async def test_other_six_defaults_are_not_type_scoped() -> None:
    """Guards the opposite regression: the footage gate must not be hoisted to cover its
    six siblings. A promo run still gets every other WS-1 seeding."""
    snap = await _build(type_="promo")
    assert snap["ship_gate"] == {"mode": "critique_only"}
    assert snap["legal"]["mode"] == "critique_only"
    assert snap["direction"]["retention"] == {"mode": "critique_only"}
    assert snap["direction"]["timing"] == {"mode": "auto"}
    assert snap["voice"]["direction"] == {"tone": "confident"}


# ── A11 ship-gate calibrated ceilings (PIPELINE_SHIP_GATE_DEFAULT_CEILINGS) ──────
#
# The flip from "advisory gate on code defaults" to "gate on CALIBRATED ceilings" has
# no ops lever without this: `final_critique._ceilings` reads
# `config_snapshot["ship_gate"]["ceilings"]`, which previously could only arrive from a
# brief or a per-run override. Every rejection path below must DROP and log, never
# raise — this parses on the run-creation path.


@pytest.mark.asyncio
async def test_ceilings_absent_by_default_is_byte_identical() -> None:
    """Unset env ⇒ exactly the historical block. This is the inert-by-default proof."""
    snap = await _build()
    assert snap["ship_gate"] == {"mode": "critique_only"}
    assert "ceilings" not in snap["ship_gate"]


@pytest.mark.asyncio
async def test_valid_ceilings_are_seeded_into_the_default_block() -> None:
    snap = await _build(
        ceilings=(
            '{"momentum_gap_max": 0.12, "continuity_l2": 0.34, '
            '"cohesion_floor": 0.71, "loudness_lufs": [-16.5, -12.1]}'
        )
    )
    assert snap["ship_gate"] == {
        "mode": "critique_only",
        "ceilings": {
            "momentum_gap_max": 0.12,
            "continuity_l2": 0.34,
            "cohesion_floor": 0.71,
            "loudness_lufs": [-16.5, -12.1],
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    ["not json at all"],
)
async def test_malformed_ceilings_are_dropped_not_raised(raw: str) -> None:
    """A bad configmap value must never break run creation."""
    snap = await _build(ceilings=raw)
    assert snap["ship_gate"] == {"mode": "critique_only"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    ['{"momentum_gap_max": true}'],
)
async def test_individually_invalid_ceilings_are_dropped(raw: str) -> None:
    snap = await _build(ceilings=raw)
    assert "ceilings" not in snap["ship_gate"]


@pytest.mark.asyncio
async def test_valid_entries_survive_alongside_dropped_ones() -> None:
    """A partially-bad mapping keeps the good keys rather than failing closed/open."""
    snap = await _build(ceilings='{"continuity_l2": 0.4, "typo": 1, "momentum_gap_max": true}')
    assert snap["ship_gate"]["ceilings"] == {"continuity_l2": 0.4}


@pytest.mark.asyncio
async def test_authored_ship_gate_block_wins_and_gets_no_ceilings() -> None:
    """Precedence: a per-run override owns the block wholesale. The calibration
    scenario overrides `ship_gate`, so injecting into it would perturb the very corpus
    the ceilings are calibrated from."""
    snap = await _build(
        overrides={"ship_gate": {"mode": "critique_only"}},
        ceilings='{"continuity_l2": 0.4}',
    )
    assert snap["ship_gate"] == {"mode": "critique_only"}


@pytest.mark.asyncio
async def test_kill_switch_off_seeds_no_ship_gate_even_with_ceilings() -> None:
    snap = await _build(enabled=False, ceilings='{"continuity_l2": 0.4}')
    assert "ship_gate" not in snap


# ── A11 ship-gate ENFORCE (PIPELINE_SHIP_GATE_DEFAULT_ENFORCE) ───────────────────


@pytest.mark.asyncio
async def test_enforce_off_by_default_is_byte_identical() -> None:
    """Default OFF ⇒ no `enforce` key, exactly the historical block (accept-best)."""
    snap = await _build()
    assert snap["ship_gate"] == {"mode": "critique_only"}
    assert "enforce" not in snap["ship_gate"]


@pytest.mark.asyncio
async def test_enforce_seeded_into_default_block_when_flag_on() -> None:
    """The flag folds `enforce: True` INTO the seeded block (like `ceilings`), so the
    gate's `_enforce` reads a truthy value — while a run keeping the seeded `critique_only`
    mode stays inert (only `auto` can enforce)."""
    snap = await _build(enforce=True)
    assert snap["ship_gate"] == {"mode": "critique_only", "enforce": True}


@pytest.mark.asyncio
async def test_enforce_rides_alongside_ceilings() -> None:
    """Both modifiers fold into the same seeded block."""
    snap = await _build(enforce=True, ceilings='{"continuity_l2": 0.4}')
    assert snap["ship_gate"] == {
        "mode": "critique_only",
        "ceilings": {"continuity_l2": 0.4},
        "enforce": True,
    }


@pytest.mark.asyncio
async def test_authored_ship_gate_block_ignores_enforce_default() -> None:
    """Precedence: a per-run override owns the block wholesale, so the enforce default
    never rides along (mirrors the ceilings precedence contract)."""
    snap = await _build(overrides={"ship_gate": {"mode": "auto"}}, enforce=True)
    assert snap["ship_gate"] == {"mode": "auto"}
    assert "enforce" not in snap["ship_gate"]


@pytest.mark.asyncio
async def test_seeded_ceilings_are_consumed_by_the_real_gate_seam() -> None:
    """END-TO-END SEAM: the producer (`run_setup`) and the consumers
    (`final_critique._ceilings`, `gestalt_metrics.loudness_window`) must agree.

    A snapshot key that the gate never reads is the "shipped but nobody wired the
    caller" failure — so this drives the REAL consumer functions over the REAL built
    snapshot rather than asserting on the dict shape alone.
    """
    from src.services.stages.demo_video.final_critique import FinalCritiqueStage
    from src.services.stages.shared.gestalt_metrics import loudness_window

    snap = await _build(
        ceilings=(
            '{"momentum_gap_max": 0.12, "continuity_l2": 0.34, '
            '"cohesion_floor": 0.71, "loudness_lufs": [-16.5, -12.1]}'
        )
    )
    ctx = SimpleNamespace(config_snapshot=snap)
    # Called unbound: `_ceilings` reads only `ctx`, so this pins the real consumer
    # without standing up the stage's full construction surface.
    ceilings = FinalCritiqueStage._ceilings(cast("Any", FinalCritiqueStage), cast("Any", ctx))
    assert ceilings["momentum_gap_max"] == 0.12
    assert ceilings["continuity_l2"] == 0.34
    assert ceilings["cohesion_floor"] == 0.71
    # The window key is parsed by the gate's own helper, not just carried.
    assert loudness_window(ceilings) == (-16.5, -12.1)


@pytest.mark.asyncio
async def test_unset_ceilings_leave_the_gate_on_its_code_default_window() -> None:
    """Inert proof, from the consumer's side: with nothing configured the gate's
    loudness window must differ from the calibrated one above (i.e. we really did
    change behaviour when set, and really did not when unset)."""
    from src.services.stages.shared.gestalt_metrics import loudness_window

    snap = await _build()
    default_window = loudness_window(snap["ship_gate"].get("ceilings") or {})
    assert default_window != (-16.5, -12.1)


# P3: build_config_snapshot pins provider slots; these tests run on a sentinel session.
pytestmark = pytest.mark.usefixtures("stub_provider_pins")
