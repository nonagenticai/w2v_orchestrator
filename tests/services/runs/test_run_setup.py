"""Unit tests for build_config_snapshot's default-injection rules.

Tested directly (not via start_run) so we can exercise the lesson default
without the lesson-needs-a-parent-course DB constraint — proj/deliv are simple
stubs; only the verbatim lookup touches the real session (returns empty).
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4, uuid7

import pytest

from src.config import get_settings
from src.services.runs.run_setup import (
    build_config_snapshot,
)


@pytest.fixture
def _claim_finish_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable the WS-1 `legal.mode` seed for the tests that assert the pre-WS-1 lift
    contract verbatim (the seed's own coverage is test_run_setup_stage_defaults.py).
    `get_settings` is lru_cached, so this patches the object run_setup reads."""
    monkeypatch.setattr(get_settings().pipeline, "claim_finish_default_enabled", False)


def _proj(
    *,
    type_: str,
    brief: dict | None = None,
    brand: dict | None = None,
    theme: dict | None = None,
    voice: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid7(),
        title="A Lesson",
        type=type_,
        brief=brief or {},
        brand=brand or {},
        voice=voice or {},
        theme=theme or {},
    )


def _deliv() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid7(),
        name="d",
        format="16x9",
        captions_mode="none",
        target_duration_seconds=None,
    )


@pytest.mark.asyncio
async def test_lesson_defaults_ai_image_gen_to_slide_prompts(db_session) -> None:
    snap = await build_config_snapshot(
        db_session,
        proj=_proj(type_="lesson"),
        deliv=_deliv(),
        tenant_id=uuid4(),
        subject_id=uuid4(),
        config_overrides=None,
    )
    cfg = snap["ai_image_gen"]
    assert cfg["prompts_from_stage"] == "lesson_slides"
    # a negative_prompt guards against the model rendering text into images
    assert "text" in cfg["negative_prompt"]


@pytest.mark.asyncio
async def test_generic_defaults_ai_image_gen_to_storyboard_prompts(db_session) -> None:
    """Live run 01a156a4-e906 (2026-10-20): with no ai_image_gen block the stage
    skipped and every shot of a generic film was a "Visual for:" text card."""
    snap = await build_config_snapshot(
        db_session,
        proj=_proj(type_="generic"),
        deliv=_deliv(),
        tenant_id=uuid4(),
        subject_id=uuid4(),
        config_overrides=None,
    )
    cfg = snap["ai_image_gen"]
    assert cfg["prompts_from_stage"] == "storyboard_draft"
    assert "text" in cfg["negative_prompt"]


@pytest.mark.asyncio
async def test_generic_brief_can_opt_out_of_default_ai_image_gen(db_session) -> None:
    snap = await build_config_snapshot(
        db_session,
        proj=_proj(type_="generic", brief={"ai_image_gen": False}),
        deliv=_deliv(),
        tenant_id=uuid4(),
        subject_id=uuid4(),
        config_overrides=None,
    )
    assert "ai_image_gen" not in snap


@pytest.mark.asyncio
async def test_lesson_default_negative_prompt_unchanged(db_session) -> None:
    snap = await build_config_snapshot(
        db_session,
        proj=_proj(type_="lesson"),
        deliv=_deliv(),
        tenant_id=uuid4(),
        subject_id=uuid4(),
        config_overrides=None,
    )
    assert snap["ai_image_gen"]["negative_prompt"] == (
        "text, words, letters, typography, captions, signage, watermark, "
        "logo, ui, interface, numbers, people, faces"
    )


@pytest.mark.asyncio
async def test_non_lesson_does_not_get_default_ai_image_gen(db_session) -> None:
    snap = await build_config_snapshot(
        db_session,
        proj=_proj(type_="promo"),
        deliv=_deliv(),
        tenant_id=uuid4(),
        subject_id=uuid4(),
        config_overrides=None,
    )
    assert "ai_image_gen" not in snap


@pytest.mark.asyncio
async def test_lesson_brief_ai_image_gen_wins_over_default(db_session) -> None:
    custom = {"provider": "local_image", "size": "1024x1024", "prompts": ["x"]}
    snap = await build_config_snapshot(
        db_session,
        proj=_proj(type_="lesson", brief={"ai_image_gen": custom}),
        deliv=_deliv(),
        tenant_id=uuid4(),
        subject_id=uuid4(),
        config_overrides=None,
    )
    assert snap["ai_image_gen"] == custom


@pytest.mark.asyncio
async def test_brief_motion_graphics_lifts_into_snapshot(db_session) -> None:
    """A truthy brief.motion_graphics block lifts to the top-level key the
    motion_graphics_gen stage reads (mirrors the ai_image_gen/ai_video_gen lift).
    Without this the FE toggle is silently dropped and the stage stays skipped."""
    cfg = {"enabled": True, "template_id": "pipeline_story"}
    snap = await build_config_snapshot(
        db_session,
        proj=_proj(type_="promo", brief={"motion_graphics": cfg}),
        deliv=_deliv(),
        tenant_id=uuid4(),
        subject_id=uuid4(),
        config_overrides=None,
    )
    assert snap["motion_graphics"] == cfg


@pytest.mark.asyncio
async def test_absent_motion_graphics_not_lifted(db_session) -> None:
    snap = await build_config_snapshot(
        db_session,
        proj=_proj(type_="promo", brief={}),
        deliv=_deliv(),
        tenant_id=uuid4(),
        subject_id=uuid4(),
        config_overrides=None,
    )
    assert "motion_graphics" not in snap


@pytest.mark.asyncio
async def test_resolution_defaults_to_2k_when_absent(db_session) -> None:
    """No brief/override → the snapshot carries the default tier so the whole
    render chain runs at 2K, never a silent 1080p."""
    snap = await build_config_snapshot(
        db_session,
        proj=_proj(type_="promo", brief={}),
        deliv=_deliv(),
        tenant_id=uuid4(),
        subject_id=uuid4(),
        config_overrides=None,
    )
    assert snap["resolution"] == "2k"


@pytest.mark.asyncio
async def test_resolution_lifts_from_brief(db_session) -> None:
    snap = await build_config_snapshot(
        db_session,
        proj=_proj(type_="promo", brief={"resolution": "4k"}),
        deliv=_deliv(),
        tenant_id=uuid4(),
        subject_id=uuid4(),
        config_overrides=None,
    )
    assert snap["resolution"] == "4k"


@pytest.mark.asyncio
async def test_resolution_override_wins_and_normalizes(db_session) -> None:
    """A per-run override beats the brief; an unknown value normalizes to the
    default rather than reaching the renderers as garbage."""
    snap = await build_config_snapshot(
        db_session,
        proj=_proj(type_="promo", brief={"resolution": "4k"}),
        deliv=_deliv(),
        tenant_id=uuid4(),
        subject_id=uuid4(),
        config_overrides={"resolution": "1080p"},
    )
    assert snap["resolution"] == "1080p"

    snap_bad = await build_config_snapshot(
        db_session,
        proj=_proj(type_="promo", brief={}),
        deliv=_deliv(),
        tenant_id=uuid4(),
        subject_id=uuid4(),
        config_overrides={"resolution": "ultra"},
    )
    assert snap_bad["resolution"] == "2k"


@pytest.mark.asyncio
async def test_brief_title_card_lifts_into_snapshot(db_session) -> None:
    """A truthy brief.title_card block lifts to the top-level key the compile-tail
    intro/outro injection reads (same opt-in lift as motion_graphics). Without it
    the FE toggle is silently dropped and no card is ever rendered."""
    cfg = {"enabled": True, "intro": True, "outro": True}
    snap = await build_config_snapshot(
        db_session,
        proj=_proj(type_="promo", brief={"title_card": cfg}),
        deliv=_deliv(),
        tenant_id=uuid4(),
        subject_id=uuid4(),
        config_overrides=None,
    )
    assert snap["title_card"] == cfg


@pytest.mark.asyncio
async def test_absent_title_card_not_lifted(db_session) -> None:
    snap = await build_config_snapshot(
        db_session,
        proj=_proj(type_="promo", brief={}),
        deliv=_deliv(),
        tenant_id=uuid4(),
        subject_id=uuid4(),
        config_overrides=None,
    )
    assert "title_card" not in snap


@pytest.mark.asyncio
async def test_brief_script_doctor_lifts_into_snapshot(db_session) -> None:
    """A truthy brief.script_doctor block lifts to the top-level key the script-doctor
    stage reads (ORCH-1.2, same presence-opt-in lift as motion_graphics/title_card).
    Without this the FE toggle is silently dropped and the stage stays skipped."""
    cfg = {"enabled": True, "passes": 2, "intent": "tighten"}
    snap = await build_config_snapshot(
        db_session,
        proj=_proj(type_="promo", brief={"script_doctor": cfg}),
        deliv=_deliv(),
        tenant_id=uuid4(),
        subject_id=uuid4(),
        config_overrides=None,
    )
    assert snap["script_doctor"] == cfg


@pytest.mark.asyncio
async def test_absent_script_doctor_is_seeded_enabled_by_default(db_session) -> None:
    """WS-2 deliberately ENDS the presence-opt-in contract this test used to pin.

    script_doctor was presence-opt-in and nothing in the product ever authored the
    block, so the stage — once registered in the demo_video blueprint — was reachable
    and then permanently skipped (zero rows of any status in prod). A brief that
    authors nothing now gets `{"enabled": True}` seeded, behind
    `PIPELINE_SCRIPT_DOCTOR_DEFAULT_ENABLED`. The seeded block must satisfy the
    handler's guard, which is a truthiness check on `.enabled` — not on the key.
    Kill-switch + precedence are pinned in test_run_setup_stage_defaults.py."""
    tenant, subject = uuid4(), uuid4()
    proj = _proj(type_="promo", brief={"motion_graphics": {"enabled": True}})
    snap = await build_config_snapshot(
        db_session,
        proj=proj,
        deliv=_deliv(),
        tenant_id=tenant,
        subject_id=subject,
        config_overrides=None,
    )
    assert snap["script_doctor"] == {"enabled": True}


@pytest.mark.asyncio
async def test_run_override_script_doctor_wins_over_brief(db_session) -> None:
    """An explicit per-run config_overrides value wins over the project brief."""
    brief_cfg = {"enabled": True, "passes": 1}
    override = {"enabled": True, "passes": 3}
    snap = await build_config_snapshot(
        db_session,
        proj=_proj(type_="promo", brief={"script_doctor": brief_cfg}),
        deliv=_deliv(),
        tenant_id=uuid4(),
        subject_id=uuid4(),
        config_overrides={"script_doctor": override},
    )
    assert snap["script_doctor"] == override


@pytest.mark.asyncio
async def test_run_override_motion_graphics_wins_over_brief(db_session) -> None:
    """An explicit per-run config_overrides value wins over the project brief."""
    brief_cfg = {"enabled": True, "template_id": "social_card"}
    override = {"enabled": True, "template_id": "pipeline_story"}
    snap = await build_config_snapshot(
        db_session,
        proj=_proj(type_="promo", brief={"motion_graphics": brief_cfg}),
        deliv=_deliv(),
        tenant_id=uuid4(),
        subject_id=uuid4(),
        config_overrides={"motion_graphics": override},
    )
    assert snap["motion_graphics"] == override


@pytest.mark.asyncio
async def test_finish_block_is_presence_opt_in(db_session) -> None:
    """The `finish` block is NOT default-on: a brief without one never adds the key,
    so build_finish_spec returns None and the compile request stays byte-identical.
    A brief block lifts verbatim (that is how the brand letterbox is opted into)."""
    snap_absent = await build_config_snapshot(
        db_session,
        proj=_proj(type_="promo", brief={}),
        deliv=_deliv(),
        tenant_id=uuid4(),
        subject_id=uuid4(),
        config_overrides=None,
    )
    assert "finish" not in snap_absent

    cfg = {"enabled": True, "letterbox_color": "#101010"}
    snap_present = await build_config_snapshot(
        db_session,
        proj=_proj(type_="promo", brief={"finish": cfg}),
        deliv=_deliv(),
        tenant_id=uuid4(),
        subject_id=uuid4(),
        config_overrides=None,
    )
    assert snap_present["finish"] == cfg


@pytest.mark.asyncio
async def test_run_override_finish_wins_over_brief(db_session) -> None:
    override = {"enabled": True, "letterbox_color": "#111111"}
    snap = await build_config_snapshot(
        db_session,
        proj=_proj(type_="promo", brief={"finish": {"enabled": True}}),
        deliv=_deliv(),
        tenant_id=uuid4(),
        subject_id=uuid4(),
        config_overrides={"finish": override},
    )
    assert snap["finish"] == override


@pytest.mark.asyncio
async def test_snapshot_propagates_project_theme_and_brand(db_session) -> None:
    """The motion-slide branding chain reads ``config_snapshot.brand`` /
    ``config_snapshot.theme``; build_config_snapshot must copy them verbatim from
    the project so a non-default brand reaches the renderer (BUG-1 regression)."""
    brand = {"primary_color": "#280983", "secondary_color": "#8460ec", "name": "Specra"}
    theme = {"accent": "#280983", "background": "#19023f", "text_color": "#F8FAFC"}
    snap = await build_config_snapshot(
        db_session,
        proj=_proj(type_="promo", brand=brand, theme=theme),
        deliv=_deliv(),
        tenant_id=uuid4(),
        subject_id=uuid4(),
        config_overrides=None,
    )
    assert snap["brand"] == brand
    assert snap["theme"] == theme


@pytest.mark.asyncio
async def test_snapshot_captures_license_id_from_identity_context(db_session) -> None:
    """The background dispatcher rebinds the three-piece identity from the
    snapshot, so build_config_snapshot must capture the request's license_id
    from the identity ContextVar (tenant/subject come from the args)."""
    from w2v_common.auth.identity_context import (
        clear_identity_context,
        set_identity_context,
    )

    tenant, subject = uuid4(), uuid4()
    set_identity_context(tenant_id=str(tenant), license_id="lic-xyz", subject_id=str(subject))
    try:
        snap = await build_config_snapshot(
            db_session,
            proj=_proj(type_="generic"),
            deliv=_deliv(),
            tenant_id=tenant,
            subject_id=subject,
            config_overrides=None,
        )
    finally:
        clear_identity_context()
    assert snap["license_id"] == "lic-xyz"
    assert snap["tenant_id"] == str(tenant)
    assert snap["subject_id"] == str(subject)


@pytest.mark.asyncio
async def test_snapshot_license_id_empty_without_context(db_session) -> None:
    """No identity context bound → license_id is "" (older runs / non-request
    callers); the dispatcher then binds tenant + subject only."""
    from w2v_common.auth.identity_context import clear_identity_context

    clear_identity_context()
    snap = await build_config_snapshot(
        db_session,
        proj=_proj(type_="generic"),
        deliv=_deliv(),
        tenant_id=uuid4(),
        subject_id=uuid4(),
        config_overrides=None,
    )
    assert snap["license_id"] == ""


# ─── brief.legal lift — shape-preservation guard ─────────────────────────────
# `legal` is typed at the API door now (LegalBrief, so a typo'd legal.mode 422s),
# but the door model is validation-ONLY: ProjectCreate never reassigns brief, and
# build_config_snapshot lifts from the persisted `proj.brief` dict. claim_finish
# reads ctx.config_snapshot["legal"] as a raw dict, so the lifted block must stay
# byte-identical to what the operator sent — nested claims dict included, and with
# NO pydantic-injected None defaults.


@pytest.mark.asyncio
async def test_legal_block_lifts_into_snapshot_unchanged(db_session) -> None:
    legal = {
        "mode": "auto",
        "posture": "conservative",
        "claims": {"on_unsupported": "flag", "iterations": 2, "score_threshold": 4.0},
    }
    snap = await build_config_snapshot(
        db_session,
        proj=_proj(type_="showcase", brief={"target_url": "https://x.example", "legal": legal}),
        deliv=_deliv(),
        tenant_id=uuid4(),
        subject_id=uuid4(),
        config_overrides=None,
    )
    # Exact dict identity of shape — not a superset. A door model that coerced the
    # brief would show up here as extra None keys (allow_superlatives=None, ...).
    assert snap["legal"] == legal
    assert snap["legal"]["claims"] == legal["claims"]
    assert snap["brief"]["legal"] == legal


@pytest.mark.asyncio
async def test_prod_legal_block_without_mode_lifts_unchanged(
    db_session, _claim_finish_default_off
) -> None:
    """The legal block already in prod: no `mode` ⇒ lifted verbatim, and
    claim_finish's guard (mode not in {auto, critique_only}) still skips.
    (With the WS-1 seed ON, `mode: critique_only` is added — see
    test_authored_legal_without_mode_still_gets_mode_seeded.)"""
    legal = {"trademarks": ["Acme®"], "defensibility": "reviewed by counsel"}
    snap = await build_config_snapshot(
        db_session,
        proj=_proj(type_="showcase", brief={"target_url": "https://x.example", "legal": legal}),
        deliv=_deliv(),
        tenant_id=uuid4(),
        subject_id=uuid4(),
        config_overrides=None,
    )
    assert snap["legal"] == legal
    assert "mode" not in snap["legal"]


# ─── ORCH-1.4: brief.voice.{direction,cast} lift into config.voice ─────────────
# These exercise the pure `_lift_brief_voice_keys` merge directly (no db_session):
# the merge is a pure dict op, and adding 3 more schema-create/drop cycles to this
# already DB-heavy file tips it into the pre-existing schema-reset deadlock. The
# helper is the single site `build_config_snapshot` delegates to, so this is the
# behaviour under test — full coverage, zero DB load.


# ─── S20: brief.voice.master_lufs → mastering.loudness_target alias ─────────────
# `voice.master_lufs` had ZERO consumers (every voice reader takes only
# cast/direction for TTS), so a brief author's loudness target was silently
# swallowed. The LIVE path is mastering.loudness_target → build_mastering_profile.


@pytest.mark.asyncio
async def test_absent_legal_not_lifted(db_session, _claim_finish_default_off) -> None:
    """A brief with no `legal` block never gets one from the LIFT (the WS-1 seed is
    off here; its own coverage asserts the seeded case)."""
    snap = await build_config_snapshot(
        db_session,
        proj=_proj(type_="showcase", brief={"target_url": "https://x.example"}),
        deliv=_deliv(),
        tenant_id=uuid4(),
        subject_id=uuid4(),
        config_overrides=None,
    )
    assert "legal" not in snap


# ─── ORCH-4.1 T1: sound default-on for showcase ──────────────────────────────


def _settings(*, sound_default_on_showcase: bool) -> SimpleNamespace:
    """A get_settings() stand-in exposing a `.pipeline` with the sound
    default-injection kill-switch (mirrors the demo_qa test idiom).

    Typed explicitly rather than `**overrides: object` — splatting an
    `object`-valued mapping into the settings model erases every field's type.
    """
    from src.config import PipelineSettings

    return SimpleNamespace(
        pipeline=PipelineSettings(sound_default_on_showcase=sound_default_on_showcase)
    )


@pytest.mark.asyncio
async def test_showcase_defaults_sound_on_when_absent(db_session, monkeypatch) -> None:
    """A showcase brief that never authors sound_effects still ships WITH sound
    — the whole point of T1 (today every showcase renders silent)."""
    monkeypatch.setattr(
        "src.services.runs.run_setup.get_settings",
        lambda: _settings(sound_default_on_showcase=True),
    )
    snap = await build_config_snapshot(
        db_session,
        proj=_proj(type_="showcase", brief={"target_url": "https://x.example"}),
        deliv=_deliv(),
        tenant_id=uuid4(),
        subject_id=uuid4(),
        config_overrides=None,
    )
    assert snap["sound_effects"] == {"enabled": True, "ambient": True}


@pytest.mark.asyncio
async def test_showcase_sound_default_off_via_kill_switch(db_session, monkeypatch) -> None:
    """PIPELINE_SOUND_DEFAULT_ON_SHOWCASE=false ⇒ byte-identical to before T1:
    no sound_effects key added, no code deploy needed to roll back a bad default."""
    monkeypatch.setattr(
        "src.services.runs.run_setup.get_settings",
        lambda: _settings(sound_default_on_showcase=False),
    )
    snap = await build_config_snapshot(
        db_session,
        proj=_proj(type_="showcase", brief={"target_url": "https://x.example"}),
        deliv=_deliv(),
        tenant_id=uuid4(),
        subject_id=uuid4(),
        config_overrides=None,
    )
    assert "sound_effects" not in snap


@pytest.mark.asyncio
async def test_non_showcase_project_never_gets_sound_default(db_session, monkeypatch) -> None:
    """The default-on injection is showcase-specific — a promo/demo_video/lesson
    run must never have sound silently switched on underneath it."""
    monkeypatch.setattr(
        "src.services.runs.run_setup.get_settings",
        lambda: _settings(sound_default_on_showcase=True),
    )
    snap = await build_config_snapshot(
        db_session,
        proj=_proj(type_="promo", brief={}),
        deliv=_deliv(),
        tenant_id=uuid4(),
        subject_id=uuid4(),
        config_overrides=None,
    )
    assert "sound_effects" not in snap


@pytest.mark.asyncio
async def test_authored_showcase_sound_effects_never_overwritten(db_session, monkeypatch) -> None:
    """An authored brief.sound_effects block always wins outright over the
    default injection — never clobbered."""
    monkeypatch.setattr(
        "src.services.runs.run_setup.get_settings",
        lambda: _settings(sound_default_on_showcase=True),
    )
    authored = {"enabled": True, "music": "jazz_lounge", "ambient": False}
    snap = await build_config_snapshot(
        db_session,
        proj=_proj(
            type_="showcase",
            brief={"target_url": "https://x.example", "sound_effects": authored},
        ),
        deliv=_deliv(),
        tenant_id=uuid4(),
        subject_id=uuid4(),
        config_overrides=None,
    )
    assert snap["sound_effects"] == authored


@pytest.mark.asyncio
async def test_run_override_sound_effects_never_overwritten(db_session, monkeypatch) -> None:
    """A per-run config_overrides value also wins outright over the default
    injection — the lift loop already spread it into config_snapshot first."""
    monkeypatch.setattr(
        "src.services.runs.run_setup.get_settings",
        lambda: _settings(sound_default_on_showcase=True),
    )
    override = {"enabled": False}
    snap = await build_config_snapshot(
        db_session,
        proj=_proj(type_="showcase", brief={"target_url": "https://x.example"}),
        deliv=_deliv(),
        tenant_id=uuid4(),
        subject_id=uuid4(),
        config_overrides={"sound_effects": override},
    )
    assert snap["sound_effects"] == override


# P3: build_config_snapshot pins provider slots; these tests run on a sentinel session.
pytestmark = pytest.mark.usefixtures("stub_provider_pins")
