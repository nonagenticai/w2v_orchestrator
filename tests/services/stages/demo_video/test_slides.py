"""showcase_slides — renders ONLY the authored card beats of a showcase brief.

A no-op for all-footage runs (returns an empty deck), so it never touches a plain
demo_video run. Footage lines (no ``card``) are bound in compile, not here.
"""

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.services import tts_storage
from src.services.runs.run_context import RunContext
from src.services.stages.demo_video.slides import ShowcaseSlidesStage
from src.services.stages.shared.design_system import DEFAULT_TOKENS, as_params
from src.services.stages.shared.slide_deck import _BRAND_DEFAULTS

# Two card beats (intro title + value-prop bullets) bracketing one footage line.
_LINES = [
    {
        "id": "intro",
        "text": "This is SpecraLabs.",
        "card": {"template": "title", "title": "SpecraLabs"},
    },
    {"id": "clip-1", "text": "A clean dashboard."},  # footage line — no card
    {
        "id": "value",
        "text": "Place devices on a live canvas.",
        "card": {"template": "bullets", "title": "What you get", "bullets": ["Audio", "Video"]},
    },
]


class _FakeStorage:
    async def initialize(self) -> None:
        pass

    async def store_stage_artifact(
        self, job_id: str, stage_name: str, data: bytes, filename: str
    ) -> str:
        return f"artifacts/{job_id}/{stage_name}/{filename}"

    async def get_presigned_url(
        self, path: str, expires: int | None = None, bucket: str | None = None
    ) -> str | None:
        return f"https://minio.test/{path}"


@pytest.fixture
def fake_storage():
    tts_storage.set_storage_for_tests(_FakeStorage())
    yield
    tts_storage.set_storage_for_tests(None)


def _ctx(
    lines: list[dict] | None = None,
    brand: dict | None = None,
    design: dict | None = None,
) -> RunContext:
    brief: dict[str, Any] = {}
    if lines is not None:
        brief["lines"] = lines
    snapshot: dict[str, Any] = {"brief": brief, "deliverable": {"title": "SpecraLabs"}}
    if brand is not None:
        snapshot["brand"] = brand
    if design is not None:
        snapshot["design"] = design
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=snapshot,
        prior_run_outputs={},
    )


def _slide_client(png_count: int = 2) -> AsyncMock:
    fake = AsyncMock()
    fake.resolve_template_id.side_effect = lambda name: {
        "title": "tpl-title",
        "cta": "tpl-cta",
        "feature_spotlight": "tpl-feature",
    }[name]
    fake.start_deck.return_value = "deck-1"
    fake.wait_for_deck.return_value = {"status": "completed", "slide_count": png_count}
    fake.get_slide_png.side_effect = [b"\x89PNG" + bytes([i]) for i in range(png_count)]
    return fake


@pytest.mark.asyncio
async def test_renders_only_card_lines_keyed_by_line_id(fake_storage) -> None:
    fake = _slide_client()
    out = await ShowcaseSlidesStage(client=fake).run({}, ctx=_ctx(_LINES))

    # Only the two card lines render; the footage line (clip-1) is skipped.
    payload = fake.start_deck.await_args.args[0]
    assert len(payload["slides"]) == 2
    assert [s.line_id for s in out.slides] == ["intro", "value"]
    # Bullets card → feature_spotlight; title card → title template.
    templates = [s["template_id"] for s in payload["slides"]]
    assert templates == ["tpl-title", "tpl-feature"]
    assert payload["slides"][1]["content"]["bullets"] == ["Audio", "Video"]
    assert payload["brand"]["primary_color"] == _BRAND_DEFAULTS["primary_color"]
    assert out.slides[0].image_url.startswith("https://minio.test/")


@pytest.mark.asyncio
async def test_per_project_brand_overrides_defaults(fake_storage) -> None:
    fake = _slide_client()
    out = await ShowcaseSlidesStage(client=fake).run(
        {}, ctx=_ctx(_LINES, brand={"primary_color": "#280983", "name": "SpecraLabs"})
    )
    payload = fake.start_deck.await_args.args[0]
    assert payload["brand"]["primary_color"] == "#280983"
    assert out.slides  # rendered


@pytest.mark.asyncio
async def test_design_block_sends_the_full_token_bundle_to_slide_service(fake_storage) -> None:
    """A ``design`` block ⇒ every derived token reaches the slide client (ORCH-3.1b).

    The design system used to be inert: only 4 brand keys were forwarded and the
    flat token map had zero callers. Assert on the call the client actually
    receives, not on an intermediate.
    """
    fake = _slide_client()
    design = {
        "type_scale_ratio": 1.5,
        "icon_set": "solid",
        "palette": {"accent": "#112233"},
        "grid": {"cols": 16},
    }
    out = await ShowcaseSlidesStage(client=fake).run({}, ctx=_ctx(_LINES, design=design))

    tokens = fake.start_deck.await_args.kwargs["tokens"]
    # Pin the key SET, not a count — the set says which token went missing; a
    # magic number just drifts when the design system grows one.
    assert set(tokens) == set(as_params(DEFAULT_TOKENS))
    assert all(isinstance(v, str) for v in tokens.values())
    # Authored overrides survive the whole path down to the client.
    assert tokens["accent"] == "#112233"
    assert tokens["icon_set"] == "solid"
    assert tokens["grid_cols"] == "16"
    assert tokens["type_scale_ratio"] == "1.5"
    # The 4-key brand channel is untouched — this PR ADDS a channel, it replaces none.
    assert fake.start_deck.await_args.args[0]["brand"]["secondary_color"] == "#112233"
    assert out.slides  # rendered


@pytest.mark.asyncio
async def test_no_design_block_sends_no_tokens(fake_storage) -> None:
    """Presence-is-opt-in: no ``design`` block ⇒ no ``tokens`` key at all."""
    fake = _slide_client()
    await ShowcaseSlidesStage(client=fake).run({}, ctx=_ctx(_LINES))

    assert fake.start_deck.await_args.kwargs.get("tokens") is None
    assert "tokens" not in fake.start_deck.await_args.args[0]


@pytest.mark.asyncio
async def test_empty_when_no_card_lines(fake_storage) -> None:
    fake = AsyncMock()
    footage_only = [{"id": "clip-1", "text": "Dashboard."}, {"id": "clip-2", "text": "Client."}]
    out = await ShowcaseSlidesStage(client=fake).run({}, ctx=_ctx(footage_only))
    assert out.slides == []
    fake.start_deck.assert_not_called()


@pytest.mark.asyncio
async def test_empty_when_no_lines(fake_storage) -> None:
    fake = AsyncMock()
    out = await ShowcaseSlidesStage(client=fake).run({}, ctx=_ctx())
    assert out.slides == []
    fake.start_deck.assert_not_called()


@pytest.mark.asyncio
async def test_deck_failure_soft_fails_with_note(fake_storage) -> None:
    fake = AsyncMock()
    fake.resolve_template_id.side_effect = lambda name: "tpl"
    fake.start_deck.side_effect = RuntimeError("slide_service down")
    out = await ShowcaseSlidesStage(client=fake).run({}, ctx=_ctx(_LINES))
    assert out.slides == []
    assert out.notes and "slide render failed" in out.notes[0]


@pytest.mark.asyncio
async def test_motion_cards_attach_video_url(fake_storage) -> None:
    fake = _slide_client()
    video = AsyncMock()
    video.record_animation.return_value = "vjob-1"
    video.wait_for_content_recording.return_value = {"status": "completed"}
    video.get_recording_video.return_value = (b"MP4BYTES", "video/mp4", "inline")

    ctx = _ctx(_LINES)
    ctx.config_snapshot["motion_graphics"] = {"enabled": True}

    out = await ShowcaseSlidesStage(client=fake, video_client=video).run({}, ctx=ctx)

    assert [s.line_id for s in out.slides] == ["intro", "value"]
    assert all(s.motion_video_url for s in out.slides)  # animated clips attached
    assert all(s.image_url.startswith("https://minio.test/") for s in out.slides)  # PNGs kept
    # The title card projects onto the animated_title_card template.
    template_ids = {c.kwargs["template_id"] for c in video.record_animation.await_args_list}
    assert "animated_title_card" in template_ids
