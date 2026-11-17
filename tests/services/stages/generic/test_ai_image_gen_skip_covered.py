"""generic ai_image_gen must not render stills video_render will discard.

Coverage is computed by ``assign_generic_line_media`` — the SAME function
video_render uses — so these tests drive both stages off one set of prior
outputs and assert they agree line for line.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.clients.schemas.ai_content_generation import AcgConceptImageJob
from src.services.runs.run_context import REGENERATE_FROM_STAGE_KEY, RunContext
from src.services.stages.base import StageSkippedError
from src.services.stages.generic import ai_image_gen
from src.services.stages.generic.ai_image_gen import AiImageGenStage, GenericAiImageGenStage
from src.services.stages.generic.video_render import VideoRenderStage
from src.services.visual_media import assign_generic_line_media

_LINES = ("l1", "l2", "l3", "l4")
_CFG = {"ai_image_gen": {"prompts_from_stage": "storyboard_draft"}}


def _storyboard() -> dict[str, Any]:
    return {
        "shots": [{"id": f"s-{lid}", "line_id": lid, "seconds": 3.0} for lid in _LINES],
        "prompts": [{"line_id": lid, "prompt": f"still for {lid}"} for lid in _LINES],
    }


def _client() -> AsyncMock:
    client = AsyncMock()
    client.create_concept_image = AsyncMock(return_value="job-1")
    client.wait_for_concept_image = AsyncMock(
        return_value=AcgConceptImageJob(
            job_id="job-1",
            asset_id="asset-1",
            status="completed",
            provider="local_image",
            image_url="https://minio.test/buckets/concept/asset-1.png",
        )
    )
    return client


def _ctx(prior: dict[str, Any], cfg: dict[str, Any] | None = None) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=cfg if cfg is not None else _CFG,
        prior_run_outputs={"storyboard_draft": _storyboard(), **prior},
    )


@pytest.fixture(autouse=True)
def _no_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    # No stored image to reuse: every prompt reaches the (fake) provider.
    monkeypatch.setattr(ai_image_gen, "find_existing_media", AsyncMock(return_value=None))


@pytest.mark.asyncio
async def test_coverage_off_renders_every_line() -> None:
    """All opt-in media stages skipped ({} / absent) → identical to before."""
    client = _client()
    ctx = _ctx({"source_ingest": {}, "media_discovery": {}, "ai_video_gen": {}})
    out = await GenericAiImageGenStage(client=client).run({}, ctx=ctx)
    assert sorted(i.line_id for i in out.images if i.line_id) == list(_LINES)
    assert out.skipped_covered_line_ids == []
    assert client.create_concept_image.await_count == len(_LINES)


@pytest.mark.asyncio
async def test_mixed_coverage_skips_only_covered_lines() -> None:
    """Pinned image → l1, discovered clip → l2, one AI clip spread → l4."""
    prior = {
        "source_ingest": {
            "pinned_media": [{"ref": "https://x/p", "composite_url": "https://minio/pin.png"}]
        },
        "media_discovery": {
            "items": [{"media_type": "video", "composite_url": "https://minio/disc.mp4"}]
        },
        "ai_video_gen": {"videos": [{"video_url": "https://minio/ai.mp4"}]},
    }
    covered = assign_generic_line_media(_ctx(prior)).covered_line_ids
    assert len(covered) == 3
    assert covered < set(_LINES)

    client = _client()
    out = await GenericAiImageGenStage(client=client).run({}, ctx=_ctx(prior))
    rendered = {i.line_id for i in out.images}
    assert rendered == set(_LINES) - covered
    assert out.skipped_covered_line_ids == sorted(covered)
    assert client.create_concept_image.await_count == len(_LINES) - len(covered)

    # video_render, fed the stage's real output, uses every rendered still and
    # loses none — the two stages agree on coverage line for line.
    fake = AsyncMock()
    fake.compile.return_value = {"artifact_url": "s3://x/y.mp4", "duration_seconds": 12.0}
    render_ctx = _ctx({**prior, "ai_image_gen": out.model_dump(mode="json")})
    await VideoRenderStage(client=fake).run({}, ctx=render_ctx)
    shots = {s["line_id"]: s for s in fake.compile.call_args.kwargs["shots"]}
    (uncovered,) = set(_LINES) - covered
    assert shots[uncovered]["image_url"] == next(i.image_url for i in out.images)
    for lid in covered:
        assert shots[lid].get("video_url") or shots[lid]["image_url"].startswith("https://minio/")


@pytest.mark.asyncio
async def test_full_coverage_skips_the_stage_without_any_provider_call() -> None:
    prior = {
        "motion_graphics_gen": {
            "clips": [
                {"line_id": lid, "video_url": f"https://minio/mg-{lid}.mp4"} for lid in _LINES
            ]
        }
    }
    client = _client()
    with pytest.raises(StageSkippedError) as exc:
        await GenericAiImageGenStage(client=client).run({}, ctx=_ctx(prior))
    assert "covered" in exc.value.reason
    assert all(lid in exc.value.reason for lid in _LINES)
    client.create_concept_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_unkeyed_static_prompts_are_never_skipped() -> None:
    """Round-robin (unkeyed) stills fill any uncovered line, so keep them."""
    prior = {
        "motion_graphics_gen": {
            "clips": [
                {"line_id": lid, "video_url": f"https://minio/mg-{lid}.mp4"} for lid in _LINES
            ]
        }
    }
    client = _client()
    cfg = {"ai_image_gen": {"prompts": ["a", "b"]}}
    out = await GenericAiImageGenStage(client=client).run({}, ctx=_ctx(prior, cfg))
    assert len(out.images) == 2
    assert out.skipped_covered_line_ids == []


@pytest.mark.asyncio
async def test_base_stage_used_by_promo_and_course_never_skips() -> None:
    prior = {
        "motion_graphics_gen": {
            "clips": [
                {"line_id": lid, "video_url": f"https://minio/mg-{lid}.mp4"} for lid in _LINES
            ]
        }
    }
    client = _client()
    out = await AiImageGenStage(client=client).run({}, ctx=_ctx(prior))
    assert len(out.images) == len(_LINES)
    assert out.skipped_covered_line_ids == []


# ── regenerate child: carry the parent's images forward ──────────────────────


class _Result:
    def __init__(self, value: object) -> None:
        self._value = value

    def scalar_one_or_none(self) -> object:
        return self._value


class _FakeSession:
    """Answers the two reads: runs.parent_run_id, then the parent's stage row."""

    def __init__(self, parent_run_id: object, parent_row: object) -> None:
        self._answers = [parent_run_id, parent_row]

    async def execute(self, _stmt: object) -> _Result:
        return _Result(self._answers.pop(0))

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None


class _Row:
    def __init__(self, output: dict[str, Any]) -> None:
        self.status = "completed"
        self.output = output


def _factory(parent_output: dict[str, Any] | None) -> Any:
    row = _Row(parent_output) if parent_output is not None else None
    return lambda: _FakeSession(uuid4() if parent_output is not None else None, row)


@pytest.fixture
def _parent_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ai_image_gen, "bind_session_tenant", AsyncMock())
    monkeypatch.setattr(ai_image_gen, "artifact_exists", AsyncMock(return_value=True))


_ONE_AI_CLIP = {"ai_video_gen": {"videos": [{"video_url": "https://minio/ai.mp4"}]}}


async def _parent_output() -> dict[str, Any]:
    """The parent run: one AI clip covers one line, three lines rendered."""
    out = await GenericAiImageGenStage(client=_client()).run({}, ctx=_ctx(_ONE_AI_CLIP))
    assert out.rendered_count == 3
    assert out.carried_forward_count == 0
    return out.model_dump(mode="json")


@pytest.mark.asyncio
@pytest.mark.usefixtures("_parent_reads")
async def test_regenerate_from_ai_video_gen_unchanged_renders_nothing() -> None:
    parent = await _parent_output()
    client = _client()
    stage = GenericAiImageGenStage(client=client, session_factory=_factory(parent))
    # Child re-ran ai_video_gen and got the same coverage; prompts unchanged.
    out = await stage.run({}, ctx=_ctx(_ONE_AI_CLIP))
    client.create_concept_image.assert_not_awaited()
    assert out.rendered_count == 0
    assert out.carried_forward_count == 3
    parent_urls = {i["line_id"]: i["image_url"] for i in parent["images"]}
    assert {i.line_id: i.image_url for i in out.images} == parent_urls


@pytest.mark.asyncio
@pytest.mark.usefixtures("_parent_reads")
async def test_regenerate_one_line_losing_coverage_renders_exactly_one() -> None:
    parent = await _parent_output()
    client = _client()
    stage = GenericAiImageGenStage(client=client, session_factory=_factory(parent))
    # Child's ai_video_gen produced no clip → the formerly covered line needs a still.
    out = await stage.run({}, ctx=_ctx({"ai_video_gen": {}}))
    assert client.create_concept_image.await_count == 1
    assert out.rendered_count == 1
    assert out.carried_forward_count == 3
    assert sorted(i.line_id for i in out.images if i.line_id) == list(_LINES)


@pytest.mark.asyncio
@pytest.mark.usefixtures("_parent_reads")
async def test_regenerate_changed_prompt_renders_only_that_line() -> None:
    parent = await _parent_output()
    client = _client()
    stage = GenericAiImageGenStage(client=client, session_factory=_factory(parent))
    ctx = _ctx(_ONE_AI_CLIP)
    uncovered = sorted(set(_LINES) - assign_generic_line_media(ctx).covered_line_ids)
    ctx = RunContext(
        run_id=ctx.run_id,
        deliverable_id=ctx.deliverable_id,
        config_snapshot=ctx.config_snapshot,
        prior_run_outputs=ctx.prior_run_outputs,
        open_annotations_by_stage={
            "ai_image_gen": [{"element_id": uncovered[0], "text": "make it blue"}]
        },
    )
    out = await stage.run({}, ctx=ctx)
    assert client.create_concept_image.await_count == 1
    assert (out.rendered_count, out.carried_forward_count) == (1, 2)


@pytest.mark.asyncio
@pytest.mark.usefixtures("_parent_reads")
async def test_non_child_run_renders_as_before() -> None:
    client = _client()
    stage = GenericAiImageGenStage(client=client, session_factory=_factory(None))
    out = await stage.run({}, ctx=_ctx({}))
    assert client.create_concept_image.await_count == len(_LINES)
    assert (out.rendered_count, out.carried_forward_count) == (len(_LINES), 0)


@pytest.mark.asyncio
async def test_parent_image_definitively_gone_is_rerendered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ai_image_gen, "bind_session_tenant", AsyncMock())
    parent = await _parent_output()
    monkeypatch.setattr(ai_image_gen, "artifact_exists", AsyncMock(return_value=False))
    client = _client()
    stage = GenericAiImageGenStage(client=client, session_factory=_factory(parent))
    out = await stage.run({}, ctx=_ctx(_ONE_AI_CLIP))
    assert (out.rendered_count, out.carried_forward_count) == (3, 0)


@pytest.mark.asyncio
@pytest.mark.usefixtures("_parent_reads")
async def test_regenerate_explicitly_from_ai_image_gen_renders_every_line() -> None:
    """The user re-ran THIS stage: fresh variants of unchanged prompts, none carried."""
    parent = await _parent_output()
    client = _client()
    stage = GenericAiImageGenStage(client=client, session_factory=_factory(parent))
    cfg = {**_CFG, REGENERATE_FROM_STAGE_KEY: "ai_image_gen"}
    out = await stage.run({}, ctx=_ctx(_ONE_AI_CLIP, cfg))
    assert client.create_concept_image.await_count == 3
    assert (out.rendered_count, out.carried_forward_count) == (3, 0)


@pytest.mark.asyncio
@pytest.mark.usefixtures("_parent_reads")
async def test_regenerate_explicitly_from_ai_video_gen_still_carries_forward() -> None:
    parent = await _parent_output()
    client = _client()
    stage = GenericAiImageGenStage(client=client, session_factory=_factory(parent))
    cfg = {**_CFG, REGENERATE_FROM_STAGE_KEY: "ai_video_gen"}
    out = await stage.run({}, ctx=_ctx(_ONE_AI_CLIP, cfg))
    client.create_concept_image.assert_not_awaited()
    assert (out.rendered_count, out.carried_forward_count) == (0, 3)


@pytest.mark.asyncio
async def test_existence_check_raising_renders_that_line_and_the_stage_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ai_image_gen, "bind_session_tenant", AsyncMock())
    parent = await _parent_output()
    first_url = parent["images"][0]["image_url"]

    async def _exists(url: str) -> bool:
        if url == first_url:
            raise RuntimeError("artifacts HEAD timed out")
        return True

    # Every parent image shares one fake url, so make them distinct per line.
    for i, img in enumerate(parent["images"]):
        img["image_url"] = f"{first_url}?n={i}" if i else first_url
    monkeypatch.setattr(ai_image_gen, "artifact_exists", _exists)
    client = _client()
    stage = GenericAiImageGenStage(client=client, session_factory=_factory(parent))
    out = await stage.run({}, ctx=_ctx(_ONE_AI_CLIP))
    assert client.create_concept_image.await_count == 1
    assert (out.rendered_count, out.carried_forward_count) == (1, 2)
