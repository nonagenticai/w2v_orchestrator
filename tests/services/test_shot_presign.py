"""Unit tests for refresh_shot_asset_urls — compile-time asset re-sign.

Verifies the helper re-signs the three top-level asset-url keys AND the nested
ones an asset hides in (``overlay_clips[].url``, ``split.second_source_url``),
leaves the rest of a shot untouched, dedupes repeated URLs to a single re-sign
*across nesting levels* (the compiler's per-job fetch dedupe is by URL, so one
source must map to one refreshed URL), keeps the original when the backend
declines (returns None), and is a no-op for a storage fake that doesn't
implement ``resign_presigned_url``.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.services.compile_budget import COMPILE_WAIT_CEILING_SECONDS, PRESIGN_TTL_SECONDS
from src.services.stages import shot_presign
from src.services.stages.shot_presign import refresh_shot_asset_urls


class _FakeStorage:
    """Re-signs by tagging ``...&fresh`` and counting calls per URL."""

    def __init__(self, *, decline: bool = False) -> None:
        self.decline = decline
        self.calls: list[str] = []

    async def resign_presigned_url(self, url: str) -> str | None:
        self.calls.append(url)
        if self.decline:
            return None
        return f"{url}&fresh"


class _NoResignStorage:
    """A storage fake (like the TTS/slide test fakes) lacking the re-sign method."""


class _TTLCapturingStorage:
    """A storage fake with BOTH the ownership-validating resign AND the
    TTL-capable presign, so the (b) mutation-proof tests below can assert on
    the ACTUAL expiry passed to the second, TTL-bearing signature — not just
    that some URL came back."""

    def __init__(self) -> None:
        self.resign_calls: list[str] = []
        self.presign_calls: list[tuple[str, int | None, str | None]] = []

    async def resign_presigned_url(self, url: str) -> str | None:
        self.resign_calls.append(url)
        return f"{url}&default-ttl"

    async def get_presigned_url(
        self, path: str, expires: int | None = None, bucket: str | None = None
    ) -> str | None:
        self.presign_calls.append((path, expires, bucket))
        return f"https://minio.test/{bucket}/{path}?sig=fresh&ttl={expires}"


@pytest.mark.asyncio
async def test_refreshes_three_asset_keys_only() -> None:
    storage = _FakeStorage()
    shots: list[dict[str, Any]] = [
        {
            "id": "s1",
            "line_id": "l1",
            "seconds": 3.0,
            "description": "a text card",
            "caption": "hi",
            "fit": "cover",
            "image_url": "http://m/w2v-orchestrator/i.png?sig=old",
            "video_url": "http://m/w2v-orchestrator/v.mp4?sig=old",
            "audio_url": "http://m/w2v-orchestrator/a.wav?sig=old",
        }
    ]

    out = await refresh_shot_asset_urls(shots, storage=storage)

    shot = out[0]
    assert shot["image_url"].endswith("&fresh")
    assert shot["video_url"].endswith("&fresh")
    assert shot["audio_url"].endswith("&fresh")
    # Non-asset keys untouched.
    assert shot["id"] == "s1"
    assert shot["line_id"] == "l1"
    assert shot["seconds"] == 3.0
    assert shot["description"] == "a text card"
    assert shot["caption"] == "hi"
    assert shot["fit"] == "cover"


@pytest.mark.asyncio
async def test_dedupes_repeated_url() -> None:
    storage = _FakeStorage()
    reused = "http://m/w2v-html-render/sections/hero.png?sig=old"
    shots = [
        {"id": "s1", "line_id": "l1", "image_url": reused},
        {"id": "s2", "line_id": "l2", "image_url": reused},
        {"id": "s3", "line_id": "l3", "image_url": reused},
    ]

    out = await refresh_shot_asset_urls(shots, storage=storage)

    # One re-sign for the shared URL, applied to all three shots.
    assert storage.calls == [reused]
    assert all(s["image_url"] == f"{reused}&fresh" for s in out)


@pytest.mark.asyncio
async def test_keeps_original_when_resign_declines() -> None:
    storage = _FakeStorage(decline=True)
    original = "https://example.com/foreign.png"
    shots = [{"id": "s1", "line_id": "l1", "image_url": original}]

    out = await refresh_shot_asset_urls(shots, storage=storage)

    assert out[0]["image_url"] == original


@pytest.mark.asyncio
async def test_no_op_when_storage_lacks_resign() -> None:
    original = "http://m/w2v-orchestrator/i.png?sig=old"
    shots = [{"id": "s1", "line_id": "l1", "image_url": original}]

    out = await refresh_shot_asset_urls(shots, storage=_NoResignStorage())

    assert out[0]["image_url"] == original


@pytest.mark.asyncio
async def test_no_op_when_storage_unavailable(monkeypatch) -> None:
    # _get_storage raising (e.g. backend can't initialise) must NOT fail compile.
    async def _boom() -> object:
        raise OSError("read-only file system")

    monkeypatch.setattr(shot_presign, "_get_storage", _boom)
    original = "http://m/w2v-orchestrator/i.png?sig=old"
    shots = [{"id": "s1", "line_id": "l1", "image_url": original}]

    out = await refresh_shot_asset_urls(shots)  # no storage= → exercises _get_storage

    assert out[0]["image_url"] == original


@pytest.mark.asyncio
async def test_keeps_original_when_resign_raises() -> None:
    class _Raising:
        async def resign_presigned_url(self, url: str) -> str:
            raise RuntimeError("signing exploded")

    original = "http://m/w2v-orchestrator/i.png?sig=old"
    shots = [{"id": "s1", "line_id": "l1", "image_url": original}]

    out = await refresh_shot_asset_urls(shots, storage=_Raising())

    assert out[0]["image_url"] == original


@pytest.mark.asyncio
async def test_refreshes_nested_overlay_clip_urls() -> None:
    """An overlay whose presign died is DROPPED by the renderer with a log line and
    the render continues — a reel with no graphics and a green run. So the nested
    url has to be re-signed like any top-level asset."""
    storage = _FakeStorage()
    shots: list[dict[str, Any]] = [
        {
            "id": "s1",
            "line_id": "l1",
            "video_url": "http://m/w2v-orchestrator/v.mp4?sig=old",
            "overlay_clips": [
                {"url": "http://m/w2v-orchestrator/lower-third.mov?sig=old", "x": 0, "y": 0},
                {"url": "http://m/w2v-orchestrator/logo.mov?sig=old", "opacity": 0.8},
            ],
        }
    ]

    out = await refresh_shot_asset_urls(shots, storage=storage)

    overlays = out[0]["overlay_clips"]
    assert [o["url"] for o in overlays] == [
        "http://m/w2v-orchestrator/lower-third.mov?sig=old&fresh",
        "http://m/w2v-orchestrator/logo.mov?sig=old&fresh",
    ]
    # The rest of the overlay entry is carried through untouched.
    assert overlays[0]["x"] == 0
    assert overlays[1]["opacity"] == 0.8


@pytest.mark.asyncio
async def test_one_source_reused_across_nesting_levels_yields_one_url() -> None:
    """The compiler dedupes its per-job fetches BY URL, so the same original must
    come back as the same new URL wherever it appears — a per-call-site signature
    would silently double the downloads of a reused clip."""
    storage = _FakeStorage()
    reused = "http://m/w2v-orchestrator/badge.mov?sig=old"
    shots: list[dict[str, Any]] = [
        {"id": "s1", "line_id": "l1", "overlay_clips": [{"url": reused}, {"url": reused}]},
        {"id": "s2", "line_id": "l2", "video_url": reused, "overlay_clips": [{"url": reused}]},
    ]

    out = await refresh_shot_asset_urls(shots, storage=storage)

    assert storage.calls == [reused]  # signed exactly once
    fresh = f"{reused}&fresh"
    assert [o["url"] for o in out[0]["overlay_clips"]] == [fresh, fresh]
    assert out[1]["video_url"] == fresh
    assert out[1]["overlay_clips"][0]["url"] == fresh


@pytest.mark.asyncio
async def test_refreshes_the_split_screen_b_side() -> None:
    """``split.second_source_url`` is emitted TODAY (promo/demo_video split-screen)
    from the same presigned pool as ``video_url`` — a live expiry hole, not a
    pre-emptive one."""
    storage = _FakeStorage()
    shots: list[dict[str, Any]] = [
        {
            "id": "s1",
            "line_id": "l1",
            "video_url": "http://m/w2v-orchestrator/a.mp4?sig=old",
            "split": {"layout": "h", "second_source_url": "http://m/w2v-orchestrator/b.mp4?sig=x"},
        }
    ]

    out = await refresh_shot_asset_urls(shots, storage=storage)

    assert out[0]["split"]["second_source_url"] == "http://m/w2v-orchestrator/b.mp4?sig=x&fresh"
    assert out[0]["split"]["layout"] == "h"  # non-url keys untouched


@pytest.mark.asyncio
async def test_malformed_nested_containers_are_left_alone() -> None:
    """The helper can only refresh a URL or leave the payload alone — never reshape
    it. A junk container is skipped, not coerced or dropped."""
    storage = _FakeStorage()
    shots: list[dict[str, Any]] = [
        {"id": "s1", "line_id": "l1", "overlay_clips": None},
        {"id": "s2", "line_id": "l2", "overlay_clips": ["not-a-dict", None]},
        {"id": "s3", "line_id": "l3", "split": "not-a-dict"},
        {"id": "s4", "line_id": "l4", "overlay_clips": [{"x": 0}]},  # entry with no url
    ]

    out = await refresh_shot_asset_urls(shots, storage=storage)

    assert storage.calls == []
    assert out[0]["overlay_clips"] is None
    assert out[1]["overlay_clips"] == ["not-a-dict", None]
    assert out[2]["split"] == "not-a-dict"
    assert out[3]["overlay_clips"] == [{"x": 0}]


@pytest.mark.asyncio
async def test_ignores_missing_and_non_string_urls() -> None:
    storage = _FakeStorage()
    shots = [
        {"id": "s1", "line_id": "l1", "description": "card only"},  # no asset urls
        {"id": "s2", "line_id": "l2", "image_url": ""},  # empty
        {"id": "s3", "line_id": "l3", "image_url": None},  # absent/None
    ]

    out = await refresh_shot_asset_urls(shots, storage=storage)

    assert storage.calls == []
    assert "image_url" not in out[0]
    assert out[1]["image_url"] == ""
    assert out[2]["image_url"] is None


@pytest.mark.asyncio
async def test_default_ttl_is_the_compile_budget_ttl() -> None:
    """(b) mutation-proof: the compile-input presign path must use
    ``compile_budget.PRESIGN_TTL_SECONDS`` — NOT the global
    ``StorageSettings.presign_expiry_seconds`` (which is back at the base 3600s
    default; see ``tests/test_config.py::test_storage_defaults``).

    This is the COMPILE-path twin of the assertion that used to live (wrongly)
    on the global setting in ``test_config.py`` — this is the path a compile
    job's inputs actually go through.

    MUTATION PROOF: reverting ``refresh_shot_asset_urls``'s ``ttl_seconds``
    default back to relying on ``resign_presigned_url`` alone (i.e. dropping the
    second, TTL-bearing ``get_presigned_url`` call) turns this red — the
    captured ``expires`` would be ``None``/absent instead of
    ``PRESIGN_TTL_SECONDS``.
    """
    storage = _TTLCapturingStorage()
    shots: list[dict[str, Any]] = [
        {"id": "s1", "line_id": "l1", "video_url": "http://m/w2v-orchestrator/v.mp4?sig=old"}
    ]

    out = await refresh_shot_asset_urls(shots, storage=storage)

    assert storage.presign_calls == [("v.mp4", PRESIGN_TTL_SECONDS, "w2v-orchestrator")]
    assert out[0]["video_url"] == (
        f"https://minio.test/w2v-orchestrator/v.mp4?sig=fresh&ttl={PRESIGN_TTL_SECONDS}"
    )
    # The whole point of defect 1: this TTL must cover the compile-wait ceiling
    # with margin, not just be "some long number".
    assert PRESIGN_TTL_SECONDS >= COMPILE_WAIT_CEILING_SECONDS + 60


@pytest.mark.asyncio
async def test_explicit_ttl_seconds_overrides_the_default() -> None:
    storage = _TTLCapturingStorage()
    shots: list[dict[str, Any]] = [
        {"id": "s1", "line_id": "l1", "image_url": "http://m/w2v-orchestrator/i.png?sig=old"}
    ]

    await refresh_shot_asset_urls(shots, storage=storage, ttl_seconds=42)

    assert storage.presign_calls == [("i.png", 42, "w2v-orchestrator")]


@pytest.mark.asyncio
async def test_falls_back_to_default_ttl_signature_when_storage_lacks_get_presigned_url() -> None:
    """A storage fake with only ``resign_presigned_url`` (the shape every
    existing test fake in this file, and the TTS/slide test fakes, use) must
    keep working exactly as before — this helper never fails a compile because
    a TTL upgrade wasn't available."""
    storage = _FakeStorage()
    shots: list[dict[str, Any]] = [
        {"id": "s1", "line_id": "l1", "image_url": "http://m/w2v-orchestrator/i.png?sig=old"}
    ]

    out = await refresh_shot_asset_urls(shots, storage=storage)

    assert out[0]["image_url"] == "http://m/w2v-orchestrator/i.png?sig=old&fresh"
