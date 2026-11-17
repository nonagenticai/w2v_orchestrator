"""Tests for the site_capture stage handler."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
import pytest

from src.services.runs.run_context import RunContext
from src.services.stages.base import StageSkippedError
from src.services.stages.generic.site_capture import (
    SiteCaptureIncompleteError,
    SiteCaptureLoginError,
    SiteCaptureStage,
    SiteCaptureUnavailableError,
)

_STORE = "src.services.stages.generic.site_capture.store_capture_clip"
_STORE_IMG = "src.services.stages.generic.site_capture.store_capture_image"


def _screens_zip(screen_ids: list[str]) -> bytes:
    """A self-describing screenshot zip (entry name = ``<screen_id>.png``)."""
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for sid in screen_ids:
            zf.writestr(f"{sid}.png", f"png-{sid}".encode())
    return buf.getvalue()


def _make_client(*, duration: float = 5.0) -> AsyncMock:
    client = AsyncMock()
    client.record_url = AsyncMock(return_value="job-1")
    client.wait_for_content_recording = AsyncMock(
        return_value={"status": "completed", "duration_seconds": duration}
    )
    client.get_recording_video = AsyncMock(return_value=(b"fake-mp4", "video/mp4", None))
    return client


def _ctx(brief: dict) -> RunContext:
    return RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={"brief": brief})


@pytest.mark.asyncio
async def test_skips_without_source_url() -> None:
    stage = SiteCaptureStage(client=_make_client())
    with pytest.raises(StageSkippedError):
        await stage.run({}, ctx=_ctx({}))


@pytest.mark.asyncio
async def test_happy_path_single_clip() -> None:
    client = _make_client(duration=7.5)
    stage = SiteCaptureStage(client=client)
    ctx = _ctx({"source_url": "https://app.internal/"})  # .internal → scrape skipped (no network)

    with patch(_STORE, new=AsyncMock(return_value="https://minio.internal/cap/clip.mp4")):
        out = await stage.run({}, ctx=ctx)

    assert len(out.clips) == 1
    assert out.clips[0].video_url == "https://minio.internal/cap/clip.mp4"
    assert out.clips[0].seconds == 7.5
    # Real recorded footage → the honesty flag is set, not inferred from len(clips).
    assert out.used_real_footage is True
    assert out.capture_status == "ok"
    client.record_url.assert_awaited_once()
    # .internal host is refused by the SSRF guard, so no page text
    assert out.page_text == ""


@pytest.mark.asyncio
async def test_multiple_paths_capped_by_max_clips() -> None:
    client = _make_client()
    stage = SiteCaptureStage(client=client)
    ctx = _ctx(
        {
            "source_url": "https://app.internal/",
            "site_capture": {"paths": ["/", "/a", "/b", "/c", "/d"], "max_clips": 3},
        }
    )
    with patch(_STORE, new=AsyncMock(return_value="https://minio.internal/cap/c.mp4")):
        out = await stage.run({}, ctx=ctx)

    assert len(out.clips) == 3
    assert client.record_url.await_count == 3


@pytest.mark.asyncio
async def test_recording_failure_is_soft_and_stage_succeeds() -> None:
    client = _make_client()
    client.record_url = AsyncMock(
        side_effect=RuntimeError("video_production job job-1 failed: page returned 404")
    )
    stage = SiteCaptureStage(client=client)
    ctx = _ctx({"source_url": "https://app.internal/"})

    out = await stage.run({}, ctx=ctx)

    # Soft-fail: no clips, but the stage SUCCEEDS (does not raise) so the run
    # degrades to AI images / text cards rather than failing.
    assert out.clips == []
    # All clips failed → the fallback is unmistakable, not silent.
    assert out.used_real_footage is False
    assert any("recording failed" in n for n in out.notes)
    assert any("no clips captured" in n for n in out.notes)
    # The recorder ANSWERED, so zero clips is a measurement, not a gap.
    assert out.capture_status == "degraded"


@pytest.mark.asyncio
async def test_clip_path_zero_clips_from_infra_is_reported_as_a_gap() -> None:
    """Clip-path failure semantics are unchanged, but the GAP is now expressible.

    A timeout means video_production never answered for this path, so ``clips: []``
    is not evidence the site had nothing to record. The stage still succeeds (the
    clip path is a best-effort enhancement), but it says ``unknown`` instead of
    silently passing an unmeasured zero downstream.
    """
    client = _make_client()
    client.record_url = AsyncMock(
        side_effect=TimeoutError("video_production job job-1 timed out after 600.0s")
    )
    stage = SiteCaptureStage(client=client)
    ctx = _ctx({"source_url": "https://app.internal/"})

    out = await stage.run({}, ctx=ctx)

    assert out.clips == []
    assert out.capture_status == "unknown"
    assert any("MEASUREMENT GAP" in n for n in out.notes)


def _ctx_with_resolution(brief: dict, resolution: str | None) -> RunContext:
    snap: dict = {"brief": brief}
    if resolution is not None:
        snap["resolution"] = resolution
    return RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot=snap)


def test_resolve_targets_dedup_and_cap() -> None:
    targets = SiteCaptureStage._resolve_targets(
        "https://x.com/", {"paths": ["/", "/", "/a"], "max_clips": 5}
    )
    assert targets == [("https://x.com/", "/", None), ("https://x.com/a", "/a", None)]


def test_resolve_targets_per_path_interactions() -> None:
    steps = [{"action": "click", "selector": '.nav >> text="Build"', "hold_ms": 3000}]
    targets = SiteCaptureStage._resolve_targets(
        "https://x.com/",
        {"paths": ["/", {"path": "/demo", "interactions": steps}], "max_clips": 5},
    )
    # The "/demo" target carries its interactions => it becomes the primary clip.
    assert targets == [
        ("https://x.com/", "/", None),
        ("https://x.com/demo", "/demo", steps),
    ]


def test_safe_to_fetch_rejects_internal_and_loopback(monkeypatch) -> None:
    # The guard resolves every host, so stub DNS: `.internal` is this estate's
    # Tailnet (CGNAT 100.64.0.0/10 — `ipaddress.is_private` is False there), and
    # an unresolvable name must fail CLOSED.
    import ipaddress
    import socket

    from w2v_common.net import safe_fetch

    resolved = {"app.internal": "100.64.0.5", "localhost": "127.0.0.1"}

    def fake_getaddrinfo(host, port, *args, **kwargs):
        addr = resolved.get(host)
        if addr is None:
            try:
                ipaddress.ip_address(host)
            except ValueError:
                raise socket.gaierror(f"stubbed DNS: unknown host {host!r}") from None
            addr = host
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (addr, port or 0))]

    monkeypatch.setattr(safe_fetch.socket, "getaddrinfo", fake_getaddrinfo)

    assert SiteCaptureStage._safe_to_fetch("https://app.internal/") is False
    assert SiteCaptureStage._safe_to_fetch("http://localhost/") is False
    assert SiteCaptureStage._safe_to_fetch("https://foo.local/") is False  # unresolvable
    assert SiteCaptureStage._safe_to_fetch("ftp://example.com/") is False
    # Loopback / private literals
    assert SiteCaptureStage._safe_to_fetch("http://127.0.0.1/") is False
    assert SiteCaptureStage._safe_to_fetch("http://10.0.0.5/") is False
    assert SiteCaptureStage._safe_to_fetch("http://169.254.169.254/") is False


def test_extract_text_strips_tags_and_scripts() -> None:
    html = "<html><head><style>x{}</style></head><body><h1>Hi</h1>"
    html += "<script>alert(1)</script><p>World</p></body></html>"
    text = SiteCaptureStage._extract_text(html)
    assert "Hi" in text
    assert "World" in text
    assert "alert" not in text
    assert "<" not in text


def test_name_and_io_models() -> None:
    from src.services.stages.generic.site_capture import (
        SiteCaptureInput,
        SiteCaptureOutput,
    )

    assert SiteCaptureStage.name == "site_capture"
    assert SiteCaptureStage.input_model is SiteCaptureInput
    assert SiteCaptureStage.output_model is SiteCaptureOutput


# ── per-screen capture mode (frame-synced demo) ────────────────────


def _screens_client() -> AsyncMock:
    client = AsyncMock()
    client.capture_screens = AsyncMock(return_value="screens-job-1")
    client.wait_for_content_recording = AsyncMock(
        return_value={"status": "completed", "duration_seconds": 0.0}
    )
    client.get_recording_video = AsyncMock(
        return_value=(_screens_zip(["home", "gov"]), "application/zip", None)
    )
    return client


@pytest.mark.asyncio
async def test_per_screen_mode_emits_screens() -> None:
    client = _screens_client()
    stage = SiteCaptureStage(client=client)
    ctx = _ctx(
        {
            "source_url": "https://app.internal/demo",
            "site_capture": {
                "screens": [
                    {"selector": '.nav-item >> text="Home"', "capture_id": "home", "label": "Home"},
                    {"selector": ".persona-switch"},  # advance only, no capture
                    {
                        "selector": '.nav-item >> text="Governance"',
                        "capture_id": "gov",
                        "label": "Gov",
                    },
                ]
            },
        }
    )

    with patch(
        _STORE_IMG, new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['image_id']}.png")
    ):
        out = await stage.run({}, ctx=ctx)

    # One SiteScreen per captured id; clips empty in per-screen mode.
    assert {s.id for s in out.screens} == {"home", "gov"}
    assert {s.label for s in out.screens} == {"Home", "Gov"}
    assert out.clips == []
    assert out.used_real_footage is True
    client.capture_screens.assert_awaited_once()
    # The persona-switch step (no capture_id) is forwarded but carries no id.
    steps = client.capture_screens.await_args.kwargs["steps"]
    assert [s.get("capture_id") for s in steps] == ["home", None, "gov"]
    # label is orchestrator-only — never sent to the recorder.
    assert all("label" not in s for s in steps)


def _screens_clip_zip(screen_ids: list[str]) -> bytes:
    """A clips-mode zip: both ``<id>.png`` stills and ``<id>.mp4`` clips."""
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for sid in screen_ids:
            zf.writestr(f"{sid}.png", f"png-{sid}".encode())
            zf.writestr(f"{sid}.mp4", f"mp4-{sid}".encode())
    return buf.getvalue()


@pytest.mark.asyncio
async def test_per_screen_mode_clips_emits_clip_url_and_requests_clips() -> None:
    client = _screens_client()
    client.get_recording_video = AsyncMock(
        return_value=(_screens_clip_zip(["home", "gov"]), "application/zip", None)
    )
    stage = SiteCaptureStage(client=client)
    ctx = _ctx(
        {
            "source_url": "https://app.internal/demo",
            "site_capture": {
                "motion": "clips",
                "screens": [
                    {"selector": ".home", "capture_id": "home", "label": "Home"},
                    {"selector": ".gov", "capture_id": "gov", "label": "Gov"},
                ],
            },
        }
    )

    with (
        patch(
            _STORE_IMG,
            new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['image_id']}.png"),
        ),
        patch(
            _STORE,
            new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['clip_id']}.mp4"),
        ),
    ):
        out = await stage.run({}, ctx=ctx)

    # record_clips=True is forwarded to the recorder.
    assert client.capture_screens.await_args.kwargs["record_clips"] is True
    by_id = {s.id: s for s in out.screens}
    assert by_id["home"].image_url == "https://minio/home.png"
    assert by_id["home"].clip_url == "https://minio/home-clip.mp4"  # clip_id = "<id>-clip"
    assert by_id["gov"].clip_url == "https://minio/gov-clip.mp4"
    assert out.used_real_footage is True


@pytest.mark.asyncio
async def test_per_screen_mode_stills_only_does_not_request_clips() -> None:
    client = _screens_client()
    stage = SiteCaptureStage(client=client)
    ctx = _ctx(
        {
            "source_url": "https://app.internal/demo",
            "site_capture": {"screens": [{"selector": ".nav", "capture_id": "home"}]},
        }
    )
    with patch(
        _STORE_IMG, new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['image_id']}.png")
    ):
        out = await stage.run({}, ctx=ctx)
    # Default mode: record_clips falsy, no clip_url.
    assert client.capture_screens.await_args.kwargs.get("record_clips") in (False, None)
    assert all(s.clip_url == "" for s in out.screens)


@pytest.mark.asyncio
async def test_per_screen_mode_soft_fails_when_the_recorder_answered() -> None:
    """A recorder-REPORTED job failure stays soft — that is a real measurement.

    ``poll_job`` raises ``RuntimeError("… job … failed: <detail>")`` only after the
    recorder ran, looked at the page and told us the capture did not work (a bad
    selector, a page that never settled). We KNOW the site yielded nothing here,
    so degrading to slides is an informed choice and the run continues.

    (This test previously used ``RuntimeError("recorder down")`` and asserted the
    same soft outcome for what was really an infrastructure failure — that is the
    fail-open this split removes; see the timeout test below.)
    """
    client = _screens_client()
    client.wait_for_content_recording = AsyncMock(
        side_effect=RuntimeError(
            "video_production job screens-job-1 failed: selector '.nav' never appeared"
        )
    )
    stage = SiteCaptureStage(client=client)
    ctx = _ctx(
        {
            "source_url": "https://app.internal/demo",
            "site_capture": {"screens": [{"selector": ".nav", "capture_id": "home"}]},
        }
    )

    out = await stage.run({}, ctx=ctx)

    # Soft-fail: no screens, run still completes, degradation recorded in notes.
    assert out.screens == []
    assert out.clips == []
    assert out.used_real_footage is False
    assert any("per-screen capture failed" in n for n in out.notes)
    # ...and it is a MEASURED shortfall, not a gap.
    assert out.capture_status == "degraded"


@pytest.mark.asyncio
async def test_per_screen_infra_timeout_fails_the_stage() -> None:
    """The 2026-09-03 outage shape: video_production never answered.

    ``poll_job`` raises ``TimeoutError`` when the recording job blows its deadline
    (a recording-slot leak, a reaped pod). Zero screens then says NOTHING about the
    site, so the stage must fail rather than return a successful
    ``screens: []`` that reads as "this site had nothing to record".
    """
    client = _screens_client()
    client.wait_for_content_recording = AsyncMock(
        side_effect=TimeoutError("video_production job screens-job-1 timed out after 600.0s")
    )
    stage = SiteCaptureStage(client=client)
    ctx = _ctx(
        {
            "source_url": "https://app.internal/demo",
            "site_capture": {"screens": [{"selector": ".nav", "capture_id": "home"}]},
        }
    )

    with pytest.raises(SiteCaptureUnavailableError) as excinfo:
        await stage.run({}, ctx=ctx)

    # The operator reading Run.error gets the gap named, not a generic timeout.
    assert "NO measurement" in str(excinfo.value)
    assert "timed out after 600.0s" in str(excinfo.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(httpx.ConnectError("connection refused"), id="transport"),
        pytest.param(
            httpx.HTTPStatusError(
                "502",
                request=httpx.Request("POST", "https://vp/api/record/capture/screens"),
                response=httpx.Response(502),
            ),
            id="5xx",
        ),
        pytest.param(
            httpx.HTTPStatusError(
                "404",
                request=httpx.Request("GET", "https://vp/api/record/jobs/x"),
                response=httpx.Response(404),
            ),
            id="job-reaped",
        ),
    ],
)
async def test_per_screen_unreachable_dependency_fails_the_stage(exc: Exception) -> None:
    """Unreachable / broken / vanished downstream is the same gap as a timeout."""
    client = _screens_client()
    client.capture_screens = AsyncMock(side_effect=exc)
    stage = SiteCaptureStage(client=client)
    ctx = _ctx(
        {
            "source_url": "https://app.internal/demo",
            "site_capture": {"screens": [{"selector": ".nav", "capture_id": "home"}]},
        }
    )

    with pytest.raises(SiteCaptureUnavailableError):
        await stage.run({}, ctx=ctx)


@pytest.mark.asyncio
async def test_per_screen_unreadable_bundle_is_a_measurement_gap() -> None:
    """The recorder answered but its bundle is unreadable → screen count UNKNOWN.

    Not a fail (the recorder did its job, this is our side) and emphatically not
    ``degraded`` — we never counted the screens, so consumers must not score this
    like a site that genuinely had nothing.
    """
    client = _screens_client()
    client.get_recording_video = AsyncMock(return_value=(b"not-a-zip", "application/zip", None))
    stage = SiteCaptureStage(client=client)
    ctx = _ctx(
        {
            "source_url": "https://app.internal/demo",
            "site_capture": {"screens": [{"selector": ".nav", "capture_id": "home"}]},
        }
    )

    out = await stage.run({}, ctx=ctx)
    assert out.screens == []
    assert out.capture_status == "unknown"
    assert any("per-screen capture failed" in n for n in out.notes)


# ── authored shotlist (deterministic per-shot capture) ─────────────────


@pytest.mark.asyncio
async def test_shotlist_synthesizes_screens_from_url_shots() -> None:
    """An authored brief.shotlist captures one screen per URL shot, keyed by the
    shot's deterministic line id; slide shots are excluded and legacy
    site_capture.paths is ignored."""
    client = AsyncMock()
    client.capture_screens = AsyncMock(return_value="screens-job-1")
    client.wait_for_content_recording = AsyncMock(
        return_value={"status": "completed", "duration_seconds": 0.0}
    )
    client.get_recording_video = AsyncMock(
        return_value=(_screens_zip(["shot-000", "shot-002"]), "application/zip", None)
    )
    stage = SiteCaptureStage(client=client)
    ctx = _ctx(
        {
            "source_url": "https://app.internal/",
            "site_capture": {"paths": ["/ignored-legacy-path"]},  # superseded by shotlist
            "shotlist": [
                {
                    "kind": "url",
                    "url": "https://app.internal/",
                    "narration": "the landing page",
                    "directives": {"hold_ms": 4000},
                },
                {"kind": "slide", "narration": "a slide", "slide": {"title": "t"}},  # no capture
                {
                    "kind": "url",
                    "url": "https://app.internal/pricing",
                    "narration": "the pricing page",
                    "directives": {"click": ".pricing", "hold_ms": 3000},
                },
            ],
        }
    )

    with patch(
        _STORE_IMG, new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['image_id']}.png")
    ):
        out = await stage.run({}, ctx=ctx)

    # One screen per URL shot, id == deterministic shot line id (the join key).
    assert {s.id for s in out.screens} == {"shot-000", "shot-002"}
    assert out.used_real_footage is True
    client.capture_screens.assert_awaited_once()
    call = client.capture_screens.await_args
    assert call is not None
    steps = call.kwargs["steps"]
    # Two capturable steps (slide shot excluded), keyed by line id; legacy path unused.
    assert [s.get("capture_id") for s in steps] == ["shot-000", "shot-002"]
    assert steps[0]["settle_ms"] == 4000  # hold_ms → settle
    assert steps[1].get("selector") == ".pricing"  # click → selector
    assert all("label" not in s for s in steps)  # label is orchestrator-only


@pytest.mark.asyncio
async def test_shotlist_url_shots_with_zero_screens_fail_the_stage() -> None:
    """A shotlist contracts its screens, so zero screens FAILS — even on the soft path.

    ``RuntimeError`` is the ANSWERED failure (``poll_job`` raises it when the
    recorder itself reports the job failed), i.e. exactly the case that stays soft
    for ``brief.source_url`` — see
    ``test_per_screen_mode_soft_fails_when_the_recorder_answered``. Under an
    authored shotlist the same failure is a MISSING DELIVERABLE: each ``kind: url``
    shot contracts a screen with ``id == line_id`` and the downstream stages bind
    narration to those ids. Reporting ``completed`` with ``screens: []`` here is
    what left dependent stages ``pending`` and parked the run in ``paused``.
    """
    client = AsyncMock()
    client.capture_screens = AsyncMock(return_value="screens-job-1")
    client.wait_for_content_recording = AsyncMock(
        side_effect=RuntimeError(
            "video_production job screens-job-1 failed: selector '.pricing' never appeared"
        )
    )
    client.get_recording_video = AsyncMock(return_value=(_screens_zip([]), "application/zip", None))
    stage = SiteCaptureStage(client=client)
    ctx = _ctx(
        {
            "source_url": "https://app.internal/",
            "shotlist": [
                {
                    "kind": "url",
                    "url": "https://app.internal/",
                    "narration": "the landing page",
                    "directives": {"hold_ms": 4000},
                },
                {
                    "kind": "url",
                    "url": "https://app.internal/pricing",
                    "narration": "the pricing page",
                    "directives": {"click": ".pricing", "hold_ms": 3000},
                },
            ],
        }
    )

    with pytest.raises(SiteCaptureIncompleteError) as excinfo:
        await stage.run({}, ctx=ctx)

    # The failure has to NAME the missing deliverable, not just say "capture failed".
    message = str(excinfo.value)
    assert "shot-000" in message
    assert "shot-001" in message
    assert "2 of 2" in message
    assert "capture_status=degraded" in message


@pytest.mark.asyncio
async def test_shotlist_base_url_falls_back_to_target_url() -> None:
    """website_review briefs carry target_url (not source_url); the shotlist
    capture base must fall back to it so relative shot paths resolve correctly."""
    client = AsyncMock()
    client.capture_screens = AsyncMock(return_value="job")
    client.wait_for_content_recording = AsyncMock(
        return_value={"status": "completed", "duration_seconds": 0.0}
    )
    client.get_recording_video = AsyncMock(
        return_value=(_screens_zip(["shot-000"]), "application/zip", None)
    )
    stage = SiteCaptureStage(client=client)
    ctx = _ctx(
        {
            "target_url": "https://acme.test/",  # no source_url
            "shotlist": [
                {"kind": "url", "url": "https://acme.test/", "narration": "the landing page"}
            ],
        }
    )

    with patch(
        _STORE_IMG, new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['image_id']}.png")
    ):
        await stage.run({}, ctx=ctx)

    call = client.capture_screens.await_args
    assert call is not None
    assert call.kwargs["url"] == "https://acme.test/"


# ── per-shot directives → recorder steps (recorder PR #54) ─────────────


# The full set of keys video_production_service's ScreenCaptureStep understands.
# Mirrors that schema (cross-repo, not importable here); the subset assertion
# below fails loudly if site_capture ever emits a key the recorder would drop at
# its ``extra="ignore"`` boundary.
_RECORDER_STEP_FIELDS = {
    "capture_id",
    "action",
    "selector",
    "settle_ms",
    "goto",
    "scroll_to",
    "wait_for",
    "viewport",
}


def _entry(url: str | None, directives: dict, base_url: str = "https://app.internal/") -> dict:
    from src.services.stages.generic.site_capture import _shot_to_screen_entry
    from src.services.stages.shared.shotlist import ShotSpec, shot_line_id

    shot = ShotSpec(
        index=0,
        line_id=shot_line_id(0),
        kind="url",
        narration="n",
        seconds=2.5,
        url=url,
        directives=directives,
    )
    return _shot_to_screen_entry(shot, base_url)


def test_shot_entry_url_without_click_becomes_goto() -> None:
    entry = _entry("https://other.test/pricing", {})
    assert entry["goto"] == "https://other.test/pricing"
    assert "selector" not in entry


def test_shot_entry_click_means_in_spa_no_goto() -> None:
    entry = _entry("https://app.internal/pricing", {"click": ".nav-pricing"})
    assert entry["selector"] == ".nav-pricing"
    assert "goto" not in entry  # click => in-SPA, never a full reload


def test_shot_entry_landing_shot_no_goto() -> None:
    entry = _entry("https://app.internal/", {}, base_url="https://app.internal/")
    assert "goto" not in entry  # url == base, already loaded


# always emitted so a later shot can reset


def test_build_screen_steps_forwards_new_keys_and_keeps_subset() -> None:
    from src.services.stages.generic.site_capture import SiteCaptureStage

    screens_cfg = [
        {
            "capture_id": "shot-000",
            "label": "landing",
            "settle_ms": 2000,
            "viewport": "mobile",
            "goto": "https://other.test/",
            "scroll_to": "bottom",
            "wait_for": "#main",
        }
    ]
    steps, labels = SiteCaptureStage._build_screen_steps(screens_cfg)
    step = steps[0]
    assert step["goto"] == "https://other.test/"
    assert step["scroll_to"] == "bottom"
    assert step["wait_for"] == "#main"
    assert step["viewport"] == "mobile"
    assert labels["shot-000"] == "landing"
    assert "label" not in step  # orchestrator-only, stripped
    # Contract guard: every emitted key must be one the recorder understands.
    assert set(step) <= _RECORDER_STEP_FIELDS


# ── ORCH-3.2 directed footage: VP-2 .track.json sidecar extraction ─────────


def _screens_track_zip(screen_ids: list[str], *, tracks: dict[str, bytes] | None = None) -> bytes:
    """A per-screen zip: ``<id>.png`` stills + optional ``<id>.track.json`` sidecars."""
    import io
    import zipfile

    tracks = tracks or {}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for sid in screen_ids:
            zf.writestr(f"{sid}.png", f"png-{sid}".encode())
            if sid in tracks:
                zf.writestr(f"{sid}.track.json", tracks[sid])
    return buf.getvalue()


def _ctx_footage(brief: dict, footage: dict) -> RunContext:
    """A ctx that opted into directed footage via the lifted ``direction.footage`` block."""
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": brief, "direction": {"footage": footage}},
    )


@pytest.mark.asyncio
async def test_per_screen_mode_extracts_track_when_footage_opted_in() -> None:
    focus = [{"t": 0.4, "bbox": {"x": 1, "y": 2, "w": 3, "h": 4}, "action": "click"}]
    cursor = [{"t": 0.0, "x": 0, "y": 0, "click": False}]
    payload = json.dumps(
        {"focus": focus, "cursor": cursor, "viewport": {"w": 1920, "h": 1080}}
    ).encode()
    client = _screens_client()
    client.get_recording_video = AsyncMock(
        return_value=(
            _screens_track_zip(["home", "gov"], tracks={"home": payload}),
            "application/zip",
            None,
        )
    )
    stage = SiteCaptureStage(client=client)
    ctx = _ctx_footage(
        {
            "source_url": "https://app.internal/demo",
            "site_capture": {
                "screens": [
                    {"selector": ".home", "capture_id": "home"},
                    {"selector": ".gov", "capture_id": "gov"},
                ]
            },
        },
        {"auto_zoom": True},
    )
    with patch(
        _STORE_IMG, new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['image_id']}.png")
    ):
        out = await stage.run({}, ctx=ctx)

    by = {s.id: s for s in out.screens}
    assert by["home"].focus_track == focus
    assert by["home"].cursor_track == cursor
    assert by["home"].viewport == {"w": 1920, "h": 1080}
    # A screen with no sidecar keeps its geometry unset (degrade, never raise).
    assert by["gov"].focus_track is None
    assert by["gov"].cursor_track is None
    # auto_zoom on => the recorder is asked to capture at 2x for sharp punch-ins.
    assert client.capture_screens.await_args.kwargs["recording_config"]["capture_scale"] == 2.0


@pytest.mark.asyncio
async def test_per_screen_track_ignored_without_direction_footage() -> None:
    # Gate: no direction.footage block => the sidecar is not extracted (geometry stays
    # None) and no capture_scale bump is requested => byte-identical to today.
    payload = json.dumps({"focus": [{"t": 0.4, "bbox": {"x": 1, "y": 2, "w": 3, "h": 4}}]}).encode()
    client = _screens_client()
    client.get_recording_video = AsyncMock(
        return_value=(
            _screens_track_zip(["home"], tracks={"home": payload}),
            "application/zip",
            None,
        )
    )
    stage = SiteCaptureStage(client=client)
    ctx = _ctx(
        {
            "source_url": "https://app.internal/demo",
            "site_capture": {"screens": [{"selector": ".home", "capture_id": "home"}]},
        }
    )
    with patch(
        _STORE_IMG, new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['image_id']}.png")
    ):
        out = await stage.run({}, ctx=ctx)

    assert out.screens[0].focus_track is None
    assert out.screens[0].cursor_track is None
    rc = client.capture_screens.await_args.kwargs.get("recording_config") or {}
    assert "capture_scale" not in rc


def test_unzip_screen_tracks_parses_good_and_drops_malformed_and_empty() -> None:
    good = json.dumps({"focus": [{"t": 0.1, "bbox": {"x": 1, "y": 2, "w": 3, "h": 4}}]}).encode()
    empty = json.dumps({"focus": [], "cursor": []}).encode()  # no keyframes → dropped
    malformed = b"{not json"  # parse error → dropped, never raises
    zb = _screens_track_zip(["a", "b", "c"], tracks={"a": good, "b": empty, "c": malformed})
    result = dict(SiteCaptureStage._unzip_screen_tracks(zb))
    assert set(result) == {"a"}
    assert result["a"]["focus"] == [{"t": 0.1, "bbox": {"x": 1, "y": 2, "w": 3, "h": 4}}]


def test_coerce_track_is_defensive_and_never_raises() -> None:
    assert SiteCaptureStage._coerce_track(b"") is None  # empty → parse error
    assert SiteCaptureStage._coerce_track(b"[]") is None  # non-object payload
    assert (
        SiteCaptureStage._coerce_track(b'{"focus": "nope"}') is None
    )  # wrong shape → no keyframes
    # A cursor-only sidecar is valid geometry; unknown/absent viewport is fine.
    assert SiteCaptureStage._coerce_track(b'{"cursor": [{"t": 0}]}') == {"cursor": [{"t": 0}]}


@pytest.mark.asyncio
async def test_credentials_ride_only_on_jobs_for_the_login_origin() -> None:
    """A bearer token bound to app.internal must never reach a path that leaves it:
    an absolute or scheme-relative `site_capture.paths` entry is recorded with NO
    credential (no bearer, no cookie, no basic-auth on a 401)."""
    client = _make_client()
    stage = SiteCaptureStage(client=client)
    login = {"type": "bearer", "token": "t-scoped"}  # pragma: allowlist secret
    ctx = _ctx(
        {
            "source_url": "https://app.internal/",
            "auth_info": login,
            "site_capture": {
                "paths": [
                    "/dashboard",
                    "https://attacker.example/x",
                    {"path": "//attacker.example"},
                ]
            },
        }
    )
    with patch(_STORE, new=AsyncMock(return_value="https://minio.internal/cap/c.mp4")):
        await stage.run({}, ctx=ctx)

    sent = {c.kwargs["url"]: c.kwargs["auth_info"] for c in client.record_url.await_args_list}
    assert sent["https://app.internal/dashboard"] == login
    assert sent["https://attacker.example/x"] is None
    assert sent["https://attacker.example"] is None


@pytest.mark.asyncio
async def test_screen_pass_leaving_the_login_origin_carries_no_credential() -> None:
    client = _make_client()
    client.capture_screens = AsyncMock(return_value="job-s")
    client.get_recording_video = AsyncMock(
        return_value=(_screens_zip(["s1", "s2"]), "application/zip", None)
    )
    stage = SiteCaptureStage(client=client)
    login = {"type": "bearer", "token": "t-scoped"}  # pragma: allowlist secret
    screens = [
        {"capture_id": "s1", "label": "home"},
        {"capture_id": "s2", "label": "away", "goto": "https://attacker.example/"},
    ]
    ctx = _ctx(
        {
            "source_url": "https://app.internal/",
            "auth_info": login,
            "site_capture": {"screens": screens},
        }
    )
    # One browser session navigates every screen, so a pass that leaves the login
    # origin gets NO credential — and recording it signed out would record a login
    # page, so the stage fails with a message instead of sending anything.
    with pytest.raises(SiteCaptureLoginError, match="another site"):
        await stage.run({}, ctx=ctx)
    client.capture_screens.assert_not_awaited()

    # Same pass kept on-origin: the login is attached.
    client.capture_screens.reset_mock()
    screens[1]["goto"] = "https://app.internal/settings"
    with patch(_STORE_IMG, new=AsyncMock(return_value="https://minio.internal/s.png")):
        await stage.run({}, ctx=ctx)
    call = client.capture_screens.await_args
    assert call is not None and call.kwargs["auth_info"] == login


# ── coded degradations + userinfo redaction in hash inputs ─────────


@pytest.mark.asyncio
async def test_recording_failure_carries_coded_degradations() -> None:
    """A degrading path writes a STABLE code, not prose only.

    Before: only ``notes`` were written, so the executor stamped
    ``unclassified_note`` ("degraded via prose notes only").
    """
    from src.services.stages.shared.degradation import (
        SITE_CAPTURE_CLIP_FAILED,
        SITE_CAPTURE_NO_FOOTAGE,
        SITE_PAGE_TEXT_MISSING,
        UNCLASSIFIED_NOTE,
        ensure_degradation_codes,
    )

    client = _make_client()
    client.record_url = AsyncMock(
        side_effect=RuntimeError("video_production job job-1 failed: page returned 404")
    )
    stage = SiteCaptureStage(client=client)
    out = await stage.run({}, ctx=_ctx({"source_url": "https://app.internal/"}))

    codes = [d["code"] for d in out.degradations]
    assert SITE_CAPTURE_CLIP_FAILED in codes
    assert SITE_CAPTURE_NO_FOOTAGE in codes
    # .internal is refused by the SSRF guard → no grounding text, coded too.
    assert SITE_PAGE_TEXT_MISSING in codes
    # Prose is kept alongside the codes.
    assert any("recording failed" in n for n in out.notes)
    dumped = out.model_dump(mode="json")
    assert ensure_degradation_codes("site_capture", dumped) is False
    assert UNCLASSIFIED_NOTE not in [d["code"] for d in dumped["degradations"]]


@pytest.mark.asyncio
async def test_per_screen_failure_carries_screens_failed_code() -> None:
    from src.services.stages.shared.degradation import SITE_CAPTURE_SCREENS_FAILED

    client = AsyncMock()
    client.capture_screens = AsyncMock(
        side_effect=RuntimeError("video_production job job-1 failed: selector not found")
    )
    stage = SiteCaptureStage(client=client)
    ctx = _ctx(
        {
            "source_url": "https://app.internal/",
            "site_capture": {"screens": [{"capture_id": "home"}]},
        }
    )
    out = await stage.run({}, ctx=ctx)

    assert out.screens == []
    assert SITE_CAPTURE_SCREENS_FAILED in [d["code"] for d in out.degradations]


def test_userinfo_does_not_reach_hash_inputs() -> None:
    from src.services.stages.generic import site_capture as sc

    with_creds = "https://alice:s3cr3t-pw@app.example.com/dash?x=1"  # pragma: allowlist secret
    other_creds = "https://bob:other-pw@app.example.com/dash?x=1"  # pragma: allowlist secret
    bare = "https://app.example.com/dash?x=1"

    redacted = sc._redact_userinfo(with_creds)
    assert redacted == bare
    assert "s3cr3t-pw" not in redacted
    assert "alice" not in redacted
    assert sc._redact_userinfo(bare) == bare

    run_id = str(uuid4())
    assert sc._stable_part_id("site", run_id, with_creds) == sc._stable_part_id(
        "site", run_id, other_creds
    )
    assert sc._stable_part_id("site", run_id, with_creds) == sc._stable_part_id(
        "site", run_id, bare
    )
    assert sc._stable_clip_id(run_id, with_creds) == sc._stable_clip_id(run_id, other_creds)


@pytest.mark.asyncio
async def test_recorder_body_hash_ignores_userinfo_but_recorder_gets_raw_url() -> None:
    """Two URLs differing only in userinfo produce the same body hash; the hash
    input carries no password; the URL SENT to the recorder is unchanged."""
    from src.services.stages.generic import site_capture as sc

    seen_hash_inputs: list[str] = []
    real_hash_inputs = sc.hash_inputs

    def _spy(*args: object) -> str:
        seen_hash_inputs.append(json.dumps(args, default=str, sort_keys=True))
        return real_hash_inputs(*args)

    body_hashes: list[str] = []
    sent_urls: list[str] = []
    url_a = "https://alice:s3cr3t-pw@app.internal/"  # pragma: allowlist secret
    url_b = "https://bob:other-pw@app.internal/"  # pragma: allowlist secret
    for url in (url_a, url_b):
        client = _make_client()
        stage = SiteCaptureStage(client=client)
        # Same run id for both so only the userinfo differs.
        ctx = RunContext(
            run_id=_FIXED_RUN_ID,
            deliverable_id=uuid4(),
            config_snapshot={"brief": {"source_url": url}},
        )
        with (
            patch.object(sc, "hash_inputs", side_effect=_spy),
            patch.object(sc, "_resumable_recording", wraps=sc._resumable_recording) as rr,
            patch(_STORE, new=AsyncMock(return_value="https://minio.internal/cap/clip.mp4")),
        ):
            await stage.run({}, ctx=ctx)
        rr_call = rr.await_args
        assert rr_call is not None
        body_hashes.append(rr_call.kwargs["body_hash"])
        sent_urls.append(client.record_url.await_args.kwargs["url"])

    assert body_hashes[0] == body_hashes[1]
    assert sent_urls == [url_a, url_b]
    joined = "\n".join(seen_hash_inputs)
    assert "s3cr3t-pw" not in joined
    assert "other-pw" not in joined


_FIXED_RUN_ID = uuid4()


@pytest.mark.asyncio
async def test_failure_prose_never_carries_url_userinfo(caplog: pytest.LogCaptureFixture) -> None:
    """PR #1078 audit: the clip-failure log + degrade note and the
    ``SiteCaptureUnavailableError`` message printed the raw ``source_url`` — a
    basic-auth password in it landed in pod logs, ``notes``/``degradations`` and
    ``Run.error``."""
    secret_url = "https://alice:s3cr3t-pw@app.internal/demo"  # pragma: allowlist secret

    client = _make_client()
    client.record_url = AsyncMock(side_effect=RuntimeError("job failed"))
    with caplog.at_level("WARNING"):
        out = await SiteCaptureStage(client=client).run({}, ctx=_ctx({"source_url": secret_url}))
    persisted = " ".join([*out.notes, *(d["detail"] for d in out.degradations)])
    assert "recording failed for https://app.internal/demo" in persisted
    assert "s3cr3t-pw" not in persisted
    assert "s3cr3t-pw" not in caplog.text

    screens = _screens_client()
    screens.wait_for_content_recording = AsyncMock(side_effect=TimeoutError("timed out"))
    with pytest.raises(SiteCaptureUnavailableError) as excinfo:
        await SiteCaptureStage(client=screens).run(
            {},
            ctx=_ctx(
                {
                    "source_url": secret_url,
                    "site_capture": {"screens": [{"selector": ".nav", "capture_id": "home"}]},
                }
            ),
        )
    assert "https://app.internal/demo" in str(excinfo.value)
    assert "s3cr3t-pw" not in str(excinfo.value)


@pytest.mark.asyncio
async def test_exception_text_never_carries_url_userinfo(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """PR #1078 audit, round 2: redacting the URL was not enough — ``{exc}`` was
    interpolated raw, and httpx's ``HTTPStatusError`` text embeds the request URL
    WITH its userinfo ("Client error '401 Unauthorized' for url
    '<url with userinfo>'"). A password-protected ``source_url`` answering 401
    wrote the password into the persisted SITE_PAGE_TEXT_MISSING note and the log.
    """
    from src.services.stages.generic import site_capture as sc

    secret_url = "https://bob:hunter2@example.com/x"  # pragma: allowlist secret
    transport = httpx.MockTransport(lambda request: httpx.Response(401, request=request))

    async def _fake_safe_get(_client: object, url: str, **kwargs: object) -> httpx.Response:
        async with httpx.AsyncClient(transport=transport) as mock_client:
            return await mock_client.get(url)

    # Precondition: the real httpx error text DOES carry the password, so the
    # assertions below exercise the scrub rather than an already-clean string.
    async with httpx.AsyncClient(transport=transport) as probe:
        resp = await probe.get(secret_url)
    with pytest.raises(httpx.HTTPStatusError) as raw:
        resp.raise_for_status()
    assert "hunter2" in str(raw.value)

    notes: list[str] = []
    degradations: list = []
    with (
        patch.object(SiteCaptureStage, "_safe_to_fetch", return_value=True),
        patch.object(sc, "safe_get", new=_fake_safe_get),
        caplog.at_level("WARNING"),
    ):
        text, status = await SiteCaptureStage._scrape_page_text(secret_url, notes, degradations)

    assert (text, status) == ("", "fetch_failed")
    persisted = " ".join([*notes, *(d["detail"] for d in degradations)])
    assert "401 Unauthorized" in persisted
    assert "https://example.com/x" in persisted
    assert "hunter2" not in persisted
    assert "bob:" not in persisted
    # Only what the scrape itself logged: the precondition probe above (and the
    # mock client) emit httpx's own INFO request line, which production keeps
    # off via configure_logging (httpx/httpcore pinned to WARNING).
    scrape_log = " ".join(r.getMessage() for r in caplog.records if r.name.startswith("src."))
    assert "hunter2" not in scrape_log
    assert "page scrape" in scrape_log


def test_configure_logging_silences_httpx_request_lines() -> None:
    """httpx logs every request URL verbatim (userinfo included) at INFO."""
    import logging

    from src.observability.logging import configure_logging

    configure_logging(level="INFO")
    assert logging.getLogger("httpx").getEffectiveLevel() >= logging.WARNING
    assert logging.getLogger("httpcore").getEffectiveLevel() >= logging.WARNING


@pytest.mark.asyncio
async def test_grounding_and_design_reuse_content_in_the_authenticated_capture_bundle():
    import io
    import zipfile

    stage = SiteCaptureStage(client=_make_client())
    content = (
        "The authenticated product page explains reporting and prioritising improvements. " * 6
    )
    html = "<main><section><h1>Improve knowledge work</h1></section></main>"
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("shot-000.png", b"fake-png")
        bundle.writestr("shot-000.page.json", json.dumps({"text": content, "html": html}))
    with patch(_STORE_IMG, new=AsyncMock(return_value="/captured.png")):
        screens = await stage._publish_screens(
            archive.getvalue(), "run", {}, record_clips=False, footage=None
        )
    with patch.object(stage, "_scrape_page_text", new=AsyncMock()) as scrape:
        out = await stage._screens_output(
            "https://example.com/", [], [], screens=screens, capture_status="ok"
        )
    scrape.assert_not_awaited()
    assert out.page_text_status == "ok"
    assert content.strip() == out.page_text
    assert out.html_content == html
    assert out.degradations == []
