"""demo_video stage handlers — gen / record / compile."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.clients.schemas.bdd_playwright import BddJobResult
from src.clients.schemas.video_analyzer import (
    VideoAnalyzerJobResult,
    VideoAnalyzerQualityMetrics,
)
from src.services.runs.run_context import RunContext
from src.services.stages.base import StageSkippedError
from src.services.stages.demo_video.compile import DemoVideoCompileStage
from src.services.stages.demo_video.gen import DemoVideoGenStage
from src.services.stages.demo_video.narrate import DemoVideoNarrateStage
from src.services.stages.demo_video.record import (
    DemoVideoClip,
    DemoVideoRecordOutput,
    DemoVideoRecordStage,
)
from src.services.stages.shared.mastering_gate import mastering_gate_notes
from tests.support.geometry_wire import focus_wire


def _ctx(brief: dict, prior: dict | None = None, extra: dict | None = None) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": brief, **(extra or {})},
        prior_run_outputs=prior or {},
    )


def _passing_analyzer() -> AsyncMock:
    """A VideoAnalyzerClient stand-in whose per-clip QA always passes.

    Keeps the default-on visual QA (O1) out of the record tests that don't
    exercise it — every clip's analysis reports ``verification_passed: True``.
    """
    analyzer = AsyncMock()
    analyzer.start_analysis.return_value = "va-job"
    analyzer.poll_analysis.return_value = {"verification_passed": True}
    return analyzer


# ── gen ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_gen_normalizes_parts_to_lines_keyed_by_clip_id():
    client = AsyncMock()
    client.start_demo_video.return_value = "bdd-job-1"
    client.wait_for_job.return_value = {}
    # Validated through the real response model: the fixture is the bdd wire
    # payload, so a drift in either side fails here rather than in prod.
    client.get_job_result.return_value = BddJobResult.model_validate(
        {
            "job_id": "bdd-job-1",
            "status": "completed",
            "features_generated": 2,
            "narration_script": {
                "parts": [
                    {
                        "part_id": "clip-1",
                        "scenario_name": "Login",
                        "text": "First we log in.",
                        "original_gherkin": "Scenario: Login",
                        "estimated_duration_seconds": 3.2,
                    },
                    {
                        "part_id": "clip-2",
                        "scenario_name": "Create",
                        "text": "Then we create a project.",
                        "original_gherkin": "Scenario: Create",
                        "estimated_duration_seconds": 4.0,
                    },
                ]
            },
        }
    )
    stage = DemoVideoGenStage(client=client)
    out = await stage.handle(stage.input_model(), _ctx({"target_url": "https://app.example"}))

    assert out.bdd_job_id == "bdd-job-1"
    assert [line.id for line in out.lines] == ["clip-1", "clip-2"]
    assert out.lines[0].text == "First we log in."
    assert out.lines[0].seconds == 3.2
    # feed_to_video must be False — the orchestrator owns the downstream.
    sent = client.start_demo_video.call_args.args[0]
    assert sent["feed_to_video"] is False
    assert sent["url"] == "https://app.example"
    assert sent["narration"]["enabled"] is True


def _completed_bdd_client() -> AsyncMock:
    client = AsyncMock()
    client.start_demo_video.return_value = "bdd-job-pages"
    client.wait_for_job.return_value = {}
    client.get_job_result.return_value = BddJobResult.model_validate(
        {
            "job_id": "bdd-job-pages",
            "status": "completed",
            "features_generated": 1,
            "narration_script": {
                "parts": [{"part_id": "clip-1", "text": "Sign in.", "scenario_name": "S"}]
            },
        }
    )
    return client


@pytest.mark.asyncio
async def test_gen_beside_a_login_forwards_only_pages_on_its_origin(monkeypatch):
    """target_pages is NOT part of a saved password's binding (editing it keeps the
    password), and bdd hands target_pages + the unsealed login to webscraper in one
    batch. So a tenant who adds their own site to target_pages must not get the
    password carried there: beside a login only pages on its bound origins travel."""
    from cryptography.fernet import Fernet
    from pydantic import SecretStr

    from src.config import get_settings
    from src.services.auth_secrets import seal_auth_secrets

    monkeypatch.setattr(
        get_settings().auth_secrets,
        "auth_config_secret_keys",
        SecretStr(Fernet.generate_key().decode()),
    )
    login = {"login_url": "https://a.example/login", "username": "u", "password": _FAKE_TW}
    sealed = seal_auth_secrets({"auth_config": login})["auth_config"]
    client = _completed_bdd_client()
    stage = DemoVideoGenStage(client=client)
    await stage.handle(
        stage.input_model(),
        _ctx(
            {
                "target_url": "https://a.example/app",
                "auth_config": sealed,
                "target_pages": [
                    "https://a.example/settings",
                    "/board",
                    "https://b.example/steal",
                    "//b.example/also-steal",
                    "HTTPS://B.example/x",
                ],
            }
        ),
    )
    sent = client.start_demo_video.call_args.args[0]
    assert sent["auth_config"]["password"] == _FAKE_TW  # the login still travels
    assert sent["target_pages"] == ["https://a.example/settings", "/board"]


@pytest.mark.asyncio
async def test_gen_without_a_login_forwards_target_pages_unchanged():
    pages = ["https://a.example/settings", "https://b.example/other", "/board"]
    client = _completed_bdd_client()
    stage = DemoVideoGenStage(client=client)
    await stage.handle(
        stage.input_model(),
        _ctx({"target_url": "https://a.example/app", "target_pages": pages}),
    )
    sent = client.start_demo_video.call_args.args[0]
    assert "auth_config" not in sent
    assert sent["target_pages"] == pages


@pytest.mark.asyncio
async def test_gen_carries_step_coverage_non_empty_onto_stage_output():
    """A grounding finding bdd reports must reach the persisted stage output —
    this is the boundary the orchestrator used to drop it at entirely."""
    client = AsyncMock()
    client.start_demo_video.return_value = "bdd-job-sc"
    client.wait_for_job.return_value = {}
    client.get_job_result.return_value = BddJobResult.model_validate(
        {
            "job_id": "bdd-job-sc",
            "status": "completed",
            "narration_script": {"parts": [{"part_id": "clip-1", "text": "Intro."}]},
            "step_coverage": {
                "steps_total": 4,
                "steps_stubbed": 1,
                "ungrounded_gherkin_concepts_flagged": 1,
                "ungrounded_gherkin_concepts": [
                    {
                        "file": "demo.feature",
                        "scenario": "Add a todo",
                        "step": "Then I should see the completed count",
                        "concept": "completed count",
                    }
                ],
            },
        }
    )
    stage = DemoVideoGenStage(client=client)
    out = await stage.handle(stage.input_model(), _ctx({"target_url": "https://app.example"}))

    assert out.step_coverage is not None
    assert out.step_coverage.ungrounded_gherkin_concepts_flagged == 1
    assert out.step_coverage.ungrounded_gherkin_concepts[0].concept == "completed count"
    # Round-trips through the stage-output JSON the same way (no manual field
    # picking downstream can miss).
    dumped = out.model_dump(mode="json")
    assert dumped["step_coverage"]["ungrounded_gherkin_concepts_flagged"] == 1


@pytest.mark.asyncio
async def test_gen_step_coverage_reported_clean_is_not_absent():
    """EMPTY (bdd ran grounding, flagged nothing) must persist as a real,
    zero-findings object — not collapse into the same ``None`` as ABSENT."""
    client = AsyncMock()
    client.start_demo_video.return_value = "bdd-job-clean"
    client.wait_for_job.return_value = {}
    client.get_job_result.return_value = BddJobResult.model_validate(
        {
            "job_id": "bdd-job-clean",
            "status": "completed",
            "narration_script": {"parts": [{"part_id": "clip-1", "text": "Intro."}]},
            "step_coverage": {"steps_total": 4, "steps_stubbed": 0},
        }
    )
    stage = DemoVideoGenStage(client=client)
    out = await stage.handle(stage.input_model(), _ctx({"target_url": "https://app.example"}))

    assert out.step_coverage is not None
    assert out.step_coverage.ungrounded_gherkin_concepts_flagged == 0
    assert out.step_coverage.ungrounded_gherkin_concepts == []
    dumped = out.model_dump(mode="json")
    assert dumped["step_coverage"] is not None
    assert dumped["step_coverage"]["ungrounded_gherkin_concepts"] == []


@pytest.mark.asyncio
async def test_gen_step_coverage_absent_when_bdd_never_reports_it():
    """The DB column empty (pre-field job, or bdd's falsy-on-the-wire ``{}``
    shape) must persist as ``None`` — ABSENT, distinguishable from a clean
    report."""
    client = AsyncMock()
    client.start_demo_video.return_value = "bdd-job-none"
    client.wait_for_job.return_value = {}
    client.get_job_result.return_value = BddJobResult.model_validate(
        {
            "job_id": "bdd-job-none",
            "status": "completed",
            "narration_script": {"parts": [{"part_id": "clip-1", "text": "Intro."}]},
        }
    )
    stage = DemoVideoGenStage(client=client)
    out = await stage.handle(stage.input_model(), _ctx({"target_url": "https://app.example"}))

    assert out.step_coverage is None
    dumped = out.model_dump(mode="json")
    assert dumped["step_coverage"] is None


@pytest.mark.asyncio
async def test_gen_normalizes_at_prefixed_generated_part_id():
    """Defensive: a bdd part_id arriving as ``@clip-<n>`` is normalized to the
    ``clip-<n>`` join key — symmetric with the authored path — so record keys the
    footage under the same tag FTR records it by."""
    client = AsyncMock()
    client.start_demo_video.return_value = "bdd-job-at"
    client.wait_for_job.return_value = {}
    client.get_job_result.return_value = BddJobResult.model_validate(
        {
            "job_id": "bdd-job-at",
            "status": "completed",
            "narration_script": {"parts": [{"part_id": "@clip-1", "text": "Intro."}]},
        }
    )
    stage = DemoVideoGenStage(client=client)
    out = await stage.handle(stage.input_model(), _ctx({"target_url": "https://app.example"}))
    assert [line.id for line in out.lines] == ["clip-1"]


@pytest.mark.asyncio
async def test_gen_requires_target_url():
    stage = DemoVideoGenStage(client=AsyncMock())
    with pytest.raises(RuntimeError, match="target_url"):
        await stage.handle(stage.input_model(), _ctx({}))


@pytest.mark.asyncio
async def test_gen_authored_mode_uses_brief_lines_and_skips_bdd():
    """brief.lines present → authored mode: no bdd call, package_url passed through."""
    client = AsyncMock()
    stage = DemoVideoGenStage(client=client)
    brief = {
        "target_url": "https://app.example",
        "package_url": "https://minio.test/specra/pkg.zip",
        "lines": [
            {"id": "clip-1", "text": "Welcome to SpecraLabs."},
            {"id": "clip-2", "text": "Twelve rooms, automatically?", "seconds": 4.0},
        ],
    }
    out = await stage.handle(stage.input_model(), _ctx(brief))

    client.start_demo_video.assert_not_called()
    assert [line.id for line in out.lines] == ["clip-1", "clip-2"]
    assert out.lines[1].seconds == 4.0
    assert out.bdd_job_id == ""
    assert out.package_url == "https://minio.test/specra/pkg.zip"
    # No dialogue flag → speakers stay unset (single narrator).
    assert all(line.speaker is None for line in out.lines)


@pytest.mark.asyncio
async def test_gen_authored_requires_some_text():
    stage = DemoVideoGenStage(client=AsyncMock())
    with pytest.raises(RuntimeError, match="no narration text"):
        await stage.handle(stage.input_model(), _ctx({"lines": [{"id": "clip-1", "text": ""}]}))


@pytest.mark.asyncio
async def test_gen_casts_explicit_speakers_when_dialogue_enabled():
    """Authored role labels (guide/new) map onto cast slots a/b and are preserved."""
    stage = DemoVideoGenStage(client=AsyncMock())
    brief = {
        "package_url": "https://minio.test/pkg.zip",
        "dialogue": {"enabled": True},
        "lines": [
            {"id": "clip-1", "text": "This is SpecraLabs.", "speaker": "guide"},
            {"id": "clip-2", "text": "Where do I start?", "speaker": "new"},
        ],
    }
    out = await stage.handle(stage.input_model(), _ctx(brief))
    assert [line.speaker for line in out.lines] == ["a", "b"]


@pytest.mark.asyncio
async def test_gen_derives_speakers_by_heuristic_when_unset():
    """dialogue.enabled with no explicit speakers → question→a, statement→b."""
    stage = DemoVideoGenStage(client=AsyncMock())
    brief = {
        "package_url": "https://minio.test/pkg.zip",
        "dialogue": {"enabled": True},
        "lines": [
            {"id": "clip-1", "text": "Twelve rooms, automatically?"},
            {"id": "clip-2", "text": "Twelve rooms, detected for you."},
        ],
    }
    out = await stage.handle(stage.input_model(), _ctx(brief))
    assert [line.speaker for line in out.lines] == ["a", "b"]


@pytest.mark.asyncio
async def test_gen_raises_when_bdd_job_failed():
    client = AsyncMock()
    client.start_demo_video.return_value = "bdd-job-2"
    client.wait_for_job.return_value = {}
    client.get_job_result.return_value = BddJobResult.model_validate(
        {"job_id": "bdd-job-2", "status": "failed", "error": "boom"}
    )
    stage = DemoVideoGenStage(client=client)
    with pytest.raises(RuntimeError, match="boom"):
        await stage.handle(stage.input_model(), _ctx({"target_url": "https://x"}))


# ── record ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_record_opts_out_when_record_disabled():
    """`record.enabled: false` in config skips the capture branch clean (raises
    StageSkippedError) BEFORE any FTR/gen work — so a plan-only run (e.g. the
    claim_finish e2e proof) reaches its plan stages without the record sibling
    hard-failing the whole run on the non-deterministic bddgen/capture path."""
    client = AsyncMock()
    stage = DemoVideoRecordStage(client=client, analyzer=_passing_analyzer())
    with pytest.raises(StageSkippedError):
        await stage.handle(
            stage.input_model(),
            _ctx(
                {"target_url": "https://app.example"},
                prior={"demo_video_gen": {"bdd_job_id": "b", "lines": [{"id": "clip-1"}]}},
                extra={"record": {"enabled": False}},
            ),
        )
    # short-circuits before dispatching any FTR job
    client.start_execution.assert_not_called()


@pytest.mark.asyncio
async def test_record_runs_when_record_block_absent_or_enabled():
    """Default (no `record` block) and explicit `enabled: true` both still record —
    the opt-out must not change the default capture behaviour."""
    for extra in (None, {"record": {"enabled": True}}):
        client = AsyncMock()
        client.start_execution.return_value = "ftr-job"
        client.wait_for_execution.return_value = {}
        client.get_job_result.return_value = {
            "scenario_clips": [{"clip_key": "clip-1", "duration_ms": 1000}]
        }
        client.get_clip_video.return_value = b"mp4bytes"
        stage = DemoVideoRecordStage(client=client, analyzer=_passing_analyzer())
        with patch(
            "src.services.stages.demo_video.record.store_capture_clip",
            new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['clip_id']}.mp4"),
        ):
            out = await stage.handle(
                stage.input_model(),
                _ctx(
                    {"target_url": "https://app.example"},
                    prior={"demo_video_gen": {"bdd_job_id": "b", "lines": [{"id": "clip-1"}]}},
                    extra=extra,
                ),
            )
        assert [c.line_id for c in out.clips] == ["clip-1"]
        client.start_execution.assert_called_once()


@pytest.mark.asyncio
async def test_record_passes_stable_idempotency_key():
    """start_execution is called with a run_id:stage Idempotency-Key (stable across a
    resumable-timeout re-dispatch) so a retried record re-uses the ORIGINAL FTR job
    instead of spawning a second paid bddgen+Playwright recording."""
    client = AsyncMock()
    client.start_execution.return_value = "ftr-job"
    client.wait_for_execution.return_value = {}
    client.get_job_result.return_value = {
        "scenario_clips": [{"clip_key": "clip-1", "duration_ms": 1000}]
    }
    client.get_clip_video.return_value = b"mp4"
    stage = DemoVideoRecordStage(client=client, analyzer=_passing_analyzer())
    rid = uuid4()
    ctx = RunContext(
        run_id=rid,
        deliverable_id=uuid4(),
        config_snapshot={"brief": {"target_url": "https://app.example"}},
        prior_run_outputs={"demo_video_gen": {"bdd_job_id": "b", "lines": [{"id": "clip-1"}]}},
    )
    with patch(
        "src.services.stages.demo_video.record.store_capture_clip",
        new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['clip_id']}.mp4"),
    ):
        await stage.handle(stage.input_model(), ctx)
    assert client.start_execution.call_args.kwargs["idempotency_key"] == f"{rid}:demo_video_record"


@pytest.mark.asyncio
async def test_record_keeps_only_clips_matching_narration_lines():
    client = AsyncMock()
    client.start_execution.return_value = "ftr-job-1"
    client.wait_for_execution.return_value = {}
    client.get_job_result.return_value = {
        "scenario_clips": [
            {"clip_key": "clip-1", "duration_ms": 2000},
            {"clip_key": "clip-2", "duration_ms": 3000},
            # Aux clip (e.g. auth setup) with no narration line → dropped.
            {"clip_key": "login-setup__authenticate", "duration_ms": 1000},
        ]
    }
    client.get_clip_video.return_value = b"mp4bytes"

    gen = {
        "bdd_job_id": "bdd-job-1",
        "lines": [{"id": "clip-1"}, {"id": "clip-2"}],
    }
    stage = DemoVideoRecordStage(client=client, analyzer=_passing_analyzer())
    with patch(
        "src.services.stages.demo_video.record.store_capture_clip",
        new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['clip_id']}.mp4"),
    ):
        out = await stage.handle(
            stage.input_model(),
            _ctx({"target_url": "https://app.example"}, prior={"demo_video_gen": gen}),
        )

    assert [c.line_id for c in out.clips] == ["clip-1", "clip-2"]
    assert out.clips[0].video_url == "https://minio/clip-1.mp4"
    assert out.clips[0].seconds == 2.0
    # FTR is handed the package by bdd_job_id, not a package_url.
    sent = client.start_execution.call_args.args[0]
    assert sent["bdd_job_id"] == "bdd-job-1"
    assert sent["target_url"] == "https://app.example"
    # A bounded per-test timeout is forwarded so broken scenarios fail fast.
    assert sent["timeout_per_test_ms"] == 180_000


@pytest.mark.asyncio
async def test_record_notes_line_clip_with_no_recorded_footage():
    """A narration line referencing a clip tag FTR never recorded gets a per-line note
    (not just the all-miss note) while the matched clips still record — so a single
    mistyped/unrecorded ref among many doesn't vanish silently into a text card."""
    client = AsyncMock()
    client.start_execution.return_value = "ftr-job-miss"
    client.wait_for_execution.return_value = {}
    client.get_job_result.return_value = {
        "scenario_clips": [
            {"clip_key": "clip-1", "duration_ms": 2000},
            # clip-9 is referenced below but FTR never produced footage for it.
        ]
    }
    client.get_clip_video.return_value = b"mp4bytes"

    gen = {
        "bdd_job_id": "bdd-job-miss",
        "lines": [{"id": "clip-1"}, {"id": "clip-9"}],
    }
    stage = DemoVideoRecordStage(client=client, analyzer=_passing_analyzer())
    with patch(
        "src.services.stages.demo_video.record.store_capture_clip",
        new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['clip_id']}.mp4"),
    ):
        out = await stage.handle(
            stage.input_model(),
            _ctx({"target_url": "https://app.example"}, prior={"demo_video_gen": gen}),
        )

    # The matched clip still records; the run is not aborted.
    assert [c.line_id for c in out.clips] == ["clip-1"]
    # The unrecorded ref is surfaced by name, and it is NOT the all-miss note.
    assert any("clip-9" in n and "no matching footage" in n for n in out.notes)
    assert not any("no recorded scenario clips matched" in n for n in out.notes)


# WebM/Matroska EBML header; MP4 carries `ftyp` at offset 4.
_WEBM_BYTES = b"\x1a\x45\xdf\xa3" + b"\x00" * 32
_MP4_BYTES = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 32


async def _record_one_clip(video_bytes: bytes) -> AsyncMock:
    """Run the record stage over a single clip whose downloaded body is
    ``video_bytes``; return the patched ``store_capture_clip`` mock."""
    client = AsyncMock()
    client.start_execution.return_value = "ftr-job-fmt"
    client.wait_for_execution.return_value = {}
    client.get_job_result.return_value = {
        "scenario_clips": [{"clip_key": "clip-1", "duration_ms": 2000}]
    }
    client.get_clip_video.return_value = video_bytes
    gen = {"bdd_job_id": "bdd-job-fmt", "lines": [{"id": "clip-1"}]}
    stage = DemoVideoRecordStage(client=client, analyzer=_passing_analyzer())
    store = AsyncMock(
        side_effect=lambda **kw: f"https://minio/{kw['clip_id']}.{kw['video_format']}"
    )
    with patch("src.services.stages.demo_video.record.store_capture_clip", new=store):
        await stage.handle(
            stage.input_model(),
            _ctx({"target_url": "https://app.example"}, prior={"demo_video_gen": gen}),
        )
    return store


@pytest.mark.asyncio
async def test_record_warns_when_clip_is_not_mp4(caplog):
    """A non-mp4 container is a visible signal that FTR's conversion is broken."""
    with caplog.at_level(logging.WARNING, logger="src.services.stages.demo_video.record"):
        await _record_one_clip(_WEBM_BYTES)
    assert any(
        "clip-1" in r.getMessage() and "webm" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("brief_value", "expected"),
    [(300_000, 300_000)],
)
async def test_record_honors_brief_test_timeout(brief_value, expected):
    """brief.timeout_per_test_ms overrides the 180s default, clamped to FTR bounds."""
    client = AsyncMock()
    client.start_execution.return_value = "ftr-job-t"
    client.wait_for_execution.return_value = {}
    client.get_job_result.return_value = {
        "scenario_clips": [{"clip_key": "clip-1", "duration_ms": 1000}]
    }
    client.get_clip_video.return_value = b"mp4"
    gen = {"bdd_job_id": "bdd-job-t", "lines": [{"id": "clip-1"}]}
    brief: dict = {"target_url": "https://app.example"}
    if brief_value is not None:
        brief["timeout_per_test_ms"] = brief_value
    stage = DemoVideoRecordStage(client=client, analyzer=_passing_analyzer())
    with patch(
        "src.services.stages.demo_video.record.store_capture_clip",
        new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['clip_id']}.mp4"),
    ):
        await stage.handle(
            stage.input_model(),
            _ctx(brief, prior={"demo_video_gen": gen}),
        )
    sent = client.start_execution.call_args.args[0]
    assert sent["timeout_per_test_ms"] == expected


@pytest.mark.asyncio
async def test_record_uses_package_url_when_present():
    """Authored mode: brief.package_url is sent to FTR in place of a bdd_job_id."""
    client = AsyncMock()
    client.start_execution.return_value = "ftr-job-pkg"
    client.wait_for_execution.return_value = {}
    client.get_job_result.return_value = {
        "scenario_clips": [{"clip_key": "clip-1", "duration_ms": 9}]
    }
    client.get_clip_video.return_value = b"mp4"
    gen = {"bdd_job_id": "", "lines": [{"id": "clip-1"}]}
    stage = DemoVideoRecordStage(client=client, analyzer=_passing_analyzer())
    with patch(
        "src.services.stages.demo_video.record.store_capture_clip",
        new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['clip_id']}.mp4"),
    ):
        await stage.handle(
            stage.input_model(),
            _ctx(
                {
                    "target_url": "https://app.example",
                    "package_url": "https://minio.test/specra/pkg.zip",
                },
                prior={"demo_video_gen": gen},
            ),
        )
    sent = client.start_execution.call_args.args[0]
    assert sent["package_url"] == "https://minio.test/specra/pkg.zip"
    assert "bdd_job_id" not in sent


@pytest.mark.asyncio
async def test_record_keeps_clip_tags_referenced_by_lines():
    """Showcase: footage lines reference a clip tag via `clip` (line id != tag).
    record must keep clips by referenced TAG, output them keyed by that tag."""
    client = AsyncMock()
    client.start_execution.return_value = "ftr-job-clip"
    client.wait_for_execution.return_value = {}
    client.get_job_result.return_value = {
        "scenario_clips": [
            {"clip_key": "clip-1", "duration_ms": 9000},
            # An aux clip no line references → dropped.
            {"clip_key": "setup__auth", "duration_ms": 500},
        ]
    }
    client.get_clip_video.return_value = b"mp4"
    # Two dialogue lines (ids dash-q/dash-a) both play over clip-1; a card line has
    # no clip. None of the line ids equal the recorded tag "clip-1".
    gen = {
        "bdd_job_id": "",
        "lines": [
            {"id": "intro", "card": {"template": "title"}},
            {"id": "dash-q", "clip": "clip-1"},
            {"id": "dash-a", "clip": "clip-1"},
        ],
    }
    stage = DemoVideoRecordStage(client=client, analyzer=_passing_analyzer())
    with patch(
        "src.services.stages.demo_video.record.store_capture_clip",
        new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['clip_id']}.mp4"),
    ):
        out = await stage.handle(
            stage.input_model(),
            _ctx(
                {"target_url": "https://x", "package_url": "https://minio.test/pkg.zip"},
                prior={"demo_video_gen": gen},
            ),
        )
    # clip-1 kept (referenced by dash-q/dash-a), keyed by its tag; aux dropped.
    assert [c.line_id for c in out.clips] == ["clip-1"]
    assert out.clips[0].video_url == "https://minio/clip-1.mp4"


@pytest.mark.asyncio
async def test_record_degrades_when_no_footage_source():
    """Authored VO spine with neither a package_url nor a bdd_job_id has nothing to
    record, so the stage DEGRADES (empty clips + footage note) instead of hard-failing
    the whole run — reconcile_timeline still runs. FTR is never called (no source).

    This must survive the total-footage-loss gate (default demo_require_footage=True):
    the gate only fires PAST this branch, where a paid capture actually happened."""
    client = AsyncMock()
    gen = {"bdd_job_id": "", "lines": [{"id": "clip-1"}]}
    stage = DemoVideoRecordStage(client=client, analyzer=_passing_analyzer())
    out = await stage.handle(
        stage.input_model(),
        _ctx({"target_url": "https://x"}, prior={"demo_video_gen": gen}),
    )
    assert out.clips == []
    assert out.notes and "no footage source" in out.notes[0]
    client.start_execution.assert_not_called()


@pytest.mark.asyncio
async def test_record_hard_fails_when_paid_source_yields_zero_clips():
    """TOTAL footage loss on a PAID source HARD-FAILS (run lands status=failed; there
    is no refund path — reconcile_timeline.py:1006).

    FTR answers status='completed' with an EMPTY scenario_clips when its Playwright
    run times out and salvages nothing, so wait_for_execution succeeds. Before this
    gate the stage returned empty clips + a soft note, compile rendered narrated text
    cards, and the run landed `completed`+`degraded` — billable for zero footage.
    The message must name the FTR job so the loss is attributable.
    """
    client = AsyncMock()
    client.start_execution.return_value = "ftr-job-2"
    client.wait_for_execution.return_value = {}
    client.get_job_result.return_value = {"scenario_clips": []}
    gen = {"bdd_job_id": "b", "lines": [{"id": "clip-1"}]}
    stage = DemoVideoRecordStage(client=client, analyzer=_passing_analyzer())
    with pytest.raises(RuntimeError, match="recorded no usable footage") as excinfo:
        await stage.handle(
            stage.input_model(),
            _ctx({"target_url": "https://x"}, prior={"demo_video_gen": gen}),
        )
    assert "ftr-job-2" in str(excinfo.value)


@pytest.mark.asyncio
async def test_record_degrades_when_ftr_returned_clips_but_tags_never_matched():
    """A tag-naming mismatch is a MAPPING defect, not a capture failure.

    FTR recorded real footage, but under keys no narration line references, so every
    clip is filtered out by `valid_clip_tags`. The footage demonstrably exists — the
    gate must NOT make that unrecoverable. It stays a soft degrade to text cards, as
    it was before the gate existed. This is why the gate keys on `returned_clips == 0`
    rather than on an empty `clips` list.
    """
    client = AsyncMock()
    client.start_execution.return_value = "ftr-job-mismatch"
    client.wait_for_execution.return_value = {}
    client.get_job_result.return_value = {
        "scenario_clips": [
            {"clip_key": "scenario-1", "duration_ms": 1000},
            {"clip_key": "scenario-2", "duration_ms": 1000},
        ]
    }
    gen = {"bdd_job_id": "b", "lines": [{"id": "clip-1"}, {"id": "clip-2"}]}
    stage = DemoVideoRecordStage(client=client, analyzer=_passing_analyzer())
    out = await stage.handle(
        stage.input_model(),
        _ctx({"target_url": "https://x"}, prior={"demo_video_gen": gen}),
    )
    assert out.clips == []
    assert any("no matching footage" in n for n in out.notes)


@pytest.mark.asyncio
async def test_record_zero_clip_failure_names_the_ftr_error():
    """The zero-clip raise must carry FTR's OWN reason, not just a count.

    With no `scenario_clips` there is no clip to hang a `_ClipDiag` on, so #719's
    blank-clip post-mortem never fires — but `test_results[].error` still holds the
    upstream cause. The identical error repeated across scenarios is deduped, so one
    nav-hook timeout is named once, not N times.
    """
    client = AsyncMock()
    client.start_execution.return_value = "ftr-job-diag"
    client.wait_for_execution.return_value = {}
    client.get_job_result.return_value = {
        "scenario_clips": [],
        "test_results": [
            {"feature": "checkout", "name": "adds an item", "error": "nav hook: no paint"},
            {"feature": "checkout", "name": "pays", "error": "nav hook: no paint"},
        ],
    }
    gen = {"bdd_job_id": "b", "lines": [{"id": "clip-1"}]}
    stage = DemoVideoRecordStage(client=client, analyzer=_passing_analyzer())
    with pytest.raises(RuntimeError, match="recorded no usable footage") as excinfo:
        await stage.handle(
            stage.input_model(),
            _ctx({"target_url": "https://x"}, prior={"demo_video_gen": gen}),
        )
    message = str(excinfo.value)
    assert "nav hook: no paint" in message
    # Deduped on the error text: named once, attributed to the first scenario that hit it.
    assert message.count("nav hook: no paint") == 1
    assert "checkout/adds an item" in message
    assert "checkout/pays" not in message


@pytest.mark.asyncio
async def test_record_hard_fails_on_zero_clips_from_a_package_url_source():
    """The gate is source-agnostic: an AUTHORED package_url capture that yields zero
    clips is just as paid-for as a bdd_job_id one, so it hard-fails too."""
    client = AsyncMock()
    client.start_execution.return_value = "ftr-job-pkg-empty"
    client.wait_for_execution.return_value = {}
    client.get_job_result.return_value = {"scenario_clips": []}
    gen = {"bdd_job_id": "", "lines": [{"id": "clip-1"}]}
    stage = DemoVideoRecordStage(client=client, analyzer=_passing_analyzer())
    with pytest.raises(RuntimeError, match="recorded no usable footage"):
        await stage.handle(
            stage.input_model(),
            _ctx(
                {"target_url": "https://x", "package_url": "https://minio.test/pkg.zip"},
                prior={"demo_video_gen": gen},
            ),
        )


@pytest.mark.asyncio
async def test_record_notes_when_no_clips_match_with_require_footage_off():
    """PIPELINE_DEMO_REQUIRE_FOOTAGE=false restores the pre-gate degrade: empty clips
    plus the all-miss soft note, so compile falls back to narrated text cards."""
    client = AsyncMock()
    client.start_execution.return_value = "ftr-job-2"
    client.wait_for_execution.return_value = {}
    client.get_job_result.return_value = {"scenario_clips": []}
    gen = {"bdd_job_id": "b", "lines": [{"id": "clip-1"}]}
    stage = DemoVideoRecordStage(client=client, analyzer=_passing_analyzer())
    with patch(
        "src.services.stages.demo_video.record.get_settings",
        return_value=_qa_settings(demo_require_footage=False),
    ):
        out = await stage.handle(
            stage.input_model(),
            _ctx({"target_url": "https://x"}, prior={"demo_video_gen": gen}),
        )
    assert out.clips == []
    assert out.notes and "no recorded scenario clips" in out.notes[0]


@pytest.mark.asyncio
async def test_record_partial_harvest_still_succeeds():
    """Only TOTAL loss hard-fails: one recorded clip + one missing tag keeps the stage
    successful with a soft per-tag footage-gap note (the degrade band is preserved)."""
    client = AsyncMock()
    client.start_execution.return_value = "ftr-job-partial"
    client.wait_for_execution.return_value = {}
    client.get_job_result.return_value = {
        "scenario_clips": [{"clip_key": "clip-1", "duration_ms": 2000}]
    }
    client.get_clip_video.return_value = b"mp4bytes"
    gen = {"bdd_job_id": "b", "lines": [{"id": "clip-1"}, {"id": "clip-2"}]}
    stage = DemoVideoRecordStage(client=client, analyzer=_passing_analyzer())
    with patch(
        "src.services.stages.demo_video.record.store_capture_clip",
        new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['clip_id']}.mp4"),
    ):
        out = await stage.handle(
            stage.input_model(),
            _ctx({"target_url": "https://x"}, prior={"demo_video_gen": gen}),
        )
    assert [c.line_id for c in out.clips] == ["clip-1"]
    assert any("clip-2" in n and "no matching footage" in n for n in out.notes)
    assert not any("no usable footage" in n for n in out.notes)


@pytest.mark.asyncio
async def test_record_forwards_auth_config_to_ftr():
    """Login-gated demos: brief.auth_config is forwarded to the FTR run so the
    recorder performs the storageState form login (the wizard control writes it)."""
    client = AsyncMock()
    client.start_execution.return_value = "ftr-job-3"
    client.wait_for_execution.return_value = {}
    client.get_job_result.return_value = {
        "scenario_clips": [{"clip_key": "clip-1", "duration_ms": 1000}]
    }
    client.get_clip_video.return_value = b"mp4"
    gen = {"bdd_job_id": "b", "lines": [{"id": "clip-1"}]}
    auth = {"type": "form", "login_url": "https://x/login", "username": "u", "password": "p"}
    stored = {**auth, "signs_into": "https://app.x", "needs_reentry": True}
    stage = DemoVideoRecordStage(client=client, analyzer=_passing_analyzer())
    with patch(
        "src.services.stages.demo_video.record.store_capture_clip",
        new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['clip_id']}.mp4"),
    ):
        await stage.handle(
            stage.input_model(),
            _ctx(
                {"target_url": "https://x", "auth_config": stored},
                prior={"demo_video_gen": gen},
            ),
        )
    # The single-login (dict) shape is stripped of orchestrator-only keys too.
    assert client.start_execution.call_args.args[0]["auth_config"] == auth
    assert client.start_execution.call_args.args[0]["trace_on_failure"] is False


_FAKE_KC = "kc-pw"  # pragma: allowlist secret
_FAKE_TW = "tw-pw"  # pragma: allowlist secret


@pytest.mark.asyncio
async def test_record_unseals_a_multi_login_auth_config_for_ftr(monkeypatch):
    """The brief stores login secrets SEALED (auth_secrets); FTR must receive the
    plaintext, one login per origin, in the list shape the brief carries. Forwarding
    the sealed blob would type a ciphertext into every login form."""
    from cryptography.fernet import Fernet
    from pydantic import SecretStr

    from src.config import get_settings
    from src.services.auth_secrets import seal_auth_secrets

    monkeypatch.setattr(
        get_settings().auth_secrets,
        "auth_config_secret_keys",
        SecretStr(Fernet.generate_key().decode()),
    )
    plain = [
        {
            "login_url": "https://marketing.example/strategy",
            "username": "kc",
            "password": _FAKE_KC,
        },
        {
            "login_url": "https://twenty.example/",
            "username": "tw@x",
            "password": _FAKE_TW,
        },
    ]
    # Orchestrator-only keys on the stored logins must never reach FTR.
    stored = [
        {**plain[0], "signs_into": "https://app.marketing.example"},
        {**plain[1], "needs_reentry": True},
    ]
    sealed = seal_auth_secrets({"auth_config": stored})["auth_config"]
    assert _FAKE_KC not in str(sealed)
    client = AsyncMock()
    client.start_execution.return_value = "ftr-job-4"
    client.wait_for_execution.return_value = {}
    client.get_job_result.return_value = {
        "scenario_clips": [{"clip_key": "clip-1", "duration_ms": 1000}]
    }
    client.get_clip_video.return_value = b"mp4"
    gen = {"bdd_job_id": "b", "lines": [{"id": "clip-1"}]}
    stage = DemoVideoRecordStage(client=client, analyzer=_passing_analyzer())
    with patch(
        "src.services.stages.demo_video.record.store_capture_clip",
        new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['clip_id']}.mp4"),
    ):
        await stage.handle(
            stage.input_model(),
            _ctx(
                {"target_url": "https://marketing.example", "auth_config": sealed},
                prior={"demo_video_gen": gen},
            ),
        )
    sent = client.start_execution.call_args.args[0]
    assert sent["auth_config"] == plain, "signs_into / needs_reentry are never forwarded"
    # A trace would record fill(password) in plaintext and ship it to object storage.
    assert sent["trace_on_failure"] is False


@pytest.mark.asyncio
async def test_record_repolls_when_a_poll_times_out():
    """A large suite can run past the 600s per-poll cap; the wait is re-polled on
    TimeoutError (the FTR job keeps running) instead of failing the stage."""
    client = AsyncMock()
    client.start_execution.return_value = "ftr-job-4"
    # First poll times out (suite still recording), second poll returns.
    client.wait_for_execution.side_effect = [TimeoutError("poll deadline"), {}]
    client.get_job_result.return_value = {
        "scenario_clips": [{"clip_key": "clip-1", "duration_ms": 1000}]
    }
    client.get_clip_video.return_value = b"mp4"
    gen = {"bdd_job_id": "b", "lines": [{"id": "clip-1"}]}
    stage = DemoVideoRecordStage(client=client, analyzer=_passing_analyzer())
    with patch(
        "src.services.stages.demo_video.record.store_capture_clip",
        new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['clip_id']}.mp4"),
    ):
        out = await stage.handle(
            stage.input_model(),
            _ctx({"target_url": "https://x"}, prior={"demo_video_gen": gen}),
        )
    assert client.wait_for_execution.await_count == 2  # re-polled once
    assert client.get_job_result.await_count == 1
    assert [c.line_id for c in out.clips] == ["clip-1"]


@pytest.mark.asyncio
async def test_record_real_failure_is_not_retried():
    """A genuine FTR job failure (non-timeout) on a run whose brief ASKED for footage
    (a bdd_job_id is present) propagates immediately and HARD-FAILS the stage — retry_on=()
    means only timeouts re-poll. This is deliberately NOT degraded: an expected footage
    loss must fail the run rather than ship a footage-free video as `completed`. No
    refund follows (reconcile_timeline.py:1006) — the spend is forfeited, which is the
    deliberate trade (see the narrow no-source degrade in handle(), which covers only
    the "nothing was ever recordable" case)."""
    client = AsyncMock()
    client.start_execution.return_value = "ftr-job-5"
    client.wait_for_execution.side_effect = RuntimeError("FTR job failed: npm install error")
    gen = {"bdd_job_id": "b", "lines": [{"id": "clip-1"}]}
    stage = DemoVideoRecordStage(client=client, analyzer=_passing_analyzer())
    with pytest.raises(RuntimeError, match="FTR job failed"):
        await stage.handle(
            stage.input_model(),
            _ctx({"target_url": "https://x"}, prior={"demo_video_gen": gen}),
        )
    assert client.wait_for_execution.await_count == 1  # NOT retried
    client.get_job_result.assert_not_awaited()


# ── narrate (multi-voice casting) ──────────────────────────────────────────


def test_narrate_voice_for_casts_by_speaker_when_dialogue_enabled():
    stage = DemoVideoNarrateStage(client=AsyncMock())
    voice = {"voice_id": "af_solo"}
    ctx = _ctx({"dialogue": {"enabled": True}})
    assert stage.voice_for({"speaker": "a", "text": "Hi?"}, voice, ctx) == "af_heart"
    assert stage.voice_for({"speaker": "b", "text": "Hi."}, voice, ctx) == "am_michael"


def test_narrate_voice_for_single_voice_when_dialogue_disabled():
    """No brief.dialogue.enabled → the single configured voice, speaker ignored."""
    stage = DemoVideoNarrateStage(client=AsyncMock())
    voice = {"voice_id": "af_solo"}
    ctx = _ctx({})
    assert stage.voice_for({"speaker": "a", "text": "Hi?"}, voice, ctx) == "af_solo"


def test_narrate_voice_for_honors_cast_override():
    stage = DemoVideoNarrateStage(client=AsyncMock())
    voice = {"voice_id": "af_solo", "cast": {"a": "am_michael", "b": "af_heart"}}
    ctx = _ctx({"dialogue": {"enabled": True}})
    assert stage.voice_for({"speaker": "a", "text": "Hi?"}, voice, ctx) == "am_michael"


# ── narrate (per-line VO rate, C13) ────────────────────────────────────────


def test_narrate_speed_for_uses_the_lines_vo_rate():
    """reconcile_timeline stamped a vo_rate → that becomes the TTS speed."""
    stage = DemoVideoNarrateStage(client=AsyncMock())
    ctx = _ctx({})
    assert stage.speed_for({"id": "clip-1", "text": "Hi.", "vo_rate": 1.08}, ctx) == 1.08
    # Ints/strings from a JSON plan coerce to float.
    assert stage.speed_for({"id": "clip-1", "text": "Hi.", "vo_rate": 1}, ctx) == 1.0


def test_narrate_speed_for_is_none_without_a_vo_rate():
    """Footage/card beats, and any run where reconcile_timeline skipped, carry no
    vo_rate → None → `speed` is omitted entirely (identical clip id + audio)."""
    stage = DemoVideoNarrateStage(client=AsyncMock())
    ctx = _ctx({})
    assert stage.speed_for({"id": "clip-1", "text": "Hi."}, ctx) is None
    assert stage.speed_for({"id": "clip-1", "text": "Hi.", "vo_rate": None}, ctx) is None


# ── narrate speed precedence (ORCH-1.4: author → vo_rate → rate_hint → 1.0) ──


def _vd(directives: list[dict]) -> dict:
    """A prior_run_outputs map whose voice_director carries a VoiceDirectorOutput-
    shaped dict (``{"directives": [PerformanceDirective, ...]}``)."""
    return {"voice_director": {"directives": directives}}


def test_narrate_speed_for_author_rate_wins_over_vo_rate_and_rate_hint():
    """The AuthoredLine per-line `rate` beats a reconcile vo_rate AND a
    voice_director rate_hint — author override is top of the precedence."""
    stage = DemoVideoNarrateStage(client=AsyncMock())
    ctx = _ctx({}, _vd([{"line_id": "clip-1", "rate_hint": 1.1}]))
    item = {"id": "clip-1", "text": "Hi.", "rate": 1.3, "vo_rate": 0.9}
    assert stage.speed_for(item, ctx) == 1.3


def test_narrate_speed_for_vo_rate_wins_over_rate_hint_without_author_rate():
    """No author rate → the reconcile vo_rate beats the voice_director rate_hint."""
    stage = DemoVideoNarrateStage(client=AsyncMock())
    ctx = _ctx({}, _vd([{"line_id": "clip-1", "rate_hint": 1.1}]))
    item = {"id": "clip-1", "text": "Hi.", "vo_rate": 0.9}
    assert stage.speed_for(item, ctx) == 0.9


def test_narrate_speed_for_falls_back_to_directive_rate_hint():
    """No author rate, no vo_rate → the voice_director's per-line rate_hint governs."""
    stage = DemoVideoNarrateStage(client=AsyncMock())
    ctx = _ctx({}, _vd([{"line_id": "clip-1", "rate_hint": 1.1}]))
    item = {"id": "clip-1", "text": "Hi."}
    assert stage.speed_for(item, ctx) == 1.1


def test_narrate_speed_for_neutral_when_director_ran_but_no_usable_rate():
    """voice_director ran but this line's directive carries no rate_hint (and no
    author/vo rate) → the neutral 1.0, not None."""
    stage = DemoVideoNarrateStage(client=AsyncMock())
    ctx = _ctx({}, _vd([{"line_id": "clip-1", "rate_hint": None}]))
    item = {"id": "clip-1", "text": "Hi."}
    assert stage.speed_for(item, ctx) == 1.0


def test_narrate_speed_for_absent_voice_director_preserves_vo_rate_path():
    """No voice_director output AND no author rate ⇒ byte-identical to today's
    vo_rate-only path (AC9): the vo_rate when present, None when absent."""
    stage = DemoVideoNarrateStage(client=AsyncMock())
    ctx = _ctx({})  # prior={} → prior_output("voice_director") is None
    assert stage.speed_for({"id": "clip-1", "text": "Hi.", "vo_rate": 0.9}, ctx) == 0.9
    assert stage.speed_for({"id": "clip-1", "text": "Hi."}, ctx) is None


# ── compile ──────────────────────────────────────────────────────────────


def test_compile_skips_clean_when_record_opted_out():
    """`record.enabled: false` (plan-only opt-out) → demo_video_record skip-cleans,
    so compile has no footage. build_shots must SKIP CLEAN (StageSkippedError), not
    RuntimeError, even if a run reaches compile without pausing first — making the
    opt-out safe unconditionally (regression guard for the record-optout footgun)."""
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        # record opted out; record output is absent/empty as it would be after a skip
        config_snapshot={"record": {"enabled": False}, "deliverable": {"captions_mode": "none"}},
        prior_run_outputs={
            "demo_video_gen": {"lines": [{"id": "clip-1", "text": "x", "seconds": 2.0}]},
            "demo_video_narrate": {"clips": []},
            "demo_video_record": {},
        },
    )
    with pytest.raises(StageSkippedError):
        DemoVideoCompileStage().build_shots(ctx)


def test_compile_build_shots_joins_video_and_narration_by_line_id():
    gen = {
        "lines": [
            {"id": "clip-1", "text": "Log in", "seconds": 2.0},
            {"id": "clip-2", "text": "Create", "seconds": 2.0},
        ]
    }
    narrate = {
        "clips": [
            {"id": "n1", "line_id": "clip-1", "artifact_url": "https://a/1.wav", "seconds": 5.0},
            {"id": "n2", "line_id": "clip-2", "artifact_url": "https://a/2.wav", "seconds": 6.0},
        ]
    }
    record = {
        "clips": [
            {"id": "clip-1", "line_id": "clip-1", "video_url": "https://v/1.mp4", "seconds": 2.0},
            {"id": "clip-2", "line_id": "clip-2", "video_url": "https://v/2.mp4", "seconds": 2.0},
        ]
    }
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"deliverable": {"captions_mode": "none"}},
        prior_run_outputs={
            "demo_video_gen": gen,
            "demo_video_narrate": narrate,
            "demo_video_record": record,
        },
    )
    shots = DemoVideoCompileStage().build_shots(ctx)

    assert [s["line_id"] for s in shots] == ["clip-1", "clip-2"]
    assert shots[0]["video_url"] == "https://v/1.mp4"
    assert shots[0]["audio_url"] == "https://a/1.wav"
    # seconds reconciled to the narration clip duration (assemble_shots).
    assert shots[0]["seconds"] == 5.0
    # captions_mode=none → no burned-in caption text.
    assert "caption" not in shots[0]


def _directed_ctx(direction: dict | None) -> RunContext:
    # Two lines share one recording (clip-1) to prove the clip→line join fans the
    # track onto every line that plays over that clip; clip-2 carries no track.
    gen = {
        "lines": [
            {"id": "clip-1", "text": "Log in", "seconds": 2.0},
            {"id": "l2", "text": "…and again", "seconds": 2.0, "clip": "clip-1"},
            {"id": "clip-2", "text": "Create", "seconds": 2.0},
        ]
    }
    narrate = {
        "clips": [
            {"id": "n1", "line_id": "clip-1", "artifact_url": "https://a/1.wav", "seconds": 5.0},
            {"id": "n2", "line_id": "l2", "artifact_url": "https://a/2.wav", "seconds": 5.0},
            {"id": "n3", "line_id": "clip-2", "artifact_url": "https://a/3.wav", "seconds": 6.0},
        ]
    }
    _focus = [{"t": 0.5, "bbox": {"x": 10, "y": 20, "w": 100, "h": 40}, "action": "click"}]
    _cursor = [{"t": 0.5, "x": 60, "y": 40, "click": True}]
    record = {
        "clips": [
            {
                "id": "clip-1",
                "line_id": "clip-1",
                "video_url": "https://v/1.mp4",
                "seconds": 2.0,
                "focus_track": _focus,
                "cursor_track": _cursor,
                "viewport": {"w": 1920, "h": 1080},
            },
            # clip-2 recorded no clicks → FTR emitted no track (fields absent/None).
            {"id": "clip-2", "line_id": "clip-2", "video_url": "https://v/2.mp4", "seconds": 2.0},
        ]
    }
    snapshot: dict = {"deliverable": {"captions_mode": "none"}}
    if direction is not None:
        snapshot["direction"] = direction
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=snapshot,
        prior_run_outputs={
            "demo_video_gen": gen,
            "demo_video_narrate": narrate,
            "demo_video_record": record,
        },
    )


def test_compile_directed_footage_threads_ftr_tracks_when_gated_on():
    shots = DemoVideoCompileStage().build_shots(_directed_ctx({"footage": {"auto_zoom": True}}))
    by = {s["line_id"]: s for s in shots}
    # The recorder shape the FTR clip carries, and what it becomes on the compile
    # wire: same keyframes in the same order, bbox as the declared 4-tuple and the
    # recorder-only ``action`` key dropped.
    recorded = [{"t": 0.5, "bbox": {"x": 10, "y": 20, "w": 100, "h": 40}, "action": "click"}]
    focus = focus_wire(recorded)
    # clip-1's geometry rides its own line AND the line that shares its recording.
    assert by["clip-1"]["focus_track"] == focus
    assert by["clip-1"]["footage_viewport"] == [1920, 1080]
    assert by["l2"]["focus_track"] == focus
    assert by["l2"]["footage_viewport"] == [1920, 1080]
    # clip-2 recorded no track → nothing rides.
    assert "focus_track" not in by["clip-2"]
    assert "footage_viewport" not in by["clip-2"]


def test_compile_directed_footage_inert_without_direction_footage():
    # No direction.footage block ⇒ gate off ⇒ no geometry rides, even though the
    # FTR clips carry tracks (byte-identical to today's demo compile).
    shots = DemoVideoCompileStage().build_shots(_directed_ctx(None))
    assert all(
        "focus_track" not in s and "cursor_track" not in s and "footage_viewport" not in s
        for s in shots
    )


def test_demo_video_clip_deserialises_without_new_geometry_fields():
    # An older persisted clip (no focus_track/cursor_track/viewport keys) still loads,
    # with the new fields defaulting to None (appended-last discipline).
    from src.services.stages.demo_video.record import DemoVideoClip

    clip = DemoVideoClip.model_validate(
        {"id": "clip-1", "line_id": "clip-1", "video_url": "https://v/1.mp4", "seconds": 2.0}
    )
    assert clip.focus_track is None
    assert clip.cursor_track is None
    assert clip.viewport is None


def test_compile_keys_showcase_card_stills_and_footage_by_line():
    """Showcase: a card line shows its still PNG, a footage line its recorded clip."""
    gen = {
        "lines": [
            {"id": "intro", "text": "Welcome", "seconds": 2.0},
            {"id": "clip-1", "text": "Dashboard", "seconds": 2.0},
        ]
    }
    narrate = {
        "clips": [
            {"id": "n0", "line_id": "intro", "artifact_url": "https://a/i.wav", "seconds": 3.0},
            {"id": "n1", "line_id": "clip-1", "artifact_url": "https://a/1.wav", "seconds": 4.0},
        ]
    }
    record = {
        "clips": [
            {"id": "clip-1", "line_id": "clip-1", "video_url": "https://v/1.mp4", "seconds": 2.0}
        ]
    }
    slides = {"slides": [{"id": "s0", "line_id": "intro", "image_url": "https://img/intro.png"}]}
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"deliverable": {"captions_mode": "none"}},
        prior_run_outputs={
            "demo_video_gen": gen,
            "demo_video_narrate": narrate,
            "demo_video_record": record,
            "showcase_slides": slides,
        },
    )
    by_line = {s["line_id"]: s for s in DemoVideoCompileStage().build_shots(ctx)}
    assert by_line["intro"]["image_url"] == "https://img/intro.png"
    assert "video_url" not in by_line["intro"]  # card line is a still, not footage
    assert by_line["clip-1"]["video_url"] == "https://v/1.mp4"


# motion clip wins over the still


def test_compile_without_showcase_slides_is_unchanged():
    """No showcase_slides prior output → no image keying (plain demo_video path)."""
    gen = {"lines": [{"id": "clip-1", "text": "Dashboard", "seconds": 2.0}]}
    narrate = {
        "clips": [
            {"id": "n1", "line_id": "clip-1", "artifact_url": "https://a/1.wav", "seconds": 4.0}
        ]
    }
    record = {
        "clips": [
            {"id": "clip-1", "line_id": "clip-1", "video_url": "https://v/1.mp4", "seconds": 2.0}
        ]
    }
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"deliverable": {"captions_mode": "none"}},
        prior_run_outputs={
            "demo_video_gen": gen,
            "demo_video_narrate": narrate,
            "demo_video_record": record,
        },
    )
    shot = DemoVideoCompileStage().build_shots(ctx)[0]
    assert shot["video_url"] == "https://v/1.mp4"
    assert "image_url" not in shot


def _timeline_ctx(lines: list[dict], vo_seconds: dict[str, float] | None = None) -> RunContext:
    """A compile ctx whose spine carries reconcile_timeline's stamps.

    ``vo_seconds`` names the lines that were narrated (and how long their VO ran);
    "clip-1" is the one recorded footage beat, measured at 6.0s.
    """
    narrated = vo_seconds or {}
    narrate = {
        "clips": [
            {
                "id": f"n-{line_id}",
                "line_id": line_id,
                "artifact_url": f"https://a/{line_id}.wav",
                "seconds": seconds,
            }
            for line_id, seconds in narrated.items()
        ]
    }
    record = {
        "clips": [
            {
                "id": "clip-1",
                "line_id": "clip-1",
                "video_url": "https://v/1.mp4",
                "seconds": 6.0,
            }
        ]
    }
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"deliverable": {"captions_mode": "none"}},
        prior_run_outputs={
            "demo_video_gen": {"lines": lines},
            "demo_video_narrate": narrate,
            "demo_video_record": record,
        },
    )


# (3, 7) window clamped the 9s VO


# the spine's own value, unchanged


def test_compile_requires_all_prior_outputs():
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
        prior_run_outputs={"demo_video_gen": {"lines": []}},
    )
    with pytest.raises(RuntimeError, match="required"):
        DemoVideoCompileStage().build_shots(ctx)


def _compile_result(mastering: dict | None) -> dict[str, Any]:
    """A minimal validated compile result build_output consumes.

    Only the keys build_output reads: the delivered artifact + duration and the
    optional post-mux ``mastering`` report. build_output never touches ctx, so
    the tests below can pass a trivial one.
    """
    result: dict[str, Any] = {
        "artifact_url": "https://minio.test/demo/final.mp4",
        "duration_seconds": 60.0,
    }
    if mastering is not None:
        result["mastering"] = mastering
    return result


def test_compile_dead_master_adds_degrade_note():
    # The compiler flags a blank/dead delivered master by appending "dead_frame"
    # to the mastering report's rerun_stages. build_output must surface that as a
    # non-empty `notes` entry, which api/routes/runs/_shared.py translates into a
    # `degraded` run (same soft-fail channel as the O1 blank-clip QA) — never a
    # hard failure. Mirrors test_record_flags_blank_clip_as_degraded_note.
    notes = mastering_gate_notes(
        {"rerun_stages": ["dead_frame"], "luma_spread": 3.0, "gate_pass": False}
    )
    assert any("blank/dead" in n and "3.0" in n for n in notes)


def test_compile_pure_banding_failure_is_not_flagged():
    # NARROWED after measuring production: 207 of 214 masters over 30 days report
    # gate_pass=false (96.7%), because banding_score awards smoothness*0.25 to ANY held
    # frame with luma range >= 60 and post_mux_verify keeps the film's WORST frame. One
    # title card condemns a fine master. Of the 99 deband-only masters, only 11 were
    # legitimately BRIGHT content (luma_spread 157-219) — pure banding, a compiler-side
    # threshold question, not grounds for degrading a paid deliverable. Those must stay
    # silent or the badge means nothing.
    assert (
        mastering_gate_notes(
            {
                "rerun_stages": ["deband"],
                "gate_pass": False,
                "luma_spread": 178.0,  # a bright, real picture
                "banding_scores": [0.31],
            }
        )
        == []
    )


def test_compile_failed_gate_without_dead_frame_names_the_real_reason():
    # REGRESSION (run 01a00b81-2576-71a5-becc-ffc4a843e6db): the compiler reported
    # gate_pass=false with rerun_stages=["deband"] (never performed) on a blank master,
    # and the run completed CLEAN because the guard matched only the literal
    # "dead_frame". A blank picture must degrade even WITHOUT the dead_frame stage, and
    # the note must NAME the failing stage and the metric that decided it.
    # CORRECTED: "blank" is now dead_frame_ratio >= 0.10, NOT luma_spread=0.0 (a min
    # over frames — one fade-to-black zeroes it; see run 01a08f66).
    notes = mastering_gate_notes(
        {
            "rerun_stages": ["deband"],
            "gate_pass": False,
            "dead_frame_ratio": 0.25,
            "luma_spread": 0.0,
            "banding_scores": [1.0, 1.0],
            "duplicate_frame_ratio": 0.338,
        }
    )
    assert len(notes) == 1
    note = notes[0]
    assert "rerun_stages=deband" in note  # the actual rerun stage
    assert "dead_frame_ratio=0.25 (cutoff 0.1)" in note  # the metric that decided
    # luma_spread is quoted as non-evidence, never as the blank signature.
    assert "luma_spread=0.0 reported, not used" in note
    # CORRECTED: duplicate frames are intentional dwell, not blank-picture evidence;
    # quoting them in a BLANK note would be contradicting noise.
    assert "duplicate_frame_ratio" not in note


def test_compile_healthy_master_adds_no_note():
    # A mastering report WITHOUT the dead_frame flag must not degrade the run:
    # notes stay empty (presence-is-opt-in, no false positive on a live master).
    assert mastering_gate_notes({"rerun_stages": [], "luma_spread": 210.0, "gate_pass": True}) == []


def test_compile_absent_mastering_report_is_unmeasured_not_passing():
    # CORRECTED: this used to assert `== []` for an absent report, i.e. it PINNED the
    # fail-open — a compiler build that ships no mastering report was indistinguishable
    # from one whose gate ran and passed. An absent report means the gate never ran;
    # it now degrades with its own UNMEASURED note. Full coverage lives in
    # tests/services/stages/shared/test_mastering_gate.py.
    notes = mastering_gate_notes(None)
    assert len(notes) == 1
    assert "UNMEASURED" in notes[0]


def _sound_ctx(sound: dict | None) -> RunContext:
    gen = {
        "lines": [
            {"id": "clip-1", "text": "Log in", "seconds": 2.0},
            {"id": "clip-2", "text": "Eleven rooms", "seconds": 2.0},
        ]
    }
    narrate = {
        "clips": [
            {"id": "n1", "line_id": "clip-1", "artifact_url": "https://a/1.wav", "seconds": 5.0},
            {"id": "n2", "line_id": "clip-2", "artifact_url": "https://a/2.wav", "seconds": 6.0},
        ]
    }
    record = {
        "clips": [
            {"id": "clip-1", "line_id": "clip-1", "video_url": "https://v/1.mp4", "seconds": 2.0},
            {"id": "clip-2", "line_id": "clip-2", "video_url": "https://v/2.mp4", "seconds": 2.0},
        ]
    }
    prior: dict = {
        "demo_video_gen": gen,
        "demo_video_narrate": narrate,
        "demo_video_record": record,
        # The director's per-line accent plan (as demo_video_sound_director emits).
        "demo_video_sound_director": {
            "cues": [{"line_id": "clip-2", "sfx_id": "ding_positive", "at_seconds": 1.0}]
        },
    }
    snapshot: dict = {"deliverable": {"captions_mode": "none"}}
    if sound is not None:
        snapshot["sound_effects"] = sound
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=snapshot,
        prior_run_outputs=prior,
    )


def test_compile_threads_sound_when_opted_in():
    ctx = _sound_ctx({"enabled": True, "accents": True, "transitions": True})
    shots = DemoVideoCompileStage().build_shots(ctx)
    # Per-line accent cue lands on its line's shot.
    assert shots[1]["sfx"] == [{"sfx_id": "ding_positive", "at_seconds": 1.0, "gain_db": -10.0}]
    assert "sfx" not in shots[0]
    # Transition whoosh rides every cut except the open.
    assert "transition_sfx_id" not in shots[0]
    assert shots[1]["transition_sfx_id"] == "whoosh_soft"


async def test_compile_brief_sfx_gain_reaches_the_transition_wire_cue(
    monkeypatch: pytest.MonkeyPatch,
):
    """A brief's `sfx_gain_db` must reach the WIRE key, not just the accents.

    Composed over BOTH stages on purpose: `build_shots` stamps the internal
    planning key `transition_sfx_gain_db`, and only `resolve_shot_sound_urls`
    turns it into `shot["transition_sfx"]["gain_db"]` (falling back to -10 dB).
    Asserting on build_shots alone would not prove the level the compiler sees.
    """
    from src.services.stages.shot_sound import resolve_shot_sound_urls

    async def _fake_presign(sfx_id: str) -> str:
        return f"https://minio.test/sound/{sfx_id}.mp3"

    monkeypatch.setattr("src.services.stages.shot_sound.presign_sound_asset", _fake_presign)

    ctx = _sound_ctx({"enabled": True, "accents": True, "transitions": True, "sfx_gain_db": -20.0})
    shots = await resolve_shot_sound_urls(DemoVideoCompileStage().build_shots(ctx))
    # The transition only attaches from the second shot on (no whoosh on the open).
    assert shots[1]["transition_sfx"]["gain_db"] == -20.0
    # …and the accents in the SAME render sit at the SAME level — the defect was
    # the internal inconsistency between the two, not either value alone.
    accents = [cue for shot in shots for cue in shot.get("sfx", [])]
    assert accents, "expected at least one accent cue to compare against"
    assert all(cue["gain_db"] == -20.0 for cue in accents)


async def test_compile_transition_keeps_neutral_gain_when_brief_sets_none(
    monkeypatch: pytest.MonkeyPatch,
):
    """No `sfx_gain_db` in the brief → the transition still rides at neutral -10 dB."""
    from src.services.stages.shot_sound import resolve_shot_sound_urls

    async def _fake_presign(sfx_id: str) -> str:
        return f"https://minio.test/sound/{sfx_id}.mp3"

    monkeypatch.setattr("src.services.stages.shot_sound.presign_sound_asset", _fake_presign)

    ctx = _sound_ctx({"enabled": True, "accents": True, "transitions": True})
    shots = await resolve_shot_sound_urls(DemoVideoCompileStage().build_shots(ctx))
    assert shots[1]["transition_sfx"]["gain_db"] == -10.0
    accents = [cue for shot in shots for cue in shot.get("sfx", [])]
    assert all(cue["gain_db"] == -10.0 for cue in accents)


def test_compile_no_sound_when_not_opted_in():
    # Sound absent → shots byte-identical to the plain path (no sfx/transition keys).
    shots = DemoVideoCompileStage().build_shots(_sound_ctx(None))
    assert all("sfx" not in s and "transition_sfx_id" not in s for s in shots)


async def test_compile_audio_bed_resolves_music_and_ambient(monkeypatch: pytest.MonkeyPatch):
    from src.services.stages.sound_catalog import SoundLibrary

    async def _fake_library(*_a, **_k) -> SoundLibrary:
        return SoundLibrary.builtin()

    async def _fake_presign(sfx_id: str) -> str:
        return f"https://minio.test/sound/{sfx_id}.mp3"

    monkeypatch.setattr("src.services.stages.shared.sound_design.load_sound_library", _fake_library)
    monkeypatch.setattr(
        "src.services.stages.shared.sound_design.presign_sound_asset", _fake_presign
    )

    ctx = _sound_ctx({"enabled": True, "music": "corporate_uplift", "ambient": True})
    shots = DemoVideoCompileStage().build_shots(ctx)
    bed = await DemoVideoCompileStage().build_audio_bed(ctx, shots)
    assert bed is not None
    assert bed["music"]["url"].endswith("corporate_uplift.mp3")
    # Ambient (default room_tone) sized to the total shot duration.
    assert bed["ambient"][0]["url"].endswith("room_tone.mp3")
    assert bed["ambient"][0]["end_seconds"] == sum(s["seconds"] for s in shots)


async def test_compile_audio_bed_none_when_sound_off():
    shots = DemoVideoCompileStage().build_shots(_sound_ctx(None))
    assert await DemoVideoCompileStage().build_audio_bed(_sound_ctx(None), shots) is None


# ── record · per-clip visual QA (O1) ───────────────────────────────────────


def _qa_settings(**overrides: Any):
    """A get_settings() stand-in exposing a `.pipeline` with QA overrides."""
    from types import SimpleNamespace

    from src.config import PipelineSettings

    return SimpleNamespace(pipeline=PipelineSettings(**overrides))


def _record_client_with_one_clip() -> AsyncMock:
    client = AsyncMock()
    client.start_execution.return_value = "ftr-qa"
    client.wait_for_execution.return_value = {}
    client.get_job_result.return_value = {
        "scenario_clips": [{"clip_key": "clip-1", "duration_ms": 2000}]
    }
    client.get_clip_video.return_value = b"mp4bytes"
    return client


@pytest.mark.asyncio
async def test_record_flags_blank_clip_as_degraded_note():
    """A clip whose QA reports verification_passed=False adds a note (→ degraded)."""
    client = _record_client_with_one_clip()
    analyzer = AsyncMock()
    analyzer.start_analysis.return_value = "va-1"
    analyzer.poll_analysis.return_value = {"verification_passed": False}
    analyzer.get_verification_result.return_value = {
        "failures": [{"criterion": "min_sharpness_score", "actual": 0.3}]
    }
    gen = {"bdd_job_id": "b", "lines": [{"id": "clip-1"}]}
    stage = DemoVideoRecordStage(client=client, analyzer=analyzer)
    with patch(
        "src.services.stages.demo_video.record.store_capture_clip",
        new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['clip_id']}.mp4"),
    ):
        out = await stage.handle(
            stage.input_model(),
            _ctx({"target_url": "https://x"}, prior={"demo_video_gen": gen}),
        )
    # The clip still records; the failure is surfaced softly as a note.
    assert [c.line_id for c in out.clips] == ["clip-1"]
    assert any("clip-1 failed visual QA" in n and "min_sharpness_score" in n for n in out.notes)


def _record_client_zero_duration() -> AsyncMock:
    """FTR fake that reports duration_ms=0 — the prod bug (races Playwright finalize)."""
    client = AsyncMock()
    client.start_execution.return_value = "ftr-qa"
    client.wait_for_execution.return_value = {}
    client.get_job_result.return_value = {
        "scenario_clips": [{"clip_key": "clip-1", "duration_ms": 0}]
    }
    client.get_clip_video.return_value = b"mp4bytes"
    return client


async def _run_record_with(client: AsyncMock, analyzer: AsyncMock) -> DemoVideoRecordOutput:
    gen = {"bdd_job_id": "b", "lines": [{"id": "clip-1"}]}
    stage = DemoVideoRecordStage(client=client, analyzer=analyzer)
    with patch(
        "src.services.stages.demo_video.record.store_capture_clip",
        new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['clip_id']}.mp4"),
    ):
        return await stage.handle(
            stage.input_model(),
            _ctx({"target_url": "https://x"}, prior={"demo_video_gen": gen}),
        )


# repaired from the analyzer, not FTR's 0


# FTR's real value wins; analyzer ignored


@pytest.mark.asyncio
async def test_record_qa_is_soft_on_analyzer_error():
    """An analyzer outage must not FAIL the run — but it must not be silent either.

    QA stays best-effort: no raise, the clips still ship. What changed is that the
    outage is now declared. `_ClipQa().blank` is False, and the blank hard-fail reads
    exactly that field — so before this, one analyzer timeout disabled the gate that
    exists to stop a 100% black video shipping, and the run rendered as a clean
    `completed` with nothing to say the check had not run.
    """
    client = _record_client_with_one_clip()
    analyzer = AsyncMock()
    analyzer.start_analysis.side_effect = RuntimeError("analyzer down")
    gen = {"bdd_job_id": "b", "lines": [{"id": "clip-1"}]}
    stage = DemoVideoRecordStage(client=client, analyzer=analyzer)
    with patch(
        "src.services.stages.demo_video.record.store_capture_clip",
        new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['clip_id']}.mp4"),
    ):
        out = await stage.handle(
            stage.input_model(),
            _ctx({"target_url": "https://x"}, prior={"demo_video_gen": gen}),
        )
    assert [c.line_id for c in out.clips] == ["clip-1"]
    # No per-clip QA verdict note — there was no verdict to report.
    assert not any("failed visual QA" in n for n in out.notes)
    # But the un-run gate is declared, which renders the run `degraded`.
    assert any("blank-recording check did not run" in n for n in out.notes)


@pytest.mark.asyncio
async def test_measured_healthy_clip_declares_no_gap():
    """The negative control for the un-run-gate note.

    Without this, the assertions above would still pass if the note were emitted
    unconditionally — which would make every run permanently `degraded` and train
    everyone to ignore it, the same way `N/N visuals` was ignored.
    """
    analyzer = AsyncMock()
    analyzer.start_analysis.return_value = "va-1"
    analyzer.poll_analysis.return_value = {"verification_passed": True}
    analyzer.get_result.return_value = VideoAnalyzerJobResult(
        job_id="va-1",
        status="completed",
        stage="done",
        progress=1.0,
        quality_metrics=VideoAnalyzerQualityMetrics.model_validate(_quality_metrics()),
    )
    out = await _run_record(analyzer)
    assert not any("blank-recording check did not run" in n for n in out.notes)


@pytest.mark.asyncio
async def test_record_qa_strict_fails_the_run():
    """Strict mode turns a QA failure into a run failure instead of a soft note."""
    client = _record_client_with_one_clip()
    analyzer = AsyncMock()
    analyzer.start_analysis.return_value = "va-1"
    analyzer.poll_analysis.return_value = {"verification_passed": False}
    analyzer.get_verification_result.return_value = {
        "failures": [{"criterion": "min_sharpness_score", "actual": 0.1}]
    }
    gen = {"bdd_job_id": "b", "lines": [{"id": "clip-1"}]}
    stage = DemoVideoRecordStage(client=client, analyzer=analyzer)
    with (
        patch(
            "src.services.stages.demo_video.record.store_capture_clip",
            new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['clip_id']}.mp4"),
        ),
        patch(
            "src.services.stages.demo_video.record.get_settings",
            return_value=_qa_settings(demo_qa_strict=True),
        ),
        pytest.raises(RuntimeError, match="strict visual QA failed"),
    ):
        await stage.handle(
            stage.input_model(),
            _ctx({"target_url": "https://x"}, prior={"demo_video_gen": gen}),
        )


@pytest.mark.asyncio
async def test_record_qa_disabled_skips_analyzer():
    """PIPELINE_DEMO_QA_ENABLED=false → no analyzer calls at all."""
    client = _record_client_with_one_clip()
    analyzer = AsyncMock()
    gen = {"bdd_job_id": "b", "lines": [{"id": "clip-1"}]}
    stage = DemoVideoRecordStage(client=client, analyzer=analyzer)
    with (
        patch(
            "src.services.stages.demo_video.record.store_capture_clip",
            new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['clip_id']}.mp4"),
        ),
        patch(
            "src.services.stages.demo_video.record.get_settings",
            return_value=_qa_settings(demo_qa_enabled=False),
        ),
    ):
        out = await stage.handle(
            stage.input_model(),
            _ctx({"target_url": "https://x"}, prior={"demo_video_gen": gen}),
        )
    assert [c.line_id for c in out.clips] == ["clip-1"]
    analyzer.start_analysis.assert_not_called()


@pytest.mark.asyncio
async def test_record_ocr_wrong_screen_degrades_but_never_strict_fails():
    """O1b: an OCR-only miss (wrong screen) adds a note + degrades but must NOT raise
    even under strict — a tesseract-less analyzer would flag it spuriously."""
    client = _record_client_with_one_clip()
    analyzer = AsyncMock()
    analyzer.start_analysis.return_value = "va-1"
    analyzer.poll_analysis.return_value = {"verification_passed": False}
    analyzer.get_verification_result.return_value = {
        "failures": [{"criterion": "required_ocr_text", "expected": "Welcome back, Alex"}]
    }
    gen = {"bdd_job_id": "b", "lines": [{"id": "clip-1", "expected_text": "Welcome back, Alex"}]}
    stage = DemoVideoRecordStage(client=client, analyzer=analyzer)
    with (
        patch(
            "src.services.stages.demo_video.record.store_capture_clip",
            new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['clip_id']}.mp4"),
        ),
        patch(
            "src.services.stages.demo_video.record.get_settings",
            return_value=_qa_settings(demo_qa_ocr_enabled=True, demo_qa_strict=True),
        ),
    ):
        # Strict is ON, but an OCR-only failure stays soft → no raise.
        out = await stage.handle(
            stage.input_model(),
            _ctx({"target_url": "https://x"}, prior={"demo_video_gen": gen}),
        )
    assert [c.line_id for c in out.clips] == ["clip-1"]
    assert any("clip-1 failed visual QA" in n and "required_ocr_text" in n for n in out.notes)
    # And the OCR assertion was actually sent (expected_text present + ocr_enabled).
    call = analyzer.start_analysis.call_args
    assert call.kwargs["verification"]["required_ocr_text"] == ["Welcome back, Alex"]
    assert call.kwargs["config_overrides"]["extract_ocr"] is True


@pytest.mark.asyncio
async def test_record_ocr_disabled_sends_no_ocr_even_with_expected_text():
    """demo_qa_ocr_enabled=False (default) → no required_ocr_text, extract_ocr stays
    False, even when the line carries an expected_text."""
    client = _record_client_with_one_clip()
    analyzer = AsyncMock()
    analyzer.start_analysis.return_value = "va-1"
    analyzer.poll_analysis.return_value = {"verification_passed": True}
    gen = {"bdd_job_id": "b", "lines": [{"id": "clip-1", "expected_text": "Welcome back, Alex"}]}
    stage = DemoVideoRecordStage(client=client, analyzer=analyzer)
    with patch(
        "src.services.stages.demo_video.record.store_capture_clip",
        new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['clip_id']}.mp4"),
    ):
        await stage.handle(
            stage.input_model(),
            _ctx({"target_url": "https://x"}, prior={"demo_video_gen": gen}),
        )
    call = analyzer.start_analysis.call_args
    assert "required_ocr_text" not in call.kwargs["verification"]
    assert call.kwargs["config_overrides"]["extract_ocr"] is False


# ── ORCH-1.3c: motion curve harvested from the QA pass ────────────────────


def _quality_metrics(timeline: list[dict[str, object]] | None = None) -> dict[str, object]:
    """A full server-shape ``QualityMetrics`` body (asdict of the dataclass) —
    every scalar field is required on the wire; only ``motion_timeline`` is
    version-dependent (omitted = an older analyzer)."""
    metrics: dict[str, object] = {
        "mean_sharpness": 50.0,
        "mean_brightness": 120.0,
        "motion_score": 0.5,
        "dominant_colors": ["#ffffff"],
        "overall_score": 3.0,
    }
    if timeline is not None:
        metrics["motion_timeline"] = timeline
    return metrics


def _motion_analyzer(timeline: list[dict[str, object]], *, passed: bool = True) -> AsyncMock:
    analyzer = AsyncMock()
    analyzer.start_analysis.return_value = "va-1"
    analyzer.poll_analysis.return_value = {"verification_passed": passed}
    analyzer.get_result.return_value = VideoAnalyzerJobResult(
        job_id="va-1",
        status="completed",
        stage="done",
        progress=1.0,
        quality_metrics=VideoAnalyzerQualityMetrics.model_validate(_quality_metrics(timeline)),
    )
    analyzer.get_verification_result.return_value = {
        "failures": [{"criterion": "min_sharpness_score", "actual": 0.3}]
    }
    return analyzer


async def _run_record(analyzer: AsyncMock) -> DemoVideoRecordOutput:
    client = _record_client_with_one_clip()
    gen = {"bdd_job_id": "b", "lines": [{"id": "clip-1"}]}
    stage = DemoVideoRecordStage(client=client, analyzer=analyzer)
    with patch(
        "src.services.stages.demo_video.record.store_capture_clip",
        new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['clip_id']}.mp4"),
    ):
        return await stage.handle(
            stage.input_model(),
            _ctx({"target_url": "https://x"}, prior={"demo_video_gen": gen}),
        )


@pytest.mark.asyncio
async def test_record_stamps_the_motion_curve_on_the_clip():
    """The QA pass already computed the curve; record must now KEEP it."""
    analyzer = _motion_analyzer([{"t": 1.0, "motion": 0.02}, {"t": 2.0, "motion": 0.5}])
    out = await _run_record(analyzer)
    (clip,) = out.clips
    assert clip.motion_timeline is not None
    assert [(s.t, s.motion) for s in clip.motion_timeline] == [(1.0, 0.02), (2.0, 0.5)]
    # Fetched from the finished job — not a second analysis run.
    analyzer.get_result.assert_awaited_once_with("va-1")
    analyzer.start_analysis.assert_awaited_once()


@pytest.mark.asyncio
async def test_record_harvests_the_curve_even_when_qa_fails():
    """A blank-ish clip still yields a curve; the failure note is independent of it."""
    analyzer = _motion_analyzer(
        [{"t": 1.0, "motion": 0.02}, {"t": 2.0, "motion": 0.5}], passed=False
    )
    out = await _run_record(analyzer)
    (clip,) = out.clips
    assert clip.motion_timeline is not None
    assert any("failed visual QA" in n for n in out.notes)


@pytest.mark.asyncio
async def test_record_leaves_curve_none_when_analyzer_predates_the_field():
    """Cross-repo version skew: an older video_analyzer emits no motion_timeline —
    the client schema defaults the field to [], so the clip's curve stays None."""
    analyzer = AsyncMock()
    analyzer.start_analysis.return_value = "va-1"
    analyzer.poll_analysis.return_value = {"verification_passed": True}
    analyzer.get_result.return_value = VideoAnalyzerJobResult(
        job_id="va-1",
        status="completed",
        stage="done",
        progress=1.0,
        quality_metrics=VideoAnalyzerQualityMetrics.model_validate(
            _quality_metrics(timeline=None)  # no motion_timeline key
        ),
    )
    out = await _run_record(analyzer)
    assert out.clips[0].motion_timeline is None
    # A missing curve is NOT a QA (blank-gate) failure...
    assert not any("failed visual QA" in n or "blank" in n for n in out.notes)
    # ...but it IS a measurement gap for the INERT gate (#inert-gate-unmeasured fix):
    # with no curve at all, the inert/active verdict for this clip cannot be judged,
    # and that must be visible rather than silently read as "fine".
    assert any("could not measure" in n and "UNPROVEN" in n for n in out.notes)


@pytest.mark.asyncio
async def test_record_curve_fetch_failure_is_soft():
    """The curve is a bonus, never a gate — a result-fetch error must not RAISE.

    It does now leave a note, and that is not about the curve: the same fetch carries
    `mean_sharpness`, so losing it leaves the blank hard-fail with no evidence to
    decide on. `is_blank` then reads False for lack of measurement, which is
    indistinguishable from a measured-and-fine clip — so the gap is declared.
    """
    analyzer = AsyncMock()
    analyzer.start_analysis.return_value = "va-1"
    analyzer.poll_analysis.return_value = {"verification_passed": True}
    analyzer.get_result.side_effect = RuntimeError("boom")
    out = await _run_record(analyzer)
    assert out.clips[0].motion_timeline is None
    assert not any("failed visual QA" in n for n in out.notes)
    assert any("blank-recording check did not run" in n for n in out.notes)


@pytest.mark.asyncio
async def test_record_garbled_curve_fails_validation_and_stays_soft():
    """A garbled curve now fails validation INSIDE the typed client (get_result
    raises ValidationError), and the record stage treats that exactly like any
    result-fetch failure: no curve, no note, never a raised run. This replaces the
    old per-entry coercion test — the cross-repo junk gate moved to the schema.

    As with the fetch-failure twin, the run does now carry a note: the same call
    supplies `mean_sharpness`, so a ValidationError leaves the blank gate unable to
    reach a verdict, and an un-run gate is declared rather than passed silently."""
    garbled = {
        "job_id": "va-1",
        "status": "completed",
        "stage": "done",
        "progress": 1.0,
        "quality_metrics": _quality_metrics(
            [{"t": "x", "motion": 0.4}, {"t": 2.0}]  # junk entries
        ),
    }
    with pytest.raises(ValidationError) as exc_info:
        VideoAnalyzerJobResult.model_validate(garbled)

    analyzer = AsyncMock()
    analyzer.start_analysis.return_value = "va-1"
    analyzer.poll_analysis.return_value = {"verification_passed": True}
    analyzer.get_result.side_effect = exc_info.value
    out = await _run_record(analyzer)
    assert out.clips[0].motion_timeline is None
    assert not any("failed visual QA" in n for n in out.notes)
    assert any("blank-recording check did not run" in n for n in out.notes)


@pytest.mark.asyncio
async def test_record_qa_disabled_means_no_curve():
    """Turning QA off turns the motion signal off with it — consumers MUST fall back."""
    analyzer = _motion_analyzer([{"t": 1.0, "motion": 0.5}, {"t": 2.0, "motion": 0.6}])
    client = _record_client_with_one_clip()
    gen = {"bdd_job_id": "b", "lines": [{"id": "clip-1"}]}
    stage = DemoVideoRecordStage(client=client, analyzer=analyzer)
    with (
        patch(
            "src.services.stages.demo_video.record.store_capture_clip",
            new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['clip_id']}.mp4"),
        ),
        patch(
            "src.services.stages.demo_video.record.get_settings",
            return_value=_qa_settings(demo_qa_enabled=False),
        ),
    ):
        out = await stage.handle(
            stage.input_model(),
            _ctx({"target_url": "https://x"}, prior={"demo_video_gen": gen}),
        )
    assert out.clips[0].motion_timeline is None
    analyzer.start_analysis.assert_not_awaited()


# ── record · blank-clip HARD-FAIL gate (O1c) ───────────────────────────────
#
# A genuinely blank recording (the app never loaded; prod blanks measure
# mean_sharpness == 0.0) must FAIL the run — independent of `demo_qa_strict` —
# instead of shipping `completed`+degraded and charging for worthless video.
# The gate reads the analyzer's typed `mean_sharpness` on EVERY QA path so a
# blank the analyzer's own softer 3.0 criterion happened to pass is still caught.


def _qm_body(sharpness: float) -> dict[str, object]:
    """A full QualityMetrics wire body with an overridden mean_sharpness."""
    body = _quality_metrics()
    body["mean_sharpness"] = sharpness
    return body


def _sharpness_analyzer(sharpness: float, *, passed: bool, ocr_only: bool = False) -> AsyncMock:
    """Analyzer whose /result reports `sharpness` and whose verdict (if it failed)
    blames min_sharpness_score by default, or required_ocr_text when `ocr_only`."""
    analyzer = AsyncMock()
    analyzer.start_analysis.return_value = "va-1"
    analyzer.poll_analysis.return_value = {"verification_passed": passed}
    analyzer.get_result.return_value = VideoAnalyzerJobResult(
        job_id="va-1",
        status="completed",
        stage="done",
        progress=1.0,
        quality_metrics=VideoAnalyzerQualityMetrics.model_validate(_qm_body(sharpness)),
    )
    criterion = "required_ocr_text" if ocr_only else "min_sharpness_score"
    analyzer.get_verification_result.return_value = {
        "failures": [{"criterion": criterion, "actual": sharpness}]
    }
    return analyzer


async def _run_record_qa(
    analyzer: AsyncMock, *, client: AsyncMock | None = None, **settings_overrides: Any
) -> DemoVideoRecordOutput:
    client = client or _record_client_with_one_clip()
    gen = {"bdd_job_id": "b", "lines": [{"id": "clip-1"}]}
    stage = DemoVideoRecordStage(client=client, analyzer=analyzer)
    with (
        patch(
            "src.services.stages.demo_video.record.store_capture_clip",
            new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['clip_id']}.mp4"),
        ),
        patch(
            "src.services.stages.demo_video.record.get_settings",
            return_value=_qa_settings(**settings_overrides),
        ),
    ):
        return await stage.handle(
            stage.input_model(),
            _ctx({"target_url": "https://x"}, prior={"demo_video_gen": gen}),
        )


@pytest.mark.asyncio
async def test_blank_clip_hard_fails_by_default_even_when_not_strict():
    """(a) mean_sharpness==0.0 (real prod value), verdict FAILED, strict OFF (default)
    → the run FAILS with the blank message, not a soft degrade note."""
    with pytest.raises(RuntimeError, match=r"blank clip.* rejected"):
        await _run_record_qa(_sharpness_analyzer(0.0, passed=False))


@pytest.mark.asyncio
async def test_blank_clip_hard_fails_even_when_analyzer_passed_it():
    """(b) THE pass-path case (adversarial R2): the analyzer's own 3.0 criterion
    PASSED the clip (verification_passed=True), but its measured mean_sharpness==0.0
    is a blank → the independent gate still hard-fails the run."""
    with pytest.raises(RuntimeError, match=r"blank clip.* rejected"):
        await _run_record_qa(_sharpness_analyzer(0.0, passed=True))


@pytest.mark.asyncio
async def test_low_but_nonblank_sharpness_only_degrades_not_fails():
    """(c) sharpness in the [fail, min) band (2.0) is low-quality, NOT blank → a soft
    note (degrade), the clip still records, no raise."""
    out = await _run_record_qa(_sharpness_analyzer(2.0, passed=False))
    assert [c.line_id for c in out.clips] == ["clip-1"]
    assert any("failed visual QA" in n for n in out.notes)
    assert not any("blank" in n for n in out.notes)


@pytest.mark.asyncio
async def test_real_clip_passes_cleanly():
    """(d) a real UI (sharpness 50.0) that the analyzer passed → no blank/QA notes, no
    raise. `_sharpness_analyzer` never populates `motion_timeline`, so the inert gate
    still (correctly) flags the curve as unmeasured — that is a SEPARATE, expected note,
    not a regression of the blank-gate contract this test exists to pin."""
    out = await _run_record_qa(_sharpness_analyzer(50.0, passed=True))
    assert [c.line_id for c in out.clips] == ["clip-1"]
    assert not any("failed visual QA" in n or "blank" in n for n in out.notes)
    assert any("could not measure" in n and "UNPROVEN" in n for n in out.notes)


@pytest.mark.asyncio
async def test_blank_gate_stays_soft_when_sharpness_is_unknown():
    """(e) verdict FAILED but the /result body is unavailable → mean_sharpness is
    unknown → the gate CANNOT fire (hard-fail needs positive near-zero evidence);
    the run only degrades. Never fires on an analyzer/result outage."""
    analyzer = _sharpness_analyzer(0.0, passed=False)
    analyzer.get_result.side_effect = RuntimeError("result body gone")
    out = await _run_record_qa(analyzer)
    assert [c.line_id for c in out.clips] == ["clip-1"]
    assert any("failed visual QA" in n for n in out.notes)  # degraded, not raised


@pytest.mark.asyncio
async def test_blank_gate_disabled_at_zero_threshold():
    """(f) demo_qa_fail_sharpness=0.0 disables the hard gate → a blank clip (0.0) that
    the analyzer passed records with no blank/QA note (pre-fix degrade-only behaviour).
    The inert gate's own unmeasured note still fires — `_sharpness_analyzer` never
    populates `motion_timeline`, independent of the blank-gate threshold."""
    out = await _run_record_qa(_sharpness_analyzer(0.0, passed=True), demo_qa_fail_sharpness=0.0)
    assert [c.line_id for c in out.clips] == ["clip-1"]
    assert not any("failed visual QA" in n or "blank" in n for n in out.notes)
    assert any("could not measure" in n and "UNPROVEN" in n for n in out.notes)


@pytest.mark.asyncio
async def test_ocr_only_failure_never_blank_fails():
    """(g) a real-sharpness clip that fails ONLY the OCR wrong-screen check must not be
    treated as blank — the OCR carve-out is preserved, it only degrades."""
    analyzer = _sharpness_analyzer(50.0, passed=False, ocr_only=True)
    out = await _run_record_qa(analyzer, demo_qa_ocr_enabled=True)
    assert [c.line_id for c in out.clips] == ["clip-1"]
    assert not any("blank" in n for n in out.notes)


@pytest.mark.asyncio
async def test_degrade_band_under_strict_raises_strict_not_blank():
    """(h) sharpness in the degrade band (2.0) under demo_qa_strict=True raises the
    STRICT message, not the blank message — proves 2.0 is not mistaken for blank and
    the pre-existing strict path is intact."""
    with pytest.raises(RuntimeError, match="strict visual QA failed"):
        await _run_record_qa(_sharpness_analyzer(2.0, passed=False), demo_qa_strict=True)


# ── record · blank-clip DIAGNOSTIC (why was it blank?) ─────────────────────
#
# The gate above kills a paid render with only `min_sharpness_score=0.0` and NO
# reason. FTR already puts the upstream cause on the wire — `test_results[].error`
# (typically the bdd nav hook: "no paint within 12000ms", page stuck on about:blank).
# The join is `(feature, scenario)` on a clip ↔ `(feature, name)` on a test result;
# `clip_key` does NOT exist on a test result. The diagnostic rides in the RAISED
# message (never in the output model — on the hard-fail path no output is persisted),
# which dag_executor writes verbatim into `StageOutput.error`.

_NAV_HOOK_ERROR = 'nav hook: no paint within 12000ms - {"label":"nav-hook"}'


def _record_client_blank(*, test_results: list[dict[str, object]] | None = None) -> AsyncMock:
    """FTR fake whose one clip carries full geometry, plus optional `test_results`."""
    client = AsyncMock()
    client.start_execution.return_value = "ftr-qa"
    client.wait_for_execution.return_value = {}
    result: dict[str, object] = {
        "scenario_clips": [
            {
                "clip_key": "clip-1",
                "scenario": "adds a todo",
                "feature": "Todos",
                "status": "failed",
                "duration_ms": 2000,
                "width": 1280,
                "height": 720,
                "size_bytes": 4096,
            }
        ]
    }
    if test_results is not None:
        result["test_results"] = test_results
    client.get_job_result.return_value = result
    client.get_clip_video.return_value = b"mp4bytes"
    return client


@pytest.mark.asyncio
async def test_blank_clip_failure_carries_ftr_error_and_geometry():
    """THE fix: a blank clip's hard-fail message names the upstream FTR error (joined on
    (feature, scenario)) plus the recorded geometry — self-diagnosing on the next
    occurrence instead of a bare mean_sharpness=0.0."""
    client = _record_client_blank(
        test_results=[
            {
                "name": "adds a todo",
                "feature": "Todos",
                "status": "failed",
                "error": _NAV_HOOK_ERROR,
            }
        ]
    )
    with pytest.raises(RuntimeError, match=r"blank clip.* rejected") as exc:
        await _run_record_qa(_sharpness_analyzer(0.0, passed=False), client=client)
    # The nav-hook text passes through INTACT (JSON braces and all).
    assert "nav hook: no paint within 12000ms" in str(exc.value)
    assert "1280x720" in str(exc.value)


@pytest.mark.asyncio
async def test_blank_diagnostic_says_none_when_no_test_result_matches():
    """Join-miss guard: `test_results` present but NOTHING matches the clip's feature or
    name → still hard-fails (semantics unchanged), and the diagnostic says so explicitly
    rather than silently attaching another scenario's error."""
    client = _record_client_blank(
        test_results=[
            {
                "name": "some other scenario",
                "feature": "Settings",
                "status": "failed",
                "error": _NAV_HOOK_ERROR,
            }
        ]
    )
    with pytest.raises(RuntimeError, match=r"blank clip.* rejected") as exc:
        await _run_record_qa(_sharpness_analyzer(0.0, passed=False), client=client)
    assert "ftr_error=<none>" in str(exc.value)
    assert "nav hook" not in str(exc.value)


@pytest.mark.asyncio
async def test_blank_diagnostic_backward_compatible_without_test_results():
    """An FTR that emits no `test_results` at all (older deploy) must not crash the
    stage: the message keeps its byte-identical prefix, with an empty ftr_error."""
    with pytest.raises(RuntimeError) as exc:
        await _run_record_qa(_sharpness_analyzer(0.0, passed=False), client=_record_client_blank())
    assert "demo_video_record: blank clip(s) rejected: " in str(exc.value)
    assert "ftr_error=<none>" in str(exc.value)


@pytest.mark.asyncio
async def test_blank_diagnostic_falls_back_to_unambiguous_name_on_feature_drift():
    """FTR SHAPE DRIFT (not Scenario Outline — an Outline's examples share one spec, so
    they carry the clip's own feature/name and the pair join hits). If FTR ever changed
    how it derives a feature title, the pair join would miss; a name claimed by exactly
    ONE test result is still safe to use."""
    client = _record_client_blank(
        test_results=[
            {
                # Different `feature` than the clip → the exact pair join MISSES.
                "name": "adds a todo",
                "feature": "Todos > Examples",
                "status": "failed",
                "error": _NAV_HOOK_ERROR,
            }
        ]
    )
    with pytest.raises(RuntimeError, match=r"blank clip.* rejected") as exc:
        await _run_record_qa(_sharpness_analyzer(0.0, passed=False), client=client)
    assert "nav hook: no paint within 12000ms" in str(exc.value)


@pytest.mark.asyncio
async def test_pair_join_wins_over_a_same_name_decoy():
    """The PRIMARY (feature, name) join must be what resolves the error — asserted
    independently of the fallback. Without this, breaking the pair join is invisible:
    the name-only map silently covers for it."""
    client = _record_client_blank(
        test_results=[
            {"name": "adds a todo", "feature": "Todos", "status": "failed", "error": "CORRECT_ERR"},
            # Same scenario name under ANOTHER feature — must never be chosen.
            {
                "name": "adds a todo",
                "feature": "Settings",
                "status": "failed",
                "error": "DECOY_ERR",
            },
        ]
    )
    with pytest.raises(RuntimeError, match=r"blank clip.* rejected") as exc:
        await _run_record_qa(_sharpness_analyzer(0.0, passed=False), client=client)
    assert "CORRECT_ERR" in str(exc.value)
    assert "DECOY_ERR" not in str(exc.value)


@pytest.mark.asyncio
async def test_ambiguous_name_refuses_to_guess_an_error():
    """Two features share a scenario title and NEITHER matches the clip's feature. The
    name-only map must refuse rather than pick suite order: a confidently-wrong
    post-mortem sends the next operator to debug the wrong scenario."""
    client = _record_client_blank(
        test_results=[
            {"name": "adds a todo", "feature": "AlphaFeat", "status": "failed", "error": "ERR_A"},
            {"name": "adds a todo", "feature": "BetaFeat", "status": "failed", "error": "ERR_B"},
        ]
    )
    # Clip's feature is "Todos", matching neither → pair miss, name poisoned.
    client.get_job_result.return_value["scenario_clips"][0]["feature"] = "Todos"
    with pytest.raises(RuntimeError, match=r"blank clip.* rejected") as exc:
        await _run_record_qa(_sharpness_analyzer(0.0, passed=False), client=client)
    assert "ftr_error=<none>" in str(exc.value)
    assert "ERR_A" not in str(exc.value)
    assert "ERR_B" not in str(exc.value)


@pytest.mark.asyncio
async def test_blank_clip_diagnostic_is_also_logged_at_error(caplog):
    """The raise is durable only if the DB write lands; the log is the belt-and-braces
    copy. Asserted so removing it cannot pass silently."""
    client = _record_client_blank(
        test_results=[
            {
                "name": "adds a todo",
                "feature": "Todos",
                "status": "failed",
                "error": _NAV_HOOK_ERROR,
            }
        ]
    )
    with caplog.at_level(logging.ERROR, logger="src.services.stages.demo_video.record"):
        with pytest.raises(RuntimeError, match=r"blank clip.* rejected"):
            await _run_record_qa(_sharpness_analyzer(0.0, passed=False), client=client)
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("clip-1" in m and "no paint within 12000ms" in m for m in errors), errors


@pytest.mark.asyncio
async def test_pathological_ftr_fields_stay_capped_and_keep_the_useful_ones():
    """A 500-char status and a 10k error must not blow the cap NOR starve the diagnostic
    down to an ellipsis: the geometry and a usable slice of the error both survive."""
    from src.services.stages.demo_video.record import _DIAG_HARD_MAX_CHARS

    client = _record_client_blank(
        test_results=[
            {
                "name": "adds a todo",
                "feature": "Todos",
                "status": "failed",
                "error": "nav hook: no paint within 12000ms " + ("X" * 10_000),
            }
        ]
    )
    # `ftr_status` is read off the CLIP, so that is where the pathological value goes.
    client.get_job_result.return_value["scenario_clips"][0]["status"] = "S" * 500
    with pytest.raises(RuntimeError, match=r"blank clip.* rejected") as exc:
        await _run_record_qa(_sharpness_analyzer(0.0, passed=False), client=client)
    msg = str(exc.value)
    # Assert the DOCUMENTED guarantee on the bracketed diagnostic itself, not on the
    # whole message — a loose bound here silently permits an unclamped field.
    diag = msg[msg.index("[") + 1 : msg.rindex("]")]
    assert len(diag) <= _DIAG_HARD_MAX_CHARS, (len(diag), diag[:120])
    assert "1280x720" in diag
    assert "nav hook: no paint within 12000ms" in diag


@pytest.mark.asyncio
async def test_malformed_ftr_geometry_does_not_crash_the_stage():
    """FTR types these int/float, so this is drift insurance: junk geometry must still
    yield the normal blank-gate RuntimeError, not a ValueError from inside the loop."""
    client = _record_client_blank(
        test_results=[
            {
                "name": "adds a todo",
                "feature": "Todos",
                "status": "failed",
                "error": _NAV_HOOK_ERROR,
            }
        ]
    )
    clip = client.get_job_result.return_value["scenario_clips"][0]
    clip["width"] = "abc"
    clip["height"] = None
    clip["duration_ms"] = "not-a-number"
    with pytest.raises(RuntimeError, match=r"blank clip.* rejected") as exc:
        await _run_record_qa(_sharpness_analyzer(0.0, passed=False), client=client)
    assert "nav hook: no paint within 12000ms" in str(exc.value)


@pytest.mark.asyncio
async def test_sharpness_exactly_at_threshold_is_not_blank():
    """Boundary: mean_sharpness EXACTLY == the fail threshold (1.0) must NOT hard-fail —
    the gate is a strict `<`, so a clip sitting on the line is not blank. The inert
    gate's own unmeasured note still fires (no `motion_timeline` in this fixture)."""
    out = await _run_record_qa(_sharpness_analyzer(1.0, passed=True))
    assert [c.line_id for c in out.clips] == ["clip-1"]
    assert not any("failed visual QA" in n or "blank" in n for n in out.notes)
    assert any("could not measure" in n and "UNPROVEN" in n for n in out.notes)


@pytest.mark.asyncio
async def test_integer_zero_sharpness_hard_fails():
    """A JSON integer 0 (not 0.0) for mean_sharpness — still blank, still hard-fails.
    Guards the float-or-None sanitiser against dropping a legitimate int-typed value."""
    with pytest.raises(RuntimeError, match=r"blank clip.* rejected"):
        await _run_record_qa(_sharpness_analyzer(0, passed=True))


# ── record · INERT-capture guard ───────────────────────────────────────────
#
# Prod run 01a00df3-772f-760b-9615-1d2bd78f7278 shipped a 61.47s "successful" demo that
# is 92% duplicate frames: the Playwright scenario's first step failed, nothing was ever
# clicked, and all four clips recorded the same static landing page — `focus_track`,
# `cursor_track` and `viewport` all null, motion 0.0412 on the paint sample then <=0.0033
# for 18-30s, and `notes == []`. The stage completed CLEAN. These tests pin the guard
# that makes that loud (some clips) or fatal (every clip).


def _inert_clip(clip_id: str, motions: list[float] | None) -> DemoVideoClip:
    """A clip with NO interaction tracks and the given motion curve (None = no curve)."""
    from src.services.stages.demo_video.record import MotionSample

    return DemoVideoClip(
        id=clip_id,
        line_id=clip_id,
        video_url=f"https://minio/{clip_id}.mp4",
        seconds=20.0,
        motion_timeline=(
            None
            if motions is None
            else [MotionSample(t=float(i), motion=m) for i, m in enumerate(motions)]
        ),
    )


# ── record · `_decide_no_scenario_passed` must not claim unmeasured == interaction ──
#
# The amplifier this whole fix targets: `_every_clip_is_inert` is the DECIDING branch
# for whether a demo_video run ships. When FTR says no scenario passed, an unmeasured
# clip used to make `_every_clip_is_inert` False, which took the "the capture DID
# record interaction" branch on ZERO evidence — a customer-facing lie, on a run that
# might genuinely be a static slideshow.


def test_inert_motion_threshold_sits_between_the_measured_populations():
    """The constant is calibrated, not magic: above the observed inert ceiling (0.0033)
    and below the page-paint spike (0.0412) from the prod run."""
    from src.services.stages.demo_video.record import _INERT_MOTION_MAX

    assert 0.0033 < _INERT_MOTION_MAX < 0.0412


@pytest.mark.asyncio
async def test_record_stage_degrades_an_all_inert_capture_end_to_end():
    """FLIPPED by #840 (was `..._hard_fails_...`). Through the real stage: the sole clip
    carries the prod curve and no tracks, so `handle` RETURNS with the clip plus inert
    notes (⇒ `degraded`) instead of raising."""
    analyzer = _motion_analyzer([{"t": 1.0, "motion": 0.0412}, {"t": 2.0, "motion": 0.0033}])
    out = await _run_record(analyzer)
    assert [c.line_id for c in out.clips] == ["clip-1"]
    assert any("clip-1 captured no interaction" in n for n in out.notes)
    assert any("inert capture — all 1 clip(s)" in n for n in out.notes)


@pytest.mark.asyncio
async def test_record_stage_leaves_a_moving_capture_unnoted():
    """Control: the same path with real motion returns normally and adds NO inert note."""
    analyzer = _motion_analyzer([{"t": 1.0, "motion": 0.0412}, {"t": 2.0, "motion": 0.42}])
    out = await _run_record(analyzer)
    assert [c.line_id for c in out.clips] == ["clip-1"]
    assert not any("captured no interaction" in n for n in out.notes)


# ── record · FTR VERDICT propagation (#840) ────────────────────────────────
#
# Measured on prod: FTR demotes a soft-failed scenario (`resilientClick`/`resilientFill`
# return WITHOUT throwing) to `failed` and reports it in `summary` + `test_results[]`.
# record read `scenario_clips` and NEVER touched `summary`, so a run whose every scenario
# failed at step 1 shipped as a clean stage with `notes == []` and a 61s static film.
# These tests pin the propagation: notes on a partial failure, a hard-fail on a total one,
# and total silence for an FTR too old to report a summary.

_STEP1_ERROR = (
    "locator.click: Timeout 30000ms exceeded waiting for "
    'getByRole("button", { name: "Sign in" }) ##SOFT_FAIL## '
) + ("x" * 600)


def _ftr_result(
    *,
    summary: dict[str, object] | None,
    test_results: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """An FTR job result carrying (or deliberately omitting) its verdict."""
    result: dict[str, object] = {"scenario_clips": [{"clip_key": "clip-1", "duration_ms": 2000}]}
    if summary is not None:
        result["summary"] = summary
    if test_results is not None:
        result["test_results"] = test_results
    return result


def _failed_test_result(name: str, *, error: str = _STEP1_ERROR) -> dict[str, object]:
    return {"name": name, "feature": "Todos", "status": "failed", "error": error}


def _record_client_with_verdict(result: dict[str, object]) -> AsyncMock:
    client = AsyncMock()
    client.start_execution.return_value = "ftr-qa"
    client.wait_for_execution.return_value = {}
    client.get_job_result.return_value = result
    client.get_clip_video.return_value = b"mp4bytes"
    return client


@pytest.mark.asyncio
async def test_record_stage_hard_fails_when_no_scenario_passed_and_nothing_was_captured():
    """Both signals agree the run is worthless → `handle` raises.

    FTR passed nothing AND the capture is inert (no tracks, flat motion after the
    page-paint sample). That is the measured 61s-static-film run, and it must still
    hard-fail — the honesty gate is not weakened, only made conditional."""
    analyzer = _motion_analyzer([{"t": 1.0, "motion": 0.0412}, {"t": 2.0, "motion": 0.0009}])
    client = _record_client_with_verdict(
        _ftr_result(
            summary={"total": 1, "passed": 0, "failed": 1},
            test_results=[_failed_test_result("adds a todo")],
        )
    )
    with pytest.raises(RuntimeError, match="passed 0 of 1 scenario"):
        await _run_record_with(client, analyzer)


@pytest.mark.asyncio
async def test_record_stage_degrades_when_no_scenario_passed_but_interaction_was_captured():
    """The recording is usable → keep it and degrade, do NOT throw it away.

    bdd #119 made a navigation visibility assert SOFT so the recording survives it,
    and FTR demotes ANY soft-failed test to `failed`. So "0 passed" can now mean
    "every scenario recorded fine but one decorative heading assert missed" —
    measured on E2E run 32253408902, where both scenarios were demoted solely for
    `getByRole('heading', { name: /TodoMVC/ })` against an app whose heading is
    `todos`. Hard-failing there destroys good footage AND the evidence that it was
    good, because the stage then stores no output at all."""
    analyzer = _motion_analyzer([{"t": 1.0, "motion": 0.5}, {"t": 2.0, "motion": 0.4}])
    client = _record_client_with_verdict(
        _ftr_result(
            summary={"total": 1, "passed": 0, "failed": 1},
            test_results=[_failed_test_result("adds a todo")],
        )
    )
    out = await _run_record_with(client, analyzer)
    assert out.clips, "usable footage must be KEPT, not discarded"
    blob = " ".join(out.notes)
    assert "capture DID record interaction" in blob
    assert "passed 0 of 1 scenario" in blob, "the operator must still see the verdict"
    assert "adds a todo" in blob, "and which scenario failed"


@pytest.mark.asyncio
async def test_record_stage_carries_a_partial_ftr_failure_into_notes_end_to_end():
    """Through the real stage: a partial failure still harvests the clip and surfaces the
    failed scenario in `notes` — the run degrades instead of reporting clean."""
    analyzer = _motion_analyzer([{"t": 1.0, "motion": 0.5}, {"t": 2.0, "motion": 0.4}])
    client = _record_client_with_verdict(
        _ftr_result(
            summary={"total": 2, "passed": 1, "failed": 1},
            test_results=[_failed_test_result("adds a todo")],
        )
    )
    out = await _run_record_with(client, analyzer)
    assert [c.line_id for c in out.clips] == ["clip-1"]
    assert any("FTR scenario Todos/adds a todo failed" in n for n in out.notes)


# ── The measured short-film defect (two real runs, 2026-09-16) ─────────────────
# A delivered film came in 7.6s short of its own footage: 62.120s of footage-pinned
# beats (min == max == the measured clip) delivered as 54.442s — every shot exactly
# its own NARRATION length, in BOTH directions (clip-5 shipped 12.333s over a 10.92s
# clip, i.e. LONGER than its footage), which is the signature of `bound is None`
# reaching `_reconcile_seconds`, not of a trim.
#
# ⚠️ THE SHAPE THAT MATTERS, and the reason the sibling tests above did not catch it:
# `_timeline_ctx` stamps the bounds on **demo_video_gen**'s lines. On a live run
# demo_video_gen / script_doctor carry `min_seconds` PRESENT with value **None**, and
# only `reconcile_timeline`'s row carries the VALUE. So the seam that actually ships
# is `resolve_plan_output` → reconcile's lines → `_min_max_by_line`, and a test that
# puts the values on gen never exercises the precedence hop at all.
_FILM_BEATS = (
    # (line_id, footage clip seconds, narration seconds)
    ("clip-1", 11.36, 8.98),
    ("clip-2", 11.28, 8.21),
    ("clip-3", 15.80, 12.53),
    ("clip-4", 12.76, 12.38),
    # ⭐ narration EXCEEDS the footage clip — the clamp must pull it DOWN to 10.92,
    # which is what proves the bound is applied in both directions.
    ("clip-5", 10.92, 12.35),
)


def _film_ctx(*, timeline_ordered: bool = True) -> RunContext:
    """The live prior-output shape: unbounded spine on demo_video_gen, the solver's
    `[min, max]` window only on reconcile_timeline (pinned min == max == clip).

    `timeline_ordered` is the run's own ORDER (`direction.timing` + a target), read
    off `config_snapshot`. It is a separate axis from reconcile_timeline's row on
    purpose: the whole defect is that the row cannot tell you whether bounds were
    ordered, and the config can.
    """
    gen_lines = [
        {
            "id": lid,
            "text": f"beat {lid}",
            "clip": lid,
            "seconds": 10.4,
            # PRESENT-but-None, exactly as the demo_video_gen row reads in prod.
            "min_seconds": None,
            "max_seconds": None,
        }
        for lid, _clip, _vo in _FILM_BEATS
    ]
    reconciled_lines = [
        {**line, "seconds": clip, "min_seconds": clip, "max_seconds": clip}
        for line, (_lid, clip, _vo) in zip(gen_lines, _FILM_BEATS, strict=True)
    ]
    deliverable: dict[str, Any] = {"captions_mode": "none"}
    direction: dict[str, Any] = {}
    if timeline_ordered:
        deliverable["target_duration_seconds"] = 60.0
        direction["timing"] = {"mode": "auto"}
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"deliverable": deliverable, "direction": direction},
        prior_run_outputs={
            "demo_video_gen": {"lines": gen_lines},
            "reconcile_timeline": {"lines": reconciled_lines},
            "demo_video_narrate": {
                "clips": [
                    {
                        "id": f"n-{lid}",
                        "line_id": lid,
                        "artifact_url": f"https://a/{lid}.wav",
                        "seconds": vo,
                    }
                    for lid, _clip, vo in _FILM_BEATS
                ]
            },
            "demo_video_record": {
                "clips": [
                    {
                        "id": lid,
                        "line_id": lid,
                        "video_url": f"https://v/{lid}.mp4",
                        "seconds": clip,
                    }
                    for lid, clip, _vo in _FILM_BEATS
                ]
            },
        },
    )


def test_compile_delivers_the_footage_length_not_the_narration_length():
    """End to end at the seam: every footage-pinned beat ships its CLIP length, and
    the total is the 62.120s footage sum — not the 54.442s narration sum that was
    actually delivered. Includes clip-5, where narration EXCEEDS the clip."""
    shots = {s["line_id"]: s for s in DemoVideoCompileStage().build_shots(_film_ctx())}
    for lid, clip, vo in _FILM_BEATS:
        assert shots[lid]["seconds"] == pytest.approx(clip), (
            f"{lid} shipped its narration ({vo}s) instead of its footage ({clip}s)"
        )
    total = sum(float(s["seconds"]) for s in shots.values())
    assert total == pytest.approx(62.12, abs=0.001)
    assert total != pytest.approx(54.442, abs=0.01)  # the delivered, narration-sum film


# ⛔ EVERY falsy state `run_dispatcher` can hand `prior_output("reconcile_timeline")`.
# `prior_run_outputs` is built at run_dispatcher.py:694-697 as
#   {so.stage_name: dict(so.output) if (completed and output) else {}
#    for so in rows if (completed and output) or skipped}
# so an ABSENT row and a COMPLETED row with a falsy output are both simply MISSING
# from the mapping, and a SKIPPED row is present as `{}`. All three are falsy and
# indistinguishable — which is why the guard must not be gated on that call. Driving
# all three against the incident's config reproduced the 7.670s shortfall with an
# EMPTY error log before this fix.
_RECONCILE_FALSY_STATES = {
    "absent_row": None,
    "completed_with_falsy_output": None,
    "skipped": {},
}


def _with_reconcile_state(ctx: RunContext, state: str) -> RunContext:
    seeded = _RECONCILE_FALSY_STATES[state]
    if seeded is None:
        ctx.prior_run_outputs.pop("reconcile_timeline", None)
    else:
        ctx.prior_run_outputs["reconcile_timeline"] = seeded
    return ctx


@pytest.mark.parametrize("state", list(sorted(_RECONCILE_FALSY_STATES))[:1])
def test_compile_names_the_short_film_in_every_falsy_reconcile_state(caplog, state: str):
    """The missing signal, in each of the three states the PR's own root cause lives
    in. The run ORDERED a reconciled timeline, every beat has paid, probed footage,
    and none has a window — so the film ships its narration sum. It must say so at
    ERROR, with the numbers, whatever reconcile's row looks like."""
    ctx = _with_reconcile_state(_film_ctx(), state)
    with caplog.at_level(logging.ERROR):
        shots = {s["line_id"]: s for s in DemoVideoCompileStage().build_shots(ctx)}
    assert "5 of 5 beat(s)" in caplog.text
    assert "62.120s of measured footage ships as 54.450s (short by 7.670s)" in caplog.text
    # Per-line, not just a total — the loss is attributable to a beat.
    for lid, clip, vo in _FILM_BEATS:
        assert f"{lid}: {clip:.3f}s -> {vo:.3f}s" in caplog.text
    # And this is what it costs, unguarded: the narration sum, not the footage sum.
    assert sum(float(s["seconds"]) for s in shots.values()) == pytest.approx(54.45, abs=0.01)


def test_compile_names_a_partial_bound_loss(caplog):
    """⛔ `_min_max_by_line` drops a line when EITHER bound is None and reconcile
    stamps PER-LINE, so the common shape is partial loss — and a map with one
    surviving entry is truthy. An all-or-nothing guard reports nothing here while
    4 of 5 beats ship short."""
    ctx = _film_ctx()
    lines = ctx.prior_run_outputs["reconcile_timeline"]["lines"]
    for line in lines[1:]:
        line["min_seconds"] = line["max_seconds"] = None
    with caplog.at_level(logging.ERROR):
        shots = {s["line_id"]: s for s in DemoVideoCompileStage().build_shots(ctx)}
    assert "4 of 5 beat(s)" in caplog.text
    assert "50.760s of measured footage ships as 45.470s (short by 5.290s)" in caplog.text
    assert "clip-1" not in caplog.text, "the beat that KEPT its window must not be blamed"
    assert sum(float(s["seconds"]) for s in shots.values()) == pytest.approx(56.83, abs=0.01)


@pytest.mark.parametrize("state", list(sorted(_RECONCILE_FALSY_STATES))[:1])
def test_compile_stays_quiet_when_no_timeline_was_ordered(caplog, state: str):
    """The case that must NOT fire, and the reason the guard reads the ORDER rather
    than reconcile's row: with no `direction.timing` block there was never a window to
    lose, so narration lengths are the CORRECT outcome (the pre-ORCH-1.4 topology).
    The footage is identical to the firing case above — only the order differs."""
    ctx = _with_reconcile_state(_film_ctx(timeline_ordered=False), state)
    with caplog.at_level(logging.ERROR):
        DemoVideoCompileStage().build_shots(ctx)
    assert "measured footage ships as" not in caplog.text


def test_compile_stays_quiet_for_beats_that_never_recorded_footage(caplog):
    """A beat with no MEASURED clip is legitimately unbounded — there is no paid
    footage for it to lose. The order is present and no beat has a window, so only
    the footage evidence keeps this quiet."""
    ctx = _with_reconcile_state(_film_ctx(), "skipped")
    # Clips harvested but never probed: `seconds` absent is NOT `seconds == 0`, and
    # neither is evidence of a length. A probe that cannot tell them apart is how
    # this defect shipped.
    for clip in ctx.prior_run_outputs["demo_video_record"]["clips"]:
        clip.pop("seconds")
    with caplog.at_level(logging.ERROR):
        DemoVideoCompileStage().build_shots(ctx)
    assert "measured footage ships as" not in caplog.text
