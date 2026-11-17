"""The zero-footage gate must be OPERABLE, and its justification must be TRUE.

Two measured defects are pinned here:

1. ``record`` never sent FTR's ``expect_recording``, which defaults ``True`` and makes
   ``_guard_zero_footage`` FAIL the job on an empty ``scenario_clips``. That upstream
   failure surfaced out of ``wait_for_execution``, so ``get_job_result`` was never
   reached and ``_require_footage`` — including the ``PIPELINE_DEMO_REQUIRE_FOOTAGE=false``
   narrated-text-card fallback its own raise message advertises — could never run.
2. Six sites justified hard-failing with a "poison-job refund" that does not exist
   (``reconcile_timeline.py:1006`` states this as measured fact).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.services.stages.demo_video import record as record_mod
from src.services.stages.demo_video.record import DemoVideoRecordStage

from .test_stages import _ctx, _passing_analyzer, _qa_settings


def _ftr_client(*, result: dict[str, Any], job_id: str = "ftr-job") -> AsyncMock:
    """An FTR client stub that ENFORCES the real ``_guard_zero_footage`` contract.

    ``frontend_test_runner_service/src/services/execution_engine.py:1688``: a run that
    persists zero ``scenario_clips`` FAILS unless the caller sent
    ``expect_recording=false``. A plain ``AsyncMock`` cannot catch defect 1 because it
    happily returns an empty result the real service would never hand back — so the
    stub raises out of ``wait_for_execution`` exactly where the service does.
    """
    client = AsyncMock()
    client.start_execution.return_value = job_id

    async def _wait(*_args: object, **_kwargs: object) -> dict[str, Any]:
        sent = client.start_execution.call_args.args[0]
        if not result.get("scenario_clips") and sent.get("expect_recording", True):
            raise RuntimeError(f"FTR job {job_id} failed: run persisted 0 scenario clips")
        return {}

    client.wait_for_execution.side_effect = _wait
    client.get_job_result.return_value = result
    client.get_clip_video.return_value = b"mp4"
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize("require_footage", [True])
async def test_record_sends_expect_recording_matching_the_footage_setting(require_footage):
    """The request carries ``expect_recording`` driven by ``demo_require_footage``.

    Both values, because the bug was the field being ABSENT: FTR then defaulted it to
    True and owned the zero-clip verdict unconditionally.
    """
    client = _ftr_client(
        result={"scenario_clips": [{"clip_key": "clip-1", "duration_ms": 1000}]},
        job_id="ftr-expect",
    )
    gen = {"bdd_job_id": "bdd-1", "lines": [{"id": "clip-1"}]}
    stage = DemoVideoRecordStage(client=client, analyzer=_passing_analyzer())
    with (
        patch(
            "src.services.stages.demo_video.record.get_settings",
            return_value=_qa_settings(demo_require_footage=require_footage),
        ),
        patch(
            "src.services.stages.demo_video.record.store_capture_clip",
            new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['clip_id']}.mp4"),
        ),
    ):
        await stage.handle(
            stage.input_model(),
            _ctx({"target_url": "https://app.example"}, prior={"demo_video_gen": gen}),
        )
    sent = client.start_execution.call_args.args[0]
    assert sent["expect_recording"] is require_footage


@pytest.mark.asyncio
async def test_zero_clips_with_require_footage_off_reaches_the_text_card_fallback():
    """require_footage=false + zero clips → the documented degrade, NOT an upstream raise.

    Before the fix this died inside ``wait_for_execution`` with a generic FTR poll
    error, so the fallback ``_require_footage`` promises was unreachable in production.
    """
    client = _ftr_client(result={"scenario_clips": []}, job_id="ftr-zero-off")
    gen = {"bdd_job_id": "bdd-1", "lines": [{"id": "clip-1"}]}
    stage = DemoVideoRecordStage(client=client, analyzer=_passing_analyzer())
    with patch(
        "src.services.stages.demo_video.record.get_settings",
        return_value=_qa_settings(demo_require_footage=False),
    ):
        out = await stage.handle(
            stage.input_model(),
            _ctx({"target_url": "https://app.example"}, prior={"demo_video_gen": gen}),
        )
    # The orchestrator's OWN gate decided, and it decided to degrade.
    assert client.get_job_result.await_count == 1
    assert out.clips == []
    assert out.notes and "no recorded scenario clips" in out.notes[0]


@pytest.mark.asyncio
async def test_zero_clips_with_require_footage_on_still_hard_fails_upstream():
    """The gate ON keeps FTR's cheaper early failure — the fix is not a weakening."""
    client = _ftr_client(result={"scenario_clips": []}, job_id="ftr-zero-on")
    gen = {"bdd_job_id": "bdd-1", "lines": [{"id": "clip-1"}]}
    stage = DemoVideoRecordStage(client=client, analyzer=_passing_analyzer())
    with (
        patch(
            "src.services.stages.demo_video.record.get_settings",
            return_value=_qa_settings(demo_require_footage=True),
        ),
        pytest.raises(RuntimeError, match="0 scenario clips"),
    ):
        await stage.handle(
            stage.input_model(),
            _ctx({"target_url": "https://app.example"}, prior={"demo_video_gen": gen}),
        )
    assert client.start_execution.call_args.args[0]["expect_recording"] is True


# ── Defect 2: no source line may CLAIM a refund exists ───────────────────────

# A mention of "refund" is only allowed when it is DENYING one. Anything else is the
# defect: six raise-justifications cited a "poison-job refund path" that
# `reconcile_timeline.py:1006` had already measured as nonexistent.
_REFUND_DENIAL = re.compile(
    r"(no (automatic )?refund|not? refund (path|follows)|grep -rn refund)", re.IGNORECASE
)


def test_no_source_line_claims_a_refund_path_exists():
    """Guard: ``grep -rn refund`` over this repo's ``src/`` finds DOCSTRINGS ONLY.

    ``run_usage_reconciler`` mirrors usage read-only; a raise here lands the run at
    ``status=failed`` and FORFEITS the spend already incurred. Raising is still correct
    — shipping a worthless deliverable is worse — but any wording implying the customer
    is made whole is false, and this is the cheap text guard that would have caught it.
    """
    # Scan the WHOLE ``src/`` package, not one file. Scoping this guard to
    # ``record.py`` is what let a false refund claim survive at
    # ``final_critique.py:147`` while the guard stayed green; a refund claim is
    # equally false in a route handler or in ``run_usage_reconciler``. Rooting at
    # ``src/`` also keeps ``tests/`` — which mentions "refund" legitimately, right
    # here — out of the scan by construction.
    src_root = Path(record_mod.__file__).parents[3]
    assert src_root.name == "src", f"expected the src/ package root, got {src_root}"
    offenders: list[str] = []
    for path in sorted(src_root.rglob("*.py")):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "refund" in line.lower() and not _REFUND_DENIAL.search(line):
                rel = path.relative_to(src_root.parent)
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, (
        "source line(s) claim a refund path that does not exist "
        "(see reconcile_timeline.py:1006):\n" + "\n".join(offenders)
    )


def test_blank_diag_text_surfaces_the_stamped_feature_and_scenario():
    """``_ClipDiag.feature``/``.scenario`` were stamped but never rendered, so the
    post-mortem named a clip id an operator had to map back to a spec by hand."""
    diag = record_mod._ClipDiag(
        feature="checkout.feature",
        scenario="user pays",
        ftr_status="timedOut",
        ftr_error="nav hook: no paint within 12000ms",
        duration_ms=1200.0,
        width=1280,
        height=720,
        size_bytes=42,
    )
    text = record_mod._blank_diag_text("clip-1", diag)
    assert "checkout.feature::user pays" in text
    assert "nav hook: no paint" in text
    # The empty sentinel (a clip with no FTR diagnostic at all) still short-circuits.
    assert record_mod._blank_diag_text("clip-9", record_mod._ClipDiag()) == (
        "no FTR diagnostic for clip clip-9"
    )
