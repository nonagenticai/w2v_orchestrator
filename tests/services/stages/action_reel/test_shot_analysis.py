"""action_reel shot_analysis — what it ASKS the analyzer for.

The submit is the expensive part of this stage and the part that is invisible
when it goes wrong: every generation flag on ``AnalysisRequest`` defaults to
``True`` server-side, so a submit with no overrides runs a vision-LLM pass, a
per-frame OCR loop and a full Whisper transcription — none of which this stage
reads. A run that "works" is exactly what that bug looks like, so the overrides
are pinned here rather than left to a reviewer to notice.
"""

from __future__ import annotations

import json
from typing import Any, cast
from uuid import uuid7

import httpx

from src.clients.schemas.video_analyzer import VideoAnalyzerJobResult
from src.clients.video_analyzer_client import VideoAnalyzerClient
from src.services.runs.run_context import RunContext
from src.services.stages.action_reel.segment_select import _MOTION_SAMPLES_PER_SOURCE
from src.services.stages.action_reel.shot_analysis import (
    _CONCURRENCY,
    _FRAME_INTERVAL_SECONDS,
    _MAX_FRAMES,
    _SAMPLING_OVERRIDES,
    ShotAnalysisInput,
    ShotAnalysisStage,
)

_RESULT = {
    "job_id": "job-1",
    "status": "completed",
    "stage": "done",
    "progress": 1.0,
    "duration_seconds": 42.0,
    "scene_boundaries": [4.0, 12.5],
    "quality_metrics": {
        "mean_sharpness": 0.7,
        "mean_brightness": 0.5,
        "motion_score": 0.6,
        "dominant_colors": ["#101010"],
        "overall_score": 0.65,
        "motion_timeline": [{"t": 1.0, "motion": 0.4}],
    },
}


# The SAME completed job, minus the quality stage. The analyzer returns this
# whenever `compute_quality_metrics` produced nothing — the job succeeded, the
# duration and the scene cuts are real, and there are no measurements.
_RESULT_WITHOUT_QUALITY = {**_RESULT, "quality_metrics": None}


class _RecordingClient:
    """Records the submit kwargs; answers the poll/fetch with a canned result.

    ``results`` is keyed by the submitted url so one fan-out can mix a measured
    source with an unmeasured one — the case the aggregate hard-fail in ``handle``
    means a single-source test can never reach.
    """

    def __init__(self, results: dict[str, dict[str, Any]] | None = None) -> None:
        self.submits: list[dict[str, Any]] = []
        self._results = results or {}
        self._jobs: dict[str, str] = {}

    async def start_analysis(self, **kwargs: Any) -> str:
        self.submits.append(kwargs)
        job_id = f"job-{len(self.submits)}"
        self._jobs[job_id] = str(kwargs.get("video_url") or "")
        return job_id

    async def poll_analysis(self, job_id: str, **_kwargs: Any) -> dict[str, Any]:
        return {"job_id": job_id, "status": "completed"}

    async def get_result(self, _job_id: str) -> VideoAnalyzerJobResult:
        url = self._jobs.get(_job_id, "")
        return VideoAnalyzerJobResult.model_validate(self._results.get(url, _RESULT))


def _ctx(sources: list[dict[str, Any]]) -> RunContext:
    return RunContext(
        run_id=uuid7(),
        deliverable_id=uuid7(),
        config_snapshot={},
        prior_run_outputs={"footage_probe": {"sources": sources}},
    )


def _stage(client: _RecordingClient) -> ShotAnalysisStage:
    return ShotAnalysisStage(client=cast("VideoAnalyzerClient", client))


async def test_the_submit_disables_every_pipeline_this_stage_never_reads() -> None:
    """THE test. Each flag is a real early-return in the analyzer's pipeline; the
    walkthrough flag additionally gates Whisper (``_needs_transcript``). A submit
    that drops these buys a vision + OCR + transcription run per source, per run,
    and throws all of it away."""
    client = _RecordingClient()
    ctx = _ctx([{"id": "src-1", "preview_url": "https://example.invalid/p.mp4"}])

    await _stage(client).handle(ShotAnalysisInput(), ctx)

    assert len(client.submits) == 1
    overrides = client.submits[0]["config_overrides"]
    assert overrides == {
        "generate_walkthrough": False,
        "generate_ui_inventory": False,
        "generate_frame_index": False,
        "extract_frame_descriptions": False,
        "extract_ocr": False,
        # ON, and named explicitly: it gates the sharpness/brightness/motion
        # numbers segment_select actually selects on.
        "compute_quality_metrics": True,
        # The sampling density. Without these the submit inherits the fleet
        # ConfigMap's 3.0 s interval, which the analyzer's MIN_FRAMES_FOR_QUALITY
        # floor turns into a 1.8 s lattice on a 9 s clip -- coarser than a beat.
        "frame_interval_seconds": 0.25,
        "max_frames": 25,
    }


async def test_the_measurements_this_stage_reads_still_arrive() -> None:
    """The overrides must not have turned off something the stage consumes — the
    three fields below are the entire reason the analyzer is called."""
    client = _RecordingClient()
    ctx = _ctx([{"id": "src-1", "preview_url": "https://example.invalid/p.mp4"}])

    out = await _stage(client).handle(ShotAnalysisInput(), ctx)

    assert [s.id for s in out.sources] == ["src-1"]
    measured = out.sources[0]
    assert measured.error == ""
    assert measured.duration_s == 42.0
    assert measured.scene_boundaries == [4.0, 12.5]
    assert measured.mean_sharpness == 0.7
    assert measured.overall_score == 0.65
    assert [(p.t, p.motion) for p in measured.motion_timeline] == [(1.0, 0.4)]


async def test_every_source_is_submitted_with_the_same_overrides() -> None:
    """A fan-out must not let one source through on the defaults."""
    client = _RecordingClient()
    ctx = _ctx(
        [
            {"id": "src-1", "preview_url": "https://example.invalid/a.mp4"},
            {"id": "src-2", "original_url": "https://example.invalid/b.mp4"},
        ]
    )

    await _stage(client).handle(ShotAnalysisInput(), ctx)

    assert len(client.submits) == 2
    assert all(s["config_overrides"]["extract_ocr"] is False for s in client.submits)
    assert all(s["config_overrides"]["compute_quality_metrics"] is True for s in client.submits)


async def test_a_result_without_quality_metrics_is_reported_as_unmeasured() -> None:
    """A completed job carrying no ``quality_metrics`` used to become an
    ``AnalyzedSource`` with every score 0.0 and an EMPTY ``error`` — "measured,
    and it scored zero" for something nobody measured. That is the exact
    conflation the ``error`` field exists to prevent, and downstream it makes a
    perfectly usable clip the worst in the corpus rather than an unknown one.

    The real numbers on that result — duration, scene cuts — are kept: they come
    off the job, not off the quality stage, and the selector cuts on them.
    """
    unmeasured_url = "https://example.invalid/b.mp4"
    client = _RecordingClient({unmeasured_url: _RESULT_WITHOUT_QUALITY})
    ctx = _ctx(
        [
            {"id": "src-1", "preview_url": "https://example.invalid/a.mp4"},
            {"id": "src-2", "preview_url": unmeasured_url},
        ]
    )

    out = await _stage(client).handle(ShotAnalysisInput(), ctx)

    measured, unmeasured = out.sources
    assert measured.error == ""
    assert measured.overall_score == 0.65
    # The whole test: this source is NOT reported as measured.
    assert unmeasured.error
    assert "quality_metrics" in unmeasured.error
    # ...and what WAS real on the result survives, so the selector can still cut.
    assert unmeasured.duration_s == 42.0
    assert unmeasured.scene_boundaries == [4.0, 12.5]
    # ...and the run says so where a reviewer reads it.
    assert any("src-2" in note for note in out.notes)


async def test_the_sampling_override_reaches_the_analyzer_request_body() -> None:
    """Not a source grep, and not the fake client's kwargs either — the assertion
    is on the JSON httpx actually PUTS ON THE WIRE.

    ``VideoAnalyzerClient.start_analysis`` merges ``config_overrides`` into the
    payload with ``payload.update(...)``, so a key that is not a real
    ``AnalysisRequest`` field would be silently dropped server-side and a
    kwargs-level assertion would still pass. This drives the stage through a REAL
    ``VideoAnalyzerClient`` over an ``httpx.MockTransport`` and reads the body,
    which is the only place the two failure modes (never passed / passed under a
    name the server ignores) are distinguishable.

    ``frame_interval_seconds`` and ``max_frames`` are declared on
    ``video_analyzer_service:src/api/schemas.py::AnalysisRequest`` and consumed at
    ``pipeline.py`` (``request.frame_interval_seconds or settings.…``), which is
    why the names below are spelled out rather than read back off the module.
    """
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/analyze" and request.method == "POST":
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={"job_id": "job-1"})
        if request.url.path.endswith("/result"):
            return httpx.Response(200, json=_RESULT)
        return httpx.Response(200, json={"job_id": "job-1", "status": "completed"})

    client = VideoAnalyzerClient(base_url="http://analyzer.invalid", license_token="tok")
    client.set_transport(httpx.MockTransport(handler))
    try:
        await ShotAnalysisStage(client=client).handle(
            ShotAnalysisInput(),
            _ctx([{"id": "src-1", "preview_url": "https://example.invalid/p.mp4"}]),
        )
    finally:
        await client.close()

    assert len(bodies) == 1, "the stage did not submit exactly one analysis job"
    body = bodies[0]
    assert body["frame_interval_seconds"] == 0.25, (
        "the sampling interval never reached the analyzer: the submit inherits "
        "PIPELINE_FRAME_INTERVAL_SECONDS=3.0 and every cut is pinned to a ~1.8s lattice"
    )
    assert body["max_frames"] == 25, (
        "the frame budget never reached the analyzer; it must stay tied to "
        "segment_select's _MOTION_SAMPLES_PER_SOURCE + 1"
    )


def test_the_frame_budget_is_derived_from_the_prompt_cap_not_a_literal() -> None:
    """The cost of this stage is ~2.0s of 4K ffmpeg decode PER FRAME, and every
    frame past ``_MOTION_SAMPLES_PER_SOURCE + 1`` is downsampled away before the
    selector ever sees it. Pinning the derivation keeps a future raise of the
    prompt cap from leaving the sampling behind (and a future literal here from
    paying for frames nobody reads)."""
    assert _MAX_FRAMES == _MOTION_SAMPLES_PER_SOURCE + 1
    assert _SAMPLING_OVERRIDES == {
        "frame_interval_seconds": _FRAME_INTERVAL_SECONDS,
        "max_frames": _MAX_FRAMES,
    }


def test_concurrency_stays_at_the_memory_safe_bound() -> None:
    """4 concurrent analyses OOMKill the analyzer. This is a MEASURED ceiling.

    Measured inside a live ``video-analyzer-service`` pod (cpu=2, memory=4Gi) on
    real 4K HEVC drone clips, once this stage began asking for a real frame budget:

        1 ffmpeg 4K seek-decode   444 MiB RSS   (pod baseline ~400 MiB)
        3 concurrent              1850 / 4096 MiB, oom_kill 0        -> safe
        4 concurrent              OOMKilled, exitCode 137, restarted -> FATAL

    Cost is ``~400 + 444 * jobs`` — linear in JOB COUNT, and near-independent of
    ``max_frames``, so trimming frames does NOT buy headroom here; only this number
    does.

    Two reasons this cannot be relaxed back on a hunch:

    * The analyzer has NO self-protection. Its ``PIPELINE_MAX_CONCURRENT_JOBS`` /
      ``settings.max_concurrent_jobs`` is declared at ``src/config.py:65`` and read
      NOWHERE — no semaphore, no queue, no ``limit_concurrency``. So this constant,
      divided by however many replicas a request lands on, is the only bound that
      exists.
    * Replica count is not a defence: there is no HPA for that Deployment and the
      count was observed drifting 2 -> 3 mid-session. A Service does not promise an
      even split either, so the worst case to reason about is ALL of them on one pod.
      At 2 that is the measured-safe case; at 4 it is the measured-fatal one.

    Raising this again means re-running the concurrency measurement first — and
    note it also doubles ``SHOT_ANALYSIS_TIMEOUT_SECONDS`` through the wave
    arithmetic, so it is never a one-line change.
    """
    assert _CONCURRENCY <= 2, (
        f"_CONCURRENCY is {_CONCURRENCY}; 3 was measured at 1850/4096 MiB and 4 "
        "OOMKilled the analyzer pod (exitCode 137). Re-measure before raising."
    )
    # Not vacuous in the other direction either: 0 or 1 would silently serialise a
    # ten-clip shoot into ten sequential 52s analyses.
    assert _CONCURRENCY >= 2
