"""FinalCritiqueStage — the A11 advisory ship gate (PR-3, LLM-stage re-drive).

Exercises the soft-gate invariant (never raise except skip / genuine missing
artifact), the WM short-circuit on a ship-ready gestalt, the directive clamp, the
loudness window, and — new in PR-3 — the bounded ``mode == "auto"`` re-drive that
re-RUNS the owning per-dimension LLM stage named by the film-critic: footage /
sound feed compile directly, voice re-drives narration via a narrate re-synth. It
threads the critic feedback into that stage, merges the regenerated items back by id
(only the flagged ids), re-compiles once, accepts an improvement, is hard-capped at
≤2 passes, soft-degrades a director / narrate / compile failure, never regresses (a
worse pass ships the earlier artifact), drops a directive naming a stage that never
ran, never re-drives the cyclic footage_critique stage, and stays byte-identical (no
director / narrate / compile) under critique_only / off. The VideoAnalyzerClient +
WMClient are mocked; ``DemoVideoCompileStage`` / the three director stages /
``DemoVideoNarrateStage`` are patched where final_critique imports them (no live
analyzer / worker / compiler / TTS).
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any, cast
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

from src.clients.schemas.video_analyzer import (
    VideoAnalyzerJobResult,
    VideoAnalyzerQualityMetrics,
)
from src.config import clear_settings_cache
from src.services.runs.run_context import RunContext
from src.services.stages.base import StagePreconditionError, StageSkippedError
from src.services.stages.demo_video.compile import DemoVideoCompileOutput
from src.services.stages.demo_video.final_critique import (
    _ANALYSIS_POLL_TIMEOUT_S,
    _LLM_LADDER_WORST_CASE_S,
    _STAGE_TIMEOUT_S,
    FinalCritiqueInput,
    FinalCritiqueStage,
    ShipGateEnforcementError,
)
from src.services.stages.demo_video.narrate import DemoVideoAudioClip, DemoVideoNarrateOutput
from src.services.stages.demo_video.voice_director import (
    PerformanceDirective,
    VoiceDirectorOutput,
)
from src.services.stages.shared.footage_direction import FootageDirectionOutput, ShotPlan
from src.services.stages.shared.gestalt_metrics import sample_frames_at_cuts
from src.services.stages.shared.sound_director import LineSfxCue, SoundDirectorOutput

_COMPILE_PATCH = "src.services.stages.demo_video.final_critique.DemoVideoCompileStage"
_FOOTAGE_PATCH = "src.services.stages.demo_video.final_critique.DemoVideoFootageDirectionStage"
_SOUND_PATCH = "src.services.stages.demo_video.final_critique.DemoVideoSoundDirectorStage"
_VOICE_PATCH = "src.services.stages.demo_video.final_critique.VoiceDirectorStage"
_NARRATE_PATCH = "src.services.stages.demo_video.final_critique.DemoVideoNarrateStage"


@pytest.fixture(autouse=True)
def _disarm_loudness_for_the_advisory_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """DISARM the loudness term for this module — an explicit opt-out, not a workaround.

    Nearly every test here sets ``integrated_lufs: -6.0`` as its "make the gestalt
    not-ship-ready so the film-critic is consulted" lever; the SUBJECT of those tests is
    the advisory behaviour (directive clamping, re-drive bookkeeping, spine evidence,
    static-input findings), not the ship decision. Once the loudness term is ARMED, that
    one fixture constant — a genuinely out-of-window master — correctly FAILS the run, and
    45 of them stopped exercising their own subject.

    So they opt out of BLOCKING and keep testing what they are about. Arming is a
    separate axis with its own tests in ``tests/unit/test_ship_gate_arming.py`` plus the
    end-to-end blocking tests at the bottom of this file, which set this env var back to
    ``true`` explicitly. ⛔ This fixture must never grow to disarm a term whose blocking
    behaviour a test in this module is asserting — the three arming tests below
    deliberately re-arm rather than rely on the default."""
    monkeypatch.setenv("PIPELINE_SHIP_GATE_LOUDNESS_ARMED", "false")
    clear_settings_cache()


def _armed(monkeypatch: pytest.MonkeyPatch, **terms: bool) -> None:
    """Arm/disarm named ship-gate terms for one test via the real env kill switches.

    Goes through ``PIPELINE_SHIP_GATE_<TERM>_ARMED`` + ``clear_settings_cache`` rather
    than patching the helper, so a test that claims the switch works is exercising the
    switch a GitOps configmap would actually flip."""
    for term, on in terms.items():
        monkeypatch.setenv(f"PIPELINE_SHIP_GATE_{term.upper()}_ARMED", "true" if on else "false")
    clear_settings_cache()


def _term_row(record: dict[str, Any], term: str) -> dict[str, Any]:
    """The one ship-gate record row for ``term``. KeyError-loud if it is missing — every
    term must appear on every record, armed or not, so an absent row is a defect."""
    rows = cast("list[dict[str, Any]]", record["terms"])
    return next(r for r in rows if r["term"] == term)


def _scored(
    overall: float,
    *,
    directives: list[dict[str, Any]] | None = None,
    findings: list[dict[str, Any]] | None = None,
) -> WMResponse:
    """A well-formed film-critic reply with a chosen overall + optional directives."""
    obj: dict[str, Any] = {
        "cohesion": 3,
        "continuity": 3,
        "momentum_experienced": 3,
        "tonal_consistency": 3,
        "overall": overall,
    }
    if directives is not None:
        obj["directives"] = directives
    if findings is not None:
        obj["findings"] = findings
    return _resp(obj)


# Film-critic directives naming an owning per-dimension LLM stage the re-drive
# re-runs (PR-3), each flagging clip-1.
_REDRIVE_DIRECTIVE = {
    "stage": "demo_video_footage_direction",
    "spine_ids": ["clip-1"],
    "reason": "weak seam",
}
_SOUND_DIRECTIVE = {
    "stage": "demo_video_sound_director",
    "spine_ids": ["clip-1"],
    "reason": "flat sfx",
}
_VOICE_DIRECTIVE = {
    "stage": "voice_director",
    "spine_ids": ["clip-1"],
    "reason": "flat delivery",
}


def _compiled(
    url: str,
    *,
    lufs: float = -6.0,
    sha: str = "sha-redriven",
    shot_timeline: list[dict[str, Any]] | None = None,
) -> DemoVideoCompileOutput:
    """A recompiled artifact. Default out-of-window loudness keeps the re-analysis
    not-ship-ready so the film-critic is consulted again on the new draft."""
    return DemoVideoCompileOutput(
        artifact_url=url,
        duration_seconds=11.0,
        loudness={"integrated_lufs": lufs},
        output_sha256=sha,
        shot_timeline=shot_timeline,
    )


def _patch_compile(handle: AsyncMock) -> Any:
    """Patch DemoVideoCompileStage where final_critique imports it; the class is
    constructed per re-compile, so bind ``handle`` on the shared return_value."""
    mgr = patch(_COMPILE_PATCH)
    mock_cls = mgr.start()
    mock_cls.return_value.handle = handle
    return mgr, mock_cls


class _Redrive:
    """The five patched re-drive stage classes. Each ``.handle`` is an ``AsyncMock``
    bound on the class's shared ``return_value`` (the class is constructed per
    re-drive), mirroring ``_patch_compile``."""

    def __init__(self, mocks: dict[str, Any]) -> None:
        self.compile = mocks["compile"]
        self.footage = mocks["footage"]
        self.sound = mocks["sound"]
        self.voice = mocks["voice"]
        self.narrate = mocks["narrate"]

    def handle(self, name: str) -> AsyncMock:
        return getattr(self, name).return_value.handle


def _patch_redrive(
    *,
    compile_handle: AsyncMock | None = None,
    footage_handle: AsyncMock | None = None,
    sound_handle: AsyncMock | None = None,
    voice_handle: AsyncMock | None = None,
    narrate_handle: AsyncMock | None = None,
) -> tuple[ExitStack, _Redrive]:
    """Patch compile + all three directors + narrate where final_critique imports
    them. Unpassed handles default to an unconfigured ``AsyncMock`` so a test can
    assert the stage was never called. Close the returned ``ExitStack`` to unpatch."""
    stack = ExitStack()
    specs = {
        "compile": (_COMPILE_PATCH, compile_handle),
        "footage": (_FOOTAGE_PATCH, footage_handle),
        "sound": (_SOUND_PATCH, sound_handle),
        "voice": (_VOICE_PATCH, voice_handle),
        "narrate": (_NARRATE_PATCH, narrate_handle),
    }
    mocks: dict[str, Any] = {}
    for name, (path, handle) in specs.items():
        mock_cls = stack.enter_context(patch(path))
        mock_cls.return_value.handle = handle if handle is not None else AsyncMock()
        mocks[name] = mock_cls
    return stack, _Redrive(mocks)


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics={"total_tokens": 10},
    )


# ── What a MEASURABLE film looks like on the REAL analyzer wire ───────────────────
# The analyzer's per-frame records carry only {index, timestamp, path, description,
# ocr_text, labels} — there is NO per-frame brightness / sharpness / dominant_color
# (video_analyzer_service pipeline.py::_frame_analysis_to_dict, mirrored in
# src/clients/schemas/video_analyzer.py). `_feature_vector` therefore substitutes
# WHOLE-VIDEO means for two components and pads [0,0,0] for colour, so **motion is the
# only component that can differ between shots**. A fixture whose motion is uniform
# therefore produces IDENTICAL feature vectors — zero dispersion — which the gate now
# (correctly) reports as `insufficient_evidence`, NOT as a coherent film.
#
# So: the default fixture varies motion across the cuts. Anything else would be a film
# the gate cannot measure, and a suite built on it would assert against fabricated
# constants. Delta 0.6-0.2 = 0.4 stays under the 0.5 continuity ceiling, so a default
# film is coherent-and-measurable rather than coherent-by-vacuity.
_TWO_CUTS: list[float] = [2.0, 6.0]
_MEASURABLE_MOTION: list[dict[str, Any]] = [
    {"t": 2.0, "motion": 0.6},
    {"t": 6.0, "motion": 0.2},
]


def _degenerate_analyzer(**kwargs: Any) -> AsyncMock:
    """The PROD shape: cuts exist but every sampled shot is identical in feature space
    (uniform motion ⇒ five constant components + one constant motion). The gate must
    report this as unmeasurable, never as a coherent film."""
    kwargs.setdefault("scene_boundaries", list(_TWO_CUTS))
    kwargs.setdefault("motion", [{"t": t, "motion": 0.0} for t in _TWO_CUTS])
    return _analyzer(**kwargs)


def _analyzer(
    *,
    scene_boundaries: list[float] | None = _TWO_CUTS,
    frame_analyses: list[dict[str, Any]] | None = None,
    motion: list[dict[str, Any]] | None = None,
    visual_timeline: list[dict[str, Any]] | None = None,
    duration: float = 10.0,
) -> AsyncMock:
    # `motion=None` means "unspecified" ⇒ the measurable default, NOT an empty timeline.
    # An empty timeline zeroes every motion sample and silently makes the film degenerate.
    motion = _MEASURABLE_MOTION if motion is None else motion
    analyzer = AsyncMock()
    analyzer.start_analysis.return_value = "job-1"
    analyzer.poll_analysis.return_value = {"status": "completed"}
    analyzer.get_result.return_value = VideoAnalyzerJobResult(
        job_id="job-1",
        status="completed",
        stage="done",
        progress=1.0,
        duration_seconds=duration,
        scene_boundaries=scene_boundaries,
        frame_analyses=frame_analyses,
        quality_metrics=VideoAnalyzerQualityMetrics.model_validate(
            {
                "motion_timeline": motion,
                # Per-frame visual curve (analyzer PR #73). Absent ⇒ empty ⇒ the consumer
                # falls back to whole-video means (constant per shot); present ⇒ each shot
                # gets its own brightness/sharpness/colour, so cohesion/continuity vary.
                "visual_timeline": visual_timeline or [],
                "mean_sharpness": 50.0,
                "mean_brightness": 120.0,
                "motion_score": 0.5,
                "dominant_colors": ["#ffffff"],
                "overall_score": 3.0,
            }
        ),
    )
    return analyzer


def _prior(
    *,
    footage: dict[str, Any] | None = None,
    loudness: dict[str, Any] | None = None,
    sha: str = "sha-abc",
    lines: tuple[str, ...] = ("clip-1", "clip-2"),
    compile_url: str = "http://v/orig.mp4",
    sound: dict[str, Any] | None = None,
    voice: dict[str, Any] | None = None,
    narrate: dict[str, Any] | None = None,
    line_overrides: dict[str, dict[str, Any]] | None = None,
    shot_timeline: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    compile_out: dict[str, Any] = {
        "artifact_url": compile_url,
        "duration_seconds": 10.0,
        "output_sha256": sha,
    }
    if loudness is not None:
        compile_out["loudness"] = loudness
    # A11 Workstream B: the compiler's AUTHORITATIVE rendered shot_timeline (opt-in).
    if shot_timeline is not None:
        compile_out["shot_timeline"] = shot_timeline
    # Per-line A11 PR-4 author overrides (pin / continuity_exempt) merged onto each
    # spine line dict, e.g. line_overrides={"clip-1": {"pin": True}}.
    overrides = line_overrides or {}
    prior: dict[str, Any] = {
        "demo_video_compile": compile_out,
        "demo_video_gen": {"lines": [{"id": lid, **overrides.get(lid, {})} for lid in lines]},
        # The L1 footage-direction plan baseline a footage re-drive merges onto.
        "demo_video_footage_direction": {
            "plans": [{"line_id": lid, "zoom": None, "reframe": None} for lid in lines]
        },
    }
    if footage is not None:
        prior["demo_video_footage_critique"] = footage
    # Optional owning-stage baselines a sound / voice re-drive merges onto. Absent by
    # default so a stage-didn't-run drop can be exercised.
    if sound is not None:
        prior["demo_video_sound_director"] = sound
    if voice is not None:
        prior["voice_director"] = voice
    if narrate is not None:
        prior["demo_video_narrate"] = narrate
    return prior


def _ctx(prior: dict[str, Any], config: dict[str, Any] | None = None) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"ship_gate": {"mode": "critique_only"}} if config is None else config,
        prior_run_outputs=prior,
    )


def _stage(wm: AsyncMock, analyzer: AsyncMock) -> FinalCritiqueStage:
    return FinalCritiqueStage(client=wm, analyzer=analyzer)


def _assert_no_redrive_stage_touched(m: _Redrive) -> None:
    """Byte-identical opt-out: NO director / narrate / compile stage constructed."""
    for name in ("compile", "footage", "sound", "voice", "narrate"):
        m.handle(name).assert_not_called()


# ── 1. Skip-clean: absent ship_gate AND mode="off" ⇒ StageSkippedError ────────
async def test_skip_clean_when_ship_gate_absent() -> None:
    wm, analyzer = AsyncMock(), AsyncMock()
    stack, m = _patch_redrive()
    try:
        with pytest.raises(StageSkippedError) as exc:
            await _stage(wm, analyzer).handle(FinalCritiqueInput(), _ctx(_prior(), config={}))
    finally:
        stack.close()
    assert exc.value.missing == ["ship_gate"]
    # Byte-identical opt-out: no analyzer, no worker, no re-drive stage.
    analyzer.start_analysis.assert_not_called()
    wm.message.assert_not_called()
    _assert_no_redrive_stage_touched(m)


async def test_skip_clean_when_mode_off() -> None:
    wm, analyzer = AsyncMock(), AsyncMock()
    ctx = _ctx(_prior(), config={"ship_gate": {"mode": "off"}})
    stack, m = _patch_redrive()
    try:
        with pytest.raises(StageSkippedError):
            await _stage(wm, analyzer).handle(FinalCritiqueInput(), ctx)
    finally:
        stack.close()
    analyzer.start_analysis.assert_not_called()
    wm.message.assert_not_called()
    _assert_no_redrive_stage_touched(m)


# ── genuine missing artifact (both footage_critique + compile empty) ⇒ fail ───
async def test_missing_artifact_raises_precondition() -> None:
    prior = {"demo_video_compile": {}, "demo_video_footage_critique": {}}
    with pytest.raises(StagePreconditionError):
        await _stage(AsyncMock(), AsyncMock()).handle(FinalCritiqueInput(), _ctx(prior))


# ── 2. Soft-gate invariant: analyzer / WM failures ⇒ accept, never raise ──────
@pytest.mark.parametrize("failure", ["analyzer"])
async def test_soft_gate_never_raises(failure: str) -> None:
    # Out-of-window loudness forces the not-ship-ready path so the WM branch runs.
    analyzer = _analyzer()
    wm = AsyncMock()
    if failure == "analyzer":
        analyzer.start_analysis.side_effect = RuntimeError("analyzer down")
    elif failure == "wm_raise":
        wm.message.side_effect = RuntimeError("worker down")
    else:  # wm_malformed — a Mapping missing every expected field
        wm.message.return_value = _resp({"unexpected": "shape"})

    out = await _stage(wm, analyzer).handle(
        FinalCritiqueInput(), _ctx(_prior(loudness={"integrated_lufs": -6.0}))
    )
    # Accepted the current artifact, no raise, no re-drive.
    assert out.artifact_url == "http://v/orig.mp4"
    assert out.duration_seconds == 10.0
    assert out.rerun_stages == []
    if failure == "analyzer":
        assert any(f.dimension == "analyzer" for f in out.findings)
    elif failure == "wm_raise":
        assert any(f.dimension == "film_critic" for f in out.findings)
    elif failure == "wm_malformed":
        # A reply carrying NOTHING usable persists NO scores — it must never be laundered
        # into a fabricated 3/3/3/3 + overall 3.0 that reads as a genuine average verdict.
        assert out.scores is None
        assert any(
            f.dimension == "film_critic" and "no usable score" in f.detail for f in out.findings
        )  # ... and says so explicitly
        assert any(
            f.dimension == "loudness" for f in out.findings
        )  # -6.0 LUFS out-of-window advisory


# ── 2b. The fail-open is VISIBLE: analyzer failure ⇒ ships + `review_skipped` ──
@pytest.mark.parametrize("where", ["poll_failed"], ids=["job-failed"])
async def test_an_analyzer_failure_ships_the_film_but_degrades_the_run(where: str) -> None:
    """Live run 01a16423-349f: a rollout orphaned the analyzer job, the poll came back
    failed/404, and the film shipped `insufficient_evidence` with only an ERROR log —
    the run was NOT degraded. The fail-open stays (the film ships, no resubmit); the
    record now says it shipped UNREVIEWED."""
    from src.services.stages.shared.degradation import REVIEW_SKIPPED, is_degraded
    from src.services.stages.upstream import PermanentUpstreamError

    analyzer = _analyzer()
    if where == "poll_failed":
        analyzer.poll_analysis.side_effect = PermanentUpstreamError(
            "video_analyzer job abc failed: worker lost the job"
        )
    elif where == "poll_404":
        analyzer.poll_analysis.side_effect = RuntimeError("video_analyzer job abc: 404 not found")
    else:
        analyzer.start_analysis.side_effect = RuntimeError("analyzer down")

    out = await _stage(AsyncMock(), analyzer).handle(FinalCritiqueInput(), _ctx(_prior()))

    assert out.artifact_url == "http://v/orig.mp4"  # the film still ships
    assert out.ship_gate["decision"] == "insufficient_evidence"
    assert [d["code"] for d in out.degradations] == [REVIEW_SKIPPED]
    detail = out.degradations[0]["detail"]
    assert "unreviewed" in detail
    assert ("404" in detail) if where == "poll_404" else True
    assert is_degraded(out.model_dump(mode="json"))  # ⇒ the run's degraded flag
    assert analyzer.start_analysis.await_count <= 1, "never an orchestrator-side resubmit"


# ── 4. WM short-circuit: ship_ready ⇒ ZERO worker calls ───────────────────────
async def test_ship_ready_short_circuits_the_worker() -> None:
    # A MEASURABLE film (two sampled shots, coherent, in-window -14 LUFS) ⇒ ship-ready.
    analyzer = _analyzer()
    wm = AsyncMock()
    out = await _stage(wm, analyzer).handle(
        FinalCritiqueInput(), _ctx(_prior(loudness={"integrated_lufs": -14.0}))
    )
    wm.message.assert_not_called()  # zero WM calls on a ship-ready gestalt
    assert out.quality_badge == "ship-ready"
    assert out.scores is None
    assert out.gestalt_summary["ship_ready"] is True
    assert out.gestalt_summary["insufficient_evidence"] is False
    # "Measurable" must MEAN something: a real dispersed cohesion (not the degenerate 1.0)
    # and a non-degenerate continuity distribution. Without these the assertion above
    # would be satisfied by a wall of fabricated constants.
    cohesion = out.gestalt_summary["cohesion"]
    assert isinstance(cohesion, float) and 0.0 < cohesion < 1.0
    stats = cast("dict[str, Any]", out.gestalt_summary["continuity_stats"])
    assert "degenerate" not in stats and stats["n"] == 1
    assert out.rerun_stages == []


# ── 4b. visual_timeline is what turns a motion-degenerate film MEASURABLE ──────
async def test_visual_timeline_makes_a_motion_degenerate_film_measurable() -> None:
    # The core of the analyzer-PR-#73 wiring. Motion is UNIFORM (0.0 at both cuts) so it
    # contributes ZERO dispersion — before per-frame visual features were consumed, five
    # of six components were whole-video constants and the sixth (motion) was flat, so the
    # gate correctly reported `insufficient_evidence`. With a per-frame `visual_timeline`
    # whose shots differ in brightness/sharpness/colour, cohesion + continuity now vary and
    # the film becomes genuinely measurable — WITHOUT any motion signal.
    uniform_motion = [{"t": t, "motion": 0.0} for t in _TWO_CUTS]  # zero motion dispersion
    visual_timeline = [
        {
            "t": 2.0,
            "index": 0,
            "sharpness": 1200.0,
            "brightness": 0.80,
            "dominant_rgb": [240, 30, 20],
        },
        {
            "t": 6.0,
            "index": 1,
            "sharpness": 150.0,
            "brightness": 0.20,
            "dominant_rgb": [10, 40, 230],
        },
    ]
    analyzer = _analyzer(motion=uniform_motion, visual_timeline=visual_timeline)
    wm = AsyncMock()
    out = await _stage(wm, analyzer).handle(
        FinalCritiqueInput(), _ctx(_prior(loudness={"integrated_lufs": -14.0}))
    )
    # Measurable off the VISUAL curve alone: real dispersed cohesion, non-degenerate
    # continuity distribution, and the honest-absence guard NOT tripped.
    assert out.gestalt_summary["insufficient_evidence"] is False
    cohesion = out.gestalt_summary["cohesion"]
    assert isinstance(cohesion, float) and 0.0 < cohesion < 1.0
    stats = cast("dict[str, Any]", out.gestalt_summary["continuity_stats"])
    assert "degenerate" not in stats and stats["n"] == 1


# ── 4c. Contrast: SAME uniform motion, NO visual_timeline ⇒ still unmeasurable ─
async def test_uniform_motion_without_visual_timeline_stays_insufficient() -> None:
    # The exact contrast to the test above: identical uniform motion but an EMPTY
    # visual_timeline ⇒ the consumer falls back to whole-video means (constant per shot),
    # so the film is degenerate and the gate honestly reports insufficient_evidence. This
    # is the honest-absence contract (#621) the back-compat fallback must preserve.
    uniform_motion = [{"t": t, "motion": 0.0} for t in _TWO_CUTS]
    analyzer = _analyzer(motion=uniform_motion, visual_timeline=None)
    out = await _stage(AsyncMock(), analyzer).handle(
        FinalCritiqueInput(), _ctx(_prior(loudness={"integrated_lufs": -14.0}))
    )
    assert out.gestalt_summary["insufficient_evidence"] is True
    assert out.gestalt_summary["cohesion"] is None
    assert out.gestalt_summary["ship_ready"] is False


# ── 5. Directive clamp: hallucinated stage + unknown spine ids dropped ────────
async def test_directive_clamp_drops_unknown_stage_and_spine_ids() -> None:
    analyzer = _analyzer()
    wm = AsyncMock()
    wm.message.return_value = _resp(
        {
            "cohesion": 3,
            "continuity": 3,
            "momentum_experienced": 3,
            "tonal_consistency": 3,
            "overall": 3.0,
            "directives": [
                # hallucinated stage → dropped
                {"stage": "not_a_real_stage", "spine_ids": ["clip-1"], "reason": "x"},
                # only-unknown spine ids → dropped
                {"stage": "demo_video_narrate", "spine_ids": ["ghost-1"], "reason": "y"},
                # valid stage, mixed ids → kept with the ghost clamped out
                {"stage": "demo_video_narrate", "spine_ids": ["clip-1", "ghost-2"], "reason": "z"},
            ],
        }
    )
    out = await _stage(wm, analyzer).handle(
        FinalCritiqueInput(), _ctx(_prior(loudness={"integrated_lufs": -6.0}))
    )
    assert out.rerun_stages == []  # re-drive is deferred regardless
    assert len(out.directives) == 1
    assert out.directives[0].stage == "demo_video_narrate"
    assert out.directives[0].spine_ids == ["clip-1"]


# ── 6. critique_only persists scores advisory-only, never re-drives ───────────
async def test_critique_only_persists_scores_and_never_redrives() -> None:
    analyzer = _analyzer()
    wm = AsyncMock()
    wm.message.return_value = _scored(
        3.5,
        directives=[_REDRIVE_DIRECTIVE],  # a redrivable directive is IGNORED in critique_only
        findings=[{"dimension": "continuity", "severity": "minor"}],
    )
    ctx = _ctx(
        _prior(loudness={"integrated_lufs": -6.0}),
        config={"ship_gate": {"mode": "critique_only"}},
    )
    stack, m = _patch_redrive(
        compile_handle=AsyncMock(return_value=_compiled("http://v/redriven.mp4"))
    )
    try:
        out = await _stage(wm, analyzer).handle(FinalCritiqueInput(), ctx)
    finally:
        stack.close()
    assert out.scores is not None
    assert out.scores.overall == 3.5
    assert out.findings  # WM finding + the deterministic out-of-window loudness one
    assert out.rerun_stages == []  # byte-identical: critique_only never re-drives
    _assert_no_redrive_stage_touched(m)  # NO director / narrate / compile in critique_only


# ── 7. Loudness window: -14 admits ship-ready; out-of-window ⇒ advisory finding ─
async def test_loudness_in_window_admits_ship_ready() -> None:
    out = await _stage(AsyncMock(), _analyzer()).handle(
        FinalCritiqueInput(), _ctx(_prior(loudness={"integrated_lufs": -14.0}))
    )
    assert out.gestalt_summary["ship_ready"] is True


async def test_loudness_out_of_window_emits_advisory_finding() -> None:
    analyzer = _analyzer()
    wm = AsyncMock()
    wm.message.return_value = _resp({"overall": 3.0})
    out = await _stage(wm, analyzer).handle(
        FinalCritiqueInput(), _ctx(_prior(loudness={"integrated_lufs": -6.0}))
    )
    wm.message.assert_called_once()  # not ship-ready ⇒ the critic was consulted
    assert any(f.dimension == "loudness" and f.severity == "advisory" for f in out.findings)


# ── REGRESSION (item 2): the frames spine uses the CONTAINMENT join, not the zip ──
# `_build_frames` used to call `sample_frames_at_cuts(scene_boundaries, duration,
# spine_ids)` with NO `beat_intervals` — the legacy POSITIONAL zip. With more cuts than
# beats the zip truncates to `min(len)`, so the whole tail of the film is never sampled,
# and every surviving pair gets the WRONG spine id. Here: 2 beats, 4 cuts.
#   positional zip ⇒ 2 frames  [(clip-1, 1.0), (clip-2, 3.0)]  ⇒ continuity_stats.n == 1
#   containment    ⇒ 4 frames  [(clip-1,1), (clip-1,3), (clip-2,6), (clip-2,8)] ⇒ n == 3
# The whole cut spine is now built ONCE and shared, so the frames and momentum evidence
# can never be measured against two different sets of cuts.
async def test_frames_spine_uses_containment_join_not_the_positional_zip() -> None:
    analyzer = _analyzer(
        scene_boundaries=[1.0, 3.0, 6.0, 8.0],
        frame_analyses=[{"timestamp": t, "description": "x"} for t in (1.0, 3.0, 6.0, 8.0)],
        motion=[{"t": t, "motion": 0.5} for t in (1.0, 3.0, 6.0, 8.0)],
    )
    wm = AsyncMock()
    wm.message.return_value = _scored(3.0)  # out-of-window loudness ⇒ critic consulted
    prior = _prior(
        loudness={"integrated_lufs": -6.0},
        line_overrides={"clip-1": {"seconds": 5.0}, "clip-2": {"seconds": 5.0}},
    )
    with patch(_SPY_FRAMES, wraps=sample_frames_at_cuts) as spy:
        out = await _stage(wm, analyzer).handle(FinalCritiqueInput(), _ctx(prior))

    # ONE join for the whole stage, and it carries the containment intervals.
    assert len(spy.call_args_list) == 1
    call = spy.call_args_list[0]
    assert call.kwargs["beat_intervals"] == [("clip-1", 0.0, 5.0), ("clip-2", 5.0, 10.0)]
    assert sample_frames_at_cuts(*call.args, **call.kwargs) == [
        ("clip-1", 1.0),
        ("clip-1", 3.0),
        ("clip-2", 6.0),
        ("clip-2", 8.0),
    ]
    # Behavioral proof: all FOUR cuts reached the gestalt frames (3 adjacent pairs).
    # The positional zip would have truncated to 2 frames ⇒ exactly 1 pair.
    stats = cast("dict[str, Any]", out.gestalt_summary["continuity_stats"])
    assert stats["n"] == 3


# ── REGRESSION (item 2): the frames spine is filtered by the STAGE duration ───────
# `_build_frames` used `as_float(result.duration_seconds)` — the ANALYZER's own idea of
# the file length — while the momentum site used the stage's `duration` (read off the
# accepted artifact). The two diverge whenever the analyzer measures a different length,
# and the cut spine is `t > duration`-filtered, so the two sites silently sampled
# different films. There is now one spine, filtered by the stage duration.
async def test_cut_spine_is_filtered_by_the_stage_duration_not_the_analyzer_duration() -> None:
    # Analyzer claims 100s; the accepted artifact (compile output) is 10s. The 12.0 cut
    # is past the REAL end of the film and must be dropped.
    analyzer = _analyzer(scene_boundaries=[1.0, 3.0, 12.0], duration=100.0)
    wm = AsyncMock()
    wm.message.return_value = _scored(3.0)
    with patch(_SPY_FRAMES, wraps=sample_frames_at_cuts) as spy:
        await _stage(wm, analyzer).handle(
            FinalCritiqueInput(), _ctx(_prior(loudness={"integrated_lufs": -6.0}))
        )
    call = spy.call_args_list[0]
    assert call.args[1] == 10.0  # the STAGE duration, not the analyzer's 100.0
    assert sample_frames_at_cuts(*call.args, **call.kwargs) == [("clip-1", 1.0), ("clip-2", 3.0)]


# ── the discarded "accept" field: deleted, not silently ignored ──────────────
def test_output_schema_does_not_request_an_unread_accept_field() -> None:
    """``_OUTPUT_SCHEMA`` used to request an ``"accept": {"type": "boolean"}``
    field that nothing downstream ever read — the ship/no-ship decision is
    derived from ``overall`` + the advisory findings (``_accept``/
    ``ship_ready``), and this stage is never allowed to gate on a new signal
    (its own system prompt says "you never block the ship"). Asking the model
    for an opinion and then silently discarding it is worse than not asking:
    it reads as though the critic's verdict matters. Pin that the schema no
    longer requests it."""
    from src.services.stages.demo_video.final_critique import _OUTPUT_SCHEMA

    properties = cast("dict[str, object]", _OUTPUT_SCHEMA["properties"])
    assert "accept" not in properties


# ── A11 Workstream B: _to_intervals reshapes the compiler's authoritative timeline ─


# ── A11 Workstream B: the momentum join PREFERS the compiler's rendered timeline ─
_SPY_FRAMES = "src.services.stages.demo_video.final_critique.sample_frames_at_cuts"


# ── inert pass-through: artifact rides through unchanged (no re-render) ────────


# ── PR-3 bounded re-drive (mode="auto"): re-run the owning per-dimension LLM stage ─
def _auto_ctx(**prior_kwargs: Any) -> RunContext:
    return _ctx(
        _prior(loudness={"integrated_lufs": -6.0}, **prior_kwargs),
        config={"ship_gate": {"mode": "auto"}},
    )


def _footage_out() -> FootageDirectionOutput:
    # An IMPROVED plan for the flagged clip-1 (hard cut + static reframe).
    return FootageDirectionOutput(
        plans=[ShotPlan(line_id="clip-1", zoom="auto", transition_signal="cut", reframe="static")]
    )


_SOUND_BASELINE = {
    "cues": [
        {"line_id": "clip-1", "sfx_id": "old", "at_seconds": 1.0, "gain_db": -10.0},
        {"line_id": "clip-2", "sfx_id": "keep", "at_seconds": 1.0, "gain_db": -10.0},
    ]
}
_VOICE_BASELINE = {
    "directives": [
        {"line_id": "clip-1", "rate_hint": 1.0},
        {"line_id": "clip-2", "rate_hint": 1.0},
    ]
}
_NARRATE_BASELINE = {
    "clips": [{"id": "n1", "line_id": "clip-1", "artifact_url": "http://a/n1.wav", "seconds": 2.0}]
}


def _sound_out() -> SoundDirectorOutput:
    return SoundDirectorOutput(cues=[LineSfxCue(line_id="clip-1", sfx_id="new", at_seconds=2.0)])


def _voice_out() -> VoiceDirectorOutput:
    return VoiceDirectorOutput(
        directives=[PerformanceDirective(line_id="clip-1", rate_hint=1.1, delivery_note="punchier")]
    )


def _narrate_out() -> DemoVideoNarrateOutput:
    return DemoVideoNarrateOutput(
        clips=[
            DemoVideoAudioClip(
                id="rn1", line_id="clip-1", artifact_url="http://a/rn1.wav", seconds=2.5
            )
        ]
    )


# ── 1. footage re-drive re-runs the stage + merges ONLY the flagged ids ────────
async def test_auto_footage_redrive_reruns_stage_and_merges_only_flagged() -> None:
    # Pass 1 scores 3.0 (< 4 ⇒ re-drive); re-running footage_direction + one recompile
    # re-scores 4.5 (converge) and is kept.
    analyzer = _analyzer()
    wm = AsyncMock()
    finding = {
        "dimension": "continuity",
        "severity": "minor",
        "seam": {"from_id": "clip-1", "to_id": "clip-2"},
        "detail": "hard whip-pan into a static frame",
    }
    wm.message.side_effect = [
        _scored(3.0, directives=[_REDRIVE_DIRECTIVE], findings=[finding]),
        _scored(4.5, directives=[_REDRIVE_DIRECTIVE]),
    ]
    compile_handle = AsyncMock(return_value=_compiled("http://v/redriven.mp4"))
    footage_handle = AsyncMock(return_value=_footage_out())
    stack, m = _patch_redrive(compile_handle=compile_handle, footage_handle=footage_handle)
    try:
        out = await _stage(wm, analyzer).handle(FinalCritiqueInput(), _auto_ctx())
    finally:
        stack.close()
    # The footage director was re-run with the critic feedback threaded in.
    footage_handle.assert_called_once()
    fin = footage_handle.call_args.args[0]
    # TIGHTENED (was `!= ""`, which passed on the directive `reason` alone even when every
    # finding bullet was literally "- [continuity/minor] "). Assert the BULLET is real.
    assert fin.redrive_feedback is not None
    assert "weak seam" in fin.redrive_feedback  # the directive reason
    assert (
        "- [continuity/minor] clip-1->clip-2: hard whip-pan into a static frame"
        in fin.redrive_feedback
    )
    compile_handle.assert_called_once()
    # No other owning stage / narrate touched.
    m.sound.return_value.handle.assert_not_called()
    m.voice.return_value.handle.assert_not_called()
    m.narrate.return_value.handle.assert_not_called()
    # Only clip-1 (flagged) changed in the merged plan; clip-2 rode through verbatim.
    merged = compile_handle.call_args.args[1].prior_run_outputs["demo_video_footage_direction"][
        "plans"
    ]
    by_id = {p["line_id"]: p for p in merged}
    assert by_id["clip-1"]["transition_signal"] == "cut"
    assert by_id["clip-2"] == {"line_id": "clip-2", "zoom": None, "reframe": None}
    assert out.artifact_url == "http://v/redriven.mp4"
    assert out.rerun_stages == ["demo_video_footage_direction"]
    assert out.scores is not None and out.scores.overall == 4.5


# ── 2. sound re-drive (analogous, on the cues list) ───────────────────────────
async def test_auto_sound_redrive_reruns_stage_and_merges_only_flagged_cues() -> None:
    analyzer = _analyzer()
    wm = AsyncMock()
    wm.message.side_effect = [
        _scored(3.0, directives=[_SOUND_DIRECTIVE]),
        _scored(4.5, directives=[_SOUND_DIRECTIVE]),
    ]
    compile_handle = AsyncMock(return_value=_compiled("http://v/redriven.mp4"))
    sound_handle = AsyncMock(return_value=_sound_out())
    stack, m = _patch_redrive(compile_handle=compile_handle, sound_handle=sound_handle)
    try:
        out = await _stage(wm, analyzer).handle(
            FinalCritiqueInput(), _auto_ctx(sound=dict(_SOUND_BASELINE))
        )
    finally:
        stack.close()
    sound_handle.assert_called_once()
    assert sound_handle.call_args.args[0].redrive_feedback is not None
    compile_handle.assert_called_once()
    m.footage.return_value.handle.assert_not_called()
    m.voice.return_value.handle.assert_not_called()
    m.narrate.return_value.handle.assert_not_called()
    merged = compile_handle.call_args.args[1].prior_run_outputs["demo_video_sound_director"]["cues"]
    by_id = {c["line_id"]: c for c in merged}
    assert by_id["clip-1"]["sfx_id"] == "new"  # flagged cue regenerated
    assert by_id["clip-2"]["sfx_id"] == "keep"  # unflagged cue verbatim
    assert out.rerun_stages == ["demo_video_sound_director"]


# ── 3. voice re-drive ALSO re-runs narrate (compile reads the fresh clips) ─────
async def test_auto_voice_redrive_reruns_voice_then_narrate() -> None:
    analyzer = _analyzer()
    wm = AsyncMock()
    wm.message.side_effect = [
        _scored(3.0, directives=[_VOICE_DIRECTIVE]),
        _scored(4.5, directives=[_VOICE_DIRECTIVE]),
    ]
    compile_handle = AsyncMock(return_value=_compiled("http://v/redriven.mp4"))
    voice_handle = AsyncMock(return_value=_voice_out())
    narrate_handle = AsyncMock(return_value=_narrate_out())
    stack, m = _patch_redrive(
        compile_handle=compile_handle, voice_handle=voice_handle, narrate_handle=narrate_handle
    )
    try:
        out = await _stage(wm, analyzer).handle(
            FinalCritiqueInput(),
            _auto_ctx(voice=dict(_VOICE_BASELINE), narrate=dict(_NARRATE_BASELINE)),
        )
    finally:
        stack.close()
    voice_handle.assert_called_once()
    narrate_handle.assert_called_once()  # voice re-drive re-synthesizes narration
    compile_handle.assert_called_once()
    m.footage.return_value.handle.assert_not_called()
    m.sound.return_value.handle.assert_not_called()
    # compile saw the re-narrated clips.
    reclips = compile_handle.call_args.args[1].prior_run_outputs["demo_video_narrate"]["clips"]
    assert reclips[0]["artifact_url"] == "http://a/rn1.wav"
    assert "voice_director" in out.rerun_stages


# ── 4. a directive naming a stage that never ran is dropped (advisory only) ────
# A stage that opted out is DROPPED whether its key is absent OR present-but-falsy:
# a StageSkippedError persists a falsy ``{}`` into prior_run_outputs
# (dag_executor.py:137), which is the realistic gate-off signal in a live run — not a
# missing key. Both must drop (never construct the stage, never count in rerun_stages).
@pytest.mark.parametrize("baseline", [None], ids=["key-absent"])
async def test_auto_drops_directive_when_owning_stage_never_ran(
    baseline: dict[str, Any] | None,
) -> None:
    analyzer = _analyzer()
    wm = AsyncMock()
    wm.message.return_value = _scored(3.0, directives=[_SOUND_DIRECTIVE])
    compile_handle = AsyncMock(return_value=_compiled("http://v/redriven.mp4"))
    stack, m = _patch_redrive(compile_handle=compile_handle)
    try:
        out = await _stage(wm, analyzer).handle(FinalCritiqueInput(), _auto_ctx(sound=baseline))
    finally:
        stack.close()
    m.sound.return_value.handle.assert_not_called()  # dropped — never constructed
    assert out.rerun_stages == []  # advisory only
    assert out.artifact_url == "http://v/orig.mp4"


# ── 5. the pass-budget cap holds even when the critic never clears the bar ─────
async def test_auto_never_exceeds_two_compile_passes() -> None:
    # Monotonic sub-threshold scores (3.0 → 3.5, both < 4.0) never converge, so the
    # loop terminates on the PASS BUDGET, not a plateau. loop_engine clamps max_passes
    # to [1, 2] and pass 1 is the initial score, so at most ONE recompile ever fires —
    # this proves the [1, 2] cap holds (compile is never called 2+ times).
    analyzer = _analyzer()
    wm = AsyncMock()
    wm.message.side_effect = [
        _scored(3.0, directives=[_REDRIVE_DIRECTIVE]),
        _scored(3.5, directives=[_REDRIVE_DIRECTIVE]),
    ]
    compile_handle = AsyncMock(return_value=_compiled("http://v/redriven.mp4"))
    footage_handle = AsyncMock(return_value=_footage_out())
    stack, _m = _patch_redrive(compile_handle=compile_handle, footage_handle=footage_handle)
    try:
        out = await _stage(wm, analyzer).handle(FinalCritiqueInput(), _auto_ctx())
    finally:
        stack.close()
    assert compile_handle.call_count == 1  # re-drive fired; [1,2] cap holds (never 2+)
    assert (
        out.scores is not None and out.scores.overall == 3.5
    )  # keep-highest kept the improved draft
    assert out.rerun_stages == ["demo_video_footage_direction"]


# ── 6. soft-degrade: a director / narrate / compile raise never fails the run ──
@pytest.mark.parametrize("raiser", ["director"])
async def test_auto_soft_degrades_when_a_redrive_stage_raises(raiser: str) -> None:
    analyzer = _analyzer()
    wm = AsyncMock()
    wm.message.return_value = _scored(3.0, directives=[_REDRIVE_DIRECTIVE])
    if raiser == "director":
        footage_handle = AsyncMock(side_effect=RuntimeError("director boom"))
        compile_handle = AsyncMock(return_value=_compiled("http://v/redriven.mp4"))
    else:  # compile raises AFTER the director succeeds
        footage_handle = AsyncMock(return_value=_footage_out())
        compile_handle = AsyncMock(side_effect=RuntimeError("compile boom"))
    stack, _m = _patch_redrive(compile_handle=compile_handle, footage_handle=footage_handle)
    try:
        out = await _stage(wm, analyzer).handle(FinalCritiqueInput(), _auto_ctx())
    finally:
        stack.close()
    assert out.artifact_url == "http://v/orig.mp4"  # kept the current best, no raise
    assert any(f.severity == "advisory" for f in out.findings)
    # TIGHTENED (item 8): the test never asserted rerun_stages. When the director raises,
    # `optimizer_fn` swallows it and returns the UNCHANGED draft, and when compile raises
    # nothing is published either — in both cases the ORIGINAL artifact ships, so NO stage
    # was re-run for the shipped draft. The old derivation (retained draft's directives ∩
    # stages with a baseline) reported ['demo_video_footage_direction'] with compile_called=0.
    assert out.rerun_stages == []
    # The swallowed failure is RECORDED on the output, not only logged: without it this
    # row was byte-identical to a re-drive that ran and simply did not improve.
    redrive = [f for f in out.findings if f.dimension == "redrive"]
    assert len(redrive) == 1
    assert f"{raiser} boom" in redrive[0].detail
    assert redrive[0].severity == "advisory"


async def test_auto_loop_error_is_recorded_on_the_accepted_output() -> None:
    # The loop itself raising (not one optimizer step) falls back to accepting the
    # initial draft. That fall-back used to leave no trace on the output either.
    analyzer = _analyzer()
    wm = AsyncMock()
    wm.message.return_value = _scored(3.0, directives=[_REDRIVE_DIRECTIVE])
    stack, _m = _patch_redrive()
    stack.enter_context(
        patch(
            "src.services.stages.demo_video.final_critique.run_evaluator_optimizer",
            AsyncMock(side_effect=RuntimeError("loop boom")),
        )
    )
    try:
        out = await _stage(wm, analyzer).handle(FinalCritiqueInput(), _auto_ctx())
    finally:
        stack.close()
    assert out.artifact_url == "http://v/orig.mp4"
    assert out.rerun_stages == []
    redrive = [f for f in out.findings if f.dimension == "redrive"]
    assert len(redrive) == 1 and "loop boom" in redrive[0].detail


# ── 7. monotonic: a worse pass ships the earlier artifact ─────────────────────
async def test_auto_is_monotonic_a_worse_pass_ships_the_earlier_artifact() -> None:
    analyzer = _analyzer()
    wm = AsyncMock()
    wm.message.side_effect = [
        _scored(3.5, directives=[_REDRIVE_DIRECTIVE]),
        _scored(3.0, directives=[_REDRIVE_DIRECTIVE]),
    ]
    compile_handle = AsyncMock(return_value=_compiled("http://v/redriven.mp4"))
    footage_handle = AsyncMock(return_value=_footage_out())
    stack, _m = _patch_redrive(compile_handle=compile_handle, footage_handle=footage_handle)
    try:
        out = await _stage(wm, analyzer).handle(FinalCritiqueInput(), _auto_ctx())
    finally:
        stack.close()
    assert compile_handle.call_count == 1
    assert out.artifact_url == "http://v/orig.mp4"  # pass-1's artifact, not the worse pass-2
    assert out.scores is not None and out.scores.overall == 3.5


# ── 8. no cycle: a directive naming a non-redrivable/cyclic stage drives nothing ─
async def test_auto_never_redrives_the_cyclic_footage_critique_stage() -> None:
    analyzer = _analyzer()
    wm = AsyncMock()
    wm.message.return_value = _scored(
        3.0,
        directives=[
            {"stage": "demo_video_footage_critique", "spine_ids": ["clip-1"], "reason": "cycle"}
        ],
    )
    stack, m = _patch_redrive(compile_handle=AsyncMock(return_value=_compiled("http://v/x.mp4")))
    try:
        out = await _stage(wm, analyzer).handle(FinalCritiqueInput(), _auto_ctx())
    finally:
        stack.close()
    # No re-compile, no director, no narrate — the cyclic directive drives nothing.
    _assert_no_redrive_stage_touched(m)
    assert "demo_video_footage_critique" not in out.rerun_stages
    assert out.rerun_stages == []


# ── A11 PR-4: per-shot pin / continuity_exempt author overrides ────────────────
# Author-set flags win over the film-critic's COMPUTED findings/directives in the
# holistic ship gate. `pin` ⇒ never a re-drive target; `continuity_exempt` ⇒ drop
# the continuity finding touching the beat. Absent ⇒ byte-identical (test 4).


async def test_pin_drops_redrive_directive_for_that_beat() -> None:
    # A footage directive naming a pinned beat is emptied by the pin-clamp ⇒ dropped;
    # nothing is re-driven and the owning director is never constructed.
    analyzer = _analyzer()
    wm = AsyncMock()
    wm.message.return_value = _scored(3.0, directives=[_REDRIVE_DIRECTIVE])  # flags clip-1
    footage_handle = AsyncMock(return_value=_footage_out())
    compile_handle = AsyncMock(return_value=_compiled("http://v/redriven.mp4"))
    stack, _m = _patch_redrive(compile_handle=compile_handle, footage_handle=footage_handle)
    try:
        out = await _stage(wm, analyzer).handle(
            FinalCritiqueInput(), _auto_ctx(line_overrides={"clip-1": {"pin": True}})
        )
    finally:
        stack.close()
    assert out.directives == []  # directive emptied by the pin ⇒ dropped
    assert out.rerun_stages == []  # nothing re-driven
    footage_handle.assert_not_called()  # a pinned beat is never a re-drive target
    compile_handle.assert_not_called()
    assert out.artifact_url == "http://v/orig.mp4"


async def test_continuity_exempt_suppresses_only_the_continuity_finding() -> None:
    # A `continuity_exempt` beat carries BOTH a continuity and a cohesion finding on
    # the SAME seam; only the continuity finding is suppressed, cohesion survives.
    analyzer = _analyzer()
    wm = AsyncMock()
    seam = {"from_id": "clip-1", "to_id": "clip-2"}
    wm.message.return_value = _scored(
        3.5,
        findings=[
            {"dimension": "continuity", "severity": "minor", "seam": seam},
            {"dimension": "cohesion", "severity": "minor", "seam": seam},
        ],
    )
    ctx = _ctx(
        _prior(
            loudness={"integrated_lufs": -6.0},
            line_overrides={"clip-1": {"continuity_exempt": True}},
        ),
        config={"ship_gate": {"mode": "critique_only"}},
    )
    out = await _stage(wm, analyzer).handle(FinalCritiqueInput(), ctx)
    dims = [f.dimension for f in out.findings]
    assert "continuity" not in dims  # exempt beat's continuity finding suppressed
    assert "cohesion" in dims  # a non-continuity finding on the SAME beat survives


async def test_override_wins_over_computed_finding_and_directive() -> None:
    # The author override wins over the critic's COMPUTED output: a beat marked both
    # `pin` and `continuity_exempt` sees its critic continuity finding AND its footage
    # directive both vanish — no re-drive, no re-compile.
    analyzer = _analyzer()
    wm = AsyncMock()
    finding = {
        "dimension": "continuity",
        "severity": "minor",
        "seam": {"from_id": "clip-1", "to_id": "clip-2"},
    }
    wm.message.return_value = _scored(3.0, directives=[_REDRIVE_DIRECTIVE], findings=[finding])
    footage_handle = AsyncMock(return_value=_footage_out())
    compile_handle = AsyncMock(return_value=_compiled("http://v/redriven.mp4"))
    stack, _m = _patch_redrive(compile_handle=compile_handle, footage_handle=footage_handle)
    try:
        out = await _stage(wm, analyzer).handle(
            FinalCritiqueInput(),
            _auto_ctx(line_overrides={"clip-1": {"pin": True, "continuity_exempt": True}}),
        )
    finally:
        stack.close()
    assert all(f.dimension != "continuity" for f in out.findings)  # exempt ⇒ suppressed
    assert out.directives == []  # pinned ⇒ directive dropped
    assert out.rerun_stages == []
    footage_handle.assert_not_called()
    compile_handle.assert_not_called()


async def test_absent_overrides_are_byte_identical_to_pr3_baseline() -> None:
    # With NO overrides, the critic's continuity finding + footage directive survive
    # exactly as PR-3: nothing suppressed, directive clamped-but-kept, no re-drive.
    analyzer = _analyzer()
    wm = AsyncMock()
    finding = {
        "dimension": "continuity",
        "severity": "minor",
        "seam": {"from_id": "clip-1", "to_id": "clip-2"},
    }
    wm.message.return_value = _scored(3.5, directives=[_REDRIVE_DIRECTIVE], findings=[finding])
    ctx = _ctx(
        _prior(loudness={"integrated_lufs": -6.0}),  # NO line_overrides
        config={"ship_gate": {"mode": "critique_only"}},
    )
    out = await _stage(wm, analyzer).handle(FinalCritiqueInput(), ctx)
    assert any(f.dimension == "continuity" for f in out.findings)  # not suppressed
    assert len(out.directives) == 1
    assert out.directives[0].stage == "demo_video_footage_direction"
    assert out.directives[0].spine_ids == ["clip-1"]  # not pin-clamped
    assert out.rerun_stages == []  # critique_only never re-drives


async def test_off_skips_clean_and_never_recompiles() -> None:
    # Byte-identical opt-out: off raises skip before any evaluation or re-drive.
    stack, m = _patch_redrive(compile_handle=AsyncMock(return_value=_compiled("http://v/x.mp4")))
    try:
        with pytest.raises(StageSkippedError):
            await _stage(AsyncMock(), AsyncMock()).handle(
                FinalCritiqueInput(), _ctx(_prior(), config={"ship_gate": {"mode": "off"}})
            )
    finally:
        stack.close()
    _assert_no_redrive_stage_touched(m)


# ── A11 ENFORCE: opt-in HARD gate (mode=auto only). Default OFF ⇒ accept-best ──
def _enforce_auto_ctx(*, enforce: bool, **prior_kwargs: Any) -> RunContext:
    """An auto-mode ctx with ``ship_gate.enforce`` set. Default loudness is OUT of window
    (-6.0) so the re-driven best draft stays not-ship-ready (ceilings MISS) unless a test
    overrides it — that is the input shape the enforce gate must act on."""
    prior_kwargs.setdefault("loudness", {"integrated_lufs": -6.0})
    return _ctx(
        _prior(**prior_kwargs),
        config={"ship_gate": {"mode": "auto", "enforce": enforce}},
    )


async def test_enforce_false_ceilings_miss_after_redrive_still_accepts_best() -> None:
    # REGRESSION LOCK: with enforce off (today's default), a best draft that STILL fails the
    # ceilings after the ≤2-pass re-drive is ACCEPTED, byte-identical to accept-best.
    analyzer = _analyzer()
    wm = AsyncMock()
    wm.message.side_effect = [
        _scored(3.0, directives=[_REDRIVE_DIRECTIVE]),
        _scored(4.5, directives=[_REDRIVE_DIRECTIVE]),
    ]
    compile_handle = AsyncMock(return_value=_compiled("http://v/redriven.mp4"))  # lufs -6.0
    footage_handle = AsyncMock(return_value=_footage_out())
    stack, _m = _patch_redrive(compile_handle=compile_handle, footage_handle=footage_handle)
    try:
        out = await _stage(wm, analyzer).handle(
            FinalCritiqueInput(), _enforce_auto_ctx(enforce=False)
        )
    finally:
        stack.close()
    # The best draft shipped even though it never cleared the ceilings.
    assert out.artifact_url == "http://v/redriven.mp4"
    assert out.gestalt_summary["ship_ready"] is False  # ceilings MISS …
    assert out.rerun_stages == ["demo_video_footage_direction"]  # … but accepted


async def test_enforce_true_ceilings_met_accepts_without_failing() -> None:
    # enforce on + ceilings MET (in-window loudness ⇒ ship-ready initial) ⇒ accept, no raise,
    # and the ship-ready short-circuit means ZERO WM calls (nothing to enforce against).
    analyzer = _analyzer()
    wm = AsyncMock()
    out = await _stage(wm, analyzer).handle(
        FinalCritiqueInput(),
        _enforce_auto_ctx(enforce=True, loudness={"integrated_lufs": -14.0}),
    )
    wm.message.assert_not_called()
    assert out.gestalt_summary["ship_ready"] is True
    assert out.quality_badge == "ship-ready"


async def test_enforce_true_ceilings_miss_after_redrive_fails_the_run() -> None:
    # The whole point: enforce on, best draft STILL fails the ceilings after the bounded
    # re-drive (out-of-window loudness on every recompile) ⇒ TERMINAL hard-fail naming the
    # ceiling. Reuses the #719 mechanism (RuntimeError → DAG persists it → run `failed`).
    analyzer = _analyzer()
    wm = AsyncMock()
    wm.message.side_effect = [
        _scored(3.0, directives=[_REDRIVE_DIRECTIVE]),
        _scored(4.5, directives=[_REDRIVE_DIRECTIVE]),
    ]
    compile_handle = AsyncMock(return_value=_compiled("http://v/redriven.mp4"))  # lufs -6.0
    footage_handle = AsyncMock(return_value=_footage_out())
    stack, _m = _patch_redrive(compile_handle=compile_handle, footage_handle=footage_handle)
    try:
        with pytest.raises(ShipGateEnforcementError) as exc:
            await _stage(wm, analyzer).handle(FinalCritiqueInput(), _enforce_auto_ctx(enforce=True))
    finally:
        stack.close()
    # It rode the #719 terminal mechanism, and the reason names the failed ceiling.
    assert isinstance(exc.value, RuntimeError)
    assert "loudness" in str(exc.value)
    # The re-drive was actually attempted (both passes) before the gate fired.
    assert wm.message.call_count == 2
    footage_handle.assert_called_once()


async def test_critique_only_now_blocks_on_an_armed_failing_term(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ⭐ THE DEFECT, INVERTED. This test previously asserted the opposite — that
    # `mode=critique_only` "does not block" even with `enforce: True` — and that pairing
    # WAS the whole defect: production seeds `{"mode": "critique_only"}`, the only
    # hard-fail site was reachable only from the `auto` re-drive, and `enforce` was never
    # seeded. Both halves false at once ⇒ `ship_ready` could not fail a run in production,
    # ever. `mode` is now the REPAIR axis only; arming is what blocks.
    #
    # An out-of-window master (-6.0 LUFS) in critique_only, loudness ARMED ⇒ the run FAILS,
    # and it does so WITHOUT attempting a re-drive (critique_only still repairs nothing).
    _armed(monkeypatch, loudness=True)
    analyzer = _analyzer()
    wm = AsyncMock()
    wm.message.return_value = _scored(3.5, findings=[{"dimension": "continuity", "severity": "x"}])
    ctx = _ctx(
        _prior(loudness={"integrated_lufs": -6.0}),  # measured OUT of the ship window
        config={"ship_gate": {"mode": "critique_only"}},  # NOTE: no `enforce` key at all
    )
    stack, m = _patch_redrive(
        compile_handle=AsyncMock(return_value=_compiled("http://v/redriven.mp4"))
    )
    try:
        with pytest.raises(ShipGateEnforcementError) as exc:
            await _stage(wm, analyzer).handle(FinalCritiqueInput(), ctx)
    finally:
        stack.close()
    # DIAGNOSABLE: the message names the term, its tier, the measured value and the limit.
    msg = str(exc.value)
    assert "loudness" in msg
    assert "-6.0 LUFS" in msg
    # DERIVED, not restated: the message must quote the band the gate actually used.
    # This pinned the literal "[-14.0, -11.0]" until 2026-09-16, when that band was
    # recentred on the target the compiler applies (see shared/loudness_contract.py) —
    # a literal here would have failed for the right reason but told the reader the
    # wrong story. The assertion's teeth are unchanged: -6.0 LUFS is far outside.
    from src.services.stages.shared.loudness_contract import ship_loudness_window

    gate_lo, gate_hi = ship_loudness_window()
    assert f"[{gate_lo}, {gate_hi}]" in msg
    assert "tunable" in msg
    # It rides the #719 terminal mechanism (RuntimeError → DAG persists → run `failed`).
    assert isinstance(exc.value, RuntimeError)
    # critique_only still repairs nothing: no director / narrate / compile was constructed.
    _assert_no_redrive_stage_touched(m)


async def test_disarming_the_kill_switch_backs_the_block_out_without_a_code_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ⭐ THE MUTATION THAT MATTERS. Byte-identical input to the test above, one env var
    # flipped — the exact flip a GitOps configmap would make. The run must now be ACCEPTED
    # advisory-only. If this and the test above ever agree, the kill switch is not wired.
    _armed(monkeypatch, loudness=False)
    analyzer = _analyzer()
    wm = AsyncMock()
    wm.message.return_value = _scored(3.5, findings=[{"dimension": "continuity", "severity": "x"}])
    ctx = _ctx(
        _prior(loudness={"integrated_lufs": -6.0}),
        config={"ship_gate": {"mode": "critique_only"}},
    )
    stack, m = _patch_redrive(
        compile_handle=AsyncMock(return_value=_compiled("http://v/redriven.mp4"))
    )
    try:
        out = await _stage(wm, analyzer).handle(FinalCritiqueInput(), ctx)
    finally:
        stack.close()
    assert out.scores is not None and out.scores.overall == 3.5  # accepted, no raise
    assert out.rerun_stages == []
    _assert_no_redrive_stage_touched(m)
    # The verdict is still MEASURED and still recorded — disarming removes the BLOCK, not
    # the diagnosis. A disarmed failing term is exactly what a calibration reader needs.
    loud = _term_row(out.ship_gate, "loudness")
    assert loud["state"] == "fail"
    assert loud["armed"] is False
    assert out.ship_gate["decision"] != "fail"
    assert out.ship_gate["failed_terms"] == []


# ── S810: a critique that RAISES must not read as a pass ──────────────────────
# `handle`'s outer catch-all used to convert ANY internal defect into an accept — so a
# gate that never reached a verdict was byte-identical to a gate that passed the film,
# including for a run that explicitly opted into the ENFORCE hard gate.
async def test_internal_defect_under_enforce_fails_the_run_instead_of_passing() -> None:
    analyzer = _analyzer()
    wm = AsyncMock()
    stage = _stage(wm, analyzer)
    # An INTERNAL defect (not an analyzer/WM outage — those have their own soft branches):
    # the evaluation itself blows up before any verdict exists.
    defect = AsyncMock(side_effect=RuntimeError("gestalt exploded"))
    with (
        patch.object(stage, "_reevaluate", defect),
        pytest.raises(ShipGateEnforcementError) as exc,
    ):
        await stage.handle(FinalCritiqueInput(), _enforce_auto_ctx(enforce=True))
    assert isinstance(exc.value, RuntimeError)  # rides the #719 terminal mechanism
    assert "could not be evaluated" in str(exc.value)
    assert "gestalt exploded" in str(exc.value)  # the swallowed cause is named, not lost
    assert exc.value.__cause__ is not None  # chained, so the traceback survives


async def test_internal_defect_without_enforce_still_soft_accepts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # DEGRADE SEMANTICS PRESERVED: with enforce off (the default) the soft gate is
    # byte-identical to before — the current artifact ships with a "stage" advisory …
    analyzer = _analyzer()
    wm = AsyncMock()
    stage = _stage(wm, analyzer)
    defect = AsyncMock(side_effect=RuntimeError("gestalt exploded"))
    with patch.object(stage, "_reevaluate", defect), caplog.at_level("ERROR"):
        out = await stage.handle(FinalCritiqueInput(), _enforce_auto_ctx(enforce=False))
    assert out.artifact_url == "http://v/orig.mp4"
    assert any(f.dimension == "stage" for f in out.findings)
    assert out.quality_badge == "needs-review"
    # … but it is no longer SILENT: ERROR level, carrying the traceback.
    records = [r for r in caplog.records if r.levelname == "ERROR" and "verdict" in r.message]
    assert records, "a swallowed gate defect must be logged at ERROR"
    assert records[0].exc_info is not None


async def test_absent_artifact_sentinel_string_is_not_a_deliverable() -> None:
    # A stringified-missing artifact_url ("None") is TRUTHY, so the old `if not url` chain
    # accepted it and the gate "evaluated" a film that does not exist — the shape that lets
    # a run reach terminal status with no encoded master.
    prior = {
        "demo_video_compile": {"artifact_url": "None", "duration_seconds": 10.0},
        "demo_video_footage_critique": {"artifact_url": "  "},
    }
    with pytest.raises(StagePreconditionError):
        await _stage(AsyncMock(), AsyncMock()).handle(FinalCritiqueInput(), _ctx(prior))


# ── A11 PR-5: assembled momentum-gap lever (ADVISORY evidence, not a gate) ─────
# UNIFORM motion at every cut. Because momentum_gap normalizes each curve by its first
# common beat, a uniform motion level CANCELS the (0.7 + 0.3*m) factor — so the gap reads
# only the residual clock drift (~0.05), regardless of the (uniform) level. Motion moves
# the gap only when it is NON-uniform across cuts (see _SAG_MOTION).
# ⚠️ UNIFORM motion ⇒ every sampled shot is IDENTICAL in feature space ⇒ the VISUAL terms
# come back `insufficient_evidence`. That is fine and intended for the tests below, which
# assert on MOMENTUM only (uniform non-zero motion is still a real momentum measurement —
# only total absence of motion is unmeasurable). Never reuse `_HIGH_MOTION` in a test that
# asserts ship_ready / cohesion / continuity: use `_MILD_VARIATION`.
_HIGH_MOTION = [{"t": 6.0, "motion": 0.9}, {"t": 9.0, "motion": 0.9}]
# NON-uniform: high motion at the first cut, collapsing to zero at the second — a genuine
# late-run momentum SAG that does NOT cancel under normalization, so it reads a materially
# larger gap (~0.31) than the uniform curve above.
_SAG_MOTION = [{"t": 6.0, "motion": 0.9}, {"t": 9.0, "motion": 0.0}]
# Motion that VARIES (so the sampled shots are not identical ⇒ the visual terms are
# measurable at all) but by LESS than the 0.5 default continuity ceiling, so the variation
# is not itself a continuity break. Needed wherever a test wants a ship-READY film:
# `_HIGH_MOTION` is uniform ⇒ zero dispersion ⇒ insufficient_evidence, and `_SAG_MOTION`
# varies by 0.9 ⇒ a continuity break. This threads between them.
_MILD_VARIATION = [{"t": 6.0, "motion": 0.9}, {"t": 9.0, "motion": 0.6}]


async def test_momentum_gap_is_present_in_gestalt_summary() -> None:
    # The measured momentum_gap rides into the gestalt summary on every evaluation.
    analyzer = _analyzer(scene_boundaries=[6.0, 9.0], motion=_HIGH_MOTION)
    out = await _stage(AsyncMock(), analyzer).handle(
        FinalCritiqueInput(), _ctx(_prior(loudness={"integrated_lufs": -14.0}))
    )
    assert "momentum_gap" in out.gestalt_summary
    assert isinstance(out.gestalt_summary["momentum_gap"], float)


async def test_momentum_gap_is_advisory_does_not_change_ship_ready() -> None:
    # A late-run motion SAG produces a materially positive momentum_gap (~0.31). The continuity
    # ceiling is relaxed so the motion delta between the two frames is NOT itself a continuity
    # break, and the cohesion floor is 0.0, so the film is otherwise ship-ready (in-window
    # loudness). With NO momentum_gap_max ceiling configured momentum stays advisory-only, so it
    # STILL ships (ship_ready True) and short-circuits the WM critic — the gap only rides as
    # evidence. (The opt-in hard gate is exercised by test_momentum_gap_over_ceiling_blocks_ship.)
    analyzer = _analyzer(scene_boundaries=[6.0, 9.0], motion=_SAG_MOTION)
    wm = AsyncMock()
    config = {
        "ship_gate": {
            "mode": "critique_only",
            "ceilings": {"continuity_l2": 5.0, "cohesion_floor": 0.0},
        }
    }
    out = await _stage(wm, analyzer).handle(
        FinalCritiqueInput(), _ctx(_prior(loudness={"integrated_lufs": -14.0}), config=config)
    )
    momentum_gap = cast("float", out.gestalt_summary["momentum_gap"])
    assert momentum_gap > 0.15  # a real sag was measured (~0.31) …
    assert out.gestalt_summary["ship_ready"] is True  # … but unconfigured momentum does NOT gate
    wm.message.assert_not_called()  # ship-ready gestalt ⇒ zero WM calls


async def test_momentum_gap_over_ceiling_blocks_ship() -> None:
    # SAME sag film as above (measured gap ~0.31), but now a momentum_gap_max=0.15 ceiling is
    # configured in ship_gate.ceilings. The gap blows the ceiling, so ship_ready flips to False
    # — proving the stage threads momentum_gap_max into gestalt_summary (the opt-in hard gate).
    analyzer = _analyzer(scene_boundaries=[6.0, 9.0], motion=_SAG_MOTION)
    config = {
        "ship_gate": {
            "mode": "critique_only",
            "ceilings": {"continuity_l2": 5.0, "cohesion_floor": 0.0, "momentum_gap_max": 0.15},
        }
    }
    # ⭐ THE MOMENTUM SEAM, GOING LIVE. `momentum` is ARMED by default and that is safe by
    # construction: with no `momentum_gap_max` configured — the live state, nothing
    # anywhere sets it — `momentum_verdict` is `unconfigured`, which is insufficient
    # evidence and can never fail. Configure the ceiling, as this test does, and the armed
    # term FAILS the run. So arming it today costs nothing and needs no code change later.
    # This test was asserting `ship_ready is False` (a flag nobody acted on); it now
    # asserts the block, which is what "blocks ship" was always supposed to mean.
    with pytest.raises(ShipGateEnforcementError) as exc:
        await _stage(AsyncMock(), analyzer).handle(
            FinalCritiqueInput(), _ctx(_prior(loudness={"integrated_lufs": -14.0}), config=config)
        )
    msg = str(exc.value)
    assert "momentum" in msg
    assert "0.150" in msg  # the CONFIGURED ceiling is named …
    assert "momentum_gap 0.3" in msg  # … alongside the measured gap that blew it


async def test_cold_open_clock_alignment_keeps_gap_small() -> None:
    # Regression for the curve time-base fix: the analyzer's scene_boundaries are
    # ABSOLUTE timestamps that include a ~6s cold-open lead-in ([6.0, 9.0]), while the
    # authored beats (seconds=3.0 each) drive the predicted watch curve from offset 0.
    # After first-cut-relative normalization both curves share offset 0, so a HEALTHY
    # film (good motion) yields a SMALL gap. Without the `t - t0` normalization the raw
    # absolute clock would decay the assembled curve ~0.12 below predicted at beat 0 — a
    # pure clock-skew artifact — so this test bites if the normalization is reverted.
    analyzer = _analyzer(scene_boundaries=[6.0, 9.0], motion=_HIGH_MOTION)
    prior = _prior(
        loudness={"integrated_lufs": -14.0},
        line_overrides={"clip-1": {"seconds": 3.0}, "clip-2": {"seconds": 3.0}},
    )
    out = await _stage(AsyncMock(), analyzer).handle(FinalCritiqueInput(), _ctx(prior))
    momentum_gap = cast("float", out.gestalt_summary["momentum_gap"])
    assert momentum_gap < 0.08  # small after clock-alignment (~0.03); ~0.12 raw


async def test_momentum_value_never_moves_ship_ready() -> None:
    # Inert-gate proof isolating momentum from the (motion-driven) continuity signal: run the
    # SAME film with the SAME uniform motion — so the measured momentum_gap is IDENTICAL — and
    # toggle ONLY the loudness in/out of the window. ship_ready flips True→False on loudness
    # alone while momentum_gap is unchanged, proving that WITH NO momentum_gap_max configured the
    # gate is loudness/continuity/cohesion-driven and momentum stays advisory. (Motion is held
    # uniform so it does not perturb the continuity signal, which shares the per-frame motion feature.)
    # NOTE: motion must VARY across the cuts. Uniform motion makes every sampled shot
    # identical in feature space (motion is the only per-shot component the analyzer wire
    # can vary — see _MEASURABLE_MOTION), which the gate reports as insufficient_evidence,
    # so neither run would be ship-ready and the comparison would be vacuous. `_SAG_MOTION`
    # is IDENTICAL across both runs, which is all this test actually needs.
    in_window = _analyzer(scene_boundaries=[6.0, 9.0], motion=_MILD_VARIATION)
    out_window = _analyzer(scene_boundaries=[6.0, 9.0], motion=_MILD_VARIATION)
    good = await _stage(AsyncMock(), in_window).handle(
        FinalCritiqueInput(), _ctx(_prior(loudness={"integrated_lufs": -14.0}))
    )
    bad = await _stage(AsyncMock(), out_window).handle(
        FinalCritiqueInput(), _ctx(_prior(loudness={"integrated_lufs": -6.0}))
    )
    # momentum_gap is identical across the two runs (same motion) …
    assert good.gestalt_summary["momentum_gap"] == bad.gestalt_summary["momentum_gap"]
    # … yet ship_ready flips purely on loudness (momentum never gates)
    assert good.gestalt_summary["ship_ready"] is True
    assert bad.gestalt_summary["ship_ready"] is False


# ── _clamp_dim hardening: inf/nan model scores are ABSENT, never crash ─────────


# ── _emit_gestalt_metrics: INERT ship-gate telemetry (never touches the gate) ──
_METRICS_PATCH = "src.services.stages.demo_video.final_critique.metrics"


def _full_gestalt() -> dict[str, Any]:
    """A fully-populated gestalt verdict as gestalt_summary returns it."""
    return {
        "cohesion": 0.82,
        "continuity_breaks": [],
        "continuity_stats": {"n": 4, "min": 0.1, "max": 0.9, "mean": 0.4},
        "tonal": {"variance": 0.05, "worst_cut": None},
        "loudness": {"integrated_lufs": -14.0, "true_peak_dbtp": -1.0},
        "momentum_gap": 0.2,
        "momentum_ok": False,
        "ship_ready": False,
    }


# ── STRICT-vs-PERMISSIVE float coercion, pinned on the REAL gate paths ────────
# `final_critique` deliberately uses the STRICT family (`src.utils.as_float` /
# `opt_float`), never the PERMISSIVE `coerce_float`. `tests/test_utils.py` pins the
# helpers in isolation; these pin the CONSEQUENCE at the call sites, so swapping in a
# permissive coercer fails HERE too. The four divergent input shapes are "1.5"
# (numeric string), True (bool), float("nan") and Decimal("2").


# ══ REGRESSION (item 3): degenerate evidence is HONEST, and never "ship-ready" ══
# The whole live corpus of this stage was two rows reading `cohesion: 1.0`,
# `tonal.variance: 0.0`, `momentum_gap: 0.0`, `continuity_stats: {"n": 0}` — four
# structurally-dead terms — because the cut spine had collapsed to one shot. The gate's
# response was `ship_ready = loud_ok and momentum_ok` (both VACUOUSLY true), so a blank
# film was the most confident ship-ready verdict it could emit.
async def test_insufficient_evidence_is_not_shippable_and_persists_nulls() -> None:
    analyzer = _analyzer(scene_boundaries=None)  # zero sampled shots
    wm = AsyncMock()
    out = await _stage(wm, analyzer).handle(
        FinalCritiqueInput(), _ctx(_prior(loudness={"integrated_lufs": -14.0}))
    )
    g = out.gestalt_summary
    assert g["insufficient_evidence"] is True
    assert g["ship_ready"] is False  # was True — in-window loudness alone is not evidence
    assert g["cohesion"] is None  # was the degenerate-perfect 1.0
    assert g["momentum_gap"] is None  # was 0.0
    assert cast("dict[str, Any]", g["tonal"])["variance"] is None  # was 0.0
    assert cast("dict[str, Any]", g["continuity_stats"])["n"] == 0
    # The badge degrades — it must NOT read "ship-ready".
    assert out.quality_badge == "needs-review"
    assert out.scores is None
    assert any(
        f.dimension == "gestalt" and "insufficient evidence" in f.detail for f in out.findings
    )
    # Nothing to critique ⇒ no WM call is made on a wall of nulls.
    wm.message.assert_not_called()


async def test_unobservable_terms_are_never_observed_into_prometheus() -> None:
    # An unobservable term must produce NO histogram sample, so the calibration corpus
    # stops being a corpus of silent zeros/ones that nothing can be calibrated from.
    with patch(_METRICS_PATCH) as m:
        await _stage(AsyncMock(), _analyzer(scene_boundaries=None)).handle(
            FinalCritiqueInput(), _ctx(_prior(loudness={"integrated_lufs": -14.0}))
        )
    m.gestalt_cohesion.observe.assert_not_called()
    m.gestalt_momentum_gap.observe.assert_not_called()
    m.gestalt_continuity_max_delta.observe.assert_not_called()
    # The MEASURED term (loudness) is still observed, and the gate counter still ticks —
    # silence is per-term, not a blanket opt-out.
    m.gestalt_loudness_lufs.observe.assert_called_once_with(-14.0)
    m.gestalt_ship_gate_total.labels.assert_called_once_with(ship_ready="false", momentum_ok="true")


# ══ REGRESSION (item 9): a missing dimension persists null, never a neutral 3 ══
async def test_missing_dimensions_persist_null_not_a_fabricated_neutral() -> None:
    # MEASURED in prod: the WM critic returned no dimension scores, so all four persisted
    # as `3` next to `overall: 1.0` — fabricated numbers that read as a real verdict.
    analyzer = _analyzer()
    wm = AsyncMock()
    wm.message.return_value = _resp({"overall": 1.0})  # overall only, no dimensions
    out = await _stage(wm, analyzer).handle(
        FinalCritiqueInput(), _ctx(_prior(loudness={"integrated_lufs": -6.0}))
    )
    assert out.scores is not None
    assert out.scores.overall == 1.0  # the REAL number the critic gave is preserved
    assert out.scores.cohesion is None  # ... and the ones it did not give are absent
    assert out.scores.continuity is None
    assert out.scores.momentum_experienced is None
    assert out.scores.tonal_consistency is None
    # It also says so, rather than leaving a silent gap.
    advisory = [f for f in out.findings if f.dimension == "film_critic"]
    assert advisory and "no usable score" in advisory[0].detail
    assert "cohesion" in advisory[0].detail and "tonal_consistency" in advisory[0].detail
    # The badge is driven by the genuine overall (1.0 ⇒ weak), not by a fabricated mean.
    assert out.quality_badge == "weak"


async def test_partial_dimensions_average_only_what_was_actually_scored() -> None:
    # A reply with two real dimensions and no overall falls back to the mean of THOSE
    # two — the two absent ones do not silently drag the mean toward a fabricated 3.
    analyzer = _analyzer()
    wm = AsyncMock()
    wm.message.return_value = _resp({"cohesion": 5, "continuity": 5})
    out = await _stage(wm, analyzer).handle(
        FinalCritiqueInput(), _ctx(_prior(loudness={"integrated_lufs": -6.0}))
    )
    assert out.scores is not None
    assert out.scores.overall == 5.0  # mean(5, 5); with neutral-3 padding it was 4.0
    assert out.scores.momentum_experienced is None
    assert out.scores.tonal_consistency is None


# ══ REGRESSION (item 10): finding `detail` exists in the schema and is populated ══
async def test_finding_detail_is_actually_demanded_of_the_worker() -> None:
    # BEHAVIOURAL (the previous version asserted the literal the same commit wrote, which
    # proves nothing): drive the real stage and inspect the schema + prompt the WORKER was
    # actually handed. A `detail` declared on a module constant that never reaches the
    # structured-output call would leave every finding detail-less exactly as before.
    analyzer = _analyzer()
    wm = AsyncMock()
    wm.message.return_value = _scored(3.0)
    await _stage(wm, analyzer).handle(
        FinalCritiqueInput(), _ctx(_prior(loudness={"integrated_lufs": -6.0}))
    )
    wm.message.assert_called_once()
    schema = wm.message.call_args.kwargs["output_schema"]
    items = schema["properties"]["findings"]["items"]
    assert items["properties"]["detail"] == {"type": "string"}
    assert "detail" in items["required"]  # the model cannot omit it
    # ... and the system prompt tells the model why it matters.
    assert "detail" in wm.message.call_args.kwargs["system_prompt"]


async def test_finding_detail_is_persisted_from_the_model_reply() -> None:
    analyzer = _analyzer()
    wm = AsyncMock()
    wm.message.return_value = _scored(
        3.5,
        findings=[
            {
                "dimension": "continuity",
                "severity": "minor",
                "seam": {"from_id": "clip-1", "to_id": "clip-2"},
                "detail": "the camera jumps 180 degrees across the cut",
            }
        ],
    )
    out = await _stage(wm, analyzer).handle(
        FinalCritiqueInput(), _ctx(_prior(loudness={"integrated_lufs": -6.0}))
    )
    continuity = [f for f in out.findings if f.dimension == "continuity"]
    assert continuity and continuity[0].detail == "the camera jumps 180 degrees across the cut"


# ══ REGRESSION (item 8): rerun_stages reports WORK DONE, not directives ══


async def test_rerun_stages_empty_when_the_named_stage_never_ran() -> None:
    # The critic names `demo_video_sound_director`, which has NO baseline output (the
    # stage never ran), so `_redrive` drops it — the recompile is a bare no-op re-render.
    # NOTHING was re-driven, and the report must say so.
    analyzer = _analyzer()
    wm = AsyncMock()
    wm.message.return_value = _scored(3.0, directives=[_SOUND_DIRECTIVE])
    compile_handle = AsyncMock(return_value=_compiled("http://v/redriven.mp4"))
    sound_handle = AsyncMock(return_value=_sound_out())
    stack, _m = _patch_redrive(compile_handle=compile_handle, sound_handle=sound_handle)
    try:
        out = await _stage(wm, analyzer).handle(FinalCritiqueInput(), _auto_ctx())  # no sound
    finally:
        stack.close()
    sound_handle.assert_not_called()  # dropped: it never ran, so it cannot be re-run
    assert out.rerun_stages == []


async def test_rerun_stages_does_not_name_what_should_be_fixed_next() -> None:
    # Pass 2's directives name what the critic wants fixed NEXT. `rerun_stages` is a
    # report of work DONE, so a pass-2 directive naming a DIFFERENT stage must not leak
    # into it — only the footage director was ever executed.
    analyzer = _analyzer()
    wm = AsyncMock()
    wm.message.side_effect = [
        _scored(3.0, directives=[_REDRIVE_DIRECTIVE]),  # pass 1: re-drive footage
        _scored(4.5, directives=[_VOICE_DIRECTIVE]),  # pass 2 verdict: "now fix voice"
    ]
    compile_handle = AsyncMock(return_value=_compiled("http://v/redriven.mp4"))
    footage_handle = AsyncMock(return_value=_footage_out())
    voice_handle = AsyncMock(return_value=_voice_out())
    stack, _m = _patch_redrive(
        compile_handle=compile_handle, footage_handle=footage_handle, voice_handle=voice_handle
    )
    try:
        out = await _stage(wm, analyzer).handle(
            FinalCritiqueInput(), _auto_ctx(voice=dict(_VOICE_BASELINE))
        )
    finally:
        stack.close()
    voice_handle.assert_not_called()  # converged at 4.5 ⇒ pass 2 never re-drove voice
    assert out.rerun_stages == ["demo_video_footage_direction"]
    assert "voice_director" not in out.rerun_stages


# ══ REGRESSION (item 1): the cut spine comes from the compiler's shot_timeline ══
# `video_analyzer.scene_boundaries` returned ONE boundary for a 29.8s multi-shot film, so
# `len(frames) < 2`. The compiler KNOWS where every shot starts — it placed them — so its
# authoritative `shot_timeline` is the spine and the analyzer's inferred boundaries are
# only the fallback.
#
# ⚠️ READ THIS BEFORE CLAIMING THE SPINE FIX AS A WIN. A correct spine raises the number
# of SAMPLED shots; it cannot create signal the analyzer never sent. On the real wire the
# per-frame records carry no brightness/sharpness/colour, so unless MOTION varies the
# extra shots are identical in feature space and every extra "sample" is a fabricated
# zero. These two tests pin both halves of that truth.
async def test_spine_fix_does_not_manufacture_evidence_on_featureless_frames() -> None:
    # THE prod shape: one detected cut, a good shot_timeline, and a static render (motion
    # flat). The spine correctly samples both shots — and the gate must still report that
    # it measured NOTHING, and must contribute NOTHING to the calibration corpus.
    tl: list[dict[str, Any]] = [
        {"line_id": "clip-1", "start": 0.0, "end": 4.0},
        {"line_id": "clip-2", "start": 4.0, "end": 10.0},
    ]
    analyzer = _degenerate_analyzer(scene_boundaries=[0.0])
    wm = AsyncMock()
    with patch(_METRICS_PATCH) as m:
        out = await _stage(wm, analyzer).handle(
            FinalCritiqueInput(),
            _ctx(_prior(loudness={"integrated_lufs": -6.0}, shot_timeline=tl)),
        )
    g = out.gestalt_summary
    # The spine DID sample both shots — the pair count is honest …
    stats = cast("dict[str, Any]", g["continuity_stats"])
    assert stats["n"] == 1
    # … but the distribution is marked degenerate and publishes NO percentile keys, so
    # nothing downstream can mistake a wall of zeros for a measured distribution.
    assert stats["degenerate"] is True
    assert "max" not in stats and "mean" not in stats and "stdev" not in stats
    assert g["cohesion"] is None  # NOT the degenerate-perfect 1.0
    assert cast("dict[str, Any]", g["tonal"])["variance"] is None
    assert cast("dict[str, Any]", g["tonal"])["worst_cut"] is None  # no invented seam
    assert g["momentum_gap"] is None
    assert g["continuity_ceiling_suggested"] is None
    assert g["insufficient_evidence"] is True
    assert g["ship_ready"] is False
    assert out.quality_badge == "needs-review"
    wm.message.assert_not_called()  # nothing to critique
    # THE corpus-poisoning assertion: zero fabricated samples.
    m.gestalt_cohesion.observe.assert_not_called()
    m.gestalt_momentum_gap.observe.assert_not_called()
    m.gestalt_continuity_max_delta.observe.assert_not_called()


async def test_spine_fix_delivers_real_evidence_when_the_signal_exists() -> None:
    # The other half: when the wire DOES carry discriminating signal (motion varying
    # across the shots), the timeline-derived spine is what lets the gate measure it —
    # the collapsed single analyzer boundary alone could not.
    tl: list[dict[str, Any]] = [
        {"line_id": "clip-1", "start": 0.0, "end": 4.0},
        {"line_id": "clip-2", "start": 4.0, "end": 10.0},
    ]
    analyzer = _analyzer(
        scene_boundaries=[0.0],
        motion=[{"t": 0.0, "motion": 0.8}, {"t": 4.0, "motion": 0.5}],
    )
    wm = AsyncMock()
    wm.message.return_value = _scored(3.0)
    with patch(_METRICS_PATCH) as m:
        out = await _stage(wm, analyzer).handle(
            FinalCritiqueInput(),
            _ctx(_prior(loudness={"integrated_lufs": -6.0}, shot_timeline=tl)),
        )
    g = out.gestalt_summary
    stats = cast("dict[str, Any]", g["continuity_stats"])
    assert stats["n"] == 1
    assert "degenerate" not in stats
    assert stats["max"] == pytest.approx(0.3)  # the REAL measured adjacent delta
    _cohesion = g["cohesion"]
    assert isinstance(_cohesion, float) and _cohesion < 1.0  # genuinely dispersed
    assert g["insufficient_evidence"] is False
    # Real measurements DO reach the corpus — the silence above is per-term, not blanket.
    m.gestalt_cohesion.observe.assert_called_once()
    m.gestalt_continuity_max_delta.observe.assert_called_once_with(pytest.approx(0.3))


async def test_cut_spine_falls_back_to_scene_boundaries_without_a_timeline() -> None:
    # BEHAVIOURAL (the previous version asserted a spy's positional argument, which a
    # refactor could satisfy while sampling something else entirely): with NO shot_timeline
    # the analyzer's boundaries must still DRIVE THE MEASURED EVIDENCE. Adding a boundary
    # adds a sampled shot, which adds an adjacent pair to the measured distribution.
    motion = [{"t": 1.0, "motion": 0.9}, {"t": 3.0, "motion": 0.6}, {"t": 7.0, "motion": 0.3}]
    # Authored beat durations so the containment join is available; without them the
    # fallback is the positional zip, which truncates to len(spine_ids) and would hide the
    # third cut for a reason that has nothing to do with the analyzer.
    seconds: dict[str, dict[str, Any]] = {
        "clip-1": {"seconds": 5.0},
        "clip-2": {"seconds": 5.0},
    }
    wm = AsyncMock()
    wm.message.return_value = _scored(3.0)
    two = await _stage(wm, _analyzer(scene_boundaries=[1.0, 3.0], motion=motion)).handle(
        FinalCritiqueInput(),
        _ctx(_prior(loudness={"integrated_lufs": -6.0}, line_overrides=seconds)),
    )
    wm2 = AsyncMock()
    wm2.message.return_value = _scored(3.0)
    three = await _stage(wm2, _analyzer(scene_boundaries=[1.0, 3.0, 7.0], motion=motion)).handle(
        FinalCritiqueInput(),
        _ctx(_prior(loudness={"integrated_lufs": -6.0}, line_overrides=seconds)),
    )
    two_stats = cast("dict[str, Any]", two.gestalt_summary["continuity_stats"])
    three_stats = cast("dict[str, Any]", three.gestalt_summary["continuity_stats"])
    assert two_stats["n"] == 1
    assert three_stats["n"] == 2
    # The extra analyzer boundary genuinely changed the measurement, not just a call arg.
    assert two.gestalt_summary["cohesion"] != three.gestalt_summary["cohesion"]


async def test_one_unparseable_beat_duration_costs_momentum_not_the_whole_gestalt() -> None:
    # `build_beat_intervals` now RAISES on an unparseable `seconds` (loud beats silent),
    # but that is a MOMENTUM-only input. With the compiler's rendered timeline available,
    # the continuity / cohesion / tonal evidence must survive intact — one malformed beat
    # duration may not blank the whole verdict.
    tl: list[dict[str, Any]] = [
        {"line_id": "clip-1", "start": 0.0, "end": 4.0},
        {"line_id": "clip-2", "start": 4.0, "end": 10.0},
    ]
    # Cuts come from the timeline (0.0 / 4.0); motion varies across them so the visual
    # terms are genuinely measurable and the assertions below are about the bad duration.
    analyzer = _analyzer(motion=[{"t": 0.0, "motion": 0.8}, {"t": 4.0, "motion": 0.2}])
    wm = AsyncMock()
    wm.message.return_value = _scored(3.0)
    prior = _prior(
        loudness={"integrated_lufs": -6.0},
        shot_timeline=tl,
        line_overrides={"clip-1": {"seconds": "soon"}, "clip-2": {"seconds": 5.0}},
    )
    out = await _stage(wm, analyzer).handle(FinalCritiqueInput(), _ctx(prior))
    g = out.gestalt_summary
    assert g["insufficient_evidence"] is False
    assert cast("dict[str, Any]", g["continuity_stats"])["n"] == 1
    assert g["cohesion"] is not None
    assert g["momentum_gap"] is None  # the ONE term the bad duration actually costs


# ══ REGRESSION: the per-shot frame join is DIRECTIONAL (at/after the cut) ══════


async def test_shot_evidence_comes_from_its_own_shot_not_the_previous_one() -> None:
    # Behavioural: two shots whose descriptions differ. With the undirected join, clip-2's
    # tone evidence was read from the frame BEFORE its cut, i.e. clip-1's description.
    tl: list[dict[str, Any]] = [
        {"line_id": "clip-1", "start": 0.0, "end": 6.0},
        {"line_id": "clip-2", "start": 6.2, "end": 10.0},
    ]
    frames = [
        {"timestamp": 0.0, "description": "welcome friendly enjoy"},
        {"timestamp": 3.0, "description": "welcome friendly enjoy"},
        {"timestamp": 6.0, "description": "welcome friendly enjoy"},  # still clip-1
        {"timestamp": 9.0, "description": "fast now launch speed"},  # clip-2's own frame
    ]
    analyzer = _analyzer(
        scene_boundaries=[0.0],
        frame_analyses=frames,
        motion=[{"t": 0.0, "motion": 0.8}, {"t": 6.2, "motion": 0.5}],
    )
    wm = AsyncMock()
    wm.message.return_value = _scored(3.0)
    out = await _stage(wm, analyzer).handle(
        FinalCritiqueInput(),
        _ctx(_prior(loudness={"integrated_lufs": -6.0}, shot_timeline=tl)),
    )
    tonal = cast("dict[str, Any]", out.gestalt_summary["tonal"])
    # clip-2 read its OWN energy-register frame ⇒ a real tonal jump was measured.
    # Reading clip-1's frame for clip-2 gives two identical warmth vectors ⇒ variance 0.0
    # (and, post-fix, `None`) — the register change would be invisible.
    assert tonal["variance"] is not None and tonal["variance"] > 0.0
    # warmth [0,0,3] vs energy [4,0,0] ⇒ L2 = 5.0
    assert tonal["worst_cut"] == {"from_id": "clip-1", "to_id": "clip-2", "jump": 5.0}


# ══ REGRESSION: the compiler's measured duplicate_frame_ratio is READ ══════════
async def test_static_render_is_flagged_from_the_compilers_own_measurement() -> None:
    # `duplicate_frame_ratio` is measured by the compiler on the real file and sat unread
    # in the compile output. A ~97.5%-duplicate render is degenerate INPUT — the reason the
    # visual terms come back unmeasured — and the gate now names it instead of leaving a
    # reader to guess why the evidence is thin.
    prior = _prior(loudness={"integrated_lufs": -6.0})
    prior["demo_video_compile"]["mastering"] = {"duplicate_frame_ratio": 0.9754}
    out = await _stage(AsyncMock(), _degenerate_analyzer()).handle(
        FinalCritiqueInput(), _ctx(prior)
    )
    assert out.gestalt_summary["static_frame_ratio"] == pytest.approx(0.9754)
    static = [f for f in out.findings if f.dimension == "static_input"]
    assert static and "97.5%" in static[0].detail
    assert static[0].severity == "advisory"  # never a gate term


async def test_a_moving_render_gets_no_static_finding() -> None:
    # Control: the finding is threshold-gated, not unconditional.
    prior = _prior(loudness={"integrated_lufs": -6.0})
    prior["demo_video_compile"]["mastering"] = {"duplicate_frame_ratio": 0.12}
    wm = AsyncMock()
    wm.message.return_value = _scored(3.0)
    out = await _stage(wm, _analyzer()).handle(FinalCritiqueInput(), _ctx(prior))
    assert out.gestalt_summary["static_frame_ratio"] == pytest.approx(0.12)
    assert not [f for f in out.findings if f.dimension == "static_input"]


# ══ REGRESSION: the compiler's measured av_gap_seconds is SURFACED ════════════
async def test_the_measured_av_gap_is_surfaced_on_the_persisted_gestalt_block() -> None:
    """The gap the compiler measured on the real file reaches the run's gestalt block.

    Drives the STAGE — the value is asserted off `out.gestalt_summary`, i.e. what a run
    actually persists — because the only thing under test is the SURFACING, and the
    reader (`_av_gap_seconds`) is already covered unit-wise in
    `tests/unit/test_av_gap_surfaced.py`. Deleting the `"av_gap_seconds": ...` entry in
    `_reevaluate` fails this and nothing else.

    EVIDENCE, never a gate term: an 8.8s gap rides alongside a `ship_ready` verdict that
    is computed without it, exactly like `static_frame_ratio`.
    """
    prior = _prior(loudness={"integrated_lufs": -6.0})
    prior["demo_video_compile"]["mastering"] = {
        "av_gap_seconds": 8.8,
        "duplicate_frame_ratio": 0.12,
    }
    wm = AsyncMock()
    wm.message.return_value = _scored(3.0)
    out = await _stage(wm, _analyzer()).handle(FinalCritiqueInput(), _ctx(prior))
    assert out.gestalt_summary["av_gap_seconds"] == pytest.approx(8.8)
    assert out.gestalt_summary["static_frame_ratio"] == pytest.approx(0.12)


async def test_a_master_whose_report_carries_no_gap_surfaces_none_not_a_zero() -> None:
    """⛔ 0.0 would read as "the picture fully covered the narration" — a QUALITY claim
    nobody measured. The key must be present and None, so a consumer can tell the
    difference between a clean master and an unmeasured one."""
    prior = _prior(loudness={"integrated_lufs": -6.0})
    prior["demo_video_compile"]["mastering"] = {"duplicate_frame_ratio": 0.12}
    wm = AsyncMock()
    wm.message.return_value = _scored(3.0)
    out = await _stage(wm, _analyzer()).handle(FinalCritiqueInput(), _ctx(prior))
    assert "av_gap_seconds" in out.gestalt_summary
    assert out.gestalt_summary["av_gap_seconds"] is None


# ══ REGRESSION: no measured motion ⇒ momentum is UNMEASURED, not a tidy 0.0 ════
async def test_zero_motion_everywhere_makes_momentum_unmeasured() -> None:
    # With no motion anywhere the assembled curve collapses to the same pure exponential
    # decay as the predicted one, so the diff reads a tidy ~0.0 — which is the analyzer
    # handing us nothing, not the film delivering its momentum. Observing that 0.0 nightly
    # drives the derived `momentum_gap_max` CEILING to 0.0.
    with patch(_METRICS_PATCH) as m:
        out = await _stage(AsyncMock(), _degenerate_analyzer()).handle(
            FinalCritiqueInput(),
            _ctx(
                _prior(
                    loudness={"integrated_lufs": -14.0},
                    line_overrides={"clip-1": {"seconds": 5.0}, "clip-2": {"seconds": 5.0}},
                )
            ),
        )
    assert out.gestalt_summary["momentum_gap"] is None
    m.gestalt_momentum_gap.observe.assert_not_called()


async def test_uniform_nonzero_motion_is_still_a_real_measurement() -> None:
    # Control: only TOTAL absence of motion is unmeasurable. Uniform non-zero motion is a
    # real (if uninformative) reading and must still be measured and observed — otherwise
    # the guard would silently delete a whole legitimate class of films from the corpus.
    out = await _stage(
        AsyncMock(), _analyzer(scene_boundaries=[6.0, 9.0], motion=_HIGH_MOTION)
    ).handle(
        FinalCritiqueInput(),
        _ctx(
            _prior(
                loudness={"integrated_lufs": -14.0},
                line_overrides={"clip-1": {"seconds": 3.0}, "clip-2": {"seconds": 3.0}},
            )
        ),
    )
    assert isinstance(out.gestalt_summary["momentum_gap"], float)


# ── gestalt_unavailable_reason: an empty gestalt must explain itself ──────────
# The live vacuous shape, copied from the 5 completed rows of 2026-08-28…31: a TRUTHY
# dict whose every calibration-bearing numeric term resolved to NULL.
_VACUOUS_GESTALT: dict[str, object] = {
    "cohesion": None,
    "momentum_gap": None,
    "continuity_stats": {"n": 0},
    "continuity_ceiling_suggested": None,
    "insufficient_evidence": True,
    "static_frame_ratio": 0.91,
    "ship_ready": False,
}
# A genuinely measured film: the control that keeps the guard from discarding good samples.
_MEASURED_GESTALT: dict[str, object] = {
    "cohesion": 0.82,
    "momentum_gap": 0.11,
    "continuity_stats": {"n": 9, "max": 0.41, "mean": 0.2, "stdev": 0.05},
    "insufficient_evidence": False,
    "ship_ready": True,
}


async def test_analyzer_failure_persists_a_readable_reason() -> None:
    """End-to-end through the real stage: the reason must reach the OUTPUT, not just a log.

    This is the assertion that would have made calibration render 019fcb32 diagnosable
    from its persisted row alone."""
    analyzer = _analyzer()
    analyzer.start_analysis.side_effect = RuntimeError("job 019fcb40 timed out after 180.0s")

    out = await _stage(AsyncMock(), analyzer).handle(
        FinalCritiqueInput(), _ctx(_prior(loudness={"integrated_lufs": -6.0}))
    )

    assert out.gestalt_summary == {}
    assert out.gestalt_unavailable_reason is not None
    assert "timed out after 180.0s" in out.gestalt_unavailable_reason


async def test_a_healthy_run_carries_no_reason() -> None:
    """Control: the field must stay None on the happy path, or it becomes noise that
    readers learn to ignore."""
    out = await _stage(AsyncMock(), _analyzer()).handle(FinalCritiqueInput(), _ctx(_prior()))
    assert out.gestalt_summary != {}
    assert out.gestalt_unavailable_reason is None


def test_analyzer_poll_timeout_is_sized_from_measurement_and_fits_the_stage() -> None:
    """The analyzer poll must fit its stage, and must spend all the room it has.

    The upper bound is the one that can hurt: a poll budget larger than the STAGE
    budget is worse than dead config — the outer wrapper kills the stage first, which
    turns a soft degrade into StageResumableTimeoutError and a re-dispatch (a second
    analyzer job, possibly a second full compile), and past `max_stage_attempts` the
    run persists `failed`. Same class as the nav-hook budget bug in
    bdd_playwright_service.

    ⚠️ The LOWER bound used to read `176.2 * 2 < poll`, where 176.2s was the analyzer's
    slowest observed job in the 7 days to 2026-08-04. That is a SAMPLE, and pinning a
    sample is how this constant went stale twice (180.0, then 420.0): re-measured
    2026-09-01 over 14 days the analyzer's completed max was 1384s, so the old floor
    was satisfied by a value that truncated 23 jobs. The floor is now the STAGE
    CONTRACT — the poll must claim essentially all the room the stage can give it,
    because every second left unclaimed is a calibration sample discarded while the
    server is still working. See the constant's own comment for why the analyzer's
    1800s deadline, the ideal key, is not reachable without re-budgeting the stage.
    """
    from src.services.blueprints.demo_video import DEMO_VIDEO_BLUEPRINT

    spec = next(s for s in DEMO_VIDEO_BLUEPRINT.stages if s.name == "final_critique")
    assert spec.default_timeout_seconds is not None
    assert spec.default_timeout_seconds == _STAGE_TIMEOUT_S, (
        "final_critique mirrors its StageSpec budget as _STAGE_TIMEOUT_S to derive the "
        "poll from it; the two have drifted apart"
    )

    # The stage also budgets an LLM ladder (~1320s worst case with its retry). The poll
    # plus that ladder must still fit, or the stage wrapper fires first.
    assert spec.default_timeout_seconds > _ANALYSIS_POLL_TIMEOUT_S + _LLM_LADDER_WORST_CASE_S, (
        "poll + LLM ladder must fit inside the stage timeout, or the poll is dead config"
    )

    # ...and it must claim everything the BINDING bound allows, or it is truncating
    # analyses for nothing.
    #
    # ⚠️ The binding bound is NOT the stage budget. It is the smaller of the stage's
    # spare room and the CLIENT CLAMP, and the clamp (600s) binds far below the stage
    # (1180s spare). Asserting against the stage alone is what produced 1100.0 — a
    # value `_clamp_timeout` RAISES on, which shipped green because every test in this
    # file mocks the analyzer client. The remaining 580s of stage slack is not
    # claimable from here at all; closing it is a w2v_common change.
    from w2v_common.clients.base import _MAX_TIMEOUT

    stage_spare = spec.default_timeout_seconds - _LLM_LADDER_WORST_CASE_S
    binding = min(stage_spare, _MAX_TIMEOUT)
    unclaimed = binding - _ANALYSIS_POLL_TIMEOUT_S
    assert unclaimed <= 100, (
        f"the poll claims {_ANALYSIS_POLL_TIMEOUT_S:.0f}s of the {binding:.0f}s the "
        f"binding constraint allows ({'client clamp' if binding == _MAX_TIMEOUT else 'stage budget'}), "
        f"leaving {unclaimed:.0f}s unclaimed; every unclaimed second is a gestalt "
        "sample discarded while video_analyzer is still working"
    )


# ── The auto re-drive's re-compile must NOT dedupe onto the original render ───
# `_redrive` drives DemoVideoCompileStage on a `replace(ctx, prior_run_outputs=...)`,
# which keeps run_id AND the stage name. Without a payload-derived discriminator the
# re-compile collides with the run's original compile: video_compile returns the
# EXISTING job, the paid director + narration re-drive is billed and thrown away, and
# `_reevaluate` reads the ORIGINAL output_sha256 out of `_result_cache` and concludes
# the re-drive did not help.
async def _redrive_compile_ctx(ctx: RunContext, footage_out: FootageDirectionOutput) -> RunContext:
    """Run one auto re-drive and return the ctx the re-compile was driven with."""
    analyzer = _analyzer()
    wm = AsyncMock()
    wm.message.side_effect = [
        _scored(3.0, directives=[_REDRIVE_DIRECTIVE]),
        _scored(4.5, directives=[_REDRIVE_DIRECTIVE]),
    ]
    compile_handle = AsyncMock(return_value=_compiled("http://v/redriven.mp4"))
    footage_handle = AsyncMock(return_value=footage_out)
    stack, _m = _patch_redrive(compile_handle=compile_handle, footage_handle=footage_handle)
    try:
        await _stage(wm, analyzer).handle(FinalCritiqueInput(), ctx)
    finally:
        stack.close()
    # Guard against a vacuous assertion: no re-compile ⇒ nothing to key.
    compile_handle.assert_called_once()
    return cast("RunContext", compile_handle.call_args.args[1])


_OTHER_FOOTAGE_OUT = FootageDirectionOutput(
    plans=[ShotPlan(line_id="clip-1", zoom="none", transition_signal="dissolve", reframe="static")]
)


# ── duration conformance overrides the ship-ready badge ──────────────────────
# Run 01a09172 (2026-09-11) delivered 68.075s against a 44.0s order — 155%, recorded
# by `duration_conformance` as `within_tolerance: false` — and `final_critique` badged
# it `quality_badge: "ship-ready"`, `ship_ready: true`, with ZERO duration findings.
# The ceilings verdict (continuity / loudness / momentum) has never known the ordered
# length, so a film of the wrong length passed a gate that cannot see length.
_ORDERED_SECONDS = 44.0
_DELIVERED_SECONDS = 68.075


def _duration_ctx(*, ordered: float | None, delivered: float) -> RunContext:
    """A MEASURABLE, otherwise-ship-ready film of ``delivered`` length against an order.

    In-window -14 LUFS + the default measurable analyzer ⇒ the ceilings pass and the
    gestalt short-circuits the WM critic, so the ONLY thing under test is the duration
    override. ``ordered=None`` omits the deliverable rung entirely (demo_video has no
    blueprint fallback) ⇒ no verdict to apply.
    """
    prior = _prior(loudness={"integrated_lufs": -14.0})
    prior["demo_video_compile"]["duration_seconds"] = delivered
    config: dict[str, Any] = {"ship_gate": {"mode": "critique_only"}}
    if ordered is not None:
        config["deliverable"] = {"target_duration_seconds": ordered}
    return _ctx(prior, config=config)


async def test_an_out_of_tolerance_master_is_not_ship_ready() -> None:
    """Run 01a09172's numbers: 155% of the order can never be badged ship-ready."""
    out = await _stage(AsyncMock(), _analyzer()).handle(
        FinalCritiqueInput(),
        _duration_ctx(ordered=_ORDERED_SECONDS, delivered=_DELIVERED_SECONDS),
    )

    assert out.quality_badge != "ship-ready"
    assert out.gestalt_summary["ship_ready"] is False
    # The ceilings verdict is preserved, not destroyed — it really did pass, and the
    # calibration corpus still needs to know that.
    assert out.gestalt_summary["ship_ready_ceilings"] is True
    # The artifact is passed through untouched: this withdraws a badge, never truncates.
    assert out.duration_seconds == pytest.approx(_DELIVERED_SECONDS)


async def test_an_out_of_tolerance_master_carries_a_duration_finding() -> None:
    """A withdrawn badge with no reason is the dead signal this replaces: the finding
    must name the ORDERED and the DELIVERED seconds, not just "too long"."""
    out = await _stage(AsyncMock(), _analyzer()).handle(
        FinalCritiqueInput(),
        _duration_ctx(ordered=_ORDERED_SECONDS, delivered=_DELIVERED_SECONDS),
    )

    findings = [f for f in out.findings if f.dimension == "duration_conformance"]
    assert len(findings) == 1, "the duration breach raised no finding of its own"
    detail = findings[0].detail
    assert "68.1s" in detail and "44.0s" in detail
    assert "155%" in detail
    assert findings[0].severity == "major"


async def test_an_in_tolerance_master_is_unchanged() -> None:
    """BLAST-RADIUS GUARD. A film delivered at its ordered length must be
    byte-identical to before: ship-ready, no duration finding, no extra gestalt keys."""
    out = await _stage(AsyncMock(), _analyzer()).handle(
        FinalCritiqueInput(), _duration_ctx(ordered=10.0, delivered=10.0)
    )

    assert out.quality_badge == "ship-ready"
    assert out.gestalt_summary["ship_ready"] is True
    assert "ship_ready_ceilings" not in out.gestalt_summary
    assert "duration_conformance" not in out.gestalt_summary
    assert [f.dimension for f in out.findings] == []


async def test_no_ordered_duration_leaves_the_badge_alone() -> None:
    """ "Nobody ordered a length" is not "the length is wrong". demo_video has no
    blueprint default, so an absent order must withdraw nothing."""
    out = await _stage(AsyncMock(), _analyzer()).handle(
        FinalCritiqueInput(), _duration_ctx(ordered=None, delivered=_DELIVERED_SECONDS)
    )

    assert out.quality_badge == "ship-ready"
    assert out.gestalt_summary["ship_ready"] is True
    assert [f for f in out.findings if f.dimension == "duration_conformance"] == []


async def test_an_empty_gestalt_keeps_its_unavailable_reason() -> None:
    """The override must not seed a key into an EMPTY gestalt.

    An empty dict already yields ``ship_ready=False`` / ``needs-review``, so there is
    nothing to withdraw — and making it truthy flips ``gestalt_unavailable_reason``
    from "not computed" to "computed but carries no usable signal", i.e. it would be
    counted as a calibration sample it is not. The finding is still recorded.
    """
    analyzer = _analyzer()
    analyzer.start_analysis.side_effect = RuntimeError("analyzer down")
    out = await _stage(AsyncMock(), analyzer).handle(
        FinalCritiqueInput(),
        _duration_ctx(ordered=_ORDERED_SECONDS, delivered=_DELIVERED_SECONDS),
    )

    assert out.gestalt_summary == {}
    assert out.gestalt_unavailable_reason is not None
    assert "no usable signal" not in out.gestalt_unavailable_reason
    assert out.quality_badge != "ship-ready"
    assert any(f.dimension == "duration_conformance" for f in out.findings)


# ── footage_below_bar withdraws the ship-ready badge ─────────────────────────
# Run 01a0ab23: footage scored 2.0 against a 4.0 bar and nothing downstream read it.
def _footage_ctx(*, degradations: list[dict[str, Any]], ordered: float | None = None) -> RunContext:
    footage = {
        "artifact_url": "http://v/orig.mp4",
        "duration_seconds": 10.0,
        "degradations": degradations,
    }
    prior = _prior(footage=footage, loudness={"integrated_lufs": -14.0})
    config: dict[str, Any] = {"ship_gate": {"mode": "critique_only"}}
    if ordered is not None:
        config["deliverable"] = {"target_duration_seconds": ordered}
    return _ctx(prior, config=config)


_BELOW = {
    "code": "footage_below_bar",
    "detail": "footage scored 2.0 against a bar of 4.0 after 2 pass(es); stop=plateau",
}


async def test_footage_below_bar_is_not_ship_ready() -> None:
    out = await _stage(AsyncMock(), _analyzer()).handle(
        FinalCritiqueInput(), _footage_ctx(degradations=[_BELOW])
    )

    assert out.quality_badge != "ship-ready"
    assert out.gestalt_summary["ship_ready"] is False
    assert out.gestalt_summary["ship_ready_ceilings"] is True
    findings = [f for f in out.findings if f.dimension == "footage_critique"]
    assert len(findings) == 1
    assert "scored 2.0" in findings[0].detail and findings[0].severity == "major"


async def test_footage_without_the_code_leaves_the_badge_alone() -> None:
    out = await _stage(AsyncMock(), _analyzer()).handle(
        FinalCritiqueInput(),
        _footage_ctx(degradations=[{"code": "footage_critique_unavailable", "detail": "x"}]),
    )

    assert out.quality_badge == "ship-ready"
    assert "ship_ready_ceilings" not in out.gestalt_summary
    assert [f for f in out.findings if f.dimension == "footage_critique"] == []


async def test_footage_and_duration_overrides_keep_the_real_ceilings_verdict() -> None:
    """Both overrides fire. The second must not overwrite ``ship_ready_ceilings`` with
    the first one's already-withdrawn ``False``."""
    ctx = _footage_ctx(degradations=[_BELOW], ordered=_ORDERED_SECONDS)
    out = await _stage(AsyncMock(), _analyzer()).handle(FinalCritiqueInput(), ctx)

    assert out.gestalt_summary["ship_ready"] is False
    assert out.gestalt_summary["ship_ready_ceilings"] is True
    dims = sorted(f.dimension for f in out.findings)
    assert dims == ["duration_conformance", "footage_critique"]
