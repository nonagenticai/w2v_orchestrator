"""The DEGRADATION CHANNEL: a run that shipped broken output cannot read clean.

Until this channel existed, degradation had exactly one carrier — a prose string
appended to ``notes`` — so an honest signal had to either masquerade as a note or
stay invisible, and six of them stayed invisible. These tests pin BOTH halves:

* each of the six signals records a structured entry with the right stable code
  at its own site (a helper nobody wired in is exactly as invisible as the defect
  it exists to surface);
* the ``degraded`` derivation reads the new channel — including the case that is
  the entire point, a populated ``degradations`` with an EMPTY ``notes``.

⚠️ The most valuable test in this file is the FALSE-POSITIVE guard: a clean stage
must report ``degraded`` false AND carry an empty channel. Every healthy run in
the fleet depends on it, and a quality signal that lights on everything is the
failure mode this codebase has already hit once (``mastering_gate_notes`` had to
be narrowed after firing on 96.7% of masters).

No ``integration`` marker anywhere: pytest ``addopts`` deselects that marker, so
an integration-marked test in here would never run in CI.
"""

from __future__ import annotations

import datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid7

import pytest

from src.services import tts_storage
from src.services.stages import base as base_mod
from src.services.stages import shot_assembly as shot_assembly_mod
from src.services.stages import shot_sound as shot_sound_mod
from src.services.stages.base import ArtifactUrl, StageInput, StageOutputWithNotes
from src.services.stages.demo_video.slides import ShowcaseSlidesStage
from src.services.stages.primitives import compile_stage as compile_stage_mod
from src.services.stages.primitives import narration as narration_mod
from src.services.stages.primitives.compile_stage import CompileStageBase
from src.services.stages.shared import ken_burns as ken_burns_mod
from src.services.stages.shared import sound_design as sound_design_mod
from src.services.stages.shared.degradation import (
    ADVISORY_CODES,
    AUDIO_UNNORMALIZED,
    CAPTIONS_MISSING,
    COMPILE_DURATION_MISMATCH,
    COMPILE_DURATION_UNVERIFIED,
    DEGRADATION_CODES,
    DURATION_NOT_ORDERED,
    DURATION_OUT_OF_TOLERANCE,
    DURATION_UNMEASURED,
    KEN_BURNS_ALL_STILLS_FROZEN,
    KEN_BURNS_SKIPPED_NO_BUDGET,
    KEN_BURNS_STILL_QUEUE_SATURATED,
    KEN_BURNS_STILL_SOFT_FAILED,
    MASTERING_GATE_FAILED,
    NARRATION_ESTIMATE_OFF_TARGET,
    SFX_CUE_DROPPED,
    SHOT_DURATION_FABRICATED,
    SHOT_DURATION_FLOORED,
    SLIDES_NOT_AUTHORED,
    SLIDES_STUBBED,
    SOUND_BED_MISSING,
    UNCLASSIFIED_NOTE,
    UNDERSPOKEN_BEAT_HELD,
    VISUAL_ASSETS_REUSED,
    advisory,
    degradation,
    ensure_degradation_codes,
    is_degraded,
)
from src.services.stages.shared.duration_conformance import (
    duration_conformance,
    duration_conformance_advisories,
    duration_conformance_degradations,
)
from src.services.stages.shared.mastering_gate import (
    mastering_gate_degradations,
    mastering_gate_notes,
)
from src.services.stages.shared.step_ledger import NullStepLedger
from src.services.stages.website_review.compile import WebsiteReviewCompileStage
from src.services.stages.website_review.slides import WebsiteReviewSlidesStage
from src.services.word_timed_captions import ClipRef


def _codes(entries: list[Any]) -> list[str]:
    return [entry["code"] for entry in entries]


# ══════════════════════════════════════════════════════════════════════════════
# The channel itself
# ══════════════════════════════════════════════════════════════════════════════


def test_an_unknown_code_is_refused_at_the_write_site() -> None:
    """``code`` is the queryable half, so a typo must not reach a persisted row."""
    with pytest.raises(ValueError, match="unknown degradation code"):
        degradation("mastering_gate_faled", "typo")
    # ...and every code this module ships is, by construction, accepted.
    for code in DEGRADATION_CODES:
        assert degradation(code, "d") == {"code": code, "detail": "d"}


def test_the_channel_is_distinct_from_notes_on_the_model() -> None:
    """``degradations`` is its own field — notes stays prose, untouched."""
    fields = base_mod.StageOutputWithNotes.model_fields
    assert "degradations" in fields and "notes" in fields
    # Both default EMPTY, so declaring the channel cannot degrade a healthy stage.
    assert base_mod.StageOutputWithNotes().degradations == []
    assert base_mod.StageOutputWithNotes().notes == []
    # The degradations mixin exists WITHOUT notes, so a stage with a quality
    # signal but no prose channel (the slide decks) can still raise one.
    assert "notes" not in base_mod.StageOutputWithDegradations.model_fields


# ══════════════════════════════════════════════════════════════════════════════
# The ADVISORY tier — reported, queryable, and NEVER degrading
# ══════════════════════════════════════════════════════════════════════════════


def test_the_two_tiers_cannot_borrow_each_others_codes() -> None:
    """A code's TIER decides whether a run reads degraded, so it cannot be
    chosen at the call site by accident.

    Both directions are wrong in a different way. A degradation built as an
    advisory ships a real defect on a field ``is_degraded`` does not read — the
    "recorded but invisible" failure this whole channel exists to remove. An
    advisory built as a degradation lights the flag on healthy runs, which is the
    96.7%-of-masters failure the tier exists to prevent. Both raise at the write
    site, where it is a programming error and not a production input.
    """
    assert not (ADVISORY_CODES & DEGRADATION_CODES)
    with pytest.raises(ValueError, match="it is an ADVISORY code"):
        degradation(NARRATION_ESTIMATE_OFF_TARGET, "pre-audio estimate")
    with pytest.raises(ValueError, match="it is a DEGRADATION code"):
        advisory(MASTERING_GATE_FAILED, "the picture is black")
    # A code in neither set is still a typo, in both constructors.
    with pytest.raises(ValueError, match="unknown advisory code"):
        advisory("narration_estimate_off_targt", "typo")
    # ...and every advisory code this module ships is accepted by advisory().
    for code in ADVISORY_CODES:
        assert advisory(code, "d") == {"code": code, "detail": "d"}


def test_an_output_carrying_only_advisories_is_not_degraded() -> None:
    """THE point of the tier, asserted on the persisted shape.

    A lesson run delivered within 0.22% of its order while its script stage
    stamped ``duration_out_of_tolerance`` from a PRE-AUDIO estimate — a
    conformant film reported broken by a guess. An advisory carries that fact
    without the verdict: populated ``advisories``, empty everything else, run
    reads CLEAN. ``is_degraded`` must never learn to read this field.
    """
    entry = advisory(NARRATION_ESTIMATE_OFF_TARGET, "lesson_script: estimated +26.4%")
    output = base_mod.StageOutputWithDegradations(advisories=[entry])
    assert output.degradations == []
    assert output.advisories == [entry]
    dumped = output.model_dump()
    assert is_degraded(dumped) is False
    # The SQL twin agrees by construction: it reads two keys, and this is not
    # one of them. (The clause itself is pinned in the last test of this file.)
    assert is_degraded({"advisories": [entry], "degradations": [], "notes": []}) is False
    # ...and a real degradation ALONGSIDE an advisory still degrades — the tier
    # must not be able to launder one.
    both = base_mod.StageOutputWithDegradations(
        advisories=[entry], degradations=[degradation(MASTERING_GATE_FAILED, "black")]
    )
    assert is_degraded(both.model_dump()) is True


# ══════════════════════════════════════════════════════════════════════════════
# Signal 1 — mastering gate (the narrowing is REMOVED on this channel)
# ══════════════════════════════════════════════════════════════════════════════


def test_a_gate_failing_only_on_tunable_ceilings_records_nothing() -> None:
    """THE ANTI-NOISE GUARD, and the whole point of the tier.

    This is the 207-of-214 population: ``gate_pass=false`` driven by a deband
    ceiling the config's own comment says fires on 96.7% of masters, on a bright,
    textured, perfectly watchable film. A tunable ceiling reasonable operators
    disagree about is exactly what "advisory" means — the verdict and the numbers
    are already on the persisted mastering report. Recording it here would make
    ``degraded`` near-universal on render runs and recreate the defect this
    channel exists to fix.

    ``duplicate_frames`` is in this list deliberately: a high duplicate ratio in
    this pipeline is INTENTIONAL DWELL, never a defect.
    """
    report = {
        "gate_pass": False,
        "rerun_stages": ["deband", "duplicate_frames", "colorspace", "loudness"],
        "luma_spread": 187.0,  # bright content — the picture is fine
        "banding_scores": [1.0],
        "duplicate_frame_ratio": 0.904,  # fleet median; dwell, not damage
        "integrated_lufs": -21.4,
    }
    assert mastering_gate_degradations(report) == []
    # The prose channel is likewise silent — neither channel cries wolf here.
    assert mastering_gate_notes(report) == []


def test_av_coverage_is_structural_and_records() -> None:
    """The compiler's own always-enforced stage: the picture stops before the
    narration does. No threshold makes shipping that right."""
    entries = mastering_gate_degradations(
        {
            "gate_pass": False,
            "rerun_stages": ["deband", "av_coverage"],
            "luma_spread": 187.0,
            "av_gap_seconds": 8.8,
            "video_stream_seconds": 135.513,
            "audio_stream_seconds": 144.293,
        }
    )
    assert _codes(entries) == [MASTERING_GATE_FAILED]
    assert "av_coverage" in entries[0]["detail"]
    assert "8.8" in entries[0]["detail"]


def test_a_blank_picture_is_structural_and_records() -> None:
    """The other reading with no advisory tier: the film is black.

    Both shapes count — the compiler's own ``dead_frame`` verdict, and a
    ``dead_frame_ratio`` at or above its 0.10 cutoff without that verdict.

    INVERTED: a floor-level ``luma_spread`` alone used to count, and that was the
    defect — it is the MIN over every frame, so one fade-to-black zeroes it on a good
    film (live run 01a08f66: dead_frame_ratio=0.0098, luma_spread=0.0, stamped BLANK).
    """
    by_verdict = mastering_gate_degradations(
        {"gate_pass": False, "rerun_stages": ["dead_frame"], "luma_spread": 0.0}
    )
    assert _codes(by_verdict) == [MASTERING_GATE_FAILED]
    assert "BLANK/dead" in by_verdict[0]["detail"]

    by_ratio = mastering_gate_degradations(
        {"gate_pass": False, "rerun_stages": ["deband"], "dead_frame_ratio": 0.25}
    )
    assert _codes(by_ratio) == [MASTERING_GATE_FAILED]
    assert "BLANK/dead" in by_ratio[0]["detail"]
    assert "dead_frame_ratio=0.25" in by_ratio[0]["detail"]

    one_flat_frame = mastering_gate_degradations(
        {
            "gate_pass": False,
            "rerun_stages": ["deband"],
            "dead_frame_ratio": 0.009825,
            "luma_spread": 0.0,
        }
    )
    assert one_flat_frame == []


def test_an_absent_mastering_report_is_unmeasured_not_a_pass() -> None:
    """Neither tier — the gate did not RUN, so nothing measured this master.

    Recorded anyway, and it costs no noise: "an absent measurement reported as a
    passing one" is this codebase's #1 defect class, and unlike a tunable breach
    it is rare.
    """
    assert _codes(mastering_gate_degradations(None)) == [MASTERING_GATE_FAILED]
    assert "UNMEASURED" in mastering_gate_degradations(object())[0]["detail"]


def test_a_passing_mastering_gate_records_nothing() -> None:
    """The false-positive guard for this signal.

    The second case is the one that makes this guard load-bearing rather than
    subsumed by the tier filter: a gate the compiler PASSED whose ``luma_spread``
    happens to sit at the floor (a deliberately dark master it measured and
    cleared). The compiler's verdict wins — the orchestrator does not re-litigate
    a check that ran and passed, or it starts overruling the only component that
    actually looked at the pixels.
    """
    assert mastering_gate_degradations({"gate_pass": True, "rerun_stages": []}) == []
    assert (
        mastering_gate_degradations({"gate_pass": True, "rerun_stages": [], "luma_spread": 0.0})
        == []
    )


def test_a_mixed_gate_names_only_the_structural_metrics() -> None:
    """Structural + tunable ⇒ ONE entry, describing the structural half.

    The detail must not quote the banding score or the duplicate ratio: neither
    says anything about why this master is structurally broken, and quoting the
    dwell ratio would read as an accusation against intentional behaviour.
    """
    entries = mastering_gate_degradations(
        {
            "gate_pass": False,
            "rerun_stages": ["deband", "duplicate_frames", "av_coverage"],
            "banding_scores": [1.0],
            "duplicate_frame_ratio": 0.904,
            "av_gap_seconds": 8.8,
            "luma_spread": 187.0,
        }
    )
    assert len(entries) == 1
    detail = entries[0]["detail"]
    assert "av_gap_seconds=8.8" in detail
    assert "0.904" not in detail, detail
    assert "banding" not in detail, detail


# ══════════════════════════════════════════════════════════════════════════════
# Signal 2 — delivered-vs-ordered duration
# ══════════════════════════════════════════════════════════════════════════════


def test_a_duration_breach_is_recorded() -> None:
    """The measured website_review case: 327.6s delivered against 120s ordered.

    Stated against a MEASURED (playable-basis) delivery, because that is the only
    kind that can be trusted to yield a breach at all. An unmeasured verdict
    additionally carries ``DURATION_UNMEASURED`` — on ``advisories``, not here
    (see the pair of tests below).
    """
    entries = duration_conformance_degradations(
        duration_conformance(120.0, 327.6, playable_seconds=327.6)
    )
    assert _codes(entries) == [DURATION_OUT_OF_TOLERANCE]
    assert "327.6s" in entries[0]["detail"] and "120.0s" in entries[0]["detail"]


def test_a_conforming_duration_records_nothing() -> None:
    """False-positive guard: an on-target MEASURED delivery carries an EMPTY channel."""
    assert (
        duration_conformance_degradations(
            duration_conformance(120.0, 121.0, playable_seconds=121.0)
        )
        == []
    )


def test_an_unprobed_conforming_duration_is_not_reported_as_clean() -> None:
    """An on-target DECLARED verdict is unverified, and must not read as a pass.

    This is the defect the probe wiring exists to close: ``duration_seconds`` is
    the nominal sum of shot durations, so a master whose picture stopped early
    reports a length that agrees with its own plan. "Nobody measured this" and
    "measured and on target" must not be byte-identical in the record.

    But it is an ADVISORY, so the degrading channel stays EMPTY. "We did not
    measure it" is a fact about our instrumentation, not about the artifact —
    and it fires on every stage that cannot reach a probe, which is 100% of
    ``demo_video``/``showcase``. As a degradation it would light the flag on
    healthy runs forever: the exact failure this file's tier test names.
    """
    result = duration_conformance(120.0, 121.0)
    assert duration_conformance_degradations(result) == []
    assert _codes(duration_conformance_advisories(result)) == [DURATION_UNMEASURED]


def test_an_unprobed_breach_records_both_the_breach_and_the_doubt() -> None:
    """Out of band AND unverified are independent facts, on their own channels."""
    result = duration_conformance(120.0, 327.6)
    assert _codes(duration_conformance_degradations(result)) == [DURATION_OUT_OF_TOLERANCE]
    assert _codes(duration_conformance_advisories(result)) == [DURATION_UNMEASURED]


def test_an_unprobed_conforming_compile_output_is_not_degraded() -> None:
    """THE POINT OF THE TIER, asserted end to end on a persisted output blob.

    A conforming master that nobody probed — the permanent state of every
    ``demo_video`` and ``showcase`` delivery, since ``encode_master`` has no
    ``actual_duration_seconds`` to pass — must read CLEAN while still carrying
    the unmeasured fact for anyone querying it. ``DURATION_UNMEASURED`` spent
    one commit on ``degradations``, where it flipped ``degraded=True`` on 100%
    of two content types indefinitely. This is the regression guard for that.
    """
    result = duration_conformance(120.0, 121.0)
    output = base_mod.StageOutputWithNotes(
        degradations=duration_conformance_degradations(result),
        advisories=duration_conformance_advisories(result),
    )
    dumped = output.model_dump()
    assert is_degraded(dumped) is False
    # ...and the fact is genuinely carried, not merely absent from `degraded`.
    assert _codes(output.advisories) == [DURATION_UNMEASURED]


def test_an_unordered_duration_is_recorded_rather_than_silent() -> None:
    """``None`` (nothing ordered) is NOT a breach — but it is not silence either.

    It used to be absent from the channel entirely, which is precisely what made
    it invisible: a null verdict cannot be found by any ``within_tolerance =
    false`` query, so a deliverable nobody ordered a duration for looked exactly
    like a deliverable that conformed. It is now recorded under its own code so
    the gap is queryable, while ``duration_conformance`` still returns ``None``
    rather than inventing an order it was never given.
    """
    entries = duration_conformance_degradations(duration_conformance(None, 327.6))
    assert _codes(entries) == [DURATION_NOT_ORDERED]
    # The detail must name WHERE the order was looked for, so the reader knows
    # which input to go fix rather than suspecting the measurement.
    assert "config_snapshot" in entries[0]["detail"]


def test_website_review_compile_carries_the_breach_out_of_build_output() -> None:
    """The WIRING, asserted on the built output — not on the helper."""
    ctx = cast(
        Any,
        SimpleNamespace(
            run_id=uuid7(),
            config_snapshot={"brief": {"duration_seconds": 120}, "deliverable": {}},
            prior_output=lambda _stage: None,
        ),
    )
    # Both results carry a `mastering` block with BOTH stream lengths — the
    # compiler's SEPARATE probes of video and audio, whose max is the PLAYABLE
    # length the verdict is now taken against. `actual_duration_seconds` is
    # deliberately NOT what is graded: it is min(video_stream, container), the
    # PICTURE length, which under-reports exactly the truncated masters this
    # channel exists to catch. Omitting the mastering block makes the delivery
    # UNMEASURED (`duration_unmeasured`, on `advisories`).
    output = WebsiteReviewCompileStage().build_output(
        {
            "artifact_url": "http://m/out.mp4",
            "duration_seconds": 327.6,
            "mastering": {"video_stream_seconds": 327.6, "audio_stream_seconds": 327.6},
        },
        ctx,
    )
    assert _codes(output.degradations) == [DURATION_OUT_OF_TOLERANCE]
    # ...and an on-target master out of the SAME stage carries an empty channel.
    on_target = WebsiteReviewCompileStage().build_output(
        {
            "artifact_url": "http://m/out.mp4",
            "duration_seconds": 121.0,
            "mastering": {"video_stream_seconds": 121.0, "audio_stream_seconds": 121.0},
        },
        ctx,
    )
    assert on_target.degradations == []


def test_website_review_compile_grades_the_measurement_not_the_declaration() -> None:
    """The wiring that makes the channel capable of catching a short master.

    Same shape as the live defect: the stage DECLARES an on-target 121.0s against
    a 120s order (a declared-basis verdict is a clean pass, empty channel) while
    the file that shipped PLAYS for 72.0s. The measurement wins, so the breach
    reaches the channel instead of being graded away by the plan it came from.
    """
    ctx = cast(
        Any,
        SimpleNamespace(
            run_id=uuid7(),
            config_snapshot={"brief": {"duration_seconds": 120}, "deliverable": {}},
            prior_output=lambda _stage: None,
        ),
    )
    output = WebsiteReviewCompileStage().build_output(
        {
            "artifact_url": "http://m/out.mp4",
            "duration_seconds": 121.0,
            # The PICTURE-only field is present and WRONG-for-grading on purpose:
            # if anything ever reads it again, this assertion moves.
            "actual_duration_seconds": 72.0,
            "mastering": {"video_stream_seconds": 72.0, "audio_stream_seconds": 72.0},
        },
        ctx,
    )

    assert output.duration_conformance is not None
    assert output.duration_conformance["delivered_basis"] == "playable"
    assert output.duration_conformance["delivered_seconds"] == pytest.approx(72.0)
    assert _codes(output.degradations) == [DURATION_OUT_OF_TOLERANCE]


# ══════════════════════════════════════════════════════════════════════════════
# Signal 3 — a dropped music / ambient bed
# ══════════════════════════════════════════════════════════════════════════════


class _FakeLibrary:
    def is_valid(self, _id: str, *, category: str) -> bool:
        return True

    # Pinned ids are checked with ``resolves`` (P6 ORCH-4: a retired tenant track
    # still resolves for a run that pinned it).
    def resolves(self, _id: str, *, category: str) -> bool:
        return True


@pytest.mark.asyncio
async def test_a_bed_that_cannot_be_presigned_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both cues were CHOSEN and both silently vanished from the payload — the
    video ships with no bed at all and nothing said so, not even a log line."""

    async def _no_library() -> Any:
        return _FakeLibrary()

    async def _no_url(_asset_id: str) -> str | None:
        return None

    monkeypatch.setattr(sound_design_mod, "load_sound_library", _no_library)
    monkeypatch.setattr(sound_design_mod, "presign_sound_asset", _no_url)

    sink: list[Any] = []
    cfg = {"enabled": True, "music": "calm-01", "ambient": "room-01"}
    bed = await sound_design_mod.resolve_audio_bed(
        cast(Any, SimpleNamespace(config_snapshot={}, prior_run_outputs={})),
        [{"seconds": 5.0}],
        cfg,
        degradations=sink,
    )
    assert bed is None  # the silent drop this records
    assert _codes(sink) == [SOUND_BED_MISSING, SOUND_BED_MISSING]
    assert "calm-01" in sink[0]["detail"] and "ambient" in sink[1]["detail"]


@pytest.mark.asyncio
async def test_a_bed_that_resolves_records_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """False-positive guard for the sound signal."""

    async def _library() -> Any:
        return _FakeLibrary()

    async def _url(asset_id: str) -> str | None:
        return f"https://minio.invalid/{asset_id}.mp3"

    monkeypatch.setattr(sound_design_mod, "load_sound_library", _library)
    monkeypatch.setattr(sound_design_mod, "presign_sound_asset", _url)

    sink: list[Any] = []
    bed = await sound_design_mod.resolve_audio_bed(
        cast(Any, SimpleNamespace(config_snapshot={}, prior_run_outputs={})),
        [{"seconds": 5.0}],
        {"enabled": True, "music": "calm-01", "ambient": "room-01"},
        degradations=sink,
    )
    assert bed is not None and "music" in bed
    assert sink == []


# ══════════════════════════════════════════════════════════════════════════════
# Signals 4+1 through the compile tail — dropped captions and the mastering gate
# ══════════════════════════════════════════════════════════════════════════════


class _CompileInput(StageInput):
    pass


class _CompileOutput(StageOutputWithNotes):
    artifact_url: ArtifactUrl
    duration_seconds: float


class _CaptionedStage(CompileStageBase[_CompileInput, _CompileOutput]):
    """A final-video compile stage that opts into word-timed captions."""

    name = "test_compile_degradation_channel"
    input_model = _CompileInput
    output_model = _CompileOutput

    def build_shots(self, ctx: Any) -> list[dict[str, Any]]:
        return [{"id": "s1", "line_id": "l1", "seconds": 2.0}]

    def build_output(self, result: dict[str, Any], ctx: Any) -> _CompileOutput:
        return _CompileOutput(
            artifact_url=str(result["artifact_url"]),
            duration_seconds=float(result["duration_seconds"]),
        )

    def compile_tags(self, ctx: Any) -> dict[str, str]:
        return {}

    def caption_clip_refs(self, ctx: Any) -> dict[str, ClipRef]:
        return {
            "l1": ClipRef(
                line_id="l1",
                text="hi",
                narrate_stage="narrate",
                clip_filename="c.wav",
                seconds=2.0,
            )
        }


class _FakeStorage:
    async def resign_presigned_url(self, url: str) -> str:
        return url


class _StubAsr:
    async def close(self) -> None:
        return None


async def _run_captioned_compile(
    monkeypatch: pytest.MonkeyPatch,
    *,
    captions_raise: bool,
    mastering: object,
    stage_cls: type[CompileStageBase[_CompileInput, _CompileOutput]] | None = None,
) -> _CompileOutput:
    """Drive the whole compile tail and return the output the run would PERSIST."""

    async def _compile(**_kwargs: Any) -> dict[str, Any]:
        return {
            "artifact_url": "http://m/out.mp4",
            "duration_seconds": 2.0,
            "declared_duration_seconds": 2.0,
            # Present and matching: otherwise the compile duration-QA guard
            # raises its own UNMEASURED note and this file's notes assertions
            # would be measuring that instead of the channel under test.
            "actual_duration_seconds": 2.0,
            "mastering": mastering,
        }

    client = AsyncMock()
    client.compile = AsyncMock(side_effect=_compile)

    monkeypatch.setattr(
        compile_stage_mod,
        "get_settings",
        lambda: SimpleNamespace(
            pipeline=SimpleNamespace(
                captions_default_on_showcase=True,
                grade_default_on_showcase=False,
                captions_karaoke=False,
            )
        ),
    )

    async def _build(**_kwargs: Any) -> list[dict[str, Any]]:
        if captions_raise:
            raise RuntimeError("transcription_worker 503")
        return [{"shot_index": 0, "cues": [{"text": "hi", "start": 0.0, "end": 1.0}]}]

    async def _storage() -> Any:
        return _FakeStorage()

    monkeypatch.setattr(compile_stage_mod, "build_shot_captions", _build)
    monkeypatch.setattr(compile_stage_mod, "_get_storage", _storage)
    monkeypatch.setattr(compile_stage_mod, "TranscriptionWorkerClient", _StubAsr)

    tts_storage.set_storage_for_tests(cast(Any, _FakeStorage()))
    try:
        return await (stage_cls or _CaptionedStage)(client).handle(
            _CompileInput(),
            ctx=cast(
                Any,
                SimpleNamespace(
                    run_id=uuid7(),
                    render_idempotency_discriminator="",
                    config_snapshot={"deliverable": {"captions_mode": "burned_in"}},
                    prior_output=lambda _stage: None,
                    step_ledger=lambda _stage: NullStepLedger(),
                ),
            ),
        )
    finally:
        tts_storage.set_storage_for_tests(None)


@pytest.mark.asyncio
async def test_dropped_captions_reach_the_persisted_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Captions were requested, the build soft-failed, EVERY subtitle is gone.

    Asserted on the returned output — the thing the run persists — never on the
    sink list, because a sink assertion still passes if ``handle`` stops folding
    the sink into the output.
    """
    output = await _run_captioned_compile(
        monkeypatch, captions_raise=True, mastering={"gate_pass": True, "rerun_stages": []}
    )
    assert _codes(output.degradations) == [CAPTIONS_MISSING]
    assert "NO subtitles" in output.degradations[0]["detail"]


@pytest.mark.asyncio
async def test_a_failed_gate_reaches_the_persisted_output_of_every_compile_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mastering wiring, on the shared tail every render blueprint runs."""
    output = await _run_captioned_compile(
        monkeypatch,
        captions_raise=False,
        mastering={
            "gate_pass": False,
            "rerun_stages": ["deband", "av_coverage"],
            "luma_spread": 187.0,
            "av_gap_seconds": 8.8,
        },
    )
    assert _codes(output.degradations) == [MASTERING_GATE_FAILED]
    # The prose channel stayed silent on this report — the whole point.
    assert output.notes == []


@pytest.mark.asyncio
async def test_a_clean_compile_carries_an_empty_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE false-positive guard for the compile tail: captions built, gate passed
    ⇒ no entry, no note, and the run reports clean."""
    output = await _run_captioned_compile(
        monkeypatch, captions_raise=False, mastering={"gate_pass": True, "rerun_stages": []}
    )
    assert output.degradations == []
    assert output.notes == []
    assert is_degraded(output.model_dump()) is False


# ══════════════════════════════════════════════════════════════════════════════
# Pre-submit prose notes carry a CODE — never the executor's `unclassified_note`
# ══════════════════════════════════════════════════════════════════════════════


class _StillStage(_CaptionedStage):
    """One bare still (a Ken Burns candidate) carrying an SFX cue and a
    fabricated-hold marker — every pre-submit note producer at once."""

    name = "test_compile_presubmit_codes"

    def ken_burns_local_only(self, ctx: Any) -> bool:
        # Pins the video_production RECORDER path, which the base no longer takes.
        return False

    def build_shots(self, ctx: Any) -> list[dict[str, Any]]:
        return [
            {
                "id": "s1",
                "line_id": "l1",
                "seconds": 2.0,
                "image_url": "https://minio.test/a.png",
                "sfx": [{"sfx_id": "whoosh", "at_seconds": 0.0, "gain_db": 0.0}],
                shot_assembly_mod.FABRICATED_HOLD_KEY: True,
            }
        ]


def _wire_failing_recorder(monkeypatch: pytest.MonkeyPatch, error: Exception) -> AsyncMock:
    """A video_production fake whose recording fails with *error*; no real I/O."""
    video = AsyncMock()
    video.record_animation.return_value = "kb-1"
    video.wait_for_content_recording.side_effect = error

    async def _fetch(url: str, client: Any) -> tuple[bytes, str]:
        return (b"\x89PNG\r\n", "image/png")

    async def _no_cache(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(ken_burns_mod, "VideoProductionClient", lambda: video)
    monkeypatch.setattr(ken_burns_mod, "_fetch_own_asset", _fetch)
    monkeypatch.setattr(ken_burns_mod, "find_existing_media", _no_cache)
    monkeypatch.setattr(ken_burns_mod, "_saturation_sleep", _no_sleep)
    monkeypatch.setattr(shot_sound_mod, "presign_sound_asset", _no_cache)
    return video


@pytest.mark.asyncio
async def test_a_ken_burns_still_fallback_reaches_the_output_with_its_own_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recorder is saturated past its retries: the shot ships as a still, and
    the persisted output names THAT — plus the dropped SFX cue and the fabricated
    hold raised in the same pre-submit block — instead of prose alone.

    Asserted on the output ``handle`` returns and then through the executor's own
    ``ensure_degradation_codes``: it must find nothing to stamp, i.e. the "prose
    notes only" path is not taken.
    """
    video = _wire_failing_recorder(
        monkeypatch,
        RuntimeError(
            "video_production job kb-1 failed: QUEUE_SATURATED: no per-pod recording "
            "slot within 300s (per-pod cap 3)"
        ),
    )
    output = await _run_captioned_compile(
        monkeypatch,
        captions_raise=False,
        mastering={"gate_pass": True, "rerun_stages": []},
        stage_cls=_StillStage,
    )
    assert video.record_animation.await_count == ken_burns_mod._SATURATION_RETRIES + 1
    codes = _codes(output.degradations)
    assert set(codes) == {
        SHOT_DURATION_FABRICATED,
        KEN_BURNS_STILL_QUEUE_SATURATED,
        KEN_BURNS_ALL_STILLS_FROZEN,
        SFX_CUE_DROPPED,
    }
    assert KEN_BURNS_STILL_SOFT_FAILED not in codes
    # The prose is NOT removed (users read it) — every note has a coded twin
    # whose detail is that same sentence.
    assert len(output.notes) == 4
    assert sorted(output.notes) == sorted(entry["detail"] for entry in output.degradations)
    dumped = output.model_dump()
    assert ensure_degradation_codes(_StillStage.name, dumped) is False
    assert UNCLASSIFIED_NOTE not in _codes(dumped["degradations"])


@pytest.mark.asyncio
async def test_a_generic_ken_burns_soft_fail_is_not_coded_as_saturation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = _wire_failing_recorder(monkeypatch, RuntimeError("chromium crashed"))
    output = await _run_captioned_compile(
        monkeypatch,
        captions_raise=False,
        mastering={"gate_pass": True, "rerun_stages": []},
        stage_cls=_StillStage,
    )
    assert video.record_animation.await_count == 1
    codes = _codes(output.degradations)
    assert KEN_BURNS_STILL_SOFT_FAILED in codes
    assert KEN_BURNS_STILL_QUEUE_SATURATED not in codes


def test_the_ken_burns_budget_skip_has_its_own_code() -> None:
    notes: list[str] = []
    degradations: list[Any] = []
    ken_burns_mod._report_stills(
        notes, candidates=3, upgraded=1, cache_only=True, degradations=degradations
    )
    assert _codes(degradations) == [KEN_BURNS_SKIPPED_NO_BUDGET]
    assert degradations[0]["detail"] == notes[0]
    # ...and a pass that upgraded at least one still, with budget, says nothing.
    clean: list[Any] = []
    ken_burns_mod._report_stills([], candidates=3, upgraded=1, cache_only=False, degradations=clean)
    assert clean == []


def test_every_drained_shot_marker_has_its_own_code() -> None:
    shots: list[dict[str, Any]] = [
        {"id": "a", shot_assembly_mod.FABRICATED_HOLD_KEY: True},
        {"id": "b", shot_assembly_mod.SHORT_BEAT_FLOOR_KEY: True},
        {"id": "c", shot_assembly_mod._VISUAL_REUSE_NOTES_KEY: ["shot visuals: reused"]},
        {"id": "d", shot_assembly_mod.UNDERSPOKEN_HOLD_KEY: 1.5},
    ]
    degradations: list[Any] = []
    notes = shot_assembly_mod.drain_fabricated_hold_notes(shots, degradations=degradations)
    assert _codes(degradations) == [
        SHOT_DURATION_FABRICATED,
        SHOT_DURATION_FLOORED,
        VISUAL_ASSETS_REUSED,
        UNDERSPOKEN_BEAT_HELD,
    ]
    assert [entry["detail"] for entry in degradations] == notes
    # The markers are drained either way, and a clean shot list records nothing.
    assert shots == [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}]
    clean: list[Any] = []
    assert shot_assembly_mod.drain_fabricated_hold_notes(shots, degradations=clean) == []
    assert clean == []


def test_compile_duration_qa_notes_get_a_code_each() -> None:
    unmeasured = compile_stage_mod._compile_duration_qa_notes({"duration_seconds": 10.0})
    short = compile_stage_mod._compile_duration_qa_notes(
        {"duration_seconds": 263.5, "actual_duration_seconds": 234.0}
    )
    assert _codes(compile_stage_mod._compile_duration_qa_degradations(unmeasured)) == [
        COMPILE_DURATION_UNVERIFIED
    ]
    entries = compile_stage_mod._compile_duration_qa_degradations(short)
    assert _codes(entries) == [COMPILE_DURATION_MISMATCH]
    assert entries[0]["detail"] == short[0]
    assert compile_stage_mod._compile_duration_qa_degradations([]) == []


# ══════════════════════════════════════════════════════════════════════════════
# Signal 5 — an empty slide deck
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_a_plain_website_review_deck_records_an_advisory_not_a_degradation() -> None:
    """A website_review that authors no slide shots is the NORMAL path.

    It must not mark the run degraded (a ``slides_stubbed`` degradation fired on
    every plain review, prod run 01a0a054) — but the empty deck must still be
    answerable from the run record, via the ``slides_not_authored`` advisory.
    """
    ctx = cast(
        Any,
        SimpleNamespace(
            run_id=uuid7(),
            config_snapshot={"brief": {}, "deliverable": {"format": "16x9"}},
            prior_output=lambda _stage: None,
        ),
    )
    out = await WebsiteReviewSlidesStage().handle(cast(Any, SimpleNamespace()), ctx)
    # ⛔ The BOOLEAN agrees with the tier, exactly as for a generated-mode showcase:
    # `slides_stubbed` means "authored lines, no cards" in every stage, and an advisory
    # with `degraded: false` beside `slides_stubbed: True` is two fields disagreeing.
    assert out.slides_stubbed is False
    assert out.slides_render_failed is False
    assert out.degradations == []
    assert _codes(out.advisories) == [SLIDES_NOT_AUTHORED]
    assert is_degraded(out.model_dump(mode="json")) is False


def _showcase_ctx(brief: dict[str, Any]) -> Any:
    return cast(
        Any,
        SimpleNamespace(
            run_id=uuid7(),
            config_snapshot={"brief": brief, "deliverable": {}},
            prior_output=lambda _stage: None,
        ),
    )


@pytest.mark.asyncio
async def test_a_generated_mode_showcase_deck_records_an_advisory_not_a_degradation() -> None:
    """A generated-mode showcase (no authored lines) has NO card source at all — nothing
    in ``src/`` writes a ``card`` block — so the empty deck is the expected outcome on
    EVERY such run. As ``slides_stubbed`` it marked every one degraded (prod 01a0ab23).

    ⛔ And the BOOLEAN must agree with the tier. Run 01a0afab recorded the
    ``slides_not_authored`` advisory with ``degraded: false`` AND ``slides_stubbed: True``
    on the same output — two fields disagreeing about whether this is a stub. The
    advisory carries the "why"; the flag is reserved for the authored-lines case."""
    out = await ShowcaseSlidesStage().handle(cast(Any, SimpleNamespace()), _showcase_ctx({}))
    assert out.slides_stubbed is False
    assert out.slides_render_failed is False
    assert out.degradations == []
    assert _codes(out.advisories) == [SLIDES_NOT_AUTHORED]
    assert is_degraded(out.model_dump(mode="json")) is False


@pytest.mark.asyncio
async def test_an_authored_showcase_without_cards_still_records_a_degradation() -> None:
    brief = {"lines": [{"id": "clip-1", "text": "A clean dashboard."}]}
    out = await ShowcaseSlidesStage().handle(cast(Any, SimpleNamespace()), _showcase_ctx(brief))
    assert out.slides_stubbed is True
    assert _codes(out.degradations) == [SLIDES_STUBBED]
    assert out.advisories == []
    assert is_degraded(out.model_dump(mode="json")) is True


# ══════════════════════════════════════════════════════════════════════════════
# Signal 6 — a clip stored at an unknown loudness
# ══════════════════════════════════════════════════════════════════════════════


def _synthesis(*, integrated_lufs: float | None) -> Any:
    return SimpleNamespace(
        audio_bytes=b"RIFFfake",
        format="wav",
        voice="af_heart",
        duration_seconds=1.5,
        integrated_lufs=integrated_lufs,
    )


@pytest.mark.asyncio
async def test_an_unproven_clip_level_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Prometheus counter is a FLEET aggregate: it can say 4% of clips shipped
    at an unknown level and it can never say which run. Now the run says so."""
    monkeypatch.setattr(narration_mod, "tts_premaster_fallback_enabled", lambda: False)
    sink: list[Any] = []
    _audio, loudness = await narration_mod.ensure_tts_loudness(
        _synthesis(integrated_lufs=None),
        stage_name="promo_narrate",
        degradations=sink,
    )
    assert loudness.normalized is False
    assert _codes(sink) == [AUDIO_UNNORMALIZED]
    assert "disabled" in sink[0]["detail"] and "promo_narrate" in sink[0]["detail"]


@pytest.mark.asyncio
async def test_a_proven_clip_level_records_nothing() -> None:
    """False-positive guard: the overwhelmingly common tts_backend-proven path."""
    sink: list[Any] = []
    _audio, loudness = await narration_mod.ensure_tts_loudness(
        _synthesis(integrated_lufs=-18.0),
        stage_name="promo_narrate",
        degradations=sink,
    )
    assert loudness.source == "tts_backend"
    assert sink == []


# ══════════════════════════════════════════════════════════════════════════════
# The `degraded` derivation
# ══════════════════════════════════════════════════════════════════════════════


def _row(output: dict[str, Any] | None, *, status: str = "completed") -> Any:
    now = datetime.datetime.now(datetime.UTC)
    return SimpleNamespace(
        id=uuid7(),
        run_id=uuid7(),
        stage_name="website_review_compile",
        status=status,
        output=output,
        error=None,
        started_at=now,
        completed_at=now,
        attempts=0,
    )


def test_the_sql_twin_reads_both_channels() -> None:
    """``degraded_stage_clause`` is what the run LIST and the live-monitor poll
    render from — a run degraded in the detail view and clean in the list is the
    same lie in a different place. Compiled against the postgresql dialect (no
    DB) so a clause that no longer mentions ``degradations`` is caught here and
    not by a reviewer diffing two files.
    """
    from sqlalchemy.dialects import postgresql

    from src.database.repositories.run_repository import degraded_stage_clause

    # No ``literal_binds``: the falsey-JSON value list binds python objects the
    # dialect cannot inline, so the JSON KEY NAMES land in the bind params and
    # the structure lands in the SQL text. Both halves are asserted.
    compiled = degraded_stage_clause().compile(dialect=postgresql.dialect())
    sql = str(compiled)
    params = list(compiled.params.values())  # some values are lists, so not a set
    assert "degradations" in params, compiled.params
    assert "notes" in params, compiled.params
    assert "completed" in params, compiled.params
    # OR, never AND: either channel alone must raise the flag.
    assert " OR " in sql, sql
    assert "jsonb_typeof" in sql, sql


# ══════════════════════════════════════════════════════════════════════════════
# The compiler's ADVISORY tier — MasteringReport.advisories (run 01a0ab23)
# ══════════════════════════════════════════════════════════════════════════════
# Run 01a0ab23 delivered `gate_pass: true`, `violations: []` while all three shots
# requested deband. The compiler now publishes that as `advisories[]`, OUTSIDE
# `violations[]`; it must reach the persisted output as an ADVISORY, never a degradation.
_BANDING_ADVISORY = {
    "stage": "banding_residual",
    "metric": "banding_scores",
    "value": 0.5935,
    "limit": 0.15,
    "tier": "tunable",
}


@pytest.mark.asyncio
async def test_a_compiler_advisory_reaches_the_persisted_output_as_an_advisory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.services.stages.shared.degradation import MASTERING_ADVISORY

    output = await _run_captioned_compile(
        monkeypatch,
        captions_raise=False,
        mastering={
            "gate_pass": True,
            "rerun_stages": [],
            "violations": [],
            "advisories": [_BANDING_ADVISORY],
        },
    )
    assert _codes(output.advisories) == [MASTERING_ADVISORY]
    detail = output.advisories[0]["detail"]
    assert "banding_residual" in detail and "banding_scores=0.5935" in detail
    assert "limit 0.15" in detail and "tier=tunable" in detail
    # NOT the violations path: nothing degrades, nothing is written as prose.
    assert output.degradations == []
    assert output.notes == []
    assert is_degraded(output.model_dump(mode="json")) is False


def test_mastering_gate_advisories_tolerates_absent_and_malformed_shapes() -> None:
    from src.services.stages.shared.mastering_gate import mastering_gate_advisories

    assert mastering_gate_advisories(None) == []
    assert mastering_gate_advisories({"gate_pass": True}) == []
    assert mastering_gate_advisories({"advisories": None}) == []
    assert mastering_gate_advisories({"advisories": "banding"}) == []
    assert mastering_gate_advisories({"advisories": []}) == []
    # A value of 0.0 is an observation, not an absence — it is still reported.
    zero = mastering_gate_advisories({"advisories": ["junk", {**_BANDING_ADVISORY, "value": 0.0}]})
    assert len(zero) == 1 and "banding_scores=0.0 " in zero[0]["detail"]
    # And the violations path never sees it.
    assert mastering_gate_degradations({"gate_pass": True, "advisories": [_BANDING_ADVISORY]}) == []
