"""action_reel music_select — the stage the whole product hangs on, untested.

The incident these tests exist for was silent, green, and total.
``VideoAnalyzerAudioGrid.meter`` was declared ``str``; the analyzer sends an
**int** (``4``). So pydantic rejected EVERY grid the analyzer ever returned,
``music_select`` caught the ValidationError and degraded exactly as designed, and
the only visible symptom was ``beats: []`` on a stage that reported success. No
reel was ever cut to a beat, every run said it was fine, and the analyzer — bpm
132.512, confidence 0.94, 98 beats — was never at fault. One field's type, on a
cross-service boundary, with nothing asserting the contract.

So the first test here is a CONTRACT test, and it is the important one: a
faithful copy of a real ``POST /api/analyze/audio`` body, taken from a live
response captured on 2026-08-16, parsed by the real model and pushed through the
real stage. It fails the moment any field's type is narrowed again. The rest pin
the four decisions the module docstring makes explicitly — the library is
authoritative, an authored pin is not reconsidered, an empty library is a
precondition failure rather than a skip, and a lost analyzer degrades VISIBLY.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid7

import httpx
import pytest
from w2v_common.clients import WMResponse
from w2v_common.stages.errors import StageRetriesExhaustedError

from src.clients.schemas.video_analyzer import VideoAnalyzerAudioGrid
from src.services.runs.run_context import RunContext
from src.services.stages.action_reel import music_select as music_select_module
from src.services.stages.action_reel.music_override import read_music_override
from src.services.stages.action_reel.music_select import MusicSelectInput, MusicSelectStage
from src.services.stages.base import StagePreconditionError, StageSkippedError
from src.services.stages.llm_budget import llm_retries
from src.services.stages.sound_catalog import SoundLibrary

# A real ``POST /api/analyze/audio`` body for the ``corporate_uplift`` bed, taken
# verbatim (beats/downbeats/onsets truncated for readability — the VALUES and,
# far more importantly, the TYPES are the live ones).
#
# ⚠️ ``meter`` is an ``int`` here on purpose. The analyzer declares
# ``meter: int`` ("beats per bar used for the downbeat inference") in its own
# ``src/api/schemas.py``, so ``4`` is what comes off the wire — never ``"4"``,
# never ``"4/4"``. That single character is the whole incident.
_LIVE_ANALYZER_BODY: dict[str, Any] = {
    "duration_s": 58.0,
    "bpm": 132.512,
    "bpm_confidence": 0.942,
    "beats": [0.09288, 0.545669, 0.986848, 1.428027, 1.880816, 2.321995, 2.774785],
    "downbeats": [0.986848, 2.774785, 4.562721, 6.362268],
    "onsets": [0.08127, 0.301859, 0.754649, 0.975238],
    "meter": 4,
    "grid_source": "librosa.beat_track+onset_phase",
    # The analyzer is separately deployed and additive: an unknown extra key must
    # ride along harmlessly rather than becoming a ValidationError storm.
    "notes": [],
}


def _ctx(
    brief: dict[str, Any] | None = None,
    *,
    grounding_text: str | None = None,
    music_override: dict[str, Any] | None = None,
) -> RunContext:
    snapshot: dict[str, Any] = {"brief": brief or {"title": "Kite session", "tone": "high energy"}}
    if music_override is not None:
        snapshot["music"] = music_override
    return RunContext(
        run_id=uuid7(),
        deliverable_id=uuid7(),
        config_snapshot=snapshot,
        prior_run_outputs=(
            {"source_ingest": {"grounding_text": grounding_text}} if grounding_text else {}
        ),
    )


def _fake_wm(track_id: str = "corporate_uplift", rationale: str = "driving pulse") -> AsyncMock:
    fake = AsyncMock()
    fake.message.return_value = WMResponse(
        output={"track_id": track_id, "rationale": rationale},
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )
    return fake


def _analyzer_returning(body: dict[str, Any]) -> AsyncMock:
    """An analyzer client that validates the WIRE BODY the way the real one does.

    ``VideoAnalyzerClient.analyze_audio`` ends in
    ``VideoAnalyzerAudioGrid.model_validate(result)``, so handing the fake a
    pre-built model would test nothing about the contract. This runs the same
    validation the live client runs, on the same JSON.
    """
    fake = AsyncMock()
    fake.analyze_audio.side_effect = lambda **_kwargs: VideoAnalyzerAudioGrid.model_validate(body)
    return fake


def _use_builtin_library(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake(*_args: object, **_kwargs: object) -> SoundLibrary:
        return SoundLibrary.builtin()

    monkeypatch.setattr(music_select_module, "load_sound_library", _fake)


def _use_empty_library(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake(*_args: object, **_kwargs: object) -> SoundLibrary:
        return SoundLibrary({})

    monkeypatch.setattr(music_select_module, "load_sound_library", _fake)


def _fake_presign(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _presign(sfx_id: str) -> str:
        return f"https://minio.test/w2v-orchestrator/sound_library/{sfx_id}.mp3?sig=x"

    monkeypatch.setattr(music_select_module, "presign_sound_asset", _presign)


# --- 1. THE CONTRACT -------------------------------------------------------


async def test_a_real_analyzer_body_parses_and_becomes_a_non_empty_beat_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE regression. Every field of a live analyzer response, at its live TYPE.

    If any field here is narrowed again — ``meter`` back to a strict ``str``,
    ``bpm`` to an ``int``, ``beats`` to anything that rejects floats — this fails
    at ``model_validate`` instead of shipping a reel cut to nothing. Asserting the
    stage OUTPUT as well as the model is deliberate: the degrade path swallows a
    ValidationError, so a model assertion alone could pass while the stage still
    silently emitted ``beats: []``.
    """
    _use_builtin_library(monkeypatch)
    _fake_presign(monkeypatch)
    analyzer = _analyzer_returning(_LIVE_ANALYZER_BODY)

    out = await MusicSelectStage(client=_fake_wm(), analyzer=analyzer).handle(
        MusicSelectInput(), _ctx()
    )

    # The one that killed the product: an int on the wire must be ACCEPTED, and
    # kept as the string this stage's own output contract promises.
    assert out.meter == "4", "the analyzer sends meter as an int; a strict str rejects EVERY grid"
    # ...and the payload the product hangs on actually arrived.
    assert out.beats == _LIVE_ANALYZER_BODY["beats"]
    assert out.beats, "an empty beat grid IS the silent failure this test exists for"
    assert out.downbeats == _LIVE_ANALYZER_BODY["downbeats"]
    assert out.onsets == _LIVE_ANALYZER_BODY["onsets"]
    assert out.bpm == 132.512
    assert out.bpm_confidence == 0.942
    assert out.duration_s == 58.0
    assert out.grid_source == "librosa.beat_track+onset_phase"
    # A grid that parsed is not a degraded run: no note may be raised for it.
    assert out.notes == []


def test_the_grid_model_accepts_every_field_at_the_type_the_analyzer_sends() -> None:
    """The same contract without the stage around it, so a failure names the
    field rather than the symptom. This is the assertion a future narrowing of
    ``video_analyzer.py`` has to get past."""
    grid = VideoAnalyzerAudioGrid.model_validate(_LIVE_ANALYZER_BODY)

    assert grid.meter == "4"
    assert grid.beats == _LIVE_ANALYZER_BODY["beats"]
    assert grid.bpm == 132.512
    assert grid.bpm_confidence == 0.942
    assert grid.duration_s == 58.0
    assert grid.grid_source == "librosa.beat_track+onset_phase"


# --- 2. the library is authoritative ---------------------------------------


async def test_a_track_the_library_does_not_have_is_refused_not_passed_downstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model is advisory. An invented id must never reach ``compile``, which
    would fail at render time on a key that cannot be presigned — so the stage
    re-prompts inside its retry budget and then fails, rather than substituting a
    track nobody chose."""
    _use_builtin_library(monkeypatch)
    _fake_presign(monkeypatch)
    wm = _fake_wm(track_id="epic_trailer_hero_2")
    analyzer = _analyzer_returning(_LIVE_ANALYZER_BODY)

    with pytest.raises(StageRetriesExhaustedError):
        await MusicSelectStage(client=wm, analyzer=analyzer).handle(MusicSelectInput(), _ctx())

    # It re-prompted rather than giving up on the first bad draw — the whole
    # configured budget, read from the same accessor the stage reads.
    assert wm.message.await_count == llm_retries() + 1
    # ...and nothing downstream was touched with the invented id.
    analyzer.analyze_audio.assert_not_awaited()


# --- 3. an authored pin is not reconsidered --------------------------------


async def test_a_pinned_track_bypasses_the_model_entirely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit human choice must never be "reconsidered" by a model — and the
    proof is that the LLM is not called AT ALL, not merely that its answer was
    discarded (which would still burn a paid call and a minute of wall clock)."""
    _use_builtin_library(monkeypatch)
    _fake_presign(monkeypatch)
    wm = _fake_wm(track_id="corporate_uplift")

    out = await MusicSelectStage(
        client=wm, analyzer=_analyzer_returning(_LIVE_ANALYZER_BODY)
    ).handle(MusicSelectInput(), _ctx(brief={"music": {"track_id": "jazz_lounge"}}))

    wm.message.assert_not_awaited()
    assert out.track_id == "jazz_lounge"
    assert out.rationale == "pinned by the brief"
    assert out.track_key == "sound_library/shared/music/jazz_lounge.mp3"
    # The pin still gets its grid — bypassing the model is not bypassing the beat.
    assert out.beats == _LIVE_ANALYZER_BODY["beats"]


# --- 4. an empty library is a FAILURE, not a skip ---------------------------


async def test_an_empty_sound_library_is_a_precondition_failure_not_a_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ "Skip-clean is deliberately NOT offered here" (module docstring): a reel cut
    to a beat has no meaningful "no music" mode, so a library with no beds must
    fail the run loud. A ``StageSkippedError`` here would let a silent, musicless
    reel count as a completed one."""
    _use_empty_library(monkeypatch)
    _fake_presign(monkeypatch)
    wm = _fake_wm()

    with pytest.raises(StagePreconditionError) as caught:
        await MusicSelectStage(client=wm, analyzer=_analyzer_returning(_LIVE_ANALYZER_BODY)).handle(
            MusicSelectInput(), _ctx()
        )

    assert not isinstance(caught.value, StageSkippedError)
    assert caught.value.missing == ["sound_library.music"]
    # It failed BEFORE spending anything.
    wm.message.assert_not_awaited()


# --- 5. a lost analyzer degrades, and says so -------------------------------


async def test_a_failed_analyzer_call_degrades_with_a_visible_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Losing the analyzer must not throw away an approved selection — but the
    degradation is exactly the thing that hid for months, so the note is the
    load-bearing half of this behaviour, not a nicety. ``beats: []`` alone is
    indistinguishable from a bed with no pulse."""
    _use_builtin_library(monkeypatch)
    _fake_presign(monkeypatch)
    analyzer = AsyncMock()
    analyzer.analyze_audio.side_effect = httpx.ConnectError("analyzer unreachable")

    out = await MusicSelectStage(client=_fake_wm(), analyzer=analyzer).handle(
        MusicSelectInput(), _ctx()
    )

    # The selection survives...
    assert out.track_id == "corporate_uplift"
    assert out.track_url.endswith("corporate_uplift.mp3?sig=x")
    # ...the grid does not...
    assert out.beats == []
    assert out.bpm == 0.0
    assert out.grid_source == ""
    # ...and the reviewer is TOLD, in the run's own output.
    assert any("beat-grid analysis failed" in note for note in out.notes)
    assert any("cut_plan will space evenly" in note for note in out.notes)


async def test_a_bed_that_cannot_be_presigned_also_leaves_a_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other route to an empty grid. It must not be silent either — and it
    must be distinguishable from an analyzer failure, since the fix is different
    (a missing object vs. a service down)."""
    _use_builtin_library(monkeypatch)

    async def _no_url(_sfx_id: str) -> str | None:
        return None

    monkeypatch.setattr(music_select_module, "presign_sound_asset", _no_url)
    analyzer = _analyzer_returning(_LIVE_ANALYZER_BODY)

    out = await MusicSelectStage(client=_fake_wm(), analyzer=analyzer).handle(
        MusicSelectInput(), _ctx()
    )

    analyzer.analyze_audio.assert_not_awaited()
    assert out.beats == []
    assert any("could not be presigned" in note for note in out.notes)


# --- 6. THE OPERATOR'S GROUNDING TEXT --------------------------------------


async def test_the_operator_grounding_text_reaches_the_mood_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``source_ingest`` DAG edge is justified — in this stage's docstring AND
    in the blueprint comment beside the edge — by making the operator's grounding
    text available to the mood prompt. It was never read: the prompt was built
    from four ``brief`` keys and nothing else, so a PAID KAG fan-out was computed
    and thrown away on every run."""
    _use_builtin_library(monkeypatch)
    _fake_presign(monkeypatch)
    fake = _fake_wm()

    await MusicSelectStage(client=fake, analyzer=_analyzer_returning(_LIVE_ANALYZER_BODY)).handle(
        MusicSelectInput(),
        _ctx(grounding_text="Big-air kiteboarding at Cape Hatteras, 35-knot squalls."),
    )

    message = fake.message.await_args.kwargs["message"]
    assert "Cape Hatteras" in message
    # Before the beds, not after them: it is context for the choice.
    assert message.index("Cape Hatteras") < message.index("ALLOWED MUSIC BEDS")


async def test_the_grounding_paste_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unbounded grounding text inflates the product's slowest call — and this
    prompt already joins the WHOLE music catalogue, against a model that charges
    its thinking to the same ``max_tokens`` the answer comes out of. The failure
    of an over-large preamble here is a 422 before any JSON, not a worse track."""
    _use_builtin_library(monkeypatch)
    _fake_presign(monkeypatch)
    fake = _fake_wm()

    await MusicSelectStage(client=fake, analyzer=_analyzer_returning(_LIVE_ANALYZER_BODY)).handle(
        MusicSelectInput(), _ctx(grounding_text="x" * 100_000)
    )

    message = fake.message.await_args.kwargs["message"]
    assert message.count("x") == music_select_module._MUSIC_GROUNDING_MAX_CHARS


async def test_a_run_with_no_ingested_sources_sends_the_message_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A brief with no ``sources`` must send byte-for-byte what this stage always
    sent — the grounding helper returns "" and nothing is interpolated."""
    _use_builtin_library(monkeypatch)
    _fake_presign(monkeypatch)
    fake = _fake_wm()
    stage = MusicSelectStage(client=fake, analyzer=_analyzer_returning(_LIVE_ANALYZER_BODY))

    await stage.handle(MusicSelectInput(), _ctx())

    library = SoundLibrary.builtin()
    assert fake.message.await_args.kwargs["message"] == stage._build_message(
        {"title": "Kite session", "tone": "high energy"}, library, library.ids_in("music")
    )


async def test_the_model_is_told_not_to_think_before_it_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same fence as ``segment_select``: reasoning ON burns the whole per-call
    budget and returns nothing on this cluster's vLLM, and a stubbed client in a
    unit test answers identically with or without the keyword — so without this
    assertion the flag could be dropped and every test here would stay green.
    """
    _use_builtin_library(monkeypatch)
    _fake_presign(monkeypatch)
    fake = _fake_wm()
    analyzer = _analyzer_returning(_LIVE_ANALYZER_BODY)

    await MusicSelectStage(client=fake, analyzer=analyzer).handle(MusicSelectInput(), _ctx())

    assert fake.message.await_args.kwargs["enable_reasoning"] is False


# --- 5. a studio music edit ------------------------------------------------
#
# The defect these pin: a reviewer submits a music change, gets a 201 and a green
# child run with the IDENTICAL soundtrack. The override used to be written to
# `config_snapshot["sound_effects"]`, which no action_reel stage reads at all
# (`ActionReelCompileStage` overrides `build_audio_bed`, bypassing the shared
# `sound_design.resolve`). It now lands on the `music` block THIS stage reads —
# and it has to be this stage, because the cut is timed to the bed.


async def test_a_music_edit_picks_the_track_and_the_grid_is_fetched_for_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point: the reviewer's track is the one the run cuts to.

    The LLM must not be consulted at all (an explicit human choice is not
    reconsidered), and the analyzer must be asked for THAT track's grid — the
    grid is what `cut_plan` snaps every shot boundary to, so a stage that
    returned the new id with the old grid would still be the original bug.
    """
    _use_builtin_library(monkeypatch)
    _fake_presign(monkeypatch)
    wm = _fake_wm(track_id="corporate_uplift")
    analyzer = _analyzer_returning(_LIVE_ANALYZER_BODY)

    out = await MusicSelectStage(client=wm, analyzer=analyzer).handle(
        MusicSelectInput(),
        _ctx(music_override={"enabled": True, "track_id": "jazz_lounge"}),
    )

    assert out.track_id == "jazz_lounge", "the edit's track must win over the model's"
    wm.message.assert_not_awaited()
    assert "jazz_lounge" in analyzer.analyze_audio.await_args.kwargs["audio_url"], (
        "the grid must be fetched for the NEW bed; the old grid is what the cut was "
        "snapped to and re-using it is the original defect"
    )
    assert out.beats, "no grid means the child reel is not beat-cut at all"


async def test_a_music_edit_naming_an_unknown_bed_is_noted_not_obeyed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same rule as the brief's pin: the library is authoritative. An id the
    compiler cannot presign would fail at RENDER time, which is the expensive
    place to find out."""
    _use_builtin_library(monkeypatch)
    _fake_presign(monkeypatch)

    out = await MusicSelectStage(
        client=_fake_wm(track_id="corporate_uplift"),
        analyzer=_analyzer_returning(_LIVE_ANALYZER_BODY),
    ).handle(
        MusicSelectInput(), _ctx(music_override={"enabled": True, "track_id": "not_a_real_bed"})
    )

    assert out.track_id == "corporate_uplift"
    assert any("not_a_real_bed" in note for note in out.notes), out.notes


async def test_music_off_is_honoured_and_said_out_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    """ "Turn the music off" used to be a no-op that shipped the same soundtrack.

    It is honoured now — no track, no grid — and it is NOTED, because a beat-cut
    reel with no beat is the one degradation nobody can spot in a green run
    summary. The empty `track_id` additionally trips the central `produces` gate,
    so the run reads `degraded`.
    """
    _use_builtin_library(monkeypatch)
    _fake_presign(monkeypatch)
    wm = _fake_wm()
    analyzer = _analyzer_returning(_LIVE_ANALYZER_BODY)

    out = await MusicSelectStage(client=wm, analyzer=analyzer).handle(
        MusicSelectInput(), _ctx(music_override={"enabled": False})
    )

    assert out.track_id == ""
    assert out.track_url == ""
    assert out.beats == []
    assert any("MUSIC OFF" in note and "SILENT" in note for note in out.notes), out.notes
    # Neither the model nor the analyzer is paid for a bed nobody wants.
    wm.message.assert_not_awaited()
    analyzer.analyze_audio.assert_not_awaited()


async def test_no_music_edit_leaves_the_stage_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The override is presence-opt-in: a run without one must reach the model and
    the analyzer exactly as before."""
    _use_builtin_library(monkeypatch)
    _fake_presign(monkeypatch)
    wm = _fake_wm(track_id="corporate_uplift")

    out = await MusicSelectStage(
        client=wm, analyzer=_analyzer_returning(_LIVE_ANALYZER_BODY)
    ).handle(MusicSelectInput(), _ctx())

    assert out.track_id == "corporate_uplift"
    assert out.notes == []
    wm.message.assert_awaited()


# --- the override block's own parsing ----------------------------------------
#
# `read_music_override` returns a MODEL, not a dict. These pin the two guards
# that a mutation proved were UNCOVERED when the parsing moved into it: the
# non-finite filter (previously `compile._mix_knob`, which no test exercised
# directly) and per-field rather than all-or-nothing validation.


def test_a_junk_mix_knob_does_not_discard_the_valid_track() -> None:
    """MUTANT: `except ValidationError: return MusicOverride()` (whole-block).

    `config_overrides` is unvalidated at the engine boundary, so one bad knob in
    an otherwise good edit is realistic. Throwing the whole block away would
    silently revert the operator's ACTUAL choice — the track — because of a knob,
    which is a worse lie than the dropped knob.
    """
    override = read_music_override(
        {"music": {"track_id": "jazz_lounge", "enabled": True, "gain_db": "loud"}}
    )

    assert override.track_id == "jazz_lounge"
    assert override.enabled is True
    assert override.gain_db is None


@pytest.mark.parametrize("bad", [float("inf")])
def test_a_non_finite_mix_knob_reads_as_unauthored(bad: float) -> None:
    """MUTANT: drop the `math.isfinite` check in `MusicOverride.mix_knob`.

    LOAD-BEARING: pydantic accepts nan/inf for a `float` field by default, and
    either interpolated into the compiler's filtergraph 422s the whole render.
    This is the last place to catch it, and the guard moved here from
    `compile._mix_knob` — where no test exercised it either.
    """
    override = read_music_override({"music": {"gain_db": bad, "fade_in_s": bad}})

    assert override.mix_knob("gain_db", -14.0) == -14.0
    assert override.mix_knob("fade_in_s", 1.0) == 1.0


def test_a_finite_authored_knob_is_still_honoured() -> None:
    """The guard must not swallow the values it exists to let through."""
    override = read_music_override({"music": {"gain_db": -6.0}})

    assert override.mix_knob("gain_db", -14.0) == -6.0


def test_a_block_that_is_not_a_dict_reads_as_no_override() -> None:
    """MUTANT: drop the `isinstance(raw, dict)` branch. A stage must never raise
    on a surprising snapshot."""
    assert read_music_override({"music": "nope"}).track_id is None
    assert read_music_override({}).track_id is None


# --- P6 ORCH-4: a tenant track deleted mid-run (RETIRED) --------------------


def _library_with_retired_tenant_track(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services.stages.sound_catalog import BUILTIN_CATALOG, SoundAsset

    retired = SoundAsset(
        key="sound_library/tenants/t1/music/x.mp3",
        description="our anthem",
        category="music",
        duration_s=30.0,
        source="tenant",
        retired=True,
    )

    async def _fake(*_args: object, **_kwargs: object) -> SoundLibrary:
        return SoundLibrary({**BUILTIN_CATALOG, "tenant_anthem": retired})

    monkeypatch.setattr(music_select_module, "load_sound_library", _fake)


async def test_a_pinned_track_retired_mid_run_still_plays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run pinned the tenant track before the tenant deleted it; the delete must
    not drop this run's bed."""
    _library_with_retired_tenant_track(monkeypatch)
    _fake_presign(monkeypatch)
    wm = _fake_wm()
    out = await MusicSelectStage(
        client=wm, analyzer=_analyzer_returning(_LIVE_ANALYZER_BODY)
    ).handle(MusicSelectInput(), _ctx(brief={"music": {"track_id": "tenant_anthem"}}))
    wm.message.assert_not_awaited()
    assert out.track_id == "tenant_anthem"
    assert out.track_key == "sound_library/tenants/t1/music/x.mp3"


async def test_the_model_can_never_pick_a_retired_track(monkeypatch: pytest.MonkeyPatch) -> None:
    _library_with_retired_tenant_track(monkeypatch)
    _fake_presign(monkeypatch)
    wm = _fake_wm(track_id="tenant_anthem")
    with pytest.raises(StageRetriesExhaustedError):
        await MusicSelectStage(client=wm, analyzer=_analyzer_returning(_LIVE_ANALYZER_BODY)).handle(
            MusicSelectInput(), _ctx()
        )
    sent = wm.message.await_args.kwargs["message"]
    assert "tenant_anthem" not in sent, "a retired track is not offered to the model"
