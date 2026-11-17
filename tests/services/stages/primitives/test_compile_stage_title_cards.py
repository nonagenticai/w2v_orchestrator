"""Who gets a SPLICED title card, and who declines it.

``CompileStageBase`` splices the opt-in intro/outro card in as a free-standing
shot at the front of the list, and NOTHING rebases the timeline afterwards. For
a beat-cut product that is fatal and invisible: the music bed is request-level
with no start/offset field (bed t=0 IS video t=0), so a 3 s card — the default —
pushes every solved cut 3 s off the beat, the compiler raises nothing, and the
duration auto-QA passes because declared and actual totals both include the card.

So there are exactly two properties to hold, and the second is the one that
guards everyone else:

* ``action_reel`` never splices, and SAYS SO when the brief asked for a card
  (its title rides as a shot-local ``graphics_design`` overlay, which cannot
  move a cut);
* every OTHER product still splices exactly as before — the hook's default is
  ``True``, and a base that stopped splicing would silently strip the intro from
  promo/demo_video/lesson.

Both are asserted on the shots that reach the COMPILE CALL, never on the hook's
return value: a hook that answers correctly and is never consulted looks
identical to a green run.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid7

import pytest

from src.services import tts_storage
from src.services.runs.run_context import RunContext
from src.services.stages.action_reel.compile import (
    ActionReelCompileInput,
    ActionReelCompileStage,
)
from src.services.stages.base import (
    ArtifactUrl,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
)
from src.services.stages.primitives.compile_stage import CompileStageBase
from src.services.stages.shared.degradation import TITLE_CARD_DROPPED, UNCLASSIFIED_NOTE

_URL = "https://example.invalid/drone-a.mp4"
_INTRO_ID = "intro-card-shot"
_OUTRO_ID = "outro-card-shot"


class _FakeStorage:
    """Enough storage for the compile tail's re-sign pass."""

    async def resign_presigned_url(self, url: str) -> str:
        return url


class _Input(StageInput):
    pass


class _Output(StageOutput):
    artifact_url: ArtifactUrl
    duration_seconds: float


class _Stage(CompileStageBase[_Input, _Output]):
    """A bare non-action_reel product: the base's own behaviour, nothing else."""

    name = "test_compile"
    input_model = _Input
    output_model = _Output

    def build_shots(self, ctx: Any) -> list[dict[str, Any]]:
        return [{"id": "s1", "line_id": "l1", "seconds": 1.0}]

    def build_output(self, result: dict[str, Any], ctx: Any) -> _Output:
        return _Output(
            artifact_url=str(result["artifact_url"]),
            duration_seconds=float(result["duration_seconds"]),
        )

    def compile_tags(self, ctx: Any) -> dict[str, str]:
        return {}  # keep this unit off ctx.config_snapshot's deliverable block


def _capturing_client(captured: dict[str, Any]) -> AsyncMock:
    """A video-compile client that records the submitted kwargs."""

    async def _compile(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "artifact_url": "http://m/w2v-compiled-videos/out.mp4",
            "duration_seconds": 2.0,
            # The PROBED encoded length, for the same reason `mastering` is here:
            # a real compiler response always carries it (`probe_output_duration`
            # runs on every render), and an absent/0.0 one is now its own finding
            # (unmeasured, not verified), so a fake without this key degrades
            # every compile in this file over something unrelated to title cards.
            "actual_duration_seconds": 2.0,
            # A PASSING mastering report, because a real compiler response carries
            # one: an ABSENT report is now its own finding (the gate could not run
            # ⇒ unmeasured, not passed — `shared/mastering_gate.py`), so a fake
            # without this key degrades every compile in this file for a reason
            # that has nothing to do with title cards.
            "mastering": {"rerun_stages": [], "gate_pass": True, "luma_spread": 210.0},
        }

    client = AsyncMock()
    client.compile = AsyncMock(side_effect=_compile)
    return client


def _stub_title_cards(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    """Replace the card RENDERER with a recorder that always yields both cards.

    Deliberately unconditional (it does not re-read ``title_card.enabled``): the
    property under test is whether the base REACHES the renderer at all, so a
    stub that could itself decline would hide a hook that never fired.
    """

    async def _fake(ctx: Any, *_a: Any, **_kw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        calls.append(str(ctx.run_id))
        return (
            {"id": _INTRO_ID, "line_id": "intro", "seconds": 3.0, "video_url": _URL},
            {"id": _OUTRO_ID, "line_id": "outro", "seconds": 3.0, "video_url": _URL},
        )

    monkeypatch.setattr(
        "src.services.stages.primitives.compile_stage.render_title_card_plan", _fake
    )


async def _run(stage: CompileStageBase[Any, Any], payload: Any, ctx: Any) -> Any:
    """Drive the compile tail with the storage singleton stubbed out."""
    tts_storage.set_storage_for_tests(cast(Any, _FakeStorage()))
    try:
        return await stage.handle(payload, ctx=ctx)
    finally:
        tts_storage.set_storage_for_tests(None)


def _reel_ctx(*, title_card: dict[str, Any] | None) -> RunContext:
    """A reel run whose brief did (or did not) ask for a title card."""
    snapshot: dict[str, Any] = {"title": "Ridgeline"}
    if title_card is not None:
        snapshot["title_card"] = title_card
    return RunContext(
        run_id=uuid7(),
        deliverable_id=uuid7(),
        config_snapshot=snapshot,
        prior_run_outputs={
            "footage_probe": {"sources": [{"id": "src-0", "fps": 24.0}]},
            "cut_plan": {
                "shots": [
                    {
                        "id": "s0",
                        "source_id": "src-0",
                        "video_url": _URL,
                        "in_seconds": 1.0,
                        "out_seconds": 3.0,
                        "seconds": 2.0,
                        "speed": 1.0,
                    }
                ],
                "total_seconds": 2.0,
                "music_track_id": "",
                "bpm": 120.0,
                "beat_grid_available": True,
            },
            "graphics_design": {"cards": []},
        },
    )


# --- action_reel: the card is DECLINED, and said out loud --------------------


@pytest.mark.asyncio
async def test_an_enabled_title_card_never_reaches_the_reels_shot_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE fix. A front-spliced 3 s card moves every beat-locked cut 3 s late,
    and nothing downstream can see that it happened."""
    calls: list[str] = []
    _stub_title_cards(monkeypatch, calls)
    captured: dict[str, Any] = {}
    stage = ActionReelCompileStage(_capturing_client(captured))

    output = await _run(
        stage,
        ActionReelCompileInput(),
        _reel_ctx(title_card={"enabled": True, "duration_seconds": 3.0}),
    )

    ids = [str(shot.get("id")) for shot in captured["shots"]]
    assert ids == ["s0"], f"a spliced card reached the reel: {ids}"
    assert calls == [], "the card was rendered (and paid for) before being dropped"
    # And the reviewer is TOLD: an authored card silently absent from the
    # delivered reel is the degradation the notes channel exists to prevent.
    assert any("spliced title card" in note for note in output.notes), output.notes


@pytest.mark.asyncio
async def test_a_reel_that_asked_for_no_card_writes_no_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declining what nobody requested is not a degradation — a note here would
    flag every normal reel as ``degraded``."""
    calls: list[str] = []
    _stub_title_cards(monkeypatch, calls)
    captured: dict[str, Any] = {}
    stage = ActionReelCompileStage(_capturing_client(captured))

    output = await _run(stage, ActionReelCompileInput(), _reel_ctx(title_card=None))

    assert [str(shot.get("id")) for shot in captured["shots"]] == ["s0"]
    assert output.notes == []


# --- the base: unchanged for every other product -----------------------------


@pytest.mark.asyncio
async def test_every_other_product_still_splices_its_title_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE regression fence for the shared primitive. ``title_cards_enabled``
    defaults to True, so promo/demo_video/lesson keep the intro and outro they
    have always had — an opt-out that leaked into the base would strip the card
    from every product with no test failing anywhere else."""
    calls: list[str] = []
    _stub_title_cards(monkeypatch, calls)
    captured: dict[str, Any] = {}
    stage = _Stage(_capturing_client(captured))

    await _run(
        stage,
        _Input(),
        SimpleNamespace(
            run_id=uuid7(),
            render_idempotency_discriminator="",
            config_snapshot={"title_card": {"enabled": True}},
            prior_output=lambda _stage: None,
        ),
    )

    ids = [str(shot.get("id")) for shot in captured["shots"]]
    assert ids == [_INTRO_ID, "s1", _OUTRO_ID], f"the base stopped splicing: {ids}"
    assert len(calls) == 1


# --- a DROPPED card leaves a trace in the PERSISTED output -------------------


class _NotesOutput(StageOutputWithNotes):
    """A final-video compile output: same shape as ``_Output`` plus the shared
    ``notes`` channel every splicing product's real output carries."""

    artifact_url: ArtifactUrl
    duration_seconds: float


class _NotesStage(CompileStageBase[_Input, _NotesOutput]):
    """``_Stage`` with a notes-carrying output — i.e. the promo/demo_video shape."""

    name = "test_compile_notes"
    input_model = _Input
    output_model = _NotesOutput

    def build_shots(self, ctx: Any) -> list[dict[str, Any]]:
        return [{"id": "s1", "line_id": "l1", "seconds": 1.0}]

    def build_output(self, result: dict[str, Any], ctx: Any) -> _NotesOutput:
        return _NotesOutput(
            artifact_url=str(result["artifact_url"]),
            duration_seconds=float(result["duration_seconds"]),
        )

    def compile_tags(self, ctx: Any) -> dict[str, str]:
        return {}


@pytest.mark.asyncio
async def test_a_dropped_title_card_reaches_the_persisted_stage_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE wiring. ``build_title_card_shots`` soft-fails a failed card render to
    ``None`` — the compile still runs, 30 s shorter, and before this the ONLY
    record of why was a ``log.warning`` that dies with the pod. That is what made
    the 2026-09-03 promo (66.725 s -> 6.725 s, exactly -60.000 s of cards) take
    three sessions to attribute.

    Asserted on the RETURNED OUTPUT, not on the notes sink and not on a call
    count: the sink is filled by ``title_cards.py``, which already worked. What is
    under test is that ``_run_presubmit`` carries it out and ``handle`` folds it
    into the output that gets PERSISTED — delete either and the sink still fills
    while the reviewer still learns nothing.
    """
    boom = RuntimeError("video_production 503: animated_title_card render failed")

    class _FailingVideoProduction:
        async def record_animation(self, **_kw: Any) -> str:
            raise boom

    # Patched at the RENDER seam inside ``title_cards`` (the client it constructs),
    # never at ``build_title_card_shots`` itself — stubbing the function would skip
    # the very soft-fail/append path this test exists to cover.
    monkeypatch.setattr(
        "src.services.stages.shared.title_cards.VideoProductionClient",
        lambda *_a, **_kw: _FailingVideoProduction(),
    )
    captured: dict[str, Any] = {}
    stage = _NotesStage(_capturing_client(captured))

    output = await _run(
        stage,
        _Input(),
        SimpleNamespace(
            run_id=uuid7(),
            render_idempotency_discriminator="",
            config_snapshot={
                "title": "Ridgeline",
                "title_card": {"enabled": True, "intro": True, "outro": False},
            },
            prior_output=lambda _stage: None,
        ),
    )

    # The card really was dropped: the compile ran WITHOUT it (soft-fail intact).
    assert [str(shot.get("id")) for shot in captured["shots"]] == ["s1"]
    # ...and the reason survived into the output the run persists, naming the card
    # part and the exception detail so the next reader can classify infra vs render.
    dropped = [note for note in output.notes if "title-intro" in note]
    assert dropped, f"the drop reason never reached the stage output: {output.notes}"
    assert "RuntimeError" in dropped[0] and str(boom) in dropped[0], dropped[0]


@pytest.mark.asyncio
async def test_a_dropped_title_card_records_a_structured_code_and_run_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(d) Defect 3 mutation-proof, end-to-end through ``handle()``: a dropped
    title card must be CLASSIFIED (``TITLE_CARD_DROPPED``), never generic
    (``unclassified_note``), and — the never-raise contract this whole feature
    is built on — the stage must still COMPLETE with a valid output.

    Before the fix, a dropped card appended ONLY to ``notes``: with
    ``degradations`` empty, ``dag_executor._enforce_degradation_codes`` (the
    central stamping logic in ``shared/degradation.py::ensure_degradation_codes``)
    would have stamped the generic ``unclassified_note`` code — indistinguishable
    from any other soft-failed stage note.

    MUTATION PROOF: reverting the ``degradations.append(degradation(...))`` call
    in ``title_cards._render``'s except clause turns this red (``output.
    degradations`` comes back empty, so ``ensure_degradation_codes`` would stamp
    ``unclassified_note`` instead of the real code).
    """
    boom = RuntimeError("video_production 503: animated_title_card render failed")

    class _FailingVideoProduction:
        async def record_animation(self, **_kw: Any) -> str:
            raise boom

    monkeypatch.setattr(
        "src.services.stages.shared.title_cards.VideoProductionClient",
        lambda *_a, **_kw: _FailingVideoProduction(),
    )
    captured: dict[str, Any] = {}
    stage = _NotesStage(_capturing_client(captured))

    # The stage must not raise: a dropped title card is a soft-fail, never a
    # hard failure of the run.
    output = await _run(
        stage,
        _Input(),
        SimpleNamespace(
            run_id=uuid7(),
            render_idempotency_discriminator="",
            config_snapshot={
                "title": "Ridgeline",
                "title_card": {"enabled": True, "intro": True, "outro": True},
            },
            prior_output=lambda _stage: None,
        ),
    )

    # Run completed: a real artifact was produced despite the dropped cards.
    assert output.artifact_url
    # Structured, queryable — not the generic catch-all.
    codes = [d["code"] for d in output.degradations]
    assert codes == [TITLE_CARD_DROPPED, TITLE_CARD_DROPPED], output.degradations
    assert UNCLASSIFIED_NOTE not in codes
    for d in output.degradations:
        assert "RuntimeError" in d["detail"]
        assert str(boom) in d["detail"]
        assert "render_error" in d["detail"]  # a content/render error, not a timeout


@pytest.mark.asyncio
async def test_a_planned_card_with_no_title_text_is_a_coded_drop_with_its_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PR #1078 audit, defect 3: a PLANNED card that is not rendered must be recorded.

    ``reconcile_timeline`` reserves the cards' seconds out of the order, so a card the
    compile silently skips ships a film that much short (~-20% at the 20% card cap) with
    no coded reason. The blank-title skip used to ``return None`` with no record at all.
    Here the run has no project title, so the planned intro has nothing to show; the
    outro still renders from its fallback text.

    Before the fix ``output.degradations`` was EMPTY on this run.
    """

    class _UnusedVideoProduction:
        async def record_animation(self, **_kw: Any) -> str:
            raise RuntimeError("outro render down")

    monkeypatch.setattr(
        "src.services.stages.shared.title_cards.VideoProductionClient",
        lambda *_a, **_kw: _UnusedVideoProduction(),
    )
    captured: dict[str, Any] = {}
    stage = _NotesStage(_capturing_client(captured))
    output = await _run(
        stage,
        _Input(),
        SimpleNamespace(
            run_id=uuid7(),
            render_idempotency_discriminator="",
            config_snapshot={
                "title": "",
                "title_card": {"enabled": True, "intro": True, "outro": False},
            },
            prior_output=lambda _stage: None,
        ),
    )

    assert output.artifact_url
    drops = [d for d in output.degradations if d["code"] == TITLE_CARD_DROPPED]
    assert len(drops) == 1, output.degradations
    assert "title-intro" in drops[0]["detail"]
    assert "no_title" in drops[0]["detail"]
    assert "s planned" in drops[0]["detail"]
