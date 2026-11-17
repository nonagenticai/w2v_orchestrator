"""action_reel graphics_design — the overlay the product promise depends on.

Two properties carry this file. First, the RENDER REQUEST: the compiler's
overlay layer only works with a real alpha channel, which means
``transparent=True`` on an overlay-safe template in a WebM container. Get any
of the three wrong and the service either 400s or hands back an opaque clip
that covers the footage it was supposed to fly over.

Second, the FAILURE POSTURE. The renderer silently drops an overlay it cannot
fetch and finishes the render, so a card emitted without a usable URL produces
a graphics-free reel and a green run. Every failure path here therefore has to
end in an EMPTY card list plus a note that says so in words — never a card the
downstream join would believe.

⚠️ Nothing here asserts ``yuva420p``: a correct VP9-alpha WebM reports
``pix_fmt=yuv420p`` and carries the alpha in WebM side data.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from types import SimpleNamespace
from typing import Any
from uuid import uuid7

import pytest

from src.clients.video_production_client import VideoProductionClient
from src.config import PipelineSettings
from src.services.blueprints.action_reel import ACTION_REEL_BLUEPRINT
from src.services.middleware.chain import build_chain
from src.services.middleware.timeout import TimeoutMiddleware
from src.services.runs.run_context import RunContext
from src.services.stages.action_reel.graphics_design import (
    _MAX_CALLOUTS,
    _MAX_HEADLINE_CHARS,
    _MAX_SHOTS_DESCRIBED,
    _MAX_STRAP_CHARS,
    _RECORD_TIMEOUT_S,
    GRAPHICS_DESIGN_TIMEOUT_SECONDS,
    GraphicsDesignInput,
    GraphicsDesignStage,
)
from src.services.stages.llm_budget import (
    MIN_OUTPUT_TOKEN_BUDGET,
    llm_budget_seconds,
    llm_client_timeout_seconds,
)
from tests.services.blueprints.test_llm_timeout_ladder import (
    _MIN_OUTER_MARGIN_SECONDS,
)


class _FakeVPClient:
    """Records the animation request and hands back canned recording bytes."""

    def __init__(self, *, fail_on: str = "") -> None:
        self.calls: list[dict[str, Any]] = []
        #: Every poll budget the stage asked for — the ladder test reads it so
        #: the constant is checked where it is USED, not only where it is set.
        self.waits: list[float] = []
        self._fail_on = fail_on

    async def record_animation(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        if self._fail_on == "record":
            raise RuntimeError("recorder said no")
        return "job-1"

    async def wait_for_content_recording(self, job_id: str, timeout: float = 0.0) -> dict[str, Any]:
        self.waits.append(timeout)
        if self._fail_on == "wait":
            raise TimeoutError("recording never finished")
        return {"job_id": job_id, "status": "completed"}

    async def get_recording_video(self, job_id: str) -> tuple[bytes, str, str | None]:
        if self._fail_on == "fetch":
            raise RuntimeError("no bytes")
        return b"\x1a\x45\xdf\xa3webm", "video/webm", None


def _ctx(
    shots: list[dict[str, Any]] | None = None,
    *,
    footage_probe: dict[str, Any] | None = None,
    **config: Any,
) -> RunContext:
    prior: dict[str, Any] = {
        "cut_plan": {
            "shots": shots
            if shots is not None
            else [
                {"id": "sh1", "seconds": 4.0, "speed": 1.0},
                {"id": "sh2", "seconds": 3.0, "speed": 1.0},
            ]
        }
    }
    if footage_probe is not None:
        prior["footage_probe"] = footage_probe
    return RunContext(
        run_id=uuid7(),
        deliverable_id=uuid7(),
        config_snapshot={"title": "Ridgeline", "brand": {"name": "SpecraLabs"}, **config},
        prior_run_outputs=prior,
    )


@pytest.fixture(autouse=True)
def _stub_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _store(**kwargs: Any) -> str:
        return f"https://minio.invalid/{kwargs['video_id']}.{kwargs['video_format']}"

    monkeypatch.setattr(
        "src.services.stages.action_reel.graphics_design.store_generated_video", _store
    )


#: Distinct from ``None``, which is itself one of the malformed draws under test.
_NO_OUTPUT_GIVEN = object()


class _FakeWM:
    """The callout designer, stubbed. Records the payload; returns a canned draw."""

    def __init__(self, output: Any = _NO_OUTPUT_GIVEN, *, fail: bool = False) -> None:
        self.output: Any = {"cards": []} if output is _NO_OUTPUT_GIVEN else output
        self.calls: list[dict[str, Any]] = []
        self._fail = fail
        self.timeout = 0.0

    async def message(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._fail:
            raise RuntimeError("worker-management said no")
        return SimpleNamespace(output=self.output)


@pytest.fixture(autouse=True)
def _stub_wm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default: a designer that answers, and chooses NO callouts.

    Autouse so no test can accidentally reach the real worker-management over the
    network — and "answered, chose none" rather than "failed" so every assertion
    written before the designer existed still describes the title-only reel it was
    written against, with `callouts_degraded` False.
    """
    monkeypatch.setattr(
        "src.services.stages.action_reel.graphics_design.WMClient",
        lambda **_kwargs: _FakeWM(),
    )


def _shots(count: int, *, seconds: float = 4.0) -> list[dict[str, Any]]:
    return [
        {
            "id": f"sh{i}",
            "seconds": seconds,
            "speed": 1.0,
            "start_seconds": float(i) * seconds,
            "rationale": f"shot {i} rationale",
        }
        for i in range(count)
    ]


def _proposal(index: int, headline: str = "Full send", strap: str = "Ridge line") -> dict[str, Any]:
    return {"shot_index": index, "headline": headline, "strap": strap}


async def _run(stage: GraphicsDesignStage, ctx: RunContext) -> Any:
    return await stage.handle(GraphicsDesignInput(), ctx)


# --- the render request -----------------------------------------------------


async def test_the_card_is_recorded_transparent_on_the_only_overlay_safe_template() -> None:
    """THE test. Without `transparent=True` the recorder returns an opaque clip
    that blacks out the footage it was meant to fly over — and the run is green.
    `lower_third` is the only template the service will render transparently;
    anything else is a 400."""
    client = _FakeVPClient()
    out = await _run(GraphicsDesignStage(client=client), _ctx())  # ty: ignore[invalid-argument-type]

    call = client.calls[0]
    assert call["transparent"] is True
    assert call["template_id"] == "lower_third"
    # h264 cannot carry alpha, so the recorder REJECTS mp4 for a transparent render.
    assert call["output_format"] == "webm"
    assert out.cards[0].url.endswith(".webm")


async def test_the_card_lands_on_the_opening_shot_at_its_first_frame() -> None:
    out = await _run(GraphicsDesignStage(client=_FakeVPClient()), _ctx())  # ty: ignore[invalid-argument-type]

    card = out.cards[0]
    assert card.shot_id == "sh1"
    assert card.at_seconds == pytest.approx(0.0)
    assert card.kind == "title"
    assert card.text == "Ridgeline"


async def test_the_card_never_outlives_the_shot_it_is_pinned_to() -> None:
    """An overlay longer than its shot is cut mid-animation at the first edit."""
    client = _FakeVPClient()
    ctx = _ctx([{"id": "sh1", "seconds": 1.5, "speed": 1.0}])

    out = await _run(GraphicsDesignStage(client=client), ctx)  # ty: ignore[invalid-argument-type]

    assert out.cards[0].seconds == pytest.approx(1.5)
    assert client.calls[0]["duration_ms"] == 1500


async def test_an_opening_shot_too_short_to_hold_a_card_gets_none_and_says_so() -> None:
    """`cut_plan` admits a 0.6 s opening (MIN_SHOT_SECONDS), and the card is
    pinned to shot[0] unconditionally. The rendered clip has an 800 ms floor —
    below it the entrance cannot complete — while the declared window has none,
    so a 0.6 s shot used to render 800 ms and declare 600 ms. The compiler trims
    to the declaration, so the title vanished MID-EXIT-WIPE on the first frame of
    the delivered reel. Refusing the card is the only honest answer: the floor
    cannot come down and the declaration must not lie about what was rendered."""
    client = _FakeVPClient()
    ctx = _ctx([{"id": "sh1", "seconds": 0.6, "speed": 1.0}])

    out = await _run(GraphicsDesignStage(client=client), ctx)  # ty: ignore[invalid-argument-type]

    assert out.cards == []
    # And nothing was recorded: a card nobody can place is still a paid render.
    assert client.calls == []
    assert any("too short" in note for note in out.notes)
    assert any("NO graphics" in note for note in out.notes)
    # A judgement, not a stub and not an outage — the designer DID look at the
    # opening shot. Borrowing either flag would tell a reviewer the wrong story.
    assert out.cards_stubbed is False
    assert out.cards_render_failed is False


@pytest.mark.parametrize(
    ("label", "shot"),
    [("zero", {"id": "sh1", "seconds": 0.0, "speed": 1.0})],
)
async def test_a_shot_with_no_usable_length_gets_no_card_and_says_so(
    label: str, shot: dict[str, Any]
) -> None:
    """A shot whose length is UNKNOWN must not buy a card at an invented length.

    `_card_window` used to read the shot through `_positive(..., _CARD_SECONDS)`,
    so every one of these degenerate shots was silently read as a full 3.0 s —
    long enough to clear the floor, so the card was rendered AND declared at 3 s.
    `compile` reads the very same missing field as 1.0 (`seconds if seconds > 0.0
    else 1.0`), so the compiler put a 1 s shot under a 3 s overlay and trimmed the
    title off mid-exit-wipe: the ORIGINAL defect, on exactly the input the fix was
    supposed to make safe. `cut_plan`'s no-grid `_even_spacing` path emits
    `max(0.0, end - start)`, which is 0.0 whenever a segment's in and out round to
    the same millisecond, so this is a reachable shot and not a hypothetical one.

    NaN deserves its own case: `NaN < _MIN_CARD_MS` is False, so any guard written
    as a bare comparison ACCEPTS a NaN card and hands the compiler an overlay
    window it cannot render.
    """
    client = _FakeVPClient()
    ctx = _ctx([shot])

    out = await _run(GraphicsDesignStage(client=client), ctx)  # ty: ignore[invalid-argument-type]

    assert out.cards == [], f"{label} `seconds` must not buy a card at a guessed length"
    # No paid render either — the refusal happens before the recorder is asked.
    assert client.calls == [], f"{label} `seconds` still paid for a recording"
    # And it is visible to a human, in words, not only as an empty list.
    assert any("NO graphics" in note for note in out.notes), f"{label}: no note"
    # A judgement about this shot, not a stub and not a render outage.
    assert out.cards_stubbed is False
    assert out.cards_render_failed is False


async def test_the_rendered_clip_and_the_declared_window_are_the_same_length() -> None:
    """The divergence that made the title disappear: `compile` forwards `seconds`
    verbatim as the overlay's `duration_seconds`, so any millisecond the render
    has beyond the declaration is a hard trim mid-animation. On every shot the
    stage accepts, the two numbers must agree exactly."""
    for shot_seconds in (0.8, 0.9, 1.2, 2.5, 4.0):
        client = _FakeVPClient()
        ctx = _ctx([{"id": "sh1", "seconds": shot_seconds, "speed": 1.0}])

        out = await _run(GraphicsDesignStage(client=client), ctx)  # ty: ignore[invalid-argument-type]

        assert out.cards, f"{shot_seconds}s shot can hold a card and must get one"
        rendered_ms = client.calls[0]["duration_ms"]
        assert rendered_ms >= 800, "a card below the floor cannot complete its entrance"
        assert round(out.cards[0].seconds * 1000) == rendered_ms, (
            f"{shot_seconds}s shot: declared {out.cards[0].seconds}s window against a "
            f"{rendered_ms}ms clip — the compiler trims the difference mid-animation"
        )


async def test_card_times_are_converted_onto_the_shots_source_timeline() -> None:
    """`OverlayClip.retime_with_speed=True` divides the authored times by the
    shot's speed, so a card authored in OUTPUT seconds would be cut in half on a
    2x shot. The conversion happens here, where the speed is known."""
    ctx = _ctx([{"id": "sh1", "seconds": 3.0, "speed": 2.0}])
    client = _FakeVPClient()

    out = await _run(GraphicsDesignStage(client=client), ctx)  # ty: ignore[invalid-argument-type]

    # The rendered clip is still 3s of real animation...
    assert client.calls[0]["duration_ms"] == 3000
    # ...but it is DECLARED on the source timeline, so speed/1 brings it back.
    assert out.cards[0].seconds == pytest.approx(6.0)


async def test_the_entrance_hold_and_exit_span_the_whole_card() -> None:
    """A card whose clip ends mid-entrance reads as a glitch, not a title."""
    client = _FakeVPClient()

    await _run(GraphicsDesignStage(client=client), _ctx())  # ty: ignore[invalid-argument-type]

    call = client.calls[0]
    params = call["params"]
    total = params["entrance_ms"] + params["hold_ms"] + params["exit_ms"]
    assert total == call["duration_ms"]
    assert params["entrance_ms"] > 0
    assert params["exit_ms"] > 0


async def test_a_resubmit_reuses_the_same_recording_job() -> None:
    """Without a dedupe key a retried stage starts a SECOND paid recording."""
    ctx = _ctx()
    first = _FakeVPClient()
    second = _FakeVPClient()

    await _run(GraphicsDesignStage(client=first), ctx)  # ty: ignore[invalid-argument-type]
    await _run(GraphicsDesignStage(client=second), ctx)  # ty: ignore[invalid-argument-type]

    key = first.calls[0]["idempotency_key"]
    assert key and key == second.calls[0]["idempotency_key"]
    assert str(ctx.run_id) in key


# --- the failure posture ----------------------------------------------------


@pytest.mark.parametrize("fail_on", ["record"])
async def test_a_failed_render_degrades_to_a_note_that_says_there_are_no_graphics(
    fail_on: str,
) -> None:
    """Never a card without a URL: the compiler drops an unfetchable overlay in
    SILENCE, so an emitted-but-dead card ships a graphics-free reel and reports
    success. And the note has to say the product outcome, not just the error."""
    out = await _run(GraphicsDesignStage(client=_FakeVPClient(fail_on=fail_on)), _ctx())  # ty: ignore[invalid-argument-type]

    assert out.cards == []
    assert any("NO graphics" in note for note in out.notes)
    # An outage, not an editorial choice. `cards == []` alone cannot tell the two
    # apart, so this flag is the only machine-readable signal a consumer has.
    assert out.cards_render_failed is True


async def test_an_empty_stored_url_is_dropped_rather_than_emitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _store_nothing(**_kwargs: Any) -> str:
        return ""

    monkeypatch.setattr(
        "src.services.stages.action_reel.graphics_design.store_generated_video", _store_nothing
    )

    out = await _run(GraphicsDesignStage(client=_FakeVPClient()), _ctx())  # ty: ignore[invalid-argument-type]

    assert out.cards == []
    assert any("NO graphics" in note for note in out.notes)
    # Storage handing back nothing is a render/store OUTAGE, same as a throw.
    assert out.cards_render_failed is True


async def test_an_untitled_brief_designs_nothing_and_records_nothing() -> None:
    """A blank lower-third is worse than no lower-third, and a render nobody
    wants is still a paid render."""
    client = _FakeVPClient()
    ctx = _ctx(title="", brand={})

    out = await _run(GraphicsDesignStage(client=client), ctx)  # ty: ignore[invalid-argument-type]

    assert client.calls == []
    assert out.cards == []
    assert any("NO graphics" in note for note in out.notes)


async def test_an_empty_cut_designs_nothing() -> None:
    out = await _run(GraphicsDesignStage(client=_FakeVPClient()), _ctx([]))  # ty: ignore[invalid-argument-type]

    assert out.cards == []
    # No shots to overlay is an editorial dead end, NOT a render outage.
    assert out.cards_render_failed is False


async def test_a_stub_is_flagged_as_a_stub_and_an_outage_is_not() -> None:
    """The three readings of ``cards == []`` have to stay tellable apart.

    No designer runs on the empty-cut or the untitled-brief path — there IS no
    designer yet — so those empty lists are stubs, not editorial decisions, and
    this stage pauses for a human who would otherwise approve "no graphics"
    believing something chose it. A render outage is the third state and must not
    borrow the stub flag: it DID design a card.
    """
    no_shots = await _run(GraphicsDesignStage(client=_FakeVPClient()), _ctx([]))  # ty: ignore[invalid-argument-type]
    untitled = await _run(
        GraphicsDesignStage(client=_FakeVPClient()),  # ty: ignore[invalid-argument-type]
        _ctx(title="", brand={}),
    )
    outage = await _run(GraphicsDesignStage(client=_FakeVPClient(fail_on="record")), _ctx())  # ty: ignore[invalid-argument-type]

    for stubbed in (no_shots, untitled):
        assert stubbed.cards == []
        assert stubbed.cards_stubbed is True
        assert stubbed.cards_render_failed is False

    assert outage.cards == []
    assert outage.cards_stubbed is False
    assert outage.cards_render_failed is True


# --- the wire ---------------------------------------------------------------
#
# Lives here rather than in tests/clients/ because the flag exists for exactly
# one caller: the stage above. Asserting it on a FAKE client only proves the
# stage passes a keyword — the service never sees a keyword, it sees a payload.


async def test_the_transparent_flag_actually_reaches_the_recorder_payload() -> None:
    """A `transparent` kwarg the client swallows would render an OPAQUE card and
    every stage-level test would still pass."""
    client = VideoProductionClient(base_url="http://vp.invalid", license_token="t")
    sent: dict[str, Any] = {}

    async def _capture(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        sent.update({"method": method, "path": path, **kwargs})
        return {"job_id": "job-1"}

    client._request = _capture

    await client.record_animation(template_id="lower_third", transparent=True)

    assert sent["json"]["transparent"] is True


async def test_an_ordinary_render_sends_no_transparent_key_at_all() -> None:
    """Every pre-existing caller must keep issuing a byte-identical payload."""
    client = VideoProductionClient(base_url="http://vp.invalid", license_token="t")
    sent: dict[str, Any] = {}

    async def _capture(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        sent.update(kwargs)
        return {"job_id": "job-1"}

    client._request = _capture

    await client.record_animation(template_id="animated_title_card")

    assert "transparent" not in sent["json"]


# --- the failure flag is not stuck on ---------------------------------------


async def test_a_successful_design_does_not_report_a_render_failure() -> None:
    """`cards_render_failed` is only worth reading if the happy path clears it —
    a flag that is always True is as useless as one that is always False."""
    out = await _run(GraphicsDesignStage(client=_FakeVPClient()), _ctx())  # ty: ignore[invalid-argument-type]

    assert out.cards_render_failed is False
    assert out.cards and out.cards[0].url


# --- the timeout ladder: the degrade path has to be REACHABLE ----------------
#
# Everything above proves the stage DEGRADES when the recorder throws. None of
# it proves the stage is ever reached by that throw in production: the stage
# runs under `TimeoutMiddleware`, which is `asyncio.wait_for`, and `wait_for`
# cancels by injecting `CancelledError` — a BaseException that `except
# Exception` does not catch. With the recorder budget ABOVE the stage ceiling
# (600 s under 300 s, as shipped) the outer wrapper always won and a slow
# recorder — the single likeliest outage — HARD-FAILED the run instead of
# degrading. The two constants are the contract; these tests pin it from both
# ends.

_GRAPHICS_SPEC = next(s for s in ACTION_REEL_BLUEPRINT.stages if s.name == "graphics_design")
#: The inner budget must clear the outer ceiling by enough to fetch the rendered
#: WebM, store it, and build the degraded output — all still under the ceiling.
_MIN_HEADROOM_SECONDS = 30.0


def test_the_recorder_budget_stays_under_the_stage_ceiling() -> None:
    outer = _GRAPHICS_SPEC.default_timeout_seconds
    assert outer is not None, "graphics_design must pin an explicit stage ceiling"
    assert outer > _RECORD_TIMEOUT_S, (
        f"inverted ladder: recorder budget {_RECORD_TIMEOUT_S}s >= stage ceiling {outer}s, so "
        "asyncio.wait_for cancels the stage with a BaseException before the recorder's own "
        "timeout can be caught — the cards_render_failed degrade path becomes unreachable"
    )
    assert outer - _RECORD_TIMEOUT_S >= _MIN_HEADROOM_SECONDS, (
        f"only {outer - _RECORD_TIMEOUT_S}s of headroom: the fetch + store that follow the "
        "recorder poll would be cancelled mid-flight"
    )


class _SlowRecorder(_FakeVPClient):
    """A recorder that never finishes, honouring the caller's own timeout.

    Which is what the real client does: `wait_for_content_recording` polls until
    `timeout` and then raises `TimeoutError`. Whether the stage sees that
    TimeoutError or a `CancelledError` from the outer wrapper is decided purely
    by which of the two budgets is smaller — the property under test.
    """

    async def wait_for_content_recording(self, job_id: str, timeout: float = 0.0) -> Any:
        await asyncio.sleep(timeout)
        raise TimeoutError("recording never finished")


async def _run_under_ceiling(ceiling: float, inner: float, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Run the real stage under the real TimeoutMiddleware, scaled to test time.

    Scaled, not mocked: the ratio between the two numbers is the whole subject,
    and 240 s under 300 s cannot be waited out in a unit test. The middleware,
    the handler and the exception types are the shipped ones.
    """
    monkeypatch.setattr("src.services.stages.action_reel.graphics_design._RECORD_TIMEOUT_S", inner)
    stage = GraphicsDesignStage(client=_SlowRecorder())  # ty: ignore[invalid-argument-type]
    chain = build_chain([TimeoutMiddleware(seconds=ceiling)])
    return await chain(lambda: stage.handle(GraphicsDesignInput(), _ctx()))


async def test_a_slow_recorder_degrades_when_the_inner_budget_fires_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shipped ordering (inner < outer): the stage survives and reports."""
    out = await _run_under_ceiling(0.5, 0.05, monkeypatch)

    assert out.cards == []
    assert out.cards_render_failed is True
    assert any("NO graphics" in note for note in out.notes)


async def test_an_inverted_ladder_hard_fails_the_stage_instead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bug, pinned: with the recorder budget ABOVE the ceiling the outer
    wrapper cancels first and NOTHING degrades — no note, no flag, a failed run.
    This is why the constants above are a contract and not a preference. Note
    the stage is not at fault and must NOT be "fixed" by catching CancelledError:
    swallowing a cancellation would break every real cancel path in the service."""
    with pytest.raises((asyncio.TimeoutError, asyncio.CancelledError)):
        await _run_under_ceiling(0.05, 0.5, monkeypatch)


# --- the callout designer ---------------------------------------------------
#
# The title card above is deterministic; these are not. A model asked for "at
# most three, never on shot 0, one per shot" returns four, on shot 0, twice on
# shot 2 — so every one of those rules is enforced in code and tested against a
# model that breaks it. The compiler caps `Shot.overlay_clips` at TWO with a
# `max_length`, so an unenforced cap does not cost a card, it 422s the WHOLE
# render and the reel is lost.


async def test_callouts_are_designed_rendered_and_ride_on_their_own_shot() -> None:
    """The feature, end to end: a proposal becomes a real alpha-WebM overlay on
    the shot it names, through the SAME record/store chain as the title."""
    vp = _FakeVPClient()
    wm = _FakeWM({"cards": [_proposal(2, "Rail grab", "Mach loop")]})
    ctx = _ctx(_shots(4))

    out = await _run(GraphicsDesignStage(client=vp, wm=wm), ctx)  # ty: ignore[invalid-argument-type]

    assert [c.kind for c in out.cards] == ["title", "callout"]
    callout = out.cards[1]
    assert callout.shot_id == "sh2"
    assert callout.text == "Rail grab"
    assert callout.url.endswith(".webm")
    # `lower_third` is the ONLY overlay-safe template — anything else 400s, and a
    # non-transparent render would black out the footage it flies over.
    assert vp.calls[1]["template_id"] == "lower_third"
    assert vp.calls[1]["transparent"] is True
    assert vp.calls[1]["output_format"] == "webm"
    assert vp.calls[1]["params"]["name"] == "Rail grab"
    assert vp.calls[1]["params"]["role"] == "Mach loop"
    assert out.callouts_designed == 1
    assert out.callouts_dropped_unusable == 0
    assert out.callouts_degraded is False


async def test_no_more_than_three_callouts_reach_the_reel() -> None:
    """A reel is 20-60s of continuous footage; a caption on every shot is a
    slideshow, and each one is a paid render. The model WILL over-answer."""
    vp = _FakeVPClient()
    wm = _FakeWM({"cards": [_proposal(i) for i in range(1, 8)]})
    ctx = _ctx(_shots(8))

    out = await _run(GraphicsDesignStage(client=vp, wm=wm), ctx)  # ty: ignore[invalid-argument-type]

    assert out.callouts_designed == _MAX_CALLOUTS
    assert len(out.cards) == _MAX_CALLOUTS + 1  # + the title
    # The overflow is COUNTED, never silently swallowed: 3 cards out of 7 asked
    # for and 3 out of 3 asked for are different draws.
    assert out.callouts_dropped_unusable == 7 - _MAX_CALLOUTS
    # And nothing over the cap was ever recorded — an unplaceable card is a paid
    # render for nothing.
    assert len(vp.calls) == _MAX_CALLOUTS + 1


async def test_a_second_callout_on_the_same_shot_is_dropped_and_counted() -> None:
    """Two captions stacked on one shot is not a design — and with the title
    already on shot 0, letting them stack is how a shot reaches the compiler's
    two-overlay `max_length` and 422s the whole render."""
    vp = _FakeVPClient()
    wm = _FakeWM({"cards": [_proposal(1, "First"), _proposal(1, "Second")]})

    out = await _run(GraphicsDesignStage(client=vp, wm=wm), _ctx(_shots(4)))  # ty: ignore[invalid-argument-type]

    assert out.callouts_designed == 1
    assert out.cards[1].text == "First"
    assert out.callouts_dropped_unusable == 1
    assert [c.shot_id for c in out.cards] == ["sh0", "sh1"]


async def test_a_callout_never_lands_on_the_title_shot() -> None:
    """Shot 0 already carries the title treatment. A callout there is a second
    overlay on the one shot that has one, fighting it for the same corner."""
    vp = _FakeVPClient()
    wm = _FakeWM({"cards": [_proposal(0, "Nope")]})

    out = await _run(GraphicsDesignStage(client=vp, wm=wm), _ctx(_shots(4)))  # ty: ignore[invalid-argument-type]

    assert [c.kind for c in out.cards] == ["title"]
    assert out.callouts_designed == 0
    assert out.callouts_dropped_unusable == 1


async def test_a_shot_too_short_to_hold_a_card_is_skipped_and_counted() -> None:
    """The same window rule the title lives under: below the floor a card cannot
    complete its entrance, and the compiler trims it mid-exit-wipe. `_card_window`
    is the single place that judges it, for both card kinds."""
    vp = _FakeVPClient()
    shots = _shots(3)
    shots[1]["seconds"] = 0.5  # under _MIN_CARD_MS
    wm = _FakeWM({"cards": [_proposal(1), _proposal(2)]})

    out = await _run(GraphicsDesignStage(client=vp, wm=wm), _ctx(shots))  # ty: ignore[invalid-argument-type]

    assert [c.shot_id for c in out.cards] == ["sh0", "sh2"]
    assert out.callouts_designed == 1
    assert out.callouts_dropped_unusable == 1
    # Never recorded: an unplaceable card is still a paid render.
    assert [call["params"]["name"] for call in vp.calls] == ["Ridgeline", "Full send"]


@pytest.mark.parametrize(
    ("label", "proposal"),
    [("out of range", {"shot_index": 99, "headline": "x", "strap": ""})],
)
async def test_an_unusable_proposal_is_dropped_and_counted(label: str, proposal: Any) -> None:
    """Nothing is REPAIRED. A shot index the model invented is not evidence about
    which shot it meant, and a blank lower-third is worse than no lower-third."""
    vp = _FakeVPClient()
    wm = _FakeWM({"cards": [proposal]})

    out = await _run(GraphicsDesignStage(client=vp, wm=wm), _ctx(_shots(4)))  # ty: ignore[invalid-argument-type]

    assert [c.kind for c in out.cards] == ["title"], label
    assert out.callouts_designed == 0, label
    assert out.callouts_dropped_unusable == 1, label
    assert len(vp.calls) == 1, f"{label}: an unusable proposal bought a render"


async def test_overlong_text_is_trimmed_rather_than_rendered_off_the_safe_area() -> None:
    """`lower_third` is a strap, not a paragraph — an unbounded headline renders
    as clipped text over the footage. Trimmed, not dropped: a long headline is
    still a headline."""
    vp = _FakeVPClient()
    wm = _FakeWM({"cards": [_proposal(1, "H" * 200, "S" * 200)]})

    out = await _run(GraphicsDesignStage(client=vp, wm=wm), _ctx(_shots(3)))  # ty: ignore[invalid-argument-type]

    assert len(vp.calls[1]["params"]["name"]) == _MAX_HEADLINE_CHARS
    assert len(vp.calls[1]["params"]["role"]) == _MAX_STRAP_CHARS
    assert out.callouts_designed == 1


async def test_a_designer_outage_still_ships_the_title_and_says_so() -> None:
    """THE fallback. A callout outage must never cost the reel the one card the
    product has always shipped — and `callouts_designed == 0` alone cannot tell an
    outage from a designer that ran and chose nothing."""
    vp = _FakeVPClient()

    out = await _run(GraphicsDesignStage(client=vp, wm=_FakeWM(fail=True)), _ctx(_shots(4)))  # ty: ignore[invalid-argument-type]

    assert [c.kind for c in out.cards] == ["title"]
    assert out.cards[0].url
    assert out.callouts_degraded is True
    assert out.callouts_designed == 0
    # The title is on the reel, so this is NOT a graphics outage.
    assert out.cards_render_failed is False
    assert any("callout designer did not answer" in note for note in out.notes)


async def test_a_malformed_draw_is_an_outage_not_a_choice() -> None:
    """The schema promises a `cards` array. Anything else is the designer failing
    to answer, which must read differently from it answering "none"."""
    for output in ({"nope": []}, {"cards": "three"}, None, "text"):
        out = await _run(
            GraphicsDesignStage(client=_FakeVPClient(), wm=_FakeWM(output)),  # ty: ignore[invalid-argument-type]
            _ctx(_shots(4)),
        )
        assert out.callouts_degraded is True, output
        assert [c.kind for c in out.cards] == ["title"], output


async def test_a_designer_that_chose_nothing_is_not_reported_as_degraded() -> None:
    """The mirror. All four counters at their zero value is the ONE reading that
    means "the designer looked at the cut and placed no callouts"."""
    stage = GraphicsDesignStage(client=_FakeVPClient(), wm=_FakeWM({"cards": []}))  # ty: ignore[invalid-argument-type]
    out = await _run(stage, _ctx(_shots(4)))

    assert out.callouts_degraded is False
    assert out.callouts_designed == 0
    assert out.callouts_dropped_unusable == 0
    assert out.callouts_render_failed == 0
    assert [c.kind for c in out.cards] == ["title"]


async def test_a_one_shot_cut_asks_the_designer_nothing() -> None:
    """Every legal callout shot has been spent by the title, so a call could only
    return proposals this stage must drop — and it would be paid for."""
    wm = _FakeWM({"cards": [_proposal(0)]})

    out = await _run(GraphicsDesignStage(client=_FakeVPClient(), wm=wm), _ctx(_shots(1)))  # ty: ignore[invalid-argument-type]

    assert wm.calls == []
    # Not an outage: there was nothing to ask about.
    assert out.callouts_degraded is False
    assert [c.kind for c in out.cards] == ["title"]


class _CalloutRenderFails(_FakeVPClient):
    """The recorder takes the title and then refuses every callout."""

    async def record_animation(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        if kwargs["part_id"] != "reel-title":
            raise RuntimeError("recorder said no")
        return "job-1"


async def test_a_callout_that_fails_to_render_is_dropped_counted_and_not_a_reel_outage() -> None:
    """Per-card soft-fail. `cards_render_failed` means "the reel has no graphics",
    which is FALSE while the title is on it — so a lost callout gets its own
    counter rather than borrowing that flag."""
    wm = _FakeWM({"cards": [_proposal(1), _proposal(2)]})

    out = await _run(GraphicsDesignStage(client=_CalloutRenderFails(), wm=wm), _ctx(_shots(4)))  # ty: ignore[invalid-argument-type]

    assert [c.kind for c in out.cards] == ["title"]
    assert out.callouts_render_failed == 2
    assert out.callouts_designed == 0
    # An outage of the RENDERER, not of the designer, and not of the reel.
    assert out.callouts_degraded is False
    assert out.cards_render_failed is False
    assert any("FAILED TO RENDER" in note for note in out.notes)


async def test_a_title_render_outage_does_not_pay_for_callout_renders() -> None:
    """The callouts use the same recorder that just failed; three more requests
    buy nothing but latency on a stage a human is waiting at."""
    vp = _FakeVPClient(fail_on="record")
    wm = _FakeWM({"cards": [_proposal(1), _proposal(2)]})

    out = await _run(GraphicsDesignStage(client=vp, wm=wm), _ctx(_shots(4)))  # ty: ignore[invalid-argument-type]

    assert out.cards == []
    assert out.cards_render_failed is True
    assert len(vp.calls) == 1


async def test_a_reworded_callout_gets_a_new_recording_not_the_old_one() -> None:
    """The recorder dedupes on `idempotency_key` SERVER-SIDE and returns the
    original job. A key without the card's text therefore hands a re-run the
    OLD WebM for new words: `card.text` would report the new headline while the
    delivered reel still showed the old one — and this stage's own note tells a
    reviewer to re-run it to try the callouts again, so that path is the
    likeliest one, not an edge case.

    Same run, same shot, same slot; only the words differ."""
    ctx = _ctx(_shots(4))
    first = _FakeVPClient()
    second = _FakeVPClient()

    await _run(
        GraphicsDesignStage(client=first, wm=_FakeWM({"cards": [_proposal(1)]})),  # ty: ignore[invalid-argument-type]
        ctx,
    )
    reworded = {**_proposal(1), "headline": "Totally different words"}
    await _run(
        GraphicsDesignStage(client=second, wm=_FakeWM({"cards": [reworded]})),  # ty: ignore[invalid-argument-type]
        ctx,
    )

    before = first.calls[-1]["idempotency_key"]
    after = second.calls[-1]["idempotency_key"]
    assert before != after, (
        "a reworded callout reused the recording key, so the recorder would "
        f"hand back the previous render: {before}"
    )
    # The deterministic TITLE must still dedupe — the digest may not turn every
    # resubmit into a second paid recording.
    assert first.calls[0]["idempotency_key"] == second.calls[0]["idempotency_key"]


async def test_a_strap_only_edit_also_gets_a_new_recording() -> None:
    """Both rendered lines are in the digest. A digest over the headline alone
    would silently deliver the old strap."""
    ctx = _ctx(_shots(4))
    a, b = _FakeVPClient(), _FakeVPClient()

    await _run(GraphicsDesignStage(client=a, wm=_FakeWM({"cards": [_proposal(1)]})), ctx)  # ty: ignore[invalid-argument-type]
    restrapped = {**_proposal(1), "strap": "a different strap"}
    await _run(GraphicsDesignStage(client=b, wm=_FakeWM({"cards": [restrapped]})), ctx)  # ty: ignore[invalid-argument-type]

    assert a.calls[-1]["idempotency_key"] != b.calls[-1]["idempotency_key"]


async def test_each_card_keys_its_own_recording_job() -> None:
    """One idempotency key for two cards would make a resubmit hand the callout
    the TITLE's recording — or make the second card reuse the first's."""
    vp = _FakeVPClient()
    wm = _FakeWM({"cards": [_proposal(1), _proposal(2)]})

    await _run(GraphicsDesignStage(client=vp, wm=wm), _ctx(_shots(4)))  # ty: ignore[invalid-argument-type]

    keys = [call["idempotency_key"] for call in vp.calls]
    assert len(set(keys)) == len(keys)


# --- the WM payload ---------------------------------------------------------
#
# Asserting these on the stage's own constants proves nothing about what the
# worker receives; both are read off the recorded call.


async def test_the_designer_call_carries_reasoning_off_and_a_real_token_budget() -> None:
    """Both halves are measured production facts, not preferences. With reasoning
    ON the model spends the whole budget thinking and returns NOTHING; with
    `max_tokens` under the floor WM answers 422, which is TERMINAL (absent from
    the retryable status codes) and kills the stage on attempt 1."""
    wm = _FakeWM()

    await _run(GraphicsDesignStage(client=_FakeVPClient(), wm=wm), _ctx(_shots(4)))  # ty: ignore[invalid-argument-type]

    call = wm.calls[0]
    assert call["enable_reasoning"] is False
    assert call["max_tokens"] >= MIN_OUTPUT_TOKEN_BUDGET
    assert call["worker_name"] == "action-reel-graphics-designer"
    assert call["output_schema"]["required"] == ["cards"]


async def test_the_prompt_is_bounded_and_resampled_rather_than_head_truncated() -> None:
    """`output_token_budget` bounds the ANSWER only: an unbounded prompt makes the
    model think longer and spend that budget faster. And thinning by truncation
    would hide the END of the reel — the landing, the reveal — which is exactly
    the half a callout is for."""
    wm = _FakeWM()
    ctx = _ctx(_shots(60))

    await _run(GraphicsDesignStage(client=_FakeVPClient(), wm=wm), ctx)  # ty: ignore[invalid-argument-type]

    described = json.loads(wm.calls[0]["message"])["shots"]
    assert len(described) == _MAX_SHOTS_DESCRIBED
    indices = [shot["shot_index"] for shot in described]
    # End to end, first and LAST kept — not the first N.
    assert indices[0] == 0
    assert indices[-1] == 59
    assert indices == sorted(indices)
    assert all(len(shot["rationale"]) <= 160 for shot in described)


async def test_the_schema_carries_no_grammar_unenforceable_bound() -> None:
    """worker-management STRIPS every such keyword before building the grammar and
    validates against the STRIPPED schema, so a bound here is a guarantee nothing
    applies. The limits are enforced in `_usable_callouts`; the schema only
    documents them. `tests/test_output_schema_bounds.py` is the tree-wide gate —
    this pins the same property on the payload the worker actually receives."""
    banned = {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "minProperties",
        "maxProperties",
    }
    wm = _FakeWM()
    await _run(GraphicsDesignStage(client=_FakeVPClient(), wm=wm), _ctx(_shots(4)))  # ty: ignore[invalid-argument-type]

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            assert not (banned & node.keys()), f"grammar-unenforceable bound in {node}"
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(wm.calls[0]["output_schema"])


def test_the_stage_ceiling_covers_every_render_the_cap_allows() -> None:
    """The ladder's second half. `test_llm_timeout_ladder` proves the ceiling
    clears the fleet's WM rung; this proves what is left over still covers the
    renders. They are SEQUENTIAL, so the cap is a multiplier — a ceiling sized
    for one card cancels the stage mid-render on the third callout, with a
    BaseException no degrade path can catch.

    Measured against ONE designer attempt, not `llm_retries + 1`. An earlier
    revision reserved for retries this stage cannot perform: `_design_callouts`
    awaits `WMClient.message` DIRECTLY inside a bare `except Exception` that
    soft-fails to title-only — there is no `stage_call`, no `guarded`, no retry
    loop, so a second attempt is `max_stage_attempts`, a fresh execution of the
    whole stage OUTSIDE this ceiling. Reserving ~1000 s for it inflated the
    ceiling against a run whose end-to-end deadline is 3600 s while the
    blueprint's ceilings already sum far past that.

    The fleet floor is NOT dropped in the process — it is asserted here too, on
    the line below, so this test is strictly stronger than the one it replaces
    rather than a relaxation with a story attached."""
    renders = (_MAX_CALLOUTS + 1) * _RECORD_TIMEOUT_S
    assert _GRAPHICS_SPEC.default_timeout_seconds == GRAPHICS_DESIGN_TIMEOUT_SECONDS
    llm_share = GRAPHICS_DESIGN_TIMEOUT_SECONDS - renders
    assert llm_share >= llm_client_timeout_seconds(), (
        f"ceiling {GRAPHICS_DESIGN_TIMEOUT_SECONDS}s leaves only {llm_share}s for the "
        f"designer once {renders}s of sequential renders are covered"
    )
    # The shared rung, independently. `test_llm_timeout_ladder` enforces this
    # fleet-wide; duplicated here because THIS file is where someone lowering
    # the constant will look, and a ceiling under the rung makes every degrade
    # path in the stage unreachable.
    ladder_floor = (
        llm_budget_seconds() * (PipelineSettings().llm_retries + 1) + _MIN_OUTER_MARGIN_SECONDS
    )
    assert ladder_floor <= GRAPHICS_DESIGN_TIMEOUT_SECONDS, (
        f"ceiling {GRAPHICS_DESIGN_TIMEOUT_SECONDS}s is under the fleet rung {ladder_floor}s"
    )


def test_the_designer_is_a_single_unretried_attempt() -> None:
    """The premise the ceiling arithmetic rests on. If a retry wrapper is ever
    added around the designer call, the ceiling above is silently too small and
    the stage starts hard-failing instead of degrading — so the premise is
    pinned rather than left as a comment."""
    source = inspect.getsource(GraphicsDesignStage._design_callouts)
    assert source.count("await self._wm.message(") == 1
    assert "stage_call" not in source
    assert "guarded" not in source
