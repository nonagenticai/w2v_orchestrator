"""reconcile_timeline's MEASURED per-line fit — replay of showcase runs 01a15b57-9403 /
01a15bda-2d21 (2026-10-20/21, 30s order, todomvc, three ~10s footage clips).

Both runs shipped ``narration_rate_floor_clamped`` + ``timeline_underspoken``: the
underspoken re-ask sized and judged WORDS at one pooled rate (176.6 / 182.6 wpm), while
the real per-line rates spread 140-187 wpm, so lines that "hit" their word counts still
spoke 1-2.6s short of their clips and narrate could only clamp the speed.

The fake TTS speaks each line at the per-line rate measured live on those runs (narrate
clips: run 1 clip-1 21 words / 7.608s, clip-4 27 / 8.933s, clip-5 24 / 10.28s at 1.0;
run 2 clip-1 32 / 10.687s, clip-4 22 / 7.071s, clip-5 30 / 10.432s). The fake model
rewrites to exactly the word count it is asked for, with words taken from the beat's
own text (so the grounding check passes).
"""

from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import pytest

from src.services.runs.run_context import RunContext
from src.services.stages.demo_video import reconcile_timeline as rt
from src.services.stages.demo_video.narrate import DemoVideoNarrateStage
from src.services.stages.demo_video.narration_fit import (
    FIT_SYSTEM,
    FIT_WORD_TOLERANCE,
    FitTarget,
    fit_message,
    needs_recount,
    recount_message,
    word_miss,
)
from src.services.stages.demo_video.reconcile_timeline import ReconcileTimelineStage
from src.services.stages.generic.line_fit import (
    ACCEPT_BAND,
    MAX_REWRITES_PER_LINE,
    LineFit,
    LineTake,
    fit_lines,
)
from src.services.stages.primitives.narration import SynthesizedClip
from src.services.stages.shared.speech_rate import SCRIPT_SAMPLE, spoken_text_digests
from src.services.tts_storage import deterministic_clip_id

if TYPE_CHECKING:
    from src.clients.tts_client import TTSClient
    from src.services.llm import WMClient

_TARGET = 30.0

_GHERKIN_1 = (
    "Scenario: Add a new todo item\n"
    '  When I type "Buy groceries" into the input field with placeholder '
    '"What needs to be done?"\n'
    '  And I press the "Enter" key\n'
    '  Then I see the text "Buy groceries" displayed in the todo list'
)
_GHERKIN_4 = (
    "Scenario: Add a todo and mark it as completed\n"
    '  When I type "Pay bills" into the input field with placeholder '
    '"What needs to be done?"\n'
    '  And I press the "Enter" key\n'
    '  And I click the checkbox next to the todo item "Pay bills"\n'
    '  Then the todo item "Pay bills" is marked as completed'
)
_GHERKIN_5 = (
    "Scenario: Verify the demo disclaimer and navigation links\n"
    '  Then I see the text "This is just a demo of TodoMVC for testing, not the real '
    'TodoMVC app."\n'
    '  And I see a link with the text "Remo H. Jansen"'
)

#: Run 1 — retention_plan's lines (what reconcile was handed), verbatim.
_RUN1: dict[str, Any] = {
    "clips": {"clip-1": 10.32, "clip-4": 10.96, "clip-5": 8.48},
    "texts": {
        "clip-1": 'I open the React TodoMVC app and type "Buy groceries" into the input '
        'field with placeholder "What needs to be done?".',
        "clip-4": 'Next, I enter "Pay bills" into the input field with placeholder "What '
        'needs to be done?" and mark it as completed.',
        "clip-5": "Finally, I verify the demo disclaimer and navigation links on the "
        "React TodoMVC page.",
    },
    "wpm": {"clip-1": 165.6, "clip-4": 181.4, "clip-5": 140.1},
    "pooled_wpm": 176.55581589032246,
}
#: Run 2 — retention_plan's lines, verbatim.
_RUN2: dict[str, Any] = {
    "clips": {"clip-1": 10.44, "clip-4": 11.2, "clip-5": 8.84},
    "texts": {
        "clip-1": 'Next, I type "Buy groceries" into the input field with placeholder '
        '"What needs to be done?". Next, I press the Enter key.',
        "clip-4": 'Next, I type "Pay bills" into the input field with placeholder "What '
        'needs to be done?". Next, I press the Enter key.',
        "clip-5": 'You should now see the heading "todos" on the page. Also, notice the '
        "small demo notice text displayed directly below the main heading.",
    },
    "wpm": {"clip-1": 179.7, "clip-4": 186.7, "clip-5": 172.6},
    "pooled_wpm": 182.60759294931793,
}
_GHERKIN = {"clip-1": _GHERKIN_1, "clip-4": _GHERKIN_4, "clip-5": _GHERKIN_5}


def _beat_of(text: str) -> str:
    if "groceries" in text:
        return "clip-1"
    if "bills" in text:
        return "clip-4"
    return "clip-5"


class _Voice:
    """Fake narrator: each beat at its live-measured rate; duration scales as 1/speed."""

    def __init__(self, wpm: dict[str, float], *, fail: bool = False) -> None:
        self.wpm = wpm
        self.fail = fail
        self.calls: list[tuple[str, str | None, float | None]] = []

    def seconds(self, text: str, speed: float | None) -> float:
        return len(text.split()) * 60.0 / self.wpm[_beat_of(text)] / float(speed or 1.0)

    async def store(self, _client: object, **kw: Any) -> SynthesizedClip:
        text, voice, speed = kw["text"], kw.get("voice"), kw.get("speed")
        self.calls.append((text, voice, speed))
        if self.fail:
            raise RuntimeError("tts /synthesize returned 503")
        return SynthesizedClip(
            deterministic_clip_id(text, voice, speed),
            f"/clips/{deterministic_clip_id(text, voice, speed)}.wav",
            self.seconds(text, speed),
            speed,
            len(text.split()),
        )


_ASK_RE = re.compile(r"rewrite it to exactly (\d+) words")
_BEAT_RE = re.compile(r"Beat \(\d+ words\): (.*)\Z", re.DOTALL)


class _Model:
    """Fake WM: answers a fit rewrite with exactly the asked word count (own words)."""

    def __init__(self) -> None:
        self.fit_calls: list[str] = []
        self.expander_calls: list[str] = []

    async def message(self, **kw: Any) -> Any:
        worker = str(kw.get("worker_name") or "")
        msg = str(kw.get("message") or "")
        ask = _ASK_RE.search(msg)
        if ask and worker in ("beat-expander", "beat-rewriter"):
            self.fit_calls.append(msg)
            want = int(ask.group(1))
            beat = _BEAT_RE.search(msg)
            words = (beat.group(1) if beat else "").split()
            grown = (words * (want // max(len(words), 1) + 1))[:want]
            return SimpleNamespace(output={"text": " ".join(grown)})
        if worker == "beat-expander":
            self.expander_calls.append(msg)
            return SimpleNamespace(output={"text": ""})
        if worker == "pacing-critic":
            return SimpleNamespace(output={"overall_score": 4}, output_type="object")
        return SimpleNamespace(output={"directives": []}, output_type="object")


def _ctx(run: dict[str, Any], **snapshot: Any) -> RunContext:
    texts = run["texts"]
    lines = [
        {
            "id": bid,
            "text": text,
            "seconds": None,
            "clip": bid,
            "scenario_name": _GHERKIN[bid].splitlines()[0].removeprefix("Scenario: "),
            "gherkin": _GHERKIN[bid],
        }
        for bid, text in texts.items()
    ]
    gen = {
        "lines": lines,
        "bdd_job_id": "",
        "features_generated": 0,
        "package_url": "",
        "resolved_wpm": run["pooled_wpm"],
        "resolved_wpm_measured": True,
        "resolved_wpm_reason": None,
        "resolved_wpm_basis": SCRIPT_SAMPLE,
        "resolved_wpm_text_digests": spoken_text_digests(list(texts.values())),
    }
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "direction": {
                "timing": {
                    "mode": "auto",
                    "relief": {"mode": "auto", "iterations": 2, "score_threshold": 4.0},
                }
            },
            "deliverable": {"target_duration_seconds": _TARGET},
            "voice": {"voice_id": "af_heart"},
            **snapshot,
        },
        prior_run_outputs={
            "demo_video_gen": gen,
            "demo_video_record": {
                "clips": [
                    {
                        "id": bid,
                        "line_id": bid,
                        "video_url": f"https://minio.test/{bid}.mp4",
                        "seconds": seconds,
                    }
                    for bid, seconds in run["clips"].items()
                ]
            },
        },
    )


async def _reconcile(
    run: dict[str, Any], monkeypatch: pytest.MonkeyPatch, *, fail: bool = False, **snap: Any
) -> tuple[Any, _Voice, _Model, RunContext]:
    voice = _Voice(run["wpm"], fail=fail)
    model = _Model()
    monkeypatch.setattr(rt, "synthesize_and_store_clip", voice.store)
    stage = ReconcileTimelineStage(
        client=cast("WMClient", model), tts_client=cast("TTSClient", object())
    )
    ctx = _ctx(run, **snap)
    out = await stage.handle(rt.ReconcileTimelineInput(), ctx)
    return out, voice, model, ctx


@pytest.mark.parametrize("run", [_RUN1], ids=["01a15b57-9403"])
async def test_replay_every_line_lands_within_5pct_of_its_footage(
    run: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    out, voice, _model, _ = await _reconcile(run, monkeypatch)
    for line in out.lines:
        window = float(line.seconds)
        # The budget IS the footage the beat plays (the clip, or its tail-trimmed out-point).
        assert window <= run["clips"][line.id] + 1e-6
        assert window >= run["clips"][line.id] - 0.6
        spoken = voice.seconds(line.text, line.vo_rate)
        assert abs(spoken - window) / window <= ACCEPT_BAND, (line.id, spoken, window)
        assert 0.9 <= float(line.vo_rate or 0) <= 1.1
    codes = {d["code"] for d in out.degradations}
    assert "timeline_underspoken" not in codes
    total_spoken = sum(voice.seconds(line.text, line.vo_rate) for line in out.lines)
    assert abs(total_spoken - sum(run["clips"].values())) / _TARGET <= ACCEPT_BAND


@pytest.mark.parametrize("run", [_RUN1], ids=["01a15b57-9403"])
async def test_narrate_requests_exactly_the_measured_clip_and_keeps_it(
    run: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    out, voice, _model, ctx = await _reconcile(run, monkeypatch)
    stored = {deterministic_clip_id(t, v, s) for t, v, s in voice.calls}
    narrate = DemoVideoNarrateStage()
    for line in out.lines:
        item = line.model_dump()
        text, ssml = narrate.synth_text_for(item, ctx)
        speed = narrate.speed_for(item, ctx)
        clip_voice = narrate.voice_for(item, ctx.config_snapshot["voice"], ctx)
        assert not ssml
        assert deterministic_clip_id(text, clip_voice, speed) in stored, line.id
        # A fitted line is accepted as measured: no fit target ⇒ no re-synthesis, no clamp.
        assert line.fitted_seconds == pytest.approx(voice.seconds(text, speed))
        assert narrate.measured_fit_target_for(item, ctx, line.fitted_seconds) is None
        assert narrate.pause_seconds_for(item, ctx) == 0.0


async def test_paid_calls_are_bounded_per_line(monkeypatch: pytest.MonkeyPatch) -> None:
    _out, voice, model, _ = await _reconcile(_RUN1, monkeypatch)
    for beat in ("clip-1", "clip-4", "clip-5"):
        synths = [c for c in voice.calls if _beat_of(c[0]) == beat]
        rewrites = [m for m in model.fit_calls if _beat_of(m.split("Beat (")[-1]) == beat]
        assert len(rewrites) <= MAX_REWRITES_PER_LINE
        # first take + one per rewrite + one speed re-synthesis
        assert len(synths) <= 1 + MAX_REWRITES_PER_LINE + 1
    # The word-count re-ask is superseded, not run alongside.
    assert model.expander_calls == []


async def test_fail_open_tts_down_runs_the_word_count_reask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, voice, model, _ = await _reconcile(_RUN1, monkeypatch, fail=True)
    assert voice.calls  # it tried
    assert model.fit_calls == []
    # Today's path ran: the underspoken re-ask was asked (its fake answers empty).
    assert model.expander_calls
    assert all(line.text == _RUN1["texts"][line.id] for line in out.lines)


async def test_verbatim_and_locked_lines_are_never_rewritten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verbatim = [{"text": "Buy groceries", "kind": None}]
    out, _voice, model, _ = await _reconcile(_RUN1, monkeypatch, verbatim_items=verbatim)
    by_id = {line.id: line for line in out.lines}
    assert by_id["clip-1"].text == _RUN1["texts"]["clip-1"]
    assert not any("groceries" in m.split("Beat (")[-1] for m in model.fit_calls)


# --- line_fit.fit_lines: independent budgets ---------------------------------


async def test_fit_lines_fits_each_line_to_its_own_budget_without_absorbing() -> None:
    def take(n: int, speed: float | None = None) -> LineTake:
        return LineTake(text=" ".join(["w"] * n), seconds=n / 2.5 / (speed or 1.0), speed=speed)

    synths: list[tuple[int, str, float | None]] = []

    async def synth(i: int, text: str, speed: float | None) -> LineTake:
        synths.append((i, text, speed))
        return take(len(text.split()), speed)

    async def rewrite(i: int, text: str, m: float, b: float, want: int) -> str | None:
        return " ".join(["w"] * want)

    # line 0 in band, line 1 speed band, line 2 rewrite band, line 3 has no budget
    fits: list[LineFit] = await fit_lines(
        [take(25, 1.0), take(23, 1.0), take(15, 1.0), take(5, 1.0)],
        [10.0, 10.0, 10.0, 0.0],
        synth=synth,
        rewrite=rewrite,
        rewritable=[True, True, True, True],
    )
    assert [f.initial_band for f in fits[:3]] == ["accept", "speed", "rewrite"]
    assert all(abs(f.miss) <= ACCEPT_BAND for f in fits[:3])
    assert fits[0].take.speed == 1.0 and not fits[0].resynthesized
    assert fits[1].resynthesized and fits[1].rewrites == 0
    assert fits[2].rewrites >= 1
    assert fits[3].take.seconds == pytest.approx(2.0)
    assert not any(i in (0, 3) for i, _t, _s in synths)


# --- narrate on FITTED lines with SSML + voice_director on (prod config) -------------


class _Settings:
    narration_ssml_enabled = True


async def _narrate_after_fit(
    monkeypatch: pytest.MonkeyPatch, *, unfit: str | None = None
) -> tuple[Any, Any, list[tuple[str, bool, bool]], dict[str, float]]:
    """Reconcile run 1, then narrate it with SSML + a director pause/emphasis on every line.

    Returns ``(reconcile_out, narrate_out, synth_calls, pads)``; each synth call is
    ``(line_beat, ssml, reused)`` where ``reused`` = the clip id was already stored.
    """
    from src.services.stages.demo_video import narrate as narrate_mod
    from src.services.stages.primitives import narration as narration_mod

    out, voice, _model, ctx = await _reconcile(_RUN1, monkeypatch)
    stored = {deterministic_clip_id(t, v, s): voice.seconds(t, s) for t, v, s in voice.calls}
    plan = out.model_dump()
    if unfit is not None:
        for line in plan["lines"]:
            if line["id"] == unfit:
                line["fitted_seconds"] = None
                line["fitted_slack_seconds"] = None
    ctx.prior_run_outputs["reconcile_timeline"] = plan
    ctx.prior_run_outputs["voice_director"] = {
        "directives": [
            {"line_id": bid, "pause_before_ms": ms, "emphasis_word_indices": [1]}
            for bid, ms in (("clip-1", 150), ("clip-4", 300), ("clip-5", 150))
        ]
    }
    calls: list[tuple[str, bool, bool]] = []

    async def _store(_client: object, **kw: Any) -> SynthesizedClip:
        text, v, speed, ssml = kw["text"], kw.get("voice"), kw.get("speed"), kw.get("ssml")
        cid = deterministic_clip_id(text, v, speed)
        calls.append((_beat_of(text), bool(ssml), cid in stored))
        seconds = stored.get(cid) or voice.seconds(re.sub(r"<[^>]+>", "", text), speed)
        return SynthesizedClip(
            cid, f"http://minio.test/clips/{cid}.wav", seconds, speed, len(text.split())
        )

    pads: dict[str, float] = {}

    async def _pad(_run: str, _stage: str, clip_id: str, pause: float) -> Any:
        pads[clip_id] = pause
        return (
            f"{clip_id}-pad",
            f"http://minio.test/clips/{clip_id}-pad.wav",
            stored[clip_id] + pause,
        )

    monkeypatch.setattr(narrate_mod, "get_settings", _Settings)
    monkeypatch.setattr(narration_mod, "synthesize_and_store_clip", _store)
    monkeypatch.setattr(narrate_mod, "pad_stored_clip_head", _pad)
    narrated = await DemoVideoNarrateStage(client=cast("TTSClient", object())).handle(
        narrate_mod.DemoVideoNarrateInput(), ctx
    )
    return out, narrated, calls, pads


async def test_fitted_lines_replay_the_measured_clip_under_ssml_and_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, narrated, calls, pads = await _narrate_after_fit(monkeypatch)
    # Exactly one call per line, every one plain text and a REUSE of the measured clip.
    assert sorted(b for b, _, _ in calls) == ["clip-1", "clip-4", "clip-5"]
    assert all(not ssml and reused for _, ssml, reused in calls)
    by_line = {c.line_id: c for c in narrated.clips}
    lines = {line.id: line for line in out.lines}
    slack = {bid: float(lines[bid].fitted_slack_seconds or 0.0) for bid in lines}
    # clip-1's 0.15s pause fits its slack -> padded silence; clip-4's 0.30s and
    # clip-5's (no slack) are dropped.
    assert slack["clip-1"] >= 0.15 > slack["clip-5"]
    assert slack["clip-4"] < 0.30
    assert list(pads.values()) == [pytest.approx(0.15)]
    assert by_line["clip-1"].id.endswith("-pad")
    assert not by_line["clip-4"].id.endswith("-pad")
    assert not by_line["clip-5"].id.endswith("-pad")
    # Length unchanged: every beat's audio still ends inside its footage window, and
    # the spine (the timeline) is narrate's input, untouched.
    for bid in ("clip-4", "clip-5"):
        assert by_line[bid].seconds == pytest.approx(float(lines[bid].fitted_seconds or 0))
    padded = by_line["clip-1"].seconds
    assert padded == pytest.approx(float(lines["clip-1"].fitted_seconds or 0) + 0.15)
    assert padded <= float(lines["clip-1"].seconds) + 1e-6  # still inside its footage


async def test_an_unfitted_line_keeps_todays_ssml_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _out, _narrated, calls, pads = await _narrate_after_fit(monkeypatch, unfit="clip-4")
    ssml_by_beat = {b: ssml for b, ssml, _ in calls}
    assert ssml_by_beat["clip-4"] is True
    assert ssml_by_beat["clip-1"] is False and ssml_by_beat["clip-5"] is False
    assert len(pads) == 1


# --- the fit rewrite's WORD COUNT: checked, re-asked once --------------------------
#
# Recorded showcase run 01a15e77-e3a4 (2026-10-21) clip-5: window 9.04s, first take 19
# words 7.27s, so the pace target asked 24 words — and the model returned the 31-word
# text below, twice (11.01s / 11.38s, +21.8% / +25.9%). At that reply's own pace
# (0.355 s/word) 24 words is 8.52s, -5.7%: inside the 0.9-1.1 clamp's reach.

_C5_TEXT = (
    "First the page displays the heading todos Next the page contains a link with the "
    "text real TodoMVC app"
)
_C5_REPLY_31 = (
    "First the page displays the heading todos Next the page contains a link with the "
    "text real TodoMVC app Then I type Check demo into the input field and press Enter"
)
_C5_REPLY_24 = (
    "First the page shows the heading todos and a link with the text real TodoMVC app "
    "Then I type Check demo into the input"
)


class _Scripted:
    """Fake WM returning the scripted replies in order; records every call."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    async def message(self, **kw: Any) -> Any:
        self.calls.append(kw)
        return SimpleNamespace(output={"text": self.replies.pop(0)})


def _fit_stage(model: _Scripted) -> ReconcileTimelineStage:
    return ReconcileTimelineStage(
        client=cast("WMClient", model), tts_client=cast("TTSClient", object())
    )


def _c5_target() -> FitTarget:
    return FitTarget(
        index=0,
        line_id="clip-5",
        text=_C5_TEXT,
        budget=9.04,
        pad=0.0,
        voice="af_heart",
        rewritable=True,
        source=None,
    )


async def _c5_rewrite(model: _Scripted) -> str | None:
    return await _fit_stage(model)._fit_rewrite(
        _c5_target(), _C5_TEXT, measured=7.27, budget=9.04, want=24
    )


def test_recorded_replies_miss_their_ask_by_more_than_the_tolerance() -> None:
    assert len(_C5_TEXT.split()) == 19
    assert len(_C5_REPLY_31.split()) == 31
    assert word_miss(_C5_REPLY_31, 24) == 7
    assert needs_recount(_C5_REPLY_31, 24)
    assert not needs_recount(_C5_REPLY_24, 24)
    assert not needs_recount(" ".join(["w"] * (24 + FIT_WORD_TOLERANCE)), 24)
    assert needs_recount(" ".join(["w"] * (24 + FIT_WORD_TOLERANCE + 1)), 24)


async def test_an_over_delivered_rewrite_is_re_asked_once_with_its_real_count() -> None:
    model = _Scripted(_C5_REPLY_31, _C5_REPLY_24)
    assert await _c5_rewrite(model) == _C5_REPLY_24
    assert len(model.calls) == 2
    first, second = (c["message"] for c in model.calls)
    assert "rewrite it to exactly 24 words (ADD 5 words)" in first
    # The re-ask is about the model's OWN reply, with its real count and the delta from IT.
    assert "has 31 words" in second
    assert "rewrite it to exactly 24 words (CUT 7 words)" in second
    assert second.endswith(f"Beat (31 words): {_C5_REPLY_31}")
    # It must shrink now: the shorten worker, and the exact-count prompt both times.
    assert [c["worker_name"] for c in model.calls] == ["beat-expander", "beat-rewriter"]
    assert all(c["system_prompt"] == FIT_SYSTEM for c in model.calls)


async def test_a_reply_within_the_tolerance_costs_no_second_call() -> None:
    model = _Scripted(_C5_REPLY_24 + " now")
    assert await _c5_rewrite(model) == _C5_REPLY_24 + " now"
    assert len(model.calls) == 1


async def test_a_re_ask_that_lands_further_off_keeps_the_first_reply() -> None:
    # Recorded: 1021c clip-4 asked 34, got 46 then 43 — a worse recount must not win.
    worse = _C5_REPLY_31 + " and then the item appears in the list below the input"
    model = _Scripted(_C5_REPLY_31, worse)
    assert await _c5_rewrite(model) == _C5_REPLY_31
    assert len(model.calls) == 2


async def test_a_refused_re_ask_keeps_the_first_reply() -> None:
    model = _Scripted(_C5_REPLY_31, "See https://demo.playwright.dev/todomvc for it")
    assert await _c5_rewrite(model) == _C5_REPLY_31


class _ReAskFails(_Scripted):
    """The first call answers; the recount re-ask hits a worker_management timeout."""

    async def message(self, **kw: Any) -> Any:
        if self.calls:
            self.calls.append(kw)
            raise TimeoutError("worker_management timed out")
        return await super().message(**kw)


async def test_a_failed_re_ask_keeps_the_paid_first_reply() -> None:
    model = _ReAskFails(_C5_REPLY_31)
    assert await _c5_rewrite(model) == _C5_REPLY_31
    assert len(model.calls) == 2


async def test_a_cancelled_re_ask_is_not_swallowed() -> None:
    class _Cancelled(_Scripted):
        async def message(self, **kw: Any) -> Any:
            if self.calls:
                raise asyncio.CancelledError
            return await super().message(**kw)

    with pytest.raises(asyncio.CancelledError):
        await _c5_rewrite(_Cancelled(_C5_REPLY_31))


def test_the_fit_prompts_ask_an_exact_count_and_offer_steps_as_optional() -> None:
    msg = fit_message(
        _C5_TEXT,
        measured=7.27,
        budget=9.04,
        want=24,
        source=_GHERKIN_5,
        uncovered=["I type Check demo", "I press Enter", "I click the checkbox"],
    )
    assert "exactly 24 words (ADD 5 words) — not more than 25, not fewer than 23" in msg
    assert "narrate only as many of them as 5 words allow and leave the rest out" in msg
    assert "MINIMUM" not in FIT_SYSTEM and "at or just above" not in FIT_SYSTEM
    assert "EXACT number of words" in FIT_SYSTEM
    cut = recount_message(_C5_REPLY_31, want=24, budget=9.04, source=None)
    assert "not more than 25, not fewer than 23" in cut
    assert "leave the rest out" not in cut  # a CUT offers no steps to add
