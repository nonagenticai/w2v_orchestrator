"""Unit tests for ScriptDoctorStage — WM + rubric both mocked (no live LLM/DB).

The rubric (:func:`score_script`) is monkeypatched per-test so the critique
scores + flags are deterministic and the loop behaviour is independent of the
parallel rubric implementation. The WM gateway is an ``AsyncMock`` whose
``message`` returns a :class:`WMResponse` keyed on ``worker_name`` (critique vs
revise), mirroring the injection pattern used by the other stage tests.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

from src.services.runs.run_context import RunContext
from src.services.stages.base import StageError, StageSkippedError
from src.services.stages.shared import script_doctor as sd_mod
from src.services.stages.shared.script_doctor import ScriptDoctorStage


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


def _wm_fake(revise_lines: list[dict[str, Any]]) -> AsyncMock:
    """AsyncMock WM: critique worker → empty critique; revise worker → revise_lines."""
    fake = AsyncMock()

    def _side(worker_name: str | None = None, **_: Any) -> WMResponse:
        if worker_name == "script-doctor-critique":
            return _resp({"critiques": [], "story_notes": {}})
        return _resp({"lines": revise_lines})

    fake.message.side_effect = _side
    return fake


def _ctx(
    *,
    sd: dict[str, Any] | None,
    prior_lines: list[dict[str, Any]] | None = None,
    verbatim: list[dict[str, Any]] | None = None,
    direction: dict[str, Any] | None = None,
    annotations: list[dict[str, Any]] | None = None,
) -> RunContext:
    cfg: dict[str, Any] = {}
    if sd is not None:
        cfg["script_doctor"] = sd
    if verbatim is not None:
        cfg["verbatim_items"] = verbatim
    if direction is not None:
        cfg["direction"] = direction
    prior = {
        "demo_video_gen": {
            "lines": prior_lines or [],
            "bdd_job_id": "",
            "features_generated": 0,
            "package_url": "",
        }
    }
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=cfg,
        prior_run_outputs=prior,
        open_annotations_by_stage={"script_doctor": annotations or []},
    )


def _patch_score(monkeypatch: pytest.MonkeyPatch, fn: Any) -> None:
    monkeypatch.setattr(sd_mod, "score_script", fn)


# ── (a) inert-by-default: absent / disabled block → skip, WM never touched ──


@pytest.mark.asyncio
@pytest.mark.parametrize("sd", [None])
async def test_disabled_block_skips_without_wm(
    monkeypatch: pytest.MonkeyPatch, sd: dict[str, Any] | None
) -> None:
    calls = {"score": 0}

    def _score(draft: Any, **_: Any) -> dict[str, Any]:
        calls["score"] += 1
        return {"composite": 0.0, "flags": {}, "per_line": {}, "per_story": {}}

    _patch_score(monkeypatch, _score)
    fake = _wm_fake([])
    stage = ScriptDoctorStage(client=fake)
    ctx = _ctx(sd=sd, prior_lines=[{"id": "clip-1", "text": "hi"}])

    with pytest.raises(StageSkippedError):
        await stage.handle(stage.input_model(), ctx)

    fake.message.assert_not_called()
    assert calls["score"] == 0  # no rubric work either before the skip


# ── (b) enabled → loop is bounded ──


@pytest.mark.asyncio
async def test_enabled_loop_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    def _score(draft: Any, **_: Any) -> dict[str, Any]:
        # Always weak + always flag clip-1 → optimizer keeps firing; the engine
        # (not the rubric) is what must clamp the pass count.
        return {"composite": 50.0, "flags": {"clip-1": ["hook"]}, "per_line": {}, "per_story": {}}

    _patch_score(monkeypatch, _score)
    fake = _wm_fake([{"id": "clip-1", "text": "revised"}])
    stage = ScriptDoctorStage(client=fake)
    ctx = _ctx(
        sd={"enabled": True, "max_passes": 2}, prior_lines=[{"id": "clip-1", "text": "orig"}]
    )

    out = await stage.handle(stage.input_model(), ctx)

    assert 1 <= fake.message.call_count <= 2 * 2  # <= 2*max_passes
    workers = {c.kwargs["worker_name"] for c in fake.message.call_args_list}
    assert workers == {"script-doctor-critique", "script-doctor-revise"}
    assert [line.id for line in out.lines] == ["clip-1"]  # spine id preserved


# ── (c) revise echoing an id replaces only text — id/clip/speaker untouched ──


# non-text field untouched


# ── (d) a locked line is never rewritten, even if the worker echoes its id ──


@pytest.mark.asyncio
async def test_locked_line_never_rewritten(monkeypatch: pytest.MonkeyPatch) -> None:
    def _score(draft: Any, **_: Any) -> dict[str, Any]:
        joined = " ".join(str(x.get("text", "")) for x in draft)
        if "NEW2" in joined:
            return {"composite": 90.0, "flags": {}, "per_line": {}, "per_story": {}}
        return {
            "composite": 50.0,
            "flags": {"clip-1": ["hook"], "clip-2": ["hook"]},
            "per_line": {},
            "per_story": {},
        }

    _patch_score(monkeypatch, _score)
    # The worker maliciously echoes the LOCKED id too — must be ignored.
    fake = _wm_fake([{"id": "clip-1", "text": "HACKED"}, {"id": "clip-2", "text": "NEW2"}])
    stage = ScriptDoctorStage(client=fake)
    ctx = _ctx(
        sd={"enabled": True},
        prior_lines=[
            {"id": "clip-1", "text": "LOCKED", "locked": True},
            {"id": "clip-2", "text": "orig2"},
        ],
    )

    out = await stage.handle(stage.input_model(), ctx)

    by_id = {line.id: line for line in out.lines}
    assert by_id["clip-1"].text == "LOCKED"  # locked → never touched
    assert by_id["clip-1"].locked is True
    assert by_id["clip-2"].text == "NEW2"  # unlocked flagged → revised


# ── (e) a verbatim CTA survives the revise pass ──


@pytest.mark.asyncio
async def test_verbatim_cta_survives(monkeypatch: pytest.MonkeyPatch) -> None:
    def _score(draft: Any, **_: Any) -> dict[str, Any]:
        joined = " ".join(str(x.get("text", "")) for x in draft)
        if "Fresh hook" in joined:
            return {"composite": 90.0, "flags": {}, "per_line": {}, "per_story": {}}
        return {"composite": 50.0, "flags": {"clip-1": ["hook"]}, "per_line": {}, "per_story": {}}

    _patch_score(monkeypatch, _score)
    # Only clip-1 is flagged/revised; the CTA line clip-2 is left verbatim.
    fake = _wm_fake([{"id": "clip-1", "text": "Fresh hook"}])
    stage = ScriptDoctorStage(client=fake)
    ctx = _ctx(
        sd={"enabled": True},
        verbatim=[{"text": "Buy now"}],
        prior_lines=[
            {"id": "clip-1", "text": "Hello there", "clip": "clip-1"},
            {"id": "clip-2", "text": "Buy now today", "clip": "clip-2"},
        ],
    )

    out = await stage.handle(stage.input_model(), ctx)

    joined = " ".join(line.text for line in out.lines)
    assert "Buy now" in joined  # verbatim pin survived
    assert out.lines[0].text == "Fresh hook"


# ── (f) a regressing revise → the earlier, better draft is returned ──


@pytest.mark.asyncio
async def test_best_draft_returned_on_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    def _score(draft: Any, **_: Any) -> dict[str, Any]:
        joined = " ".join(str(x.get("text", "")) for x in draft)
        if "WORSE" in joined:
            return {"composite": 40.0, "flags": {}, "per_line": {}, "per_story": {}}
        return {"composite": 70.0, "flags": {"clip-1": ["hook"]}, "per_line": {}, "per_story": {}}

    _patch_score(monkeypatch, _score)
    fake = _wm_fake([{"id": "clip-1", "text": "WORSE"}])
    stage = ScriptDoctorStage(client=fake)
    ctx = _ctx(sd={"enabled": True}, prior_lines=[{"id": "clip-1", "text": "orig"}])

    out = await stage.handle(stage.input_model(), ctx)

    # pass1 (orig) scored 70, pass2 (WORSE) scored 40 → keep the original draft.
    assert out.lines[0].text == "orig"
    assert fake.message.call_count >= 2  # the revise pass did run


# ── (g) a reversed revise response must NOT reorder the spine (C0) ──


@pytest.mark.asyncio
async def test_revise_response_order_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    def _score(draft: Any, **_: Any) -> dict[str, Any]:
        joined = " ".join(str(x.get("text", "")) for x in draft)
        if "R1" in joined:  # already revised → converge, no flags
            return {"composite": 95.0, "flags": {}, "per_line": {}, "per_story": {}}
        return {
            "composite": 50.0,
            "flags": {"clip-1": ["hook"], "clip-2": ["hook"], "clip-3": ["hook"]},
            "per_line": {},
            "per_story": {},
        }

    _patch_score(monkeypatch, _score)
    # Worker returns the lines in REVERSED order — the merge must ignore it and
    # keep the original spine order, mapping text back by id.
    fake = _wm_fake(
        [
            {"id": "clip-3", "text": "R3"},
            {"id": "clip-2", "text": "R2"},
            {"id": "clip-1", "text": "R1"},
        ]
    )
    stage = ScriptDoctorStage(client=fake)
    ctx = _ctx(
        sd={"enabled": True},
        prior_lines=[
            {"id": "clip-1", "text": "a"},
            {"id": "clip-2", "text": "b"},
            {"id": "clip-3", "text": "c"},
        ],
    )

    out = await stage.handle(stage.input_model(), ctx)

    assert [line.id for line in out.lines] == ["clip-1", "clip-2", "clip-3"]  # order preserved
    assert [line.text for line in out.lines] == ["R1", "R2", "R3"]  # text mapped by id


# ── (h) a hallucinated id in the revise response is dropped (no-add, C0) ──


@pytest.mark.asyncio
async def test_hallucinated_id_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    def _score(draft: Any, **_: Any) -> dict[str, Any]:
        joined = " ".join(str(x.get("text", "")) for x in draft)
        if "NEW" in joined:
            return {"composite": 95.0, "flags": {}, "per_line": {}, "per_story": {}}
        return {"composite": 50.0, "flags": {"clip-1": ["hook"]}, "per_line": {}, "per_story": {}}

    _patch_score(monkeypatch, _score)
    # Worker invents a line id that isn't in the spine — it must never appear.
    fake = _wm_fake([{"id": "clip-1", "text": "NEW"}, {"id": "clip-999", "text": "HALLUCINATED"}])
    stage = ScriptDoctorStage(client=fake)
    ctx = _ctx(sd={"enabled": True}, prior_lines=[{"id": "clip-1", "text": "orig"}])

    out = await stage.handle(stage.input_model(), ctx)

    assert [line.id for line in out.lines] == ["clip-1"]  # no invented id
    assert out.lines[0].text == "NEW"


def test_critique_msg_serializes_real_metricresult_subscores() -> None:
    """``_critique_msg`` must render a report whose ``per_line`` values are REAL
    ``MetricResult`` objects — the shape ``score_script`` actually returns.

    Every other test in this module monkeypatches ``score_script`` to return
    ``per_line: {}`` (plain dicts), so none exercised serialization. In prod the
    real rubric returns ``{line_id: {metric: MetricResult}}``; a bare
    ``json.dumps`` on it raised ``TypeError: Object of type MetricResult is not
    JSON serializable`` inside ``_critique_msg`` — BEFORE the WM call, so the
    loop's soft-degrade never caught it — and hard-failed the stage on first
    end-to-end contact. This asserts the model is dumped, not stringified.
    """
    from src.services.evaluation.models import MetricResult

    mr = MetricResult(
        name="abstraction_ratio", score=12.0, weight=0.2, threshold="fail", detail="too abstract"
    )
    report = {
        "composite": 40.0,
        "per_line": {"clip-1": {"abstraction_ratio": mr}},
        "per_story": {},
        "flags": {"clip-1": ["abstraction_ratio"]},
    }
    msg = sd_mod._critique_msg(
        [{"id": "clip-1", "text": "some text"}], report, director=None, notes=[]
    )
    assert "abstraction_ratio" in msg
    # model_dump()'d (structured), not str()'d — the score field is present as a number.
    assert '"score": 12.0' in msg


def test_json_default_rejects_truly_unserializable() -> None:
    """The fallback dumps pydantic models but must NOT silently stringify a
    surprise type — an unexpected object still raises, so a real serialization
    regression is loud rather than hidden behind ``str()``."""
    with pytest.raises(TypeError):
        sd_mod._json_default(object())


# ── soft-degrade: a WM/upstream failure must NOT hard-fail a paid render ──
#
# script_doctor is default-ON on every demo_video run. Before this guard it
# HARD-FAILED the whole run on a WM error — and in prod it did: of 4 live
# script_doctor rows, 3 were `failed`, 2 of them WM 401/402 (expired downstream
# license token / byo-mode misconfig). The stage is a quality *enhancement*, so
# an unavailable enhancer must yield the un-doctored draft, never kill the
# deliverable after the human gen-gate. Mirrors final_critique's soft-degrade.


@pytest.mark.asyncio
async def test_wm_upstream_failure_soft_degrades_to_undoctored_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.services.stages.upstream import PermanentUpstreamError

    _patch_score(
        monkeypatch,
        lambda draft, **_: {
            "composite": 50.0,
            "flags": {"clip-1": ["hook"]},
            "per_line": {},
            "per_story": {},
        },
    )
    fake = AsyncMock()
    fake.message.side_effect = PermanentUpstreamError(
        "upstream 401 from worker-management: Invalid or expired license token"
    )
    stage = ScriptDoctorStage(client=fake)
    original = [{"id": "clip-1", "text": "the ORIGINAL un-doctored line"}]
    ctx = _ctx(sd={"enabled": True, "max_passes": 2}, prior_lines=original)

    # Does NOT raise — the WM 401 is absorbed…
    out = await stage.handle(stage.input_model(), ctx)

    # …and the caller gets the untouched draft, exactly what shipped pre-stage.
    fake.message.assert_awaited()  # the loop really did reach WM (not a skip)
    assert [line.id for line in out.lines] == ["clip-1"]
    assert [line.text for line in out.lines] == ["the ORIGINAL un-doctored line"]


@pytest.mark.asyncio
async def test_malformed_enabled_block_soft_degrades_not_hard_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An author-supplied enabled block with a bad `max_passes` type used to
    hard-fail via `int(...)`; it must now degrade to the un-doctored draft."""
    _patch_score(
        monkeypatch,
        lambda draft, **_: {"composite": 0.0, "flags": {}, "per_line": {}, "per_story": {}},
    )
    fake = _wm_fake([])
    stage = ScriptDoctorStage(client=fake)
    original = [{"id": "clip-1", "text": "keep me verbatim"}]
    ctx = _ctx(sd={"enabled": True, "max_passes": "not-an-int"}, prior_lines=original)

    out = await stage.handle(stage.input_model(), ctx)

    assert [line.text for line in out.lines] == ["keep me verbatim"]
    fake.message.assert_not_called()  # coercion blew up before any WM traffic


@pytest.mark.asyncio
async def test_wm_failure_soft_degrades_when_raw_draft_still_carries_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A WM failure degrades to the raw draft — but only after confirming that raw
    draft still carries the author-mandated verbatim text (the common case)."""
    from src.services.stages.upstream import PermanentUpstreamError

    _patch_score(
        monkeypatch,
        lambda draft, **_: {
            "composite": 50.0,
            "flags": {"clip-1": ["hook"]},
            "per_line": {},
            "per_story": {},
        },
    )
    fake = AsyncMock()
    fake.message.side_effect = PermanentUpstreamError("upstream 402: byo not configured")
    stage = ScriptDoctorStage(client=fake)
    original = [{"id": "clip-1", "text": "Grab it today — SUBSCRIBE NOW to win."}]
    ctx = _ctx(
        sd={"enabled": True, "max_passes": 2},
        prior_lines=original,
        verbatim=[{"text": "SUBSCRIBE NOW"}],
    )

    out = await stage.handle(stage.input_model(), ctx)

    assert [line.text for line in out.lines] == ["Grab it today — SUBSCRIBE NOW to win."]


@pytest.mark.asyncio
async def test_soft_degrade_refuses_to_ship_when_raw_draft_lacks_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """script_doctor is the ONLY verbatim gate in the demo_video chain. If WM is down
    AND the raw draft is missing a mandated phrase, the stage must NOT silently ship
    it — it fails loud (StageError) instead of degrading."""
    from src.services.stages.upstream import PermanentUpstreamError

    _patch_score(
        monkeypatch,
        lambda draft, **_: {
            "composite": 50.0,
            "flags": {"clip-1": ["hook"]},
            "per_line": {},
            "per_story": {},
        },
    )
    fake = AsyncMock()
    fake.message.side_effect = PermanentUpstreamError("upstream 401: expired token")
    stage = ScriptDoctorStage(client=fake)
    # Raw draft does NOT contain the mandated phrase.
    original = [{"id": "clip-1", "text": "Grab it today to win."}]
    ctx = _ctx(
        sd={"enabled": True, "max_passes": 2},
        prior_lines=original,
        verbatim=[{"text": "SUBSCRIBE NOW"}],
    )

    with pytest.raises(StageError, match="verbatim"):
        await stage.handle(stage.input_model(), ctx)


# ── N6 — the rubric is scored against the run's MEASURED speaking rate ──
#
# THE CALL-SITE PROOF for `script_craft.score_script(..., wpm=)`. The rubric's
# `word_budget`/`duration_budget` ask a physical question ("can this voice say
# these words in these seconds?"), and answering it against a fixed 150 wpm while
# the live voice delivers ~120 does not merely mis-score: it tells THIS stage's
# revise loop that an over-packed beat is UNDER budget and should grow.
#
# A defined-but-never-called seam is this repo's signature failure, so these
# assert the value actually arrives at the rubric — not that the parameter exists.


@pytest.mark.asyncio
async def test_the_rubric_is_scored_against_the_measured_speaking_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []

    def _score(draft: Any, **kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs.get("wpm"))
        return {"composite": 99.0, "flags": {}, "per_line": {}, "per_story": {}}

    async def _measured(*_: Any, **__: Any) -> float:
        return 118.5

    _patch_score(monkeypatch, _score)
    monkeypatch.setattr(sd_mod, "resolve_natural_wpm", _measured)
    stage = ScriptDoctorStage(client=_wm_fake([]))
    ctx = _ctx(sd={"enabled": True}, prior_lines=[{"id": "clip-1", "text": "orig"}])

    await stage.handle(stage.input_model(), ctx)

    assert seen, "the rubric was never scored — the call site did not run"
    assert seen == [118.5] * len(seen)


# ── the revise request's ``lines`` is grammar-non-empty (prefixItems) ──


@pytest.mark.asyncio
async def test_revise_request_lines_schema_is_grammar_non_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WM strips ``minItems``; an items-only array lets the local model answer
    ``{"lines": []}`` — a silent no-op revision. The request SENT to the revise
    worker must carry ``prefixItems == [items]``."""

    def _score(draft: Any, **_: Any) -> dict[str, Any]:
        return {"composite": 50.0, "flags": {"clip-1": ["hook"]}, "per_line": {}, "per_story": {}}

    _patch_score(monkeypatch, _score)
    fake = _wm_fake([{"id": "clip-1", "text": "revised"}])
    stage = ScriptDoctorStage(client=fake)
    ctx = _ctx(sd={"enabled": True}, prior_lines=[{"id": "clip-1", "text": "orig"}])

    await stage.handle(stage.input_model(), ctx)

    revise_calls = [
        c for c in fake.message.call_args_list if c.kwargs["worker_name"] == "script-doctor-revise"
    ]
    assert revise_calls, "the revise worker was never called — test would be vacuous"
    for call in revise_calls:
        lines = call.kwargs["output_schema"]["properties"]["lines"]
        assert lines["type"] == "array"
        assert lines["prefixItems"] == [lines["items"]]
        assert lines["items"]["required"] == ["id", "text"]


# ── grounding + word budget on the revise step (run 01a156a6, 2026-10-20) ──

_TODO_GHERKIN = (
    "Scenario: Add a new todo item\n"
    '  When I type "Buy groceries" into the input field\n'
    "  And I press the Enter key\n"
    '  Then I see the text "Buy groceries" displayed on the page'
)


def _grounded_lines() -> list[dict[str, Any]]:
    return [
        {
            "id": cid,
            "text": "First, I open the React TodoMVC demo.",
            "scenario_name": "Add a new todo item",
            "gherkin": _TODO_GHERKIN,
        }
        for cid in ("clip-1", "clip-4")
    ]


@pytest.mark.asyncio
async def test_revise_is_grounded_budgeted_and_refuses_invented_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FAILS BEFORE: the invented clip-1 revision shipped (nothing checked it), and the
    revise prompt carried neither the recording's steps nor a word budget."""

    def _score(draft: Any, **_: Any) -> dict[str, Any]:
        # A revised draft scores higher, so the engine keeps it.
        revised = any("groceries" in str(line.get("text", "")) for line in draft)
        return {
            "composite": 90.0 if revised else 50.0,
            "flags": {"clip-1": ["hook"], "clip-4": ["hook"]},
            "per_line": {},
            "per_story": {},
        }

    _patch_score(monkeypatch, _score)
    invented = (
        "I am about to break the React TodoMVC demo in three specific ways, refactoring "
        "its component tree to eliminate redundant re-renders"
    )
    grounded = 'I type "Buy groceries" into the input field and press Enter to add it.'
    fake = _wm_fake([{"id": "clip-1", "text": invented}, {"id": "clip-4", "text": grounded}])
    stage = ScriptDoctorStage(client=fake)
    ctx = _ctx(sd={"enabled": True, "max_passes": 2}, prior_lines=_grounded_lines())
    ctx.prior_run_outputs["demo_video_gen"]["narration_words_per_beat"] = {
        "clip-1": 25,
        "clip-4": 25,
    }

    out = await stage.handle(stage.input_model(), ctx)

    texts = {line.id: line.text for line in out.lines}
    assert texts["clip-1"] == "First, I open the React TodoMVC demo.", "invented -> unrevised"
    assert texts["clip-4"] == grounded
    revise = next(
        c.kwargs["message"]
        for c in fake.message.call_args_list
        if c.kwargs["worker_name"] == "script-doctor-revise"
    )
    assert '"recording_shows"' in revise and "Buy groceries" in revise
    assert '"word_budget": 25' in revise


def test_revise_prompt_is_unchanged_for_lines_without_scenario_or_budget() -> None:
    msg = sd_mod._revise_msg([{"id": "clip-1", "text": "orig"}], None, [])
    assert "recording_shows" not in msg
    assert "word_budget" not in msg
    assert sd_mod._lines_json([{"id": "clip-1", "text": "orig"}]) in msg


# ── revise must not cut the script under its word band (a film IS its narration) ──


def _wordy(n: int, tag: str) -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


def _always_flagged_improving(draft: Any, **_: Any) -> dict[str, Any]:
    # A shorter draft scores HIGHER, so without the guard the engine keeps the cut.
    words = sum(len(str(line.get("text", "")).split()) for line in draft)
    return {
        "composite": 50.0 - words * 0.1,
        "flags": {"clip-1": ["hook"], "clip-2": ["hook"]},
        "per_line": {},
        "per_story": {},
    }


@pytest.mark.asyncio
async def test_revise_that_cuts_the_script_under_its_floor_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FAILS BEFORE: the revise halved a 40-word script planned at 40 words and the
    shorter (higher-scoring) draft shipped — a film ~50% short of its order."""
    _patch_score(monkeypatch, _always_flagged_improving)
    original = [
        {"id": "clip-1", "text": _wordy(20, "a")},
        {"id": "clip-2", "text": _wordy(20, "b")},
    ]
    fake = _wm_fake([{"id": "clip-1", "text": _wordy(10, "c")}, {"id": "clip-2", "text": "d"}])
    stage = ScriptDoctorStage(client=fake)
    ctx = _ctx(sd={"enabled": True, "max_passes": 2}, prior_lines=original)
    ctx.prior_run_outputs["demo_video_gen"]["narration_words_per_beat"] = {
        "clip-1": 20,
        "clip-2": 20,
    }

    out = await stage.handle(stage.input_model(), ctx)

    assert [line.text for line in out.lines] == [line["text"] for line in original]
    revise = next(
        c.kwargs["message"]
        for c in fake.message.call_args_list
        if c.kwargs["worker_name"] == "script-doctor-revise"
    )
    # 40-word budget ⇒ band floor floor(40 x 0.95) = 38.
    assert "The whole script holds 40 words and must keep at least 38" in revise


@pytest.mark.asyncio
async def test_revise_floor_comes_from_the_ordered_duration_without_beat_budgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No per-beat budgets: the EXPLICIT order at the resolved rate sets the floor."""

    async def _rate(*_: Any, **__: Any) -> float:
        return 120.0

    _patch_score(monkeypatch, _always_flagged_improving)
    monkeypatch.setattr(sd_mod, "resolve_natural_wpm", _rate)
    original = [
        {"id": "clip-1", "text": _wordy(10, "a")},
        {"id": "clip-2", "text": _wordy(10, "b")},
    ]
    fake = _wm_fake([{"id": "clip-1", "text": "c"}, {"id": "clip-2", "text": "d"}])
    stage = ScriptDoctorStage(client=fake)
    ctx = _ctx(sd={"enabled": True}, prior_lines=original)
    ctx.config_snapshot["target_duration_seconds"] = 10  # 10s x 120 wpm = 20 words

    out = await stage.handle(stage.input_model(), ctx)

    assert [line.text for line in out.lines] == [line["text"] for line in original]


@pytest.mark.asyncio
async def test_revise_that_keeps_length_or_grows_a_short_draft_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard refuses LOST length only: a short draft that grows is still revised."""

    def _score(draft: Any, **_: Any) -> dict[str, Any]:
        words = sum(len(str(line.get("text", "")).split()) for line in draft)
        return {
            "composite": float(words),
            "flags": {"clip-1": ["hook"]},
            "per_line": {},
            "per_story": {},
        }

    _patch_score(monkeypatch, _score)
    original = [{"id": "clip-1", "text": _wordy(5, "a")}]
    grown = _wordy(12, "g")
    stage = ScriptDoctorStage(client=_wm_fake([{"id": "clip-1", "text": grown}]))
    ctx = _ctx(sd={"enabled": True}, prior_lines=original)
    ctx.prior_run_outputs["demo_video_gen"]["narration_words_per_beat"] = {"clip-1": 30}

    out = await stage.handle(stage.input_model(), ctx)

    assert [line.text for line in out.lines] == [grown]


def test_no_order_and_no_beat_budgets_states_no_floor() -> None:
    assert sd_mod._script_word_floor({}, {}, 150.0) is None
    assert "must keep at least" not in sd_mod._revise_msg([{"id": "c", "text": "o"}], None, [])
