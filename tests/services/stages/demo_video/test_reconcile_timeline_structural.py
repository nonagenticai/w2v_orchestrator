"""C13 phase 2 — structural timing directives (cut / merge / reallocate).

The Phase-1 relief loop can only SHORTEN over-long narration in place; a beat that
cannot be honestly shortened (claim-bearing / author-locked / footage-pinned) was merely
flagged for an upstream re-run. Phase 2 adds a deterministic STRUCTURAL pre-pass that runs
BEFORE the trim optimizer loop (``ReconcileTimelineStage._apply_structural_prepass``): one
``timing-director`` call proposes ``cut`` / ``merge`` / ``reallocate`` over the currently-
infeasible spine, the pre-pass applies them, and the deterministic solver re-runs over the
REDUCED spine — so the loop-engine's C0 id-set guard (which would trip if an id were removed
mid-loop) never fires.

Contracts pinned here:

* the pre-pass is armed ONLY when at least one infeasible beat cannot be trimmed (structural
  surgery is a last resort); an all-eligible infeasible timeline is left to the trim loop;
* ``cut`` drops an unprotected beat and the freed seconds water-fill the survivors;
* ``cut`` / ``merge`` targeting a PROTECTED beat is rejected-with-log, never applied;
* ``merge`` survivor = the protected beat when exactly one side is protected, else the beat with
  the higher ``_merge_survivor_rank`` — carries an ``expected_text`` (verified-screen) assertion,
  then larger ``demand.ideal``, then earlier spine index as the tiebreak; text is ALWAYS the two
  texts joined in READING (spine) order whichever side survives; the survivor keeps its own
  footage/clip fields; the beat-rewriter is NOT called for a merge;
* ``reallocate`` pins the beat (``time_lock`` + ``seconds``) clamped into its pre-pin
  ``[demand.min, demand.max]``; the solver water-fills the rest;
* the deterministic solver runs FOR REAL — only ``self._wm.message`` is mocked, keyed on
  ``worker_name`` + prompt (the pre-pass call is distinguished by the word ``STRUCTURAL``).
"""

from __future__ import annotations

import logging
import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4, uuid7

import httpx
import pytest
from w2v_common.clients import WMResponse

from src.services.runs.run_context import RunContext
from src.services.runs.structural_relief import maybe_spawn_structural_relief_run
from src.services.stages.demo_video.reconcile_timeline import (
    ReconcileTimelineStage,
    _classify_convergence,
    _TimingConfig,
)
from src.services.stages.shared.time_budget import (
    DEFAULT_WPM_BAND,
    DEFAULT_WPM_NATURAL,
    VO_RATE_DRAG_FLOOR,
    collect_demands,
)

_RELIEF_LOG = "src.services.stages.demo_video.reconcile_timeline"


# ── fixtures/helpers ──────────────────────────────────────────────────────────


def _phrase(tag: str, n: int) -> str:
    """``n``-word narration whose every token is ``tag`` (word_count == n, recognisable)."""
    return " ".join([tag] * n)


def _min_seconds(words: int) -> float:
    """A VO beat's demand floor: ``words / wpm_max(170) * 60``."""
    return words / 170.0 * 60.0


def _max_seconds(words: int) -> float:
    """A VO beat's demand ceiling: ``words / wpm_min(130) * 60``."""
    return words / 130.0 * 60.0


def _clips(seconds: float = 4.0, line_id: str = "clip-2") -> list[dict[str, Any]]:
    return [
        {
            "id": line_id,
            "line_id": line_id,
            "video_url": f"https://minio.test/{line_id}.mp4",
            "seconds": seconds,
        }
    ]


def _relief_timing() -> dict[str, Any]:
    return {"mode": "auto", "relief": {"mode": "auto", "iterations": 2, "score_threshold": 4.0}}


def _ctx(
    *,
    lines: list[dict[str, Any]],
    clips: list[dict[str, Any]] | None = None,
    target_seconds: float,
) -> RunContext:
    cfg: dict[str, Any] = {
        "direction": {"timing": _relief_timing()},
        "deliverable": {"target_duration_seconds": target_seconds},
    }
    prior: dict[str, Any] = {
        "demo_video_gen": {
            "lines": lines,
            "bdd_job_id": "",
            "features_generated": 0,
            "package_url": "",
        },
        "demo_video_record": {"clips": clips or []},
    }
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=cfg,
        prior_run_outputs=prior,
    )


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


def _structural_wm(directives: list[dict[str, Any]], *, pacing_score: int = 5) -> AsyncMock:
    """WM stub for the structural pre-pass, keyed on ``worker_name`` + prompt.

    * ``pacing-critic`` → a rhythm dict scoring ``pacing_score``.
    * ``timing-director`` **structural pre-pass call** (its prompt carries ``STRUCTURAL``) →
      the canned ``directives`` under test; the ``timing-director`` **loop** call (the trim
      optimizer, no ``STRUCTURAL``) → an empty directive set so the loop proposes no trim.
    * ``beat-rewriter`` → the beat's own text truncated to the requested word target (only
      reached if the loop ever trims — none of these tests expect it to).
    """
    fake = AsyncMock()

    def _side(worker_name: str | None = None, message: str = "", **_: Any) -> WMResponse:
        if worker_name == "pacing-critic":
            return _resp({"overall_score": pacing_score})
        if worker_name == "beat-rewriter":
            match = re.search(r"to (\d+) words or fewer", message)
            target = int(match.group(1)) if match else 0
            text = message.partition("words): ")[2]
            return _resp({"text": " ".join(text.split()[:target])})
        # timing-director: the structural pre-pass prompt is the one carrying "STRUCTURAL".
        if "STRUCTURAL" in message:
            return _resp({"directives": directives})
        return _resp({"directives": []})

    fake.message.side_effect = _side
    return fake


# ── (1) cut drops an unprotected beat and the freed time water-fills the rest ─────────


@pytest.mark.asyncio
async def test_cut_drops_unprotected_beat_and_reallocates_time() -> None:
    # An infeasible timeline whose over-share set mixes a claim beat (INELIGIBLE to trim — it
    # arms the structural pre-pass) with two shortenable beats. The director cuts the big
    # unprotected beat; removing it drops Σmin below target, so the solver water-fills the
    # freed seconds into the survivors (both grow above their floor).
    lines: list[dict[str, Any]] = [
        {
            "id": "clip-claim",
            "text": _phrase("claim", 20),
            "seconds": None,
            "clip": None,
            "claim_status": "grounded",
        },
        {"id": "clip-big", "text": _phrase("big", 40), "seconds": None, "clip": None},
        {"id": "clip-small", "text": _phrase("small", 15), "seconds": None, "clip": None},
    ]
    fake = _structural_wm([{"op": "cut", "target_ids": ["clip-big"]}])
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(lines=lines, target_seconds=15.0)

    out = await stage.handle(stage.input_model(), ctx)

    ids = [ln.id for ln in out.lines]
    assert ids == ["clip-claim", "clip-small"]  # clip-big cut, order/ids otherwise intact
    by_id = {ln.id: ln for ln in out.lines}
    # Freed time redistributed: both survivors grew above their word-budget floor.
    assert (by_id["clip-small"].seconds or 0.0) > _min_seconds(15)
    assert (by_id["clip-claim"].seconds or 0.0) > _min_seconds(20)
    # The re-solved (reduced) timeline is FEASIBLE and the loop persisted its manifest.
    assert not any("INFEASIBLE" in note for note in out.notes)
    assert any("timeline_manifest=" in note for note in out.notes)


# ── (2) cut targeting a protected beat is rejected — spine unchanged ───────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("protected", ["locked"])
async def test_cut_rejected_on_protected_beat(
    protected: str, caplog: pytest.LogCaptureFixture
) -> None:
    target_line: dict[str, Any] = {
        "id": "clip-target",
        "text": _phrase("target", 30),
        "seconds": None,
        "clip": None,
    }
    clips: list[dict[str, Any]] = []
    if protected == "footage":
        target_line["text"] = ""
        target_line["clip"] = "clip-target"
        clips = _clips(seconds=4.0, line_id="clip-target")
    elif protected == "locked":
        target_line["locked"] = True
    else:  # claim
        target_line["claim_status"] = "grounded"
    expected_text = target_line["text"]

    # clip-arm (author-locked, over-long) is the ineligible-infeasible beat that ARMS the
    # structural pre-pass; the director then (wrongly) proposes cutting the protected target.
    lines: list[dict[str, Any]] = [
        {
            "id": "clip-arm",
            "text": _phrase("arm", 30),
            "seconds": None,
            "clip": None,
            "locked": True,
        },
        target_line,
    ]
    fake = _structural_wm([{"op": "cut", "target_ids": ["clip-target"]}])
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(lines=lines, clips=clips, target_seconds=8.0)

    with caplog.at_level(logging.WARNING, logger=_RELIEF_LOG):
        out = await stage.handle(stage.input_model(), ctx)

    ids = [ln.id for ln in out.lines]
    assert ids == ["clip-arm", "clip-target"]  # the protected beat is NEVER cut
    by_id = {ln.id: ln for ln in out.lines}
    assert by_id["clip-target"].text == expected_text  # untouched
    assert "protected" in caplog.text  # the rejection is diagnosable, not silent


# ── (3) merge that would drop a protected beat (both sides protected) is rejected ──────


@pytest.mark.asyncio
async def test_merge_rejected_on_protected_target(caplog: pytest.LogCaptureFixture) -> None:
    lines: list[dict[str, Any]] = [
        {
            "id": "clip-a",
            "text": _phrase("aa", 30),
            "seconds": None,
            "clip": None,
            "claim_status": "grounded",
        },
        {"id": "clip-b", "text": _phrase("bb", 30), "seconds": None, "clip": None, "locked": True},
    ]
    fake = _structural_wm([{"op": "merge", "target_ids": ["clip-a", "clip-b"]}])
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(lines=lines, target_seconds=15.0)

    with caplog.at_level(logging.WARNING, logger=_RELIEF_LOG):
        out = await stage.handle(stage.input_model(), ctx)

    ids = [ln.id for ln in out.lines]
    assert ids == ["clip-a", "clip-b"]  # neither protected beat dropped
    by_id = {ln.id: ln for ln in out.lines}
    assert by_id["clip-a"].text == _phrase("aa", 30)
    assert by_id["clip-b"].text == _phrase("bb", 30)
    assert "protected" in caplog.text


# ── (3b) a merge naming the SAME id twice must not self-merge into an unintended cut ───


@pytest.mark.asyncio
async def test_merge_rejected_on_duplicate_target_ids(caplog: pytest.LogCaptureFixture) -> None:
    # A small model can echo the same id twice. ["clip-dup","clip-dup"] must NOT collapse
    # into a self-merge that silently deletes the beat — it dedupes to one distinct id and
    # is rejected like any under-specified merge; the beat (and its content) survives.
    lines: list[dict[str, Any]] = [
        {
            "id": "clip-arm",
            "text": _phrase("arm", 30),
            "seconds": None,
            "clip": None,
            "claim_status": "grounded",
        },
        {"id": "clip-dup", "text": _phrase("dup", 30), "seconds": None, "clip": None},
    ]
    fake = _structural_wm([{"op": "merge", "target_ids": ["clip-dup", "clip-dup"]}])
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(lines=lines, target_seconds=15.0)

    with caplog.at_level(logging.WARNING, logger=_RELIEF_LOG):
        out = await stage.handle(stage.input_model(), ctx)

    ids = [ln.id for ln in out.lines]
    assert "clip-dup" in ids  # the beat is NOT dropped by a duplicate-id "merge"
    assert {ln.id for ln in out.lines} == {"clip-arm", "clip-dup"}
    assert "DISTINCT" in caplog.text


# ── (4a) merge survivor is the PROTECTED beat when exactly one side is protected ───────


@pytest.mark.asyncio
async def test_merge_survivor_is_protected_when_one_side_protected() -> None:
    lines: list[dict[str, Any]] = [
        {
            "id": "clip-claim",
            "text": _phrase("claim", 30),
            "seconds": None,
            "clip": None,
            "claim_status": "grounded",
        },
        {"id": "clip-plain", "text": _phrase("plain", 30), "seconds": None, "clip": None},
    ]
    fake = _structural_wm([{"op": "merge", "target_ids": ["clip-claim", "clip-plain"]}])
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(lines=lines, target_seconds=15.0)

    out = await stage.handle(stage.input_model(), ctx)

    ids = [ln.id for ln in out.lines]
    assert ids == ["clip-claim"]  # the protected (claim) beat SURVIVES; the plain one is dropped
    # Text = the two texts joined in reading order, space-joined, stripped.
    assert out.lines[0].text == f"{_phrase('claim', 30)} {_phrase('plain', 30)}"


# ── (4b) neither side protected ⇒ survivor is chosen by QUALITY RANK, position is the tiebreak ─
#
# ``_merge_survivor_rank`` orders an unprotected pair on
# ``(carries expected_text, demand.ideal, -spine index)`` — highest wins. The three tests below
# pin one key each: the positional tiebreak (quality ties ⇒ unchanged, earlier survives), the
# verified-screen assertion, and the larger wanted-seconds. In EVERY case the merged text stays
# in READING (spine) order — the survivor changing must never reorder narration.


@pytest.mark.asyncio
async def test_merge_survivor_is_earlier_index_when_quality_keys_tie() -> None:
    # clip-arm (locked) merely arms the pre-pass; the director merges the two plain beats,
    # naming them in REVERSE (y before x). clip-x and clip-y are indistinguishable on quality
    # (same word count ⇒ same demand.ideal, neither carries expected_text), so the rank falls
    # through to its final key — spine index — and the EARLIER beat survives, exactly as the
    # pre-quality positional rule did. Text is joined in READING order, not directive order.
    lines: list[dict[str, Any]] = [
        {
            "id": "clip-arm",
            "text": _phrase("arm", 30),
            "seconds": None,
            "clip": None,
            "locked": True,
        },
        {"id": "clip-x", "text": _phrase("xx", 30), "seconds": None, "clip": None},
        {"id": "clip-y", "text": _phrase("yy", 30), "seconds": None, "clip": None},
    ]
    fake = _structural_wm([{"op": "merge", "target_ids": ["clip-y", "clip-x"]}])
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(lines=lines, target_seconds=20.0)

    out = await stage.handle(stage.input_model(), ctx)

    ids = [ln.id for ln in out.lines]
    assert ids == ["clip-arm", "clip-x"]  # tie ⇒ earlier-index survivor kept, clip-y dropped
    by_id = {ln.id: ln for ln in out.lines}
    assert by_id["clip-x"].text == f"{_phrase('xx', 30)} {_phrase('yy', 30)}"  # reading order


@pytest.mark.asyncio
async def test_merge_survivor_is_verified_screen_beat_over_earlier_beat() -> None:
    # clip-y is the LATER beat but carries an O1b ``expected_text`` OCR assertion — it is the
    # beat whose recorded screen is verified. Dropping it would discard the pair's only
    # wrong-screen check purely because of position, so it must SURVIVE over the earlier
    # clip-x. The narration is still concatenated in reading order (x before y).
    lines: list[dict[str, Any]] = [
        {
            "id": "clip-arm",
            "text": _phrase("arm", 30),
            "seconds": None,
            "clip": None,
            "locked": True,
        },
        {"id": "clip-x", "text": _phrase("xx", 30), "seconds": None, "clip": None},
        {
            "id": "clip-y",
            "text": _phrase("yy", 30),
            "seconds": None,
            "clip": None,
            "expected_text": "Dashboard",
        },
    ]
    fake = _structural_wm([{"op": "merge", "target_ids": ["clip-x", "clip-y"]}])
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(lines=lines, target_seconds=20.0)

    out = await stage.handle(stage.input_model(), ctx)

    ids = [ln.id for ln in out.lines]
    assert ids == ["clip-arm", "clip-y"]  # verified-screen beat survives, earlier clip-x dropped
    by_id = {ln.id: ln for ln in out.lines}
    # Survivor changed, narration order did NOT: still x's text then y's text.
    assert by_id["clip-y"].text == f"{_phrase('xx', 30)} {_phrase('yy', 30)}"
    assert by_id["clip-y"].expected_text == "Dashboard"  # the assertion is retained, not dropped


@pytest.mark.asyncio
async def test_merge_survivor_is_larger_ideal_demand_over_earlier_beat() -> None:
    # Neither beat carries expected_text, so the rank falls to demand.ideal: clip-y is a 40-word
    # beat, clip-x a 20-word one, so the solver wants materially more time for clip-y — it is
    # the substantive beat and must survive even though clip-x is earlier in the spine.
    lines: list[dict[str, Any]] = [
        {
            "id": "clip-arm",
            "text": _phrase("arm", 30),
            "seconds": None,
            "clip": None,
            "locked": True,
        },
        {"id": "clip-x", "text": _phrase("xx", 20), "seconds": None, "clip": None},
        {"id": "clip-y", "text": _phrase("yy", 40), "seconds": None, "clip": None},
    ]
    fake = _structural_wm([{"op": "merge", "target_ids": ["clip-x", "clip-y"]}])
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(lines=lines, target_seconds=20.0)

    out = await stage.handle(stage.input_model(), ctx)

    ids = [ln.id for ln in out.lines]
    assert ids == ["clip-arm", "clip-y"]  # larger-ideal beat survives, earlier clip-x dropped
    by_id = {ln.id: ln for ln in out.lines}
    assert by_id["clip-y"].text == f"{_phrase('xx', 20)} {_phrase('yy', 40)}"  # reading order


# ── (5) merge keeps the survivor's footage/clip fields; the beat-rewriter is NOT called ─


@pytest.mark.asyncio
async def test_merge_concatenates_text_reading_order_keeps_survivor_footage() -> None:
    # Merge a footage-pinned beat (protected ⇒ survivor) with a plain VO beat: the survivor
    # keeps its ``clip`` field, absorbs the VO's narration text (footage text is empty, so the
    # concatenation is the VO text), and the merge NEVER fires a beat-rewriter call.
    lines: list[dict[str, Any]] = [
        {
            "id": "clip-arm",
            "text": _phrase("arm", 30),
            "seconds": None,
            "clip": None,
            "locked": True,
        },
        {"id": "clip-foot", "text": "", "seconds": 5.0, "clip": "clip-foot"},
        {"id": "clip-vo", "text": "Narration for the footage.", "seconds": None, "clip": None},
    ]
    fake = _structural_wm([{"op": "merge", "target_ids": ["clip-foot", "clip-vo"]}])
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(lines=lines, clips=_clips(seconds=4.0, line_id="clip-foot"), target_seconds=15.0)

    out = await stage.handle(stage.input_model(), ctx)

    ids = [ln.id for ln in out.lines]
    assert ids == ["clip-arm", "clip-foot"]  # footage survivor kept, VO beat dropped
    by_id = {ln.id: ln for ln in out.lines}
    assert by_id["clip-foot"].clip == "clip-foot"  # survivor's footage/clip fields preserved
    assert by_id["clip-foot"].text == "Narration for the footage."  # "" + " " + vo, stripped
    workers = {c.kwargs["worker_name"] for c in fake.message.call_args_list}
    assert "beat-rewriter" not in workers  # a merge never rewrites text


# ── (6) reallocate pins the beat; the solver water-fills the rest ─────────────────────


@pytest.mark.asyncio
async def test_reallocate_pins_beat_then_water_fills_rest() -> None:
    # clip-claim arms the pre-pass; the director cuts the big beat to reach feasibility AND
    # reallocates (pins) clip-pin at 10.0s. The pinned beat holds EXACTLY 10.0s and the
    # remaining elastic budget (target - pin) water-fills into clip-claim.
    lines: list[dict[str, Any]] = [
        {
            "id": "clip-claim",
            "text": _phrase("claim", 35),
            "seconds": None,
            "clip": None,
            "claim_status": "grounded",
        },
        {"id": "clip-cut", "text": _phrase("cut", 40), "seconds": None, "clip": None},
        {"id": "clip-pin", "text": _phrase("pin", 25), "seconds": None, "clip": None},
    ]
    fake = _structural_wm(
        [
            {"op": "cut", "target_ids": ["clip-cut"]},
            {"op": "reallocate", "target_ids": ["clip-pin"], "seconds": 10.0},
        ]
    )
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(lines=lines, target_seconds=24.0)

    out = await stage.handle(stage.input_model(), ctx)

    ids = [ln.id for ln in out.lines]
    assert "clip-cut" not in ids
    assert ids == ["clip-claim", "clip-pin"]
    by_id = {ln.id: ln for ln in out.lines}
    assert (by_id["clip-pin"].seconds or 0.0) == pytest.approx(10.0)  # pinned at the directive
    # The rest water-fills to (target - pin): the single elastic survivor takes the balance.
    assert (by_id["clip-claim"].seconds or 0.0) == pytest.approx(24.0 - 10.0, abs=0.1)
    assert not any("INFEASIBLE" in note for note in out.notes)


# ── (7) a reallocate pin is CLAMPED into the beat's pre-pin demand bounds ──────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pin_seconds, expect",
    [(1.0, "min")],
)
async def test_reallocate_pin_clamped_into_demand_bounds(pin_seconds: float, expect: str) -> None:
    # A pin below the beat's floor / above its ceiling is clamped into [demand.min, demand.max]
    # so it can never invert the solver's repair floor into an unsayable window.
    lines: list[dict[str, Any]] = [
        {
            "id": "clip-arm",
            "text": _phrase("arm", 30),
            "seconds": None,
            "clip": None,
            "claim_status": "grounded",
        },
        {"id": "clip-pin", "text": _phrase("pin", 25), "seconds": None, "clip": None},
    ]
    fake = _structural_wm(
        [{"op": "reallocate", "target_ids": ["clip-pin"], "seconds": pin_seconds}]
    )
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(lines=lines, target_seconds=12.0)

    out = await stage.handle(stage.input_model(), ctx)

    by_id = {ln.id: ln for ln in out.lines}
    expected = _min_seconds(25) if expect == "min" else _max_seconds(25)
    assert (by_id["clip-pin"].seconds or 0.0) == pytest.approx(expected)


# ── (8) prepass-then-optimizer: a cut never trips the loop-engine C0 id-set guard ─────


@pytest.mark.asyncio
async def test_prepass_then_optimizer_c0_never_trips() -> None:
    # A cut removes clip-cut in the PRE-pass, then the (still-infeasible) trim loop runs its
    # real run_evaluator_optimizer over the REDUCED spine. Because ``_spine_ids`` is derived
    # AFTER the pre-pass, the loop's C0 id-set guard sees a stable id-set and never raises
    # ValueError — the whole handle() completes normally with clip-cut gone.
    lines: list[dict[str, Any]] = [
        {
            "id": "clip-claim",
            "text": _phrase("claim", 30),
            "seconds": None,
            "clip": None,
            "claim_status": "grounded",
        },
        {"id": "clip-cut", "text": _phrase("cut", 40), "seconds": None, "clip": None},
        {"id": "clip-elastic", "text": _phrase("el", 40), "seconds": None, "clip": None},
    ]
    fake = _structural_wm([{"op": "cut", "target_ids": ["clip-cut"]}])
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(lines=lines, target_seconds=12.0)

    out = await stage.handle(stage.input_model(), ctx)  # must NOT raise ValueError

    ids = [ln.id for ln in out.lines]
    assert "clip-cut" not in ids
    assert ids == ["clip-claim", "clip-elastic"]


# ── (7) the STRUCTURAL director call is offered ONLY cut/merge/reallocate (never trim) ──


def _op_enum(schema: dict[str, Any]) -> list[str]:
    """Pull the ``op`` enum out of a timing-director directive schema."""
    return schema["properties"]["directives"]["items"]["properties"]["op"]["enum"]


@pytest.mark.asyncio
async def test_structural_director_schema_excludes_trim() -> None:
    # Regression guard: the structural pre-pass shares NEITHER schema NOR op-set with the trim
    # loop. If the structural director is ever handed a schema that advertises ``trim`` again,
    # guided decoding periodically emits a ``trim`` the pre-pass silently rejects — a paid WM
    # pass spent for zero relief (observed live). Assert the ``output_schema`` passed on the
    # STRUCTURAL call offers exactly cut/merge/reallocate and NOT trim.
    lines: list[dict[str, Any]] = [
        {
            "id": "clip-claim",
            "text": _phrase("claim", 20),
            "seconds": None,
            "clip": None,
            "claim_status": "grounded",
        },
        {"id": "clip-big", "text": _phrase("big", 40), "seconds": None, "clip": None},
        {"id": "clip-small", "text": _phrase("small", 15), "seconds": None, "clip": None},
    ]
    fake = _structural_wm([{"op": "cut", "target_ids": ["clip-big"]}])
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(lines=lines, target_seconds=15.0)

    await stage.handle(stage.input_model(), ctx)

    structural_calls = [
        c for c in fake.message.call_args_list if "STRUCTURAL" in (c.kwargs.get("message") or "")
    ]
    assert structural_calls, "the structural pre-pass director call must have fired"
    for call in structural_calls:
        enum = _op_enum(call.kwargs["output_schema"])
        assert "trim" not in enum
        assert set(enum) == {"cut", "merge", "reallocate"}


# ── (8) convergence classification + structural telemetry on the typed manifest (PR-B) ──


def _footage_pinned_ctx(target_seconds: float) -> RunContext:
    """Two footage-pinned clips whose measured seconds overrun the target — a PURE
    footage-pinned infeasible timeline: every over-budget beat is protected, so it provably
    cannot converge in place (the ``cant_converge`` / degrade case).

    30s each (was 12s): ``trim_footage_tails`` now cuts footage tails down to 60% of the
    clip, so 2 x 12s (floors 2 x 7.2s) FITS a 15s order. 2 x 30s keeps both floors (18s)
    over it on their own, which is the shape these tests exist for."""
    lines: list[dict[str, Any]] = [
        {"id": "clip-1", "text": "", "seconds": None, "clip": "clip-1"},
        {"id": "clip-2", "text": "", "seconds": None, "clip": "clip-2"},
    ]
    clips = [
        {
            "id": "clip-1",
            "line_id": "clip-1",
            "video_url": "https://minio.test/1.mp4",
            "seconds": 30.0,
        },
        {
            "id": "clip-2",
            "line_id": "clip-2",
            "video_url": "https://minio.test/2.mp4",
            "seconds": 30.0,
        },
    ]
    return _ctx(lines=lines, clips=clips, target_seconds=target_seconds)


@pytest.mark.asyncio
async def test_mixed_timeline_converged_outcome_and_patch_telemetry() -> None:
    # The cut scenario (a claim beat arms the pre-pass; clip-big is cut) reaches feasibility
    # in place. The typed manifest must report it: converged, one structural patch applied,
    # no gen re-run requested.
    lines: list[dict[str, Any]] = [
        {
            "id": "clip-claim",
            "text": _phrase("claim", 20),
            "seconds": None,
            "clip": None,
            "claim_status": "grounded",
        },
        {"id": "clip-big", "text": _phrase("big", 40), "seconds": None, "clip": None},
        {"id": "clip-small", "text": _phrase("small", 15), "seconds": None, "clip": None},
    ]
    fake = _structural_wm([{"op": "cut", "target_ids": ["clip-big"]}])
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(lines=lines, target_seconds=15.0)

    out = await stage.handle(stage.input_model(), ctx)

    assert out.timeline_manifest is not None
    run = out.timeline_manifest.run
    assert run.convergence_outcome == "converged"
    assert run.structural_relief_requested is False
    assert run.overrun_seconds == pytest.approx(0.0)
    assert run.structural_patches_applied == 1  # the one cut
    assert run.infeasible_beat_ids == []


@pytest.mark.asyncio
async def test_footage_pinned_timeline_cant_converge_and_degrades() -> None:
    # A pure footage-pinned infeasible timeline. The director (wrongly) proposes cutting a
    # protected clip; the pre-pass rejects it (protected), the loop cannot shorten a footage
    # beat, so the run stays infeasible. The manifest must classify this HONESTLY as
    # cant_converge, request structural relief, and surface the rejected op — and it must NOT
    # loop or raise.
    fake = _structural_wm([{"op": "cut", "target_ids": ["clip-1"]}])
    stage = ReconcileTimelineStage(client=fake)
    ctx = _footage_pinned_ctx(target_seconds=15.0)

    out = await stage.handle(stage.input_model(), ctx)  # must NOT raise

    assert out.timeline_manifest is not None
    run = out.timeline_manifest.run
    assert run.convergence_outcome == "cant_converge"
    assert run.structural_relief_requested is True
    assert run.overrun_seconds > 0.0
    assert run.structural_patches_applied == 0  # the protected cut was rejected
    assert "clip-1" in run.structural_rejected_ids
    # The human-facing degrade note is present (auto-marks the run degraded downstream).
    assert any("STRUCTURAL RELIEF NEEDED" in note for note in out.notes)


@pytest.mark.asyncio
async def test_structural_wm_error_surfaces_soft_degrade_note() -> None:
    # When the timing-director call ERRORS, the pre-pass soft-degrades (spine unchanged) — a
    # previously SILENT fallback. The manifest note must now surface it so a live no-op is
    # visible in the run record, not only the pod log. Still no raise, still honest classify.
    fake = AsyncMock()

    def _side(worker_name: str | None = None, message: str = "", **_: Any) -> WMResponse:
        if worker_name == "pacing-critic":
            return _resp({"overall_score": 5})
        if "STRUCTURAL" in message:
            raise RuntimeError("worker-management unreachable")
        return _resp({"directives": []})

    fake.message.side_effect = _side
    stage = ReconcileTimelineStage(client=fake)
    ctx = _footage_pinned_ctx(target_seconds=15.0)

    out = await stage.handle(stage.input_model(), ctx)  # must NOT raise

    assert any("structural relief unavailable" in note for note in out.notes)
    assert out.timeline_manifest is not None
    # Spine unchanged + still infeasible → honest cant_converge, zero patches.
    assert out.timeline_manifest.run.convergence_outcome == "cant_converge"
    assert out.timeline_manifest.run.structural_patches_applied == 0


@pytest.mark.asyncio
async def test_reallocate_surfaces_in_manifest_telemetry() -> None:
    # A reallocate pin on the armed timeline is surfaced as a reallocated id + counts as a patch.
    lines: list[dict[str, Any]] = [
        {
            "id": "clip-claim",
            "text": _phrase("claim", 20),
            "seconds": None,
            "clip": None,
            "claim_status": "grounded",
        },
        {"id": "clip-big", "text": _phrase("big", 40), "seconds": None, "clip": None},
        {"id": "clip-small", "text": _phrase("small", 15), "seconds": None, "clip": None},
    ]
    fake = _structural_wm([{"op": "reallocate", "target_ids": ["clip-small"], "seconds": 2.0}])
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(lines=lines, target_seconds=15.0)

    out = await stage.handle(stage.input_model(), ctx)

    assert out.timeline_manifest is not None
    run = out.timeline_manifest.run
    assert "clip-small" in run.structural_reallocated_ids
    assert run.structural_patches_applied >= 1


@pytest.mark.asyncio
async def test_reallocate_then_cut_same_id_not_double_counted() -> None:
    # A director that emits BOTH a reallocate AND a cut on the same id: the beat is removed
    # from the spine, so it must NOT also appear as a reallocated id (it no longer exists) and
    # must NOT be counted twice in structural_patches_applied. Telemetry-accuracy regression.
    lines: list[dict[str, Any]] = [
        {
            "id": "clip-claim",
            "text": _phrase("claim", 20),
            "seconds": None,
            "clip": None,
            "claim_status": "grounded",
        },
        {"id": "clip-x", "text": _phrase("x", 40), "seconds": None, "clip": None},
        {"id": "clip-keep", "text": _phrase("keep", 15), "seconds": None, "clip": None},
    ]
    fake = _structural_wm(
        [
            {"op": "reallocate", "target_ids": ["clip-x"], "seconds": 2.0},
            {"op": "cut", "target_ids": ["clip-x"]},
        ]
    )
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(lines=lines, target_seconds=15.0)

    out = await stage.handle(stage.input_model(), ctx)

    assert out.timeline_manifest is not None
    run = out.timeline_manifest.run
    ids = [ln.id for ln in out.lines]
    assert "clip-x" not in ids  # the cut removed it
    assert "clip-x" not in run.structural_reallocated_ids  # so it is not "reallocated"
    # Exactly one net structural patch (the cut), never two.
    assert run.structural_patches_applied == 1


# ── (9) _classify_convergence — the Σ(eligible mins) >= overrun feasibility gate (PR-C fix) ─
#
# Regression: `progress` is not "≥1 eligible beat exists" — it is "cutting the ELIGIBLE
# (unprotected) beats can ACTUALLY close the overrun". When a footage/claim/locked FLOOR
# dominates, the small elastic beats cannot cover `Σmin - target`, so the honest outcome is
# `cant_converge` (a gen re-run that drops PROTECTED beats is the only lever) — NOT the
# over-optimistic `progress` that would loop the in-stage trimmer to no effect.


def _cfg(target: float) -> _TimingConfig:
    return _TimingConfig(
        mode="auto",
        target=target,
        tolerance_pct=2.0,
        wpm_band=DEFAULT_WPM_BAND,
        wpm_natural=DEFAULT_WPM_NATURAL,
        freeze_pad_ceiling_seconds=1.5,
        vo_rate_drag_floor=VO_RATE_DRAG_FLOOR,
        relief_mode="auto",
        relief_iterations=2,
        relief_threshold=4.0,
    )


def _classify(lines: list[dict[str, Any]], clip_seconds: dict[str, float], target: float) -> str:
    """Build the demand vector the stage builds, then classify — exactly as handle() does."""
    cfg = _cfg(target)
    demands = collect_demands(
        lines, clip_seconds=clip_seconds, wpm_band=cfg.wpm_band, wpm_natural=cfg.wpm_natural
    )
    return _classify_convergence(lines, demands, cfg)


def test_classify_footage_dominated_floor_is_cant_converge() -> None:
    # One pinned 60s footage beat (target 30) + two tiny elastic VO beats (min ~3s each).
    # The VO beats ARE eligible (no claim/lock, not footage), but overrun = 66-30 = 36 and the
    # removable eligible floor is only ~6s < 36 — cutting them can never close the gap. Was
    # WRONGLY `progress` before the fix (eligible-non-empty ⇒ progress); now `cant_converge`.
    lines: list[dict[str, Any]] = [
        {"id": "clip-foot", "text": "", "seconds": None, "clip": "clip-foot"},
        {"id": "clip-vo1", "text": _phrase("vo", 9), "seconds": None, "clip": None},
        {"id": "clip-vo2", "text": _phrase("vo", 9), "seconds": None, "clip": None},
    ]
    assert _classify(lines, {"clip-foot": 60.0}, 30.0) == "cant_converge"


def test_classify_vo_dominated_is_progress() -> None:
    # Two elastic VO beats whose mins (~25s each) exceed the overrun: Σmin ~50, target 30,
    # overrun ~20, removable ~50 >= 20 — cutting the eligible beats CAN reach feasibility.
    lines: list[dict[str, Any]] = [
        {"id": "clip-a", "text": _phrase("aa", 71), "seconds": None, "clip": None},
        {"id": "clip-b", "text": _phrase("bb", 71), "seconds": None, "clip": None},
    ]
    assert _classify(lines, {}, 30.0) == "progress"


def test_classify_feasible_is_converged() -> None:
    # Σmin (~6s) well under the 30s target ⇒ not infeasible at all.
    lines: list[dict[str, Any]] = [
        {"id": "clip-a", "text": _phrase("aa", 9), "seconds": None, "clip": None},
        {"id": "clip-b", "text": _phrase("bb", 9), "seconds": None, "clip": None},
    ]
    assert _classify(lines, {}, 30.0) == "converged"


def test_classify_all_protected_infeasible_is_cant_converge() -> None:
    # A lone footage-pinned beat over budget: no eligible beat exists at all ⇒ cant_converge.
    lines: list[dict[str, Any]] = [
        {"id": "clip-foot", "text": "", "seconds": None, "clip": "clip-foot"},
    ]
    assert _classify(lines, {"clip-foot": 60.0}, 30.0) == "cant_converge"


def test_classify_time_lock_floor_is_cant_converge() -> None:
    # Two authored ``time_lock`` beats pinned at 10s each (collect_demands pins them
    # min==ideal==max==authored seconds, INDEPENDENT of word count) sum to 20s > the 15s
    # target, plus one tiny elastic VO beat. time_lock beats are non-footage yet PROTECTED
    # (trimming their text cannot move their pinned seconds) — so the irreducible protected
    # floor (20s) already exceeds target and no in-place op can converge. Regression: before
    # time_lock was added to the protected floor these beats were invisible to
    # _classify_convergence, the protected floor read 0, and the timeline was WRONGLY
    # `progress` (looping the trimmer over an immovable spine). Now `cant_converge`.
    lines: list[dict[str, Any]] = [
        {"id": "lock-1", "text": "", "seconds": 10.0, "clip": None, "time_lock": True},
        {"id": "lock-2", "text": "", "seconds": 10.0, "clip": None, "time_lock": True},
        {"id": "clip-vo", "text": _phrase("vo", 9), "seconds": None, "clip": None},
    ]
    assert _classify(lines, {}, 15.0) == "cant_converge"


# ── (10) the positional OPENER (cold-open) is never structurally cut or merge-dropped ──
#
# Only gen.py's ``_apply_target_beats`` lever protects index 0; this in-place structural
# pass had no such guard, so a director ``cut`` / ``merge`` could silently drop the opening
# beat. The opener = the earliest-order LIVE beat (min spine index), not a protected flag.


@pytest.mark.asyncio
async def test_cut_rejected_on_opening_beat(caplog: pytest.LogCaptureFixture) -> None:
    # clip-arm (author-locked, over-long) is the ineligible-infeasible beat that ARMS the
    # structural pre-pass (an all-trim-eligible infeasible spine skips the pre-pass entirely).
    # clip-open is a small, UNPROTECTED elastic opener at index 0. The director wrongly proposes
    # cutting the opener; the guard rejects it — the positional cold-open must never be
    # structurally dropped, even when unprotected. The state spans the full spine, so index 0 is
    # the opener regardless of which beats are flagged infeasible.
    lines: list[dict[str, Any]] = [
        {"id": "clip-open", "text": _phrase("open", 9), "seconds": None, "clip": None},
        {
            "id": "clip-arm",
            "text": _phrase("arm", 30),
            "seconds": None,
            "clip": None,
            "locked": True,
        },
    ]
    fake = _structural_wm([{"op": "cut", "target_ids": ["clip-open"]}])
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(lines=lines, target_seconds=8.0)

    with caplog.at_level(logging.WARNING, logger=_RELIEF_LOG):
        out = await stage.handle(stage.input_model(), ctx)

    ids = [ln.id for ln in out.lines]
    assert ids == ["clip-open", "clip-arm"]  # the opener is NEVER cut
    assert "opening beat" in caplog.text  # the rejection is diagnosable, not silent
    assert out.timeline_manifest is not None
    run = out.timeline_manifest.run
    assert "clip-open" in run.structural_rejected_ids  # surfaced in telemetry
    assert run.structural_patches_applied == 0  # the opener cut was rejected, no patch applied


@pytest.mark.asyncio
async def test_merge_never_drops_opening_beat() -> None:
    # clip-claim (claim, over-long) arms the pre-pass. The director merges the OPENER (clip-open,
    # index 0) with a later plain beat (clip-late), naming them in REVERSE — neither is
    # claim/lock protected. The opener must SURVIVE as the retained beat, the other is dropped,
    # and the merged text lands on the opener in READING (spine) order.
    lines: list[dict[str, Any]] = [
        {"id": "clip-open", "text": _phrase("open", 20), "seconds": None, "clip": None},
        {
            "id": "clip-claim",
            "text": _phrase("claim", 30),
            "seconds": None,
            "clip": None,
            "claim_status": "grounded",
        },
        {"id": "clip-late", "text": _phrase("late", 20), "seconds": None, "clip": None},
    ]
    fake = _structural_wm([{"op": "merge", "target_ids": ["clip-late", "clip-open"]}])
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(lines=lines, target_seconds=15.0)

    out = await stage.handle(stage.input_model(), ctx)

    ids = [ln.id for ln in out.lines]
    assert ids == ["clip-open", "clip-claim"]  # opener survives, clip-late dropped
    by_id = {ln.id: ln for ln in out.lines}
    # Merged text on the opener, joined in reading order (open before late), not directive order.
    assert by_id["clip-open"].text == f"{_phrase('open', 20)} {_phrase('late', 20)}"


@pytest.mark.asyncio
async def test_merge_opener_with_protected_beat_is_rejected(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The director merges the UNPROTECTED opener (clip-open, index 0) with a PROTECTED beat
    # (clip-claim). A merge drops exactly one beat; here neither can be dropped — keeping the
    # opener would drop the protected beat, keeping the protected beat would drop the opener. The
    # merge is unsatisfiable, so it must be REJECTED (both survive), not resolved by dropping the
    # protected beat. Regression guard: an earlier opener-survival cut dropped the protected side.
    lines: list[dict[str, Any]] = [
        {"id": "clip-open", "text": _phrase("open", 20), "seconds": None, "clip": None},
        {
            "id": "clip-claim",
            "text": _phrase("claim", 30),
            "seconds": None,
            "clip": None,
            "claim_status": "grounded",
        },
    ]
    fake = _structural_wm([{"op": "merge", "target_ids": ["clip-open", "clip-claim"]}])
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(lines=lines, target_seconds=8.0)

    with caplog.at_level(logging.WARNING, logger=_RELIEF_LOG):
        out = await stage.handle(stage.input_model(), ctx)

    ids = [ln.id for ln in out.lines]
    assert ids == ["clip-open", "clip-claim"]  # both survive — the merge was rejected
    by_id = {ln.id: ln for ln in out.lines}
    assert by_id["clip-claim"].text == _phrase("claim", 30)  # protected beat untouched
    assert by_id["clip-open"].text == _phrase("open", 20)  # opener text not concatenated
    assert "opening beat" in caplog.text  # rejection is diagnosable
    assert out.timeline_manifest is not None
    assert out.timeline_manifest.run.structural_patches_applied == 0


# ── (11) protected_beat_count telemetry (PR-B floor) + 3+-id merge discards the extras ──
#
# The auto-converge hook clamps a gen re-run's ``target_beats`` to the un-cuttable floor —
# footage-pinned / claim-bearing / author-locked / time_lock beats gen physically cannot cut.
# The manifest must carry that count. And a merge that names 3+ ids fuses exactly two; the
# extras must be recorded in ``structural_rejected_ids`` + logged, not silently discarded.


@pytest.mark.asyncio
async def test_manifest_carries_protected_beat_count_footage_pinned() -> None:
    # A pure footage-pinned infeasible timeline: BOTH beats are footage-pinned ⇒ protected. The
    # re-solved manifest must report protected_beat_count == 2 — the count of beats no
    # IN-PLACE op in THIS stage may drop. (It is telemetry only: the PR-C spawn hook no
    # longer clamps target_beats to it, because gen._apply_target_beats CAN drop a
    # footage-pinned beat — see structural_relief._next_target_beats.) The director's cut of
    # a protected clip is rejected here, so the spine stays at 2.
    fake = _structural_wm([{"op": "cut", "target_ids": ["clip-1"]}])
    stage = ReconcileTimelineStage(client=fake)
    ctx = _footage_pinned_ctx(target_seconds=15.0)

    out = await stage.handle(stage.input_model(), ctx)

    assert out.timeline_manifest is not None
    assert out.timeline_manifest.run.protected_beat_count == 2


@pytest.mark.asyncio
async def test_merge_three_ids_discards_extras_and_records_reject(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A small model can name 3+ ids on one merge. A merge fuses exactly TWO; the surplus ids must
    # be DISCARDED with a log AND recorded into structural_rejected_ids (the same channel a
    # rejected cut uses) — never silently dropped. The spine stays valid: the first two fuse
    # (earlier-index survivor, reading-order text) and the surplus beat survives untouched.
    lines: list[dict[str, Any]] = [
        {
            "id": "clip-arm",
            "text": _phrase("arm", 30),
            "seconds": None,
            "clip": None,
            "locked": True,
        },
        {"id": "clip-x", "text": _phrase("xx", 30), "seconds": None, "clip": None},
        {"id": "clip-y", "text": _phrase("yy", 30), "seconds": None, "clip": None},
        {"id": "clip-z", "text": _phrase("zz", 30), "seconds": None, "clip": None},
    ]
    fake = _structural_wm([{"op": "merge", "target_ids": ["clip-x", "clip-y", "clip-z"]}])
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(lines=lines, target_seconds=8.0)

    with caplog.at_level(logging.WARNING, logger=_RELIEF_LOG):
        out = await stage.handle(stage.input_model(), ctx)

    ids = [ln.id for ln in out.lines]
    assert ids == ["clip-arm", "clip-x", "clip-z"]  # x,y fused → x kept; z (the extra) survives
    by_id = {ln.id: ln for ln in out.lines}
    assert by_id["clip-x"].text == f"{_phrase('xx', 30)} {_phrase('yy', 30)}"  # reading order
    assert by_id["clip-z"].text == _phrase("zz", 30)  # the discarded surplus beat is untouched
    assert "discarded" in caplog.text  # the extra-id discard is diagnosable, not silent
    assert out.timeline_manifest is not None
    assert "clip-z" in out.timeline_manifest.run.structural_rejected_ids


# ── (11) the cant_converge DEGRADE NOTE reaches the user (regression, orch #8xx) ──────────
#
# The note used to be gated on ``result.rerun_stages``. That flag is only set when a
# *flagged* beat is ineligible, and ``flagged`` excludes pinned beats — so a footage-DOMINATED
# MIXED spine (footage floor > target + a few elastic VO beats) left it FALSE and shipped an
# over-target picture with the user told NOTHING, while the spawn policy had already stopped
# trusting the same flag. The gate is now the convergence OUTCOME. These tests assert the
# user-visible ``notes`` field, not an internal boolean — the old tests asserted only the
# outcome LABEL, which is why the silent degrade shipped green.

_CANT_CONVERGE_MARKER = "STRUCTURAL RELIEF NEEDED"


def _footage_dominated_mixed_ctx(target_seconds: float) -> RunContext:
    """One pinned 60s footage beat + two tiny elastic VO beats — the MIXED spine whose
    protected floor alone blows the target. ``_classify_convergence`` calls it
    ``cant_converge``; the relief loop leaves ``rerun_stages`` EMPTY (the elastic beats are
    eligible, so nothing gets flagged-and-ineligible)."""
    lines: list[dict[str, Any]] = [
        {"id": "clip-foot", "text": "", "seconds": None, "clip": "clip-foot"},
        {"id": "clip-vo1", "text": _phrase("vo", 9), "seconds": None, "clip": None},
        {"id": "clip-vo2", "text": _phrase("vo", 9), "seconds": None, "clip": None},
    ]
    clips = [
        {
            "id": "clip-foot",
            "line_id": "clip-foot",
            "video_url": "https://minio.test/foot.mp4",
            "seconds": 60.0,
        }
    ]
    return _ctx(lines=lines, clips=clips, target_seconds=target_seconds)


@pytest.mark.asyncio
async def test_footage_dominated_cant_converge_still_notes_the_user() -> None:
    # REGRESSION: cant_converge with a FALSY rerun_stages. Before the fix this run shipped
    # over target with an empty degrade channel — the manifest said cant_converge and the
    # user saw nothing.
    fake = _structural_wm([])
    stage = ReconcileTimelineStage(client=fake)
    ctx = _footage_dominated_mixed_ctx(target_seconds=30.0)

    out = await stage.handle(stage.input_model(), ctx)

    assert out.timeline_manifest is not None
    run = out.timeline_manifest.run
    assert run.convergence_outcome == "cant_converge"
    assert run.structural_relief_requested is False  # the flag the note used to depend on
    notes = [n for n in out.notes if _CANT_CONVERGE_MARKER in n]
    assert len(notes) == 1, out.notes
    # Kill-switch OFF (prod default) ⇒ nothing happens automatically, so the user IS the lever.
    assert "no automatic re-render will follow" in notes[0]
    assert "demo_video_gen" in notes[0]


@pytest.mark.asyncio
async def test_cant_converge_with_rerun_stages_still_notes_the_user() -> None:
    # The pure footage-pinned case (rerun_stages TRUTHY) keeps its note and, with the
    # kill-switch off, keeps the act-by-hand wording.
    fake = _structural_wm([{"op": "cut", "target_ids": ["clip-1"]}])
    stage = ReconcileTimelineStage(client=fake)
    ctx = _footage_pinned_ctx(target_seconds=15.0)

    out = await stage.handle(stage.input_model(), ctx)

    assert out.timeline_manifest is not None
    assert out.timeline_manifest.run.convergence_outcome == "cant_converge"
    assert out.timeline_manifest.run.structural_relief_requested is True
    notes = [n for n in out.notes if _CANT_CONVERGE_MARKER in n]
    assert len(notes) == 1, out.notes
    assert "no automatic re-render will follow" in notes[0]


# ── (11b) the note's promise is the SPAWNER's decision, not a second copy of it ────────
#
# The first cut of this note forecast the spawn from ``decide_structural_relief`` alone.
# That is only the FIRST of four gates: ``maybe_spawn_structural_relief_run`` then applies
# ``next_target_beats`` + ``protected_seconds_floor``, and REFUSES to bill a render that
# provably cannot converge. So on the commonest shape — a footage-pinned spine whose floor
# already exceeds the target, i.e. exactly when the user is the only lever — the note said
# "no action needed" while the spawner logged "refusing to bill an unconvergeable render".
# These tests drive the REAL spawner over the REAL manifest the stage just emitted and
# assert the note against what actually happens.


def _all_footage_ctx(*, seconds: list[float], target_seconds: float) -> RunContext:
    """An all-footage-pinned spine: every beat's min IS its measured clip, so the floor
    cannot shrink below the smallest clip. With ``min(seconds) > target`` NO subset of any
    size fits the budget — the spawner's unconvergeable guard fires."""
    lines: list[dict[str, Any]] = [
        {"id": f"clip-{i}", "text": "", "seconds": None, "clip": f"clip-{i}"}
        for i in range(len(seconds))
    ]
    clips = [
        {
            "id": f"clip-{i}",
            "line_id": f"clip-{i}",
            "video_url": f"https://minio.test/{i}.mp4",
            "seconds": sec,
        }
        for i, sec in enumerate(seconds)
    ]
    return _ctx(lines=lines, clips=clips, target_seconds=target_seconds)


def _stub_run_engine() -> Any:
    """A RunEngine stand-in whose only exercised method is ``start_run`` — the spawn hook's
    single side effect. Every DECISION in the hook still runs for real."""
    return SimpleNamespace(start_run=AsyncMock(return_value=SimpleNamespace(id=uuid7())))


async def _real_spawner_child(out: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Feed the stage's OWN manifest to the real post-completion spawn hook and return the
    child run id it would create (``None`` when it refuses). No mock of the decision — the
    hook's every guard runs; only ``RunEngine.start_run`` is stubbed."""
    monkeypatch.setattr(
        "src.services.runs.structural_relief.get_settings",
        lambda: SimpleNamespace(
            pipeline=SimpleNamespace(
                structural_relief_autoconverge_enabled=True,
                structural_relief_max_passes=2,
                # 0.0 = the underfull clamp disabled, exactly as it disables
                # reconcile_timeline's own gate — these tests predate it and must keep
                # exercising the shape guards they were written for.
                demo_timeline_underfull_fail_ratio=0.0,
            )
        ),
    )
    assert out.timeline_manifest is not None
    rows: list[Any] = [
        SimpleNamespace(
            stage_name="reconcile_timeline",
            output={
                "timeline_manifest": out.timeline_manifest.model_dump(),
                "lines": [{"id": ln.id, "text": ln.text or ""} for ln in out.lines],
            },
        )
    ]
    return await maybe_spawn_structural_relief_run(
        run_engine=_stub_run_engine(),
        parent_run_id=uuid7(),
        deliverable_id=uuid7(),
        # The parent's own snapshot: the SAME relief opt-in the stage ran under, which
        # the spawner now requires before it will bill a child.
        config_snapshot={
            "tenant_id": str(uuid7()),
            "subject_id": str(uuid7()),
            "direction": {"timing": _relief_timing()},
        },
        stage_rows=rows,
    )


def _arm_stage_killswitch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.services.stages.demo_video.reconcile_timeline.get_settings",
        lambda: SimpleNamespace(
            pipeline=SimpleNamespace(
                structural_relief_autoconverge_enabled=True,
                structural_relief_max_passes=2,
                # 0.0 = the underfull clamp disabled, exactly as it disables
                # reconcile_timeline's own gate — these tests predate it and must keep
                # exercising the shape guards they were written for.
                demo_timeline_underfull_fail_ratio=0.0,
            )
        ),
    )


@pytest.mark.asyncio
async def test_unconvergeable_spine_note_promises_nothing_the_spawner_will_not_do(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # D1 REGRESSION (the worst one). Three footage-pinned beats at 60s each against a 30s
    # target, kill-switch ARMED. (60s, not 40s: footage tails are now trimmed to 60% of the
    # clip, so a 40s clip's 24s floor would fit a 1-beat child.) The spawner refuses
    # ("cannot converge at any 1-beat spine, floor=36.00s > target=30.00s — refusing to
    # bill an unconvergeable render"), so the
    # note must NOT say a re-render will happen: the user is the ONLY remaining lever.
    _arm_stage_killswitch(monkeypatch)
    stage = ReconcileTimelineStage(client=_structural_wm([]))
    ctx = _all_footage_ctx(seconds=[60.0, 60.0, 60.0], target_seconds=30.0)

    out = await stage.handle(stage.input_model(), ctx)

    assert out.timeline_manifest is not None
    assert out.timeline_manifest.run.convergence_outcome == "cant_converge"
    assert await _real_spawner_child(out, monkeypatch) is None  # the ground truth
    notes = [n for n in out.notes if _CANT_CONVERGE_MARKER in n]
    assert len(notes) == 1, out.notes
    assert "re-render will be attempted" not in notes[0]
    assert "no automatic re-render will follow" in notes[0]
    assert "demo_video_gen" in notes[0]


@pytest.mark.asyncio
async def test_single_beat_spine_note_promises_nothing_the_spawner_will_not_do(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # D2 REGRESSION. A ONE-beat footage spine at 60s against a 30s target: the spawner
    # cannot shrink it ("beats=1, target_beats=1 — no spawn"), so the note must not promise
    # an automatic re-render.
    _arm_stage_killswitch(monkeypatch)
    stage = ReconcileTimelineStage(client=_structural_wm([]))
    ctx = _all_footage_ctx(seconds=[60.0], target_seconds=30.0)

    out = await stage.handle(stage.input_model(), ctx)

    assert out.timeline_manifest is not None
    assert out.timeline_manifest.run.convergence_outcome == "cant_converge"
    assert await _real_spawner_child(out, monkeypatch) is None
    notes = [n for n in out.notes if _CANT_CONVERGE_MARKER in n]
    assert len(notes) == 1, out.notes
    assert "re-render will be attempted" not in notes[0]
    assert "no automatic re-render will follow" in notes[0]


@pytest.mark.asyncio
async def test_cant_converge_with_autospawn_armed_promises_a_conditional_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # REWRITTEN — this was `test_cant_converge_with_autospawn_armed_does_not_tell_user_to_act`,
    # which PINNED the bug: it asserted the stage's own forecast against itself and required
    # the wording "requested automatically — no action needed", never once consulting the
    # spawner. Here the spawner really does spawn for this shape (one 60s pinned beat + two
    # elastic VO beats: the cheapest 1-beat spine is a VO beat, well under the 30s target),
    # and the note is asserted TOGETHER with that fact. The promise is conditional on
    # completion because the hook only runs on the dispatcher's `mark_completed` path (D3).
    _arm_stage_killswitch(monkeypatch)
    stage = ReconcileTimelineStage(client=_structural_wm([]))
    ctx = _footage_dominated_mixed_ctx(target_seconds=30.0)

    out = await stage.handle(stage.input_model(), ctx)

    assert await _real_spawner_child(out, monkeypatch) is not None  # the spawner DOES spawn
    notes = [n for n in out.notes if _CANT_CONVERGE_MARKER in n]
    assert len(notes) == 1, out.notes
    assert "If this run completes" in notes[0]  # an attempt, not a completed action
    assert "re-render will be attempted" in notes[0]
    assert "no action needed" not in notes[0]


@pytest.mark.asyncio
async def test_cant_converge_notes_never_contradict_the_rewrite_the_copy_advice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # D4. `_infeasible_note` fires on this same path and tells the user the narration "must
    # be shortened (rewrite the copy)". Shipping "no action needed" beside it in the same
    # `notes[]` was self-contradictory, so that wording is gone entirely — on BOTH the
    # spawn and the no-spawn branch.
    _arm_stage_killswitch(monkeypatch)
    stage = ReconcileTimelineStage(client=_structural_wm([]))

    for ctx in (
        _footage_dominated_mixed_ctx(target_seconds=30.0),  # spawner WOULD spawn
        _all_footage_ctx(seconds=[60.0, 60.0, 60.0], target_seconds=30.0),  # it would not
    ):
        out = await stage.handle(stage.input_model(), ctx)
        assert any("rewrite the copy" in n for n in out.notes), out.notes
        assert not any("no action needed" in n for n in out.notes), out.notes


@pytest.mark.asyncio
async def test_cant_converge_at_the_pass_cap_tells_the_user_to_act(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Kill-switch armed but the paid-render CAP is reached ⇒ no child will spawn. The note
    # must fall back to the act-by-hand wording rather than promising a re-render.
    monkeypatch.setattr(
        "src.services.stages.demo_video.reconcile_timeline.get_settings",
        lambda: SimpleNamespace(
            pipeline=SimpleNamespace(
                structural_relief_autoconverge_enabled=True,
                structural_relief_max_passes=2,
                # 0.0 = the underfull clamp disabled, exactly as it disables
                # reconcile_timeline's own gate — these tests predate it and must keep
                # exercising the shape guards they were written for.
                demo_timeline_underfull_fail_ratio=0.0,
            )
        ),
    )
    fake = _structural_wm([])
    stage = ReconcileTimelineStage(client=fake)
    ctx = _footage_dominated_mixed_ctx(target_seconds=30.0)
    ctx.config_snapshot["structural_relief"] = {"pass": 2}

    out = await stage.handle(stage.input_model(), ctx)

    notes = [n for n in out.notes if _CANT_CONVERGE_MARKER in n]
    assert len(notes) == 1, out.notes
    assert "no automatic re-render will follow" in notes[0]


@pytest.mark.asyncio
async def test_converged_timeline_emits_no_cant_converge_note() -> None:
    # No false alarm: a run that reaches feasibility in place must carry NO degrade note
    # (it would otherwise mark every healthy relief run `degraded` downstream).
    lines: list[dict[str, Any]] = [
        {
            "id": "clip-claim",
            "text": _phrase("claim", 20),
            "seconds": None,
            "clip": None,
            "claim_status": "grounded",
        },
        {"id": "clip-big", "text": _phrase("big", 40), "seconds": None, "clip": None},
        {"id": "clip-small", "text": _phrase("small", 15), "seconds": None, "clip": None},
    ]
    fake = _structural_wm([{"op": "cut", "target_ids": ["clip-big"]}])
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(lines=lines, target_seconds=15.0)

    out = await stage.handle(stage.input_model(), ctx)

    assert out.timeline_manifest is not None
    assert out.timeline_manifest.run.convergence_outcome == "converged"
    assert not [n for n in out.notes if _CANT_CONVERGE_MARKER in n]


# ── (8) the structural pre-pass obeys the ONE upstream policy ────────────────────────
#
# ``_apply_structural_prepass`` is the only WM call in this stage that runs OUTSIDE
# ``run_evaluator_optimizer`` (which routes its injected closures through
# ``upstream.guarded``). It used to be a bare ``self._wm.message`` under a blanket
# ``except Exception``, so EVERY upstream failure looked identical: zero retries spent on
# a transient blip, and an anonymous "call failed" note on a permanent fault whose real
# sentence was the only actionable thing about it.
#
# These three pin the corrected policy on OBSERVABLE outcomes — whether the structural op
# actually landed on the spine, and what the run's own note says — not on a call count for
# its own sake. The one attempt-count assertion below is the fail-fast property itself.


def _http_error(status: int, payload: dict[str, Any]) -> httpx.HTTPStatusError:
    """A WM-shaped ``HTTPStatusError`` (the double-nested envelope ``_human`` digs into)."""
    request = httpx.Request("POST", "https://worker-management.test/workers/x/message")
    response = httpx.Response(status, json=payload, request=request)
    return httpx.HTTPStatusError(f"{status}", request=request, response=response)


_BYO_402 = {
    "error": {
        "code": "ERROR",
        "message": {
            "code": "byo_llm_not_configured",
            "message": "Configure your LLM provider in Settings before running agents.",
        },
        "status": 402,
    }
}
_BLIP_503 = {"error": {"code": "INTERNAL_ERROR", "message": "Connection error.", "status": 503}}


def _cut_lines() -> list[dict[str, Any]]:
    """The armed mixed timeline: a claim beat (ineligible to trim) + two shortenable beats."""
    return [
        {
            "id": "clip-claim",
            "text": _phrase("claim", 20),
            "seconds": None,
            "clip": None,
            "claim_status": "grounded",
        },
        {"id": "clip-big", "text": _phrase("big", 40), "seconds": None, "clip": None},
        {"id": "clip-small", "text": _phrase("small", 15), "seconds": None, "clip": None},
    ]


def _flaky_structural_wm(
    errors: list[Exception | None], directives: list[dict[str, Any]]
) -> tuple[AsyncMock, list[int]]:
    """WM stub whose STRUCTURAL call raises ``errors[n]`` on attempt n (None ⇒ succeed).

    Returns the stub plus a one-element list carrying the STRUCTURAL attempt count, so a
    test can assert that a PERMANENT fault consumed exactly one attempt.
    """
    attempts = [0]

    def _side(worker_name: str | None = None, message: str = "", **_: Any) -> WMResponse:
        if worker_name == "pacing-critic":
            return _resp({"overall_score": 5})
        if worker_name == "beat-rewriter":
            return _resp({"text": ""})
        if "STRUCTURAL" in message:
            index = attempts[0]
            attempts[0] += 1
            exc = errors[index] if index < len(errors) else None
            if exc is not None:
                raise exc
            return _resp({"directives": directives})
        return _resp({"directives": []})

    fake = AsyncMock()
    fake.message.side_effect = _side
    return fake, attempts


@pytest.mark.asyncio
async def test_permanent_402_fails_the_prepass_fast_with_the_real_upstream_sentence() -> None:
    # A WM 402 "configure your LLM provider" can never be fixed by repeating the request.
    # OBSERVABLE: the run's own degrade note carries the upstream sentence a human can act
    # on, and it is marked PERMANENT so it is distinguishable from an outage that was
    # merely retried to exhaustion. And the retry budget is NOT spent: exactly one attempt.
    fake, attempts = _flaky_structural_wm(
        [_http_error(402, _BYO_402)], [{"op": "cut", "target_ids": ["clip-big"]}]
    )
    stage = ReconcileTimelineStage(client=fake)
    assert stage._llm_retries >= 1, "a zero-retry budget would make this assertion vacuous"

    out = await stage.handle(stage.input_model(), _ctx(lines=_cut_lines(), target_seconds=15.0))

    notes = [n for n in out.notes if "structural relief unavailable" in n]
    assert notes, f"expected a structural soft-degrade note, got {out.notes}"
    assert "permanent upstream fault" in notes[0]
    assert "Configure your LLM provider in Settings" in notes[0]
    # FAIL FAST: one attempt, not `llm_retries + 1`.
    assert attempts[0] == 1
    # Still a DEGRADE, never a raise: the spine is untouched, zero patches.
    assert out.timeline_manifest is not None
    assert out.timeline_manifest.run.structural_patches_applied == 0


@pytest.mark.asyncio
async def test_transient_upstream_is_retried_and_the_structural_cut_still_lands() -> None:
    # A 503 "Connection error." is exactly what the retry budget exists for. OBSERVABLE:
    # the structural cut ACTUALLY LANDS on the spine (one patch applied, converged) and the
    # run carries NO structural-degrade note — i.e. the blip cost the run nothing.
    # Before guarded()/stage_call, the first hiccup threw the whole pre-pass away.
    fake, attempts = _flaky_structural_wm(
        [_http_error(503, _BLIP_503), None], [{"op": "cut", "target_ids": ["clip-big"]}]
    )
    stage = ReconcileTimelineStage(client=fake)

    out = await stage.handle(stage.input_model(), _ctx(lines=_cut_lines(), target_seconds=15.0))

    assert attempts[0] == 2, "the transient attempt was not retried"
    assert not [n for n in out.notes if "structural relief unavailable" in n], out.notes
    assert out.timeline_manifest is not None
    run = out.timeline_manifest.run
    assert run.structural_patches_applied == 1
    assert run.convergence_outcome == "converged"


@pytest.mark.asyncio
async def test_transient_that_outlives_the_budget_still_soft_degrades() -> None:
    # The degrade contract is PRESERVED, not traded away for the retry: an upstream that
    # stays down through every attempt must still fall back to the trim-only loop with a
    # note — never hard-fail a paid render — and it must NOT claim a permanent fault.
    fake, attempts = _flaky_structural_wm(
        [_http_error(503, _BLIP_503)] * 8, [{"op": "cut", "target_ids": ["clip-big"]}]
    )
    stage = ReconcileTimelineStage(client=fake)

    out = await stage.handle(stage.input_model(), _ctx(lines=_cut_lines(), target_seconds=15.0))

    assert attempts[0] == stage._llm_retries + 1, "the whole retry budget should be spent"
    notes = [n for n in out.notes if "structural relief unavailable" in n]
    assert notes, f"expected a structural soft-degrade note, got {out.notes}"
    assert "permanent upstream fault" not in notes[0]
    assert out.timeline_manifest is not None
    assert out.timeline_manifest.run.structural_patches_applied == 0
