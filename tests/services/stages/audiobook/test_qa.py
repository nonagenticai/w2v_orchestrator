"""Tests for audiobook_qa (PR-7) — the listen-back QA loop.

The WM judge, the ASR transcriber, and the clip storage are all injected as fakes
(no network): storage hands back per-clip marker bytes, the ASR fake reads that
marker to return a per-clip transcript, and the WM fake answers a verdict for the
chunk_ids present in each call. Coverage:

* opt-out (no brief.audiobook.qa) → skip-clean, ZERO ASR/WM/storage traffic;
* all-clean transcripts → converged, no notes, no re-roll;
* one mispronounced chunk → flagged in notes (DEGRADE), re-roll recorded for
  exactly that chunk_id, bounded to ≤2 passes, ASR memoised across passes;
* ASR outage / WM outage → SOFT accept-current, run NOT failed, no raise;
* id_extractor guard: an optimizer returning a changed chunk-id set raises.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from w2v_common.clients import WMResponse

import src.services.stages.audiobook.qa as qa_module
from src.clients.schemas.transcription_worker import TranscriptionResult
from src.services.runs.run_context import RunContext
from src.services.stages.audiobook.qa import (
    AudiobookQAOutput,
    AudiobookQAStage,
    _QADraft,
)
from src.services.stages.shared.loop_engine import Critique, LoopResult


# ── fakes ────────────────────────────────────────────────────────────────────
def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics={"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
    )


def _make_wm(verdict_by_chunk: dict[str, dict[str, Any]]) -> AsyncMock:
    """WM fake: answers one verdict per chunk_id present in the call's message."""
    wm = AsyncMock()

    def _side(*, message: str, **_kw: Any) -> WMResponse:
        ids = re.findall(r'"chunk_id":\s*"([^"]+)"', message)
        judgments = [{"chunk_id": cid, **verdict_by_chunk[cid]} for cid in ids]
        return _resp({"judgments": judgments})

    wm.message.side_effect = _side
    return wm


def _make_storage(clip_bytes: dict[str, bytes]) -> AsyncMock:
    """Storage fake: returns marker bytes for a known ``{clip_id}.wav`` key, else None."""
    storage = AsyncMock()

    def _retrieve(_job_id: str, _stage_name: str, filename: str) -> bytes | None:
        return clip_bytes.get(filename)

    storage.retrieve_stage_artifact.side_effect = _retrieve
    return storage


def _make_asr(transcript_by_clip: dict[str, str]) -> AsyncMock:
    """ASR fake: reads the temp file's ``AUDIO:<clip_id>`` marker → per-clip text."""
    asr = AsyncMock()

    def _transcribe(*, file_path: Path, **_kw: Any) -> TranscriptionResult:
        marker = Path(file_path).read_bytes().decode()
        clip_id = marker.split(":", 1)[1]
        return TranscriptionResult(
            backend="fake",
            model="fake",
            duration=1.0,
            text=transcript_by_clip.get(clip_id, ""),
        )

    asr.transcribe_file.side_effect = _transcribe
    return asr


def _plan(chunks: list[tuple[str, str]]) -> dict[str, Any]:
    return {
        "chapters": [
            {
                "id": "ch1",
                "sequence": 0,
                "title": "C1",
                "chunks": [
                    {"id": cid, "text": txt, "sequence": i} for i, (cid, txt) in enumerate(chunks)
                ],
            }
        ],
        "total_chunks": len(chunks),
    }


def _narration(
    clips: list[tuple[str, str]],
    *,
    voice: str | None = None,
    speed: float | None = None,
) -> dict[str, Any]:
    """Narrate output fixture. ``voice``/``speed`` add the re-roll synth
    provenance (absent by default so the detection-only tests are unchanged)."""
    out: list[dict[str, Any]] = []
    for i, (clip_id, chunk_id) in enumerate(clips):
        clip: dict[str, Any] = {
            "id": clip_id,
            "chunk_id": chunk_id,
            "chapter_id": "ch1",
            "chapter_sequence": 0,
            "chunk_sequence": i,
            "artifact_url": f"http://x/{clip_id}.wav",
            "seconds": 1.0,
        }
        if voice is not None:
            clip["voice"] = voice
        if speed is not None:
            clip["speed"] = speed
        out.append(clip)
    return {"clips": out}


def _ctx(
    *,
    qa: bool = True,
    reroll: bool = False,
    plan: dict[str, Any] | None = None,
    narration: dict[str, Any] | None = None,
) -> RunContext:
    audiobook: dict[str, Any] = {"output_format": "mp3"}
    if qa:
        audiobook["qa"] = {"enabled": True, "reroll": reroll} if reroll else {"enabled": True}
    prior: dict[str, dict[str, Any]] = {}
    if plan is not None:
        prior["audiobook_plan"] = plan
    if narration is not None:
        prior["audiobook_narrate"] = narration
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": {"audiobook": audiobook}},
        prior_run_outputs=prior,
    )


def _clip_bytes(clip_ids: list[str]) -> dict[str, bytes]:
    return {f"{cid}.wav": f"AUDIO:{cid}".encode() for cid in clip_ids}


# ── (a) opt-out → skip-clean, zero traffic, byte-identical passthrough ────────
async def test_opt_out_skips_clean_with_zero_traffic() -> None:
    wm, asr, storage = AsyncMock(), AsyncMock(), AsyncMock()
    stage = AudiobookQAStage(client=wm, asr=asr, storage=storage)
    ctx = _ctx(
        qa=False,
        plan=_plan([("c1", "hello world")]),
        narration=_narration([("clipA", "c1")]),
    )
    from src.services.stages.base import StageSkippedError

    with pytest.raises(StageSkippedError) as exc:
        await stage.run({}, ctx=ctx)
    assert exc.value.missing == ["brief.audiobook.qa"]
    # Byte-identical opt-out: no ASR, no worker, no storage read.
    wm.message.assert_not_called()
    asr.transcribe_file.assert_not_called()
    storage.retrieve_stage_artifact.assert_not_called()


# ── (b) all-clean transcripts → converged, no notes, no re-roll ──────────────
async def test_all_clean_converges_without_reroll() -> None:
    wm = _make_wm({"c1": {"ok": True, "severity_score": 0.0}, "c2": {"ok": True}})
    asr = _make_asr({"clipA": "hello world", "clipB": "goodbye moon"})
    storage = _make_storage(_clip_bytes(["clipA", "clipB"]))
    stage = AudiobookQAStage(client=wm, asr=asr, storage=storage)
    ctx = _ctx(
        plan=_plan([("c1", "hello world"), ("c2", "goodbye moon")]),
        narration=_narration([("clipA", "c1"), ("clipB", "c2")]),
    )
    out = await stage.run({}, ctx=ctx)
    assert isinstance(out, AudiobookQAOutput)
    assert out.qa.converged is True
    assert out.qa.flagged_chunks == []
    assert out.notes == []  # clean ⇒ run stays healthy, not degraded
    assert out.reroll_requests == []
    assert wm.message.call_count == 1  # converged on pass 1, single bounded batch
    assert {j.chunk_id for j in out.qa.judgments} == {"c1", "c2"}
    assert all(j.ok for j in out.qa.judgments)


# ── (c) one mispronounced chunk → DEGRADE + re-roll recorded, ≤2 passes ──────
async def test_one_mispronounced_chunk_degrades_and_records_reroll() -> None:
    wm = _make_wm(
        {
            "c1": {"ok": True, "severity_score": 0.0},
            "c2": {
                "ok": False,
                "severity_score": 3.0,
                "issue": "read 'lead' as the metal, not the verb",
                "suggested_pronunciation": {"word": "lead", "ipa_or_phonetic": "leed"},
            },
        }
    )
    asr = _make_asr({"clipA": "hello world", "clipB": "the metal led the way"})
    storage = _make_storage(_clip_bytes(["clipA", "clipB"]))
    stage = AudiobookQAStage(client=wm, asr=asr, storage=storage)
    ctx = _ctx(
        plan=_plan([("c1", "hello world"), ("c2", "the lead led the way")]),
        narration=_narration([("clipA", "c1"), ("clipB", "c2")]),
    )
    out = await stage.run({}, ctx=ctx)

    # Exactly c2 flagged; surfaced as a degraded note naming it.
    assert out.qa.flagged_chunks == ["c2"]
    assert out.notes and "c2" in out.notes[0]
    assert out.qa.converged is False
    # Detection-only ⇒ a SINGLE scoring pass. A second pass would only re-judge
    # byte-identical audio (nothing to optimize), doubling WM spend (bug #3).
    assert out.qa.passes == 1
    # Re-roll recorded for EXACTLY the offending chunk, carrying the judge's fix.
    assert len(out.reroll_requests) == 1
    req = out.reroll_requests[0]
    assert req["chunk_id"] == "c2"
    assert req["suggested_word"] == "lead"
    assert req["suggested_pronunciation"] == "leed"
    # ASR warmed ONCE up front (hoisted out of the timed critic): one per clip.
    assert asr.transcribe_file.call_count == 2
    # A single WM scoring round, one bounded batch (2 chunks < _JUDGE_BATCH_SIZE).
    assert wm.message.call_count == 1


# ── (d) ASR outage → SOFT accept-current, run NOT failed ─────────────────────
async def test_asr_outage_soft_accepts_current_without_raising() -> None:
    wm = _make_wm({"c1": {"ok": True}})
    asr = AsyncMock()
    asr.transcribe_file.side_effect = RuntimeError("asr unreachable")
    storage = _make_storage(_clip_bytes(["clipA"]))
    stage = AudiobookQAStage(client=wm, asr=asr, storage=storage)
    ctx = _ctx(
        plan=_plan([("c1", "hello world")]),
        narration=_narration([("clipA", "c1")]),
    )
    out = await stage.run({}, ctx=ctx)  # must NOT raise
    assert out.qa.flagged_chunks == []
    # SOFT, but NOT a clean pass: nothing was listened back to, so the verdict must
    # not serialise as "converged, 5.0, no issues".
    assert out.qa.converged is False
    assert out.qa.assessed is False
    assert out.qa.reason == "no_judge_coverage"
    assert out.qa.score == qa_module._SCORE_MIN
    assert out.qa.unjudged_chunks == ["c1"]
    assert out.notes and "UNJUDGED" in out.notes[0]
    # No transcript ⇒ nothing judgeable ⇒ the judge worker is never even reached.
    wm.message.assert_not_called()
    assert out.reroll_requests == []


# ── (d') WM outage → SOFT accept-current, run NOT failed ─────────────────────
async def test_wm_outage_soft_accepts_current_without_raising() -> None:
    wm = AsyncMock()
    wm.message.side_effect = TimeoutError("judge worker timed out")
    asr = _make_asr({"clipA": "hello world"})
    storage = _make_storage(_clip_bytes(["clipA"]))
    stage = AudiobookQAStage(client=wm, asr=asr, storage=storage)
    ctx = _ctx(
        plan=_plan([("c1", "hello world")]),
        narration=_narration([("clipA", "c1")]),
    )
    out = await stage.run({}, ctx=ctx)  # must NOT raise
    assert out.qa.flagged_chunks == []
    # The judge never answered ⇒ zero coverage ⇒ degrade, not a silent green.
    assert out.qa.converged is False
    assert out.qa.assessed is False
    assert out.qa.reason == "no_judge_coverage"
    assert out.qa.unjudged_chunks == ["c1"]
    assert out.notes and "UNJUDGED" in out.notes[0]
    assert out.reroll_requests == []


# ── (e) id_extractor guard: a changed chunk-id set raises ─────────────────────


# ── no-op optimizer seam: returns the same draft (re-roll deferred) ──────────


# ── (f) timeout inversion (bug #1): whole-book ASR must NOT run inside the loop's
#        per-critic timeout, or QA times out and no-ops on any real-length book ──
async def test_whole_book_asr_is_hoisted_out_of_the_loop_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whole-book ASR must complete BEFORE the timed loop is entered.

    Before the fix ``_transcribe_all`` ran inside the loop's ``timeout_seconds``
    wait, so a real book (~100 chunks, ~minutes of ASR) blew the 660s budget →
    StageTimeoutError → SOFT ``loop_error`` → QA silently no-opped on every
    non-trivial book while passing 2-3-chunk fixtures.

    This asserts the invariant STRUCTURALLY (no wall-clock sleep, so it cannot
    flake on a loaded runner): a spy on ``run_evaluator_optimizer`` samples the
    ASR call count at loop entry. Hoisted ⇒ the whole book is already transcribed
    (``n``); ASR back inside the critic ⇒ ``0``."""
    n = 8
    clips = [(f"clip{i}", f"c{i}") for i in range(n)]
    asr = _make_asr({f"clip{i}": "hello" for i in range(n)})
    storage = _make_storage(_clip_bytes([c for c, _ in clips]))
    wm = _make_wm({f"c{i}": {"ok": True, "severity_score": 0.0} for i in range(n)})
    stage = AudiobookQAStage(client=wm, asr=asr, storage=storage)
    ctx = _ctx(
        plan=_plan([(c, "hello") for _, c in clips]),
        narration=_narration(clips),
    )

    asr_calls_at_loop_entry: list[int] = []
    real_loop = qa_module.run_evaluator_optimizer

    async def _spy_loop(
        initial: _QADraft,
        *,
        critic_fn: Callable[[_QADraft], Awaitable[Critique]] | None,
        optimizer_fn: Callable[[_QADraft, Critique], Awaitable[_QADraft]],
        threshold: float = 4.0,
        epsilon: float = 0.25,
        max_passes: int = 2,
        stage_name: str = "loop_engine",
        timeout_seconds: float | None = None,
        retries: int | None = None,
        id_extractor: Callable[[_QADraft], frozenset[str]] | None = None,
    ) -> LoopResult[_QADraft]:
        asr_calls_at_loop_entry.append(int(asr.transcribe_file.call_count))
        return await real_loop(
            initial,
            critic_fn=critic_fn,
            optimizer_fn=optimizer_fn,
            threshold=threshold,
            epsilon=epsilon,
            max_passes=max_passes,
            stage_name=stage_name,
            timeout_seconds=timeout_seconds,
            retries=retries,
            id_extractor=id_extractor,
        )

    monkeypatch.setattr(qa_module, "run_evaluator_optimizer", _spy_loop)

    out = await stage.run({}, ctx=ctx)

    # THE invariant: the whole book was transcribed before the timed loop started.
    assert asr_calls_at_loop_entry == [n]
    # A REAL verdict, not the SOFT loop_error the timeout inversion produced.
    assert out.qa.reason != "loop_error"
    assert out.qa.converged is True
    assert {j.chunk_id for j in out.qa.judgments} == {f"c{i}" for i in range(n)}
    # Every clip transcribed exactly once (warmed up front, not re-run per pass).
    assert asr.transcribe_file.call_count == n


# ── (g) one slow batch must NOT eat the whole critic budget (partial survives) ──
async def test_one_hanging_judge_batch_skips_only_that_batch() -> None:
    """A 50-chunk book is 10 sequential WM judge calls. Before the fix EACH was
    allowed the FULL outer budget inside ONE outer budget, so a single hung batch
    blew the loop timeout ⇒ ``loop_error`` with ZERO judgments — a silent no-op on
    a paid stage. Each batch now gets its own slice; a batch that blows it is
    SKIPPED and the loop continues, so the other 45 chunks are still judged."""
    n = 50
    clips = [(f"clip{i}", f"c{i}") for i in range(n)]
    hung = {f"c{i}" for i in range(5, 10)}  # exactly the 2nd batch of 5
    never = asyncio.Event()  # never set ⇒ the batch hangs until its slice expires

    verdicts = {f"c{i}": {"ok": True, "severity_score": 0.0} for i in range(n)}
    inner = _make_wm(verdicts)

    async def _maybe_hang(*, message: str, **kw: Any) -> WMResponse:
        if any(f'"chunk_id": "{cid}"' in message for cid in hung):
            await never.wait()
        return inner.message.side_effect(message=message, **kw)

    wm = AsyncMock()
    wm.message.side_effect = _maybe_hang
    asr = _make_asr({f"clip{i}": "hello" for i in range(n)})
    storage = _make_storage(_clip_bytes([c for c, _ in clips]))
    stage = AudiobookQAStage(client=wm, asr=asr, storage=storage)
    # 10 batches ⇒ each gets outer/10 = 0.4s. The hung batch burns 0.4s of the 4.0s
    # outer budget (10x headroom), the other 9 answer instantly.
    stage._llm_timeout = 4.0
    stage._llm_retries = 0
    stage._min_judge_batch_seconds = 0.05
    ctx = _ctx(
        plan=_plan([(c, "hello") for _, c in clips]),
        narration=_narration(clips),
    )

    out = await stage.run({}, ctx=ctx)

    # NOT a hard fail and NOT the zero-judgment soft no-op.
    assert out.qa.reason != "loop_error"
    judged = {j.chunk_id for j in out.qa.judgments}
    assert judged == {f"c{i}" for i in range(n)} - hung
    assert len(judged) == n - len(hung)
    # Every batch was attempted — the hung one did not abort the loop.
    assert wm.message.call_count == 10
    # …and the 5 skipped chunks are SURFACED, not silently dropped into a clean
    # pass: partial coverage gets its own reason, converged=False, and a note that
    # names the count and the ids.
    assert out.qa.reason == "partial_judge_coverage"
    assert out.qa.converged is False
    assert out.qa.assessed is True  # 45 chunks WERE judged
    assert sorted(out.qa.unjudged_chunks) == sorted(hung)
    assert out.notes, "5 unjudged chunks must degrade the run"
    coverage_note = out.notes[0]
    assert f"{len(hung)} of {n}" in coverage_note
    assert "UNJUDGED" in coverage_note
    for cid in hung:
        assert cid in coverage_note


# ── (h) a hung whole-book ASR must degrade SOFT, not hard-kill a paid stage ────
async def test_hanging_whole_book_asr_degrades_softly() -> None:
    """The hoisted ``_transcribe_all`` was bare — bounded only by the blueprint's
    ``default_timeout_seconds=3600``, which fires as a HARD ``StageTimeoutError``
    from the stage middleware, OUTSIDE this stage's soft-degrade guard. A hung ASR
    box therefore killed a paid stage. It is now bounded inside the guard."""
    never = asyncio.Event()

    async def _hang(*, file_path: Path, **_kw: Any) -> TranscriptionResult:
        assert Path(file_path).exists()
        await never.wait()
        raise AssertionError("unreachable")

    asr = AsyncMock()
    asr.transcribe_file.side_effect = _hang
    storage = _make_storage(_clip_bytes(["clipA"]))
    wm = _make_wm({"c1": {"ok": True}})
    stage = AudiobookQAStage(client=wm, asr=asr, storage=storage)
    stage._asr_timeout = 0.05
    ctx = _ctx(
        plan=_plan([("c1", "hello world")]),
        narration=_narration([("clipA", "c1")]),
    )

    out = await stage.run({}, ctx=ctx)  # must NOT raise

    assert out.qa.reason == "asr_timeout"
    # Nothing was listened back to ⇒ "not assessed", never "converged + perfect".
    assert out.qa.converged is False
    assert out.qa.assessed is False
    assert out.qa.score == qa_module._SCORE_MIN
    assert out.qa.unjudged_chunks == ["c1"]
    assert out.qa.judgments == []
    assert out.notes and "ASR" in out.notes[0]
    wm.message.assert_not_called()


# ── (i) DEFECT 1: a FULLY-skipped judge loop must NOT report a clean pass ──────
async def test_all_judge_batches_skipped_is_not_a_clean_pass() -> None:
    """The regression this guards: every batch cut ⇒ ``records == []`` ⇒
    ``min(..., default=_SCORE_MAX)`` ⇒ 5.0 ⇒ the loop converged on ``threshold``
    and the stage emitted ``judgments=[] flagged_chunks=[] notes=[]`` — a
    "no issues found" verdict on a PAID stage having judged NOTHING. That is
    strictly worse than the visible ``loop_error`` degrade it replaced."""
    n = 10  # 2 batches of _JUDGE_BATCH_SIZE
    clips = [(f"clip{i}", f"c{i}") for i in range(n)]

    # Every batch times out INSTANTLY (the WM round-trip blowing its own budget —
    # the same `except TimeoutError: continue` arm a burnt batch slice takes). No
    # wall-clock sleep, so the outcome cannot race the outer wait.
    wm = AsyncMock()
    wm.message.side_effect = TimeoutError("judge batch exceeded its budget")
    asr = _make_asr({f"clip{i}": "hello" for i in range(n)})
    storage = _make_storage(_clip_bytes([c for c, _ in clips]))
    stage = AudiobookQAStage(client=wm, asr=asr, storage=storage)
    stage._llm_retries = 0
    ctx = _ctx(plan=_plan([(c, "hello") for _, c in clips]), narration=_narration(clips))

    out = await stage.run({}, ctx=ctx)  # SOFT: must NOT raise

    # THE assertion: zero coverage may not masquerade as a converged clean pass.
    assert out.qa.judgments == []
    assert out.notes, "a fully-skipped judge loop MUST degrade the run"
    assert out.qa.reason != "threshold"
    assert out.qa.score != qa_module._SCORE_MAX
    assert out.qa.converged is False
    assert out.qa.assessed is False
    assert out.qa.reason == "no_judge_coverage"
    assert out.qa.score == qa_module._SCORE_MIN
    assert sorted(out.qa.unjudged_chunks) == sorted(f"c{i}" for i in range(n))
    assert f"{n} of {n}" in out.notes[0]
    assert "UNJUDGED" in out.notes[0]
    assert wm.message.call_count == 2  # both batches attempted, both cut


# ── (j) DEFECT 2a: the _MIN_JUDGE_BATCH_SECONDS FLOOR overruns — the deadline
#        clamp, not the nominal share, is what bounds the loop ──────────────────
def test_judge_batch_floor_overruns_but_the_deadline_clamp_bounds_the_sum() -> None:
    """``n * max(floor, phase/n) >= phase`` ALWAYS, and strictly exceeds it once the
    floor engages (a 100-batch book: 100 x 30s = 3000s against a ~460s phase). The
    nominal share was the only bound before, so the judge loop was guaranteed to
    overrun the critic budget and be cancelled — discarding the pass. This pins the
    floor branch AND the property that the clamped slices can never sum past it."""
    stage = AudiobookQAStage(client=AsyncMock(), asr=AsyncMock(), storage=AsyncMock())
    stage._llm_timeout = 660.0
    phase = stage._judge_phase_seconds()
    assert phase < stage._llm_timeout  # headroom for ASR + validation is reserved

    batch_count = 100  # a ~500-chunk book
    share = stage._judge_batch_budget(batch_count)
    # The FLOOR branch (phase/100 == 4.6s < 30s), untested until now.
    assert share == qa_module._MIN_JUDGE_BATCH_SECONDS
    assert batch_count * share > phase  # the overrun, made explicit

    budgets: list[float] = []
    remaining = phase
    for _ in range(batch_count):
        budget = stage._next_batch_budget(share, remaining)
        if budget is None:
            break
        budgets.append(budget)
        remaining -= budget

    assert sum(budgets) <= phase + 1e-9  # never past the judge phase…
    assert sum(budgets) <= stage._llm_timeout  # …and never past the outer budget
    assert len(budgets) < batch_count  # the loop STOPS instead of starting doomed batches
    assert stage._next_batch_budget(share, 0.0) is None
    assert stage._next_batch_budget(share, -1.0) is None
    # A remaining window smaller than the nominal share is clamped, never rounded up.
    assert stage._next_batch_budget(share, 3.0) == 3.0


# ── (k) DEFECT 2b: an OUTER cancel mid-loop must preserve completed judgments ──
async def test_outer_cancellation_preserves_already_completed_judgments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``raw_judgments`` used to be a critic-local list written into
    ``records_by_key`` only AFTER the loop, so ANY outer cancel unwound the critic
    before that line and ``handle`` returned ``loop_error`` with ZERO judgments —
    the exact defect the per-batch bound claimed to fix. Records are now published
    and mutated in place, so a cancel keeps what was already judged.

    The outer cancel is forced by neutering the judge-phase deadline (simulating any
    cause of an over-budget pass); batch 1 answers, batch 2 hangs past the outer
    ``timeout_seconds``."""
    n = 10
    clips = [(f"clip{i}", f"c{i}") for i in range(n)]
    hung = {f"c{i}" for i in range(5, 10)}  # exactly the 2nd batch of 5
    never = asyncio.Event()

    verdicts = {f"c{i}": {"ok": True, "severity_score": 0.0} for i in range(n)}
    inner = _make_wm(verdicts)

    async def _maybe_hang(*, message: str, **kw: Any) -> WMResponse:
        if any(f'"chunk_id": "{cid}"' in message for cid in hung):
            await never.wait()
        return inner.message.side_effect(message=message, **kw)

    wm = AsyncMock()
    wm.message.side_effect = _maybe_hang
    asr = _make_asr({f"clip{i}": "hello" for i in range(n)})
    storage = _make_storage(_clip_bytes([c for c, _ in clips]))
    stage = AudiobookQAStage(client=wm, asr=asr, storage=storage)
    stage._llm_timeout = 1.5  # stage_call floors timeout_seconds at 1.0s
    stage._llm_retries = 0
    stage._min_judge_batch_seconds = 0.01
    # Neuter the per-batch deadline so the hung batch runs past the OUTER wait and
    # the loop engine cancels the critic mid-pass.
    monkeypatch.setattr(stage, "_judge_phase_seconds", lambda: 100.0)
    ctx = _ctx(plan=_plan([(c, "hello") for _, c in clips]), narration=_narration(clips))

    out = await stage.run({}, ctx=ctx)  # SOFT: must NOT raise

    # The cancel is real (this is the degrade path, not a normal completion)…
    assert out.qa.reason == "loop_error"
    # …and batch 1's five judgments SURVIVED it.
    assert {j.chunk_id for j in out.qa.judgments} == {f"c{i}" for i in range(5)}
    assert out.qa.assessed is True
    assert out.qa.converged is False
    assert sorted(out.qa.unjudged_chunks) == sorted(hung)
    assert len(out.notes) == 2  # the loop-error degrade + the coverage degrade
    assert "UNJUDGED" in out.notes[1]


# ── surgical re-roll (PR-8) ───────────────────────────────────────────────────
def _make_faithful_wm() -> AsyncMock:
    """A transcript-aware judge: a chunk is ``ok`` iff its ASR transcript matches
    its manuscript (case-insensitive). A mismatch is flagged with a respelling fix
    ``Merkee → Mer-kee`` — so the re-roll has something actionable to apply."""
    wm = AsyncMock()

    def _side(*, message: str, **_kw: Any) -> WMResponse:
        judgments: list[dict[str, Any]] = []
        for line in message.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            row = json.loads(line)
            faithful = row["transcript"].strip().lower() == row["manuscript"].strip().lower()
            j: dict[str, Any] = {
                "chunk_id": row["chunk_id"],
                "ok": faithful,
                "severity_score": 0.0 if faithful else 3.0,
            }
            if not faithful:
                j["issue"] = "mispronounced proper noun"
                j["suggested_pronunciation"] = {"word": "Merkee", "ipa_or_phonetic": "Mer-kee"}
            judgments.append(j)
        return _resp({"judgments": judgments})

    wm.message.side_effect = _side
    return wm


def _fake_synth(new_clip_id: str, url: str = "", seconds: float = 1.5):
    """A synthesize_and_store double that records its call and returns a fixed
    (clip_id, url, seconds) so the test can pre-register the corrected clip's
    bytes/transcript under a known id."""
    calls: list[dict[str, Any]] = []

    async def _synth(_client: Any, **kwargs: Any) -> tuple[str, str, float]:
        calls.append(kwargs)
        return new_clip_id, url or f"http://x/{new_clip_id}.wav", seconds

    return _synth, calls


async def test_reroll_off_does_not_synthesize(monkeypatch: pytest.MonkeyPatch) -> None:
    # reroll flag absent ⇒ the optimizer is a no-op: no re-synth is ever issued,
    # a single scoring pass, and no corrected clips are emitted (byte-identical to
    # the detection-only path).
    synth, calls = _fake_synth("nope")
    monkeypatch.setattr(qa_module, "synthesize_and_store", synth)
    wm = _make_faithful_wm()
    asr = _make_asr({"clipB": "the mercky stood"})  # wrong ⇒ flagged
    storage = _make_storage(_clip_bytes(["clipB"]))
    stage = AudiobookQAStage(client=wm, asr=asr, storage=storage, tts=AsyncMock())
    ctx = _ctx(
        reroll=False,
        plan=_plan([("c2", "the Merkee stood")]),
        narration=_narration([("clipB", "c2")], voice="af_heart", speed=1.1),
    )
    out = await stage.run({}, ctx=ctx)
    assert calls == []  # NO re-synth when reroll is off
    assert out.qa.passes == 1
    assert out.corrected_clips == []
    assert out.qa.flagged_chunks == ["c2"]


async def test_reroll_resynthesizes_flagged_chunk_and_reconverges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # c2's ORIGINAL clip transcribes wrong (flagged pass 1); the re-roll applies
    # the judge's respelling to the chunk text, re-synthesizes as PLAIN TEXT (a new
    # clip id), whose CORRECTED audio transcribes right ⇒ pass 2 converges.
    synth, calls = _fake_synth("clipB2")
    monkeypatch.setattr(qa_module, "synthesize_and_store", synth)
    wm = _make_faithful_wm()
    # clipB (original) → wrong; clipB2 (re-rolled) → correct.
    asr = _make_asr({"clipB": "the mercky stood", "clipB2": "the Merkee stood"})
    storage = _make_storage(_clip_bytes(["clipB", "clipB2"]))
    stage = AudiobookQAStage(client=wm, asr=asr, storage=storage, tts=AsyncMock())
    ctx = _ctx(
        reroll=True,
        plan=_plan([("c2", "the Merkee stood")]),
        narration=_narration([("clipB", "c2")], voice="af_heart", speed=1.1),
    )
    out = await stage.run({}, ctx=ctx)

    # The re-synth applied the respelling to the SYNTH text, kept the same
    # (voice, speed), and sent PLAIN TEXT (no ssml=True).
    assert len(calls) == 1
    assert calls[0]["text"] == "the Mer-kee stood"
    assert calls[0]["voice"] == "af_heart"
    assert calls[0]["speed"] == 1.1
    assert calls[0].get("ssml", False) is False
    # Two scoring passes: the corrected audio re-judged.
    assert out.qa.passes == 2
    # Converged: the corrected clip transcribes faithfully now.
    assert out.qa.converged is True
    assert out.qa.flagged_chunks == []
    # The corrected clip is emitted for stitch to prefer (keyed by chunk_id).
    assert len(out.corrected_clips) == 1
    corrected = out.corrected_clips[0]
    assert corrected["chunk_id"] == "c2"
    assert corrected["id"] == "clipB2"
    assert corrected["seconds"] == 1.5


async def test_reroll_regression_does_not_report_a_discarded_fix_as_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two chunks flagged. The re-roll FIXES cA (corrected audio transcribes right)
    # but cB stays bad (its re-synth did not help), so the candidate's min-score
    # ties the initial's and the loop KEEPS the initial draft (ships BOTH originals,
    # corrected_clips empty). The reported verdict MUST match the shipped audio: both
    # chunks flagged. A spine-shared record set would leak cA's pass-2 "ok" judgment
    # into the retained report — a false-clean on a bad shipped clip.
    def _judge(*, message: str, **_kw: Any) -> WMResponse:
        judgments: list[dict[str, Any]] = []
        for line in message.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            row = json.loads(line)
            faithful = row["transcript"].strip().lower() == row["manuscript"].strip().lower()
            j: dict[str, Any] = {
                "chunk_id": row["chunk_id"],
                "ok": faithful,
                "severity_score": 0.0 if faithful else 3.0,
            }
            if not faithful:
                word = {"cA": "Alpha", "cB": "Bravo"}[row["chunk_id"]]
                j["issue"] = "x"
                j["suggested_pronunciation"] = {
                    "word": word,
                    "ipa_or_phonetic": f"{word[:2]}-{word[2:]}",
                }
            judgments.append(j)
        return _resp({"judgments": judgments})

    wm = AsyncMock()
    wm.message.side_effect = _judge

    def _synth(_client: Any, *, text: str, **kw: Any) -> Any:
        async def _run() -> tuple[str, str, float]:
            new_id = "clipA2" if "Al-pha" in text else "clipB2"
            return new_id, f"http://x/{new_id}.wav", 1.5

        return _run()

    monkeypatch.setattr(qa_module, "synthesize_and_store", _synth)
    # clipA fixed by the re-roll; clipB still wrong after its re-roll.
    asr = _make_asr(
        {
            "clipA": "hello alfa world",
            "clipA2": "hello Alpha world",
            "clipB": "hello bravoo world",
            "clipB2": "hello bravoo world",
        }
    )
    storage = _make_storage(_clip_bytes(["clipA", "clipA2", "clipB", "clipB2"]))
    stage = AudiobookQAStage(client=wm, asr=asr, storage=storage, tts=AsyncMock())
    ctx = _ctx(
        reroll=True,
        plan=_plan([("cA", "hello Alpha world"), ("cB", "hello Bravo world")]),
        narration=_narration([("clipA", "cA"), ("clipB", "cB")], voice="af_heart", speed=1.1),
    )
    out = await stage.run({}, ctx=ctx)

    # The winning draft is the ORIGINAL (tie ⇒ keep initial): nothing shipped corrected.
    assert out.corrected_clips == []
    # …so the report must flag BOTH original chunks — not silently clean cA.
    assert set(out.qa.flagged_chunks) == {"cA", "cB"}
    assert out.qa.converged is False


async def test_reroll_skips_chunk_without_synth_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A clip from a pre-provenance narrate output (or the gateway path) carries no
    # voice/speed ⇒ it cannot be reproduced, so the re-roll SKIPS it (no synth).
    synth, calls = _fake_synth("clipB2")
    monkeypatch.setattr(qa_module, "synthesize_and_store", synth)
    wm = _make_faithful_wm()
    asr = _make_asr({"clipB": "the mercky stood"})
    storage = _make_storage(_clip_bytes(["clipB"]))
    stage = AudiobookQAStage(client=wm, asr=asr, storage=storage, tts=AsyncMock())
    ctx = _ctx(
        reroll=True,
        plan=_plan([("c2", "the Merkee stood")]),
        narration=_narration([("clipB", "c2")]),  # NO voice/speed provenance
    )
    out = await stage.run({}, ctx=ctx)
    assert calls == []  # cannot reproduce the synth ⇒ no re-roll
    assert out.corrected_clips == []
    assert out.qa.flagged_chunks == ["c2"]


# ── (m) THE SUBDIVIDING SITE: audiobook_qa is the one stage on the LLM ladder
#        whose WM call does NOT run on the global per-attempt budget ───────────
#
# The client rung (src/config.py LLM_CLIENT_TIMEOUT_DELTA_S) is derived from
# ``llm_budget_seconds`` and is therefore correct only for a PLAIN site — one whose
# effective per-attempt timeout IS the global budget. This stage is not one: its judge
# loop slices ``llm_budget * _JUDGE_PHASE_SHARE`` across N batches and bounds each with
# its own ``asyncio.timeout(budget)`` (qa.py, the judge loop).
#
# ⚠️ AND IT CANNOT BE FIXED BY SIZING THE CLIENT DOWN. ``WMClient.message`` takes no
# per-call timeout — the read timeout is fixed at construction — and this stage shares
# ONE ``self._wm`` between the per-batch judge slice AND the full-budget
# evaluator-optimizer critic. No single client value can sit below both. So the
# guarantee that has to hold here is the other one: the SITE's own clamp fires first.
#
# That ordering is also the DESIRABLE one, which is the part a "just make client <
# effective everywhere" fix would have inverted. Read the judge loop's two handlers:
#
#     except TimeoutError:      -> log, `continue`  (skip ONE batch, keep judging)
#     except (httpx.HTTPError, ...) -> log, `break` (abandon EVERY remaining batch)
#
# ``asyncio.timeout(budget)`` raises ``TimeoutError`` and a client-side read timeout
# raises ``httpx.ReadTimeout``. Making the client fire first would convert a one-batch
# skip into a whole-loop abandonment — strictly worse coverage on a paid run.


def test_the_judge_batch_clamp_fires_before_the_client_rung() -> None:
    """The subdividing site's own clamp must be strictly below the shared client rung.

    Pinned across the batch-count range because the per-batch slice is dynamic: it is
    ``phase/n`` floored at ``_MIN_JUDGE_BATCH_SECONDS``, so the LARGEST slice is n == 1
    (the whole judge phase) and that is the case that has to clear the rung.

    Mutation-check: raise ``_JUDGE_PHASE_SHARE`` to 1.0 (making the judge phase equal to
    the full budget, i.e. above the client rung) and this goes red at n == 1.
    """
    from src.services.stages.llm_budget import llm_budget_seconds, llm_client_timeout_seconds

    # NO injected client: the point is the timeout the stage gives its OWN WMClient.
    stage = AudiobookQAStage(asr=AsyncMock(), storage=AsyncMock(), tts=AsyncMock())
    assert stage._llm_timeout == llm_budget_seconds()
    rung = llm_client_timeout_seconds()
    assert stage._wm.timeout == rung

    for batch_count in (1, 2, 5, 20, 100):
        slice_seconds = stage._judge_batch_budget(batch_count)
        assert slice_seconds < rung, (
            f"judge batch slice {slice_seconds}s for {batch_count} batch(es) is at or "
            f"above the shared WMClient rung {rung}s — the client would raise "
            "httpx.ReadTimeout first, which this loop handles by BREAKING out of every "
            "remaining batch instead of skipping one"
        )


def test_the_judge_loop_skips_one_batch_on_its_own_clamp_and_breaks_on_a_client_error() -> None:
    """The asymmetry the ordering above protects, read off the source.

    This is what makes "the client must always fire first" the WRONG fix at this site.
    """
    import inspect as _inspect

    src = _inspect.getsource(qa_module.AudiobookQAStage)
    loop = src[src.index("async with asyncio.timeout(budget):") :]
    # Anchor on the httpx handler and take the timeout handler that IMMEDIATELY
    # precedes it, rather than matching a literal `except TimeoutError:`. #872 widened
    # that handler to `except TimeoutError, StageTimeoutError:`, and a literal match
    # silently slid to a LATER handler outside the judge loop, producing an EMPTY slice
    # that failed for the wrong reason. The invariant this pins is an ORDERING of two
    # adjacent handlers, so locate them by adjacency, not by exact spelling.
    http_at = loop.index("except (httpx.HTTPError")
    timeout_at = loop.rindex("except TimeoutError", 0, http_at)
    on_timeout = loop[timeout_at:http_at]
    on_http = loop[http_at:][:1200]
    assert "continue" in on_timeout, "the site's own clamp no longer skips just one batch"
    assert "break" in on_http, "the client-error handler no longer breaks the loop"


# ── (j) upstream faults on the judge call: classified, not swallowed ──────────
#
# BEFORE: the judge call was wrapped in ``except (httpx.HTTPError, ...): break``.
# EVERY upstream failure — a 402 the tenant must act on, a 503 blip, a 429 — was the
# same event: stop the loop, spend ZERO of the configured retries, and report only
# "N chunks went UNJUDGED (judge budget exhausted, worker error, or no transcript)".
# AFTER: ``guarded`` classifies it, ``stage_call`` spends the batch's own slice on
# the transient half, and the note names which half it was. The stage stays SOFT in
# every branch — this is a paid run and the QA critic is advisory.
def _http_error(status: int, detail: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://wm.test/workers/x/message")
    response = httpx.Response(
        status, json={"error": {"code": "E", "message": detail, "status": status}}, request=request
    )
    return httpx.HTTPStatusError(detail, request=request, response=response)


@pytest.fixture
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse stage_call's retry backoff so the ladder runs instantly."""
    real_sleep = asyncio.sleep

    async def _noop(delay: float, *args: Any, **kwargs: Any) -> Any:
        return await real_sleep(0, *args, **kwargs)

    monkeypatch.setattr(asyncio, "sleep", _noop)


def _one_chunk_stage(wm: AsyncMock) -> tuple[AudiobookQAStage, RunContext]:
    asr = _make_asr({"clipA": "hello world"})
    storage = _make_storage(_clip_bytes(["clipA"]))
    stage = AudiobookQAStage(client=wm, asr=asr, storage=storage)
    ctx = _ctx(plan=_plan([("c1", "hello world")]), narration=_narration([("clipA", "c1")]))
    return stage, ctx


async def test_permanent_upstream_fault_fails_fast_and_names_the_sentence(
    _no_backoff: None,
) -> None:
    """A 402 ("configure your LLM provider") is not an outage: no retry can fix it,
    so it must cost exactly ONE call — and the run must SAY so, not bury it under
    generic unjudged-coverage prose."""
    wm = AsyncMock()
    wm.message.side_effect = _http_error(
        402, "Configure your LLM provider in Settings before running agents."
    )
    stage, ctx = _one_chunk_stage(wm)
    stage._llm_retries = 3  # a budget that must NOT be spent

    out = await stage.run({}, ctx=ctx)  # SOFT: must not raise

    assert wm.message.call_count == 1, "a permanent fault must not spend the retry budget"
    assert out.qa.assessed is False
    assert out.qa.converged is False
    assert out.qa.reason == "no_judge_coverage"
    assert out.qa.unjudged_chunks == ["c1"]
    fault_notes = [n for n in out.notes if "PERMANENT" in n]
    assert fault_notes, f"the permanent fault must be reported, got {out.notes}"
    assert "Configure your LLM provider" in fault_notes[0]


async def test_transient_upstream_fault_is_retried_and_the_batch_still_judges(
    _no_backoff: None,
) -> None:
    """A 503 blip used to end the judge loop with zero coverage on its first
    occurrence. It is now retried inside the batch's own slice, so the chunk gets a
    real verdict and the run is CLEAN — no degrade note at all."""
    verdicts = _make_wm({"c1": {"ok": True, "severity_score": 0.0}})
    calls = {"n": 0}

    async def _flaky(*, message: str, **kw: Any) -> WMResponse:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _http_error(503, "upstream connect error")
        return verdicts.message.side_effect(message=message, **kw)

    wm = AsyncMock()
    wm.message.side_effect = _flaky
    stage, ctx = _one_chunk_stage(wm)
    stage._llm_retries = 2  # 3 attempts: two blips then a verdict

    out = await stage.run({}, ctx=ctx)

    assert calls["n"] == 3
    assert out.qa.unjudged_chunks == []
    assert out.qa.assessed is True
    assert out.qa.converged is True
    assert out.notes == [], f"a survived blip must not degrade the run: {out.notes}"


async def test_exhausted_transient_degrades_to_partial_coverage_never_a_hard_failure(
    _no_backoff: None,
) -> None:
    """The budget is spent and the worker is still down. The stage must degrade
    (unjudged coverage + a note naming the attempts), NOT raise — ``qa`` runs at the
    end of a paid audiobook and its critic is advisory."""
    wm = AsyncMock()
    wm.message.side_effect = _http_error(503, "service unavailable")
    stage, ctx = _one_chunk_stage(wm)
    stage._llm_retries = 2

    out = await stage.run({}, ctx=ctx)  # SOFT: must not raise

    assert wm.message.call_count == 3, "the whole ladder is spent on a transient"
    assert out.qa.assessed is False
    assert out.qa.reason == "no_judge_coverage"
    assert out.qa.unjudged_chunks == ["c1"]
    exhausted = [n for n in out.notes if "attempt(s)" in n]
    assert exhausted, f"the exhausted ladder must be reported, got {out.notes}"
    assert "3 attempt(s)" in exhausted[0]


async def test_a_batch_slice_too_small_to_split_buys_no_extra_attempts(
    _no_backoff: None,
) -> None:
    """The SHARED-DEADLINE property, observably: retries come out of the batch's OWN
    slice. With 10 batches sharing a 4s critic budget each slice is ~0.28s — far too
    short for even one WM round-trip — so the ladder collapses to a single attempt
    per batch instead of 3x-ing a deadline every sibling batch draws on."""
    n = 10
    clips = [(f"clip{i}", f"c{i}") for i in range(n)]
    wm = AsyncMock()
    wm.message.side_effect = _http_error(503, "service unavailable")
    asr = _make_asr({f"clip{i}": "hello" for i in range(n)})
    storage = _make_storage(_clip_bytes([c for c, _ in clips]))
    stage = AudiobookQAStage(client=wm, asr=asr, storage=storage)
    stage._llm_timeout = 4.0
    stage._llm_retries = 2
    stage._min_judge_batch_seconds = 0.05
    ctx = _ctx(plan=_plan([(c, "hello") for _, c in clips]), narration=_narration(clips))

    out = await stage.run({}, ctx=ctx)

    # ONE attempt, and the loop stops on the outage rather than paying it per batch.
    assert wm.message.call_count == 1
    assert sorted(out.qa.unjudged_chunks) == sorted(c for _, c in clips)
    assert out.notes
