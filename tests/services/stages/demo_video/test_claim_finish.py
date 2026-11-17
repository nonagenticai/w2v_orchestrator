"""Unit tests for ClaimFinishStage — WM gateway mocked (no live LLM / DB).

The ``claim-grounder`` and ``claim-softener`` WM workers are an ``AsyncMock`` whose
``message`` returns a :class:`WMResponse` keyed on ``worker_name``; the deterministic
:func:`analyze_claims` evidence pass runs for real (it is pure). Mirrors the injection
pattern of ``tests/services/stages/demo_video/test_retention_plan.py``.

Contracts pinned here:
* presence-opt-in — no ``legal`` block, a ``legal`` block with NO ``mode`` key (the
  shape already shipping in prod), or no ingested corpus all skip before any WM
  traffic and leave the spine untouched;
* spine safety — a pinned or locked line is BYTE-UNCHANGED even when the softener
  returns a rewrite for it; hallucinated ids are dropped; ids / order / clip survive;
* anti-laundering — the softener's word is never trusted: the lint is RE-RUN on the
  final draft, and a swap of one unsourced superlative for another is stamped
  ``unsupported_accepted`` (never ``softened``) and recorded in ``notes``;
* soft gate — the worst possible score still returns a valid output; nothing blocks.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

from src.services.runs.run_context import RunContext
from src.services.stages.base import StageSkippedError
from src.services.stages.demo_video.claim_finish import ClaimFinishStage

# A corpus that supports "fastest onboarding" and NOTHING else the drafts assert —
# no "only", no "leading", no "guaranteed", no "40%".
_CORPUS = (
    "Acme is a project tracker for small teams. "
    "Independent benchmarks rank it the fastest onboarding on the market. "
    "Setup takes about ten minutes."
)
_FASTEST_SENTENCE = "Independent benchmarks rank it the fastest onboarding on the market."

_SHIPPED_LEGAL = {"trademarks": ["Acme®"], "defensibility": "trademark review pending"}


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


def _wm_fake(
    critic_outputs: list[dict[str, Any]] | dict[str, Any],
    softener_lines: list[dict[str, Any]],
) -> AsyncMock:
    """AsyncMock WM: claim-grounder → verdict dict(s); claim-softener → rewrites.

    ``critic_outputs`` may be a single dict (returned every critic call) or a list
    consumed one-per-call so a test can score below-then-above threshold.
    """
    fake = AsyncMock()
    critic_queue = list(critic_outputs) if isinstance(critic_outputs, list) else None
    critic_single = critic_outputs if isinstance(critic_outputs, dict) else None

    def _side(worker_name: str | None = None, **_: Any) -> WMResponse:
        if worker_name == "claim-grounder":
            if critic_queue is not None:
                # Repeat the final verdict once the queue is drained.
                return _resp(critic_queue.pop(0) if len(critic_queue) > 1 else critic_queue[0])
            return _resp(critic_single)
        return _resp({"lines": softener_lines})

    fake.message.side_effect = _side
    return fake


def _ctx(
    *,
    legal: dict[str, Any] | None,
    prior_lines: list[dict[str, Any]],
    grounding_text: str | None = _CORPUS,
) -> RunContext:
    cfg: dict[str, Any] = {}
    if legal is not None:
        cfg["legal"] = legal
    prior: dict[str, Any] = {
        "demo_video_gen": {
            "lines": prior_lines,
            "bdd_job_id": "",
            "features_generated": 0,
            "package_url": "",
        }
    }
    if grounding_text is not None:
        prior["source_ingest"] = {"grounding_text": grounding_text}
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=cfg,
        prior_run_outputs=prior,
    )


def _lines() -> list[dict[str, Any]]:
    """A spine whose l1 asserts an unsourced superlative the corpus cannot back."""
    return [
        {"id": "l1", "text": "We are the only platform that does this.", "clip": None},
        {"id": "l2", "text": "Here is the dashboard.", "clip": "clip-2"},
        {"id": "l3", "text": "Thanks for watching.", "clip": "clip-3"},
    ]


def _soften_l1() -> dict[str, Any]:
    return {"overall_score": 2, "directives": [{"op": "soften", "target_ids": ["l1"]}]}


_CLEAN_VERDICT = {"overall_score": 5, "directives": []}


# ── (a) presence-opt-in: no legal block at all → skip before any WM call ──────


@pytest.mark.asyncio
async def test_skips_without_legal_block() -> None:
    fake = _wm_fake(_CLEAN_VERDICT, [])
    stage = ClaimFinishStage(client=fake)
    ctx = _ctx(legal=None, prior_lines=_lines())

    with pytest.raises(StageSkippedError):
        await stage.handle(stage.input_model(), ctx)

    fake.message.assert_not_called()


# ── (b) the legal block ALREADY shipping in prod carries no `mode` → must not wake ──


@pytest.mark.asyncio
async def test_skips_when_legal_present_but_no_mode() -> None:
    fake = _wm_fake(_CLEAN_VERDICT, [])
    stage = ClaimFinishStage(client=fake)
    ctx = _ctx(legal=dict(_SHIPPED_LEGAL), prior_lines=_lines())

    with pytest.raises(StageSkippedError):
        await stage.handle(stage.input_model(), ctx)

    fake.message.assert_not_called()


# ── (c) grounding needs a corpus: nothing ingested → skip ─────────────────────


@pytest.mark.asyncio
async def test_skips_without_grounding_corpus() -> None:
    fake = _wm_fake(_soften_l1(), [{"id": "l1", "text": "SHOULD NOT BE APPLIED"}])
    stage = ClaimFinishStage(client=fake)
    ctx = _ctx(legal={"mode": "auto"}, prior_lines=_lines(), grounding_text=None)

    with pytest.raises(StageSkippedError):
        await stage.handle(stage.input_model(), ctx)

    fake.message.assert_not_called()


# ── (d) an author-pinned claim is byte-unchanged even when the softener rewrites it ──


@pytest.mark.asyncio
async def test_pinned_line_is_never_rewritten() -> None:
    pinned_text = "We are the only platform that does this."
    lines = [
        {"id": "l1", "text": pinned_text, "clip": None, "claim_status": "pinned"},
        {"id": "l2", "text": "Guaranteed 40% faster.", "clip": "clip-2"},
    ]
    fake = _wm_fake(
        [
            {"overall_score": 2, "directives": [{"op": "soften", "target_ids": ["l1", "l2"]}]},
            _CLEAN_VERDICT,
        ],
        # The softener happily rewrites the pinned beat too — it must be ignored.
        [
            {"id": "l1", "text": "LAUNDERED PINNED LINE"},
            {"id": "l2", "text": "Teams tell us ramp-up feels quicker."},
        ],
    )
    stage = ClaimFinishStage(client=fake)
    ctx = _ctx(legal={"mode": "auto"}, prior_lines=lines)

    out = await stage.handle(stage.input_model(), ctx)

    # Pinned: text AND status byte-unchanged — never rewritten, never re-stamped.
    assert out.lines[0].text == pinned_text
    assert out.lines[0].claim_status == "pinned"
    # The unpinned unsourced beat WAS softened.
    assert out.lines[1].text == "Teams tell us ramp-up feels quicker."
    assert out.lines[1].claim_status == "softened"


# ── (e) an author-locked line is byte-unchanged even when the softener rewrites it ──


@pytest.mark.asyncio
async def test_locked_line_is_never_rewritten() -> None:
    locked_text = "We are the only platform that does this."
    lines = [
        {"id": "l1", "text": locked_text, "clip": None, "locked": True},
        {"id": "l2", "text": "Guaranteed 40% faster.", "clip": "clip-2"},
    ]
    fake = _wm_fake(
        [
            {"overall_score": 2, "directives": [{"op": "soften", "target_ids": ["l1", "l2"]}]},
            _CLEAN_VERDICT,
        ],
        [
            {"id": "l1", "text": "LAUNDERED LOCKED LINE"},
            {"id": "l2", "text": "Teams tell us ramp-up feels quicker."},
        ],
    )
    stage = ClaimFinishStage(client=fake)
    ctx = _ctx(legal={"mode": "auto"}, prior_lines=lines)

    out = await stage.handle(stage.input_model(), ctx)

    assert out.lines[0].text == locked_text
    assert out.lines[0].locked is True
    assert out.lines[1].text == "Teams tell us ramp-up feels quicker."


# ── (f) happy path: unsourced superlative softened; ids / order / clip intact ──


@pytest.mark.asyncio
async def test_softens_unsourced_superlative_and_preserves_ids() -> None:
    fake = _wm_fake(
        [_soften_l1(), _CLEAN_VERDICT],
        [{"id": "l1", "text": "Built for teams who want fewer clicks."}],
    )
    stage = ClaimFinishStage(client=fake)
    ctx = _ctx(legal={"mode": "auto"}, prior_lines=_lines())

    out = await stage.handle(stage.input_model(), ctx)

    # Spine invariants: id-set, order and clip bindings untouched.
    assert [ln.id for ln in out.lines] == ["l1", "l2", "l3"]
    assert [ln.clip for ln in out.lines] == [None, "clip-2", "clip-3"]
    # Only the offending beat changed, and it is now clean ⇒ softened.
    assert out.lines[0].text == "Built for teams who want fewer clicks."
    assert out.lines[0].claim_status == "softened"
    assert out.lines[1].text == "Here is the dashboard."
    assert out.lines[2].text == "Thanks for watching."
    # Both workers were exercised.
    workers = {c.kwargs["worker_name"] for c in fake.message.call_args_list}
    assert workers == {"claim-grounder", "claim-softener"}
    # A softened claim is clean — nothing to degrade the run with.
    assert out.notes == []


# ── (g) a claim the corpus DOES echo is left alone (and costs zero WM traffic) ──


@pytest.mark.asyncio
async def test_grounded_claim_is_left_alone() -> None:
    grounded_text = "It is the fastest onboarding around."
    lines = [
        {"id": "l1", "text": grounded_text, "clip": None},
        {"id": "l2", "text": "Here is the dashboard.", "clip": "clip-2"},
    ]
    fake = _wm_fake(_soften_l1(), [{"id": "l1", "text": "SHOULD NOT BE APPLIED"}])
    stage = ClaimFinishStage(client=fake)
    ctx = _ctx(legal={"mode": "auto"}, prior_lines=lines)

    out = await stage.handle(stage.input_model(), ctx)

    assert out.lines[0].text == grounded_text
    assert out.lines[0].claim_status == "grounded"
    assert out.lines[0].claim_source == _FASTEST_SENTENCE
    # Nothing was unsourced ⇒ the loop never ran ⇒ no WM traffic at all.
    fake.message.assert_not_called()
    assert out.notes == []


# ── (h) ANTI-LAUNDERING: "only" → "leading" is still unsourced ────────────────


@pytest.mark.asyncio
async def test_relaunders_are_marked_unsupported_accepted() -> None:
    # The critic is fooled (scores the rewrite 5/5); the deterministic lint is not.
    fake = _wm_fake(
        [_soften_l1(), _CLEAN_VERDICT],
        [{"id": "l1", "text": "We are the leading platform that does this."}],
    )
    stage = ClaimFinishStage(client=fake)
    ctx = _ctx(legal={"mode": "auto"}, prior_lines=_lines())

    out = await stage.handle(stage.input_model(), ctx)

    # The rewrite was applied, but the verdict is the LINT's, not the softener's.
    assert out.lines[0].text == "We are the leading platform that does this."
    assert out.lines[0].claim_status == "unsupported_accepted"
    assert out.lines[0].claim_source == ""
    # Recorded, not silently accepted: a note (⇒ degraded run) naming the rerun hint.
    assert len(out.notes) == 1
    assert "unsupported_accepted" in out.notes[0]
    assert "leading" in out.notes[0]
    assert "script_doctor" in out.notes[0]


# ── (i) soft gate: the worst possible score still returns a valid output ──────


@pytest.mark.asyncio
async def test_soft_gate_never_raises() -> None:
    # The critic never crosses threshold and the softener never fixes anything.
    fake = _wm_fake(
        {"overall_score": 1, "directives": [{"op": "drop_claim", "target_ids": ["l1"]}]},
        [{"id": "l1", "text": "We are the only platform, guaranteed."}],
    )
    stage = ClaimFinishStage(client=fake)
    ctx = _ctx(legal={"mode": "auto", "claims": {"iterations": 2}}, prior_lines=_lines())

    out = await stage.handle(stage.input_model(), ctx)

    # Bounded loop, valid output, run proceeds to compile — never blocked.
    assert [ln.id for ln in out.lines] == ["l1", "l2", "l3"]
    assert out.lines[0].claim_status == "unsupported_accepted"
    assert out.notes, "an accepted unsupported claim must be recorded"
    assert 1 <= fake.message.call_count <= 2 * 2


# ── (j) C0: an id the softener invented is never merged into the spine ────────


@pytest.mark.asyncio
async def test_hallucinated_id_ignored() -> None:
    fake = _wm_fake(
        [_soften_l1(), _CLEAN_VERDICT],
        [
            {"id": "l1", "text": "Built for teams who want fewer clicks."},
            {"id": "l999", "text": "HALLUCINATED"},
        ],
    )
    stage = ClaimFinishStage(client=fake)
    ctx = _ctx(legal={"mode": "auto"}, prior_lines=_lines())

    out = await stage.handle(stage.input_model(), ctx)

    assert [ln.id for ln in out.lines] == ["l1", "l2", "l3"]
    assert all(ln.text != "HALLUCINATED" for ln in out.lines)


# ── (k) critique_only: the critic scores, but NO rewrite is applied ───────────


@pytest.mark.asyncio
async def test_critique_only_applies_no_rewrite() -> None:
    fake = _wm_fake(_soften_l1(), [{"id": "l1", "text": "SHOULD NOT BE APPLIED"}])
    stage = ClaimFinishStage(client=fake)
    ctx = _ctx(legal={"mode": "critique_only"}, prior_lines=_lines())

    out = await stage.handle(stage.input_model(), ctx)

    assert [ln.text for ln in out.lines] == [ln["text"] for ln in _lines()]
    # Still HONEST about what it found: the claim is unsupported and recorded.
    assert out.lines[0].claim_status == "unsupported_accepted"
    workers = [c.kwargs["worker_name"] for c in fake.message.call_args_list]
    assert "claim-grounder" in workers
    assert "claim-softener" not in workers
