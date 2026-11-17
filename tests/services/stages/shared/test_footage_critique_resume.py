"""FootageCritiqueStage resume-determinism proof — the loop-position pass ordinal.

The compile discriminator handed to ``rerender`` is a 1-based LOOP-POSITION ordinal
(``pass_idx``), NOT a hash of the (non-deterministic, LLM-derived) nudged plan. So a
wholesale handler re-run after a resumable timeout submits its Nth re-render under the
SAME ``pass{n}`` key even when the critic's verdict — and therefore the nudged plan —
came out different on the re-run. The compiler dedupes purely on that key, so the
re-submit returns the first paid render instead of firing a duplicate PAID render.

REVERT-PROOF (cannot be executed here, documented instead): if the discriminator were
reverted to a plan-hash (``critique:<sha256(nudged plans)>``), run 1 and run 2 below
produce DIFFERENT nudged plans (they flag different shots), so their discriminators —
and thus their compile keys — would DIFFER. The resume-stability assertion
(``run1_passes == run2_passes == [1]`` ⇒ identical key) would then FAIL. That the two
runs' nudged plans genuinely differ (asserted below) is what makes this proof real
rather than vacuous.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from tests.services.stages.shared.test_footage_critique import (
    _analyzer,
    _ctx,
    _resp,
    _StubCritiqueStage,
)

# INTENT plans whose flip is VISIBLE: a flagged shot's ``zoom`` goes off→auto, so the
# two runs' nudged rows are byte-distinguishable (run 1 nudges clip-1, run 2 clip-2).
_PLANS = [
    {"line_id": "clip-1", "zoom": "off", "reframe": "static"},
    {"line_id": "clip-2", "zoom": "off", "reframe": "static"},
]

# Run 1 flags clip-1 (framing 2.0 < 4.0), clip-2 clean; the re-rendered pass scores
# clean so the loop converges after exactly one re-render.
_RUN1_VERDICTS = [
    _resp(
        {
            "shots": [
                {"shot_id": "clip-1", "score": 2.0, "framing": 2.0},
                {"shot_id": "clip-2", "score": 5.0},
            ]
        }
    ),
    _resp({"shots": [{"shot_id": "clip-1", "score": 5.0}, {"shot_id": "clip-2", "score": 5.0}]}),
]
# Run 2 flags a DIFFERENT shot (clip-2) — the resume produced a different verdict, so
# the nudged plan differs — yet the re-render is still the FIRST submit of this loop.
_RUN2_VERDICTS = [
    _resp(
        {
            "shots": [
                {"shot_id": "clip-1", "score": 5.0},
                {"shot_id": "clip-2", "score": 2.0, "framing": 2.0},
            ]
        }
    ),
    _resp({"shots": [{"shot_id": "clip-1", "score": 5.0}, {"shot_id": "clip-2", "score": 5.0}]}),
]


async def test_rerender_pass_ordinal_is_resume_stable_despite_differing_verdicts() -> None:
    # Same run_id across both drives — this models a resumed re-run of the SAME run.
    ctx = _ctx()

    wm1 = AsyncMock()
    wm1.message.side_effect = list(_RUN1_VERDICTS)
    stage1 = _StubCritiqueStage(wm=wm1, analyzer=_analyzer(), plans=[dict(p) for p in _PLANS])
    await stage1.run({}, ctx=ctx)

    wm2 = AsyncMock()
    wm2.message.side_effect = list(_RUN2_VERDICTS)
    stage2 = _StubCritiqueStage(wm=wm2, analyzer=_analyzer(), plans=[dict(p) for p in _PLANS])
    await stage2.run({}, ctx=ctx)

    # THE INVARIANT: both drives submit exactly one re-render and the loop hands it the
    # SAME ordinal despite the differing verdicts — so the production override stamps the
    # identical ``critique:pass1`` discriminator, and _compile_idempotency_key builds the
    # identical ``{run_id}:{stage}:critique:pass1`` key across the resume (compiler
    # dedupes ⇒ no duplicate paid render). A bug that failed to reset the per-handle
    # counter (e.g. ``[2]`` on the second drive) would fail this list equality.
    assert stage1.rerender_passes == [1]
    assert stage2.rerender_passes == stage1.rerender_passes
    # The ordinal→``critique:pass{n}`` KEY mapping itself is pinned (revert-proof) by the
    # demo_video/promo ``rerender`` tests' ``== "critique:pass1"`` asserts — this test
    # pins that the LOOP hands a resume-stable ordinal into that mapping.

    # NON-VACUITY: the two runs actually produced DIFFERENT nudged plans (run 1 flipped
    # clip-1's zoom, run 2 flipped clip-2's) — so the stability above is real, not the
    # trivial consequence of identical inputs. Under a plan-hash discriminator these
    # differing plans would have yielded different keys (the bug this fix removes).
    run1_nudged = {str(p["line_id"]): p["zoom"] for p in stage1.rerender_calls[0]}
    run2_nudged = {str(p["line_id"]): p["zoom"] for p in stage2.rerender_calls[0]}
    assert run1_nudged != run2_nudged
    assert run1_nudged["clip-1"] == "auto"  # run 1 flagged clip-1 → zoom flipped on
    assert run2_nudged["clip-2"] == "auto"  # run 2 flagged clip-2 → zoom flipped on
