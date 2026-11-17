"""EncodeMasterStage — the S19 encode/mux mastering evaluator-optimizer (spec 13 §5).

Exercises the DISABLED-BY-DEFAULT byte-identity (no `mastering.critic` ⇒ zero WM
calls, unchanged artifact), pass-1 convergence, monotone keep-highest across two
passes, the §5 clamp + provenance recording, authored-override-wins, the soft
never-raise invariant, the S20 compose re-compile (fires even after footage_critique
re-rendered, since footage nudges now persist), the per-shot banding channel, and the
downstream artifact-chain preference. The WMClient (`encode-critic` +
`mux-mastering-advisor`) and `DemoVideoCompileStage` are mocked/patched where
encode_master imports them — no live worker / compiler.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
import pytest
from w2v_common.clients import WMResponse

from src.services.blueprints.demo_video import DEMO_VIDEO_BLUEPRINT
from src.services.runs.run_context import RunContext
from src.services.stages.demo_video.compile import DemoVideoCompileOutput
from src.services.stages.demo_video.encode_master import (
    _CRITIC_OUTPUT_SCHEMA,
    EncodeMasterInput,
    EncodeMasterOutput,
    EncodeMasterStage,
)
from src.services.stages.demo_video.final_critique import FinalCritiqueStage
from src.services.stages.primitives.compile_stage import _compile_idempotency_key
from src.services.stages.shared.mastering_plan import ship_loudness_window

_COMPILE_PATCH = "src.services.stages.demo_video.encode_master.DemoVideoCompileStage"


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics={"total_tokens": 10},
    )


def _critic_resp(
    overall: float,
    *,
    profile_delta: dict[str, Any] | None = None,
    rerun_stages: list[str] | None = None,
) -> WMResponse:
    obj: dict[str, Any] = {"overall_score": overall}
    if profile_delta is not None:
        obj["profile_delta"] = profile_delta
    if rerun_stages is not None:
        obj["rerun_stages"] = rerun_stages
    return _resp(obj)


def _wm(
    critic: list[WMResponse] | WMResponse,
    advisor: WMResponse | Exception | None = None,
) -> AsyncMock:
    """A WMClient mock routing by ``worker_name``: the encode-critic responses are
    served in order from ``critic``; every other call returns/raises ``advisor``."""
    wm = AsyncMock()
    critic_list = critic if isinstance(critic, list) else [critic]
    critic_iter = iter(critic_list)

    async def _message(*_args: Any, **kwargs: Any) -> WMResponse:
        if kwargs.get("worker_name") == "encode-critic":
            return next(critic_iter)
        if isinstance(advisor, Exception):
            raise advisor
        assert advisor is not None
        return advisor

    wm.message.side_effect = _message
    return wm


def _patch_compile(handle: AsyncMock) -> tuple[Any, Any]:
    """Patch DemoVideoCompileStage where encode_master imports it; the class is
    constructed per re-compile, so bind ``handle`` on the shared return_value."""
    mgr = patch(_COMPILE_PATCH)
    mock_cls = mgr.start()
    mock_cls.return_value.handle = handle
    return mgr, mock_cls


def _ctx(
    *,
    mastering: dict[str, Any] | None = None,
    report: dict[str, Any] | None = None,
    footage: dict[str, Any] | None = None,
    compile_url: str = "http://v/orig.mp4",
    duration: float = 10.0,
) -> RunContext:
    compile_out: dict[str, Any] = {"artifact_url": compile_url, "duration_seconds": duration}
    if report is not None:
        compile_out["mastering"] = report
    prior: dict[str, Any] = {"demo_video_compile": compile_out}
    if footage is not None:
        prior["demo_video_footage_critique"] = footage
    config: dict[str, Any] = {}
    if mastering is not None:
        config["mastering"] = mastering
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=config,
        prior_run_outputs=prior,
    )


def _stage(wm: AsyncMock | None = None) -> EncodeMasterStage:
    return EncodeMasterStage(client=wm or AsyncMock(), compile_client=AsyncMock())


# ── 1. Disabled-by-default: no mastering.critic ⇒ byte-identical, zero WM calls ─
async def test_noop_byte_identity() -> None:
    wm = AsyncMock()
    report = {"gate_pass": True, "integrated_lufs": -14.0}
    out = await _stage(wm).handle(EncodeMasterInput(), _ctx(report=report))
    assert out.artifact_url == "http://v/orig.mp4"
    assert out.duration_seconds == 10.0
    assert out.mastering == report  # existing report passed through unchanged
    assert out.encode_critic is None
    wm.message.assert_not_called()  # zero worker traffic when disabled


# ── 2. Convergence at threshold on pass 1 ⇒ no advisor / no re-compile ─────────
async def test_converge_threshold() -> None:
    wm = _wm(critic=_critic_resp(5.0))
    compile_handle = AsyncMock()
    mgr, _compile_cls = _patch_compile(compile_handle)
    try:
        ctx = _ctx(
            mastering={"critic": {"iterations": 2}},
            report={"gate_pass": False, "rerun_stages": ["mux"]},
        )
        out = await _stage(wm).handle(EncodeMasterInput(), ctx)
    finally:
        mgr.stop()
    assert out.artifact_url == "http://v/orig.mp4"
    assert out.encode_critic is not None and out.encode_critic["score"] == 5.0
    compile_handle.assert_not_called()  # converged pass 1, optimizer never ran


# ── 2b. A REAL pass at exactly the threshold must not look like a dead critic ──
async def test_genuine_threshold_pass_is_not_marked_unavailable() -> None:
    """The discriminator test for the soft-degrade fallback.

    This is the case that made `score` unusable as evidence of assessment: a critic
    that RAN and scored exactly at the threshold with no rerun produces the same
    `score`, the same empty `provenance` and the same empty `rerun_stages` as the
    degrade path in test 6b. Only `critic_unavailable` separates them, so both
    directions have to be asserted or the flag proves nothing.
    """
    threshold = 3.5
    wm = _wm(critic=_critic_resp(threshold))
    compile_handle = AsyncMock()
    mgr, _cls = _patch_compile(compile_handle)
    try:
        ctx = _ctx(
            mastering={"critic": {"iterations": 2, "score_threshold": threshold}},
            report={"gate_pass": True},
        )
        out = await _stage(wm).handle(EncodeMasterInput(), ctx)
    finally:
        mgr.stop()

    assert out.encode_critic is not None
    assert out.encode_critic["score"] == threshold  # identical to the degrade path
    assert out.encode_critic["critic_unavailable"] is False  # ...but assessed
    assert out.encode_critic["critic_failed_passes"] == 0
    assert not any("WITHOUT being assessed" in n for n in out.notes)


# ── 2c. The advisor failing is recorded too, without failing the render ───────
async def test_advisor_failure_is_recorded() -> None:
    """A dead advisor plateaus the loop; that must be visible, not inferred."""
    wm = AsyncMock()

    async def _message(*_args: Any, **kwargs: Any) -> WMResponse:
        if kwargs.get("worker_name") == "encode-critic":
            return _critic_resp(2.0, profile_delta={"loudness_target": -14.0})
        raise RuntimeError("mux-mastering-advisor unregistered")

    wm.message.side_effect = _message
    compile_handle = AsyncMock()
    mgr, _cls = _patch_compile(compile_handle)
    try:
        ctx = _ctx(
            mastering={"critic": {"iterations": 2}},
            report={"gate_pass": False, "rerun_stages": ["mux"]},
        )
        out = await _stage(wm).handle(EncodeMasterInput(), ctx)
    finally:
        mgr.stop()

    assert out.artifact_url == "http://v/orig.mp4"  # render survived
    assert out.encode_critic is not None
    assert out.encode_critic["advisor_failed_passes"] != 0
    assert any("mux-mastering-advisor failed" in n for n in out.notes)
    compile_handle.assert_not_called()  # advisor died before any re-compile


# ── 3. Two passes, best (higher-scoring) master retained ──────────────────────
async def test_ran_to_iterations_monotone() -> None:
    wm = _wm(
        critic=[_critic_resp(2.0, profile_delta={"loudness_target": -14.0}), _critic_resp(4.5)],
        advisor=_resp({"loudness_target": -14.0}),
    )
    recompiled = DemoVideoCompileOutput(
        artifact_url="http://v/master.mp4",
        duration_seconds=10.0,
        mastering={"gate_pass": True, "integrated_lufs": -14.0},
    )
    compile_handle = AsyncMock(return_value=recompiled)
    mgr, _cls = _patch_compile(compile_handle)
    try:
        ctx = _ctx(
            mastering={"critic": {"iterations": 2}},
            report={"gate_pass": False, "rerun_stages": ["mux"]},
        )
        out = await _stage(wm).handle(EncodeMasterInput(), ctx)
    finally:
        mgr.stop()
    assert out.artifact_url == "http://v/master.mp4"  # improved master retained
    assert out.encode_critic is not None and out.encode_critic["score"] == 4.5
    compile_handle.assert_awaited_once()


# ── 4. §5 clamp: out-of-range knobs pulled back + recorded in provenance ──────
async def test_clamp_out_of_range_recorded() -> None:
    # gate passes + no rerun asked ⇒ the optimizer runs but does NOT re-compile
    # (advisory only), so the clamped recommendation + provenance surface without
    # touching the compiler.
    delta = {"loudness_target": -24.0, "crf": 8}
    wm = _wm(
        critic=[_critic_resp(2.0, profile_delta=delta), _critic_resp(2.0)],
        advisor=_resp(dict(delta)),
    )
    compile_handle = AsyncMock()
    mgr, _cls = _patch_compile(compile_handle)
    try:
        ctx = _ctx(
            mastering={"critic": {"iterations": 2}},
            report={"gate_pass": True},
        )
        out = await _stage(wm).handle(EncodeMasterInput(), ctx)
    finally:
        mgr.stop()
    compile_handle.assert_not_called()
    assert out.encode_critic is not None
    reco = cast("dict[str, Any]", out.encode_critic["profile_recommendation"])
    # DERIVED, not restated: the advisor's floor IS the ship gate's floor (see
    # `mastering_plan.ship_loudness_window` + tests/.../test_loudness_contract.py).
    # This used to pin the literal -16.0, which was 2 LU BELOW the band the ship gate
    # convicts at — i.e. this test asserted that the advisor was allowed to aim exactly
    # where `final_critique` would fail the run.
    gate_lo, _gate_hi = ship_loudness_window()
    assert reco["loudness_target"] == gate_lo  # clamped from -24.0
    assert reco["crf"] == 16  # clamped from 8
    prov_list = cast("list[dict[str, Any]]", out.encode_critic["provenance"])
    prov = {p["knob"]: p for p in prov_list}
    assert prov["loudness_target"]["proposed"] == -24.0
    assert prov["loudness_target"]["clamped"] == gate_lo
    assert prov["crf"]["proposed"] == 8
    assert prov["crf"]["clamped"] == 16


# ── 5. Authored knob wins over the advisor's proposal ─────────────────────────
async def test_authored_override_wins() -> None:
    wm = _wm(
        critic=[
            _critic_resp(2.0, profile_delta={"crf": 20, "loudness_target": -14.0}),
            _critic_resp(2.0),
        ],
        advisor=_resp({"crf": 20, "loudness_target": -14.0}),
    )
    compile_handle = AsyncMock()
    mgr, _cls = _patch_compile(compile_handle)
    try:
        ctx = _ctx(
            mastering={"critic": {"iterations": 2}, "crf": 19},  # author pinned crf=19
            report={"gate_pass": True},  # gate ok + no rerun ⇒ advisory-only, no recompile
        )
        out = await _stage(wm).handle(EncodeMasterInput(), ctx)
    finally:
        mgr.stop()
    assert out.encode_critic is not None
    reco = cast("dict[str, Any]", out.encode_critic["profile_recommendation"])
    assert reco["crf"] == 19  # authored value wins over the advisor's 20
    assert reco["loudness_target"] == -14.0  # advisor fills the author-unset knob


# ── 6. Soft gate: an advisor outage never raises; rerun_stages still surface ───
async def test_soft_never_raises_emits_rerun_stages() -> None:
    wm = _wm(
        critic=[_critic_resp(2.0), _critic_resp(2.0)],
        advisor=RuntimeError("advisor down"),
    )
    compile_handle = AsyncMock()
    mgr, _cls = _patch_compile(compile_handle)
    try:
        ctx = _ctx(
            mastering={"critic": {"iterations": 2}},
            report={"gate_pass": False, "rerun_stages": ["mux", "encode"]},
        )
        out = await _stage(wm).handle(EncodeMasterInput(), ctx)
    finally:
        mgr.stop()
    assert out.artifact_url == "http://v/orig.mp4"  # accepted current, no raise
    assert out.encode_critic is not None
    assert out.encode_critic["rerun_stages"] == ["mux", "encode"]
    compile_handle.assert_not_called()  # advisor failed before any re-compile


# ── 6b. Soft-degrade: the encode-critic WORKER raises ⇒ fallback recorded, not swallowed ─
async def test_soft_degrade_when_critic_worker_raises() -> None:
    # The encode-critic worker errors (unregistered worker, or schema-invalid output the
    # WMClient surfaces as an exception). The stage MUST NOT raise: critic_fn's
    # `except Exception` (encode_master.py:471-473) degrades to a threshold-score Critique
    # so the loop converges on the accept-current artifact, and the fallback is RECORDED on
    # `encode_critic` (score == threshold, empty provenance / rerun_stages) — a soft-degrade
    # is surfaced, never silently swallowed. NON-VACUITY: delete that except block and the
    # exception propagates to the outer belt-and-suspenders (handle():550-563), which returns
    # `encode_critic=None` — the `encode_critic is not None` / score asserts below then fail.
    threshold = 3.5
    wm = AsyncMock()

    async def _message(*_args: Any, **kwargs: Any) -> WMResponse:
        # Same worker_name routing as `_wm`, but the critic RAISES instead of returning.
        if kwargs.get("worker_name") == "encode-critic":
            raise RuntimeError("encode-critic worker error")
        raise AssertionError("advisor must never be reached — the critic degraded first")

    wm.message.side_effect = _message
    compile_handle = AsyncMock()
    mgr, _cls = _patch_compile(compile_handle)
    try:
        ctx = _ctx(
            # gate failed + a rerun asked: had the critic returned a low score, the optimizer
            # would run + re-compile — so a not-called compiler proves the critic short-circuited.
            mastering={"critic": {"iterations": 2, "score_threshold": threshold}},
            report={"gate_pass": False, "rerun_stages": ["mux"]},
        )
        out = await _stage(wm).handle(EncodeMasterInput(), ctx)
    finally:
        mgr.stop()
    assert out.artifact_url == "http://v/orig.mp4"  # accept-current, never raised
    assert out.encode_critic is not None  # fallback RECORDED, not swallowed
    assert out.encode_critic["score"] == threshold  # degrade == threshold ⇒ converge
    assert out.encode_critic["provenance"] == []  # no advisor ran ⇒ no provenance
    assert out.encode_critic["rerun_stages"] == []  # degrade path emits empty rerun_stages
    compile_handle.assert_not_called()  # critic degraded before any optimize / re-compile
    # `score == threshold` is NOT evidence of assessment: a critic that ran and passed at
    # exactly the threshold with no rerun emits an identical score, provenance and
    # rerun_stages. These two fields are the only thing that separates "approved" from
    # "never assessed", and the note is what makes the run read as degraded.
    assert out.encode_critic["critic_unavailable"] is True
    assert out.encode_critic["critic_failed_passes"] == 1  # converged on the degrade
    assert any("WITHOUT being assessed" in n for n in out.notes)


# ── 7. Guard removed (S20): re-compile FIRES even after footage_critique re-rendered ─
async def test_recompile_fires_even_when_footage_critique_rerendered() -> None:
    wm = _wm(
        critic=[_critic_resp(2.0, profile_delta={"loudness_target": -14.0}), _critic_resp(4.5)],
        advisor=_resp({"loudness_target": -14.0}),
    )
    recompiled = DemoVideoCompileOutput(
        artifact_url="http://v/master.mp4",
        duration_seconds=11.0,
        mastering={"gate_pass": True, "integrated_lufs": -14.0},
    )
    compile_handle = AsyncMock(return_value=recompiled)
    mgr, _cls = _patch_compile(compile_handle)
    try:
        ctx = _ctx(
            mastering={"critic": {"iterations": 2}},
            report={"gate_pass": False, "rerun_stages": ["mux"]},
            # footage-critique produced its OWN artifact (differs from compile's). The
            # old S19 guard blocked a re-master here; S20 removed it — footage_critique
            # persists its nudged plans (footage_plans) which compile prefers, so the
            # re-compile reproduces the improved footage AND applies the mastering delta.
            footage={"artifact_url": "http://v/fc.mp4", "duration_seconds": 11.0},
        )
        out = await _stage(wm).handle(EncodeMasterInput(), ctx)
    finally:
        mgr.stop()
    compile_handle.assert_awaited_once()  # guard gone ⇒ the re-compile fires
    assert out.artifact_url == "http://v/master.mp4"  # re-mastered artifact retained
    assert out.encode_critic is not None and out.encode_critic["score"] == 4.5
    # No advisory-only note any more — the re-compile is truthful.
    assert not any("advisory-only" in n for n in out.notes)
    # Compose: the mastering delta (advisor's loudness fill) reached the re-compile's
    # ctx, on top of the ctx that still carries footage_critique's persisted plans.
    assert compile_handle.await_args is not None
    _in, recompile_ctx = compile_handle.await_args.args
    assert recompile_ctx.config_snapshot["mastering"]["loudness_target"] == -14.0
    assert "demo_video_footage_critique" in recompile_ctx.prior_run_outputs


# ── 8. Downstream: final_critique prefers the encode_master artifact ──────────
def test_downstream_reads_encode_master_artifact() -> None:
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
        prior_run_outputs={
            "demo_video_encode_master": {
                "artifact_url": "http://v/master.mp4",
                "duration_seconds": 12.0,
            },
            "demo_video_footage_critique": {
                "artifact_url": "http://v/fc.mp4",
                "duration_seconds": 11.0,
            },
            "demo_video_compile": {"artifact_url": "http://v/orig.mp4", "duration_seconds": 10.0},
        },
    )
    stage = FinalCritiqueStage(client=AsyncMock(), analyzer=AsyncMock())
    assert stage._starting_artifact(ctx) == ("http://v/master.mp4", 12.0)


# ── 9. per_shot channel: surfaced when present, defensively None on old reports ─


def test_critic_schema_per_shot_is_optional() -> None:
    # The per_shot channel is available on the critic schema but NOT required, so a
    # critic response that omits it still validates (defensive against old critics).
    props = cast("dict[str, Any]", _CRITIC_OUTPUT_SCHEMA["properties"])
    required = cast("list[str]", _CRITIC_OUTPUT_SCHEMA["required"])
    assert "per_shot" in props  # channel exposed
    assert "per_shot" not in required  # but optional
    assert required == ["overall_score"]  # only overall_score is mandatory
    item_props = cast("dict[str, Any]", props["per_shot"]["items"]["properties"])
    assert set(item_props) == {"line_id", "banding", "needs_deband"}


# ── DAG registration in both blueprints ───────────────────────────────────────
@pytest.mark.parametrize("blueprint", [DEMO_VIDEO_BLUEPRINT])
def test_encode_master_registered_and_wired(blueprint: Any) -> None:
    # The encode-master stage hangs off footage_critique …
    spec = blueprint.stage("demo_video_encode_master")
    assert "demo_video_footage_critique" in spec.depends_on
    # … and final_critique is repointed to observe its output.
    assert "demo_video_encode_master" in blueprint.stage("final_critique").depends_on


# ── 12. Upstream faults on the critic/advisor calls: classified, ledgers KEPT ──
#
# BEFORE: both closures caught a bare ``Exception``. A 402 the tenant must act on, a
# 503 blip and a 429 were one undifferentiated `"<Type>: <msg>"` ledger entry, and a
# transient one spent ZERO retries — the master went unassessed on a blip.
#
# AFTER: ``guarded`` + ``stage_call`` run INSIDE the closure. Doing it outside would
# route exhaustion to the loop's ``on_loop_error="raise"`` → the stage's outer
# ``except Exception`` → ``_accept_current(encode_critic=None)``, which discards
# ``critic_failures`` / ``advisor_failures`` — the run would read CLEAN (no note) on
# a master nobody assessed. These tests pin that the ledgers survive.
def _wm_http_error(status: int, detail: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://wm.test/workers/encode-critic/message")
    response = httpx.Response(
        status, json={"error": {"code": "E", "message": detail, "status": status}}, request=request
    )
    return httpx.HTTPStatusError(detail, request=request, response=response)


@pytest.fixture
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    real_sleep = asyncio.sleep

    async def _noop(delay: float, *args: Any, **kwargs: Any) -> Any:
        return await real_sleep(0, *args, **kwargs)

    monkeypatch.setattr(asyncio, "sleep", _noop)


async def _run_with_critic_error(exc: Exception, *, retries: int) -> Any:
    wm = AsyncMock()
    calls = {"n": 0}

    async def _message(*_args: Any, **kwargs: Any) -> WMResponse:
        if kwargs.get("worker_name") == "encode-critic":
            calls["n"] += 1
            raise exc
        raise AssertionError("advisor must never be reached — the critic degraded first")

    wm.message.side_effect = _message
    compile_handle = AsyncMock()
    mgr, _cls = _patch_compile(compile_handle)
    stage = EncodeMasterStage(client=wm, compile_client=AsyncMock())
    stage._llm_retries = retries
    stage._llm_timeout = 60.0
    try:
        ctx = _ctx(
            mastering={"critic": {"iterations": 1, "score_threshold": 3.5}},
            report={"gate_pass": False, "rerun_stages": ["mux"]},
        )
        out = await stage.handle(EncodeMasterInput(), ctx)
    finally:
        mgr.stop()
    return out, calls["n"]


async def test_permanent_upstream_critic_fault_spends_no_retries_and_keeps_the_ledger(
    _no_backoff: None,
) -> None:
    """A 402 cannot be retried away. It must cost ONE call — and the master must
    still ship with ``critic_unavailable`` + the degrade note (the ledger survived)."""
    out, calls = await _run_with_critic_error(
        _wm_http_error(402, "Configure your LLM provider in Settings."), retries=3
    )
    assert calls == 1, "a permanent fault must not spend the retry budget"
    assert out.artifact_url == "http://v/orig.mp4"  # accept-current, never raised
    assert out.encode_critic is not None, "the ledger must not be discarded"
    assert out.encode_critic["critic_unavailable"] is True
    assert out.encode_critic["critic_failed_passes"] == 1
    assert any("WITHOUT being assessed" in n for n in out.notes)
    assert any("permanent upstream fault" in n for n in out.notes)


async def test_transient_upstream_critic_fault_is_retried_then_recorded_as_transient(
    _no_backoff: None,
) -> None:
    """A 503 spends the whole ladder (it is exactly what the budget is for), and the
    exhausted ladder is recorded as TRANSIENT — a different operator action from a
    402, and previously indistinguishable."""
    out, calls = await _run_with_critic_error(_wm_http_error(503, "unavailable"), retries=2)
    assert calls == 3, "a transient fault spends the configured retries"
    assert out.encode_critic is not None
    assert out.encode_critic["critic_unavailable"] is True
    notes = " ".join(out.notes)
    assert "transient upstream failure after 3 attempt(s)" in notes
    assert "permanent" not in notes


async def test_a_recovered_blip_leaves_the_master_assessed_and_the_run_clean(
    _no_backoff: None,
) -> None:
    """The point of the retry: a 503 on the first draw used to leave the master
    unassessed and the run degraded. Now the second draw lands and the verdict is
    real — ``critic_unavailable`` False, no note."""
    wm = AsyncMock()
    calls = {"n": 0}

    async def _message(*_args: Any, **kwargs: Any) -> WMResponse:
        if kwargs.get("worker_name") == "encode-critic":
            calls["n"] += 1
            if calls["n"] == 1:
                raise _wm_http_error(503, "unavailable")
            return _critic_resp(4.5)
        raise AssertionError("advisor must never be reached — the critic passed")

    wm.message.side_effect = _message
    compile_handle = AsyncMock()
    mgr, _cls = _patch_compile(compile_handle)
    stage = EncodeMasterStage(client=wm, compile_client=AsyncMock())
    stage._llm_retries = 2
    stage._llm_timeout = 60.0
    try:
        ctx = _ctx(
            mastering={"critic": {"iterations": 1, "score_threshold": 3.5}},
            report={"gate_pass": True},
        )
        out = await stage.handle(EncodeMasterInput(), ctx)
    finally:
        mgr.stop()
    assert calls["n"] == 2
    assert out.encode_critic is not None
    assert out.encode_critic["critic_unavailable"] is False
    assert out.encode_critic["score"] == 4.5
    assert out.notes == [], f"a survived blip must not degrade the run: {out.notes}"


async def test_permanent_upstream_advisor_fault_keeps_its_own_ledger(
    _no_backoff: None,
) -> None:
    """Same contract on the optimizer side: the advisor's 402 keeps the current draft,
    is recorded as permanent, and reaches the run as its own note."""
    wm = _wm(
        critic=[_critic_resp(2.0, profile_delta={"loudness_target": -14.0}), _critic_resp(2.0)],
        advisor=_wm_http_error(402, "Configure your LLM provider in Settings."),
    )
    compile_handle = AsyncMock()
    mgr, _cls = _patch_compile(compile_handle)
    stage = EncodeMasterStage(client=wm, compile_client=AsyncMock())
    stage._llm_retries = 3
    stage._llm_timeout = 60.0
    try:
        ctx = _ctx(
            mastering={"critic": {"iterations": 2, "score_threshold": 3.5}},
            report={"gate_pass": False, "rerun_stages": ["mux"]},
        )
        out = await stage.handle(EncodeMasterInput(), ctx)
    finally:
        mgr.stop()
    assert out.encode_critic is not None
    assert out.encode_critic["advisor_failed_passes"] == 1
    assert any("mux-mastering-advisor failed" in n for n in out.notes)
    assert any("permanent upstream fault" in n for n in out.notes)
    compile_handle.assert_not_called()  # no delta ⇒ nothing re-compiled


# ── 12. The mastering re-compile must NOT dedupe onto the run's original render ─
# `optimizer_fn` drives DemoVideoCompileStage on a `replace(ctx, ...)`, which keeps
# run_id AND the stage name. Without a payload-derived discriminator every draft
# collides onto the original compile key, video_compile returns the EXISTING job, and
# `_EncodeDraft.report` reports the new crf/loudnorm/deband as applied to a video that
# was never re-encoded.
async def _recompile_key(ctx: RunContext, delta: dict[str, Any]) -> str:
    """Run the stage once, forcing exactly one re-compile, and return the idempotency
    key the compile would have submitted under."""
    wm = _wm(
        critic=[_critic_resp(2.0, profile_delta=dict(delta)), _critic_resp(2.0)],
        advisor=_resp(dict(delta)),
    )
    seen: list[RunContext] = []

    async def _handle(_inp: Any, new_ctx: RunContext) -> DemoVideoCompileOutput:
        seen.append(new_ctx)
        return DemoVideoCompileOutput(
            artifact_url="http://v/master.mp4",
            duration_seconds=10.0,
            mastering={"gate_pass": True},
        )

    compile_handle = AsyncMock(side_effect=_handle)
    mgr, _cls = _patch_compile(compile_handle)
    try:
        await _stage(wm).handle(EncodeMasterInput(), ctx)
    finally:
        mgr.stop()
    # Guard against a vacuous assertion: if the loop never re-compiled there is no key
    # to compare and the test below would pass on any implementation.
    assert seen, "the stage never re-compiled — this test would prove nothing"
    return _compile_idempotency_key(seen[0], "demo_video_compile")


def _recompiling_ctx() -> RunContext:
    return _ctx(
        mastering={"critic": {"iterations": 2}},
        report={"gate_pass": False, "rerun_stages": ["mux"]},
    )


async def test_identical_mastering_recompile_still_dedupes() -> None:
    """Idempotency preserved: the same payload must reuse the render already paid for.

    This is the half a per-pass ordinal / timestamp / uuid discriminator would break —
    it would silently multiply the render bill for byte-identical work.
    """
    ctx = _recompiling_ctx()
    delta = {"loudness_target": -14.0, "crf": 20}
    assert await _recompile_key(ctx, delta) == await _recompile_key(ctx, delta)


# ── 13. The OUTER belt-and-suspenders catch must not emit the DISABLED shape ───
#
# The two closures bank their own failures, but a fault raised by the LOOP itself —
# a loop-level timeout, a defect in `run_evaluator_optimizer`, the
# `stage_call`/`guarded` wrapper — lands in `handle()`'s outer `except Exception`,
# one level ABOVE both ledgers. That catch used to return
# `_accept_current(encode_critic=None)` with a hardcoded `notes=[]`, i.e. an output
# BYTE-IDENTICAL to the critic-deliberately-DISABLED path (test 1). Since
# `api/routes/runs/_shared.py` derives a run's `degraded` flag from a non-empty
# `notes`, the run then reported a clean, assessed master that was never assessed —
# for the final deliverable on the main path.
async def test_loop_level_failure_is_degraded_not_the_disabled_shape() -> None:
    """A raise from `run_evaluator_optimizer` must still ACCEPT the current master
    (soft gate, no raise) while naming the failure in `notes`, so the run reads
    DEGRADED — and must not be confusable with the critic being switched off."""
    wm = _wm(critic=_critic_resp(5.0), advisor=_resp({}))
    compile_handle = AsyncMock()
    mgr, _cls = _patch_compile(compile_handle)
    loop_patch = patch(
        "src.services.stages.demo_video.encode_master.run_evaluator_optimizer",
        AsyncMock(side_effect=TimeoutError("loop deadline exceeded")),
    )
    loop_patch.start()
    try:
        ctx = _ctx(
            mastering={"critic": {"iterations": 2, "score_threshold": 3.5}},
            report={"gate_pass": True},
        )
        out = await _stage(wm).handle(EncodeMasterInput(), ctx)
    finally:
        loop_patch.stop()
        mgr.stop()

    # Soft gate preserved: the paid master still ships, the stage did not raise.
    assert out.artifact_url == "http://v/orig.mp4"
    assert out.duration_seconds == 10.0

    # …but the run must read DEGRADED, and the note must NAME the failure.
    assert out.notes, "a loop-level failure must not emit an empty (clean) note list"
    joined = " ".join(out.notes)
    assert "WITHOUT being assessed" in joined
    assert "loop" in joined
    assert "TimeoutError: loop deadline exceeded" in joined

    # …and it must be distinguishable from the DISABLED path, which emits
    # `encode_critic=None` (test 1). Same vocabulary as the in-closure degrades.
    assert out.encode_critic is not None
    assert out.encode_critic["critic_unavailable"] is True
    assert out.encode_critic["loop_failed"] is True
    assert out.encode_critic["score"] is None  # no verdict exists, not a threshold pass


# ── 13. The re-master loop is SCOPED to what a re-master can fix ──────────────
# Covers `encode_master.optimizer_fn`'s `should_recompile` expression. Before A/V
# wave 7 it was `bool(critique.rerun_stages) or gate_pass is False`, so a
# `colorspace` verdict — 23.3% of masters, an upstream video-TAGGING defect the
# compiler's own `mastering.py` says re-mastering re-emits identically — bought a
# full real-ffmpeg recompile plus 2 LLM calls per pass that could not move it.


def _gate_ctx(stages: list[str]) -> RunContext:
    return _ctx(
        mastering={"critic": {"iterations": 2}},
        report={"gate_pass": False, "rerun_stages": stages},
    )


async def _deliver(stages: list[str]) -> tuple[EncodeMasterOutput, AsyncMock]:
    """Drive a full two-pass loop against a failing gate naming ``stages``.

    Returns the stage OUTPUT — the master this stage actually ships downstream
    (`final_critique` / `render_manifest` read `artifact_url` from here) — plus the
    patched compile handle as secondary evidence.

    The two possible masters are distinguishable on every field that matters, so the
    assertions can read the delivered artifact rather than the call pattern:

    * ORIGINAL  — `http://v/orig.mp4`, 10.0s, report `{gate_pass: False, rerun_stages}`
    * REMASTERED — `http://v/master.mp4`, 12.0s, report `{gate_pass: True, ...}`
    """
    wm = _wm(
        critic=[_critic_resp(2.0, profile_delta={"loudness_target": -14.0}), _critic_resp(4.5)],
        advisor=_resp({"loudness_target": -14.0}),
    )
    compile_handle = AsyncMock(
        return_value=DemoVideoCompileOutput(
            artifact_url="http://v/master.mp4",
            duration_seconds=12.0,
            mastering={"gate_pass": True, "integrated_lufs": -14.0},
        )
    )
    mgr, _cls = _patch_compile(compile_handle)
    try:
        out = await _stage(wm).handle(EncodeMasterInput(), _gate_ctx(stages))
    finally:
        mgr.stop()
    return out, compile_handle


def _assert_delivered_the_original(out: EncodeMasterOutput, stages: list[str]) -> None:
    """The stage shipped the master it was given, untouched: same artifact, same
    length, and the SAME failing report — no second master was produced for it to
    choose between. Stated as an outcome, so it stays true under any restructuring
    that still declines the futile re-compile."""
    assert out.artifact_url == "http://v/orig.mp4"
    assert out.duration_seconds == 10.0
    assert out.mastering is not None
    assert out.mastering["gate_pass"] is False
    assert out.mastering["rerun_stages"] == stages
    # The advisory recommendation is still produced and stashed — declining the
    # re-compile must not cost the operator the advice.
    assert out.encode_critic is not None
    assert out.encode_critic["profile_recommendation"]


def _assert_delivered_the_remaster(out: EncodeMasterOutput) -> None:
    """The stage shipped the RE-MASTERED file: the new artifact, its new length, and
    the new report whose gate now passes."""
    assert out.artifact_url == "http://v/master.mp4"
    assert out.duration_seconds == 12.0
    assert out.mastering is not None
    assert out.mastering["gate_pass"] is True
    assert out.mastering["integrated_lufs"] == -14.0


# secondary: the spend itself


async def test_remediable_gate_failure_delivers_the_remaster() -> None:
    # ⛔ THE FIX IS NOT "delete the gate term". Loudness drift is what the compiler
    # calls "the real, remediable finding" and the advisor's `loudness_target` moves it,
    # so the run must come back carrying the IMPROVED master, not the failing one.
    out, compile_handle = await _deliver(["loudness"])
    _assert_delivered_the_remaster(out)
    assert compile_handle.await_count >= 1


async def test_a_mixed_verdict_delivers_the_remaster() -> None:
    # An unremediable stage must not VETO a remediable one named in the same report:
    # the loudness half is still fixed and the fixed master is what ships.
    out, compile_handle = await _deliver(["colorspace", "loudness"])
    _assert_delivered_the_remaster(out)
    assert compile_handle.await_count >= 1


async def test_an_unclassified_stage_delivers_the_remaster() -> None:
    # DENYLIST, not allowlist: a stage nobody has classified keeps today's behaviour —
    # it is still re-mastered and the re-mastered file is still delivered — so this
    # change can only ever remove a provably-futile recompile, never withhold a fix.
    out, compile_handle = await _deliver(["some_future_stage"])
    _assert_delivered_the_remaster(out)
    assert compile_handle.await_count >= 1
