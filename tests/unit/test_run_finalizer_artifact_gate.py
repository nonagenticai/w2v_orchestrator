"""Unit tests for run_finalizer.verify_terminal_artifact — the run-level
"completed only if the artifact exists" backstop.

Exercises the verdict matrix directly against the helper, patching its imported
``artifact_exists`` with the probe verdict; no DB / Redis / MinIO needed. Only a
definitive ``artifact_exists -> False`` blocks completion; everything else (no
terminal stage, skipped terminal stage, missing URL, indeterminate probe, probe
error) lets the run complete — we never turn a good run red on a blip.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

from src.services.runs.run_finalizer import (
    RunDecision,
    decide_run_terminal_state,
    verify_terminal_artifact,
)


@dataclass
class _Row:
    stage_name: str
    status: str
    output: dict | None


@dataclass
class _BP:
    terminal_artifact_stage: str | None
    # Mirrors DAGBlueprint's defaults, so every pre-existing case below still
    # describes an ``artifact_url`` / object-store blueprint verbatim.
    terminal_artifact_field: str = "artifact_url"
    terminal_artifact_probe: str = "object_store"


@dataclass
class _BPTopo:
    stages: list[str]

    def topological_order(self) -> list[str]:
        return self.stages


def _patch_probe(*, return_value: bool | None = None, side_effect: BaseException | None = None):
    mock = AsyncMock(return_value=return_value, side_effect=side_effect)
    return patch("src.services.runs.run_finalizer.artifact_exists", mock), mock


def _patch_design_probe(
    *, return_value: bool | None = None, side_effect: BaseException | None = None
):
    mock = AsyncMock(return_value=return_value, side_effect=side_effect)
    return patch("src.services.runs.run_finalizer.design_job_exists", mock), mock


#: A blueprint shaped like ``web_design_v2``: the terminal stage publishes NO
#: ``artifact_url`` — its user-facing links are proxy routes — and the verifiable
#: handle is the html_render job id.
_DESIGN_BP = _BP(
    "assemble", terminal_artifact_field="redesign_job_id", terminal_artifact_probe="design_job"
)


@pytest.mark.asyncio
async def test_missing_artifact_blocks_completion() -> None:
    ctx, mock = _patch_probe(return_value=False)
    with ctx:
        rows = [_Row("video_render", "completed", {"artifact_url": "http://minio.test/b/x.mp4"})]
        reason = await verify_terminal_artifact(_BP("video_render"), rows)
    assert reason is not None
    payload = json.loads(reason)
    assert payload["reason"] == "artifact_verification_failed"
    assert payload["stage"] == "video_render"
    assert payload["artifact_url"] == "http://minio.test/b/x.mp4"
    mock.assert_awaited_once_with("http://minio.test/b/x.mp4")


@pytest.mark.asyncio
async def test_present_artifact_completes() -> None:
    ctx, _ = _patch_probe(return_value=True)
    with ctx:
        rows = [_Row("video_render", "completed", {"artifact_url": "http://minio.test/b/x.mp4"})]
        assert await verify_terminal_artifact(_BP("video_render"), rows) is None


@pytest.mark.asyncio
async def test_indeterminate_probe_completes() -> None:
    ctx, _ = _patch_probe(return_value=None)
    with ctx:
        rows = [_Row("video_render", "completed", {"artifact_url": "http://x/y.mp4"})]
        assert await verify_terminal_artifact(_BP("video_render"), rows) is None


@pytest.mark.asyncio
async def test_probe_error_completes() -> None:
    ctx, mock = _patch_probe(side_effect=RuntimeError("probe boom"))
    with ctx:
        rows = [_Row("video_render", "completed", {"artifact_url": "http://x/y.mp4"})]
        assert await verify_terminal_artifact(_BP("video_render"), rows) is None
    mock.assert_awaited_once()  # the probe was attempted before erroring


@pytest.mark.asyncio
async def test_no_terminal_stage_completes_without_probing() -> None:
    ctx, mock = _patch_probe(return_value=False)
    with ctx:
        rows = [_Row("growth_bundle", "completed", {"snippets": []})]
        assert await verify_terminal_artifact(_BP(None), rows) is None
    mock.assert_not_awaited()  # no terminal stage → no probe at all


@pytest.mark.asyncio
async def test_skipped_terminal_stage_completes() -> None:
    # A terminal stage with no output (e.g. it was skipped) can't be probed;
    # the run still completes rather than failing on absence-of-output.
    ctx, mock = _patch_probe(return_value=False)
    with ctx:
        rows = [_Row("video_render", "skipped", None)]
        assert await verify_terminal_artifact(_BP("video_render"), rows) is None
    mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_url_in_output_completes() -> None:
    ctx, mock = _patch_probe(return_value=False)
    with ctx:
        rows = [_Row("video_render", "completed", {"duration_seconds": 30.0})]
        assert await verify_terminal_artifact(_BP("video_render"), rows) is None
    mock.assert_not_awaited()


# --- decide_run_terminal_state: the pure post-execution verdict --------------


def test_decision_continue_when_stages_pending() -> None:
    bp = _BPTopo(["a", "b", "c"])
    rows = [_Row("a", "completed", {}), _Row("b", "running", None)]
    assert decide_run_terminal_state(bp, rows, set()) is RunDecision.CONTINUE


def test_decision_dag_complete_when_all_terminal() -> None:
    # 'skipped' counts toward "the DAG is done" alongside 'completed'.
    bp = _BPTopo(["a", "b", "c"])
    rows = [_Row("a", "completed", {}), _Row("b", "skipped", None), _Row("c", "completed", {})]
    assert decide_run_terminal_state(bp, rows, set()) is RunDecision.DAG_COMPLETE


def test_decision_paused_when_pause_stage_completed() -> None:
    bp = _BPTopo(["a", "b", "c"])
    rows = [_Row("a", "completed", {}), _Row("b", "running", None)]
    assert decide_run_terminal_state(bp, rows, {"a"}) is RunDecision.PAUSED


def test_decision_completion_wins_over_stale_pause() -> None:
    # DAG_COMPLETE wins over PAUSED. When every stage is terminal, the run
    # completes even if a pause stage is among the completed ones — otherwise a
    # RESUMED run re-paused forever on its already-consumed checkpoint (the
    # checkpoint stays `completed` and keeps matching pause_set after the DAG
    # drained). A genuine fresh pause never reaches all-terminal (downstream
    # stays pending), so this never swallows a real review gate.
    bp = _BPTopo(["a", "b"])
    rows = [_Row("a", "completed", {}), _Row("b", "completed", {})]
    assert decide_run_terminal_state(bp, rows, {"b"}) is RunDecision.DAG_COMPLETE


def test_decision_resumed_run_past_checkpoint_completes() -> None:
    # Regression: pause@a → resume → DAG drains. `a` is still completed and in
    # pause_set, but every stage is terminal now → DAG_COMPLETE, not a second
    # PAUSED. (The bug left such runs stuck `paused` despite producing the
    # artifact.)
    bp = _BPTopo(["a", "b", "c"])
    rows = [_Row("a", "completed", {}), _Row("b", "completed", {}), _Row("c", "completed", {})]
    assert decide_run_terminal_state(bp, rows, {"a"}) is RunDecision.DAG_COMPLETE


def test_decision_fresh_pause_still_pauses_with_pending_downstream() -> None:
    # A genuine pause: the checkpoint completed but downstream is still pending
    # (the executor stopped at the gate), so not-all-terminal → PAUSED stands.
    bp = _BPTopo(["a", "b", "c"])
    rows = [_Row("a", "completed", {}), _Row("b", "pending", None), _Row("c", "pending", None)]
    assert decide_run_terminal_state(bp, rows, {"a"}) is RunDecision.PAUSED


def test_decision_pause_only_on_completed_not_skipped() -> None:
    # A pause stage that was *skipped* (not completed) must not pause — pause is
    # keyed off real completions only.
    bp = _BPTopo(["a", "b"])
    rows = [_Row("a", "completed", {}), _Row("b", "skipped", None)]
    assert decide_run_terminal_state(bp, rows, {"b"}) is RunDecision.DAG_COMPLETE


# --- the subset predicate: `declared <= terminal`, in BOTH directions --------
#
# The gate asks "is every DECLARED stage terminal?" — a SUBSET test, not set
# equality and not the reverse subset. The three tests below pin all three
# corners so neither inversion can be reintroduced as a "simplification".


def test_decision_dag_complete_with_terminal_row_for_removed_stage() -> None:
    """A terminal row for a stage NO LONGER in the blueprint must not block
    completion.

    WHY: when a stage is deleted from a blueprint, in-flight runs keep the row
    it already wrote. Under the old `terminal == declared` set-EQUALITY the
    terminal set was then a strict SUPERSET of the declared set, equality
    failed, and a run that had genuinely finished every declared stage could
    never reach DAG_COMPLETE — the reconciler re-dispatched it 5x and then
    marked it `failed` with a wedged error, a spurious failure of completed
    work. `declared <= terminal` ignores the extra ghost row.
    """
    bp = _BPTopo(["a", "b"])
    rows = [
        _Row("a", "completed", {}),
        _Row("b", "completed", {}),
        _Row("ghost", "completed", {}),  # stage since removed from the blueprint
    ]
    assert decide_run_terminal_state(bp, rows, set()) is RunDecision.DAG_COMPLETE


def test_decision_continue_when_a_declared_stage_has_no_row() -> None:
    """A declared stage with no row at all keeps the run CONTINUE-ing.

    WHY (the vacuity guard): this is what distinguishes the correct predicate
    `declared <= terminal` from the naive inversion `terminal <= declared`.
    Under that inversion `{a} <= {a, b}` holds, so a run whose stage `b` has
    never executed would be declared DAG_COMPLETE and shipped with an unrun
    stage. The subset direction matters: every DECLARED stage must be terminal,
    not merely every observed row.
    """
    bp = _BPTopo(["a", "b"])
    rows = [_Row("a", "completed", {})]
    assert decide_run_terminal_state(bp, rows, set()) is RunDecision.CONTINUE


def test_decision_dag_complete_with_skipped_declared_stage_and_ghost_row() -> None:
    """`skipped` still counts as terminal when a ghost row is also present.

    WHY: the removed-stage fix touches the same expression that consults
    ``TERMINAL_STAGE_STATUSES``. This pins that widening the predicate did not
    narrow the status set — an opt-in stage that was `skipped` (its
    precondition was absent) must keep satisfying the declared-stage
    requirement, otherwise every run with an opt-in stage hangs in `running`.
    """
    bp = _BPTopo(["a", "b"])
    rows = [
        _Row("a", "completed", {}),
        _Row("b", "skipped", None),
        _Row("ghost", "completed", {}),
    ]
    assert decide_run_terminal_state(bp, rows, set()) is RunDecision.DAG_COMPLETE


# --- a blueprint whose deliverable is NOT an `artifact_url` ------------------
#
# `web_design_v2`'s `assemble` emits `live_html_url` / `pdf_url` (orchestrator
# PROXY routes over html_render) and `redesign_job_id`, and no `artifact_url` at
# all. Before the blueprint declared its own deliverable field, the probe looked
# for `artifact_url`, found nothing, and disabled ITSELF for the whole blueprint:
# every web_design_v2 run reported `completed` with the deliverable never checked.


@pytest.mark.asyncio
async def test_design_blueprint_is_verified_via_its_declared_handle() -> None:
    """The whole point: a terminal stage with no `artifact_url` IS verified."""
    ctx, mock = _patch_design_probe(return_value=False)
    with ctx:
        rows = [
            _Row(
                "assemble",
                "completed",
                {
                    "redesign_job_id": "job-123",
                    "live_html_url": "/api/redesign/jobs/job-123/html",
                    "pdf_url": "/api/redesign/jobs/job-123/pdf",
                },
            )
        ]
        reason = await verify_terminal_artifact(_DESIGN_BP, rows)
    assert reason is not None, "a design deliverable that html_render lost must fail the run"
    payload = json.loads(reason)
    assert payload["reason"] == "artifact_verification_failed"
    assert payload["stage"] == "assemble"
    assert payload["field"] == "redesign_job_id"
    assert payload["probe"] == "design_job"
    assert payload["artifact_url"] == "job-123"
    mock.assert_awaited_once_with("job-123")


@pytest.mark.asyncio
async def test_healthy_design_run_completes_clean() -> None:
    """No false positives: a live design job completes, undegraded."""
    ctx, mock = _patch_design_probe(return_value=True)
    with ctx:
        rows = [_Row("assemble", "completed", {"redesign_job_id": "job-123"})]
        assert await verify_terminal_artifact(_DESIGN_BP, rows) is None
    mock.assert_awaited_once_with("job-123")


@pytest.mark.asyncio
async def test_design_blueprint_never_uses_the_object_store_probe() -> None:
    """The probe KIND is the blueprint's call, not the finalizer's.

    An object-store HEAD of `/api/redesign/jobs/{id}/html` is permanently
    indeterminate (foreign host, no bucket) — i.e. exactly as inert as no probe.
    Pinning that the storage probe is not even reached keeps a future
    "simplification" from routing everything through it again.
    """
    store_ctx, store_mock = _patch_probe(return_value=False)
    design_ctx, _ = _patch_design_probe(return_value=True)
    with store_ctx, design_ctx:
        rows = [_Row("assemble", "completed", {"redesign_job_id": "job-123"})]
        assert await verify_terminal_artifact(_DESIGN_BP, rows) is None
    store_mock.assert_not_awaited()


# --- a MISSING deliverable is not the same thing as a WIRING GAP -------------


@pytest.mark.asyncio
async def test_wiring_gap_is_loud_but_never_a_verdict(caplog) -> None:
    """A row that carries no declared handle: logged at ERROR, run still completes.

    The previous session's reasoning stands — a wiring gap must not fail a run
    that did all its work. What changed is that it can no longer hide: it logs,
    and (unlike a missing deliverable) it returns no verdict.
    """
    ctx, mock = _patch_design_probe(return_value=False)
    with ctx, caplog.at_level(logging.ERROR):
        rows = [_Row("assemble", "completed", {"live_html_url": "/api/redesign/jobs/x/html"})]
        assert await verify_terminal_artifact(_DESIGN_BP, rows) is None
    mock.assert_not_awaited()  # nothing to probe — NOT "probed and present"
    assert any("redesign_job_id" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_missing_deliverable_and_wiring_gap_are_distinguishable() -> None:
    """Same blueprint, same stage: absent handle → None; probe says gone → verdict."""
    ctx, _ = _patch_design_probe(return_value=False)
    with ctx:
        gap = await verify_terminal_artifact(
            _DESIGN_BP, [_Row("assemble", "completed", {"pdf_url": "/x"})]
        )
        gone = await verify_terminal_artifact(
            _DESIGN_BP, [_Row("assemble", "completed", {"redesign_job_id": "job-9"})]
        )
    assert gap is None
    assert gone is not None
    assert json.loads(gone)["reason"] == "artifact_verification_failed"


@pytest.mark.asyncio
async def test_unknown_probe_kind_is_loud_but_never_a_verdict(caplog) -> None:
    bp = _BP("assemble", terminal_artifact_field="redesign_job_id", terminal_artifact_probe="moon")
    with caplog.at_level(logging.ERROR):
        rows = [_Row("assemble", "completed", {"redesign_job_id": "job-123"})]
        assert await verify_terminal_artifact(bp, rows) is None
    assert any("unknown probe kind" in r.getMessage() for r in caplog.records)
