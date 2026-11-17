"""render_manifest terminal stage (A20) — against real postgres.

Covers the always-emit contract (absent / off reproducibility still emits the
override ledger + completeness, but withholds the reproducibility spine), the
spine-on path (reproducibility on ⇒ a manifest carrying the master seed, the
override ledger, the delivered artifact and full completeness), and the JSONB
persistence round-trip through the real ``StageOutputRepository`` on the CI
postgres — the whole point of the stage is that the manifest survives as an
auditable record, so it must come back whole after a re-read from postgres.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.run_repository import RunRepository
from src.database.repositories.stage_output_repository import StageOutputRepository
from src.services.runs.run_context import RunContext
from src.services.stages.demo_video.manifest import (
    RenderManifestOutput,
    RenderManifestStage,
)

pytestmark = pytest.mark.asyncio

_PLAN_DOCUMENT: dict[str, Any] = {
    "provenance": {
        "resolution": {"value": "2k", "source": "default"},
        "music_gain_db": {"value": -22, "source": "override"},
    }
}
_COMPILE_OUTPUT: dict[str, Any] = {
    "artifact_url": "https://minio.test/x.mp4",
    "duration_seconds": 42.0,
}


def _ctx(
    config_snapshot: dict[str, Any], *, compile_output: dict[str, Any] | None = None
) -> RunContext:
    prior: dict[str, dict[str, Any]] = {}
    if compile_output is not None:
        prior["demo_video_compile"] = compile_output
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=config_snapshot,
        prior_run_outputs=prior,
    )


# ── Always-emit — absent / off reproducibility still emits the ledger, no spine ─


@pytest.mark.parametrize(
    "config_snapshot",
    [{"plan_document": _PLAN_DOCUMENT}],
)
async def test_absent_or_off_reproducibility_emits_ledger_without_spine(
    config_snapshot: dict[str, Any],
) -> None:
    """The manifest ALWAYS emits: absent/off reproducibility still yields the
    override ledger + completeness (the audit record the UI surfaces), but the
    reproducibility spine (master_seed + input_hash) stays None — the stage makes
    no reproducibility promise it can't keep."""
    stage = RenderManifestStage()
    ctx = _ctx(config_snapshot, compile_output=_COMPILE_OUTPUT)

    out = await stage.handle(stage.input_model(), ctx)

    assert isinstance(out, RenderManifestOutput)
    # Ledger + completeness populated; delivered artifact rides in.
    assert len(out.manifest.override_ledger) == 2
    assert out.manifest.completeness == 1.0
    assert out.manifest.artifact_url == "https://minio.test/x.mp4"
    # Spine withheld — no reproducibility opt-in.
    assert out.manifest.master_seed is None
    assert out.manifest.input_hash is None


# ── Active: reproducibility on ⇒ a fully-populated manifest ───────────────────


async def test_active_manifest_populated() -> None:
    stage = RenderManifestStage()
    ctx = _ctx(
        {
            "reproducibility": {"mode": "on", "master_seed": 12345},
            "plan_document": _PLAN_DOCUMENT,
        },
        compile_output=_COMPILE_OUTPUT,
    )

    out = await stage.handle(stage.input_model(), ctx)

    assert isinstance(out, RenderManifestOutput)
    assert out.manifest.master_seed == 12345
    assert len(out.manifest.override_ledger) == 2
    assert out.manifest.artifact_url == "https://minio.test/x.mp4"
    assert out.manifest.duration_seconds == 42.0
    # Both provenance sources ("default", "override") are recognised ⇒ complete.
    assert out.manifest.completeness == 1.0


async def test_active_manifest_model_dump_round_trips() -> None:
    """The output survives a pydantic dump→revalidate (JSONB-equivalent)."""
    stage = RenderManifestStage()
    ctx = _ctx(
        {
            "reproducibility": {"mode": "on", "master_seed": 12345},
            "plan_document": _PLAN_DOCUMENT,
        },
        compile_output=_COMPILE_OUTPUT,
    )

    out = await stage.handle(stage.input_model(), ctx)
    reloaded = RenderManifestOutput.model_validate(out.model_dump())

    assert reloaded.manifest.master_seed == 12345
    assert reloaded.manifest.override_ledger == out.manifest.override_ledger


# ── Per-field resolution of mastering / output_sha256 across the stage chain ──

_MASTERING: dict[str, Any] = {"integrated_lufs": -16.0, "true_peak_dbtp": -1.5}
_SHA = "e6d585fb0000000000000000000000000000000000000000000000000000db83"
_SPINE_ON: dict[str, Any] = {
    "reproducibility": {"mode": "on", "master_seed": 12345},
    "plan_document": _PLAN_DOCUMENT,
}


def _ctx_priors(priors: dict[str, dict[str, Any]]) -> RunContext:
    """A ctx with arbitrary prior stage outputs (not just demo_video_compile)."""
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=_SPINE_ON,
        prior_run_outputs=priors,
    )


async def test_mastering_and_sha_resolved_per_field_across_chain() -> None:
    """REGRESSION (prod run 01a00b81-2576-71a5-becc-ffc4a843e6db).

    `final_critique` wins the artifact walk but declares only artifact_url +
    duration_seconds. First-mapping-wins therefore dropped `mastering` and
    `output_sha256` to None even though `demo_video_compile` carried both for the
    SAME file. That is the production shape: `final_critique` (and the 9ms
    `encode_master` pass-through) republish compile's artifact URL unchanged. The
    manifest must carry ALL FOUR.
    """
    stage = RenderManifestStage()
    published = "https://minio.test/compiled.mp4"
    ctx = _ctx_priors(
        {
            "final_critique": {"artifact_url": published, "duration_seconds": 42.0},
            "demo_video_compile": {
                "artifact_url": published,
                "duration_seconds": 41.0,
                "mastering": _MASTERING,
                "output_sha256": _SHA,
            },
        }
    )

    out = await stage.handle(stage.input_model(), ctx)

    assert out.manifest.artifact_url == published
    assert out.manifest.duration_seconds == 42.0
    assert out.manifest.mastering == _MASTERING
    assert out.manifest.output_sha256 == _SHA


async def test_sha_is_never_borrowed_from_a_different_artifact() -> None:
    """A hash must describe the file it is published beside — else null.

    Adversarial review broke the first cut here: it took `output_sha256` from the
    first prior that had one (only `demo_video_compile` emits it) regardless of
    which file won `artifact_url`. So whenever a later stage genuinely RE-RENDERS,
    the manifest attested the SUPERSEDED file's hash under the new file's name. A
    stale hash in an audit record is strictly worse than a missing one: `None` is
    honest, a wrong sha is a false attestation.
    """
    stage = RenderManifestStage()
    ctx = _ctx_priors(
        {
            "final_critique": {
                "artifact_url": "https://minio.test/rerendered.mp4",
                "duration_seconds": 42.0,
            },
            "demo_video_compile": {
                "artifact_url": "https://minio.test/superseded.mp4",
                "duration_seconds": 41.0,
                "mastering": _MASTERING,
                "output_sha256": _SHA,
            },
        }
    )

    out = await stage.handle(stage.input_model(), ctx)

    assert out.manifest.artifact_url == "https://minio.test/rerendered.mp4"
    assert out.manifest.output_sha256 is None
    assert out.manifest.mastering is None


async def test_compile_only_chain_is_unchanged() -> None:
    """Only demo_video_compile present ⇒ behaviour identical to before."""
    stage = RenderManifestStage()
    ctx = _ctx_priors(
        {
            "demo_video_compile": {
                "artifact_url": "https://minio.test/compiled.mp4",
                "duration_seconds": 41.0,
                "mastering": _MASTERING,
                "output_sha256": _SHA,
            }
        }
    )

    out = await stage.handle(stage.input_model(), ctx)

    assert out.manifest.artifact_url == "https://minio.test/compiled.mp4"
    assert out.manifest.duration_seconds == 41.0
    assert out.manifest.mastering == _MASTERING
    assert out.manifest.output_sha256 == _SHA


async def test_no_stage_has_mastering_leaves_fields_null() -> None:
    """No stage carries mastering/sha ⇒ both stay None, no crash (the manifest
    never invents a value)."""
    stage = RenderManifestStage()
    ctx = _ctx_priors(
        {
            "final_critique": {
                "artifact_url": "https://minio.test/accepted.mp4",
                "duration_seconds": 42.0,
            },
            "demo_video_compile": {
                "artifact_url": "https://minio.test/compiled.mp4",
                "duration_seconds": 41.0,
            },
        }
    )

    out = await stage.handle(stage.input_model(), ctx)

    assert out.manifest.artifact_url == "https://minio.test/accepted.mp4"
    assert out.manifest.mastering is None
    assert out.manifest.output_sha256 is None


# ── Persistence round-trip through the real StageOutputRepository ─────────────


async def _new_run(session: Any, config_snapshot: dict[str, Any]) -> Any:
    proj = await ProjectRepository(session).create_with_deliverables(
        tenant_id=uuid4(),
        owner_subject_id=uuid4(),
        type_="demo_video",
        title="a20",
        brief={},
        brand={},
        voice={},
        theme={},
        visibility="tenant",
        deliverables=[
            {
                "name": "d",
                "format": "16x9",
                "captions_mode": "none",
                "target_duration_seconds": 60,
            }
        ],
    )
    await session.commit()
    run = await RunRepository(session).create(
        deliverable_id=proj.deliverables[0].id,
        blueprint_id="demo_video_v1",
        config_snapshot=config_snapshot,
        pause_after_stages=[],
    )
    await session.commit()
    return run


async def test_manifest_persists_through_stage_output_repository(db_session) -> None:
    config_snapshot = {
        "reproducibility": {"mode": "on", "master_seed": 12345},
        "plan_document": _PLAN_DOCUMENT,
    }
    stage = RenderManifestStage()
    ctx = _ctx(config_snapshot, compile_output=_COMPILE_OUTPUT)
    out = await stage.handle(stage.input_model(), ctx)

    run = await _new_run(db_session, config_snapshot)
    run_id = run.id  # captured before expire_all() to avoid a sync lazy-refresh
    repo = StageOutputRepository(db_session)
    await repo.upsert_pending(run_id, "render_manifest")
    await repo.mark_completed(run_id, "render_manifest", out.model_dump(mode="json"))
    await db_session.commit()
    db_session.expire_all()  # re-read from postgres, not the identity map

    stored = await repo.get(run_id, "render_manifest")
    assert stored is not None
    stored_output = stored.output
    assert stored_output is not None

    manifest = stored_output["manifest"]
    # The seed + ledger survived the JSONB round-trip intact.
    assert manifest["master_seed"] == 12345
    assert len(manifest["override_ledger"]) == 2
    assert {row["source"] for row in manifest["override_ledger"]} == {"default", "override"}
    assert manifest["artifact_url"] == "https://minio.test/x.mp4"

    # And the typed model re-validates from the persisted JSONB.
    reloaded = RenderManifestOutput.model_validate(stored_output)
    assert reloaded.manifest.master_seed == 12345
