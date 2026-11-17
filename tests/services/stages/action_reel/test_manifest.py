"""action_reel render_manifest — the ledger reads the artifact off `compile`.

The shared handler's artifact chain names demo_video stages, none of which exist
in an action_reel run, so reusing it unchanged left every reel's manifest with an
empty ``artifact_url``/``duration_seconds`` — the ledger emitted, but promised no
deliverable. These cover the fix (the subclass reads `compile`) and, just as
load-bearing, that demo_video's existing precedence is untouched by the
refactor from a hardcoded `or` chain to the ``artifact_stage_names`` walk.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from src.services.blueprints.action_reel import ACTION_REEL_BLUEPRINT
from src.services.runs.run_context import RunContext
from src.services.stages.action_reel.manifest import ActionReelRenderManifestStage
from src.services.stages.demo_video.manifest import RenderManifestStage

_PLAN_DOCUMENT: dict[str, Any] = {
    "provenance": {
        "resolution": {"value": "2k", "source": "default"},
        "music_gain_db": {"value": -22, "source": "override"},
    }
}
_SNAPSHOT: dict[str, Any] = {
    "reproducibility": {"mode": "on", "master_seed": 12345},
    "plan_document": _PLAN_DOCUMENT,
}


def _ctx(prior: dict[str, dict[str, Any]]) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=_SNAPSHOT,
        prior_run_outputs=prior,
    )


# ── action_reel: the artifact comes off `compile` ─────────────────────────────


@pytest.mark.asyncio
async def test_action_reel_manifest_reads_compile_stage() -> None:
    """An action_reel-shaped run (the render stage is named `compile`) yields a
    manifest carrying the delivered artifact — the whole point of the subclass."""
    stage = ActionReelRenderManifestStage()
    ctx = _ctx(
        {
            "compile": {
                "artifact_url": "https://minio.test/reel.mp4",
                "duration_seconds": 31.5,
            }
        }
    )

    out = await stage.handle(stage.input_model(), ctx)

    assert out.manifest.artifact_url == "https://minio.test/reel.mp4"
    assert out.manifest.duration_seconds == 31.5
    # The audit half is unchanged by the override.
    assert len(out.manifest.override_ledger) == 2
    assert out.manifest.completeness == 1.0
    assert out.manifest.master_seed == 12345


@pytest.mark.asyncio
async def test_action_reel_manifest_without_compile_output_still_emits() -> None:
    """A partial run (compile never produced) still emits the ledger, with the
    artifact slot empty rather than raising."""
    stage = ActionReelRenderManifestStage()

    out = await stage.handle(stage.input_model(), _ctx({}))

    assert out.manifest.artifact_url is None
    assert out.manifest.duration_seconds is None
    assert len(out.manifest.override_ledger) == 2


@pytest.mark.asyncio
async def test_base_stage_ignores_action_reel_compile() -> None:
    """The regression this fixes, pinned: the UNSUBCLASSED handler finds nothing
    in an action_reel run — which is exactly why the blueprint must register the
    subclass, not the base."""
    stage = RenderManifestStage()

    out = await stage.handle(
        stage.input_model(),
        _ctx({"compile": {"artifact_url": "https://minio.test/reel.mp4"}}),
    )

    assert out.manifest.artifact_url is None


def test_blueprint_registers_the_action_reel_subclass() -> None:
    built = ACTION_REEL_BLUEPRINT.stage("render_manifest").handler_factory({})
    assert isinstance(built, ActionReelRenderManifestStage)
    assert built.artifact_stage_names == ("compile",)
    # Still the shared stage as far as the DAG is concerned.
    assert built.name == "render_manifest"


# ── demo_video precedence is byte-identical after the refactor ────────────────


@pytest.mark.parametrize(
    ("prior", "expected"),
    [
        (
            {
                "final_critique": {"artifact_url": "u:final"},
                "demo_video_encode_master": {"artifact_url": "u:master"},
                "demo_video_footage_critique": {"artifact_url": "u:footage"},
                "demo_video_compile": {"artifact_url": "u:compile"},
            },
            "u:final",
        )
    ],
)
@pytest.mark.asyncio
async def test_demo_video_precedence_unchanged(
    prior: dict[str, dict[str, Any]], expected: str
) -> None:
    stage = RenderManifestStage()

    out = await stage.handle(stage.input_model(), _ctx(prior))

    assert out.manifest.artifact_url == expected
