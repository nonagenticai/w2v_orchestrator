"""Unit tests for the pure render/provenance manifest (A20).

No DB — these are pure functions. The real :mod:`provenance` helpers are called
directly (no mocks).
"""

from __future__ import annotations

from src.services.stages.shared.render_manifest import (
    KnobRecord,
    RenderManifest,
    assemble_manifest,
    manifest_completeness,
    override_ledger,
    reproducibility_drift,
)


def _snapshot_with_provenance(resolution_value: str = "1080p") -> dict[str, object]:
    """A snapshot carrying a plan_document with three resolved knobs.

    ``resolution_value`` lets a caller vary one knob's resolved value without
    subscripting into the nested ``object``-typed dict (keeps the test ty-clean).
    """
    return {
        "project_id": "proj-abc",
        "plan_document": {
            "provenance": {
                "resolution": {
                    "value": resolution_value,
                    "source": "override",
                    "computed_by": None,
                },
                "ken_burns_zoom": {
                    "value": 1.15,
                    "source": "smart-auto",
                    "computed_by": "motion_planner",
                },
                "preset": {"value": "default", "source": "default", "computed_by": None},
            }
        },
    }


def test_override_ledger_flattens_three_knobs() -> None:
    ledger = override_ledger(_snapshot_with_provenance())
    assert len(ledger) == 3
    by_key = {r.key: r for r in ledger}
    assert by_key["resolution"].source == "override"
    assert by_key["resolution"].value == "1080p"
    assert by_key["ken_burns_zoom"].source == "smart-auto"
    assert by_key["ken_burns_zoom"].computed_by == "motion_planner"
    assert by_key["preset"].source == "default"


def test_override_ledger_empty_without_plan_document() -> None:
    assert override_ledger({"project_id": "proj-abc"}) == []
    # Malformed plan_document / provenance also yields an empty ledger.
    assert override_ledger({"plan_document": {}}) == []
    assert override_ledger({"plan_document": {"provenance": None}}) == []


def test_manifest_completeness_all_valid_is_one() -> None:
    ledger = override_ledger(_snapshot_with_provenance())
    assert manifest_completeness(ledger) == 1.0
    # Empty ledger is vacuously complete.
    assert manifest_completeness([]) == 1.0


def test_manifest_completeness_unknown_source_drops_below_one() -> None:
    ledger = [
        KnobRecord(key="a", value=1, source="override"),
        KnobRecord(key="b", value=2, source=""),
    ]
    assert manifest_completeness(ledger) == 0.5
    ledger_unknown = [
        KnobRecord(key="a", value=1, source="override"),
        KnobRecord(key="b", value=2, source="unknown"),
    ]
    assert manifest_completeness(ledger_unknown) == 0.5


def test_reproducibility_drift_identical_is_empty() -> None:
    snap = _snapshot_with_provenance()
    snap["reproducibility"] = {"master_seed": 777}
    other = _snapshot_with_provenance()
    other["reproducibility"] = {"master_seed": 777}
    assert reproducibility_drift(snap, other) == []


def test_reproducibility_drift_reports_changed_knob() -> None:
    snap = _snapshot_with_provenance()
    snap["reproducibility"] = {"master_seed": 777}
    changed = _snapshot_with_provenance(resolution_value="720p")
    changed["reproducibility"] = {"master_seed": 777}
    assert reproducibility_drift(snap, changed) == ["resolution"]


def test_reproducibility_drift_includes_master_seed_when_seeds_differ() -> None:
    snap = _snapshot_with_provenance()
    snap["reproducibility"] = {"master_seed": 111}
    other = _snapshot_with_provenance()
    other["reproducibility"] = {"master_seed": 222}
    drift = reproducibility_drift(snap, other)
    assert "master_seed" in drift
    # Knobs are identical, so master_seed is the only entry.
    assert drift == ["master_seed"]


def test_assemble_manifest_populates_fields() -> None:
    snap = _snapshot_with_provenance()
    snap["reproducibility"] = {"master_seed": 4242}
    manifest = assemble_manifest(
        snap,
        stage_outputs={
            "demo_video_compile": {
                "artifact_url": "https://minio.test/x.mp4",
                "duration_seconds": 42.0,
                "mastering": {"integrated_lufs": -14.1, "true_peak_db": -1.2},
                "output_sha256": "a" * 64,
            }
        },
    )
    assert isinstance(manifest, RenderManifest)
    assert manifest.master_seed == 4242
    assert len(manifest.override_ledger) == 3
    assert manifest.override_ledger[0].keys() == {"key", "value", "source", "computed_by"}
    assert manifest.input_hash is not None
    assert manifest.completeness == 1.0
    assert manifest.artifact_url == "https://minio.test/x.mp4"
    assert manifest.duration_seconds == 42.0
    # Mastering report + content hash pass through from the compile stage output.
    assert manifest.mastering == {"integrated_lufs": -14.1, "true_peak_db": -1.2}
    assert manifest.output_sha256 == "a" * 64


def test_assemble_manifest_without_compile_output() -> None:
    snap = _snapshot_with_provenance()
    snap["reproducibility"] = {"master_seed": 9}
    manifest = assemble_manifest(snap, stage_outputs={})
    assert manifest.master_seed == 9
    assert manifest.artifact_url is None
    assert manifest.duration_seconds is None
    # Absent compile output ⇒ mastering / output_sha256 degrade to None, never invented.
    assert manifest.mastering is None
    assert manifest.output_sha256 is None
    assert len(manifest.override_ledger) == 3


def test_assemble_manifest_compile_output_without_enrichment() -> None:
    """A compile output lacking mastering/output_sha256 (older compiler) ⇒ None."""
    snap = _snapshot_with_provenance()
    snap["reproducibility"] = {"master_seed": 7}
    manifest = assemble_manifest(
        snap,
        stage_outputs={
            "demo_video_compile": {
                "artifact_url": "https://minio.test/y.mp4",
                "duration_seconds": 12.0,
            }
        },
    )
    assert manifest.artifact_url == "https://minio.test/y.mp4"
    assert manifest.mastering is None
    assert manifest.output_sha256 is None
