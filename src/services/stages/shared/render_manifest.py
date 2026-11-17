"""Render / provenance manifest (A20) — the run's audit artifact.

The pipeline resolves every knob down the provenance ladder (override →
smart-auto → default, see :mod:`provenance`) but historically kept no single
artifact that explains *how the delivered video was configured*. This module is
that artifact: given a run's ``config_snapshot`` it computes a flat override
ledger (one record per resolved knob), a completeness score (are all sources
recognised?), a reproducibility-drift diff between two snapshots, and an
assembled :class:`RenderManifest` a stage handler can persist.

Pure — no I/O, no DB, no run context. Every function takes plain mappings and
returns plain values/pydantic models so it is trivially unit-testable and safe
to call from any stage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field

from src.services.stages.shared.provenance import VALID_SOURCES, input_hash, master_seed


@dataclass(frozen=True)
class KnobRecord:
    """One resolved knob in the override ledger.

    ``key`` is the knob name, ``value`` its resolved value, ``source`` where it
    came from (should be one of :data:`provenance.VALID_SOURCES`), and
    ``computed_by`` the heuristic/model that produced a computed/clamped value.
    """

    key: str
    value: object
    source: str
    computed_by: str | None = None


def _as_str_mapping(value: object) -> dict[str, object]:
    """Coerce an untyped nested config value into a ``dict[str, object]``.

    Rebuilds the mapping with string keys so downstream ``.get`` calls are
    well-typed (a value narrowed from ``object`` via ``isinstance`` is
    ``Mapping[Unknown, Unknown]``, whose ``.get`` rejects a ``str`` key). A
    non-mapping yields ``{}`` — so a run with no ``plan_document`` or a malformed
    block degrades to an empty ledger rather than raising.
    """
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    return {}


def _provenance_block(config_snapshot: Mapping[str, object]) -> dict[str, object]:
    """Return ``config_snapshot['plan_document']['provenance']`` or an empty map.

    Guards every level via :func:`_as_str_mapping` so a run that never had a
    ``VideoIntent`` (no ``plan_document``) — or a malformed one — yields an empty
    ledger rather than raising.
    """
    plan_document = _as_str_mapping(config_snapshot.get("plan_document"))
    return _as_str_mapping(plan_document.get("provenance"))


def override_ledger(config_snapshot: Mapping[str, object]) -> list[KnobRecord]:
    """Flatten ``plan_document.provenance`` into one :class:`KnobRecord` per knob.

    Reads the serialized :class:`KnobProvenance` entries. Returns an empty list
    when the plan document (or its provenance) is absent. Missing ``source`` is
    coerced to ``""`` so :func:`manifest_completeness` can flag it.
    """
    records: list[KnobRecord] = []
    for key, entry in _provenance_block(config_snapshot).items():
        if not isinstance(entry, Mapping):
            continue
        entry_map = _as_str_mapping(entry)
        source = entry_map.get("source")
        computed_by = entry_map.get("computed_by")
        records.append(
            KnobRecord(
                key=str(key),
                value=entry_map.get("value"),
                source=str(source) if source is not None else "",
                computed_by=str(computed_by) if computed_by is not None else None,
            )
        )
    return records


def manifest_completeness(ledger: Sequence[KnobRecord]) -> float:
    """Fraction of records whose ``source`` is a recognised provenance source.

    An empty ledger is vacuously complete (``1.0``). Any record carrying an
    unknown or empty source drops the score below ``1.0``.
    """
    if not ledger:
        return 1.0
    valid = sum(1 for r in ledger if r.source in VALID_SOURCES)
    return valid / len(ledger)


def _snapshot_master_seed(snapshot: Mapping[str, object]) -> int:
    """Resolve a snapshot's master seed: prefer ``reproducibility.master_seed``.

    Falls back to :func:`provenance.master_seed` when the reproducibility block
    is absent or lacks an integer ``master_seed``.
    """
    reproducibility = _as_str_mapping(snapshot.get("reproducibility"))
    seed = reproducibility.get("master_seed")
    if isinstance(seed, int) and not isinstance(seed, bool):
        return seed
    return master_seed(dict(snapshot))


def reproducibility_drift(
    snapshot_a: Mapping[str, object], snapshot_b: Mapping[str, object]
) -> list[str]:
    """Return the sorted knob keys whose resolved value differs between snapshots.

    Compares the resolved values from each snapshot's :func:`override_ledger`; a
    key present in only one snapshot counts as drift. Appends the literal
    ``"master_seed"`` when the two snapshots' master seeds differ (via
    :func:`_snapshot_master_seed`). Identical snapshots yield ``[]``.
    """
    values_a = {r.key: r.value for r in override_ledger(snapshot_a)}
    values_b = {r.key: r.value for r in override_ledger(snapshot_b)}

    differing: set[str] = set()
    for key in values_a.keys() | values_b.keys():
        if values_a.get(key) != values_b.get(key):
            differing.add(key)

    drift = sorted(differing)
    if _snapshot_master_seed(snapshot_a) != _snapshot_master_seed(snapshot_b):
        drift.append("master_seed")
    return drift


class RenderManifest(BaseModel):
    """The assembled render/provenance manifest for one run.

    ``mastering`` (the compiler's post-mux loudness/limiter report) and
    ``output_sha256`` (the delivered file's content hash) are surfaced from the
    ``demo_video_compile`` stage output when present, and stay ``None`` otherwise
    (an older compiler build or a non-compile run). All fields are computed by
    :func:`assemble_manifest`.
    """

    master_seed: int | None = None
    input_hash: str | None = None
    override_ledger: list[dict[str, object]] = Field(default_factory=list)
    completeness: float = 1.0
    output_sha256: str | None = None  # delivered-file content hash (compiler-supplied)
    mastering: dict[str, object] | None = None  # compiler post-mux loudness/limiter report
    artifact_url: str | None = None
    duration_seconds: float | None = None


def assemble_manifest(
    config_snapshot: Mapping[str, object],
    *,
    stage_outputs: Mapping[str, Mapping[str, object]],
    include_spine: bool = True,
) -> RenderManifest:
    """Compute the :class:`RenderManifest` from a run's snapshot + stage outputs.

    The override ledger and its completeness come from :func:`override_ledger` /
    :func:`manifest_completeness` and are ALWAYS populated (the audit record that
    lights up the UI on every run). ``artifact_url`` / ``duration_seconds`` /
    ``mastering`` / ``output_sha256`` are taken best-effort from the
    ``demo_video_compile`` stage output when present. ``mastering`` (the compiler's
    post-mux loudness/limiter report) and ``output_sha256`` (the delivered file's
    content hash) degrade to ``None`` when the compile output lacks them — an older
    compiler build or a non-compile run — so the manifest never invents a value.

    ``include_spine`` gates the reproducibility spine (``master_seed`` +
    ``input_hash``): when ``True`` the master seed prefers
    ``config_snapshot['reproducibility']['master_seed']`` and falls back to
    :func:`provenance.master_seed`, and ``input_hash`` is a stable hash over a
    deterministic ``(key, source, value)`` projection of the ledger. When ``False``
    (a run that did NOT opt into reproducibility) both stay ``None`` — the ledger
    is still emitted, but the manifest makes no reproducibility promise it can't keep.
    """
    ledger = override_ledger(config_snapshot)
    completeness = manifest_completeness(ledger)

    ledger_payload: list[dict[str, object]] = [
        {"key": r.key, "value": r.value, "source": r.source, "computed_by": r.computed_by}
        for r in ledger
    ]

    # Deterministic, order-insensitive projection of the ledger for hashing.
    projection = sorted((r.key, r.source, str(r.value)) for r in ledger)
    ledger_hash = input_hash(projection)

    artifact_url: str | None = None
    duration_seconds: float | None = None
    mastering: dict[str, object] | None = None
    output_sha256: str | None = None
    compile_output = stage_outputs.get("demo_video_compile")
    if isinstance(compile_output, Mapping):
        raw_url = compile_output.get("artifact_url")
        if isinstance(raw_url, str):
            artifact_url = raw_url
        raw_duration = compile_output.get("duration_seconds")
        if isinstance(raw_duration, (int, float)) and not isinstance(raw_duration, bool):
            duration_seconds = float(raw_duration)
        raw_mastering = compile_output.get("mastering")
        if isinstance(raw_mastering, Mapping):
            mastering = _as_str_mapping(raw_mastering)
        raw_sha = compile_output.get("output_sha256")
        if isinstance(raw_sha, str):
            output_sha256 = raw_sha

    return RenderManifest(
        master_seed=_snapshot_master_seed(config_snapshot) if include_spine else None,
        input_hash=ledger_hash if include_spine else None,
        override_ledger=ledger_payload,
        completeness=completeness,
        output_sha256=output_sha256,
        mastering=mastering,
        artifact_url=artifact_url,
        duration_seconds=duration_seconds,
    )
