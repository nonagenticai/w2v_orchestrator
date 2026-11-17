"""Config-driven audit blueprint family — seo_audit / cro_audit / marketing_scorecard.

These three product types are the **same DAG shape**: a synchronous
analysis_service scorecard scan (``<type>_scan`` → ``POST /api/{cro,seo,marketing}``)
whose structured output IS the deliverable, plus an OPTIONAL narrated-video
sub-chain (``script → slides → {narrate ‖ record} → compile``) gated on
``brief.generate_video`` (off by default — each video stage ``require_*_video``-gates
first, so a scan-only run skips the whole chain and completes on the scan output).

They previously lived as three near-identical blueprint modules. Declared here as
``AuditFamily`` config + one :func:`build_audit_blueprint` factory instead: a new
**scan-only** audit type is a single ``AuditFamily`` entry. The video sub-chain
still references the per-type stage *classes* (the script prompt and compile shots
genuinely differ); genericising those into parameterised stages is a separate step.

Every family also gets an OPT-IN ``source_ingest`` leaf that the scan depends on:
when the brief carries ``sources`` it scrapes them into ``grounding_text`` that the
scan forwards to the scorecard's claim-grounding; with no ``sources`` it skips, so a
plain audit run is byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.services.blueprints._factories import SOURCE_INGEST_FACTORY
from src.services.blueprints.base import DAGBlueprint, StageSpec, handler
from src.services.blueprints.registry import get_registry
from src.services.stages.base import StageHandler
from src.services.stages.cro_audit.compile import CroAuditCompileStage
from src.services.stages.cro_audit.narrate import CroAuditNarrateStage
from src.services.stages.cro_audit.record import CroAuditRecordStage
from src.services.stages.cro_audit.scan import CroAuditScanStage
from src.services.stages.cro_audit.script import CroAuditScriptStage
from src.services.stages.cro_audit.slides import CroAuditSlidesStage
from src.services.stages.marketing_scorecard.scan import MarketingScorecardScanStage
from src.services.stages.primitives.compile_stage import COMPILE_STAGE_OUTER_TIMEOUT_SECONDS
from src.services.stages.seo_audit.compile import SeoAuditCompileStage
from src.services.stages.seo_audit.narrate import SeoAuditNarrateStage
from src.services.stages.seo_audit.record import SeoAuditRecordStage
from src.services.stages.seo_audit.scan import SeoAuditScanStage
from src.services.stages.seo_audit.script import SeoAuditScriptStage
from src.services.stages.seo_audit.slides import SeoAuditSlidesStage


@dataclass(frozen=True)
class VideoStages:
    """The five stage classes of the optional narrated-video sub-chain."""

    script: type[StageHandler]
    slides: type[StageHandler]
    narrate: type[StageHandler]
    record: type[StageHandler]
    compile: type[StageHandler]


@dataclass(frozen=True)
class AuditFamily:
    """One scorecard product type. ``video=None`` → scan-only (terminal=None)."""

    project_type: str
    scan_stage: type[StageHandler]
    video: VideoStages | None = None


def build_audit_blueprint(f: AuditFamily) -> DAGBlueprint:
    """Assemble the DAGBlueprint for one audit family (scan-only or scan+video)."""
    p = f.project_type
    stages = [
        # OPT-IN multi-source grounding: source_ingest scrapes/parses brief.sources
        # into grounding_text. It StageSkips (completed-empty) when the brief carries
        # no sources, so a plain audit run is byte-identical. The scan depends on it
        # so the ingested text is available to ground the recommendation-writer's
        # claims against the operator's sources (in addition to the scanned page).
        StageSpec(
            name="source_ingest",
            handler_factory=SOURCE_INGEST_FACTORY,
            default_timeout_seconds=900,
        ),
        # Sync scorer: scan(60s) retried once → ~120s worst case; give margin.
        StageSpec(
            name=f"{p}_scan",
            depends_on=("source_ingest",),
            handler_factory=handler(f.scan_stage),
            default_timeout_seconds=180,
        ),
    ]
    terminal: str | None = None
    if f.video is not None:
        v = f.video
        # OPTIONAL narrated-video sub-chain (brief.generate_video; off by default).
        # Each stage require_*_video()-gates first → skipped when the flag is absent,
        # so a plain scan-only run is unchanged (skipped terminal → run completes).
        stages += [
            StageSpec(
                name=f"{p}_script",
                depends_on=(f"{p}_scan",),
                handler_factory=handler(v.script),
            ),
            StageSpec(
                name=f"{p}_slides",
                depends_on=(f"{p}_script",),
                handler_factory=handler(v.slides),
                default_timeout_seconds=60,
            ),
            StageSpec(
                # narrate and record both depend on slides so they become eligible in
                # the same scheduler round and run concurrently.
                name=f"{p}_narrate",
                depends_on=(f"{p}_slides",),
                handler_factory=handler(v.narrate),
                default_timeout_seconds=1300,
            ),
            StageSpec(
                name=f"{p}_record",
                depends_on=(f"{p}_slides",),
                handler_factory=handler(v.record),
                default_timeout_seconds=900,
            ),
            StageSpec(
                # Outer timeout must cover the compile-wait backstop below it —
                # see compile_stage.COMPILE_STAGE_OUTER_TIMEOUT_SECONDS.
                name=f"{p}_compile",
                depends_on=(f"{p}_narrate", f"{p}_record"),
                handler_factory=handler(v.compile),
                default_timeout_seconds=COMPILE_STAGE_OUTER_TIMEOUT_SECONDS,
            ),
        ]
        terminal = f"{p}_compile"
    return DAGBlueprint(
        id=f"{p}_v1",
        project_type=p,
        terminal_artifact_stage=terminal,
        stages=stages,
    )


AUDIT_FAMILIES: list[AuditFamily] = [
    AuditFamily(
        project_type="seo_audit",
        scan_stage=SeoAuditScanStage,
        video=VideoStages(
            script=SeoAuditScriptStage,
            slides=SeoAuditSlidesStage,
            narrate=SeoAuditNarrateStage,
            record=SeoAuditRecordStage,
            compile=SeoAuditCompileStage,
        ),
    ),
    AuditFamily(
        project_type="cro_audit",
        scan_stage=CroAuditScanStage,
        video=VideoStages(
            script=CroAuditScriptStage,
            slides=CroAuditSlidesStage,
            narrate=CroAuditNarrateStage,
            record=CroAuditRecordStage,
            compile=CroAuditCompileStage,
        ),
    ),
    # Scan-only (no video variant yet) — a single config entry, terminal=None.
    AuditFamily(project_type="marketing_scorecard", scan_stage=MarketingScorecardScanStage),
]

# The audit family's project-type strings — the single source of truth. New audit
# types are added ONLY to ``AUDIT_FAMILIES`` above; api-layer consumers derive from
# this tuple (one-way import: this module imports only ``services.*``, never ``api``
# or ``database``). The copies that CANNOT import this without a cycle or that need
# static members — the ``ProjectType`` Literal, the ``ck_projects_type`` CHECK, the
# alembic migration — are locked to this tuple by
# ``tests/services/blueprints/test_audit_family_consistency.py``.
AUDIT_PROJECT_TYPES: tuple[str, ...] = tuple(f.project_type for f in AUDIT_FAMILIES)

for _family in AUDIT_FAMILIES:
    get_registry().register(build_audit_blueprint(_family))
