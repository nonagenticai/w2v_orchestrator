"""Shared config + leaf Literal aliases for the schemas package.

These are the leaf types with no model dependencies — the base of the
package's import order (``_shared`` ← everything else).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BeforeValidator, ConfigDict

from src.models.failure_class import FailureClass as _FailureClass

_camel = ConfigDict(populate_by_name=True, str_strip_whitespace=True)


# ─── Projects ────────────────────────────────────────────────────────────

ProjectType = Literal[
    "promo",
    "course",
    "app_review",
    "generic",
    "lesson",
    "website_review",
    "audiobook",
    "web_design",
    "web_design_v2",
    "demo_video",
    "showcase",
    "growth_bundle",
    "seo_audit",
    "cro_audit",
    "marketing_scorecard",
    "action_reel",
]
ProjectStatus = Literal["drafting", "producing", "completed", "failed", "archived"]
Visibility = Literal["private", "tenant", "public"]
DeliverableFormat = Literal["16x9", "9x16", "1x1", "4x5", "audio"]
CaptionsMode = Literal["none", "burned_in", "overlay"]
#: Input-only spellings folded onto the canonical ``CaptionsMode`` at the request
#: boundary, so storage (``ck_deliverables_captions_mode``), the compile stages and
#: every response only ever see ``burned_in``. ``burned`` is what the pre-v3
#: schema (migration 0005) called it and what callers still naturally type.
CAPTIONS_MODE_ALIASES: dict[str, str] = {"burned": "burned_in"}


def _normalize_captions_mode(value: object) -> object:
    if isinstance(value, str):
        stripped = value.strip()
        return CAPTIONS_MODE_ALIASES.get(stripped, stripped)
    return value


#: Request-side ``captions_mode``: accepts ``burned`` as an alias of
#: ``burned_in``; anything outside the canonical set still 422s.
CaptionsModeInput = Annotated[CaptionsMode, BeforeValidator(_normalize_captions_mode)]
# Derived on read from the deliverable's latest run (never persisted — the
# `deliverables.status` column stays at its 'idle' default). See
# `DeliverableRepository.derive_deliverable_status`.
# ``paused`` (0102) = the latest run was paused by a person, NOT at a review gate;
# ``awaiting_review`` stays the checkpoint case (and a pre-0102 paused row).
DeliverableStatus = Literal["idle", "running", "awaiting_review", "paused", "completed", "failed"]


# ─── Stage outputs ───────────────────────────────────────────────────────

# Must stay in EXACT parity with the DB's ``ck_stage_outputs_status`` CHECK
# (see `StageOutput.__table_args__` + migration 0059, which added 'cancelled'
# for `StageOutputRepository.bulk_cancel_non_terminal`). A DB-only value 500s
# every read that validates it; a schema-only value is a dead branch the DB
# rejects on write. Guarded by
# `tests/contracts/test_stage_status_check_constraint_parity.py`.
StageStatus = Literal["pending", "running", "completed", "failed", "skipped", "cancelled"]


# ─── Runs ────────────────────────────────────────────────────────────────

RunStatus = Literal["pending", "running", "paused", "completed", "failed", "cancelled"]

# Fault attribution for a FAILED run — "was this ours or theirs?". RE-EXPORTED,
# not redefined: the canonical Literal (and the reasoning for each member) lives
# in `src/models/failure_class.py`, because the database repository and the
# dispatcher need the same vocabulary and neither may import the API package.
# Aliasing here rather than retyping means the API and the writer can never
# drift. `None` means the run predates the column or was never classified, which
# is NOT the same statement as `'unknown'` ("classified, and the evidence did
# not attribute"). Parity with the DB CHECK is guarded by
# `tests/contracts/test_failure_class_check_constraint_parity.py`, which runs
# its DB-vs-Python assertion against THIS export as well as the canonical one.
FailureClass = _FailureClass


# ─── SSE event payloads ──────────────────────────────────────────────────

RunEventName = Literal[
    "stage_started",
    "stage_completed",
    "stage_failed",
    # `stage_skipped` is what the DAG executor publishes live
    # (`dag_executor._run_stage`); `stage_cancelled` has no live publisher —
    # cancellation is a DB-level sweep (`bulk_cancel_non_terminal`), so the
    # catch-up snapshot is the only emitter. Both exist so a reconnecting
    # client never sees a terminal stage as a ghost 'pending' badge.
    "stage_skipped",
    "stage_cancelled",
    "run_started",
    "run_paused",
    "run_completed",
    "run_failed",
    "run_cancelled",
]


# ─── Annotations ────────────────────────────────────────────────────────

AnnotationStatus = Literal["open", "applied", "dismissed"]


# ─── Verbatim items + growth bundle ─────────────────────────────────────

VerbatimKind = Literal["slide", "tts"]
VerbatimScope = Literal["pipeline", "chapter", "scene", "shot", "slide"]
VerbatimStatus = Literal["pending", "placed", "rejected"]
