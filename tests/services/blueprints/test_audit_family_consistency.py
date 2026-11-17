"""Guard test: the audit-type enumeration cannot silently drift across layers.

The URL→scorecard audit family (``seo_audit`` / ``cro_audit`` /
``marketing_scorecard``) has one source of truth — ``AUDIT_FAMILIES`` in
``src.services.blueprints.audit_family`` — but the same set of project-type
strings is *also* spelled out in three places that cannot import that module
(a cycle risk, or they need static/literal members): the ``ProjectType``
``Literal``, the ``ck_projects_type`` CHECK constraint on the ``projects`` table,
and the ``_BRIEF_DOOR_MODELS`` create-door mapping.

These asserts turn "currently consistent" into "CI-enforced": adding an audit
type to ``AUDIT_FAMILIES`` without threading it through the Literal / CHECK /
brief-door (the exact class of drift that once cost the marketing type its
hero + Download-PDF) fails here instead of shipping a half-wired type.

The migration copy is locked too. The other asserts read the CHECK from the
SQLAlchemy *model*, and the suite builds its schema with ``create_all`` (never
alembic) — so the alembic migration's ``ck_projects_type`` string was the one
hand-maintained copy with no lock. A type added to the model + Literal but
forgotten/typo'd in the migration passes every model-based assert yet ships a
prod-only ``IntegrityError`` on the first insert (prod runs ``alembic upgrade
head``, not ``create_all``). ``test_migration_check_constraint_matches_model``
closes that: model + Literal + migration are now one enforced set.

Pure metadata test — no DB connection required (the migration check walks the
alembic revision graph and reads the head-most migration's CHECK statically).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast, get_args

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import CheckConstraint, Table

from src.api.schemas import _BRIEF_DOOR_MODELS, AuditBrief, ProjectType
from src.database.models import Project
from src.services.blueprints.audit_family import AUDIT_PROJECT_TYPES

# tests/services/blueprints/ → repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _check_constraint_types() -> set[str]:
    """The allowed ``type`` values from the ``ck_projects_type`` CHECK constraint."""
    table = cast("Table", Project.__table__)
    for constraint in table.constraints:
        if isinstance(constraint, CheckConstraint) and constraint.name == "ck_projects_type":
            # sqltext is ``type IN ('promo','course',...)`` — pull the quoted values.
            return set(re.findall(r"'([^']+)'", str(constraint.sqltext)))
    raise AssertionError("ck_projects_type constraint not found on the projects table")


def _migration_check_constraint_types() -> set[str]:
    """The final ``ck_projects_type`` type set the alembic chain produces.

    Walks the real revision graph (head→base) and returns the type list from the
    head-most migration that (re)defines ``ck_projects_type``. Each type-adding
    migration declares the full list as a module-level ``_TYPE_NEW`` (see 0042);
    that convention is what this locks, so a drifted migration string fails CI
    instead of as a prod-only insert error.
    """
    cfg = Config()
    cfg.set_main_option("script_location", str(_REPO_ROOT / "src/database/migrations"))
    script = ScriptDirectory.from_config(cfg)
    for rev in script.walk_revisions():  # head → base
        if "ck_projects_type" not in Path(rev.path).read_text():
            continue
        type_new = getattr(rev.module, "_TYPE_NEW", None)
        if not isinstance(type_new, str):
            raise AssertionError(
                f"migration {rev.revision} ({Path(rev.path).name}) redefines "
                "ck_projects_type but exposes no module-level `_TYPE_NEW` string — "
                "follow the 0042 convention so the CHECK stays statically lockable."
            )
        return set(re.findall(r"'([^']+)'", type_new))
    raise AssertionError("no migration defines ck_projects_type")


def test_audit_types_are_valid_project_types() -> None:
    """Every audit family must be a member of the ProjectType Literal."""
    assert set(AUDIT_PROJECT_TYPES) <= set(get_args(ProjectType))


def test_project_type_literal_matches_db_check_constraint() -> None:
    """The Pydantic Literal and the DB CHECK must list the SAME project types.

    Locks the two hand-maintained copies (schemas Literal ↔ models CHECK) to
    each other for ALL types, so the model constraint cannot drift from the API
    contract (and, transitively, from the migration that must reproduce it)."""
    assert set(get_args(ProjectType)) == _check_constraint_types()


def test_audit_types_allowed_by_db_check_constraint() -> None:
    """Every audit family must be accepted by the projects CHECK constraint."""
    assert set(AUDIT_PROJECT_TYPES) <= _check_constraint_types()


def test_audit_types_have_typed_brief_door() -> None:
    """Every audit family must map to the AuditBrief door model."""
    for project_type in AUDIT_PROJECT_TYPES:
        assert _BRIEF_DOOR_MODELS.get(project_type) is AuditBrief


def test_migration_check_constraint_matches_model() -> None:
    """The alembic migration's ck_projects_type must match the model CHECK.

    The other asserts read the CHECK from the model and the suite builds schema
    via ``create_all`` (never alembic), so the migration string was unverified.
    Locking it to the model (already locked to the Literal) makes model + Literal
    + migration one enforced set: a forgotten/typo'd migration fails here instead
    of as a prod-only IntegrityError on the first insert of the new type.
    """
    assert _migration_check_constraint_types() == _check_constraint_types()
