"""``clone_from_project_id`` — copy a source project's settings into a new project.

The wizard's "Clone settings from" select promises (FE ``BriefStep.svelte``):
"Copies brand / voice / theme configs and brief defaults from the source project."
Until UI review 2026-10-17 (X3) the orchestrator declared the field and ignored it
on every door, so the select was inert.

Rules, in one place so the three doors (``POST /projects``, ``POST /projects/draft``,
``PATCH /projects/{id}``) cannot disagree:

* **Tenant-scoped.** The source is read with the caller's tenant (and the request's
  RLS-bound session); another tenant's id, or an unknown one, is the same
  :class:`CloneSourceNotFoundError` → 404, so existence cannot be probed.
* **Explicit wins.** Every dict is merged KEY-WISE with the cloned values as the
  base and the request's (or, on PATCH, the project's current) values on top.
* **Brief defaults only, same type only.** A brief is type-specific; copying one
  across types would feed a pipeline keys it does not read (or fail its door model).
  Content and per-project bindings are never copied: login blocks (sealed secrets
  belong to their own project), and every source/target URL or reference — the new
  project is about something else; only its *style* is cloned.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from src.services.auth_secrets import AUTH_BLOCK_KEYS

if TYPE_CHECKING:
    from src.database.repositories.project_repository import ProjectRepository

__all__ = [
    "CLONE_SOURCE_NOT_FOUND",
    "CloneSourceNotFoundError",
    "ClonedSettings",
    "load_clone_source",
    "merge_under",
]

#: ``error.details.code`` of the 404 for an unknown / other-tenant clone source.
CLONE_SOURCE_NOT_FOUND = "clone_source_not_found"

# Brief keys that bind a project to ITS content, never copied by a clone. Anything
# ending in ``_url`` is excluded too (target_url, source_url, package_url, login_url…).
_CONTENT_BRIEF_KEYS: frozenset[str] = frozenset(
    {
        *AUTH_BLOCK_KEYS,
        "url",
        "ref",
        "urls",
        "sources",
        "video_sources",
        "panels",
        "lines",
        "shotlist",
        "intent",
        "manuscript",
        "manuscript_ref",
        "document_ref",
    }
)


class CloneSourceNotFoundError(LookupError):
    """The clone source does not exist or belongs to another tenant (→ 404)."""

    def __init__(self, source_id: UUID) -> None:
        super().__init__(f"clone source project {source_id} not found")
        self.source_id = source_id

    @property
    def detail(self) -> dict[str, str]:
        return {
            "code": CLONE_SOURCE_NOT_FOUND,
            "message": "The project to clone settings from was not found.",
            "clone_from_project_id": str(self.source_id),
        }


@dataclass(frozen=True, slots=True)
class ClonedSettings:
    """What a clone contributes: the BASE layer every door merges under."""

    brief: dict[str, object]
    brand: dict[str, object]
    voice: dict[str, object]
    theme: dict[str, object]


def _brief_defaults(brief: Mapping[str, object]) -> dict[str, object]:
    return {
        k: v for k, v in brief.items() if k not in _CONTENT_BRIEF_KEYS and not k.endswith("_url")
    }


async def load_clone_source(
    project_repo: ProjectRepository,
    source_id: UUID,
    *,
    tenant_id: UUID,
    target_type: str,
) -> ClonedSettings:
    """The cloneable settings of ``source_id`` for a ``target_type`` project.

    Raises :class:`CloneSourceNotFoundError` for an unknown or foreign source.
    """
    source = await project_repo.get(source_id, tenant_id=tenant_id)
    if source is None:
        raise CloneSourceNotFoundError(source_id)
    same_type = source.type == target_type
    return ClonedSettings(
        brief=_brief_defaults(source.brief or {}) if same_type else {},
        brand=dict(source.brand or {}),
        voice=dict(source.voice or {}),
        theme=dict(source.theme or {}),
    )


def merge_under(
    cloned: Mapping[str, object], explicit: Mapping[str, object] | None
) -> dict[str, object]:
    """``explicit`` on top of ``cloned``, key-wise (explicit wins)."""
    return {**cloned, **(explicit or {})}
