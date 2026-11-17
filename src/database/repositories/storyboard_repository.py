"""Storyboard repositories — tenant-scoped CRUD + panel reorder.

Mirrors ``VerbatimItemRepository``: constructor ``(self, session)``, a project
scope-check, auto-increment ``position`` at ``max+1`` when the caller doesn't pin
one, tenant enforced via a join to ``projects`` on item reads, and ``update``'s
``setattr`` + ``flush`` + ``refresh``. Panel ``reorder`` is permutation-exact
(mirrors ``services.runs.elements.reorder_elements_in_output`` — on DB rows).

``storyboard_panels``/``storyboard_scenes`` carry a NOT-NULL ``storyboard_id``,
but Phase 0 exposes no storyboard CRUD route, so ``StoryboardRepository.
ensure_for_project`` get-or-creates the single project-level ``storyboards`` row
that panels/scenes hang off. Model attribute names are the WS-A contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID, uuid7

from sqlalchemy import asc, delete, desc, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer
from sqlalchemy.orm.exc import StaleDataError

from src.database.models import (
    Project,
    Storyboard,
    StoryboardAsset,
    StoryboardBranch,
    StoryboardPanel,
    StoryboardPanelAsset,
    StoryboardPanelEvent,
    StoryboardScene,
    StoryboardSnapshot,
)

# The three storyboard row kinds that carry a ``lock_version`` optimistic-concurrency
# token. Spelled as a union of concrete model classes (not ``type[Any]``) so the shared
# board-CAS helpers keep their ``.id``/``.project_id``/``.lock_version`` typing.
_VersionedModel = type[StoryboardScene] | type[StoryboardPanel] | type[StoryboardAsset]


class UnknownAssetError(Exception):
    """An incoming asset id is not a known asset OF THIS PROJECT (→ 422).

    Covers three cases, all the caller's fault and none of them a 500: the id
    doesn't exist, it belongs to another project (the FK alone would happily
    accept another tenant's asset — this is the cross-tenant guard), or the
    string isn't even UUID-shaped (``characters`` is ``list[str]`` on the wire).
    """

    def __init__(self, ids: list[str]) -> None:
        super().__init__(f"unknown asset id for this project: {', '.join(ids)}")
        self.ids = ids


class UnknownSceneError(Exception):
    """An incoming ``scene_id`` is not a known scene OF THIS PROJECT (→ 422).

    The panel's single ``scene_id`` FK sibling of :class:`UnknownAssetError`. Covers
    the same three caller-fault cases, none of them a 500: the id doesn't exist, it
    belongs to another project (the FK alone would happily bind another project's scene
    — this is the cross-project guard), or the string isn't UUID-shaped. ``scene_id:
    None`` is a legitimate clear and never reaches here (the caller guards on it).
    """

    def __init__(self, ids: list[str]) -> None:
        super().__init__(f"unknown scene id for this project: {', '.join(ids)}")
        self.ids = ids


class AssetInUseError(Exception):
    """An asset still referenced by storyboard panels cannot be deleted (→ 409).

    ``panels`` carries the referencing panels as ``{"id": ..., "shot_key": ...}``
    so the UI can name them ("used by clip-1, clip-3") instead of showing a bare
    failure.
    """

    def __init__(self, panels: list[dict[str, str]]) -> None:
        super().__init__(f"asset is referenced by {len(panels)} storyboard panel(s)")
        self.panels = panels


class ProjectArchivedError(Exception):
    """A storyboard mutation was attempted on an ``archived`` (read-only) project (→ 409).

    ``archived`` is the sole terminal state of ``projects.status`` — a merged/abandoned
    branch child, or a root the user soft-deleted — and it is read-only by governance:
    its panels/scenes/assets are the frozen record of finished (or discarded) work, and a
    forked/merged branch must not keep mutating a project the lifecycle has closed. Reads
    (list/diff/export) stay open; only WRITES raise this. The ASGI handler maps it to a
    single uniform 409 so the by-id mutator routes (which have no project_id in the path
    and reach the project only through the repository JOIN) need no per-route try/except.
    """

    def __init__(self, project_id: UUID | str) -> None:
        super().__init__(f"project {project_id} is archived and read-only")
        self.project_id = str(project_id)


class StoryboardVersionConflictError(Exception):
    """An opt-in storyboard write lost the compare-and-set on ``lock_version`` (→ 409).

    Raised ONLY when the caller supplied an ``expected_lock_version`` and the row's
    live ``lock_version`` no longer matches it — i.e. another write landed between the
    caller's read and this write (classic PATCH-vs-PATCH lost update). A caller that
    omits ``expected_lock_version`` never triggers this: the write stays last-write-wins
    (unchanged behaviour). It is distinct from the merge delete-race (which stays a 404):
    the repo re-checks existence on a 0-row CAS and only raises this when the row is
    still there. The ASGI handler maps it to a uniform 409, naming ``entity``.

    Panels, scenes and assets each raise their own subclass so a caller can still catch
    a specific one, but the app-level handler is registered on THIS base and catches all
    three via the MRO.
    """

    def __init__(self, entity: str, entity_id: UUID | str) -> None:
        super().__init__(f"{entity} {entity_id} was modified concurrently")
        self.entity = entity
        self.entity_id = str(entity_id)


class PanelVersionConflictError(StoryboardVersionConflictError):
    """Panel CAS conflict. Keeps ``self.panel_id`` for existing callers; the message and
    handler output ("panel was modified concurrently") are unchanged from migration 0055."""

    def __init__(self, panel_id: UUID | str) -> None:
        super().__init__("panel", panel_id)
        self.panel_id = str(panel_id)


class SceneVersionConflictError(StoryboardVersionConflictError):
    """Scene CAS conflict (migration 0056)."""

    def __init__(self, scene_id: UUID | str) -> None:
        super().__init__("scene", scene_id)
        self.scene_id = str(scene_id)


class AssetVersionConflictError(StoryboardVersionConflictError):
    """Asset CAS conflict (migration 0056)."""

    def __init__(self, asset_id: UUID | str) -> None:
        super().__init__("asset", asset_id)
        self.asset_id = str(asset_id)


class BoardVersionConflictError(StoryboardVersionConflictError):
    """WHOLE-BOARD CAS conflict (→ 409, entity ``storyboard``).

    The row-level subclasses guard ONE row; the destructive/bulk routes (panel + scene
    reorder, snapshot restore) rewrite the ordering or the entire graph, so the unit the
    caller must be current on is the BOARD, not a row. The caller sends the
    ``{row id: lock_version}`` map it last read and this is raised when the live board no
    longer matches it exactly — a row was added, deleted, or written by a sibling since
    that read. Reuses the ``StoryboardVersionConflictError`` app-level handler, so it is
    the SAME 409 shape as a row conflict ("storyboard was modified concurrently").
    """

    def __init__(self, project_id: UUID | str) -> None:
        super().__init__("storyboard", project_id)
        self.project_id = str(project_id)


class _ScopedRepo:
    """Shared project-scope + tenant-join helpers (verbatim parity)."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def _scoped_project(self, project_id: UUID, tenant_id: UUID) -> Project | None:
        return (
            await self.session.execute(
                select(Project).where(Project.id == project_id, Project.tenant_id == tenant_id)
            )
        ).scalar_one_or_none()

    async def _assert_editable(self, project_id: UUID, tenant_id: UUID) -> Project | None:
        """The tenant-scoped project, raising :class:`ProjectArchivedError` when it is
        ``archived`` (the sole read-only terminal state).

        The write-path counterpart of :meth:`_scoped_project`: same single, PK-indexed
        query (``_scoped_project`` already loads the whole Project row, so ``status`` is
        in hand — ZERO extra query for the create/reorder paths that already called it),
        but it enforces the lifecycle guard. Returns ``None`` for a missing/cross-tenant
        project exactly like ``_scoped_project`` so existing "``is None`` → 404/no-op"
        callers keep working; it only ever RAISES for the archived case. READ paths keep
        calling ``_scoped_project`` directly so an archived project stays fully listable.
        """
        project = await self._scoped_project(project_id, tenant_id)
        if project is not None and project.status == "archived":
            raise ProjectArchivedError(project_id)
        return project

    async def _locked_lock_versions(
        self, model: _VersionedModel, project_id: UUID
    ) -> dict[UUID, int]:
        """``{row id: lock_version}`` for every ``model`` row of the project, taken under
        ``SELECT ... FOR UPDATE``.

        The lock is the point: a board-level CAS that only READ the versions would be a
        TOCTOU window — a sibling could commit between the check and the rewrite and lose
        its work anyway. Holding the row locks until this transaction commits makes the
        loser of two concurrent board writes BLOCK, then re-read the winner's fresh
        versions and 409. Ordered by ``id`` so two concurrent lockers take the rows in the
        same order and cannot deadlock each other.
        """
        rows = (
            await self.session.execute(
                select(model.id, model.lock_version)
                .where(model.project_id == project_id)
                .order_by(asc(model.id))
                .with_for_update()
            )
        ).all()
        return {row[0]: row[1] for row in rows}

    async def _assert_rows_at(
        self, model: _VersionedModel, project_id: UUID, expected: Mapping[UUID, int]
    ) -> None:
        """Raise :class:`BoardVersionConflictError` unless the project's ``model`` rows are
        EXACTLY ``expected`` — same id set, same ``lock_version`` on each. An added or
        deleted row is a conflict too, not just a bumped token: a reorder computed against
        a board that has since gained a row is stale in the same way a bumped row is."""
        if await self._locked_lock_versions(model, project_id) != dict(expected):
            raise BoardVersionConflictError(project_id)

    async def _next_position(self, model: Any, project_id: UUID) -> int:
        max_pos = (
            await self.session.execute(
                select(func.coalesce(func.max(model.position), -1)).where(
                    model.project_id == project_id
                )
            )
        ).scalar_one()
        return int(max_pos) + 1


class StoryboardRepository(_ScopedRepo):
    """The ``storyboards`` grouping row that panels/scenes belong to.

    Phase 0 has no storyboard CRUD surface — one storyboard per project is
    get-or-created on the first panel/scene write so the NOT-NULL
    ``storyboard_id`` FK is always satisfiable.
    """

    async def assert_board_at(self, project_id: UUID, expected: Mapping[UUID, int]) -> None:
        """Whole-graph CAS for the routes that rewrite EVERYTHING (snapshot restore).

        ``expected`` is one FLAT ``{row id: lock_version}`` map over the project's scenes,
        panels and assets — row ids are uuid7 and unique across the three tables, so they
        share one namespace and the caller does not have to bucket them. Every row is
        locked ``FOR UPDATE`` before the comparison (see ``_locked_lock_versions``), so a
        concurrent restore/edit either finishes first and makes us 409, or waits for us.
        Raises :class:`BoardVersionConflictError` (→ 409) on any difference.
        """
        live: dict[UUID, int] = {}
        for model in (StoryboardScene, StoryboardPanel, StoryboardAsset):
            live.update(await self._locked_lock_versions(model, project_id))
        if live != dict(expected):
            raise BoardVersionConflictError(project_id)

    async def ensure_for_project(self, project_id: UUID, *, tenant_id: UUID) -> Storyboard | None:
        # Write chokepoint: get-or-creating the storyboard row is the first thing every
        # panel/scene CREATE does, so guarding here blocks create on an archived project
        # (→ 409) with no extra query — ``_assert_editable`` reuses the scope select.
        if (await self._assert_editable(project_id, tenant_id)) is None:
            return None
        # Serialize CREATE against a concurrent storyboard MERGE. Merge holds FOR UPDATE on
        # the ROOT ``projects`` row (``MergeService._lock_root``) while it mints shot_keys,
        # but a plain INSERT of a NEW panel is a phantom the merge's existing-row FOR UPDATE
        # cannot cover: the two independently mint the same ``clip-N`` and collide at COMMIT
        # on the DEFERRED ``uq_storyboard_panels_project_shot`` (an uncaught IntegrityError
        # 500 on a legitimate merge). Taking the SAME projects-row lock here at the create
        # chokepoint makes create-vs-merge — and create-vs-create — serialize on that one
        # row (merge already holds it re-entrantly when it calls this during apply). Locked
        # BEFORE the storyboard/panel INSERT so the lock precedes the shot_key mint.
        await self.session.execute(
            select(Project.id).where(Project.id == project_id).with_for_update()
        )
        # Idempotent get-or-create: a UNIQUE(project_id) constraint lets a
        # concurrent writer's insert collapse to a no-op instead of racing a
        # SELECT-then-INSERT into two rows. The caller commits.
        await self.session.execute(
            pg_insert(Storyboard)
            .values(id=uuid7(), project_id=project_id)
            .on_conflict_do_nothing(index_elements=["project_id"])
        )
        await self.session.flush()
        return (
            await self.session.execute(
                select(Storyboard).where(Storyboard.project_id == project_id)
            )
        ).scalar_one()


class StoryboardPanelRepository(_ScopedRepo):
    """Panels, plus the ``storyboard_panel_assets`` join rows behind
    ``characters`` / ``props`` / ``location_asset_id``.

    Those three are READ-ONLY properties on the model (they project the join
    collection), so they must never reach ``StoryboardPanel(**fields)`` or the
    generic ``setattr`` loop — a property with no setter raises ``AttributeError``
    there. ``create`` takes them as explicit kwargs; ``update`` routes them through
    :meth:`_apply_panel_assets`. The wire contract is unchanged.
    """

    @staticmethod
    def _coerce_ids(values: Sequence[str | UUID], *, bad: list[str]) -> list[UUID]:
        """Ordered, first-occurrence-deduped ids. Non-UUID-shaped strings are
        collected into ``bad`` (→ ``UnknownAssetError``), never a ``ValueError`` 500 —
        the API has always accepted ``characters: ["hero"]``."""
        out: list[UUID] = []
        seen: set[UUID] = set()
        for value in values:
            if isinstance(value, UUID):
                asset_id = value
            else:
                try:
                    asset_id = UUID(str(value))
                except ValueError:
                    bad.append(str(value))
                    continue
            if asset_id not in seen:
                seen.add(asset_id)
                out.append(asset_id)
        return out

    async def _validate_assets(self, project_id: UUID, ids: Sequence[UUID]) -> list[str]:
        """Ids that are NOT an asset of ``project_id`` (missing OR foreign-project)."""
        if not ids:
            return []
        known = set(
            (
                await self.session.execute(
                    select(StoryboardAsset.id).where(
                        StoryboardAsset.id.in_(list(ids)),
                        StoryboardAsset.project_id == project_id,
                    )
                )
            )
            .scalars()
            .all()
        )
        return [str(i) for i in ids if i not in known]

    async def _validate_scene(self, project_id: UUID, scene_id: str | UUID) -> None:
        """Raise :class:`UnknownSceneError` when ``scene_id`` is not a scene of
        ``project_id`` (missing OR foreign-project). Mirrors :meth:`_validate_assets`:
        a project-scoped existence SELECT. ``None`` is a legitimate clear and is never
        passed here — the caller (create/update) guards on it. A non-UUID-shaped string
        is the caller's fault → 422, never a ``ValueError`` 500."""
        try:
            sid = scene_id if isinstance(scene_id, UUID) else UUID(str(scene_id))
        except ValueError as exc:
            raise UnknownSceneError([str(scene_id)]) from exc
        exists = (
            await self.session.execute(
                select(StoryboardScene.id).where(
                    StoryboardScene.id == sid,
                    StoryboardScene.project_id == project_id,
                )
            )
        ).scalar_one_or_none()
        if exists is None:
            raise UnknownSceneError([str(scene_id)])

    async def _resolve_roles(
        self,
        project_id: UUID,
        *,
        characters: Sequence[str | UUID] | None,
        props: Sequence[str | UUID] | None,
        location_asset_id: str | UUID | None,
        location_present: bool,
    ) -> dict[str, list[UUID]]:
        """``role -> ordered asset ids``, ONLY for the roles the caller supplied.

        A role absent from the result is left untouched by the apply step (partial
        PATCH safety: ``{"props": []}`` must not clobber characters/location); a
        supplied role with an empty list CLEARS it. ``location_asset_id: None`` is a
        legitimate clear — the column is nullable — so it is distinguished by
        ``location_present``, not by the value.
        """
        bad: list[str] = []
        desired: dict[str, list[UUID]] = {}
        if characters is not None:
            desired["character"] = self._coerce_ids(characters, bad=bad)
        if props is not None:
            desired["prop"] = self._coerce_ids(props, bad=bad)
        if location_present:
            desired["location"] = (
                [] if location_asset_id is None else self._coerce_ids([location_asset_id], bad=bad)
            )
        unknown = bad + await self._validate_assets(
            project_id, [i for ids in desired.values() for i in ids]
        )
        if unknown:
            raise UnknownAssetError(unknown)
        return desired

    @staticmethod
    def _apply_desired(panel: StoryboardPanel, desired: dict[str, list[UUID]]) -> None:
        """Diff the desired ids against the live join rows, per role.

        Rows are RECYCLED IN PLACE wherever possible — an id that is already present
        keeps its row (only ``position`` moves), and a leftover row of the same role is
        re-pointed at a new ``asset_id`` rather than dropped and re-inserted. Two
        reasons, one of them load-bearing:

        * The join-row id survives an edit, so the row's identity is stable across a
          re-order or an asset swap instead of churning on every PATCH. Note this is a
          nicety, NOT the branch/merge contract: a join row carries no ``origin_id`` of
          its own — its merge identity is a pure function of
          ``(panel.origin_id, asset.origin_id, role)``, re-derived at merge time.
        * ``uq_storyboard_panel_assets_one_location`` is a partial unique **index**, not
          a deferrable constraint, so it is checked per-statement and
          ``SET CONSTRAINTS ALL DEFERRED`` cannot relax it. On a location swap A → B,
          remove-then-append would leave the unit of work emitting the INSERT of B
          *before* the DELETE of A (SQLAlchemy orders INSERTs ahead of DELETEs within a
          table when no FK forces otherwise) — two ``role='location'`` rows for one
          panel mid-flush, and the index fires: a 500 on a perfectly legal edit.
          Re-pointing the existing row makes it a single UPDATE, with nothing transient.

        Rows still unwanted after recycling are removed (delete-orphan does the DELETE).
        """
        for role, ids in desired.items():
            by_asset = {row.asset_id: row for row in panel.panel_assets if row.role == role}
            # Rows whose asset_id is NOT wanted any more — reusable carcasses.
            spare = [row for asset_id, row in by_asset.items() if asset_id not in set(ids)]
            for pos, asset_id in enumerate(ids):
                row = by_asset.get(asset_id)
                if row is not None:
                    row.position = pos
                elif spare:
                    row = spare.pop()
                    row.asset_id = asset_id
                    row.position = pos
                else:
                    panel.panel_assets.append(
                        StoryboardPanelAsset(id=uuid7(), asset_id=asset_id, role=role, position=pos)
                    )
            for row in spare:
                panel.panel_assets.remove(row)

    async def _apply_panel_assets(self, panel: StoryboardPanel, fields: dict[str, Any]) -> None:
        """POP the three asset keys out of ``fields`` and apply them to ``panel``.

        Popping is the point: whatever is left is safe for the generic ``setattr``
        loop. Callers must pass a dict they own (``update`` copies the caller's).
        """
        location_present = "location_asset_id" in fields
        desired = await self._resolve_roles(
            panel.project_id,
            characters=fields.pop("characters", None),
            props=fields.pop("props", None),
            location_asset_id=fields.pop("location_asset_id", None),
            location_present=location_present,
        )
        self._apply_desired(panel, desired)

    async def create(
        self,
        *,
        project_id: UUID,
        storyboard_id: UUID,
        position: int | None,
        shot_key: str | None,
        characters: Sequence[str | UUID] | None = None,
        props: Sequence[str | UUID] | None = None,
        location_asset_id: str | UUID | None = None,
        **fields: Any,
    ) -> StoryboardPanel:
        if position is None:
            position = await self._next_position(StoryboardPanel, project_id)
        # AuthoredLine rule: blank spine key → clip-<position+1>.
        if not shot_key:
            shot_key = f"clip-{position + 1}"
        # Validate BEFORE the panel exists in the session: an unknown asset id is a
        # 422 with nothing half-written.
        desired = await self._resolve_roles(
            project_id,
            characters=characters,
            props=props,
            location_asset_id=location_asset_id,
            location_present=location_asset_id is not None,
        )
        # Validate scene_id the same way (project-scoped): an unknown/foreign scene is a
        # 422, not the FK's 500 (missing) or a silent cross-project bind (foreign). None
        # is a legitimate absence — only validate a supplied, non-null id.
        scene_id = fields.get("scene_id")
        if scene_id is not None:
            await self._validate_scene(project_id, scene_id)
        panel = StoryboardPanel(
            id=uuid7(),
            project_id=project_id,
            storyboard_id=storyboard_id,
            position=position,
            shot_key=shot_key,
            **fields,
        )
        # Initialise the collection while the instance is still TRANSIENT (no IO).
        # Skipping this when the caller passed no refs would leave it un-initialised;
        # after the flush below the panel is PERSISTENT, and the response serializer's
        # ``row.characters`` read would then emit a lazy load on the async session →
        # ``MissingGreenlet``. ``selectin`` only eager-loads rows fetched by a QUERY,
        # and this one was never queried.
        panel.panel_assets = []
        # Through the ORM collection, never a Core INSERT: create() flushes but does
        # NOT refresh, so a Core-inserted row would be invisible to the freshly
        # loaded (empty) selectin collection and the response would say characters: [].
        self._apply_desired(panel, desired)
        self.session.add(panel)
        await self.session.flush()
        return panel

    async def list_for_project(self, project_id: UUID, *, tenant_id: UUID) -> list[StoryboardPanel]:
        if (await self._scoped_project(project_id, tenant_id)) is None:
            return []
        stmt = (
            select(StoryboardPanel)
            .where(StoryboardPanel.project_id == project_id)
            .order_by(asc(StoryboardPanel.position), asc(StoryboardPanel.created_at))
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def get(self, panel_id: UUID, *, tenant_id: UUID) -> StoryboardPanel | None:
        stmt = (
            select(StoryboardPanel)
            .join(Project, StoryboardPanel.project_id == Project.id)
            .where(StoryboardPanel.id == panel_id, Project.tenant_id == tenant_id)
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def _cas_guard(
        self, panel: StoryboardPanel, expected_lock_version: int, *, tenant_id: UUID
    ) -> bool:
        """Opt-in compare-and-set on ``lock_version`` — called ONLY when the caller
        supplied an ``expected_lock_version``, and BEFORE any field is mutated.

        A guarded Core ``UPDATE ... SET lock_version = lock_version + 1 WHERE id = :id
        AND lock_version = :expected`` that atomically LOCKS the row and advances the
        token, so two concurrent CAS writers serialize and the loser matches 0 rows. On
        0 rows we re-check existence (the same PK the caller already read): gone ⇒ merge
        delete-race ⇒ ``False`` (→ 404); still there ⇒ stale token ⇒
        :class:`PanelVersionConflictError` (→ 409). ``synchronize_session=False`` keeps
        the SQL-expression bump off the ORM identity map (the caller ``refresh``es the
        bumped value back). Running BEFORE the setattrs means a lost CAS leaves the panel
        untouched — nothing was written to roll back.

        The last-write-wins path never calls this: it folds ``panel.lock_version =
        StoryboardPanel.lock_version + 1`` (an atomic in-DB increment, NOT a Python
        read-modify-write) into the SAME flush as the fields, just before the flush.
        """
        result = await self.session.execute(
            update(StoryboardPanel)
            .where(
                StoryboardPanel.id == panel.id,
                StoryboardPanel.lock_version == expected_lock_version,
            )
            .values(lock_version=StoryboardPanel.lock_version + 1),
            execution_options={"synchronize_session": False},
        )
        if (getattr(result, "rowcount", 0) or 0) > 0:
            return True
        if await self.get(panel.id, tenant_id=tenant_id) is None:
            return False
        raise PanelVersionConflictError(panel.id)

    async def update(
        self,
        panel_id: UUID,
        *,
        tenant_id: UUID,
        fields: dict[str, Any],
        expected_lock_version: int | None = None,
    ) -> StoryboardPanel | None:
        panel = await self.get(panel_id, tenant_id=tenant_id)
        if panel is None:
            return None
        # Lifecycle guard: the by-id route has no project_id, so the archived check MUST
        # live here — one cheap PK-indexed select on the owning project (→ 409 archived).
        await self._assert_editable(panel.project_id, tenant_id)
        # Opt-in compare-and-set BEFORE any field mutation: on a lost CAS this locks/raises
        # (409) or returns 404 with nothing half-written. The asset-validation SELECT below
        # would autoflush a pending non-CAS bump early, so the last-write-wins increment is
        # deferred to just before the flush instead (single UPDATE with the fields).
        if expected_lock_version is not None and not await self._cas_guard(
            panel, expected_lock_version, tenant_id=tenant_id
        ):
            return None
        # characters/props/location_asset_id are read-only projections of the join
        # table — pop them out (on a COPY; the caller keeps its dict) and apply them
        # to the collection before the generic setattr loop, which would otherwise
        # hit a property with no setter.
        remaining = dict(fields)
        # scene_id gets the same project-scoped validation as create: a supplied,
        # non-null scene must be a scene of THIS project (→ 422), else an unknown id is a
        # 500 (FK violation) and a foreign id is a silent cross-project bind. An explicit
        # None is a legitimate clear (nullable column) — skip it.
        if remaining.get("scene_id") is not None:
            await self._validate_scene(panel.project_id, remaining["scene_id"])
        await self._apply_panel_assets(panel, remaining)
        for k, v in remaining.items():
            setattr(panel, k, v)
        if expected_lock_version is None:
            panel.lock_version = StoryboardPanel.lock_version + 1
        try:
            await self.session.flush()
        except StaleDataError:
            # A concurrent storyboard MERGE deleted this panel between our (unlocked)
            # tenant-scoped SELECT and this UPDATE. Merge holds FOR UPDATE on the board
            # rows, so our UPDATE blocked, then unblocked to find 0 matching rows and
            # SQLAlchemy raised. The panel is genuinely gone → surface as 404 (None), not
            # an unhandled 500. The caller (route) then raises HTTPException(404) and the
            # request session rolls back.
            return None
        # onupdate=func.now() expires updated_at; refresh repopulates (including the
        # selectin join collection) so the response serializer doesn't trigger a sync
        # lazy-load.
        await self.session.refresh(panel)
        return panel

    async def approve(
        self,
        panel_id: UUID,
        *,
        tenant_id: UUID,
        actor_subject_id: UUID,
        note: str | None = None,
        events: StoryboardPanelEventRepository,
        expected_lock_version: int | None = None,
    ) -> StoryboardPanel | None:
        """Approve a panel + append an ``approved`` trail event.

        Idempotent: an already-approved panel is a no-op (NO event, current
        ``approved_at`` / ``approved_by`` preserved). ``revision`` is NOT bumped —
        it is the content-edit counter owned by AI rewrite; the event only
        SNAPSHOTS it. 404 (None) if the panel is missing / cross-tenant.

        ``expected_lock_version`` opts into the same compare-and-set as ``update``:
        a real state change advances ``lock_version`` and 409s on a stale token. The
        idempotent no-op neither bumps nor CAS-checks — nothing is written, so there is
        no lost update to guard against.
        """
        panel = await self.get(panel_id, tenant_id=tenant_id)
        if panel is None:
            return None
        # Lifecycle guard before the idempotent short-circuit: approving on an archived
        # project is a mutation attempt and is refused (→ 409), not silently no-op'd.
        await self._assert_editable(panel.project_id, tenant_id)
        if panel.status == "approved":
            return panel
        # Advance the token for this real state change: CAS/409 when opted in (before the
        # setattrs, so a lost CAS leaves the panel untouched), else fold the in-DB bump
        # into the coming flush.
        if expected_lock_version is not None and not await self._cas_guard(
            panel, expected_lock_version, tenant_id=tenant_id
        ):
            return None
        panel.status = "approved"
        panel.approved_at = func.now()
        panel.approved_by = str(actor_subject_id)
        if expected_lock_version is None:
            panel.lock_version = StoryboardPanel.lock_version + 1
        try:
            await self.session.flush()
        except StaleDataError:
            # Same delete-race as ``update``: a concurrent merge deleted this panel while
            # our approve UPDATE blocked on the merge's board FOR UPDATE lock. → 404, not
            # 500; no trail event is appended for a panel that no longer exists.
            return None
        await events.append(
            project_id=panel.project_id,
            panel_id=panel.id,
            event_type="approved",
            revision=panel.revision,
            actor_subject_id=actor_subject_id,
            note=note,
        )
        # func.now() expires approved_at on flush; refresh repopulates so the
        # response serializer doesn't trigger a sync lazy-load.
        await self.session.refresh(panel)
        return panel

    async def unapprove(
        self,
        panel_id: UUID,
        *,
        tenant_id: UUID,
        actor_subject_id: UUID,
        note: str | None = None,
        events: StoryboardPanelEventRepository,
        expected_lock_version: int | None = None,
    ) -> StoryboardPanel | None:
        """Revert a panel to ``draft`` + append an ``unapproved`` trail event.

        Idempotent: a not-currently-approved panel is a no-op (NO event). Clears
        ``approved_at`` / ``approved_by``. ``revision`` is NOT bumped. 404 (None)
        if the panel is missing / cross-tenant.

        ``expected_lock_version`` opts into the same compare-and-set as ``update`` —
        the no-op path neither bumps nor CAS-checks (nothing is written).
        """
        panel = await self.get(panel_id, tenant_id=tenant_id)
        if panel is None:
            return None
        # Lifecycle guard before the idempotent short-circuit: same as ``approve``.
        await self._assert_editable(panel.project_id, tenant_id)
        if panel.status != "approved":
            return panel
        # Advance the token for this real state change: CAS/409 when opted in, else fold
        # the in-DB bump into the coming flush.
        if expected_lock_version is not None and not await self._cas_guard(
            panel, expected_lock_version, tenant_id=tenant_id
        ):
            return None
        panel.status = "draft"
        panel.approved_at = None
        panel.approved_by = None
        if expected_lock_version is None:
            panel.lock_version = StoryboardPanel.lock_version + 1
        try:
            await self.session.flush()
        except StaleDataError:
            # Same delete-race as ``update``: a concurrent merge deleted this panel while
            # our unapprove UPDATE blocked on the merge's board FOR UPDATE lock. → 404, not
            # 500; no trail event is appended for a panel that no longer exists.
            return None
        await events.append(
            project_id=panel.project_id,
            panel_id=panel.id,
            event_type="unapproved",
            revision=panel.revision,
            actor_subject_id=actor_subject_id,
            note=note,
        )
        await self.session.refresh(panel)
        return panel

    async def delete(self, panel_id: UUID, *, tenant_id: UUID, expected_lock_version: int) -> bool:
        """Delete a panel, MANDATORILY compare-and-set on ``lock_version``.

        Unlike ``update``, the lock is NOT opt-in here: a delete cannot be merged or
        replayed, so a last-write-wins delete destroys a sibling's edit with no trace.
        ``False`` (→ 404) when the panel is missing/cross-tenant or lost the CAS to a
        concurrent DELETE; :class:`PanelVersionConflictError` (→ 409) when it is still
        there at a different version — i.e. somebody edited what you are deleting.
        """
        panel = await self.get(panel_id, tenant_id=tenant_id)
        if panel is None:
            return False
        await self._assert_editable(panel.project_id, tenant_id)  # → 409 on archived
        # The guarded UPDATE also LOCKS the row, so a concurrent CAS writer serialises
        # behind this delete instead of racing it.
        if not await self._cas_guard(panel, expected_lock_version, tenant_id=tenant_id):
            return False
        await self.session.delete(panel)
        await self.session.flush()
        return True

    async def delete_for_project(self, project_id: UUID, *, tenant_id: UUID) -> int:
        """Delete every panel of a (tenant-scoped) project; used by AI re-draft."""
        if (await self._scoped_project(project_id, tenant_id)) is None:
            return 0
        rows = (
            (
                await self.session.execute(
                    select(StoryboardPanel).where(StoryboardPanel.project_id == project_id)
                )
            )
            .scalars()
            .all()
        )
        for row in rows:
            await self.session.delete(row)
        await self.session.flush()
        return len(rows)

    async def reorder(
        self,
        project_id: UUID,
        ordered_ids: Sequence[UUID],
        *,
        tenant_id: UUID,
        expected_lock_versions: Mapping[UUID, int],
    ) -> list[StoryboardPanel] | None:
        """Permutation-exact reorder on panel row-ids (mirrors elements.reorder).

        Returns the panels in their new position order on success, or ``None`` when
        ``ordered_ids`` is not an exact permutation of the project's panels (→ 400).
        An empty board legitimately returns ``[]`` (a valid no-op reorder), so callers
        must distinguish ``None`` from ``[]``.

        ``expected_lock_versions`` is MANDATORY: the ``{panel id: lock_version}`` map the
        caller last read. It is asserted (under row locks) BEFORE the permutation check,
        so a board that moved under the caller is a :class:`BoardVersionConflictError`
        (409 "storyboard was modified concurrently") and not a misleading 400 — the
        ordering isn't malformed, it is stale. Without it a reorder was pure
        last-write-wins: it BUMPED every moved row's token while never COMPARING one, so a
        stale client silently reverted a sibling's reorder.

        A reorder MUTATES every moved row's ``position``, so it also advances that row's
        ``lock_version``. Without this, a client holding a pre-reorder token could PATCH
        ``position`` with a now-stale ``expected_lock_version`` that the CAS guard would
        still match, silently reverting the reorder with no 409. Unmoved rows keep their
        token, so a reorder never spuriously 409s an edit to a panel that didn't move.
        """
        if (await self._scoped_project(project_id, tenant_id)) is None:
            return None
        await self._assert_rows_at(StoryboardPanel, project_id, expected_lock_versions)
        rows = (
            (
                await self.session.execute(
                    select(StoryboardPanel).where(StoryboardPanel.project_id == project_id)
                )
            )
            .scalars()
            .all()
        )
        present = {r.id for r in rows}
        # Missing, extra, or duplicate id → not a permutation → 400.
        if sorted(ordered_ids) != sorted(present) or len(ordered_ids) != len(set(ordered_ids)):
            return None
        rank = {pid: i for i, pid in enumerate(ordered_ids)}
        for row in rows:
            new_pos = rank[row.id]
            if row.position != new_pos:
                row.position = new_pos
                # Atomic in-DB increment (SQL expression, not a Python read-modify-write),
                # matching ``update()``. The ``list_for_project`` re-read below materialises
                # the bumped value back as an int for serialization.
                row.lock_version = StoryboardPanel.lock_version + 1
        await self.session.flush()
        return await self.list_for_project(project_id, tenant_id=tenant_id)


class StoryboardPanelEventRepository(_ScopedRepo):
    """Append-only per-panel approval trail (Boords pattern).

    Writes are ONLY appends (mint ``id``, add, flush); there is no update/delete
    path — the panel row holds the current state, this is the immutable history.
    Reads are tenant-scoped via the panel/project join, mirroring the sibling
    repos. Ordered by ``created_at`` (then ``id``, a uuid7 tiebreak, so two events
    that share a transaction-clock ``now()`` still return in insert order).
    """

    async def append(
        self,
        *,
        project_id: UUID,
        panel_id: UUID,
        event_type: str,
        revision: int,
        actor_subject_id: UUID | None = None,
        note: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> StoryboardPanelEvent:
        row = StoryboardPanelEvent(
            id=uuid7(),
            project_id=project_id,
            panel_id=panel_id,
            event_type=event_type,
            revision=revision,
            actor_subject_id=actor_subject_id,
            note=note,
            # NOT-NULL JSONB with a '{}' default — only pass a value when non-empty
            # so the approve/unapprove callers keep the server default.
            **({"detail": detail} if detail else {}),
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def append_all(self, rows: list[dict[str, Any]]) -> list[StoryboardPanelEvent]:
        """Batch sibling of :meth:`append`: ONE ``session.add_all`` + ONE ``flush`` for N
        events instead of N single-row flushes.

        Each dict carries the SAME fields :meth:`append` takes as kwargs
        (``project_id``/``panel_id``/``event_type``/``revision`` required;
        ``actor_subject_id``/``note``/``detail`` optional). ``detail`` is only set when
        truthy — identical to :meth:`append`, so the NOT-NULL JSONB keeps its ``'{}'``
        server default otherwise. ``id`` is minted per row via ``uuid7()`` in input order,
        so the created rows' uuid7 tiebreak preserves the caller's order under
        ``list_for_*``'s ``(created_at, id)`` sort. Returns the rows in input order.

        Added for merge's per-touched-panel trail loop; the single-row :meth:`append` stays
        the path for every other caller.
        """
        events = [
            StoryboardPanelEvent(
                id=uuid7(),
                project_id=r["project_id"],
                panel_id=r["panel_id"],
                event_type=r["event_type"],
                revision=r["revision"],
                actor_subject_id=r.get("actor_subject_id"),
                note=r.get("note"),
                **({"detail": r["detail"]} if r.get("detail") else {}),
            )
            for r in rows
        ]
        self.session.add_all(events)
        await self.session.flush()
        return events

    async def list_for_project(
        self, project_id: UUID, *, tenant_id: UUID
    ) -> list[StoryboardPanelEvent]:
        if (await self._scoped_project(project_id, tenant_id)) is None:
            return []
        stmt = (
            select(StoryboardPanelEvent)
            .where(StoryboardPanelEvent.project_id == project_id)
            .order_by(asc(StoryboardPanelEvent.created_at), asc(StoryboardPanelEvent.id))
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def list_for_panel(
        self, panel_id: UUID, *, tenant_id: UUID
    ) -> list[StoryboardPanelEvent] | None:
        """None when the panel is missing / cross-tenant (→ 404); else the trail."""
        if await StoryboardPanelRepository(self.session).get(panel_id, tenant_id=tenant_id) is None:
            return None
        stmt = (
            select(StoryboardPanelEvent)
            .where(StoryboardPanelEvent.panel_id == panel_id)
            .order_by(asc(StoryboardPanelEvent.created_at), asc(StoryboardPanelEvent.id))
        )
        return list((await self.session.execute(stmt)).scalars().all())


class StoryboardSceneRepository(_ScopedRepo):
    async def create(
        self, *, project_id: UUID, storyboard_id: UUID, position: int | None, **fields: Any
    ) -> StoryboardScene:
        if position is None:
            position = await self._next_position(StoryboardScene, project_id)
        scene = StoryboardScene(
            id=uuid7(),
            project_id=project_id,
            storyboard_id=storyboard_id,
            position=position,
            **fields,
        )
        self.session.add(scene)
        await self.session.flush()
        return scene

    async def list_for_project(self, project_id: UUID, *, tenant_id: UUID) -> list[StoryboardScene]:
        if (await self._scoped_project(project_id, tenant_id)) is None:
            return []
        stmt = (
            select(StoryboardScene)
            .where(StoryboardScene.project_id == project_id)
            .order_by(asc(StoryboardScene.position), asc(StoryboardScene.created_at))
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def get(self, scene_id: UUID, *, tenant_id: UUID) -> StoryboardScene | None:
        stmt = (
            select(StoryboardScene)
            .join(Project, StoryboardScene.project_id == Project.id)
            .where(StoryboardScene.id == scene_id, Project.tenant_id == tenant_id)
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def _cas_guard(
        self, scene: StoryboardScene, expected_lock_version: int, *, tenant_id: UUID
    ) -> bool:
        """Opt-in compare-and-set on ``lock_version`` — mirror of
        ``StoryboardPanelRepository._cas_guard``. Called ONLY when the caller supplied an
        ``expected_lock_version``, and BEFORE any field is mutated. A guarded Core
        ``UPDATE ... SET lock_version = lock_version + 1 WHERE id AND lock_version = :expected``
        atomically locks the row and advances the token, so two concurrent CAS writers
        serialize and the loser matches 0 rows. On 0 rows we re-check existence: gone ⇒
        ``False`` (→ 404); still there ⇒ stale token ⇒ :class:`SceneVersionConflictError`
        (→ 409). ``synchronize_session=False`` keeps the SQL-expression bump off the ORM
        identity map (the caller ``refresh``es the bumped value back)."""
        result = await self.session.execute(
            update(StoryboardScene)
            .where(
                StoryboardScene.id == scene.id,
                StoryboardScene.lock_version == expected_lock_version,
            )
            .values(lock_version=StoryboardScene.lock_version + 1),
            execution_options={"synchronize_session": False},
        )
        if (getattr(result, "rowcount", 0) or 0) > 0:
            return True
        if await self.get(scene.id, tenant_id=tenant_id) is None:
            return False
        raise SceneVersionConflictError(scene.id)

    async def update(
        self,
        scene_id: UUID,
        *,
        tenant_id: UUID,
        fields: dict[str, Any],
        expected_lock_version: int | None = None,
    ) -> StoryboardScene | None:
        scene = await self.get(scene_id, tenant_id=tenant_id)
        if scene is None:
            return None
        await self._assert_editable(scene.project_id, tenant_id)  # → 409 on archived
        # Opt-in CAS BEFORE any field mutation: a lost CAS locks/raises (409) or returns
        # 404 with nothing half-written.
        if expected_lock_version is not None and not await self._cas_guard(
            scene, expected_lock_version, tenant_id=tenant_id
        ):
            return None
        for k, v in fields.items():
            setattr(scene, k, v)
        # Last-write-wins path: fold the atomic in-DB increment into the SAME flush as the
        # fields (the CAS path already bumped via ``_cas_guard``).
        if expected_lock_version is None:
            scene.lock_version = StoryboardScene.lock_version + 1
        await self.session.flush()
        await self.session.refresh(scene)
        return scene

    async def delete(self, scene_id: UUID, *, tenant_id: UUID, expected_lock_version: int) -> bool:
        """Delete a scene with a MANDATORY compare-and-set on ``lock_version`` (mirror of
        ``StoryboardPanelRepository.delete`` — same rationale, same outcomes: ``False`` →
        404, :class:`SceneVersionConflictError` → 409)."""
        scene = await self.get(scene_id, tenant_id=tenant_id)
        if scene is None:
            return False
        await self._assert_editable(scene.project_id, tenant_id)  # → 409 on archived
        if not await self._cas_guard(scene, expected_lock_version, tenant_id=tenant_id):
            return False
        await self.session.delete(scene)
        await self.session.flush()
        return True

    async def delete_for_project(self, project_id: UUID, *, tenant_id: UUID) -> int:
        """Delete every scene of a (tenant-scoped) project; used by AI re-draft."""
        if (await self._scoped_project(project_id, tenant_id)) is None:
            return 0
        rows = (
            (
                await self.session.execute(
                    select(StoryboardScene).where(StoryboardScene.project_id == project_id)
                )
            )
            .scalars()
            .all()
        )
        for row in rows:
            await self.session.delete(row)
        await self.session.flush()
        return len(rows)

    async def reorder(
        self,
        project_id: UUID,
        ordered_ids: Sequence[UUID],
        *,
        tenant_id: UUID,
        expected_lock_versions: Mapping[UUID, int],
    ) -> list[StoryboardScene] | None:
        """Permutation-exact reorder on scene row-ids (mirror of
        ``StoryboardPanelRepository.reorder``). Atomic: every scene's ``position`` is
        reassigned in one flush, so a mid-sequence failure can never persist a partial
        order (unlike N per-scene PATCHes). ``position`` has no UNIQUE constraint, so the
        transient duplicate positions during reassignment are harmless — no deferrable
        dance needed.

        Returns the scenes in new position order on success, ``None`` on a non-permutation
        (→ 400); an empty board returns ``[]``. Every moved scene's ``lock_version`` is
        advanced so a stale-token ``position`` PATCH cannot silently revert the reorder —
        identical rationale to the panel reorder.

        ``expected_lock_versions`` (``{scene id: lock_version}``) is MANDATORY and asserted
        under row locks before the permutation check — same contract and same
        :class:`BoardVersionConflictError` (409) as ``StoryboardPanelRepository.reorder``.
        """
        if (await self._scoped_project(project_id, tenant_id)) is None:
            return None
        await self._assert_rows_at(StoryboardScene, project_id, expected_lock_versions)
        rows = (
            (
                await self.session.execute(
                    select(StoryboardScene).where(StoryboardScene.project_id == project_id)
                )
            )
            .scalars()
            .all()
        )
        present = {r.id for r in rows}
        # Missing, extra, or duplicate id → not a permutation → 400.
        if sorted(ordered_ids) != sorted(present) or len(ordered_ids) != len(set(ordered_ids)):
            return None
        rank = {sid: i for i, sid in enumerate(ordered_ids)}
        for row in rows:
            new_pos = rank[row.id]
            if row.position != new_pos:
                row.position = new_pos
                row.lock_version = StoryboardScene.lock_version + 1
        await self.session.flush()
        return await self.list_for_project(project_id, tenant_id=tenant_id)


class StoryboardAssetRepository(_ScopedRepo):
    async def create(
        self, *, project_id: UUID, position: int | None, **fields: Any
    ) -> StoryboardAsset:
        if position is None:
            position = await self._next_position(StoryboardAsset, project_id)
        asset = StoryboardAsset(id=uuid7(), project_id=project_id, position=position, **fields)
        self.session.add(asset)
        await self.session.flush()
        return asset

    async def list_for_project(self, project_id: UUID, *, tenant_id: UUID) -> list[StoryboardAsset]:
        if (await self._scoped_project(project_id, tenant_id)) is None:
            return []
        stmt = (
            select(StoryboardAsset)
            .where(StoryboardAsset.project_id == project_id)
            .order_by(asc(StoryboardAsset.position), asc(StoryboardAsset.created_at))
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def get(self, asset_id: UUID, *, tenant_id: UUID) -> StoryboardAsset | None:
        stmt = (
            select(StoryboardAsset)
            .join(Project, StoryboardAsset.project_id == Project.id)
            .where(StoryboardAsset.id == asset_id, Project.tenant_id == tenant_id)
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def _cas_guard(
        self, asset: StoryboardAsset, expected_lock_version: int, *, tenant_id: UUID
    ) -> bool:
        """Opt-in compare-and-set on ``lock_version`` — mirror of
        ``StoryboardPanelRepository._cas_guard``. Called ONLY when the caller supplied an
        ``expected_lock_version``, and BEFORE any field is mutated. A guarded Core
        ``UPDATE ... SET lock_version = lock_version + 1 WHERE id AND lock_version = :expected``
        atomically locks the row and advances the token, so two concurrent CAS writers
        serialize and the loser matches 0 rows. On 0 rows we re-check existence: gone ⇒
        ``False`` (→ 404); still there ⇒ stale token ⇒ :class:`AssetVersionConflictError`
        (→ 409). ``synchronize_session=False`` keeps the SQL-expression bump off the ORM
        identity map (the caller ``refresh``es the bumped value back)."""
        result = await self.session.execute(
            update(StoryboardAsset)
            .where(
                StoryboardAsset.id == asset.id,
                StoryboardAsset.lock_version == expected_lock_version,
            )
            .values(lock_version=StoryboardAsset.lock_version + 1),
            execution_options={"synchronize_session": False},
        )
        if (getattr(result, "rowcount", 0) or 0) > 0:
            return True
        if await self.get(asset.id, tenant_id=tenant_id) is None:
            return False
        raise AssetVersionConflictError(asset.id)

    async def update(
        self,
        asset_id: UUID,
        *,
        tenant_id: UUID,
        fields: dict[str, Any],
        expected_lock_version: int | None = None,
    ) -> StoryboardAsset | None:
        asset = await self.get(asset_id, tenant_id=tenant_id)
        if asset is None:
            return None
        await self._assert_editable(asset.project_id, tenant_id)  # → 409 on archived
        # Opt-in CAS BEFORE any field mutation: a lost CAS locks/raises (409) or returns
        # 404 with nothing half-written.
        if expected_lock_version is not None and not await self._cas_guard(
            asset, expected_lock_version, tenant_id=tenant_id
        ):
            return None
        for k, v in fields.items():
            setattr(asset, k, v)
        # Last-write-wins path: fold the atomic in-DB increment into the SAME flush as the
        # fields (the CAS path already bumped via ``_cas_guard``).
        if expected_lock_version is None:
            asset.lock_version = StoryboardAsset.lock_version + 1
        await self.session.flush()
        await self.session.refresh(asset)
        return asset

    async def _referencing_panels(self, asset_id: UUID) -> list[dict[str, str]]:
        """Panels that still reference the asset, in board order (id + shot_key).

        ``position`` is in the select list on purpose: Postgres rejects a
        ``SELECT DISTINCT ... ORDER BY`` on an expression that isn't projected.
        """
        rows = (
            await self.session.execute(
                select(StoryboardPanel.id, StoryboardPanel.shot_key, StoryboardPanel.position)
                .join(StoryboardPanelAsset, StoryboardPanelAsset.panel_id == StoryboardPanel.id)
                .where(StoryboardPanelAsset.asset_id == asset_id)
                .distinct()
                .order_by(asc(StoryboardPanel.position), asc(StoryboardPanel.id))
            )
        ).all()
        return [{"id": str(panel_id), "shot_key": shot_key} for panel_id, shot_key, _ in rows]

    async def delete(self, asset_id: UUID, *, tenant_id: UUID, expected_lock_version: int) -> bool:
        """Delete an asset. Raises :class:`AssetInUseError` (→ 409) when a panel
        still references it — the FK would refuse anyway, this just names the panels.

        ``expected_lock_version`` is MANDATORY (mirror of the panel/scene deletes):
        ``False`` → 404, :class:`AssetVersionConflictError` → 409. The CAS runs BEFORE the
        in-use probe so the row is locked while we read its referencing panels — otherwise
        a panel could gain a reference between the probe and the DELETE.
        """
        asset = await self.get(asset_id, tenant_id=tenant_id)
        if asset is None:
            return False
        await self._assert_editable(asset.project_id, tenant_id)  # → 409 on archived
        if not await self._cas_guard(asset, expected_lock_version, tenant_id=tenant_id):
            return False
        panels = await self._referencing_panels(asset_id)
        if panels:
            raise AssetInUseError(panels)
        await self.session.delete(asset)
        await self.session.flush()
        return True


class StoryboardSnapshotRepository(_ScopedRepo):
    """Version snapshots of the storyboard graph (Phase 5).

    Append-only from the caller's view: a snapshot row is created (never mutated)
    and either listed (metadata only — ``payload`` is deferred so a listing scan
    doesn't drag every blob into memory) or fetched whole (with ``payload``) to
    drive a restore. Tenant scoping mirrors the sibling repos: reads join to
    ``projects`` on ``tenant_id`` (RLS enforces the same in prod).
    """

    async def create(
        self,
        *,
        project_id: UUID,
        storyboard_id: UUID,
        kind: str,
        payload: dict[str, Any],
        note: str | None = None,
        actor_subject: str | None = None,
    ) -> StoryboardSnapshot:
        row = StoryboardSnapshot(
            id=uuid7(),
            project_id=project_id,
            storyboard_id=storyboard_id,
            kind=kind,
            note=note,
            actor_subject=actor_subject,
            payload=payload,
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def list_for_project(
        self, project_id: UUID, *, tenant_id: UUID, limit: int = 20, offset: int = 0
    ) -> list[StoryboardSnapshot]:
        """One newest-first PAGE of snapshot metadata; ``payload`` is deferred.

        Bounded on purpose: a project accumulates a snapshot per restore, so an
        unlimited scan is a slow-query / memory vector even with ``payload``
        deferred. Pair with :meth:`count_for_project` for the true total.
        """
        if (await self._scoped_project(project_id, tenant_id)) is None:
            return []
        stmt = (
            select(StoryboardSnapshot)
            .where(StoryboardSnapshot.project_id == project_id)
            .order_by(desc(StoryboardSnapshot.created_at), desc(StoryboardSnapshot.id))
            .options(defer(StoryboardSnapshot.payload))
            .limit(limit)
            .offset(offset)
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def count_for_project(self, project_id: UUID, *, tenant_id: UUID) -> int:
        """Total snapshots for the project (both kinds) — the page-independent total."""
        if (await self._scoped_project(project_id, tenant_id)) is None:
            return 0
        return int(
            (
                await self.session.execute(
                    select(func.count(StoryboardSnapshot.id)).where(
                        StoryboardSnapshot.project_id == project_id
                    )
                )
            ).scalar_one()
        )

    async def get(self, snapshot_id: UUID, *, tenant_id: UUID) -> StoryboardSnapshot | None:
        """A single snapshot WITH its ``payload`` (tenant-scoped), or None."""
        stmt = (
            select(StoryboardSnapshot)
            .join(Project, StoryboardSnapshot.project_id == Project.id)
            .where(StoryboardSnapshot.id == snapshot_id, Project.tenant_id == tenant_id)
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def get_meta(
        self, snapshot_id: UUID, *, project_id: UUID, tenant_id: UUID
    ) -> StoryboardSnapshot | None:
        """A single snapshot WITHOUT its ``payload``, scoped to project + tenant.

        Used by the delete route, which needs only ``kind`` — dragging the whole
        graph blob in to decide a 204/409 would be wasteful.
        """
        stmt = (
            select(StoryboardSnapshot)
            .join(Project, StoryboardSnapshot.project_id == Project.id)
            .where(
                StoryboardSnapshot.id == snapshot_id,
                StoryboardSnapshot.project_id == project_id,
                Project.tenant_id == tenant_id,
            )
            .options(defer(StoryboardSnapshot.payload))
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def delete(self, snapshot_id: UUID, *, project_id: UUID, tenant_id: UUID) -> bool:
        """Delete one snapshot (project + tenant scoped). False when not found."""
        row = await self.get_meta(snapshot_id, project_id=project_id, tenant_id=tenant_id)
        if row is None:
            return False
        await self.session.delete(row)
        await self.session.flush()
        return True

    async def prune_auto_for_project(
        self, project_id: UUID, *, keep: int, protect_id: UUID | None = None
    ) -> int:
        """Drop every ``kind='auto'`` snapshot beyond the newest ``keep`` ones.

        Returns the number of rows removed. ``manual`` snapshots are NEVER touched
        — they are user-authored and kept forever; only the machine-minted
        pre-restore safety copies are capped. ``keep == 0`` means pruning is
        DISABLED (returns 0 immediately) — it does NOT mean "delete them all".

        ``protect_id`` is exempt from the prune: ``restore`` reads its payload from
        that snapshot and appends trail events referencing it, so pruning must never
        delete the row the in-flight restore depends on (a restore OF an old auto
        snapshot would otherwise be able to delete its own source).

        Victims are chosen by a ``created_at DESC, id DESC`` subselect with
        ``OFFSET keep`` — the same "LIMIT-ed id subselect fed to a DELETE" idiom as
        ``RequestAuditRepository.prune_older_than``, and covered by
        ``idx_storyboard_snapshots_project_created``. Unlike that one this is a
        single statement with no commit: the prune runs inside the caller's restore
        transaction and must roll back with it. ``synchronize_session=False`` keeps
        the bulk DML off the ORM identity map (``restore`` follows it with
        ``expunge_all()``, so no stale identity survives).
        """
        if keep <= 0:
            return 0
        conditions = [
            StoryboardSnapshot.project_id == project_id,
            StoryboardSnapshot.kind == "auto",
        ]
        victims = (
            select(StoryboardSnapshot.id)
            .where(*conditions)
            .order_by(desc(StoryboardSnapshot.created_at), desc(StoryboardSnapshot.id))
            .offset(keep)
            .scalar_subquery()
        )
        stmt = delete(StoryboardSnapshot).where(StoryboardSnapshot.id.in_(victims))
        if protect_id is not None:
            stmt = stmt.where(StoryboardSnapshot.id != protect_id)
        result = await self.session.execute(stmt, execution_options={"synchronize_session": False})
        return int(getattr(result, "rowcount", 0) or 0)


class StoryboardBranchRepository(_ScopedRepo):
    """The branch↔root link + merge lifecycle (migration 0052).

    A branch's *content* lives in its hidden child project (``projects.is_branch``),
    reusing the whole panel/scene/asset machinery; THIS repo owns only the link row.

    Tenant scoping climbs ``root_project_id -> projects.tenant_id`` (the same 1-hop
    join RLS enforces in prod). Reads deliberately gate on the ROOT project, not the
    child: the child inherits the root's ``tenant_id`` at fork, so both resolve to the
    same tenant, and the root is the stable, user-visible side of the link.
    """

    async def create(
        self,
        *,
        root_project_id: UUID,
        child_project_id: UUID,
        name: str,
        base_snapshot_id: UUID,
        parent_base_snapshot_id: UUID | None = None,
        actor_subject: str | None = None,
    ) -> StoryboardBranch:
        row = StoryboardBranch(
            id=uuid7(),
            root_project_id=root_project_id,
            child_project_id=child_project_id,
            name=name,
            base_snapshot_id=base_snapshot_id,
            parent_base_snapshot_id=parent_base_snapshot_id,
            status="open",
            actor_subject=actor_subject,
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def shared_parent_base(self, root_project_id: UUID, *, tenant_id: UUID) -> UUID | None:
        """The snapshot all SIBLINGS of ``root_project_id`` share as their common ancestor,
        or ``None`` when this root has no branch carrying one yet (i.e. this is the first
        fork). The first fork of a root establishes it; every later fork reuses THIS value so
        every sibling three-way-merges against one base. Tenant-scoped via the root project.
        """
        stmt = (
            select(StoryboardBranch.parent_base_snapshot_id)
            .join(Project, StoryboardBranch.root_project_id == Project.id)
            .where(
                StoryboardBranch.root_project_id == root_project_id,
                Project.tenant_id == tenant_id,
                StoryboardBranch.parent_base_snapshot_id.is_not(None),
            )
            .order_by(StoryboardBranch.created_at.asc(), StoryboardBranch.id.asc())
            .limit(1)
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def get(self, branch_id: UUID, *, tenant_id: UUID) -> StoryboardBranch | None:
        """A single branch, tenant-scoped via its ROOT project. None when missing or
        cross-tenant (the route turns both into the same 404 — an existence oracle
        across tenants is itself a leak)."""
        stmt = (
            select(StoryboardBranch)
            .join(Project, StoryboardBranch.root_project_id == Project.id)
            .where(StoryboardBranch.id == branch_id, Project.tenant_id == tenant_id)
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def get_for_child(
        self, child_project_id: UUID, *, tenant_id: UUID
    ) -> StoryboardBranch | None:
        """The branch record whose CHILD is this project — i.e. "am I a branch, and of
        what?". Backs the branch banner the child project's workspace renders."""
        stmt = (
            select(StoryboardBranch)
            .join(Project, StoryboardBranch.root_project_id == Project.id)
            .where(
                StoryboardBranch.child_project_id == child_project_id,
                Project.tenant_id == tenant_id,
            )
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def list_for_root(
        self,
        root_project_id: UUID,
        *,
        tenant_id: UUID,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[StoryboardBranch]:
        """One newest-first PAGE of a root project's branches, optionally filtered to a
        single ``status``.

        Bounded like the snapshot listing: a long-lived project accumulates a branch per
        experiment, so an unlimited scan is a slow-query / memory vector. Pair with
        :meth:`count_for_root` for the true total — an ``len(items)`` "total" is a bug,
        not a shortcut, and it silently caps the UI's pager at one page.

        ``status=None`` means NO filter (all three lifecycle states), which is NOT the
        same as "the open ones" — the caller asks for what it wants.
        """
        if (await self._scoped_project(root_project_id, tenant_id)) is None:
            return []
        stmt = (
            select(StoryboardBranch)
            .where(*self._root_conditions(root_project_id, status))
            # ``created_at`` DESC leads so the newest branch is page 1; the uuid7 ``id``
            # tiebreak keeps the order total when two forks share a transaction clock.
            # Covered by ``idx_storyboard_branches_root (root_project_id, created_at)``.
            .order_by(desc(StoryboardBranch.created_at), desc(StoryboardBranch.id))
            .limit(limit)
            .offset(offset)
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def count_for_root(
        self, root_project_id: UUID, *, tenant_id: UUID, status: str | None = None
    ) -> int:
        """Total branches of the root under the SAME filter as :meth:`list_for_root` —
        the page-independent total. The ``status`` argument must be passed through
        identically by the caller, or the pager's total describes a different set than
        the rows it is paging."""
        if (await self._scoped_project(root_project_id, tenant_id)) is None:
            return 0
        return int(
            (
                await self.session.execute(
                    select(func.count(StoryboardBranch.id)).where(
                        *self._root_conditions(root_project_id, status)
                    )
                )
            ).scalar_one()
        )

    @staticmethod
    def _root_conditions(root_project_id: UUID, status: str | None) -> list[Any]:
        """The WHERE shared by list + count, so the two can never drift apart."""
        conditions: list[Any] = [StoryboardBranch.root_project_id == root_project_id]
        if status is not None:
            conditions.append(StoryboardBranch.status == status)
        return conditions

    async def mark_merged(self, branch: StoryboardBranch) -> None:
        """Close a branch as ``merged`` and stamp ``merged_at``.

        ``func.now()`` (transaction clock, server-side) rather than a Python
        ``datetime``: the stamp then agrees with every other timestamp written in the
        merge transaction, and it cannot skew with the app pod's clock. It expires the
        attribute on flush, so ``refresh`` repopulates — otherwise a caller reading
        ``branch.merged_at`` to build its response would emit a sync lazy-load on the
        async session (``MissingGreenlet``), the same trap ``StoryboardPanelRepository.
        approve`` documents.

        The caller commits, and owns the "is this branch still ``open``?" guard — a
        double-merge is a 409 at the route, not a silent re-stamp here.
        """
        branch.status = "merged"
        branch.merged_at = func.now()
        await self.session.flush()
        await self.session.refresh(branch)

    async def mark_abandoned(self, branch: StoryboardBranch) -> None:
        """Close a branch as ``abandoned``. ``merged_at`` stays NULL on purpose — it
        records a merge, and an abandoned branch never had one; a non-NULL ``merged_at``
        on a non-``merged`` row would make "when was this merged?" unanswerable."""
        branch.status = "abandoned"
        await self.session.flush()
