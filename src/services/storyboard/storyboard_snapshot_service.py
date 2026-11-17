"""Storyboard version snapshots + restore (Phase 5).

A snapshot is a self-contained JSONB copy of a project's WHOLE storyboard graph
(storyboard row + scenes + panels + assets + the panel→asset join rows — NOT the
append-only ``storyboard_panel_events`` trail) captured at a point in time, the same
blob idiom as ``Run.config_snapshot``. Restore rolls the live graph back to a snapshot:

* auto-snapshot the CURRENT state first (a ``kind='auto'`` safety copy) so a
  restore is itself reversible, then prune that project's ``auto`` snapshots back
  to the newest N (``PIPELINE_STORYBOARD_SNAPSHOT_MAX_AUTO_PER_PROJECT``; 0
  disables). ``manual`` snapshots are user-authored and never pruned;
* delete rows that did not exist at snapshot time (panels → ALL of the project's
  panel→asset join rows → scenes → assets; the join wipe is what makes the asset
  prune legal at all — see ``_delete_panel_assets_for_project``);
* upsert every captured row back (storyboard → scenes → panels → assets → join
  rows last, once both of their FK parents exist), so a restore recreates deleted
  rows, revives content edits, and un-does reorders;
* append one ``restored`` trail event per restored panel (the panel's
  ``approved_at`` / ``approved_by`` are restored verbatim from the payload, never
  synthesised).

The whole restore runs in the request's single transaction. ``SET CONSTRAINTS ALL
DEFERRED`` defers the (now DEFERRABLE) ``uq_storyboard_panels_project_shot`` check
to COMMIT, so a shot_key swap/reorder that is transiently duplicate mid-upsert
settles before it is validated — no two-pass shuffle needed.

Tenant scoping mirrors the sibling repos: every read/write is project-scoped and
the project is verified tenant-owned before any mutation (RLS enforces the same in
prod). ``StoryboardSnapshotService(session)`` — constructed with the request
session, methods take ``tenant_id`` (the ``StoryboardAssistService`` pattern).
"""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from decimal import Decimal
from typing import cast
from uuid import UUID, uuid7

from sqlalchemy import DateTime, delete, select, text
from sqlalchemy.dialects.postgresql import UUID as PgUUID  # noqa: N811
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.database.models import (
    Base,
    Project,
    Storyboard,
    StoryboardAsset,
    StoryboardPanel,
    StoryboardPanelAsset,
    StoryboardScene,
    StoryboardSnapshot,
)
from src.database.repositories.storyboard_repository import (
    ProjectArchivedError,
    StoryboardAssetRepository,
    StoryboardPanelEventRepository,
    StoryboardPanelRepository,
    StoryboardRepository,
    StoryboardSceneRepository,
    StoryboardSnapshotRepository,
)


class SnapshotNotFoundError(Exception):
    """Target snapshot is missing or belongs to another tenant (route → 404)."""


def _json_safe(value: object) -> object:
    """Coerce a single ORM column value to a JSON-serialisable form.

    UUID → str, datetime → ISO-8601 str, Decimal → float. JSONB columns already
    hold plain lists/dicts and primitives pass through untouched.
    """
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def _row_to_dict(row: Base) -> dict[str, object]:
    """Every mapped column of an ORM row → a JSON-safe dict (nothing dropped)."""
    return {col.name: _json_safe(getattr(row, col.name)) for col in row.__table__.columns}


def _coerce_row(model: type[Base], data: dict[str, object]) -> dict[str, object]:
    """Reverse :func:`_json_safe` for an upsert: str → UUID / str → datetime per
    the model's column types. JSONB / primitive columns pass through."""
    out: dict[str, object] = {}
    for col in model.__table__.columns:
        if col.name not in data:
            continue
        v = data[col.name]
        if v is not None:
            if isinstance(col.type, PgUUID) and isinstance(v, str):
                v = UUID(v)
            elif isinstance(col.type, DateTime) and isinstance(v, str):
                v = datetime.datetime.fromisoformat(v)
        out[col.name] = v
    return out


def _as_uuid(value: object) -> UUID | None:
    """A payload value → ``UUID``, or ``None`` when it is not UUID-shaped.

    Pre-0051 ``characters`` / ``props`` were free-form ``list[str]`` with no FK behind
    them, so the API accepted literal names (``["hero"]``). Those strings can never
    satisfy the new FK — shape-check before trusting any legacy id."""
    if isinstance(value, UUID):
        return value
    if not isinstance(value, str):
        return None
    try:
        return UUID(value)
    except ValueError:
        return None


async def upsert_rows(
    session: AsyncSession, model: type[Base], rows: list[dict[str, object]]
) -> None:
    """INSERT … ON CONFLICT (id) DO UPDATE for a batch of already-JSON-safe rows.

    THE write primitive for anything that re-applies a captured graph onto a LIVE one —
    restore and merge both go through it, and neither may delete-then-reinsert a panel:
    ``storyboard_panel_events.panel_id`` is ON DELETE CASCADE, so a reinsert (even with
    the same content) silently destroys the panel's approval/event trail. Upserting onto
    the EXISTING id keeps the row — and therefore the trail — alive. (That is the bug
    caught in PR #468; it is the reason this function exists as a shared primitive rather
    than being re-typed per caller.)

    Every dict must carry the SAME key set: ``values([...])`` compiles ONE statement, and
    a column omitted from the VALUES list would take its DEFAULT on insert while its
    ``excluded.<col>`` on the update path is not the caller's data. Pass whole rows.
    """
    if not rows:
        return
    values = [_coerce_row(model, r) for r in rows]
    stmt = pg_insert(model).values(values)
    update = {
        col.name: getattr(stmt.excluded, col.name)
        for col in model.__table__.columns
        if col.name != "id"
    }
    await session.execute(stmt.on_conflict_do_update(index_elements=["id"], set_=update))


class StoryboardSnapshotService:
    def __init__(self, session: AsyncSession, *, max_auto_per_project: int | None = None) -> None:
        self.session = session
        # None → read the operational cap from settings
        # (``PIPELINE_STORYBOARD_SNAPSHOT_MAX_AUTO_PER_PROJECT``). Injectable so a
        # test can pin a small cap without touching process env.
        self._max_auto_per_project = max_auto_per_project

    def _auto_retention(self) -> int:
        if self._max_auto_per_project is not None:
            return self._max_auto_per_project
        return get_settings().pipeline.storyboard_snapshot_max_auto_per_project

    async def prune_autos(self, project_id: UUID, *, protect_id: UUID | None = None) -> int:
        """Cap ``project_id``'s machine-minted ``kind='auto'`` snapshots. Returns rows removed.

        EVERY path that mints an auto must call this. There are two — ``restore`` (a
        pre-restore safety copy) and ``merge`` (a pre-merge safety copy of the ROOT) — and
        for its first release only ``restore`` did, so the cap silently did not exist on the
        merge path: a root merged N times accumulated N unbounded graph blobs. The cap is a
        property of MINTING an auto, not of restoring, so it lives here rather than being
        re-derived at each call site.

        ``protect_id`` is exempt from the prune. It is NOT always the row just minted:
        ``restore`` protects the snapshot it is restoring FROM (restoring an old auto must
        not let the prune delete its own source), while ``merge`` protects the auto it just
        minted (that id is the merge's undo handle). Passing the wrong one is a silent
        data-loss bug, so each caller states its own.

        ``keep == 0`` disables pruning outright — it does NOT mean "delete them all"
        (see :meth:`StoryboardSnapshotRepository.prune_auto_for_project`). Runs in the
        caller's transaction, so a later failure rolls the prune back with it.
        """
        return await StoryboardSnapshotRepository(self.session).prune_auto_for_project(
            project_id, keep=self._auto_retention(), protect_id=protect_id
        )

    async def serialize(self, project_id: UUID, *, tenant_id: UUID) -> dict[str, object]:
        """Build the self-contained payload for ``project_id``'s current graph.

        ``ensure_for_project`` returns the project's single storyboard row (it is
        get-or-created on the first panel/scene write, so it exists whenever there
        is anything to snapshot). Returns ``{}`` when the project is not
        tenant-owned (the route gates existence first, so this is defensive)."""
        storyboard = await StoryboardRepository(self.session).ensure_for_project(
            project_id, tenant_id=tenant_id
        )
        if storyboard is None:
            return {}
        scenes = await StoryboardSceneRepository(self.session).list_for_project(
            project_id, tenant_id=tenant_id
        )
        panels = await StoryboardPanelRepository(self.session).list_for_project(
            project_id, tenant_id=tenant_id
        )
        assets = await StoryboardAssetRepository(self.session).list_for_project(
            project_id, tenant_id=tenant_id
        )
        return {
            "storyboard": _row_to_dict(storyboard),
            "scenes": [_row_to_dict(s) for s in scenes],
            "panels": [_row_to_dict(p) for p in panels],
            "assets": [_row_to_dict(a) for a in assets],
            # Panel→asset references live in their OWN table since 0051, and
            # ``_row_to_dict`` only walks ``__table__.columns`` — so the panel dicts
            # above no longer carry ``characters`` / ``props`` / ``location_asset_id``
            # (they are now Python properties, not columns). Without this flat section
            # a snapshot would silently lose every cast/prop/location assignment.
            #
            # ``panel.panel_assets`` is ``lazy="selectin"``, so the collection is
            # already eager-loaded by ``list_for_project`` — no query per panel here.
            #
            # ALWAYS emit the key, even as ``[]``. Its presence is the flag ``restore``
            # uses to tell a post-0051 snapshot (authoritative, possibly empty) from a
            # pre-0051 one (join rows must be reconstructed from the legacy per-panel
            # keys) — see ``_legacy_panel_asset_rows``.
            "panel_assets": [_row_to_dict(pa) for p in panels for pa in p.panel_assets],
        }

    async def create_snapshot(
        self,
        project_id: UUID,
        *,
        tenant_id: UUID,
        kind: str,
        note: str | None = None,
        actor_subject: UUID | None = None,
    ) -> StoryboardSnapshot:
        payload = await self.serialize(project_id, tenant_id=tenant_id)
        storyboard = cast("dict[str, object]", payload.get("storyboard") or {})
        return await StoryboardSnapshotRepository(self.session).create(
            project_id=project_id,
            storyboard_id=UUID(str(storyboard["id"])),
            kind=kind,
            payload=payload,
            note=note,
            actor_subject=str(actor_subject) if actor_subject is not None else None,
        )

    async def _delete_not_in(
        self, model: type[Base], project_id: UUID, keep_ids: list[UUID]
    ) -> None:
        columns = model.__table__.c
        stmt = delete(model).where(columns.project_id == project_id)
        if keep_ids:
            stmt = stmt.where(columns.id.notin_(keep_ids))
        await self.session.execute(stmt)

    async def _delete_panel_assets_for_project(self, project_id: UUID) -> None:
        """Wipe EVERY panel→asset join row of this project's panels.

        ``storyboard_panel_assets`` carries no ``project_id``, so the generic
        ``_delete_not_in`` cannot serve it — hence the sub-select climb through
        ``storyboard_panels``.

        This is a full wipe, not a conditional prune, and it is load-bearing for the
        asset delete that follows it in ``restore``: postgres cannot defer an FK
        violation caused by deleting a still-referenced asset unless the referencing
        child rows are already gone. ``_delete_not_in(StoryboardPanel, ...)`` only
        CASCADE-clears the join rows of DROPPED panels; the rows hanging off SURVIVING
        panels are precisely the ones that would break the asset prune, and nothing
        removes those implicitly. Wiping all of them and re-inserting from the payload
        is both simpler and strictly more correct than diffing."""
        await self.session.execute(
            delete(StoryboardPanelAsset).where(
                StoryboardPanelAsset.panel_id.in_(
                    select(StoryboardPanel.id).where(StoryboardPanel.project_id == project_id)
                )
            )
        )

    @staticmethod
    def _legacy_panel_asset_rows(
        panels: list[dict[str, object]], assets: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        """Reconstruct join rows from a PRE-0051 payload's per-panel legacy keys.

        A pre-0051 snapshot has no top-level ``panel_assets`` key, but its panel dicts
        DO still carry ``characters`` / ``props`` / ``location_asset_id`` — ``_row_to_dict``
        captured every column that existed at the time. The data was never lost; only
        the container changed. ``_coerce_row`` would now silently discard those keys
        (they are no longer columns), so unless we rebuild here, restoring an old
        snapshot would leave the panels with today's cast still attached — a fabricated
        hybrid the user never saw, not a rollback.

        Candidate ids are filtered against the payload's OWN asset set: that set IS the
        post-restore asset set, so every surviving row is FK-safe by construction with
        zero extra queries, and refs that were ALREADY dangling at capture time (legal
        back then — there was no FK) are dropped rather than resurrected into a 23503.
        Non-UUID-shaped strings are skipped too: the pre-0051 API accepted free text
        (``{"characters": ["hero"]}``).

        ``position`` is the index in the captured list, preserving author order. Ids are
        minted fresh with ``uuid7`` (repo convention — never ``uuid4``); the old rows had
        no identity of their own to preserve."""
        known: set[UUID] = set()
        for a in assets:
            aid = _as_uuid(a.get("id"))
            if aid is not None:
                known.add(aid)

        rows: list[dict[str, object]] = []
        for p in panels:
            panel_id = _as_uuid(p.get("id"))
            if panel_id is None:
                continue
            # Role order mirrors the migration's backfill: character, prop, location.
            legacy: list[tuple[str, object]] = [
                ("character", p.get("characters") or []),
                ("prop", p.get("props") or []),
                ("location", [p["location_asset_id"]] if p.get("location_asset_id") else []),
            ]
            for role, raw_ids in legacy:
                if not isinstance(raw_ids, list):
                    continue
                # Dedupe per (panel, asset, role): a same-asset-twice legacy array would
                # otherwise violate ``uq_storyboard_panel_assets_triple``.
                seen: set[UUID] = set()
                position = 0
                for raw in raw_ids:
                    asset_id = _as_uuid(raw)
                    if asset_id is None or asset_id not in known or asset_id in seen:
                        continue
                    seen.add(asset_id)
                    rows.append(
                        {
                            "id": str(uuid7()),
                            "panel_id": str(panel_id),
                            "asset_id": str(asset_id),
                            "role": role,
                            "position": position,
                        }
                    )
                    position += 1
        return rows

    async def _upsert(self, model: type[Base], rows: list[dict[str, object]]) -> None:
        await upsert_rows(self.session, model, rows)

    async def restore(
        self,
        snapshot_id: UUID,
        *,
        tenant_id: UUID,
        expected_lock_versions: Mapping[UUID, int],
        actor_subject: UUID | None = None,
    ) -> dict[str, UUID]:
        """Roll the project's storyboard graph back to ``snapshot_id``.

        Raises :class:`SnapshotNotFoundError` when the snapshot is missing or
        cross-tenant. Runs entirely in the caller's transaction (the route session
        commits), so a failure rolls back the auto-snapshot too — atomic.

        ``expected_lock_versions`` is MANDATORY — the flat ``{row id: lock_version}`` map
        of the live board as the caller last read it. Restore is the sharpest write in the
        storyboard: it rewrites scenes, panels and assets wholesale, so without a
        compare-and-set a restore fired against a stale view silently discards every edit a
        sibling landed since that read (and a second concurrent restore discards the
        first). Asserted under row locks, BEFORE the safety auto-snapshot is minted, so a
        losing restore writes nothing at all: :class:`BoardVersionConflictError` → 409."""
        repo = StoryboardSnapshotRepository(self.session)
        target = await repo.get(snapshot_id, tenant_id=tenant_id)
        if target is None:
            raise SnapshotNotFoundError(str(snapshot_id))

        project_id = target.project_id
        payload = target.payload or {}

        # Restore rewrites the whole live graph — a mutation — so it is refused on an
        # archived (read-only) project. Checked here (the route has only snapshot_id in its
        # path; the project is resolved from the snapshot). → 409.
        project_status = (
            await self.session.execute(
                select(Project.status).where(
                    Project.id == project_id, Project.tenant_id == tenant_id
                )
            )
        ).scalar_one_or_none()
        if project_status == "archived":
            raise ProjectArchivedError(str(project_id))

        # Whole-board CAS before ANY write (including the auto-snapshot): a losing restore
        # must leave zero trace, not a stray "auto before restore" blob.
        await StoryboardRepository(self.session).assert_board_at(project_id, expected_lock_versions)

        # 1) Safety snapshot of the CURRENT state before we mutate anything.
        auto = await self.create_snapshot(
            project_id,
            tenant_id=tenant_id,
            kind="auto",
            note=f"auto before restore of {snapshot_id}",
            actor_subject=actor_subject,
        )
        auto_id = auto.id

        # 1b) Cap the machine-minted autos: every restore mints one, so without a
        # cap a repeatedly-rolled-back project accumulates unbounded graph blobs.
        # Manual snapshots are never pruned, and the snapshot being restored FROM is
        # protected (a restore of an old auto must not delete its own source). Runs
        # in this same transaction, so a later failure rolls the prune back too.
        await self.prune_autos(project_id, protect_id=snapshot_id)

        # 2) Defer the DEFERRABLE unique check to COMMIT so shot_key swaps settle.
        await self.session.execute(text("SET CONSTRAINTS ALL DEFERRED"))

        scenes = payload.get("scenes", [])
        panels = payload.get("panels", [])
        assets = payload.get("assets", [])
        # ABSENT ≠ EMPTY. ``serialize`` has emitted ``panel_assets`` unconditionally
        # since 0051 (``[]`` when a project genuinely has no assignments), so the key
        # missing entirely can only mean a PRE-0051 payload — whose panel dicts still
        # carry the legacy ``characters`` / ``props`` / ``location_asset_id`` columns.
        # ``[]`` → prune and insert nothing; absent → rebuild from those legacy keys.
        captured_panel_assets = payload.get("panel_assets")
        panel_assets = (
            self._legacy_panel_asset_rows(panels, assets)
            if captured_panel_assets is None
            else captured_panel_assets
        )

        # 3) Drop rows that did not exist at snapshot time (panels → join rows →
        # scenes → assets). A deleted panel cascade-drops only ITS OWN events
        # (acceptable) and its own join rows — which is exactly why the join wipe
        # below is a separate step: the rows on SURVIVING panels are the ones that
        # would make the asset prune an undeferrable FK violation (see
        # ``_delete_panel_assets_for_project``). It must therefore run BEFORE the
        # asset delete, and the join rows are re-inserted from the payload in step 4.
        await self._delete_not_in(StoryboardPanel, project_id, [UUID(p["id"]) for p in panels])
        await self._delete_panel_assets_for_project(project_id)
        await self._delete_not_in(StoryboardScene, project_id, [UUID(s["id"]) for s in scenes])
        await self._delete_not_in(StoryboardAsset, project_id, [UUID(a["id"]) for a in assets])

        # 4) Upsert the captured rows back. Storyboard + scenes before panels so
        # the ``storyboard_id`` / ``scene_id`` FKs resolve; assets next; the join rows
        # LAST — both of their FK parents (panel AND asset) must exist first.
        storyboard = payload.get("storyboard")
        if storyboard:
            await self._upsert(Storyboard, [storyboard])
        await self._upsert(StoryboardScene, scenes)
        await self._upsert(StoryboardPanel, panels)
        await self._upsert(StoryboardAsset, assets)
        await self._upsert(StoryboardPanelAsset, panel_assets)
        await self.session.flush()

        # The upserts above are bulk DML that bypasses the ORM, so any
        # identity-mapped Storyboard/Scene/Panel/Asset instances still in this
        # session hold their PRE-restore attribute values. Read what we still need
        # off ORM instances FIRST, then drop the identity map so every later read
        # re-hydrates from the restored rows.
        #
        # expunge_all (detach), NOT expire_all (invalidate): an expired attribute
        # lazy-loads on next access, which raises MissingGreenlet under async and
        # would blow up any caller still holding a pre-restore object. Detached
        # instances keep their already-loaded values readable instead. Safe because
        # the flush above already emitted every pending INSERT (incl. the auto
        # snapshot), so detaching drops no unwritten work.
        from_created_at = target.created_at.isoformat() if target.created_at is not None else None
        self.session.expunge_all()

        # 5) One ``restored`` trail event per restored panel (approved_at/by were
        # restored verbatim by the upsert — never rewritten here).
        events = StoryboardPanelEventRepository(self.session)
        for p in panels:
            await events.append(
                project_id=project_id,
                panel_id=UUID(p["id"]),
                event_type="restored",
                revision=int(p.get("revision", 1)),
                actor_subject_id=actor_subject,
                detail={"snapshot_id": str(snapshot_id), "from_created_at": from_created_at},
            )

        return {"restored_snapshot_id": snapshot_id, "auto_snapshot_id": auto_id}
