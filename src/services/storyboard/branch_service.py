"""Storyboard BRANCH (fork) — mint a hidden child project seeded from the root.

A branch is a real ``projects`` row with ``is_branch=true`` whose panels/scenes/
assets live in their OWN ``project_id`` namespace. That is the whole trick: the
existing panel/scene/asset/snapshot/restore machinery (and the DEFERRABLE
``uq_storyboard_panels_project_shot``, which is per-project) is reused UNCHANGED,
and two people can hold conflicting ``clip-3``s without the database caring.

Forking is the deep-copy analog of :mod:`storyboard_snapshot_service`, run in the
opposite direction: instead of writing the live graph into a JSONB blob, it reads a
blob back out into a *different, empty* project. Three invariants make merge possible
later, and every one of them is set HERE:

* **The base snapshot is pinned.** The fork mints a ``kind='branch_base'`` snapshot of
  the root and seeds the child from THAT payload — not from a second live read. The
  seed is therefore byte-identical to the merge base by construction, so a
  three-way diff can never be skewed by a write that landed between the two reads.
  ``branch_base`` is a distinct kind on purpose: ``auto`` would be swept by
  ``prune_auto_for_project`` (whose ``protect_id`` shields exactly one id, not the
  arbitrarily many open branches) and ``manual`` is user-deletable. A branch whose
  base vanished is a branch that can never be merged.
* **Every copied row re-mints its ``id``** (a uuid PK is global — two projects cannot
  share one) and carries ``origin_id`` = the SOURCE row's ``origin_id`` if set, else
  its ``id``. That "if set, else" is what keeps lineage stable across a RE-fork: a
  branch-of-a-branch chain would otherwise re-root lineage at each hop. This is what
  MAKES the chain work — ``C``'s panels keep pointing at ``R``'s origins across both
  hops, so the same lineage key matches on the way back up. Branch-of-branch is
  supported; the rule that keeps it sound is :class:`OpenChildBranchesError` (merge
  bottom-up), not a ban on forking.
* **The child inherits ``tenant_id``.** RLS keys off it; a branch that does not
  inherit it is a cross-tenant leak, not a cosmetic bug.

``parent_project_id`` is deliberately NOT reused for the link — that column is the
course→lesson hierarchy (``ProjectRepository`` enforces "lesson must have a parent"
and lists children off it), so a branch hung there would surface as a
lesson-in-a-course. The link lives in ``storyboard_branches``.

Everything runs in the CALLER's transaction (the route commits) → the snapshot, the
child project, the whole seeded graph and the branch row land atomically or not at
all. ``StoryboardBranchService(session)`` — constructed with the request session,
methods take ``tenant_id`` (the ``StoryboardSnapshotService`` pattern).
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid7

from sqlalchemy import func, insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import (
    Project,
    StoryboardAsset,
    StoryboardBranch,
    StoryboardPanel,
    StoryboardPanelAsset,
    StoryboardScene,
)
from src.database.repositories.storyboard_repository import (
    ProjectArchivedError,
    StoryboardBranchRepository,
    StoryboardRepository,
)
from src.services.storyboard.storyboard_snapshot_service import (
    StoryboardSnapshotService,
    _coerce_row,
)


class ProjectNotFoundError(Exception):
    """Root project is missing or belongs to another tenant (route → 404).

    Missing and cross-tenant collapse to the SAME error on purpose: a distinguishable
    403 would be an existence oracle across tenants, which is itself a leak.
    """


class OpenChildBranchesError(Exception):
    """The branch still has OPEN branches of its own (route → 409 on merge/abandon).

    This is what makes branch-of-a-branch safe, and it replaced the blanket refusal that
    used to live here. A fork of a branch is a real chain — root ``R`` ← branch ``B`` ←
    branch ``C`` — and every hop is 1-hop by construction: ``C``'s link row points at
    ``B``'s project, so diff/merge/lock/RLS resolve ``C``'s "root" as ``B`` with no
    change. Lineage survives the re-fork too (``_lineage`` keeps the source's
    ``origin_id`` rather than re-rooting), so ``C``'s keys still match ``R``'s panels
    across both hops.

    What is NOT safe is merging a chain out of order. ``merge`` marks ``B`` ``merged``,
    which is TERMINAL (:class:`~...merge_service.BranchNotOpenError` refuses a re-merge).
    So if ``B`` merged into ``R`` while ``C`` was still open, ``C`` could still merge into
    ``B`` — into a branch that can never carry it upward again. ``C``'s work would be
    silently STRANDED: the API reports a successful merge, and the content never reaches
    the root. That is the "rebase a chain of bases" problem the old blanket 409 cited.

    We refuse the out-of-order merge instead of building the rebase: merge/abandon the
    children FIRST, bottom-up (``C`` → ``B``, then ``B`` → ``R``), and every hop is the
    single-root merge that is already proven. Abandon is guarded for the same reason —
    abandoning ``B`` strands ``C`` just as thoroughly as merging it does.
    """

    def __init__(self, branch_id: str, open_children: int) -> None:
        super().__init__(
            f"branch {branch_id} has {open_children} open child branch(es); "
            "merge or abandon them first"
        )
        self.branch_id = branch_id
        self.open_children = open_children


class BranchNotFoundError(Exception):
    """Target branch is missing or belongs to another tenant (route → 404)."""


class BranchAlreadyMergedError(Exception):
    """Refusing to abandon a branch that was already merged (route → 409).

    ``merged`` is terminal: its content is already in the root, and ``merge`` itself
    already archived the child project (mirroring ``abandon``). Abandoning it would be
    at best a no-op re-archive of a project the merge's provenance (the ``merged`` panel
    trail events, ``base_snapshot_id``) still points at — so it is refused loudly rather
    than accepted as a silent no-op.
    """


async def assert_no_open_children(
    session: AsyncSession, branch: StoryboardBranch, *, tenant_id: UUID
) -> None:
    """Guard the way OUT of a branch (merge or abandon). See :class:`OpenChildBranchesError`.

    A branch's CHILDREN are the branches rooted at its own child project — ``C``'s link row
    carries ``root_project_id = B.child_project_id`` — so this is the same 1-hop count the
    listing uses, not a recursive walk. It only has to be 1-hop: a grandchild cannot be open
    unless ``C`` is open too, and ``C`` being open is already enough to refuse ``B``.

    Lives beside the exception (not on the repo) because both exits need it and neither
    owns it: ``merge`` is in ``merge_service``, ``abandon`` is below.
    """
    open_children = await StoryboardBranchRepository(session).count_for_root(
        branch.child_project_id, tenant_id=tenant_id, status="open"
    )
    if open_children:
        raise OpenChildBranchesError(str(branch.id), open_children)


class StoryboardBranchService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def _scoped_project(self, project_id: UUID, tenant_id: UUID) -> Project | None:
        return (
            await self.session.execute(
                select(Project).where(Project.id == project_id, Project.tenant_id == tenant_id)
            )
        ).scalar_one_or_none()

    async def _insert_rows(self, model: type[Any], rows: list[dict[str, Any]]) -> None:
        """Bulk Core INSERT of already-remapped rows. No ON CONFLICT clause — the
        child project is EMPTY, so nothing can collide."""
        if not rows:
            return
        await self.session.execute(insert(model).values([_coerce_row(model, r) for r in rows]))

    async def create_branch(
        self,
        project_id: UUID,
        *,
        tenant_id: UUID,
        name: str,
        actor_subject: UUID | None = None,
    ) -> tuple[StoryboardBranch, Project]:
        """Fork ``project_id``'s storyboard into a new hidden child project.

        Returns ``(branch, child_project)``. Raises :class:`ProjectNotFoundError` when
        the root is missing/cross-tenant.

        ``project_id`` MAY itself be a branch — a fork of a branch is a chain, and every
        hop stays 1-hop: the link row inserted below points at ``project_id`` whatever it
        is, so diff/merge/lock/RLS resolve this child's "root" as its IMMEDIATE parent
        with no special case. The ordering rule that keeps the chain sound lives on the
        way back UP, not here: see :class:`OpenChildBranchesError`.
        """
        root = await self._scoped_project(project_id, tenant_id)
        if root is None:
            raise ProjectNotFoundError(str(project_id))
        # An archived root (or archived parent branch) is read-only: forking FROM it would
        # spawn a live, mergeable child of a project the lifecycle has already closed. → 409.
        if root.status == "archived":
            raise ProjectArchivedError(str(project_id))

        # 1) Pin the merge base FIRST, and seed from ITS payload (below) rather than
        # from a second live read — see the module docstring's base-pinning invariant.
        base = await StoryboardSnapshotService(self.session).create_snapshot(
            project_id,
            tenant_id=tenant_id,
            kind="branch_base",
            note=f"branch base: {name}",
            actor_subject=actor_subject,
        )
        payload: dict[str, Any] = base.payload or {}

        # 2) The hidden child project. ``tenant_id`` is inherited (RLS keys off it);
        # ``parent_project_id`` stays NULL (that is the course→lesson hierarchy, not
        # this link — the link row in step 5 is the only place the fork is recorded).
        # ``brief`` / ``brand`` / ``voice`` / ``theme`` are copied, not defaulted: they are
        # the creative context every downstream stage reads, so a branch that dropped them
        # would render its panels against a DIFFERENT brand and voice than the root — the
        # diff would then be measuring the fork's own data loss rather than the author's
        # edits. A branch is an alternative CUT of the same project, not a new project.
        child = Project(
            id=uuid7(),
            tenant_id=root.tenant_id,
            owner_subject_id=root.owner_subject_id,
            visibility=root.visibility,
            type=root.type,
            title=name,
            brief=root.brief,
            brand=root.brand,
            voice=root.voice,
            theme=root.theme,
            is_branch=True,
            parent_project_id=None,
            status="drafting",
        )
        self.session.add(child)
        await self.session.flush()

        # 3) The child's own storyboards row — the NOT-NULL ``storyboard_id`` every
        # copied scene/panel re-points at.
        child_storyboard = await StoryboardRepository(self.session).ensure_for_project(
            child.id, tenant_id=tenant_id
        )
        if child_storyboard is None:  # pragma: no cover — we just created the project
            raise ProjectNotFoundError(str(child.id))

        scenes: list[dict[str, Any]] = payload.get("scenes", [])
        panels: list[dict[str, Any]] = payload.get("panels", [])
        assets: list[dict[str, Any]] = payload.get("assets", [])
        panel_assets: list[dict[str, Any]] = payload.get("panel_assets", [])

        # 4) Re-mint every id and remap every FK. One flat old→new map covers scenes,
        # panels and assets: uuids are globally unique, so the three key spaces cannot
        # collide and a single dict keeps the panel/join remaps a plain lookup.
        remap: dict[UUID, UUID] = {}
        for src in (*scenes, *panels, *assets):
            remap[UUID(src["id"])] = uuid7()

        def _lineage(src: dict[str, Any]) -> UUID:
            """The source's ``origin_id`` if it has one, else its own id — so lineage
            survives a re-fork instead of re-rooting at every hop."""
            origin = src.get("origin_id")
            return UUID(origin) if origin else UUID(src["id"])

        new_scenes = [
            {
                **s,
                "id": remap[UUID(s["id"])],
                "project_id": child.id,
                "storyboard_id": child_storyboard.id,
                "origin_id": _lineage(s),
            }
            for s in scenes
        ]
        new_assets = [
            {
                **a,
                "id": remap[UUID(a["id"])],
                "project_id": child.id,
                "origin_id": _lineage(a),
            }
            for a in assets
        ]
        new_panels = [
            {
                **p,
                "id": remap[UUID(p["id"])],
                "project_id": child.id,
                "storyboard_id": child_storyboard.id,
                # None-safe: an unscened panel stays unscened.
                "scene_id": remap[UUID(p["scene_id"])] if p.get("scene_id") else None,
                "origin_id": _lineage(p),
            }
            for p in panels
        ]
        # Join rows carry NO ``origin_id`` — the column does not exist on them. A join
        # row's identity is DERIVED at merge time from
        # ``(panel.origin_id, asset.origin_id, role)``, so it needs no lineage of its
        # own. Rows whose panel or asset is somehow absent from the payload are dropped
        # rather than resurrected into a 23503 (defensive: ``serialize`` builds them by
        # walking the panels it captured, so this should be unreachable).
        new_panel_assets = [
            {
                "id": uuid7(),
                "panel_id": remap[UUID(pa["panel_id"])],
                "asset_id": remap[UUID(pa["asset_id"])],
                "role": pa["role"],
                "position": pa["position"],
            }
            for pa in panel_assets
            if UUID(pa["panel_id"]) in remap and UUID(pa["asset_id"]) in remap
        ]

        # FK dependency order: scenes → assets → panels (scene_id FK) → join rows (both
        # parents). NO ``SET CONSTRAINTS ALL DEFERRED`` here, unlike ``restore``: the
        # child project is EMPTY, so an ordered insert cannot transiently violate
        # ``uq_storyboard_panels_project_shot`` (every shot_key is new to this project),
        # the ``asset_id`` FK, or the ``uq_storyboard_panel_assets_one_location`` partial
        # index (nothing is being swapped — each is inserted exactly once). Restore needs
        # the deferral only because it upserts over a LIVE graph.
        await self._insert_rows(StoryboardScene, new_scenes)
        await self._insert_rows(StoryboardAsset, new_assets)
        await self._insert_rows(StoryboardPanel, new_panels)
        await self._insert_rows(StoryboardPanelAsset, new_panel_assets)
        await self.session.flush()

        # 5) The link row. Last, so a failed seed leaves no branch record behind (moot
        # inside one transaction, but it keeps the ordering honest).
        #
        # ``parent_base_snapshot_id`` is the SHARED ancestor for SIBLING merges (this branch
        # <-> another branch of the same ``project_id``). The FIRST fork of this parent has no
        # sibling to inherit from, so it establishes the shared base from its OWN branch_base;
        # every LATER fork reuses whatever the earliest sibling pinned, so all siblings merge
        # against one common base. (``base_snapshot_id`` stays per-branch — see the model.)
        branch_repo = StoryboardBranchRepository(self.session)
        shared_parent_base = await branch_repo.shared_parent_base(project_id, tenant_id=tenant_id)
        branch = await branch_repo.create(
            root_project_id=project_id,
            child_project_id=child.id,
            name=name,
            base_snapshot_id=base.id,
            parent_base_snapshot_id=shared_parent_base or base.id,
            actor_subject=str(actor_subject) if actor_subject is not None else None,
        )
        return branch, child

    async def abandon(self, branch_id: UUID, *, tenant_id: UUID) -> StoryboardBranch:
        """Close a branch WITHOUT merging it, and archive its child project.

        Raises :class:`BranchNotFoundError` (→ 404) when missing/cross-tenant,
        :class:`BranchAlreadyMergedError` (→ 409) when the branch already merged, and
        :class:`OpenChildBranchesError` (→ 409) when it still has open branches of its own
        — archiving this project would strand their only route back to the root, and they
        would go on merging into a parent that can never carry them upward.
        The child project is archived rather than deleted: its panels are the only
        record of the discarded work, and ``projects.status``'s CHECK already permits
        ``'archived'`` (no schema change) — every user-facing listing filters
        ``is_branch = false`` anyway, so an archived branch is invisible either way.
        """
        repo = StoryboardBranchRepository(self.session)
        branch = await repo.get(branch_id, tenant_id=tenant_id)
        if branch is None:
            raise BranchNotFoundError(str(branch_id))
        if branch.status == "merged":
            raise BranchAlreadyMergedError(str(branch_id))
        await assert_no_open_children(self.session, branch, tenant_id=tenant_id)

        child = await self._scoped_project(branch.child_project_id, tenant_id)
        if child is not None:
            child.status = "archived"
            child.archived_at = func.now()
        # Idempotent for an already-``abandoned`` branch: the status write is a no-op
        # re-assignment, not an error.
        await repo.mark_abandoned(branch)
        return branch
