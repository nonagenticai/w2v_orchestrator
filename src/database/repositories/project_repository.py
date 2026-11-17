"""Project repository — atomic create with deliverables, tenant-scoped reads."""

from __future__ import annotations

import datetime
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from typing import cast as type_cast
from uuid import UUID, uuid7

from sqlalchemy import Text, case, cast, func, literal, null, select, update
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased, selectinload
from sqlalchemy.orm.attributes import set_committed_value

from src.database.models import Deliverable, Project, Run, StageOutput, StoryboardBranch

# A branch-of-branch chain is walked UP one hop per link row; this bounds the
# recursive CTE so a corrupt cycle in ``storyboard_branches`` (structurally
# impossible - a fork always mints a NEW child that points at an OLDER project -
# but cheap to guard) can never spin forever. Real chains are 1-3 deep.
_MAX_BRANCH_DEPTH = 64

# The ONLY project statuses the e2e retention sweep may archive. Terminal by
# construction: `drafting`/`producing` mean a run is still working the project,
# and `archived` is already the destination (excluding it makes a re-sweep a
# genuine no-op instead of restamping `archived_at`). See
# `ProjectRepository.archive_stale_for_subject`.
_PRUNABLE_PROJECT_STATUSES = ("completed", "failed")


@dataclass(frozen=True)
class BranchAncestorRow:
    """One hop of a branch's lineage: a parent project on the way back to the root.

    ``root_project_id`` in ``storyboard_branches`` is really the branch's IMMEDIATE
    parent (a fork's link row points at whatever it forked — see
    ``StoryboardBranchService.create_branch``), so a branch-of-branch chain is a linked
    list of these rows. ``is_branch`` lets the FE tell an intermediate parent (itself a
    branch) from the ultimate root (a normal project) when it renders the chain.
    """

    project_id: UUID
    title: str
    is_branch: bool


@dataclass(frozen=True)
class BranchContextRow:
    """The branch↔root link for a HIDDEN branch project, plus the root's title.

    A flat value object on purpose: the route needs five scalars, and returning the
    ``StoryboardBranch`` ORM row instead would push the caller into a lazy
    ``branch.root.title`` access (MissingGreenlet under AsyncSession). The root title
    is selected in the same join, so there is nothing left to lazy-load.

    ``root_project_id`` / ``root_title`` name the IMMEDIATE parent (the merge target and
    the back-navigation), kept for backward compatibility with the pre-chain contract.
    ``ancestors`` is the full lineage from that immediate parent up to the ultimate root
    (``ancestors[0]`` == the immediate parent, ``ancestors[-1]`` == the root); for a
    direct branch it is a single element. It lets the banner say "branch of B, which is a
    branch of A" instead of stopping one hop up now that branch-of-branch is supported.
    """

    branch_id: UUID
    name: str
    status: str
    root_project_id: UUID
    root_title: str
    ancestors: tuple[BranchAncestorRow, ...]


@dataclass
class ProjectListResult:
    items: list[Project]
    total: int
    # Per-project metadata for the list view, keyed by project id. Computed in
    # two page-scoped companion queries (no N+1); the route merges them onto
    # each ProjectResponse.
    deliverable_counts: dict[UUID, int]
    last_run_statuses: dict[UUID, str | None]
    # The latest run's ``pause_reason`` (0102) where that run is paused, so a card
    # can tell "needs review" (checkpoint) from "paused" (someone pressed Pause).
    last_run_pause_reasons: dict[UUID, str | None]
    # Per-project poster: project_id → (run_id, raw minio thumbnail url) from the
    # most recent COMPLETED run that produced a thumbnail. run_id is carried so
    # the route can rewrite the minio url to the browser-reachable
    # /api/runs/{id}/artifact proxy path. Absent when no completed run has one.
    thumbnails: dict[UUID, tuple[UUID, str]]


# Whitelist of sortable columns for ``list()`` — the API passes a raw string,
# so this dict is also the injection guard (unknown keys fall back to
# created_at).
_SORT_COLUMNS = {
    "created_at": Project.created_at,
    "finished_at": Project.finished_at,
    "updated_at": Project.updated_at,
    "title": Project.title,
    "status": Project.status,
    "type": Project.type,
}


def _derive_status(*, with_run: int, active: int, done: int, bad: int, deliv_total: int) -> str:
    """Map per-deliverable latest-run counts to a project status (see
    :meth:`ProjectRepository.recompute_status`)."""
    if with_run == 0:
        return "drafting"
    if active > 0:
        return "producing"
    if done == deliv_total:
        return "completed"
    if bad > 0:
        return "failed"
    return "producing"


class ProjectRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_with_deliverables(
        self,
        *,
        tenant_id: UUID,
        owner_subject_id: UUID,
        type_: str,
        title: str,
        brief: dict[str, Any],
        brand: dict[str, Any],
        voice: dict[str, Any],
        theme: dict[str, Any],
        visibility: str,
        deliverables: Sequence[dict[str, Any]],
        parent_project_id: UUID | None = None,
    ) -> Project:
        if type_ == "lesson" and parent_project_id is None:
            raise ValueError("lesson projects must have a parent_project_id pointing to a course")
        proj = Project(
            id=uuid7(),
            tenant_id=tenant_id,
            owner_subject_id=owner_subject_id,
            type=type_,
            title=title,
            brief=brief,
            brand=brand,
            voice=voice,
            theme=theme,
            visibility=visibility,
            parent_project_id=parent_project_id,
            status="drafting",
        )
        proj.deliverables = [
            Deliverable(
                id=uuid7(),
                name=d["name"],
                format=d["format"],
                captions_mode=d.get("captions_mode", "none"),
                target_duration_seconds=d.get("target_duration_seconds"),
                status="idle",
            )
            for d in deliverables
        ]
        self.session.add(proj)
        await self.session.flush()
        return proj

    async def create_draft(
        self,
        *,
        tenant_id: UUID,
        owner_subject_id: UUID,
        type_: str,
        title: str,
        visibility: str,
        brief: dict | None = None,
        brand: dict | None = None,
        voice: dict | None = None,
        theme: dict | None = None,
    ) -> Project:
        """Create a project row with NO deliverables yet — the create-project wizard's
        early-draft path (type is chosen, deliverables/brief come later).

        Thin wrapper over :meth:`create_with_deliverables` with an empty deliverables
        list so the insert logic lives in exactly one place: ``status`` still defaults
        to ``drafting`` and zero deliverable child rows is legal. brief/brand/voice/theme
        start as empty dicts; the wizard PATCHes them + adds deliverables via the
        existing endpoints before starting runs. The optional config dicts carry a
        ``clone_from_project_id`` seed (``services.project_clone``); omitted ⇒ ``{}``.
        """
        return await self.create_with_deliverables(
            tenant_id=tenant_id,
            owner_subject_id=owner_subject_id,
            type_=type_,
            title=title,
            brief=brief or {},
            brand=brand or {},
            voice=voice or {},
            theme=theme or {},
            visibility=visibility,
            deliverables=[],
        )

    async def add_deliverable(
        self,
        project_id: UUID,
        *,
        tenant_id: UUID,
        name: str,
        format_: str,
        captions_mode: str,
        target_duration_seconds: int | None,
    ) -> Deliverable | None:
        """Append ONE deliverable to an existing project — the wizard's finalize step.

        Returns ``None`` when the project is not visible to *tenant_id* (unknown OR
        another tenant's), so the route can 404 both cases identically and never leak
        a foreign project id. Ownership is resolved through :meth:`get`, the same
        tenant-scoped lookup every other project sub-route uses.

        Mirrors the child rows :meth:`create_with_deliverables` builds (same defaults,
        same ``status='idle'``) — the derived-on-read status contract is unchanged.
        FLUSHes only; the route commits before responding (see routes/projects.py).
        """
        proj = await self.get(project_id, tenant_id=tenant_id)
        if proj is None:
            return None
        deliverable = Deliverable(
            id=uuid7(),
            project_id=proj.id,
            name=name,
            format=format_,
            captions_mode=captions_mode,
            target_duration_seconds=target_duration_seconds,
            status="idle",
        )
        self.session.add(deliverable)
        await self.session.flush()
        await self.session.refresh(deliverable)
        return deliverable

    async def create_branch_project(self, root: Project, *, title: str) -> Project:
        """Create the HIDDEN child project that backs a storyboard branch (0052).

        Deliberately NOT routed through :meth:`create_with_deliverables`, for two
        independent reasons:

        1. That method raises ``ValueError`` for ``type_ == "lesson"`` without a
           ``parent_project_id`` — and a branch of a LESSON project is exactly that
           shape (``type`` is copied from the root, ``parent_project_id`` is
           deliberately NULL). Reusing it would hard-fail every branch of a lesson.
        2. ``parent_project_id`` MUST stay NULL. It is the course→lesson hierarchy;
           hanging a branch off it would surface the branch as a lesson inside a
           course and pollute four child-listing queries. Branch linkage lives
           SOLELY in ``storyboard_branches``.

        ``tenant_id`` is copied from the root and is NOT optional — RLS keys off it,
        and a branch that does not inherit it is a cross-tenant leak. ``owner_subject_id``
        / ``visibility`` / ``type`` are copied so the branch behaves like its root
        everywhere else. No deliverables: a branch is a storyboard working copy, not a
        thing that renders.
        """
        proj = Project(
            id=uuid7(),
            tenant_id=root.tenant_id,
            owner_subject_id=root.owner_subject_id,
            type=root.type,
            title=title,
            brief=dict(root.brief or {}),
            brand=dict(root.brand or {}),
            voice=dict(root.voice or {}),
            theme=dict(root.theme or {}),
            visibility=root.visibility,
            parent_project_id=None,
            status="drafting",
            is_branch=True,
        )
        self.session.add(proj)
        await self.session.flush()
        return proj

    async def get(self, project_id: UUID, *, tenant_id: UUID) -> Project | None:
        stmt = (
            select(Project)
            .where(Project.id == project_id, Project.tenant_id == tenant_id)
            .options(selectinload(Project.deliverables))
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def get_with_children(self, project_id: UUID, *, tenant_id: UUID) -> Project | None:
        proj = await self.get(project_id, tenant_id=tenant_id)
        if proj is None:
            return None
        # ``is_branch`` filter is defense-in-depth, not the load-bearing guard: a fork
        # leaves ``parent_project_id`` NULL (branch linkage lives in
        # ``storyboard_branches``), so a branch cannot match this parent-keyed query
        # today. It is asserted here anyway so the fork invariant is not the ONLY
        # thing standing between a branch and the children list.
        children_stmt = (
            select(Project)
            .where(
                Project.parent_project_id == project_id,
                Project.tenant_id == tenant_id,
                Project.is_branch.is_(False),
            )
            .order_by(Project.position.nulls_last(), Project.created_at)
        )
        children = list((await self.session.execute(children_stmt)).scalars().all())
        # Bypass the InstrumentedAttribute lazy-load trigger that fires on a
        # plain attribute assignment under AsyncSession (MissingGreenlet).
        set_committed_value(proj, "children", children)
        return proj

    async def branch_context(self, project_id: UUID, *, tenant_id: UUID) -> BranchContextRow | None:
        """ "Am I a branch, and of what?" — the ``storyboard_branches`` row whose CHILD is
        this project, joined to the root project's title.

        Kept OUT of :meth:`get_with_children` deliberately: the overwhelming majority of
        project reads are non-branch, and this would add a join to every one of them. The
        route calls it only when ``project.is_branch`` is true.

        Tenant scoping climbs ``root_project_id -> projects.tenant_id`` — the same 1-hop
        join RLS enforces in prod (``RLS_JOIN_PREDICATES_BRANCHES``), and the same side
        ``StoryboardBranchRepository`` gates on. The child inherits the root's tenant at
        fork, so both resolve to the same tenant; the root is the stable, user-visible
        side. A foreign tenant gets ``None`` (the route never reaches here anyway — the
        child project itself is already a tenant-scoped 404).

        Returns ``None`` for a normal project, and also for the pathological case of an
        ``is_branch`` row with no linkage record (the FK is ON DELETE CASCADE, so this
        should be unreachable) — the detail read degrades to ``branch: null`` rather
        than 500ing.

        The lineage is walked in ONE recursive CTE, not a per-hop loop: a branch-of-branch
        chain is a linked list up ``storyboard_branches.root_project_id`` (the immediate
        parent), and the CTE follows it to the root in a single round-trip. Every hop is
        tenant-scoped by the outer join (a chain never crosses tenants — the child inherits
        the root's tenant at fork — so this is a safety belt that also collapses a foreign
        tenant to ``None``, matching the pre-chain behaviour). ``depth < _MAX_BRANCH_DEPTH``
        bounds the recursion against a (structurally impossible) cycle.
        """
        # Anchor: the branch row whose CHILD is this project. It carries the branch's own
        # identity (id / name / status) at depth 0; the recursive term climbs one link per
        # hop and carries NULLs for those identity columns (only depth 0 is a real branch
        # from this project's point of view). The columns must line up across the UNION, so
        # the NULLs are cast to the anchor's column types.
        anchor = (
            select(
                StoryboardBranch.id.label("branch_id"),
                StoryboardBranch.name.label("branch_name"),
                StoryboardBranch.status.label("branch_status"),
                StoryboardBranch.root_project_id.label("parent_id"),
                literal(0).label("depth"),
            )
            .where(StoryboardBranch.child_project_id == project_id)
            .cte("branch_ancestry", recursive=True)
        )
        up = aliased(StoryboardBranch)
        ancestry = anchor.union_all(
            select(
                cast(null(), postgresql.UUID(as_uuid=True)).label("branch_id"),
                cast(null(), Text()).label("branch_name"),
                cast(null(), Text()).label("branch_status"),
                up.root_project_id.label("parent_id"),
                (anchor.c.depth + 1).label("depth"),
            ).where(
                up.child_project_id == anchor.c.parent_id,
                anchor.c.depth < _MAX_BRANCH_DEPTH,
            )
        )
        stmt = (
            select(
                ancestry.c.branch_id,
                ancestry.c.branch_name,
                ancestry.c.branch_status,
                ancestry.c.parent_id,
                ancestry.c.depth,
                Project.title,
                Project.is_branch,
            )
            .join(Project, Project.id == ancestry.c.parent_id)
            .where(Project.tenant_id == tenant_id)
            .order_by(ancestry.c.depth)
        )
        rows = (await self.session.execute(stmt)).all()
        if not rows:
            return None
        head = rows[0]
        # Defensive: if the immediate parent (depth 0) was tenant-filtered out, there is no
        # coherent context to return — degrade to None rather than a chain missing its head.
        if head.depth != 0 or head.branch_id is None:
            return None
        ancestors = tuple(
            BranchAncestorRow(project_id=r.parent_id, title=r.title, is_branch=r.is_branch)
            for r in rows
        )
        return BranchContextRow(
            branch_id=head.branch_id,
            name=head.branch_name,
            status=head.branch_status,
            root_project_id=head.parent_id,
            root_title=head.title,
            ancestors=ancestors,
        )

    async def list(
        self,
        *,
        tenant_id: UUID,
        type_: str | None = None,
        status: str | None = None,
        parent_id: UUID | None = None,
        include_archived: bool = False,
        sort_by: str = "created_at",
        order: str = "desc",
        limit: int,
        offset: int,
    ) -> ProjectListResult:
        # A storyboard BRANCH is a real (hidden) project row, so it would otherwise
        # show up here — this is THE projects-page leak the branch design calls out.
        # Unlike the parent-keyed queries below, this one is only tenant-scoped, so
        # nothing else excludes it. There is deliberately no ``include_branches``
        # escape hatch: branches are listed by the dedicated
        # ``GET /projects/{pid}/storyboard/branches`` route (which reads
        # ``storyboard_branches``), never through the generic project list.
        stmt = select(Project).where(
            Project.tenant_id == tenant_id,
            Project.is_branch.is_(False),
        )
        if type_:
            stmt = stmt.where(Project.type == type_)
        if status:
            stmt = stmt.where(Project.status == status)
        if parent_id:
            stmt = stmt.where(Project.parent_project_id == parent_id)
        if not include_archived:
            stmt = stmt.where(Project.status != "archived")
        total = (
            await self.session.execute(select(func.count()).select_from(stmt.subquery()))
        ).scalar_one()

        col = _SORT_COLUMNS.get(sort_by, Project.created_at)
        direction = col.desc() if order == "desc" else col.asc()
        # finished_at is nullable — keep unfinished projects last regardless of
        # direction so they don't dominate an ascending sort. Project.id is a
        # stable tiebreak for deterministic paging.
        order_clause = direction.nulls_last() if sort_by == "finished_at" else direction
        items = list(
            (
                await self.session.execute(
                    stmt.order_by(order_clause, Project.id.desc()).limit(limit).offset(offset)
                )
            )
            .scalars()
            .all()
        )

        # Two page-scoped companion queries for the list-view metadata: a grouped
        # count and a windowed latest-run lookup. Both reuse existing indexes
        # (idx_deliverables_project, idx_runs_deliverable) and add exactly two
        # round-trips per page — never per-row.
        ids = [p.id for p in items]
        deliverable_counts: dict[UUID, int] = {}
        last_run_statuses: dict[UUID, str | None] = {}
        last_run_pause_reasons: dict[UUID, str | None] = {}
        thumbnails: dict[UUID, tuple[UUID, str]] = {}
        if ids:
            dc_rows = await self.session.execute(
                select(Deliverable.project_id, func.count(Deliverable.id))
                .where(Deliverable.project_id.in_(ids))
                .group_by(Deliverable.project_id)
            )
            deliverable_counts = {pid: cnt for pid, cnt in dc_rows.all()}
            lr_sub = (
                select(
                    Deliverable.project_id.label("pid"),
                    Run.status.label("status"),
                    Run.pause_reason.label("pause_reason"),
                    func.row_number()
                    .over(
                        partition_by=Deliverable.project_id,
                        order_by=(Run.created_at.desc(), Run.id.desc()),
                    )
                    .label("rn"),
                )
                .join(Run, Run.deliverable_id == Deliverable.id)
                .where(Deliverable.project_id.in_(ids))
                .subquery()
            )
            lr_rows = (
                await self.session.execute(
                    select(lr_sub.c.pid, lr_sub.c.status, lr_sub.c.pause_reason).where(
                        lr_sub.c.rn == 1
                    )
                )
            ).all()
            last_run_statuses = {pid: st for pid, st, _ in lr_rows}
            last_run_pause_reasons = {
                pid: reason if st == "paused" else None for pid, st, reason in lr_rows
            }

            # Third companion query: per-project poster. Pick the thumbnail from
            # the most recent COMPLETED run that produced one, preferring an
            # operator override over the auto-generated cover. Page-scoped via the
            # same project-id filter (no N+1); tenant isolation comes from `ids`
            # (already tenant-scoped) plus RLS on the joined tables when enforced.
            override = StageOutput.output["thumbnail_override_url"].astext
            thumb_expr = func.coalesce(override, StageOutput.output["thumbnail_url"].astext)
            th_sub = (
                select(
                    Deliverable.project_id.label("pid"),
                    Run.id.label("run_id"),
                    thumb_expr.label("thumb"),
                    func.row_number()
                    .over(
                        partition_by=Deliverable.project_id,
                        order_by=(
                            Run.completed_at.desc().nulls_last(),
                            Run.id.desc(),
                            override.desc().nulls_last(),  # prefer override over auto cover
                        ),
                    )
                    .label("rn"),
                )
                .join(Run, StageOutput.run_id == Run.id)
                .join(Deliverable, Run.deliverable_id == Deliverable.id)
                .where(
                    Deliverable.project_id.in_(ids),
                    Run.status == "completed",
                    StageOutput.status == "completed",
                    thumb_expr.isnot(None),
                )
                .subquery()
            )
            th_rows = await self.session.execute(
                select(th_sub.c.pid, th_sub.c.run_id, th_sub.c.thumb).where(th_sub.c.rn == 1)
            )
            thumbnails = {pid: (run_id, thumb) for pid, run_id, thumb in th_rows.all()}

        return ProjectListResult(
            items=items,
            total=total,
            deliverable_counts=deliverable_counts,
            last_run_statuses=last_run_statuses,
            last_run_pause_reasons=last_run_pause_reasons,
            thumbnails=thumbnails,
        )

    async def recompute_status(self, project_id: UUID) -> str | None:
        """Re-derive a project's status from its deliverables' latest runs AND
        its child projects (parent/course rollup).

        ``deliverables.status`` is never written anywhere in the codebase, so
        "all deliverables completed" is interpreted as "each deliverable's
        most-recent run is completed". Each deliverable's latest run and each
        non-archived child project are treated as one *unit*; the units are
        unioned and reduced to a status:

        * ``drafting``  — no unit has any progress yet
        * ``producing`` — at least one unit is active, or a mix that is neither
          all-done nor failed
        * ``completed`` — every unit is completed
        * ``failed``    — a unit failed/cancelled and none are active

        Rollup: a course owns both its own pipeline deliverable AND its lesson
        children, so a failed/incomplete lesson surfaces on the course even if
        the course's own run finished. Child statuses map: producing→active,
        completed→done, failed→bad, drafting→no-progress (archived excluded).
        The dispatcher recomputes the child first, then the parent, so the
        parent sees fresh child statuses within the same transaction.

        ``archived`` is owned by :meth:`soft_delete` and never overwritten.
        Idempotent: only issues an UPDATE when a value actually changes, and
        stays inside the caller's transaction (no commit) so status commits
        atomically with the run-status change that triggered it.
        """
        proj = (
            await self.session.execute(select(Project).where(Project.id == project_id))
        ).scalar_one_or_none()
        if proj is None:
            return None
        if proj.status == "archived":
            return proj.status

        # One row per deliverable (partition by the never-null Deliverable.id so
        # run-less deliverables each keep their own NULL-run row), carrying that
        # deliverable's latest run status.
        latest = (
            select(
                Run.status.label("run_status"),
                func.row_number()
                .over(
                    partition_by=Deliverable.id,
                    order_by=(Run.created_at.desc().nulls_last(), Run.id.desc().nulls_last()),
                )
                .label("rn"),
            )
            .select_from(Deliverable)
            .outerjoin(Run, Run.deliverable_id == Deliverable.id)
            .where(Deliverable.project_id == project_id)
            .subquery()
        )
        counts = (
            await self.session.execute(
                select(
                    func.count().label("deliv_total"),
                    func.count(latest.c.run_status).label("with_run"),
                    func.count(
                        case((latest.c.run_status.in_(("pending", "running", "paused")), 1))
                    ).label("active"),
                    func.count(case((latest.c.run_status == "completed", 1))).label("done"),
                    func.count(case((latest.c.run_status.in_(("failed", "cancelled")), 1))).label(
                        "bad"
                    ),
                ).where(latest.c.rn == 1)
            )
        ).one()

        # Parent/course rollup: fold each non-archived child project in as an
        # extra unit alongside this project's own deliverable runs. No-op for
        # leaf projects (lessons have no children).
        kids = (
            await self.session.execute(
                select(
                    func.count().label("total"),
                    func.count(case((Project.status != "drafting", 1))).label("with_signal"),
                    func.count(case((Project.status == "producing", 1))).label("active"),
                    func.count(case((Project.status == "completed", 1))).label("done"),
                    func.count(case((Project.status == "failed", 1))).label("bad"),
                ).where(
                    Project.parent_project_id == project_id,
                    Project.status != "archived",
                    # Defense-in-depth (see get_with_children): a branch leaves
                    # parent_project_id NULL, so it cannot inflate a parent's status
                    # rollup today — asserted anyway rather than relying on that alone.
                    Project.is_branch.is_(False),
                )
            )
        ).one()

        new = _derive_status(
            with_run=counts.with_run + kids.with_signal,
            active=counts.active + kids.active,
            done=counts.done + kids.done,
            bad=counts.bad + kids.bad,
            deliv_total=counts.deliv_total + kids.total,
        )

        values: dict[str, Any] = {}
        if new != proj.status:
            values["status"] = new
        if new == "completed":
            if proj.finished_at is None:
                values["finished_at"] = func.now()
        elif proj.finished_at is not None:
            values["finished_at"] = None
        if values:
            await self.session.execute(
                update(Project).where(Project.id == project_id).values(**values)
            )
        return new

    async def update(
        self,
        project_id: UUID,
        *,
        tenant_id: UUID,
        fields: dict[str, Any],
    ) -> Project | None:
        proj = await self.get(project_id, tenant_id=tenant_id)
        if proj is None:
            return None
        for k, v in fields.items():
            setattr(proj, k, v)
        await self.session.flush()
        await self.session.refresh(proj)
        return proj

    async def soft_delete(self, project_id: UUID, *, tenant_id: UUID) -> bool:
        proj = await self.get(project_id, tenant_id=tenant_id)
        if proj is None:
            return False
        proj.status = "archived"
        proj.archived_at = func.now()
        await self.session.flush()
        return True

    async def reorder_children(
        self,
        parent_id: UUID,
        ordered_child_ids: Sequence[UUID],
        *,
        tenant_id: UUID,
    ) -> bool:
        for position, child_id in enumerate(ordered_child_ids):
            stmt = select(Project).where(
                Project.id == child_id,
                Project.tenant_id == tenant_id,
                Project.parent_project_id == parent_id,
                # Defense-in-depth (see get_with_children): a branch leaves
                # parent_project_id NULL, so it cannot be reordered into a parent's
                # child list today — asserted anyway rather than relying on that alone.
                Project.is_branch.is_(False),
            )
            child = (await self.session.execute(stmt)).scalar_one_or_none()
            if child is None:
                return False
            child.position = position
        await self.session.flush()
        return True

    async def archive_stale_for_subject(
        self,
        *,
        tenant_id: UUID,
        owner_subject_id: UUID,
        older_than: datetime.datetime,
        limit: int,
        # `list[UUID]` cannot be spelled here: `ProjectRepository.list` shadows the
        # builtin in class scope, so ty reads the annotation as a subscript of the
        # METHOD. `Sequence` is the honest contract anyway — callers only iterate,
        # index and len() the result.
    ) -> Sequence[UUID]:
        """Archive up to ``limit`` terminal projects owned by ONE subject.

        The e2e retention sweep's only write. Returns the ids it archived, in the
        order it archived them, so the caller can log exactly what moved.

        Scope — all four predicates are load-bearing, none is an optimisation:

        * ``tenant_id`` AND ``owner_subject_id`` together are the whole safety
          argument. The owner's real projects live in the SAME tenant under a
          different subject, so a tenant-only sweep would archive them. There is
          deliberately no code path here that omits either one.
        * ``status IN ('completed', 'failed')`` — terminal only. A ``drafting``
          or ``producing`` project belongs to a live run and is never eligible at
          any age, which is what keeps the sweep from racing a running scenario.
          ``archived`` is excluded by the same clause, so a re-sweep is a no-op
          rather than a churn of ``archived_at``.
        * ``updated_at < older_than`` — the age window. ``updated_at`` (not
          ``created_at``) is the right clock: a long course run keeps touching
          the row, and the window should start when the project went quiet.

        ``is_branch`` rows are skipped: a storyboard branch is a hidden child that
        no listing shows, so it is not the UI clutter this sweep exists for, and
        archiving one would put a status the branch machinery never writes onto a
        row it owns.

        CHILDREN BEFORE PARENTS. Ordering by ``parent_project_id IS NULL``
        ascending puts lessons (a non-NULL parent → ``false``) ahead of courses
        (``true``). Two reasons, and the second is the one that bites:

        1. ``trg_lesson_parent_is_course`` (migration 0015) is ``BEFORE INSERT OR
           UPDATE``, and ``projects.parent_project_id`` is ``ON DELETE SET
           NULL`` — so a HARD delete of a course raises on the surviving lesson's
           cascade UPDATE. This sweep only soft-archives (no type or parent
           changes), so the trigger cannot fire either way; ordering here keeps
           the invariant true for anything that later reuses this query.
        2. ``limit`` can truncate a batch mid-family. Cut after the children, a
           course is left listed with its lessons archived — coherent. Cut the
           other way and the lessons are stranded in the listing with no parent
           above them, which is exactly the clutter being removed.
        """
        stmt = (
            select(Project.id)
            .where(
                Project.tenant_id == tenant_id,
                Project.owner_subject_id == owner_subject_id,
                Project.status.in_(_PRUNABLE_PROJECT_STATUSES),
                Project.updated_at < older_than,
                Project.is_branch.is_(False),
            )
            # `.is_(None)` yields a boolean expression; false sorts before true,
            # so children (parent IS NOT NULL) come first. `Project.id` is a
            # stable tiebreak so a truncated batch pages deterministically.
            .order_by(Project.parent_project_id.is_(None).asc(), Project.id.asc())
            .limit(limit)
        )
        project_ids = list((await self.session.execute(stmt)).scalars().all())
        if not project_ids:
            return []
        # One UPDATE per id, issued in the order above, so the children-first
        # ordering survives into the write and is not re-planned away by a
        # single set-based UPDATE ... WHERE id = ANY(...).
        archived: list[UUID] = []
        for project_id in project_ids:
            # `AsyncSession.execute` is typed as returning `Result`, but an
            # UPDATE genuinely yields a `CursorResult`, which is where
            # `rowcount` lives. The cast is the type-checker's problem only.
            result = type_cast(
                "CursorResult[Any]",
                await self.session.execute(
                    update(Project)
                    .where(
                        Project.id == project_id,
                        Project.tenant_id == tenant_id,
                        # ⛔ RE-ASSERT THE SUBJECT AT WRITE TIME TOO. Nothing in
                        # `src/` rewrites `owner_subject_id` today, so this cannot
                        # fire — but it is the predicate carrying the ENTIRE safety
                        # argument, and leaving it out of the write while tenant and
                        # status are both re-asserted is precisely the asymmetry that
                        # made the mid-run archive race possible. Symmetry here is one
                        # line and removes the need to re-derive "is it reachable?"
                        # the next time someone adds a writer.
                        Project.owner_subject_id == owner_subject_id,
                        # ⛔ RE-ASSERT THE STATUS AT WRITE TIME. The SELECT above is a
                        # separate statement, so a run can start between the two and
                        # flip this project to `producing`. Without this clause the
                        # sweep archives a LIVE project — and that is not
                        # self-healing: `recompute_status` returns early on
                        # `archived` (see its guard), so the project never climbs back
                        # out and the run's result is stranded behind an archived
                        # parent. The window is sub-second and confined to the e2e
                        # subject, but the damage is permanent, so the anti-race the
                        # docstring promises has to be enforced HERE, not only in the
                        # candidate query.
                        Project.status.in_(_PRUNABLE_PROJECT_STATUSES),
                    )
                    .values(status="archived", archived_at=func.now()),
                ),
            )
            # Report only what the UPDATE actually matched. A row that lost the
            # race matched zero and must not appear in the returned ids — the
            # caller logs that list as "archived", and the post-condition check
            # re-reads exactly these ids.
            if result.rowcount:
                archived.append(project_id)
        await self.session.flush()
        return archived

    async def foreign_owned_ids(
        self, project_ids: Sequence[UUID], *, owner_subject_id: UUID
    ) -> Sequence[UUID]:
        """Of ``project_ids``, those NOT owned by ``owner_subject_id``.

        The read half of the sweep's pre-commit post-condition. Empty is the only
        acceptable answer; anything else means the batch reached beyond the
        configured subject and the transaction must be rolled back.

        Why this exists on top of the scoping already in
        :meth:`archive_stale_for_subject`: the per-sweep cap bounds the blast per
        TICK, not in total. At 50/tick on an hourly loop a wrong subject would
        still take the owner's ~95 real projects inside ~2 hours — slowly enough
        that nobody is watching when it starts. Re-reading what is about to be
        committed turns that from a gradual catastrophe into a loud no-op.
        """
        if not project_ids:
            return []
        stmt = select(Project.id).where(
            Project.id.in_(list(project_ids)),
            # `is_distinct_from`, not `!=`: SQL `!=` is NULL-blind, so a NULL
            # owner would compare UNKNOWN and drop out of this result — i.e. the
            # one row least provably in scope would be reported as clean. The
            # column is NOT NULL today, so this is unreachable; it also costs
            # nothing, and this is the check whose whole job is to be true when
            # the rest of the reasoning has failed.
            Project.owner_subject_id.is_distinct_from(owner_subject_id),
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def open_notes_count(self, project_id: UUID, *, tenant_id: UUID) -> int:
        from src.database.models import Annotation, Run, StageOutput

        stmt = (
            select(func.count(Annotation.id))
            .join(StageOutput, Annotation.stage_output_id == StageOutput.id)
            .join(Run, StageOutput.run_id == Run.id)
            .join(Deliverable, Run.deliverable_id == Deliverable.id)
            .join(Project, Deliverable.project_id == Project.id)
            .where(
                Project.id == project_id,
                Project.tenant_id == tenant_id,
                Annotation.status == "open",
            )
        )
        return (await self.session.execute(stmt)).scalar_one()
