"""Storyboard branch MERGE — the three-way diff and its apply (the other half of PR #475).

``branch_service.create_branch`` established the three invariants that make this file
possible; this file is the only consumer of all three:

* the fork pinned a ``kind='branch_base'`` snapshot of the ROOT and seeded the child from
  THAT payload, so ``base`` is byte-identical to what the child started from — the diff
  can never be skewed by a write that landed between two live reads;
* every copied row carries ``origin_id`` lineage (source's ``origin_id`` if set, else its
  ``id``), so a panel is identifiable ACROSS the three projects even though its ``id`` was
  re-minted; and
* the child inherited ``tenant_id``, so one tenant scope covers all three sides.

The three sides:

* ``base``   — the branch's ``base_snapshot_id`` payload (the board AT fork time),
* ``main``   — the ROOT project's live board NOW,
* ``branch`` — the CHILD project's live board NOW.

Identity is the LINEAGE KEY — ``origin_id or id`` — never the panel id. A root panel is
its own origin; the fork stamped that id onto the child's copy; so both sides key to the
same value. Keying on ``id`` would report every panel as added-on-one-side and
deleted-on-the-other, i.e. a diff that is 100% noise.

═══ The two invariants the APPLY must not violate ═══════════════════════════════════

1. **UPSERT onto the root's EXISTING panel ids — never delete-then-reinsert.**
   ``storyboard_panel_events.panel_id`` is ON DELETE CASCADE. Re-inserting a panel (even
   with identical content, even with the same id in the same transaction) reaps its whole
   approval/event trail. Merge therefore maps branch panel → root panel by lineage and
   writes through :func:`storyboard_snapshot_service.upsert_rows` — the SAME primitive
   snapshot RESTORE uses, for the same reason. An explicit ``deleted`` resolution is the
   ONE sanctioned delete: the author asked for it, and the trail dies with the panel it
   describes.

2. **Auto-snapshot the ROOT before applying** (``kind='auto'``, exactly like restore), so
   a merge is reversible by restoring that id. It is minted inside the same transaction,
   so a failed merge rolls the snapshot back too.

Approval state (``status`` / ``approved_at`` / ``approved_by``) is neither diffed nor
merged. Approval is the ROOT's governance record: carrying the branch's copy over would
either forge an approval on the root (a branch author approving their own fork) or
silently unapprove a root panel — in both cases with no root-side trail event to explain
it. Merge carries CONTENT; ``revision`` is bumped on every panel it touches (the
content-edit counter), and a ``merged`` trail event is appended per touched panel.

Concurrency: :meth:`diff` returns a ``root_fingerprint`` — a sha256 over the root board's
NORMALISED content (panels + scenes + assets). :meth:`merge` takes it back and raises
:class:`StaleRootError` (→ 409) when the root has moved since, rather than clobbering
whatever landed on main while the human was reading the diff. Content-hashed on purpose:
a timestamp/``updated_at`` etag skews across the orchestrator's replicas and fires on
writes that changed nothing.

That fingerprint is a compare-and-set, and its compare and its set are two separate
statements under READ COMMITTED — so it is NOT self-sufficient on a service that runs
``replicas:2`` + HPA. There are TWO distinct lost-update holes, and the fingerprint sees
neither:

* **merge vs merge** — two merges of the same root each read the pre-merge board, each
  find the fingerprint fresh, and the second upserts its panels from an in-memory copy of
  a root row the first has already rewritten. :meth:`merge` closes this by taking a row
  lock on the ROOT ``projects`` row (:meth:`_lock_root`, ``SELECT … FOR UPDATE OF
  projects``, tenant-scoped) BEFORE it reads any board. The loser blocks there, reads the
  root only AFTER the winner committed, and its fingerprint is then correctly stale → 409.

* **merge vs PATCH** — a concurrent ``PATCH /panels/{id}`` (status/approve/reorder) commits
  between merge's read and merge's upsert; the upsert's ``ON CONFLICT DO UPDATE SET <all
  cols>`` then silently reverts it. The ``projects``-row lock does NOT close this: a PATCH
  never touches the ``projects`` row, so it slips straight past that lock. :meth:`merge`
  therefore ALSO takes ``FOR UPDATE`` on the root's panel/scene/asset rows themselves
  (:meth:`_lock_root_board`) BEFORE reading — the PATCH's UPDATE blocks on those row locks,
  runs only after the merge commits, and re-applies on top instead of being clobbered.

Both locks are transaction-scoped: held for exactly the caller's transaction and released
by its COMMIT/ROLLBACK.

Everything runs in the CALLER's transaction (the route commits) — snapshot, panel writes,
trail events and the branch's ``status='merged'`` land atomically or not at all.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid7

from sqlalchemy import delete, func, insert, select, text
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
    StoryboardAssetRepository,
    StoryboardBranchRepository,
    StoryboardPanelEventRepository,
    StoryboardPanelRepository,
    StoryboardRepository,
    StoryboardSceneRepository,
    StoryboardSnapshotRepository,
)
from src.services.storyboard.branch_service import BranchNotFoundError, assert_no_open_children
from src.services.storyboard.storyboard_snapshot_service import (
    StoryboardSnapshotService,
    _coerce_row,
    _row_to_dict,
    upsert_rows,
)

# The JSON-safe result dict both merge methods return; the ROUTE validates it into its
# response model (Merge/SiblingMergeResponse), keeping the service decoupled from the API
# schema layer. Named once here rather than repeating the ``dict[str, Any]`` boundary per
# method — the two payloads have different keys, so a shared TypedDict would not fit.
_MergeResult = dict[str, Any]


class BranchNotOpenError(Exception):
    """The branch is already ``merged`` or ``abandoned`` (route → 409).

    Terminal in both directions. Re-merging a ``merged`` branch would replay stale content
    over a root that has since moved on; merging an ``abandoned`` one would resurrect work
    its author explicitly discarded. The resource exists and the caller may see it — its
    STATE refuses, which is exactly what 409 means.
    """

    def __init__(self, status: str) -> None:
        super().__init__(f"branch is {status}, not open")
        self.status = status


class BaseSnapshotMissingError(Exception):
    """The branch's ``base_snapshot_id`` no longer resolves (route → 409).

    Should be unreachable: ``kind='branch_base'`` is immune to the auto-prune and 409s on
    a hand DELETE. If it ever fires, the branch cannot be merged — there is no base, so a
    three-way diff would silently degrade into "take everything from the branch", which is
    a data-loss bug wearing a merge's clothes. Refuse instead.
    """


class StaleRootError(Exception):
    """The root board moved since the diff was taken (route → 409).

    Carries both fingerprints so the client can tell "somebody else merged" from "I sent
    a garbage token".
    """

    def __init__(self, expected: str, actual: str) -> None:
        super().__init__("root project changed since the diff was computed")
        self.expected = expected
        self.actual = actual


class MergeRoutingError(Exception):
    """An entry reached the writer with no branch-side row (a BUG in this module → 500).

    Unreachable by construction: :meth:`StoryboardMergeService.merge` routes on the
    BRANCH SIDE'S ABSENCE, so every entry handed to ``_panel_rows`` has a row to write.
    It exists because the code that used to sit where it is raised ``continue``d instead
    — and that silent skip was DATA LOSS: the key was already in ``applied`` (the API
    reported success), the panel was neither written nor deleted, and ``_shot_key_claims``
    had already RELEASED its shot_key to somebody else. A crash is the correct failure for
    an invariant this file owns.
    """

    def __init__(self, key: UUID) -> None:
        super().__init__(f"panel {key} was routed to the writer with no branch row")
        self.key = key


class UnresolvedConflictError(Exception):
    """A ``conflict`` entry has no explicit resolution (route → 422).

    NOT a 409: nothing about the server's state refuses the request — the REQUEST is
    incomplete. Choosing a side is precisely the decision the human opened the diff to
    make, and defaulting it (either way) silently discards somebody's work.
    """

    def __init__(self, keys: list[UUID], missing: dict[UUID, list[str]] | None = None) -> None:
        super().__init__(f"{len(keys)} conflicting panel(s) need an explicit resolution")
        self.keys = keys
        # For a partially field-resolved conflict: key → the conflicting fields still
        # without a branch/main pick. Empty for a wholly-unresolved conflict.
        self.missing = missing or {}


class FieldResolutionTargetError(Exception):
    """A ``field_resolutions`` entry targets a key that is not a field-resolvable panel
    conflict (route → 422).

    Per-field resolution only means something on a panel that changed in BOTH lanes whose
    disagreements are all plain content columns. Aimed at a non-conflict entry, a scene/asset
    key, a delete-vs-edit conflict, or a conflict whose conflicting fields include a fenced
    lineage/shot_key/position member, there is no coherent per-field merge to perform — that
    is a malformed request, not a silent no-op. Such a conflict must use a whole-row verdict.
    """

    def __init__(self, keys: list[UUID]) -> None:
        super().__init__(
            f"{len(keys)} field_resolution(s) do not target a field-resolvable conflict"
        )
        self.keys = keys


class KeepBothTargetError(Exception):
    """A ``keep_both`` resolution targets a key that is not a resolvable panel conflict
    (route → 422).

    ``keep_both`` keeps main's row AND re-materialises the branch's conflicting row as a
    new panel — it only means anything on a panel that changed in BOTH lanes. Aimed at a
    non-conflict entry, or at a scene/asset key (which never produces a resolvable entry),
    there is no second version to keep. That is a malformed request, not a silent no-op.
    """

    def __init__(self, keys: list[UUID]) -> None:
        super().__init__(f"{len(keys)} keep_both resolution(s) do not target a panel conflict")
        self.keys = keys


class NotSiblingsError(Exception):
    """A sibling merge names a source and target that are not siblings (route → 422).

    A sibling merge (branch A into branch B) requires both to be branches of the SAME parent
    — they share ``root_project_id``, which is what makes the parent's ``branch_base`` their
    common ancestor. A target that is the source itself, or a branch of a different parent,
    has no shared base to three-way against. That is a malformed request, not a 409.
    """

    def __init__(self, source_id: UUID, target_id: UUID) -> None:
        super().__init__(f"branches {source_id} and {target_id} are not siblings")
        self.source_id = source_id
        self.target_id = target_id


class SharedBaseMissingError(Exception):
    """The siblings have no shared ``parent_base_snapshot_id`` to merge against (route → 409).

    Every branch forked after migration 0057 pins one; only branches predating the backfill
    can lack it. Without a common base a three-way would silently degrade into "take
    everything from the source", which is data loss wearing a merge's clothes — refuse.
    """

    def __init__(self, source_id: UUID) -> None:
        super().__init__(f"branch {source_id} has no shared parent base for a sibling merge")
        self.source_id = source_id


# The panel columns whose content merge compares and carries. Deliberately EXCLUDED:
# ``id`` / ``project_id`` / ``storyboard_id`` (per-project identity), ``origin_id`` (the
# key itself), ``created_at`` / ``updated_at`` (bookkeeping), ``scene_id`` +
# ``characters`` / ``props`` / ``location_asset_id`` (raw ids differ per project — they are
# compared as LINEAGE keys, see ``_view``), and ``status`` / ``approved_at`` /
# ``approved_by`` / ``revision`` (the root's governance + its own edit counter — see the
# module docstring).
_CONTENT_FIELDS: tuple[str, ...] = (
    "shot_key",
    "shot_number",
    "panel_index",
    "position",
    "action_description",
    "dialogue",
    "vo_text",
    "sfx_cue",
    "music_cue",
    "image_ref",
    "media_ref",
    "visual_type",
    "shot_size",
    "composition_notes",
    "lighting_mood",
    "art_style",
    "aspect_ratio",
    "angle",
    "movement",
    "movement_path",
    "lens",
    "duration_seconds",
    "transition_out",
    "gen",
    "source",
)

# ── per-field conflict resolution ────────────────────────────────────────────
# A conflict may be resolved branch-vs-main PER FIELD instead of whole-row, but only for
# plain scalar content columns. ``shot_key`` (project-unique + keep_both machinery) and
# ``position`` (board-global order) are special-cased in the apply and are NEVER per-field
# mergeable. Lineage keys (scene/characters/props/location) are auto-fenced: they are not in
# ``_CONTENT_FIELDS`` and the apply overlay only iterates ``_CONTENT_FIELDS``, so they can
# only ever merge whole-unit (see ``_field_resolvable``).
_FIELD_MERGE_FENCED: frozenset[str] = frozenset({"shot_key", "position"})
FIELD_MERGEABLE: frozenset[str] = frozenset(_CONTENT_FIELDS).difference(_FIELD_MERGE_FENCED)
# The non-column lineage keys ``view()`` folds into a panel's comparable payload.
_LINEAGE_KEYS: frozenset[str] = frozenset({"scene", "characters", "props", "location"})

_SCENE_FIELDS: tuple[str, ...] = (
    "name",
    "int_ext",
    "location",
    "time_of_day",
    "description",
    "position",
)

_ASSET_FIELDS: tuple[str, ...] = (
    "kind",
    "name",
    "canonical_descriptor",
    "reference_images",
    "seed",
    "style_token",
    "lora_path",
    "attributes",
    "position",
)


def _key(row: dict[str, Any]) -> UUID:
    """A row's LINEAGE key: ``origin_id`` if it has one, else its own ``id``.

    The one function that makes the three sides comparable. A root row has no lineage of
    its own (it IS the origin), and the fork stamped its id onto the child's copy — so
    both sides land on the same key. Panel-created-in-branch has no lineage either, so it
    keys to its own child-side id, which by construction exists on no other side: an
    ``added``.
    """
    origin = row.get("origin_id")
    return UUID(str(origin)) if origin else UUID(str(row["id"]))


@dataclass
class _Board:
    """One side of the three-way diff, indexed by lineage key.

    ``panels`` / ``scenes`` / ``assets`` map key → the raw row dict (JSON-safe, exactly as
    ``StoryboardSnapshotService.serialize`` writes it, so the live sides and the snapshot
    side are literally the same shape). ``roles`` maps a panel key → ``role`` → the ordered
    asset LINEAGE keys it references — reconstructed from the ``storyboard_panel_assets``
    join rows, which is the only place cast/props/location live since migration 0051.
    """

    panels: dict[UUID, dict[str, Any]] = field(default_factory=dict)
    scenes: dict[UUID, dict[str, Any]] = field(default_factory=dict)
    assets: dict[UUID, dict[str, Any]] = field(default_factory=dict)
    roles: dict[UUID, dict[str, list[UUID]]] = field(default_factory=dict)
    # Reverse maps for the apply step: lineage key → the row id ON THIS SIDE.
    panel_ids: dict[UUID, UUID] = field(default_factory=dict)
    scene_ids: dict[UUID, UUID] = field(default_factory=dict)
    asset_ids: dict[UUID, UUID] = field(default_factory=dict)
    # INVERSE of ``scene_ids`` (scene row id → lineage key), built ONCE by
    # ``_build_board``. ``scene_key_of`` is called once per panel inside ``view()`` (and
    # ``view()`` once per panel inside ``fingerprint()``/``_entries``) plus in
    # ``_ensure_scenes`` / ``_panel_rows`` — a linear scan of ``scene_ids`` per call made the
    # whole diff O(panels * scenes). A dict lookup makes ``scene_key_of`` O(1) with
    # byte-identical results: it is the exact inverse of the map the old scan searched.
    _scene_key_by_id: dict[UUID, UUID] = field(default_factory=dict)

    def scene_key_of(self, panel: dict[str, Any]) -> UUID | None:
        """The panel's scene as a LINEAGE key (None-safe — an unscened panel stays so).

        O(1) lookup into ``_scene_key_by_id`` (the inverse of ``scene_ids``). Semantics are
        identical to the former linear scan: a falsy ``scene_id`` → ``None``; a ``scene_id``
        absent from ``scene_ids``' values → ``None`` (``dict.get`` default), exactly the
        scan's ``return None`` fall-through; otherwise the matching lineage key.
        """
        scene_id = panel.get("scene_id")
        if not scene_id:
            return None
        return self._scene_key_by_id.get(UUID(str(scene_id)))

    def view(self, key: UUID) -> dict[str, Any]:
        """The panel's NORMALISED, comparable payload — the unit of the whole diff.

        Raw ids are useless across projects, so ``scene`` and the three asset roles are
        expressed as lineage keys. Two panels with identical content in two projects
        therefore produce IDENTICAL views, which is the entire point: without this,
        every forked panel would read as modified.
        """
        panel = self.panels[key]
        out: dict[str, Any] = {f: panel.get(f) for f in _CONTENT_FIELDS}
        scene = self.scene_key_of(panel)
        out["scene"] = str(scene) if scene else None
        roles = self.roles.get(key, {})
        out["characters"] = [str(a) for a in roles.get("character", [])]
        out["props"] = [str(a) for a in roles.get("prop", [])]
        location = roles.get("location") or []
        out["location"] = str(location[0]) if location else None
        return out

    def fingerprint(self) -> str:
        """Content hash of the WHOLE board (panels + scenes + assets), lineage-keyed.

        Scenes and assets are in it because merge can materialise them onto the root; a
        panels-only hash would let a concurrent scene rename slip through the 409 gate.
        """
        canonical = {
            "panels": {str(k): self.view(k) for k in sorted(self.panels, key=lambda u: str(u))},
            "scenes": {
                str(k): {f: self.scenes[k].get(f) for f in _SCENE_FIELDS}
                for k in sorted(self.scenes, key=lambda u: str(u))
            },
            "assets": {
                str(k): {f: self.assets[k].get(f) for f in _ASSET_FIELDS}
                for k in sorted(self.assets, key=lambda u: str(u))
            },
        }
        blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _build_board(
    *,
    panels: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    assets: list[dict[str, Any]],
    panel_assets: list[dict[str, Any]],
) -> _Board:
    """Index one side's raw rows by lineage key. Shape-identical for a snapshot payload
    and for a live read — which is exactly why both go through here."""
    board = _Board()
    panel_id_to_key: dict[UUID, UUID] = {}
    asset_id_to_key: dict[UUID, UUID] = {}

    for row in scenes:
        key = _key(row)
        board.scenes[key] = row
        board.scene_ids[key] = UUID(str(row["id"]))
    for row in assets:
        key = _key(row)
        board.assets[key] = row
        row_id = UUID(str(row["id"]))
        board.asset_ids[key] = row_id
        asset_id_to_key[row_id] = key
    for row in panels:
        key = _key(row)
        board.panels[key] = row
        row_id = UUID(str(row["id"]))
        board.panel_ids[key] = row_id
        panel_id_to_key[row_id] = key

    # Join rows carry no ``origin_id`` of their own — a join row's merge identity is a
    # pure function of ``(panel.origin_id, asset.origin_id, role)``, re-derived HERE.
    # Ordered by ``position`` so ``characters``/``props`` compare as ORDERED lists (the
    # author's cast order is content, not incidental).
    ordered: dict[UUID, dict[str, list[tuple[int, UUID]]]] = {}
    for row in panel_assets:
        pid = UUID(str(row["panel_id"]))
        aid = UUID(str(row["asset_id"]))
        pkey, akey = panel_id_to_key.get(pid), asset_id_to_key.get(aid)
        if pkey is None or akey is None:  # pragma: no cover — FK guarantees both
            continue
        ordered.setdefault(pkey, {}).setdefault(str(row["role"]), []).append(
            (int(row["position"]), akey)
        )
    for pkey, roles in ordered.items():
        board.roles[pkey] = {
            role: [akey for _pos, akey in sorted(entries)] for role, entries in roles.items()
        }

    # Inverse of ``scene_ids`` for O(1) ``scene_key_of``. Built from the FINAL ``scene_ids``
    # so it reflects last-write-wins exactly as the old scan (which searched ``scene_ids``)
    # did; row ids are PKs so the values are unique and the inversion is lossless.
    board._scene_key_by_id = {row_id: key for key, row_id in board.scene_ids.items()}
    return board


@dataclass
class _Entry:
    """A diff entry before it becomes a Pydantic ``DiffEntry`` (the route maps it)."""

    key: UUID
    kind: str
    shot_key: str | None
    base: dict[str, Any] | None
    main: dict[str, Any] | None
    branch: dict[str, Any] | None
    changed_fields: list[str]
    main_changed_fields: list[str]


@dataclass
class _Diff:
    branch_id: UUID
    root_project_id: UUID
    child_project_id: UUID
    status: str
    base_snapshot_id: UUID
    root_fingerprint: str
    entries: list[_Entry]

    @property
    def conflict_count(self) -> int:
        return sum(1 for e in self.entries if e.kind == "conflict")


def _changed(left: dict[str, Any] | None, right: dict[str, Any] | None) -> list[str]:
    """Field names that differ between two views (``[]`` when either side is absent —
    presence itself is the difference, and it is carried by the entry's ``kind``)."""
    if left is None or right is None:
        return []
    return sorted(f for f in left if left[f] != right[f])


def _conflicting_fields(entry: _Entry) -> set[str]:
    """The fields a conflict genuinely disagrees on — changed on BOTH sides vs base.

    Empty for a delete-vs-edit conflict (one side's view is ``None`` → ``_changed`` returns
    ``[]``), which is exactly why an empty intersection must NEVER be treated as
    field-resolvable: it would let an empty/junk ``field_resolutions`` vacuously "cover"
    the conflict and route it to a silent delete.
    """
    return set(entry.changed_fields) & set(entry.main_changed_fields)


def _field_resolvable(entry: _Entry) -> bool:
    """Whether a conflict may be resolved field-by-field (branch-vs-main per column).

    True iff BOTH sides still have the panel (never a delete-vs-edit), there is at least one
    genuinely-conflicting field, EVERY conflicting field is a plain content column (a
    lineage/shot_key/position member poisons the whole conflict to whole-row), AND main made
    no edit the content overlay could not carry per-field. The overlay carries only the
    ``FIELD_MERGEABLE`` columns; ``shot_key``/``position`` and the lineage keys
    (scene/characters/props/location) are re-sourced UNCONDITIONALLY from the branch in the
    apply, so a main-ONLY change to any of them would be silently reverted. Such a conflict is
    fenced to a whole-row verdict. Otherwise per-field is a malformed target.
    """
    if entry.kind != "conflict" or entry.branch is None or entry.main is None:
        return False
    conflicting = _conflicting_fields(entry)
    if not conflicting or not conflicting <= FIELD_MERGEABLE:
        return False
    # Fields the content overlay cannot carry per-field (branch-sourced regardless): a
    # main-ONLY change to any of them would be reverted, so it must go whole-row.
    branch_sourced = _LINEAGE_KEYS | _FIELD_MERGE_FENCED
    main_only = set(entry.main_changed_fields) - set(entry.changed_fields)
    return not (main_only & branch_sourced)


def _auto_mergeable(entry: _Entry) -> bool:
    """Whether a two-sided divergence merges CLEANLY with no human input — branch's changed
    fields overlay, main's changed fields are kept — because the two sides touched DISJOINT
    fields (no field changed on BOTH sides vs base).

    This is the same "non-conflicting hunks auto-merge" a line-based VCS does: main changed
    ``duration`` while the branch changed ``caption`` is not a conflict, it is both edits.
    Only a field changed on both sides is a genuine disagreement a human must arbitrate.

    Intentionally does NOT gate on ``entry.kind`` — it is called BOTH while ``_entries`` is
    still deciding the kind (to demote a would-be conflict to ``modified``) AND in the apply
    (to drive the field-wise overlay). Returns False unless:

    * both sides still have the panel (never a delete-vs-edit — a side's view is ``None``),
    * BOTH sides changed something vs base (a one-sided change is ``modified``, not this),
    * NO field changed on both sides (a real conflict → per-field or whole-row), AND
    * every main-ONLY edit is a field the content overlay can carry. ``shot_key``/``position``
      and the lineage keys (scene/characters/props/location) are re-sourced from the BRANCH
      unconditionally in the apply, so a main-only change to any of them would be silently
      reverted — such a divergence stays a whole-row conflict.
    """
    if entry.branch is None or entry.main is None or entry.base is None:
        return False
    changed = set(entry.changed_fields)
    main_changed = set(entry.main_changed_fields)
    if not changed or not main_changed:
        return False
    if changed & main_changed:  # a field changed on BOTH sides → a genuine conflict
        return False
    branch_sourced = _LINEAGE_KEYS | _FIELD_MERGE_FENCED
    return not ((main_changed - changed) & branch_sourced)


class StoryboardMergeService:
    """``StoryboardMergeService(session)`` — the request session; methods take
    ``tenant_id`` (the ``StoryboardBranchService`` / ``StoryboardSnapshotService`` pattern).
    """

    def __init__(self, session: AsyncSession, *, max_auto_per_project: int | None = None) -> None:
        self.session = session
        # Passed straight through to the ``StoryboardSnapshotService`` this service mints
        # its pre-merge auto with. ``None`` → that service reads the operational cap from
        # settings. Injectable for the same reason it is there: a test pins a small cap
        # without touching process env.
        self._max_auto_per_project = max_auto_per_project

    # ─── loading the three sides ─────────────────────────────────────────────
    async def _live_board(self, project_id: UUID, *, tenant_id: UUID) -> _Board:
        """The live board of a project, in the SAME dict shape as a snapshot payload —
        so ``_build_board`` is the one indexer for all three sides."""
        panels = await StoryboardPanelRepository(self.session).list_for_project(
            project_id, tenant_id=tenant_id
        )
        scenes = await StoryboardSceneRepository(self.session).list_for_project(
            project_id, tenant_id=tenant_id
        )
        assets = await StoryboardAssetRepository(self.session).list_for_project(
            project_id, tenant_id=tenant_id
        )
        # ``panel.panel_assets`` is lazy="selectin" — already eager-loaded by
        # ``list_for_project``, so this is not a query per panel.
        return _build_board(
            panels=[_row_to_dict(p) for p in panels],
            scenes=[_row_to_dict(s) for s in scenes],
            assets=[_row_to_dict(a) for a in assets],
            panel_assets=[_row_to_dict(pa) for p in panels for pa in p.panel_assets],
        )

    @staticmethod
    def _payload_board(payload: dict[str, Any]) -> _Board:
        """The ``branch_base`` snapshot payload → a board.

        ``panel_assets`` ABSENT means a PRE-0051 payload, whose panel dicts still carry the
        legacy ``characters``/``props``/``location_asset_id`` columns. No ``branch_base``
        snapshot can predate 0051 (the kind did not exist until 0052), so this cannot fire
        for a real branch — but defaulting to ``[]`` rather than reconstructing would
        silently report every cast assignment as "added in branch", so the guard stays and
        is explicit rather than accidental.
        """
        return _build_board(
            panels=payload.get("panels", []),
            scenes=payload.get("scenes", []),
            assets=payload.get("assets", []),
            panel_assets=payload.get("panel_assets") or [],
        )

    async def _load(
        self, branch_id: UUID, *, tenant_id: UUID
    ) -> tuple[StoryboardBranch, _Board, _Board, _Board]:
        """``(branch, base, main, branch_board)`` — tenant-scoped, or raise."""
        branch = await StoryboardBranchRepository(self.session).get(branch_id, tenant_id=tenant_id)
        if branch is None:
            raise BranchNotFoundError(str(branch_id))

        base_row = await StoryboardSnapshotRepository(self.session).get(
            branch.base_snapshot_id, tenant_id=tenant_id
        )
        if base_row is None:
            raise BaseSnapshotMissingError(str(branch.base_snapshot_id))

        base = self._payload_board(base_row.payload or {})
        main = await self._live_board(branch.root_project_id, tenant_id=tenant_id)
        child = await self._live_board(branch.child_project_id, tenant_id=tenant_id)
        return branch, base, main, child

    async def _lock_root(self, branch_id: UUID, *, tenant_id: UUID) -> UUID:
        """Row-lock the ROOT project of ``branch_id`` and return its id, or raise 404.

        ``SELECT … FOR UPDATE OF projects`` — the serialisation point for the fingerprint
        gate (see the module docstring). Deliberately the ROOT ``projects`` row and not the
        branch row: two DIFFERENT branches of the same root race for the same board, so a
        per-branch lock would let them straight through. It is a plain row lock, not an
        advisory one, so it needs no separate key space and dies with the transaction even
        if the process does.

        Tenant-scoped through the same ``root_project_id -> projects.tenant_id`` hop
        ``StoryboardBranchRepository`` uses (and RLS enforces in prod). Column-only select
        on purpose: it takes the lock WITHOUT populating this session's identity map, so
        the ORM read in ``_load`` — which runs after the lock and must see the winner's
        committed state, never a pre-lock cache of it — is the first fetch of every row.
        """
        root_id = (
            await self.session.execute(
                select(StoryboardBranch.root_project_id)
                .join(Project, StoryboardBranch.root_project_id == Project.id)
                .where(StoryboardBranch.id == branch_id, Project.tenant_id == tenant_id)
                .with_for_update(of=Project)
            )
        ).scalar_one_or_none()
        if root_id is None:
            raise BranchNotFoundError(str(branch_id))
        return UUID(str(root_id))

    async def _lock_root_board(self, root_id: UUID) -> None:
        """Row-lock the root's panels/scenes/assets ``FOR UPDATE`` so a concurrent panel
        PATCH's UPDATE blocks until this merge commits, then re-applies on top under
        READ COMMITTED (no silent revert of ``status``/``approved_*``/``position``).

        The ``projects``-row lock in :meth:`_lock_root` only serialises merge-vs-merge; a
        PATCH never touches the ``projects`` row, so it slips straight past that lock,
        commits between this merge's read and its upsert, and the upsert's
        ``ON CONFLICT DO UPDATE SET <all cols>`` reverts it — a lost update the fingerprint
        gate never sees (it compares this merge's OWN stale snapshot). Taking ``FOR UPDATE``
        on the board rows themselves is what the PATCH's UPDATE actually contends on.

        Column-only selects — take the locks WITHOUT populating the identity map, exactly
        like :meth:`_lock_root` — so ``_load``'s ORM read is still the first fetch of the
        winner's committed state. ``ORDER BY id`` gives a deterministic acquisition order
        (no lock-ordering deadlock between two mergers of the same root).

        RESIDUAL GAP (documented, not fixed here): a pure cast/props/location PATCH edits
        only ``storyboard_panel_assets`` join rows and may emit NO UPDATE on the panel row,
        so this panel-row ``FOR UPDATE`` won't block it. Merge rebuilds those join rows
        wholesale in :meth:`_apply` (delete-then-insert of the touched panels' joins), so a
        concurrent join-only PATCH is a narrow, low-harm edge, not a silent-revert of
        governed state. ``root_id`` already fully identifies the rows and is tenant-verified
        by :meth:`_lock_root` upstream (RLS scopes it in prod), so this takes no tenant_id.
        """
        for model in (StoryboardPanel, StoryboardScene, StoryboardAsset):
            await self.session.execute(
                select(model.id)
                .where(model.project_id == root_id)
                .order_by(model.id)
                .with_for_update()
            )

    # ─── the three-way diff ──────────────────────────────────────────────────
    async def diff(self, branch_id: UUID, *, tenant_id: UUID) -> _Diff:
        """Three-way diff of ``base`` (fork point) / ``main`` (root now) / ``branch`` (child
        now), keyed by lineage. Works for a ``merged``/``abandoned`` branch too — reading a
        closed branch's diff is how you audit what it did; only APPLYING one is refused."""
        branch, base, main, child = await self._load(branch_id, tenant_id=tenant_id)
        entries = self._entries(base, main, child)
        return _Diff(
            branch_id=branch.id,
            root_project_id=branch.root_project_id,
            child_project_id=branch.child_project_id,
            status=branch.status,
            base_snapshot_id=branch.base_snapshot_id,
            root_fingerprint=main.fingerprint(),
            entries=entries,
        )

    async def _lock_project_row(self, project_id: UUID, *, tenant_id: UUID) -> None:
        """Row-lock a ``projects`` row ``FOR UPDATE`` — the merge-vs-merge serialisation point
        for a SIBLING merge, where the board being written is the target sibling's CHILD
        project (not a branch's root). Column-only select so it takes the lock WITHOUT
        populating the identity map; the ORM reads in :meth:`_load_sibling` run after it and
        must see the winner's committed state. Tenant-scoped, like :meth:`_lock_root`."""
        await self.session.execute(
            select(Project.id)
            .where(Project.id == project_id, Project.tenant_id == tenant_id)
            .with_for_update(of=Project)
        )

    async def _load_sibling(
        self, source_branch_id: UUID, target_branch_id: UUID, *, tenant_id: UUID
    ) -> tuple[StoryboardBranch, StoryboardBranch, UUID, _Board, _Board, _Board]:
        """``(source, target, base_snapshot_id, base, target_board, source_board)`` for a
        SIBLING merge, or raise. ``base`` is the parent's shared ``branch_base`` (the common
        ancestor of both siblings); ``target_board`` is the board being merged INTO (the main
        side); ``source_board`` is the board being merged FROM (the branch side). Both siblings
        share ``parent_base_snapshot_id`` by construction, so the source's is authoritative."""
        repo = StoryboardBranchRepository(self.session)
        source = await repo.get(source_branch_id, tenant_id=tenant_id)
        if source is None:
            raise BranchNotFoundError(str(source_branch_id))
        target = await repo.get(target_branch_id, tenant_id=tenant_id)
        if target is None:
            raise BranchNotFoundError(str(target_branch_id))
        if source.id == target.id or source.root_project_id != target.root_project_id:
            raise NotSiblingsError(source.id, target.id)
        base_snapshot_id = source.parent_base_snapshot_id
        if base_snapshot_id is None:
            raise SharedBaseMissingError(source.id)
        base_row = await StoryboardSnapshotRepository(self.session).get(
            base_snapshot_id, tenant_id=tenant_id
        )
        if base_row is None:
            raise SharedBaseMissingError(source.id)
        base = self._payload_board(base_row.payload or {})
        target_board = await self._live_board(target.child_project_id, tenant_id=tenant_id)
        source_board = await self._live_board(source.child_project_id, tenant_id=tenant_id)
        return source, target, base_snapshot_id, base, target_board, source_board

    async def diff_sibling(
        self, source_branch_id: UUID, target_branch_id: UUID, *, tenant_id: UUID
    ) -> _Diff:
        """Three-way diff for a SIBLING merge (branch A into branch B): ``base`` = the parent's
        shared ancestor, ``main`` = the TARGET sibling's board now, ``branch`` = the SOURCE
        sibling's board now. Symmetric — the base is common to both — unlike a branch->root
        diff whose base is the source's own fork point. ``root_fingerprint`` is the TARGET
        board's, the compare-and-set token :meth:`merge_into_sibling` takes back."""
        (
            source,
            target,
            base_snapshot_id,
            base,
            target_board,
            source_board,
        ) = await self._load_sibling(source_branch_id, target_branch_id, tenant_id=tenant_id)
        entries = self._entries(base, target_board, source_board)
        return _Diff(
            branch_id=source.id,
            root_project_id=target.child_project_id,
            child_project_id=source.child_project_id,
            status=source.status,
            base_snapshot_id=base_snapshot_id,
            root_fingerprint=target_board.fingerprint(),
            entries=entries,
        )

    @staticmethod
    def _entries(base: _Board, main: _Board, child: _Board) -> list[_Entry]:
        """The entry matrix. BRANCH-CENTRIC: a panel only main touched is NOT an entry —
        there is nothing to merge from it (it still moves the fingerprint, which is how the
        409 gate protects it).

        * ``added``    — present in branch, no lineage in base
        * ``modified`` — in base and branch, branch content differs from base
        * ``deleted``  — in base, gone from branch
        * ``conflict`` — branch changed vs base AND main changed vs base, and the two sides
          disagree. Note this catches the asymmetric cases too: branch-deleted +
          main-modified, and branch-modified + main-deleted, are both conflicts, because
          the resulting states differ.
        """
        entries: list[_Entry] = []
        for key in sorted(
            set(base.panels) | set(main.panels) | set(child.panels), key=lambda u: str(u)
        ):
            base_view = base.view(key) if key in base.panels else None
            main_view = main.view(key) if key in main.panels else None
            child_view = child.view(key) if key in child.panels else None

            if child_view == base_view:
                continue  # the branch did nothing to this panel — nothing to merge

            if key not in base.panels:
                kind = "added"
            elif key not in child.panels:
                kind = "deleted"
            else:
                kind = "modified"

            main_changed = main_view != base_view
            if main_changed and child_view != main_view:
                kind = "conflict"

            source = child.panels.get(key) or main.panels.get(key) or base.panels.get(key) or {}
            entry = _Entry(
                key=key,
                kind=kind,
                shot_key=source.get("shot_key"),
                base=base_view,
                main=main_view,
                branch=child_view,
                changed_fields=_changed(child_view, base_view),
                main_changed_fields=_changed(main_view, base_view),
            )
            # A would-be conflict whose two sides touched DISJOINT fields — nothing changed on
            # BOTH sides — and whose main-only edits the content overlay can carry is NOT a
            # conflict: it merges cleanly (branch's fields overlay, main's are kept). Demote it
            # to 'modified' so it needs no human pick; :meth:`_build_panel_rows` combines both
            # sides field-wise. A genuine two-sided disagreement stays 'conflict'.
            if entry.kind == "conflict" and _auto_mergeable(entry):
                entry.kind = "modified"
            entries.append(entry)
        return entries

    @staticmethod
    def _route_entries(
        entries: list[_Entry], resolutions: dict[UUID, str]
    ) -> tuple[list[UUID], list[UUID], list[_Entry], list[_Entry], set[UUID]]:
        """Bucket diff entries by their resolution verdict → ``(applied, skipped, to_write,
        to_delete, keep_both)``. Non-conflict entries default to ``branch`` (the author asked
        to merge their branch); conflicts are guaranteed resolved by ``merge`` before this.

        Routes on the BRANCH SIDE'S ABSENCE, never on ``kind``. "The branch deleted this
        panel" is spelled ``kind='deleted'`` ONLY while main left it alone — the moment main
        also edited it, ``_entries`` overrides the kind to ``conflict``, and resolving THAT to
        "branch" still means "delete it". Routing on the string would send it to the writer,
        which has no branch row to write, drop it, and report it in ``applied`` anyway — the
        panel silently surviving the merge that asked to remove it. ``entry.branch is None`` is
        the real predicate and covers both spellings.

        ``keep_both`` (conflicts only) keeps main's row AND re-materialises the branch's row as
        a NEW panel, so the branch entry joins ``to_write`` AND ``keep_both``. A keep_both whose
        branch side is absent (a delete-vs-edit conflict) has nothing to carry over → it
        degenerates to "keep main": mint nothing, main's row already stands.
        """
        applied: list[UUID] = []
        skipped: list[UUID] = []
        to_write: list[_Entry] = []
        to_delete: list[_Entry] = []
        keep_both: set[UUID] = set()
        for entry in entries:
            verdict = resolutions.get(entry.key, "branch")
            if verdict == "keep_both":
                if entry.branch is not None:
                    keep_both.add(entry.key)
                    to_write.append(entry)
                    applied.append(entry.key)
                else:
                    skipped.append(entry.key)
                continue
            if verdict != "branch":
                skipped.append(entry.key)
                continue
            (to_delete if entry.branch is None else to_write).append(entry)
            applied.append(entry.key)
        return applied, skipped, to_write, to_delete, keep_both

    @staticmethod
    def _validate_resolutions(
        entries: list[_Entry],
        resolutions: dict[UUID, str],
        field_resolutions: Mapping[UUID, Mapping[str, str]],
    ) -> None:
        """Reject a malformed or incomplete resolution set, BEFORE any write. Three 422s:

        * ``field_resolutions`` aimed at a key that is not a FIELD-RESOLVABLE conflict — a
          non-conflict / scene / asset key, a delete-vs-edit conflict, or a conflict poisoned
          by a fenced lineage/shot_key/position member → ``FieldResolutionTargetError``.
          Checked FIRST so a malformed target is never mistaken for a merely-incomplete one.
        * a conflict left without a whole-row verdict AND without a ``field_resolutions``
          covering EXACTLY its conflicting fields → ``UnresolvedConflictError`` (carrying the
          still-missing fields); applying it would silently default them to branch, discarding
          main's side without a choice.
        * ``keep_both`` on a non-conflict key → ``KeepBothTargetError``.
        """
        entries_by_key = {e.key: e for e in entries}
        conflict_keys = {e.key for e in entries if e.kind == "conflict"}

        bad_field_targets = [
            k
            for k in field_resolutions
            if k not in entries_by_key or not _field_resolvable(entries_by_key[k])
        ]
        if bad_field_targets:
            raise FieldResolutionTargetError(bad_field_targets)

        unresolved: list[UUID] = []
        missing: dict[UUID, list[str]] = {}
        for e in entries:
            if e.kind != "conflict" or e.key in resolutions:
                continue
            fr = field_resolutions.get(e.key)
            if fr is None:
                unresolved.append(e.key)
                continue
            conflicting = _conflicting_fields(e)  # non-empty: e passed ``_field_resolvable``
            if set(fr) != conflicting:
                unresolved.append(e.key)
                missing[e.key] = sorted(conflicting - set(fr))
        if unresolved:
            raise UnresolvedConflictError(unresolved, missing)

        misused_keep_both = [
            k for k, v in resolutions.items() if v == "keep_both" and k not in conflict_keys
        ]
        if misused_keep_both:
            raise KeepBothTargetError(misused_keep_both)

    # ─── the apply ───────────────────────────────────────────────────────────
    async def merge(
        self,
        branch_id: UUID,
        *,
        tenant_id: UUID,
        root_fingerprint: str,
        resolutions: dict[UUID, str] | None = None,
        field_resolutions: Mapping[UUID, Mapping[str, str]] | None = None,
        actor_subject: UUID | None = None,
    ) -> _MergeResult:
        """Apply the branch onto the root.

        Raises :class:`BranchNotFoundError` (404), :class:`BranchNotOpenError` (409),
        :class:`BaseSnapshotMissingError` (409), :class:`StaleRootError` (409) and
        :class:`UnresolvedConflictError` (422). Runs in the caller's transaction.
        """
        resolutions = resolutions or {}
        field_resolutions = field_resolutions or {}

        # 0) Serialise on the ROOT before READING anything. A lock taken after the read it
        # is meant to protect protects nothing: the whole point is that the second merge of
        # this root must observe the first one's COMMITTED board, so that its fingerprint
        # comes out stale and it 409s instead of clobbering. See the module docstring.
        # The projects-row lock only serialises merge-vs-merge; the board-row locks close
        # merge-vs-PATCH (a PATCH never touches the projects row) — both BEFORE any read.
        root_id = await self._lock_root(branch_id, tenant_id=tenant_id)
        await self._lock_root_board(root_id)

        branch, base, main, child = await self._load(branch_id, tenant_id=tenant_id)
        if branch.status != "open":
            raise BranchNotOpenError(branch.status)
        # Merging INTO an archived root would resurrect writes on a project the lifecycle
        # has closed (read-only). The root row is not loaded by ``_load`` (it builds boards,
        # not the Project), so read its status off the locked root id. → 409.
        root_status = (
            await self.session.execute(
                select(Project.status).where(Project.id == root_id, Project.tenant_id == tenant_id)
            )
        ).scalar_one_or_none()
        if root_status == "archived":
            raise ProjectArchivedError(str(root_id))
        # Bottom-up or not at all. ``mark_merged`` below is TERMINAL, so merging this
        # branch while it still has open children of its own would leave them merging into
        # a parent that can never carry them upward again — their work silently stranded,
        # with the API reporting success. Merge/abandon the children FIRST.
        await assert_no_open_children(self.session, branch, tenant_id=tenant_id)

        # ``root_id`` already holds this value from ``_lock_root`` above (same branch row);
        # keep only ``base_snapshot_id`` here.
        base_snapshot_id: UUID = branch.base_snapshot_id

        # 1) Optimistic concurrency FIRST — before a single write, including the safety
        # snapshot. A stale merge must leave the root byte-identical, autos included.
        actual = main.fingerprint()
        if actual != root_fingerprint:
            raise StaleRootError(expected=root_fingerprint, actual=actual)

        entries = self._entries(base, main, child)
        # Reject a malformed or incomplete resolution set BEFORE any write (all three modes
        # are 422s; see the helper). Extracted so ``merge`` stays under the complexity gate.
        self._validate_resolutions(entries, resolutions, field_resolutions)

        # 2) The pre-merge safety copy of the ROOT (kind='auto', exactly like restore), so
        # a merge is reversible by restoring this id. Minted BEFORE any mutation and inside
        # this transaction, so a later failure rolls it back with everything else.
        snapshots = StoryboardSnapshotService(
            self.session, max_auto_per_project=self._max_auto_per_project
        )
        auto = await snapshots.create_snapshot(
            root_id,
            tenant_id=tenant_id,
            kind="auto",
            note=f"auto before merge of branch {branch_id}",
            actor_subject=actor_subject,
        )
        auto_id = auto.id

        # 2b) …and cap them. A root is merged into repeatedly — once per branch, and a
        # branch can be re-merged after a 409/rework — so every merge minting an
        # unpruned auto grew the root's snapshot table by a whole graph blob forever.
        # ``restore`` capped its autos from day one; this path did not, which made the
        # configured cap a no-op for the busiest auto producer we have. Protect the id we
        # just minted: it is this merge's undo handle, and a cap of 1 would otherwise be
        # free to evict the very snapshot that makes the merge reversible.
        await snapshots.prune_autos(root_id, protect_id=auto_id)

        # Non-conflict entries default to ``branch`` — the author asked to merge their
        # branch. Conflicts have NO default (rejected above).
        # Bucket every entry by its verdict. Extracted so ``merge`` stays under the
        # complexity gate; the routing rules (and why they route on branch-absence, not the
        # kind string) live in ``_route_entries``.
        applied, skipped, to_write, to_delete, keep_both = self._route_entries(entries, resolutions)

        if to_write or to_delete:
            # A reorder or a shot_key rename produces a TRANSIENT duplicate on the
            # DEFERRABLE ``uq_storyboard_panels_project_shot`` mid-write that only settles
            # once the whole set is in. Same reason restore does this; ``INITIALLY
            # IMMEDIATE`` means every other write path is unaffected.
            await self.session.execute(text("SET CONSTRAINTS ALL DEFERRED"))
            await self._apply(
                root_id=root_id,
                tenant_id=tenant_id,
                base_snapshot_id=base_snapshot_id,
                branch_id=branch_id,
                main=main,
                child=child,
                to_write=to_write,
                to_delete=to_delete,
                keep_both=keep_both,
                field_resolutions=field_resolutions,
                actor_subject=actor_subject,
            )

        # 3) Close the branch. Re-fetched because ``_apply`` expunges the identity map
        # (its bulk DML bypasses the ORM, so every instance held here is stale).
        repo = StoryboardBranchRepository(self.session)
        fresh = await repo.get(branch_id, tenant_id=tenant_id)
        if fresh is None:  # pragma: no cover — it resolved moments ago in this transaction
            raise BranchNotFoundError(str(branch_id))
        await repo.mark_merged(fresh)

        # 3b) Archive the child project, exactly as ``abandon`` does. A merged branch's
        # content now lives on the root, so leaving the child live+editable would strand a
        # second, forkable copy of the merged work that no listing hides (``is_branch`` only
        # hides it from USER listings, not from the branch machinery). ``abandon`` refuses an
        # already-merged branch, so without this the child could never be archived at all.
        # Runs AFTER ``_apply`` (which expunged the identity map), so this is a fresh read.
        child_proj = (
            await self.session.execute(
                select(Project).where(
                    Project.id == fresh.child_project_id, Project.tenant_id == tenant_id
                )
            )
        ).scalar_one_or_none()
        if child_proj is not None:
            child_proj.status = "archived"
            child_proj.archived_at = func.now()

        after = await self._live_board(root_id, tenant_id=tenant_id)
        return {
            "branch_id": branch_id,
            "status": fresh.status,
            "merged_at": fresh.merged_at,
            "auto_snapshot_id": auto_id,
            "applied": applied,
            "skipped": skipped,
            "root_fingerprint": after.fingerprint(),
        }

    async def merge_into_sibling(
        self,
        source_branch_id: UUID,
        target_branch_id: UUID,
        *,
        tenant_id: UUID,
        target_fingerprint: str,
        resolutions: dict[UUID, str] | None = None,
        field_resolutions: Mapping[UUID, Mapping[str, str]] | None = None,
        actor_subject: UUID | None = None,
    ) -> _MergeResult:
        """Apply SOURCE branch's changes onto its SIBLING TARGET branch (A -> B), three-way
        against the parent's shared base.

        Unlike a branch->root merge this is a CROSS-POLLINATE, not a lifecycle close: BOTH
        branches stay ``open``, neither child is archived, and the source is NOT marked
        ``merged`` — its content never reached the root, it was copied to a sibling. Only the
        TARGET board is written (and auto-snapshotted for reversibility). The diff is
        BRANCH-CENTRIC on the SOURCE, so the target's OWN edits are preserved: a panel only the
        target touched is not an entry, and a panel both touched differently is a conflict.

        Raises :class:`BranchNotFoundError` (404), :class:`NotSiblingsError` (422),
        :class:`BranchNotOpenError` (409), :class:`SharedBaseMissingError` (409),
        :class:`StaleRootError` (409), :class:`UnresolvedConflictError` (422). Runs in the
        caller's transaction.
        """
        resolutions = resolutions or {}
        field_resolutions = field_resolutions or {}

        # Resolve the target's CHILD project id (the board written onto) column-only — no
        # identity map — then lock its projects row + board BEFORE any read, the same
        # merge-vs-merge / merge-vs-PATCH serialisation the branch->root path takes, but on the
        # target sibling. The real ORM reads happen in ``_load_sibling`` AFTER these locks.
        target_project_id = (
            await self.session.execute(
                select(StoryboardBranch.child_project_id)
                .join(Project, StoryboardBranch.root_project_id == Project.id)
                .where(StoryboardBranch.id == target_branch_id, Project.tenant_id == tenant_id)
            )
        ).scalar_one_or_none()
        if target_project_id is None:
            raise BranchNotFoundError(str(target_branch_id))
        target_project_id = UUID(str(target_project_id))
        await self._lock_project_row(target_project_id, tenant_id=tenant_id)
        await self._lock_root_board(target_project_id)

        (
            source,
            target,
            base_snapshot_id,
            base,
            target_board,
            source_board,
        ) = await self._load_sibling(source_branch_id, target_branch_id, tenant_id=tenant_id)
        # Both must be OPEN: merging FROM a closed source replays discarded/landed work, and
        # merging INTO a closed target writes onto a branch the lifecycle has finished with.
        if source.status != "open":
            raise BranchNotOpenError(source.status)
        if target.status != "open":
            raise BranchNotOpenError(target.status)

        # Optimistic concurrency FIRST — a stale merge leaves the target byte-identical.
        actual = target_board.fingerprint()
        if actual != target_fingerprint:
            raise StaleRootError(expected=target_fingerprint, actual=actual)

        entries = self._entries(base, target_board, source_board)
        self._validate_resolutions(entries, resolutions, field_resolutions)

        # Pre-merge safety copy of the TARGET (kind='auto'), so the sibling merge is reversible
        # by restoring this id. Minted before any mutation, inside this transaction.
        snapshots = StoryboardSnapshotService(
            self.session, max_auto_per_project=self._max_auto_per_project
        )
        auto = await snapshots.create_snapshot(
            target_project_id,
            tenant_id=tenant_id,
            kind="auto",
            note=f"auto before sibling merge of branch {source.id} into {target.id}",
            actor_subject=actor_subject,
        )
        auto_id = auto.id
        await snapshots.prune_autos(target_project_id, protect_id=auto_id)

        applied, skipped, to_write, to_delete, keep_both = self._route_entries(entries, resolutions)
        if to_write or to_delete:
            await self.session.execute(text("SET CONSTRAINTS ALL DEFERRED"))
            await self._apply(
                root_id=target_project_id,
                tenant_id=tenant_id,
                base_snapshot_id=base_snapshot_id,
                branch_id=source.id,
                main=target_board,
                child=source_board,
                to_write=to_write,
                to_delete=to_delete,
                keep_both=keep_both,
                field_resolutions=field_resolutions,
                actor_subject=actor_subject,
            )

        after = await self._live_board(target_project_id, tenant_id=tenant_id)
        return {
            "source_branch_id": source.id,
            "target_branch_id": target.id,
            "auto_snapshot_id": auto_id,
            "applied": applied,
            "skipped": skipped,
            "target_fingerprint": after.fingerprint(),
        }

    async def _apply(
        self,
        *,
        root_id: UUID,
        tenant_id: UUID,
        base_snapshot_id: UUID,
        branch_id: UUID,
        main: _Board,
        child: _Board,
        to_write: list[_Entry],
        to_delete: list[_Entry],
        keep_both: set[UUID],
        field_resolutions: Mapping[UUID, Mapping[str, str]],
        actor_subject: UUID | None,
    ) -> None:
        storyboard = await StoryboardRepository(self.session).ensure_for_project(
            root_id, tenant_id=tenant_id
        )
        if storyboard is None:  # pragma: no cover — the root was resolved by ``_load``
            raise BranchNotFoundError(str(branch_id))

        # Scene/asset counterparts the branch panels reference. A panel that gained a new
        # asset in the branch has NO root-side asset to point at, so the reference would
        # merge dangling (or drop the whole cast). Materialise the counterpart on the root
        # instead, stamping ``origin_id`` = the branch row's lineage key so a SECOND merge
        # of the same branch finds it rather than minting a duplicate.
        scene_map = await self._ensure_scenes(
            root_id=root_id, storyboard_id=storyboard.id, main=main, child=child, entries=to_write
        )
        asset_map = await self._ensure_assets(
            root_id=root_id, main=main, child=child, entries=to_write
        )

        # ``to_delete`` (every entry with NO branch-side row — a plain ``deleted``, or a
        # ``conflict`` where the branch deleted what main edited and the caller chose the
        # branch) is the ONE sanctioned delete: the author removed the panel and asked for
        # that to land. Its trail dies with it (CASCADE), which is correct — an approval
        # trail for a panel that no longer exists describes nothing. Every OTHER path
        # upserts (see the module docstring). An entry whose panel main ALSO deleted has no
        # ``main.panel_ids`` row and simply drops out here: nothing left to delete.
        deleted_ids = [main.panel_ids[e.key] for e in to_delete if e.key in main.panel_ids]
        if deleted_ids:
            await self.session.execute(
                delete(StoryboardPanel).where(StoryboardPanel.id.in_(deleted_ids))
            )

        rows, joins, touched = self._panel_rows(
            root_id=root_id,
            storyboard_id=storyboard.id,
            main=main,
            child=child,
            entries=to_write,
            deleted_ids=set(deleted_ids),
            keep_both=keep_both,
            field_resolutions=field_resolutions,
            scene_map=scene_map,
            asset_map=asset_map,
        )
        if rows:
            # UPSERT onto the root's EXISTING panel ids — the trail-preserving primitive
            # restore uses. Never delete-then-reinsert.
            await upsert_rows(self.session, StoryboardPanel, rows)
            # The join rows of the touched panels are REBUILT (they carry no lineage of
            # their own — their identity is derived). Delete-then-insert is safe here and
            # ONLY here: no event table hangs off them, and the two statements are executed
            # in order, so ``uq_storyboard_panel_assets_one_location`` (a partial unique
            # INDEX — ``SET CONSTRAINTS`` cannot defer it) never sees two locations at once.
            await self.session.execute(
                delete(StoryboardPanelAsset).where(
                    StoryboardPanelAsset.panel_id.in_([UUID(str(r["id"])) for r in rows])
                )
            )
            if joins:
                await self.session.execute(
                    insert(StoryboardPanelAsset).values(
                        [_coerce_row(StoryboardPanelAsset, j) for j in joins]
                    )
                )
        await self.session.flush()

        # The upserts above are bulk DML that bypasses the ORM, so any identity-mapped
        # instance still in this session holds PRE-merge values. Detach (expunge, NOT
        # expire — an expired attribute lazy-loads on next access, which is a MissingGreenlet
        # under async), so every later read re-hydrates from the merged rows. Safe: the
        # flush above already emitted every pending INSERT.
        self.session.expunge_all()

        # One ``merged`` trail event per touched panel, on the root's SURVIVING panel row —
        # which is exactly the row the upsert kept alive. Collected and written through the
        # BATCH ``append_all`` (one ``add_all`` + one ``flush``) instead of a flush per
        # panel: same rows, same fields, same order (``touched`` order → uuid7 order →
        # ``list_for_*`` sort order). Empty ``touched`` → no call → zero flushes, exactly as
        # the old loop's zero iterations did.
        detail = {"branch_id": str(branch_id), "base_snapshot_id": str(base_snapshot_id)}
        event_rows = [
            {
                "project_id": root_id,
                "panel_id": panel_id,
                "event_type": "merged",
                "revision": revision,
                "actor_subject_id": actor_subject,
                "detail": detail,
            }
            for panel_id, revision in touched
        ]
        if event_rows:
            await StoryboardPanelEventRepository(self.session).append_all(event_rows)

    async def _ensure_scenes(
        self,
        *,
        root_id: UUID,
        storyboard_id: UUID,
        main: _Board,
        child: _Board,
        entries: list[_Entry],
    ) -> dict[UUID, UUID]:
        """Lineage key → ROOT scene id, materialising the ones the root does not have."""
        mapping = dict(main.scene_ids)
        new_rows: list[dict[str, Any]] = []
        for entry in entries:
            panel = child.panels.get(entry.key)
            if panel is None:
                continue
            scene_key = child.scene_key_of(panel)
            if scene_key is None or scene_key in mapping:
                continue
            src = child.scenes[scene_key]
            row_id = uuid7()
            mapping[scene_key] = row_id
            new_rows.append(
                {
                    **src,
                    "id": row_id,
                    "project_id": root_id,
                    "storyboard_id": storyboard_id,
                    "origin_id": scene_key,
                }
            )
        if new_rows:
            await upsert_rows(self.session, StoryboardScene, new_rows)
        return mapping

    async def _ensure_assets(
        self, *, root_id: UUID, main: _Board, child: _Board, entries: list[_Entry]
    ) -> dict[UUID, UUID]:
        """Lineage key → ROOT asset id, materialising the ones the root does not have.

        Only assets a merged panel actually REFERENCES are materialised — an asset created
        in the branch but never used by any merged panel is branch-local scratch, and
        dragging it onto the root would pollute the root's cast list with rows nobody asked
        for. (It survives in the child project, which is archived, not destroyed.)
        """
        mapping = dict(main.asset_ids)
        new_rows: list[dict[str, Any]] = []
        for entry in entries:
            for role_ids in child.roles.get(entry.key, {}).values():
                for asset_key in role_ids:
                    if asset_key in mapping or asset_key not in child.assets:
                        continue
                    src = child.assets[asset_key]
                    row_id = uuid7()
                    mapping[asset_key] = row_id
                    new_rows.append(
                        {**src, "id": row_id, "project_id": root_id, "origin_id": asset_key}
                    )
        if new_rows:
            await upsert_rows(self.session, StoryboardAsset, new_rows)
        return mapping

    def _shot_key_claims(
        self,
        *,
        main: _Board,
        entries: list[_Entry],
        deleted_ids: set[UUID],
        keep_both: set[UUID],
    ) -> tuple[dict[str, UUID], int]:
        """Who still owns which shot_key once this merge settles, + the next free position.

        Final-state ownership, so the merged board satisfies
        ``uq_storyboard_panels_project_shot`` at COMMIT (the deferral tolerates a TRANSIENT
        duplicate, not a permanent one — and an IntegrityError at COMMIT is a 500 the user
        cannot act on).

        Only panels that SURVIVE UNTOUCHED hold a claim. A panel this merge is about to
        rewrite RELEASES its current shot_key first: otherwise a plain swap (clip-1 ⇄
        clip-2, the commonest reorder there is) would see the key it wants still held by its
        own counterpart and suffix BOTH panels into ``clip-2-2`` / ``clip-1-2`` — mangling a
        rename that is perfectly legal in the settled state.

        ``entries`` is the WRITE set ONLY — ``merge`` routes every branch-side absence to
        the DELETE set — so each key released here is re-claimed by the row ``_panel_rows``
        builds for it, and each key released by ``deleted_ids`` belongs to a row that is
        gone. A release with no counterpart write would be a permanent hole another panel
        could claim, i.e. a duplicate the DEFERRED unique constraint only discovers at
        COMMIT, as a 500 that aborts the entire merge.
        """
        # A ``keep_both`` entry does NOT rewrite main's row — it keeps it verbatim and mints
        # the branch's version alongside it. So main's kept row must STAY in ``taken`` holding
        # its shot_key: excluding it here is what makes the new panel collide and get suffixed
        # (``clip-N-2``) instead of being handed main's key while main's row still holds it —
        # a permanent duplicate the DEFERRED unique constraint would only catch at COMMIT (500).
        rewritten: set[UUID] = {
            main.panel_ids[e.key]
            for e in entries
            if e.key in main.panel_ids
            and main.panel_ids[e.key] not in deleted_ids
            and e.key not in keep_both
        }
        taken: dict[str, UUID] = {
            row["shot_key"]: main.panel_ids[key]
            for key, row in main.panels.items()
            if main.panel_ids[key] not in deleted_ids and main.panel_ids[key] not in rewritten
        }
        next_position = 1 + max(
            (int(row.get("position", 0)) for row in main.panels.values()), default=-1
        )
        return taken, next_position

    def _panel_rows(
        self,
        *,
        root_id: UUID,
        storyboard_id: UUID,
        main: _Board,
        child: _Board,
        entries: list[_Entry],
        deleted_ids: set[UUID],
        keep_both: set[UUID],
        field_resolutions: Mapping[UUID, Mapping[str, str]],
        scene_map: dict[UUID, UUID],
        asset_map: dict[UUID, UUID],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[tuple[UUID, int]]]:
        """Build the panel upsert rows + their join rows + ``(panel_id, revision)`` touched.

        Every row carries the SAME key set (all panel columns) — ``upsert_rows`` compiles
        ONE statement, so a ragged key set would silently take column DEFAULTS.

        ``keep_both`` entries are minted as NEW panels (never upserted onto main's row), so
        main's row survives untouched beside the branch's version.
        """
        taken, next_position = self._shot_key_claims(
            main=main, entries=entries, deleted_ids=deleted_ids, keep_both=keep_both
        )

        rows: list[dict[str, Any]] = []
        joins: list[dict[str, Any]] = []
        touched: list[tuple[UUID, int]] = []

        # Existing panels first: they own their shot_keys, so an ADDED panel that collides
        # is the one that gets renamed — not the panel that was already on the board.
        for entry in sorted(entries, key=lambda e: e.kind == "added"):
            src = child.panels.get(entry.key)
            if src is None:  # pragma: no cover — ``merge`` routes branch-absent to DELETE
                # Loud, not silent. ``merge`` sends every entry with ``branch is None`` to
                # ``to_delete``, so a writer entry ALWAYS has a branch row. If that ever
                # stops holding, the entry is already counted in ``applied`` and has already
                # released its shot_key in ``_shot_key_claims`` above — skipping it would
                # report a merge that did not happen and could hand its shot_key to another
                # panel. Crash instead of lying.
                raise MergeRoutingError(entry.key)
            root_panel = main.panels.get(entry.key)
            existing_id = main.panel_ids.get(entry.key)

            reuses_root_row = (
                root_panel is not None
                and existing_id is not None
                and existing_id not in deleted_ids
                # ``keep_both`` KEEPS main's row and mints the branch's alongside it, so it
                # must take the ADDED path below — never upsert onto main's id.
                and entry.key not in keep_both
            )
            if reuses_root_row and root_panel is not None:
                # MODIFIED / CONFLICT-resolved-to-branch. Start from the ROOT's row so
                # ``id``, ``created_at``, ``origin_id`` and the whole approval block
                # (``status`` / ``approved_at`` / ``approved_by``) survive verbatim, then
                # overlay the branch's CONTENT. This IS the upsert-onto-the-existing-id
                # invariant, expressed as data.
                row = dict(root_panel)
                revision = int(root_panel.get("revision", 1)) + 1
            else:
                # ADDED — or CONFLICT where main deleted the panel and the caller chose the
                # branch: re-materialise it, with lineage so a re-merge recognises it.
                row = dict(src)
                row["id"] = uuid7()
                # For an ADDED/branch-only row the lineage key IS ``entry.key``. But a
                # ``keep_both`` row is minted BESIDE main's kept row, whose lineage key is
                # ALSO ``entry.key`` (a conflict shares one key across both lanes). Giving the
                # new row that same key would make ``_build_board`` collapse the two into one
                # (last writer wins) on the next diff/merge. ``None`` makes it a fresh native
                # root that self-lineages off its own id — two independent panels, as intended.
                row["origin_id"] = None if entry.key in keep_both else entry.key
                row["position"] = next_position
                next_position += 1
                # A branch author's approval does not travel: the root has no trail event
                # for it, so an "approved" panel nobody approved on the root would be a
                # forged governance record.
                row["status"] = "draft"
                row["approved_at"] = None
                row["approved_by"] = None
                revision = int(src.get("revision", 1))

            panel_id = UUID(str(row["id"]))
            row["project_id"] = root_id
            row["storyboard_id"] = storyboard_id
            row["revision"] = revision
            # A field-resolved conflict — OR an auto-merged disjoint divergence — does a TRUE
            # 3-way overlay instead of blanket-copying the branch: ``row`` already holds MAIN's
            # row, so keeping a field is a no-op and overlaying takes the branch. Fields main
            # changed but the branch did not MUST be kept — the blanket copy below would revert
            # them to base, silently discarding main's edit (the whole reason per-field exists,
            # and the whole reason disjoint edits can auto-merge). Conflicting fields honour the
            # caller's per-column pick. Auto-merged entries carry NO ``fr`` (nothing to pick —
            # the sides are disjoint by construction) but still take the field-wise path so
            # main's disjoint fields survive; every whole-row/default entry keeps the blanket
            # overlay.
            fr = field_resolutions.get(entry.key)
            field_merge = fr is not None or _auto_mergeable(entry)
            main_only = (
                set(entry.main_changed_fields) - set(entry.changed_fields) if field_merge else set()
            )
            for f in _CONTENT_FIELDS:
                if f in ("shot_key", "position"):
                    continue
                if field_merge and ((fr or {}).get(f) == "main" or f in main_only):
                    continue  # keep main's value (a "main" pick, or a main-only edit)
                row[f] = src.get(f)
            if reuses_root_row:
                # A reorder IS content — carry the branch's position onto the root row.
                # An ADDED panel keeps the append-at-max+1 slot assigned above instead:
                # the branch's own position numbers a DIFFERENT board and would collide.
                row["position"] = int(src.get("position", row.get("position", 0)))

            scene_key = child.scene_key_of(src)
            row["scene_id"] = scene_map.get(scene_key) if scene_key else None

            desired = str(src.get("shot_key") or f"clip-{row['position'] + 1}")
            owner = taken.get(desired)
            if owner is not None and owner != panel_id:
                desired = self._free_shot_key(desired, taken)
            row["shot_key"] = desired
            taken[desired] = panel_id

            rows.append(row)
            touched.append((panel_id, revision))

            joins.extend(
                self._join_rows(panel_id=panel_id, key=entry.key, child=child, asset_map=asset_map)
            )
        return rows, joins, touched

    def _join_rows(
        self,
        *,
        panel_id: UUID,
        key: UUID,
        child: _Board,
        asset_map: dict[UUID, UUID],
    ) -> list[dict[str, Any]]:
        """The panel↔asset join rows for one merged panel, asset keys mapped to root ids."""
        return [
            {
                "id": uuid7(),
                "panel_id": panel_id,
                "asset_id": asset_map[asset_key],
                "role": role,
                "position": position,
            }
            for role, asset_keys in child.roles.get(key, {}).items()
            for position, asset_key in enumerate(asset_keys)
            # ``_ensure_assets`` materialises every referenced asset, so a miss is
            # unreachable — skipped rather than KeyError'd if that ever stops holding.
            if asset_key in asset_map
        ]

    @staticmethod
    def _free_shot_key(desired: str, taken: dict[str, UUID]) -> str:
        """``clip-3`` → ``clip-3-2`` when the root already has a DIFFERENT panel on that key.

        Two projects can legally hold conflicting ``clip-3``s (the unique constraint is
        per-project — that is the whole reason a branch is its own project), so a merge that
        brings a NEW panel across can collide. Renaming the incoming panel is the only
        non-destructive answer: refusing the merge would strand the branch, and overwriting
        the root's ``clip-3`` would silently destroy a panel nobody's diff mentioned.
        """
        n = 2
        while f"{desired}-{n}" in taken:
            n += 1
        return f"{desired}-{n}"
