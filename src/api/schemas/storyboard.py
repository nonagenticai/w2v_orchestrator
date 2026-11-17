"""Storyboard schemas — panels, scenes, assets (Phase 0 CRUD).

Mirrors ``schemas/verbatim.py``: the shared ``_camel`` config (NO alias
generator → JSON keys are snake_case) and ``*Create/Update/Response/
ListResponse`` per resource. All authoring fields are OPTIONAL so a partial or
AI-drafted board still persists (spec §3).

Response models are populated by the route's hand-built ``_resp(row) -> dict``
mapper (mirror ``routes/verbatim.py``), NOT ``from_attributes`` — the route maps
ORM columns → response fields explicitly.

Domain enums (``ShotSize``/``CameraAngle``/…) live in ``src.models.showcase_line``
(owned by WS-A); the ``storyboards.storyboard_id`` grouping is resolved
server-side and is not part of the external surface.
"""

from __future__ import annotations

import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

from src.models.showcase_line import (
    AssetKind,
    CameraAngle,
    CameraMovement,
    MediaRef,
    PanelGen,
    PanelStatus,
    ShotSize,
    TransitionOut,
    VisualType,
)

from ._shared import _camel


def _reject_null(v: Any) -> Any:
    """Reject an explicit ``null`` on a non-nullable field (→ 422).

    ``mode="before"`` validators fire ONLY when the key is present in the
    payload, so ``model_dump(exclude_unset=True)`` still lets an omitted field
    no-op. Nullable columns (e.g. ``canonical_descriptor``) skip this guard and
    clear to ``null`` fine.
    """
    if v is None:
        raise ValueError("field is non-nullable; omit it to leave unchanged")
    return v


# ─── Panels ──────────────────────────────────────────────────────────────
class StoryboardPanelCreate(BaseModel):
    model_config = _camel
    # Spine key (clip-<n>); blank/None → repo assigns clip-<position+1> (AuthoredLine rule).
    shot_key: Annotated[str | None, Field(max_length=64)] = None
    scene_id: UUID | None = None
    shot_number: Annotated[str | None, Field(max_length=16)] = None
    panel_index: int | None = None
    status: PanelStatus | None = None
    revision: int | None = None
    # Action / audio
    action_description: Annotated[str | None, Field(max_length=4000)] = None
    dialogue: Annotated[str | None, Field(max_length=4000)] = None
    vo_text: Annotated[str | None, Field(max_length=4000)] = None
    sfx_cue: Annotated[str | None, Field(max_length=1000)] = None
    music_cue: Annotated[str | None, Field(max_length=1000)] = None
    # Visual
    image_ref: Annotated[str | None, Field(max_length=512)] = None
    media_ref: dict[str, object] | None = None
    visual_type: VisualType | None = None
    shot_size: ShotSize | None = None
    composition_notes: Annotated[str | None, Field(max_length=2000)] = None
    # Asset references. On the wire these are still flat id lists / a scalar id, but
    # they persist as ``storyboard_panel_assets`` rows (role character|prop|location)
    # behind a real FK — an id that is not an asset OF THIS PROJECT is a 422, and an
    # asset a panel references cannot be silently deleted out from under it (409).
    characters: list[str] | None = None  # asset ids, order-preserving
    location_asset_id: UUID | None = None  # asset id; null clears it (nullable)
    props: list[str] | None = None  # asset ids, order-preserving
    lighting_mood: Annotated[str | None, Field(max_length=256)] = None
    art_style: Annotated[str | None, Field(max_length=256)] = None
    aspect_ratio: Annotated[str | None, Field(max_length=16)] = None
    # Camera
    angle: CameraAngle | None = None
    movement: CameraMovement | None = None
    movement_path: dict[str, object] | None = None
    lens: Annotated[str | None, Field(max_length=64)] = None
    # Timing / transition
    duration_seconds: float | None = None
    transition_out: TransitionOut | None = None
    # Gen provenance (populated by later phases; passthrough here)
    gen: dict[str, object] | None = None
    source: Annotated[str | None, Field(max_length=32)] = None  # ai|user|uploaded
    # Ordering (verbatim parity: None → append at max+1)
    position: int | None = None


class StoryboardPanelUpdate(BaseModel):
    model_config = _camel
    shot_key: Annotated[str | None, Field(max_length=64)] = None
    scene_id: UUID | None = None
    shot_number: Annotated[str | None, Field(max_length=16)] = None
    panel_index: int | None = None
    status: PanelStatus | None = None
    revision: int | None = None
    action_description: Annotated[str | None, Field(max_length=4000)] = None
    dialogue: Annotated[str | None, Field(max_length=4000)] = None
    vo_text: Annotated[str | None, Field(max_length=4000)] = None
    sfx_cue: Annotated[str | None, Field(max_length=1000)] = None
    music_cue: Annotated[str | None, Field(max_length=1000)] = None
    image_ref: Annotated[str | None, Field(max_length=512)] = None
    media_ref: dict[str, object] | None = None
    visual_type: VisualType | None = None
    shot_size: ShotSize | None = None
    composition_notes: Annotated[str | None, Field(max_length=2000)] = None
    characters: list[str] | None = None
    location_asset_id: UUID | None = None
    props: list[str] | None = None
    lighting_mood: Annotated[str | None, Field(max_length=256)] = None
    art_style: Annotated[str | None, Field(max_length=256)] = None
    aspect_ratio: Annotated[str | None, Field(max_length=16)] = None
    angle: CameraAngle | None = None
    movement: CameraMovement | None = None
    movement_path: dict[str, object] | None = None
    lens: Annotated[str | None, Field(max_length=64)] = None
    duration_seconds: float | None = None
    transition_out: TransitionOut | None = None
    gen: dict[str, object] | None = None
    source: Annotated[str | None, Field(max_length=32)] = None
    position: int | None = None
    # Opt-in optimistic concurrency: echo the ``lock_version`` last read to get
    # compare-and-set (409 on a concurrent write). Omit it for last-write-wins. It is
    # NOT a column — the route pops it out before the field ``setattr`` loop.
    expected_lock_version: int | None = None

    # Reject explicit null (→ 422); omit the key to leave the field unchanged.
    # Most of these are NOT-NULL columns. ``characters``/``props`` no longer are
    # columns at all — they project the ``storyboard_panel_assets`` join rows — but
    # the guard stays: a null there has no meaning distinct from "leave alone", and
    # the way to CLEAR a role is an empty list. (``location_asset_id`` is deliberately
    # absent: it is a scalar role, and null IS its legitimate clear.)
    _nn = field_validator(
        "shot_key",
        "position",
        "panel_index",
        "status",
        "revision",
        "characters",
        "props",
        "media_ref",
        "gen",
        mode="before",
    )(_reject_null)


class StoryboardPanelResponse(BaseModel):
    model_config = _camel
    id: UUID
    project_id: UUID
    # Lineage key that survives a fork/merge (the source row's ``origin_id`` if set, else its
    # own id at fork). ``None`` on a native, never-forked panel. Exposed so a branch-diff
    # client can correlate a child panel to its lineage-keyed diff entry.
    origin_id: UUID | None = None
    shot_key: str
    scene_id: UUID | None = None
    shot_number: str | None = None
    panel_index: int | None = None
    status: PanelStatus
    revision: int
    # Optimistic-concurrency token: read it here, echo it as ``expected_lock_version``
    # on the next write to opt into compare-and-set.
    lock_version: int
    approved_at: datetime.datetime | None = None
    approved_by: str | None = None
    action_description: str | None = None
    dialogue: str | None = None
    vo_text: str | None = None
    sfx_cue: str | None = None
    music_cue: str | None = None
    image_ref: str | None = None
    media_ref: dict[str, object]
    visual_type: VisualType | None = None
    shot_size: ShotSize | None = None
    composition_notes: str | None = None
    characters: list[str]
    location_asset_id: UUID | None = None
    props: list[str]
    lighting_mood: str | None = None
    art_style: str | None = None
    aspect_ratio: str | None = None
    angle: CameraAngle | None = None
    movement: CameraMovement | None = None
    movement_path: dict[str, object] | None = None
    lens: str | None = None
    duration_seconds: float | None = None
    transition_out: TransitionOut | None = None
    gen: dict[str, object]
    source: str | None = None
    position: int
    created_at: datetime.datetime
    updated_at: datetime.datetime


class StoryboardPanelListResponse(BaseModel):
    model_config = _camel
    items: list[StoryboardPanelResponse]
    total: int


class ReorderPanels(BaseModel):
    """Permutation of the project's panel row-ids (mirror ``ReorderChildren``)."""

    model_config = _camel
    ordered_panel_ids: list[UUID]
    # MANDATORY optimistic-concurrency map: ``{row id: lock_version}`` exactly as the
    # caller last read the board. Typed optional ONLY so an omitted field lands as an
    # actionable 400 from the route ("send expected_lock_versions …") instead of pydantic's
    # generic 422 — the route rejects ``None``. A stale/incomplete map is a 409.
    expected_lock_versions: dict[UUID, int] | None = None


class ReorderScenes(BaseModel):
    """Permutation of the project's scene row-ids (mirror ``ReorderPanels``)."""

    model_config = _camel
    ordered_scene_ids: list[UUID]
    # MANDATORY optimistic-concurrency map: ``{row id: lock_version}`` exactly as the
    # caller last read the board. Typed optional ONLY so an omitted field lands as an
    # actionable 400 from the route ("send expected_lock_versions …") instead of pydantic's
    # generic 422 — the route rejects ``None``. A stale/incomplete map is a 409.
    expected_lock_versions: dict[UUID, int] | None = None


# ─── Approval trail (Boords pattern) ─────────────────────────────────────
class PanelApproveRequest(BaseModel):
    """Body for approve/unapprove — an optional free-text note recorded on the event."""

    model_config = _camel
    note: Annotated[str | None, Field(max_length=2000)] = None
    # Opt-in optimistic concurrency for the approve/unapprove state flip — same
    # semantics as ``StoryboardPanelUpdate.expected_lock_version``.
    expected_lock_version: int | None = None


class PanelEventResponse(BaseModel):
    model_config = _camel
    id: UUID
    project_id: UUID
    panel_id: UUID
    event_type: str
    revision: int
    actor_subject_id: UUID | None = None
    note: str | None = None
    created_at: datetime.datetime


class PanelEventListResponse(BaseModel):
    model_config = _camel
    items: list[PanelEventResponse]
    total: int


# ─── Scenes ──────────────────────────────────────────────────────────────
class StoryboardSceneCreate(BaseModel):
    model_config = _camel
    name: Annotated[str | None, Field(max_length=256)] = None
    int_ext: Annotated[str | None, Field(max_length=32)] = None  # free text INT/EXT/INT_EXT
    location: Annotated[str | None, Field(max_length=256)] = None
    time_of_day: Annotated[str | None, Field(max_length=64)] = None
    description: Annotated[str | None, Field(max_length=4000)] = None
    position: int | None = None


class StoryboardSceneUpdate(BaseModel):
    model_config = _camel
    name: Annotated[str | None, Field(max_length=256)] = None
    int_ext: Annotated[str | None, Field(max_length=32)] = None
    location: Annotated[str | None, Field(max_length=256)] = None
    time_of_day: Annotated[str | None, Field(max_length=64)] = None
    description: Annotated[str | None, Field(max_length=4000)] = None
    position: int | None = None
    # Opt-in optimistic concurrency — same semantics as
    # ``StoryboardPanelUpdate.expected_lock_version``. Not a column; the route pops it
    # out before the field ``setattr`` loop.
    expected_lock_version: int | None = None

    # NOT-NULL int col: reject explicit null (→ 422); omit to leave unchanged.
    _nn = field_validator("position", mode="before")(_reject_null)


class StoryboardSceneResponse(BaseModel):
    model_config = _camel
    id: UUID
    project_id: UUID
    name: str | None = None
    int_ext: str | None = None
    location: str | None = None
    time_of_day: str | None = None
    description: str | None = None
    position: int
    # Optimistic-concurrency token: read it here, echo it as ``expected_lock_version``
    # on the next write to opt into compare-and-set.
    lock_version: int
    created_at: datetime.datetime
    updated_at: datetime.datetime


class StoryboardSceneListResponse(BaseModel):
    model_config = _camel
    items: list[StoryboardSceneResponse]
    total: int


# ─── Assets (Character | Location | Prop) ────────────────────────────────
class StoryboardAssetCreate(BaseModel):
    model_config = _camel
    kind: AssetKind
    name: Annotated[str, Field(min_length=1, max_length=256)]  # consistency key
    canonical_descriptor: Annotated[str | None, Field(max_length=4000)] = None
    reference_images: list[str] | None = None  # MinIO asset ids
    seed: int | None = None
    style_token: Annotated[str | None, Field(max_length=128)] = None
    lora_path: Annotated[str | None, Field(max_length=512)] = None
    attributes: dict[str, object] | None = None
    position: int | None = None


class StoryboardAssetUpdate(BaseModel):
    model_config = _camel
    kind: AssetKind | None = None
    name: Annotated[str | None, Field(min_length=1, max_length=256)] = None
    canonical_descriptor: Annotated[str | None, Field(max_length=4000)] = None
    reference_images: list[str] | None = None
    seed: int | None = None
    style_token: Annotated[str | None, Field(max_length=128)] = None
    lora_path: Annotated[str | None, Field(max_length=512)] = None
    attributes: dict[str, object] | None = None
    position: int | None = None
    # Opt-in optimistic concurrency — same semantics as
    # ``StoryboardPanelUpdate.expected_lock_version``. Not a column; the route pops it
    # out before the field ``setattr`` loop.
    expected_lock_version: int | None = None

    # NOT-NULL cols: reject explicit null (→ 422); omit to leave unchanged.
    _nn = field_validator(
        "kind",
        "name",
        "position",
        "reference_images",
        "attributes",
        mode="before",
    )(_reject_null)


class StoryboardAssetResponse(BaseModel):
    model_config = _camel
    id: UUID
    project_id: UUID
    kind: AssetKind
    name: str
    canonical_descriptor: str | None = None
    reference_images: list[str]
    seed: int | None = None
    style_token: str | None = None
    lora_path: str | None = None
    attributes: dict[str, object]
    position: int
    # Optimistic-concurrency token: read it here, echo it as ``expected_lock_version``
    # on the next write to opt into compare-and-set.
    lock_version: int
    created_at: datetime.datetime
    updated_at: datetime.datetime


class StoryboardAssetListResponse(BaseModel):
    model_config = _camel
    items: list[StoryboardAssetResponse]
    total: int


# ─── AI authoring (Phase 1 — LLM draft / rewrite) ────────────────────────
class StoryboardDraftRequest(BaseModel):
    model_config = _camel
    script_or_idea: Annotated[str | None, Field(max_length=20000)] = None
    instructions: Annotated[str | None, Field(max_length=4000)] = None
    replace: bool = False
    target_scene_count: int | None = None
    target_panel_count: int | None = None


class StoryboardDraftResponse(BaseModel):
    model_config = _camel
    scenes: list[StoryboardSceneResponse]
    panels: list[StoryboardPanelResponse]
    assets: list[StoryboardAssetResponse]


class PanelRewriteRequest(BaseModel):
    model_config = _camel
    instruction: Annotated[str, Field(min_length=1, max_length=2000)]
    fields: list[str] | None = None


# ─── Panel image generation (Phase 2) ────────────────────────────────────
PanelImageStatus = Literal["queued", "running", "completed", "failed"]


class PanelImageRequest(BaseModel):
    """Body for ``POST /api/storyboard/panels/{id}/image``.

    Everything is optional: the prompt is composed server-side from the panel's
    own direction plus its referenced assets' canonical descriptors, and the
    seed is taken from the referenced assets so a character stays on-model
    across the board. Overrides exist for the operator who wants a different
    aspect, provider, or an explicit re-roll.

    There is deliberately NO reference-image field. The image runner is
    text→image plus a seed; it cannot be conditioned on a picture, so accepting
    one here would be a lie in the API surface.
    """

    model_config = _camel
    # Opt-in optimistic concurrency, asserted at SUBMIT time: a client working
    # from a stale panel gets a 409 before anything is paid for.
    expected_lock_version: int | None = None
    size: Annotated[str | None, Field(pattern=r"^\d{2,5}x\d{2,5}$")] = None
    provider: Annotated[str | None, Field(max_length=64)] = None
    # Explicit re-roll / pin. Bounded to **int32**, which is what the image
    # runner accepts (`image_gen_backend/src/models.py`: `le=2**31 - 1`). This
    # said "uint32 — what FLUX accepts" and allowed 4_294_967_295; every value
    # above 2147483647 was accepted here and then rejected downstream with a
    # 422, so the caller's own override could not work.
    seed: Annotated[int | None, Field(ge=0, le=2_147_483_647)] = None
    negative_prompt: Annotated[str | None, Field(max_length=2000)] = None


class PanelImageJobResponse(BaseModel):
    """Status of one panel-image generation.

    ``queued``/``running`` are transient; ``completed`` carries the panel's new
    ``media_ref`` + ``gen`` (so a client that polls need not re-fetch the panel),
    ``failed`` carries ``error``. The durable record is the PANEL, not this — the
    job is TTL-bounded progress metadata.

    A generated ``media_ref.url`` is served as the panel's image-proxy path
    (``/api/storyboard/panels/{id}/image/content``) rather than the stored
    ``minio.test`` URL, which no browser outside the cluster can resolve; a url
    the proxy cannot serve (an external image) is passed through untouched. The
    path addresses the PANEL's current image rather than this job's snapshot of
    it, so a client that caches an image url is caching a pointer to the durable
    record — which is the panel, not this.
    """

    model_config = _camel
    job_id: str
    panel_id: UUID
    project_id: UUID
    status: PanelImageStatus = "queued"
    error: str | None = None
    # Typed, not a passthrough blob: this pair IS the generation contract, and
    # the ``Any`` ratchet exists to stop new JSON dicts leaking into the surface.
    media_ref: MediaRef | None = None
    gen: PanelGen | None = None
    updated_at: str | None = None


# ─── Version snapshots + restore (Phase 5) ───────────────────────────────
class SnapshotCreateRequest(BaseModel):
    """Body for a manual snapshot — an optional free-text note."""

    model_config = _camel
    note: Annotated[str | None, Field(max_length=2000)] = None


class SnapshotResponse(BaseModel):
    model_config = _camel
    id: UUID
    # 'manual' (user-authored) | 'auto' (pre-restore/pre-merge safety copy) |
    # 'branch_base' (the fork's pinned merge base). All three are permitted by
    # ``ck_storyboard_snapshots_kind`` (models.py); the last two are machine-managed
    # and the DELETE route 409s on them.
    kind: str
    note: str | None = None
    actor_subject: str | None = None
    created_at: datetime.datetime
    # Cheap graph-size hints, populated only when the payload is already in hand
    # (create/get) — omitted (None) on the payload-deferred list scan.
    scene_count: int | None = None
    panel_count: int | None = None
    asset_count: int | None = None


class SnapshotListResponse(BaseModel):
    model_config = _camel
    items: list[SnapshotResponse]
    total: int


class RestoreSnapshotRequest(BaseModel):
    """Body for a snapshot restore — the whole-board optimistic-concurrency map.

    Restore rewrites scenes + panels + assets wholesale, so the caller must state the board
    state it is restoring OVER: one flat ``{row id: lock_version}`` map covering every live
    scene, panel and asset (uuid7 ids share one namespace, so no bucketing). Anything added,
    deleted or edited since that read → 409, instead of silently discarding it.
    """

    model_config = _camel
    # MANDATORY optimistic-concurrency map: ``{row id: lock_version}`` exactly as the
    # caller last read the board. Typed optional ONLY so an omitted field lands as an
    # actionable 400 from the route ("send expected_lock_versions …") instead of pydantic's
    # generic 422 — the route rejects ``None``. A stale/incomplete map is a 409.
    expected_lock_versions: dict[UUID, int] | None = None


class RestoreResponse(BaseModel):
    model_config = _camel
    restored_snapshot_id: UUID
    auto_snapshot_id: UUID


# ─── Branch/merge (PR2) ──────────────────────────────────────────────────
class BranchCreateRequest(BaseModel):
    """Body for opening a branch off a root project's current state."""

    model_config = _camel
    name: Annotated[str, Field(min_length=1, max_length=256)]


class BranchResponse(BaseModel):
    model_config = _camel
    id: UUID
    root_project_id: UUID
    child_project_id: UUID
    name: str
    base_snapshot_id: UUID
    status: str  # 'open' | 'merged' | 'abandoned'
    created_at: datetime.datetime
    merged_at: datetime.datetime | None = None


class BranchListResponse(BaseModel):
    model_config = _camel
    items: list[BranchResponse]
    total: int


# ─── Branch three-way diff + merge ───────────────────────────────────────
class DiffEntry(BaseModel):
    """One panel's three-way story, keyed by LINEAGE — never by panel id.

    ``key`` is the panel's ``origin_id`` if it has one, else its own id. Every side
    resolves to the same key: a root panel is its own origin, and the fork stamped that
    id onto the child's copy. Panel ids are re-minted per project, so keying on them
    would make every panel look "added" on one side and "deleted" on the other.

    ``base`` / ``main`` / ``branch`` are the normalised panel payloads of the three
    sides (``None`` where the panel does not exist on that side), so the UI can render
    three columns without re-fetching. They are NORMALISED, not raw rows: ``scene`` and
    the ``characters`` / ``props`` / ``location`` roles are expressed as LINEAGE keys,
    because the raw ids differ per project and would otherwise report a spurious edit on
    every forked panel.

    Approval state (``status`` / ``approved_at`` / ``approved_by``) is deliberately NOT
    part of the compared surface: approval is the ROOT's governance record. Diffing it
    would let a merge forge an approval on the root (a branch author approving their own
    copy) or silently unapprove a root panel — neither has a corresponding root-side
    trail event. Merge carries CONTENT; approvals stay where they were granted.
    """

    model_config = _camel
    key: UUID
    kind: str  # 'added' | 'modified' | 'deleted' | 'conflict'
    shot_key: str | None = None
    base: dict[str, object] | None = None
    main: dict[str, object] | None = None
    branch: dict[str, object] | None = None
    # Field names that differ branch-vs-base and main-vs-base. The intersection is what
    # makes a ``conflict`` a conflict, and it is what the UI highlights.
    changed_fields: list[str] = Field(default_factory=list)
    main_changed_fields: list[str] = Field(default_factory=list)


class BranchDiffResponse(BaseModel):
    """The diff, plus the optimistic-concurrency token the merge must hand back.

    ``root_fingerprint`` is a content hash of the ROOT's whole board (panels + scenes +
    assets, normalised exactly as above). It is NOT a timestamp: a clock-based etag skews
    across pods and a ``updated_at`` bump fires on writes that changed nothing. The merge
    request echoes it; if the root moved in between, the merge is a 409 rather than a
    silent clobber of whatever landed on main while the author was reading the diff.

    Entries are BRANCH-CENTRIC: a panel that only main changed is not an entry (there is
    nothing to merge). It still moves the fingerprint, which is the point.
    """

    model_config = _camel
    branch_id: UUID
    root_project_id: UUID
    child_project_id: UUID
    status: str  # 'open' | 'merged' | 'abandoned'
    base_snapshot_id: UUID
    root_fingerprint: str
    entries: list[DiffEntry]
    conflict_count: int
    # The server's ALLOWLIST of per-field-mergeable columns — the single source of truth for
    # which conflicting fields a client may resolve branch-vs-main per column. Everything not
    # in it (shot_key/position and the lineage keys) is fenced to a whole-row verdict. Emitted
    # so the frontend derives its fence from the server instead of hand-mirroring a denylist
    # that drifts LOOSER as new content columns are added.
    mergeable_fields: list[str]


class MergeResolution(BaseModel):
    """Per-entry verdict. ``branch`` applies the branch side onto the root; ``main``
    keeps the root's version; ``skip`` leaves the panel alone (identical effect to
    ``main`` — both are a no-op write — but they say different things in the audit and
    the UI, so the vocabulary keeps them apart). ``keep_both`` (conflicts only) keeps the
    root's version AND brings the branch's conflicting panel across as a NEW panel with a
    fresh, non-colliding ``shot_key`` — the "I want both cuts" resolution."""

    model_config = _camel
    key: UUID
    # Whole-row verdict. Mutually exclusive with ``field_resolutions`` — exactly one must be
    # present (a resolution either takes a whole side or picks per column, never both).
    resolution: str | None = None  # 'branch' | 'main' | 'skip' | 'keep_both'
    # Per-column branch/main picks for a field-resolvable conflict (both sides edited an
    # overlapping set of plain content columns). Keys must be the conflict's actually-
    # conflicting fields; the server rejects a set that does not cover them exactly (422).
    field_resolutions: dict[str, Literal["branch", "main"]] | None = None

    @field_validator("resolution")
    @classmethod
    def _known(cls, v: str | None) -> str | None:
        if v is not None and v not in ("branch", "main", "skip", "keep_both"):
            raise ValueError("resolution must be one of: branch, main, skip, keep_both")
        return v

    @field_validator("field_resolutions")
    @classmethod
    def _known_fields(cls, v: dict[str, str] | None) -> dict[str, str] | None:
        if v is None:
            return v
        if not v:
            raise ValueError("field_resolutions must not be empty")
        # Single source of truth for the mergeable-column allowlist — lazily imported so the
        # schema module never couples to the merge service at import time. shot_key/position
        # and the lineage keys (scene/characters/props/location) are NEVER per-field mergeable.
        from src.services.storyboard.merge_service import FIELD_MERGEABLE

        bad = sorted(k for k in v if k not in FIELD_MERGEABLE)
        if bad:
            raise ValueError(f"field_resolutions names non-mergeable field(s): {', '.join(bad)}")
        return v

    @model_validator(mode="after")
    def _exactly_one(self) -> MergeResolution:
        if (self.resolution is None) == (self.field_resolutions is None):
            raise ValueError("provide exactly one of resolution or field_resolutions")
        return self


class MergeRequest(BaseModel):
    """``root_fingerprint`` is REQUIRED — it is the whole concurrency contract.

    ``resolutions`` need only cover what the caller wants to override: a non-conflict
    entry defaults to ``branch`` (the author asked to merge their branch). A ``conflict``
    entry has NO default — merging one unresolved is a 422, because guessing which side
    wins is precisely the decision the human opened the diff to make.
    """

    model_config = _camel
    root_fingerprint: Annotated[str, Field(min_length=1, max_length=128)]
    resolutions: list[MergeResolution] = Field(default_factory=list)


class MergeResponse(BaseModel):
    model_config = _camel
    branch_id: UUID
    status: str  # always 'merged' on success
    merged_at: datetime.datetime | None = None
    # The ROOT's pre-merge safety snapshot (kind='auto'), exactly like restore mints.
    # A merge is therefore reversible: restore this id.
    auto_snapshot_id: UUID
    applied: list[UUID]  # lineage keys written onto the root
    skipped: list[UUID]  # lineage keys the caller resolved to 'main'/'skip'
    # The root's fingerprint AFTER the merge — a second merge into the same root can
    # chain straight off it without re-reading the diff.
    root_fingerprint: str


class SiblingMergeRequest(BaseModel):
    """Merge one branch INTO a SIBLING — both forked off the same parent, so the parent's
    shared ``branch_base`` is their common ancestor for a symmetric three-way.

    ``target_branch_id`` is the sibling to merge into; ``target_fingerprint`` (from the
    sibling diff) is the concurrency contract on the TARGET board — a 409 if it moved.
    ``resolutions`` follow the same rules as a branch->root merge (a conflict has no default).
    """

    model_config = _camel
    target_branch_id: UUID
    target_fingerprint: Annotated[str, Field(min_length=1, max_length=128)]
    resolutions: list[MergeResolution] = Field(default_factory=list)


class SiblingMergeResponse(BaseModel):
    """Both branches stay OPEN — a sibling merge cross-pollinates, it does not close a
    lifecycle: the source is not ``merged`` (its content never reached the root) and neither
    child is archived. Only the TARGET board changed; ``target_fingerprint`` is its post-merge
    hash, so a chained sibling merge can proceed without re-reading the diff."""

    model_config = _camel
    source_branch_id: UUID
    target_branch_id: UUID
    # The TARGET's pre-merge safety snapshot (kind='auto') — restore this id to undo.
    auto_snapshot_id: UUID
    applied: list[UUID]  # lineage keys written onto the target
    skipped: list[UUID]  # lineage keys the caller resolved to 'main'/'skip'
    target_fingerprint: str
