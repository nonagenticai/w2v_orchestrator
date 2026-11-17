"""Blueprints + Stage outputs + Runs schemas."""

from __future__ import annotations

import datetime
from typing import Literal
from uuid import UUID

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from src.services.auth_secrets import redact_auth_secrets

from ._shared import FailureClass, ProjectType, RunStatus, StageStatus, _camel

#: ``runs.pause_reason`` (migration 0102). ``unknown`` = paused before 0102 with no
#: evidence of which kind (0102's backfill); show it as a plain "paused".
PauseReason = Literal["checkpoint", "manual", "unknown"]


class BlueprintStage(BaseModel):
    model_config = _camel
    name: str
    depends_on: list[str]
    parallel_group: str | None
    default_timeout_seconds: int
    default_pause_after: bool


class BlueprintSummary(BaseModel):
    model_config = _camel
    id: str
    project_type: ProjectType
    stages: list[BlueprintStage]


# ─── Stage outputs ───────────────────────────────────────────────────────


class StageOutputResponse(BaseModel):
    model_config = _camel
    id: UUID
    run_id: UUID
    stage_name: str
    status: StageStatus
    output: dict | None
    error: str | None
    started_at: datetime.datetime | None
    completed_at: datetime.datetime | None
    # Timeout-resume counter (`stage_outputs.attempts`). 0 on the first and only
    # attempt; >0 means the stage burnt an earlier attempt against its budget and
    # was re-dispatched. Surfaced because `started_at` alone cannot distinguish a
    # slow stage from a stage on its third go at the same work.
    attempts: int = 0

    # ⚠️ `None` is coerced to 0, and that is not defensive padding — it is the
    # UNFLUSHED-INSTANCE window. The column is `nullable=False, server_default="0"`,
    # so every row IN THE DATABASE has an int. But `server_default` (and a Python
    # `default=`) only fire at INSERT/flush, so a `StageOutput` built in Python and
    # serialised BEFORE flush has `attempts is None` — which `Mapped[int]` says is
    # impossible and `model_validate(..., from_attributes=True)` then rejects with
    # `int_type`. Six action_reel footage_probe tests construct exactly that shape,
    # and so can production code that responds from an un-flushed row.
    @field_validator("attempts", mode="before")
    @classmethod
    def _attempts_defaults_to_zero(cls, v: object) -> object:
        return 0 if v is None else v

    # True when the stage finished `completed` but soft-failed a downstream step
    # — "completed with warnings" rather than clean. Derived, not stored, from
    # EITHER channel on the persisted output (see
    # ``services/stages/shared/degradation.py::is_degraded``): the structured
    # ``degradations`` list (a stable code + detail: failed mastering gate,
    # out-of-tolerance duration, dropped captions, missing sound bed, stubbed
    # deck, unnormalized audio) or the legacy prose ``notes`` list.
    degraded: bool = False

    # WHICH degradations, not merely whether there were any. Nine stable codes
    # exist (``services/stages/shared/degradation.py::DEGRADATION_CODES``), every
    # one of them is emitted, and until this field ALL NINE collapsed into the
    # single `degraded` boolean above — a failed mastering gate, a silent video
    # missing its music bed and a stubbed deck were the same wire value. The
    # codes already ship inside `output["degradations"]`; this promotes them to a
    # named, typed field a consumer can bind to without walking a free-form blob.
    #
    # Deliberate properties:
    #   * PURELY ADDITIVE — `degraded` keeps its exact semantics and value. This
    #     field arms nothing: no consumer is required to read it, and an
    #     unrecognised code is reported, never rejected (a rolling deploy can put
    #     a newer pod's code in a row an older pod reads back).
    #   * `[]`, never None, when there is nothing to report — so a reader may
    #     iterate unconditionally.
    #   * DE-DUPLICATED in FIRST-SEEN order, not sorted and not a set: the same
    #     code raised by two stages of one run is one fact, and the order it is
    #     reported in must be deterministic for tests and diffs.
    #   * Sourced from the structured `degradations` channel ONLY. The legacy
    #     prose `notes` channel still feeds `degraded` but has no stable identity
    #     to promote, so a notes-only stage is `degraded=True` with `[]` here.
    #     That asymmetry is the point: it names exactly the runs whose degradation
    #     is still unqueryable prose.
    degradation_codes: list[str] = Field(default_factory=list)

    # The output keys holding EDITABLE elements (``PATCH .../elements/{id}``) on this
    # row — ``elements.populated_element_paths``, judged by the content. Empty for a
    # stage with no element addressing. Lets the Studio offer "edit this step"
    # without guessing which arrays the server accepts.
    element_paths: list[str] = Field(default_factory=list)
    # When a user last edited this stage's output in place (0102). A stage edited
    # after it completed keeps the edit on Restart; its dependents re-run.
    edited_at: datetime.datetime | None = None


class StageElementPatch(BaseModel):
    """Shallow-merge patch payload for one element inside a stage's JSONB output.

    Keys present here override the matching keys on the element; nested keys
    are NOT recursed into (last-write-wins on the whole nested value). The
    element's `id` field is preserved regardless of what's in `patch`."""

    model_config = _camel
    patch: dict


class StageElementRestamp(BaseModel):
    """Full-replacement payload for one element + dirty-mark downstream stages.

    Unlike `StageElementPatch` (shallow-merge in place, no downstream effect),
    a restamp replaces the addressed element with `value` and resets every
    downstream stage_output of the same run to `status='pending'` with
    `output=null`. After a restamp, a subsequent `POST /runs/{id}/regenerate`
    re-executes from the first dirty stage.

    The element's `id` is preserved regardless of `value` content."""

    model_config = _camel
    value: dict


class ChapterTitleOverrides(BaseModel):
    """Operator edits to a course's chapter titles and sub-chapter visibility.

    ``overrides`` is keyed by each chapter's stable ``chapter_key`` (stable across
    re-stitches, unlike sequence): the bare ``lesson_project_id`` for a per-lesson
    (level-0) marker, ``f"{lesson_project_id}#{ordinal}"`` for a per-slide
    (level-1) sub-chapter. Both levels are editable. Each title is trimmed and
    must be 1-120 chars after trimming. The route 404s a key that no chapter
    carries.

    ``hidden`` is the full set of level-1 ``chapter_key``\\ s the operator wants
    dropped from the published YouTube chapters and the in-app player (lesson
    markers are the backbone and can never be hidden). It has full-replacement
    semantics: ``None``/omitted leaves the stored hide set untouched (so a
    title-only edit never wipes hides); ``[]`` clears all hides; a list sets the
    hidden set to exactly those keys. The route 404s a key that is unknown or not
    a level-1 sub-chapter."""

    model_config = _camel
    overrides: dict[str, str]
    hidden: list[str] | None = None

    @field_validator("overrides")
    @classmethod
    def _validate_titles(cls, value: dict[str, str]) -> dict[str, str]:
        cleaned: dict[str, str] = {}
        for key, title in value.items():
            trimmed = (title or "").strip()
            if not trimmed:
                raise ValueError(f"chapter title for {key!r} is blank")
            if len(trimmed) > 120:
                raise ValueError(f"chapter title for {key!r} exceeds 120 chars")
            cleaned[key] = trimmed
        return cleaned

    @field_validator("hidden")
    @classmethod
    def _validate_hidden(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        # Strip blanks and dedupe, preserving first-seen order.
        return list(dict.fromkeys(k for raw in value if (k := (raw or "").strip())))


class ThumbnailOverride(BaseModel):
    """Operator override for a course's video thumbnail / poster.

    ``thumbnail_url`` is an artifact URL the orchestrator can presign (a MinIO
    object it owns or a local ``file://`` artifact — validated route-side).
    ``None`` clears the override and reverts to the auto AI cover."""

    model_config = _camel
    thumbnail_url: str | None = None


class RunUsageResponse(BaseModel):
    model_config = _camel
    stage_name: str
    model: str | None
    provider: str | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    # Which tenant provider connection (and pinned VERSION) served this row; both
    # None = the platform path (WM / gateway / billing-mirrored).
    connection_id: UUID | None = None
    connection_version: int | None = None


class RunUsageReport(BaseModel):
    """``POST /api/runs/{run_id}/usage`` — one LLM call a downstream service made on
    the platform vLLM for this run. Snake-case wire names, no aliases: this is an S2S
    contract (w2v_common's reporter), not an SPA payload."""

    model_config = ConfigDict(extra="forbid")
    #: The reporter's idempotency key: a retried report of the same call is a no-op.
    call_id: str = Field(min_length=1, max_length=128)
    stage: str | None = None
    model: str
    provider: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    #: The calling service's name (e.g. ``analysis-service``).
    service: str


class RunUsageReportResult(BaseModel):
    """``recorded=False`` ⇒ this ``call_id`` was already recorded for the run."""

    recorded: bool


CostStatus = Literal["ok", "unavailable"]


class RunUsageTotal(BaseModel):
    """A run's usage + cost.

    ``cost_status="unavailable"`` means the cost could not be priced (e.g.
    ``cost_unavailable_reason="billing_credential_missing"``) — ``cost_usd`` is
    then ``None``, never ``0``: an unknown cost must not be renderable as $0. The
    validator makes the two states mutually exclusive by construction.
    ``by_stage`` still lists the rows that ARE known locally.
    """

    model_config = _camel
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    cost_status: CostStatus = "ok"
    cost_unavailable_reason: str | None = None
    # True once the run's cost is FINAL: the run is terminal and its billing
    # mirror has been reconciled past the settle window. Until then
    # ``cost_usd`` is at best a partial figure (a live run, or billing's async
    # cost events still flushing) and a viewer should say so rather than show it
    # as the run's price. False for every non-terminal run.
    cost_settled: bool = Field(
        default=False,
        description=(
            "True once the cost is final (terminal run, billing mirror reconciled). "
            "While False, cost_usd is partial or not yet known."
        ),
    )
    by_stage: list[RunUsageResponse]

    @model_validator(mode="after")
    def _cost_state_is_consistent(self) -> RunUsageTotal:
        if self.cost_status == "unavailable":
            if self.cost_usd is not None:
                raise ValueError("cost_status=unavailable must not carry a cost_usd figure")
            if not self.cost_unavailable_reason:
                raise ValueError("cost_status=unavailable requires cost_unavailable_reason")
        else:
            if self.cost_usd is None:
                raise ValueError("cost_status=ok requires cost_usd")
            if self.cost_unavailable_reason is not None:
                raise ValueError("cost_unavailable_reason is only valid when unavailable")
        return self


# ─── Runs ────────────────────────────────────────────────────────────────


class RunStart(BaseModel):
    """Body for ``POST /api/deliverables/{id}/runs``.

    ``pause_after_stages`` semantics:

    * ``None`` (field omitted) → seed from the blueprint's
      ``default_pause_after`` stage specs (so e.g. promo runs pause after
      script + storyboard for creative review by default, matching the
      blueprint declaration). This is the SPA default — ``runs.start()``
      without args sends an empty body.
    * ``[]`` (explicit empty list) → autopilot, no checkpoints.
    * ``[stage_name, ...]`` → user-chosen checkpoint set.

    The empty-default sentinel changed in 2026-05-31: previously the
    field defaulted to ``[]`` which silently turned every run into
    autopilot, contradicting the blueprint's declared pause defaults
    and the run-monitor UI's intent to pause for the user to review.
    """

    model_config = _camel
    parent_run_id: UUID | None = None
    # ``_camel`` is populate_by_name with NO alias_generator, so a camelCase
    # client's ``pauseAfterStages``/``configOverrides`` was silently DROPPED to
    # the defaults below — and a dropped ``pause_after_stages`` takes the
    # "omitted: seed from blueprint defaults" branch, i.e. the caller's explicit
    # autopilot request silently reverts to review-gate pauses. Same hole, same
    # fix, as ``RegenerateRequest.from_stage``.
    pause_after_stages: list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices("pause_after_stages", "pauseAfterStages"),
    )
    config_overrides: dict = Field(
        default_factory=dict,
        validation_alias=AliasChoices("config_overrides", "configOverrides"),
    )


class CheckpointsResponse(BaseModel):
    model_config = _camel
    run_id: UUID
    pause_after_stages: list[str]


class CheckpointsUpdate(BaseModel):
    model_config = _camel
    pause_after_stages: list[str]


class RegenerateRequest(BaseModel):
    """Create a child run that re-runs part of the pipeline.

    ``from_stage`` (optional) targets a specific stage: re-run it and everything
    depending on it, reuse every other completed stage — the "rerun this stage"
    action. When omitted, the per-stage restart plan decides (see
    ``GET /runs/{id}/restart-plan``): unfinished, fed-back, input-changed and
    edit-dependent stages re-run, the rest is reused. 422 if ``from_stage`` is not a
    stage of this run's blueprint."""

    model_config = _camel
    config_overrides: dict = Field(default_factory=dict)
    # ``_camel`` is populate_by_name with NO alias_generator, so a camelCase client's
    # ``fromStage`` was silently DROPPED to None — and a dropped target takes the
    # "no notes, no target: re-run all" fallback in ``build_regenerate_state``, i.e. the
    # caller's narrow request becomes a FULL, re-billed re-run with no error anywhere.
    # Same hole, same fix, as ``projects._TARGET_DURATION_ALIASES``: accept both
    # spellings, validation-only. Without this the RegenerateScopeError refusal is
    # unreachable for the very client that hit the defect.
    from_stage: str | None = Field(
        default=None, validation_alias=AliasChoices("from_stage", "fromStage")
    )


class RestartPlanReason(BaseModel):
    """Why a stage is reused / re-run. ``code`` is stable (``restart_plan.ReasonCode``);
    ``stage`` names the stage it depends on, ``inputs`` the changed run inputs it
    reads; ``message`` is a plain-English fallback."""

    model_config = _camel
    code: str
    message: str
    stage: str | None = None
    inputs: list[str] = Field(default_factory=list)


class RestartPlanStage(BaseModel):
    """One stage of a restart plan: reused as-is, or re-run — and why."""

    model_config = _camel
    stage_name: str
    action: Literal["reuse", "rerun"]
    reasons: list[RestartPlanReason]
    #: The user edited this stage's output after it completed.
    edited: bool = False


class RestartPlanResponse(BaseModel):
    """What ``POST /runs/{id}/regenerate {}`` would reuse and re-run, right now.

    ``mode="partial"`` reuses every ``reuse`` stage; ``mode="full"`` re-runs every
    stage and ``full_reasons`` says why. ``changed_inputs`` lists the run inputs that
    differ from the ones the run was started with (``live:<x>`` = project data a
    stage reads live, e.g. uploaded footage). ``stages`` is in pipeline order.
    """

    model_config = _camel
    run_id: UUID
    mode: Literal["partial", "full"]
    full_reasons: list[RestartPlanReason]
    changed_inputs: list[str]
    stages: list[RestartPlanStage]


class ReorderClips(BaseModel):
    """Rearrange the clip timeline and re-stitch the video.

    ``stage_name`` is the run's clip-spine stage (the storyboard / gen stage whose
    output holds the ordered clip array). ``ordered_element_ids`` is the full set
    of that stage's element ids in the desired new order — a permutation of the
    current ids (missing, extra, or duplicate ids → 400). The route spawns a
    child run that preserves every upstream stage (spine reordered) and re-runs
    only the compile stage onward, reusing all generated clip assets."""

    model_config = _camel
    stage_name: str
    ordered_element_ids: list[str] = Field(min_length=1)


class ClipTransition(BaseModel):
    """The cut ENTERING a clip, applied on a Studio re-stitch.

    Mirrors the compositor's ``Shot.transition``: ``type`` is the mix/wipe family
    and ``duration`` is the crossfade length in seconds. It rides the spine element
    like the other per-clip edits (``patch_elements_in_output``) and reaches the
    compositor unchanged as a ``{type, duration}`` dict. Describes the incoming cut,
    so the first shot's transition is a no-op (there is no cut before it)."""

    model_config = _camel
    type: Literal["fade", "dissolve", "wipeleft", "slideright"]
    duration: float = Field(gt=0, le=5.0, default=0.5)


class ColorAdjust(BaseModel):
    """Per-clip colour grade applied on a Studio re-stitch.

    Mirrors the compiler's per-clip colour contract: ``brightness`` shifts
    exposure, ``contrast`` / ``saturation`` / ``gamma`` are multiplicative dials
    (``1.0`` = identity), and an optional ``preset`` names a canned look the
    compiler expands into the same param space. It rides the spine element like
    the other per-clip edits (``patch_elements_in_output``) and reaches the
    compiler unchanged as an ``{brightness, contrast, saturation, gamma, preset}``
    dict. All-defaults (or ``null`` on the ``ClipEdit`` field) is an identity
    grade the compiler omits."""

    model_config = _camel
    brightness: float = Field(default=0.0, ge=-1.0, le=1.0)
    contrast: float = Field(default=1.0, ge=0.0, le=2.0)
    saturation: float = Field(default=1.0, ge=0.0, le=3.0)
    gamma: float = Field(default=1.0, ge=0.1, le=10.0)
    preset: Literal["warm", "cool", "bw", "vivid", "cinematic"] | None = None


class ClipAudio(BaseModel):
    """Per-clip audio level on a Studio re-stitch. Applies to the shot's PRIMARY
    audio (narration/clip audio_url) only — NOT the music bed (see MusicEdit).
    All-defaults is a no-op the compiler omits (byte-identical)."""

    model_config = _camel
    gain_db: float = Field(default=0.0, ge=-60, le=12)
    mute: bool = False
    fade_in_s: float = Field(default=0.0, ge=0, le=30)
    fade_out_s: float = Field(default=0.0, ge=0, le=30)


class TextOverlay(BaseModel):
    """A burned-in text title/lower-third layered over a clip on re-stitch.

    Mirrors the compiler's per-clip overlay contract: ``text`` is the string to
    draw, ``position`` picks a nine-grid anchor (an explicit ``x`` / ``y`` pixel
    pair overrides the anchor when both are set), ``font_size`` / ``color`` style
    it, and ``start`` / ``end`` window it in seconds (absent = the whole clip). A
    clip carries a list of these; the compiler draws them in order over the
    graded frame."""

    model_config = _camel
    text: str = Field(min_length=1, max_length=200)
    position: Literal[
        "top-left",
        "top-center",
        "top-right",
        "middle-left",
        "middle-center",
        "middle-right",
        "bottom-left",
        "bottom-center",
        "bottom-right",
    ] = "bottom-center"
    x: int | None = None
    y: int | None = None
    font_size: int = Field(default=48, ge=8, le=200)
    color: str = "white"
    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, gt=0)


class ClipEdit(BaseModel):
    """Per-clip edit overrides merged onto a spine element before re-stitch.

    Every field is optional; only fields the client actually sends are patched
    (the route uses ``model_dump(exclude_unset=True)``), so omitting a field
    leaves that clip's current value untouched. Sending a field as ``null``
    explicitly CLEARS a previously-authored override back to the computed default
    — the override ladder's "unset" step.

    - ``clip_lead_seconds`` — seconds of recorded footage to skip at the head
      (drop the app's blank load-in). Only meaningful for a footage clip; a no-op
      on an image/text-card shot (``assemble_shots._attach_clip_framing``).
    - ``clip_out_seconds`` — the source-timeline OUT-point (seconds from the head
      of the recording) closing the window ``clip_lead_seconds`` opened:
      ``clip_lead_seconds=4.0`` + ``clip_out_seconds=7.5`` shows exactly 4.0s-7.5s
      of the source, which is the only way to cut SEVERAL shots out of ONE long
      recording. Measured on the SOURCE timeline, so it is independent of
      ``speed``. Unlike ``clip_lead_seconds`` (start-anchor only), the compiler
      honours it under BOTH anchors. Must be strictly greater than
      ``clip_lead_seconds`` — the compiler declares ``exclusiveMinimum: 0`` and
      rejects a non-increasing window, so we reject it here as a 422 rather than
      pay for a failed compile.
    - ``clip_anchor`` — ``start`` shows the head (after any lead), ``end`` shows
      the settled tail of the recording. ``auto`` defers to build_shots, which
      resolves the motion-settled hero-frame lead from the clip's
      ``motion_timeline`` (emitting ``start`` + a numeric lead), degrading to the
      default framing when the curve cannot localise a moment.
    - ``caption`` — burned-in caption text for this clip; wins over the computed
      caption and emits even when the deliverable's captions are off. Empty
      string or ``null`` clears the override back to the computed default.
    - ``fit`` — how the clip's visual fills the frame (``contain`` = letterbox,
      ``cover`` = crop-fill, ``scroll`` = pan a tall capture).
    - ``speed`` — playback rate multiplier (``1.0`` = real time, ``2.0`` = 2x fast,
      ``0.5`` = half-speed slow-mo). Orthogonal to the narration-duration reconcile:
      the compositor retimes the visual, it does not shorten the shot's ``seconds``.
    - ``transition`` — the mix/wipe cut ENTERING this clip; ``null`` restores the
      default hard cut.
    - ``adjust`` — a per-clip colour grade (brightness / contrast / saturation /
      gamma + optional look preset); ``null`` restores the ungraded source.
    - ``overlays`` — burned-in text titles/lower-thirds layered over this clip;
      ``null`` or an empty list clears any authored overlays.
    - ``audio`` — a per-clip audio-level edit (gain / mute / fade-in / fade-out)
      on the shot's PRIMARY audio (narration / clip audio); ``null`` restores the
      unmodified audio.
    - ``audio_url`` — an uploaded voiceover (``POST /api/uploads/voiceover``) that
      REPLACES this shot's TTS narration; ``null`` clears it back to TTS. Strictly
      orthogonal to ``audio``: ``audio`` LEVELS whatever the primary audio is,
      ``audio_url`` changes what the primary audio *is* — both may be set and they
      compose.
    - ``audio_seconds`` — the server-probed duration of that upload. It DECLARES
      this shot's length (the compiler ffprobes the audio and freeze-pads the
      visual to it), so it must ride along with ``audio_url``; the upload route is
      the only thing that sets it (never hand-authored)."""

    model_config = _camel
    clip_lead_seconds: float | None = Field(default=None, ge=0)
    # gt=0 (NOT ge=0) — mirrors the compiler's own ``exclusiveMinimum: 0`` on
    # ``Shot.clip_out_seconds``. A zero out-point is a zero-length window, which the
    # compiler would reject after we had already paid for the render.
    clip_out_seconds: float | None = Field(default=None, gt=0)
    clip_anchor: Literal["start", "end", "auto"] | None = None
    caption: str | None = Field(default=None, max_length=500)
    fit: Literal["contain", "cover", "scroll"] | None = None
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    transition: ClipTransition | None = None
    adjust: ColorAdjust | None = None
    overlays: list[TextOverlay] | None = None
    audio: ClipAudio | None = None
    audio_url: str | None = None
    audio_seconds: float | None = Field(default=None, gt=0, le=3600)

    @model_validator(mode="after")
    def _window_must_open_before_it_closes(self) -> ClipEdit:
        # The compiler enforces ``clip_out_seconds > clip_lead_seconds`` (an
        # empty or reversed source window has no frames to show). Catching it at
        # OUR boundary turns a failed compile — minutes of render, an errored run,
        # and a user who has to guess which clip was wrong — into an immediate 422
        # naming the field. Only checked when BOTH are authored: a lone out-point
        # rides against the compiler's own 0.0 lead default, which is always lower.
        if (
            self.clip_out_seconds is not None
            and self.clip_lead_seconds is not None
            and self.clip_out_seconds <= self.clip_lead_seconds
        ):
            raise ValueError(
                "clipOutSeconds must be greater than clipLeadSeconds "
                f"(got out={self.clip_out_seconds}, lead={self.clip_lead_seconds})"
            )
        return self


class MusicEdit(BaseModel):
    """Run-level music-bed override applied on a Studio re-stitch.

    ``track`` is a sound-catalog **music** id (e.g. the built-ins ``corporate_uplift``
    / ``ambient_tech`` / ``jazz_lounge``); ``null`` or ``""`` turns the music bed
    OFF (other sound design — transitions / ambient — is untouched). The remaining
    fields tune the bed; each omitted field keeps the run's current value. Applied
    by merging into the parent run's ``sound_effects`` config and re-stitching, so a
    per-run override never mutates the project brief. An unknown ``track`` (not a
    music id in ``GET /api/music/tracks`` for the caller's tenant — e.g. the track's
    human LABEL instead of its id) is rejected with 422 by ``apply-edits``.

    ``action_reel`` (and any pipeline that times its cut to the bed) handles the
    same payload differently, because it must: the child run re-runs from
    ``music_select`` so the new track's beat grid is fetched and the cut re-snapped
    to it, and the override rides the ``music`` config block rather than
    ``sound_effects`` — merged onto the parent's block, so the keeps-the-current-value
    rule above holds on that product too. On that product ``duck_db`` cannot be applied at all and is
    reported as an ignored field in the run's notes; a falsy ``track`` renders a
    SILENT, evenly-spaced reel and is likewise noted (the run reads ``degraded``)."""

    model_config = _camel
    track: str | None = None
    # Bounds MIRROR `ClipAudio` (this module's other per-render level control) —
    # deliberately not new limits of their own. Every one of these numbers is
    # interpolated straight into an ffmpeg filtergraph (reel: `_build_music_bed`
    # via `_mix_knob`; everything else: the shared sound-design resolver), and
    # `_mix_knob` only rejects NON-FINITE values, so a finite-but-absurd
    # `gain_db: 1e9` reached the compiler as `gain_db: 1000000000.0`.
    gain_db: float | None = Field(default=None, ge=-60, le=12)
    fade_in_s: float | None = Field(default=None, ge=0, le=30)
    fade_out_s: float | None = Field(default=None, ge=0, le=30)
    # How deep the music dips under narration (dB, <=0). 0 = no ducking, more
    # negative = deeper; the compiler maps it to the sidechain-compressor ratio.
    # Floor mirrors `ClipAudio.gain_db`'s -60 dB — the same "silence" bottom.
    duck_db: float | None = Field(default=None, ge=-60, le=0)


class ApplyClipEdits(BaseModel):
    """Apply per-clip edits and/or a music-bed change; re-stitch as a new version.

    ``stage_name`` is the run's clip-spine stage (the storyboard / gen stage whose
    output holds the clip array). ``edits`` maps each edited clip's element id →
    the fields to override (may be empty for a music-only change). ``music`` is an
    optional run-level music override. At least one of ``edits`` / ``music`` must be
    present. The route spawns a child run that preserves every upstream stage
    (spine edited) and re-runs only the compile stage onward, reusing all generated
    clip assets. Clip ids absent from the current timeline are ignored; a non-empty
    edit set with no live id → 400."""

    model_config = _camel
    stage_name: str
    edits: dict[str, ClipEdit] = Field(default_factory=dict)
    music: MusicEdit | None = None

    @model_validator(mode="after")
    def _require_an_edit(self) -> ApplyClipEdits:
        if not self.edits and self.music is None:
            raise ValueError("apply-edits needs at least one clip edit or a music change")
        return self


class RegenerateClip(BaseModel):
    """Regenerate a single clip's image and re-stitch the video.

    ``stage_name`` is the run's clip-spine stage; ``element_id`` is the clip's
    spine element id (``shot.id`` / ``line.id``) that open feedback is pinned to.
    The route spawns a child run that preserves the spine verbatim (siblings
    untouched) and re-runs only the per-clip image stage onward, folding the
    clip's feedback into just that clip's prompt so exactly one image
    regenerates. Falls back to annotation-driven regenerate when the pipeline has
    no per-clip image stage (422)."""

    model_config = _camel
    stage_name: str
    element_id: str


class Renarrate(BaseModel):
    """Re-narrate a single line: new narration text and/or a new voice.

    ``stage_name`` is the run's clip-spine stage; ``element_id`` is the clip's
    spine element id. An omitted ``narration_text`` keeps the line's current text;
    an omitted ``voice`` uses the run's configured voice. At least one must be
    given (otherwise the request is a no-op that would still burn a TTS call).

    The route spawns a child run that re-synthesizes EXACTLY ONE clip and re-runs
    only the compile stage — every sibling clip's audio is reused verbatim. 422 if
    the clip already carries an uploaded voiceover (which would win over the TTS)."""

    model_config = _camel
    stage_name: str
    element_id: str
    narration_text: str | None = None
    voice: str | None = None

    @model_validator(mode="after")
    def _require_a_change(self) -> Renarrate:
        if self.narration_text is None and self.voice is None:
            raise ValueError("renarrate needs narration_text and/or voice")
        return self


class RunSummary(BaseModel):
    model_config = _camel
    id: UUID
    project_id: UUID
    deliverable_id: UUID
    parent_run_id: UUID | None
    blueprint_id: str
    status: RunStatus
    current_stage: str | None
    started_at: datetime.datetime | None
    completed_at: datetime.datetime | None
    created_at: datetime.datetime
    # When this run began waiting at a review checkpoint (``runs.paused_at``),
    # None unless it is paused. Lets every surface say "waiting for your review
    # since ..." instead of showing a parked run as an old, still-working one.
    paused_at: datetime.datetime | None = None
    # WHY it is paused, None unless paused: ``checkpoint`` = a review gate tripped
    # and a person must approve the stage output ("needs review"); ``manual`` =
    # someone pressed Pause; ``unknown`` = paused before migration 0102 recorded it
    # and its rows could not tell (show a plain "paused", never "needs review").
    pause_reason: PauseReason | None = None
    # True if ANY stage completed-with-warnings (see StageOutputResponse.degraded)
    # — distinguishes a clean run from one that finished green but degraded, so a
    # viewer can stop silently treating a partial result as fully successful.
    #
    # It lives on the SUMMARY, not just RunDetail: a run that shipped a broken
    # deliverable used to read as a clean "Completed" in every run LIST and in the
    # live-monitor poll, because only the detail view carried the flag. Defaults
    # to False so the newly-created-child responses (which have no stage rows yet)
    # are unaffected.
    degraded: bool = False
    # Token totals summed over the run's ``run_usage`` rows, so a run LIST (the
    # deliverable page's History) can show tokens without fetching each run's
    # detail. Computed for a whole page in one grouped query; 0 for a run with no
    # usage rows yet (and for just-created child runs).
    input_tokens: int = Field(
        default=0, description="Sum of input (prompt) tokens across the run's usage rows."
    )
    output_tokens: int = Field(
        default=0, description="Sum of output (completion) tokens across the run's usage rows."
    )


class RunFailureDetail(BaseModel):
    """``RunDetail.failure_detail`` — stable, frontend-branchable failure codes."""

    model_config = _camel
    #: ``byo_provider_required`` | ``upstream_auth`` (an upstream refused the call's
    #: credentials/licence — never retried; see ``stages.upstream.UpstreamAuthError``).
    code: str
    #: ``image`` | ``tts`` | ``asr`` for ``byo_provider_required``.
    modality: str | None = None
    #: ``platform`` (our licence/credential — contact support) | ``tenant`` (the
    #: tenant's own provider key — fix it on the Providers page) for ``upstream_auth``.
    scope: str | None = None
    #: ``site_login_failed``: the host the capture could not sign in to
    #: (``app.example.com``) and WHY, as a clause ("the sign-in page did not accept
    #: the saved login"). Structured so the run page never parses ``message``.
    site: str | None = None
    reason: str | None = None
    #: User-facing sentence, e.g. "Configure your own TTS provider to continue".
    message: str


class RunDetail(RunSummary):
    config_snapshot: dict
    pause_after_stages: list[str]
    error: str | None
    # Fault attribution for a failed run — the structured half of ``error``.
    # ``None`` means never classified (every run that predates migration 0066);
    # ``'unknown'`` means classified and unattributable, which is a real answer,
    # not a gap. Defaulted so a run built before the column existed, and every
    # non-failed run, serialize unchanged.
    failure_class: FailureClass | None = None
    # The machine-readable WHAT-TO-DO beside ``error`` (#1106). ``None`` when the
    # failure carried no structured detail (and for every non-failed run). Today
    # one code exists — ``byo_provider_required`` with ``modality`` image|tts|asr:
    # the tenant is licences ``byo_required`` and has no provider of its own.
    failure_detail: RunFailureDetail | None = None
    # The stage a FAILED run failed in (``None`` for every non-failed run). Written by
    # ``RunRepository.mark_failed`` into ``current_stage`` in the same statement as
    # ``status='failed'``; surfaced as its own field so the frontend need not guess it
    # from stage statuses (UI review 2026-10-17, X7).
    failed_stage: str | None = None
    stage_outputs: list[StageOutputResponse]
    # The ONLY authority on which stage may be passed as ``stage_name`` to
    # ``POST /runs/{id}/reorder-clips``. The frontend must NOT re-derive it by
    # inspecting output shape: several stages of one run legitimately carry
    # identically-shaped element arrays (four stages of a ``demo_video_v1`` run
    # emit the same ``lines[]``), and only the blueprint's topological position
    # distinguishes the real reorder spine from its look-alikes — which is why
    # shape-sniffing 422s. ``None`` means the run has no reorderable timeline
    # (audio-only, critique, redesign) OR its blueprint is unknown.
    #
    # ⚠️ WIRE NAME IS ``clip_spine_stage``, NOT ``clipSpineStage``. This class
    # sets ``model_config = _camel`` (inherited from RunSummary), but that name
    # is a LIE: ``_shared._camel`` is
    # ``ConfigDict(populate_by_name=True, str_strip_whitespace=True)`` with NO
    # ``alias_generator``, so this package serialises snake_case — exactly like
    # the sibling ``stage_outputs``/``failure_class`` the frontend already reads.
    # (``src/api/routes/audit.py`` defines its own, genuinely camelising
    # ``_camel``; do not confuse the two.)
    clip_spine_stage: str | None = None
    # The ONLY authority on WHICH stage holds this run's delivered artifact, and
    # the URL of that artifact. Same rationale as ``clip_spine_stage`` above: a
    # client must NOT re-derive it by inspecting output shape. FOUR stages of a
    # ``demo_video_v1`` run publish an ``artifact_url`` (compile, footage
    # critique, encode master, final critique) and only the blueprint's
    # topological position says which one is the file the customer gets — the
    # other three are intermediates that a re-master legitimately supersedes.
    # Binding an assertion to one of those passes green while the shipped master
    # is broken, which is precisely the hole this closes.
    #
    # ``delivered_artifact_stage`` is DECLARED (``DAGBlueprint.delivered_stage``)
    # — it is the stage that OWNS the deliverable for this blueprint, and it does
    # not move as the run progresses. ``delivered_artifact_url`` is RESOLVED,
    # walking the same skip-clean fallback chain the pipeline itself walks
    # (``final_critique._starting_artifact``), so a stage that skipped clean
    # falls through to the artifact upstream of it instead of reporting nothing.
    # The two therefore answer different questions: "who is responsible" and
    # "what is there right now". ``None`` on either means the blueprint declares
    # no artifact (audit, redesign) / is unknown, or nothing has been produced
    # yet — never "go and guess".
    #
    # ⚠️ WIRE NAMES ARE ``delivered_artifact_stage`` / ``delivered_artifact_url``,
    # snake_case, for the same reason spelled out in the ``clip_spine_stage``
    # comment above: ``_camel`` here has NO ``alias_generator``.
    delivered_artifact_stage: str | None = None
    delivered_artifact_url: str | None = None
    usage_total: RunUsageTotal

    @field_validator("config_snapshot", mode="after")
    @classmethod
    def _redact_login_secrets(cls, v: dict) -> dict:
        # The snapshot copies the project brief, login password included (sealed at
        # rest). It is NEVER returned — see src/services/auth_secrets.py.
        return redact_auth_secrets(v)


class StageStatusLite(BaseModel):
    """Per-stage status without the (potentially multi-MB) ``output`` blob.

    The run-monitor's safety-net poll only needs each stage's lifecycle state;
    shipping the full ``StageOutputResponse.output`` JSONB every poll tick
    grew the payload with each completed stage and saturated the browser main
    thread on long runs. This is the trimmed shape it polls instead.
    """

    model_config = _camel
    stage_name: str
    status: StageStatus
    started_at: datetime.datetime | None
    completed_at: datetime.datetime | None
    # See StageOutputResponse.attempts — carried on the lite shape too because
    # this is what the e2e harness prints in its stage table at timeout.
    attempts: int = 0

    # ⚠️ `None` is coerced to 0, and that is not defensive padding — it is the
    # UNFLUSHED-INSTANCE window. The column is `nullable=False, server_default="0"`,
    # so every row IN THE DATABASE has an int. But `server_default` (and a Python
    # `default=`) only fire at INSERT/flush, so a `StageOutput` built in Python and
    # serialised BEFORE flush has `attempts is None` — which `Mapped[int]` says is
    # impossible and `model_validate(..., from_attributes=True)` then rejects with
    # `int_type`. Six action_reel footage_probe tests construct exactly that shape,
    # and so can production code that responds from an un-flushed row.
    @field_validator("attempts", mode="before")
    @classmethod
    def _attempts_defaults_to_zero(cls, v: object) -> object:
        return 0 if v is None else v


class RunStatusLite(BaseModel):
    """Lightweight run status for the live-monitor poll.

    Field names mirror the subset of ``RunDetail`` that the frontend
    ``StateReconciler`` reads (``status``/``current_stage``/``error`` +
    ``stage_outputs`` statuses), so the client reconciler is unchanged — only
    the wire payload shrinks. Omits ``config_snapshot``, ``usage_total`` and
    every stage ``output`` blob.
    """

    model_config = _camel
    id: UUID
    status: RunStatus
    current_stage: str | None
    error: str | None
    stage_outputs: list[StageStatusLite]
    # Same fields as ``RunSummary.paused_at`` / ``pause_reason``.
    paused_at: datetime.datetime | None = None
    pause_reason: PauseReason | None = None
    # Same completed-with-warnings signal RunDetail/RunSummary carry, derived by
    # the SAME rule — the poll is what the live monitor renders from, so without
    # it a degraded run displays as cleanly "Completed" until (if ever) someone
    # opens the detail view. Computed with a SQL predicate over stage_outputs, so
    # the lite payload's no-blob contract is preserved.
    degraded: bool = False
