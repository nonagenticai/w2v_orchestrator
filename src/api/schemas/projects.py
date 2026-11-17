"""Projects + Deliverables + Briefs schemas."""

from __future__ import annotations

import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from src.models import AuthoredLine
from src.services.auth_secrets import (
    redact_auth_secrets,
    validate_auth_block,
    validate_brief_logins,
)
from src.services.blueprints.audit_family import AUDIT_PROJECT_TYPES
from src.services.package_refs import require_uploaded_package_for_logins
from src.services.stages.audiobook.voice_catalog import VOICE_CATALOG
from src.services.stages.website_review.order import validate_website_review_order

from ._shared import (
    CaptionsMode,
    CaptionsModeInput,
    DeliverableFormat,
    DeliverableStatus,
    ProjectStatus,
    ProjectType,
    RunStatus,
    Visibility,
    _camel,
)
from .intent import VideoIntent

#: ``target_duration_seconds`` is the ONE request field the order floor
#: (``MINIMUM_ORDER_SECONDS``, #1024) is computed from, and it is the only field whose
#: ABSENCE is indistinguishable from "no order" — a missing value validates to ``None``
#: and the floor then has nothing to refuse. ``_camel`` is a misnomer: it is
#: ``populate_by_name=True`` with **no** ``alias_generator``, so a camelCase client's
#: ``targetDurationSeconds`` was silently DROPPED (measured 2026-09-16: a 9s order sent
#: as ``targetDurationSeconds`` returned **201 with a null order** instead of the typed
#: 400 — the floor never applied).
#:
#: Decision: ACCEPT BOTH SPELLINGS. Adding ``alias_generator=to_camel`` to ``_camel``
#: would flip the wire format of every schema that shares it (including RESPONSE
#: bodies), so the alias is declared per-field here instead — the same
#: ``alias=``/``populate_by_name`` shape already used by ``intent.py``'s ``target_url``,
#: widened to ``AliasChoices`` so snake_case keeps working. Validation-only: no
#: serialization alias, so response bodies are byte-unchanged.
_TARGET_DURATION_ALIASES = AliasChoices("target_duration_seconds", "targetDurationSeconds")

#: THE FIFTH SPELLING. ``brief`` is a free-form ``dict``, so no pydantic alias reaches
#: inside it — and ``website_review_ordered_seconds`` reads
#: ``brief["duration_seconds"]`` as the fallback order (``order.py:122``), as does
#: ``duration_conformance`` (``:896``). A brief authored as
#: ``{"durationSeconds": 9}`` therefore matched nothing and the order fell back to the
#: 120s default: **201, run started, the user's stated duration silently ignored**.
#: It fails OPEN, so it is not a floor bypass — but a stated duration being dropped is
#: its own defect and it is defect 2's exact shape.
#:
#: Normalised HERE, at the door, rather than in either reader: the brief is persisted
#: verbatim and read by more than one consumer, so fixing a reader leaves the next one
#: broken. Canonicalising one known alias of our own contract key is what
#: ``populate_by_name`` does for typed fields; nothing else in the brief is touched.
_BRIEF_DURATION_CAMEL = "durationSeconds"
_BRIEF_DURATION_SNAKE = "duration_seconds"


def _canonicalise_brief_duration(brief: dict) -> dict:
    """Move a top-level ``durationSeconds`` onto ``duration_seconds``.

    A brief that already carries the canonical key keeps it (an explicit snake_case
    value wins); the camelCase key is removed so a downstream consumer cannot read a
    second, contradicting order.
    """
    if _BRIEF_DURATION_CAMEL not in brief:
        return brief
    value = brief.pop(_BRIEF_DURATION_CAMEL)
    brief.setdefault(_BRIEF_DURATION_SNAKE, value)
    return brief


class DeliverableCreate(BaseModel):
    model_config = _camel
    name: Annotated[str, Field(min_length=1, max_length=200)]
    format: DeliverableFormat
    #: ``None`` (omitted) → the tenant's default captions mode, else ``"none"``;
    #: resolved by the route at creation (``src.services.tenant_defaults``).
    captions_mode: CaptionsModeInput | None = None
    target_duration_seconds: Annotated[
        int | None, Field(ge=1, le=3600, validation_alias=_TARGET_DURATION_ALIASES)
    ] = None


class ShowcaseLine(AuthoredLine):
    """API request contract for one authored clip↔beat: narration bound to a recorded
    ``@clip-<n>`` footage tag.

    Shares the ``AuthoredLine`` field set (single source of truth in
    ``src.models.showcase_line``) with the demo_video stage-output twin
    ``services/stages/demo_video/gen.py::DemoVideoLine`` — so the two can no longer
    drift. This variant adds the request-door strictness the pipeline relies on
    (non-empty narration, ``clip_anchor`` admits ``'end'`` plus ``'auto'``) plus the
    ``_camel`` config; the stage twin stays permissive.

    ``clip_anchor="auto"`` is a content-aware request — build_shots resolves it to
    the motion-settled hero-frame lead (``clip_anchor="start"`` + numeric
    ``clip_lead_seconds``) from the clip's ``motion_timeline`` at compile time; an
    unresolvable curve degrades to the default framing.
    """

    model_config = _camel
    text: Annotated[str, Field(min_length=1)]
    seconds: Annotated[float, Field(ge=0)] = 0.0
    clip_anchor: Literal["end", "auto"] | None = None
    # A22/C6 timeline fields — TIGHTENED at the door only (overrides of inherited
    # fields, so the OpenAPI property order is unchanged). A negative floor/ceiling
    # or a zero/negative speaking rate is never a valid authored value.
    min_seconds: Annotated[float | None, Field(ge=0)] = None
    max_seconds: Annotated[float | None, Field(ge=0)] = None
    vo_rate: Annotated[float | None, Field(gt=0)] = None


class LegalBrief(BaseModel):
    """Typed door-view of ``brief.legal`` — the ``claim_finish`` (A18) opt-in gate.

    ``claim_finish`` wakes ONLY on an explicit ``legal.mode`` of ``auto`` or
    ``critique_only`` (``stages/demo_video/claim_finish.py`` ``_ACTIVE_MODES``);
    anything else silently skips the stage for the whole run. Until now ``mode`` was
    validated NOWHERE, so an operator who typed ``"critique-only"`` (hyphen) or
    ``"Auto"`` got a 201 at the door and a legal gate that never ran — discoverable
    only by reading stage rows after the fact. A present-but-invalid ``mode`` now
    422s here instead.

    ⚠️ ``mode`` is OPTIONAL and the model is ``extra='allow'``, deliberately. The
    ``legal`` block ALREADY ships in prod carrying
    ``{"trademarks": [...], "defensibility": "..."}`` for other consumers and NO
    ``mode`` key; that shape must keep validating (and must keep leaving claim_finish
    inert — the existing, intended opt-in behaviour). Only an explicitly-supplied bad
    ``mode`` is rejected.

    ``mode`` is the ONLY typed field, also deliberately. ``posture`` and the
    ``claims`` sub-block (``allow_superlatives`` / ``on_unsupported`` / ``iterations``
    / ``score_threshold``) ride on ``extra='allow'`` untyped: ``claim_finish`` itself
    defends with ``if not isinstance(claims_cfg, dict): claims_cfg = {}``, which is
    direct evidence that a NON-dict ``legal.claims`` (e.g. a list of claim strings in
    a legal/trademark block) is anticipated in the wild. Typing that sub-block would
    422 such a brief at create — a regression on payloads that are valid today. The
    stage already defaults every claims knob, so a bad one degrades to a default; a
    bad ``mode`` silently disables the whole stage. Only the latter is worth a 422.
    """

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True, extra="allow")
    mode: Literal["auto", "critique_only"] | None = None


class ShowcaseBrief(BaseModel):
    """Typed view of a showcase project's brief.

    Documents + validates the keys the showcase blueprint reads; the persisted shape
    stays the free-form ``brief: dict`` (same contract as ``ProjectResponse.source_url``).
    ``extra='allow'`` keeps the many other brief keys (title_card / ken_burns / dialogue /
    sound_effects / …) round-tripping untouched — only the typed fields are validated.
    """

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True, extra="allow")
    # Required in both modes (the Source step). min_length=1 so an empty/whitespace
    # URL 422s at the door (str_strip_whitespace runs first) instead of surviving to
    # fail three stages deep in gen/record — the failure this typed brief exists to kill.
    target_url: Annotated[str, Field(min_length=1)]
    package_url: str | None = None  # authored mode (from POST /api/uploads/package)
    lines: list[ShowcaseLine] | None = None  # authored spine; presence ⇒ authored mode
    focus_areas: list[str] | None = None  # generated-mode steering
    max_scenarios_per_feature: int | None = None
    # auth_config stays opaque: the stages read brief['auth_config'] raw and the richer
    # models.AuthConfig has a different shape — typing it here would reject valid briefs.
    # A dict is one login; a LIST is one login per origin (a demo that signs into
    # several apps in one recording — marketing via Keycloak, Twenty, the console).
    # Secrets inside are sealed at rest and redacted on read (services/auth_secrets).
    auth_config: dict | list[dict] | None = None
    timeout_per_test_ms: int | None = None
    # Silent, captions-only render: skip TTS, burn each line's text in as a caption, and
    # time unauthored beats by reading speed (services/stages/shared/captions_only.py).
    captions_only: bool | None = None
    # Which same-origin login bdd scaffolds the app with, when several logins share
    # target_url's origin (told apart by `label`); see services/auth_secrets.py.
    login_label: str | None = None
    # C23 design-token + motion-signature author blocks (ORCH-3.1). Kept as OPEN
    # dicts so open-ended palette / signature overrides round-trip without a 422;
    # optional + presence-opt-in (absent ⇒ byte-identical run, no tokens derived).
    design: dict | None = None
    motion: dict | None = None
    # C-cold-open presence-opt-in render switch (ORCH-1.3a). A scalar style tag
    # (`cold_open` | `logo_open` | `auto`), left OPEN/unrestricted for forward-compat
    # — absent ⇒ byte-identical run, no cold-open derived (cold_open_window degrades
    # gracefully on any unknown value). The open `direction.hook` sub-block needs NO
    # schema edit here — it round-trips via the whole-`direction` lift in run_setup.
    open_style: str | None = None
    # A18 claim_finish opt-in gate. Typed ONLY to reject a typo'd `legal.mode` at the
    # door — a bad mode silently skips the whole legal stage for the run (see
    # LegalBrief). Absent `legal`, or a `legal` block with no `mode`, stays valid and
    # leaves claim_finish inert ⇒ byte-identical run.
    legal: LegalBrief | None = None

    @field_validator("auth_config", mode="after")
    @classmethod
    def _valid_logins(cls, v: dict | list[dict] | None) -> dict | list[dict] | None:
        validate_auth_block(v)
        return v

    @model_validator(mode="after")
    def _authored_coherence(self) -> ShowcaseBrief:
        # A hand-authored spine needs footage to play over: lines ⇒ package_url.
        if self.lines and not self.package_url:
            raise ValueError("authored showcase (brief.lines present) requires brief.package_url")
        return self


class AuditBrief(BaseModel):
    """Typed door-view of an ``seo_audit`` / ``cro_audit`` / ``marketing_scorecard`` brief.

    All three audit blueprints scan a single URL, so they share this shape. Same contract
    as ``ShowcaseBrief``: the persisted brief stays the free-form ``brief: dict`` (only
    validated here at the door), and ``extra='allow'`` round-trips every other key.
    """

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True, extra="allow")
    # min_length=1 so an empty/whitespace URL 422s at the door (str_strip_whitespace runs
    # first) instead of surviving to a RuntimeError one stage deep in the scan — the same
    # fail-at-the-door contract ShowcaseBrief.target_url enforces.
    target_url: Annotated[str, Field(min_length=1)]
    include_llm: bool = False
    # Opt into the narrated-video variant (off by default → scan-only deliverable).
    # Previously an undocumented ``extra='allow'`` magic key read only by the video
    # gate; typing it here documents the flag, gives door-level coercion, and surfaces
    # it in the OpenAPI schema so the FE gets a typed brief contract. Zero runtime
    # change — the brief is still persisted un-reassigned; only validated here.
    generate_video: bool = False


class DemoVideoBrief(BaseModel):
    """Typed door-view of a ``demo_video`` brief — ``legal.mode`` ONLY.

    ``claim_finish`` (A18) sits in BOTH the ``showcase`` and the ``demo_video``
    blueprints, so typing ``legal`` on ``ShowcaseBrief`` alone would leave half the
    stage's surface silently skippable on a typo'd mode. This door exists purely to
    close that half.

    Deliberately has NO required field and NO other typed field: ``demo_video`` had no
    door model at all until now, so anything stricter would start 422-ing briefs that
    create fine today. With ``extra='allow'`` + an all-optional ``LegalBrief``, the ONLY
    payload this can reject is an explicitly-supplied-but-invalid ``legal.mode`` — which
    is exactly the bug. Not referenced by any response model, so the OpenAPI response
    contract is unchanged.
    """

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True, extra="allow")
    legal: LegalBrief | None = None


class AppReviewBrief(BaseModel):
    """Typed door-view of an ``app_review`` brief.

    Exists because of a MEASURED production failure, not a hypothetical: on 2026-08-05
    the live orchestrator DB showed ``app_inspect`` at **194 completed / 62 failed**, and
    all 62 failures carried one signature —
    ``RuntimeError: brief.target_url required for app_review``. That is raised by
    ``stages/app_review/inspect.py`` *after* the run has been accepted, dispatched and
    billed, so a missing field burned a paid stage instead of 422-ing at the door.

    Same contract as ``AuditBrief``: the persisted brief stays the free-form
    ``brief: dict`` (only validated here), and ``extra='allow'`` round-trips every other
    key. The stage-level ``RuntimeError`` stays as a backstop but should now be
    unreachable from the create path.

    THREE keys are accepted, not one. ``target_url`` is canonical, but ``source_url`` and
    the legacy ``url`` are a documented, tested input shape for this project type —
    ``ProjectResponse.source_url`` projects from exactly these three
    (``api/routes/projects.py:95``), and ``demo_video``'s stages read the same trio.
    Requiring ``target_url`` alone would 422 a shape the API has always accepted. The
    stage now resolves the same trio, so accepting any one of them is honest end-to-end.
    """

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True, extra="allow")
    # Each is optional ALONE; the model validator below requires at least one. min_length=1
    # so an empty/whitespace URL cannot satisfy the requirement (str_strip_whitespace runs
    # first) and instead 422s here rather than surviving to the stage's RuntimeError.
    target_url: Annotated[str, Field(min_length=1)] | None = None
    source_url: Annotated[str, Field(min_length=1)] | None = None
    url: Annotated[str, Field(min_length=1)] | None = None
    # Advisory context, genuinely optional: the stage degrades each competitor to None on
    # failure. Typed only so a non-list 422s at the door instead of blowing up in
    # `list(brief.get("competitor_urls") or [])`.
    competitor_urls: list[str] = []

    @model_validator(mode="after")
    def _require_a_target(self) -> AppReviewBrief:
        """Reject a brief carrying none of the three accepted target keys.

        This is the whole point of the door: ``stages/app_review/inspect.py`` raises
        ``RuntimeError`` for this case *after* the run has been accepted, dispatched and
        billed. 62 of 256 observed ``app_inspect`` stage rows died exactly that way.
        """
        if not (self.target_url or self.source_url or self.url):
            msg = "brief requires one of target_url / source_url / url for app_review"
            raise ValueError(msg)
        return self


class WebsiteReviewBrief(BaseModel):
    """Typed door-view of a ``website_review`` brief.

    Same contract as ``AuditBrief``: the persisted brief stays the free-form
    ``brief: dict`` (only validated here), and ``extra='allow'`` round-trips every other
    key. Typing a flag here documents it, coerces it at the door, and surfaces it in the
    OpenAPI schema so the FE gets a generated contract.
    """

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True, extra="allow")
    # ⚠️ INERT since 2026-09-15 — accepted, persisted, and IGNORED. It used to arm human
    # review gates after website_redesign and website_review_script; the owner had those
    # gates removed (the UI never announced them, so a gated run read as stalled), and
    # `_seed_pause_after_stages` no longer reads this key for website_review. Neither
    # value changes anything: a website_review always runs straight through.
    #
    # Kept typed, not deleted, because the FE client is GENERATED from this schema and
    # dropping the field would break every caller that still sends it. The description
    # below ships INTO that client, so it must say "no-op" rather than describe gates
    # that are gone.
    pause_for_review: bool = Field(
        default=True,
        description=(
            "No-op. Retained for backwards compatibility: website_review has no review "
            "gates, so a run never pauses regardless of this value."
        ),
    )


# Per-type brief models validated at the create door. The persisted shape stays the
# free-form ``brief: dict``; these only reject malformed input up front instead of
# letting it fail deep in the run.
_BRIEF_DOOR_MODELS: dict[str, type[BaseModel]] = {
    "showcase": ShowcaseBrief,
    # legal.mode-only door — see DemoVideoBrief. demo_video runs claim_finish too.
    "demo_video": DemoVideoBrief,
    # target_url is REQUIRED — 62/62 observed app_inspect failures were its absence.
    "app_review": AppReviewBrief,
    "website_review": WebsiteReviewBrief,
    # Every audit-family type takes the same typed brief door — derived from the
    # single source of truth so a new audit type is picked up without editing here.
    **{_pt: AuditBrief for _pt in AUDIT_PROJECT_TYPES},
}


class Voice(BaseModel):
    """Typed door-view of a project's ``voice`` dict.

    Same contract as ``ShowcaseBrief``: the persisted shape stays the free-form
    ``voice: dict`` (only validated here at the door), and ``extra='allow'``
    round-trips every other key. Its job is to catch an out-of-catalog kokoro
    voice id BEFORE it reaches the TTS backend — the backend 500s on an unknown
    voice (see ``voice_catalog.py``), so today a typo'd ``voice_id`` fails a run
    three stages deep in narrate, and a typo'd ``cast`` entry silently falls back
    to the wrong voice. Rejecting at the door turns both into a clean 422.

    Only kokoro ids are drawn from ``VOICE_CATALOG``; the openai/elevenlabs
    providers use their own voice namespaces, so catalog checks are skipped for
    a non-kokoro ``provider``.
    """

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True, extra="allow")
    provider: str | None = None
    voice_id: str | None = None
    speed: float | None = None
    descriptor: str | None = None
    # Speaker-role → voice_id (showcase/promo dialogue). Audiobook casting uses a
    # separate brief.voice_cast field and is NOT validated here.
    cast: dict[str, str] | None = None

    @model_validator(mode="after")
    def _validate_kokoro_ids(self) -> Voice:
        if (self.provider or "kokoro").lower() != "kokoro":
            return self
        bad: list[str] = []
        if self.voice_id and self.voice_id not in VOICE_CATALOG:
            bad.append(self.voice_id)
        for role, vid in (self.cast or {}).items():
            if vid and vid not in VOICE_CATALOG:
                bad.append(f"{role}→{vid}")
        if bad:
            raise ValueError(
                f"unknown kokoro voice id(s) {bad}: the TTS backend rejects ids "
                f"outside the {len(VOICE_CATALOG)}-voice catalog"
            )
        return self


class ProjectCreate(BaseModel):
    model_config = _camel
    type: ProjectType
    title: Annotated[str, Field(min_length=1, max_length=200)]
    brief: dict = Field(default_factory=dict)
    brand: dict = Field(default_factory=dict)
    voice: dict = Field(default_factory=dict)
    theme: dict = Field(default_factory=dict)
    visibility: Visibility = "tenant"
    parent_project_id: UUID | None = None
    clone_from_project_id: UUID | None = None
    deliverables: Annotated[list[DeliverableCreate], Field(min_length=1, max_length=20)]

    @field_validator("deliverables")
    @classmethod
    def _unique_names(cls, v: list[DeliverableCreate]) -> list[DeliverableCreate]:
        names = [d.name for d in v]
        if len(set(names)) != len(names):
            raise ValueError("deliverables must have unique names")
        return v

    @model_validator(mode="after")
    def _normalise_brief_duration(self) -> ProjectCreate:
        # Before _validate_brief: a typed door model should see the canonical key.
        _canonicalise_brief_duration(self.brief)
        return self

    @model_validator(mode="after")
    def _validate_brief(self) -> ProjectCreate:
        # Validate a typed brief at the API door so a malformed brief 422s here instead of
        # failing one/three stages deep. brief is NOT reassigned — it stays the persisted
        # dict; this only rejects bad input. Types without a door model skip validation.
        model = _BRIEF_DOOR_MODELS.get(self.type)
        if model is not None:
            model.model_validate(self.brief)
        validate_brief_logins(self.brief)
        require_uploaded_package_for_logins(self.brief)
        return self

    def check_website_review_order(self) -> None:
        """Raise ``OrderBelowFloorError`` for a website_review below the order floor.

        Deliberately NOT a ``model_validator``: pydantic wraps a validator's exception
        in a ``ValidationError``, which FastAPI renders as a **422** whose body is a
        validator traceback — the floor and the ordered value never reach the operator.
        ``create_project`` calls this and turns it into the typed **400** the owner
        asked for. The run door (``RunEngine.start_run``) enforces the same floor for a
        deliverable whose order was shortened after creation.
        """
        if self.type != "website_review":
            return
        for d in self.deliverables:
            validate_website_review_order(
                {
                    "brief": self.brief,
                    "deliverable": {"target_duration_seconds": d.target_duration_seconds},
                }
            )

    @model_validator(mode="after")
    def _validate_voice(self) -> ProjectCreate:
        # Reject an out-of-catalog kokoro voice id at the door (see Voice). voice is
        # NOT reassigned — it stays the persisted dict; this only rejects bad input.
        Voice.model_validate(self.voice)
        return self


class FromIntentRequest(BaseModel):
    """The A0 authoring-front-door request (great-video PR 2.2): a declarative
    ``VideoIntent`` + one deliverable's packaging knobs. The orchestrator resolves the
    intent into a computed plan (``resolve_plan``) at run-start and drives the existing
    DAG — the author supplies ~15 lines of intent instead of a hand-tuned brief."""

    model_config = _camel
    intent: VideoIntent
    format: DeliverableFormat = "16x9"
    #: ``None`` (omitted) → the tenant's default captions mode, else ``"none"``.
    captions_mode: CaptionsModeInput | None = None
    target_duration_seconds: Annotated[
        int | None, Field(ge=1, le=3600, validation_alias=_TARGET_DURATION_ALIASES)
    ] = None
    # Deterministic slice: default autopilot (no plan-checkpoint stage yet — that lands
    # with the WM director loop). False seeds the blueprint's declared pause set.
    autopilot: bool = True


class FromIntentResponse(BaseModel):
    """The created project + deliverable + started run, with URLs to inspect the
    computed plan (``config_snapshot['plan_document']``) and stream live progress."""

    model_config = _camel
    project_id: UUID
    deliverable_id: UUID
    run_id: UUID
    plan_preview_url: str
    events_url: str
    pause_after_stages: list[str]


class ProjectDraftCreate(BaseModel):
    """Minimal create-door for the wizard's EARLY draft: a real project row with a
    chosen ``type`` but NO deliverables yet.

    Deliberately kept separate from (and does NOT relax) the strict ``ProjectCreate``
    used by the normal final-submit / n8n path — that one still requires >=1
    deliverable + runs the typed-brief door. A draft has no brief/deliverables yet;
    the wizard adds them + PATCHes + starts runs via the existing endpoints later.

    ``type`` stays REQUIRED so the DB CHECK still enforces a valid project type (an
    unknown value 422s at the Literal door before it ever reaches the DB) — no
    migration or schema change is needed. ``status`` server-defaults to ``drafting``;
    zero deliverables is legal (deliverables are a child table).
    """

    model_config = _camel
    type: ProjectType
    title: Annotated[str, Field(min_length=1, max_length=200)] = "Untitled"
    visibility: Visibility = "tenant"
    #: Seed brand/voice/theme (+ same-type brief defaults) from this project of the
    #: SAME tenant (404 ``clone_source_not_found`` otherwise). See ``services.project_clone``.
    clone_from_project_id: UUID | None = None


class ProjectUpdate(BaseModel):
    model_config = _camel
    title: Annotated[str | None, Field(min_length=1, max_length=200)] = None
    brief: dict | None = None
    brand: dict | None = None
    voice: dict | None = None
    theme: dict | None = None
    status: ProjectStatus | None = None
    #: The draft-finalize door for "Clone settings from": the cloned brand/voice/theme
    #: (+ same-type brief defaults) are merged UNDER this PATCH's own fields — and
    #: under the project's current ones for a field the PATCH omits. Never persisted.
    clone_from_project_id: UUID | None = None

    @model_validator(mode="after")
    def _normalise_brief_duration(self) -> ProjectUpdate:
        if self.brief is not None:
            _canonicalise_brief_duration(self.brief)
        return self

    @model_validator(mode="after")
    def _validate_logins(self) -> ProjectUpdate:
        # Every login URL (and the URL a login is bound to) is strict at the door —
        # see auth_secrets.strict_login_url for the parser-differential it closes.
        if self.brief is not None:
            validate_brief_logins(self.brief)
            require_uploaded_package_for_logins(self.brief)
        return self

    @model_validator(mode="after")
    def _validate_voice(self) -> ProjectUpdate:
        # Same door check as ProjectCreate, but voice is optional on PATCH.
        if self.voice is not None:
            Voice.model_validate(self.voice)
        return self


class DeliverableResponse(BaseModel):
    model_config = _camel
    id: UUID
    project_id: UUID
    name: str
    format: DeliverableFormat
    captions_mode: CaptionsMode
    target_duration_seconds: int | None
    status: DeliverableStatus
    created_at: datetime.datetime
    updated_at: datetime.datetime


class BranchAncestor(BaseModel):
    """One hop of a branch's lineage on the way back to the ultimate root.

    Since branch-of-branch is supported (a fork's link row points at whatever it forked,
    so ``root`` is really the IMMEDIATE parent), a branch can sit several hops below the
    root. ``is_branch`` distinguishes an intermediate parent (itself a branch) from the
    ultimate root (a normal project) so the FE can render the chain and its terminal.
    """

    model_config = _camel
    project_id: UUID
    title: str
    is_branch: bool


class BranchContext(BaseModel):
    """Where-am-I context for a storyboard BRANCH project (migration 0052).

    A branch is a real but HIDDEN project row (``projects.is_branch``); its linkage to
    the root lives solely in ``storyboard_branches``. Without this block a user who
    opens a branch has nothing telling them they are in one and no way back to the
    root — a dead end. Populated on ``GET /api/projects/{id}`` for branch projects
    only; ``null`` for every normal project.

    ``root_project_id`` / ``root_title`` name the IMMEDIATE parent (the merge target and
    the back-navigation link); ``root_title`` is denormalised so the banner can render
    "branch of <parent>" without a second round-trip. ``ancestors`` is the full lineage
    from that immediate parent up to the ultimate root (``ancestors[0]`` == the immediate
    parent, ``ancestors[-1]`` == the root; a single element for a direct branch), so the
    banner can show the whole chain — "branch of B, which is a branch of A".
    """

    model_config = _camel
    branch_id: UUID
    name: str
    status: str  # 'open' | 'merged' | 'abandoned' (mirrors BranchResponse.status)
    root_project_id: UUID
    root_title: str
    ancestors: list[BranchAncestor] = Field(default_factory=list)


class ProjectResponse(BaseModel):
    model_config = _camel
    id: UUID
    tenant_id: UUID
    owner_subject_id: UUID
    visibility: Visibility
    type: ProjectType
    title: str
    status: ProjectStatus
    parent_project_id: UUID | None
    position: int | None
    # True when this project is the hidden child backing a storyboard branch. Branch
    # rows are excluded from the projects LIST but are readable by id, so the FE needs
    # this flag to know it is NOT looking at a normal project. See BranchContext.
    is_branch: bool = False
    # Stable, typed source URL for the project. The persisted shape is still
    # the free-form `brief: dict`, but populating a typed top-level field at
    # serialization time gives the frontend a contract it can rely on without
    # casting `brief` to `Record<string, unknown>`.
    source_url: str | None = None
    # Typed projection of a showcase project's brief (populated only when
    # type=="showcase"), same pattern as source_url: the persisted shape stays the
    # free-form brief: dict, but referencing ShowcaseBrief here pulls it + ShowcaseLine
    # into the OpenAPI schema so the frontend gets a generated contract instead of
    # casting brief to Record<string, unknown>.
    showcase_brief: ShowcaseBrief | None = None
    # Typed projection of an seo_audit / cro_audit brief (populated only when
    # type in {"seo_audit","cro_audit"}), same pattern as showcase_brief: the
    # persisted shape stays the free-form brief: dict, but referencing AuditBrief
    # here pulls it into the OpenAPI schema so the frontend gets a generated
    # contract for the typed flags (target_url / include_llm / generate_video).
    audit_brief: AuditBrief | None = None
    created_at: datetime.datetime
    updated_at: datetime.datetime
    # Set when the project reached 'completed' (all deliverables done); null
    # otherwise. Sortable on the list endpoint.
    finished_at: datetime.datetime | None = None
    # List-view metadata. Defaulted so the create/get paths (which don't supply
    # them) still validate; the list handler populates them per page.
    deliverable_count: int = 0
    last_run_status: RunStatus | None = None
    # Why the latest run is paused (only when ``last_run_status == 'paused'``):
    # ``checkpoint`` ⇒ the card reads "Needs review"; ``manual`` / ``unknown`` ⇒
    # just "Paused". Same vocabulary as ``RunSummary.pause_reason``.
    last_run_pause_reason: Literal["checkpoint", "manual", "unknown"] | None = None
    # Poster for the list view: the most recent completed run's video thumbnail
    # (operator override preferred, else auto cover), rewritten to a browser-
    # reachable /api/runs/{id}/artifact proxy path. Null when no completed run
    # produced one. Populated by the list handler per page.
    thumbnail_url: str | None = None


class ProjectDetail(ProjectResponse):
    brief: dict
    brand: dict
    voice: dict
    theme: dict
    deliverables: list[DeliverableResponse]
    children: list[ProjectResponse]
    open_notes_count: int
    # Non-null ONLY when is_branch is true (the branch↔root link + the root's title).
    # Gives the branch workspace its banner and its way back to the root project.
    branch: BranchContext | None = None

    @field_validator("brief", mode="after")
    @classmethod
    def _redact_login_secrets(cls, v: dict) -> dict:
        # A login password (sealed at rest) is NEVER returned — every construction of
        # this response passes here, so no route can forget it. See auth_secrets.
        return redact_auth_secrets(v)


class ProjectList(BaseModel):
    model_config = _camel
    items: list[ProjectResponse]
    total: int
    limit: int
    offset: int


class CreateChildProject(BaseModel):
    model_config = _camel
    type: ProjectType
    title: Annotated[str, Field(min_length=1, max_length=200)]
    brief: dict = Field(default_factory=dict)
    # Optional branding overrides. When omitted (None), the child inherits the
    # parent's brand / voice / theme so a course's lessons render on-brand by
    # default; an explicit value (even {}) wins over inheritance.
    brand: dict | None = None
    voice: dict | None = None
    theme: dict | None = None
    deliverables: Annotated[list[DeliverableCreate], Field(min_length=1, max_length=20)]

    @model_validator(mode="after")
    def _validate_logins(self) -> CreateChildProject:
        validate_brief_logins(self.brief)
        require_uploaded_package_for_logins(self.brief)
        return self


class ReorderChildren(BaseModel):
    model_config = _camel
    ordered_child_ids: list[UUID]


# ─── Deliverables ────────────────────────────────────────────────────────


class DeliverableUpdate(BaseModel):
    model_config = _camel
    name: Annotated[str | None, Field(min_length=1, max_length=200)] = None
    format: DeliverableFormat | None = None
    target_duration_seconds: Annotated[
        int | None, Field(ge=1, le=3600, validation_alias=_TARGET_DURATION_ALIASES)
    ] = None
    captions_mode: CaptionsModeInput | None = None
