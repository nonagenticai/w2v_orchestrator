"""Canonical authored clip↔beat field set — the single source of truth shared by
the API request contract (``api.schemas.ShowcaseLine``) and the demo_video stage
output twin (``services.stages.demo_video.gen.DemoVideoLine``), so the two can no
longer drift.

Permissive on purpose. The API model tightens these at the request door (non-empty
narration, ``clip_anchor`` restricted to ``'end'``); the stage layer, however,
legitimately constructs looser lines — a footage-only beat with empty ``text``, or a
free-form ``clip_anchor`` carried through from a hand-authored brief — so the shared
base imposes no such constraints. Both layers depend on this module; neither depends
on the other.
"""

from __future__ import annotations

import re
from enum import StrEnum
from hashlib import blake2s
from typing import Any, Literal

from pydantic import BaseModel, field_validator

_WHITESPACE = re.compile(r"\s+")


def _normalize_content(value: str | None) -> str:
    """Casefold + collapse runs of whitespace, so a trivial re-wrap / re-casing of the
    SAME beat is not a different beat. Returns ``""`` for None/blank."""
    return _WHITESPACE.sub(" ", (value or "").strip()).casefold()


def beat_fingerprint(
    *,
    scenario_name: str | None = None,
    gherkin: str | None = None,
    text: str | None = None,
) -> str:
    """A stable CONTENT id for one narration beat — the position-free join key.

    The spine's ``id`` (``clip-<n>``) is a POSITIONAL ORDINAL: it is the scenario's
    document-order tag in the generated bdd suite. Any hint that has to survive a
    regeneration (the structural-relief loop re-runs ``demo_video_gen``, producing a
    fresh suite with a different scenario count/order) therefore cannot key on it —
    ``clip-3`` in the child run is different CONTENT from ``clip-3`` in the parent.
    This digest keys on what the beat *is* instead.

    Precedence — the most stable identifying field wins: the bdd ``scenario_name``
    (regenerated verbatim for an unchanged scenario), else the ``gherkin`` steps, else
    the narration ``text`` (the only field an authored/dialogue beat carries). The
    chosen field is domain-separated into the digest so the same string appearing as a
    scenario name and as narration never collides.

    Deterministic across processes and runs: ``blake2s`` over UTF-8 bytes, no ``hash()``,
    no set iteration, no salt. A beat with NO content-bearing field returns ``""`` — an
    empty fingerprint is never a match, so contentless beats can never collide with each
    other (which a digest of the empty string would have made them do).
    """
    for kind, raw in (("scenario", scenario_name), ("gherkin", gherkin), ("text", text)):
        normalized = _normalize_content(raw)
        if normalized:
            return blake2s(f"{kind}\x00{normalized}".encode(), digest_size=8).hexdigest()
    return ""


def normalize_clip_key(value: str) -> str:
    """Normalize a footage-clip / line join key to the frontend_test_runner's form.

    Scenarios are tagged ``@clip-<n>`` in the recording package, but FTR keys each
    recorded clip by that tag with the leading ``@`` stripped (``clip-<n>``), and
    ``demo_video_record`` matches a line's ``clip``/``id`` against that key verbatim
    (no normalization). So accept either form on input — a hand-authored or
    API-submitted ``@clip-1`` still joins to its footage — by trimming surrounding
    whitespace and a single leading ``@``.
    """
    value = value.strip()
    return value[1:] if value.startswith("@") else value


class AuthoredSplit(BaseModel):
    """An author-pinned two-pane layout for one beat: this line's footage beside a
    SECOND recorded clip (``h`` = left/right, ``v`` = top/bottom).

    ``second_clip`` is a footage-bearing line id / ``@clip-<n>`` tag — never a URL; the
    compiler resolves it (``shared/footage_plans``). A pinned split bypasses the LLM
    footage director for that line and wins over whatever the director or the critique
    nudged."""

    layout: Literal["h", "v"]
    second_clip: str

    @field_validator("second_clip")
    @classmethod
    def _normalize_second_clip(cls, v: str) -> str:
        v = normalize_clip_key(v)
        if not v:
            raise ValueError("split.second_clip must name a footage line")
        return v


class AuthoredLine(BaseModel):
    """One narration beat, optionally bound to a recorded ``@clip-<n>`` footage tag."""

    id: str = ""  # blank → stage falls back to clip-<idx+1> (authored_line_id)
    text: str = ""
    seconds: float = 0.0
    scenario_name: str = ""
    gherkin: str = ""
    # Two-voice dialogue slot ('a'/'b'), populated only when ``brief.dialogue.enabled``;
    # demo_video_narrate reads it to cast each line. On input it may also be an alias
    # (guide/new/…) — the stage normalizes it.
    speaker: str | None = None
    # The recorded footage clip (``@clip-<n>`` tag) this line plays over. Authored mode
    # only — lets several dialogue lines share one footage clip (an exchange over a
    # single beat). Defaults to the line id, so the recorded-footage 1:1 join is
    # unchanged when unset.
    clip: str | None = None
    # Where within the footage clip the shown window sits (compositor Shot.clip_*):
    # ``clip_anchor='end'`` shows the settled tail (skips a slow app load / zoom-in at
    # the head → no leading white); ``clip_lead_seconds`` skips a fixed head then plays
    # forward. Authored/footage only; unset → play from the start unchanged.
    # ``clip_out_seconds`` is the OTHER end of that window: the source-timeline OUT-point
    # (seconds from the head of the recording), so ``clip_lead_seconds=4.0`` +
    # ``clip_out_seconds=7.5`` shows exactly 4.0s-7.5s of the source. That pair is the
    # only way to cut SEVERAL shots out of ONE long recording — without an out-point the
    # window's end is merely implied by ``seconds`` and every shot replays from the same
    # in-point. Measured on the SOURCE timeline, so it is independent of ``speed``, and —
    # unlike ``clip_lead_seconds``, which the compiler honours only under the ``start``
    # anchor — it is honoured under BOTH anchors. Must be greater than
    # ``clip_lead_seconds``; unset → the out-point stays implied (byte-identical).
    clip_anchor: str | None = None
    clip_lead_seconds: float | None = None
    clip_out_seconds: float | None = None
    # Expected on-screen text for this beat's footage clip (O1b). Authored/knowledge-pack
    # only. When set, demo_video_record asserts it via video_analyzer OCR
    # (``required_ocr_text``) so a clip that recorded the WRONG screen (e.g. a
    # "Logging out" interstitial) at the right length is flagged — the OCR half of the
    # v12 fix. Unset → no OCR pass (the sharpness/blank check in O1 still runs).
    expected_text: str | None = None
    # ── Script-doctor per-line overrides (C6). All optional → absent leaves behavior
    # unchanged; pure-Pydantic JSON round-trip, no DB migration. ──
    # Author-pinned: the script doctor must not rewrite this beat's narration.
    locked: bool = False
    # Holistic ship-gate per-shot overrides (A11 PR-4). Author-set; win over the film-critic's
    # COMPUTED findings. Absent -> byte-identical; no DB migration (pure-pydantic JSON round-trip).
    # NOTE: `pin` here is a BOOLEAN ship-gate exemption — DISTINCT from `pin_position: int | None`
    # (~below), which is a retention-pass ORDINAL (a target spine slot), not a re-drive guard.
    pin: bool = False  # NEVER a re-drive target: no ship-gate directive may flag this beat
    continuity_exempt: bool = False  # suppress the ship-gate continuity finding touching this beat
    # The dramatic beat this line plays (e.g. 'hook'/'payoff') — doctor guidance only.
    beat_role: str | None = None
    # A free-form note from the script doctor about this line (rationale / suggestion).
    doctor_note: str | None = None
    # ── Retention pass per-line overrides (C6). Both optional → absent leaves the
    # schema `required` unchanged (["text"]); pure-Pydantic JSON round-trip. ──
    # The retention role this line plays. Enum semantics (guidance only):
    # cold_open | loop_open | loop_payoff | pattern_interrupt | none.
    hook_role: str | None = None
    # Target ordinal position this line should be pinned to in the spine (int).
    pin_position: int | None = None
    # ── Timeline reconciliation per-line fields (A22 / contract C6). All optional →
    # absent leaves `required` (["text"]) and every existing run byte-identical. The
    # `reconcile_timeline` stage (C13) stamps seconds/vo_rate/min_seconds/max_seconds
    # per line; an AUTHORED value always wins over the computed default and over a
    # worker-management proposal. ──
    # Author-pinned duration: this beat's `seconds` is a fixed point — the timeline
    # solver must not stretch or squeeze it when redistributing the budget.
    time_lock: bool | None = None
    # Floor/ceiling the solver must keep this beat's duration within (seconds).
    # Authoring controls: unset → the solver derives them from the narration length.
    min_seconds: float | None = None
    max_seconds: float | None = None
    # COMPUTED OUTPUT of the solver (not an authoring control): the speaking-rate
    # multiplier the TTS must use so this beat's narration fits its allotted `seconds`.
    vo_rate: float | None = None
    # ── Claim-grounding fields (A18 / ORCH-4.4). Both optional → absent leaves the
    # schema `required` (["text"]) and every existing payload validating unchanged;
    # pure-Pydantic JSON round-trip, no DB migration. Stamped by the `claim_finish`
    # stage against the `source_ingest` grounding corpus. ──
    # How this line's factual claims fared against the corpus. Enum semantics:
    #   grounded             — every claim matched a grounding sentence.
    #   softened             — an unsupported claim was hedged/rewritten by claim_finish.
    #   pinned               — AUTHOR-PINNED: claim_finish must NEVER rewrite this line's
    #                          text, whatever the corpus says (it may still annotate).
    #   unsupported_accepted — unsupported, but explicitly let through (no rewrite).
    #   none                 — no factual claim to check (or the gate did not run).
    claim_status: str | None = None
    # Provenance for a `grounded` (or softened-from) line: the grounding sentence /
    # source id from the source_ingest corpus that backs the claim.
    claim_source: str | None = None
    # ── Voice-performance per-line overrides (ORCH-1.4, spec 03 §6+§8.6): all
    # optional → absent leaves the schema unchanged; present → author WINS over the
    # voice_director-computed PerformanceDirective. Pure-Pydantic JSON round-trip,
    # no DB migration. ──
    provider: str | None = None
    delivery: str | None = None
    emphasis: list[int] | None = None
    pause_before_ms: int | None = None
    rate: float | None = None
    ssml: str | None = None
    pronunciations: dict[str, str] | None = None
    # ── Timeline solver output, APPENDED (never inserted beside `vo_rate`): field order
    # here feeds OpenAPI → FE codegen, so a new field goes on the end. Optional ⇒
    # `required` stays ["text"]; pure-Pydantic JSON round-trip, no DB migration. ──
    # COMPUTED OUTPUT of the solver (not an authoring control): how far past `seconds` a
    # FOOTAGE beat's last frame may be held frozen, capped at
    # `time_budget.DEFAULT_FREEZE_PAD_CEILING`. 0.0 on every non-footage / already-fits
    # beat, absent on a spine `reconcile_timeline` never stamped.
    #
    # ⛔ IT IS PART OF THE WINDOW THE RATE WAS CHOSEN AGAINST. `solve_timeline` picks a
    # footage beat's `vo_rate` against `seconds + freeze_pad_seconds`, not against
    # `seconds` — so a consumer that re-fits the line to `seconds` alone re-synthesizes
    # it SLOWER than the solver intended, by up to the whole ceiling (1.5s on a beat).
    # That is why it rides the line instead of staying inside the solver: the fit pass in
    # `demo_video_narrate` cannot reconstruct it from `min_seconds` / `max_seconds`
    # (`max_seconds` is `demand.max + pad`, and `demand.max` is not on the line).
    freeze_pad_seconds: float | None = None
    # COMPUTED OUTPUT of the solver (not an authoring control), APPENDED after
    # `freeze_pad_seconds` for the same reason: field order feeds OpenAPI → FE
    # codegen. An UNDERSPOKEN beat's residual — the seconds its allotment holds
    # that its words cannot fill even at the natural-rate floor
    # (`time_budget.underspoken_hold`, capped at the same
    # `DEFAULT_FREEZE_PAD_CEILING`). 0.0 on every beat that is not underspoken,
    # absent on a spine `reconcile_timeline` never stamped.
    #
    # Declared here (rather than left to pydantic's default `extra="ignore"`) is
    # the whole point: `Allotment.hold_seconds` (time_budget.py) already computes
    # this, but until this field existed a stamped value was silently DROPPED on
    # the way back through `AuthoredLine` — a vacuous stamp that would assert
    # green and deliver nothing. See `shot_assembly._apply_underspoken_hold` for
    # the consumer.
    #
    # ⛔ DISTINCT FROM `freeze_pad_seconds` — do not conflate the two, and never
    # add this into that field. `freeze_pad_seconds` is EXTRA WINDOW GRANTED TO
    # THE VOICE (narrate re-synthesizes to fill it); `hold_seconds` is the
    # opposite direction — the voice's words ran out, so the residual must be
    # spent as a HELD FRAME, never as more speaking time, or the beat drags
    # exactly the way this field exists to stop. See the extended comment on
    # `Allotment.hold_seconds` (time_budget.py) for the full reasoning.
    hold_seconds: float | None = None
    # AUTHORING control, APPENDED last (field order feeds OpenAPI → FE codegen): pins a
    # two-pane split for this beat, so the LLM footage director neither plans nor can
    # override it. Absent → the director decides, byte-identical to before.
    split: AuthoredSplit | None = None

    @field_validator("clip")
    @classmethod
    def _normalize_clip(cls, v: str | None) -> str | None:
        # A user may bind a beat with the ``@clip-1`` tag they see in the .feature;
        # normalize to the ``clip-1`` join key so it matches the recorded footage.
        return normalize_clip_key(v) if v else v


def authored_line_fingerprint(line: AuthoredLine) -> str:
    """:func:`beat_fingerprint` for a model-shaped beat (``AuthoredLine`` or a subclass
    such as ``DemoVideoLine`` / ``StoryboardPanelBase``). The dict-shaped counterpart —
    a beat read back off a persisted stage output — calls ``beat_fingerprint`` directly.
    """
    return beat_fingerprint(scenario_name=line.scenario_name, gherkin=line.gherkin, text=line.text)


# ── Storyboard Phase 0 — industry panel/scene/asset vocabulary (Appendix B) ──
#
# Stored in the DB as ``Text`` + ``CheckConstraint`` (never a native pg ENUM —
# repo convention). These ``str, Enum`` classes exist at the Pydantic door only,
# and the panel fields that use them stay PERMISSIVE (``Enum | str | None``, like
# every other ``AuthoredLine`` field) so a hand-authored or AI-drafted board with
# an out-of-vocabulary value still round-trips instead of raising at the model.


class ShotSize(StrEnum):
    EWS = "EWS"  # extreme wide
    WS = "WS"  # wide
    LS = "LS"  # long (== wide)
    MLS = "MLS"  # medium-long
    MS = "MS"  # medium
    MCU = "MCU"  # medium close-up
    CU = "CU"  # close-up
    ECU = "ECU"  # extreme close-up
    ESTABLISHING = "establishing"
    INSERT = "insert"
    OTS = "OTS"  # over-the-shoulder
    POV = "POV"
    TWO_SHOT = "two_shot"


class CameraAngle(StrEnum):
    EYE_LEVEL = "eye_level"
    HIGH = "high"
    LOW = "low"
    DUTCH = "dutch"
    OVERHEAD = "overhead"  # bird's-eye
    WORM = "worm"


class CameraMovement(StrEnum):
    STATIC = "static"
    PAN = "pan"
    TILT = "tilt"
    DOLLY = "dolly"  # track
    ZOOM = "zoom"
    CRANE = "crane"  # jib
    HANDHELD = "handheld"
    PUSH_IN = "push_in"
    PULL_OUT = "pull_out"


class TransitionOut(StrEnum):
    CUT = "cut"
    DISSOLVE = "dissolve"
    WIPE = "wipe"
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    MATCH_CUT = "match_cut"


class VisualType(StrEnum):
    """Media provenance of a panel's picture (Appendix B)."""

    FOOTAGE = "footage"
    SLIDE = "slide"
    TITLE_CARD = "title_card"
    AI_IMAGE = "ai_image"
    UPLOAD = "upload"


class AssetKind(StrEnum):
    CHARACTER = "character"
    LOCATION = "location"
    PROP = "prop"


class PanelStatus(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"


class MediaRef(BaseModel):
    """Where a panel's picture lives — serialized into the JSONB ``media_ref`` blob.
    All optional so a text-only (Phase-0) panel persists with an empty ``{}``."""

    type: str | None = None  # footage | slide | ai_image | upload | …
    asset_id: str | None = None  # MinIO object id (generated/uploaded)
    # Object IDENTITY, not a durable link: for generated media this is a
    # presigned URL whose signature expires in an hour, so the stored value is
    # re-signed on every read (src/services/storyboard/panel_presign.py). Never
    # hand this string straight to a browser: its ``minio.test`` host does
    # not resolve outside the cluster either. Storyboard panel reads replace it
    # with the panel's image-proxy path (src/api/panel_image_url.py).
    url: str | None = None


class PanelGen(BaseModel):
    """Generation provenance for a panel's picture — serialized into the JSONB
    ``gen`` blob alongside :class:`MediaRef`.

    ``seed`` is the reproducibility contract: the image runner is text→image plus
    a seed (no img2img, no ControlNet, no IP-Adapter), so the prompt and the seed
    together are the ONLY things that can reproduce the picture or keep a
    character on-model across panels. ``media_id`` is the content-addressed id
    the artifact store keyed the object under, so a regenerate with identical
    inputs can find it instead of paying for a second generation.

    ``size`` is the image's MEASURED ``"{width}x{height}"``, read off the
    delivered bytes — not the size that was requested. The two differ: the
    generator overrides the request (a 1024x1024 ask came back 1536x1536 in
    prod), and recording the ask made this field state something the picture
    beside it contradicted. It is ``None`` whenever the bytes never passed
    through this service (a reused stored object, or a provider CDN url served
    straight through) or carried no readable PNG header — unknown, not guessed.
    """

    prompt: str
    seed: int
    model: str | None = None
    provider: str | None = None
    size: str | None = None
    style_token: str | None = None
    asset_ids: list[str] = []
    media_id: str | None = None
    reused: bool = False
    generated_at: str | None = None


class StoryboardPanelBase(AuthoredLine):
    """The single-source panel field set: extends the shipped ``AuthoredLine`` (so
    ``id`` stays the sacred ``clip-<n>`` spine key, normalized) with the industry
    panel fields (Appendix B). Serializes straight into ``brief.storyboard`` and
    feeds the existing seed path. All added fields are optional and PERMISSIVE →
    partial / AI-drafted boards persist. Inherits: id, text, seconds, scenario_name,
    gherkin, speaker, clip, clip_anchor, clip_lead_seconds, clip_out_seconds,
    expected_text.

    Pydantic↔DB name map (applied by the repo layer, WS-B): the inherited ``id``
    (``clip-<n>``) ⇒ column ``shot_key`` (DB ``id`` is a surrogate uuid7);
    ``location`` (asset id) ⇒ column ``location_asset_id``; ``media_ref`` ⇒ JSONB
    ``media_ref``; ``gen`` ⇒ JSONB ``gen``; ``source`` ⇒ its own ``source`` column.
    """

    # ── Meta ──
    scene_id: str | None = None
    shot_number: str | None = None  # e.g. "12A"
    panel_index: int = 0
    status: PanelStatus | str = PanelStatus.DRAFT
    revision: int = 1
    position: int = 0  # board order within the project

    # ── Action / audio ──
    action_description: str | None = None
    dialogue: str | None = None
    vo_text: str | None = None
    sfx_cue: str | None = None
    music_cue: str | None = None

    # ── Visual ──
    image_ref: str | None = None  # convenience mirror of media_ref.asset_id/url
    media_ref: MediaRef | None = None
    visual_type: VisualType | str | None = None
    shot_size: ShotSize | str | None = None
    composition_notes: str | None = None
    characters: list[str] = []  # Asset ids
    location: str | None = None  # Asset id (⇒ column ``location_asset_id``)
    props: list[str] = []  # Asset ids
    lighting_mood: str | None = None
    art_style: str | None = None
    aspect_ratio: str | None = None

    # ── Camera ──
    angle: CameraAngle | str | None = None
    movement: CameraMovement | str | None = None
    movement_path: Any = None  # JSONB keyframe hint for the animatic
    lens: str | None = None

    # ── Timing / transition ──
    duration_seconds: float | None = None
    transition_out: TransitionOut | str | None = None

    # ── Gen provenance (retry/reproducibility) ⇒ JSONB ``gen`` column ──
    gen: dict[str, object] = {}  # holds prompt / seed / model
    source: str | None = None  # ai | user | uploaded
