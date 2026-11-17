"""Pure setup helpers extracted from ``RunEngine.start_run``.

Each is independently unit-testable without a dispatch stub: assemble the run's
``config_snapshot``, fold a course's child lessons into it, and insert the
initial ``stage_output`` rows. ``start_run`` orchestrates these; it no longer
inlines them.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, MutableMapping, Sequence
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid7

from sqlalchemy import func, insert
from w2v_common.auth.identity_context import get_license_id_from_context

from src.auth.run_attribution import snapshot_run_attribution
from src.auth.run_providers import MODEL_TIER_KEY
from src.auth.tenant_provenance import TENANT_SOURCE_KEY, snapshot_tenant_source
from src.config import PipelineSettings, get_settings
from src.database.models import Project, StageOutput
from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.storyboard_repository import StoryboardPanelRepository
from src.database.repositories.verbatim_item_repository import VerbatimItemRepository
from src.services.planning.presets import load_preset
from src.services.planning.resolve_plan import resolve_plan
from src.services.providers.adapters import build_resolver_deps
from src.services.providers.registry import PINNED_SLOT_NAMES
from src.services.providers.resolver import (
    PROVIDER_SLOTS_KEY,
    SlotRef,
    pin_slots,
    pins_to_snapshot,
)
from src.services.runs.run_context import REGENERATE_FROM_STAGE_KEY
from src.services.stages.shared.motion_slides import quality_tier
from src.services.stages.shared.provenance import master_seed
from src.services.stages.shared.retention import resolve_retention_weights
from src.services.tenant_defaults import (
    CAPTIONS_KARAOKE_KEY,
    TENANT_DEFAULTS_KEY,
    ResolvedTenantDefaults,
    merge_block,
    resolve_run_defaults,
    retention_seed_mode,
    tenant_captions_karaoke,
    timing_director_tier,
    voice_direction_tone,
)
from src.utils import coerce_float

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.database.repositories.stage_output_repository import StageOutputRepository
    from src.services.blueprints.base import DAGBlueprint

logger = logging.getLogger(__name__)

#: The advisory (strictly non-mutating) mode every WS-1 default opts into: the stage
#: runs, scores, and reports, but never rewrites the delivered artifact or the spine.
_CRITIQUE_ONLY = "critique_only"

#: Project types whose blueprints carry the terminal ``render_manifest`` audit
#: stage — the only place a recorded provenance ledger is ever surfaced. Recording
#: it for other types would persist an audit record no stage reads.
_PROVENANCE_LEDGER_TYPES = frozenset({"demo_video", "showcase"})

#: Project types whose blueprints wire the ``demo_video_footage_{direction,critique}``
#: stages — the only ones a seeded ``direction.footage`` is meant to activate. Scoped
#: because the block is ALSO read by the site_capture/render path that promo and
#: website_review run: seeding it there silently un-gates VP-2 focus/cursor sidecar
#: extraction + track binding on those blueprints, and turns promo's terminal
#: ``promo_footage_critique`` (``skip_when_unrequested=False``, i.e. a documented
#: pass-through) into a live multi-pass re-render loop. Both are behaviour no author
#: asked for. An explicitly AUTHORED ``direction.footage`` still works everywhere —
#: only the automatic default is narrowed.
_FOOTAGE_DIRECTION_TYPES = frozenset({"demo_video", "showcase"})


def _storyboard_panel_to_dict(panel: Any) -> Mapping[str, Any]:
    """Serialize a persisted ``StoryboardPanel`` row into the ``brief.storyboard``
    panel dict the ``authored_storyboard`` helper consumes.

    Only the fields the no-GPU authored path reads are surfaced: the ``shot_key``
    spine id (== ``clip-<n>``), narration (``vo_text``/``dialogue``/
    ``action_description``), timing (``duration_seconds``), and the ``shot_number``
    label. Visual/camera/gen columns are carried by the persistence layer for later
    phases and are intentionally omitted here. ``speaker``/``expected_text`` and the
    A22/C6 timeline slots (``time_lock``/``min_seconds``/``max_seconds``/``vo_rate``)
    are ``AuthoredLine``-inherited schema fields with no dedicated DB column, so they
    are read defensively via ``getattr`` — a board that gains those columns starts
    feeding ``reconcile_timeline`` with no change here. (Until then the panel API
    contract — ``schemas/storyboard.StoryboardPanelCreate`` — cannot set them; a
    hand-inlined ``brief["storyboard"]`` can.)
    """
    return {
        "shot_key": panel.shot_key,
        "shot_number": panel.shot_number,
        "vo_text": panel.vo_text,
        "dialogue": panel.dialogue,
        "action_description": panel.action_description,
        "duration_seconds": panel.duration_seconds,
        "speaker": getattr(panel, "speaker", None),
        "expected_text": getattr(panel, "expected_text", None),
        "time_lock": getattr(panel, "time_lock", None),
        "min_seconds": getattr(panel, "min_seconds", None),
        "max_seconds": getattr(panel, "max_seconds", None),
        "vo_rate": getattr(panel, "vo_rate", None),
    }


# Project-brief blocks lifted verbatim to the top of the config snapshot. Each is
# presence-is-opt-in: the consuming stage skips a falsy/absent block, so a brief
# without the block (and no per-run override) leaves the key out → byte-identical
# snapshot. A per-run ``config_overrides`` value (spread into the snapshot first)
# always wins — the lift only fills a key the override did not set.
#   ai_image_gen / ai_video_gen / motion_graphics / title_card / ken_burns — visual gen.
#   sound_effects — transitions + accent SFX + music/ambient beds.
#   direction — art-direction knobs.  reproducibility — deterministic-seeding mode.
#   script_doctor — script-doctor critique/revise intent (ORCH-1.2, presence-opt-in).
#   mastering — deterministic final-encode mastering intent (ORCH-0.5).
#   cadence — master frame-rate / retime intent (deterministic compile-tail resolver).
#   design / motion — C23 design-token + motion-signature author blocks (ORCH-3.1).
#   open_style — cold-open presence-opt-in author block (ORCH-1.3a). direction (incl.
#     direction.hook) is already lifted whole above, so no separate key is needed for it.
#: Belt-and-braces against the image model rendering text/UI into a default-seeded
#: still (lesson + generic): the prompts are already wordless. Lessons also keep
#: people out of their abstract slide heroes; a generic film may need them.
_NO_TEXT_NEGATIVE_PROMPT = (
    "text, words, letters, typography, captions, signage, watermark, logo, ui, interface, numbers"
)

_BRIEF_LIFT_KEYS = (
    "ai_image_gen",
    "ai_video_gen",
    "motion_graphics",
    "title_card",
    "ken_burns",
    "sound_effects",
    "direction",
    "script_doctor",
    "reproducibility",
    "legal",
    "mastering",
    "cadence",
    "accessibility",
    "design",
    "motion",
    "open_style",
    # ship_gate — A11 advisory ship-gate intent (presence-opt-in). The final_critique
    # stage skips-clean when this block is absent or `mode == "off"`, so a brief
    # without it (and no per-run override) leaves the key out → byte-identical snapshot.
    "ship_gate",
    # A0 (PR 2.2): the authoring front door's declarative `intent` block. Lifted
    # presence-opt-in like its siblings; consumed by `_apply_intent_plan` below.
    "intent",
)


def _lift_brief_config_keys(
    config_snapshot: MutableMapping[str, Any], brief_cfg: Mapping[str, Any]
) -> None:
    """Fill each ``_BRIEF_LIFT_KEYS`` block from the brief when a truthy value is
    present and no per-run override already set it (mutates ``config_snapshot``)."""
    for key in _BRIEF_LIFT_KEYS:
        if key not in config_snapshot and brief_cfg.get(key):
            config_snapshot[key] = brief_cfg[key]


def _lift_brief_voice_keys(
    config_snapshot: MutableMapping[str, Any], brief_cfg: Mapping[str, Any]
) -> None:
    """Fold the brief's ``voice.{direction,cast}`` sub-blocks INTO the
    already-seeded ``config_snapshot["voice"]`` (from the project row), presence-is-
    opt-in per key — a sub-key already present from the project row always WINS, and a
    brief with no ``voice`` leaves the block byte-identical to ``dict(proj.voice)`` (AC9).
    (``voice.master_lufs`` is NOT folded here — it has no voice consumer; it is aliased
    onto ``mastering.loudness_target`` by :func:`_alias_brief_master_lufs`.)
    Mutates ``config_snapshot``."""
    brief_voice = brief_cfg.get("voice")
    snap_voice = config_snapshot.get("voice")
    if not isinstance(brief_voice, dict) or not isinstance(snap_voice, dict):
        return
    for sub in ("direction", "cast"):
        if sub in brief_voice and sub not in snap_voice:
            snap_voice[sub] = brief_voice[sub]


def _alias_brief_master_lufs(
    config_snapshot: MutableMapping[str, Any], brief_cfg: Mapping[str, Any]
) -> None:
    """Alias a brief-authored ``voice.master_lufs`` onto ``mastering.loudness_target``.

    A brief author who sets ``voice.master_lufs`` means the final-mix integrated-
    loudness target, but every ``voice`` reader consumes only ``cast``/``direction``
    for TTS — so the key had ZERO consumers and was silently swallowed. The LIVE
    loudness path is ``mastering.loudness_target`` → ``build_mastering_profile``, so
    fold the voice knob into the mastering block here.

    Precedence: an explicit ``mastering.loudness_target`` ALWAYS wins. Because
    ``build_mastering_profile`` returns ``None`` when the ``mastering`` block is
    ABSENT, the alias must CREATE the block and assign it back onto the snapshot — a
    ``setdefault`` into a missing dict (or mutating a ``.get(..., {})`` temporary)
    would silently no-op. The block is copied before mutation (copy-on-write: a lifted
    block is the ORM-attached brief dict by reference). Non-numeric values are dropped
    (``coerce_float`` ⇒ ``None``). No clamping here — ``MasteringProfile.__post_init__``
    already clamps ``loudness_target`` to -30..-5. Mutates ``config_snapshot``.

    Called immediately AFTER :func:`_lift_brief_voice_keys` and AFTER
    :func:`_lift_brief_config_keys` (which lifts the ``mastering`` block), so an
    explicit ``mastering`` block is already present when this runs."""
    brief_voice = brief_cfg.get("voice")
    if not isinstance(brief_voice, dict) or "master_lufs" not in brief_voice:
        return
    target = coerce_float(brief_voice["master_lufs"])
    if target is None:
        return
    existing = config_snapshot.get("mastering")
    if isinstance(existing, dict) and "loudness_target" in existing:
        return  # an explicit mastering.loudness_target always wins over the voice alias
    block = dict(existing) if isinstance(existing, dict) else {}
    block["loudness_target"] = target
    config_snapshot["mastering"] = block


def _inject_showcase_grade_default(config_snapshot: MutableMapping[str, Any]) -> None:
    """Seed a default enabled ``finish.grade`` so the colorist runs (ORCH-3.3).

    Called only for a showcase run with the kill-switch on. ``setdefault("enabled")``
    respects an author who explicitly set ``finish.enabled=False`` (grade stays inert);
    an author-supplied ``finish.grade`` is left untouched. Mutates ``config_snapshot``."""
    existing_finish = config_snapshot.get("finish")
    if isinstance(existing_finish, dict):
        if "grade" not in existing_finish:
            finish_block = dict(existing_finish)
            finish_block.setdefault("enabled", True)
            finish_block["grade"] = {}
            config_snapshot["finish"] = finish_block
    else:
        config_snapshot["finish"] = {"enabled": True, "grade": {}}


def _setdefault_leaf(
    config_snapshot: MutableMapping[str, Any], parent: str, leaf: str, value: object
) -> None:
    """``config_snapshot[parent][leaf] = value`` only when that LEAF is absent.

    Leaf-level so a partially-authored parent block keeps every sub-key it authored
    (seeding ``direction.timing`` must never wipe an authored ``direction.retention``).
    Copy-on-write: the parent block was lifted BY REFERENCE off ``proj.brief``, so it
    is replaced with a mutated copy rather than mutated in place — the ORM-attached
    brief dict is never touched. A non-dict authored parent is left alone entirely.
    Mutates ``config_snapshot``."""
    existing = config_snapshot.get(parent)
    if existing is not None and not isinstance(existing, dict):
        return
    if isinstance(existing, dict) and leaf in existing:
        return
    block = dict(existing) if isinstance(existing, dict) else {}
    block[leaf] = value
    config_snapshot[parent] = block


def _seed_footage_critique(config_snapshot: MutableMapping[str, Any]) -> None:
    """Seed ``direction.footage.critique = True`` on an EXISTING footage block.

    Two levels deep, so ``_setdefault_leaf`` cannot express it. Callers must apply the
    ``_FOOTAGE_DIRECTION_TYPES`` gate themselves: this only fires on a footage block
    that already exists, and on promo such a block arrives hand-authored, where
    ``promo_footage_critique`` is the terminal stage with ``skip_when_unrequested=False``
    — injecting ``critique`` there flips a documented pass-through into a multi-pass
    re-render. An author who wants that on promo can still say ``critique: true``.
    Mutates ``config_snapshot``."""
    direction = config_snapshot.get("direction")
    if not isinstance(direction, dict):
        return
    footage = direction.get("footage")
    if not isinstance(footage, dict) or "critique" in footage:
        return
    # Copy BOTH levels — neither dict is necessarily ours to mutate.
    updated = dict(direction)
    updated["footage"] = {**footage, "critique": True}
    config_snapshot["direction"] = updated


def _seed_timing_relief(
    config_snapshot: MutableMapping[str, Any], relief_default: dict[str, object]
) -> None:
    """Seed ``direction.timing.relief`` on an EXISTING ``direction.timing`` block
    that omitted it — independently of whether THIS run created that block.

    ``_setdefault_leaf(config_snapshot, "direction", "timing", timing_default)``
    only fires when the ``timing`` LEAF is absent, so an author who authors ANY
    ``direction.timing`` block (even one that says nothing about ``relief``) wins
    the whole leaf and silently loses the seeded ``relief`` sub-key — the #1006
    re-ask (which shortens over-long narration on an infeasible timeline) never
    arms on such a run, with no error or note. Two levels deep, same shape as
    ``_seed_footage_critique``: seed the grandchild key on whatever ``timing``
    block exists (author-authored or just seeded above), only when it doesn't
    already have one. Mutates ``config_snapshot``."""
    direction = config_snapshot.get("direction")
    if not isinstance(direction, dict):
        return
    timing = direction.get("timing")
    if not isinstance(timing, dict) or "relief" in timing:
        return
    # Copy BOTH levels — neither dict is necessarily ours to mutate.
    updated = dict(direction)
    updated["timing"] = {**timing, "relief": relief_default}
    config_snapshot["direction"] = updated


def _seed_timing_defaults(
    config_snapshot: MutableMapping[str, Any], pipeline: PipelineSettings
) -> None:
    """Seed ``direction.timing`` (mode + opt-in relief), extracted purely to keep
    ``_inject_stage_defaults`` under the C901 complexity ceiling. Same gating and
    behaviour as before the extraction: ``timing_default_enabled`` gates the
    whole block, ``timing_relief_default_enabled`` gates ``relief`` within it —
    seeded on the freshly-created block AND, independently, on an authored
    ``direction.timing`` block that omitted it (see ``_seed_timing_relief``).
    Mutates ``config_snapshot``."""
    if not pipeline.timing_default_enabled:
        return
    timing_default: dict[str, object] = {"mode": "auto"}
    relief_default: dict[str, object] = {"mode": "auto", "iterations": 2, "score_threshold": 4.0}
    if pipeline.timing_relief_default_enabled:
        # Arm the opt-in WM relief loop (shortens over-long narration on an
        # INFEASIBLE timeline) on the default-seeded block.
        timing_default["relief"] = relief_default
    _setdefault_leaf(config_snapshot, "direction", "timing", timing_default)
    if pipeline.timing_relief_default_enabled:
        # ⛔ AND INDEPENDENTLY of the leaf seed above: `_setdefault_leaf` is a
        # no-op the instant an author supplies ANY `direction.timing` block, so
        # an authored block that says nothing about `relief` silently lost the
        # re-ask with no error — see `_seed_timing_relief`. This second call is
        # a no-op on the block this function just seeded (it already carries
        # `relief`) and only does real work on an AUTHORED `direction.timing`.
        _seed_timing_relief(config_snapshot, relief_default)


def _default_ship_gate(pipeline: PipelineSettings) -> dict[str, object]:
    """Build the seeded A11 ``ship_gate`` block, folding in calibrated ceilings.

    Extracted rather than inlined purely to keep ``_inject_stage_defaults`` under the
    C901 complexity ceiling. With no configured ceilings AND enforce disabled this
    returns exactly the historical ``{"mode": critique_only}``, so the default path
    stays byte-identical. ``enforce`` (like ``ceilings``) is folded in only when the
    pipeline default asks for it, so an absent key reads as ``False`` on the gate side.
    """
    block: dict[str, object] = {"mode": _CRITIQUE_ONLY}
    ceilings = pipeline.resolved_ship_gate_ceilings()
    if ceilings:
        block["ceilings"] = ceilings
    if pipeline.ship_gate_default_enforce:
        block["enforce"] = True
    return block


def _inject_stage_defaults(
    config_snapshot: MutableMapping[str, Any],
    pipeline: PipelineSettings,
    tenant_defaults: ResolvedTenantDefaults | None = None,
) -> None:
    """Seed the presence-is-opt-in blocks the eight opt-in stages guard on (WS-1/WS-2).

    Each of those stages raises ``StageSkippedError`` when its config key is absent,
    and nothing in the product ever authored one — so none of them had ever completed
    in production. Every default is behind its OWN kill-switch (see ``PipelineSettings``)
    so one bad stage is killable from the configmap without taking the other seven down.

    Precedence — per-run ``config_overrides`` > brief-authored > this default — falls
    out of ordering plus ``setdefault``: both of the former already landed in
    ``config_snapshot`` before this runs, and injection is a strict no-op on a key that
    already exists. Called AFTER ``_lift_brief_voice_keys`` (so an authored
    ``brief.voice.direction`` still folds in first) and BEFORE the ``retention_weights``
    resolve (so an injected ``direction.retention`` still materializes its weights).

    P6: the tenant's ``retention_mode`` / ``voice_direction.tone`` replace the
    deployment value of the two SEEDED blocks (``src.services.tenant_defaults``);
    a run/brief-authored block still wins (``setdefault``), and the kill-switches stay
    global — an operator turning a default off turns it off for every tenant. A tenant
    ``retention_mode=off`` seeds nothing, so ``retention_plan`` skips.
    Mutates ``config_snapshot``."""
    if pipeline.ship_gate_default_enabled:
        # Calibrated ceilings ride INSIDE the seeded default block, not as a separate
        # leaf injection: an authored/overridden `ship_gate` must stay byte-identical
        # (the calibration scenario overrides this block, and perturbing it would change
        # the very corpus the ceilings are derived from).
        config_snapshot.setdefault("ship_gate", _default_ship_gate(pipeline))
    retention_mode = retention_seed_mode(tenant_defaults)
    if pipeline.retention_default_enabled and retention_mode is not None:
        _setdefault_leaf(config_snapshot, "direction", "retention", {"mode": retention_mode})
    _seed_timing_defaults(config_snapshot, pipeline)
    if pipeline.voice_direction_default_enabled:
        _setdefault_leaf(
            config_snapshot, "voice", "direction", {"tone": voice_direction_tone(tenant_defaults)}
        )
    if (
        pipeline.footage_direction_default_enabled
        and config_snapshot.get("type") in _FOOTAGE_DIRECTION_TYPES
    ):
        # The guard is a plain truthiness check on `direction.footage`, so the seeded
        # block must be non-empty. `enabled` is inert filler (no consumer reads it) —
        # the critique sub-flag below is what actually carries meaning. `auto_zoom` is
        # seeded WITH it because the punch-ins this block un-gates are otherwise cropped
        # out of a 1x capture: site_capture only raises `capture_scale` to 2.0 when
        # `direction.footage.auto_zoom` is truthy, so seeding the block alone would ship
        # visibly upscaled pixels. Type-scoped — see `_FOOTAGE_DIRECTION_TYPES`.
        _setdefault_leaf(
            config_snapshot, "direction", "footage", {"enabled": True, "auto_zoom": True}
        )
    if (
        pipeline.footage_critique_default_enabled
        and config_snapshot.get("type") in _FOOTAGE_DIRECTION_TYPES
    ):
        _seed_footage_critique(config_snapshot)
    if pipeline.claim_finish_default_enabled:
        _setdefault_leaf(config_snapshot, "legal", "mode", _CRITIQUE_ONLY)
    if pipeline.script_doctor_default_enabled:
        # TOP-LEVEL key (not a `direction.*` leaf), so a plain `setdefault` — an
        # authored `script_doctor` block, however partial, comes through untouched.
        # The stage's guard is a truthiness check on `.enabled` (not on the block's
        # presence), so the seeded block MUST carry `enabled: True` — an empty dict
        # here would leave the stage reachable-but-permanently-skipped, which is the
        # exact stranded-stage defect this seeding exists to close.
        config_snapshot.setdefault("script_doctor", {"enabled": True})


def _pin_tenant_defaults(
    config_snapshot: MutableMapping[str, Any], tenant_defaults: ResolvedTenantDefaults | None
) -> None:
    """Record the pinned tenant-defaults row and the tenant's karaoke choice.

    Both keys are added ONLY when the tenant has a row (resp. set karaoke), so a
    tenant without defaults gets a byte-identical snapshot. A run override of
    ``captions_karaoke`` (already spread in) wins over the tenant's."""
    if tenant_defaults is None:
        return
    config_snapshot[TENANT_DEFAULTS_KEY] = tenant_defaults.to_snapshot()
    karaoke = tenant_captions_karaoke(tenant_defaults)
    if karaoke is not None:
        config_snapshot.setdefault(CAPTIONS_KARAOKE_KEY, karaoke)


def _apply_intent_plan(config_snapshot: MutableMapping[str, Any]) -> bool:
    """A0 (PR 2.2): resolve the authoring-front-door ``intent`` block into a
    ``PlanDocument`` and merge its computed knobs INTO ``config_snapshot["brief"]``
    presence-opt-in — an authored brief value always WINS per knob. Runs BEFORE the
    ``_lift_brief_config_keys`` pass so every computed knob (audio gains, resolution,
    voice cast, and the footage-bound ``lines``) flows to the top level through the
    SAME lift machinery a hand-authored brief uses.

    Reads ``intent`` from a per-run override (spread above) or the brief. Materializes
    footage-bound ``lines`` ONLY when a ``package_url`` resolves (the resolver enforces
    this): with no footage ``lines`` stays absent so the existing generated pipeline
    builds footage from ``target_url`` — emitting authored lines against an empty
    ``package_url`` would die stage-deep at ``demo_video_record``. Stashes the full plan
    (with per-knob provenance) at ``config_snapshot["plan_document"]`` for audit.

    Returns ``True`` when a plan was applied. Absent ``intent`` (or the kill-switch
    off) ⇒ ``False`` and no mutation ⇒ byte-identical snapshot (AC-1). The settings
    read is gated behind the intent-presence check so a run WITHOUT an intent never
    touches the ``planning_from_intent_enabled`` flag.
    """
    brief = config_snapshot.get("brief")
    if not isinstance(brief, dict):
        return False
    intent_cfg = config_snapshot.get("intent") or brief.get("intent")
    if not (isinstance(intent_cfg, dict) and intent_cfg):
        return False
    if not get_settings().pipeline.planning_from_intent_enabled:
        return False
    plan = resolve_plan(
        intent_cfg,
        brief=brief,
        preset=load_preset(str(intent_cfg.get("preset", "default"))),
    )
    for key, value in plan.to_brief_patch().items():
        existing = brief.get(key)
        if key == "sound_effects" and isinstance(existing, dict) and isinstance(value, dict):
            # Per-knob presence-opt-in: an authored gain sub-key always wins. Build a
            # NEW dict rather than mutating `existing` in place — `config_snapshot["brief"]`
            # is only a SHALLOW copy of `proj.brief`, so its nested dicts are shared with
            # the persisted ORM instance; an in-place edit would corrupt `proj.brief`.
            brief[key] = {**value, **existing}
        elif not existing:
            brief[key] = value
    config_snapshot["plan_document"] = plan.model_dump(mode="json")
    return True


def _record_provenance_ledger(config_snapshot: MutableMapping[str, Any]) -> None:
    """Record an AUDIT-ONLY provenance ledger for a NON-intent demo_video/showcase run.

    Companion to :func:`_apply_intent_plan` for the ordinary (no-``intent``) path.
    Resolves the authored brief against an EMPTY intent so every knob reads its
    override (brief-authored) or ``default`` rung, and writes ONLY
    ``config_snapshot["plan_document"]`` — the pure audit record ``render_manifest``
    reads. It NEVER touches ``config_snapshot["brief"]`` (the behavioural bridge
    :func:`_apply_intent_plan` uses via ``PlanDocument.to_brief_patch``), so the
    delivered video is byte-identical; only a new audit key appears in the snapshot.

    Scoped to types whose blueprints actually carry ``render_manifest``
    (:data:`_PROVENANCE_LEDGER_TYPES`) so unrelated runs don't accrue a ledger no
    stage surfaces. Best-effort: a resolution failure is swallowed (logged) because
    an audit artifact must never break a run that would otherwise deliver.
    Kill-switch: ``pipeline.provenance_ledger_universal_enabled``.
    """
    if not get_settings().pipeline.provenance_ledger_universal_enabled:
        return
    if config_snapshot.get("type") not in _PROVENANCE_LEDGER_TYPES:
        return
    brief = config_snapshot.get("brief")
    if not isinstance(brief, dict):
        return
    try:
        plan = resolve_plan({}, brief=brief, preset=load_preset("default"))
    except Exception:  # audit must never break a would-be-successful run
        logger.warning("provenance ledger recording failed; continuing without it", exc_info=True)
        return
    config_snapshot["plan_document"] = plan.model_dump(mode="json")


async def build_config_snapshot(
    session: AsyncSession,
    *,
    proj: Any,
    deliv: Any,
    tenant_id: UUID,
    subject_id: UUID,
    config_overrides: Mapping[str, Any] | None,
    inherit_slots: Mapping[str, Any] | None = None,
    inherit_defaults_from: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the run config snapshot from the project + deliverable.

    The single assembler of what a run's stages read: ``RunEngine.start_run``
    persists it, and the failed-parent resume check re-derives it to decide whether
    the order changed (see :func:`order_fingerprint`).

    Includes the project brief/brand/voice/theme, the deliverable descriptor,
    the project's verbatim items, and the AI-visual config lifted from the brief
    (an explicit per-run override wins over the project default).

    Provider pins (P3): ``provider_slots`` + ``model_tier`` are written LAST and can
    never come from ``config_overrides`` (stripped) — see :func:`_pin_run_providers`.
    ``inherit_slots`` is the PARENT's ``provider_slots`` for a system-spawned child.

    Tenant defaults (P6): ``inherit_defaults_from`` is the PARENT run's snapshot for
    any run spawned by a run — the child pins its parent's tenant defaults (or none,
    when the parent pinned none) instead of reading the tenant's current row.
    """
    config_overrides = {k: v for k, v in (config_overrides or {}).items() if k not in _RUN_PIN_KEYS}
    # P6: the tenant's defaults, read ONCE here and folded into the snapshot, so the run
    # keeps them for its whole life. Precedence (src.services.tenant_defaults):
    # run override / project / brief > tenant default > deployment default.
    # A run spawned by a run passes its parent's snapshot and inherits its pin.
    tenant_defaults = await resolve_run_defaults(session, tenant_id, inherit_defaults_from)
    config_snapshot: dict[str, Any] = {
        "project_id": str(proj.id),
        "tenant_id": str(tenant_id),
        "subject_id": str(subject_id),
        # Captured from the request's identity ContextVar so the background
        # dispatcher can rebind the full three-piece identity (tenant +
        # license + subject) on its S2S calls. Empty for runs created before
        # this field existed — the dispatcher then binds tenant + subject only.
        "license_id": get_license_id_from_context() or "",
        # HOW that tenant was established. Captured HERE because this is the
        # only place it is still bound: the background dispatcher has no
        # inbound request, and w2v-common 0.67.0 REFUSES to delegate a tenant
        # of unrecorded provenance on a metered hop (every LLM call). Omitting
        # it is not a type error and not a lint error — it is an
        # UnvouchedTenantError at runtime, on the first worker_management call
        # of every background run.
        TENANT_SOURCE_KEY: snapshot_tenant_source(),
        # The caller's ACTOR + fr project (w2v-common 0.88): re-bound by the
        # dispatcher so a background run bills the caller, not the orchestrator's
        # own SERVICE_ACTOR_ID. See src/auth/run_attribution.py.
        **snapshot_run_attribution(),
        "title": proj.title,
        "type": proj.type,
        "brief": dict(proj.brief),
        "brand": merge_block(proj.brand, tenant_defaults.brand if tenant_defaults else None),
        "voice": merge_block(proj.voice, tenant_defaults.voice if tenant_defaults else None),
        "theme": dict(proj.theme),
        "deliverable": {
            "id": str(deliv.id),
            "name": deliv.name,
            "format": deliv.format,
            "captions_mode": deliv.captions_mode,
            "target_duration_seconds": deliv.target_duration_seconds,
        },
        **(config_overrides or {}),
    }
    _pin_tenant_defaults(config_snapshot, tenant_defaults)

    verbatim_rows = await VerbatimItemRepository(session).list_for_project(
        proj.id, tenant_id=tenant_id
    )
    # The snapshot is the ONLY source of verbatim for stages — no stage queries the
    # repository — so it must carry the discriminators stages actually read.
    # Dropping `kind` here is what made a slide-verbatim phrase get hard-asserted
    # against the NARRATION script (assert_verbatim_present → StageError → run failed),
    # i.e. a tester using the two-track UI as designed could hard-fail their own run.
    # `id` is stringified so the snapshot stays JSON-serializable; it identifies the
    # item in the drop warning stages emit. `scope`/`scope_target` are deliberately
    # NOT carried: nothing in src/ reads them outside the CRUD route, and a
    # half-implemented payload invites the next author to assume it works. Add them
    # here in the same commit as their first real consumer.
    config_snapshot["verbatim_items"] = [
        {
            "id": str(v.id),
            "text": v.text,
            "position": v.position,
            "kind": v.kind,
        }
        for v in verbatim_rows
    ]

    # Fold the project's PERSISTED storyboard into brief.storyboard so the authored
    # short-circuits (generic script_draft, demo_video gen) render FROM the approved
    # board. Persisted WINS over any inline brief.storyboard: only overwrite when rows
    # exist, so a project with no persisted board keeps its inline value (if any) and a
    # project with neither leaves the key absent → byte-identical legacy snapshot /
    # render path. Same session + tenant_id (same RLS txn) as the verbatim load above.
    storyboard_rows = await StoryboardPanelRepository(session).list_for_project(
        proj.id, tenant_id=tenant_id
    )
    if storyboard_rows:
        # Approval gate: if ANY panel in the board is approved, seed ONLY the
        # approved panels; otherwise seed every panel (byte-identical legacy
        # behavior for boards that never approve). Ordering (position/created_at)
        # is already applied by the repo, so filtering preserves it.
        approved = [p for p in storyboard_rows if getattr(p, "status", None) == "approved"]
        seed_rows = approved or storyboard_rows
        config_snapshot["brief"]["storyboard"] = [
            _storyboard_panel_to_dict(panel) for panel in seed_rows
        ]

    # Lift the presence-is-opt-in config blocks (AI-visual, sound, art-direction,
    # reproducibility) from the project brief to the top level a per-run override
    # already spread into config_snapshot wins over. Factored into a helper so this
    # assembler stays under the cyclomatic-complexity gate as more blocks accrue.
    brief_cfg = proj.brief or {}
    # A0 (PR 2.2): materialize the authoring-front-door `intent` into the brief BEFORE
    # the lift, so the computed knobs flow to the top level through the same machinery.
    # When a plan is applied, the lifts read from the seeded brief copy (a superset of
    # proj.brief). Absent `intent` (or the kill-switch off) ⇒ no-op ⇒ byte-identical.
    if _apply_intent_plan(config_snapshot):
        brief_cfg = config_snapshot["brief"]
    else:
        # No intent ⇒ no behavioural brief-merge; still record an audit-only
        # provenance ledger (byte-identical — writes plan_document, not brief).
        _record_provenance_ledger(config_snapshot)
    _lift_brief_config_keys(config_snapshot, brief_cfg)

    # Voice-performance config lift (ORCH-1.4). The project row already seeded
    # `config_snapshot["voice"]` unconditionally above; fold the brief's
    # `voice.{direction,cast}` sub-blocks INTO it (see helper).
    _lift_brief_voice_keys(config_snapshot, brief_cfg)

    # `voice.master_lufs` has no voice consumer — alias it onto the LIVE loudness
    # path (`mastering.loudness_target` → `build_mastering_profile`). Runs AFTER
    # `_lift_brief_config_keys` above, so an explicit `mastering` block is already
    # lifted and wins; creates the block when absent (see helper).
    _alias_brief_master_lufs(config_snapshot, brief_cfg)

    # WS-1: activate the seven opt-in stages that had never once completed in prod
    # (0 rows in live `stage_outputs`) because nothing ever authored the config block
    # each of them guards on. Deliberately placed HERE: after both brief lifts (so an
    # authored block always wins) and before the `retention_weights` resolve below (so
    # an injected `direction.retention` still materializes its weights). Each default
    # has its own kill-switch; all seven off ⇒ byte-identical snapshot.
    _inject_stage_defaults(config_snapshot, get_settings().pipeline, tenant_defaults)

    # ORCH-4.1 T1: turn sound ON by default for showcase runs. A showcase brief
    # that never authors a `sound_effects` block (and no per-run override — the
    # lift above already left the key absent in that case) would otherwise
    # render silent, since `sound_effects` is presence-is-opt-in. Only fills the
    # block when the project type is "showcase" AND the key is still absent —
    # an authored/overridden block always wins outright. `music` is
    # deliberately omitted so `resolve_audio_bed` falls through to the
    # music_supervisor's deterministic pick rather than a hardcoded track.
    # Kill-switch: PIPELINE_SOUND_DEFAULT_ON_SHOWCASE=false, no code deploy.
    if (
        config_snapshot["type"] == "showcase"
        and "sound_effects" not in config_snapshot
        and get_settings().pipeline.sound_default_on_showcase
    ):
        config_snapshot["sound_effects"] = {"enabled": True, "ambient": True}

    # When reproducibility is active (a block is present AND its mode != "off"),
    # stamp a deterministic master seed derived from a stable project field so
    # every downstream subseed is reproducible across re-runs. Copy the block
    # before mutating so the source brief/override dict is never touched. Absent
    # block (or mode == "off") ⇒ nothing added ⇒ byte-identical snapshot.
    repro = config_snapshot.get("reproducibility")
    if isinstance(repro, dict) and repro.get("mode") != "off":
        repro = dict(repro)
        repro["master_seed"] = master_seed(config_snapshot)
        config_snapshot["reproducibility"] = repro

    # Retention direction weights (ORCH-0.3). Presence-is-opt-in like the sibling
    # lifts above: only when the author supplied a `direction.retention` block do
    # we resolve its pacing / 31s-cliff thresholds (via resolve_with_provenance,
    # inside resolve_retention_weights) into a flat top-level `retention_weights`
    # dict that the advisory growth_bundle retention insights read. `direction`
    # itself was already lifted whole by the prov_key loop above; this only adds
    # the resolved-weights convenience block. Absent/empty block ⇒ key never added
    # ⇒ byte-identical snapshot.
    direction_cfg = config_snapshot.get("direction")
    retention_cfg = direction_cfg.get("retention") if isinstance(direction_cfg, dict) else None
    if isinstance(retention_cfg, dict) and retention_cfg:
        config_snapshot["retention_weights"] = resolve_retention_weights(retention_cfg)

    # Output resolution tier (1080p|2k|4k) for the whole render chain — motion
    # clips, slide PNGs, and the terminal compile canvas all scale off this one
    # value so the final video is sharp end-to-end (not a 1080p upscale). A
    # per-run override wins (spread above); else the project brief; else the
    # default. Normalized via quality_tier so an unknown/missing value resolves
    # to the default tier rather than silently dropping to 1080p.
    if "resolution" not in config_snapshot and brief_cfg.get("resolution"):
        config_snapshot["resolution"] = brief_cfg["resolution"]
    config_snapshot["resolution"] = quality_tier(config_snapshot)

    # Color-finish config lift (C5.b `finish` block). Same presence-is-opt-in
    # contract as the visual keys above: a brief without a `finish` block (and no
    # per-run override) never adds the key, so the snapshot stays byte-identical
    # and `build_finish_spec` returns None — no `finish` block on the compile
    # request. The brand letterbox is therefore OPT-IN: supply a `finish` block and
    # `build_finish_spec` resolves `letterbox_color` from `brand.primary_color`.
    if "finish" not in config_snapshot and brief_cfg.get("finish"):
        config_snapshot["finish"] = brief_cfg["finish"]

    # ORCH-3.3 whole-film colorist default (kill-switch, OFF by default). For a
    # showcase run, when PIPELINE_GRADE_DEFAULT_ON_SHOWCASE is on AND the brief did
    # NOT author its own `finish.grade`, inject a default enabled `finish` block so
    # `build_finish_spec` runs the deterministic colorist. `setdefault("enabled")`
    # respects an author who explicitly set `finish.enabled=False` (the grade then
    # stays inert). Non-showcase, flag-off, or an author-supplied `finish.grade`
    # leave the snapshot untouched → byte-identical.
    if proj.type == "showcase" and get_settings().pipeline.grade_default_on_showcase:
        _inject_showcase_grade_default(config_snapshot)

    _seed_default_ai_image_gen(config_snapshot, proj=proj)
    await _fold_course_lessons(session, config_snapshot, proj=proj, tenant_id=tenant_id)
    await _pin_run_providers(
        session,
        config_snapshot,
        tenant_id=tenant_id,
        inherit_slots=inherit_slots,
        tenant_defaults=tenant_defaults,
    )
    return config_snapshot


def _seed_default_ai_image_gen(config_snapshot: MutableMapping[str, Any], *, proj: Any) -> None:
    """Default-seed ``ai_image_gen`` for the types whose films need a still per beat.

    Mutates ``config_snapshot``. Never clobbers a lifted brief / override block.
    """
    # Lessons get an on-brand hero image per slide by default: lesson_slides
    # emits one prompt per slide and ai_image_gen renders them. A project brief
    # or per-run override still wins (handled above), so this only fills the
    # default — it never clobbers an explicit config.
    if proj.type == "lesson" and "ai_image_gen" not in config_snapshot:
        config_snapshot["ai_image_gen"] = {
            "prompts_from_stage": "lesson_slides",
            # Belt-and-braces against the image model rendering text/UI into the
            # picture (the prompts are already wordless + abstract).
            "negative_prompt": f"{_NO_TEXT_NEGATIVE_PROMPT}, people, faces",
        }
    # Generic films get a generated still per shot by default, through the SAME
    # ai_image_gen → ACG path lessons use: storyboard_draft emits one wordless
    # prompt per narration line. Without this the stage skipped ("no ai_image_gen
    # config block") and every shot of a generic film was a text card (live run
    # 01a156a4-e906, 2026-10-20). Brief / override still win (lifted above), and an
    # explicit falsy `brief.ai_image_gen` (e.g. `false`) opts the run out.
    if (
        proj.type == "generic"
        and "ai_image_gen" not in config_snapshot
        and "ai_image_gen" not in (proj.brief or {})
    ):
        config_snapshot["ai_image_gen"] = {
            "prompts_from_stage": "storyboard_draft",
            "negative_prompt": _NO_TEXT_NEGATIVE_PROMPT,
        }


#: Snapshot keys only the server writes: the run's provider pins (P3) and the pinned
#: tenant-defaults version (P6).
_RUN_PIN_KEYS: frozenset[str] = frozenset({PROVIDER_SLOTS_KEY, MODEL_TIER_KEY, TENANT_DEFAULTS_KEY})


async def _pin_run_providers(
    session: AsyncSession,
    config_snapshot: MutableMapping[str, Any],
    *,
    tenant_id: UUID,
    inherit_slots: Mapping[str, Any] | None,
    tenant_defaults: ResolvedTenantDefaults | None = None,
) -> None:
    """Pin every provider slot + the per-stage model tiers for this run.

    ``model_tier.timing_director``: the tenant's tier alias, else the deployment env.

    Written AFTER everything else (incl. the ``config_overrides`` spread, from which
    both keys are stripped) so no caller can choose another tenant's connection or a
    tier. A system-spawned child passes its parent's ``provider_slots`` and keeps
    them (a paused course resumed next week must not have its lessons switch
    provider); slots the parent never pinned (legacy parent, registry growth) are
    pinned fresh. Each inherited entry is validated so a malformed parent pin fails
    the child at start, not mid-run.
    """
    inherited: dict[str, Any] = {}
    if inherit_slots:
        for slot, entry in inherit_slots.items():
            if slot in PINNED_SLOT_NAMES and isinstance(entry, Mapping):
                inherited[slot] = SlotRef.from_dict(slot, entry).to_dict()
    pins: dict[str, Any] = {}
    if any(slot not in inherited for slot in PINNED_SLOT_NAMES):
        deps = build_resolver_deps(session, get_settings())
        pins = pins_to_snapshot(
            await pin_slots(deps.repo, tenant_id, policy=deps.policy, posture=deps.posture)
        )
    config_snapshot[PROVIDER_SLOTS_KEY] = {**pins, **inherited}
    config_snapshot[MODEL_TIER_KEY] = {
        "timing_director": timing_director_tier(
            tenant_defaults, get_settings().pipeline.timing_director_model_tier
        ),
    }


#: Snapshot keys that vary per REQUEST or per RUN rather than per order, dropped
#: before comparing a would-be child snapshot to its parent's. Everything else —
#: brief (incl. the folded storyboard), brand/voice/theme, deliverable (incl.
#: ``target_duration_seconds``), verbatim items, lifted/injected config — is kept.
RUN_SCOPED_SNAPSHOT_KEYS: frozenset[str] = frozenset(
    {
        # WHO clicked: a teammate restarting the run is a different subject. Read
        # only by the dispatcher to rebind S2S identity, never by a stage's content.
        "subject_id",
        # The caller's license from the request ContextVar (rotates; empty on
        # pre-field runs). Dispatcher identity rebinding only.
        "license_id",
        # HOW the tenant was authenticated on this request (header vs cookie).
        # Dispatcher provenance only.
        TENANT_SOURCE_KEY,
        # Folded from the PARENT run's open annotations by start_run, not from the
        # project; a resume check already refuses when annotations are open, and a
        # parent's own notes are already baked into the outputs being preserved.
        "regenerate_notes",
        # WHICH stage the caller explicitly re-ran from — a property of this child
        # run's request, not of the order (see REGENERATE_FROM_STAGE_KEY).
        REGENERATE_FROM_STAGE_KEY,
        # P3 provider pins: WHICH provider connection version / model tier this run
        # executes on — a run-scoped choice, not the order. Excluded so a legacy
        # (pre-P3) parent still fingerprints equal to its would-be resumed child,
        # and a tenant switching provider does not make every paused run "changed".
        PROVIDER_SLOTS_KEY,
        MODEL_TIER_KEY,
        # P6: WHICH tenant-defaults row the run pinned. The resolved VALUES
        # (brand/voice/…) stay in the fingerprint — a changed default is a changed
        # order — but the pin itself (version, raw row) is not.
        TENANT_DEFAULTS_KEY,
    }
)


def order_fingerprint(config_snapshot: Mapping[str, object]) -> str:
    """Canonical JSON of a snapshot minus :data:`RUN_SCOPED_SNAPSHOT_KEYS`.

    Serialised with sorted keys and ``default=str`` so a freshly built dict (UUIDs,
    tuples) and the JSONB round-tripped parent compare by content, not by type.
    """
    order = {k: v for k, v in config_snapshot.items() if k not in RUN_SCOPED_SNAPSHOT_KEYS}
    return json.dumps(json.loads(json.dumps(order, default=str)), sort_keys=True)


async def _fold_course_lessons(
    session: AsyncSession,
    config_snapshot: MutableMapping[str, object],
    *,
    proj: Project,
    tenant_id: UUID,
) -> None:
    """Carry a course's lessons (each child's title + brief + default_deliverable_id).

    course_compile_lessons dispatches from them without re-resolving the children.
    Folded INSIDE ``build_config_snapshot``, not by its caller, so every reader of
    "the snapshot a new run gets" — ``start_run`` AND the failed-parent resume check
    — sees the same inputs.
    """
    if proj.type == "course":
        config_snapshot["lessons"] = await build_course_lessons(
            ProjectRepository(session), proj, tenant_id
        )


async def build_course_lessons(
    proj_repo: ProjectRepository, proj: Any, tenant_id: UUID
) -> list[dict[str, Any]]:
    """Fold a course's child lesson projects into the dispatch-ready shape.

    Each lesson carries its title + brief + default_deliverable_id so the
    course_compile_lessons stage can dispatch child runs without re-resolving
    the children.
    """
    course_with_children = await proj_repo.get_with_children(proj.id, tenant_id=tenant_id)
    lessons: list[dict[str, Any]] = []
    if course_with_children is not None:
        for child in course_with_children.children or []:
            full = await proj_repo.get(child.id, tenant_id=tenant_id)
            if full is None or not full.deliverables:
                continue
            lessons.append(
                {
                    "lesson_project_id": str(child.id),
                    "title": child.title,
                    "brief": dict(child.brief),
                    "default_deliverable_id": str(full.deliverables[0].id),
                }
            )
    return lessons


async def insert_initial_stage_rows(
    session: AsyncSession,
    stage_repo: StageOutputRepository,
    *,
    run_id: UUID,
    blueprint: DAGBlueprint,
    preserved_outputs: dict[str, dict[str, Any]],
) -> None:
    """Insert one stage_output row per blueprint stage in topological order.

    A preserved (regenerated) stage is inserted already-``completed`` with its
    parent output; every other stage is upserted ``pending``.
    """
    for stage_name in blueprint.topological_order():
        if stage_name in preserved_outputs:
            await session.execute(
                insert(StageOutput).values(
                    id=uuid7(),
                    run_id=run_id,
                    stage_name=stage_name,
                    status="completed",
                    output=preserved_outputs[stage_name],
                    started_at=func.now(),
                    completed_at=func.now(),
                )
            )
        else:
            await stage_repo.upsert_pending(run_id, stage_name)


async def reconcile_blueprint_stage_rows(
    stage_repo: StageOutputRepository,
    *,
    run_id: UUID,
    blueprint: DAGBlueprint,
    stage_rows: Sequence[StageOutput],
) -> list[StageOutput]:
    """Seed the ``stage_output`` rows a pre-existing run is missing for its blueprint.

    Rows are inserted exactly once, at run creation. A stage ADDED to a blueprint
    afterwards therefore has no row on any older run — and every writer
    (``mark_running``/``mark_completed``/``mark_skipped``) is a bare ``UPDATE``,
    so each one silently touches zero rows. The stage never appears in
    ``terminal_stage_names``, so ``decide_run_terminal_state``'s set-equality
    against ``blueprint.topological_order()`` can never hold: the run returns
    CONTINUE and sits ``running`` forever.

    The fix is to CREATE the missing rows. Returns the reconciled rows; when
    nothing is missing (the overwhelmingly common case) the caller's rows are
    returned as-is, at the cost of zero extra queries.

    **The two directions of that gate are not symmetric — do not conflate them.**
    ``decide_run_terminal_state`` now asks ``declared <= terminal`` rather than
    ``terminal == declared``:

    * ``declared <= terminal`` (what it does) — every stage the blueprint
      declares must be terminal. This function is what makes that sound: it
      guarantees a row exists for every declared stage, so "no row" can never
      masquerade as "not required".
    * ``terminal <= declared`` (the inversion) — would let a genuinely-unrun
      stage count as done. **Never write this.** It is the loosening the
      original version of this docstring warned about.

    Tolerating rows for stages the blueprint no longer declares is deliberate,
    not a loosening: an orphan row is not evidence about any declared stage.
    Reconcile is INSERT-ONLY and must never DELETE such a row — that would
    destroy a completed stage's ``output``, which ``prior_run_outputs`` reads.

    **A missing stage is not always safe to seed as ``pending``.** The blueprint
    resolved here is always the CURRENT (live) one — there is no per-run pinned
    version — so a stage inserted upstream of one of this run's stages that has
    already run (``config_snapshot`` frozen to the OLD blueprint) would either be
    orphaned work (its dependent already completed and will never consume it —
    the ``website_review_slide_plan``/``website_review_slides`` incident) or, worse,
    feed a stale-``config_snapshot`` dependent a new-blueprint output it was never
    validated against (a hybrid deliverable). Genuinely downstream-only new
    stages (no dependent has started) are unaffected by either hazard and must
    keep seeding ``pending`` exactly as before.

    So each missing stage is classified by whether ANY of its stages
    transitively downstream (``blueprint.descendants``, not just direct
    dependents — a two-hop dependent that already started is exactly as
    contaminated) has left ``pending``:

    * no started descendant → insert ``pending`` (today's behaviour, unchanged).
    * at least one started descendant → insert ``skipped`` with a recorded
      reason, never ``pending``. This is not a special case downstream: a
      ``skipped`` dependency already satisfies ``dag_executor``'s eligibility
      check (``prior_run_outputs`` seeds a skip as ``{}``, same as any other
      opt-out stage) and every stage handler already has to tolerate a
      dependency being skipped, since ``StageSkippedError`` can be raised by any
      stage at runtime. So a stage still-``pending`` downstream of a
      retrofit-skip runs exactly as it would if the new stage had legitimately
      skipped itself.

    "Started" means any status other than ``pending`` — ``running``,
    ``completed``, ``failed``, ``skipped``, or ``cancelled`` all mean the
    dependent's execution decision is already made and must not be revisited.
    A descendant that is ALSO missing (itself new, not yet a row) cannot have
    started and is not treated as one.
    """
    have = {row.stage_name for row in stage_rows}
    missing = [name for name in blueprint.topological_order() if name not in have]
    if not missing:
        return list(stage_rows)
    status_by_name = {row.stage_name: row.status for row in stage_rows}
    to_pending: list[str] = []
    to_skip: list[tuple[str, str]] = []
    for name in missing:
        started_descendants = sorted(
            d for d in blueprint.descendants(name) if status_by_name.get(d, "pending") != "pending"
        )
        if started_descendants:
            to_skip.append(
                (
                    name,
                    f"stage '{name}' was added to the blueprint after this run started; "
                    f"downstream stage(s) {started_descendants} already left 'pending' "
                    "under the frozen (older) blueprint, so retrofitting this stage would "
                    "either orphan its output or feed a stage that was never validated "
                    "against it — skipping instead of running",
                )
            )
        else:
            to_pending.append(name)
    if to_pending:
        logger.warning(
            "run %s has no stage_output rows for %s — blueprint gained stages after "
            "run creation with no started descendant; seeding them 'pending' so the "
            "completion gate can close",
            run_id,
            to_pending,
        )
        await stage_repo.insert_pending_bulk(run_id, to_pending)
    if to_skip:
        logger.warning(
            "run %s has no stage_output rows for %s — blueprint gained stages after "
            "run created UPSTREAM of already-started work; seeding them 'skipped' "
            "instead of 'pending' to avoid retrofitting a hybrid deliverable",
            run_id,
            [name for name, _ in to_skip],
        )
        await stage_repo.insert_skipped_bulk(run_id, to_skip)
    return await stage_repo.list_for_run(run_id)
