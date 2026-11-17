"""The A0 resolver — a ``VideoIntent`` (+ authored brief) → a ``PlanDocument``.

Pure, deterministic, no-LLM. For every knob it walks the provenance ladder via
``resolve_with_provenance``: an authored ``brief`` value structurally WINS its
rung (override), else a computed smart-auto value, else the preset/global
default. Each resolved knob is recorded in the ``PlanDocument.provenance`` map so
the UI/audit can explain where every value came from.

Make-or-break correctness rule (see the ``package_url`` branch below):
authored ``lines`` are materialized ONLY when a real ``package_url`` is
resolvable. With no footage, ``lines`` stays ``[]`` so the existing generated
pipeline builds footage from ``target_url`` — emitting authored lines against an
empty ``package_url`` dies stage-deep at ``demo_video_record``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.api.schemas.intent import KnobProvenance, PlanDocument
from src.services.planning import compute_levers
from src.services.planning.presets import Preset, load_preset
from src.services.stages.shared.provenance import (
    _UNSET,
    ProvenanceValue,
    input_hash,
    resolve_with_provenance,
)

#: Global fallback resolution when neither the brief nor the intent asks for one.
_DEFAULT_RESOLUTION = "1080p"


def _canonical(intent: Mapping[str, Any]) -> dict[str, object]:
    """Stable projection of the intent for hashing.

    ``input_hash`` canonicalizes with ``sort_keys=True``, so a shallow copy is
    enough to make the hash order-insensitive over the intent's top-level keys.
    """
    return dict(intent)


def _brief_get(brief: Mapping[str, Any], *path: str) -> object:
    """Authored value at ``brief[path...]``, or the ``_UNSET`` sentinel if absent.

    Returning ``_UNSET`` (not ``None``) lets an explicitly authored ``None`` /
    ``0`` / ``False`` still WIN the override rung in ``resolve_with_provenance``.
    """
    node: object = brief
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return _UNSET
        node = node[key]
    return node


def _emit(pv: ProvenanceValue) -> bool:
    """A knob is written into the brief ONLY when it was authored or computed from a
    real signal. A pure preset/global DEFAULT must not be emitted — otherwise it
    silently overrides the pipeline's own default for that knob (e.g. resolution
    "1080p" would beat the system's 2k tier, and a default dialogue/voice block would
    force two-voice mode on a plain single-narrator run)."""
    return pv.source != "default"


def _smart_auto_from_mix(pv: ProvenanceValue) -> object:
    """A mix target as a smart-auto candidate — or ``_UNSET`` when it's a default.

    ``mix_gain_targets`` already distinguishes a computed (``"smart-auto"``) run
    from the byte-identical neutral literals (``"default"``). We only forward the
    value as a smart-auto candidate when it was actually computed; a neutral
    literal must fall through to the ``default`` rung so it reads ``"default"``.
    """
    return pv.value if pv.source == "smart-auto" else _UNSET


def resolve_plan(
    intent: Mapping[str, Any], *, brief: Mapping[str, Any], preset: Preset | None = None
) -> PlanDocument:
    """Resolve a high-level intent + authored brief into a computed plan."""
    preset = preset or load_preset(intent.get("preset", "default"))
    ih = input_hash(_canonical(intent))

    prov: dict[str, ProvenanceValue] = {}

    # --- Audio bed gains (music / ambient / duck / sfx) ---------------------
    mood = intent.get("mood")
    mix = compute_levers.mix_gains(preset, mood)
    for knob, target in (
        ("music_gain_db", mix.music),
        ("ambient_gain_db", mix.ambient),
        ("duck_db", mix.duck),
        ("sfx_gain_db", mix.sfx),
    ):
        prov[knob] = resolve_with_provenance(
            knob,
            override=_brief_get(brief, "sound_effects", knob),
            smart_auto=_smart_auto_from_mix(target),
            default=target.value,
            computed_by="resolve_plan",
        )

    # --- Resolution ---------------------------------------------------------
    prov["resolution"] = resolve_with_provenance(
        "resolution",
        override=_brief_get(brief, "resolution"),
        smart_auto=intent.get("resolution", _UNSET),
        default=_DEFAULT_RESOLUTION,
        computed_by="resolve_plan",
    )

    # --- Dialogue on/off ----------------------------------------------------
    prov["dialogue_enabled"] = resolve_with_provenance(
        "dialogue_enabled",
        override=_brief_get(brief, "dialogue", "enabled"),
        default=preset.dialogue_enabled,
        computed_by="resolve_plan",
    )

    # --- Voice cast ---------------------------------------------------------
    # Pins come from `intent.cast` ({role: voice_id}); a brief-level `voice.cast`
    # override still wins per role. Speakers ARE the pinned roles — a bare intent
    # (cast="auto") yields no speakers and no forced cast (source stays "default").
    cast_intent = intent.get("cast")
    intent_pins = (
        {str(k): str(v) for k, v in cast_intent.items()} if isinstance(cast_intent, Mapping) else {}
    )
    authored_cast = _brief_get(brief, "voice", "cast")
    brief_pins = (
        {str(k): str(v) for k, v in authored_cast.items()}
        if isinstance(authored_cast, Mapping)
        else {}
    )
    pins = {**intent_pins, **brief_pins} or None  # brief overrides intent per role
    speakers = list(pins) if pins else []
    computed_cast = compute_levers.cast_assignment(speakers, pins)
    prov["voice_cast"] = resolve_with_provenance(
        "voice_cast",
        override=authored_cast if brief_pins else _UNSET,
        smart_auto=computed_cast or _UNSET,
        default={},
        computed_by="resolve_plan",
    )

    # --- Package URL (footage bundle) — gates the make-or-break lines branch -
    package = intent.get("package")
    prov["package_url"] = resolve_with_provenance(
        "package_url",
        override=_brief_get(brief, "package_url"),
        smart_auto=package if package else _UNSET,
        default=None,
        computed_by="resolve_plan",
    )
    package_url = prov["package_url"].value

    # --- CORRECTNESS RULE (make-or-break) -----------------------------------
    # Materialize authored `lines` ONLY when a real package_url is resolvable.
    # With NO footage we leave `lines = []` so the existing generated pipeline
    # builds footage from target_url; emitting authored lines against an empty
    # package_url dies stage-deep at demo_video_record.
    lines: list[dict[str, Any]]
    if package_url:
        must_show = list(intent.get("must_show") or [])
        must_say = list(intent.get("must_say") or [])
        n_beats = compute_levers.beat_count(
            float(intent.get("duration_seconds") or 0.0), preset
        ).value
        lines = compute_levers.build_beats(must_show, must_say, n_beats, speakers=speakers)
        bindings = compute_levers.clip_bindings(lines, must_show)
        for line in lines:
            bound = bindings.get(line["id"])
            if bound is not None:
                line["clip"] = bound
    else:
        lines = []

    # Emit ONLY the knobs that carry a real signal (authored or computed) — a pure
    # default is left OFF the plan so it never pre-empts the pipeline's own default
    # (see `_emit`). `provenance` still records every knob, defaults included, for the
    # audit trail. `lines`/`package_url` are already presence-gated above. Raw beat
    # dicts are coerced to AuthoredLine by `model_validate`.
    fields: dict[str, object] = {
        "lines": lines,
        "preset": preset.name,
        "input_hash": ih,
        "provenance": {
            k: KnobProvenance(
                value=pv.value, source=pv.source, computed_by=pv.computed_by, clamp_of=pv.clamp_of
            )
            for k, pv in prov.items()
        },
    }
    if package_url:
        fields["package_url"] = package_url
    gain_keys = ("music_gain_db", "ambient_gain_db", "duck_db", "sfx_gain_db")
    if any(_emit(prov[g]) for g in gain_keys):
        fields["sound_effects"] = {**{g: prov[g].value for g in gain_keys}, "enabled": True}
    if _emit(prov["resolution"]):
        fields["resolution"] = prov["resolution"].value
    if _emit(prov["dialogue_enabled"]):
        fields["dialogue"] = {"enabled": prov["dialogue_enabled"].value}
    if _emit(prov["voice_cast"]) and prov["voice_cast"].value:
        fields["voice"] = {"cast": prov["voice_cast"].value}

    return PlanDocument.model_validate(fields)
