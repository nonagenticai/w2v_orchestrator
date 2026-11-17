"""Content-derived pacing for the audiobook pipeline (spec L3.1).

Pure functions — no I/O — that turn a structural boundary kind plus the
narrator's ``speed`` and an optional director ``tempo`` into a pause length in
milliseconds. This replaces the hand-tuned pacing constant on the audiobook
path (the ``700`` ms inter-chapter gap) with a computed, direction-aware value.

Presence-is-opt-in: :func:`resolve_chapter_gap_ms` returns ``None`` when the
brief carries no ``audiobook.direction`` / ``audiobook.pacing`` block, so the
caller keeps its legacy constant and a run that opts into nothing renders
byte-identical to today (acceptance #4).

Inter-SENTENCE / inter-piece pacing is deliberately NOT computed here. That
seam lives in ``tts_backend`` (TTSB-2 — content-derived from each piece's
trailing punctuation, behind ``TTS_CONTENT_PACING_ENABLED``). Computing it here
too would double-pace (the audio analogue of the double-master hazard). This
module owns only the boundary the orchestrator controls at stitch time — the
inter-chapter gap — plus the per-boundary bases that PR-6's directive clamp
reuses to bound ``pause_before_ms``.
"""

from __future__ import annotations

from typing import Literal, cast

BoundaryKind = Literal["sentence", "paragraph", "scene", "dialogue_turn", "chapter"]

#: Base pause per structural boundary, in milliseconds (spec L3.1) at neutral
#: tempo and narrator speed 1.0. A longer structural boundary ⇒ a longer breath.
_BASE_MS: dict[str, int] = {
    "sentence": 150,
    "paragraph": 400,
    "scene": 900,
    "dialogue_turn": 250,
    "chapter": 1400,
}

#: Director tempo → multiplier over every base (smart-auto). Absent/unknown ⇒ 1.0.
_TEMPO_SCALE: dict[str, float] = {
    "brisk": 0.8,
    "measured": 1.25,
}

#: Speed-derived pause scaling bounds. A faster narrator (speed > 1) gets
#: slightly shorter pauses, a slower one slightly longer, clamped so pauses
#: never collapse to nothing or balloon.
_RATE_FACTOR_MIN = 0.7
_RATE_FACTOR_MAX = 1.4


def _as_int(value: object) -> int | None:
    """Coerce a brief-supplied value to int, or None if absent/malformed.

    The brief is authored outside this codebase; a bad value (e.g.
    ``chapter_gap_ms: "700ms"``) must be IGNORED — treated as absent so the
    control ladder falls through — never raise ``ValueError`` out of the terminal
    stitch stage. Mirrors ingest's defensive coercion of ``brief.pronunciations``."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return int(value)
    except TypeError, ValueError:
        return None


def _rate_factor(speed: float) -> float:
    """Narrator speed → pause multiplier ``1/speed`` clamped to [0.7, 1.4].

    A non-positive speed is meaningless for a rate; it maps to the neutral 1.0.
    """
    if speed <= 0:
        return 1.0
    return min(_RATE_FACTOR_MAX, max(_RATE_FACTOR_MIN, 1.0 / speed))


def _tempo_scale(tempo: str | None) -> float:
    if not tempo:
        return 1.0
    return _TEMPO_SCALE.get(tempo.strip().lower(), 1.0)


def compute_pause_ms(
    boundary_kind: BoundaryKind | str,
    *,
    speed: float = 1.0,
    tempo: str | None = None,
    override_ms: int | None = None,
) -> int:
    """Pause for one boundary kind: ``base * tempo * rate_factor`` (ms).

    Control ladder: an explicit ``override_ms`` wins outright (clamped ``>= 0``);
    otherwise the base for ``boundary_kind`` is scaled by the director ``tempo``
    (smart-auto) and the narrator ``speed``. An unknown boundary kind falls back
    to the ``sentence`` base. Always a non-negative integer.
    """
    if override_ms is not None:
        return max(0, int(override_ms))
    base = _BASE_MS.get(str(boundary_kind), _BASE_MS["sentence"])
    return max(0, round(base * _tempo_scale(tempo) * _rate_factor(speed)))


def _tempo_from_direction(direction: object) -> str | None:
    """Extract a tempo hint (``brisk`` / ``measured``) from a brief direction.

    ``direction`` is either a dict carrying an explicit ``tempo`` or the freeform
    ~15-line intent string (e.g. "warm literary narrator, measured pace, …").
    Returns ``None`` (neutral tempo) when no cue is present.
    """
    if isinstance(direction, dict):
        # isinstance-narrowing an `object` to `dict` gives ty a `Never` key type;
        # cast to the real string-keyed shape so `.get` type-checks.
        tempo = cast("dict[str, object]", direction).get("tempo")
        return str(tempo) if tempo else None
    if isinstance(direction, str):
        low = direction.lower()
        for cue in ("brisk", "measured"):
            if cue in low:
                return cue
    return None


def resolve_chapter_gap_ms(audiobook_cfg: dict, *, speed: float = 1.0) -> int | None:
    """Resolve the inter-chapter gap (ms) from the brief's ``audiobook`` block.

    Presence-is-opt-in, control-ladder ordered:

    * an explicit ``audiobook.chapter_gap_ms`` (incl. an intentional ``0``) wins —
      override;
    * else an explicit ``audiobook.pacing.chapter_ms`` wins — override;
    * else, when the brief opted into pacing at all (a truthy ``direction`` or a
      ``pacing`` dict is present), COMPUTE the gap from the ``chapter`` base
      scaled by the director tempo and narrator ``speed`` — smart-auto;
    * else return ``None`` — the caller keeps its legacy constant, so a run that
      opts into no pacing renders byte-identical to today.
    """
    explicit = _as_int(audiobook_cfg.get("chapter_gap_ms"))
    if explicit is not None:
        return max(0, explicit)

    pacing = audiobook_cfg.get("pacing")
    if isinstance(pacing, dict):
        chapter_override = _as_int(pacing.get("chapter_ms"))
        if chapter_override is not None:
            return max(0, chapter_override)

    direction = audiobook_cfg.get("direction")
    if not (direction or isinstance(pacing, dict)):
        return None

    return compute_pause_ms("chapter", speed=speed, tempo=_tempo_from_direction(direction))
