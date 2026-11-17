"""Color-finish spec — the deterministic tail of the grade/finishing chain.

A run's ``finish`` config block describes how the final video should be graded
and framed. The spec splits cleanly into two layers:

* **Deterministic letterbox layer** (ORCH-0.4) — ``letterbox_color``, the brand
  pillarbox/letterbox fill. A pure config decision the compiler applies
  byte-deterministically, with no model in the loop, and always threaded (see
  :meth:`FinishSpec.as_finish_payload`). ``broadcast_safe`` remains an
  author-facing C8 field but is NEVER sent: the legal-luma clamp was empirically
  falsified (a pre-encode ``lutyuv`` clamp cannot survive libx264 ringing, and
  the concat is ``-c copy``), so the compiler has no such field.
* **Whole-film grade layer** (ORCH-3.3) — the finishing-exclusive knobs
  ``look``/tints/``vignette``/``grain``/``glow``/``sharpen``/``grade_intensity``
  (+ ``lut_url``), filled by the deterministic :mod:`colorist` off the run's
  mood/style + design tokens when the grade is opted in (``finish.enabled``).
  They ride :meth:`FinishSpec.as_finish_payload` ONLY when ``enabled`` AND a grade
  is actually set. The tonal dials ``contrast``/``saturation``/``gamma``/
  ``temperature`` stay at their C8 identity defaults and are NEVER threaded — the
  per-shot ``ColorAdjust`` (Studio re-stitch) owns those, and a whole-film tonal
  layer would double-apply on top of a user's per-clip grade. :class:`FinishSpec`
  clamps every bounded knob on construction, matching the compiler's ranges.

:class:`FinishSpec` is a frozen dataclass (sibling convention:
``shotlist.ShotSpec`` / ``provenance.ProvenanceValue`` — NOT pydantic). It
clamps every bounded knob on construction via :func:`provenance.clamp`, so an
out-of-range value is pulled back at the boundary and can never reach the
compiler. :func:`build_finish_spec` is presence-is-opt-in: a snapshot WITHOUT a
``finish`` block returns ``None`` so the compiler stays on ``-c:v copy`` and the
output is byte-identical to before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.services.stages.shared.provenance import clamp, resolve_with_provenance

__all__ = ["FinishSpec", "build_finish_spec"]

_COMPUTED_BY = "finish_spec"

# Default tints/fills. Neutral so the deterministic layers alone (no grade) leave
# a picture visually untouched apart from the legal-luma clamp + letterbox fill.
_DEFAULT_SHADOW_TINT = "#000000"
_DEFAULT_HIGHLIGHT_TINT = "#ffffff"
_DEFAULT_LETTERBOX_COLOR = "#000000"
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


@dataclass(frozen=True)
class FinishSpec:
    """A resolved color-finish spec. Bounded knobs clamp on construction.

    Field names + ranges match contract C8 verbatim — including
    ``broadcast_safe``, which an author may still set but which is never threaded
    to the compiler (it has no such field). ``enabled`` gates the WM grade layers
    only; ``letterbox_color`` applies regardless. Clamped fields:
    ``contrast`` 0.8..1.4, ``saturation`` 0.6..1.4, ``gamma`` 0.5..2.0,
    ``temperature`` -1..1, ``vignette``/``grain``/``glow``/``sharpen`` 0..1, and
    ``grade_intensity`` 0..1. The ``gamma`` and ``grade_intensity`` bounds match
    the compiler's grade chain exactly (ORCH-3.3).
    """

    enabled: bool = False
    look: str = "none"
    contrast: float = 1.0  # clamp 0.8..1.4
    saturation: float = 1.0  # clamp 0.6..1.4
    gamma: float = 1.0  # clamp 0.5..2.0
    temperature: float = 0.0  # clamp -1..1
    shadow_tint: str = _DEFAULT_SHADOW_TINT
    highlight_tint: str = _DEFAULT_HIGHLIGHT_TINT
    vignette: float = 0.0  # clamp 0..1
    grain: float = 0.0  # clamp 0..1
    glow: float = 0.0  # clamp 0..1
    sharpen: float = 0.0  # clamp 0..1
    grade_intensity: float = 1.0  # clamp 0..1
    match_sources: bool = False
    broadcast_safe: bool = False
    letterbox_color: str = _DEFAULT_LETTERBOX_COLOR
    lut_url: str | None = None

    def __post_init__(self) -> None:
        # Clamp bounded knobs at the boundary so an out-of-range value can never
        # reach the compiler. The dataclass is frozen, so write back via
        # object.__setattr__ (the same escape hatch frozen dataclasses use).
        for field_name, lo, hi in (
            ("contrast", 0.8, 1.4),
            ("saturation", 0.6, 1.4),
            ("gamma", 0.5, 2.0),
            ("temperature", -1.0, 1.0),
            ("vignette", 0.0, 1.0),
            ("grain", 0.0, 1.0),
            ("glow", 0.0, 1.0),
            ("sharpen", 0.0, 1.0),
            ("grade_intensity", 0.0, 1.0),
        ):
            resolved = clamp(float(getattr(self, field_name)), lo, hi, computed_by=_COMPUTED_BY)
            object.__setattr__(self, field_name, resolved.value)

    @property
    def grade_is_set(self) -> bool:
        """True iff this spec carries a real whole-film grade (a look or an effect).

        A ``look`` other than ``none``, a ``lut_url``, or any positive film-texture
        knob (``vignette`` / ``grain`` / ``glow`` / ``sharpen``) means the colorist
        (or an author) asked for a grade. All-identity ⇒ ``False`` ⇒ nothing to
        thread beyond the letterbox.
        """
        return (
            self.look != "none"
            or bool(self.lut_url)
            or any(v > 0.0 for v in (self.vignette, self.grain, self.glow, self.sharpen))
        )

    def as_finish_payload(self) -> dict[str, Any]:
        """The dict threaded to the compiler as its ``finish`` block.

        Always emits ``letterbox_color`` — the deterministic layer, a real resolved
        default sourced from the brand's ``primary_color`` (the same always-threads
        rule as ``MasteringProfile.as_mastering_payload``'s ``loudness_target``).

        Additionally emits the whole-film GRADE knobs under a nested ``grade`` key
        (the compiler's ``FinishProfile.grade: GradeSpec`` — flat top-level knobs
        are dropped by its ``extra=ignore``) — but ONLY when
        ``self.enabled and self.grade_is_set``. An all-identity spec, or one with
        ``enabled=False`` (the flag that gates the grade layer), threads only the
        letterbox, so a run that did not opt into a grade is byte-identical to the
        pre-ORCH-3.3 payload. The grade set is the *finishing-exclusive* knobs only
        — ``look`` / tints / ``vignette`` / ``grain`` / ``glow`` / ``sharpen`` /
        ``grade_intensity`` / ``match_sources`` (+ ``lut_url`` when set). The
        tonal dials (``contrast`` / ``saturation`` / ``gamma`` / ``temperature``) are
        DELIBERATELY not sent: per-shot ``ColorAdjust`` owns those and a whole-film
        tonal layer would double-apply on top of a user's per-clip grade.

        ``broadcast_safe`` is deliberately NOT sent. The legal-luma clamp was
        empirically falsified on real ffmpeg: a pre-encode ``lutyuv`` clamp cannot
        deliver decoded Y∈[16,235] because libx264 ringing around sharp edges
        (captions) re-introduces out-of-range luma, worsening with CRF, and the
        concat is ``-c copy`` so nothing downstream can fix it. The compiler
        therefore has no such field, and the flag would be a lie. The dataclass
        field stays (it is part of the C8 author contract) — it just never ships.
        """
        payload: dict[str, Any] = {"letterbox_color": self.letterbox_color}
        if self.enabled and self.grade_is_set:
            # The grade knobs are NESTED under ``grade`` — the compiler's
            # ``FinishProfile`` has only ``letterbox_color`` + a nested
            # ``grade: GradeSpec`` and ignores unknown top-level keys
            # (``extra=ignore``), so flat knobs are silently dropped and the
            # grade never applies. Every emitted name is a real ``GradeSpec``
            # field (no renames) — but the emitted SET is a strict subset of
            # ``GradeSpec``: the tonal dials and ``broadcast_safe`` are
            # deliberately withheld (see this docstring), so "one-for-one" runs
            # one way only. Key order follows ``GradeSpec``'s field order.
            grade: dict[str, object] = {
                "enabled": True,
                "look": self.look,
                "shadow_tint": _normalize_hex(self.shadow_tint),
                "highlight_tint": _normalize_hex(self.highlight_tint),
                "vignette": self.vignette,
                "grain": self.grain,
                "glow": self.glow,
                "sharpen": self.sharpen,
                "grade_intensity": self.grade_intensity,
                "match_sources": self.match_sources,
            }
            if self.lut_url:
                grade["lut_url"] = self.lut_url
            payload["grade"] = grade
        return payload


def build_finish_spec(
    config_snapshot: dict[str, Any], *, grade_enabled: bool = False
) -> FinishSpec | None:
    """Build a :class:`FinishSpec` from a run's ``config_snapshot``.

    Presence-is-opt-in: an ABSENT ``finish`` block returns ``None`` so the
    compiler stays on ``-c:v copy`` (byte-identical output). A present block
    always yields a spec — ``enabled=False`` still carries the deterministic
    ``letterbox_color`` layer but no grade. The block is presence-opt-in end to
    end: ``run_setup`` only lifts it when the brief (or a per-run override)
    supplies one, so a run with no ``finish`` block sends no ``finish`` payload.

    The deterministic ``letterbox_color`` layer always resolves (sourced from the
    brand's ``primary_color`` via :func:`resolve_with_provenance`, falling back to
    ``#000000``). The whole-film GRADE knobs are filled by the deterministic
    :mod:`colorist` (ORCH-3.3) — but ONLY when the grade layer is opted in
    (``finish.enabled`` is truthy) AND ``grade_enabled`` is set. ``grade_enabled``
    is the GitOps kill-switch (``pipeline.grade_default_on_showcase``): with it
    off, NO run emits a grade — not even one whose brief authored
    ``finish.enabled: true`` — so the feature is fully killable without a code
    deploy. A disabled/kill-switched grade keeps the knobs at their identity
    defaults, so ``as_finish_payload`` threads only the letterbox (byte-identical
    to the deterministic-only path).
    """
    finish = config_snapshot.get("finish")
    if not isinstance(finish, dict):
        return None

    brand = config_snapshot.get("brand")
    brand_color = (brand.get("primary_color") if isinstance(brand, dict) else None) or (
        _DEFAULT_LETTERBOX_COLOR
    )

    # An explicit finish.letterbox_color wins; else the brand primary colour; else
    # the neutral fallback. resolve_with_provenance records where the value came
    # from (override vs resolved default) for the audit trail.
    if "letterbox_color" in finish:
        letterbox = resolve_with_provenance(
            "letterbox_color",
            override=str(finish["letterbox_color"]),
            default=brand_color,
            computed_by=_COMPUTED_BY,
        ).value
    else:
        letterbox = resolve_with_provenance(
            "letterbox_color", default=brand_color, computed_by=_COMPUTED_BY
        ).value

    enabled = bool(finish.get("enabled", False))
    broadcast_safe = bool(finish.get("broadcast_safe", False))
    letterbox_color = _normalize_hex(letterbox)
    # Grade emits only when opted in AND the kill-switch is on. Off ⇒ letterbox
    # only, regardless of an author-set finish.enabled (true global kill-switch).
    if not enabled or not grade_enabled:
        return FinishSpec(
            enabled=enabled, broadcast_safe=broadcast_safe, letterbox_color=letterbox_color
        )

    # Grade layer opted in: the deterministic colorist fills the finishing-exclusive
    # knobs (look/tints/vignette/grain/glow/sharpen/grade_intensity) off the run's
    # mood/tone/style + design tokens; a brief-authored finish.grade knob wins. The
    # tonal dials (contrast/saturation/gamma/temperature) stay at identity — per-shot
    # ColorAdjust owns those. Lazy import keeps the shared-module graph acyclic.
    from src.services.stages.shared.colorist import grade_from_snapshot

    grade = grade_from_snapshot(config_snapshot)
    lut_raw = finish.get("lut_url")
    # ``match_sources`` is a boolean switch, not a colorist-resolved dial (the
    # colorist owns look/tints/texture only), so it is lifted straight off the
    # author's ``finish.grade`` block — the same place the colorist reads its
    # own overrides from.
    authored_grade = finish.get("grade")
    match_sources = bool(
        authored_grade.get("match_sources", False) if isinstance(authored_grade, dict) else False
    )
    return FinishSpec(
        enabled=enabled,
        broadcast_safe=broadcast_safe,
        letterbox_color=letterbox_color,
        look=grade.look,
        shadow_tint=grade.shadow_tint,
        highlight_tint=grade.highlight_tint,
        vignette=grade.vignette,
        grain=grade.grain,
        glow=grade.glow,
        sharpen=grade.sharpen,
        grade_intensity=grade.grade_intensity,
        match_sources=match_sources,
        lut_url=lut_raw if isinstance(lut_raw, str) else None,
    )


def _normalize_hex(value: str) -> str:
    """Coerce a brand/author colour to the 6- or 8-digit hex the compiler accepts.

    The compiler validates ``letterbox_color`` with a strict ``#rrggbb[aa]``
    pattern, and ``CompileRequest`` validation is atomic — so one malformed colour
    422s the ENTIRE compile, not just the letterbox. Brand colours are not
    trustworthy input: ``design_system._hex_to_rgb`` deliberately accepts 3-digit
    ``#rgb``, so a brand whose ``primary_color`` is ``#fff`` is legitimate here and
    would kill the run.

    Expand 3-digit to 6; fall back to the neutral default for anything we cannot
    parse. This is the colour analogue of the numeric ``clamp()`` the bounded
    knobs already get: bad values are corrected at the boundary, never forwarded.
    """
    s = value.strip()
    body = s[1:] if s.startswith("#") else s
    if len(body) == 3 and all(c in _HEX_DIGITS for c in body):
        return "#" + "".join(c * 2 for c in body)
    if len(body) in (6, 8) and all(c in _HEX_DIGITS for c in body):
        return "#" + body
    return _DEFAULT_LETTERBOX_COLOR
