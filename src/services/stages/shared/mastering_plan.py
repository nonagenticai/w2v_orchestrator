"""Mastering profile — the deterministic audio/video mastering intent block.

A run's ``mastering`` config block describes the final-encode mastering the
video compiler should apply: colour handling (``colorspace`` / ``deband``),
rate control (``bitrate_mode`` / ``maxrate`` / ``crf``), and loudness targets
(``loudness_target`` / ``true_peak``) plus a delivery ``platform`` hint. Every
field is a pure config decision the compiler can apply deterministically, with
no model in the loop.

:class:`MasteringProfile` is a frozen dataclass (sibling convention:
``finish.FinishSpec`` / ``shotlist.ShotSpec`` — NOT pydantic). It clamps the
two bounded numeric knobs on construction via :func:`provenance.clamp`, so an
out-of-range value is pulled back at the boundary and can never reach the
compiler. :func:`build_mastering_profile` is presence-is-opt-in: a snapshot
WITHOUT a ``mastering`` block returns ``None`` so the compiler request is
byte-identical to today.

The threaded dict (:meth:`MasteringProfile.as_mastering_payload`) uses EXACTLY
the field names of the compiler's ``MasteringProfile`` contract (all optional):
``colorspace`` / ``deband`` / ``bitrate_mode`` / ``maxrate`` / ``crf`` /
``loudness_target`` / ``true_peak`` / ``platform``. ``deband`` is a STRING
("auto"/"on"/"off"), not a bool.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeVar, cast

from src.services.stages.shared.loudness_contract import (
    COMPILER_MASTER_LUFS,
    ship_loudness_window,
)
from src.services.stages.shared.provenance import clamp, resolve_with_provenance

__all__ = [
    "MasteringProfile",
    "build_mastering_profile",
    "default_loudness_target",
    "ship_loudness_window",
]

_T = TypeVar("_T")

_COMPUTED_BY = "mastering_profile"


def default_loudness_target() -> float:
    """The master integrated-loudness target (LUFS) — the COMPILER's, not ours.

    ⭐ THE ORCHESTRATOR DOES NOT GET TO PICK THIS NUMBER. It is whatever
    ``video_compiler_service`` will actually apply, vendored as
    :data:`loudness_contract.COMPILER_MASTER_LUFS` (-14.0, compiler PR #217,
    2026-09-12). Threading anything else means asserting a target the compiler is
    about to ignore or, worse, overriding its shipped default with a number this
    repo invented.

    ⚠️ CORRECTION, 2026-09-16. This function briefly derived the target as the CENTRE
    of the ship gate's window, which evaluated to -12.5. That was backwards on two
    counts. The window is downstream of the target, not upstream of it — deriving one
    from the other in that direction lets a band edit silently move the delivered
    audio. And -12.5 specifically is the value compiler PR #217 ABANDONED: it asked a
    bus already arriving at -18 LUFS per narration clip for ~5.5 dB of gain riding,
    "which is what made the narration breathe" (that PR's own words). Re-imposing it
    fleet-wide would have reintroduced the owner's original artefact complaint. The
    arrow now points the only honest way: compiler default → orchestrator target →
    ship window.
    """
    return COMPILER_MASTER_LUFS


# The integrated-loudness target (LUFS) threaded whenever a run authors a ``mastering``
# block. It is the compiler's OWN shipped default, so a run that states the block and a
# run that omits it are mastered identically — the orchestrator can no longer assert a
# target that disagrees with what the compiler applies. Held in step by
# ``tests/services/stages/shared/test_loudness_contract.py``.
_DEFAULT_LOUDNESS_TARGET = default_loudness_target()

# Bounded-knob ranges. crf is the standard x264/x265 0..51 range; loudness_target
# is clamped to a sane integrated-loudness window so a typo can't ask for silence
# or clipping.
_CRF_LO, _CRF_HI = 0, 51
_LOUDNESS_LO, _LOUDNESS_HI = -30.0, -5.0

# The optional (may-be-None) contract fields, in payload order. loudness_target is
# handled separately because it always threads.
_OPTIONAL_FIELDS = (
    "colorspace",
    "deband",
    "bitrate_mode",
    "maxrate",
    "crf",
    "true_peak",
    "platform",
)


@dataclass(frozen=True)
class MasteringProfile:
    """A resolved mastering profile. The two bounded knobs clamp on construction.

    Field names match the video_compiler_service ``MasteringProfile`` contract
    verbatim. All fields are optional except ``loudness_target``, which carries
    the COMPILER's own shipped default (:data:`_DEFAULT_LOUDNESS_TARGET`). Clamped fields:
    ``crf`` 0..51 and ``loudness_target`` -30..-5 — both only when not ``None``.
    """

    colorspace: str | None = None
    deband: str | None = None  # "auto" | "on" | "off" — a STRING, not a bool
    bitrate_mode: str | None = None
    maxrate: str | None = None  # e.g. "12M"
    crf: int | None = None  # clamp 0..51 when set
    loudness_target: float | None = _DEFAULT_LOUDNESS_TARGET  # clamp -30..-5
    true_peak: float | None = None
    platform: str | None = None

    def __post_init__(self) -> None:
        # Clamp bounded knobs at the boundary so an out-of-range value can never
        # reach the compiler. Skip None (an absent knob stays absent). The
        # dataclass is frozen, so write back via object.__setattr__.
        if self.crf is not None:
            resolved = clamp(int(self.crf), _CRF_LO, _CRF_HI, computed_by=_COMPUTED_BY)
            object.__setattr__(self, "crf", resolved.value)
        if self.loudness_target is not None:
            resolved = clamp(
                float(self.loudness_target), _LOUDNESS_LO, _LOUDNESS_HI, computed_by=_COMPUTED_BY
            )
            object.__setattr__(self, "loudness_target", resolved.value)

    def as_mastering_payload(self) -> dict[str, object]:
        """The dict threaded to the compiler as its ``mastering`` block.

        Always includes ``loudness_target`` (it carries a real default); every
        other field is included only when set, so an absent knob is omitted and
        the compiler keeps its own default for it.
        """
        payload: dict[str, object] = {"loudness_target": self.loudness_target}
        for name in _OPTIONAL_FIELDS:
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return payload


def build_mastering_profile(config_snapshot: Mapping[str, object]) -> MasteringProfile | None:
    """Build a :class:`MasteringProfile` from a run's ``config_snapshot``.

    Presence-is-opt-in: an ABSENT (or non-dict) ``mastering`` block returns
    ``None`` so the compiler request stays byte-identical to today. A present
    block always yields a profile — each field resolves via
    :func:`resolve_with_provenance` (an explicit block value overrides the
    default) for the audit trail.
    """
    mastering_raw = config_snapshot.get("mastering")
    if not isinstance(mastering_raw, dict):
        return None
    # ``config_snapshot.get`` is ``object``; ``isinstance``-narrowing to ``dict``
    # yields ``dict[Unknown, Unknown]`` (Never-typed keys), so cast to the
    # opaque-JSON shape for the read-only ``key in``/subscript below.
    mastering = cast("Mapping[str, object]", mastering_raw)

    def _resolve(key: str, default: _T) -> _T:
        if key in mastering:
            return resolve_with_provenance(
                key, override=mastering[key], default=default, computed_by=_COMPUTED_BY
            ).value
        return resolve_with_provenance(key, default=default, computed_by=_COMPUTED_BY).value

    return MasteringProfile(
        colorspace=_resolve("colorspace", None),
        deband=_resolve("deband", None),
        bitrate_mode=_resolve("bitrate_mode", None),
        maxrate=_resolve("maxrate", None),
        crf=_resolve("crf", None),
        loudness_target=_resolve("loudness_target", _DEFAULT_LOUDNESS_TARGET),
        true_peak=_resolve("true_peak", None),
        platform=_resolve("platform", None),
    )
