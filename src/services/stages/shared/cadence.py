"""Cadence spec — the deterministic frame-rate reconciler for the compile tail.

A run's ``cadence`` config block describes the master frame rate the final video
should be conformed to, and how sources at a different rate should be retimed. It
is the temporal sibling of :mod:`finish` (the spatial/colour finisher): both are
pure, deterministic config resolvers the compiler applies with no model in the
loop.

The reconciler answers three questions:

* **target_fps** — the master rate. An explicit ``cadence.target_fps`` override
  wins; else the dominant source rate; else the neutral ``30`` default. Every
  candidate is snapped to a legal broadcast rate via :func:`clamp_cadence`
  (``23.976 → 24``, ``29.97 → 30``, ``59.94 → 60``; legal rates are identity).
* **retime** — ``"interpolate"`` ONLY when a source↔master ratio is non-integer
  (a genuine cadence conversion, e.g. 24→60) AND that source is marked
  ``high_motion`` (where judder/blending actually shows); otherwise ``"clean"``
  (a plain conform / integer frame-duplication that needs no interpolation).
* **is_cfr** — READ from the source profiles, NEVER re-asserted here (contract
  C10 split-verify: the reconciler reports what the source is, it does not force
  constant-frame-rate).

:class:`CadenceSpec` is a frozen dataclass (sibling convention:
``finish.FinishSpec`` / ``shotlist.ShotSpec`` — NOT pydantic).
:func:`build_cadence_spec` is presence-is-opt-in: a snapshot WITHOUT a
``cadence`` block AND with no source profiles returns ``None`` so the compiler
stays exactly on its current path and the output is byte-identical to before.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["CadenceSpec", "build_cadence_spec", "clamp_cadence", "reconcile_cadence"]

# The closed set of legal broadcast frame rates. Every resolved rate snaps to one
# of these, so a source's NTSC fractional rate (23.976/29.97/59.94) or an
# out-of-set override can never reach the compiler.
_LEGAL: tuple[int, ...] = (24, 25, 30, 50, 60)

_DEFAULT_FPS = 30


def clamp_cadence(x: float) -> int:
    """Snap ``x`` to the nearest legal broadcast frame rate in :data:`_LEGAL`.

    NTSC fractional rates land on their integer sibling (``23.976 → 24``,
    ``29.97 → 30``, ``59.94 → 60``); a rate already legal is returned unchanged.
    Nearest is by absolute distance, so an off-set value (e.g. 48) resolves to the
    closest legal rate deterministically.
    """
    return min(_LEGAL, key=lambda rate: abs(rate - x))


@dataclass(frozen=True)
class CadenceSpec:
    """A resolved cadence spec. Deterministic — same inputs ⇒ equal specs.

    ``target_fps`` is the master rate every source conforms to. ``retime`` is
    ``clean`` (integer conform) or ``interpolate`` (non-integer + high-motion
    source). ``is_cfr`` is read from the source profiles, never re-asserted.
    ``source_fps`` records the (legal-snapped) source rates the spec reconciled,
    for the audit trail.
    """

    target_fps: int = 30
    retime: str = "clean"  # clean | interpolate
    is_cfr: bool = True  # READ from source, never re-asserted (C10 split-verify)
    source_fps: tuple[int, ...] = ()


def reconcile_cadence(
    source_profiles: Sequence[Mapping[str, Any]],
    intent: Mapping[str, Any] | None,
) -> CadenceSpec | None:
    """Reconcile source frame rates against a cadence ``intent`` into a spec.

    The master gate: return ``None`` when ``intent`` is falsy AND there are no
    ``source_profiles`` — nothing asked for a cadence and nothing carries one, so
    the compiler stays inert (byte-identical output). Otherwise:

    * ``target_fps`` = clamp(``intent.target_fps`` override) → clamped dominant
      source fps → ``30``.
    * ``retime`` = ``"interpolate"`` ONLY when some source↔master ratio is
      non-integer AND that source profile is marked ``high_motion``; else
      ``"clean"``.
    * ``is_cfr`` = the AND of every source profile's ``is_cfr`` (default ``True``)
      — read from source, never forced.

    Each profile is a mapping that may carry ``fps`` (numeric), ``high_motion``
    (bool) and ``is_cfr`` (bool). Absent keys take neutral defaults.
    """
    if not intent and not source_profiles:
        return None

    # Legal-snapped source rates (skip profiles with no/zero fps).
    source_fps = tuple(clamp_cadence(float(p["fps"])) for p in source_profiles if p.get("fps"))

    # target_fps ladder: explicit override → dominant source rate → default.
    override = intent.get("target_fps") if intent else None
    if override is not None:
        target_fps = clamp_cadence(float(override))
    elif source_fps:
        target_fps = Counter(source_fps).most_common(1)[0][0]
    else:
        target_fps = _DEFAULT_FPS

    # retime: interpolate only on a genuine (non-integer) cadence conversion of a
    # high-motion source; a clean integer conform never needs blending.
    retime = "clean"
    for profile in source_profiles:
        raw = profile.get("fps")
        if not raw:
            continue
        sfps = clamp_cadence(float(raw))
        hi, lo = max(target_fps, sfps), min(target_fps, sfps)
        non_integer = lo == 0 or hi % lo != 0
        if non_integer and profile.get("high_motion"):
            retime = "interpolate"
            break

    is_cfr = all(bool(p.get("is_cfr", True)) for p in source_profiles)

    return CadenceSpec(
        target_fps=target_fps,
        retime=retime,
        is_cfr=is_cfr,
        source_fps=source_fps,
    )


def build_cadence_spec(config_snapshot: dict[str, Any]) -> CadenceSpec | None:
    """Build a :class:`CadenceSpec` from a run's ``config_snapshot``.

    Presence-is-opt-in: an ABSENT ``cadence`` block AND no ``source_profiles``
    yields ``None`` so the compile tail stays byte-identical. A present ``cadence``
    block (or any carried source profile) yields a reconciled spec. Callers stay
    one-liners: ``spec = build_cadence_spec(ctx.config_snapshot)`` (mirror
    :func:`finish.build_finish_spec`).
    """
    cadence = config_snapshot.get("cadence")
    intent = cadence if isinstance(cadence, dict) else None
    raw_profiles = config_snapshot.get("source_profiles")
    source_profiles = raw_profiles if isinstance(raw_profiles, list) else []
    return reconcile_cadence(source_profiles, intent)
