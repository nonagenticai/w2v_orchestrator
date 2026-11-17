"""The one place `action_reel` decides what frame rate the reel is built on.

Two consumers, and they MUST agree:

* ``cut_plan`` quantises every cut boundary to a frame grid. A boundary the
  master cannot express is re-rounded by the renderer, one-sidedly, per segment —
  which is exactly the prefix-sum drift ``cut_plan``'s docstring exists to prevent.
* ``compile`` sets the master rate the compiler conforms every source to.

Before this module the two disagreed by construction: ``solve_cut_plan`` defaulted
to 24000/1001 while the compile request carried no ``cadence`` block at all, so the
compiler used its own 30 fps default. Planning on a 23.976 grid and rendering on a
30 grid is strictly worse than planning on the grid you render on, so the fix is
not "pick the right number" — it is "ask the same function twice".

Resolution order is inherited wholesale from :mod:`stages.shared.cadence`, which
already implements it for every other product: an explicit ``cadence.target_fps``
in the run config wins, else the dominant measured source rate, else 30. Every
candidate is snapped to a legal broadcast rate (``23.976 → 24``), so the drone
corpus resolves to a 24 fps master and the 5:4 duplicate-frame judder disappears.

``high_motion`` is deliberately NOT asserted on the profiles. It only selects
``retime="interpolate"`` on a non-integer conversion, and this stage has no
measurement of it at the point the spec is built — ``shot_analysis``'s motion
timeline is per-segment, not per-source. Claiming it would be inventing a signal.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.services.stages.shared.cadence import reconcile_cadence

__all__ = ["DEFAULT_MASTER_FPS", "resolve_master_fps", "source_profiles_from_probe"]

#: Used only when nothing measured or configured a rate — the same neutral value
#: ``shared.cadence`` falls back to, restated here so a reader of this module does
#: not have to guess what "no answer" means.
DEFAULT_MASTER_FPS = 30


def source_profiles_from_probe(footage: Mapping[str, Any] | None) -> list[dict[str, float]]:
    """``footage_probe`` output → the profile shape :mod:`shared.cadence` reads.

    Sources with an unmeasured rate are DROPPED rather than passed through as
    ``fps=0``: a zero is not a frame rate, and letting it into the dominant-rate
    vote would let unprobed clips outvote probed ones and silently resolve to the
    30 fps default. Dropping them means "we have no opinion from this clip".

    Every field is re-checked at runtime even though the annotation promises a
    mapping: this reads a PRIOR STAGE OUTPUT, which is JSON that has round-tripped
    through the database, not a value any type checker saw constructed.
    """
    if not isinstance(footage, Mapping):
        return []
    sources = footage.get("sources")
    if not isinstance(sources, list):
        return []
    profiles: list[dict[str, float]] = []
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        try:
            fps = float(source.get("fps") or 0.0)
        except TypeError, ValueError:
            continue
        if fps > 0:
            profiles.append({"fps": fps})
    return profiles


def resolve_master_fps(
    config_snapshot: Mapping[str, Any] | None,
    footage: Mapping[str, Any] | None,
) -> int:
    """The master frame rate both ``cut_plan`` and ``compile`` must build on.

    Returns an int on the legal broadcast set. Never raises: an unusable snapshot
    or an unprobed shoot degrades to :data:`DEFAULT_MASTER_FPS`, which is the rate
    the compiler would have used anyway — so the worst case is today's behaviour,
    not a new failure mode.
    """
    snapshot = config_snapshot if isinstance(config_snapshot, Mapping) else {}
    cadence = snapshot.get("cadence")
    intent = cadence if isinstance(cadence, Mapping) else None
    spec = reconcile_cadence(source_profiles_from_probe(footage), intent)
    return spec.target_fps if spec is not None else DEFAULT_MASTER_FPS
