"""The ONE loudness number, and the ship window DERIVED from it.

⭐ WHAT THIS MODULE EXISTS TO MAKE IMPOSSIBLE. Loudness was, until 2026-09-16,
three numbers in three files that had to agree and were held together by nothing
but prose:

* ``gestalt_metrics._DEFAULT_LOUDNESS_WINDOW`` — the band the ship gate convicts
  against, ``(-14.0, -11.0)``;
* ``mastering_plan._DEFAULT_LOUDNESS_TARGET`` — the target the orchestrator
  threads when a run authors a ``mastering`` block, ``-12.5``;
* ``encode_master._LOUDNESS_LO/_HI`` — the range the mux-mastering-advisor may
  clamp a proposed target into, ``(-16.0, -13.0)``.

Every pair of those disagreed, and each file carried a comment asserting the
relationship it did not enforce. Now all three DERIVE from the two constants
below, so a window that does not contain the target is not a bug you can write:
:func:`ship_loudness_window` *computes* the band as target ± a positive
half-width. There is no second place to edit.

⛔ THE TARGET IS NOT OURS TO CHOOSE. It belongs to ``video_compiler_service``:
see :data:`COMPILER_MASTER_LUFS`.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "COMPILER_MASTER_LUFS",
    "SHIP_TOLERANCE_LU",
    "ship_loudness_window",
]

# ⭐ VENDORED CROSS-REPO CONSTANT — the integrated-loudness target (LUFS) that the
# compiler ACTUALLY applies. This orchestrator does not get a vote; it may only
# agree or be wrong.
#
# SOURCE OF TRUTH, both on ``video_compiler_service`` ``origin/main``:
#   * ``src/config.py``            → ``Settings.master_lufs``      default -14.0
#   * ``src/services/mastering.py``→ ``DEFAULT_MASTER_LUFS``       = -14.0
# Landed together in compiler PR #217 (2026-09-12, ``e007e14``), an AUDIO-QUALITY
# PR, replacing the earlier -12.5. Its own docstring states the reason verbatim:
#
#   "-14 is the streaming-platform reference (YouTube/Spotify both normalise TO
#    it, so anything hotter is turned back DOWN on playback and only the extra
#    compression survives). The previous -12.5 asked a bus that already arrives
#    at -18 LUFS per narration clip for another ~5.5 dB of gain riding, which is
#    what made the narration breathe."
#
# So -12.5 is not an alternative opinion — it is the ABANDONED value, and the
# audible artefact it caused is the owner's original complaint. Any comment in
# this repo calling -14 "stale" is itself the stale one; two such comments were
# removed in the commit that created this file.
#
# WHY A VENDORED LITERAL AND NOT A DERIVED READ. The CI job "Vendored compiler
# contract is fresh" (``.github/workflows/ci-cd.yaml``) compares
# ``tests/contracts/compile_request.schema.json`` against the compiler's promoted
# image — but that schema is the REQUEST-OVERRIDE shape: both ``master_lufs`` and
# ``MasteringProfile.loudness_target`` carry ``"default": null`` there, because a
# request that omits them means "use your own". The compiler's Settings default
# is deliberately NOT in the wire contract, so the freshness job cannot pin this
# number and re-vendoring the schema will never update it. It is a hand-carried
# literal, and the honest thing is to say so rather than to imply a gate that
# does not exist. If the compiler moves ``master_lufs`` again, THIS LINE is what
# a human must change; ``tests/services/stages/shared/test_loudness_contract.py``
# then holds everything else in step automatically.
COMPILER_MASTER_LUFS: Final[float] = -14.0

# Half-width (LU) of the ship gate's integrated-loudness window around the target.
#
# ⚠️ THIS IS NOT A WIDENING. The band was ±1.5 LU before this change and is ±1.5 LU
# after it; only its CENTRE moved, from an orchestrator-local -12.5 to the target
# the compiler actually applies. The three live demo_video failures on 2026-09-16
# (-14.2 / -14.1 / -14.1 LUFS convicted against ``(-14.0, -11.0)``) were not loud
# films and not a too-tight tolerance: the band's FLOOR sat exactly ON the target,
# so single-pass loudnorm's ordinary 0.1-0.2 LU undershoot convicted every
# correctly-mastered master. Recentring fixes the disagreement; keeping the width
# is what keeps this a fix rather than a goalpost move.
#
# ±1.5 LU is retained (not re-derived) on purpose: it is the width the gate has
# always enforced, it is ~7x the measured single-pass residual, and nothing in
# this change is evidence about width. Narrowing it is a separate, measured
# decision — as is widening it, which is NEVER the right answer to a conviction.
SHIP_TOLERANCE_LU: Final[float] = 1.5


def ship_loudness_window() -> tuple[float, float]:
    """The ship gate's DEFAULT integrated-LUFS window: the target ± the tolerance.

    Structural, not asserted: because the band is COMPUTED from
    :data:`COMPILER_MASTER_LUFS` and a strictly positive half-width, a window that
    excludes the target the compiler aims at cannot be expressed. That is the
    property the old three-literals arrangement could only claim in prose.

    Returns ``(lo, hi)`` with ``lo < hi``, both LUFS (negative).
    """
    return (COMPILER_MASTER_LUFS - SHIP_TOLERANCE_LU, COMPILER_MASTER_LUFS + SHIP_TOLERANCE_LU)
