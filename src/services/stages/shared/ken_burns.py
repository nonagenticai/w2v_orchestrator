"""Ken Burns (on by default) — animate a compile shot's still image into a pan/zoom clip.

After ``build_shots``, any shot whose final visual is a *still image*
(``image_url`` present, no ``video_url``) can be upgraded to a deterministic
pan/zoom clip rendered through the ``ken_burns`` motion template on
video_production_service — the exact render path ``motion_graphics_gen`` and the
animated slides use (record → poll → fetch → store). One post-pass over the shot
list, called from ``CompileStageBase.handle``, so it covers every still-bearing
video type uniformly (generic AI/curated images, promo still slides, the lesson
slide PNG, …) without per-stage wiring.

Per-shot soft-fail: a shot whose clip fails keeps its still ``image_url`` and the
compile proceeds. Motion is ON BY DEFAULT for stills; it is tuned (or turned off)
via ``config_snapshot["ken_burns"]``::

    ken_burns:
      enabled: false         # explicit OPT-OUT — the only way to get frozen stills
      preset: zoom_in        # zoom_in (default) | zoom_out | pan_{left,right,up,down}
      duration_seconds: 5.0   # FALLBACK only — a shot's own ``seconds`` wins

The clip is cut to the SHOT's own ``seconds`` (clamped to
``[_MIN_DURATION_S, _MAX_DURATION_S]``); ``duration_seconds`` above is only the
fallback for a shot that carries none. A clip shorter than its shot is
freeze-padded by the compositor, which is the frozen-still defect all over again.

Absent block, empty block, or a block without ``enabled`` → ENABLED, with every
sub-knob on its ``_DEFAULT_*``. Only an explicitly falsy ``enabled`` returns the
shot list unchanged. The block was presence-opt-in until 2026-08-17, but nothing
ever set it (``PlanDocument.ken_burns`` defaults to ``{}``, ``to_brief_patch``
skips empty dicts, the A0 planner never resolves it, gitops never sets it), so
the gate was unreachable and every still-based film shipped as frozen single
frames.

The shot's ``image_url`` is *expected* to be the run's OWN storage URL (a
presigned MinIO artifact), but it is a value carried on a shot dict — the same
dict real-media / operator-pinned URLs flow through — so it is NOT fetched
blind. It goes through ``w2v_common.net.safe_get`` like every other URL sink in
this tree: resolve once, pin the validated IP (no DNS-rebinding TOCTOU), and
re-validate every redirect hop (the client no longer runs
``follow_redirects=True``, which used to make a single ``302`` into the Tailnet
escape the fetch entirely).

``visual_media.fetch_image_bytes`` is deliberately NOT reused here: it has no
``allow_hosts``, so it would reject our own ``minio.test`` artifact (a CGNAT address) and silently disable ken_burns for every shot. Instead the
configured storage endpoint(s) are passed as ``allow_hosts`` — the shared
guard's documented escape hatch for presigned artifact hosts that legitimately
resolve internal. It is re-checked on every redirect hop, so the trusted host
cannot 302 onward to postgres/k3s. A ``file://`` / relative local-backend URL is
skipped (left as the still).
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import math
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urlsplit
from uuid import uuid7

import httpx
from pydantic import BaseModel, ConfigDict, field_validator
from w2v_common.auth import get_stage_from_context
from w2v_common.net import safe_get
from w2v_common.stages import stage_gather

from src.cache.redis_client import RedisClient, get_redis_client
from src.clients.video_production_client import VideoProductionClient
from src.config import get_settings
from src.services.stages.generic.site_capture import _resolve_recording_quality_preset
from src.services.stages.shared.cadence import build_cadence_spec
from src.services.stages.shared.degradation import (
    KEN_BURNS_ALL_STILLS_FROZEN,
    KEN_BURNS_SKIPPED_NO_BUDGET,
    KEN_BURNS_STILL_QUEUE_SATURATED,
    KEN_BURNS_STILL_SOFT_FAILED,
    Degradation,
    degradation,
)
from src.services.stages.shared.motion_slides import (
    format_dims,
    quality_tier,
    token_fp_for_snapshot,
)
from src.services.stages.shared.step_ledger import (
    JsonValue,
    NullStepLedger,
    StepLedger,
    hash_inputs,
)
from src.services.tts_storage import (
    deterministic_media_id,
    find_existing_media,
    store_generated_video,
)
from src.services.visual_media import MEDIA_FETCH_USER_AGENT

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

log = logging.getLogger(__name__)

_TEMPLATE = "ken_burns"
# ONE poll call's deadline — the client's own ceiling (``poll_job`` raises above
# 600s, see tests/services/stages/test_poll_timeouts_within_client_ceiling.py).
_RECORD_TIMEOUT_S = 600.0
# The WHOLE pass's wall budget, measured from its start. A shot's wait is no
# longer one 600s poll: video_production's job wall time includes its fleet-wide
# recording QUEUE (``max_concurrent_recordings``), and a job still queued or
# rendering at 600s was abandoned — kept as a still while video_production went
# on rendering it for nobody. Run 01a10d84 (2026-10-05): 2 of 4 stills at 4K
# timed out exactly that way, the film shipped 61% duplicate frames, and the two
# frozen stills are also the two shots that breached the banding limit. Now a
# shot keeps polling the SAME job (no re-submit, no second bill) until this
# deadline. The compile stage hands this pass its OWN budget (this value) and
# moves its aggregate pre-submit deadline out to hold it
# (``compile_stage._ensure_ken_burns_room``). It used to hand over only what was
# left of that deadline after title cards, and on an uncontended run the third
# recorder wave (9 stills / 3 slots) was starved: 3 of 9 shots kept as stills
# after waiting only 280-395s.
_PHASE_BUDGET_S = 1000.0
# stage_gather's per-item ceiling over the phase budget: fetch + prescale + upload.
_PER_ITEM_OVERHEAD_S = 120.0
# Per-item ceiling for ``stage_gather`` at the full budget.
_PER_ITEM_TIMEOUT_S = _PHASE_BUDGET_S + _PER_ITEM_OVERHEAD_S
# Below this there is no time to record even one clip: keep every still, say so.
_MIN_PHASE_BUDGET_S = 30.0
# How video_production says "no recording slot freed up in time". Its
# ``JobStatusResponse`` carries NO structured error code — only a free-text
# ``error`` that STARTS with this marker (``QUEUE_SATURATED: no per-pod
# recording slot within 300s ...``), which ``w2v_common`` ``poll_job`` re-raises
# as a plain ``RuntimeError("video_production job <id> failed: QUEUE_SATURATED:
# ...")``. So the exception TEXT is the only discriminator there is. The string
# is pinned on the other side by video_production_service itself
# (``QUEUE_SATURATED_ERROR_PREFIX`` in ``src/services/recording/concurrency.py``
# and the tests that assert on it): if it is ever renamed there, this stops
# matching and a saturated shot silently degrades to the generic soft-fail —
# exactly the pre-retry behaviour, never a crash.
_VP_QUEUE_SATURATED_MARKER = "QUEUE_SATURATED"
# What precedes the job's ``error`` text in the ``RuntimeError`` the locked
# ``w2v_common`` ``poll_job`` raises (``f"{service} job {id} failed: {error}"``).
_POLL_JOB_FAILED_SEP = " failed: "
# How long video_production lets a job WAIT for a recording slot before failing
# it QUEUE_SATURATED: ``recording.queue_wait_timeout_seconds`` in
# video_production_service ``src/config.py`` (default 300.0). A re-submitted job
# may spend all of it queued before a single frame is recorded, so a retry that
# has less than this (plus the record floor) left is a recording paid for nobody.
_VP_QUEUE_WAIT_TIMEOUT_S = 300.0
# Remaining pass budget, AFTER the backoff, below which no saturation retry is
# started: the worst-case queue wait plus the floor for recording one clip.
_SATURATION_RETRY_MIN_BUDGET_S = _VP_QUEUE_WAIT_TIMEOUT_S + _MIN_PHASE_BUDGET_S
# Re-submissions of ONE shot after a saturation failure (so up to 1 + this many
# recordings are attempted). Saturation is capacity, not a defect in the shot:
# the same request succeeds once a slot frees.
_SATURATION_RETRIES = 3
# Backoff before re-submission ``n`` (1-based): ``base * 2**(n-1)`` plus up to
# one ``base`` of jitter, so the shots of one pass — which all saturate within
# the same second — do not re-queue as one wave. Short on purpose: the failed
# attempt has already waited video_production's full 300s slot timeout.
_SATURATION_BACKOFF_BASE_S = 5.0
# Stills render in parallel up to video_production's GUARANTEED capacity: its
# per-pod recording cap (``per_pod_max: 3``, gitops ``w2v/deployments.yaml``)
# times its MINIMUM replica count (``video-production-service-hpa`` min 1 / max 2
# — read live 2026-10-06: 1 replica). At 2, a promo's 4-5 stills ran in two or
# three serial waves and the later waves timed out; at 4 (sized for "2 pods x
# 3") a one-pod fleet always queued the fourth still, and run 01a10fcf's fourth
# failed ``QUEUE_SATURATED: no per-pod recording slot within 300s (per-pod cap 3)``.
_RECORDER_PER_POD_MAX = 3
_RECORDER_MIN_REPLICAS = 1
_CONCURRENCY = _RECORDER_PER_POD_MAX * _RECORDER_MIN_REPLICAS
_FETCH_TIMEOUT_S = 20.0
_MAX_IMAGE_BYTES = 12 * 1024 * 1024  # mirror visual_media's image cap
_DEFAULT_DURATION_S = 5.0
# Per-shot duration bounds. A ken_burns clip SHORTER than the shot it backs gets
# freeze-padded by the compositor (``_freeze_pad_suffix`` clones the last frame),
# which reintroduces exactly the frozen-still defect this pass exists to remove —
# so the shot's own ``seconds`` wins over the one config value. The clamp bounds
# what a browser recording may cost: under 1.5s a pan/zoom reads as a jitter and
# the record round-trip dominates the render; over 20s the recording grows without
# any editorial gain (a still held that long is a storyboard problem, not a motion
# one) and starts crowding ``_RECORD_TIMEOUT_S``.
_MIN_DURATION_S = 1.5
_MAX_DURATION_S = 20.0
_DEFAULT_PRESET = "zoom_in"
_VALID_PRESETS = frozenset({"zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down"})
_DEFAULT_FIT = "contain"
# "cover" fills the frame and crops the over-scan — correct for photos whose
# source is >= the frame. "contain" (default) keeps a frame-sized designed still
# (slide / text card) fully in view: a blurred backdrop + a sharp layer whose
# motion never scales past frame-fill, so edge content (e.g. a title) is never
# cropped. Set `ken_burns.fit: cover` for photo/footage-bearing videos.
#
# The default was "cover" while this pass was opt-in and unreachable. Motion is
# now ON BY DEFAULT for every bare still, so the default has to be the SAFE mode,
# not the pretty one: `ken_burns.html` runs zoom_in as a raw `scale(1 -> 1.14)`
# with NO safe-area inset on the cover path, so the last frames lose ~6.1% off
# every side (~118px per side at 1920 wide). Applied by default to a lesson
# slide, a chart or a website screenshot, that silently crops edge text. The
# contain path is bounded (SAFE_MIN 0.92 -> 1.0, "never exceeds the frame"), so
# it is the only mode that is correct for stills we did not choose. This also
# stops the module contradicting its own guidance, which already said to use
# contain for slide-bearing videos.
_VALID_FITS = frozenset({"cover", "contain"})
# Recording cadence fallback. The ``record_animation`` call HARDCODED ``fps=30``
# until 2026-09-15, while the compiler conforms the master to whatever
# ``_resolve_master_cadence`` picks out of {24, 25, 30, 50, 60}. That mismatch is
# not merely a judder risk, it makes the compiler's own conform arithmetic WRONG:
#
# * The clip reaches the compiler on the footage path with no ``Shot.target_fps``
#   (``action_reel/compile.py`` declines to stamp one and nothing else passes
#   ``build_shots(target_fps=...)``), so ``_cadence_fragment`` sees ``source_fps
#   is None`` and emits a bare decimating ``fps=<master>`` — the interpolation
#   branch is structurally unreachable for a source whose rate is unknown.
# * A Ken Burns clip is, by construction, ONE continuous linear transform for its
#   whole length. It is the single clip class where uneven decimation reads as
#   visible cadence judder rather than as nothing: 30→24 drops 6 of every 30
#   frames on a smooth ramp.
#
# Recording AT the master rate makes the conform an exact 1:1 no-op instead of a
# lossy resample, which is also what the deck motion clips already do
# (``motion_slides``: ``fps = int(motion_cfg.get("fps") or master_fps or 30)``).
_DEFAULT_FPS = 30


def _master_fps(config_snapshot: Mapping[str, Any], override: int | None = None) -> int:
    """The cadence to RECORD a ken_burns clip at, so the compiler need not resample.

    *override* is the plumbing seam: ``CompileStageBase.resolve_cadence`` is a
    HOOK precisely because a product whose master rate is a property of its
    FOOTAGE rather than of its config has nowhere else to say so
    (``action_reel`` overrides it to fold in ``footage_probe``'s measured source
    rates). A caller that has already resolved the master rate passes it here and
    it wins outright; ``None`` falls back to the base's own config-only
    resolution, which is byte-identical to what ``resolve_cadence`` returns for
    every other product.

    ``build_cadence_spec`` is presence-is-opt-in (absent ``cadence`` block AND no
    carried ``source_profiles`` ⇒ ``None``), so an un-opted run keeps 30 and its
    clip bytes are unchanged. Duplicated rather than imported from
    ``slide_stage._cadence_master_fps`` because ``slide_stage`` pulls in the
    compile-stage surface and this module is on ``CompileStageBase``'s own import
    path.

    A snapshot shape ``build_cadence_spec`` cannot read must not cost the run its
    motion: this is a TUNING knob and the pre-existing behaviour (30) is itself a
    legal cadence, so a raising resolver degrades to the default rather than
    failing the whole compile stage the way an unguarded call would.
    """
    if override is not None:
        return int(override)
    try:
        spec = build_cadence_spec(dict(config_snapshot))
    except Exception as exc:  # noqa: BLE001 -- tuning only; 30 stays valid
        log.warning("ken_burns: cadence unresolvable, recording at %dfps: %r", _DEFAULT_FPS, exc)
        return _DEFAULT_FPS
    return spec.target_fps if spec else _DEFAULT_FPS


def _data_uri(data: bytes, content_type: str) -> str:
    """Base64 data-URI for the recorder's ``window.PARAMS.image`` (offline, no fetch)."""
    mime = content_type if content_type.startswith("image/") else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


# --- O12: pre-upscale the still so the BROWSER never resamples it up ----------
#
# The largest upscale in the still path is not performed by any filter we chose —
# it is performed by Chromium's default image smoothing, and the template has no
# ``image-rendering`` hint to steer it. An AI still arrives at 1536 long edge
# (``scripts/e2e_validate.py`` hard-asserts that), the recorder sets the viewport
# to the full target dims and rasterises at ``device_scale_factor = 2``, so a
# default 2K run rasterises 5120x2880 — the still is smoothed up 3.33x before a
# single frame exists. video_compiler already records the cost of exactly this
# class of upscale (``renderer.py`` ~3143: variance-of-Laplacian 28.22 bilinear
# vs 47.57 lanczos on a 1.33x upscale).
#
# MEASURED, not argued (3 real images, 2 timepoints, content-region only, the
# REAL template through the REAL recorder geometry, vs a lanczos-downscaled
# ground truth). Pre-upscaling to the recording dims:
#
#   image      variance-of-Laplacian      PSNR vs ground truth
#   photo      154.24 ->  178.24  (+15.6%)   31.14 -> 32.22 dB  (+1.07)
#   photo2     333.18 ->  384.12  (+15.3%)   26.26 -> 26.91 dB  (+0.65)
#   graphic     41.65 ->   50.24  (+20.6%)   41.35 -> 44.46 dB  (+3.11)
#
# BOTH metrics move the same way, which is the load-bearing part: a sharpness
# score alone rewards ringing, so "sharper" is only believable when fidelity to
# the ground truth improves too. This is NOT the killed ``_zoompan_prescale``
# (which measured WORSE, 17.08 -> 13.58, because it prescaled an input that was
# already frame-sized) — here the input is genuinely 1536 against a 5120 raster.
#
# O12b: taken further, and WITHOUT the coupling that blocked it. Targeting the
# recorder's full CAPTURE RASTER rather than the ordered frame is better again,
# because it removes a whole bilinear round-trip from the path: at the ordered
# dims Chromium still has to smooth the still UP to its supersampled surface and
# the recorder then lanczos-downscales it back, and that round-trip is where the
# detail goes. Re-MEASURED on this branch (4 real stills at 1536 long edge, the
# real 2k geometry — order 2560x1440, reported raster 5120x2880 — vs a
# single-lanczos reference render of the same source):
#
#   Chromium smoothing modelled as BILINEAR   lapvar            PSNR
#     promo frame (1536x1016)      56.78 ->  77.17 (+35.9%)   +4.85 dB
#     photo  m_086 (1536x864)     196.47 -> 299.59 (+52.5%)  +10.06 dB
#     ui     fs_system            161.83 -> 237.89 (+47.0%)  +10.09 dB
#     ui     homehold_a           179.57 -> 267.23 (+48.8%)   +9.32 dB
#   Chromium smoothing modelled as BICUBIC (the conservative model)
#     promo frame                  72.01 ->  77.17  (+7.2%)   +4.89 dB
#     photo  m_086                269.43 -> 299.59 (+11.2%)   +4.16 dB
#     ui     fs_system            216.43 -> 237.89  (+9.9%)   +4.99 dB
#     ui     homehold_a           241.12 -> 267.23 (+10.8%)   +4.65 dB
#
# Sharpness AND fidelity improve in all 8 cells, under both smoothing models —
# the same both-metrics bar the O12 measurement above had to clear, and the
# reason this is not another ``_zoompan_prescale`` (which moved them opposite
# ways). JPEG q95 payloads land at 0.84-1.18MB, well inside
# ``_MAX_UPSCALED_BYTES``.
#
# The coupling that refused this before is gone: the raster is no longer GUESSED
# from the recorder's private ``_CAPTURE_SCALE``, it is REPORTED by the recorder
# on its own recording result (``capture_width`` / ``capture_height``,
# video_production_service #182) and observed here. Unreported (NULL) keeps
# today's behaviour exactly. See :func:`_observed_capture_raster`.
_UPSCALE_JPEG_QUALITY = 95
# Encoded budget for the pre-upscaled still. Measured JPEG q95 payloads at
# 2560x1440 are 1.0-1.6MB and keep the PNG's whole quality win (178.24 vs 176.84
# lapvar, 32.22 vs 32.26 dB) at a quarter of the bytes — but the ALPHA path has
# to stay PNG and a 2560x1440 PNG measured 4.4-6.8MB. Past this budget the
# upscale is DROPPED and the original still is sent, i.e. today's behaviour: a
# soft picture is a far better outcome than a multi-megabyte JSON body.
_MAX_UPSCALED_BYTES = 4 * 1024 * 1024
# Output-pixel ceiling for the resize, so a pathological source aspect (a 1x4000
# strip, a mis-tagged sprite sheet) cannot turn a cover-fit factor into a
# gigapixel resize, and so no still can make this stage a memory event.
#
# ONE clause, because there is only one cost. An earlier revision of this branch
# expressed the bound as a multiple of the TARGET area (4x) AND an absolute pixel
# count, refusing only when both were exceeded. That conjunction could not state
# the memory property at all: a 2000x4000 portrait cover-fitted to a 5120x2880
# raster is 5120x10240 = 52.4M px (150 MiB of RGB destination) but only 3.56x the
# target area, so the relative clause was false and the resize was ADMITTED —
# past an absolute ceiling written to stop exactly that. The measured geometries
# show why the two can never be AND-ed: they anti-correlate.
#
#   750x2000  -> 1920x1080   9.83M px   4.74x target    28 MiB   cheap, high ratio
#   2000x4000 -> 5120x2880  52.43M px   3.56x target   150 MiB   costly, low ratio
#
# The ratio is not a proxy for the cost, so the ratio cannot guard the cost. And
# the relative clause is not needed for its OWN stated purpose either: an
# absolute ceiling already bounds an unbounded cover fit by construction (the
# 1x4000 strip is 1.47e10 px at 1080p — over any sane number). Keeping it would
# have forced a headroom of >=16x, so that 720p x headroom still cleared the
# 14.75M the old ceiling admitted; at 16x it is inert at 1080p and above, which
# is precisely where the expensive geometries are. A constant that is inert where
# the cost lives is not a guard, so it is gone rather than left to be read as one.
#
# This RETIRES the cross-repo duplicate that motivated the relative form. The
# constant here used to be ``5120 * 2880``, which restated
# video_production_service's private ``_MAX_CAPTURE_RASTER_PIXELS``
# (``2560*1440*_CAPTURE_SCALE**2``) from another repo — two copies of one number,
# with nothing to keep them equal. The fix for that is to state the ceiling on
# ITS OWN terms (the cost of the resize) rather than to make it relative: this
# number is derived from memory below and is deliberately NOT ``5120 * 2880`` nor
# a multiple of it, so nothing should make a reader think it tracks
# ``_CAPTURE_SCALE``.
#
# WHERE THE NUMBER COMES FROM — the memory budget, not a test. Pillow's LANCZOS
# resize allocates a horizontal-pass intermediate (``out_w x src_h``) and the
# destination (``out_w x out_h``), 3 B/px for RGB and 4 for RGBA; the destination
# dominates. 28M px is 80 MiB of RGB destination (107 MiB RGBA), ~110-140 MiB
# peak with the intermediate and the encode buffer. That is the largest resize
# that is not a latency-and-memory event on a pod sized for a compile stage,
# independent of what any recorder would have asked for.
#
# What it admits and refuses, against the measured set:
#   * NON-REGRESSION, structurally: 28M > 14_745_600, the old ceiling, and the
#     test is a single comparison on ``out_px`` that does not mention the target.
#     So EVERY geometry the old ceiling admitted is still admitted at EVERY
#     target, whatever its aspect — including the tall captures a relative bound
#     would have dropped (750x2000 -> 1080p at 9.83M, 640x1800 at 10.37M).
#   * The measured win survives with 1.61x of margin: the 1536x1016 (1.512:1)
#     promo frame at the 5120x2880 raster is 5120x3387 = 17.34M px (50 MiB). The
#     old 14.75M ceiling DROPPED exactly that image and silently shipped it
#     un-upscaled — i.e. the duplicate was the binding constraint on the win.
#   * An ordinary 1:1 still at the largest raster is 5120x5120 = 26.2M (75 MiB)
#     and is admitted; a square screenshot must not lose its upscale.
#   * A 3:4 portrait at that raster is 34.95M (100 MiB) and a 1:2 portrait is
#     52.43M (150 MiB): both refused. Under a cover fit the delivered picture is
#     never more than the target area, so those extra pixels are cropped — the
#     clip pays the memory and shows none of it. Refusal costs only the sharpen:
#     the original still is sent, which is today's behaviour.
#   * A 1x4000 strip is refused at every target (1.05e11 px at the raster,
#     1.47e10 at 1080p).
_MAX_UPSCALE_ABS_PIXELS = 28_000_000


# Capture rasters the RECORDER has told us it used, keyed by the order it was
# given. Process-lifetime, bounded by the number of distinct orders this pod
# ever places (a handful: 4 formats x 3 tiers x a few presets).
#
# Why a cache and not a request parameter: ``capture_width``/``capture_height``
# are reported on a recording RESULT, so the raster for a recording cannot be
# known before that recording exists, and video_production_service exposes no
# preflight for it. But the recorder resolves the raster with
# ``_resolve_capture_scale(width, height)`` — a pure function of the order — so
# ONE completed recording at a given order reports the raster for EVERY
# recording at that order, until the recorder itself changes. So: observe it,
# key it by the order, and apply it to subsequent stills. The first batch of a
# cold start prescales to the ordered dims exactly as before; a still whose
# FIRST dispatch comes after the first completed recording is sized to the real
# raster. (A still that was already dispatched keeps the target it was first
# sent with — see ``_pin_shot_prescale_target`` for why it may not move.)
#
# This is the whole point of the contract being reported rather than assumed:
# the day ``_CAPTURE_SCALE`` moves, or an order gets clamped down to 1x, the
# NEXT observation carries the new number and the prescale follows it. Nothing
# here restates it.
_OBSERVED_CAPTURE_RASTER: dict[tuple[int, int, str], tuple[int, int]] = {}


def _raster_key(width: int, height: int, quality_preset: str | None) -> tuple[int, int, str]:
    """Cache key: the ORDER, which is what the recorder resolves the raster from.

    ``quality_preset`` is part of the key because it can override the recorder's
    viewport dims, so two orders with the same ``width``/``height`` but different
    presets are not guaranteed to capture at the same raster.
    """
    return (width, height, quality_preset or "")


def _observed_capture_raster(
    width: int, height: int, quality_preset: str | None
) -> tuple[int, int] | None:
    """The raster the recorder REPORTED for this order, or ``None`` if unobserved.

    ``None`` is the "not reported" case the upstream contract defines, and every
    caller must treat it as today's behaviour — fall back to the ordered dims.
    """
    return _OBSERVED_CAPTURE_RASTER.get(_raster_key(width, height, quality_preset))


def _observe_capture_raster(
    width: int, height: int, quality_preset: str | None, status: Mapping[str, Any]
) -> None:
    """Record ``capture_width``/``capture_height`` off a completed recording.

    Deliberately total: anything that is not a pair of positive ints at least as
    large as the order is DISCARDED rather than stored, so a NULL (the recorder
    did not report — old rows, non-supersampling recorders), a malformed value,
    or a nonsensical raster smaller than the frame it produced all leave the
    cache empty and the prescale on its pre-feature path. A reported raster
    EQUAL to the order is a legitimate, meaningful observation — it is how the
    contract expresses "supersampling was clamped down" (a 4k order) — so it is
    stored, and it makes the prescale a no-op change rather than an upscale to a
    raster that does not exist.
    """
    raw_w, raw_h = status.get("capture_width"), status.get("capture_height")
    if not isinstance(raw_w, int) or not isinstance(raw_h, int):
        return
    if isinstance(raw_w, bool) or isinstance(raw_h, bool):
        return
    if raw_w < width or raw_h < height:
        return
    _OBSERVED_CAPTURE_RASTER[_raster_key(width, height, quality_preset)] = (raw_w, raw_h)


# ---------------------------------------------------------------------------
# The observed raster, SHARED across pods.
#
# ``_OBSERVED_CAPTURE_RASTER`` alone is process-local, and that made the
# prescale target a function of pod warmth — which reaches the recorder's
# idempotency cache. video_production_service keys a replay on the
# ``idempotency_key`` AND a hash of the request BODY (w2v_common
# ``middleware/idempotency.py``, by design), and the body carries the prescaled
# still as a data URI. So a warm pod submitted a clip prescaled to the real 2x
# raster, the pod rolled mid-compile, and the cold pod resubmitted the SAME key
# with the still prescaled to the ordered dims: different bytes, different body
# hash, no replay — a SECOND paid recording of a clip already in flight.
#
# The media id cannot absorb this (see the ⛔ block in ``apply_ken_burns``), so
# the observation itself is made to outlive the process: written to the
# orchestrator's Redis on observation, read back ONCE per ``apply_ken_burns``
# call when the process dict misses. A cold pod then prescales to the raster the
# warm pod used, and its resubmit is byte-identical ⇒ a replay.
#
# Fail SOFT in both directions. This is an optimisation of a soft-degrading
# sharpen; nothing here may fail a compile. Redis down, slow, unconnected, or
# holding garbage all mean exactly the pre-feature behaviour (cold = ordered
# dims), announced once per process.
#
# The shared raster ALONE did not close it: the very first pod to place an order
# when the shared key is absent (first ever, or >24h since the last observation)
# submits its first ``_CONCURRENCY`` shots at the ordered dims, and the raster
# THOSE recordings publish then moved the target of those same shots under the
# successor pod after a roll — and under an in-process ``stage_gather`` retry of
# a first-batch shot once a sibling had completed. Same key, different bytes,
# paid twice. The per-SHOT pin below (``_pin_shot_prescale_target``) closes
# both: the observed/shared raster now only chooses the target a shot is FIRST
# dispatched with; it never changes the target of a shot already dispatched.
# ---------------------------------------------------------------------------
_SHARED_RASTER_PREFIX: Final = "w2v_orchestrator:ken_burns:capture_raster:v1"
_SHARED_RASTER_TTL_S: Final = 24 * 60 * 60
# A hung Redis must cost one bounded wait, not a compile stage.
_SHARED_RASTER_IO_TIMEOUT_S: Final = 2.0
# Upper bound on a raster axis we will believe from the cache. The recorder's
# largest real raster is 5120 wide; anything past this is not a capture raster,
# and ``_prescale_still`` would only refuse it on cost anyway.
_SHARED_RASTER_MAX_AXIS: Final = 16_384
# Re-publish cadence. Every completed recording re-observes the same value, so
# without a throttle each shot would cost a SET; with one that never refreshes,
# a long-lived pod would let the 24h key expire under it. Success is refreshed
# hourly; a failed write is retried after a minute, so a Redis outage costs one
# bounded attempt per minute rather than one per shot.
_SHARED_RASTER_REFRESH_S: Final = 60 * 60.0
_SHARED_RASTER_RETRY_S: Final = 60.0
# raster key -> (value last offered to the shared cache, monotonic time of the
# next allowed attempt). Process-local bookkeeping only.
_SHARED_RASTER_NEXT_PUBLISH: dict[tuple[int, int, str], tuple[tuple[int, int], float]] = {}
_shared_raster_warned = False


# ---------------------------------------------------------------------------
# The prescale target, PINNED per shot at its first dispatch.
#
# The invariant the recorder's replay needs is narrow: for a given
# (run, media id) — i.e. for a given idempotency key — the request body is
# byte-identical on every attempt, in every pod. The observed raster breaks it
# because it appears mid-run, the moment the first recording completes. So a
# shot reads it exactly ONCE: at its first dispatch it resolves a target
# (observed/shared raster if one is known by then, else the ordered dims),
# claims it with a set-if-absent under a key carrying the run id and the media
# id, BEFORE the paid ``record_animation`` call, and every later attempt for
# that shot — a ``stage_gather`` retry, a re-entered compile, a successor pod —
# reads the claim and prescales to it, whatever its own process believes by then.
#
# Per SHOT, not per run, on purpose: a run-wide pin would hold every still of a
# run that started cold at the ordered dims. Pinned per shot, picture quality is
# exactly what it was before the pin existed — only the first ``_CONCURRENCY``
# shots of a cold start are soft; a shot first dispatched after the raster is
# known pins the sharp raster.
#
# Where the pin LIVES (2026-10-06): the run's step ledger is the source of
# truth — row ``kb_pin:{media_id}`` of the calling stage (``stage_step_results``,
# Postgres, no TTL, lives as long as the run). Redis is consulted only when the
# ledger has no row for the shot: its set-if-absent arbitrates two dispatches
# racing the FIRST claim, and it carries pins written before the ledger held
# them. Whatever is resolved there is then written to the ledger, so a re-entry
# — on any pod, after any Redis outage, at any age — reads the ledger and sends
# the same bytes. ``_SHOT_PIN_TTL_S`` only bounds the Redis copy now.
#
# What can STILL be paid twice, stated plainly:
#   * The ledger write failing at a shot's FIRST dispatch (Postgres down — the
#     run itself cannot persist then either) AND Redis unreachable, then a pod
#     roll: the pin was held in this process only (``_SHOT_PRESCALE_PIN``), so
#     the successor resolves its own target and — if the raster became visible
#     in between — sends different bytes.
#   * Two pods dispatching the same shot concurrently with Redis down: the
#     ledger has no in-process dedupe, both offer their own target.
#   * Anything else that changes the body under the same key: a source image
#     whose BYTES changed at the same URL path, or a Pillow/encoder difference
#     between the two pods' images altering the re-encoded still. Neither is new.
#
# A user restamp deletes the calling stage's ledger rows — the pin AND the
# recording generation (``_recording_generation``) together — so the re-run
# both re-pins and submits under a NEW idempotency key. The ledger rows move as
# one. (The Redis copy is not generation-scoped and may hand the previous target
# back to that re-pin — harmless: it is the FIRST dispatch under the new key, and
# it is recorded in the ledger from then on.)
#
# What is REFUSED rather than risked: the claim is lost (another dispatch of
# this shot holds the pin) and the winner cannot be read back — read error,
# timeout, or a value that fails validation. That is the one state where we
# KNOW a paid job for this key probably exists and do not know its bytes, so
# after one full re-try of claim + read the shot raises instead of submitting
# blind; ``_gen`` soft-fails it to its still (a note, no bill). A film with one
# frozen still is recoverable on the next attempt; a second recording is not.
# ---------------------------------------------------------------------------
_SHOT_PIN_PREFIX: Final = "w2v_orchestrator:ken_burns:shot_prescale:v1"
# Same lifetime as the shared raster: long enough to outlive any compile and its
# pod-roll resumes, short enough that finished runs do not accumulate.
_SHOT_PIN_TTL_S: Final = _SHARED_RASTER_TTL_S
# Claim + read-back rounds before a lost-but-unreadable pin refuses the shot.
_SHOT_PIN_ROUNDS: Final = 2
# (run id, media id) -> pinned target. The in-process half of the pin: what a
# retry on this pod uses when Redis is down. Bounded FIFO — one entry per shot
# this pod dispatches.
_SHOT_PRESCALE_PIN: dict[tuple[str, str], tuple[int, int]] = {}
_SHOT_PRESCALE_PIN_MAX: Final = 4096


class _ShotPinUnreadableError(RuntimeError):
    """Another dispatch holds this shot's pin and its value cannot be read."""


def _shot_pin_cache_key(run_id: str, media_id: str) -> str:
    """Redis key for one shot's pinned prescale target.

    ``run_id`` + ``media_id`` is exactly the recorder idempotency key's identity
    (``f"{run_id}:ken_burns:{media_id}"``); the media id already folds the order,
    the preset and the source image, so nothing else distinguishes the body.
    """
    return f"{_SHOT_PIN_PREFIX}:{run_id}:{media_id}"


def _remember_shot_pin(local_key: tuple[str, str], target: tuple[int, int]) -> tuple[int, int]:
    _SHOT_PRESCALE_PIN.pop(local_key, None)
    while len(_SHOT_PRESCALE_PIN) >= _SHOT_PRESCALE_PIN_MAX:
        del _SHOT_PRESCALE_PIN[next(iter(_SHOT_PRESCALE_PIN))]
    _SHOT_PRESCALE_PIN[local_key] = target
    return target


def _ken_burns_ledger(ctx: RunContext) -> StepLedger:
    """The step ledger of the stage this pass runs INSIDE (a compile stage).

    ``apply_ken_burns`` / ``rescue_frozen_stills`` are helpers called from a
    compile stage's ``handle`` and are not handed its name, so they read the
    executing stage from the cost-correlation ContextVar ``DAGExecutor._run_one``
    sets (``set_correlation_context(run_id, spec.name)``) before every handler.
    It MUST be that stage's real name: a restamp deletes ledger rows by the
    blueprint's stage names, and that delete is what moves the recording
    generation. Outside an executor (unit tests, hand-built contexts) the
    ContextVar is empty and the rows go under ``"ken_burns"``.

    A context without ``step_ledger`` (a duck-typed test double) gets a
    :class:`NullStepLedger` — the ledger may never be the reason a pass fails.
    """
    stage_name = get_stage_from_context() or "ken_burns"
    factory = getattr(ctx, "step_ledger", None)
    if factory is None:
        return NullStepLedger(stage_name)
    return factory(stage_name)


def _dump_pin(target: tuple[int, int]) -> JsonValue:
    return [target[0], target[1]]


def _load_pin(width: int, height: int) -> Callable[[JsonValue], tuple[int, int]]:
    def _load(value: JsonValue) -> tuple[int, int]:
        target = _valid_shared_raster(value, width, height)
        if target is None:
            # The ledger treats a raising ``load`` as unreadable and resolves anew.
            raise ValueError(f"stored prescale pin is not a usable raster: {value!r}")
        return target

    return _load


async def _pin_shot_prescale_target(
    run_id: str,
    media_id: str,
    width: int,
    height: int,
    quality_preset: str | None,
    *,
    ledger: StepLedger | None = None,
) -> tuple[int, int]:
    """The ONE prescale target every dispatch of this shot uses.

    Must be awaited BEFORE the paid ``record_animation`` call, on EVERY attempt.

    ``ledger`` (the calling stage's step ledger) is the durable record: a stored
    pin for ``media_id`` is returned without touching Redis; otherwise the target
    is resolved through Redis as below and then stored. ``None`` = Redis only.
    """
    if ledger is None:
        return await _resolve_shot_prescale_target(run_id, media_id, width, height, quality_preset)
    resolved = False

    async def _resolve() -> tuple[int, int]:
        nonlocal resolved
        resolved = True
        return await _resolve_shot_prescale_target(run_id, media_id, width, height, quality_preset)

    target = await ledger.once(
        f"kb_pin:{media_id}",
        hash_inputs("kb_pin", width, height, quality_preset),
        _resolve,
        dump=_dump_pin,
        load=_load_pin(width, height),
    )
    if not resolved:
        # A stored row answered: the target an EARLIER dispatch was sent with.
        _log_shot_pin(media_id, "ledger", target, width, height)
    return _remember_shot_pin((run_id, media_id), target)


def _log_shot_pin(
    media_id: str, source: str, target: tuple[int, int], width: int, height: int
) -> None:
    """One line per kb_pin lookup: WHERE the shot's prescale target came from.

    ``ledger`` = a stored pin (re-entry / retry); ``redis-won`` = this dispatch
    claimed the shot first; ``redis-lost`` = an earlier dispatch's claim was read
    back; ``local`` = the cache was unavailable, the pin is this process's only.
    """
    log.info(
        "ken_burns: kb_pin media=%s source=%s target=%dx%d (ordered %dx%d)",
        media_id,
        source,
        target[0],
        target[1],
        width,
        height,
    )


async def _resolve_shot_prescale_target(
    run_id: str, media_id: str, width: int, height: int, quality_preset: str | None
) -> tuple[int, int]:
    """Resolve a shot's pin through Redis (the claim arbiter for a FIRST dispatch).

    The shared pin always wins when it can be read: this process's own choice
    (its earlier in-process pin for the shot, else the observed raster, else the
    ordered dims) is only what it OFFERS. ``set_nx`` is tri-state (w2v_common
    ``RedisClient``): ``True`` = the offer is now the shot's pin; ``False`` =
    an earlier dispatch got there first, so read the winner back; ``None`` = the
    cache is unavailable, so the offer stands for this process only.

    Never raises for an unavailable cache (soft: in-process pin). Raises
    :class:`_ShotPinUnreadableError` only when the pin is HELD and unreadable
    after ``_SHOT_PIN_ROUNDS`` rounds — the caller must not submit.
    """
    local_key = (run_id, media_id)
    offer = (
        _SHOT_PRESCALE_PIN.get(local_key)
        or _observed_capture_raster(width, height, quality_preset)
        or (width, height)
    )
    pin_key = _shot_pin_cache_key(run_id, media_id)
    problem = "no attempt made"
    for _round in range(_SHOT_PIN_ROUNDS):
        try:
            async with asyncio.timeout(_SHARED_RASTER_IO_TIMEOUT_S):
                won = await _shared_raster_cache().set_nx(pin_key, list(offer), ex=_SHOT_PIN_TTL_S)
        except Exception as exc:  # noqa: BLE001 -- fail soft: in-process pin, never a failed compile
            _warn_shared_raster_once("shot-pin claim", exc)
            _log_shot_pin(media_id, "local", offer, width, height)
            return _remember_shot_pin(local_key, offer)
        if won is None:
            _warn_shared_raster_once("shot-pin claim", RuntimeError("cache unavailable"))
            _log_shot_pin(media_id, "local", offer, width, height)
            return _remember_shot_pin(local_key, offer)
        if won:
            _log_shot_pin(media_id, "redis-won", offer, width, height)
            return _remember_shot_pin(local_key, offer)
        # Lost: a dispatch of this shot already exists. Its target is the only
        # one that replays, so it must be READ — never guessed.
        try:
            async with asyncio.timeout(_SHARED_RASTER_IO_TIMEOUT_S):
                held = await _shared_raster_cache().get(pin_key)
        except Exception as exc:  # noqa: BLE001 -- retried below, then refused
            problem = f"read failed ({type(exc).__name__}: {exc})"
            continue
        winner = _valid_shared_raster(held, width, height)
        if winner is not None:
            _log_shot_pin(media_id, "redis-lost", winner, width, height)
            return _remember_shot_pin(local_key, winner)
        # ``None`` here is also what an expiry between the two calls looks like,
        # in which case the next round's claim wins cleanly.
        problem = f"held value unusable ({held!r})"
    raise _ShotPinUnreadableError(
        f"prescale pin for this shot is held by an earlier dispatch but {problem}; "
        "not submitting a recording whose bytes could not match it"
    )


def _shared_raster_cache() -> RedisClient:
    """The orchestrator's ONE Redis client (the instance ``app.state.redis`` is).

    Reused rather than opened: ``get_redis_client`` is the lifespan-connected
    singleton. Before/without ``init_redis`` it is an unconnected wrapper whose
    ``get``/``set`` answer ``None``/``False`` — i.e. the cold path. Kept as a
    function so tests can substitute the cache without touching the singleton.
    """
    return get_redis_client()


def _shared_raster_cache_key(width: int, height: int, quality_preset: str | None) -> str:
    """Redis key for an order's observed raster — derived from :func:`_raster_key`.

    Global, not per-tenant: the raster is a property of the recorder and the
    order (``_resolve_capture_scale(width, height)``), and carries no tenant data.
    """
    key_w, key_h, key_preset = _raster_key(width, height, quality_preset)
    return f"{_SHARED_RASTER_PREFIX}:{key_w}x{key_h}:{key_preset}"


def _warn_shared_raster_once(action: str, exc: BaseException) -> None:
    global _shared_raster_warned
    if _shared_raster_warned:
        return
    _shared_raster_warned = True
    log.warning(
        "ken_burns: shared capture-raster cache %s failed (%s: %s) — falling back to the "
        "process-local observation and in-process shot pin; a pod roll mid-compile may re-record "
        "in-flight clips. "
        "Further failures are not logged.",
        action,
        type(exc).__name__,
        exc,
    )


def _valid_shared_raster(value: object, width: int, height: int) -> tuple[int, int] | None:
    """A cached raster we are willing to prescale to, or ``None``.

    The cache is another process's write, so it gets the same test an
    observation gets (two real ints, not smaller than the order) plus an upper
    bound, since nothing else stands between a corrupt value and a resize.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    raw_w, raw_h = value
    if not isinstance(raw_w, int) or not isinstance(raw_h, int):
        return None
    if isinstance(raw_w, bool) or isinstance(raw_h, bool):
        return None
    if raw_w < width or raw_h < height:
        return None
    if raw_w > _SHARED_RASTER_MAX_AXIS or raw_h > _SHARED_RASTER_MAX_AXIS:
        return None
    return (raw_w, raw_h)


async def _load_shared_capture_raster(width: int, height: int, quality_preset: str | None) -> None:
    """Warm the process dict from the shared cache. Never raises.

    Called ONCE per ``apply_ken_burns`` call, before the per-shot fan-out, and
    only does I/O when this process has no observation for the order — so a warm
    pod pays nothing and no shot pays a round-trip.
    """
    key = _raster_key(width, height, quality_preset)
    if key in _OBSERVED_CAPTURE_RASTER:
        return
    try:
        async with asyncio.timeout(_SHARED_RASTER_IO_TIMEOUT_S):
            cached = await _shared_raster_cache().get(
                _shared_raster_cache_key(width, height, quality_preset)
            )
    except Exception as exc:  # noqa: BLE001 -- fail soft: cold path, never a failed compile
        _warn_shared_raster_once("read", exc)
        return
    raster = _valid_shared_raster(cached, width, height)
    if raster is not None:
        # ``setdefault``: an observation that landed in THIS process meanwhile is
        # fresher than anything a peer wrote.
        _OBSERVED_CAPTURE_RASTER.setdefault(key, raster)


async def _publish_observed_capture_raster(
    width: int, height: int, quality_preset: str | None
) -> None:
    """Offer this process's observation for the order to the shared cache. Never raises."""
    key = _raster_key(width, height, quality_preset)
    raster = _OBSERVED_CAPTURE_RASTER.get(key)
    if raster is None:
        return
    now = time.monotonic()
    last = _SHARED_RASTER_NEXT_PUBLISH.get(key)
    if last is not None and last[0] == raster and now < last[1]:
        return
    # Claim the slot BEFORE awaiting, so the concurrent shots of one fan-out do
    # not all write the same value.
    _SHARED_RASTER_NEXT_PUBLISH[key] = (raster, now + _SHARED_RASTER_RETRY_S)
    try:
        async with asyncio.timeout(_SHARED_RASTER_IO_TIMEOUT_S):
            stored = await _shared_raster_cache().set(
                _shared_raster_cache_key(width, height, quality_preset),
                list(raster),
                ttl=_SHARED_RASTER_TTL_S,
            )
    except Exception as exc:  # noqa: BLE001 -- fail soft: the observation stays process-local
        _warn_shared_raster_once("write", exc)
        return
    if stored:
        _SHARED_RASTER_NEXT_PUBLISH[key] = (raster, now + _SHARED_RASTER_REFRESH_S)


def _prescale_still(data: bytes, content_type: str, width: int, height: int) -> tuple[bytes, str]:
    """Lanczos-upscale a still that is SMALLER than the recording frame.

    Returns ``(bytes, content_type)`` — the original pair unchanged whenever the
    upscale is unnecessary, unaffordable, or unsafe. Every failure mode degrades
    to the input: this is a picture-quality optimisation, and the pre-existing
    behaviour (let Chromium smooth it) is a perfectly serviceable clip, so it
    must never be able to cost a shot its motion.

    The scale factor is the COVER factor (``max(w/sw, h/sh)``), not the contain
    one, so the result is at least 1:1 under BOTH template fits — ``cover``
    displays the image at frame-fill or beyond, and a contain-fit still would be
    left under-resolved by a contain-factor resize.

    Three traps this has to sidestep, all of which are silent:

    * **EXIF orientation.** Chromium APPLIES a JPEG's orientation tag; Pillow does
      not. Re-encoding without honouring it first would rotate the delivered
      picture 90 degrees on any portrait-tagged camera still. ``exif_transpose``
      bakes the rotation into the pixels and drops the tag, so both agree.
    * **Animation.** An animated GIF/WebP still plays in the recorder. Pillow
      would silently take frame 0 and the clip would lose its animation, which is
      a strictly worse picture, not a sharper one — so multi-frame images are
      left alone.
    * **Alpha.** A JPEG re-encode of an RGBA still flattens transparency against
      nothing. Alpha keeps PNG (and usually exceeds the byte budget, which is the
      correct outcome — it degrades to the original).
    """
    try:
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(data)) as src:
            if getattr(src, "n_frames", 1) > 1:
                return data, content_type
            has_alpha = src.mode in {"RGBA", "LA", "PA"} or (
                src.mode == "P" and "transparency" in src.info
            )
            # Transpose BEFORE measuring. Every dimension below has to come from
            # the DISPLAY orientation: a portrait-tagged 200x100 still is shown
            # by the browser as 100x200, so sizing the resize off the stored
            # 200x100 would squash the picture into the wrong aspect — which is a
            # worse defect than the softness this function exists to fix.
            image = ImageOps.exif_transpose(src) or src
            image = image.convert("RGBA" if has_alpha else "RGB")
            src_w, src_h = image.size
            if src_w <= 0 or src_h <= 0:
                return data, content_type
            factor = max(width / src_w, height / src_h)
            # Already at or above the frame: the browser only ever downsizes it,
            # which is the case ``_zoompan_prescale`` proved does NOT benefit.
            if factor <= 1.0:
                return data, content_type
            out_w, out_h = max(1, round(src_w * factor)), max(1, round(src_h * factor))
            # The cost of this resize is its OUTPUT pixels — not its output
            # pixels relative to the target, which anti-correlates with the cost
            # (see the constant above). So the bound does not mention ``width``
            # or ``height``, and is therefore the same at every tier: whatever
            # the old 14.75M ceiling admitted is admitted here too.
            out_px = out_w * out_h
            if out_px > _MAX_UPSCALE_ABS_PIXELS:
                return data, content_type
            # ``Image.Resampling.LANCZOS``, never the bare ``Image.LANCZOS``
            # alias: the alias is an untyped plain ``int`` (1) that the type
            # checker cannot resolve, and a resample filter that silently
            # degrades to bilinear would cost the CPU and deliver none of the
            # measured win. Verified on the pinned Pillow (>=12.1.0, 12.3.0
            # resolved) that the two produce byte-identical output and that both
            # differ from BILINEAR.
            resized = image.resize((out_w, out_h), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        if has_alpha:
            resized.save(buf, format="PNG", optimize=True)
            out_type = "image/png"
        else:
            resized.save(buf, format="JPEG", quality=_UPSCALE_JPEG_QUALITY, optimize=True)
            out_type = "image/jpeg"
        blob = buf.getvalue()
        if len(blob) > _MAX_UPSCALED_BYTES:
            return data, content_type
        return blob, out_type
    except Exception as exc:  # noqa: BLE001 -- quality only; the original still works
        log.warning("ken_burns: still pre-upscale skipped (%s): %r", content_type, exc)
        return data, content_type


def _endpoint_host_keys(endpoint: str | None) -> set[str]:
    """Allow-list keys for a configured storage endpoint.

    Accepts every shape ``StorageSettings`` permits — ``host``, ``host:port``, or
    ``scheme://host[:port]`` — and yields both the bare hostname and the
    ``host:port`` form, the two keys ``w2v_common.net`` matches on. Empty/blank
    (local backend, no external endpoint) yields nothing, so nothing is trusted.
    """
    raw = (endpoint or "").strip()
    if not raw:
        return set()
    parsed = urlsplit(raw if "//" in raw else f"//{raw}")
    host = (parsed.hostname or "").lower()
    if not host:
        return set()
    keys: set[str] = {host}
    if parsed.port is not None:
        keys.add(f"{host}:{parsed.port}")
    return keys


def _artifact_allow_hosts() -> frozenset[str]:
    """Storage hosts whose internal address is legitimate for OUR artifacts.

    Only the endpoints this service is configured to presign against — the
    in-cluster MinIO and, when set, its external rewrite target. Everything else
    (postgres, the registry, k3s, any other Tailnet host) stays blocked, and the
    allow-list is re-applied on every redirect hop by ``safe_get``, so a trusted
    host cannot 302 onward into the estate.
    """
    storage = get_settings().storage
    return frozenset(
        _endpoint_host_keys(storage.minio_endpoint)
        | _endpoint_host_keys(storage.minio_external_endpoint)
    )


def _content_key(url: str) -> str:
    """The stable identity of a presigned artifact URL: its PATH, query stripped.

    A presigned MinIO URL carries a rotating ``X-Amz-Signature``/``X-Amz-Date``
    query, and ``refresh_shot_asset_urls`` re-mints it on every compile — so
    hashing the whole URL made the content-address differ on every run and
    ``find_existing_media`` could never hit: every clip was re-recorded and
    re-paid on every retry/resume. The path (``/bucket/key``) is exactly what
    ``StorageService.resign_presigned_url`` itself keys on, so it survives
    re-signing while still naming a distinct object per distinct image.
    """
    split = urlsplit(url)
    return split.path or url


_ABSENT: Final = object()


class KenBurnsConfig(BaseModel):
    """The author-facing ``ken_burns`` block, normalised to usable values.

    Every knob DEGRADES to its ``_DEFAULT_*`` rather than raising: this block is
    hand-authored (brief / config snapshot / gitops), it is read on a hot compile
    path, and a bad value must cost the run its *tuning*, never the run itself.
    Validation lives here rather than at the call site so the fallbacks cannot
    drift away from the fields they guard.

    ``extra="ignore"``: forward/unknown keys in an authored block must not
    explode — an author adding a knob this build does not know yet still gets
    motion, on defaults.
    """

    model_config = ConfigDict(extra="ignore")

    # Opt-OUT is resolved in ``_resolve_cfg`` (a falsy ``enabled`` yields ``None``
    # rather than a disabled model), so a constructed config is always enabled.
    enabled: bool = True
    preset: str = _DEFAULT_PRESET
    fit: str = _DEFAULT_FIT
    # Run-wide FALLBACK only — a shot's own ``seconds`` wins (``_shot_duration_ms``).
    duration_seconds: float = _DEFAULT_DURATION_S

    @field_validator("enabled", mode="before")
    @classmethod
    def _coerce_enabled(cls, value: object) -> bool:
        """Any truthy authored value means ON — never a validation error."""
        return bool(value)

    @field_validator("preset", mode="before")
    @classmethod
    def _coerce_preset(cls, value: object) -> str:
        """Unknown/absent/non-string preset ⇒ ``_DEFAULT_PRESET``."""
        preset = str(value) if value else _DEFAULT_PRESET
        return preset if preset in _VALID_PRESETS else _DEFAULT_PRESET

    @field_validator("fit", mode="before")
    @classmethod
    def _coerce_fit(cls, value: object) -> str:
        """Unknown/absent/non-string fit ⇒ ``_DEFAULT_FIT``; authored fit lowercased."""
        fit = (str(value) if value else _DEFAULT_FIT).lower()
        return fit if fit in _VALID_FITS else _DEFAULT_FIT

    @field_validator("duration_seconds", mode="before")
    @classmethod
    def _coerce_duration(cls, value: object) -> float:
        """Non-numeric / non-positive / absent duration ⇒ ``_DEFAULT_DURATION_S``.

        A bad value used to raise straight out of the resolver and fail the whole
        compile stage; a mistyped fallback duration now just degrades to the
        default.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            return _DEFAULT_DURATION_S
        try:
            seconds = float(value)
        except ValueError:
            return _DEFAULT_DURATION_S
        # NaN/inf survive ``float()`` and would blow up the ``int(... * 1000)``
        # at the call site, so they degrade here too.
        return seconds if math.isfinite(seconds) and seconds > 0 else _DEFAULT_DURATION_S


def _resolve_cfg(snapshot: Mapping[str, Any]) -> KenBurnsConfig | None:
    """The effective ``ken_burns`` config, or ``None`` when the author opted OUT.

    Absent block ⇒ a default ``KenBurnsConfig`` (ENABLED on every
    ``_DEFAULT_*``), because nothing in
    the product ever authored this block and the old presence-opt-in gate was
    therefore unreachable.

    Two shapes have to be read out of the SNAPSHOT and the BRIEF, not just the
    snapshot, or an explicit opt-out silently inverts into opt-IN:

    * ``_lift_brief_config_keys`` (``runs/run_setup.py``) copies a brief block up
      to the snapshot top level only when it is TRUTHY, so an authored
      ``ken_burns: false`` is dropped by the lift and would read as "absent"
      here — i.e. the user asks for OFF and gets ON. So when the key is missing
      from the snapshot we consult ``config_snapshot["brief"]`` before defaulting.
    * the project brief is ``extra="allow"`` and accepts ``ken_burns`` as a BARE
      SCALAR (not only as a block), so ``ken_burns: false`` must be honoured as
      an opt-out rather than falling through the "not a Mapping" branch.
    """
    raw: object = snapshot.get("ken_burns", _ABSENT)
    if raw is _ABSENT:
        brief = snapshot.get("brief")
        if isinstance(brief, Mapping):
            raw = brief.get("ken_burns", _ABSENT)
    if raw is _ABSENT:
        return KenBurnsConfig()
    if not isinstance(raw, Mapping):
        # Bare scalar: `ken_burns: false` is an opt-out, `ken_burns: true` is a
        # plain opt-in carrying no knobs.
        return None if not raw else KenBurnsConfig()
    if "enabled" in raw and not raw["enabled"]:
        return None
    return KenBurnsConfig.model_validate(dict(raw))


def _shot_duration_ms(shot: Mapping[str, object], default_ms: int) -> int:
    """This shot's clip length in ms — the SHOT's own ``seconds`` when it has one.

    ``seconds`` is the duration field every ``build_shots`` stamps on a compile
    shot (it is the compiler's ``Shot.seconds``). A single config-wide
    ``duration_seconds`` would give a 12s narration shot a 5s pan followed by 7s
    of cloned last frame. Falls back to *default_ms* (config ``duration_seconds``
    → ``_DEFAULT_DURATION_S``) when the shot carries no usable duration, and
    clamps to ``[_MIN_DURATION_S, _MAX_DURATION_S]``.
    """
    raw = shot.get("seconds")
    if not isinstance(raw, (int, float)) or isinstance(raw, bool) or raw <= 0:
        return default_ms
    return int(min(max(float(raw), _MIN_DURATION_S), _MAX_DURATION_S) * 1000)


async def _fetch_own_asset(url: str, client: httpx.AsyncClient) -> tuple[bytes, str]:
    """GET the shot's image URL → ``(bytes, content_type)``, SSRF-guarded + size-capped.

    Routed through ``safe_get`` with the configured storage endpoints as
    ``allow_hosts``: our own presigned MinIO artifact is fetched un-pinned (so the
    SigV4 ``Host`` signature survives byte-for-byte), while any other host is
    resolved, IP-pinned, and rejected if it lands in a private/loopback/link-local/
    **CGNAT** range — on the first hop and on every redirect.

    Raises ``SSRFError`` for a blocked target and ``ResponseTooLargeError`` past
    ``_MAX_IMAGE_BYTES`` (both ``ValueError`` subclasses, and the caller catches
    ``Exception``), so a rejected shot soft-fails to its still exactly as before.

    Sends ``visual_media.MEDIA_FETCH_USER_AGENT``. The name of this function says
    "own asset" and the happy path is a presigned MinIO artifact — but that is an
    EXPECTATION, not a guard: as this module's docstring says, ``image_url`` is a
    value carried on a shot dict, the same dict operator-pinned and discovered
    third-party URLs flow through, which is exactly why it goes through
    ``safe_get`` at all. Nothing anywhere asserts the URL is ours. So a Wikimedia
    still that reached a shot without being re-published 403s httpx's default UA
    here and the shot silently degrades to a frozen frame. Sending the UA removes
    the dependency on an invariant the code does not enforce; it costs nothing on
    the MinIO path, where a presigned SigV4 signature covers ``Host`` and the
    signed-headers list, never ``User-Agent``.
    """
    resp = await safe_get(
        client,
        url,
        allow_hosts=_artifact_allow_hosts(),
        max_fetch_bytes=_MAX_IMAGE_BYTES,
        headers={"User-Agent": MEDIA_FETCH_USER_AGENT},
    )
    resp.raise_for_status()
    content_type = (resp.headers.get("content-type") or "image/jpeg").split(";")[0].strip()
    return resp.content, content_type.lower()


async def _wait_for_recording(
    vp_client: VideoProductionClient, job_id: str, deadline: float
) -> dict[str, object]:
    """Poll ONE recording job until it settles or the pass deadline passes.

    Each poll call is bounded by the client's 600s ceiling; a call that times out
    while the pass still has budget polls the SAME job again — the job is still
    queued or rendering on video_production, and abandoning it there wastes the
    render and keeps the shot frozen. Only the pass deadline turns a slow job
    into a soft-failed still. Any non-timeout error propagates unchanged.
    """
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"video_production job {job_id} still unfinished at the ken_burns pass deadline"
            )
        try:
            return await vp_client.wait_for_content_recording(
                job_id, timeout=min(_RECORD_TIMEOUT_S, remaining)
            )
        except TimeoutError:
            if deadline - time.monotonic() <= 0:
                raise
            log.info("ken_burns: job %s still running at a poll deadline; waiting on", job_id)


class _SaturatedDeadlineError(TimeoutError):
    """The pass deadline expired on a RE-SUBMITTED job: its shot was saturated first."""


def _is_queue_saturated(exc: BaseException) -> bool:
    """True when *exc* is video_production's "no recording slot in time" failure.

    Exactly the shape ``poll_job`` produces and nothing looser: the text after
    ``" failed: "`` must START with ``QUEUE_SATURATED:`` (see
    :data:`_VP_QUEUE_SATURATED_MARKER` for why nothing more structured exists).
    An unrelated error that merely quotes the word is NOT saturation. The one
    non-text case is this module's own :class:`_SaturatedDeadlineError`.
    """
    if isinstance(exc, _SaturatedDeadlineError):
        return True
    _, sep, error = str(exc).partition(_POLL_JOB_FAILED_SEP)
    return bool(sep) and error.startswith(f"{_VP_QUEUE_SATURATED_MARKER}:")


def _saturation_backoff_seconds(attempt: int) -> float:
    """Delay before saturation re-submission ``attempt`` (1-based), with jitter."""
    base = _SATURATION_BACKOFF_BASE_S
    # Jitter only de-synchronises a retry wave; it is not a security decision.
    return base * 2 ** (attempt - 1) + random.uniform(0.0, base)  # noqa: S311


async def _saturation_sleep(seconds: float) -> None:
    """The backoff sleep, as a module seam so tests never really sleep."""
    await asyncio.sleep(seconds)


async def _record_with_saturation_retry(
    submit: Callable[[str], Awaitable[str]],
    vp_client: VideoProductionClient,
    *,
    base_key: str,
    deadline: float,
    shot_id: object,
) -> tuple[str, dict[str, object]]:
    """Submit ONE shot's recording and wait for it, re-submitting on saturation.

    ``submit(idempotency_key)`` starts the recording and returns its job id; the
    caller builds it ONCE so the request body (the prescaled still, the pinned
    target, ``part_id``) is identical on every attempt — only the key differs.

    Attempt 0 uses ``base_key`` unchanged (byte-identical to the pre-retry key,
    so an already-recorded clip still dedupes). video_production answers a
    repeated key with the ORIGINAL job — a failed one included — so re-submitting
    under the same key would replay the saturation forever; attempt ``n`` uses
    ``f"{base_key}:sat{n}"`` to get a NEW job. Numbering restarts at 0 on every
    entry, so a successor pod replays the same keys in the same order: the failed
    ``base_key`` / ``:sat1`` jobs answer "failed" at once and it lands on the
    first key that is still running or succeeded, never on a fresh paid job for
    work a predecessor already bought.

    A retry is started ONLY while the pass deadline can still hold it: after the
    backoff there must be at least ``_SATURATION_RETRY_MIN_BUDGET_S`` left —
    video_production's own queue wait (a re-submitted job may sit queued that
    long) plus this module's floor for recording a clip at all. With less, the
    job would be submitted, the pass would time out on it, and the recording
    would be paid for nobody. Otherwise the saturation error
    propagates and the shot keeps its still. Any NON-saturation failure
    propagates on the spot, exactly as before. A pass-deadline timeout on a
    RE-SUBMITTED job is raised as :class:`_SaturatedDeadlineError`, so it is
    still attributed to the saturated queue rather than to a slow render.
    """
    saturated = 0
    for attempt in range(_SATURATION_RETRIES + 1):
        key = base_key if attempt == 0 else f"{base_key}:sat{attempt}"
        job_id: str | None = None
        try:
            job_id = await submit(key)
            return job_id, await _wait_for_recording(vp_client, job_id, deadline)
        except TimeoutError as exc:
            # A plain timeout is NOT re-submitted, on purpose. Unlike a
            # QUEUE_SATURATED job (terminal: it failed, it will never deliver),
            # a timed-out job is still QUEUED or RENDERING on video_production —
            # it will most likely complete. A new key would start a SECOND paid
            # recording of the same clip while the first one finishes for
            # nobody; the same key replays the same job, which is exactly what
            # ``_wait_for_recording`` already polls. And there is no budget to
            # retry IN: ``_wait_for_recording`` only gives up once the pass
            # deadline itself has passed. Logged so the gap is visible.
            log.info(
                "ken_burns: shot %s job %s (key %s) unfinished at the pass deadline; "
                "NOT re-submitting — the job is still in flight on video_production and "
                "a new key would pay for a second recording of the same clip",
                shot_id,
                job_id,
                key,
            )
            if not saturated:
                # THIS shot never saw a saturation, so it is coded a plain
                # soft-fail — even when it was starved by siblings that did.
                raise
            raise _SaturatedDeadlineError(
                f"{exc} (after {saturated} {_VP_QUEUE_SATURATED_MARKER} attempt(s))"
            ) from exc
        except Exception as exc:
            if not _is_queue_saturated(exc):
                raise
            saturated += 1
            if attempt == _SATURATION_RETRIES:
                raise
            delay = _saturation_backoff_seconds(attempt + 1)
            remaining = deadline - time.monotonic()
            if remaining - delay < _SATURATION_RETRY_MIN_BUDGET_S:
                log.info(
                    "ken_burns: shot %s saturated on attempt %d; NOT retrying — %.0fs of the "
                    "pass budget left, a retry needs %.0fs backoff + %.0fs",
                    shot_id,
                    attempt,
                    max(remaining, 0.0),
                    delay,
                    _SATURATION_RETRY_MIN_BUDGET_S,
                )
                raise
            log.info(
                "ken_burns: shot %s recording hit %s; retry %d/%d in %.1fs",
                shot_id,
                _VP_QUEUE_SATURATED_MARKER,
                attempt + 1,
                _SATURATION_RETRIES,
                delay,
            )
            await _saturation_sleep(delay)
    raise AssertionError("unreachable: the last saturation attempt re-raises")  # pragma: no cover


def _log_pass_start(budget: float, offered: float | None, stills: int, *, cache_only: bool) -> None:
    """The budget this pass RECEIVED — the first thing to read when shots stay stills."""
    log.info(
        "ken_burns: pass start — budget %.0fs (offered %s), %d still(s), concurrency %d%s",
        budget,
        "default" if offered is None else f"{offered:.0f}s",
        stills,
        _CONCURRENCY,
        " — CACHE ONLY, below the record floor" if cache_only else "",
    )


def _log_shot_outcome(shot_id: object, outcome: str, shot_t0: float, pass_t0: float) -> None:
    """One line per shot: how long IT waited, when it started, and how it ended.

    ``started_at`` (seconds into the pass) separates a slow job from a starved
    one: a shot of the last recorder wave starts late and gets the smallest
    share of the pass deadline.
    """
    now = time.monotonic()
    log.info(
        "ken_burns: shot %s outcome=%s waited=%.1fs started_at=%.1fs",
        shot_id,
        outcome,
        now - shot_t0,
        shot_t0 - pass_t0,
    )


def _pass_budget(budget_seconds: float | None) -> tuple[float, bool]:
    """``(wall budget, cache_only)`` for the pass.

    ``cache_only`` when less than ``_MIN_PHASE_BUDGET_S`` is left: no clip can be
    RECORDED in that time, but clips a previous attempt already stored (a resumed
    compile after a pod roll) cost one lookup each and are still used.
    """
    budget = _PHASE_BUDGET_S if budget_seconds is None else min(budget_seconds, _PHASE_BUDGET_S)
    if budget >= _MIN_PHASE_BUDGET_S:
        return budget, False
    return 0.0, True


def _note(
    notes: list[str] | None,
    reason: str,
    *,
    degradations: list[Degradation] | None = None,
    code: str | None = None,
) -> None:
    """Record one pass-level sentence: log + prose note + (given a code) its coded twin."""
    log.warning("%s", reason)
    if notes is not None:
        notes.append(reason)
    if degradations is not None and code is not None:
        degradations.append(degradation(code, reason))


def _record_soft_fail(
    shot: Mapping[str, object],
    exc: Exception,
    notes: list[str] | None,
    degradations: list[Degradation] | None,
) -> None:
    """Record ONE shot that keeps its still: a log line, a prose note, a code."""
    # The exception CLASS is load-bearing: it separates an infra timeout (record
    # poll / fetch) from a 403 on an expired presign from a render error, and
    # the run detail is the only place anyone reads it after the pod rolls.
    reason = (
        f"ken_burns: shot {shot.get('id')} soft-failed, kept as a still: "
        f"{type(exc).__name__}: {exc}"
    )
    log.warning("%s", reason)
    if notes is not None:
        notes.append(reason)
    if degradations is not None:
        # Same sentence, now with an identity: a still lost to a saturated
        # recording queue is a CAPACITY fact about video_production, every other
        # soft-fail is about this shot.
        code = (
            KEN_BURNS_STILL_QUEUE_SATURATED
            if _is_queue_saturated(exc)
            else KEN_BURNS_STILL_SOFT_FAILED
        )
        degradations.append(degradation(code, reason))


def _report_stills(
    notes: list[str] | None,
    *,
    candidates: int,
    upgraded: int,
    cache_only: bool,
    degradations: list[Degradation] | None = None,
) -> None:
    """Record the shots that stayed stills — the budget skip, or ALL soft-failed."""
    if cache_only and upgraded < candidates:
        _note(
            notes,
            f"ken_burns: skipped, {candidates - upgraded} uncached still shot(s) kept as "
            f"stills — less than {_MIN_PHASE_BUDGET_S:.0f}s of the compile's pre-submit "
            "budget was left (title cards ran long)",
            degradations=degradations,
            code=KEN_BURNS_SKIPPED_NO_BUDGET,
        )
    elif not upgraded:
        # EVERY still stayed a still. Strictly worse than one dropped clip: the
        # delivered film is a slideshow of frozen frames, and it ships green.
        _note(
            notes,
            f"ken_burns: ALL {candidates} still shot(s) soft-failed — no motion was "
            "applied and the video is a sequence of frozen stills (~92% duplicate frames)",
            degradations=degradations,
            code=KEN_BURNS_ALL_STILLS_FROZEN,
        )


def _load_generation(value: JsonValue) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"stored recording generation is not a token: {value!r}")
    return value


async def _recording_generation(ledger: StepLedger) -> str:
    """This run's ken_burns RECORDING generation — folded into every recorder key.

    Why it exists: video_production's idempotency cache (w2v_common) answers a
    replayed SUBMIT with the ORIGINAL job id for 24h. After a user restamp the
    downstream compile re-runs under the SAME ``run_id``, so a shot whose first
    recording FAILED was re-submitted under the same key and got that failed job
    back — the shot stayed a still with no way to re-record it inside the run.

    Source: a token minted once and stored in the CALLING stage's step ledger
    (row ``kb_generation``). It is the one thing that changes exactly when it
    must:
      * pod death / drain / resumable timeout → same row → same token → the
        re-submit replays the in-flight job (no second bill);
      * restamp → ``_reset_downstream_stages`` deletes the stage's ledger rows in
        the restamp transaction → next entry mints a new token → new keys;
      * failed stage → regenerate is a CHILD run → new ``run_id`` anyway.
    The stage row's ``attempts`` / ``interruptions`` counters do NOT move on a
    restamp (see ``step_ledger`` "Key scope"), so neither can serve here.

    ``""`` (the pre-generation key, byte-identical) when there is no durable
    ledger behind the context: a token that cannot be stored would change on
    every call and break replay instead of enabling it.
    """
    if ledger.run_id is None:
        return ""
    return await ledger.once(
        "kb_generation",
        hash_inputs("kb_generation", 1),
        _mint_generation,
        load=_load_generation,
    )


async def _mint_generation() -> str:
    return uuid7().hex[-12:]


def _recording_base_key(run_id: str, media_id: str, generation: str) -> str:
    """The recorder idempotency key of one shot; ``:satN`` retries derive from it."""
    base = f"{run_id}:ken_burns:{media_id}"
    return f"{base}:g{generation}" if generation else base


async def apply_ken_burns(
    ctx: RunContext,
    shots: list[dict[str, Any]],
    *,
    client: VideoProductionClient | None = None,
    notes: list[str] | None = None,
    master_fps: int | None = None,
    budget_seconds: float | None = None,
    cache_hits: list[int] | None = None,
    degradations: list[Degradation] | None = None,
) -> list[dict[str, Any]]:
    """Upgrade still-image shots to pan/zoom clips when ``ken_burns`` is enabled.

    ``budget_seconds`` is the wall time the caller can give the whole pass
    (``None`` = ``_PHASE_BUDGET_S``; it is capped at that either way). With less
    than ``_MIN_PHASE_BUDGET_S`` left, no clip is attempted: every still stays a
    still and ``notes`` says the pre-submit budget ran out — a DEGRADED film,
    never a hard-failed render.

    Returns a (possibly new) shot list with ``video_url`` set on each upgraded
    shot — the still ``image_url`` stays as the degrade path (the compositor
    honours ``video_url`` > ``image_url``). An absent or empty ``ken_burns``
    block means ENABLED — motion is the default for stills; only an explicit
    ``enabled: false`` returns the shot list unchanged. Also unchanged when no
    shot is a bare still, or when every clip soft-fails.

    ``master_fps`` is the run's resolved master cadence, when the caller already
    knows it (``CompileStageBase.resolve_cadence``). ``None`` — every caller
    today — resolves the same value from the run config, so behaviour is
    unchanged for every product except ``action_reel``, whose rate is
    footage-derived and therefore only reachable through this argument. See
    :func:`_master_fps`.

    ``cache_hits`` is an OPTIONAL sink: the shot index of every clip served from
    the content-addressed store (``find_existing_media``) instead of being
    re-recorded is appended — the evidence that a resumed compile reused, rather
    than re-paid for, the clips a rolled pod already stored.

    ``notes`` is the OPTIONAL degrade channel (the same ``list[str]`` sink
    ``shared/title_cards`` uses): a caller that owns a stage output with a
    ``notes`` field passes its list in, a soft-failed shot appends ONE sentence
    naming the shot and the exception, and the ALL-degraded case — every still
    stayed a still — appends its own, louder line. That case is the one that
    matters: with video_production down, every clip soft-fails, the compile
    succeeds, and the delivered film is the ~92%-duplicate-frame slideshow this
    function exists to prevent — previously with ZERO durable record, since a
    ``log.warning`` evaporates with the pod. Behaviour is unchanged: shots are
    still soft-failed to their still, and ``None`` (no caller sink) is a no-op.

    ``degradations`` is the STRUCTURED twin of ``notes`` (``shared/degradation``):
    every sentence appended to ``notes`` here is also appended as a coded entry
    whose ``detail`` is that same sentence, so the stage output carries a code a
    query can bind to instead of the DAG executor's ``unclassified_note``. A shot
    lost to a saturated recording queue (after its retries — see
    ``_record_with_saturation_retry``) is told apart from every other soft-fail.
    """
    # Motion is the DEFAULT for stills: an absent/empty ``ken_burns`` block means
    # ENABLED. Only an EXPLICIT falsy ``enabled`` opts out. The block was never
    # resolved by the planner and never set in gitops, so the old
    # presence-opt-in gate was unreachable and every still-based film rendered
    # as frozen single frames (~92% duplicate frames).
    cfg = _resolve_cfg(ctx.config_snapshot)
    if cfg is None:
        return shots

    # Already normalised + fallen back by ``KenBurnsConfig`` — no re-validation here.
    preset = cfg.preset
    fit = cfg.fit
    # Run-wide FALLBACK only: the per-shot duration below wins whenever the shot
    # carries its own ``seconds`` (see ``_shot_duration_ms``).
    default_duration_ms = int(cfg.duration_seconds * 1000)
    deliverable = ctx.config_snapshot.get("deliverable") or {}
    width, height = format_dims(deliverable.get("format"), quality_tier(ctx.config_snapshot))
    # C23 design/motion fingerprint for this run — folded into each ken_burns clip
    # media id ONLY when present. ``None`` (no design/motion block) → conditional
    # omission keeps the media id byte-identical (this is the ONLY path a design
    # change reaches a ken_burns clip, since its params don't carry tokens).
    token_fp = token_fp_for_snapshot(ctx.config_snapshot)
    # Encode preset for the run's resolution tier — the SAME resolver site_capture
    # uses, so a ken_burns clip is recorded at the tier the film is compiled at
    # instead of VP's ``standard`` (CRF 23) default, which the compiler then
    # re-encodes (generational loss). Empty cfg: ken_burns has no per-brief
    # ``recording_tier`` pin, so it always follows the run tier.
    quality_preset = _resolve_recording_quality_preset(ctx, {})
    # Record AT the master cadence so the compiler's per-clip ``fps=`` conform is
    # a 1:1 no-op instead of an uneven decimation of a continuous pan.
    fps = _master_fps(ctx.config_snapshot, master_fps)

    # A still-only shot: a fetchable http(s) image and no clip already on it
    # (footage / curated / AI video / motion-graphics / animated slide all win).
    candidates = [
        (index, shot)
        for index, shot in enumerate(shots)
        if isinstance(shot.get("image_url"), str)
        and shot["image_url"].startswith(("http://", "https://"))
        and not shot.get("video_url")
    ]
    if not candidates:
        return shots
    # Too little of the caller's budget left to RECORD → cache lookups only.
    budget, cache_only = _pass_budget(budget_seconds)

    vp_client = client or VideoProductionClient()
    run_id = str(ctx.run_id)
    # A cold pod adopts the raster a peer already observed, so shots FIRST
    # dispatched here start sharp. Once, here — not per shot. Soft: a
    # miss/outage is the ordered-dims path. This only feeds a shot's OFFER (see
    # ``_pin_shot_prescale_target``); it never decides the target of a shot that
    # has already been dispatched, by this pod or a predecessor.
    await _load_shared_capture_raster(width, height, quality_preset)
    # Once per pass, BEFORE the fan-out: every shot (and every stage_gather retry
    # of one) must address the recorder under the same generation.
    ledger = _ken_burns_ledger(ctx)
    generation = await _recording_generation(ledger)
    pass_t0 = time.monotonic()
    deadline = pass_t0 + budget
    _log_pass_start(budget, budget_seconds, len(candidates), cache_only=cache_only)

    # No ``follow_redirects``: ``safe_get`` follows them MANUALLY so every hop is
    # re-validated. httpx-level redirect following would skip the guard entirely.
    async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_S) as http_client:

        async def _gen(item: tuple[int, dict[str, Any]]) -> tuple[int, str | None]:
            index, shot = item
            shot_t0 = time.monotonic()
            try:
                # This SHOT's length, not one run-wide constant — a clip shorter
                # than its shot gets freeze-padded back into a frozen still.
                duration_ms = _shot_duration_ms(shot, default_duration_ms)
                # Content-address by the source image + motion params so a
                # resumed/retried render reuses an already-recorded clip and skips
                # both the asset re-fetch and the video_production_service record.
                # The image is keyed by its URL PATH: the presigned query rotates on
                # every re-sign, so hashing the full URL made this cache 0%-hit.
                # O12b: the prescale target is resolved BELOW, after the cache
                # lookup, by ``_pin_shot_prescale_target`` — never read straight
                # off the observed raster here.
                media_id = deterministic_media_id(
                    "ken_burns",
                    _content_key(shot["image_url"]),
                    preset,
                    fit,
                    duration_ms,
                    # ``width``/``height`` ARE folded (verified), and the ORDER is
                    # ALL that is folded — the observed raster deliberately is NOT.
                    #
                    # ⛔ THE PRESCALE TARGET MUST NOT REACH THIS KEY. It is the one
                    # input here that is not a property of the ORDER: it depends on
                    # whether a recording at this order had completed when this
                    # shot was FIRST dispatched, so the same shot would address
                    # differently depending on pod warmth. (It is constant per
                    # (run, shot) — the pin below — which is all the idempotency
                    # key needs; this id is also a CROSS-RUN address.) Folding it
                    # — even conditionally — makes every warm process re-address
                    # every clip the estate recorded under O12, and
                    # ``find_existing_media`` has a tenant-scoped CROSS-RUN lane
                    # (``_cross_run_tenant``), so that is not one resumed run's
                    # bill, it is the tenant's whole cached ken_burns estate
                    # re-recorded to re-sharpen work already delivered. That is
                    # the exact cost the O12 comment this replaced weighed and
                    # declined, and O12b does not get to reverse that policy in
                    # silence.
                    #
                    # What it costs to leave it out, stated plainly: the shots
                    # first dispatched before the first observation lands (at
                    # most ``_CONCURRENCY`` of them per cold start) are prescaled
                    # to the ordered dims and share a media id with the
                    # sharp render
                    # of the same order, so a later lookup can serve the soft one
                    # back. That is the SAME trade O12 already made explicitly for
                    # a resume across its own deploy, and it is bounded: a soft
                    # clip is a valid clip at the ordered dims, only interpolated
                    # worse. ``deterministic_media_id`` content-addresses the
                    # ORDER, never the bytes, and every other conditional fold
                    # here (``fps``) holds the same line.
                    #
                    # ⚠️ Consequence for MEASURING the O12b win: a before/after
                    # comparison must run against a tenant/run whose cache is
                    # cold for these ids, or it measures the cache.
                    width,
                    height,
                    # The encode preset determines the bytes: without it a resume
                    # would reuse a clip recorded at the old CRF-23 default.
                    quality_preset,
                    *((token_fp,) if token_fp else ()),
                    # The recording cadence determines the frame COUNT and so the
                    # bytes, but it is folded ONLY when it differs from the old
                    # hardcoded 30: at 30 the clip is byte-for-byte what the
                    # previous code produced, so reusing that media id is CORRECT,
                    # and an unconditional fold would invalidate every cached clip
                    # in the estate and re-pay for a re-record that changes nothing.
                    *((f"fps{fps}",) if fps != _DEFAULT_FPS else ()),
                )
                existing = await find_existing_media(run_id, "ken_burns", media_id, "mp4")
                if existing is not None:
                    if cache_hits is not None:
                        cache_hits.append(index)
                    _log_shot_outcome(shot.get("id"), "cache_hit", shot_t0, pass_t0)
                    return (index, existing)
                if cache_only:
                    _log_shot_outcome(shot.get("id"), "skipped_no_budget", shot_t0, pass_t0)
                    return (index, None)  # no time to record; reported once below
                data, content_type = await _fetch_own_asset(shot["image_url"], http_client)
                # O12/O12b: hand the recorder a still that is already at the
                # resolution it will RASTERISE at, so Chromium's default
                # smoothing never performs the biggest upscale in this path.
                # ``to_thread`` because Pillow's
                # resize is a few hundred ms of CPU per shot and this coroutine
                # otherwise owns the event loop for the whole compile stage.
                # ⛔ Resolved through the per-shot pin on EVERY attempt, before
                # the paid call: the recorder replays on (idempotency key + BODY
                # hash) and the body carries this still, so the target a shot
                # was first dispatched with is the only one a retry or a
                # successor pod may use. Raises (→ soft-fail, no bill) only when
                # an earlier dispatch's pin exists and cannot be read.
                prescale_w, prescale_h = await _pin_shot_prescale_target(
                    run_id, media_id, width, height, quality_preset, ledger=ledger
                )
                data, content_type = await asyncio.to_thread(
                    _prescale_still, data, content_type, prescale_w, prescale_h
                )
                # Built ONCE, outside the saturation-retry loop: ``part_id`` falls
                # back to a random id, and the recorder replays on (key + BODY
                # hash), so every attempt must send the same body.
                part_id = f"kb-{(shot.get('id') or uuid7().hex)[:16]}"
                animation_params: dict[str, object] = {
                    "image": _data_uri(data, content_type),
                    "preset": preset,
                    "fit": fit,
                    "duration_ms": duration_ms,
                }

                async def _submit(idempotency_key: str) -> str:
                    return await vp_client.record_animation(
                        template_id=_TEMPLATE,
                        params=animation_params,
                        part_id=part_id,
                        fps=fps,
                        width=width,
                        height=height,
                        duration_ms=duration_ms,
                        output_format="mp4",
                        quality_preset=quality_preset,
                        idempotency_key=idempotency_key,
                    )

                # Idempotency: stage_gather retries the WHOLE callable on a
                # transient wait failure (per_item_retries=1). Without a dedupe
                # key the re-submit starts a SECOND paid recording for the same
                # clip (double-bill). The per-item key (run + stage +
                # content-addressed media_id) makes the service return the
                # ORIGINAL job, so only the wait is actually retried. Same
                # construction as motion_graphics_gen / ai_video_gen. A
                # QUEUE_SATURATED failure is the one case that key must NOT
                # replay — see ``_record_with_saturation_retry``.
                job_id, status = await _record_with_saturation_retry(
                    _submit,
                    vp_client,
                    # ``generation`` changes on a restamp (see
                    # ``_recording_generation``) so a re-run does not get a
                    # FAILED original job back from the 24h submit cache.
                    base_key=_recording_base_key(run_id, media_id, generation),
                    deadline=deadline,
                    shot_id=shot.get("id"),
                )
                # The settled job payload carries the recorder's OWN account of
                # the raster it captured at (video_production_service #182). It
                # was already being returned here and thrown away; reading it is
                # the whole unblock for O12b, and costs no extra request.
                _observe_capture_raster(width, height, quality_preset, status)
                await _publish_observed_capture_raster(width, height, quality_preset)
                video_bytes, _ctype, _disp = await vp_client.get_recording_video(job_id)
                url = await store_generated_video(
                    run_id=run_id,
                    stage_name="ken_burns",
                    video_id=media_id,
                    video_bytes=video_bytes,
                    video_format="mp4",
                )
                _log_shot_outcome(shot.get("id"), "recorded", shot_t0, pass_t0)
                return (index, url)
            except Exception as exc:  # noqa: BLE001 -- soft-fail: this shot keeps its still
                _log_shot_outcome(
                    shot.get("id"), f"soft_failed:{type(exc).__name__}", shot_t0, pass_t0
                )
                _record_soft_fail(shot, exc, notes, degradations)
                return (index, None)

        results = await stage_gather(
            candidates,
            _gen,
            concurrency=_CONCURRENCY,
            per_item_timeout_seconds=budget + _PER_ITEM_OVERHEAD_S,
            per_item_retries=1,
            stage_name="ken_burns",
        )

    url_by_index = {index: url for (index, url) in results if url}
    _report_stills(
        notes,
        candidates=len(candidates),
        upgraded=len(url_by_index),
        cache_only=cache_only,
        degradations=degradations,
    )
    if not url_by_index:
        return shots
    return [
        {**shot, "video_url": url_by_index[index]} if index in url_by_index else shot
        for index, shot in enumerate(shots)
    ]
