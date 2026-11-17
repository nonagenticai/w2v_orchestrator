"""Local, deterministic Ken Burns — the fallback when video_production cannot animate a still.

Why this exists (run 01a10fcf, 2026-10-06): one of a promo's four stills failed its
recording with ``QUEUE_SATURATED: no per-pod recording slot within 300s`` and shipped
as a frozen frame; with a near-static neighbour that left ~12s in the middle of the
film with no motion at all. A frozen still is the defect Ken Burns exists to remove,
so when the recorder cannot deliver a clip the orchestrator renders a plain centred
push-in itself with ffmpeg (present in the orchestrator image) instead of giving up.

Two pure-ish helpers, both bounded by a timeout and both soft (they raise; the caller
decides what a failure costs):

* :func:`render_local_ken_burns` — still bytes -> mp4 bytes: a linear zoom about the
  frame CENTRE over the clip (a per-frame ``perspective`` source window: sub-pixel, so
  a slow push does not judder the way an integer-stepping ``zoompan`` does). A
  ``cover`` still fills the frame and is pushed in; a ``contain`` still (a designed
  slide whose aspect is NOT the frame's) starts inset and ends filling the frame, so
  none of it is ever cropped; a ``full`` still (a designed slide that already has
  the frame's aspect) fills the frame from the first frame to the last with no
  geometric move at all — only a slow brightness breath.
  ``ken_burns.preset`` steers the ``cover``/``contain`` move: ``zoom_in`` (the
  default, above), ``zoom_out`` (the same zoom reversed) and ``pan_{left,right,
  up,down}`` (a held zoom whose window drifts back to centre) — the recorder
  template's semantics, never-crop rule included.
* :func:`clip_motion_psnr` — PSNR between a clip's first and last frame on a small
  grey thumbnail. High PSNR = the two ends look alike = little visible motion. Run
  01a10fcf's line 4 clip measured 38 dB: a pan nobody could see.
"""

from __future__ import annotations

import asyncio
import io
import logging
import math
import shutil
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from PIL import Image

from src.services.stages.shared import ken_burns as kb
from src.services.stages.shared.degradation import (
    KEN_BURNS_LOCAL_FALLBACK,
    KEN_BURNS_LOCAL_FALLBACK_FAILED,
    KEN_BURNS_NEAR_STATIC,
    KEN_BURNS_PRESET_UNKNOWN,
    STILL_COVER_CROP_FAILED,
    Degradation,
    advisory,
    degradation,
)
from src.services.stages.shared.motion_slides import (
    format_dims,
    motion_per_line_requested,
    quality_tier,
)
from src.services.stages.shared.step_ledger import JsonValue, hash_inputs
from src.services.stages.shot_assembly import ShotDict
from src.services.still_fit import bound_to_frame, encode_still_under
from src.services.tts_storage import (
    deterministic_media_id,
    store_generated_image,
    store_generated_video,
)

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

log = logging.getLogger(__name__)

# Wall ceiling of one local render (4K, ~8s clip: ~30s measured on a laptop; the
# orchestrator pod has 3 CPUs). Bounded further by the caller's budget.
_LOCAL_RENDER_TIMEOUT_S = 240.0
_LOCAL_PROBE_TIMEOUT_S = 60.0
_FETCH_TIMEOUT_S = 20.0

#: Zoom travel of the local push-in (1.0 -> 1.0 + this). Matches the recorder
#: template's cover-path travel (``scale(1 -> 1.14)``) closely enough to cut in
#: beside its clips without a visible change of style.
LOCAL_ZOOM_TRAVEL = 0.12
#: Travel of the never-crop ``contain`` move: the still starts inset by this much
#: (1/1.06 = 94% of the frame) and ends filling it.
LOCAL_CONTAIN_TRAVEL = 0.06
#: Peak of the ``full`` move's brightness breath (ffmpeg ``eq`` units, -1..1): the
#: slide brightens by this much at mid-clip and is back where it started at the end.
LOCAL_FULL_BREATH = 0.03
#: Long edge the local clip is rendered at, at most (see :func:`local_render_dims`).
LOCAL_MAX_LONG_EDGE = 1920
#: First-vs-last-frame PSNR at or above which a Ken Burns clip counts as
#: near-static. A visible 12-14% push on a detailed still measures well under
#: 30 dB at the thumbnail size below; the near-static clip of run 01a10fcf
#: measured 38 dB.
NEAR_STATIC_PSNR_DB = 35.0
_THUMB_W, _THUMB_H = 160, 90


class LocalKenBurnsError(RuntimeError):
    """The local render or measurement could not complete."""


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise LocalKenBurnsError("ffmpeg is not installed in this image")
    return path


async def _run(cmd: list[str], timeout_s: float) -> bytes:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except TimeoutError as exc:
        raise LocalKenBurnsError(f"ffmpeg exceeded {timeout_s:.0f}s") from exc
    finally:
        # Timeout AND cancellation (the stage's own deadline, a pod shutdown): the
        # child must never outlive the coroutine that owns it.
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
    if proc.returncode != 0:
        tail = err.decode(errors="replace").strip()[-300:]
        raise LocalKenBurnsError(f"ffmpeg exited {proc.returncode}: {tail}")
    return out


def _even(n: float) -> int:
    return max(2, round(n / 2.0) * 2)


def local_render_dims(width: int, height: int) -> tuple[int, int]:
    """The raster the local clip is rendered at: the frame, long edge capped.

    One 4K render peaked at 2.67 GB RSS against the pod's 4Gi limit. The fallback
    is rendered at no more than :data:`LOCAL_MAX_LONG_EDGE` and the compiler
    scales it into the film like any other clip — a slightly softer fallback
    shot instead of an OOM-killed orchestrator.
    """
    long_edge = max(width, height)
    if long_edge <= LOCAL_MAX_LONG_EDGE:
        return _even(width), _even(height)
    ratio = LOCAL_MAX_LONG_EDGE / long_edge
    return _even(width * ratio), _even(height * ratio)


def _centred_zoom(
    width: int, height: int, frames: int, travel: float, *, reverse: bool = False
) -> str:
    """A zoom from 1.0 to ``1 + travel`` about the frame CENTRE, per frame.

    ``perspective`` with ``sense=source`` maps a source window onto the whole
    output; the window is the frame shrunk about its centre by the frame's zoom,
    so the centre pixel never moves and all four edges are cut equally. (The
    first version cropped a per-frame ``scale``: ``crop`` clamps x/y against the
    link's config-time width, so the "centred" crop sat at 0,0 and the picture
    slid toward the top-left.) ``on`` is the output frame number in ffmpeg 4.4
    and 7.1 alike.

    ``reverse`` runs the same zoom backwards (``1 + travel`` -> 1.0): the
    ``zoom_out`` preset. The forward string is byte-identical to before presets.
    """
    span = max(frames - 1, 1)
    progress = f"(1-on/{span})" if reverse else f"on/{span}"
    inset = f"(1-1/(1+{travel}*{progress}))/2"  # share of each side outside the window
    left, top = f"W*{inset}", f"H*{inset}"
    right, bottom = f"W*(1-{inset})", f"H*(1-{inset})"
    return (
        f"perspective=x0='{left}':y0='{top}':x1='{right}':y1='{top}'"
        f":x2='{left}':y2='{bottom}':x3='{right}':y3='{bottom}'"
        ":sense=source:eval=frame:interpolation=cubic"
    )


#: The presets the local move renders — the recorder template's own set
#: (``video_production`` ``ken_burns.html``), so ``ken_burns.preset`` means the
#: same motion on either path. ``zoom_in`` is the default.
LOCAL_PRESETS = kb._VALID_PRESETS
DEFAULT_PRESET = kb._DEFAULT_PRESET
_PANS = {
    # preset: (axis, sign of the source window's starting offset from centre).
    # pan_left: the window starts LEFT of centre, so the picture starts shifted
    # right and drifts left to centre — the recorder's "start right -> centre".
    "pan_left": ("x", -1),
    "pan_right": ("x", 1),
    "pan_up": ("y", -1),
    "pan_down": ("y", 1),
}


def _num(value: float) -> str:
    return f"{value:.6f}"


def _panned_window(frames: int, zoom: float, slack: float, preset: str) -> str:
    """A constant ``zoom`` source window whose centre drifts back to the frame centre.

    The window is ``1/zoom`` of the source on each axis. Along the pan axis its
    centre starts ``slack`` (a share of the source) off centre and travels
    linearly to the centre by the last frame — the recorder template's pans, which
    also end centred. ``slack`` is chosen by the caller so the window never leaves
    what it may show (the still for ``cover``, the inset slide for ``contain``).
    """
    axis, sign = _PANS[preset]
    span = max(frames - 1, 1)
    half = 0.5 / zoom
    lo, hi = _num(0.5 - half), _num(0.5 + half)
    drift = f"{_num(slack)}*(1-on/{span})"
    op = "+" if sign > 0 else "-"
    moving_lo, moving_hi = f"({lo}{op}{drift})", f"({hi}{op}{drift})"
    if axis == "x":
        left, right = f"W*{moving_lo}", f"W*{moving_hi}"
        top, bottom = f"H*{lo}", f"H*{hi}"
    else:
        left, right = f"W*{lo}", f"W*{hi}"
        top, bottom = f"H*{moving_lo}", f"H*{moving_hi}"
    return (
        f"perspective=x0='{left}':y0='{top}':x1='{right}':y1='{top}'"
        f":x2='{left}':y2='{bottom}':x3='{right}':y3='{bottom}'"
        ":sense=source:eval=frame:interpolation=cubic"
    )


def _cover_move(width: int, height: int, frames: int, preset: str) -> str:
    """The cover-fit move for ``preset`` (the still fills the frame; edges may crop).

    Pans hold the zoom-in's end scale (``1 + LOCAL_ZOOM_TRAVEL``, the recorder's
    ``1 + PAD``) and travel the whole per-side slack, so no frame ever shows past
    the still's edge.
    """
    if preset == "zoom_out":
        return _centred_zoom(width, height, frames, LOCAL_ZOOM_TRAVEL, reverse=True)
    if preset in _PANS:
        zoom = 1.0 + LOCAL_ZOOM_TRAVEL
        return _panned_window(frames, zoom, 0.5 - 0.5 / zoom, preset)
    return _centred_zoom(width, height, frames, LOCAL_ZOOM_TRAVEL)


def _contain_move(width: int, height: int, frames: int, travel: float, preset: str) -> str:
    """The never-crop move for ``preset`` over the inset composite (see the filter).

    The inset slide spans ``1/(1+travel)`` of the composite. ``zoom_in`` ends with
    it filling the frame and ``zoom_out`` starts there; a pan holds the midpoint
    zoom and drifts exactly as far as keeps the slide's far edge flush with the
    frame edge — the recorder's ``panScale``/``panTravel`` rule — so no part of
    the slide is ever cut.
    """
    if preset == "zoom_out":
        return _centred_zoom(width, height, frames, travel, reverse=True)
    if preset in _PANS:
        zoom = 1.0 + travel / 2.0
        return _panned_window(frames, zoom, 0.5 / zoom - 0.5 / (1.0 + travel), preset)
    return _centred_zoom(width, height, frames, travel)


def local_ken_burns_filter(
    width: int, height: int, fps: int, frames: int, fit: str, preset: str = DEFAULT_PRESET
) -> str:
    """The ffmpeg ``-filter_complex`` graph of the local push-in (pure, for tests).

    ``cover`` (a photo-like still): fill the frame, then push in by
    :data:`LOCAL_ZOOM_TRAVEL` — the edges are cropped, as the compositor would.

    ``contain`` (a DESIGNED still — a slide whose text runs to its safe margin):
    nothing of the still may ever leave the frame. It starts inset at
    ``1 / (1 + LOCAL_CONTAIN_TRAVEL)`` of the frame over a blurred copy of itself
    and the same centred zoom ends with it exactly filling the frame — the same
    never-crop rule the recorder's ``contain`` path keeps (``ken_burns._DEFAULT_FIT``).

    ``full`` (a designed still that ALREADY has the frame's aspect): the slide
    fills the frame on every frame — no inset, no blurred backdrop. The inset
    drew a darker ring around every slide (frame QA of run 01a11082), and a slide
    cannot be zoomed IN without cropping its text, so there is no geometric move
    at all: the only change over the clip is a slow brightness breath
    (:data:`LOCAL_FULL_BREATH`), which leaves the scale at the edges untouched.

    ``preset`` (``ken_burns.preset``, :data:`LOCAL_PRESETS`) picks the move of the
    ``cover`` and ``contain`` paths; an unknown value takes the default
    ``zoom_in`` (the caller records it — :func:`resolve_local_preset`). ``full``
    has no geometric move to steer, so the preset does not apply to it.
    """
    tail = f"fps={fps},format=yuv420p[out]"
    if fit == "full":
        span = max(frames - 1, 1)
        return (
            f"[0:v]scale={width}:{height}:flags=lanczos,setsar=1,"
            f"eq=brightness='{LOCAL_FULL_BREATH}*sin(PI*n/{span})':eval=frame,{tail}"
        )
    if fit == "cover":
        zoom = _cover_move(width, height, frames, preset)
        return (
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={width}:{height},setsar=1,{zoom},{tail}"
        )
    inset_w = _even(width / (1.0 + LOCAL_CONTAIN_TRAVEL))
    inset_h = _even(height / (1.0 + LOCAL_CONTAIN_TRAVEL))
    # Zoom exactly as far as brings the inset still to the frame edge, never past.
    travel = min(width / inset_w, height / inset_h) - 1.0
    zoom = _contain_move(width, height, frames, travel, preset)
    return (
        "[0:v]split[bg_in][fg_in];"
        f"[bg_in]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},boxblur=20:2[bg];"
        f"[fg_in]scale={inset_w}:{inset_h}:force_original_aspect_ratio=decrease:flags=lanczos[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,{zoom},{tail}"
    )


async def render_local_ken_burns(
    image: bytes,
    *,
    width: int,
    height: int,
    fps: int,
    duration_s: float,
    timeout_s: float,
    fit: str = "cover",
    preset: str = DEFAULT_PRESET,
) -> bytes:
    """Render ``image`` as a ``duration_s`` mp4 of the ``preset`` move (see the filter)."""
    if duration_s <= 0 or width <= 0 or height <= 0 or fps <= 0:
        raise LocalKenBurnsError("non-positive render geometry")
    frames = max(math.ceil(duration_s * fps), 2)
    out_w, out_h = local_render_dims(width, height)
    ffmpeg = _ffmpeg()
    with tempfile.TemporaryDirectory(prefix="kb-local-") as tmp:
        src = Path(tmp) / "still"
        dst = Path(tmp) / "out.mp4"
        src.write_bytes(image)
        await _run(
            [
                ffmpeg,
                "-nostdin",
                "-y",
                "-loglevel",
                "error",
                "-loop",
                "1",
                "-framerate",
                str(fps),
                "-i",
                str(src),
                # Two threads everywhere: the pod has a 3-CPU quota but SEES 16
                # cores, and x264's per-thread lookahead is what the memory is.
                "-filter_complex_threads",
                "2",
                "-filter_complex",
                local_ken_burns_filter(out_w, out_h, fps, frames, fit, preset),
                "-map",
                "[out]",
                "-frames:v",
                str(frames),
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "2",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-movflags",
                "+faststart",
                str(dst),
            ],
            timeout_s,
        )
        data = dst.read_bytes()
    if not data:
        raise LocalKenBurnsError("ffmpeg produced an empty file")
    return data


async def _thumb(source: str, *, last: bool, timeout_s: float) -> bytes:
    cmd = [_ffmpeg(), "-nostdin", "-loglevel", "error"]
    if last:
        cmd += ["-sseof", "-0.25"]
    cmd += [
        "-i",
        source,
        "-vf",
        f"scale={_THUMB_W}:{_THUMB_H},format=gray",
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    data = await _run(cmd, timeout_s)
    if len(data) != _THUMB_W * _THUMB_H:
        raise LocalKenBurnsError(f"thumbnail decode returned {len(data)} bytes")
    return data


def psnr(a: bytes, b: bytes) -> float:
    """PSNR (dB) of two equal-length 8-bit grey buffers; ``inf`` when identical."""
    if len(a) != len(b) or not a:
        raise LocalKenBurnsError("PSNR needs two equal, non-empty buffers")
    mse = sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) / len(a)
    return math.inf if mse == 0 else 10.0 * math.log10(255.0**2 / mse)


async def clip_motion_psnr(source: str, *, timeout_s: float = 60.0) -> float:
    """First-vs-last-frame PSNR of the clip at ``source`` (a path or URL)."""
    first = await _thumb(source, last=False, timeout_s=timeout_s)
    last = await _thumb(source, last=True, timeout_s=timeout_s)
    return psnr(first, last)


# ---------------------------------------------------------------------------
# The caller-side passes (``CompileStageBase._run_presubmit``). They live here,
# not in ``ken_burns.py``: that module owns the PAID recorder path; these are
# the orchestrator's own free, local work on either side of it.
# ---------------------------------------------------------------------------
_SOFT_FAIL_NOTE = "ken_burns: shot {shot_id} soft-failed, kept as a still: "
_ALL_FROZEN_NOTE_PREFIX = "ken_burns: ALL "


def _degrade(
    notes: list[str] | None, degradations: list[Degradation] | None, code: str, reason: str
) -> None:
    """One sentence on BOTH channels: the prose note and its coded twin."""
    if notes is not None:
        notes.append(reason)
    if degradations is not None:
        degradations.append(degradation(code, reason))


def cover_crop_still(image: bytes, width: int, height: int) -> bytes | None:
    """``image`` centre-cropped to the ``width:height`` aspect, as JPEG bytes.

    ``None`` when the still already has that aspect (within one pixel) or cannot
    be decoded — the caller keeps the original. A still within
    ``STILL_HEADROOM`` x the frame is only cut, never resampled, so a 1536x960
    (16:10) still becomes a native 1536x864 (16:9) one; a larger one (a pinned
    4032x3024 phone photo) is also downscaled to that bound.

    JPEG, not PNG, and bounded: the crop is converted to RGB anyway (no alpha to
    keep), and a LOSSLESS re-encode of a camera photo is several times the
    camera's own JPEG. Live run 01a1639e (2026-10-22): a 3.7 MB 4032x3024 pin
    came out of here as a >12 MiB 4032x2268 PNG, which the local render's
    size-capped re-fetch refused — the shot shipped frozen
    (``ken_burns_local_fallback_failed``).
    """
    try:
        with Image.open(io.BytesIO(image)) as opened:
            src = opened.convert("RGB")
    except Exception:  # noqa: BLE001 -- an undecodable still is the recorder's problem, not ours
        return None
    src_w, src_h = src.size
    if src_w <= 0 or src_h <= 0 or width <= 0 or height <= 0:
        return None
    target_w = min(src_w, round(src_h * width / height))
    target_h = min(src_h, round(src_w * height / width))
    if src_w - target_w <= 1 and src_h - target_h <= 1:
        return None
    left, top = (src_w - target_w) // 2, (src_h - target_h) // 2
    cropped = bound_to_frame(src.crop((left, top, left + target_w, top + target_h)), width, height)
    # Under the cap the local render's re-fetch enforces, stepping quality and
    # then size down if needed (2x a 4K frame is 33 MP). ``None`` (never fits)
    # keeps the original still, exactly like an undecodable one.
    encoded = encode_still_under(cropped, kb._MAX_IMAGE_BYTES)
    return None if encoded is None else encoded[0]


def has_frame_aspect(image: bytes, width: int, height: int) -> bool:
    """Whether ``image`` already has the ``width:height`` aspect, within one pixel.

    ``False`` for an undecodable image: the caller then takes the never-crop
    ``contain`` move, which is safe for any aspect.
    """
    try:
        with Image.open(io.BytesIO(image)) as opened:
            src_w, src_h = opened.size
    except Exception:  # noqa: BLE001 -- unknown aspect: not "same aspect"
        return False
    if src_w <= 0 or src_h <= 0 or width <= 0 or height <= 0:
        return False
    return abs(src_w - src_h * width / height) <= 1.0 or abs(src_h - src_w * height / width) <= 1.0


#: What each local move does, for the advisory that records it. ``cover`` and
#: ``contain`` read exactly as they always did.
_MOVE_BY_FIT = {
    "cover": "cover-safe centred push-in",
    "contain": "contain-safe centred push-in",
    "full": "full-frame, no geometric move, slow brightness breath",
}


def _move_description(fit: str, preset: str) -> str:
    """The advisory/log wording of one local move — unchanged for ``zoom_in``."""
    if fit == "full" or preset == DEFAULT_PRESET or preset not in LOCAL_PRESETS:
        return _MOVE_BY_FIT[fit]
    return f"{fit}-safe {preset.replace('_', ' ')}"


_ABSENT = object()


def resolve_local_preset(snapshot: Mapping[str, Any], resolved: str) -> tuple[str, str | None]:
    """``(preset to render, authored value that was NOT honoured or None)``.

    ``resolved`` is ``KenBurnsConfig.preset``, which coerces an unknown authored
    value to ``zoom_in`` without a word. That is right for the config model (a bad
    knob must not fail the run) and wrong for the run record: the author asked for
    a motion and got another. So the authored block is read again — from the
    snapshot, else the brief, the same lookup as ``ken_burns._resolve_cfg`` — and
    a non-empty value the build does not know is returned for the caller to record.
    """
    raw: object = snapshot.get("ken_burns", _ABSENT)
    if raw is _ABSENT:
        brief = snapshot.get("brief")
        if isinstance(brief, Mapping):
            raw = brief.get("ken_burns", _ABSENT)
    authored = raw.get("preset") if isinstance(raw, Mapping) else None
    preset = resolved if resolved in LOCAL_PRESETS else DEFAULT_PRESET
    if authored and str(authored) not in LOCAL_PRESETS:
        return preset, str(authored)
    return preset, None


def local_fit(
    shot: Mapping[str, Any], image: bytes, width: int, height: int, *, full_frame: bool = False
) -> str:
    """Which local move a still takes.

    Only a shot the compositor cover-fits may be cropped by a zoom. Every other
    still is a designed slide and takes the never-crop ``contain`` move — unless
    the run opted in to per-line motion (``full_frame``) AND the slide already
    has the frame's aspect: then ``full`` (it fills the frame, nothing is inset).
    """
    if shot.get("fit") == "cover":
        return "cover"
    if full_frame and has_frame_aspect(image, width, height):
        return "full"
    return "contain"


def _is_frozen_still(shot: Mapping[str, Any]) -> bool:
    url = shot.get("image_url")
    return (
        isinstance(url, str)
        and url.startswith(("http://", "https://"))
        and not shot.get("video_url")
    )


async def crop_cover_stills_to_frame(
    ctx: RunContext,
    shots: list[ShotDict],
    *,
    notes: list[str] | None = None,
    degradations: list[Degradation] | None = None,
) -> list[ShotDict]:
    """Give every COVER-fit still the frame's own aspect before it is animated.

    The Ken Burns recorder letterboxes a still whose aspect differs from the
    frame inside a blurred copy of itself (its run-wide ``contain`` default), and
    the edge where the sharp layer meets the blur reads as a hard vertical seam
    (run 01a10fcf: 1536x960 AI stills in a 16:9 film). A shot the compositor
    would cover-crop anyway loses nothing by being cropped first — and a still
    that already HAS the frame's aspect has no pillar to seam. Soft: any failure
    keeps the original still.
    """
    targets = [
        (index, shot)
        for index, shot in enumerate(shots)
        if _is_frozen_still(shot) and shot.get("fit") == "cover"
    ]
    if not targets or kb._resolve_cfg(ctx.config_snapshot) is None:
        return shots
    deliverable = ctx.config_snapshot.get("deliverable") or {}
    width, height = format_dims(deliverable.get("format"), quality_tier(ctx.config_snapshot))
    out = list(shots)
    async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_S) as http_client:
        for index, shot in targets:
            try:
                data, _content_type = await kb._fetch_own_asset(shot["image_url"], http_client)
                cropped = await asyncio.to_thread(cover_crop_still, data, width, height)
                if cropped is None:
                    continue
                image_id = deterministic_media_id(
                    "still_cover_crop", kb._content_key(shot["image_url"]), width, height
                )
                url = await store_generated_image(
                    run_id=str(ctx.run_id),
                    stage_name="ken_burns",
                    image_id=image_id,
                    image_bytes=cropped,
                    image_format="jpg",
                )
            except Exception as exc:  # noqa: BLE001 -- the original still stands
                log.warning("still crop for shot %s failed: %s", shot.get("id"), exc)
                _degrade(
                    notes,
                    degradations,
                    STILL_COVER_CROP_FAILED,
                    f"ken_burns: shot {shot.get('id')} could not be cropped to the frame "
                    f"aspect, its still keeps its own: {type(exc).__name__}: {exc}",
                )
                continue
            out[index] = {**shot, "image_url": url}
    return out


def _record(advisories: list[Degradation] | None, code: str, detail: str) -> None:
    log.info("ken_burns: %s: %s", code, detail)
    if advisories is not None:
        advisories.append(advisory(code, detail))


def _dump_local_ref(ref: dict[str, str]) -> JsonValue:
    # ``fit`` is stored only for the ``full`` move (per-line motion runs): for
    # every other still the move IS its pre-fetch class, so the stored reference
    # is the two keys it always was.
    stored: dict[str, JsonValue] = {"media_id": ref["media_id"], "url": ref["url"]}
    if ref.get("fit"):
        stored["fit"] = ref["fit"]
    return stored


def _load_local_ref(value: JsonValue) -> dict[str, str]:
    """A stored local-render reference, or raise (→ the ledger renders anew)."""
    if not isinstance(value, dict):
        raise ValueError(f"stored local Ken Burns reference is not an object: {value!r}")
    media_id, url = value.get("media_id"), value.get("url")
    if not isinstance(media_id, str) or not isinstance(url, str) or not url:
        raise ValueError(f"stored local Ken Burns reference is incomplete: {value!r}")
    fit = value.get("fit")
    ref = {"media_id": media_id, "url": url}
    if isinstance(fit, str) and fit:
        ref["fit"] = fit
    return ref


def _pass_preset(
    snapshot: Mapping[str, Any],
    resolved: str,
    notes: list[str] | None,
    degradations: list[Degradation] | None,
) -> str:
    """The preset of this pass; an authored preset it cannot honour is RECORDED."""
    preset, unknown = resolve_local_preset(snapshot, resolved)
    if unknown is not None:
        _degrade(
            notes,
            degradations,
            KEN_BURNS_PRESET_UNKNOWN,
            f"ken_burns: preset {unknown!r} is not one of "
            f"{', '.join(sorted(LOCAL_PRESETS))}; every still took the default "
            f"{DEFAULT_PRESET} move instead",
        )
    return preset


async def rescue_frozen_stills(
    ctx: RunContext,
    shots: list[ShotDict],
    *,
    notes: list[str] | None = None,
    advisories: list[Degradation] | None = None,
    master_fps: int | None = None,
    budget_seconds: float,
    degradations: list[Degradation] | None = None,
    local_first: bool = False,
) -> list[ShotDict]:
    """Animate every still the Ken Burns pass left frozen. Runs AFTER that pass.

    ``local_first`` — the product sends NO still to the recorder
    (``CompileStageBase.ken_burns_local_only``), so every still is rendered here by
    design: no recorder failure to report, hence no ``KEN_BURNS_LOCAL_FALLBACK``
    advisory (a failed local render is still degraded, as below).

    A still shot that comes back from ``apply_ken_burns`` with no ``video_url``
    is a recording that failed (run 01a10fcf: ``QUEUE_SATURATED``), was skipped
    for lack of budget, or was refused. Each is rendered here as a deterministic
    centred push-in (:func:`render_local_ken_burns`), stored like a recorded
    clip and recorded as :data:`KEN_BURNS_LOCAL_FALLBACK` with the recorder's
    own error; its soft-fail note is withdrawn, because the film is no longer
    frozen there. When the local render fails too the note stays and a second
    one says so — that shot IS a frozen still.

    ``degradations`` is the coded twin of ``notes`` and is kept in step with it
    in BOTH directions: a rescued shot's coded soft-fail entries are withdrawn
    with its note (their ``detail`` is that same sentence), and a local render
    that fails too adds :data:`KEN_BURNS_LOCAL_FALLBACK_FAILED`.

    Sequential on purpose: ffmpeg at 4K saturates the pod's CPUs. Bounded by
    ``budget_seconds`` overall, and each render by its SHARE of what is left
    (remaining / stills still to render), so one slow render cannot starve the
    stills after it. Ken Burns opted out (``ken_burns: false``) →
    nothing is animated.
    """
    targets = [(index, shot) for index, shot in enumerate(shots) if _is_frozen_still(shot)]
    cfg = kb._resolve_cfg(ctx.config_snapshot)
    if not targets or cfg is None:
        return shots
    deliverable = ctx.config_snapshot.get("deliverable") or {}
    width, height = format_dims(deliverable.get("format"), quality_tier(ctx.config_snapshot))
    fps = kb._master_fps(ctx.config_snapshot, master_fps)
    default_ms = int(cfg.duration_seconds * 1000)
    deadline = time.monotonic() + budget_seconds
    out = list(shots)
    ledger = kb._ken_burns_ledger(ctx)
    # The full-frame move for same-aspect slides is part of per-line motion.
    full_frame = motion_per_line_requested(ctx.config_snapshot)
    preset = _pass_preset(ctx.config_snapshot, cfg.preset, notes, degradations)
    # ``zoom_in`` keeps the address and ledger hash it always had, so a run that
    # resumes across this deploy reuses its clips; any other preset is part of both.
    preset_key: tuple[str, ...] = () if preset == DEFAULT_PRESET else (preset,)
    async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_S) as http_client:
        for position, (index, shot) in enumerate(targets):
            shot_id = shot.get("id")
            prefix = _SOFT_FAIL_NOTE.format(shot_id=shot_id)
            failure = next((n[len(prefix) :] for n in (notes or []) if n.startswith(prefix)), "")
            remaining = deadline - time.monotonic()
            try:
                duration_s = kb._shot_duration_ms(shot, default_ms) / 1000.0
                # Only a shot the compositor cover-fits may be cropped by the zoom;
                # every other still (a designed slide) takes a never-crop move —
                # WHICH one (``full`` / ``contain``) needs the image's own aspect,
                # so it is resolved after the fetch (``local_fit``). The address
                # and the ledger hash use this pre-fetch class: for one image the
                # resolved move is a pure function of it.
                fit = "cover" if shot.get("fit") == "cover" else "contain"
                content_key = kb._content_key(shot["image_url"])
                share = remaining / (len(targets) - position)
                media_id = deterministic_media_id(
                    "ken_burns_local",
                    fit,
                    content_key,
                    int(duration_s * 1000),
                    width,
                    height,
                    fps,
                    *preset_key,
                )

                async def _render_and_store(
                    shot: ShotDict = shot,
                    remaining: float = remaining,
                    duration_s: float = duration_s,
                    fit: str = fit,
                    share: float = share,
                    media_id: str = media_id,
                ) -> dict[str, str]:
                    if remaining <= 1.0:
                        raise LocalKenBurnsError("no pre-submit budget left for a local render")
                    data, _content_type = await kb._fetch_own_asset(shot["image_url"], http_client)
                    move = local_fit(shot, data, width, height, full_frame=full_frame)
                    video = await render_local_ken_burns(
                        data,
                        width=width,
                        height=height,
                        fps=fps,
                        duration_s=duration_s,
                        # Its SHARE of what is left, so one slow render cannot
                        # starve the stills behind it.
                        timeout_s=min(_LOCAL_RENDER_TIMEOUT_S, share),
                        fit=move,
                        preset=preset,
                    )
                    url = await store_generated_video(
                        run_id=str(ctx.run_id),
                        stage_name="ken_burns",
                        video_id=media_id,
                        video_bytes=video,
                        video_format="mp4",
                    )
                    ref = {"media_id": media_id, "url": url}
                    if move != fit:
                        ref["fit"] = move
                    return ref

                # Re-entry (pod death mid-compile) must not fetch + re-render a
                # still it already rescued: the ledger keeps the STORED object's
                # reference — never the bytes. The url is re-signed from its object
                # path by the compile's ``presign_2`` pass, so an aged signature
                # here is harmless. Image keyed by URL PATH (``_content_key``): the
                # presigned query rotates on every re-sign.
                stored = await ledger.once(
                    f"kb_local:{shot_id or content_key}",
                    hash_inputs(content_key, width, height, fps, duration_s, fit, *preset_key),
                    _render_and_store,
                    dump=_dump_local_ref,
                    load=_load_local_ref,
                )
                url = stored["url"]
                fit = stored.get("fit", fit)
            except Exception as exc:  # noqa: BLE001 -- the still (and its note) stands
                log.warning("ken_burns: local fallback for shot %s failed: %s", shot_id, exc)
                _degrade(
                    notes,
                    degradations,
                    KEN_BURNS_LOCAL_FALLBACK_FAILED,
                    f"ken_burns: local fallback for shot {shot_id} failed too, the still "
                    f"stays frozen: {type(exc).__name__}: {exc}",
                )
                continue
            out[index] = {**shot, "video_url": url}
            if notes is not None:
                # No longer true once one still moves: the per-shot soft-fail and
                # the pass's "ALL N still shot(s) soft-failed ... frozen stills".
                notes[:] = [n for n in notes if not n.startswith((prefix, _ALL_FROZEN_NOTE_PREFIX))]
            if degradations is not None:
                # The coded twins of exactly those notes — withdrawn together, or
                # a film that now moves would still be CODED as frozen.
                degradations[:] = [
                    d
                    for d in degradations
                    if not d["detail"].startswith((prefix, _ALL_FROZEN_NOTE_PREFIX))
                ]
            if local_first:
                log.info(
                    "ken_burns: shot %s animated locally (%s)",
                    shot_id,
                    _move_description(fit, preset),
                )
                continue
            _record(
                advisories,
                KEN_BURNS_LOCAL_FALLBACK,
                f"shot {shot_id}: recorder clip failed ({failure or 'no clip'}); animated "
                f"locally ({_move_description(fit, preset)}) instead of a frozen still",
            )
    return out


async def flag_near_static_clips(
    shots: list[ShotDict],
    was_still: set[int],
    *,
    advisories: list[Degradation] | None,
) -> None:
    """Record :data:`KEN_BURNS_NEAR_STATIC` for each animated still that barely moves.

    ``was_still`` are the indexes that were stills BEFORE the Ken Burns pass; the
    ones now carrying a clip are measured (first vs last frame). Run 01a10fcf's
    line-4 clip measured 38 dB and nobody knew. A measurement never costs a clip:
    every failure is swallowed.
    """
    for index in sorted(was_still):
        url = shots[index].get("video_url") if index < len(shots) else None
        if not isinstance(url, str) or not url:
            continue
        try:
            score = await clip_motion_psnr(url, timeout_s=_LOCAL_PROBE_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001 -- see docstring
            log.warning(
                "ken_burns: motion probe of shot %s failed: %s", shots[index].get("id"), exc
            )
            continue
        if score >= NEAR_STATIC_PSNR_DB:
            _record(
                advisories,
                KEN_BURNS_NEAR_STATIC,
                f"shot {shots[index].get('id')}: clip is near-static (first vs last frame "
                f"PSNR {score:.1f} dB >= {NEAR_STATIC_PSNR_DB:.0f} dB)",
            )
