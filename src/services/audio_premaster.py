"""Loudness pre-master for timeline audio — uploaded takes AND TTS narration.

The timeline's level contract is **-18 LUFS / -1.5 dBTP**. Two sources feed it
and NEITHER is guaranteed to arrive at that level:

* An operator-uploaded voiceover take arrives at whatever the operator's mic/DAW
  produced — a phone recording at -30 LUFS, a hot podcast bounce at -8 LUFS.
* ⚠️ A ``tts_backend`` clip is *usually*, but **not always**, pre-mastered.
  This module's docstring used to assert "every TTS-generated clip is already
  pre-mastered by tts_backend" — that is FALSE, and the assumption is exactly
  what let un-normalized narration onto the timeline unnoticed. The orchestrator
  learns the level from the ``X-TTS-LUFS`` response header, and that header is
  absent tens of times a day. See ``src/clients/tts_client.py`` for what an
  absent header actually means; the orchestrator-side enforcement point is
  ``src/services/stages/primitives/narration.py::ensure_tts_loudness``, which
  calls back into ``premaster_core`` below for any clip whose level is unknown.

The compiler puts those bytes straight on the timeline (``renderer.py`` fetches
``shot.audio_url`` and applies no gain), so an un-normalized clip sits next to
-18 LUFS neighbours as a 10-20 dB level jump mid-video.

This module closes that gap by running the SAME two-pass ffmpeg ``loudnorm``
tts_backend runs (ported from its ``services/post_processor.py``): pass 1
measures the input, pass 2 applies a ``linear=true`` loudnorm seeded with those
measurements — an exact, non-dynamic gain, so the audio is re-levelled without
pumping or squashing its dynamics.

⚠️ **Fail-open, always.** This runs inside an interactive HTTP upload. Every
failure path — the kill-switch off, no ffmpeg on PATH, an unknown container, a
non-zero ffmpeg exit, unparseable measurement JSON, a timeout, an empty output,
an unexpected exception — returns the INPUT bytes unchanged with empty metrics
and NEVER raises. A degraded outcome here is "the take is un-normalized" (exactly
today's behaviour), never "the upload 500s".

**Two independent kill-switches, one shared engine.** The engine is
``premaster_core`` — it checks NO flag; each entry point owns exactly one:

* ``premaster`` (upload path) checks ``VOICEOVER_PREMASTER_ENABLED``.
* ``ensure_tts_loudness`` (narration path, in
  ``services/stages/primitives/narration.py``) checks
  ``TTS_PREMASTER_FALLBACK_ENABLED`` and then calls ``premaster_core`` DIRECTLY.

That direct call is the whole point and is not an accident to be "tidied up" by
routing it through ``premaster``: doing so is what made the switches secretly
dependent. Flipping ``VOICEOVER_PREMASTER_ENABLED`` off used to also un-normalise
every narration clip, and the counter blamed ``premaster_failed_open`` — an
ffmpeg fault — for what was an operator config change. Both flags default ON and
are read per call, so a ConfigMap flip takes effect without a pod restart.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.utils import coerce_float

logger = logging.getLogger(__name__)

# The level tts_backend masters every synthesized clip to. These two numbers are
# the contract between the two services: change them here and the uploads drift
# away from their TTS neighbours, which is the entire bug this module fixes.
TARGET_LUFS = -18.0
TRUE_PEAK_DBTP = -1.5

# ffmpeg decodes the whole clip on each pass. A voiceover take is seconds-to-minutes
# long, so this is a generous ceiling on a pathological input rather than a budget
# — a hang must not tie up the worker (same posture as the upload route's ffprobe).
_PREMASTER_TIMEOUT_S = 120.0

# Duration probes are a single ffprobe header read, not a decode — a much tighter
# ceiling than the two-pass premaster above. Moved here verbatim from the upload
# route (`_VOICEOVER_PROBE_TIMEOUT_S`) when `probe_seconds` became shared.
_PROBE_TIMEOUT_S = 20.0

# The one fail-open reason callers are expected to BRANCH on, so it is a shared
# constant rather than prose inside a log line.
#
# EBU R128 integrated loudness is gated: blocks below the relative gate are
# excluded, and when nothing survives the gate ffmpeg reports ``input_i: "-inf"``.
# Measured with real ffmpeg on 24 kHz PCM: a 0.2 s and a 0.3 s tone both report
# ``-inf`` (too few 400 ms gating blocks to integrate over) while 0.5 s and up
# measure normally, and a 2 s tone attenuated to roughly -80 LUFS reports ``-inf``
# too (it sits under the -70 LUFS absolute gate). Feeding that back into pass 2 as
# ``measured_I=-inf`` makes the filter die with "Failed to inject frame into
# filter network: Result too large".
#
# Nothing has FAILED in that case — the clip simply has no integrated loudness to
# normalize toward, and normalizing silence up to -18 LUFS would be actively
# wrong. So it is a distinct, honest reason and the audio passes through untouched.
UNMEASURABLE_LEVEL = "unmeasurable_level"

# Explicit encoder per container. Deliberately NOT left to ffmpeg's extension
# guess: the stored artifact must keep the container the caller uploaded (an
# ``.mp3`` that silently became PCM would balloon a 25 MiB upload into ~250 MiB
# of WAV in the bucket), and pinning the encoder makes a missing-codec build fail
# loudly into the fail-open path instead of muxing something unexpected.
_ENCODERS: dict[str, list[str]] = {
    "wav": ["-c:a", "pcm_s16le"],
    "mp3": ["-c:a", "libmp3lame", "-q:a", "2"],
    "m4a": ["-c:a", "aac", "-b:a", "192k"],
    "aac": ["-c:a", "aac", "-b:a", "192k"],
    "ogg": ["-c:a", "libvorbis", "-q:a", "5"],
    "flac": ["-c:a", "flac"],
}


@dataclass(frozen=True, slots=True)
class LoudnessMetrics:
    """Measured loudness of one clip. Every field is nullable.

    Measurement is best-effort — on the fail-open path (and whenever ffmpeg's
    ``loudnorm`` JSON can't be read) the fields stay ``None`` rather than the
    caller getting a lie.
    """

    integrated_lufs: float | None = None
    true_peak_dbtp: float | None = None
    loudness_range_lu: float | None = None

    @property
    def measured(self) -> bool:
        """True when an integrated-loudness reading was actually obtained."""
        return self.integrated_lufs is not None


@dataclass(frozen=True, slots=True)
class PremasterResult:
    """What ``premaster_core`` did, with the WHY carried alongside the bytes.

    ``premaster``'s ``(bytes, metrics)`` tuple forces the caller to infer the
    outcome from object identity, which collapses every distinct fail-open cause
    into one indistinguishable "it did nothing". The narration path has to tell
    those causes apart to report an honest reason, so it consumes this instead.

    ``reason`` is ``""`` exactly when ``normalized`` is ``True``. Callers that
    branch should compare against ``UNMEASURABLE_LEVEL``; every other value is
    free-form diagnostic prose and must NOT become a metric label.
    """

    audio: bytes
    metrics: LoudnessMetrics
    normalized: bool
    reason: str = ""


def premaster_enabled() -> bool:
    """Whether uploaded audio is loudness-normalized (env ``VOICEOVER_PREMASTER_ENABLED``).

    Default ON. Read per call so the kill-switch takes effect on a ConfigMap
    change without waiting for a pod restart.
    """
    return _flag_on("VOICEOVER_PREMASTER_ENABLED")


def tts_premaster_fallback_enabled() -> bool:
    """Whether the orchestrator re-masters TTS clips of UNKNOWN loudness.

    Env ``TTS_PREMASTER_FALLBACK_ENABLED``, default ON, read per call so the
    kill-switch takes effect on a ConfigMap change without a pod restart.

    Genuinely SEPARATE from ``VOICEOVER_PREMASTER_ENABLED`` — the caller of this
    flag reaches ``premaster_core`` directly, so neither flag can gate the other's
    path. The two paths have different blast radii: the upload path runs once per
    human action; the TTS fallback runs inside a fan-out narrate on the paid render
    path, so it must be possible to shed that ffmpeg cost (or a bad interaction
    with a specific tts_backend build) without also un-normalizing operator
    uploads, and to kill uploads without silently un-normalizing all narration.
    """
    return _flag_on("TTS_PREMASTER_FALLBACK_ENABLED")


def _flag_on(name: str) -> bool:
    """``True`` unless *name* is explicitly set to a falsey word. Default ON."""
    return os.getenv(name, "true").strip().lower() not in ("false", "0", "no", "off")


async def probe_seconds(data: bytes, ext: str, *, demuxer: str | None = None) -> float | None:
    """ffprobe *data* and return its duration in seconds, or ``None``.

    ``None`` means "not decodable audio" — no ffprobe on PATH, a non-zero exit (a
    corrupt/empty container), a probe that reports no/zero duration, or a probe
    that hangs past the timeout. Callers treat ``None`` as indeterminate and take
    their own conservative branch: the upload route 400s rather than store a file
    the compiler would later resolve to SILENCE; ``ensure_tts_loudness`` discards
    the re-encoded bytes rather than let an unverifiable duration move the
    timeline.

    Probed from a temp FILE rather than stdin: a seekable input is required for
    the MP4/M4A family (whose ``moov`` atom may sit at the end), so piping would
    reject valid inputs. The write is off-loaded to a thread so the event loop is
    never blocked on disk IO.

    Lives here rather than in ``api/routes/uploads.py`` (where it started) so the
    upload path and the TTS timing guard share ONE implementation — two probes
    that could disagree about a duration is exactly the drift this guards against.

    The input is untrusted, so ffprobe is confined to the temp file
    (``-protocol_whitelist file``): a playlist-shaped input (HLS, concat) cannot
    make it open anything else. ``demuxer`` forces the container a caller already
    SNIFFED from the magic bytes — without it ffprobe auto-detects, and it detects
    AFTER skipping an ID3v2 tag, so ``ID3`` + an m3u8 playlist sniffs as ``mp3`` yet
    probes as HLS and reports the duration of whatever the playlist points at.
    """
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    with tempfile.TemporaryDirectory(prefix="w2v-probe-") as tmpdir:
        path = Path(tmpdir) / f"probe.{ext}"
        await asyncio.to_thread(path.write_bytes, data)
        forced = ["-f", demuxer] if demuxer else []
        proc = await asyncio.create_subprocess_exec(
            ffprobe,
            "-v",
            "error",
            "-protocol_whitelist",
            "file",
            *forced,
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            async with asyncio.timeout(_PROBE_TIMEOUT_S):
                stdout, _ = await proc.communicate()
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return None
    if proc.returncode != 0:
        return None
    try:
        seconds = float(stdout.decode("utf-8", "replace").strip())
    except ValueError:
        return None
    return seconds if seconds > 0 else None


async def _run(ffmpeg: str, args: list[str]) -> tuple[int, str] | None:
    """Run ffmpeg with *args* and return ``(returncode, stderr)``; ``None`` on timeout.

    ffmpeg writes everything interesting (including the ``loudnorm`` JSON block)
    to stderr. Uses ``create_subprocess_exec`` + ``asyncio.timeout`` — the same
    non-blocking, bounded shape the upload route's ffprobe helper uses, so a
    pathological input can never wedge the event loop or the worker.
    """
    proc = await asyncio.create_subprocess_exec(
        ffmpeg,
        *args,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        async with asyncio.timeout(_PREMASTER_TIMEOUT_S):
            _, stderr = await proc.communicate()
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return None
    return proc.returncode or 0, stderr.decode("utf-8", "replace")


def _parse_loudnorm_json(stderr: str) -> dict[str, object] | None:
    """Parse the trailing ``print_format=json`` block ffmpeg prints to stderr."""
    start = stderr.rfind("{")
    end = stderr.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(stderr[start:end])
    except json.JSONDecodeError, ValueError, TypeError:
        return None
    return parsed if isinstance(parsed, dict) else None


async def _probe_sample_rate(path: Path) -> int | None:
    """The input's sample rate via ffprobe; ``None`` if it can't be read.

    Load-bearing, NOT cosmetic: ffmpeg's ``loudnorm`` filter runs at 192 kHz
    internally and emits at 192 kHz unless the output rate is pinned. Left
    unpinned, a 24 kHz take would be stored as a 192 kHz artifact (8x the bytes),
    and lossy encoders would simply refuse it (libmp3lame tops out at 48 kHz) —
    turning the whole premaster into a silent no-op for mp3. So we probe the input
    rate and hand it back to ffmpeg via ``-ar``; failing to read it is a fail-open
    path, not something to guess at.
    """
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    proc = await asyncio.create_subprocess_exec(
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        async with asyncio.timeout(_PREMASTER_TIMEOUT_S):
            stdout, _ = await proc.communicate()
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return None
    if proc.returncode != 0:
        return None
    try:
        rate = int(stdout.decode("utf-8", "replace").strip())
    except ValueError:
        return None
    return rate if rate > 0 else None


async def _measure(ffmpeg: str, path: Path) -> dict[str, object] | None:
    """Pass 1: run ``loudnorm`` in measurement mode and return its parsed JSON.

    Output is discarded to ``-f null``; only the printed ``input_i`` / ``input_tp``
    / ``input_lra`` / ``input_thresh`` readings matter — they seed pass 2. Returns
    ``None`` on a non-zero exit, a timeout, or unparseable output.
    """
    outcome = await _run(
        ffmpeg,
        [
            "-hide_banner",
            "-nostdin",
            "-i",
            str(path),
            "-af",
            f"loudnorm=I={TARGET_LUFS}:TP={TRUE_PEAK_DBTP}:print_format=json",
            "-f",
            "null",
            "-",
        ],
    )
    if outcome is None:
        return None
    returncode, stderr = outcome
    if returncode != 0:
        return None
    return _parse_loudnorm_json(stderr)


async def measure_loudness(data: bytes, audio_format: str) -> LoudnessMetrics:
    """Measure integrated LUFS / true-peak / LRA of *data*, best-effort.

    Never raises: on any ffmpeg or parse failure the fields stay ``None``.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return LoudnessMetrics()
    try:
        with tempfile.TemporaryDirectory(prefix="w2v-lufs-") as tmpdir:
            path = Path(tmpdir) / f"measure.{audio_format}"
            await asyncio.to_thread(path.write_bytes, data)
            measured = await _measure(ffmpeg, path)
    except OSError:
        return LoudnessMetrics()
    if measured is None:
        return LoudnessMetrics()
    return LoudnessMetrics(
        integrated_lufs=coerce_float(measured.get("input_i")),
        true_peak_dbtp=coerce_float(measured.get("input_tp")),
        loudness_range_lu=coerce_float(measured.get("input_lra")),
    )


def _fail_open(
    data: bytes, reason: str, *, exc_info: bool = False
) -> tuple[bytes, LoudnessMetrics]:
    """Pass the input bytes through unchanged and say why, at WARNING.

    Logged every time (not latched like tts_backend's per-process synth path):
    an upload is a rare, human-driven event, so one line per un-normalized take is
    exactly the signal an operator needs to explain a level jump after the fact.
    """
    logger.warning(
        "voiceover premaster passing audio through unchanged (%s)", reason, exc_info=exc_info
    )
    return data, LoudnessMetrics()


def _fail_open_result(data: bytes, reason: str, *, exc_info: bool = False) -> PremasterResult:
    """``_fail_open`` for ``premaster_core``: same passthrough, reason KEPT.

    Returns the caller's own ``data`` object (not a copy) so the identity-based
    fail-open detection that ``premaster``'s tuple contract relies on survives the
    delegation.
    """
    data, metrics = _fail_open(data, reason, exc_info=exc_info)
    return PremasterResult(audio=data, metrics=metrics, normalized=False, reason=reason)


async def _two_pass(
    ffmpeg: str,
    data: bytes,
    audio_format: str,
    encoder: list[str],
    target_lufs: float,
    true_peak: float,
) -> tuple[bytes | None, str]:
    """Run the two ffmpeg passes in a temp dir.

    Returns ``(normalized_bytes, "")`` on success or ``(None, reason)`` on any
    handled failure — the caller turns a reason into a fail-open passthrough.
    """
    with tempfile.TemporaryDirectory(prefix="w2v-premaster-") as tmpdir:
        in_path = Path(tmpdir) / f"in.{audio_format}"
        out_path = Path(tmpdir) / f"out.{audio_format}"
        await asyncio.to_thread(in_path.write_bytes, data)

        sample_rate = await _probe_sample_rate(in_path)
        if sample_rate is None:
            return None, "could not probe the input sample rate"

        measured = await _measure(ffmpeg, in_path)
        if measured is None:
            return None, "pass-1 loudnorm measurement failed"
        # Bail BEFORE pass 2 on a non-finite reading (see UNMEASURABLE_LEVEL).
        # Left to run, pass 2 dies on "Result too large" and the caller reports an
        # ffmpeg fault for a clip that is merely too short or too quiet to have an
        # integrated loudness at all.
        input_i = coerce_float(measured.get("input_i"))
        if input_i is None or not math.isfinite(input_i):
            return None, UNMEASURABLE_LEVEL
        try:
            audio_filter = (
                f"loudnorm=I={target_lufs}:TP={true_peak}"
                f":measured_I={measured['input_i']}"
                f":measured_TP={measured['input_tp']}"
                f":measured_LRA={measured['input_lra']}"
                f":measured_thresh={measured['input_thresh']}"
                f":linear=true"
            )
        except KeyError as exc:
            return None, f"pass-1 loudnorm JSON missing {exc}"

        outcome = await _run(
            ffmpeg,
            [
                "-hide_banner",
                "-nostdin",
                "-y",
                "-i",
                str(in_path),
                "-af",
                audio_filter,
                # Pin the output rate back to the input's — loudnorm emits
                # 192 kHz otherwise (see _probe_sample_rate).
                "-ar",
                str(sample_rate),
                *encoder,
                str(out_path),
            ],
        )
        if outcome is None:
            return None, "pass-2 loudnorm timed out"
        returncode, stderr = outcome
        if returncode != 0:
            return None, f"pass-2 loudnorm exit {returncode}: {stderr[-300:]}"
        if not out_path.exists() or out_path.stat().st_size == 0:
            return None, "pass-2 loudnorm produced empty output"

        return await asyncio.to_thread(out_path.read_bytes), ""


async def premaster_core(
    data: bytes,
    audio_format: str,
    *,
    target_lufs: float = TARGET_LUFS,
    true_peak: float = TRUE_PEAK_DBTP,
) -> PremasterResult:
    """Two-pass ``loudnorm`` *data* to *target_lufs* / *true_peak*. Fail-open.

    ⚠️ **Checks no kill-switch.** This is the shared engine; every entry point owns
    exactly one flag and calls in here (see the module docstring). Calling this
    from a new place without a flag of its own creates an un-killable ffmpeg path.

    Pass 1 measures the input; pass 2 re-encodes it with a ``linear=true``
    loudnorm seeded with those measurements — a single exact gain across the whole
    clip, so the audio is re-levelled to sit with its -18 LUFS neighbours without
    dynamic pumping. The container is preserved (see ``_ENCODERS``), so the
    caller's format/extension stays valid.

    On success: ``normalized=True``, ``audio`` is the re-encoded bytes, ``metrics``
    measures the OUTPUT, ``reason`` is ``""``. On EVERY failure path it returns the
    input bytes unchanged with empty metrics, ``normalized=False`` and a reason,
    and never raises:

    * ``UNMEASURABLE_LEVEL`` — pass 1 read a non-finite integrated loudness, i.e.
      the clip is under ~0.5 s or near-silent. Not a failure; see the constant.
    * ffmpeg is not on PATH
    * the container has no pinned encoder
    * the input's sample rate can't be probed (see ``_probe_sample_rate``)
    * pass 1 exits non-zero / times out / prints unparseable JSON
    * pass 1's JSON is missing a measurement key
    * pass 2 exits non-zero / times out / produces an empty file
    * any unexpected exception (temp-dir / IO / OS)
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return _fail_open_result(data, "ffmpeg not found on PATH")
    encoder = _ENCODERS.get(audio_format)
    if encoder is None:
        return _fail_open_result(data, f"no pinned encoder for format {audio_format!r}")

    try:
        out_bytes, reason = await _two_pass(
            ffmpeg, data, audio_format, encoder, target_lufs, true_peak
        )
    except OSError, ValueError:
        return _fail_open_result(data, "unexpected error", exc_info=True)
    if out_bytes is None:
        return _fail_open_result(data, reason)

    # The two-pass normalization already SUCCEEDED — keep the normalized bytes even
    # if this trailing measurement (advisory only, for the log line) blows up. A
    # measurement failure must never discard a good normalization or raise into the
    # upload handler.
    try:
        metrics = await measure_loudness(out_bytes, audio_format)
    except OSError, ValueError:
        logger.warning("voiceover premaster: output measurement failed", exc_info=True)
        metrics = LoudnessMetrics()
    logger.info(
        "audio premastered format=%s bytes=%d->%d integrated_lufs=%s true_peak_dbtp=%s",
        audio_format,
        len(data),
        len(out_bytes),
        metrics.integrated_lufs,
        metrics.true_peak_dbtp,
    )
    return PremasterResult(audio=out_bytes, metrics=metrics, normalized=True)


async def premaster(
    data: bytes,
    audio_format: str,
    *,
    target_lufs: float = TARGET_LUFS,
    true_peak: float = TRUE_PEAK_DBTP,
) -> tuple[bytes, LoudnessMetrics]:
    """The UPLOAD path's premaster: ``premaster_core`` gated by its own kill-switch.

    Gated by ``VOICEOVER_PREMASTER_ENABLED`` and by NOTHING else — in particular it
    has no say over narration, which runs its own flag against ``premaster_core``.

    Returns ``(normalized_bytes, metrics_of_the_output)`` on success and
    ``(data, LoudnessMetrics())`` — the byte-identical INPUT OBJECT, empty metrics —
    on every failure path including the flag being off. Callers detect fail-open by
    identity (``out is data``); a caller that needs to know WHY should use
    ``premaster_core`` behind its own flag instead of guessing.
    """
    if not premaster_enabled():
        return _fail_open(data, "VOICEOVER_PREMASTER_ENABLED is off")
    result = await premaster_core(data, audio_format, target_lufs=target_lufs, true_peak=true_peak)
    return result.audio, result.metrics
