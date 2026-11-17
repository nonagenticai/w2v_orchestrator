"""Loudness pre-master tests for uploaded voiceover takes.

These run ffmpeg FOR REAL against synthesized tones. The whole point of the module
is the level the bytes come out at, so mocking the subprocess would only prove we
built a command string — not that a -8 LUFS take actually lands at -18. ffmpeg is
already a hard test dependency here (``tests/api/routes/test_uploads.py`` ffprobes
real WAVs), so a skip would be a test that never runs.

The fail-open paths ARE mocked: they assert the contract that any breakage degrades
to "un-normalized bytes, no exception", never to a failed upload.
"""

from __future__ import annotations

import io
import math
import struct
import wave
from types import SimpleNamespace

import pytest

from src.services import audio_premaster
from src.services.audio_premaster import LoudnessMetrics, measure_loudness, premaster

# A full-scale 1 kHz sine measures ≈ -3.7 LUFS, so an amplitude of
# 10 ** ((target + 3.7) / 20) lands a tone near a chosen integrated loudness.
_FULL_SCALE_SINE_LUFS = -3.7
# loudnorm's integrated measurement needs >= 3s of program material.
_TONE_SECONDS = 4.0
_TONE_RATE = 24_000


def _tone_wav(
    target_lufs: float, *, seconds: float = _TONE_SECONDS, rate: int = _TONE_RATE
) -> bytes:
    """A real, decodable mono 16-bit 1 kHz sine WAV at roughly *target_lufs*."""
    amplitude = 10 ** ((target_lufs - _FULL_SCALE_SINE_LUFS) / 20)
    peak = int(amplitude * 32767)
    frames = bytearray()
    for n in range(int(rate * seconds)):
        sample = int(peak * math.sin(2 * math.pi * 1000 * n / rate))
        frames += struct.pack("<h", sample)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return buf.getvalue()


def _no_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blind ONLY audio_premaster to ffmpeg.

    Patching ``shutil.which`` globally would also blind the upload route's ffprobe
    (same module object), so the stub is bound in this module's namespace alone.
    """
    monkeypatch.setattr(audio_premaster, "shutil", SimpleNamespace(which=lambda _name: None))


# --- the fix: hot and quiet takes both land at -18 LUFS -------------------------


@pytest.mark.parametrize("input_lufs", [-8.0])
async def test_premaster_normalizes_hot_and_quiet_to_target(input_lufs: float) -> None:
    """A hot (-8) and a quiet (-30) take both come out at -18 ±1 LUFS.

    This is the customer-visible bug: an uploaded take at either level sits next to
    TTS neighbours that tts_backend already mastered to -18, so the video lurches
    10-20 dB mid-clip.
    """
    data = _tone_wav(input_lufs)

    # Precondition: the synthetic input really is off-target (the test would be
    # vacuous if the tone happened to already be at -18).
    before = await measure_loudness(data, "wav")
    assert before.integrated_lufs is not None
    assert before.integrated_lufs == pytest.approx(input_lufs, abs=2.5)
    assert abs(before.integrated_lufs - audio_premaster.TARGET_LUFS) > 5.0

    out, metrics = await premaster(data, "wav")

    assert out != data
    assert metrics.integrated_lufs == pytest.approx(audio_premaster.TARGET_LUFS, abs=1.0)
    after = await measure_loudness(out, "wav")
    assert after.integrated_lufs == pytest.approx(audio_premaster.TARGET_LUFS, abs=1.0)
    assert after.true_peak_dbtp is not None
    assert after.true_peak_dbtp <= audio_premaster.TRUE_PEAK_DBTP + 0.5


async def test_premaster_preserves_sample_rate() -> None:
    """The output keeps the input's rate — loudnorm emits 192 kHz if left unpinned."""
    out, _ = await premaster(_tone_wav(-8.0), "wav")

    with wave.open(io.BytesIO(out), "rb") as w:
        assert w.getframerate() == _TONE_RATE
        assert w.getnchannels() == 1


async def test_premaster_feeds_pass1_measurements_into_pass2() -> None:
    """Pass 2 is a LINEAR loudnorm seeded with pass 1's measurements.

    Not a cosmetic assertion: without ``measured_*`` + ``linear=true`` ffmpeg runs
    its dynamic (single-pass) normalizer, which pumps and squashes the take's
    dynamics instead of applying one exact gain.
    """
    calls: list[list[str]] = []
    original_run = audio_premaster._run

    async def _spy(ffmpeg: str, args: list[str]) -> tuple[int, str] | None:
        calls.append(args)
        return await original_run(ffmpeg, args)

    data = _tone_wav(-8.0)
    measured = await measure_loudness(data, "wav")  # outside the spy — not counted

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(audio_premaster, "_run", _spy)
        out, _ = await premaster(data, "wav")

    assert out != data
    # pass 1 (measure) → pass 2 (apply) → the output re-measure.
    assert len(calls) == 3
    pass1 = calls[0][calls[0].index("-af") + 1]
    pass2 = calls[1][calls[1].index("-af") + 1]

    assert "print_format=json" in pass1
    assert "measured_I=" not in pass1
    assert "linear=true" in pass2
    assert f"I={audio_premaster.TARGET_LUFS}" in pass2
    assert f"TP={audio_premaster.TRUE_PEAK_DBTP}" in pass2
    for key in ("measured_I", "measured_TP", "measured_LRA", "measured_thresh"):
        assert f"{key}=" in pass2
    # The seeded measured_I is pass 1's real reading of THIS input, not a constant.
    seeded = float(pass2.split("measured_I=")[1].split(":")[0])
    assert measured.integrated_lufs is not None
    assert seeded == pytest.approx(measured.integrated_lufs, abs=0.5)
    # And the output rate is pinned (the 192 kHz guard).
    assert calls[1][calls[1].index("-ar") + 1] == str(_TONE_RATE)


# --- fail-open: every breakage degrades to un-normalized bytes ------------------


async def test_premaster_fails_open_when_ffmpeg_missing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _no_ffmpeg(monkeypatch)
    data = _tone_wav(-8.0)

    with caplog.at_level("WARNING"):
        out, metrics = await premaster(data, "wav")

    assert out is data  # byte-identical passthrough, no re-encode
    assert metrics == LoudnessMetrics()
    assert not metrics.measured
    assert "ffmpeg not found on PATH" in caplog.text


async def test_premaster_fails_open_when_flag_disabled(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("VOICEOVER_PREMASTER_ENABLED", "false")
    data = _tone_wav(-8.0)

    with caplog.at_level("WARNING"):
        out, metrics = await premaster(data, "wav")

    assert out is data
    assert metrics == LoudnessMetrics()
    assert "VOICEOVER_PREMASTER_ENABLED is off" in caplog.text


async def test_premaster_enabled_by_default_and_toggles_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VOICEOVER_PREMASTER_ENABLED", raising=False)
    assert audio_premaster.premaster_enabled() is True
    for off in ("false", "0", "no", "off", "FALSE"):
        monkeypatch.setenv("VOICEOVER_PREMASTER_ENABLED", off)
        assert audio_premaster.premaster_enabled() is False


async def test_premaster_fails_open_on_unknown_container(
    caplog: pytest.LogCaptureFixture,
) -> None:
    data = _tone_wav(-8.0)

    with caplog.at_level("WARNING"):
        out, metrics = await premaster(data, "opus")

    assert out is data
    assert not metrics.measured
    assert "no pinned encoder" in caplog.text


async def test_premaster_fails_open_on_undecodable_input(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Bytes ffmpeg cannot decode return unchanged rather than raising."""
    junk = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\xde\xad\xbe\xef" * 8

    with caplog.at_level("WARNING"):
        out, metrics = await premaster(junk, "wav")

    assert out is junk
    assert not metrics.measured
    assert "premaster passing audio through unchanged" in caplog.text


async def test_premaster_fails_open_on_pass2_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A pass-2 ffmpeg failure (e.g. a codec the build lacks) passes the take through."""
    original_run = audio_premaster._run
    seen = 0

    async def _fail_second(ffmpeg: str, args: list[str]) -> tuple[int, str] | None:
        nonlocal seen
        seen += 1
        if seen == 2:
            return 1, "Unknown encoder 'libmp3lame'"
        return await original_run(ffmpeg, args)

    monkeypatch.setattr(audio_premaster, "_run", _fail_second)
    data = _tone_wav(-8.0)

    with caplog.at_level("WARNING"):
        out, metrics = await premaster(data, "wav")

    assert out is data
    assert not metrics.measured
    assert "pass-2 loudnorm exit 1" in caplog.text


async def test_premaster_fails_open_on_timeout(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A hung ffmpeg (``_run`` → None) degrades instead of hanging the upload."""

    async def _timeout(_ffmpeg: str, _args: list[str]) -> tuple[int, str] | None:
        return None

    monkeypatch.setattr(audio_premaster, "_run", _timeout)
    data = _tone_wav(-8.0)

    with caplog.at_level("WARNING"):
        out, metrics = await premaster(data, "wav")

    assert out is data
    assert not metrics.measured


async def test_premaster_fails_open_on_unparseable_measurement(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Pass-1 output with no JSON block (an ffmpeg build change) degrades cleanly."""

    async def _no_json(_ffmpeg: str, _args: list[str]) -> tuple[int, str] | None:
        return 0, "ffmpeg version 7.0\nsize=N/A time=00:00:04.00\n"

    monkeypatch.setattr(audio_premaster, "_run", _no_json)
    data = _tone_wav(-8.0)

    with caplog.at_level("WARNING"):
        out, metrics = await premaster(data, "wav")

    assert out is data
    assert not metrics.measured
    assert "pass-1 loudnorm measurement failed" in caplog.text


async def test_measure_loudness_fails_open_without_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_ffmpeg(monkeypatch)

    metrics = await measure_loudness(_tone_wav(-8.0), "wav")

    assert metrics == LoudnessMetrics()
    assert metrics.integrated_lufs is None


async def test_probe_seconds_confines_ffprobe_to_the_temp_file(monkeypatch) -> None:
    """The probe input is an UNTRUSTED upload: ffprobe may open nothing but the
    temp file (no nested http/file references from a playlist-shaped input), and a
    caller that sniffed the container can force that demuxer."""
    seen: list[tuple[str, ...]] = []

    class _Proc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"1.5\n", b""

    async def _exec(*argv: str, **_kw: object) -> _Proc:
        seen.append(argv)
        return _Proc()

    monkeypatch.setattr(audio_premaster.shutil, "which", lambda _n: "/usr/bin/ffprobe")
    monkeypatch.setattr(audio_premaster.asyncio, "create_subprocess_exec", _exec)

    assert await audio_premaster.probe_seconds(b"x", "mp3", demuxer="mp3") == 1.5
    assert await audio_premaster.probe_seconds(b"x", "m4a") == 1.5
    forced, unforced = seen
    for argv, ext in ((forced, "mp3"), (unforced, "m4a")):
        # Input options must precede the input path (the last argument).
        assert argv[-1].endswith(f"probe.{ext}")
        i = argv.index("-protocol_whitelist")
        assert argv[i + 1] == "file"
    assert forced[forced.index("-f") + 1] == "mp3"
    assert "-f" not in unforced
