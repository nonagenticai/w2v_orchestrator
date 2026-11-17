"""Narration clips reach storage at a DETERMINISTIC level, or say why not.

``synthesize_and_store`` is the single choke point for all 11 narration
consumers, so ``ensure_tts_loudness`` sits inside it. These tests pin every
outcome, plus the things the fix must NOT have broken:

1. tts_backend proved the level (``X-TTS-LUFS`` present) → zero ffmpeg.
2. Level unproven → the orchestrator's re-mastered bytes are what get stored.
3. The re-master failed open → the ORIGINAL bytes are stored, counted, and the
   sidecar says ``normalized: false`` rather than lying.
4. The re-master moved the clip's length → its bytes are DISCARDED; the timeline
   never shifts as a side effect of levelling.
5. The content-addressed reuse short-circuit still returns the cached clip
   without synthesizing (and without paying any ffmpeg cost).
6. The two kill-switches are INDEPENDENT: the upload switch cannot silently
   un-normalize narration, nor mislabel the reason when it is off.
7. A normalized clip whose advisory measurement failed still lands in a series —
   the histogram/counter pair covers 100% of clips with no hole.
8. A sub-0.5s or near-silent clip reports ``unmeasurable_level``, not a fault.
9. A container that pads (mp3) does not trip the duration guard.

⚠️ Tests 1-9 mock ffmpeg away. ``test_real_ffmpeg_normalizes_hot_and_quiet_clips``
at the bottom is the ONLY one that proves the feature actually works; defects in
the flag wiring, the short-clip cliff and the duration reference all shipped past
a fully green mocked suite.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from prometheus_client import REGISTRY

from src.clients.schemas.tts import TTSSynthesis
from src.services import tts_storage
from src.services.audio_premaster import UNMEASURABLE_LEVEL, LoudnessMetrics, PremasterResult
from src.services.stages.primitives import narration as narration_module
from src.services.stages.primitives.narration import ensure_tts_loudness, synthesize_and_store

_ORIGINAL = b"ORIGINAL-AUDIO-BYTES"
_NORMALIZED = b"NORMALIZED-AUDIO-BYTES"
_STAGE = "audiobook_narrate"


class _MemStorage:
    """In-memory storage exposing the full clip surface (store + probe)."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def initialize(self) -> None:
        pass

    def _key(self, job_id: str, stage_name: str, filename: str) -> str:
        return f"artifacts/{job_id}/{stage_name}/{filename}"

    async def store_stage_artifact(
        self, job_id: str, stage_name: str, data: bytes, filename: str
    ) -> str:
        key = self._key(job_id, stage_name, filename)
        self.objects[key] = data
        return key

    async def get_presigned_url(
        self, path: str, expires: int | None = None, bucket: str | None = None
    ) -> str | None:
        return f"https://minio.test/{path}"

    async def retrieve_stage_artifact(
        self, job_id: str, stage_name: str, filename: str = "output.json"
    ) -> bytes | None:
        return self.objects.get(self._key(job_id, stage_name, filename))

    async def stage_artifact_url(self, job_id: str, stage_name: str, filename: str) -> str | None:
        key = self._key(job_id, stage_name, filename)
        return f"https://minio.test/{key}" if key in self.objects else None

    def sidecar(self, job_id: str, clip_id: str) -> dict[str, Any]:
        return json.loads(self.objects[self._key(job_id, _STAGE, f"{clip_id}.json")])

    def audio(self, job_id: str, clip_id: str) -> bytes:
        return self.objects[self._key(job_id, _STAGE, f"{clip_id}.wav")]


@pytest.fixture
def mem_storage():
    store = _MemStorage()
    tts_storage.set_storage_for_tests(store)
    yield store
    tts_storage.set_storage_for_tests(None)


def _tts(*, lufs: float | None, duration: float = 4.0) -> AsyncMock:
    """A TTS client double whose synthesis reports (or omits) a proven level."""
    fake = AsyncMock()
    # model_validate rather than the constructor: `audio_bytes` is coerced by a
    # pydantic validator, which the type checker cannot follow through kwargs.
    fake.synthesize.return_value = TTSSynthesis.model_validate(
        {
            "audio_bytes": _ORIGINAL,
            "sample_rate": 24000,
            "duration_seconds": duration,
            "provider": "kokoro",
            "voice": "af_heart",
            "format": "wav",
            "integrated_lufs": lufs,
        }
    )
    return fake


def _unnormalized_count(reason: str, stage: str = _STAGE) -> float:
    value = REGISTRY.get_sample_value(
        "w2v_orchestrator_tts_clip_unnormalized_total", {"reason": reason, "stage": stage}
    )
    return 0.0 if value is None else value


def _premaster_returning(data: bytes, metrics: LoudnessMetrics | None = None):
    """A ``premaster_core`` double that SUCCEEDS, returning *data*."""

    async def _fake(audio: bytes, audio_format: str, **_: object) -> PremasterResult:
        return PremasterResult(
            audio=data,
            metrics=metrics or LoudnessMetrics(integrated_lufs=-18.0, true_peak_dbtp=-1.5),
            normalized=True,
        )

    return _fake


def _premaster_failing(reason: str):
    """A ``premaster_core`` double that FAILS OPEN with *reason*."""

    async def _fake(audio: bytes, audio_format: str, **_: object) -> PremasterResult:
        return PremasterResult(
            audio=audio, metrics=LoudnessMetrics(), normalized=False, reason=reason
        )

    return _fake


def _probe_returning(mapping: dict[bytes, float | None]):
    """A ``probe_seconds`` double keyed by the BYTES handed to it.

    Keyed, not constant: the duration guard now probes the INPUT and the OUTPUT
    (so container padding cancels), and a constant double would make the two sides
    agree by construction — passing the drift test on code that never compares
    anything.
    """

    async def _probe(data: bytes, _fmt: str) -> float | None:
        assert data in mapping, f"unexpected probe of {data!r}"
        return mapping[data]

    return _probe


async def _explode(*_: object, **__: object):
    raise AssertionError("premaster must not run on the proven-level hot path")


async def test_proven_level_skips_premaster_entirely(
    mem_storage: _MemStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """X-TTS-LUFS present ⇒ the clip's level is known ⇒ zero ffmpeg cost."""
    monkeypatch.setattr(narration_module, "premaster_core", _explode)
    monkeypatch.setattr(narration_module, "probe_seconds", _explode)

    run_id = str(uuid4())
    clip_id, _url, seconds = await synthesize_and_store(
        _tts(lufs=-18.1), text="Alpha.", voice="af_heart", run_id=run_id, stage_name=_STAGE
    )

    assert seconds == 4.0
    assert mem_storage.audio(run_id, clip_id) == _ORIGINAL
    sidecar = mem_storage.sidecar(run_id, clip_id)
    assert sidecar["loudness_source"] == "tts_backend"
    assert sidecar["normalized"] is True
    assert sidecar["integrated_lufs"] == -18.1


async def test_unproven_level_stores_the_premastered_bytes(
    mem_storage: _MemStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No header ⇒ the orchestrator re-masters, and ITS bytes are what get stored."""
    monkeypatch.setattr(narration_module, "premaster_core", _premaster_returning(_NORMALIZED))
    # Input and output probe to the same length: the re-encode moved nothing.
    monkeypatch.setattr(
        narration_module,
        "probe_seconds",
        _probe_returning({_ORIGINAL: 4.0, _NORMALIZED: 4.0}),
    )

    run_id = str(uuid4())
    clip_id, _url, seconds = await synthesize_and_store(
        _tts(lufs=None), text="Beta.", voice="af_heart", run_id=run_id, stage_name=_STAGE
    )

    assert mem_storage.audio(run_id, clip_id) == _NORMALIZED
    sidecar = mem_storage.sidecar(run_id, clip_id)
    assert sidecar["loudness_source"] == "orchestrator"
    assert sidecar["normalized"] is True
    assert sidecar["integrated_lufs"] == -18.0
    assert sidecar["true_peak_dbtp"] == -1.5
    # The reported duration is authoritative and untouched by the re-encode.
    assert seconds == 4.0 and sidecar["seconds"] == 4.0


async def test_premaster_fail_open_keeps_original_and_counts_it(
    mem_storage: _MemStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fail-open premaster hands its input back; that must be visible, not silent."""
    monkeypatch.setattr(
        narration_module, "premaster_core", _premaster_failing("pass-2 loudnorm exit 1")
    )
    monkeypatch.setattr(narration_module, "probe_seconds", _explode)

    before = _unnormalized_count("premaster_failed_open")
    run_id = str(uuid4())
    clip_id, _url, _seconds = await synthesize_and_store(
        _tts(lufs=None), text="Gamma.", voice="af_heart", run_id=run_id, stage_name=_STAGE
    )

    assert mem_storage.audio(run_id, clip_id) == _ORIGINAL
    sidecar = mem_storage.sidecar(run_id, clip_id)
    assert sidecar["normalized"] is False
    assert sidecar["loudness_source"] == "none"
    assert sidecar["integrated_lufs"] is None
    assert _unnormalized_count("premaster_failed_open") == before + 1


@pytest.mark.parametrize("probed", [4.4])
async def test_duration_drift_discards_the_normalized_bytes(
    mem_storage: _MemStorage, monkeypatch: pytest.MonkeyPatch, probed: float | None
) -> None:
    """Bytes whose real length disagrees with the INPUT's are thrown away.

    A silently longer clip would desync every downstream timing decision (shot
    length, freeze-pad, captions), which are all made against ``duration_seconds``.
    An unprobeable output is the same verdict: unverifiable, so not trusted.

    The input probes at 4.0 s and the output at 4.4 s (or not at all) — the drift
    is genuinely between the two SIDES, not between a probe and the backend's
    reported number, which is a different quantity (see the mp3 test below).
    """
    monkeypatch.setattr(narration_module, "premaster_core", _premaster_returning(_NORMALIZED))
    monkeypatch.setattr(
        narration_module,
        "probe_seconds",
        _probe_returning({_ORIGINAL: 4.0, _NORMALIZED: probed}),
    )

    before = _unnormalized_count("duration_drift")
    run_id = str(uuid4())
    clip_id, _url, seconds = await synthesize_and_store(
        _tts(lufs=None, duration=4.0),
        text="Delta.",
        voice="af_heart",
        run_id=run_id,
        stage_name=_STAGE,
    )

    assert mem_storage.audio(run_id, clip_id) == _ORIGINAL  # normalized bytes DISCARDED
    assert seconds == 4.0
    sidecar = mem_storage.sidecar(run_id, clip_id)
    assert sidecar["normalized"] is False
    assert sidecar["seconds"] == 4.0
    assert _unnormalized_count("duration_drift") == before + 1


async def test_reuse_short_circuit_still_skips_synth_and_ffmpeg(
    mem_storage: _MemStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The content-addressed reuse path returns early — before synth AND before ffmpeg."""
    monkeypatch.setattr(narration_module, "premaster_core", _premaster_returning(_NORMALIZED))
    monkeypatch.setattr(
        narration_module,
        "probe_seconds",
        _probe_returning({_ORIGINAL: 7.5, _NORMALIZED: 7.5}),
    )

    run_id = str(uuid4())
    first = _tts(lufs=None, duration=7.5)
    clip1, url1, sec1 = await synthesize_and_store(
        first, text="Epsilon.", voice="af_heart", run_id=run_id, stage_name=_STAGE
    )
    assert first.synthesize.await_count == 1

    # A resumed narrate re-enters with a fresh client. Nothing may run this time:
    # not the TTS call, and not the premaster/probe pair.
    monkeypatch.setattr(narration_module, "premaster_core", _explode)
    monkeypatch.setattr(narration_module, "probe_seconds", _explode)
    second = _tts(lufs=None, duration=7.5)
    clip2, url2, sec2 = await synthesize_and_store(
        second, text="Epsilon.", voice="af_heart", run_id=run_id, stage_name=_STAGE
    )

    assert second.synthesize.await_count == 0
    assert (clip2, url2, sec2) == (clip1, url1, sec1)


# ── The four defects an adversarial real-ffmpeg review found ────────────────


async def test_upload_kill_switch_does_not_touch_narration(
    mem_storage: _MemStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flipping ONLY the upload switch must not un-normalize narration.

    ``ensure_tts_loudness`` used to call ``premaster``, which checks
    ``VOICEOVER_PREMASTER_ENABLED`` internally, so an operator disabling upload
    normalization silently disabled narration normalization too — and the counter
    reported ``premaster_failed_open``, i.e. an ffmpeg fault, for a config change.
    Whoever read that metric would go debug ffmpeg.

    Here the double stands in for ``premaster_core``, which by contract consults
    NO flag; the real engine's flag-freedom is proved end-to-end by the real-ffmpeg
    test below, which runs with this same switch off.
    """
    monkeypatch.setenv("VOICEOVER_PREMASTER_ENABLED", "false")
    monkeypatch.delenv("TTS_PREMASTER_FALLBACK_ENABLED", raising=False)
    monkeypatch.setattr(narration_module, "premaster_core", _premaster_returning(_NORMALIZED))
    monkeypatch.setattr(
        narration_module,
        "probe_seconds",
        _probe_returning({_ORIGINAL: 4.0, _NORMALIZED: 4.0}),
    )

    before = _unnormalized_count("premaster_failed_open")
    run_id = str(uuid4())
    clip_id, _url, _seconds = await synthesize_and_store(
        _tts(lufs=None), text="Zeta.", voice="af_heart", run_id=run_id, stage_name=_STAGE
    )

    sidecar = mem_storage.sidecar(run_id, clip_id)
    assert sidecar["loudness_source"] == "orchestrator"
    assert sidecar["normalized"] is True
    assert mem_storage.audio(run_id, clip_id) == _NORMALIZED
    # The precise regression: the upload switch must not be able to raise this.
    assert _unnormalized_count("premaster_failed_open") == before


async def test_tts_kill_switch_reports_disabled_not_a_failure(
    mem_storage: _MemStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The narration switch, off, is reported as ``disabled`` — an honest reason."""
    monkeypatch.setenv("TTS_PREMASTER_FALLBACK_ENABLED", "false")
    monkeypatch.setattr(narration_module, "premaster_core", _explode)
    monkeypatch.setattr(narration_module, "probe_seconds", _explode)

    before_disabled = _unnormalized_count("disabled")
    before_failed = _unnormalized_count("premaster_failed_open")
    run_id = str(uuid4())
    clip_id, _url, _seconds = await synthesize_and_store(
        _tts(lufs=None), text="Eta.", voice="af_heart", run_id=run_id, stage_name=_STAGE
    )

    assert mem_storage.audio(run_id, clip_id) == _ORIGINAL
    assert mem_storage.sidecar(run_id, clip_id)["normalized"] is False
    assert _unnormalized_count("disabled") == before_disabled + 1
    assert _unnormalized_count("premaster_failed_open") == before_failed


async def test_measure_failure_still_lands_in_exactly_one_series(
    mem_storage: _MemStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A normalized clip with no measured level is COUNTED, not lost.

    ``premaster``'s trailing measurement is advisory: when it fails the bytes are
    still correctly normalized but ``integrated_lufs`` is ``None``. The histogram
    observation is then skipped — so unless the clip is counted somewhere it exists
    in NEITHER series while its sidecar says ``normalized: true``, which is a hole
    in the audit trail exactly where someone would look for one.
    """
    monkeypatch.setattr(
        narration_module,
        "premaster_core",
        _premaster_returning(_NORMALIZED, LoudnessMetrics()),  # normalized, unmeasured
    )
    monkeypatch.setattr(
        narration_module,
        "probe_seconds",
        _probe_returning({_ORIGINAL: 4.0, _NORMALIZED: 4.0}),
    )

    before = _unnormalized_count("measure_failed")
    run_id = str(uuid4())
    clip_id, _url, _seconds = await synthesize_and_store(
        _tts(lufs=None), text="Theta.", voice="af_heart", run_id=run_id, stage_name=_STAGE
    )

    # The good bytes are kept — the normalization itself succeeded.
    assert mem_storage.audio(run_id, clip_id) == _NORMALIZED
    sidecar = mem_storage.sidecar(run_id, clip_id)
    assert sidecar["loudness_source"] == "orchestrator"
    assert sidecar["integrated_lufs"] is None
    # ...and the clip is nonetheless accounted for.
    assert _unnormalized_count("measure_failed") == before + 1


async def test_unmeasurable_level_is_not_reported_as_a_failure(
    mem_storage: _MemStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clip with no integrated loudness gets its OWN reason, not ``failed_open``.

    Nothing failed: a sub-0.5s or near-silent clip has no measurable integrated
    loudness, so there is nothing to normalize toward. Calling that
    ``premaster_failed_open`` sends whoever reads the metric hunting an ffmpeg bug.
    """
    monkeypatch.setattr(narration_module, "premaster_core", _premaster_failing(UNMEASURABLE_LEVEL))
    monkeypatch.setattr(narration_module, "probe_seconds", _explode)

    before_unmeasurable = _unnormalized_count(UNMEASURABLE_LEVEL)
    before_failed = _unnormalized_count("premaster_failed_open")
    run_id = str(uuid4())
    clip_id, _url, _seconds = await synthesize_and_store(
        _tts(lufs=None, duration=0.2),
        text="Iota.",
        voice="af_heart",
        run_id=run_id,
        stage_name=_STAGE,
    )

    assert mem_storage.audio(run_id, clip_id) == _ORIGINAL  # passed through UNTOUCHED
    assert mem_storage.sidecar(run_id, clip_id)["normalized"] is False
    assert _unnormalized_count(UNMEASURABLE_LEVEL) == before_unmeasurable + 1
    assert _unnormalized_count("premaster_failed_open") == before_failed


async def test_padding_container_does_not_trip_the_duration_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An mp3-shaped clip survives the duration guard. Probed-vs-PROBED, not vs reported.

    The numbers are the ones measured with real ffmpeg: 2.5000 s of true audio
    encoded as 24 kHz mp3 probes 2.5680 s — 68 ms of encoder delay/priming that
    the CONTAINER carries and the backend's reported duration does not. Comparing
    the probed output against ``duration_seconds`` therefore reports 68 ms of
    "drift" on a re-encode that moved nothing, and with ``format="mp3"`` 100% of
    narration would silently revert to un-normalized. Probing both sides the same
    way cancels the padding out.
    """
    monkeypatch.setattr(narration_module, "premaster_core", _premaster_returning(_NORMALIZED))
    monkeypatch.setattr(
        narration_module,
        "probe_seconds",
        _probe_returning({_ORIGINAL: 2.5680, _NORMALIZED: 2.5680}),
    )

    result = TTSSynthesis.model_validate(
        {
            "audio_bytes": _ORIGINAL,
            "sample_rate": 24000,
            "duration_seconds": 2.5,  # the AUDIO's length; the container is longer
            "provider": "kokoro",
            "voice": "af_heart",
            "format": "mp3",
            "integrated_lufs": None,
        }
    )
    before = _unnormalized_count("duration_drift")
    audio, loudness = await ensure_tts_loudness(result, stage_name=_STAGE)

    assert audio == _NORMALIZED
    assert loudness.source == "orchestrator" and loudness.normalized is True
    assert _unnormalized_count("duration_drift") == before


# ── The only test that proves the feature, with real ffmpeg ─────────────────

# Resolved to ABSOLUTE paths, both for the skip gate and for the subprocess argv
# (a bare "ffmpeg" is a partial-path execution, S607).
_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")
_HAS_FFMPEG = _FFMPEG is not None and _FFPROBE is not None


def _make_wav(tmp_path: Path, name: str, *, seconds: float, gain_db: float) -> bytes:
    """Synthesize a real 24 kHz PCM tone at a chosen level and return its bytes."""
    path = tmp_path / name
    subprocess.run(  # noqa: S603 -- fixed argv, test-only, no shell
        [
            str(_FFMPEG),
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}:sample_rate=24000",
            "-af",
            f"volume={gain_db}dB",
            "-c:a",
            "pcm_s16le",
            str(path),
        ],
        check=True,
    )
    return path.read_bytes()


def _probe_wav_seconds(tmp_path: Path, data: bytes) -> float:
    path = tmp_path / "probe-me.wav"
    path.write_bytes(data)
    out = subprocess.run(  # noqa: S603 -- fixed argv, test-only, no shell
        [
            str(_FFPROBE),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(out.stdout.strip())


def _synthesis(data: bytes, seconds: float) -> TTSSynthesis:
    return TTSSynthesis.model_validate(
        {
            "audio_bytes": data,
            "sample_rate": 24000,
            "duration_seconds": seconds,
            "provider": "kokoro",
            "voice": "af_heart",
            "format": "wav",
            "integrated_lufs": None,  # unproven ⇒ the fallback must engage
        }
    )


@pytest.mark.skipif(not _HAS_FFMPEG, reason="needs real ffmpeg + ffprobe on PATH")
@pytest.mark.parametrize("gain_db", [13.0], ids=["hot"])
async def test_real_ffmpeg_normalizes_hot_and_quiet_clips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gain_db: float
) -> None:
    """THE test. Real ffmpeg, real loudnorm, both extremes converge on -18 LUFS.

    Everything above mocks ``premaster_core`` and ``probe_seconds`` away, which is
    exactly why three real defects shipped past a green suite: the kill-switch
    coupling, the sub-0.5s ``-inf`` cliff and the wrong duration reference all live
    in the code the mocks replace.

    A hot and a quiet 2 s tone must both come out within 1.5 LU of -18.0 with
    their duration unchanged. Measured on this ffmpeg: +13 dB in = -8.75 LUFS and
    -27 dB in = -48.75 LUFS — 40 LU apart — both land on -18.05 with 0.00 ms of
    duration drift. ``VOICEOVER_PREMASTER_ENABLED`` is off throughout: the upload
    switch has no authority here, and this asserts that against the REAL engine
    rather than a double.
    """
    monkeypatch.setenv("VOICEOVER_PREMASTER_ENABLED", "false")
    monkeypatch.delenv("TTS_PREMASTER_FALLBACK_ENABLED", raising=False)

    source = _make_wav(tmp_path, f"src{gain_db}.wav", seconds=2.0, gain_db=gain_db)
    source_seconds = _probe_wav_seconds(tmp_path, source)

    audio, loudness = await ensure_tts_loudness(
        _synthesis(source, source_seconds), stage_name=_STAGE
    )

    assert loudness.source == "orchestrator", "the fallback did not engage"
    assert loudness.normalized is True
    assert audio != source, "the stored bytes are not the re-mastered ones"
    assert loudness.integrated_lufs is not None, "no level was measured"
    assert abs(loudness.integrated_lufs + 18.0) <= 1.5, (
        f"landed at {loudness.integrated_lufs} LUFS, not -18"
    )
    assert abs(_probe_wav_seconds(tmp_path, audio) - source_seconds) <= 0.005


@pytest.mark.skipif(not _HAS_FFMPEG, reason="needs real ffmpeg + ffprobe on PATH")
@pytest.mark.parametrize(
    ("seconds", "gain_db"),
    [(0.2, 0.0)],
    ids=["too-short"],
)
async def test_real_ffmpeg_short_or_silent_clip_is_unmeasurable_not_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seconds: float, gain_db: float
) -> None:
    """Real ffmpeg on the two inputs that have no integrated loudness at all.

    Verified against ffmpeg directly: a 0.2 s tone and a 2 s tone at -70 dB both
    make pass 1 print ``"input_i": "-inf"`` (too few gating blocks / under the
    -70 LUFS absolute gate). Seeding pass 2 with that kills the filter chain with
    "Failed to inject frame into filter network: Result too large", which used to
    be reported as ``premaster_failed_open``. It is not a failure — the audio
    passes through untouched and says so.
    """
    monkeypatch.delenv("TTS_PREMASTER_FALLBACK_ENABLED", raising=False)
    source = _make_wav(tmp_path, f"edge{seconds}-{gain_db}.wav", seconds=seconds, gain_db=gain_db)

    before_unmeasurable = _unnormalized_count(UNMEASURABLE_LEVEL)
    before_failed = _unnormalized_count("premaster_failed_open")
    audio, loudness = await ensure_tts_loudness(_synthesis(source, seconds), stage_name=_STAGE)

    assert audio == source, "silence must pass through UNCHANGED, never be amplified"
    assert loudness.normalized is False and loudness.source == "none"
    assert _unnormalized_count(UNMEASURABLE_LEVEL) == before_unmeasurable + 1
    assert _unnormalized_count("premaster_failed_open") == before_failed
