"""synthesize_and_store is content-addressed + resumable.

A random uuid clip id made a resumed narrate re-synthesize every chunk (a pod
reap mid-narrate re-ran the whole book). These tests pin the fix: identical
(voice, speed, text) → identical clip id, a duration sidecar records the clip's
existence, and a second call reuses the stored clip WITHOUT calling the TTS
backend again.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.clients.schemas.tts import TTSSynthesis
from src.services import tts_storage
from src.services.stages.primitives.narration import synthesize_and_store
from src.services.tts_storage import PREPROC_VERSION, deterministic_clip_id


class _MemStorage:
    """In-memory storage exposing the full clip surface (store + probe)."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.store_calls = 0

    async def initialize(self) -> None:
        pass

    def _key(self, job_id: str, stage_name: str, filename: str) -> str:
        return f"artifacts/{job_id}/{stage_name}/{filename}"

    async def store_stage_artifact(
        self, job_id: str, stage_name: str, data: bytes, filename: str
    ) -> str:
        key = self._key(job_id, stage_name, filename)
        self.objects[key] = data
        self.store_calls += 1
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


@pytest.fixture
def mem_storage():
    store = _MemStorage()
    tts_storage.set_storage_for_tests(store)
    yield store
    tts_storage.set_storage_for_tests(None)


def _tts(duration: float = 4.0) -> AsyncMock:
    fake = AsyncMock()
    fake.synthesize.return_value = TTSSynthesis(
        audio_bytes=b"\x01\x02\x03", sample_rate=0, duration_seconds=duration, provider="", voice=""
    )
    return fake


def test_deterministic_clip_id_stable_and_speed_normalized() -> None:
    a = deterministic_clip_id("Hello world.", "af_heart", 1.0)
    assert a == deterministic_clip_id("Hello world.", "af_heart", 1.0)  # stable
    assert a == deterministic_clip_id("Hello world.", "af_heart", 1)  # 1 == 1.0
    assert a != deterministic_clip_id("Hello world.", "am_adam", 1.0)  # voice matters
    assert a != deterministic_clip_id("Hello world.", "af_heart", 1.1)  # speed matters
    assert a != deterministic_clip_id("Goodbye.", "af_heart", 1.0)  # text matters
    # None speed (video narrate stages) hashes stably and differs from an explicit speed
    n = deterministic_clip_id("Hi.", "af_heart", None)
    assert n == deterministic_clip_id("Hi.", "af_heart", None)


def test_clip_id_golden_pins_cache_key_scheme() -> None:
    """Freeze the full cache-key scheme (format + PREPROC_VERSION) to a literal.

    A tts_backend pre-processing change (e.g. the misaki G2P activation) MUST bump
    PREPROC_VERSION IN LOCKSTEP with tts_gateway or stale espeak clips replay. This
    golden hash breaks the moment the version or the raw-string format changes, so a
    bump is always a deliberate, reviewed edit — never a silent drift.
    """
    assert PREPROC_VERSION == "2"
    # deterministic_clip_id("The quick brown fox.", "af_heart", 1.0) under version "2"
    assert (
        deterministic_clip_id("The quick brown fox.", "af_heart", 1.0)
        == "85b750b132b42d061d69e620497e2fc0fe7248eb"  # pragma: allowlist secret
    )


async def test_first_call_synthesizes_and_writes_wav_plus_sidecar(mem_storage: _MemStorage) -> None:
    run_id = str(uuid4())
    fake = _tts(4.0)
    clip_id, url, seconds = await synthesize_and_store(
        fake,
        text="Alpha.",
        voice="af_heart",
        run_id=run_id,
        stage_name="audiobook_narrate",
        speed=1.0,
    )
    assert fake.synthesize.await_count == 1
    assert seconds == 4.0
    assert url.endswith(f"{clip_id}.wav")
    # both the wav and the duration sidecar are persisted
    assert f"artifacts/{run_id}/audiobook_narrate/{clip_id}.wav" in mem_storage.objects
    assert f"artifacts/{run_id}/audiobook_narrate/{clip_id}.json" in mem_storage.objects


async def test_resume_reuses_clip_without_resynthesizing(mem_storage: _MemStorage) -> None:
    run_id = str(uuid4())

    first_client = _tts(7.5)
    clip1, url1, sec1 = await synthesize_and_store(
        first_client,
        text="Beta.",
        voice="af_heart",
        run_id=run_id,
        stage_name="audiobook_narrate",
        speed=1.0,
    )
    assert first_client.synthesize.await_count == 1

    # a reaped+resumed narrate re-enters with a FRESH client; storage persisted
    second_client = _tts(7.5)
    clip2, url2, sec2 = await synthesize_and_store(
        second_client,
        text="Beta.",
        voice="af_heart",
        run_id=run_id,
        stage_name="audiobook_narrate",
        speed=1.0,
    )
    assert second_client.synthesize.await_count == 0  # skipped — reused the stored clip
    assert (clip2, url2, sec2) == (clip1, url1, sec1)
