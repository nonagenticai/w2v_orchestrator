"""``store_tts_clip`` sidecar provenance — WHO spoke the clip, and how fast.

Every TTS clip persists its measured ``seconds`` but used to drop the voice,
provider and requested speed, so the system could not answer "how fast does
voice X actually talk?" from its own history — the reason a hardcoded wpm
constant exists elsewhere. These tests pin the additive contract:

* the four provenance keys land in the ``{clip_id}.json`` sidecar when given;
* a value left ``None`` is OMITTED, never written as a JSON ``null`` a reader
  could mistake for a measured zero;
* ``natural_seconds`` (= ``seconds * speed``, the duration at speed 1.0) is
  derived only for a positive speed;
* an OLD sidecar carrying none of these keys still parses through
  ``find_existing_clip``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from src.services.tts_storage import (
    ClipLoudness,
    find_existing_clip,
    set_storage_for_tests,
    store_tts_clip,
)

_RUN = "run-prov-1"
_STAGE = "demo_video_narrate"
_CLIP = "line-7"


class _MemStorage:
    """In-memory stand-in for ``StorageService`` (per-run stage-artifact API)."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def initialize(self) -> None: ...

    async def store_stage_artifact(
        self, job_id: str, stage_name: str, data: bytes, filename: str
    ) -> str:
        key = f"artifacts/{job_id}/{stage_name}/{filename}"
        self.objects[key] = data
        return key

    async def stage_artifact_url(self, job_id: str, stage_name: str, filename: str) -> str:
        return f"mem://own/artifacts/{job_id}/{stage_name}/{filename}"

    async def retrieve_stage_artifact(
        self, job_id: str, stage_name: str, filename: str
    ) -> bytes | None:
        return self.objects.get(f"artifacts/{job_id}/{stage_name}/{filename}")

    async def get_presigned_url(
        self, path: str, expires: int | None = None, bucket: str | None = None
    ) -> str | None:
        return f"mem://own/{path}"


@pytest.fixture
def storage() -> Iterator[_MemStorage]:
    mem = _MemStorage()
    set_storage_for_tests(mem)
    yield mem
    set_storage_for_tests(None)


def _sidecar(mem: _MemStorage) -> dict[str, object]:
    raw = mem.objects[f"artifacts/{_RUN}/{_STAGE}/{_CLIP}.json"]
    loaded = json.loads(raw)
    assert isinstance(loaded, dict)
    return loaded


@pytest.mark.asyncio
async def test_provenance_keys_are_persisted_into_the_sidecar(storage: _MemStorage) -> None:
    """voice / provider / speed / word_count reach the sidecar verbatim."""
    await store_tts_clip(
        _RUN,
        _STAGE,
        _CLIP,
        b"RIFFfake",
        seconds=8.0,
        voice="af_heart",
        provider="kokoro",
        speed=1.25,
        word_count=25,
    )

    payload = _sidecar(storage)
    assert payload["voice"] == "af_heart"
    assert payload["provider"] == "kokoro"
    assert payload["speed"] == 1.25
    assert payload["word_count"] == 25
    # the pre-existing keys are untouched
    assert payload["seconds"] == 8.0
    assert payload["format"] == "wav"


@pytest.mark.asyncio
async def test_natural_seconds_is_the_duration_at_speed_one(storage: _MemStorage) -> None:
    """``natural_seconds`` = seconds * speed — what the clip WOULD have lasted at 1.0x.

    A clip rendered at 1.25x is not evidence the voice talks 25% faster, so the
    rate-bearing quantity is the de-sped duration, not the wall duration.
    """
    await store_tts_clip(_RUN, _STAGE, _CLIP, b"RIFFfake", seconds=8.0, speed=1.25)

    payload = _sidecar(storage)
    assert payload["natural_seconds"] == pytest.approx(10.0)
    # and it is DERIVED, not a copy of either input
    assert payload["natural_seconds"] != payload["seconds"]


@pytest.mark.asyncio
@pytest.mark.parametrize("speed", [None])
async def test_natural_seconds_absent_without_a_positive_speed(
    storage: _MemStorage, speed: float | None
) -> None:
    """No speed (or a nonsensical one) ⇒ no derived duration, rather than a wrong one."""
    await store_tts_clip(_RUN, _STAGE, _CLIP, b"RIFFfake", seconds=8.0, speed=speed)

    assert "natural_seconds" not in _sidecar(storage)


@pytest.mark.asyncio
async def test_unset_provenance_is_omitted_not_written_as_null(storage: _MemStorage) -> None:
    """An omitted value must not land as ``null`` — a reader could read it as measured.

    ``word_count: null`` is indistinguishable from ``word_count: 0`` ("no words")
    to a lenient reader, so absence has to be encoded as ABSENCE.
    """
    await store_tts_clip(
        _RUN, _STAGE, _CLIP, b"RIFFfake", seconds=8.0, voice="af_heart", speed=None
    )

    payload = _sidecar(storage)
    assert payload["voice"] == "af_heart"
    for key in ("provider", "speed", "word_count", "natural_seconds"):
        assert key not in payload, f"{key} was written despite being unset"


@pytest.mark.asyncio
async def test_existing_callers_are_byte_identical(storage: _MemStorage) -> None:
    """A caller passing none of the new params writes exactly the old sidecar."""
    await store_tts_clip(
        _RUN,
        _STAGE,
        _CLIP,
        b"RIFFfake",
        seconds=3.5,
        loudness=ClipLoudness(
            integrated_lufs=-18.0, true_peak_dbtp=-1.5, source="tts_backend", normalized=True
        ),
        word_timings=[{"word": "hi", "start": 0.0, "end": 0.4}],
    )

    assert _sidecar(storage) == {
        "seconds": 3.5,
        "format": "wav",
        "integrated_lufs": -18.0,
        "true_peak_dbtp": -1.5,
        "loudness_source": "tts_backend",
        "normalized": True,
        "word_timings": [{"word": "hi", "start": 0.0, "end": 0.4}],
    }


@pytest.mark.asyncio
async def test_old_sidecar_without_provenance_still_resumes(storage: _MemStorage) -> None:
    """A sidecar written before these keys existed still short-circuits a narrate."""
    storage.objects[f"artifacts/{_RUN}/{_STAGE}/{_CLIP}.json"] = json.dumps(
        {"seconds": 4.25, "format": "wav"}
    ).encode("utf-8")

    found = await find_existing_clip(_RUN, _STAGE, _CLIP)

    assert found is not None
    url, seconds = found
    assert seconds == 4.25
    assert url.endswith(f"{_CLIP}.wav")


@pytest.mark.asyncio
async def test_new_sidecar_round_trips_through_find_existing_clip(storage: _MemStorage) -> None:
    """The provenance keys do not disturb the resume probe's duration read."""
    await store_tts_clip(
        _RUN,
        _STAGE,
        _CLIP,
        b"RIFFfake",
        seconds=4.25,
        voice="af_heart",
        provider="kokoro",
        speed=1.1,
        word_count=11,
    )

    found = await find_existing_clip(_RUN, _STAGE, _CLIP)

    assert found is not None
    assert found[1] == 4.25
