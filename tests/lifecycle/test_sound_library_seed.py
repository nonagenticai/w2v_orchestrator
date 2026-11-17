"""Startup self-heal of the shared sound library (bucket wiped 2026-10-14)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from src.lifecycle import sound_library_seed
from src.services.stages.sound_library_sync import (
    DEFAULT_SOUND_SRC,
    SHARED_MANIFEST_KEY,
    seed_shared_library_if_missing,
)


class _FakeStorage:
    """In-memory own-bucket: ``object_size`` is None only for a genuinely absent key."""

    def __init__(self, existing: dict[str, bytes] | None = None, probe_error: bool = False) -> None:
        self.objects: dict[str, bytes] = dict(existing or {})
        self.puts: list[str] = []
        self.deletes: list[str] = []
        self.probe_error = probe_error
        self.closed = False

    async def initialize(self) -> None:
        pass

    async def close(self) -> None:
        self.closed = True

    async def object_size(self, key: str) -> int | None:
        if self.probe_error:
            raise ConnectionError("minio unreachable")
        data = self.objects.get(key)
        return None if data is None else len(data)

    async def put_object(
        self, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> str:
        self.objects[key] = data
        self.puts.append(key)
        return key

    async def delete_object(self, key: str) -> bool:  # pragma: no cover - must never be called
        self.deletes.append(key)
        return True


def _tree(root: Path) -> Path:
    (root / "accents").mkdir(parents=True)
    (root / "accents" / "foo.mp3").write_bytes(b"ID3fakeaudio")
    return root


def test_bundled_library_is_where_the_image_copies_it() -> None:
    # Dockerfile: COPY assets/ ./assets/ — the seed reads from this path in-pod.
    assert DEFAULT_SOUND_SRC.is_dir()
    assert any(DEFAULT_SOUND_SRC.rglob("*.mp3")) or any(DEFAULT_SOUND_SRC.rglob("*.wav"))
    dockerfile = (Path(__file__).resolve().parents[2] / "Dockerfile").read_text()
    assert "COPY --chown=orchestrator:orchestrator assets/ ./assets/" in dockerfile


async def test_seeds_when_manifest_absent(tmp_path: Path) -> None:
    store = _FakeStorage()
    assert await seed_shared_library_if_missing(store, _tree(tmp_path)) is True
    assert "sound_library/shared/accents/foo.mp3" in store.objects
    manifest = json.loads(store.objects[SHARED_MANIFEST_KEY])
    assert manifest["foo"]["key"] == "sound_library/shared/accents/foo.mp3"
    assert store.deletes == []


async def test_noop_when_manifest_present(tmp_path: Path) -> None:
    store = _FakeStorage({SHARED_MANIFEST_KEY: b'{"custom": {}}'})
    assert await seed_shared_library_if_missing(store, _tree(tmp_path)) is False
    assert store.puts == []
    assert store.objects[SHARED_MANIFEST_KEY] == b'{"custom": {}}'


async def test_idempotent_second_call_is_noop(tmp_path: Path) -> None:
    store = _FakeStorage()
    src = _tree(tmp_path)
    assert await seed_shared_library_if_missing(store, src) is True
    n = len(store.puts)
    assert await seed_shared_library_if_missing(store, src) is False
    assert len(store.puts) == n


async def test_probe_error_is_not_read_as_absent(tmp_path: Path) -> None:
    store = _FakeStorage(probe_error=True)
    with pytest.raises(ConnectionError):
        await seed_shared_library_if_missing(store, _tree(tmp_path))
    assert store.puts == []


async def test_empty_source_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="seed"):
        await seed_shared_library_if_missing(_FakeStorage(), tmp_path)


async def test_startup_task_swallows_failure_and_closes(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    store = _FakeStorage(probe_error=True)
    monkeypatch.setattr(sound_library_seed, "StorageService", lambda: store)
    with caplog.at_level(logging.WARNING, logger=sound_library_seed.__name__):
        await sound_library_seed._seed_shared_sound_library()  # must not raise
    assert "startup seed failed" in caplog.text
    assert store.closed


async def test_startup_task_seeds_and_clears_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _FakeStorage()
    cleared: list[bool] = []
    monkeypatch.setattr(sound_library_seed, "StorageService", lambda: store)
    monkeypatch.setattr(
        sound_library_seed, "clear_sound_library_cache", lambda: cleared.append(True)
    )
    await sound_library_seed._seed_shared_sound_library()
    assert SHARED_MANIFEST_KEY in store.objects
    assert cleared == [True]
    assert store.closed
