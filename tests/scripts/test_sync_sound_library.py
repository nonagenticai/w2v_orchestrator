"""sync_sound_library — uploads a sound dir + regenerates the manifest via StorageService."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.sync_sound_library import sync


class _CapturingStorage:
    """Duck-typed StorageService capturing every ``put_object``."""

    def __init__(self) -> None:
        self.puts: dict[str, tuple[bytes, str]] = {}
        self.closed = False

    async def initialize(self) -> None:  # pragma: no cover - injected, not called
        pass

    async def put_object(
        self, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> str:
        self.puts[key] = (data, content_type)
        return key

    async def close(self) -> None:  # pragma: no cover - injected, must NOT be called
        self.closed = True


def _seed_tree(root: Path) -> None:
    (root / "accents").mkdir(parents=True)
    (root / "music").mkdir(parents=True)
    (root / "accents" / "foo.mp3").write_bytes(b"ID3fakeaudio")
    (root / "music" / "bar.mp3").write_bytes(b"ID3fakeaudio")


async def test_sync_uploads_assets_and_writes_manifest(tmp_path: Path) -> None:
    _seed_tree(tmp_path)
    storage = _CapturingStorage()

    rc = await sync(tmp_path, "shared", None, storage=storage)

    assert rc == 0
    assert "sound_library/shared/accents/foo.mp3" in storage.puts
    assert "sound_library/shared/music/bar.mp3" in storage.puts
    assert storage.puts["sound_library/shared/accents/foo.mp3"][1] == "audio/mpeg"

    manifest_bytes, manifest_ct = storage.puts["sound_library/shared/manifest.json"]
    assert manifest_ct == "application/json"
    manifest = json.loads(manifest_bytes)
    assert manifest["foo"]["category"] == "accent"
    assert manifest["foo"]["key"] == "sound_library/shared/accents/foo.mp3"
    assert manifest["bar"]["category"] == "music"


async def test_description_precedence(tmp_path: Path) -> None:
    # click_ui is a built-in id; foo is not. Override wins over both.
    (tmp_path / "accents").mkdir(parents=True)
    (tmp_path / "accents" / "click_ui.mp3").write_bytes(b"ID3fakeaudio")  # built-in id
    (tmp_path / "accents" / "foo.mp3").write_bytes(b"ID3fakeaudio")  # unknown id
    (tmp_path / "accents" / "ding_positive.mp3").write_bytes(
        b"ID3fakeaudio"
    )  # built-in, overridden
    (tmp_path / "descriptions.json").write_text(json.dumps({"ding_positive": "override wins"}))
    storage = _CapturingStorage()

    await sync(tmp_path, "shared", None, storage=storage)

    manifest = json.loads(storage.puts["sound_library/shared/manifest.json"][0])
    # built-in id, no override → curated catalog text
    assert manifest["click_ui"]["description"] == "crisp UI click for feature reveals"
    # unknown id, no override → humanised filename
    assert manifest["foo"]["description"] == "Foo"
    # explicit override beats the built-in text
    assert manifest["ding_positive"]["description"] == "override wins"


async def test_injected_storage_not_closed(tmp_path: Path) -> None:
    _seed_tree(tmp_path)
    storage = _CapturingStorage()

    await sync(tmp_path, "shared", None, storage=storage)

    # The tool closes only storage it created; an injected one is the caller's.
    assert storage.closed is False


async def test_sync_tenant_scope_is_refused_and_writes_nothing(tmp_path: Path) -> None:
    """P6 ORCH-4: the tenant manifest is derived from ``tenant_sound_assets``; a CLI
    write there would be overwritten on the next upload, so the scope is retired."""
    _seed_tree(tmp_path)
    storage = _CapturingStorage()

    with pytest.raises(SystemExit, match="/api/tenant/music"):
        await sync(tmp_path, "tenant", "tenant-xyz", storage=storage)
    assert storage.puts == {}

    # Control: the same tree under the shared scope does write.
    assert await sync(tmp_path, "shared", None, storage=storage) == 0
    assert storage.puts


async def test_shared_sync_skips_the_reserved_tenant_prefix(tmp_path: Path) -> None:
    _seed_tree(tmp_path)
    (tmp_path / "music" / "tenant_x.mp3").write_bytes(b"ID3fakeaudio")
    storage = _CapturingStorage()

    assert await sync(tmp_path, "shared", None, storage=storage) == 0
    manifest = json.loads(storage.puts["sound_library/shared/manifest.json"][0])
    assert "tenant_x" not in manifest
    assert "bar" in manifest


async def test_sync_empty_dir_returns_nonzero(tmp_path: Path) -> None:
    storage = _CapturingStorage()
    rc = await sync(tmp_path, "shared", None, storage=storage)
    assert rc == 1
    assert storage.puts == {}
