"""sound_catalog — built-in baseline + the data-driven, tenant-scoped loader."""

from __future__ import annotations

import json
import logging

import pytest

from src.services.stages import sound_catalog as _sound_catalog
from src.services.stages.sound_catalog import (
    DEFAULT_AMBIENT_ID,
    DEFAULT_TRANSITION_ID,
    SHARED_PREFIX,
    SoundLibrary,
    clear_sound_library_cache,
    load_sound_library,
    shared_key,
    tenant_prefix,
)


class _FakeStorage:
    """Duck-typed StorageService: only ``get_object_bytes`` is exercised here."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self._objects = objects

    async def get_object_bytes(self, key: str) -> bytes | None:
        return self._objects.get(key)


class _ExplodingStorage:
    """A storage whose read RAISES — the genuinely unexpected failure.

    Distinct from ``_FakeStorage({})``, which returns ``None``: that is what the
    real facade does for an absent object (it never raises for not-found), so
    the two fixtures are the two sides of the DEBUG-vs-WARNING split below.
    """

    async def get_object_bytes(self, key: str) -> bytes | None:
        raise RuntimeError("minio client not initialized")


def _manifest(entries: dict[str, dict[str, object]]) -> bytes:
    return json.dumps(entries).encode("utf-8")


#: Derived from the module rather than hardcoded, so a move cannot leave the
#: caplog filters silently matching nothing (which would pass vacuously).
_CATALOG_LOGGER = _sound_catalog.__name__


# --- built-in baseline ---


def test_builtin_categories_are_populated() -> None:
    lib = SoundLibrary.builtin()
    for category in ("transition", "accent", "music", "ambient"):
        assert lib.ids_in(category), f"no ids in category {category}"


def test_is_valid_enforces_category() -> None:
    lib = SoundLibrary.builtin()
    assert lib.is_valid("click_ui")
    assert lib.is_valid("click_ui", category="accent")
    assert not lib.is_valid("click_ui", category="music")
    assert not lib.is_valid("not_a_real_id")


def test_defaults_are_in_baseline_and_correct_category() -> None:
    lib = SoundLibrary.builtin()
    assert lib.is_valid(DEFAULT_TRANSITION_ID, category="transition")
    assert lib.is_valid(DEFAULT_AMBIENT_ID, category="ambient")


def test_key_for_points_at_shared_prefix() -> None:
    lib = SoundLibrary.builtin()
    assert lib.key_for("whoosh_soft") == shared_key("transitions/whoosh_soft.mp3")
    key = lib.key_for("whoosh_soft")
    assert key is not None and key.startswith(f"{SHARED_PREFIX}/")
    assert lib.key_for("missing") is None


def test_accent_prompt_block_lists_only_accents() -> None:
    block = SoundLibrary.builtin().accent_prompt_block()
    assert "click_ui" in block
    # Music/ambient/transition ids must not leak into the accent-picker prompt.
    assert "corporate_uplift" not in block
    assert "whoosh_soft" not in block


# --- data-driven loader ---


async def test_shared_manifest_extends_builtin() -> None:
    clear_sound_library_cache()
    storage = _FakeStorage(
        {
            f"{SHARED_PREFIX}/manifest.json": _manifest(
                {
                    "brand_whoosh": {
                        "key": shared_key("transitions/brand_whoosh.mp3"),
                        "description": "brand whoosh",
                        "category": "transition",
                        "duration_s": 0.5,
                    }
                }
            )
        }
    )
    lib = await load_sound_library(storage, tenant_id="", use_cache=False)
    assert lib.is_valid("brand_whoosh", category="transition")  # from manifest
    assert lib.is_valid("whoosh_soft")  # built-in floor still present


async def test_tenant_manifest_layers_over_shared() -> None:
    clear_sound_library_cache()
    storage = _FakeStorage(
        {
            f"{tenant_prefix('t1')}/manifest.json": _manifest(
                {
                    "tenant_sting": {
                        "key": f"{tenant_prefix('t1')}/accent/a.mp3",
                        "description": "tenant sting",
                        "category": "accent",
                        "duration_s": 0.4,
                    }
                }
            )
        }
    )
    shared_lib = await load_sound_library(storage, tenant_id="", use_cache=False)
    assert not shared_lib.is_valid("tenant_sting")  # tenant id invisible to other tenants
    tenant_lib = await load_sound_library(storage, tenant_id="t1", use_cache=False)
    assert tenant_lib.is_valid("tenant_sting", category="accent")
    asset = tenant_lib.get("tenant_sting")
    assert asset is not None and asset.source == "tenant"
    builtin = tenant_lib.get("whoosh_soft")
    assert builtin is not None and builtin.source == "builtin"


async def test_tenant_layer_is_namespace_and_key_confined() -> None:
    """A tenant manifest can only ADD ``tenant_`` ids pointing under that tenant's
    own prefix: an entry aimed at another tenant's object, at the shared library, or
    trying to shadow a bundled id is dropped (the good entry is the control)."""
    clear_sound_library_cache()
    t1 = tenant_prefix("t1")
    storage = _FakeStorage(
        {
            f"{t1}/manifest.json": _manifest(
                {
                    "tenant_good": {"key": f"{t1}/music/g.mp3", "category": "music"},
                    "tenant_steal": {
                        "key": f"{tenant_prefix('t2')}/music/x.mp3",
                        "category": "music",
                    },
                    "tenant_lookalike": {"key": f"{t1}x/music/x.mp3", "category": "music"},
                    "tenant_shared": {"key": shared_key("music/x.mp3"), "category": "music"},
                    "corporate_uplift": {"key": f"{t1}/music/c.mp3", "category": "music"},
                }
            ),
            f"{SHARED_PREFIX}/manifest.json": _manifest(
                {"tenant_fake": {"key": shared_key("music/f.mp3"), "category": "music"}}
            ),
        }
    )
    lib = await load_sound_library(storage, tenant_id="t1", use_cache=False)
    assert lib.is_valid("tenant_good", category="music")
    for sid in ("tenant_steal", "tenant_lookalike", "tenant_shared", "tenant_fake"):
        assert lib.get(sid) is None, sid
    assert lib.key_for("corporate_uplift") == shared_key("music/corporate_uplift.mp3")


async def test_retired_tenant_track_resolves_but_is_never_offered() -> None:
    clear_sound_library_cache()
    t1 = tenant_prefix("t1")
    storage = _FakeStorage(
        {
            f"{t1}/manifest.json": _manifest(
                {
                    "tenant_old": {
                        "key": f"{t1}/music/o.mp3",
                        "category": "music",
                        "retired": True,
                    },
                    "tenant_new": {"key": f"{t1}/music/n.mp3", "category": "music"},
                }
            )
        }
    )
    lib = await load_sound_library(storage, tenant_id="t1", use_cache=False)
    assert "tenant_old" not in lib.ids_in("music")
    assert not lib.is_valid("tenant_old", category="music")
    assert lib.resolves("tenant_old", category="music")
    assert lib.key_for("tenant_old") == f"{t1}/music/o.mp3"
    assert "tenant_new" in lib.ids_in("music") and lib.is_valid("tenant_new")


async def test_malformed_manifest_is_non_fatal() -> None:
    clear_sound_library_cache()
    storage = _FakeStorage({f"{SHARED_PREFIX}/manifest.json": b"{ not valid json"})
    lib = await load_sound_library(storage, tenant_id="", use_cache=False)
    assert lib.is_valid("whoosh_soft")  # degrades to the built-in floor, never errors


async def test_bad_category_entry_skipped() -> None:
    clear_sound_library_cache()
    storage = _FakeStorage(
        {
            f"{SHARED_PREFIX}/manifest.json": _manifest(
                {
                    "weird": {"key": "k", "category": "not_a_category", "duration_s": 1.0},
                    "good": {
                        "key": shared_key("accents/good.mp3"),
                        "category": "accent",
                        "duration_s": 0.2,
                    },
                }
            )
        }
    )
    lib = await load_sound_library(storage, tenant_id="", use_cache=False)
    assert not lib.is_valid("weird")
    assert lib.is_valid("good", category="accent")


# --- log level: an EXPECTED miss must not look like a failure ---


async def test_absent_manifests_log_debug_and_never_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No manifest for either layer is the normal fresh-env / reserved-tenant
    state. It degrades to the built-in floor and must emit ZERO warnings — a
    warning that fires for a designed-for condition trains people to ignore
    warnings, which is the actual defect."""
    clear_sound_library_cache()
    storage = _FakeStorage({})
    with caplog.at_level(logging.DEBUG, logger=_CATALOG_LOGGER):
        lib = await load_sound_library(storage, tenant_id="t1", use_cache=False)

    assert lib.is_valid("whoosh_soft")  # built-in floor still stands
    ours = [r for r in caplog.records if r.name == _CATALOG_LOGGER]
    assert [r.getMessage() for r in ours if r.levelno >= logging.WARNING] == []
    debug_msgs = [r.getMessage() for r in ours if r.levelno == logging.DEBUG]
    # Both layers reported, and the optional tenant layer is worded as such.
    assert any(f"{SHARED_PREFIX}/manifest.json" in m for m in debug_msgs)
    assert any(
        f"{tenant_prefix('t1')}/manifest.json" in m and "optional layer" in m for m in debug_msgs
    )


async def test_read_failure_still_warns(caplog: pytest.LogCaptureFixture) -> None:
    """A read that RAISES is the unexpected case the warning is reserved for.

    The facade returns ``None`` for a missing object, so anything reaching the
    broad handler is a real read failure (uninitialised client, broken backend)
    — it must keep its WARNING, and must still degrade rather than raise.
    """
    clear_sound_library_cache()
    with caplog.at_level(logging.DEBUG, logger=_CATALOG_LOGGER):
        lib = await load_sound_library(_ExplodingStorage(), tenant_id="t1", use_cache=False)

    assert lib.is_valid("whoosh_soft")  # never fails a run on the manifest
    warnings = [
        r.getMessage()
        for r in caplog.records
        if r.name == _CATALOG_LOGGER and r.levelno == logging.WARNING
    ]
    assert len(warnings) == 2  # shared + tenant, both genuine failures
    assert all("could not read" in m for m in warnings)


async def test_cache_returns_same_instance_within_ttl() -> None:
    clear_sound_library_cache()
    storage = _FakeStorage({})
    first = await load_sound_library(storage, tenant_id="", use_cache=True)
    second = await load_sound_library(storage, tenant_id="", use_cache=True)
    assert first is second
    clear_sound_library_cache()
    third = await load_sound_library(storage, tenant_id="", use_cache=True)
    assert third is not first
