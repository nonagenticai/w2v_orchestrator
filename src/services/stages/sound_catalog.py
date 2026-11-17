"""Sound-effects library — the set of sound ids the compiler can play, each
mapping to an object key in the orchestrator's **own** ``w2v-orchestrator`` MinIO
bucket under the ``sound_library/`` prefix.

Storage model (no dedicated bucket, no extra IAM — just ``StorageService``):

* ``sound_library/shared/<category>/<id>.<ext>``     — the curated, all-tenant library
* ``sound_library/shared/manifest.json``             — its data-driven index
* ``sound_library/tenants/<tenant_id>/<category>/<uuid>.<ext>`` — a tenant's own uploads
* ``sound_library/tenants/<tenant_id>/manifest.json`` — DERIVED from the
  ``tenant_sound_assets`` table (``services.tenant_sound_library``); never hand-edited

The runtime catalog is **data-driven**: ``load_sound_library`` reads the
``manifest.json`` object(s) and merges them over a small built-in baseline, so a
library of hundreds of sounds is added by uploading files + regenerating the
manifest (``scripts/sync_sound_library.py``) with **no code change**. Tenant
separation is "as usual" — the loader layers the current tenant's manifest over
the shared one (tenant ids read from the ambient identity ContextVar, never
threaded).

All BUNDLED assets (built-in + shared library) are **royalty-free for commercial
use with no attribution and no per-use licensing** (Mixkit Free License and/or CC0;
see ``assets/sound_effects/CREDITS.md``). Tenant uploads are NOT: their rights are
TENANT-DECLARED (a tenant_admin attested them on upload), which is why every tenant
id carries the reserved ``tenant_`` prefix — the asset-rights ledger tells the two
apart by id alone, and a tenant id can never shadow a bundled one. The tenant layer
is also key-confined: an entry whose object key is not under that tenant's own
prefix is dropped on load. ``promo_sound_director`` validates every
LLM-emitted accent id against the loaded library before it can reach a shot —
the model is advisory, the library is authoritative (mirrors
``audiobook/voice_catalog.py``).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Literal, Protocol, get_args

from w2v_common.auth.identity_context import get_tenant_id_from_context

from src.auth.tenant_scope import tenant_scope_key

logger = logging.getLogger(__name__)


class _SoundStorage(Protocol):
    """The slice of ``StorageService`` the loader needs — read one object by key."""

    async def get_object_bytes(self, key: str) -> bytes | None: ...


SoundCategory = Literal["transition", "accent", "music", "ambient"]
_CATEGORIES: frozenset[str] = frozenset(get_args(SoundCategory))

#: Where an asset came from. ``tenant`` = uploaded by a tenant_admin (rights
#: tenant-declared); ``builtin``/``shared`` = the bundled royalty-free library.
SoundSource = Literal["builtin", "shared", "tenant"]

#: Every tenant-uploaded sound id starts with this; bundled ids never may.
TENANT_SOUND_ID_PREFIX = "tenant_"


def is_tenant_sound_id(sfx_id: str) -> bool:
    """True for an id in the reserved tenant-upload namespace."""
    return sfx_id.startswith(TENANT_SOUND_ID_PREFIX)


#: Object-key prefixes in the orchestrator's own bucket.
LIBRARY_ROOT = "sound_library"
SHARED_PREFIX = f"{LIBRARY_ROOT}/shared"
MANIFEST_NAME = "manifest.json"


def shared_key(rel_path: str) -> str:
    """Full object key for a shared-library asset (``<category>/<file>``)."""
    return f"{SHARED_PREFIX}/{rel_path}"


def tenant_prefix(tenant_id: str) -> str:
    """Object-key prefix for a tenant's private library."""
    return f"{LIBRARY_ROOT}/tenants/{tenant_id}"


def tenant_manifest_key(tenant_id: str) -> str:
    """Object key of a tenant's derived manifest."""
    return f"{tenant_prefix(tenant_id)}/{MANIFEST_NAME}"


@dataclass(frozen=True)
class SoundAsset:
    """One catalogued royalty-free sound.

    ``key`` is the object key inside the orchestrator's own bucket. ``category``
    gates where the id may be used (a director picks ``accent`` ids only; the
    config block picks ``transition``/``music``/``ambient``).
    """

    key: str
    description: str
    category: SoundCategory
    duration_s: float
    source: SoundSource = "builtin"
    #: A tenant upload that was DELETED while a run of the tenant was active. It
    #: still resolves (``get``/``key_for``/``resolves``) so a run that already
    #: pinned it renders unchanged, but it is never offered for a new pick
    #: (``ids_in``/``is_valid``).
    retired: bool = False
    #: Categories a tenant upload HELD while a run was active, before a PATCH
    #: re-categorised it. ``resolves`` (an id a run already picked) accepts them;
    #: ``is_valid``/``ids_in`` (a new pick) never do.
    pinned_categories: frozenset[str] = frozenset()


# Built-in baseline (the curated 11). Kept in code so a fresh/un-seeded env still
# validates the core ids; the bucket manifest extends/overrides this at runtime.
# (id, "<category-dir>/<file>", description, category, duration_s)
#
# ⚠ ``duration_s`` is the ffprobe'd length of the file shipped in
# ``assets/sound_effects/<rel>`` — a MEASUREMENT, never a nominal/intended
# length. It is rendered verbatim into the sound-director prompt (``describe``),
# so a wrong value is a lie told to the model at the exact moment it is choosing
# a bed for a piece of a known length: every one of these was originally a round
# guess (120.0 for a 58.0s bed, 0.6 for a 2.4s whoosh) and the director was
# picking beds it believed were twice as long as they are.
# ``scripts/sync_sound_library.py`` already ffprobes when it regenerates the
# bucket manifest; this in-code baseline is the copy that had no such gate, hence
# ``tests/services/stages/test_sound_catalog_durations.py`` — re-measure and
# update BOTH when an asset is replaced.
_BUILTIN_DEFS: tuple[tuple[str, str, str, SoundCategory, float], ...] = (
    # --- transition (played at a shot's cut-in) ---
    (
        "whoosh_soft",
        "transitions/whoosh_soft.mp3",
        "soft airy whoosh for scene cuts",
        "transition",
        2.352,
    ),
    (
        "swoosh_fast",
        "transitions/swoosh_fast.mp3",
        "quick punchy swoosh for snappy cuts",
        "transition",
        1.68,
    ),
    # --- accent (placed at a moment within a line) ---
    ("click_ui", "accents/click_ui.mp3", "crisp UI click for feature reveals", "accent", 0.24),
    (
        "ding_positive",
        "accents/ding_positive.mp3",
        "bright positive ding / confirmation",
        "accent",
        2.832,
    ),
    ("chime_soft", "accents/chime_soft.mp3", "gentle chime for emphasis", "accent", 1.704),
    ("pop_subtle", "accents/pop_subtle.mp3", "subtle pop for bullet points", "accent", 1.152),
    # --- music (continuous bed, ducked under narration) ---
    (
        "corporate_uplift",
        "music/corporate_uplift.mp3",
        "upbeat corporate background bed",
        "music",
        58.032,
    ),
    ("ambient_tech", "music/ambient_tech.mp3", "calm tech ambient music bed", "music", 58.032),
    ("jazz_lounge", "music/jazz_lounge.mp3", "smooth lounge jazz background bed", "music", 58.032),
    # --- ambient (continuous environment bed) ---
    ("room_tone", "ambient/room_tone.mp3", "neutral room tone", "ambient", 42.552),
    ("office_hum", "ambient/office_hum.mp3", "soft office ambience", "ambient", 42.12),
)

#: The curated baseline, id → asset (keys under ``sound_library/shared/``).
BUILTIN_CATALOG: dict[str, SoundAsset] = {
    sid: SoundAsset(shared_key(rel), desc, cat, dur, source="builtin")
    for sid, rel, desc, cat, dur in _BUILTIN_DEFS
}

#: Default ids used when a config block opts in with ``true`` (rather than a
#: specific id). Both are in the baseline so they always resolve.
DEFAULT_TRANSITION_ID = "whoosh_soft"
DEFAULT_AMBIENT_ID = "room_tone"


class SoundLibrary:
    """An immutable, resolved view of the sounds available to one caller
    (built-in + shared manifest + current-tenant manifest)."""

    __slots__ = ("_assets",)

    def __init__(self, assets: dict[str, SoundAsset]) -> None:
        self._assets = dict(assets)

    @classmethod
    def builtin(cls) -> SoundLibrary:
        """The baseline library with no manifest layers (fresh-env / test floor)."""
        return cls(BUILTIN_CATALOG)

    def get(self, sfx_id: str) -> SoundAsset | None:
        return self._assets.get(sfx_id)

    def is_valid(self, sfx_id: str, *, category: SoundCategory | None = None) -> bool:
        """True when ``sfx_id`` may be CHOSEN now: in the library, not retired, and
        (optionally) of a given category."""
        asset = self._assets.get(sfx_id)
        return (
            asset is not None
            and not asset.retired
            and (category is None or asset.category == category)
        )

    def resolves(self, sfx_id: str, *, category: SoundCategory | None = None) -> bool:
        """Like :meth:`is_valid` but also true for a RETIRED tenant upload.

        For an id a run already PINNED (a brief/edit pin in the snapshot, a stage
        output): deleting the track after the run started must not drop that run's
        bed. Likewise a category the track held when the run picked it still
        resolves after a PATCH re-categorised it (``pinned_categories``): a run pins
        its config for its whole lifetime. Never use this to validate a new choice."""
        asset = self._assets.get(sfx_id)
        return asset is not None and (
            category is None or asset.category == category or category in asset.pinned_categories
        )

    def ids_in(self, category: SoundCategory) -> list[str]:
        """Sorted pickable (non-retired) ids in a category."""
        return sorted(
            sid for sid, a in self._assets.items() if a.category == category and not a.retired
        )

    def key_for(self, sfx_id: str) -> str | None:
        """Object key for a catalog id, or ``None`` if unknown."""
        asset = self._assets.get(sfx_id)
        return asset.key if asset else None

    def describe(self, sfx_id: str) -> str:
        """Human/LLM-readable '<id> (<category>, <dur>s): <desc>' hint."""
        asset = self._assets[sfx_id]
        return f"{sfx_id} ({asset.category}, {asset.duration_s:.1f}s): {asset.description}"

    def accent_prompt_block(self) -> str:
        """Accent-only listing for the sound-director LLM (it picks accents)."""
        return "\n".join(self.describe(sid) for sid in self.ids_in("accent"))


# --- Runtime loading (manifest-driven, tenant-scoped, cached) ---

_LIB_TTL_SECONDS = 60.0
# Process cache keyed by tenant id ("" = no tenant / shared only). Short TTL so a
# freshly-synced manifest is picked up without a restart.
_LIB_CACHE: dict[str, tuple[float, SoundLibrary]] = {}


def clear_sound_library_cache() -> None:
    """Drop the process cache (tests + after a re-seed)."""
    _LIB_CACHE.clear()


async def _load_manifest(
    storage: _SoundStorage,
    prefix: str,
    *,
    optional: bool = False,
    source: SoundSource = "shared",
) -> dict[str, SoundAsset]:
    """Parse ``<prefix>/manifest.json`` into assets. Non-fatal: a missing or
    malformed manifest yields an empty layer (the library degrades, never errors).

    ``optional`` marks a layer whose ABSENCE is the designed-for normal state —
    i.e. the per-tenant library, which exists only once that tenant has uploaded
    a sound. Only the CALLER can know that, because a
    missing object is **not distinguishable by exception type** at this level:
    ``get_object_bytes`` lands in
    ``w2v_common.storage.service.StorageService.get_bytes``, which returns
    ``None`` for an absent object on BOTH backends — the local branch returns
    ``None`` when the path is not a file (``service.py:342-348``) and the minio
    branch's ``_retrieve_minio`` swallows ``OSError``/``S3Error``/
    ``MinioException`` and returns ``None`` (``service.py:674-685``). So
    not-found reaches us as a falsy ``raw``, never as a raise, and there is no
    not-found type to catch ahead of the broad handler. Hence the flag rather
    than an ``except NotFound``.

    ⚠️ The facade logs its OWN ``"Failed to retrieve from MinIO: <key>"`` at
    WARNING inside that swallow (``service.py:684``), so on the minio backend a
    routine absent tenant manifest still produces one warning — from w2v_common,
    not from here. Nothing in this module can suppress it; silencing that one
    means fixing ``_retrieve_minio`` to tell NoSuchKey apart from a real
    transport error, in the w2v_common repo.
    """
    object_key = f"{prefix}/{MANIFEST_NAME}"
    try:
        raw = await storage.get_object_bytes(object_key)
    except Exception as exc:  # noqa: BLE001 -- manifest is best-effort; never fail a run on it
        # Deliberately still WARNING. Per the docstring above, an absent object
        # CANNOT reach this handler — the storage facade returns None for that
        # case. Anything that does raise here is an unexpected read failure (an
        # uninitialised client, a backend breaking its own contract), which is
        # precisely what a warning should be reserved for.
        logger.warning("sound_library: could not read %s: %s", object_key, exc)
        return {}
    if not raw:
        # The absent-manifest path — DEBUG, never WARNING. For a tenant that never
        # uploaded, this is the expected state on every single load, and a
        # warning that fires for a designed-for condition is exactly how a team
        # learns to ignore warnings. An absent SHARED manifest is non-fatal too
        # (the built-in baseline is the floor), but the two are worth telling
        # apart when reading a log, hence the distinct wording.
        logger.debug(
            "sound_library: no manifest at %s — %s",
            object_key,
            "optional layer, nothing to overlay"
            if optional
            else "layer adds nothing; the built-in baseline stands",
        )
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError) as exc:
        logger.warning("sound_library: malformed manifest %s: %s", object_key, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("sound_library: manifest %s is not an object", object_key)
        return {}
    out: dict[str, SoundAsset] = {}
    for sid, entry in data.items():
        if not isinstance(entry, dict):
            continue
        key = entry.get("key")
        category = entry.get("category")
        if not key or category not in _CATEGORIES:
            continue
        if not _entry_admissible(str(sid), str(key), prefix, source):
            logger.warning(
                "sound_library: dropping %s entry %r (key %r) from %s — outside its namespace",
                source,
                sid,
                key,
                object_key,
            )
            continue
        try:
            out[str(sid)] = SoundAsset(
                key=str(key),
                description=str(entry.get("description", "")),
                category=category,  # narrowed to SoundCategory by the membership check
                duration_s=float(entry.get("duration_s", 0.0) or 0.0),
                source=source,
                retired=source == "tenant" and entry.get("retired") is True,
                pinned_categories=_pinned_categories(entry) if source == "tenant" else frozenset(),
            )
        except TypeError, ValueError:
            continue
    return out


def _pinned_categories(entry: dict[object, object]) -> frozenset[str]:
    """A tenant entry's ``pinned_categories``, restricted to the real vocabulary."""
    raw = entry.get("pinned_categories")
    if not isinstance(raw, list):
        return frozenset()
    return frozenset(c for c in raw if isinstance(c, str) and c in _CATEGORIES)


def _entry_admissible(sid: str, key: str, prefix: str, source: SoundSource) -> bool:
    """The namespace rules that keep tenant uploads apart from the bundled library.

    * A tenant entry must use a ``tenant_`` id AND point under that tenant's own
      prefix — a manifest (however it was written) can never make one tenant's
      library resolve another tenant's object, or a bundled object under a tenant
      id.
    * A shared entry may not use the reserved ``tenant_`` namespace, so a bundled
      asset can never be mistaken for a tenant upload (or vice versa) by id.
    """
    if source == "tenant":
        return is_tenant_sound_id(sid) and key.startswith(f"{prefix}/")
    return not is_tenant_sound_id(sid)


async def load_sound_library(
    storage: _SoundStorage | None = None,
    *,
    tenant_id: str | None = None,
    use_cache: bool = True,
) -> SoundLibrary:
    """Resolve the sounds available to the current caller.

    Layers (later overrides earlier on id collision):
    built-in baseline → ``shared/manifest.json`` → ``tenants/<tenant>/manifest.json``.

    ``tenant_id`` defaults to the ambient identity ContextVar; ``storage`` to the
    process-wide singleton. Cached per tenant for ``_LIB_TTL_SECONDS``.
    """
    if tenant_id is None:
        # w2v-common 0.67.0: the ambient ContextVar is `uuid.UUID | None`, not a
        # str. `tenant_scope_key` renders it and refuses the nil uuid BY VALUE --
        # nil was the pre-0.67 `auth.default_tenant_id` fallback, and letting it
        # through would key `_LIB_CACHE` and `tenants/00000000-.../manifest.json`
        # to a single bucket shared by every unattributed caller. An explicitly
        # passed `tenant_id` is in-process caller-supplied and is left verbatim.
        tenant_id = tenant_scope_key(get_tenant_id_from_context()) or ""
    cache_key = tenant_id

    if use_cache:
        hit = _LIB_CACHE.get(cache_key)
        if hit is not None and (time.monotonic() - hit[0]) < _LIB_TTL_SECONDS:
            return hit[1]

    store: _SoundStorage
    if storage is not None:
        store = storage
    else:
        from src.services.tts_storage import _get_storage

        # why: _get_storage returns the structural _StorageLike singleton, which the
        # concrete StorageService satisfies (and which carries get_object_bytes).
        store = await _get_storage()  # ty: ignore[invalid-assignment]

    assets: dict[str, SoundAsset] = dict(BUILTIN_CATALOG)
    assets.update(await _load_manifest(store, SHARED_PREFIX, source="shared"))
    if tenant_id:
        # optional=True: a tenant that never uploaded has no manifest — the normal
        # state. The caller is the only place that knows which layer is expected
        # to exist, so it is the caller that decides whether a miss is log-worthy.
        # Tenant ids live in the reserved ``tenant_`` namespace, so this layer only
        # ADDS ids; it cannot shadow a bundled one.
        assets.update(
            await _load_manifest(store, tenant_prefix(tenant_id), optional=True, source="tenant")
        )

    library = SoundLibrary(assets)
    if use_cache:
        _LIB_CACHE[cache_key] = (time.monotonic(), library)
    return library
