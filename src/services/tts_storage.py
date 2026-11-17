"""TTS clip persistence helper.

The TTS service is sync and returns raw WAV bytes — there is no
artifact store on the tts_backend side. Stage handlers (``lesson_narrate``,
``app_review_record``) need an ``artifact_url`` so downstream stages
(``lesson_compile``) can reference the audio.

This helper wraps ``StorageService`` with:

* a process-wide lazy singleton (``StageHandler``\\s receive ``RunContext``,
  not ``app.state``, so the storage layer can't be threaded through the
  registry without a wider refactor);
* a single ``store_tts_clip`` entry point that writes WAV bytes under a
  stable per-run/clip path and returns the best URL the backend can give
  (presigned http(s):// for MinIO, ``file://`` for the local backend).

Tests can inject a fake by calling ``set_storage_for_tests``.
"""

from __future__ import annotations

import io
import json
import logging
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal, Protocol, cast

from w2v_common.auth.identity_context import get_tenant_id_from_context

from src.auth.tenant_scope import tenant_scope_key
from src.config import get_settings
from src.services._lazy_singleton import AsyncLazySingleton
from src.services.compile_budget import PRESIGN_TTL_SECONDS
from src.services.storage_service import MEDIA_CACHE_ROOT, StorageService

logger = logging.getLogger(__name__)


class _StorageLike(Protocol):
    async def initialize(self) -> None: ...
    async def store_stage_artifact(
        self, job_id: str, stage_name: str, data: bytes, filename: str
    ) -> str: ...
    async def get_presigned_url(
        self, path: str, expires: int | None = ..., bucket: str | None = ...
    ) -> str | None: ...


async def _make_storage() -> _StorageLike:
    svc = StorageService()
    await svc.initialize()
    return svc


_storage_singleton: AsyncLazySingleton[_StorageLike] = AsyncLazySingleton(_make_storage)


async def _get_storage() -> _StorageLike:
    return await _storage_singleton.get()


def set_storage_for_tests(storage: _StorageLike | None) -> None:
    """Override (or reset) the process-wide storage singleton.

    Tests use this to inject an in-memory fake so ``store_tts_clip`` does
    not require a real MinIO/local-disk backend. Pass ``None`` to clear.
    """
    _storage_singleton.set_for_tests(storage)


async def _store_asset(
    run_id: str,
    stage_name: str,
    asset_id: str,
    data: bytes,
    ext: str,
    expires: int | None = None,
) -> str:
    """Persist a stage asset and return a fetchable URL.

    The returned URL is whatever the backend can produce (MinIO presigned
    http(s):// when configured; ``file://`` for the local backend). Stages
    pass this verbatim to ``lesson_compile`` so the video-compile service
    can fetch the asset anonymously.

    ``expires`` overrides the presign TTL (seconds); ``None`` uses the
    backend default (``presign_expiry_seconds``, 1h). Assets consumed long
    after upload (e.g. a recording package fetched by the record stage of a
    later/paused run) pass an explicit longer window.
    """
    storage = await _get_storage()
    filename = f"{asset_id}.{ext}"
    path = await storage.store_stage_artifact(run_id, stage_name, data, filename)
    presigned = await storage.get_presigned_url(path, expires=expires)
    if presigned:
        return presigned
    # Local-backend fallback: store_stage_artifact returns an absolute disk path.
    return f"file://{path}"


# Version of the tts_backend synthesis pre-processing (normalize + split + gaps +
# misaki G2P). Folded into the clip id so improving the pre-processing forces a
# clean re-synth instead of replaying stale clips: the tts_backend change does NOT alter
# ``text``, so without this token the id is unchanged and the existence check would reuse
# the old audio and the fix would be inaudible.
# ⚠️ Keep in lockstep with tts_gateway's job_manager.PREPROC_VERSION and bump on every
# tts_backend pre-processing change. Old clips stay cacheable under their old ids.
# "1" → "2" (2026-07-04): misaki G2P activated on the tts_backend live path
# (TTS_G2P=misaki on both minis) — phonemization changed without changing request text.
PREPROC_VERSION = "2"


def deterministic_clip_id(text: str, voice: str | None, speed: float | None) -> str:
    """Content-addressed clip id: identical ``(voice, speed, text, preproc-version)`` → identical id.

    Replaces a random uuid so a resumed/re-run narrate reuses the already-stored
    clip instead of re-synthesizing it — a pod reap mid-narrate then costs only
    the in-flight chunk(s), not the whole book — and identical repeated lines
    dedupe to a single synth. ``speed`` is normalised so ``1`` and ``1.0``
    collide; ``None`` (the video narrate stages omit speed, relying on the
    backend default) hashes to a stable empty token. ``PREPROC_VERSION`` invalidates
    every id when the backend pre-processing changes.
    """
    speed_token = "" if speed is None else format(float(speed), ".3f")
    raw = f"{voice or ''}|{speed_token}|{PREPROC_VERSION}|{text}"
    return sha256(raw.encode("utf-8")).hexdigest()[:40]


async def find_existing_clip(
    run_id: str, stage_name: str, clip_id: str, audio_format: str = "wav"
) -> tuple[str, float] | None:
    """Return ``(artifact_url, seconds)`` for an already-stored clip, else ``None``.

    Presence + duration come from the ``{clip_id}.json`` sidecar
    ``store_tts_clip`` writes AFTER the wav — so a present sidecar implies a
    present wav (no half-written-clip race). Any probe failure (no sidecar, a
    test fake without the probe methods, a transient MinIO error, or storage not
    yet initialised) returns ``None`` — indeterminate, so the caller
    synthesizes. An existence check must never raise or a storage blip would
    fail an otherwise-good narrate.

    ``retrieve_stage_artifact`` / ``stage_artifact_url`` are resolved via
    ``getattr`` rather than declared on ``_StorageLike`` — that Protocol
    describes the minimal *clip-storage* surface every TTS/slide test fake
    implements, and this resume probe must not force those partial fakes to grow
    two methods they never use (same rationale as ``artifact_exists`` below). The
    real ``StorageService`` always provides them; a fake without them reads as
    "can't probe" → synthesize.
    """
    try:
        storage = await _get_storage()
        retrieve = getattr(storage, "retrieve_stage_artifact", None)
        locate = getattr(storage, "stage_artifact_url", None)
        if retrieve is None or locate is None:
            return None
        sidecar = await retrieve(run_id, stage_name, f"{clip_id}.json")
        if not sidecar:
            return None
        seconds = float(json.loads(sidecar).get("seconds") or 0.0)
        url = await locate(run_id, stage_name, f"{clip_id}.{audio_format}")
        if not url:
            return None
        return url, seconds
    except Exception as exc:  # noqa: BLE001 -- a probe failure is "don't know", never a raise
        logger.warning(
            "find_existing_clip(%s/%s/%s) indeterminate: %s", run_id, stage_name, clip_id, exc
        )
        return None


def _pad_wav_head(audio: bytes, pad_seconds: float) -> tuple[bytes, float] | None:
    """``audio`` (PCM WAV) with ``pad_seconds`` of digital silence prepended, and the
    silence actually added (whole frames); ``None`` when ``audio`` is not plain PCM WAV."""
    try:
        with wave.open(io.BytesIO(audio), "rb") as src:
            params = src.getparams()
            frames = src.readframes(src.getnframes())
    except wave.Error, EOFError:
        return None
    pad_frames = round(pad_seconds * params.framerate)
    if pad_frames <= 0:
        return None
    out = io.BytesIO()
    with wave.open(out, "wb") as dst:
        dst.setparams(params)
        dst.writeframes(b"\x00" * (pad_frames * params.sampwidth * params.nchannels) + frames)
    return out.getvalue(), pad_frames / params.framerate


async def pad_stored_clip_head(
    run_id: str, stage_name: str, clip_id: str, pad_seconds: float
) -> tuple[str, str, float] | None:
    """A copy of a stored WAV clip with leading silence: ``(clip_id, url, seconds)``.

    No synthesis: the stored audio is padded locally, so the words are exactly the
    measured take's. The sidecar is copied (level keys unchanged — leading digital
    silence moves neither integrated loudness nor true peak), ``seconds`` grows by the
    pad, word timings shift by it, and ``head_pad_seconds`` / ``source_clip_id`` record
    the provenance. Content-addressed on ``(clip_id, pad ms)``, so a resumed narrate
    reuses it. ``None`` on ANY failure (no such clip, not PCM WAV, a storage error) —
    the caller then ships the unpadded clip.
    """
    try:
        storage = await _get_storage()
        retrieve = getattr(storage, "retrieve_stage_artifact", None)
        if retrieve is None:
            return None
        pad_ms = round(pad_seconds * 1000)
        padded_id = sha256(f"{clip_id}|headpad|{pad_ms}".encode()).hexdigest()[:40]
        existing = await find_existing_clip(run_id, stage_name, padded_id)
        if existing is not None:
            return padded_id, existing[0], existing[1]
        audio = await retrieve(run_id, stage_name, f"{clip_id}.wav")
        sidecar_raw = await retrieve(run_id, stage_name, f"{clip_id}.json")
        if not audio or not sidecar_raw:
            return None
        padded = _pad_wav_head(audio, pad_ms / 1000.0)
        if padded is None:
            return None
        data, added = padded
        sidecar = dict(json.loads(sidecar_raw))
        seconds = float(sidecar.get("seconds") or 0.0) + added
        sidecar["seconds"] = seconds
        timings = sidecar.get("word_timings")
        if isinstance(timings, list):
            sidecar["word_timings"] = [
                {
                    **w,
                    "start": float(w.get("start") or 0.0) + added,
                    "end": float(w.get("end") or 0.0) + added,
                }
                for w in timings
                if isinstance(w, dict)
            ]
        speed = sidecar.get("speed")
        if isinstance(speed, int | float) and speed > 0:
            sidecar["natural_seconds"] = seconds * float(speed)
        sidecar["head_pad_seconds"] = added
        sidecar["source_clip_id"] = clip_id
        url = await _store_asset(run_id, stage_name, padded_id, data, "wav")
        await _store_asset(
            run_id, stage_name, padded_id, json.dumps(sidecar).encode("utf-8"), "json"
        )
        return padded_id, url, seconds
    except Exception as exc:  # noqa: BLE001 -- a failed pad ships the unpadded clip
        logger.warning(
            "pad_stored_clip_head(%s/%s/%s) failed: %s", run_id, stage_name, clip_id, exc
        )
        return None


# Version of the media-generation inputs folded into ``deterministic_media_id``.
# Bump when the meaning of an id's inputs changes (e.g. a new prompt-shaping pass)
# so a regenerate can't silently replay a stale image/video/slide.
_MEDIA_GEN_VERSION = "1"


def deterministic_media_id(*parts: object) -> str:
    """Content-addressed id for a generated media item (image / video / slide).

    Identical generative inputs → identical id, so a resumed or retried fan-out
    stage reuses the already-stored item instead of re-hitting ACG / slide-service
    / video_production_service for it — a pod reap mid-fan-out then costs only the
    in-flight item(s), not the whole batch, and identical repeated inputs dedupe.
    Replaces the random ``uuid7()`` these stages minted per item. Pass the inputs
    that actually determine the output (prompt, model/engine, size, style, source
    ref, ...); ``None`` hashes to a stable empty token. ``_MEDIA_GEN_VERSION``
    invalidates every id when the input semantics change.
    """
    raw = "|".join("" if p is None else str(p) for p in parts) + f"|v{_MEDIA_GEN_VERSION}"
    return sha256(raw.encode("utf-8")).hexdigest()[:40]


def _cross_run_tenant() -> str | None:
    """Tenant id to key the cross-run media cache under, or ``None`` to fall back.

    P3b makes the media cache reuse an item across *different* runs, not just a
    single run's retries (P2). Returns the tenant id only when the feature flag
    is on AND a tenant is bound in the ambient identity ContextVar (which
    ``run_dispatcher`` binds for the whole run task via ``set_identity_context``).
    Off / no tenant / any error → ``None`` → callers use per-run storage exactly
    as P2. Reading the tenant from the ambient context (not threading it through
    every ``store_*`` call site) follows the repo's identity-via-ContextVar rule
    and mirrors ``sound_catalog``'s tenant-scoped library. Cross-run reuse is
    scoped per tenant deliberately — a content hash is not an auth boundary, so
    media is never shared across tenants.
    """
    try:
        if not get_settings().storage.media_cache_cross_run:
            return None
        # w2v-common 0.67.0: the ambient ContextVar is `uuid.UUID | None`. The
        # nil uuid is refused BY VALUE, not just None -- keying the cross-run
        # media cache under nil would share generated media between every
        # unattributed caller, which is exactly what this docstring promises
        # never happens ("a content hash is not an auth boundary").
        return tenant_scope_key(get_tenant_id_from_context())
    except Exception:  # flag/context lookup must never break storage
        # Degrading to per-run storage is CORRECT here; being invisible about it is not.
        # Silently returning None switches cross-run media caching OFF, so every image /
        # video / slide is REGENERATED at real provider cost on the next run — and the
        # symptom is indistinguishable from "the feature flag is off". Log it.
        logger.warning(
            "media cache: cross-run tenant lookup failed — falling back to PER-RUN media "
            "storage for this call, so cached images/videos/slides will be REGENERATED at "
            "cost. This is a degrade, not the feature flag being off.",
            exc_info=True,
        )
        return None


def _media_cache_key(tenant_id: str, media_id: str, ext: str) -> str:
    """Global (cross-run), tenant-scoped object key for a content-addressed item.

    ``media_id`` already folds every generative input + ``_MEDIA_GEN_VERSION`` and
    carries no run/stage identity, so the same inputs under the same tenant map to
    one blob across runs. Sharded by ``media_id[:2]`` to avoid a single fat prefix.
    Lives under ``MEDIA_CACHE_ROOT`` so the bucket lifecycle rule can GC it.
    """
    return f"{MEDIA_CACHE_ROOT}/tenants/{tenant_id}/{media_id[:2]}/{media_id}.{ext}"


def _ingest_cache_key(tenant_id: str, fingerprint: str, idx: int, ext: str) -> str:
    """Durable, tenant-scoped object key for a source-ingest cached image.

    Lives under ``MEDIA_CACHE_ROOT`` (so the bucket lifecycle rule GCs it) in an
    ``ingest/`` subtree keyed by the source ``fingerprint`` — NOT the originating
    run's ``artifacts/{run}/`` prefix, which is reclaimed with the run and could
    not be re-fetched by a later run reusing the cache. Sharded by ``fingerprint[:2]``.
    """
    return (
        f"{MEDIA_CACHE_ROOT}/tenants/{tenant_id}/ingest/{fingerprint[:2]}/{fingerprint}/{idx}.{ext}"
    )


async def store_ingest_cache_image(
    tenant_id: str, fingerprint: str, idx: int, data: bytes, ext: str
) -> str | None:
    """Persist a source-ingest extracted image to the durable cache prefix.

    Returns the stored object KEY (to record in the cache row — NEVER a presigned
    URL, which expires) or ``None`` if the backend can't take a keyed put (a partial
    test fake) — same defensive posture as ``_store_media_asset``. The persistent
    source-ingest cache stores this key and re-mints a fresh URL at consume time via
    ``presign_ingest_cache_image`` so cross-run reuse survives presign expiry.
    """
    storage = await _get_storage()
    put = getattr(storage, "put_object", None)
    if put is None:
        return None
    key = _ingest_cache_key(tenant_id, fingerprint, idx, ext)
    await put(key, data)
    return key


async def presign_ingest_cache_image(key: str) -> str | None:
    """Mint a FRESH fetchable URL for a cached-ingest image ``key`` at consume time.

    ``None`` if the backend lacks the generic keyed-URL method (partial fake). The
    caller (source_ingest consume path) gates on a separate ``object_exists`` and
    drops the image on a miss (soft-fail), so a GC'd blob never crashes a compile.
    """
    storage = await _get_storage()
    url_for = getattr(storage, "object_url", None)
    if url_for is None:
        return None
    return await url_for(key)


async def ingest_cache_image_url(key: str) -> str | None:
    """Re-mint a fresh URL for a cached-ingest image ``key`` IF its blob still exists.

    The consume-path resolver: mints the URL (``presign_ingest_cache_image``) and
    probes ``object_exists`` so a lifecycle-GC'd blob (the media-cache TTL expired
    under a still-``ready`` cache row) is DROPPED rather than handed to the compiler
    as a 404. ``object_exists`` returns True/False/None — only a definitive ``False``
    drops (``None`` is indeterminate → keep the re-minted URL, best-effort, matching
    ``find_existing_media``). ``None`` here means "no usable URL" (backend can't
    presign, or the blob is gone) so the caller drops the image (soft-fail).
    """
    url = await presign_ingest_cache_image(key)
    if not url:
        return None
    storage = await _get_storage()
    exists = getattr(storage, "object_exists", None)
    if exists is not None and await exists(url) is False:
        return None
    return url


async def _store_media_asset(
    run_id: str,
    stage_name: str,
    asset_id: str,
    data: bytes,
    ext: str,
) -> str:
    """Persist a fan-out media item, cross-run cached when enabled, else per-run.

    When the cross-run cache is active (flag on + tenant bound), writes under the
    tenant-scoped ``media-cache/`` prefix via the generic keyed-object API so an
    identical item reuses one blob across runs; otherwise defers to ``_store_asset``
    for exact P2 per-run behavior. Falls back to per-run if the storage backend
    lacks the generic methods (e.g. a partial test fake) — same defensive posture
    as ``find_existing_media``.
    """
    tenant_id = _cross_run_tenant()
    if tenant_id is None:
        return await _store_asset(run_id, stage_name, asset_id, data, ext)
    storage = await _get_storage()
    put = getattr(storage, "put_object", None)
    url_for = getattr(storage, "object_url", None)
    if put is None or url_for is None:
        return await _store_asset(run_id, stage_name, asset_id, data, ext)
    key = _media_cache_key(tenant_id, asset_id, ext)
    stored = await put(key, data)
    url = await url_for(key)
    return url or f"file://{stored}"


async def find_existing_media(run_id: str, stage_name: str, media_id: str, ext: str) -> str | None:
    """Return the artifact URL for an already-stored media item, else ``None``.

    Mirrors ``find_existing_clip`` for non-audio assets: a media object is a
    single atomic put (present object ⇒ complete), so existence is probed
    directly via ``object_exists`` (a signature-independent ``stat_object``, not a
    full download) on the reconstructed URL rather than a duration sidecar. Any
    probe failure — a test fake without the methods, a missing object, a transient
    storage error, storage not yet initialised — returns ``None`` (indeterminate,
    so the caller regenerates). An existence check must never raise or a storage
    blip would fail an otherwise-good fan-out.

    ``stage_artifact_url`` / ``object_exists`` are resolved via ``getattr`` rather
    than declared on ``_StorageLike`` — same rationale as ``find_existing_clip``:
    that Protocol is the minimal clip-storage surface the test fakes implement,
    and this resume probe must not force those partial fakes to grow methods they
    never use. The real ``StorageService`` always provides them; a fake without
    them reads as "can't probe" → regenerate.
    """
    try:
        storage = await _get_storage()
        exists = getattr(storage, "object_exists", None)
        if exists is None:
            return None
        # Cross-run cache (P3b): when enabled + tenant bound + backend supports
        # the generic keyed URL, probe ONLY the tenant-scoped cache key (that is
        # where ``_store_media_asset`` wrote it); a miss regenerates.
        tenant_id = _cross_run_tenant()
        if tenant_id is not None:
            url_for = getattr(storage, "object_url", None)
            if url_for is not None:
                key = _media_cache_key(tenant_id, media_id, ext)
                url = await url_for(key)
                if url and await exists(url) is True:
                    return url
                return None
        # Per-run probe (flag off / no tenant / fake without object_url) — P2 path.
        locate = getattr(storage, "stage_artifact_url", None)
        if locate is None:
            return None
        url = await locate(run_id, stage_name, f"{media_id}.{ext}")
        if not url:
            return None
        if await exists(url) is True:
            return url
        return None
    except Exception as exc:  # noqa: BLE001 -- a probe failure is "don't know", never a raise
        logger.warning(
            "find_existing_media(%s/%s/%s) indeterminate: %s", run_id, stage_name, media_id, exc
        )
        return None


LoudnessSource = Literal["tts_backend", "orchestrator", "none"]


@dataclass(frozen=True, slots=True)
class ClipLoudness:
    """What is KNOWN about a stored clip's level, and who established it.

    Written into the clip's sidecar by ``store_tts_clip`` so the level of every
    narration artifact is answerable from storage alone — the audit trail behind
    the ``tts_clip_*`` metrics. ``source="none"`` with ``normalized=False`` is the
    honest "we could not establish this clip's level" record; it must never be
    confused with "-18, as designed".
    """

    integrated_lufs: float | None = None
    true_peak_dbtp: float | None = None
    source: LoudnessSource = "none"
    normalized: bool = False


async def store_tts_clip(
    run_id: str,
    stage_name: str,
    clip_id: str,
    audio_bytes: bytes,
    audio_format: str = "wav",
    *,
    seconds: float | None = None,
    loudness: ClipLoudness | None = None,
    word_timings: Sequence[Mapping[str, object]] | None = None,
    voice: str | None = None,
    provider: str | None = None,
    speed: float | None = None,
    word_count: int | None = None,
) -> str:
    """Persist a TTS-rendered clip and return an artifact URL.

    When ``seconds`` is given, also writes a ``{clip_id}.json`` duration sidecar
    AFTER the wav — the presence marker + duration source that lets a resumed
    narrate reuse this clip via ``find_existing_clip`` without re-synthesizing.

    ``loudness``, when given, adds ``integrated_lufs`` / ``true_peak_dbtp`` /
    ``loudness_source`` / ``normalized`` to that same sidecar. Strictly ADDITIVE:
    ``find_existing_clip`` reads only ``seconds``, so sidecars already in the
    bucket (written before these keys existed) still load and still short-circuit
    a resumed narrate. Absent keys read as "unknown", not as "un-normalized".

    ``word_timings``, when given, adds the per-word ``{word, start, end}`` spans
    tts_backend measured, in the clip's own timebase. Same additive contract: it
    is absent for every clip synthesized by the ONNX engine (which cannot emit
    timings) and for every clip written before this key existed, and an absent
    key means "ask ASR", never "this clip has no words". Storing it here rather
    than in a new object is deliberate — a resumed narrate that short-circuits on
    this sidecar then also recovers the timings, so captions never silently fall
    back to ASR just because the clip was reused.

    ``voice`` / ``provider`` / ``speed`` / ``word_count`` record WHO spoke the
    clip and under what request. Without them the sidecar answers "how long was
    this clip?" but never "how fast does this voice actually talk?", which is why
    a hardcoded words-per-minute constant exists elsewhere in the pipeline: the
    system could not observe its own speech rate from its own history. Each key
    is written ONLY when its value is not ``None`` — an omitted value must never
    land as a JSON ``null`` a reader could mistake for a measured zero (a
    ``word_count`` of 0 means "no words", ``speed`` 0 would mean a stopped
    clip).

    ``natural_seconds`` = ``seconds * speed`` is derived and written only when
    ``speed`` is a positive number: the duration this clip WOULD have had at
    speed 1.0, which is the quantity a rate calculation needs (a clip rendered at
    1.25x is not evidence that the voice talks 25% faster). No words-per-minute
    is computed here — that belongs to the consumer, which owns the window and
    the tokenization.

    All four, plus ``natural_seconds``, are strictly ADDITIVE on the same
    contract as ``loudness`` / ``word_timings``: every existing caller passes
    none of them, ``find_existing_clip`` reads only ``seconds``, and
    ``word_timed_captions`` reads only ``word_timings``, so sidecars written
    before these keys existed still parse and still short-circuit a resumed
    narrate. An absent key reads as "not recorded", never as a measured value.
    """
    url = await _store_asset(run_id, stage_name, clip_id, audio_bytes, audio_format)
    if seconds is not None:
        payload: dict[str, object] = {"seconds": float(seconds), "format": audio_format}
        if loudness is not None:
            payload["integrated_lufs"] = loudness.integrated_lufs
            payload["true_peak_dbtp"] = loudness.true_peak_dbtp
            payload["loudness_source"] = loudness.source
            payload["normalized"] = loudness.normalized
        if word_timings:
            payload["word_timings"] = [dict(w) for w in word_timings]
        if voice is not None:
            payload["voice"] = voice
        if provider is not None:
            payload["provider"] = provider
        if speed is not None:
            payload["speed"] = float(speed)
        if word_count is not None:
            payload["word_count"] = int(word_count)
        if speed is not None and speed > 0:
            payload["natural_seconds"] = float(seconds) * float(speed)
        meta = json.dumps(payload).encode("utf-8")
        await _store_asset(run_id, stage_name, clip_id, meta, "json")
    return url


async def presign_sound_asset(sfx_id: str) -> str | None:
    """Presign a sound-library asset by catalog id.

    The library lives in the orchestrator's **own** bucket under
    ``sound_library/`` (no foreign bucket, no extra IAM), so this is a plain
    own-bucket presign. The object key is resolved from the loaded, tenant-scoped
    library. Returns ``None`` for an unknown id or a non-MinIO backend, so the
    caller drops that cue rather than passing a dead reference.

    Signed with ``compile_budget.PRESIGN_TTL_SECONDS``, NOT the storage default
    (3600s). Minting "at compile time" does NOT sidestep expiry: the compiler
    fetches the music/ambient bed when it reaches the audio mux, which on a
    compile that legitimately runs toward ``COMPILE_WAIT_CEILING_SECONDS`` is
    well past an hour after submit — the bed GET then 403s and the compiler
    ships the master with no bed (``music_bed.applied=False``). Every caller of
    this helper feeds a compile job (the shared sound-design bed on every
    blueprint including website_review's ``build_audio_bed`` override, the
    action_reel bed, per-shot SFX), and catalog audio is not tenant content, so
    the longer TTL is scoped here rather than per call site.
    """
    from src.services.stages.sound_catalog import load_sound_library

    library = await load_sound_library()
    key = library.key_for(sfx_id)
    if not key:
        return None
    storage = await _get_storage()
    return await storage.get_presigned_url(key, expires=PRESIGN_TTL_SECONDS)


async def store_captions_track(
    run_id: str,
    stage_name: str,
    track_id: str,
    data: bytes,
    ext: str = "srt",
) -> str:
    """Persist a soft closed-caption track (SRT/VTT) and return a fetchable URL.

    The YouTube publish path mints an accurate caption track from the run's
    narration timing and stores it here so ``youtube_upload_service`` can fetch
    it anonymously by URL and attach it via ``captions().insert`` — same storage
    path as the per-clip TTS audio / slide images.
    """
    return await _store_asset(run_id, stage_name, track_id, data, ext)


async def artifact_exists(url: str) -> bool | None:
    """Probe whether ``url`` resolves to a stored object (run-finalizer gate).

    Thin wrapper over the process-wide storage singleton so the run dispatcher
    can verify a terminal artifact without threading ``StorageService`` through
    the registry. Returns ``True`` / ``False`` / ``None`` with the same contract
    as ``StorageService.object_exists`` — only ``False`` (definitively absent)
    is safe to fail a run on.

    ``object_exists`` is resolved via ``getattr`` rather than declared on
    ``_StorageLike``: that Protocol describes the *clip-storage* surface the
    TTS/slide test fakes implement, and the finalizer must not force every one
    of those partial fakes to grow a method they never use. The real
    ``StorageService`` always provides it; a fake without it reads as "can't
    probe" → ``None`` (indeterminate, never fails a run).
    """
    storage = await _get_storage()
    probe = getattr(storage, "object_exists", None)
    if probe is None:
        return None
    return await probe(url)


async def store_capture_clip(
    run_id: str,
    stage_name: str,
    clip_id: str,
    video_bytes: bytes,
    video_format: str = "mp4",
) -> str:
    """Persist a recorded site-capture clip and return an artifact URL.

    ``site_capture`` downloads each recorded clip from video_production_service
    (which is license-gated) and re-publishes it here so the video-compile
    service can fetch it anonymously by URL — the same pattern as
    ``store_slide_image``.
    """
    return await _store_asset(run_id, stage_name, clip_id, video_bytes, video_format)


async def store_capture_image(
    run_id: str,
    stage_name: str,
    image_id: str,
    image_bytes: bytes,
    image_format: str = "png",
) -> str:
    """Persist one per-screen capture screenshot and return an artifact URL.

    ``site_capture`` (per-screen mode) downloads a zip of screenshots from
    video_production_service (license-gated), unpacks it, and re-publishes each
    PNG here so the anonymous video-compile service can fetch it by URL — the
    same pattern as ``store_capture_clip`` / ``store_slide_image``.
    """
    return await _store_asset(run_id, stage_name, image_id, image_bytes, image_format)


async def store_slide_image(
    run_id: str,
    stage_name: str,
    image_id: str,
    image_bytes: bytes,
    image_format: str = "png",
) -> str:
    """Persist a rendered slide image and return an artifact URL.

    ``lesson_record`` downloads each slide PNG from slide-service (which is
    license-gated) and re-publishes it here so the video-compile service can
    fetch it anonymously by URL when compositing the final video.
    """
    return await _store_media_asset(run_id, stage_name, image_id, image_bytes, image_format)


async def store_generated_image(
    run_id: str,
    stage_name: str,
    image_id: str,
    image_bytes: bytes,
    image_format: str = "png",
) -> str:
    """Persist an AI-generated concept image and return a fetchable artifact URL.

    ``ai_image_gen`` downloads each image from ACG (license-gated, returns an
    ACG-relative download URL the anonymous video-compile service cannot fetch)
    and re-publishes it here — same pattern as slides / site-capture clips.
    """
    return await _store_media_asset(run_id, stage_name, image_id, image_bytes, image_format)


async def store_generated_video(
    run_id: str,
    stage_name: str,
    video_id: str,
    video_bytes: bytes,
    video_format: str = "mp4",
) -> str:
    """Persist an AI-generated video clip and return a fetchable artifact URL.

    ``ai_video_gen`` downloads each clip from ACG (license-gated, returns an
    ACG-relative download URL the anonymous video-compile service cannot fetch)
    and re-publishes it here — same pattern as ``store_generated_image`` /
    ``store_capture_clip``.
    """
    return await _store_media_asset(run_id, stage_name, video_id, video_bytes, video_format)


async def store_thumbnail_override(
    namespace: str,
    image_id: str,
    image_bytes: bytes,
    image_format: str = "png",
) -> str:
    """Persist an operator-chosen thumbnail (captured video frame or upload).

    Namespaced by tenant (the upload happens before the PATCH that sets it, so
    it is tenant-scoped like the manuscript upload). Returns a fetchable artifact
    URL the orchestrator owns, so the course thumbnail PATCH route can store it
    as ``thumbnail_override_url`` and the YouTube passthrough can re-sign it —
    same storage path as the auto AI cover.
    """
    return await _store_asset(
        f"thumbnails/{namespace}", "course_thumbnail_override", image_id, image_bytes, image_format
    )


async def store_storyboard_asset_image(
    namespace: str,
    image_id: str,
    image_bytes: bytes,
    image_format: str = "png",
) -> str:
    """Persist a storyboard-asset reference image (Character / Location / Prop).

    Namespaced by tenant exactly like ``store_thumbnail_override``: the upload
    happens before the asset row that references it exists (the operator uploads,
    then POSTs/PATCHes the asset with the returned URL onto
    ``reference_images``), so it cannot be project- or run-scoped.

    Load-bearing, same reason as ``store_voiceover_override``: this MUST land in an
    owned ``w2v-*`` bucket. ``_store_asset`` writes to the orchestrator's own
    bucket, so ``storage_service.resign_presigned_url`` can re-mint the URL when the
    1h presign has rotted — a reference image is read back long after upload (every
    later panel render conditions on it), and a foreign bucket would re-sign to
    nothing and silently drop the character's likeness.
    """
    return await _store_asset(
        f"storyboard_assets/{namespace}",
        "asset_reference_image",
        image_id,
        image_bytes,
        image_format,
    )


async def store_voiceover_override(
    namespace: str,
    upload_id: str,
    audio_bytes: bytes,
    audio_format: str = "mp3",
    *,
    seconds: float | None = None,
) -> str:
    """Persist an uploaded per-clip voiceover (Studio narration replacement).

    Namespaced by tenant like ``store_thumbnail_override`` (the upload happens
    before the clip-edit PATCH that references it, so it cannot be run-scoped),
    and — like ``store_tts_clip`` — writes a ``{upload_id}.json`` duration sidecar
    AFTER the audio when ``seconds`` is given, so the probed length stays
    recoverable from storage alone (the same presence-marker ordering: a present
    sidecar implies a present take).

    Load-bearing: this MUST land in an owned ``w2v-*`` bucket. ``_store_asset``
    writes to the orchestrator's own bucket, so ``resign_presigned_url``
    (``storage_service.resign_presigned_url`` → the compile-time
    ``refresh_shot_asset_urls`` pass) can re-mint the URL. A foreign bucket would
    re-sign to nothing, the 1h presign would rot before the re-stitch, and the
    compiler would silently degrade that shot to SILENCE.
    """
    url = await _store_asset(
        f"voiceovers/{namespace}", "clip_voiceover_override", upload_id, audio_bytes, audio_format
    )
    if seconds is not None:
        meta = json.dumps({"seconds": float(seconds), "format": audio_format}).encode("utf-8")
        await _store_asset(
            f"voiceovers/{namespace}", "clip_voiceover_override", upload_id, meta, "json"
        )
    return url


# Recording packages are consumed by the record stage, which may start well after
# upload (a queued or paused showcase run). Presign for 7 days rather than the 1h
# default so the ``package_url`` on the brief is still fetchable when the run reaches
# ``demo_video_record``.
_PACKAGE_PRESIGN_SECONDS = 7 * 24 * 3600


#: Where recording packages live: ``artifacts/video-packages/<tenant>/recording_package/``.
PACKAGE_STAGE = "recording_package"


def package_namespace(tenant: str) -> str:
    return f"video-packages/{tenant}"


class _PackageStorage(Protocol):
    async def retrieve_stage_artifact(
        self, job_id: str, stage_name: str, filename: str
    ) -> bytes | None: ...
    async def stage_artifact_url(
        self, job_id: str, stage_name: str, filename: str
    ) -> str | None: ...


async def storage_for_packages() -> _PackageStorage:
    """The process storage, typed for reading a stored package back by key."""
    return cast("_PackageStorage", await _get_storage())


async def store_package(namespace: str, upload_id: str, data: bytes) -> str:
    """Persist a Playwright recording-package zip and return a fetchable URL.

    Tenant-scoped (``namespace`` is the tenant id): the upload happens before the
    showcase/demo_video project is created, so — like the manuscript and thumbnail
    uploads — it cannot be project-scoped. The returned presigned URL is dropped onto
    ``brief.package_url``; ``demo_video_record`` forwards it to the frontend test
    runner, which fetches the zip anonymously and re-runs the recording live.
    """
    return await _store_asset(
        package_namespace(namespace),
        PACKAGE_STAGE,
        upload_id,
        data,
        "zip",
        expires=_PACKAGE_PRESIGN_SECONDS,
    )
