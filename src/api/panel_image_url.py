"""Rewrite a storyboard panel's in-cluster MinIO image URL to a browser-reachable path.

A generated panel's ``media_ref.url`` is a presigned ``http://minio.test:8389/…``
URL. That host resolves inside the cluster (Tailnet split-DNS) and server-side
consumers reach it fine — but the user's browser cannot resolve it at all, so an
``<img src>`` pointed at it never loads. Stage outputs already solved exactly this
(:mod:`src.api.artifact_url_rewriter`, whose docstring states the same problem);
storyboard panels never got the same treatment, so generated panels were invisible
to real users no matter how well the generation worked.

Panels carry no ``run_id`` — the storyboard tables deliberately have none — so the
run-scoped artifact proxy cannot serve them. The browser path is the panel-scoped
``GET /api/storyboard/panels/{panel_id}/image/content`` instead.

**Why the key confinement below is load-bearing.** ``media_ref`` is client-writable
(it is a field on both ``StoryboardPanelCreate`` and ``StoryboardPanelUpdate``), and
the proxy reads MinIO with the ORCHESTRATOR's credentials, which can read every
tenant's objects. Authorising the *panel* therefore proves nothing about the
*object*: without a key check, tenant A PATCHes its own panel's ``media_ref.url`` to
a key under tenant B's prefix and the proxy streams B's bytes back — a confused
deputy. (The pre-existing re-sign path has the same shape, but only ever produced a
presigned URL no browser outside the cluster could fetch; a proxy turns that latent
exposure into a live read.) So :func:`panel_object_ref` requires the key to sit under
a prefix derived from the panel's OWN row plus the authenticated tenant — never from
the URL and never from caller input.

Those prefixes are exactly the two places a panel image is ever written:

* ``artifacts/storyboard/{project_id}/`` — the per-project path
  (``render.StoryboardPanelImageService._namespace`` → ``storage_service.store_stage_artifact``);
* ``media-cache/tenants/{tenant_id}/`` — the cross-run content-addressed cache
  (``tts_storage._media_cache_key``), used instead of the first whenever
  ``storage.media_cache_cross_run`` is on and a tenant is bound. It is
  tenant-scoped rather than project-scoped **by the writer's design** (a content
  hash is not an auth boundary, so a blob is never shared across tenants), and this
  guard matches the writer rather than being stricter than it — being stricter would
  404 every real panel thumbnail the day that flag is flipped.

Storyboard ASSET reference images (``tts_storage.store_storyboard_asset_image``,
under ``artifacts/storyboard_assets/{tenant_id}/``) are deliberately NOT allowed: they
land on ``StoryboardAsset.reference_images``, and nothing in this service ever writes
one onto a panel's ``media_ref``. Add that prefix explicitly if a panel-frame upload
path is ever built — never by loosening these to a substring match.

**The DB representation is deliberately unchanged**, the same split
``artifact_url_rewriter`` documents: the stored presigned URL remains the durable
object identity, and the server-side PDF export keeps reading it (re-signed) via
``routes/storyboard._resolve_panel_images`` — inside the cluster ``minio.test``
is reachable and an absolute URL is what that fetch needs. Only the JSON leaving
the public API is rewritten.

:func:`panel_object_ref` is the SINGLE predicate for "is this panel image
proxyable", used both by the rewrite (what the API emits a proxy path for) and by
the route (what the proxy will serve), so a URL the API advertises and a URL the
proxy accepts cannot drift apart.
"""

from __future__ import annotations

import logging
from uuid import UUID

from src.api.artifact_url_rewriter import internal_minio_ref
from src.api.schemas import PanelImageJobResponse
from src.config import get_settings
from src.services.object_key_guard import (
    KeyConfinementError,
    decode_ref_path,
    is_within,
    normalize_key,
)
from src.services.storage_service import MEDIA_CACHE_ROOT
from src.services.storyboard.panel_presign import MediaRefJson

log = logging.getLogger(__name__)


def panel_image_proxy_path(panel_id: UUID | str) -> str:
    """The relative path a browser loads this panel's image from.

    Relative, with no host: the SPA and the orchestrator share the
    ``w2v.nonagentic.ai`` origin via the SSR proxy, so a path is all an
    ``<img src>`` needs — and it stays correct across environments.
    """
    return f"/api/storyboard/panels/{panel_id}/image/content"


def _owned_prefixes(project_id: UUID | str, tenant_id: UUID | str) -> tuple[str, ...]:
    """The object-key prefixes a panel of this project/tenant may address.

    Both are derived from trusted inputs — the panel row's ``project_id`` and the
    authenticated tenant — so neither can be steered by a caller-supplied URL. See
    the module docstring for which writer mints each.
    """
    return (
        f"artifacts/storyboard/{project_id}",
        f"{MEDIA_CACHE_ROOT}/tenants/{tenant_id}",
    )


def panel_object_ref(
    url: object, *, project_id: UUID | str, tenant_id: UUID | str
) -> tuple[str, str] | None:
    """The ``(bucket, object_key)`` a panel's stored URL addresses, if it may serve it.

    ``None`` — meaning "leave this URL alone, and refuse to proxy it" — for anything
    that is not provably this panel's own image: an operator-pasted external image, a
    ``file://`` URL from the local backend, an already-relative path, an object
    outside the orchestrator's own bucket, a key that fails :func:`normalize_key`,
    or **a well-formed key belonging to another project or tenant**. Those are
    returned to the client unchanged rather than rewritten into a proxy path.

    Containment is segment-wise (:func:`object_key_guard.is_within`), so a sibling
    prefix that merely shares a string prefix — ``artifacts/storyboard/{id}evil/x`` —
    is refused, and ``..`` segments are collapsed before the comparison rather than
    after it.

    Keyword-only ``project_id`` / ``tenant_id`` on purpose: two positional UUIDs
    would swap silently, and swapping them here would silently widen the guard.

    The stored URL's query string (its expired signature) is discarded:
    ``internal_minio_ref`` keeps only the path, and the proxy re-reads the object
    with the orchestrator's own credentials, which carry no TTL.
    """
    ref = internal_minio_ref(url)
    if ref is None:
        return None
    bucket, _, key = ref.partition("/")
    # Both writers put panel images in the orchestrator's OWN bucket; nothing else
    # is a panel image, whatever the key looks like.
    if not key or bucket != get_settings().storage.minio_bucket:
        return None
    try:
        # A URL path carries exactly one encoding layer; strip that one and no
        # more, then normalise — see ``object_key_guard``.
        normalized = normalize_key(decode_ref_path(key))
    except KeyConfinementError as exc:
        log.info("panel image url is not a serveable object key (%s): %s", exc, url)
        return None
    if not any(is_within(prefix, normalized) for prefix in _owned_prefixes(project_id, tenant_id)):
        # Not this panel's object. INFO, not a refusal to the caller: the read path
        # simply leaves the url alone and the route 404s, so a probe cannot tell
        # "someone else's key" from "no image".
        log.info(
            "panel image url outside project %s / tenant %s prefixes: %s",
            project_id,
            tenant_id,
            normalized,
        )
        return None
    return bucket, normalized


def public_media_ref(
    panel_id: UUID | str,
    media_ref: MediaRefJson | None,
    *,
    project_id: UUID | str,
    tenant_id: UUID | str,
) -> MediaRefJson | None:
    """*media_ref* as the PUBLIC API may emit it. Use this on every read path.

    Supersedes ``refresh_panel_media_url(...)`` + :func:`proxied_media_ref` as a
    pair. Composing those two was unsafe in that ORDER, and the order was not
    obvious at the call site:

    ``refresh_panel_media_url`` re-signs on BUCKET ownership alone —
    ``StorageService.resign_presigned_url`` accepts "own bucket or any ``w2v-*``"
    and applies no project/tenant key confinement. It therefore minted a FRESH,
    valid, 1-hour presigned GET for a key :func:`panel_object_ref` was about to
    refuse, and ``proxied_media_ref`` then returned that media_ref *unchanged* —
    handing the caller a working pointer to the very object the guard exists to
    withhold. ``media_ref`` is client-writable (it is a field on both
    ``StoryboardPanelCreate`` and ``StoryboardPanelUpdate``), so a tenant could
    PATCH its own panel's url to another tenant's key and read back a live
    signature for it. The confused deputy the module docstring describes, entered
    through the re-sign rather than through the proxy.

    Three outcomes, ownership decided FIRST:

    * **ours** → the proxy path. No signature is minted at all: the proxy re-reads
      the object with the orchestrator's own credentials, so the re-sign this
      path used to perform was pure waste even when it was safe.
    * **an internal object we may NOT serve** → ``url: None``. Never re-signed and
      never emitted. Previously this leaked both a live signature and the key
      itself; it was also unreachable from any browser, so nothing could render it.
    * **anything else** (operator-pasted external image, ``file://``, an already
      relative path) → returned untouched. ``resign_presigned_url`` declined these
      already, so dropping the re-sign changes nothing for them.

    Never mutates the argument: the caller's dict is the SQLAlchemy row's JSONB
    value, and rewriting it in place would mark the ORM object dirty and persist a
    URL the DB must not hold.

    NOT for the server-side export path. ``routes/storyboard._resolve_panel_images``
    genuinely needs an absolute, freshly-signed URL — inside the cluster
    ``minio.test`` is reachable and that fetch requires one. It already checks
    :func:`panel_object_ref` against the URL it is about to fetch before
    allow-listing the host, which is the same ownership-first discipline in the
    shape that path needs.
    """
    if not media_ref:
        return media_ref
    url = media_ref.get("url")
    if panel_object_ref(url, project_id=project_id, tenant_id=tenant_id) is not None:
        return {**media_ref, "url": panel_image_proxy_path(panel_id)}
    if internal_minio_ref(url) is not None:
        # Addresses our MinIO but failed confinement. Withhold it: emitting the
        # stored string would disclose another tenant's object key, and emitting a
        # re-signed one would disclose the object.
        return {**media_ref, "url": None}
    return media_ref


def proxied_job(record: PanelImageJobResponse, *, tenant_id: UUID | str) -> PanelImageJobResponse:
    """*record* with its completed ``media_ref.url`` rewritten to the proxy path.

    ``project_id`` comes off the record, which the runner populated from the panel
    row; ``tenant_id`` is the authenticated caller's (or, on the publish path, the
    tenant the job was submitted for). A copy — the stored job record keeps the
    presigned URL, for the same reason the panel row does.
    """
    media = record.media_ref
    if media is None:
        return record
    if panel_object_ref(media.url, project_id=record.project_id, tenant_id=tenant_id) is None:
        return record
    fresh = media.model_copy(update={"url": panel_image_proxy_path(record.panel_id)})
    return record.model_copy(update={"media_ref": fresh})
