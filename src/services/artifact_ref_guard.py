"""Which MinIO objects a run may name — enforced on the WRITE path.

The artifact proxy authorises a download by asking "does this run's own persisted
stage output reference that object?". That question is only worth asking if the
answer is not attacker-controlled, and it very nearly was: ``PATCH
/runs/{id}/stages/{stage}/elements/{el}`` merges a free-form ``dict`` straight
into ``StageOutput.output``, and the course-thumbnail PATCH stored any URL that
merely *looked* like an artifact. Owning one worthless run was therefore enough
to write ``http://minio.test:8389/w2v-audiobooks/artifacts/{victim}/…`` into
your OWN output and then stream the victim's audiobook through the proxy — the
read gate matched, because the read gate was matching against a record the
attacker had just authored.

This module closes that: every artifact ref a caller writes must already be one
the caller owns, on one of exactly two grounds.

1. **The run already references it.** Legitimate element edits re-point at
   artifacts the pipeline produced for this run; they never introduce a new one
   (only stage handlers do that, and they write via the repository, not here).
2. **It is the caller's own tenant-scoped upload.** ``POST /uploads/thumbnail``
   (and the storyboard-asset / voiceover uploads) deliberately land *outside* any
   run — the upload precedes the PATCH that adopts it — under
   ``artifacts/{namespace}/{tenant_id}/…`` in the orchestrator's own bucket. The
   rule is the third path segment, not a list of namespaces, so a new upload kind
   is covered the day it ships instead of the day someone remembers this file.

Anything else is refused with a 4xx and never stored — reject, never sanitise, so
there is no half-trusted value to reason about later. The proxy's read-side gate
stays exactly as it was: it is the backstop for any write path this misses, and
"defence in depth" is only true while both layers are actually present.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import select

from src.api.artifact_url_rewriter import collect_internal_minio_refs
from src.config import get_settings
from src.database.models import Deliverable, Run, StageOutput, VideoSource
from src.services.object_key_guard import (
    KeyConfinementError,
    decode_ref_path,
    key_segments,
    normalize_key,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# ``artifacts/{namespace}/{tenant_id}/{stage}/{file}`` — the shape every
# pre-run, tenant-scoped upload lands at (``tts_storage._store_asset`` with a
# ``{namespace}/{tenant}`` pseudo-job). Index 2 is the tenant slot.
_UPLOAD_ROOT = "artifacts"
_UPLOAD_TENANT_SEGMENT = 2
_MIN_UPLOAD_SEGMENTS = 4

# ``video_sources.preview_status`` value that means the browser-playable proxy
# object exists in MinIO. Any other value means there is nothing to serve.
_PREVIEW_READY = "ready"


class ArtifactRefError(ValueError):
    """A caller tried to write an artifact ref it does not own."""


def normalize_stored_ref(ref: str) -> str | None:
    """A stored ``{bucket}/{key}`` in true-object-key form; ``None`` if unusable.

    Stored refs are URL *paths* (they come off a presigned URL), so exactly one
    layer of percent-encoding is stripped before normalisation — that is what
    makes a stored ref comparable to the same key arriving as a path parameter,
    which Starlette has already decoded once. A stored ref we cannot normalise is
    one we will never match, and a single junk value must not poison a whole run's
    allowlist, so it drops out rather than raising.
    """
    try:
        return normalize_key(decode_ref_path(ref))
    except KeyConfinementError:
        return None


async def _ready_preview_refs(session: AsyncSession, run_id: UUID) -> set[str]:
    """The ``{bucket}/{key}`` refs of the READY preview proxies of ``run_id``'s project.

    These objects are real and browser-playable, but they are addressed by a
    COLUMN (``video_sources.preview_key``) rather than by a URL string inside a
    stage output — and for every run whose ``footage_probe`` executed before the
    preview-proxy worker existed there is no such string at all (the stage stored
    ``preview_url: ""``). Without this the read path could hand the SPA a proxy
    path the proxy itself would refuse, i.e. a visible-but-dead player.

    ⚠️ The confinement property CHANGED here, and saying otherwise would be the
    kind of comment this repo has been burned by. For preview keys the allowlist is
    PROJECT-scoped, not RUN-scoped: the join climbs ``run -> deliverable ->
    project`` and then admits EVERY ready proxy of that project, including sources
    uploaded for a sibling run and sources no stage output of this run references.
    The stage-output half of :func:`run_artifact_refs` stays run-scoped.

    Why project-scoping is sound rather than merely convenient:

    * It matches the PRODUCER. ``video_sources`` has no run column at all — footage
      belongs to a project (``footage_probe`` selects the run's footage with exactly
      ``VideoSource.project_id == project_id``), so "the proxies this run may serve"
      has no narrower honest definition than the project's.
    * It cannot cross a tenant. ``project_id`` comes from the run's own row, never
      from a caller-supplied string, and RLS on ``video_sources``
      (project_id -> projects.tenant_id) is the second fence under enforce.
    * The widening is same-tenant, same-project, and bounded to objects that are
      ``preview_status = 'ready'`` — i.e. proxies this tenant's own footage produced
      in a project the caller already has to be authorized on to reach this run.

    ONE query, no matter how many sources the project has.
    """
    bucket = get_settings().storage.minio_bucket
    rows = await session.execute(
        select(VideoSource.preview_key)
        .join(Deliverable, Deliverable.project_id == VideoSource.project_id)
        .join(Run, Run.deliverable_id == Deliverable.id)
        .where(
            Run.id == run_id,
            VideoSource.preview_status == _PREVIEW_READY,
            VideoSource.preview_key.isnot(None),
        )
    )
    refs: set[str] = set()
    for (preview_key,) in rows.all():
        if not preview_key:
            continue
        try:
            # ``normalize_key``, NOT ``normalize_stored_ref``: this is a true
            # object key straight off a column, not a URL path, so it must NOT be
            # percent-decoded — that would turn a literal ``%2D`` in a key into a
            # different key (the ``discount%2Ddeal.pdf`` lesson).
            refs.add(normalize_key(f"{bucket}/{preview_key}"))
        except KeyConfinementError:
            continue
    return refs


async def run_artifact_refs(session: AsyncSession, run_id: UUID) -> set[str]:
    """The ``{bucket}/{key}`` refs ``run_id``'s persisted stage outputs address.

    This is the run's artifact list: the orchestrator has no artifacts table —
    every artifact URL lives as a string inside ``stage_outputs.output`` JSONB —
    so the run's own outputs are the record of what it owns. That record is only
    authoritative because :func:`assert_refs_owned` guards every caller-facing
    write into it. The same walk (``collect_internal_minio_refs``) that produces
    this set is the one ``rewrite_payload`` uses to hand the SPA its proxy URLs,
    so the proxy serves exactly what the API advertised and nothing else.

    Plus the run's project's READY preview proxies (:func:`_ready_preview_refs`) —
    the one class of object this run legitimately serves that no stage output
    references as a URL. ⚠️ That half is PROJECT-scoped, not run-scoped: the
    returned set is therefore "this run's stage-output refs, plus this run's
    PROJECT's ready proxies". Same tenant either way; see that function for why
    the project is the only honest scope for footage.
    """
    rows = await session.execute(
        select(StageOutput.output).where(
            StageOutput.run_id == run_id,
            # Pending / restamped-dirty rows carry a NULL output and can only
            # contribute nothing; skipping them at the DB keeps a reset run's
            # rows off the JSONB parse entirely.
            StageOutput.output.isnot(None),
        )
    )
    refs: set[str] = set()
    for (output,) in rows.all():
        collect_internal_minio_refs(output, refs)
    normalized = {
        normalized for ref in refs if (normalized := normalize_stored_ref(ref)) is not None
    }
    return normalized | await _ready_preview_refs(session, run_id)


def is_own_tenant_upload(normalized_ref: str, tenant_id: UUID | str) -> bool:
    """True iff ``normalized_ref`` is one of ``tenant_id``'s own staged uploads.

    ``{own_bucket}/artifacts/{namespace}/{tenant_id}/…`` and nothing else — the
    bucket must be the orchestrator's own (an upload never lands in a sibling
    service's ``w2v-*`` bucket), and the tenant slot must be the caller's, matched
    as a whole path segment so ``{tenant}evil`` cannot ride a string prefix.
    """
    bucket, _, key = normalized_ref.partition("/")
    if not key or bucket != get_settings().storage.minio_bucket:
        return False
    try:
        segments = key_segments(key)
    except KeyConfinementError:
        return False
    return (
        len(segments) >= _MIN_UPLOAD_SEGMENTS
        and segments[0] == _UPLOAD_ROOT
        and segments[_UPLOAD_TENANT_SEGMENT] == str(tenant_id)
    )


def assert_refs_owned(payload: object, *, tenant_id: UUID | str, owned_refs: set[str]) -> None:
    """Refuse ``payload`` if it names a MinIO object the caller does not own.

    ``owned_refs`` is :func:`run_artifact_refs` for the run being written, read
    BEFORE the write — so a patch can re-point at any artifact the run already
    carries, but cannot bootstrap a foreign one and then "prove" ownership of it
    with the very row it just wrote.

    Only ``minio.test`` URLs are policed, because those are exactly the values
    the proxy's read gate will later honour; a proxy path (``/api/runs/{other}/…``)
    stored in your own output escalates nothing — fetching it is still gated by
    *that* run's ownership check.
    """
    for ref in sorted(collect_internal_minio_refs(payload, set())):
        normalized = normalize_stored_ref(ref)
        if normalized is None:
            raise ArtifactRefError(f"unusable artifact reference: {ref!r}")
        if normalized in owned_refs or is_own_tenant_upload(normalized, tenant_id):
            continue
        raise ArtifactRefError(f"artifact reference not owned by this run: {ref!r}")
