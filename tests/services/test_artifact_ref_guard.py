"""The write-path gate: a caller may not store an artifact ref it does not own.

This closes the hole that made the artifact proxy's read gate circular. The proxy
authorises a download by asking "does this run's own persisted stage output
reference that object?" — and every caller-facing write into
``StageOutput.output`` (the free-form element ``patch``, the element ``restamp``
value, the course-thumbnail override) let the caller answer that question for
itself. Own one throwaway run, PATCH the victim's audiobook URL into your own
output, and the proxy streams the victim's book with every other gate satisfied.

These are the pure-function tests; ``tests/api/routes/test_runs.py`` drives the
same exploit through the real routes against a real DB.
"""

from __future__ import annotations

from unittest.mock import patch
from uuid import UUID

import pytest

from src.services.artifact_ref_guard import (
    ArtifactRefError,
    assert_refs_owned,
    is_own_tenant_upload,
    normalize_stored_ref,
)

_OWN_BUCKET = "w2v-orchestrator"
# Fixed, NOT uuid4(): both ids appear inside parametrize node ids below, so a
# value generated at import time makes every pytest-xdist worker collect a
# different set of test ids and aborts the whole run with "Different tests were
# collected between gw0 and gwN" before a single test executes.
_TENANT = UUID("00000000-0000-4000-8000-00000000a001")
_OTHER_TENANT = UUID("00000000-0000-4000-8000-00000000a002")

_VICTIM_RUN = "00000000-0000-0000-0000-0000000000ff"
_VICTIM_URL = (
    f"http://minio.test:8389/w2v-audiobooks/artifacts/{_VICTIM_RUN}/audiobook_stitch/book.mp3"
)
_OWNED_URL = "http://minio.test:8389/w2v-audiobooks/artifacts/mine/audiobook_stitch/book.mp3"
_OWNED_REFS = {"w2v-audiobooks/artifacts/mine/audiobook_stitch/book.mp3"}

# What POST /uploads/thumbnail returns: tenant-scoped, in the OWN bucket, and
# deliberately outside any run (the upload precedes the PATCH that adopts it).
_UPLOAD_URL = (
    f"http://minio.test:8389/{_OWN_BUCKET}/artifacts/thumbnails/{_TENANT}"
    "/course_thumbnail_override/abc.png"
)


@pytest.fixture(autouse=True)
def own_bucket():
    with patch("src.services.artifact_ref_guard.get_settings") as mock_settings:
        mock_settings.return_value.storage.minio_bucket = _OWN_BUCKET
        yield


def _assert(payload: object, *, tenant=_TENANT, owned=None) -> None:
    assert_refs_owned(payload, tenant_id=tenant, owned_refs=_OWNED_REFS if owned is None else owned)


# ---------------------------------------------------------------------------
# The exploit
# ---------------------------------------------------------------------------


def test_foreign_run_artifact_url_is_refused() -> None:
    """THE bug: writing another run's object into your own output must not store.

    Mutation killed: removing the ``assert_refs_owned`` call from any write route
    (or making it return unconditionally) — this ref would be persisted, the
    proxy's ``_run_artifact_refs`` would then return it, and the victim's
    audiobook streams.
    """
    with pytest.raises(ArtifactRefError, match="not owned by this run"):
        _assert({"artifact_url": _VICTIM_URL})


def test_foreign_ref_is_caught_however_deeply_it_is_buried() -> None:
    """The patch body is free-form ``dict``; the ref can sit anywhere in it.

    Mutation killed: validating only top-level string values instead of walking
    the payload with the same collector the proxy uses.
    """
    with pytest.raises(ArtifactRefError):
        _assert({"clips": [{"variants": [{"url": _VICTIM_URL}]}]})


def test_a_bare_string_payload_is_validated() -> None:
    """The thumbnail route hands a bare ``str``, not a dict.

    Mutation killed: an ``isinstance(payload, dict)`` early-return in the guard —
    the entire course-thumbnail route would stop being checked.
    """
    with pytest.raises(ArtifactRefError):
        _assert(_VICTIM_URL)


def test_own_tenant_upload_of_a_different_tenant_is_refused() -> None:
    """The upload arm is scoped to the CALLER's tenant, matched as a segment.

    Mutation killed: comparing with ``startswith`` or dropping the tenant check —
    ``artifacts/thumbnails/{someone-else}/…`` would be adoptable.
    """
    with pytest.raises(ArtifactRefError):
        _assert(_UPLOAD_URL, tenant=_OTHER_TENANT)


def test_tenant_prefixed_lookalike_segment_is_refused() -> None:
    """``{tenant}evil`` shares a string prefix, not a path segment."""
    lookalike = _UPLOAD_URL.replace(f"/{_TENANT}/", f"/{_TENANT}evil/")
    with pytest.raises(ArtifactRefError):
        _assert(lookalike)


def test_upload_arm_requires_the_orchestrators_own_bucket() -> None:
    """A tenant-shaped key in a SIBLING w2v bucket is not our upload namespace.

    ``is_readable_w2v_bucket`` lets the proxy read every ``w2v-*`` bucket, so
    without the own-bucket condition the upload arm would authorise a
    tenant-shaped key in any of them — buckets other services write.

    Mutation killed: dropping the ``bucket != minio_bucket`` test.
    """
    foreign_bucket = _UPLOAD_URL.replace(f"/{_OWN_BUCKET}/", "/w2v-compiled-videos/")
    with pytest.raises(ArtifactRefError):
        _assert(foreign_bucket)


def test_unnormalisable_ref_is_refused_not_skipped() -> None:
    """Fail closed: a ref we cannot normalise must not be quietly waved through.

    Mutation killed: ``if normalized is None: continue`` — a caller could store a
    traversing ref that some downstream consumer decodes differently than we do.
    """
    with pytest.raises(ArtifactRefError, match="unusable artifact reference"):
        _assert({"url": "http://minio.test:8389/w2v-audiobooks/../../etc/passwd"})


# ---------------------------------------------------------------------------
# What must keep working
# ---------------------------------------------------------------------------


def test_re_pointing_at_an_artifact_the_run_already_owns_is_allowed() -> None:
    """Legitimate element edits re-point at the run's own artifacts."""
    _assert({"artifact_url": _OWNED_URL, "text": "new caption"})


def test_this_tenants_own_staged_upload_is_allowed() -> None:
    """The thumbnail flow: upload (outside any run) then PATCH it onto the stitch.

    Mutation killed: dropping the ``is_own_tenant_upload`` arm — the product's
    own "set a custom course thumbnail" flow would 422 on every legitimate use.
    """
    _assert(_UPLOAD_URL)


def test_non_minio_values_are_not_policed() -> None:
    """Only ``minio.test`` URLs are what the proxy's read gate honours.

    A proxy path naming ANOTHER run escalates nothing — fetching it is still
    gated by that run's own ownership check — and external URLs / plain text are
    ordinary content. Policing them would break every normal patch.
    """
    _assert(
        {
            "text": "just words",
            "external": "https://cdn.example.com/img.png",
            "already_proxied": f"/api/runs/{_VICTIM_RUN}/artifact/w2v-audiobooks/x.mp3",
            "wrong_host": "http://minio-other.test:8389/w2v-audiobooks/x.mp3",
            "count": 3,
            "nothing": None,
        }
    )


def test_empty_payload_and_empty_owned_set_are_fine() -> None:
    """A run with no artifacts yet still accepts a patch that names none."""
    _assert({"text": "x"}, owned=set())


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ref", "expected"),
    [(f"{_OWN_BUCKET}/artifacts/thumbnails/{_TENANT}/course_thumbnail_override/a.png", True)],
)
def test_is_own_tenant_upload(ref: str, expected: bool) -> None:
    assert is_own_tenant_upload(ref, _TENANT) is expected


def test_normalize_stored_ref_drops_junk_instead_of_raising() -> None:
    """A single junk value must not poison a whole run's allowlist."""
    assert normalize_stored_ref("w2v-audiobooks/../../etc/passwd") is None
    assert normalize_stored_ref("w2v-audiobooks/a.mp3") == "w2v-audiobooks/a.mp3"
