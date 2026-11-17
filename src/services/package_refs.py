"""Immutable, tenant-bound recording-package references for login runs.

A showcase/demo_video brief that carries a LOGIN hands the tenant's credentials to
whatever code the recording package contains — the package's ``auth.setup.ts``
types them. So for a login run the package must be code the orchestrator itself
stored for THIS tenant, pinned by content hash, never a URL that can be swapped:

    uploads://package/<upload_id>@sha256:<64 hex>

``POST /api/uploads/package`` returns it as ``package_ref``. At record time
:func:`resolve_package_ref` reads the object from the run tenant's own namespace
(so another tenant's upload id resolves to nothing), verifies the sha256, and only
then mints the short-lived URL the test runner fetches. An external
``package_url`` (or a presigned URL — it names a key, not content) is refused for
any brief with a login (422 at the door, a hard stage failure at record time).
"""

from __future__ import annotations

import hashlib
import re

from src.services.auth_secrets import AUTH_BLOCK_KEYS, auth_entries
from src.services.tts_storage import PACKAGE_STAGE, package_namespace, storage_for_packages

_REF = re.compile(
    r"uploads://package/(?P<id>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
    r"@sha256:(?P<sha>[0-9a-f]{64})"
)


class PackageRefError(RuntimeError):
    """A package reference that cannot be trusted with a login."""


def make_package_ref(upload_id: str, data: bytes) -> str:
    return f"uploads://package/{upload_id}@sha256:{hashlib.sha256(data).hexdigest()}"


def parse_package_ref(raw: object) -> tuple[str, str] | None:
    """``(upload_id, sha256)`` of a well-formed package ref, else None."""
    if not isinstance(raw, str):
        return None
    match = _REF.fullmatch(raw)
    return (match["id"], match["sha"]) if match else None


def brief_has_login(doc: object) -> bool:
    """True when any auth block anywhere in ``doc`` holds at least one login."""
    if isinstance(doc, list):
        return any(brief_has_login(item) for item in doc)
    if not isinstance(doc, dict):
        return False
    if any(auth_entries(doc.get(k)) for k in AUTH_BLOCK_KEYS):
        return True
    return any(brief_has_login(v) for k, v in doc.items() if k not in AUTH_BLOCK_KEYS)


def require_uploaded_package_for_logins(brief: dict) -> None:
    """Door check (ValueError ⇒ 422): a login brief's package must be a package ref."""
    package = brief.get("package_url")
    if package and brief_has_login(brief) and parse_package_ref(package) is None:
        raise ValueError(
            "a brief with a login must reference an orchestrator-uploaded package by "
            "its immutable package_ref (uploads://package/<id>@sha256:<hash> from POST "
            "/api/uploads/package); an external package_url would receive the credentials"
        )


async def resolve_package_ref(ref: str, tenant_id: object) -> str:
    """Verify ``ref`` against the run tenant's own stored upload; return a fetch URL.

    Raises :class:`PackageRefError` for a malformed ref, an upload absent from THIS
    tenant's namespace (another tenant's id included), or content whose sha256 no
    longer matches — never falls back to anything else."""
    parsed = parse_package_ref(ref)
    if parsed is None or tenant_id is None:
        raise PackageRefError("not a tenant-bound package_ref")
    upload_id, sha = parsed
    storage = await storage_for_packages()
    namespace = package_namespace(str(tenant_id))
    filename = f"{upload_id}.zip"
    data = await storage.retrieve_stage_artifact(namespace, PACKAGE_STAGE, filename)
    if data is None:
        raise PackageRefError(f"package {upload_id} was not uploaded by this tenant")
    if hashlib.sha256(data).hexdigest() != sha:
        raise PackageRefError(f"package {upload_id} does not match its pinned sha256")
    url = await storage.stage_artifact_url(namespace, PACKAGE_STAGE, filename)
    if not url:
        raise PackageRefError(f"package {upload_id} has no fetchable URL")
    return url
