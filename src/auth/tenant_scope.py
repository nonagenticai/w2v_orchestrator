"""Normalise an ambient tenant id into a *scoping key*, or refuse it.

w2v-common 0.67.0 made the identity ContextVar carry ``uuid.UUID | None``
instead of ``str`` / ``""``. Three orchestrator subsystems key **shared
storage** off that value and each needs a ``str``:

* the sound-catalog library cache (``_LIB_CACHE`` + ``tenants/<t>/manifest.json``),
* the cross-run media cache (``tts_storage._cross_run_tenant``),
* the ``X-Owner-Tenant-Id`` header the youtube_upload_service scopes upload
  jobs by.

For all three, "no usable tenant" must mean *do not scope* -- never *scope
everyone into one shared bucket*.

The nil uuid is therefore rejected **by value**, in every spelling
(``uuid.UUID.int == 0`` catches ``00000000-0000-0000-0000-000000000000``,
the braced/urn forms, and the unhyphenated ``32 * "0"`` alike). This is not
defensive decoration: before 0.67.0 the nil uuid *was* the configured
``auth.default_tenant_id`` fallback, so it genuinely reached these call sites.
It is exactly the value that must not become a scoping key --
``tenants/00000000-.../`` is a storage prefix EVERY unattributed caller would
share, and an ``X-Owner-Tenant-Id`` of nil is the sentinel that "can never
match a real caller", the documented cause of 35 of 48 prod ``upload_jobs``
rows that 404 forever. A ``None``-only guard leaves that hole wide open,
because the nil uuid is perfectly truthy.

A malformed / non-uuid tenant is also refused rather than used verbatim: an
unparseable value is not an authorization boundary, and degrading it to
"unscoped" is strictly safer than minting a cache key out of it.

This helper is for cache/storage/header *scoping* only. Request-handling code
that must not proceed without a tenant should raise instead -- use
``w2v_common.tenant.require_tenant_id``.
"""

from __future__ import annotations

from w2v_common.tenant import TenantError, parse_tenant_id

__all__ = ["tenant_scope_key"]


def tenant_scope_key(value: object) -> str | None:
    """Return ``value`` as a canonical tenant string, or ``None`` if unusable.

    ``None`` is returned for: an absent tenant, an empty/blank string, a value
    that is not a valid uuid, and the nil uuid. Callers treat ``None`` as
    "fall back to unscoped / per-run behaviour".
    """
    try:
        parsed = parse_tenant_id(value)
    except TenantError:
        return None
    if parsed.int == 0:
        return None
    return str(parsed)
