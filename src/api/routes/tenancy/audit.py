"""Tenant-scoped audit view (P6 ORCH-5).

- GET /api/tenant/audit → the ``admin_audit_log`` rows whose TARGET is the effective
  tenant, newest first, keyset paged. ``TenantAdminDep`` + ``LICENSE_GATE_ONLY`` + the
  RLS bind to the authorized effective tenant (the tenant-admin tier posture, same as
  ``/api/tenant/connections``). ``admin_audit_log`` has no RLS policy of its own, so
  the explicit ``target_tenant_id`` predicate in the repository IS the isolation
  boundary; the bind keeps the roster lookup (``tenant_members``) on the same tenant.

Who is named (blueprint ORCH-5: super_admin actions on the tenant show as "platform
operator"): an actor is named only when they are on the tenant's roster and the action
is one a tenant member can take (``member_actor_clause``). Everyone else — a
super_admin scope-switched in, a partner, the platform — is ``platform_operator`` with
no subject id. The ``actor`` filter only matches named rows, so it cannot be used to
learn which rows an operator wrote.

``detail`` is already names-only at every writer (slot/kind/version/fingerprint, never
a secret value, see ``connections.py`` / ``youtube._audit_accepted``). As a second
line, keys that name secret material are dropped on every row, and keys that point at
another tenant or at platform internals (``from_tenant``, ``partner_licence_id``) are
dropped from operator rows.

Not shown: ``request_audit`` (per-request access log) — the blueprint names only
``admin_audit_log``, and request rows carry raw paths/ips of operator traffic.
"""

from __future__ import annotations

import base64
import binascii
import datetime
import json
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from fastapi import params as fastapi_params
from sqlalchemy.engine import Row

from src.api.routes._common import LICENSE_GATE_ONLY, AuditRepoDep, TenantAdminDep
from src.api.routes.tenancy.connections import bind_admin_tenant_rls
from src.api.routes.tenancy.errors import TenancyError
from src.api.schemas.tenant_audit import (
    PLATFORM_OPERATOR_LABEL,
    ActorKind,
    AuditAction,
    TenantAuditEntry,
    TenantAuditPage,
)

router = APIRouter(prefix="/api/tenant/audit", tags=["tenancy"])

#: Hard page-size ceiling.
MAX_PAGE_SIZE = 100
#: Widest ``since``..``until`` window a single query may ask for.
MAX_WINDOW = datetime.timedelta(days=366)

#: A ``detail`` key containing any of these is dropped on every row (defence in depth;
#: no writer stores one today). Matched against the key lower-cased with every
#: non-alphanumeric removed, so ``privateKey``, ``x-api-key``, ``client.secret`` and
#: ``aws_access_key`` all match — the parts are written in that normal form.
_SECRET_KEY_PARTS = (
    "secret",
    "password",
    "passwd",
    "token",
    "credential",
    "ciphertext",
    "privatekey",
    "apikey",
    "accesskey",
    "authorization",
    "bearer",
    "cookie",
    "jwt",
)
#: ``detail`` keys dropped from platform-operator rows: another tenant, platform internals.
_OPERATOR_ONLY_KEYS = frozenset({"from_tenant", "partner_licence_id"})

_ADMIN_GATE: list[fastapi_params.Depends] = [
    *LICENSE_GATE_ONLY,
    Depends(bind_admin_tenant_rls),
]


def _bad_request(code: str, message: str) -> TenancyError:
    return TenancyError(status.HTTP_422_UNPROCESSABLE_CONTENT, code, message, None)


def encode_cursor(created_at: datetime.datetime, row_id: UUID) -> str:
    raw = json.dumps({"t": created_at.isoformat(), "i": str(row_id)}).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str) -> tuple[datetime.datetime, UUID]:
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        data = json.loads(raw)
        created_at = datetime.datetime.fromisoformat(data["t"])
        row_id = UUID(data["i"])
    except (binascii.Error, ValueError, TypeError, KeyError, UnicodeDecodeError) as exc:
        raise _bad_request("INVALID_CURSOR", "cursor is not one this endpoint issued") from exc
    if created_at.tzinfo is None:
        raise _bad_request("INVALID_CURSOR", "cursor is not one this endpoint issued")
    return created_at, row_id


def _is_secret_key(key: str) -> bool:
    normal = "".join(c for c in key.lower() if c.isalnum())
    return any(part in normal for part in _SECRET_KEY_PARTS)


def scrub_detail(value: object, *, operator: bool) -> object:
    """``detail`` as a tenant admin may see it (module doc). Recurses into nested
    dicts/lists so a secret-named key cannot hide one level down."""
    if isinstance(value, dict):
        return {
            k: scrub_detail(v, operator=operator)
            for k, v in value.items()
            if not _is_secret_key(str(k)) and not (operator and k in _OPERATOR_ONLY_KEYS)
        }
    if isinstance(value, list):
        return [scrub_detail(v, operator=operator) for v in value]
    return value


def _scrub_top(detail: dict[str, object], *, operator: bool) -> dict[str, object]:
    scrubbed = scrub_detail(detail, operator=operator)
    return scrubbed if isinstance(scrubbed, dict) else {}


def _entry(row: Row[Any]) -> TenantAuditEntry:
    member = bool(row.is_member)
    kind: ActorKind = "member" if member else "platform_operator"
    return TenantAuditEntry(
        id=row.id,
        created_at=row.created_at,
        actor_subject_id=row.actor_subject_id if member else None,
        actor_kind=kind,
        actor_label=row.member_email if member else PLATFORM_OPERATOR_LABEL,
        action=row.action,
        target_tenant_id=row.target_tenant_id,
        detail=_scrub_top(row.detail or {}, operator=not member),
    )


@router.get(
    "",
    response_model=TenantAuditPage,
    operation_id="tenant.audit.list",
    dependencies=_ADMIN_GATE,
)
async def list_tenant_audit(
    tenant_id: TenantAdminDep,
    audit: AuditRepoDep,
    limit: int = Query(50, ge=1, le=MAX_PAGE_SIZE),
    cursor: str | None = Query(None, max_length=256),
    action: AuditAction | None = Query(None),
    actor: UUID | None = Query(None, description="A member's subject id."),
    actor_kind: ActorKind | None = Query(None),
    since: datetime.datetime | None = Query(None, description="Inclusive, timezone-aware."),
    until: datetime.datetime | None = Query(None, description="Exclusive, timezone-aware."),
) -> TenantAuditPage:
    """This tenant's admin audit trail, newest first. Tenant-admin only."""
    for name, value in (("since", since), ("until", until)):
        if value is not None and value.tzinfo is None:
            raise _bad_request("NAIVE_DATETIME", f"{name} must carry a timezone offset")
    if since is not None and until is not None:
        if until <= since:
            raise _bad_request("INVALID_RANGE", "until must be after since")
        if until - since > MAX_WINDOW:
            raise _bad_request(
                "RANGE_TOO_WIDE", f"since..until may span at most {MAX_WINDOW.days} days"
            )
    before = decode_cursor(cursor) if cursor is not None else None
    rows = await audit.list_tenant_events(
        tenant_id=tenant_id,
        limit=limit + 1,
        before=before,
        action=action,
        actor_subject_id=actor,
        actor_kind=actor_kind,
        since=since,
        until=until,
    )
    page = rows[:limit]
    next_cursor = (
        encode_cursor(page[-1].created_at, page[-1].id) if len(rows) > limit and page else None
    )
    return TenantAuditPage(items=[_entry(r) for r in page], limit=limit, next_cursor=next_cursor)
