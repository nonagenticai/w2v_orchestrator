"""P6 ORCH-5 schemas: the tenant-scoped audit view (``GET /api/tenant/audit``).

An entry is the ``/api/admin/audit`` serializer (``AdminAuditEntry``) plus who did it,
as the tenant is allowed to see it: a member of the tenant is named; anyone else (a
super_admin acting into the tenant, a partner, the platform itself) is the
``platform operator`` and carries no subject id.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from ._shared import _camel
from .admin import AdminAuditEntry

#: Every ``admin_audit_log.action`` value — must equal the ``ck_admin_audit_action``
#: CHECK set (``models.AdminAuditLog``); ``tests/unit/test_tenant_audit_vocabulary.py``
#: fails when they drift.
AuditAction = Literal[
    "tenant.update",
    "scope.switch",
    "scope.reset",
    "member.grant_role",
    "member.revoke_role",
    "member.invite",
    "member.remove",
    "member.set_role",
    "invite.revoke",
    "apikey.create",
    "apikey.revoke",
    "tenant.create",
    "connection.set",
    "connection.clear_secret",
    "connection.delete",
    "connection.test",
    "connection.test_failed",
    "youtube.oauth_client.set",
    "youtube.oauth_client.clear",
    "youtube.platform_channel.set",
    "youtube.tenant_policy.set",
    "defaults.update",
    "music.upload",
    "music.update",
    "music.delete",
    "offboarding.request",
    "offboarding.export",
    "offboarding.delete",
    "offboarding.reject",
    "connection.bulk_set",
    "tenant.partner_bind",
    "tenant.licence_token_set",
    "run.cancel",
    "signup.remove",
]

ActorKind = Literal["member", "platform_operator"]

PLATFORM_OPERATOR_LABEL = "platform operator"


class TenantAuditEntry(AdminAuditEntry):
    #: ``member``: the actor is (or was) a member of this tenant and is named.
    #: ``platform_operator``: ``actor_subject_id`` is withheld (``None``).
    actor_kind: ActorKind
    #: The member's email when known, else ``None``; ``"platform operator"`` otherwise.
    actor_label: str | None = None


class TenantAuditPage(BaseModel):
    model_config = _camel
    items: list[TenantAuditEntry]
    limit: int
    #: Opaque keyset cursor for the next (older) page; ``None`` on the last page.
    next_cursor: str | None = None
