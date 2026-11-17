"""SQLAlchemy ORM models for w2v_orchestrator (Studio v3)."""

from __future__ import annotations

import copy
import datetime
from collections.abc import Callable
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    Numeric,
    Text,
    UniqueConstraint,
    event,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID  # noqa: N811
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from src.services.auth_secrets import has_plaintext_secret, seal_auth_secrets


class Base(DeclarativeBase):
    pass


#: ``tenants.status`` values an OPERATOR set: they block access, and the licence
#: refresh never overwrites them. ``deleted`` is the offboarding tombstone (0085) —
#: the row stays so the id can never be re-provisioned or re-adopted.
OPERATOR_HELD_TENANT_STATUSES: tuple[str, ...] = ("suspended", "deleted")


class Tenant(Base):
    """Canonical tenant registry — a name + status annotation layer.

    Keyed on the claim-derived ``tenant_id`` (NOT minted here; w2v is downstream
    of licenses.test / Keycloak). Backfilled from ``projects.tenant_id`` in
    migration 0021. ``name`` is an optional human label (UUID shown until set);
    ``status`` blocks a super-admin scope-switch into a suspended tenant.

    P2 tenancy (migration 0072) adds the licence + provenance columns:
    ``status`` gains ``licence_lapsed`` (written by the licence refresh);
    ``licence_ref``/``licence_checked_at`` record the entitling w2v licence;
    ``created_via`` ∈ {super_admin, partner, backfill} (pre-P2 rows =
    ``backfill``); ``partner_licence_id`` is the partner-key licence that
    created the tenant.

    Licence introspection (migration 0089): ``licence_token_ciphertext`` is the
    tenant's OWN w2v licence token, AES-GCM sealed with the provider keyring under
    AAD ``("tenant-licence", tenant_id)`` (``licence_token_kid`` = the sealing key,
    ``licence_token_set_at`` = when it was stored, ``licence_token_expires_at`` =
    the token's own expiry — an expired stored token counts as none). Entitlement is the
    introspection of that token. Never returned by any API, never logged.
    ``licence_op_id`` / ``licence_op_until`` are the per-tenant licence-change
    LEASE (one conditional UPDATE, no lock held across the licences calls).
    """

    __tablename__ = "tenants"
    __table_args__ = (
        CheckConstraint(
            "status IN ('active','suspended','licence_lapsed','deleted')",
            name="ck_tenants_status",
        ),
        CheckConstraint(
            "created_via IN ('super_admin','partner','backfill')", name="ck_tenants_created_via"
        ),
        CheckConstraint(
            "((licence_token_ciphertext IS NULL) = (licence_token_kid IS NULL)) "
            "AND ((licence_token_ciphertext IS NULL) = (licence_token_set_at IS NULL))",
            name="ck_tenants_licence_token_complete",
        ),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    name: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="active")
    licence_ref: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    licence_checked_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_via: Mapped[str] = mapped_column(Text, nullable=False, server_default="super_admin")
    partner_licence_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    licence_token_ciphertext: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    licence_token_kid: Mapped[str | None] = mapped_column(Text, nullable=True)
    licence_token_set_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    licence_token_expires_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    licence_op_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    licence_op_until: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    def __repr__(self) -> str:  # never render the sealed token
        return f"Tenant(id={self.id!s}, status={self.status!r})"


class AdminAuditLog(Base):
    """Durable record of deliberate admin actions (Phase B, 4th increment).

    One row per *deliberate* action (a tenant rename/suspend, or a super-admin
    scope-switch recorded on the UI toggle) — NOT one per request. ``action`` is
    a small controlled vocabulary; ``detail`` carries action-specific payload
    (before/after for an update, from-tenant for a switch). Written inside the
    acting super-admin route's session, so the audit row commits atomically with
    the action it records. See migration 0022.
    """

    __tablename__ = "admin_audit_log"
    __table_args__ = (
        CheckConstraint(
            "action IN ('tenant.update','scope.switch','scope.reset',"
            "'member.grant_role','member.revoke_role',"
            "'member.invite','member.remove','member.set_role','invite.revoke',"
            "'apikey.create','apikey.revoke','tenant.create',"
            "'connection.set','connection.clear_secret','connection.delete',"
            "'connection.test','connection.test_failed',"
            "'youtube.oauth_client.set','youtube.oauth_client.clear',"
            "'youtube.platform_channel.set','youtube.tenant_policy.set',"
            "'defaults.update','music.upload','music.update','music.delete',"
            "'offboarding.request','offboarding.export','offboarding.delete',"
            "'offboarding.reject','connection.bulk_set','tenant.partner_bind',"
            "'tenant.licence_token_set','run.cancel','signup.remove')",
            name="ck_admin_audit_action",
        ),
        Index("idx_admin_audit_created", text("created_at DESC")),  # DESC mirrors migration 0022
        Index("idx_admin_audit_tenant", "target_tenant_id"),
        Index("idx_admin_audit_actor", "actor_subject_id"),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    actor_subject_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    action: Mapped[str] = mapped_column(Text, nullable=False)
    target_tenant_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    detail: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False, default=dict)


class TenantMember(Base):
    """Per-tenant roster of identities that have signed in (Phase C, C-1).

    Adopts fr's DB-backed roster model (``user_roles``): one row per
    ``(subject_id, tenant_id)``, upserted best-effort on ``GET /api/auth/me`` so
    the tenant Members screen reads a real roster instead of "observed authors".
    ``roles`` snapshots the realm roles seen at last login. Not minted/owned here
    (identity is downstream of licenses.test / Keycloak); no FK to ``tenants``.
    See migration 0024.

    P2 tenancy (migration 0072) makes this table the membership AUTHORITY:
    ``role`` ∈ {member, tenant_admin} is the tenant-scoped role (replacing the
    realm-wide Keycloak roles for w2v), ``status`` ∈ {active, removed} with
    ``removed_at`` set iff removed, plus ``invited_by``/``created_at``. ``roles``
    stays as the observed realm-role snapshot. The partial index on
    ``subject_id`` WHERE active serves "sub → active memberships".
    """

    __tablename__ = "tenant_members"
    __table_args__ = (
        Index("idx_tenant_members_tenant", "tenant_id"),
        Index(
            "idx_tenant_members_subject_active",
            "subject_id",
            postgresql_where=text("status = 'active'"),
        ),
        CheckConstraint("role IN ('member','tenant_admin')", name="ck_tenant_members_role"),
        CheckConstraint("status IN ('active','removed')", name="ck_tenant_members_status"),
        CheckConstraint(
            "(status = 'removed') = (removed_at IS NOT NULL)",
            name="ck_tenant_members_removed_at",
        ),
    )

    subject_id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    tenant_id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    email: Mapped[str | None] = mapped_column(Text, nullable=True)
    display_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    roles: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    first_seen_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_seen_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    role: Mapped[str] = mapped_column(Text, nullable=False, server_default="member")
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="active")
    invited_by: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    removed_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class TenantInvite(Base):
    """A pending/settled invitation of an email address into a tenant (P2, 0073).

    Only the SHA-256 of the one-time token is stored (``token_hash``). At most one
    PENDING invite per ``(tenant_id, lower(email))``; a resend rotates the token on
    the same row. ``accepted_subject_id`` is set iff ``status = 'accepted'``. No
    RLS — acceptance looks invites up across tenants by token hash / email.
    """

    __tablename__ = "tenant_invites"
    __table_args__ = (
        UniqueConstraint("token_hash", name="uq_tenant_invites_token_hash"),
        CheckConstraint("role IN ('member','tenant_admin')", name="ck_tenant_invites_role"),
        CheckConstraint(
            "status IN ('pending','accepted','revoked','expired')",
            name="ck_tenant_invites_status",
        ),
        CheckConstraint(
            "(status = 'accepted') = (accepted_subject_id IS NOT NULL)",
            name="ck_tenant_invites_accepted",
        ),
        CheckConstraint("octet_length(token_hash) = 32", name="ck_tenant_invites_token_hash_len"),
        Index("idx_tenant_invites_tenant", "tenant_id"),
        Index(
            "uq_tenant_invites_pending_email",
            "tenant_id",
            text("lower(email)"),
            unique=True,
            postgresql_where=text("status = 'pending'"),
        ),
        Index(
            "idx_tenant_invites_pending_email",
            text("lower(email)"),
            postgresql_where=text("status = 'pending'"),
        ),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    tenant_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    email: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False, server_default="member")
    token_hash: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    expires_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    invited_by: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    kc_user_created: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    accepted_subject_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    accepted_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    resent_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class SignupApproval(Base):
    """An early-access signup request / super_admin invite (migration 0098).

    ``pending`` → ``approved`` (a tenant is founded for it; ``steps`` records the
    provisioning, retried by the tick) or ``rejected`` (nothing was created);
    ``approved`` → ``removed`` (0100: the super_admin removal undoes the approve,
    recorded in ``removal``). At
    most ONE open (pending or approved) row per ``lower(email)``: one workspace per person. No RLS: a platform table read by
    super_admin surfaces and the service doors only. See
    ``src/services/signup/core.py``.
    """

    __tablename__ = "signup_approvals"
    __table_args__ = (
        CheckConstraint(
            "message IS NULL OR char_length(message) <= 2000",
            name="ck_signup_approvals_message_len",
        ),
        CheckConstraint(
            "locale IS NULL OR locale IN ('ar','de','en','es','fr','he','nl','pl')",
            name="ck_signup_approvals_locale",
        ),
        CheckConstraint("source IN ('web','chat','invite')", name="ck_signup_approvals_source"),
        CheckConstraint(
            "status IN ('pending','approved','rejected','removed')",
            name="ck_signup_approvals_status",
        ),
        CheckConstraint(
            "status <> 'removed' OR removal IS NOT NULL", name="ck_signup_approvals_removal"
        ),
        CheckConstraint(
            "decided_via IS NULL OR decided_via IN ('onboarding','telegram')",
            name="ck_signup_approvals_decided_via",
        ),
        CheckConstraint(
            "(status = 'pending') = (decided_at IS NULL)", name="ck_signup_approvals_decided"
        ),
        CheckConstraint(
            "status <> 'approved' OR (tenant_id IS NOT NULL AND tenant_slug IS NOT NULL)",
            name="ck_signup_approvals_approved_tenant",
        ),
        CheckConstraint(
            "tenant_slug IS NULL OR tenant_slug ~ '^[a-z0-9-]{2,40}$'",
            name="ck_signup_approvals_tenant_slug",
        ),
        CheckConstraint("steps_attempts >= 0", name="ck_signup_approvals_steps_attempts"),
        CheckConstraint(
            "llm_access IS NULL OR llm_access IN ('local_allowed','byo_required','platform_only')",
            name="ck_signup_approvals_llm_access",
        ),
        Index(
            "uq_signup_approvals_open_email",
            text("lower(email)"),
            unique=True,
            # 0100: a removed row keeps the address until its cleanup is finished.
            postgresql_where=text(
                "status IN ('pending','approved') OR (status = 'removed' "
                "AND (removal->>'state') NOT IN ('done','resolved_manually'))"
            ),
        ),
        Index(
            "uq_signup_approvals_tenant_slug",
            "tenant_slug",
            unique=True,
            postgresql_where=text("tenant_slug IS NOT NULL"),
        ),
        Index("idx_signup_approvals_status_created", "status", "created_at"),
        Index(
            "idx_signup_approvals_steps_due",
            "steps_next_attempt_at",
            postgresql_where=text("steps_next_attempt_at IS NOT NULL"),
        ),
        Index(
            "idx_signup_approvals_removal_state",
            text("(removal->>'state')"),
            postgresql_where=text("removal IS NOT NULL"),
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    email: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    company: Mapped[str | None] = mapped_column(Text, nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    locale: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    intake_client: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    decided_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    decided_by: Mapped[str | None] = mapped_column(Text, nullable=True)
    decided_via: Mapped[str | None] = mapped_column(Text, nullable=True)
    decider_actor: Mapped[str | None] = mapped_column(Text, nullable=True)
    approving_until: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    approving_claim: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    tenant_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    tenant_slug: Mapped[str | None] = mapped_column(Text, nullable=True)
    tenant_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    #: The approver's LLM access choice for the new tenant (0099), recorded with
    #: the decision; NULL while nothing is decided.
    llm_access: Mapped[str | None] = mapped_column(Text, nullable=True)
    steps: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    steps_lease_until: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    steps_lease_token: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    steps_next_attempt_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    steps_attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    #: The early-access REMOVAL (0100): set when the row moves to ``removed``;
    #: ``{state, attempts, steps, blockers, last_error, ...}`` — see
    #: ``src/services/signup/removal.py``.
    removal: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class TenantApiKey(Base):
    """A tenant-scoped API key ``w2v_sk_<prefix>_<secret>`` (P2, 0074).

    ``prefix`` (8 lowercase base32 chars) is public and UNIQUE — the lookup key;
    ``key_hash`` is the SHA-256 of the FULL key (the secret is shown once and never
    stored). Format + verification live in ``src/auth/api_key_format.py``.

    RLS (0090): ENABLE + ``tenant_isolation`` on ``tenant_id`` like every
    direct-tenant table. A presented key is resolved BEFORE any tenant is known,
    so that one lookup goes through the ``resolve_api_key`` SECURITY DEFINER
    resolver (``src.database.rls``), never a plain SELECT.
    """

    __tablename__ = "tenant_api_keys"
    __table_args__ = (
        UniqueConstraint("prefix", name="uq_tenant_api_keys_prefix"),
        CheckConstraint("prefix ~ '^[a-z2-7]{8}$'", name="ck_tenant_api_keys_prefix"),
        CheckConstraint("octet_length(key_hash) = 32", name="ck_tenant_api_keys_key_hash_len"),
        # P8 (0086): the closed scope vocabulary (``API_KEY_SCOPES``).
        CheckConstraint(
            "scopes <@ ARRAY['connections:write']::text[]", name="ck_tenant_api_keys_scopes"
        ),
        Index("idx_tenant_api_keys_tenant", "tenant_id"),
        # P9 (0087): one LIVE partner-minted key per (tenant, name) — the partner
        # mint is idempotent by name and two concurrent mints cannot both land.
        Index(
            "uq_tenant_api_keys_partner_live_name",
            "tenant_id",
            "name",
            unique=True,
            postgresql_where=text("revoked_at IS NULL AND partner_licence_id IS NOT NULL"),
        ),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    tenant_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    prefix: Mapped[str] = mapped_column(Text, nullable=False)
    key_hash: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_by: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_used_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    expires_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    revoked_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    #: Extra grants beyond the ``member`` role a key always has (P8, 0086). Today
    #: only ``connections:write`` (bulk ``PUT /api/tenant/connections``).
    scopes: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, default=list, server_default=text("'{}'::text[]")
    )
    #: The partner licence that minted this key via the partner lane
    #: (``POST /api/tenants/{tenant_id}/api-keys``, P9, 0087); NULL = a tenant_admin
    #: minted it. A partner may revoke only the keys it minted.
    partner_licence_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)


class TenantConnection(Base):
    """A tenant's binding of one provider slot to its own connection (P3, 0077).

    ``current_version`` names the live :class:`TenantConnectionVersion`. Soft
    delete flips ``status`` to ``deleted`` and stamps ``deleted_at`` (the slot then
    falls back to the deployment default); at most one ``active`` row per
    ``(tenant_id, slot)`` (partial unique index). RLS ``tenant_isolation`` on
    ``tenant_id`` — registered for ``create_all`` by
    ``register_tenant_connections_rls_ddl``, frozen inline in migration 0077.
    """

    __tablename__ = "tenant_connections"
    __table_args__ = (
        CheckConstraint("status IN ('active','deleted')", name="ck_tenant_connections_status"),
        CheckConstraint(
            "(status = 'deleted') = (deleted_at IS NOT NULL)",
            name="ck_tenant_connections_deleted_at",
        ),
        CheckConstraint("current_version >= 1", name="ck_tenant_connections_current_version"),
        CheckConstraint(
            "slot ~ '^[a-z][a-z0-9_]*(\\.[a-z0-9_]+)*$'", name="ck_tenant_connections_slot"
        ),
        Index(
            "uq_tenant_connections_tenant_slot_active",
            "tenant_id",
            "slot",
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    tenant_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    slot: Mapped[str] = mapped_column(Text, nullable=False)
    current_version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="active")
    last_test_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_test_ok: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    last_test_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    updated_by: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
    deleted_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class TenantConnectionVersion(Base):
    """One immutable configuration of a :class:`TenantConnection` (P3, 0077).

    Runs pin ``(connection_id, version)``, so the config columns are never
    rewritten. Only lifecycle stamps (``superseded_at``/``purged_at``) and the
    secret pair change after insert: a rewrap re-seals under a newer key, a GC
    purge NULLs it. ``secret_ciphertext`` is ``v1|kid|nonce|ct+tag`` AES-256-GCM
    bound to tenant|connection|version (``src/services/providers/crypto.py``);
    ``secret_fingerprint`` is the 12-hex public fingerprint and survives a purge.
    No ``tenant_id``: RLS climbs one hop to the connection.
    """

    __tablename__ = "tenant_connection_versions"
    __table_args__ = (
        CheckConstraint("version >= 1", name="ck_tenant_connection_versions_version"),
        CheckConstraint(
            "(secret_ciphertext IS NULL) = (key_id IS NULL)",
            name="ck_tenant_connection_versions_secret_key",
        ),
        CheckConstraint(
            "purged_at IS NULL OR secret_ciphertext IS NULL",
            name="ck_tenant_connection_versions_purged",
        ),
        CheckConstraint(
            "secret_fingerprint IS NULL OR secret_fingerprint ~ '^[0-9a-f]{12}$'",
            name="ck_tenant_connection_versions_fingerprint",
        ),
        Index(
            "idx_tenant_connection_versions_purgeable",
            "superseded_at",
            postgresql_where=text("superseded_at IS NOT NULL AND secret_ciphertext IS NOT NULL"),
        ),
    )

    connection_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("tenant_connections.id", ondelete="CASCADE"),
        primary_key=True,
    )
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    base_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    model: Mapped[str | None] = mapped_column(Text, nullable=True)
    options: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    secret_ciphertext: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    key_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    secret_fingerprint: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    superseded_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    purged_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class TenantDefaults(Base):
    """A tenant's production defaults (P6 ORCH-1, migration 0081).

    One row per tenant. Every setting column is nullable and NULL means "deployment
    default"; a row whose settings are all NULL behaves exactly like no row. A run
    reads it once at creation (``build_config_snapshot``) and pins the values in its
    snapshot. ``version`` is the optimistic-concurrency counter of
    ``PUT /api/tenant/defaults``. The jsonb columns are ``none_as_null`` so a Python
    ``None`` is SQL NULL (a JSON ``null`` would fail the object CHECKs). RLS
    ``tenant_isolation`` on ``tenant_id``,
    registered for ``create_all`` by ``register_tenant_defaults_rls_ddl``, frozen
    inline in migration 0081.
    """

    __tablename__ = "tenant_defaults"
    __table_args__ = (
        CheckConstraint("version >= 1", name="ck_tenant_defaults_version"),
        CheckConstraint(
            "retention_mode IS NULL OR retention_mode IN ('auto','critique_only','off')",
            name="ck_tenant_defaults_retention_mode",
        ),
        CheckConstraint(
            "voice IS NULL OR jsonb_typeof(voice) = 'object'", name="ck_tenant_defaults_voice"
        ),
        CheckConstraint(
            "brand IS NULL OR jsonb_typeof(brand) = 'object'", name="ck_tenant_defaults_brand"
        ),
        CheckConstraint(
            "captions IS NULL OR jsonb_typeof(captions) = 'object'",
            name="ck_tenant_defaults_captions",
        ),
        CheckConstraint(
            "voice_direction IS NULL OR jsonb_typeof(voice_direction) = 'object'",
            name="ck_tenant_defaults_voice_direction",
        ),
        CheckConstraint(
            "model_tier IS NULL OR jsonb_typeof(model_tier) = 'object'",
            name="ck_tenant_defaults_model_tier",
        ),
    )

    tenant_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        primary_key=True,
    )
    voice: Mapped[dict[str, object] | None] = mapped_column(JSONB(none_as_null=True), nullable=True)
    brand: Mapped[dict[str, object] | None] = mapped_column(JSONB(none_as_null=True), nullable=True)
    captions: Mapped[dict[str, object] | None] = mapped_column(
        JSONB(none_as_null=True), nullable=True
    )
    voice_direction: Mapped[dict[str, object] | None] = mapped_column(
        JSONB(none_as_null=True), nullable=True
    )
    retention_mode: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_tier: Mapped[dict[str, object] | None] = mapped_column(
        JSONB(none_as_null=True), nullable=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    updated_by: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class TenantSoundAsset(Base):
    """One sound a tenant_admin uploaded into the tenant's music library (P6 ORCH-4,
    migration 0084).

    The source of truth for the tenant library; ``sound_library/tenants/{t}/manifest.json``
    is derived from these rows (``services.tenant_sound_library``). ``sound_id`` is
    always ``tenant_<slug>`` (CHECK) so it can never shadow a built-in/shared id, and
    ``object_key`` is CHECKed to sit under the row's own tenant prefix. ``deleted_at``
    is a soft delete: the row stays as a RETIRED manifest entry while any run of the
    tenant is active (a run that pinned it keeps resolving it), then the object and the
    row are purged. ``pinned_categories`` applies the same rule to a re-categorising
    PATCH. Rights are TENANT-DECLARED (``rights_attested_by``), never "verified
    royalty-free". RLS ``tenant_isolation`` on ``tenant_id``, registered for
    ``create_all`` by ``register_tenant_sound_assets_rls_ddl``, frozen inline in 0084.
    """

    __tablename__ = "tenant_sound_assets"
    __table_args__ = (
        UniqueConstraint("object_key", name="uq_tenant_sound_assets_object_key"),
        CheckConstraint(
            "category IN ('music','ambient','accent','transition')",
            name="ck_tenant_sound_assets_category",
        ),
        CheckConstraint(
            "sound_id ~ '^tenant_[a-z0-9_]{1,56}$'", name="ck_tenant_sound_assets_sound_id"
        ),
        CheckConstraint(
            "starts_with(object_key, 'sound_library/tenants/' || tenant_id::text || '/')",
            name="ck_tenant_sound_assets_object_key",
        ),
        CheckConstraint("duration_s > 0", name="ck_tenant_sound_assets_duration"),
        CheckConstraint("bytes > 0", name="ck_tenant_sound_assets_bytes"),
        CheckConstraint(
            "pinned_categories <@ ARRAY['music','ambient','accent','transition']::text[]",
            name="ck_tenant_sound_assets_pinned_categories",
        ),
        Index(
            "uq_tenant_sound_assets_live_sound_id",
            "tenant_id",
            "sound_id",
            unique=True,
            postgresql_where=text("deleted_at IS NULL"),
        ),
        Index("idx_tenant_sound_assets_tenant", "tenant_id"),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    tenant_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    sound_id: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''"))
    object_key: Mapped[str] = mapped_column(Text, nullable=False)
    duration_s: Mapped[float] = mapped_column(Float, nullable=False)
    bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    content_type: Mapped[str] = mapped_column(Text, nullable=False)
    rights_attested_by: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    deleted_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    #: Categories the track held while a run of the tenant was active (kept across a
    #: re-categorising PATCH so a run that picked it keeps resolving it); cleared when
    #: no run is active. See ``services.tenant_sound_library``.
    pinned_categories: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, default=list, server_default=text("'{}'::text[]")
    )


class TenantOffboarding(Base):
    """One tenant export / delete job (P6 ORCH-6, migration 0085).

    A tenant_admin can only REQUEST (``status='requested'``); a super_admin approves
    (runs it), rejects, or starts one directly. ``artifact_key`` is the export
    archive's object key (CHECKed under ``exports/tenants/<tenant_id>/``) — a download
    URL is presigned per read, never stored. ``progress`` is the resumable-delete
    ledger; ``result`` the final report. At most one OPEN (requested|running) job per
    ``(tenant_id, kind)``. FK to ``tenants`` without cascade: a deleted tenant keeps
    its tombstone row and these job rows as the proof of deletion. No RLS (the
    super_admin queue reads across tenants). See ``services.tenant_offboarding``.
    """

    __tablename__ = "tenant_offboarding"
    __table_args__ = (
        CheckConstraint("kind IN ('export','delete')", name="ck_tenant_offboarding_kind"),
        CheckConstraint(
            "status IN ('requested','running','done','failed','rejected')",
            name="ck_tenant_offboarding_status",
        ),
        CheckConstraint(
            "artifact_key IS NULL "
            "OR starts_with(artifact_key, 'exports/tenants/' || tenant_id::text || '/')",
            name="ck_tenant_offboarding_artifact_key",
        ),
        Index(
            "uq_tenant_offboarding_open",
            "tenant_id",
            "kind",
            unique=True,
            postgresql_where=text("status IN ('requested','running')"),
        ),
        Index("idx_tenant_offboarding_tenant", "tenant_id", "created_at"),
        Index("idx_tenant_offboarding_status", "status", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    tenant_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False
    )
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="requested")
    requested_by: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    decided_by: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    force: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    artifact_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    progress: Mapped[dict[str, object]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    result: Mapped[dict[str, object] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    started_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class ProvisioningAudit(Base):
    """One row per super-admin provision attempt (admin tier Phase C-2).

    Adopts fr's ``provisioning_audit``: a ``pending`` row is written (and
    committed on its own) BEFORE the Keycloak/licenses side effects, then updated
    to ``success`` or ``partial`` — so a crash mid-flight always leaves a trail an
    operator can inspect (KC user + license may exist with no ``success`` row).
    ``payload`` is the inbound request; ``result`` the created ids (NOT the raw
    one-time secrets, which are returned to the caller and never stored). See
    migration 0025.
    """

    __tablename__ = "provisioning_audit"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','success','partial')",
            name="ck_provisioning_audit_status",
        ),
        Index("idx_provisioning_audit_created", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
    actor_subject_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    action: Mapped[str] = mapped_column(Text, nullable=False, server_default="provision")
    target_email: Mapped[str] = mapped_column(Text, nullable=False)
    target_subject_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    tenant_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    result: Mapped[dict[str, object] | None] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class RequestAudit(Base):
    """High-volume per-request security audit (middleware-wide audit).

    One row per *mutation* (POST/PUT/PATCH/DELETE), *auth failure* (401/403), or
    *cross-tenant access* (a super-admin acting into a foreign tenant via
    ``X-Active-Tenant-Id`` — recorded even on a GET, since a cross-tenant read is
    the most isolation-sensitive access). Successful same-tenant GETs are NOT
    recorded, bounding volume. Distinct from ``admin_audit_log`` (migration 0022):
    that table is one row per *deliberate* admin action with a controlled-vocab
    ``action`` written atomically inside the acting route; this is the firehose of
    raw request activity, written **fail-safe off the hot path** by
    ``AuditMiddleware`` (a failed insert never breaks the request). ``event`` is
    the trigger class; ``cross_tenant`` is an orthogonal dimension (a super-admin
    cross-tenant POST is ``mutation`` + ``cross_tenant=true``). See migration 0026.
    """

    __tablename__ = "request_audit"
    __table_args__ = (
        CheckConstraint(
            "event IN ('mutation','auth_failure','read')",
            name="ck_request_audit_event",
        ),
        Index("idx_request_audit_created", "created_at", "id"),
        Index("idx_request_audit_tenant_created", "effective_tenant_id", "created_at"),
        Index("idx_request_audit_actor", "actor_subject_id"),
        Index("idx_request_audit_event", "event"),
        # Partial index for the super-admin "cross-tenant only" filter
        # (GET /api/admin/request-audit?cross_tenant=true → list_events). Cross-
        # tenant rows are a tiny minority (legitimate scope-switches), so a plain
        # WHERE cross_tenant ORDER BY created_at DESC would scan many recent rows
        # to fill a page; this partial index stays small and serves it directly.
        Index(
            "idx_request_audit_cross_tenant",
            "created_at",
            "id",
            postgresql_where=text("cross_tenant"),
        ),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    actor_subject_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    claim_tenant_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    effective_tenant_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    cross_tenant: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    event: Mapped[str] = mapped_column(Text, nullable=False)
    method: Mapped[str] = mapped_column(Text, nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    resource: Mapped[str | None] = mapped_column(Text, nullable=True)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    ip: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    credential_fingerprint: Mapped[str | None] = mapped_column(Text, nullable=True)
    request_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    detail: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False, default=dict)


class Project(Base):
    __tablename__ = "projects"
    __table_args__ = (
        Index("idx_projects_tenant", "tenant_id", "status"),
        Index(
            "idx_projects_parent",
            "parent_project_id",
            "position",
            postgresql_where="parent_project_id IS NOT NULL",
        ),
        CheckConstraint(
            "visibility IN ('private','tenant','public')",
            name="ck_projects_visibility",
        ),
        CheckConstraint(
            "type IN ('promo','course','app_review','generic','lesson','website_review',"
            "'audiobook','web_design','web_design_v2','demo_video','showcase',"
            "'growth_bundle','seo_audit','cro_audit','marketing_scorecard',"
            "'action_reel')",
            name="ck_projects_type",
        ),
        CheckConstraint(
            "status IN ('drafting','producing','completed','failed','archived')",
            name="ck_projects_status",
        ),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    tenant_id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), nullable=False)
    owner_subject_id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), nullable=False)
    visibility: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    brief: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    brand: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False, default=dict)
    voice: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False, default=dict)
    theme: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False, default=dict)
    parent_project_id: Mapped[UUID | None] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="SET NULL"),
        nullable=True,
    )
    # A storyboard BRANCH is a real, hidden child project (migration 0052): its
    # panels/scenes/assets live in their own ``project_id`` namespace so the
    # existing DEFERRABLE ``uq_storyboard_panels_project_shot`` never collides and
    # the whole panel/scene/asset/snapshot/restore machinery is reused unchanged.
    # The branch↔root link lives in ``storyboard_branches`` — NOT in
    # ``parent_project_id`` (that is the course→lesson hierarchy;
    # ``ProjectRepository`` enforces "lesson must have a parent" and lists children
    # off it, so reusing it would surface every branch as a lesson-in-a-course).
    # Every user-facing projects LISTING filters ``is_branch = false`` so branches
    # do not pollute the Projects page.
    is_branch: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    position: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="drafting")
    archived_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Set when the project transitions to 'completed' (all deliverables done);
    # cleared if it ever leaves 'completed' (e.g. a regenerate run reopens it).
    finished_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    deliverables: Mapped[list[Deliverable]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    children: Mapped[list[Project]] = relationship(
        "Project", remote_side=lambda: [Project.parent_project_id]
    )
    verbatim_items: Mapped[list[VerbatimItem]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class Deliverable(Base):
    __tablename__ = "deliverables"
    __table_args__ = (
        Index("idx_deliverables_project", "project_id"),
        CheckConstraint(
            "format IN ('16x9','9x16','1x1','4x5','audio')",
            name="ck_deliverables_format",
        ),
        CheckConstraint(
            "captions_mode IN ('none','burned_in','overlay')",
            name="ck_deliverables_captions",
        ),
        CheckConstraint(
            "status IN ('idle','running','awaiting_review','completed','failed')",
            name="ck_deliverables_status",
        ),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    project_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    format: Mapped[str] = mapped_column(Text, nullable=False)
    captions_mode: Mapped[str] = mapped_column(Text, nullable=False, server_default="none")
    target_duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="idle")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    project: Mapped[Project] = relationship(back_populates="deliverables")
    runs: Mapped[list[Run]] = relationship(
        back_populates="deliverable", cascade="all, delete-orphan"
    )


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = (
        Index(
            "idx_runs_deliverable", "deliverable_id", text("created_at DESC")
        ),  # 2nd col DESC mirrors migration 0015
        Index(
            "idx_runs_parent",
            "parent_run_id",
            postgresql_where="parent_run_id IS NOT NULL",
        ),
        Index(
            "idx_runs_active",
            "status",
            postgresql_where="status IN ('pending','running','paused')",
        ),
        # Spawn-once guard for the demo_video structural-relief auto-convergence
        # regenerate (PR-C): at most one child run per (parent, relief pass), so
        # neither the two orchestrator replicas nor the inline gate + the reconciler
        # backstop can double-insert a child = a double paid render. Partial: only
        # relief-spawned children carry a non-NULL structural_relief_pass.
        Index(
            "uq_runs_parent_relief_pass",
            "parent_run_id",
            "structural_relief_pass",
            unique=True,
            postgresql_where="structural_relief_pass IS NOT NULL",
        ),
        CheckConstraint(
            "status IN ('pending','running','paused','completed','failed','cancelled')",
            name="ck_runs_status",
        ),
        # Fault attribution vocabulary — byte-for-byte the predicate in migration
        # 0066 (this repo's convention: copy, don't retype; the models-vs-
        # migrations CHECK parity guard compares the normalized predicates).
        CheckConstraint(
            "failure_class IS NULL OR failure_class IN "
            "('our_defect','customer_input','external','unknown')",
            name="ck_runs_failure_class",
        ),
        # A fault attribution is only meaningful for a run that actually failed.
        # Cheap to hold: `failure_class` has exactly one writer (`mark_failed`),
        # which sets it in the SAME statement that writes `status='failed'`, and
        # terminal statuses are write-once, so nothing can strand a class on a
        # row that later reads 'completed'.
        CheckConstraint(
            "failure_class IS NULL OR status = 'failed'",
            name="ck_runs_failure_class_only_on_failed",
        ),
        # Why a run is paused (migration 0102) — byte-for-byte that predicate.
        CheckConstraint(
            "pause_reason IS NULL OR pause_reason IN ('checkpoint', 'manual', 'unknown')",
            name="ck_runs_pause_reason",
        ),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    deliverable_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("deliverables.id", ondelete="CASCADE"),
        nullable=False,
    )
    parent_run_id: Mapped[UUID | None] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    # PR-C: the structural-relief regenerate pass number (1-based) when this run was
    # auto-spawned to shorten an over-budget demo_video timeline; NULL for every
    # normal run. Unique with parent_run_id (see uq_runs_parent_relief_pass) so a
    # relief child is spawned at most once per (parent, pass).
    structural_relief_pass: Mapped[int | None] = mapped_column(Integer, nullable=True)
    blueprint_id: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    current_stage: Mapped[str | None] = mapped_column(Text, nullable=True)
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    pause_after_stages: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default="{}"
    )
    # The BLUEPRINT-seeded subset of ``pause_after_stages`` at creation (migration
    # 0070). NULL = provenance unknown (legacy row). ``resume_run`` drops a pause
    # only if it is in here AND the blueprint no longer declares it a gate, so a
    # user-chosen checkpoint is never dropped.
    seeded_pause_after_stages: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Fault attribution on the OURS-vs-THEIRS axis, written by
    # ``RunRepository.mark_failed`` alongside ``error``. NULL means "never
    # classified" (every row that predates migration 0066); ``'unknown'`` means
    # "classified, and the evidence did not attribute" — a real answer, not a
    # gap. See ``src/models/failure_class.py``.
    failure_class: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Machine-readable WHAT-TO-DO for a failed run (migration 0091), e.g.
    # ``{"code": "byo_provider_required", "modality": "tts", "message": ...}``.
    # NULL when the failure carried no structured detail. Writers:
    # ``RunRepository.mark_failed`` (see ``models.failure_class.failure_detail_for``)
    # and ``RunRepository.mark_cancelled_by_parent``, which stamps
    # ``{"code": "cancelled_by_parent", ...}`` on a CANCELLED sub-DAG child so a
    # cascade cancel is distinguishable from a user's.
    failure_detail: Mapped[dict[str, str] | None] = mapped_column(JSONB, nullable=True)
    started_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Pause accounting (migration 0094). ``paused_at`` is set while the run sits
    # ``paused`` (stamped by ``RunRepository.mark_paused``); ``mark_resumed``
    # folds ``now() - paused_at`` into ``paused_seconds`` and clears it. The run
    # deadline is ``started_at + run_deadline_seconds + paused_seconds`` so time
    # spent waiting on a human reviewer is not charged to the run's budget.
    paused_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    paused_seconds: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    # Why the run is paused (migration 0102): ``checkpoint`` = a review gate
    # tripped and a person must approve the stage output; ``manual`` = someone
    # pressed Pause; ``unknown`` = paused before 0102 with no evidence either way
    # (0102's backfill). NULL whenever the run is not paused. Only ``checkpoint``
    # is "needs review"; ``manual`` and ``unknown`` are just "paused".
    pause_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    # The owner dismissed this pause's in-app "needs your review" notice (0104).
    # The notice returns once the run has waited 24h; resume clears this.
    review_notice_dismissed_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    # Set once the billing reconciler has mirrored this terminal run's cost into
    # ``run_usage``. Distinguishes a genuinely zero-cost run (reconciled, no rows)
    # from a not-yet-reconciled one, so the tenant-wide usage sweep stops
    # re-querying billing for zero-cost runs on every /usage and /admin load.
    usage_reconciled_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Durable execution claim — the second guard behind the Redis execution
    # lease (``RunLease``). Populated ONLY on the fail-open path: when Redis is
    # unavailable, ``RunLease.acquire`` returns None and every replica would
    # otherwise execute the run, double-running its non-idempotent submit
    # stages. An atomic conditional UPDATE on ``exec_claimed_at`` lets exactly
    # one pod win; a heartbeat refreshes it and finalize/drain clears it. Stale
    # (older than the claim TTL) ⇒ reclaimable, so a hard-killed pod's run
    # resumes. NULL for the normal Redis-lease path — no behaviour change there.
    exec_pod: Mapped[str | None] = mapped_column(Text, nullable=True)
    exec_claimed_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    deliverable: Mapped[Deliverable] = relationship(back_populates="runs")
    parent_run: Mapped[Run | None] = relationship("Run", remote_side=lambda: [Run.id])
    stage_outputs: Mapped[list[StageOutput]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    run_usages: Mapped[list[RunUsage]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class StageOutput(Base):
    __tablename__ = "stage_outputs"
    __table_args__ = (
        UniqueConstraint("run_id", "stage_name", name="uq_stage_outputs_run_stage"),
        Index("idx_stage_outputs_run", "run_id"),
        CheckConstraint(
            "status IN ('pending','running','completed','failed','skipped','cancelled')",
            name="ck_stage_outputs_status",
        ),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    stage_name: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    output: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Count of genuine-timeout resume attempts for this stage (durability P1). A
    # slow stage that times out is re-dispatched/resumed rather than failing the
    # run; this bounds that resume — over PIPELINE_MAX_STAGE_ATTEMPTS the stage
    # terminal-fails so an always-slow stage can't loop forever. NOT reset by
    # mark_running/mark_pending, so it survives resumes and pod death.
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    # Count of DRAIN/CANCEL interruptions (pod roll, pause, user cancel) this stage
    # survived. The twin of ``attempts``, which counts only genuine TIMEOUT resumes.
    # Both paths funnel through ``mark_pending``, so before this column existed the
    # two were indistinguishable and ``attempts == 0`` read as "this stage ran once,
    # cleanly" for a stage that had in fact been drained repeatedly.
    #
    # NULLABLE with NO server default on purpose: rows written before 0067 have an
    # UNKNOWN interruption history, and a server default would have stamped 0 onto
    # every one of them and asserted a clean history the database does not have.
    # NULL therefore means "not recorded" and is reserved for pre-0067 rows; the
    # INSERT writers seed 0 explicitly, so a row created after 0067 always reads
    # "recorded, never drained" rather than "unknown". Bumps use
    # ``coalesce(interruptions, 0) + 1`` so an old row can still start counting.
    interruptions: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # When the stage FIRST entered 'running' — stamped once, via COALESCE, and
    # deliberately preserved across resumes so ``completed_at - started_at`` keeps
    # measuring first-dispatch → final completion. See ``mark_running``/``mark_pending``.
    started_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # When the CURRENT (or last) execution entered 'running' — re-stamped
    # UNCONDITIONALLY on every ``mark_running``. This is the missing half of the
    # pair: ``started_at`` spans burnt executions, re-dispatch gaps and pod
    # downtime, so ``completed_at - started_at`` is NOT handler runtime and cannot
    # be compared against a stage ceiling. ``completed_at - last_execution_started_at``
    # is the per-execution runtime that can. NULLABLE, no backfill: pre-0067 rows
    # keep only their ambiguous span rather than an invented one.
    last_execution_started_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # 0102: when a user last edited this row's output in place (element patch /
    # restamp, course-stitch overrides). A stage edited after it completed keeps its
    # edit on Restart and its DESCENDANTS re-run (services/runs/restart_plan.py).
    # NULL = never edited. A preserved row a child inherits starts NULL: the edit is
    # baked into the inherited output and the child's dependents ran on it.
    edited_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # 0102: the top-level run-snapshot keys this stage read while it ran
    # (services/runs/config_reads.py); ``["*"]`` = read wholesale. NULL = unknown
    # (pre-0102 row, or a path that does not record): any input change re-runs it.
    config_reads: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    run: Mapped[Run] = relationship(back_populates="stage_outputs")
    annotations: Mapped[list[Annotation]] = relationship(
        back_populates="stage_output", cascade="all, delete-orphan"
    )

    @property
    def last_execution_seconds(self) -> float | None:
        """Runtime of the FINAL execution of this stage, or ``None`` if unknowable.

        The number to compare against a stage ceiling. ``None`` — not a guess and
        not the ambiguous span — whenever the stage has not completed or predates
        migration 0067, because a caller silently substituting
        ``completed_at - started_at`` there is exactly the mistake this column
        exists to stop.
        """
        if self.completed_at is None or self.last_execution_started_at is None:
            return None
        return (self.completed_at - self.last_execution_started_at).total_seconds()

    @property
    def elapsed_seconds(self) -> float | None:
        """First dispatch → completion, INCLUDING burnt executions and downtime.

        Deliberately named for what it is. This is a scheduling/latency figure, NOT
        handler runtime — see :attr:`last_execution_seconds`.
        """
        if self.completed_at is None or self.started_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()

    @property
    def duration_is_ambiguous(self) -> bool:
        """Whether :attr:`elapsed_seconds` over-reports this stage's runtime.

        True when the stage was drained or timed out at least once, so its span
        covers work that was thrown away. ``interruptions is None`` (a pre-0067
        row) also counts as ambiguous: not-recorded is not the same as
        did-not-happen, and treating it as the latter is the false ``attempts == 0``
        filter all over again.
        """
        return self.interruptions is None or self.interruptions > 0 or self.attempts > 0


class StageStepResult(Base):
    """One finished SUB-STEP of a stage — a row of the **step ledger** (0071).

    ``stage_outputs`` holds one row per (run, stage) and is written only when the
    whole stage ends, so a stage that dies half-way (pod roll, OOM, node loss) is
    re-entered from the top with nothing to show for the paid calls it already
    made. A ledger row records the result of ONE such call, keyed
    ``(run_id, stage_name, step_key)``, so the re-entry can reuse it.

    * ``input_hash`` — digest of everything the sub-step's result depends on. A row
      is reused only when the caller's hash matches; a different hash REPLACES the
      row (upsert), it never accumulates history.
    * ``result`` — opaque JSONB, NOT NULL. The ledger helper
      (``src/services/stages/shared/step_ledger.py``) stores an envelope
      ``{"value": ...}`` so a sub-step that legitimately returned ``None`` is still
      distinguishable from "no row", and never has to rely on JSON-null vs SQL-NULL.
    * No ``status``: only SUCCESSFUL sub-steps are recorded. A row's existence is
      the whole fact.

    Not a user pause gate — that vocabulary ("checkpoint") is taken by
    ``runs.pause_after_stages``. Deliberately not ``stage_outputs.output`` either:
    ``record_progress`` replaces that dict wholesale and ``mark_completed``
    overwrites it, and the write-once trigger (0063) freezes it at terminal.

    RLS: ENABLE + join-based ``tenant_isolation`` climbing ``run_id -> runs ->
    deliverables -> projects.tenant_id`` (``RLS_JOIN_PREDICATES_STAGE_STEP_RESULTS``);
    frozen inline in migration 0071, registered for ``create_all`` below.
    """

    __tablename__ = "stage_step_results"
    __table_args__ = (
        UniqueConstraint(
            "run_id", "stage_name", "step_key", name="uq_stage_step_results_run_stage_step"
        ),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    stage_name: Mapped[str] = mapped_column(Text, nullable=False)
    step_key: Mapped[str] = mapped_column(Text, nullable=False)
    input_hash: Mapped[str] = mapped_column(Text, nullable=False)
    result: Mapped[object] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class Annotation(Base):
    __tablename__ = "annotations"
    __table_args__ = (
        Index("idx_annotations_stage_output", "stage_output_id", "status"),
        Index(
            "idx_annotations_open",
            "stage_output_id",
            postgresql_where="status = 'open'",
        ),
        CheckConstraint(
            "status IN ('open','applied','dismissed')",
            name="ck_annotations_status",
        ),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    stage_output_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("stage_outputs.id", ondelete="CASCADE"),
        nullable=False,
    )
    element_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="open")
    applied_in_run_id: Mapped[UUID | None] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    stage_output: Mapped[StageOutput] = relationship(back_populates="annotations")


class RunUsage(Base):
    __tablename__ = "run_usage"
    __table_args__ = (
        Index("idx_run_usage_run", "run_id"),
        Index("idx_run_usage_stage", "run_id", "stage_name"),
        # Supports the admin usage time-series: date_trunc bucketing + the
        # optional last-N-days window predicate on created_at (migration 0023).
        Index("idx_run_usage_created", "created_at"),
        # P6 ORCH-3 (migration 0083): per-user usage reads group on it.
        Index("idx_run_usage_subject", "subject_id"),
        # Migration 0093: a service-reported row (POST /api/runs/{id}/usage) carries
        # the reporter's ``metadata.call_id``; one row per (run, call) makes a retried
        # report a no-op (INSERT … ON CONFLICT DO NOTHING). Rows without one (billing
        # mirror, in-process) are outside the index.
        Index(
            "uq_run_usage_run_call_id",
            "run_id",
            text("(metadata ->> 'call_id')"),
            unique=True,
            postgresql_where=text("(metadata ->> 'call_id') IS NOT NULL"),
        ),
        # Migration 0105: ONE billing-mirrored row per (run, stage, model, provider).
        # The mirror is delete-then-insert; this makes a concurrent duplicate
        # impossible even if a writer skips RunUsageRepository.try_lock_mirror.
        Index(
            "uq_run_usage_mirrored_group",
            "run_id",
            "stage_name",
            text("COALESCE(model, '')"),
            text("COALESCE(provider, '')"),
            unique=True,
            postgresql_where=text("(metadata ->> 'source') = 'billing_mirror'"),
        ),
        CheckConstraint(
            "unit_kind IS NULL OR unit_kind IN ('tokens','chars','images','seconds')",
            name="ck_run_usage_unit_kind",
        ),
        CheckConstraint(
            "funding IS NULL OR funding IN ('local_gpu','platform_external','tenant_byo')",
            name="ck_run_usage_funding",
        ),
        # Migration 0088: a connection id without its version (or vice versa) is
        # unrepresentable — both NULL = platform path, both set = tenant slot.
        CheckConstraint(
            "(connection_id IS NULL) = (connection_version IS NULL)",
            name="ck_run_usage_connection_pair",
        ),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    stage_name: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str | None] = mapped_column(Text, nullable=True)
    provider: Mapped[str | None] = mapped_column(Text, nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False, server_default="0")
    metadata_: Mapped[dict[str, object] | None] = mapped_column("metadata", JSONB, nullable=True)
    # P6 ORCH-3 attribution (migration 0083). All nullable: NULL = not attributed
    # (pre-0083 rows the backfill could not attribute) — reported, never dropped.
    #: Provider slot of an in-process tenant-BYO call; NULL = the platform path.
    slot: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: Who the usage is attributed to (the run's subject).
    subject_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    #: ``local_gpu`` | ``platform_external`` | ``tenant_byo``; NULL = not known
    #: (billing-mirrored rows: billing cannot tell who paid) → reported unattributed.
    funding: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: Billable quantity, in ``unit_kind`` (tokens | chars | images | seconds).
    units: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)
    unit_kind: Mapped[str | None] = mapped_column(Text, nullable=True)
    # P7 / R3 (migration 0088): the tenant connection + VERSION that served an
    # in-process call. Both NULL = the platform path (WM / gateway / billing-mirrored).
    # No FK: usage outlives an offboarded/purged connection.
    connection_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    connection_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    run: Mapped[Run] = relationship(back_populates="run_usages")


class VerbatimItem(Base):
    __tablename__ = "verbatim_items"
    __table_args__ = (
        # Predicates + names copied VERBATIM from migration 0017, which re-added
        # `kind`/`scope` after the Studio v3 reset (0015) dropped the whole
        # 0009-era table CASCADE — so 0009's other three CHECKs (status,
        # visibility, scope_xor) and their columns no longer exist at head, and
        # these two are the entire live-at-head set. They were missing here for
        # every release since 0017: prod rejected a bad `kind`, while the
        # create_all test schema accepted it, so no test could catch the write.
        # Vocabulary mirrors `VerbatimKind` / `VerbatimScope` in
        # `src/api/schemas/_shared.py`. Widen all three together, never one.
        # Guarded by tests/contracts/test_stage_status_check_constraint_parity.py
        # (`test_check_constraints_match_between_models_and_migrations`).
        CheckConstraint("kind IN ('slide', 'tts')", name="ck_verbatim_items_kind"),
        CheckConstraint(
            "scope IN ('pipeline', 'chapter', 'scene', 'shot', 'slide')",
            name="ck_verbatim_items_scope",
        ),
        Index("idx_verbatim_project", "project_id", "position"),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    project_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    kind: Mapped[str] = mapped_column(Text, nullable=False, server_default="tts")
    text: Mapped[str] = mapped_column(Text, nullable=False)
    hint: Mapped[str | None] = mapped_column(Text, nullable=True)
    scope: Mapped[str] = mapped_column(Text, nullable=False, server_default="pipeline")
    scope_target: Mapped[str | None] = mapped_column(Text, nullable=True)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    project: Mapped[Project] = relationship(back_populates="verbatim_items")


class SourceIngestCache(Base):
    """Persistent, cross-run, per-source ingest cache.

    Memoizes each ``brief.sources[]`` ingest (scrape/parse + republished
    media) so N runs — notably many concurrent course runs — over the SAME source
    don't each re-fetch it. Keyed by a content/identity ``fingerprint`` (tenant +
    kind + normalized ref + options + auth identity — see
    ``src.services.source_cache.source_fingerprint``), which folds ``tenant_id``
    in, so a second tenant's identical source is a different PK → an automatic miss
    even before RLS. RLS (``tenant_isolation``) is defense-in-depth + existence-leak
    prevention, applied in migration 0045 (ENABLE-not-FORCE, direct-tenant policy).

    Single-flight: a cold-cache row is INSERTed ``pending`` (claimed) so concurrent
    identical ingests coalesce onto one ingest; the winner UPDATEs it to ``ready``
    with the ``payload`` (the per-source contribution: text, references, image
    descriptors carrying a DURABLE media-cache object key). ``expires_at`` gives
    lazy TTL expiry; a ``pending`` claim older than the claim TTL (a crashed holder)
    is reclaimable. Grants to the runtime role ride ``ALTER DEFAULT PRIVILEGES``
    (gitops), like every other table — no per-table GRANT here.
    """

    __tablename__ = "source_ingest_cache"
    __table_args__ = (
        CheckConstraint("state IN ('pending','ready')", name="ck_source_ingest_cache_state"),
        Index("ix_source_ingest_cache_tenant", "tenant_id"),
        Index("ix_source_ingest_cache_expires_at", "expires_at"),
    )

    # 64-hex sha256 fingerprint; globally unique (tenant folded in) → the PK.
    fingerprint: Mapped[str] = mapped_column(Text, primary_key=True)
    tenant_id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), nullable=False)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    # The source ref (URL or document key) — for observability only; the
    # fingerprint, not this, is the cache identity.
    ref: Mapped[str] = mapped_column(Text, nullable=False)
    state: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    claimed_by: Mapped[str | None] = mapped_column(Text, nullable=True)
    claimed_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # The cached per-source contribution (text / references / image descriptors).
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    expires_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class Storyboard(Base):
    """One visual blueprint per deliverable/project (Storyboard Phase 0).

    Project-scoped for RLS (join predicate climbs ``project_id → projects.tenant_id``).
    Mirrors the ``verbatim_items`` / ``source_ingest_cache`` conventions: surrogate
    uuid PK, ``Text`` + ``CheckConstraint`` enum columns, ``timestamptz`` server
    defaults. RLS attaches via ``register_late_join_rls_ddl`` (create_all) + inline
    in migration 0046 (prod).
    """

    __tablename__ = "storyboards"
    __table_args__ = (
        CheckConstraint("status IN ('draft','approved')", name="ck_storyboards_status"),
        # One storyboard per project — backs ensure_for_project's ON CONFLICT
        # and provides the project_id index (replaces the plain idx). Tests use
        # Base.metadata.create_all, so this must live on the model, not only the
        # migration.
        UniqueConstraint("project_id", name="uq_storyboards_project"),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    project_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    deliverable_id: Mapped[UUID | None] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("deliverables.id", ondelete="SET NULL"),
        nullable=True,
    )
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="draft")
    aspect_ratio: Mapped[str | None] = mapped_column(Text, nullable=True)
    art_style: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class StoryboardScene(Base):
    """Continuous time+place block; ``position``-ordered within a storyboard."""

    __tablename__ = "storyboard_scenes"
    __table_args__ = (
        Index("idx_storyboard_scenes_project", "project_id", "position"),
        Index("idx_storyboard_scenes_storyboard", "storyboard_id"),
        # Merge resolves EVERY row of both sides by ``origin_id`` within a project;
        # unindexed that is a seq scan per lookup over the whole branch+root row set.
        # PARTIAL on ``origin_id IS NOT NULL`` because only forked rows carry lineage —
        # the index stays proportional to the branches in flight, not to the table.
        Index(
            "idx_storyboard_scenes_origin",
            "project_id",
            "origin_id",
            postgresql_where=text("origin_id IS NOT NULL"),
        ),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    project_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    storyboard_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("storyboards.id", ondelete="CASCADE"),
        nullable=False,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str | None] = mapped_column(Text, nullable=True)
    int_ext: Mapped[str | None] = mapped_column(Text, nullable=True)  # free text INT/EXT/INT_EXT
    location: Mapped[str | None] = mapped_column(Text, nullable=True)
    time_of_day: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Optimistic-concurrency token (opt-in CAS), mirror of ``StoryboardPanel.lock_version``
    # (migration 0056). A plain counter the repository bumps by hand on every scene write —
    # NOT SQLAlchemy ``version_id_col``. A client echoing the ``lock_version`` it last read
    # as ``expected_lock_version`` gets compare-and-set (409 on a lost update); omitting it
    # keeps last-write-wins.
    lock_version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    # Branch lineage (migration 0052). When a project is forked into a branch, every
    # copied row re-mints its ``id`` (a global uuid — two projects cannot share one),
    # so merge needs an identity that survives the copy. ``origin_id`` is stamped at
    # fork time to the SOURCE row's ``origin_id`` if set, else its ``id`` — so lineage
    # stays stable across re-forks. Deliberately NO FK: it references a row in a
    # DIFFERENT project that may be deleted independently; it is a lineage marker, not
    # a constraint. NULL on every non-branch row.
    origin_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class StoryboardPanel(Base):
    """The spine element. Surrogate uuid ``id`` PK; ``shot_key`` holds the sacred
    ``clip-<n>`` join key (= the Pydantic ``StoryboardPanelBase.id``)."""

    __tablename__ = "storyboard_panels"
    __table_args__ = (
        CheckConstraint("status IN ('draft','approved')", name="ck_storyboard_panels_status"),
        CheckConstraint(
            "shot_size IS NULL OR shot_size IN "
            "('EWS','WS','LS','MLS','MS','MCU','CU','ECU',"
            "'establishing','insert','OTS','POV','two_shot')",
            name="ck_storyboard_panels_shot_size",
        ),
        CheckConstraint(
            "angle IS NULL OR angle IN ('eye_level','high','low','dutch','overhead','worm')",
            name="ck_storyboard_panels_angle",
        ),
        CheckConstraint(
            "movement IS NULL OR movement IN "
            "('static','pan','tilt','dolly','zoom','crane','handheld','push_in','pull_out')",
            name="ck_storyboard_panels_movement",
        ),
        CheckConstraint(
            "transition_out IS NULL OR transition_out IN "
            "('cut','dissolve','wipe','fade_in','fade_out','match_cut')",
            name="ck_storyboard_panels_transition_out",
        ),
        CheckConstraint(
            "visual_type IS NULL OR visual_type IN "
            "('footage','slide','title_card','ai_image','upload')",
            name="ck_storyboard_panels_visual_type",
        ),
        Index("idx_storyboard_panels_project", "project_id", "position"),
        # ``shot_key`` is the NOT-NULL per-project spine (``clip-<n>``) join key —
        # it must be UNIQUE per project, not just indexed. The unique constraint
        # supplies its own backing index, so the plain
        # ``idx_storyboard_panels_project_shot`` is dropped (migration 0048; same
        # swap 0047 did to ``storyboards``). Tests build the schema via
        # ``Base.metadata.create_all``, so this must live on the model, not only
        # the migration.
        #
        # DEFERRABLE INITIALLY IMMEDIATE (migration 0050): a snapshot RESTORE
        # upserts every panel in one transaction, and a shot_key swap/reorder
        # produces a TRANSIENT duplicate mid-statement that only resolves once the
        # whole set is written. ``SET CONSTRAINTS ALL DEFERRED`` defers the check to
        # COMMIT so the settled state is validated, not the intermediate one.
        # ``INITIALLY IMMEDIATE`` keeps every other write path checking per-statement
        # exactly as before — only an explicit ``SET CONSTRAINTS DEFERRED`` defers it.
        UniqueConstraint(
            "project_id",
            "shot_key",
            name="uq_storyboard_panels_project_shot",
            deferrable=True,
            initially="IMMEDIATE",
        ),
        Index("idx_storyboard_panels_scene", "scene_id"),
        Index("idx_storyboard_panels_storyboard", "storyboard_id"),
        # See ``StoryboardScene``'s origin index — merge matches panel-by-panel on
        # ``origin_id``, and the panel table is the widest of the three.
        Index(
            "idx_storyboard_panels_origin",
            "project_id",
            "origin_id",
            postgresql_where=text("origin_id IS NOT NULL"),
        ),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    project_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    storyboard_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("storyboards.id", ondelete="CASCADE"),
        nullable=False,
    )
    scene_id: Mapped[UUID | None] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("storyboard_scenes.id", ondelete="SET NULL"),
        nullable=True,
    )
    shot_key: Mapped[str] = mapped_column(Text, nullable=False)  # clip-<n> spine key
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    shot_number: Mapped[str | None] = mapped_column(Text, nullable=True)
    panel_index: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="draft")
    revision: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    # Optimistic-concurrency token (opt-in CAS). A plain counter bumped by every
    # panel write — NOT SQLAlchemy ``version_id_col`` (that would force a CAS on
    # every flush and collide with the merge delete-race ``StaleDataError`` guard).
    # A client that echoes the ``lock_version`` it last read as ``expected_lock_version``
    # gets compare-and-set (409 on a lost update); one that omits it keeps last-write-wins.
    lock_version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    # Current approval state (Boords trail). The append-only history lives in
    # ``storyboard_panel_events``; these two hold the CURRENT snapshot so a plain
    # panel read needs no join. Set on approve, cleared on unapprove.
    approved_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    approved_by: Mapped[str | None] = mapped_column(Text, nullable=True)

    # action / audio
    action_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    dialogue: Mapped[str | None] = mapped_column(Text, nullable=True)
    vo_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    sfx_cue: Mapped[str | None] = mapped_column(Text, nullable=True)
    music_cue: Mapped[str | None] = mapped_column(Text, nullable=True)

    # visual
    image_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    visual_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    shot_size: Mapped[str | None] = mapped_column(Text, nullable=True)
    composition_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    # ``characters`` / ``props`` / ``location_asset_id`` were JSONB arrays + a bare
    # UUID with NO foreign key ("no FK — asset may be deleted"), i.e. undeclared FKs
    # into ``storyboard_assets``: deleting an asset silently gutted every panel that
    # referenced it. Migration 0051 normalizes all three into the
    # ``storyboard_panel_assets`` join table below; they survive on the wire as the
    # read-only projections at the bottom of this class, so the API contract
    # (``characters: list[str]``, ``props: list[str]``,
    # ``location_asset_id: UUID | None``) is unchanged.
    lighting_mood: Mapped[str | None] = mapped_column(Text, nullable=True)
    art_style: Mapped[str | None] = mapped_column(Text, nullable=True)
    aspect_ratio: Mapped[str | None] = mapped_column(Text, nullable=True)

    # camera
    angle: Mapped[str | None] = mapped_column(Text, nullable=True)
    movement: Mapped[str | None] = mapped_column(Text, nullable=True)
    movement_path: Mapped[Any | None] = mapped_column(JSONB, nullable=True)  # keyframe hint
    lens: Mapped[str | None] = mapped_column(Text, nullable=True)

    # timing / transition
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    transition_out: Mapped[str | None] = mapped_column(Text, nullable=True)

    # media + gen provenance (JSONB blobs)
    media_ref: Mapped[dict[str, object]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    source: Mapped[str | None] = mapped_column(Text, nullable=True)  # ai | user | uploaded
    gen: Mapped[dict[str, object]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )

    # Branch lineage (migration 0052) — see ``StoryboardScene.origin_id``. Merge
    # matches panels PRIMARILY by ``origin_id`` (falls back to ``shot_key``, which is
    # added independently in both lanes), so a shot_key REORDER inside a branch merges
    # as an EDIT, not delete+add.
    origin_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # The normalized asset references (migration 0051). ``lazy="selectin"`` is
    # MANDATORY, not a tuning knob: the default ``lazy="select"`` would emit an
    # implicit IO-on-attribute-access lazy load on the ASYNC session the first time
    # a serializer touches the collection → ``MissingGreenlet``. ``passive_deletes``
    # hands panel deletion to the DB's ON DELETE CASCADE (no SELECT-then-DELETE
    # round trip); ``delete-orphan`` keeps a detached join row from surviving.
    panel_assets: Mapped[list[StoryboardPanelAsset]] = relationship(
        back_populates="panel",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
        order_by="(StoryboardPanelAsset.role, StoryboardPanelAsset.position)",
    )

    # ─── Wire-contract projections over ``panel_assets`` ────────────────────────
    # Read-only ON PURPOSE: a writable property would silently re-open the
    # undeclared-FK hole. Writes go through ``StoryboardPanelRepository``, which
    # diffs the join rows (and validates the ids against the project's assets).
    #
    # The sort is done IN PYTHON, not left to the relationship's ``order_by``: that
    # only orders the collection as LOADED, and never re-sorts an in-memory mutated
    # collection — so a create/PATCH response rendered from the flushed-but-not-
    # reloaded instance would emit the wrong order.

    @property
    def characters(self) -> list[str]:
        """Character asset ids, in author order (response schema is ``list[str]``)."""
        return self._asset_ids("character")

    @property
    def props(self) -> list[str]:
        """Prop asset ids, in author order (response schema is ``list[str]``)."""
        return self._asset_ids("prop")

    @property
    def location_asset_id(self) -> UUID | None:
        """The single location asset, if any — kept scalar by the partial unique
        index ``uq_storyboard_panel_assets_one_location``."""
        rows = [row for row in self.panel_assets if row.role == "location"]
        return rows[0].asset_id if rows else None

    def _asset_ids(self, role: str) -> list[str]:
        rows = sorted(
            (row for row in self.panel_assets if row.role == role),
            key=lambda row: row.position,
        )
        return [str(row.asset_id) for row in rows]


class StoryboardPanelEvent(Base):
    """Append-only per-panel approval trail (Boords pattern).

    One immutable row per approve/unapprove transition. The panel row
    (``storyboard_panels``) holds the CURRENT state (``status`` +
    ``approved_at`` / ``approved_by``); THIS table is the history — never updated
    in place, only appended. ``revision`` snapshots ``panel.revision`` at the
    event time (the content-edit counter owned by AI rewrite — NOT bumped by an
    approval). Mirrors ``AdminAuditLog`` (caller-minted ``id`` with no column
    default, controlled-vocab ``event_type``, ``detail`` JSONB) + the
    ``StoryboardPanel`` project-FK idiom (``project_id`` is the RLS climb key,
    ``project_id -> projects.tenant_id``).

    ``panel_id`` FK is ON DELETE CASCADE, so deleting a panel reaps its trail
    (acceptable — the trail is meaningless without its panel).
    """

    __tablename__ = "storyboard_panel_events"
    __table_args__ = (
        CheckConstraint(
            # 'restored' added in migration 0050: a snapshot RESTORE appends one
            # ``restored`` trail event per restored panel (approve/unapprove stay
            # the only other transitions). 'merged' added in migration 0052: a branch
            # MERGE appends one ``merged`` trail event per touched panel
            # (``detail={branch_id, base_snapshot_id}``). Merge UPSERTs panels — it
            # never delete-then-reinserts — so the ``panel_id`` CASCADE never reaps
            # the trail (the exact bug caught in PR #468).
            "event_type IN ('approved','unapproved','restored','merged')",
            name="ck_storyboard_panel_events_type",
        ),
        Index("idx_storyboard_panel_events_panel", "panel_id", "created_at"),
        Index("idx_storyboard_panel_events_project", "project_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    project_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    panel_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("storyboard_panels.id", ondelete="CASCADE"),
        nullable=False,
    )
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    actor_subject_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    detail: Mapped[dict[str, object]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class StoryboardAsset(Base):
    """Per-project Character/Location/Prop — the consistency backbone."""

    __tablename__ = "storyboard_assets"
    __table_args__ = (
        CheckConstraint(
            "kind IN ('character','location','prop')", name="ck_storyboard_assets_kind"
        ),
        Index("idx_storyboard_assets_project", "project_id", "kind"),
        Index("idx_storyboard_assets_project_name", "project_id", "name"),
        # See ``StoryboardScene``'s origin index. The ``(kind, name)`` fallback match is
        # already served by ``idx_storyboard_assets_project_name``.
        Index(
            "idx_storyboard_assets_origin",
            "project_id",
            "origin_id",
            postgresql_where=text("origin_id IS NOT NULL"),
        ),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    project_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)  # exact match = consistency key
    canonical_descriptor: Mapped[str | None] = mapped_column(Text, nullable=True)
    reference_images: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    style_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    lora_path: Mapped[str | None] = mapped_column(Text, nullable=True)  # v2, nullable
    attributes: Mapped[dict[str, object]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    # Optimistic-concurrency token (opt-in CAS), mirror of ``StoryboardPanel.lock_version``
    # (migration 0056). A plain counter the repository bumps by hand on every asset write —
    # NOT SQLAlchemy ``version_id_col``. A client echoing the ``lock_version`` it last read
    # as ``expected_lock_version`` gets compare-and-set (409 on a lost update); omitting it
    # keeps last-write-wins.
    lock_version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    # Branch lineage (migration 0052) — see ``StoryboardScene.origin_id``. Merge
    # matches assets PRIMARILY by ``origin_id`` (falls back to ``(kind, name)``, the
    # declared consistency key), so a renamed-in-branch asset still merges as an edit.
    origin_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class StoryboardPanelAsset(Base):
    """Panel → asset reference (migration 0051) — the real FK the three JSONB/bare-UUID
    columns on ``StoryboardPanel`` never were.

    Declared AFTER ``StoryboardAsset`` so ``create_all``'s FK-dependency ordering
    resolves both parents (``storyboard_panels``, ``storyboard_assets``) before this
    table is created.

    ``panel_id`` is ON DELETE CASCADE (a join row is meaningless without its panel).
    ``asset_id`` is **ON DELETE NO ACTION DEFERRABLE INITIALLY IMMEDIATE**, NOT
    RESTRICT: postgres checks RESTRICT immediately and CANNOT defer it even under
    ``SET CONSTRAINTS ALL DEFERRED``, which is exactly what
    ``storyboard_snapshot_service.restore()`` does — it deletes assets while surviving
    panels still reference them, and only reaches a consistent state at COMMIT. NO
    ACTION honours the deferral, so restore works; and because it stays INITIALLY
    IMMEDIATE, an ordinary asset DELETE still raises at the statement (SQLSTATE
    23503, matched on the constraint NAME) → the clean 409 "asset in use".

    Deliberately NO backref on ``StoryboardAsset``: a default (non-passive) cascade
    would make ``session.delete(asset)`` emit ``UPDATE storyboard_panel_assets SET
    asset_id = NULL`` and blow up on the NOT NULL constraint instead of surfacing the
    FK violation the 409 path matches on.

    RLS: this table carries neither ``tenant_id`` nor ``project_id``, so its policy is
    a **2-hop** climb ``panel_id -> storyboard_panels.project_id -> projects.tenant_id``
    (see ``RLS_JOIN_PREDICATES_PANEL_ASSETS`` in ``src/database/rls.py``; applied inline
    in migration 0051 for prod and via ``register_late_join_panel_assets_rls_ddl`` for
    ``create_all``).
    """

    __tablename__ = "storyboard_panel_assets"
    __table_args__ = (
        CheckConstraint(
            "role IN ('character','prop','location')", name="ck_storyboard_panel_assets_role"
        ),
        UniqueConstraint("panel_id", "asset_id", "role", name="uq_storyboard_panel_assets_triple"),
        # A panel has at most ONE location — the old column was scalar and the
        # ``location_asset_id`` projection stays scalar. A partial unique index is the
        # only way to express "unique per panel, but only for this role".
        Index(
            "uq_storyboard_panel_assets_one_location",
            "panel_id",
            unique=True,
            postgresql_where=text("role = 'location'"),
        ),
        # The FK check and the 409 "which panels reference this asset?" listing both
        # scan by asset_id.
        Index("idx_storyboard_panel_assets_asset", "asset_id"),
        Index("idx_storyboard_panel_assets_panel_role", "panel_id", "role", "position"),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    panel_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("storyboard_panels.id", ondelete="CASCADE"),
        nullable=False,
    )
    asset_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey(
            "storyboard_assets.id",
            ondelete="NO ACTION",
            deferrable=True,
            initially="IMMEDIATE",
        ),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(Text, nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    panel: Mapped[StoryboardPanel] = relationship(back_populates="panel_assets")


class StoryboardSnapshot(Base):
    """Immutable version snapshot of a project's whole storyboard graph (Phase 5).

    One row per snapshot; ``payload`` is a self-contained JSONB copy of the
    storyboard + scenes + panels + assets (NOT the append-only
    ``storyboard_panel_events`` trail) at capture time — the same blob idiom as
    ``Run.config_snapshot``. ``kind`` is ``manual`` (user-requested) or ``auto``
    (the pre-restore safety snapshot). Project-scoped for RLS (join predicate
    climbs ``project_id -> projects.tenant_id``); RLS attaches via
    ``register_late_join_snapshots_rls_ddl`` (create_all) + inline in migration
    0050 (prod). ``id`` is caller-minted (uuid7) like every sibling storyboard row.
    """

    __tablename__ = "storyboard_snapshots"
    __table_args__ = (
        CheckConstraint(
            # 'branch_base' added in migration 0052: the snapshot a FORK mints to pin
            # its merge base. It is a THIRD kind rather than reusing the existing two
            # because both would lose it under an open branch. ``auto`` is swept by
            # ``StoryboardSnapshotRepository.prune_auto_for_project`` (it deletes every
            # ``kind='auto'`` row beyond the newest N, and its ``protect_id`` shields
            # exactly ONE id — not the arbitrarily many open branches). ``manual`` is
            # user-deletable (the DELETE route only 409s on ``auto``). A distinct kind
            # is immune to the prune for FREE — it filters on ``kind='auto'`` — and the
            # DELETE route's 409 gate is widened to cover it alongside ``auto``.
            "kind IN ('manual','auto','branch_base')",
            name="ck_storyboard_snapshots_kind",
        ),
        # Backs the newest-first listing scan (``ORDER BY created_at DESC`` uses a
        # backward index scan over this composite btree).
        Index("idx_storyboard_snapshots_project_created", "project_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    project_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    storyboard_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("storyboards.id", ondelete="CASCADE"),
        nullable=False,
    )
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    actor_subject: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )


class StoryboardBranch(Base):
    """A storyboard branch (migration 0052) — the branch↔root link + its lifecycle.

    A branch is a real, HIDDEN child ``projects`` row (``projects.is_branch=true``);
    this table records only the linkage and merge lifecycle. Kept OFF
    ``parent_project_id`` on purpose — that column is the course→lesson hierarchy
    (``ProjectRepository`` enforces "lesson must have a parent" and lists children
    off it), so reusing it would surface every branch as a lesson-in-a-course.

    ``root_project_id`` / ``child_project_id`` are BOTH ``ON DELETE CASCADE``: deleting
    either project reaps the branch record. ``child_project_id`` is UNIQUE (one branch
    record per child project). ``base_snapshot_id`` pins the fork point — the ``base``
    side of merge's three-way diff — and is ``ON DELETE CASCADE`` as well. That reads
    dangerous but is the SAFE choice given ``kind='branch_base'``: the base snapshot is
    no longer reachable by the auto-prune or by a hand DELETE, so the ONLY thing that
    can delete it is deleting the whole project — which already CASCADE-reaps this
    branch row through ``root_project_id``/``child_project_id`` anyway. A non-cascading
    FK would not protect the branch there; it would just make the project-delete cascade
    hit a NO ACTION violation on this row, depending on which side postgres reaps first.

    RLS: 1-hop climb ``root_project_id -> projects.tenant_id`` (see
    ``RLS_JOIN_PREDICATES_BRANCHES`` in ``src/database/rls.py``; applied inline in
    migration 0052 for prod and via ``register_late_join_branches_rls_ddl`` for
    ``create_all``). Rooting the climb on ``root_project_id`` (not ``child_``) matters:
    the child project inherits the root's ``tenant_id`` at fork, so either climbs to
    the same tenant, but ``root_`` is the stable, user-visible side.
    """

    __tablename__ = "storyboard_branches"
    __table_args__ = (
        CheckConstraint(
            "status IN ('open','merged','abandoned')",
            name="ck_storyboard_branches_status",
        ),
        UniqueConstraint("child_project_id", name="uq_storyboard_branches_child"),
        Index("idx_storyboard_branches_root", "root_project_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    root_project_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    child_project_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    base_snapshot_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("storyboard_snapshots.id", ondelete="CASCADE"),
        nullable=False,
    )
    # The SHARED ancestor for SIBLING merges (branch A <-> branch B forked off the same
    # parent) — the parent's state before ANY sibling diverged (migration 0057). Distinct
    # from ``base_snapshot_id``: that is per-branch (the base for merging THIS branch into
    # its parent, which has moved on independently); this is ONE snapshot shared by all
    # siblings of a root, so an A<->B three-way is symmetric against a single common base.
    # The first fork of a root establishes it (its own ``branch_base``); every later fork of
    # the same parent reuses it. NULL only for branches predating 0057's backfill, or a root
    # with a single branch that never needed one. ``ON DELETE SET NULL`` (not CASCADE like
    # ``base_snapshot_id``): it points at a DIFFERENT branch's base, so a stale ref must
    # null out (→ sibling merge unavailable) rather than reap this branch row.
    parent_base_snapshot_id: Mapped[UUID | None] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("storyboard_snapshots.id", ondelete="SET NULL"),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="open")
    actor_subject: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    merged_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class VideoSource(Base):
    """A raw video the user uploaded into a project — the input side of ``action_reel``.

    One row per uploaded original (4K drone footage and friends). The bytes
    themselves never live here: ``object_key`` is the MinIO key of the untouched
    original, and the derived artefacts each get their own key
    (``preview_key`` = a small, seekable proxy the editor/LLM can actually stream;
    ``poster_key`` = a single still for the picker; ``sprite_key`` = the scrub
    filmstrip). They are nullable because they are produced ASYNCHRONOUSLY: the
    upload commits the row immediately with ``preview_status='pending'`` so the UI
    can show the file the instant it lands, and a later worker fills the three keys
    and flips the status to ``'ready'`` (or ``'failed'`` + ``error``). Reading a
    derived key without first checking ``preview_status`` is therefore always a bug.

    ``probe`` is the raw ffprobe result (duration / fps / codec / width / height).
    Kept as JSONB rather than promoted to columns because the cut planner feeds the
    whole blob to the LLM and ffprobe's shape varies per container — promoting a
    fixed subset would silently drop what a new codec reports. Defaults to ``'{}'``
    (NOT NULL) so a pending row is a plain empty object, never a NULL to guard on.

    Project-scoped for RLS (the join predicate climbs
    ``project_id -> projects.tenant_id``), exactly like every storyboard table:
    deliberately NO ``tenant_id`` column, so a source can never disagree with the
    tenant of the project that owns it. See ``RLS_JOIN_PREDICATES_VIDEO_SOURCES`` in
    ``src/database/rls.py``; applied inline in migration 0060 for prod and via
    ``register_late_join_video_sources_rls_ddl`` for ``create_all``.

    ``object_key`` is UNIQUE: the key is content/upload-addressed, so two rows
    pointing at one object would let deleting either row orphan or double-reap the
    same MinIO object. ``filename`` is the user's original name and is DISPLAY ONLY —
    never used to build a storage key or a filesystem path.
    """

    __tablename__ = "video_sources"
    __table_args__ = (
        CheckConstraint(
            "preview_status IN ('pending','ready','failed')",
            name="ck_video_sources_preview_status",
        ),
        UniqueConstraint("object_key", name="uq_video_sources_object_key"),
        Index("idx_video_sources_project", "project_id"),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    project_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Display only — the user's original name. Never a storage key, never a path.
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    object_key: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(Text, nullable=False)
    # BigInteger, not Integer: a 4K drone original passes the 2 GiB int4 ceiling
    # routinely, and overflowing it here would reject the upload at the DB.
    bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    preview_status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    preview_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    poster_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    sprite_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    # ``dict[str, object]``, NOT ``dict[str, Any]``: the neighbouring JSONB columns
    # predate the ``test_any_ratchet`` budget, which is LOWER-ONLY and may never be
    # raised. ``object`` costs a consumer one explicit narrow at the point of use —
    # which is precisely the hygiene the ratchet exists to force — and is already
    # the fleet's majority spelling for a JSON boundary.
    probe: Mapped[dict[str, object]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    # Why ``preview_status='failed'`` — operator-facing, cleared on a successful retry.
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # The preview-proxy worker's DURABLE claim. The advisory lock it also takes dies
    # with the connection, so a pod SIGKILLed mid-transcode (OOMKill of PID 1, node
    # drain, node reboot) left the row ``pending`` with its attempt counter unwritten
    # and every later tick re-picked it forever — the in-Python ``max_attempts`` cap
    # only ever ran on the caught-exception path, so it could not engage for that
    # class. The worker now stamps this (counter bumped, COMMITTED) before it starts
    # ffmpeg; ``pending_video_sources`` skips rows leased into the future, and a dead
    # pod's lease simply EXPIRES instead of needing a reaper. NULL means unleased —
    # every terminal write (ready/retry/failed) clears it back.
    preview_leased_until: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class VideoSourceUpload(Base):
    """A resumable CHUNKED upload session for one ``VideoSource`` (migration 0062).

    Cloudflare rejects any request body over 100 MB with a 413, so a 134 MB drone
    clip cannot arrive as a single PUT. The client instead slices the file into
    ``part_size`` parts, PUTs them one at a time, and the server composes the final
    object. This table is the session state for that.

    ⚠️ **Why this is a TABLE and not a dict in the pod.** The orchestrator runs
    ``replicas: 2`` behind an HPA (min 2 / max 5), so part *n* and part *n+1* land on
    DIFFERENT pods essentially every time. Any in-memory or on-pod-disk record of
    "which parts have arrived and how many bytes they carried" is invisible to the
    pod that serves the next PUT — the running-total cap would reset to zero on every
    hop and a client could stream unbounded bytes. The state must be in Postgres and
    it must be COMMITTED per part, not at the end. If you ever find yourself caching
    this row in a module-level dict "for speed", you have reintroduced exactly that
    hole.

    ``parts`` is the per-part ledger: ``{"1": 33554432, "2": 33554432, ...}`` mapping
    part number → byte count actually stored. JSONB (keys are part numbers as STRINGS
    — JSON object keys cannot be integers) rather than a child table because the whole
    map is read and rewritten atomically on every part PUT; a child table would buy
    nothing and cost a join on the hot path. NOT NULL with a ``'{}'::jsonb`` default so
    a freshly-opened session is an empty object, never a NULL to guard on. It is also
    what makes a re-PUT of the same part idempotent (the key is simply overwritten)
    and what the running total is summed from to enforce the cumulative cap.

    ``source_id`` is PRE-ALLOCATED at create time, before a single byte arrives, so the
    final object key is fixed up-front: every part is written under a key derived from
    an id that already exists, and completing the upload inserts the ``VideoSource``
    row with THAT id. Minting it at completion instead would mean the parts were
    written under one key and the row published under another.

    ``bytes_declared`` is the client's claim about the total size, checked at create
    against the per-upload ceiling. It is NOT trusted afterwards — the authoritative
    number is the sum of ``parts``, because a client can declare 1 MiB and then stream
    forever. ``BigInteger`` for the same reason ``VideoSource.bytes`` is: a 4K original
    passes the 2 GiB int4 ceiling routinely.

    ``status`` is ``Text`` + ``CheckConstraint`` rather than a native pg ENUM — the
    convention every table here follows (0046/0051/0060) — so widening the vocabulary
    later is a drop/create CHECK, not a type migration. ``aborted`` is a real terminal
    state, distinct from a row simply left ``open``: the 24 h sweeper reaps abandoned
    ``open`` sessions, and an explicit abort must not have to wait a day.

    Project-scoped for RLS (join predicate climbs ``project_id -> projects.tenant_id``),
    exactly like ``VideoSource``: deliberately NO ``tenant_id`` column, so a session can
    never disagree with the tenant of the project that owns it. See
    ``RLS_JOIN_PREDICATES_VIDEO_SOURCE_UPLOADS`` in ``src/database/rls.py``; applied
    inline in migration 0062 for prod and via
    ``register_late_join_video_source_uploads_rls_ddl`` for ``create_all``. Both paths
    are load-bearing: under enforce, an unbound session reads ZERO rows, which presents
    to the user not as a permission error but as "my upload vanished mid-transfer".

    ``filename`` is the user's original name and is DISPLAY ONLY — never used to build a
    storage key or a filesystem path (the key comes from ``source_id`` + ``ext``).
    """

    __tablename__ = "video_source_uploads"
    __table_args__ = (
        CheckConstraint(
            "status IN ('open','completed','aborted')",
            name="ck_video_source_uploads_status",
        ),
        # The 24 h garbage sweeper's only query: "open sessions older than X". Leading
        # column is ``status`` so the scan starts from the small ``open`` slice rather
        # than walking every completed session ever recorded.
        Index("idx_video_source_uploads_status_created", "status", "created_at"),
    )

    # The ``upload_id`` handed back to the client and echoed on every part PUT.
    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    # The RLS climb key. CASCADE: a half-finished upload is meaningless without its
    # project, and deleting a project must not leave a dangling session behind.
    project_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Pre-allocated id of the ``VideoSource`` this upload will BECOME. Deliberately a
    # bare UUID and NOT an FK: the row does not exist yet (it is inserted at completion),
    # so an FK would reject every open session.
    source_id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), nullable=False)
    # Display only — the user's original name. Never a storage key, never a path.
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(Text, nullable=False)
    # Sanitized extension, carried separately so the final object key never has to be
    # re-derived from the untrusted ``filename`` at completion time.
    ext: Mapped[str] = mapped_column(Text, nullable=False)
    bytes_declared: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # Part size in bytes (32 MiB in practice) — fits int4 comfortably; only the TOTAL
    # needs BigInteger.
    part_size: Mapped[int] = mapped_column(Integer, nullable=False)
    part_count: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="open")
    # part number (as a JSON string key) → bytes stored for that part. Rewritten and
    # COMMITTED on every part PUT — this is what survives the pod hop.
    # ``dict[str, int]`` rather than ``dict[str, Any]``: the ``test_any_ratchet`` budget
    # is LOWER-ONLY and may never be raised, and the value here really is just a count.
    parts: Mapped[dict[str, int]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ServiceRun(Base):
    """One "test this GPU service" bench call (migration 0107).

    The service lab performs a REAL provider call for one slot (image / tts / asr /
    llm / vlm / ocr / video) and returns the media inline. THIS table is the history
    the bench lists: who asked for what, which provider served it, how long it took,
    and where the produced media lives.

    Media bytes are NOT stored here — ``media_object_key`` points at an object under
    ``media-cache/tenants/{tenant_id}/service-lab/`` (a prefix the bucket lifecycle
    rule already expires after ``media_cache_ttl_days``), and
    ``services.service_lab.store`` is the only writer. ``request`` carries the
    parameters only (NEVER uploaded bytes); ``provider`` is the ``ProviderInfo``
    (``source`` + ``kind`` + ``model``) that actually served the call.

    Rows are pruned opportunistically by ``delete_older_than`` on the same 30-day
    horizon as the media, so the table cannot outlive its objects. RLS
    ``tenant_isolation`` on ``tenant_id``, registered for ``create_all`` by
    ``register_service_runs_rls_ddl``, frozen inline in migration 0107.
    """

    __tablename__ = "service_runs"
    __table_args__ = (
        CheckConstraint("status IN ('succeeded','failed')", name="ck_service_runs_status"),
        CheckConstraint(
            "media_object_key IS NULL "
            "OR starts_with(media_object_key, 'media-cache/tenants/' || tenant_id::text "
            "|| '/service-lab/')",
            name="ck_service_runs_media_key",
        ),
        Index(
            "idx_service_runs_tenant_slot_created",
            "tenant_id",
            "slot",
            text("created_at DESC"),
        ),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    tenant_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    #: The subject that ran the bench call; NULL for a token without one.
    subject_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    #: Bench slot name — the invoker's vocabulary (``video``, not ``video_gen``).
    slot: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    #: image | audio | text | transcript | markdown | video (NULL for a failed run).
    output_kind: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: The request parameters. NEVER the uploaded bytes (see ``request_payload``).
    request: Mapped[dict[str, object]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    text_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    media_object_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    media_content_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: ``ProviderInfo`` (``source``/``kind``/``model``) of the provider that served it.
    provider: Mapped[dict[str, object]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    #: ``{"code", "message"}`` for a failed run; NULL when it succeeded.
    error: Mapped[dict[str, object] | None] = mapped_column(JSONB(none_as_null=True), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


# Phase D (RLS) — attach the tenant-isolation policies via ``after_create`` so a
# freshly ``create_all``-ed schema (tests/dev) carries the SAME RLS the Alembic
# migration applies in prod. Scoped to the tables that carry ``tenant_id``
# directly (``projects``). See ``src/database/rls.py``.
from src.database.rls import (  # noqa: E402  (must follow model defs)
    register_admin_functions_ddl,
    register_late_join_branches_rls_ddl,
    register_late_join_events_rls_ddl,
    register_late_join_panel_assets_rls_ddl,
    register_late_join_rls_ddl,
    register_late_join_snapshots_rls_ddl,
    register_late_join_video_source_uploads_rls_ddl,
    register_late_join_video_sources_rls_ddl,
    register_late_rls_ddl,
    register_provider_secret_gc_ddl,
    register_rls_ddl,
    register_run_functions_ddl,
    register_service_runs_rls_ddl,
    register_stage_step_results_rls_ddl,
    register_tenant_api_keys_rls_ddl,
    register_tenant_connections_rls_ddl,
    register_tenant_defaults_rls_ddl,
    register_tenant_occupancy_functions_ddl,
    register_tenant_sound_assets_rls_ddl,
    register_video_source_functions_ddl,
)
from src.database.stage_output_guard import (  # noqa: E402  (must follow model defs)
    register_stage_output_write_once_ddl,
)

register_rls_ddl(Base.metadata)
# Direct-tenant tables added after 0027 (source_ingest_cache) — same RLS via a
# separate registration so historical migrations keep iterating only projects.
register_late_rls_ddl(Base.metadata)
# Join-based tables added after 0028 (storyboard Phase 0) — ENABLE + join policy
# via a SEPARATE registration so historical migration 0028 keeps iterating only
# its own ``RLS_JOIN_PREDICATES`` set (0028 runs before 0046 on a fresh upgrade).
register_late_join_rls_ddl(Base.metadata)
# The append-only storyboard approval trail (storyboard_panel_events) added after
# 0046 — same 1-hop join RLS via its OWN separate registration so historical
# migration 0046 keeps iterating only its own RLS_JOIN_PREDICATES_LATE set.
register_late_join_events_rls_ddl(Base.metadata)
# The storyboard version-snapshot table (storyboard_snapshots) added after 0049 —
# same 1-hop join RLS via its OWN separate registration so historical migrations
# 0046/0049 keep iterating only their own late sets.
register_late_join_snapshots_rls_ddl(Base.metadata)
# The panel→asset join table (storyboard_panel_assets) added after 0050 — a 2-hop
# join RLS (panel_id -> storyboard_panels.project_id -> projects.tenant_id) via its
# OWN separate registration so historical migrations 0046/0049/0050 keep iterating
# only their own late sets.
register_late_join_panel_assets_rls_ddl(Base.metadata)
# The branch↔root link table (storyboard_branches) added after 0051 — a 1-hop join
# RLS (root_project_id -> projects.tenant_id) via its OWN separate registration so
# historical migrations 0046/0049/0050/0051 keep iterating only their own late sets.
register_late_join_branches_rls_ddl(Base.metadata)
# The uploaded-footage table (video_sources) added after 0052 — a 1-hop join RLS
# (project_id -> projects.tenant_id) via its OWN separate registration so historical
# migrations 0046/0049/0050/0051/0052 keep iterating only their own late sets.
register_late_join_video_sources_rls_ddl(Base.metadata)
# The chunked-upload session table (video_source_uploads) added after 0061 — a 1-hop
# join RLS (project_id -> projects.tenant_id) via its OWN separate registration so
# historical migrations 0046/0049/0050/0051/0052/0060 keep iterating only their own
# late sets. Without this, a create_all-ed test/dev schema would carry NO policy on
# the table while prod carries one — the drift that hides a tenant leak until prod.
register_late_join_video_source_uploads_rls_ddl(Base.metadata)
# The step ledger (stage_step_results) added in 0071 — the same 3-hop run-join RLS as
# stage_outputs, via its OWN registration so the frozen 0028/0029 snapshots (pinned
# to RLS_JOIN_PREDICATES) stay byte-identical.
register_stage_step_results_rls_ddl(Base.metadata)
# The P3 provider-connection tables (0077) — direct tenant policy on the connection,
# one-hop EXISTS on its versions, via their OWN registration (frozen inline in 0077).
register_tenant_connections_rls_ddl(Base.metadata)
# P6 tenant defaults (0081) — direct tenant policy via its OWN registration (frozen
# inline in 0081).
register_tenant_defaults_rls_ddl(Base.metadata)
# P6 tenant music library (0084) — direct tenant policy via its OWN registration
# (frozen inline in 0084).
register_tenant_sound_assets_rls_ddl(Base.metadata)
# Service lab run history (0107) — direct tenant policy via its OWN registration
# (frozen inline in 0107).
register_service_runs_rls_ddl(Base.metadata)
# Tenant API keys (0090) — direct tenant policy + the pre-tenant key resolvers
# (resolve_api_key / api_key_prefix_taken), via their OWN registration (frozen
# inline in 0090).
register_tenant_api_keys_rls_ddl(Base.metadata)
# The SECURITY DEFINER bootstrap/bypass resolvers (run_tenant / orphan_runs) the
# background sessions use under enforce — attached so create_all schemas get them.
register_run_functions_ddl(Base.metadata)
# The cross-tenant admin rollup resolver (admin_tenant_rollup) the platform
# tenants list uses under enforce — attached so create_all schemas get it too.
register_admin_functions_ddl(Base.metadata)
# The pending-video-source resolver (pending_video_sources) the preview-proxy
# sweeper uses under enforce — attached so create_all schemas carry it too. Without
# this a test/dev schema would have NO function and the sweeper would hard-error
# there while working in prod.
register_video_source_functions_ddl(Base.metadata)
# The tenant-occupancy resolver (tenant_has_projects) partner tenant creation uses
# to refuse a tenant id that already owns data — across RLS, so create_all too.
register_tenant_occupancy_functions_ddl(Base.metadata)
# P3 WP3: superseded provider-secret GC (SECURITY DEFINER, reads runs + versions).
register_provider_secret_gc_ddl(Base.metadata)

# The DB-level write-once guard on stage_outputs.output (migration 0063 in prod).
# Registered here for the same reason as everything above it: the whole test
# suite runs against a create_all-built schema, so a guard installed only by the
# migration chain would leave every test exercising an UNGATED table.
register_stage_output_write_once_ddl(Base.metadata)


# ── Login secrets never reach a row in plaintext ─────────────────────────────────
# `brief.auth_config` / `brief.auth_info` carry the password a recording stage types
# into a tenant's login form, and `run_setup` copies the brief into every run's
# `config_snapshot`. Sealing at the mapper (not per route) means no write path —
# create, PATCH, child, branch clone, regenerate, audit — can persist one readable.
# Idempotent (a sealed value is skipped) and fail-closed (no key ⇒ AuthSecretError,
# the flush aborts). See src/services/auth_secrets.py.
def _seal_column(attr: str) -> Callable[[object, object, object], None]:
    def _listener(_mapper: object, _connection: object, target: object) -> None:
        value = getattr(target, attr, None)
        if isinstance(value, dict) and has_plaintext_secret(value):
            setattr(target, attr, seal_auth_secrets(copy.deepcopy(value)))

    return _listener


for _model, _attr in ((Project, "brief"), (Run, "config_snapshot")):
    event.listen(_model, "before_insert", _seal_column(_attr))
    event.listen(_model, "before_update", _seal_column(_attr))
