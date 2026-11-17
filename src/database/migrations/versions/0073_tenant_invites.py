"""tenant_invites: invite-by-email into a w2v tenant (P2, WP1)

A tenant_admin invites an email address with a role; acceptance (WP4a) binds the
accepting Keycloak ``sub`` to a ``tenant_members`` row when its VERIFIED email
matches. Only the SHA-256 of the one-time token is stored (``token_hash``), never
the token itself.

* ``status`` ∈ {pending, accepted, revoked, expired}; ``accepted_subject_id`` is
  set iff ``status = 'accepted'`` (``ck_tenant_invites_accepted``).
* at most ONE pending invite per ``(tenant_id, lower(email))`` — the partial
  unique index; a resend rotates the token on the same row.
* ``email`` is ``text`` + ``lower(email)`` indexes, NOT ``citext``: ``citext``
  needs ``CREATE EXTENSION`` on the database, and the case-insensitive match
  this table needs is exactly what the ``lower(email)`` indexes give.
* ``kc_user_created`` — w2v created the Keycloak user for this invite (so a
  later member removal may disable it if it has no other memberships).
* FK to ``tenants`` ON DELETE CASCADE: an invite is meaningless without its
  tenant (tenants are suspended, not deleted, in practice).

No RLS: acceptance looks an invite up by token hash / email ACROSS tenants by
design. Drift-tolerant (``IF NOT EXISTS``); DDL is inline literal text.

Revision ID: 0073_tenant_invites
Revises: 0072_tenancy_membership
Create Date: 2026-10-08
"""

from alembic import op

# ``alembic_version.version_num`` is varchar(32); this id is 19 characters.
revision = "0073_tenant_invites"
down_revision = "0072_tenancy_membership"
branch_labels = None
depends_on = None

_UPGRADE: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS tenant_invites (
        id uuid NOT NULL,
        tenant_id uuid NOT NULL,
        email text NOT NULL,
        role text NOT NULL DEFAULT 'member',
        token_hash bytea NOT NULL,
        expires_at timestamptz NOT NULL,
        status text NOT NULL DEFAULT 'pending',
        invited_by uuid,
        kc_user_created boolean NOT NULL DEFAULT false,
        accepted_subject_id uuid,
        accepted_at timestamptz,
        created_at timestamptz NOT NULL DEFAULT now(),
        resent_at timestamptz,
        CONSTRAINT tenant_invites_pkey PRIMARY KEY (id),
        CONSTRAINT tenant_invites_tenant_id_fkey FOREIGN KEY (tenant_id)
            REFERENCES tenants (id) ON DELETE CASCADE,
        CONSTRAINT uq_tenant_invites_token_hash UNIQUE (token_hash),
        CONSTRAINT ck_tenant_invites_role CHECK (role IN ('member','tenant_admin')),
        CONSTRAINT ck_tenant_invites_status
            CHECK (status IN ('pending','accepted','revoked','expired')),
        CONSTRAINT ck_tenant_invites_accepted
            CHECK ((status = 'accepted') = (accepted_subject_id IS NOT NULL)),
        CONSTRAINT ck_tenant_invites_token_hash_len CHECK (octet_length(token_hash) = 32)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_tenant_invites_tenant ON tenant_invites (tenant_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_tenant_invites_pending_email "
    "ON tenant_invites (tenant_id, lower(email)) WHERE status = 'pending'",
    "CREATE INDEX IF NOT EXISTS idx_tenant_invites_pending_email "
    "ON tenant_invites (lower(email)) WHERE status = 'pending'",
)

_DOWNGRADE: tuple[str, ...] = ("DROP TABLE IF EXISTS tenant_invites",)


def upgrade() -> None:
    for statement in _UPGRADE:
        op.execute(statement)


def downgrade() -> None:
    for statement in _DOWNGRADE:
        op.execute(statement)
