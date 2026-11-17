"""``signup_approvals`` — early-access signup v2 (one row per request / invite)

A visitor asks for access on nonagentic.ai/early-access/w2v; the shared
early-access-intake handler relays it with a Keycloak service-account token
(``POST /api/signup-approvals/intake``); George approves with one Telegram tap
(chat-bridge → ``POST /api/signup-approvals/{id}/telegram-decision``) or on the
super_admin screen; the approve founds a licensed tenant, seats the requester's
tenant_admin invite and only THEN sends the one set-password email.

* One OPEN row per address: partial unique index on ``lower(email)`` WHERE
  ``status IN ('pending','approved')`` — a re-submit is an idempotent no-op, and
  early access is ONE workspace per person (an approved row blocks a second
  request); a REJECTED row does not block a new request.
* ``locale`` (C6) is CHECKed to the eight theme locales; NULL = English.
* ``approving_until`` / ``approving_claim`` — the approve claim (one approver at a
  time, across replicas); the decision UPDATE requires the claim token.
* ``tenant_slug`` is reserved on the row (partial UNIQUE) before anything is
  created; ``tenant_id`` is minted ONCE with the decision so every retry founds
  the same tenant.
* ``steps`` — per provisioning step ``{status, at, code, error, detail}``;
  ``steps_lease_*`` is the ONE lease approve, the 5-minute tick and the
  super_admin Retry take before running a step (the email is never sent twice in
  parallel); ``steps_next_attempt_at`` / ``steps_attempts`` the tick's backoff.

No RLS: a platform table, read only by super_admin surfaces and the two service
doors (never tenant-scoped). Runtime-role privileges ride ``ALTER DEFAULT
PRIVILEGES`` (gitops), like every table.

Downgrade REFUSES while any row exists (a signup record has no regeneration
path); on an empty table it drops it. Frozen literal SQL; no application code.

Revision ID: 0098_signup_approvals
Revises: 0097_audit_run_cancel
Create Date: 2026-10-20
"""

from alembic import op

# ``alembic_version.version_num`` is varchar(32); this id is 21 characters.
revision = "0098_signup_approvals"
down_revision = "0097_audit_run_cancel"
branch_labels = None
depends_on = None

_UPGRADE: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS signup_approvals (
        id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        email text NOT NULL,
        name text NOT NULL,
        company text NULL,
        message text NULL,
        locale text NULL,
        source text NOT NULL,
        intake_client text NULL,
        status text NOT NULL DEFAULT 'pending',
        decided_at timestamptz NULL,
        decided_by text NULL,
        decided_via text NULL,
        decider_actor text NULL,
        approving_until timestamptz NULL,
        approving_claim uuid NULL,
        tenant_name text NULL,
        tenant_slug text NULL,
        tenant_id uuid NULL,
        steps jsonb NOT NULL DEFAULT '{}'::jsonb,
        steps_lease_until timestamptz NULL,
        steps_lease_token uuid NULL,
        steps_next_attempt_at timestamptz NULL,
        steps_attempts integer NOT NULL DEFAULT 0,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT ck_signup_approvals_message_len
            CHECK (message IS NULL OR char_length(message) <= 2000),
        CONSTRAINT ck_signup_approvals_locale
            CHECK (locale IS NULL OR locale IN ('ar','de','en','es','fr','he','nl','pl')),
        CONSTRAINT ck_signup_approvals_source CHECK (source IN ('web','chat','invite')),
        CONSTRAINT ck_signup_approvals_status
            CHECK (status IN ('pending','approved','rejected')),
        CONSTRAINT ck_signup_approvals_decided_via
            CHECK (decided_via IS NULL OR decided_via IN ('onboarding','telegram')),
        CONSTRAINT ck_signup_approvals_decided
            CHECK ((status = 'pending') = (decided_at IS NULL)),
        CONSTRAINT ck_signup_approvals_approved_tenant
            CHECK (status <> 'approved' OR (tenant_id IS NOT NULL AND tenant_slug IS NOT NULL)),
        CONSTRAINT ck_signup_approvals_tenant_slug
            CHECK (tenant_slug IS NULL OR tenant_slug ~ '^[a-z0-9-]{2,40}$'),
        CONSTRAINT ck_signup_approvals_steps_attempts CHECK (steps_attempts >= 0)
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_signup_approvals_open_email "
    "ON signup_approvals (lower(email)) WHERE status IN ('pending','approved')",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_signup_approvals_tenant_slug "
    "ON signup_approvals (tenant_slug) WHERE tenant_slug IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_signup_approvals_status_created "
    "ON signup_approvals (status, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_signup_approvals_steps_due "
    "ON signup_approvals (steps_next_attempt_at) WHERE steps_next_attempt_at IS NOT NULL",
)

#: Refuse to destroy signup records: there is no way to regenerate them.
_DOWNGRADE_GUARD = """
DO $$
BEGIN
    IF to_regclass('signup_approvals') IS NOT NULL
       AND EXISTS (SELECT 1 FROM signup_approvals) THEN
        RAISE EXCEPTION
            'refusing to drop signup_approvals: it holds rows (signup records cannot be '
            'regenerated); export and delete them deliberately first';
    END IF;
END $$
"""


def upgrade() -> None:
    for statement in _UPGRADE:
        op.execute(statement)


def downgrade() -> None:
    op.execute(_DOWNGRADE_GUARD)
    op.execute("DROP TABLE IF EXISTS signup_approvals")
