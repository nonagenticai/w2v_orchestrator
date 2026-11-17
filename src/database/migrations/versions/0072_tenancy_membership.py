"""tenancy membership: tenant_members becomes authoritative (P2, WP1)

P2 "tenancy owned by w2v" moves the tenant + role decision from the Keycloak claim
(``tenant_id`` attribute + ``member``/``tenant_admin`` realm roles) to w2v's own
``tenant_members`` table. This revision only adds the SCHEMA and the pure-SQL part
of the backfill; nothing on the request path reads the new columns yet (the
resolver ships in WP3 behind ``W2V_TENANCY_SOURCE``). ``record_seen`` keeps
working unchanged: every new column has a server default (or is nullable).

``tenant_members`` (created in 0024):

* ``role`` ∈ {member, tenant_admin}, default ``member`` — the tenant-scoped role
  that replaces the realm-wide Keycloak roles for w2v.
* ``status`` ∈ {active, removed}, default ``active``; ``removed_at`` is set iff
  ``status = 'removed'`` (``ck_tenant_members_removed_at``).
* ``invited_by`` (subject uuid, nullable) and ``created_at`` (seeded from
  ``first_seen_at`` for existing rows).
* partial index on ``subject_id`` WHERE active — the request-path lookup
  "sub → active memberships".

``tenants`` (created in 0021):

* ``status`` CHECK widened with ``licence_lapsed`` (set by the WP4b licence
  refresh; nothing writes it yet).
* ``licence_ref`` / ``licence_checked_at`` — the w2v licence entitling the
  tenant, and when it was last checked.
* ``created_via`` ∈ {super_admin, partner, backfill}. Pre-existing rows are
  stamped ``backfill``; the column default for new rows is ``super_admin``
  (today the only writers are the super-admin PATCH + provision routes).
* ``partner_licence_id`` — the partner-key licence that created the tenant.

Backfill (pure SQL, idempotent):

* ``role = 'tenant_admin'`` only for rows whose ``roles`` snapshot contains an
  explicit ``tenant_admin``. A ``super_admin``-only row stays ``member``:
  super_admin authority comes from the realm role at request time, never from
  membership, and a super_admin's roster holds a row for EVERY tenant it merely
  visited (scope switch) — mapping those to tenant_admin would grant tenant
  admin across tenants once membership mode is on. The migration cannot know a
  subject's home tenant; ``scripts/backfill_tenancy.py`` (Keycloak-aware) marks
  the scope-switch rows ``removed`` and sets the home row's role from the live
  ``tenant_admin`` realm role before the cutover.
* a ``tenants`` row (``created_via='backfill'``) for every tenant that has a
  member, so a membership can never point at an unknown tenant.

No RLS: membership lookup is cross-tenant by design (a subject's memberships
span tenants). Drift-tolerant: ``IF NOT EXISTS`` / ``DROP ... IF EXISTS`` on
every object; DDL is inline literal text (no live metadata / rls.py import).

Revision ID: 0072_tenancy_membership
Revises: 0071_stage_step_results
Create Date: 2026-10-08
"""

from alembic import op

# ``alembic_version.version_num`` is varchar(32); this id is 23 characters.
revision = "0072_tenancy_membership"
down_revision = "0071_stage_step_results"
branch_labels = None
depends_on = None

_UPGRADE: tuple[str, ...] = (
    # ── tenant_members ──────────────────────────────────────────────────────
    "ALTER TABLE tenant_members ADD COLUMN IF NOT EXISTS role text NOT NULL DEFAULT 'member'",
    "ALTER TABLE tenant_members ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT 'active'",
    "ALTER TABLE tenant_members ADD COLUMN IF NOT EXISTS invited_by uuid",
    "ALTER TABLE tenant_members ADD COLUMN IF NOT EXISTS created_at timestamptz "
    "NOT NULL DEFAULT now()",
    "ALTER TABLE tenant_members ADD COLUMN IF NOT EXISTS removed_at timestamptz",
    # Existing rows: membership began when the subject was first seen.
    "UPDATE tenant_members SET created_at = first_seen_at WHERE created_at > first_seen_at",
    "ALTER TABLE tenant_members DROP CONSTRAINT IF EXISTS ck_tenant_members_role",
    "ALTER TABLE tenant_members ADD CONSTRAINT ck_tenant_members_role "
    "CHECK (role IN ('member','tenant_admin'))",
    "ALTER TABLE tenant_members DROP CONSTRAINT IF EXISTS ck_tenant_members_status",
    "ALTER TABLE tenant_members ADD CONSTRAINT ck_tenant_members_status "
    "CHECK (status IN ('active','removed'))",
    "ALTER TABLE tenant_members DROP CONSTRAINT IF EXISTS ck_tenant_members_removed_at",
    "ALTER TABLE tenant_members ADD CONSTRAINT ck_tenant_members_removed_at "
    "CHECK ((status = 'removed') = (removed_at IS NOT NULL))",
    "CREATE INDEX IF NOT EXISTS idx_tenant_members_subject_active "
    "ON tenant_members (subject_id) WHERE status = 'active'",
    # Role backfill from the realm-role snapshot. ``?`` = the string is a
    # top-level element of the jsonb array. ONLY tenant_admin — never
    # super_admin (scope-switch rows would become cross-tenant admin grants).
    "UPDATE tenant_members SET role = 'tenant_admin' "
    "WHERE role <> 'tenant_admin' AND jsonb_typeof(roles) = 'array' "
    "AND roles ? 'tenant_admin'",
    # ── tenants ─────────────────────────────────────────────────────────────
    "ALTER TABLE tenants DROP CONSTRAINT IF EXISTS ck_tenants_status",
    "ALTER TABLE tenants ADD CONSTRAINT ck_tenants_status "
    "CHECK (status IN ('active','suspended','licence_lapsed'))",
    "ALTER TABLE tenants ADD COLUMN IF NOT EXISTS licence_ref uuid",
    "ALTER TABLE tenants ADD COLUMN IF NOT EXISTS licence_checked_at timestamptz",
    # Added with DEFAULT 'backfill' so every pre-existing row is stamped as such
    # in the same statement, then the default for NEW rows becomes 'super_admin'.
    "ALTER TABLE tenants ADD COLUMN IF NOT EXISTS created_via text NOT NULL DEFAULT 'backfill'",
    "ALTER TABLE tenants ALTER COLUMN created_via SET DEFAULT 'super_admin'",
    "ALTER TABLE tenants ADD COLUMN IF NOT EXISTS partner_licence_id uuid",
    "ALTER TABLE tenants DROP CONSTRAINT IF EXISTS ck_tenants_created_via",
    "ALTER TABLE tenants ADD CONSTRAINT ck_tenants_created_via "
    "CHECK (created_via IN ('super_admin','partner','backfill'))",
    # Every tenant that has a member gets a registry row.
    "INSERT INTO tenants (id, created_via) "
    "SELECT DISTINCT tenant_id, 'backfill' FROM tenant_members "
    "ON CONFLICT (id) DO NOTHING",
)

_DOWNGRADE: tuple[str, ...] = (
    "ALTER TABLE tenants DROP CONSTRAINT IF EXISTS ck_tenants_created_via",
    "ALTER TABLE tenants DROP COLUMN IF EXISTS partner_licence_id",
    "ALTER TABLE tenants DROP COLUMN IF EXISTS created_via",
    "ALTER TABLE tenants DROP COLUMN IF EXISTS licence_checked_at",
    "ALTER TABLE tenants DROP COLUMN IF EXISTS licence_ref",
    # A lapsed tenant has no pre-P2 spelling; the nearest is "blocked".
    "UPDATE tenants SET status = 'suspended' WHERE status = 'licence_lapsed'",
    "ALTER TABLE tenants DROP CONSTRAINT IF EXISTS ck_tenants_status",
    "ALTER TABLE tenants ADD CONSTRAINT ck_tenants_status CHECK (status IN ('active','suspended'))",
    "DROP INDEX IF EXISTS idx_tenant_members_subject_active",
    "ALTER TABLE tenant_members DROP CONSTRAINT IF EXISTS ck_tenant_members_removed_at",
    "ALTER TABLE tenant_members DROP CONSTRAINT IF EXISTS ck_tenant_members_status",
    "ALTER TABLE tenant_members DROP CONSTRAINT IF EXISTS ck_tenant_members_role",
    "ALTER TABLE tenant_members DROP COLUMN IF EXISTS removed_at",
    "ALTER TABLE tenant_members DROP COLUMN IF EXISTS created_at",
    "ALTER TABLE tenant_members DROP COLUMN IF EXISTS invited_by",
    "ALTER TABLE tenant_members DROP COLUMN IF EXISTS status",
    "ALTER TABLE tenant_members DROP COLUMN IF EXISTS role",
)


def upgrade() -> None:
    for statement in _UPGRADE:
        op.execute(statement)


def downgrade() -> None:
    for statement in _DOWNGRADE:
        op.execute(statement)
