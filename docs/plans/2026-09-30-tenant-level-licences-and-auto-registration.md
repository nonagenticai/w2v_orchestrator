# Tenants with only tenant-level licences: auto-register, and stop minting per-user licences

Date: 2026-09-30
Repo: nonagenticai/w2v_orchestrator

**Staleness warning.** HTTPS fetch does not work on m4 (`could not read Username for 'https://github.com'`), so m4's `origin/main` may be behind GitHub. This document was checked against `origin/main` at `3393e66f` (commit date 2026-09-18). The local `main` checkout is `d874155c` (2026-09-15), 33 commits behind that. All file:line references below are for `3393e66f`. Re-check them against the real `main` before starting work. The w2v_frontend references come from m4's checkout at `9f2f415` (branch `feat/raise-footage-upload-cap-4gib`, which has no `origin` refs).

## Context

On 2026-09-29/30 we provisioned tenant `netcare` with tenant-level licences only:

- One licence per app, issued from bundles. The licence subject is `svc:<app>@<tenant>`.
- `byo_mode` is `local_allowed`.
- One admin user with the Keycloak `portal` realm role `tenant_admin` and attribute `tenant_id=<tenant uuid>`.
- No per-member licences. No `platform_operator` or `super_admin`.

Future customer tenants will be set up the same way. w2v assumes two things this model does not provide: a row in the local tenant registry, and licences issued per user.

## Findings

### A. A licensed tenant gets 403 "tenant not provisioned" until a super-admin adds it by hand

**Current behaviour**

- `require_known_tenant` (`src/api/routes/_common.py:438`) raises 403 "tenant not provisioned" at `:485` when enforcement mode is `enforce`. It lets a tenant through only if it has a registry row (`Tenant`, `src/database/models.py:34`, migration `src/database/migrations/versions/0021_tenants_registry.py`) or a project. That check is `AdminRepository.tenant_switchable` (`src/database/repositories/admin_repository.py:167`).
- The gate runs first in `TENANT_GATE` (`_common.py:639-643`): known tenant, then RLS, then `require_license` (`:600`). A licensed tenant is refused before its licence is even looked at.
- Only a super_admin can create a registry row: `PATCH /api/admin/tenants/{id}` (`src/api/routes/admin.py:211-248`, upsert at `:234`) or `POST /api/admin/provision` (`admin.py:484`, upsert at `:536`).
- Calls from a subject in `AUTH_DELEGATING_SUBJECTS` (for example marketing sending `X-Effective-Tenant-Id`) skip this check (`_common.py:454-477`). The comment there says ≥37 of 39 licensed tenants have no registry row. Direct users of a new tenant do not get this bypass.
- On 2026-09-30 a super-admin had to add netcare by hand.

**Why it matters**

Every new customer tenant's users get locked out of w2v until someone with super_admin steps in. The licence is already the entitlement, so the registry row adds nothing except a manual step.

**Recommended change**

- In `require_known_tenant`, when the effective tenant **is the caller's own tenant claim** (not a scope switch) and it is unknown, check licence status for `app_key=w2v`. If the tenant holds an entitled w2v licence (a tenant-level `svc:` licence counts), upsert the `Tenant` row and write an audit row `tenant.auto_register` (actor = subject, detail = licence id and app key). Then continue.
- Update `_known_tenant_cache` only after the upsert succeeds. The existing comment at `:469-476` explains why this cache must never be written on an unverified path.
- A super-admin switching into an unknown tenant must still get 404 (`_common.py:399-400`). Auto-registration never applies to a scope switch.
- Optional: add `PUT /api/admin/tenants/{id}` for an explicit allowlist of provisioning subjects. Suggested setting name `PROVISIONING_SUBJECTS`; it does not exist yet. The provisioner can then create the row up front. Never give service accounts super_admin to do this.

**Tests**

- Unknown tenant, own claim, licence status shows an entitled w2v licence → 200, a `Tenant` row exists, and a `tenant.auto_register` audit row exists.
- Unknown tenant, own claim, no licence → 403 "tenant not provisioned", and no row is created.
- A super-admin scope switch to an unknown tenant → 404, and no row is created.
- In `warn` and `off` modes, behaviour does not change.

### B. When `/licenses/active` is refused, the licence status fallback looks up the user, not the tenant

**Current behaviour**

`_resolve_uncached` (`src/services/license_status.py:376`) calls `GET /licenses/active?tenant_id=&app_key=` (`:386`). If that returns 401 or 403, it logs `license_gate_misconfigured` and works out entitlement from `GET /licenses/user/{sub}` (`_fetch_inventory`, `:405`, which uses `src/clients/licenses_client.py:133`). If `/licenses/active` returns 404, it refines the answer from the same per-user inventory (`:414-418`). A tenant whose only licences are `svc:` licences has an empty per-user inventory, so this path resolves to `none`, and the gate returns 402.

Measured on 2026-09-30: the running `w2v-orchestrator` pod (started 00:13Z) logged 0 `license_gate_misconfigured` lines, so the primary route currently works and this bug is latent. It fires as soon as the w2v service account loses its licenses role.

**Why it matters**

A licences-service role change would quietly lock out every tenant that uses tenant-level licences, while tenants with per-user licences keep working. That makes it hard to spot.

**Recommended change**

- Codify the internal-service role that `/licenses/active` needs for the w2v-backend client, so the primary route is guaranteed.
- Make the fallback query tenant plus app (a tenant-scoped inventory, or `svc:w2v@<tenant>`) instead of `/licenses/user/{sub}`.

**Tests**

- `/licenses/active` returns 403, and the tenant holds only `svc:w2v@<tenant>` → state `active`, not `none`.
- `/licenses/active` returns 404 and the tenant has no licences → `none`.

### C. Self-serve auto-issue and admin provisioning mint licences per user

**Current behaviour**

- Self-serve: `src/api/routes/auth.py:56-69` calls `ensure_active_license` when state is `none` (`src/services/license_provisioning.py:38`, which calls `create_license` at `:81`). That becomes `POST /licenses/user/{sub}` (`src/clients/licenses_client.py:142`, request at `:157`).
- Admin provision: `admin.py:546-590` lists the user's licences (`:557`) and mints one licence per app for that user (`create_license(subject_id=user_uuid, …)` at `:578`).

**Why it matters**

This contradicts the tenant-level model. A netcare user who resolves to `none` (see B) would be given a personal licence. That duplicates entitlement, affects billing, and hides the real problem.

**Recommended change**

In both places, first check whether the tenant already holds a tenant-level w2v licence. If it does, mint nothing. If it does not, mint a tenant-level `svc:w2v@<tenant>` licence instead of a per-user one.

**Tests**

- The tenant holds `svc:w2v@<tenant>` → no `POST /licenses/user/*`.
- The tenant holds no w2v licence → exactly one tenant-level licence is created, and a retry does not create a second.
- An expired or revoked licence is still never resurrected (existing behaviour).

### D. Dead `/api/pipeline/ws?license=` route is still listed at the edge (gitops follow-up; no w2v change)

- The orchestrator has no `/api/pipeline/ws` route. It was removed with the Studio v3 reset (`8800a4cf`, "PR-1: Studio v3 foundation", which deleted the legacy WS and connection manager), and a grep of `src/` at `3393e66f` finds no match.
- The SPA could not send `?license=` anyway: for Keycloak logins `license_token` is `null` (w2v_frontend `src/lib/api/auth.ts:22-29`, `src/lib/api/client.ts:31-52`), and w2v_frontend has no `pipeline/ws` caller.
- The edge still carries it: gitops-production `clusters/strato/policies/envoy-filters/ext-authz-enforce.yaml` (`WS_LICENSE_PATHS` at `:354-369`, used at `:740-741`) and `ci/ingress-auth-exempt.txt:102`. Whoever owns the edge filter should remove it.

## Out of scope / related

- Marketing has the matching problems on its side (static per-tenant token maps, and operator-only UI calls): `nonagenticai/marketing` `docs/plans/2026-09-30-tenant-admin-operator-gating.md`.
- Licences-service role assignment for the w2v service account (B) is a licenses/Keycloak change and is tracked there.

## Verification

- For a fresh tenant provisioned like netcare (tenant-level licences only, no registry row), its tenant admin loads w2v with no super-admin step. A `tenants` row and a `tenant.auto_register` audit row appear.
- `GET /api/license/status` for that user returns `active`, and no per-user licence is created in licenses.
- A super-admin switch into an unrelated unknown tenant still returns 404.
