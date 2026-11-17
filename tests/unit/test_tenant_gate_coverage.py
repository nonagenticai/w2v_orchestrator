"""Every mounted route is either GATED or EXPLICITLY exempt — no third state.

The tenant gate (`_common.TENANT_GATE`: require_known_tenant → apply_tenant_rls →
require_license) is attached PER ROUTER on each `include_router(...)` in
`src/asgi.py`, and the admin tier takes `LICENSE_GATE_ONLY` PER ROUTE. Nothing
structural forces a router (or a single route) added later to carry either, so a
forgotten `dependencies=` is a silent open hole: it serves tenant data to an
unknown / unentitled tenant and every other test stays green.

This test reads the REAL app (not a test-local reconstruction) and classifies
every route into exactly one of three explicit sets. A new route that is in none
of them fails here, and the author must consciously choose: gate it, or add it to
`_EXEMPT` with the reason. Stale entries also fail, so the lists cannot rot.

This test found `GET /api/admin/request-audit` ungated: the SuperAdmin request
audit pane, sibling of `GET /api/admin/audit`, which #1003 gave LICENSE_GATE_ONLY
and missed this one. It now carries the same gate.
"""

from __future__ import annotations

from collections.abc import Iterator

from fastapi.routing import APIRoute

#: Entitlement-only (`_common.LICENSE_GATE_ONLY`) — the admin tier + the
#: tenant-admin surface (incl. P3 provider connections). Deliberately NOT the full gate; see `_common.LICENSE_GATE_ONLY` and
#: tests/api/routes/test_admin_license_gate.py for why.
_LICENSE_ONLY: frozenset[tuple[str, str]] = frozenset(
    {
        ("GET", "/api/admin/overview"),
        ("GET", "/api/admin/usage"),
        ("GET", "/api/admin/runs"),
        # super_admin cross-tenant cancel: binds the RUN's tenant via run_tenant().
        ("POST", "/api/admin/runs/{run_id}/cancel"),
        ("GET", "/api/admin/audit"),
        ("GET", "/api/admin/request-audit"),
        # operator YouTube platform channel (super_admin, platform-wide config)
        ("GET", "/api/admin/youtube/platform-channel"),
        ("PUT", "/api/admin/youtube/platform-channel"),
        ("GET", "/api/admin/youtube/tenants"),
        # P2 WP4a tenant-admin surface (same posture as the admin tier)
        ("GET", "/api/tenant/members"),
        ("PATCH", "/api/tenant/members/{subject_id}"),
        ("DELETE", "/api/tenant/members/{subject_id}"),
        ("POST", "/api/tenant/invites"),
        ("GET", "/api/tenant/invites"),
        ("POST", "/api/tenant/invites/{invite_id}/resend"),
        ("DELETE", "/api/tenant/invites/{invite_id}"),
        ("PATCH", "/api/tenant"),
        # P2 WP4b: tenant_admin key management is product capability.
        ("GET", "/api/tenant/api-keys"),
        ("POST", "/api/tenant/api-keys"),
        ("DELETE", "/api/tenant/api-keys/{key_id}"),
        # P3 WP4: tenant provider connections (replaced /api/llm-config).
        ("GET", "/api/tenant/connections"),
        ("PUT", "/api/tenant/connections/{slot}"),
        ("POST", "/api/tenant/connections/{slot}/test"),
        ("DELETE", "/api/tenant/connections/{slot}"),
        ("POST", "/api/tenant/connections/{slot}/models"),
        # P8: bulk subtree PUT (tenant_admin OR an API key with connections:write).
        ("PUT", "/api/tenant/connections"),
        # P6 ORCH-1: tenant defaults WRITE is tenant-admin tier (the GET is a
        # member-readable data route and carries the full gate).
        ("PUT", "/api/tenant/defaults"),
        # P6 ORCH-3: read-only limits view, tenant-admin tier (reads no tenant table).
        ("GET", "/api/tenant/limits"),
        # P6 ORCH-5: tenant-scoped audit view, tenant-admin tier (+ RLS bind).
        ("GET", "/api/tenant/audit"),
        # P6 ORCH-4: tenant music library WRITES are tenant-admin tier (the GET is a
        # member-readable data route and carries the full gate).
        ("POST", "/api/tenant/music"),
        ("PATCH", "/api/tenant/music/{track_id}"),
        ("DELETE", "/api/tenant/music/{track_id}"),
    }
)

#: Ungated BY DESIGN. Every entry needs a reason; adding one is a security decision.
_EXEMPT: dict[tuple[str, str], str] = {
    # probes — must answer before/without any tenant
    ("GET", "/health"): "probe",
    ("GET", "/healthz"): "probe",
    ("GET", "/health/live"): "probe",
    ("GET", "/health/ready"): "probe",
    ("GET", "/readyz"): "probe",
    ("GET", "/metrics"): "prometheus scrape (PUBLIC_ENDPOINTS)",
    # a blocked tenant must still load identity → FE tenant-unavailable banner
    ("GET", "/api/auth/me"): "identity for the blocked-tenant banner",
    ("GET", "/api/license/status"): "tells an unlicensed tenant WHY it is blocked",
    # tenant-agnostic catalogue
    ("GET", "/api/blueprints"): "tenant-agnostic catalogue",
    ("GET", "/api/blueprints/{blueprint_id}"): "tenant-agnostic catalogue",
    # P8: static provider requirements manifest (registry + stage declarations)
    ("GET", "/api/provider-requirements"): "tenant-agnostic catalogue",
    # SSE: own RLS-bound ownership check; a router gate would pin a pool
    # connection for the stream's lifetime (the prod pool-exhaustion bug)
    ("GET", "/api/runs/{run_id}/events"): "SSE, self-checked ownership",
    ("GET", "/api/projects/{project_id}/events"): "SSE, self-checked ownership",
    # S2S usage report: route-local delegation (USAGE_REPORT_DELEGATING_SUBJECTS) —
    # the router gate would RLS-bind / known-check the SERVICE ACCOUNT's tenant. The
    # handler binds RLS to the effective tenant and 404s a run that tenant does not
    # own (tests/api/routes/test_run_usage_report_s2s.py).
    ("POST", "/api/runs/{run_id}/usage"): "S2S usage report, self-bound RLS + ownership",
    # circular: provisioning mints the licence the gate would demand
    ("POST", "/api/admin/provision"): "circular (mints the licence)",
    ("GET", "/api/admin/tenants"): "circular (operator reaches unlicensed tenant)",
    ("PATCH", "/api/admin/tenants/{tenant_id}"): "circular",
    ("POST", "/api/admin/scope-switch"): "circular",
    ("PUT", "/api/admin/tenants/{tenant_id}/youtube"): (
        "operator sets platform-channel policy on ANY tenant, licensed or not "
        "(same posture as PATCH /api/admin/tenants/{tenant_id}); super_admin-gated"
    ),
    # P2 WP4a: reached BEFORE the caller has a membership (NO_MEMBERSHIP_EXEMPT_PATHS)
    ("GET", "/api/me/tenants"): "caller's own memberships; tenant-less by design",
    ("POST", "/api/invites/accept"): "joins a tenant; caller has none yet",
    ("POST", "/api/invites/{invite_id}/accept"): "explicit join of a /me pending invite",
    # P2 WP4b: the partner licence lane CREATES tenants (circular) and checks
    # its own allow-listed, active licence in the handler.
    ("POST", "/api/tenants"): "circular (partner creates the tenant)",
    # P9-3: the partner mints/revokes machine keys for a tenant bound to ITS licence.
    # The caller's tenant is the PARTNER's own (licence lane), so the tenant gate
    # would gate the wrong tenant; the handler checks the allow-listed active
    # partner licence + the target tenant's binding. The minted key's USE is gated.
    ("POST", "/api/tenants/{tenant_id}/api-keys"): "partner lane; handler-checked binding",
    ("DELETE", "/api/tenants/{tenant_id}/api-keys/{key_id}"): (
        "partner lane; handler-checked binding"
    ),
    # Licence introspection (0089): the owning partner (handler-checked binding) or a
    # super_admin rotates the tenant's stored licence token — the entitlement this
    # gate would demand is exactly what is being repaired.
    ("PUT", "/api/tenants/{tenant_id}/licence-token"): (
        "partner lane / super_admin; handler-checked binding"
    ),
    # P9-5: operator binds ANY tenant (licensed or not) to a partner; super_admin-gated
    # (the PATCH /api/admin/tenants/{tenant_id} posture).
    ("PUT", "/api/admin/tenants/{tenant_id}/partner"): "operator; super_admin-gated",
    # Sets the posture of a tenant that may not be licensed yet (provision /
    # early-access approve) — gating it on entitlement would deadlock onboarding.
    ("GET", "/api/admin/tenants/{tenant_id}/llm-access"): "operator; super_admin-gated",
    ("PUT", "/api/admin/tenants/{tenant_id}/llm-access"): "operator; super_admin-gated",
    # P6 ORCH-6 offboarding. Tenant side: asking for your data back (or for its
    # erasure) must keep working after the licence lapses; TenantAdminDep in the
    # handler, and the POST only records a request a super_admin decides.
    ("POST", "/api/tenant/offboarding"): "data-subject request; reachable after licence lapse",
    ("GET", "/api/tenant/offboarding"): "own export download; reachable after licence lapse",
    ("GET", "/api/tenant/offboarding/{job_id}/download"): (
        "own export bytes; reachable after licence lapse"
    ),
    # Operator side: super_admin acts on ANY tenant, licensed or not (the
    # PATCH /api/admin/tenants/{tenant_id} posture).
    ("GET", "/api/admin/offboarding"): "operator queue across tenants; super_admin-gated",
    ("GET", "/api/admin/offboarding/{job_id}"): "operator; super_admin-gated",
    ("GET", "/api/admin/offboarding/{job_id}/download"): "operator; super_admin-gated",
    ("POST", "/api/admin/offboarding/{job_id}/approve"): "operator; super_admin-gated",
    ("POST", "/api/admin/offboarding/{job_id}/reject"): "operator; super_admin-gated",
    ("POST", "/api/admin/tenants/{tenant_id}/export"): "operator; super_admin-gated",
    ("GET", "/api/admin/tenants/{tenant_id}/offboarding-plan"): "operator; super_admin-gated",
    ("DELETE", "/api/admin/tenants/{tenant_id}"): "operator; super_admin-gated",
    # Early-access signup. The two service doors carry a Keycloak SERVICE-ACCOUNT
    # token (no tenant) and verify it themselves (azp/aud/role, never super_admin);
    # the super_admin routes are how a tenant comes to exist (circular, like
    # POST /api/admin/provision). All 503 unless W2V_SIGNUP_ENABLED=true.
    ("POST", "/api/signup-approvals/intake"): "SA door; own token check, no tenant",
    ("POST", "/api/signup-approvals/{signup_id}/telegram-decision"): (
        "SA door (chat-bridge); own token check, no tenant"
    ),
    ("GET", "/api/admin/signup-approvals"): "operator; super_admin-gated",
    ("GET", "/api/admin/signup-approvals/{signup_id}"): "operator; super_admin-gated",
    ("POST", "/api/admin/signup-approvals/invite"): "circular (founds the tenant)",
    ("POST", "/api/admin/signup-approvals/{signup_id}/approve"): "circular (founds the tenant)",
    ("POST", "/api/admin/signup-approvals/{signup_id}/reject"): "operator; super_admin-gated",
    ("POST", "/api/admin/signup-approvals/{signup_id}/retry"): "circular (founds the tenant)",
    ("POST", "/api/admin/signup-approvals/{signup_id}/remove"): (
        "operator; super_admin-gated, removes a tenant (it suspends + revokes its licence)"
    ),
    ("POST", "/api/admin/signup-approvals/{signup_id}/removal/retry"): (
        "operator; super_admin-gated, the tenant is suspended / unlicensed / deleted"
    ),
    ("POST", "/api/admin/signup-approvals/{signup_id}/removal/resolve"): (
        "operator; super_admin-gated, no tenant data"
    ),
}

#: Non-APIRoute routes FastAPI itself installs (schema/docs). Anything else that
#: is not an APIRoute (app.mount, a websocket, a raw Starlette Route) bypasses
#: dependency injection entirely and must be reviewed by hand.
_FRAMEWORK_ROUTES: frozenset[str] = frozenset(
    {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
)


def _walk(routes: list, inherited: tuple = ()) -> Iterator[tuple[object, tuple]]:
    """Yield (route, effective deps). include_router is lazy here: router-level
    deps live on `include_context`, per-route ones on the route — a caller is
    gated by the UNION, so walk both (same read as test_admin_license_gate)."""
    for route in routes:
        ctx = getattr(route, "include_context", None)
        if ctx is not None:
            yield from _walk(ctx.included_router.routes, inherited + tuple(ctx.dependencies))
            continue
        deps = inherited + tuple(route.dependencies) if isinstance(route, APIRoute) else ()
        yield route, deps


def _classify() -> tuple[dict[tuple[str, str], list[object]], list[str]]:
    from src.asgi import app  # lazy: autouse env fixtures must run first

    api: dict[tuple[str, str], list[object]] = {}
    other: list[str] = []
    for route, deps in _walk(app.routes):
        if isinstance(route, APIRoute):
            for method in route.methods or ():
                api[(method, route.path)] = [d.dependency for d in deps]
        else:
            other.append(getattr(route, "path", repr(route)))
    return api, other


def test_every_api_route_is_gated_or_explicitly_exempt() -> None:
    from src.api.routes._common import LICENSE_GATE_ONLY, TENANT_GATE

    full = [d.dependency for d in TENANT_GATE]
    lic = [d.dependency for d in LICENSE_GATE_ONLY]
    api, _ = _classify()

    ungated: list[str] = []
    wrong: list[str] = []
    for key, deps in sorted(api.items()):
        if key in _EXEMPT:
            if any(d in deps for d in full):
                wrong.append(f"{key} is listed EXEMPT but carries a gate dep: {deps}")
            continue
        expected = lic if key in _LICENSE_ONLY else full
        missing = [getattr(d, "__name__", repr(d)) for d in expected if d not in deps]
        if missing:
            ungated.append(f"{key} missing {missing}")
        if key in _LICENSE_ONLY and any(d in deps for d in full if d not in lic):
            wrong.append(f"{key} is LICENSE_ONLY but carries the full gate: {deps}")

    assert not ungated, (
        "OPEN HOLE — route(s) without the tenant gate and not in _EXEMPT. Mount the "
        "router with dependencies=_tenant_gate in src/asgi.py, or add an "
        "explicit, reasoned _EXEMPT entry:\n  " + "\n  ".join(ungated)
    )
    assert not wrong, "\n".join(wrong)


def test_no_route_bypasses_dependency_injection() -> None:
    """app.mount / websocket / raw Starlette routes never run route dependencies."""
    _, other = _classify()
    unexpected = sorted(set(other) - _FRAMEWORK_ROUTES)
    assert not unexpected, f"non-APIRoute route(s) bypass the tenant gate: {unexpected}"


def test_allowlists_have_no_stale_entries() -> None:
    """A removed/renamed route must leave the lists too, or they silently widen."""
    api, _ = _classify()
    stale = sorted((set(_EXEMPT) | _LICENSE_ONLY) - set(api))
    assert not stale, f"allowlist entries match no mounted route: {stale}"
    assert not (set(_EXEMPT) & _LICENSE_ONLY), "a route cannot be both EXEMPT and gated"


def test_the_gate_list_still_has_teeth() -> None:
    """Guard against the check going vacuous: most routes must be FULL-gated."""
    from src.api.routes._common import TENANT_GATE

    full = [d.dependency for d in TENANT_GATE]
    assert len(full) == 3
    api, _ = _classify()
    gated = [k for k, deps in api.items() if all(d in deps for d in full)]
    assert len(gated) >= 50, f"only {len(gated)} fully gated routes — mounts lost their gate?"
