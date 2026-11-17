"""Lock the OpenAPI surface so accidental schema changes fail CI.

The frontend's openapi-typescript codegen consumes this surface. Any new
PR adding/changing routes MUST update this test in lockstep so the FE
PR is forced to regenerate types (T15 enforces that with a
``git diff --exit-code`` after codegen).
"""

import pytest


@pytest.fixture
def openapi() -> dict:
    from src.asgi import app

    return app.openapi()


def test_audit_brief_generate_video_in_schema(openapi: dict) -> None:
    """AuditBrief.generate_video is the typed flag the FE consumes to drive the
    optional narrated-video chain — lock it in the published surface so codegen
    picks it up."""
    props = openapi["components"]["schemas"]["AuditBrief"]["properties"]
    assert "generate_video" in props


def test_tenant_connections_routes_present(openapi: dict) -> None:
    """P3 tenant provider connections (the Providers page) — replaced the WM BYO
    proxy ``/api/llm-config``, which must stay gone (no shim).
    """
    paths = openapi["paths"]
    expected = [
        ("/api/tenant/connections", "get"),
        ("/api/tenant/connections/{slot}", "put"),
        ("/api/tenant/connections/{slot}", "delete"),
        ("/api/tenant/connections/{slot}/test", "post"),
        ("/api/tenant/connections/{slot}/models", "post"),
    ]
    for path, method in expected:
        assert path in paths, f"missing path {path}"
        assert method in paths[path], f"missing {method} on {path}"
    assert not [p for p in paths if p.startswith("/api/llm-config")]


def test_project_create_request_includes_deliverables(openapi: dict) -> None:
    schema = openapi["components"]["schemas"]["ProjectCreate"]
    assert "deliverables" in schema["properties"]
    assert schema["properties"]["deliverables"]["minItems"] == 1


def test_no_legacy_pipeline_paths(openapi: dict) -> None:
    bad = [p for p in openapi["paths"] if p.startswith("/api/pipeline")]
    assert bad == [], f"legacy /api/pipeline paths must be gone: {bad}"


def test_unique_operation_ids(openapi: dict) -> None:
    seen: dict[str, str] = {}
    for path, ops in openapi["paths"].items():
        for method, op in ops.items():
            if not isinstance(op, dict):
                continue
            opid = op.get("operationId")
            assert opid is not None, f"{method} {path} missing operationId"
            assert opid not in seen, f"duplicate operationId {opid}: {seen[opid]} + {method} {path}"
            seen[opid] = f"{method} {path}"


def test_admin_members_and_role_routes_are_gone(openapi: dict) -> None:
    """P2 WP4b: the Keycloak realm-role route and the admin roster moved to
    w2v-owned membership (/api/tenant/members, WP4a) — they must not linger."""
    paths = openapi["paths"]
    assert "/api/admin/members" not in paths
    assert "/api/admin/members/{subject_id}/roles" not in paths
    schemas = openapi["components"]["schemas"]
    for name in ("AdminMember", "AdminMemberList", "AdminMemberRoleUpdate"):
        assert name not in schemas, f"stale schema {name}"


def test_admin_provision_surface_present(openapi: dict) -> None:
    """Super-admin provisioning route + schemas; no temporary password any more."""
    paths = openapi["paths"]
    assert "/api/admin/provision" in paths, "missing /api/admin/provision"
    assert "post" in paths["/api/admin/provision"], "missing post on /api/admin/provision"
    assert paths["/api/admin/provision"]["post"]["operationId"] == "admin.provision"
    schemas = openapi["components"]["schemas"]
    for name in ("AdminProvisionRequest", "AdminProvisionResult"):
        assert name in schemas, f"missing schema {name}"
    result = schemas["AdminProvisionResult"]["properties"]
    assert "temporary_password" not in result
    assert "invite_id" in result


def test_tenancy_wp4b_surface_present(openapi: dict) -> None:
    """Tenant API keys + partner tenant creation (FE codegen lockstep, P2 WP4b)."""
    paths = openapi["paths"]
    keys = paths["/api/tenant/api-keys"]
    assert keys["get"]["operationId"] == "tenant.api_keys.list"
    assert keys["post"]["operationId"] == "tenant.api_keys.create"
    revoke = paths["/api/tenant/api-keys/{key_id}"]["delete"]
    assert revoke["operationId"] == "tenant.api_keys.revoke"
    assert paths["/api/tenants"]["post"]["operationId"] == "tenants.partner_create"
    schemas = openapi["components"]["schemas"]
    for name in ("ApiKeySummary", "ApiKeyCreated", "ApiKeyList", "PartnerTenantCreate"):
        assert name in schemas, f"missing schema {name}"
    # The list shape must never carry the secret.
    assert "key" not in schemas["ApiKeySummary"]["properties"]


def test_admin_usage_surface_present(openapi: dict) -> None:
    """Admin usage breakdown route + schemas (FE codegen lockstep)."""
    paths = openapi["paths"]
    assert "/api/admin/usage" in paths, "missing /api/admin/usage"
    assert "get" in paths["/api/admin/usage"], "missing get on /api/admin/usage"
    assert paths["/api/admin/usage"]["get"]["operationId"] == "admin.usage.detail"
    schemas = openapi["components"]["schemas"]
    for name in (
        "AdminUsageDetail",
        "AdminUsageTotals",
        "AdminUsageByType",
        "AdminUsageByStage",
        "AdminUsageByModel",
        "AdminUsagePoint",
    ):
        assert name in schemas, f"missing schema {name}"
    # The time-series is part of the detail response (increment 7).
    assert "time_series" in schemas["AdminUsageDetail"]["properties"]
    # Increment 8: the chart's bucket + window-day query params (FE codegen lockstep).
    params = {
        p["name"] for p in paths["/api/admin/usage"]["get"].get("parameters", []) if "name" in p
    }
    assert "bucket" in params, "missing bucket query param"
    assert "window_days" in params, "missing window_days query param"


def test_admin_tenants_summary_carries_usage_rollup(openapi: dict) -> None:
    """The platform tenants list surfaces per-tenant run_count + cost (increment 7)."""
    props = openapi["components"]["schemas"]["AdminTenantSummary"]["properties"]
    assert "run_count" in props
    assert "total_cost_usd" in props


@pytest.mark.asyncio
async def test_blueprints_endpoint_lists_all_registered(client) -> None:
    """At runtime ``/api/blueprints`` enumerates every registered blueprint."""
    res = await client.get("/api/blueprints")
    assert res.status_code == 200
    body = res.json()
    ids = {b["id"] for b in body}
    assert {
        "generic_v1",
        "promo_v1",
        "lesson_v1",
        "course_v1",
        "app_review_v1",
    } <= ids
    promo = next(b for b in body if b["id"] == "promo_v1")
    assert promo["project_type"] == "promo"
    stage_names = [s["name"] for s in promo["stages"]]
    assert stage_names == [
        "promo_analyze",
        "promo_script",
        "promo_storyboard",
        "promo_narrate",
        "promo_render",
        "promo_footage_critique",
        "promo_review",
        "growth_bundle",
        "promo_image_prompts",
        "ai_image_gen",
        "promo_visual_director",
        "promo_slides",
        "promo_footage_direction",
        "promo_sound_director",
        "promo_html",
        "ai_video_gen",
        "promo_motion_brief",
        "motion_graphics_gen",
        "site_capture",
        "source_ingest",
        "media_discovery",
    ]
    app_review = next(b for b in body if b["id"] == "app_review_v1")
    assert app_review["project_type"] == "app_review"
    assert [s["name"] for s in app_review["stages"]] == [
        "app_inspect",
        "app_critique",
        "app_review_script",
        "app_review_record",
        "app_review_compile",
        "growth_bundle",
    ]
    # action_reel: raw footage -> LLM segment pick -> beat-cut render. Pinned in
    # DECLARATION order (which is what the route serves), not topological order.
    action_reel = next(b for b in body if b["id"] == "action_reel_v1")
    assert action_reel["project_type"] == "action_reel"
    assert [s["name"] for s in action_reel["stages"]] == [
        "source_ingest",
        "footage_probe",
        "shot_analysis",
        "segment_select",
        "music_select",
        "cut_plan",
        "graphics_design",
        "narrate",
        "compile",
        "render_manifest",
    ]


def test_branch_context_surface_present(openapi: dict) -> None:
    """A branch project is readable by id but hidden from the list — the FE codegen
    needs ``is_branch`` + the ``branch`` block or it cannot render the "you are in a
    branch, go back to <root>" banner."""
    schemas = openapi["components"]["schemas"]
    assert "BranchContext" in schemas
    for field in ("branch_id", "name", "status", "root_project_id", "root_title", "ancestors"):
        assert field in schemas["BranchContext"]["properties"], f"missing {field}"
    # The lineage element the FE renders the branch-of-branch chain from.
    assert "BranchAncestor" in schemas
    for field in ("project_id", "title", "is_branch"):
        assert field in schemas["BranchAncestor"]["properties"], f"missing {field}"
    assert "is_branch" in schemas["ProjectResponse"]["properties"]
    assert "is_branch" in schemas["ProjectDetail"]["properties"]
    assert "branch" in schemas["ProjectDetail"]["properties"]


def test_explicit_invite_acceptance_surface(openapi: dict) -> None:
    """/api/auth/me lists pending invites (never accepts); joining is the
    explicit by-id accept. No Keycloak user ids in partner/provision results."""
    paths = openapi["paths"]
    accept = paths["/api/invites/{invite_id}/accept"]["post"]
    assert accept["operationId"] == "invites.accept_by_id"
    schemas = openapi["components"]["schemas"]
    me = schemas["AuthMe"]["properties"]
    assert "pending_invites" in me and "accepted_invites" not in me
    assert set(schemas["PendingInvite"]["properties"]) == {
        "id",
        "tenant_id",
        "tenant_name",
        "role",
        "expires_at",
    }
    assert "admin_user_id" not in schemas["PartnerTenantResult"]["properties"]
    assert "user_id" not in schemas["AdminProvisionResult"]["properties"]


def test_run_usage_report_route_present(openapi: dict) -> None:
    """S2S contract consumed by w2v_common's VLLMUsageReporter."""
    op = openapi["paths"]["/api/runs/{run_id}/usage"]["post"]
    assert op["operationId"] == "runs.report_usage"
    assert "202" in op["responses"]
    schemas = openapi["components"]["schemas"]
    body = schemas["RunUsageReport"]
    assert set(body["properties"]) == {
        "call_id",
        "stage",
        "model",
        "provider",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "service",
    }
    assert body.get("additionalProperties") is False
    assert set(schemas["RunUsageReportResult"]["properties"]) == {"recorded"}
