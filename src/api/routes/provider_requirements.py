"""``GET /api/provider-requirements`` — the static P8 requirements manifest.

Tenant-agnostic (no tenant table is read): any authenticated caller — a member's
JWT, a tenant API key, a partner licence — may read it; ``AuthMiddleware`` still
rejects anonymous calls. Exempt from the tenant gate for the same reason as
``/api/blueprints`` (``tests/unit/test_tenant_gate_coverage.py``).
"""

from __future__ import annotations

from fastapi import APIRouter

import src.services.blueprints  # noqa: F401  triggers blueprint registration
from src.api.routes.tenancy.connections import PolicyDep
from src.api.schemas.provider_requirements import ProviderRequirements
from src.services.blueprints.registry import get_registry
from src.services.providers.manifest import build_manifest

router = APIRouter(prefix="/api/provider-requirements", tags=["providers"])


@router.get("", response_model=ProviderRequirements, operation_id="provider_requirements.get")
async def get_provider_requirements(policy: PolicyDep) -> ProviderRequirements:
    return build_manifest(policy, get_registry())
