"""P2 tenancy routers (WP4): members, invites, the caller's tenants, API keys,
partner tenant creation; P3 WP4 tenant provider connections; P6 tenant defaults,
the read-only limits view, the tenant-scoped audit view, the tenant music
library and tenant offboarding (export + delete).

``src/asgi.py`` mounts every router in ``ROUTERS`` WITHOUT a router-level gate —
each WP4 route declares its own (``TenantAdminDep``, ``LICENSE_GATE_ONLY``, the
partner licence lane, …). A route a caller with NO membership must reach
(invite acceptance, ``/api/me/tenants``) also has to be listed in
``src/auth/tenancy_middleware.NO_MEMBERSHIP_EXEMPT_PATHS``.
"""

from fastapi import APIRouter

from src.api.routes.tenancy import api_keys as api_keys
from src.api.routes.tenancy import audit as audit
from src.api.routes.tenancy import connections as connections
from src.api.routes.tenancy import defaults as defaults
from src.api.routes.tenancy import invites as invites
from src.api.routes.tenancy import limits as limits
from src.api.routes.tenancy import me as me
from src.api.routes.tenancy import members as members
from src.api.routes.tenancy import music as music
from src.api.routes.tenancy import offboarding as offboarding
from src.api.routes.tenancy import partner as partner
from src.api.routes.tenancy import tenant as tenant

ROUTERS: list[APIRouter] = [
    members.router,
    invites.router,
    invites.accept_router,
    me.router,
    tenant.router,
    api_keys.router,
    partner.router,
    connections.router,
    defaults.router,
    limits.router,
    audit.router,
    music.router,
    offboarding.router,
    offboarding.admin_router,
]
