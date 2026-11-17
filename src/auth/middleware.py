"""License token authentication middleware — delegates to w2v_common.

This module re-exports the shared auth middleware so that existing imports
throughout the service (``from src.auth.middleware import ...``) continue to
work without modification.
"""

from w2v_common.auth.middleware import (
    PUBLIC_ENDPOINTS,
    PUBLIC_PREFIXES,
    TRUSTED_NETWORKS,
    AuthMiddleware,
    _get_license_cache,
    _introspect_license,
    _is_trusted_network,
)

__all__ = [
    "PUBLIC_ENDPOINTS",
    "PUBLIC_PREFIXES",
    "TRUSTED_NETWORKS",
    "AuthMiddleware",
    "_get_license_cache",
    "_introspect_license",
    "_is_trusted_network",
]
