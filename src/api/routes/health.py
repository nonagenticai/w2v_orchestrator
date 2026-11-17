"""K8s probes — /healthz (liveness) and /readyz (DB + Redis reachability)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy import text

router = APIRouter(tags=["health"])


@router.get("/healthz", operation_id="health.live")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


# Legacy probe-path aliases. The live probes are declared in gitops
# (`clusters/strato/services/w2v/deployments.yaml`) +
# the CI smoke deploy yaml (.github/workflows/ci-cd.yaml) still configure
# startupProbe / readinessProbe / livenessProbe against /health, /health/live,
# /health/ready. Keep them as hidden aliases so the Studio v3 rewrite doesn't
# require a coordinated probe-config rollout. Not exposed via OpenAPI; not
# asserted by the contract gate. Drop after every probe config moves to
# /healthz + /readyz.
@router.get("/health", include_in_schema=False)
async def health_legacy() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/live", include_in_schema=False)
async def health_live_legacy() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz", operation_id="health.ready")
async def readyz(request: Request) -> dict[str, str]:
    eng = getattr(request.app.state, "db_engine", None)
    if eng is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="db engine not initialized")
    try:
        async with eng.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"db unhealthy: {exc}"
        ) from exc
    redis = getattr(request.app.state, "redis", None)
    if redis is not None:
        try:
            await redis.ping()
        except Exception as exc:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"redis unhealthy: {exc}",
            ) from exc
    return {"status": "ready"}


@router.get("/health/ready", include_in_schema=False)
async def health_ready_legacy(request: Request) -> dict[str, str]:
    # Delegate to the canonical /readyz so the legacy path runs the same
    # DB+Redis reachability check as the modern path. Without delegation
    # the probe would return 200 even when dependencies are down.
    return await readyz(request)
