# IMPORTANT: do NOT add `from __future__ import annotations` to this file.
#
# A submodule named `annotations.py` (PR-6) collides with the future-import
# feature flag if this package eagerly evaluates `from __future__ import
# annotations`, producing an opaque
# ``ImportError: cannot import name 'annotations' from '__future__'``
# at startup. The fix is to never import the future flag at the package
# level. Submodules may still use it locally without conflict.
#
# A pytest test (``tests/api/test_no_future_annotations_in_routes_init.py``)
# enforces this rule.

"""Studio v3 router package — explicit submodule re-exports for asgi mounting."""

from src.api.routes.admin import router as admin_router
from src.api.routes.annotations import router as annotations_router
from src.api.routes.artifact_proxy import router as artifact_proxy_router
from src.api.routes.audit import router as audit_router
from src.api.routes.auth import license_router as license_router
from src.api.routes.auth import router as auth_router
from src.api.routes.blueprints import router as blueprints_router
from src.api.routes.brand import router as brand_router
from src.api.routes.critique import router as critique_router
from src.api.routes.deliverables import router as deliverables_router
from src.api.routes.fixes import router as fixes_router
from src.api.routes.health import router as health_router
from src.api.routes.internal_provider_slots import router as internal_provider_slots_router
from src.api.routes.media import router as media_router
from src.api.routes.media_understanding import router as media_understanding_router
from src.api.routes.music import router as music_router
from src.api.routes.projects import router as projects_router
from src.api.routes.projects import sse_router as projects_sse_router
from src.api.routes.provider_requirements import router as provider_requirements_router
from src.api.routes.redesign import router as redesign_router
from src.api.routes.review_inbox import router as review_inbox_router
from src.api.routes.runs import router as runs_router
from src.api.routes.runs import sse_router as runs_sse_router
from src.api.routes.runs import usage_router as runs_usage_router
from src.api.routes.service_lab import router as service_lab_router
from src.api.routes.storyboard import router as storyboard_router
from src.api.routes.tts import router as tts_router
from src.api.routes.uploads import router as uploads_router
from src.api.routes.uploads_video import router as uploads_video_router
from src.api.routes.verbatim import router as verbatim_router
from src.api.routes.youtube import admin_router as youtube_admin_router
from src.api.routes.youtube import router as youtube_router

__all__ = [
    "admin_router",
    "annotations_router",
    "artifact_proxy_router",
    "audit_router",
    "auth_router",
    "blueprints_router",
    "brand_router",
    "critique_router",
    "deliverables_router",
    "fixes_router",
    "health_router",
    "internal_provider_slots_router",
    "license_router",
    "media_router",
    "media_understanding_router",
    "music_router",
    "projects_router",
    "projects_sse_router",
    "provider_requirements_router",
    "redesign_router",
    "review_inbox_router",
    "runs_router",
    "runs_sse_router",
    "runs_usage_router",
    "service_lab_router",
    "storyboard_router",
    "tts_router",
    "uploads_router",
    "uploads_video_router",
    "verbatim_router",
    "youtube_admin_router",
    "youtube_router",
]
