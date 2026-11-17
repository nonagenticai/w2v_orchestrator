"""DAG blueprint declarations per project type."""

from src.services.blueprints import (
    action_reel as _action_reel,  # noqa: F401  triggers register
)
from src.services.blueprints import (
    app_review as _app_review,  # noqa: F401  triggers register
)
from src.services.blueprints import audiobook as _audiobook  # noqa: F401  triggers register
from src.services.blueprints import (
    audit_family as _audit_family,  # noqa: F401  registers seo_audit/cro_audit/marketing_scorecard
)
from src.services.blueprints import course as _course  # noqa: F401  triggers register
from src.services.blueprints import (
    demo_video as _demo_video,  # noqa: F401  triggers register
)
from src.services.blueprints import generic as _generic  # noqa: F401  triggers register
from src.services.blueprints import (
    growth_bundle as _growth_bundle,  # noqa: F401  triggers register
)
from src.services.blueprints import lesson as _lesson  # noqa: F401  triggers register
from src.services.blueprints import promo as _promo  # noqa: F401  triggers register
from src.services.blueprints import showcase as _showcase  # noqa: F401  triggers register
from src.services.blueprints import web_design as _web_design  # noqa: F401  triggers register
from src.services.blueprints import (
    web_design_v2 as _web_design_v2,  # noqa: F401  triggers register
)
from src.services.blueprints import (
    website_review as _website_review,  # noqa: F401  triggers register
)
from src.services.blueprints.base import DAGBlueprint, StageSpec  # noqa: F401
from src.services.blueprints.registry import get_registry  # noqa: F401
