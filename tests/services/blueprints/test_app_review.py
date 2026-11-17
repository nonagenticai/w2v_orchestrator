from src.services.blueprints.app_review import APP_REVIEW_BLUEPRINT
from src.services.blueprints.registry import get_registry


def test_app_review_registered() -> None:
    assert get_registry().resolve("app_review") is APP_REVIEW_BLUEPRINT


def test_app_review_topology() -> None:
    assert APP_REVIEW_BLUEPRINT.topological_order() == [
        "app_inspect",
        "app_critique",
        "app_review_script",
        "app_review_record",
        "app_review_compile",
        "growth_bundle",
    ]
