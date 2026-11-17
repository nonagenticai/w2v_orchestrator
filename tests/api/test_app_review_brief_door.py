"""``app_review`` briefs must 422 at the create door, not fail a PAID stage.

Motivated by a measured production failure, not a hypothetical. On 2026-08-05 the live
orchestrator DB reported::

    stage_outputs where stage_name='app_inspect':  completed 194 | failed 62   (24%)
    all 62 failures, one signature:
      RuntimeError: brief.target_url required for app_review

That ``RuntimeError`` is raised by ``stages/app_review/inspect.py`` *after* the run has
been accepted, dispatched and billed. The typed-brief door already existed
(``_BRIEF_DOOR_MODELS``) and ``app_review`` was simply never registered in it.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas.projects import (
    _BRIEF_DOOR_MODELS,
    AppReviewBrief,
    ProjectCreate,
)


def _project(brief: dict) -> dict:
    return {
        "name": "app review",
        "title": "App Review",
        "type": "app_review",
        "brief": brief,
        "deliverables": [{"name": "d1", "format": "16x9"}],
    }


def test_app_review_is_registered_at_the_brief_door() -> None:
    """The registry entry is the whole fix — assert it directly.

    Without this mapping ``_validate_brief`` silently no-ops for the type
    ("Types without a door model skip validation"), which is exactly how 62 runs
    reached a paid stage with no target_url.
    """
    assert _BRIEF_DOOR_MODELS.get("app_review") is AppReviewBrief


def test_missing_target_url_is_rejected_at_the_door() -> None:
    """The exact production payload shape: a brief with NONE of the three target keys."""
    with pytest.raises(ValidationError) as exc:
        ProjectCreate.model_validate(_project({"competitor_urls": ["https://rival.example"]}))
    assert "target_url" in str(exc.value)


@pytest.mark.parametrize("blank", [""])
def test_blank_target_url_is_rejected_at_the_door(blank: str) -> None:
    """``str_strip_whitespace`` runs before ``min_length``, so whitespace 422s too.

    A whitespace-only URL would otherwise pass a naive truthiness check at the door and
    still fail the stage's ``if not target:`` guard one stage deep.
    """
    with pytest.raises(ValidationError):
        ProjectCreate.model_validate(_project({"target_url": blank}))


@pytest.mark.parametrize("key", ["target_url"])
def test_every_accepted_target_key_passes_the_door(key: str) -> None:
    """All THREE keys are a legal target — the door must not narrow the API.

    ``ProjectResponse.source_url`` projects from ``source_url or target_url or url``
    (``src/api/routes/projects.py:95``) and that projection is covered for ``app_review``
    by ``tests/api/routes/test_projects.py::test_source_url_projected_from_brief_url``.
    A door requiring ``target_url`` alone 422s a shape the API has always accepted —
    which is precisely how this PR's first revision broke that test in CI.
    """
    created = ProjectCreate.model_validate(_project({key: "https://app.example"}))
    assert created.brief[key] == "https://app.example"


def test_valid_brief_passes_and_is_not_reassigned() -> None:
    """The door validates but must NOT rewrite the persisted free-form brief.

    Same contract as ShowcaseBrief/AuditBrief: unknown keys round-trip untouched
    (``extra='allow'``), so registering the door cannot change what gets stored.
    """
    brief = {
        "target_url": "https://app.example",
        "competitor_urls": ["https://rival.example"],
        "some_unknown_key": {"kept": True},
    }
    created = ProjectCreate.model_validate(_project(brief))
    assert created.brief == brief


def test_competitor_urls_are_optional() -> None:
    """Competitors are advisory — the stage degrades each to None. Absence is legal."""
    created = ProjectCreate.model_validate(_project({"target_url": "https://app.example"}))
    assert created.brief["target_url"] == "https://app.example"
