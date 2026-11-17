"""``RunDetail.delivered_artifact_url`` must resolve the SAME file the pipeline does.

The read path and ``final_critique._starting_artifact`` are two independent
implementations of one question: of the four stages on a ``demo_video_v1`` run
that publish an ``artifact_url``, which one holds the file that shipped? If they
disagree, every comparison between what the API reports and what the pipeline
graded is noise — and an assertion bound to the API's answer can pass while the
delivered master is broken.

These pin the resolver against the pipeline's own chain, including the
skipped-clean fallbacks (a stage that did not run publishes a falsy ``{}``, and
the artifact upstream of it is the delivery) and the absent-URL sentinels.

⚠️ Nothing here is a gate. No assertion anywhere claims a resolved URL is good,
only that it is the right one to look at.

No ``integration`` marker: pytest addopts deselects that marker, so a test
wearing it would never execute in CI.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid7

from src.api.routes.runs._shared import (
    _ABSENT_ARTIFACT_URL_SENTINELS,
)
from src.api.schemas import StageOutputResponse
from src.api.schemas.runs import StageStatus
from src.services.blueprints.registry import get_registry
from src.services.stages.demo_video.final_critique import _ABSENT_URL_SENTINELS

_RUN_ID = uuid7()


def _sr(
    stage_name: str, output: object, *, status: StageStatus = "completed"
) -> StageOutputResponse:
    return StageOutputResponse(
        id=uuid7(),
        run_id=_RUN_ID,
        stage_name=stage_name,
        status=status,
        output=cast(Any, output),
        error=None,
        started_at=None,
        completed_at=None,
    )


def _demo_video():
    return get_registry().resolve_by_id("demo_video_v1")


def test_the_api_sentinels_are_byte_identical_to_the_pipelines() -> None:
    """The two copies of "present but meaning absent" must not drift.

    The API layer may not import a stage module, so the sentinel set is
    RESTATED in ``_shared``. A drift of one entry means a run whose
    ``artifact_url`` is the string ``"none"`` is reported as delivered by one
    side and undelivered by the other.
    """
    assert _ABSENT_ARTIFACT_URL_SENTINELS == _ABSENT_URL_SENTINELS
