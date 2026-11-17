"""A slide render outage must keep its FAULT, not just its prose.

``render_deck_slide_stage``'s soft-fail boundary is the last place the original
exception exists. Before this, it kept ``str(exc)`` in a note and dropped the
object, so the ``failure_class`` that ``upstream.guarded`` had already stamped
where the HTTP status was in hand — the only place it is knowable — died there
and ``attribute_exception`` at the persistence site had nothing left to read. A
slide_service 503 is ``external`` and was known to be; recording it ``unknown``
threw a fact away rather than never having had it.

These tests drive the REAL attribution path (``upstream.guarded`` over a real
``httpx`` 503) rather than hand-setting the attribute, because the defect was
never in ``attribute_exception`` — it was in whether anything still holds an
exception for it to read.

The negative cases are the load-bearing half. ``models/failure_class.py``'s
cardinal rule is that a wrong attribution is worse than an absent one, so a
``ValueError`` must stay ``unknown`` and a healthy render must attribute nothing
at all: a boundary that answered ``external`` unconditionally would pass the
first test here and be worse than the bug.
"""

from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest

from src.clients.slide_service_client import SlideServiceClient
from src.models.failure_class import EXTERNAL, OUR_DEFECT, UNKNOWN, attribute_exception
from src.services import tts_storage
from src.services.runs.dag_executor import DAGExecutor
from src.services.runs.run_context import RunContext
from src.services.stages.shared.slide_stage import DeckEntry, render_deck_slide_stage
from src.services.stages.upstream import guarded
from src.services.stages.website_review.slides import (
    WebsiteReviewSlide,
    WebsiteReviewSlidesInput,
    WebsiteReviewSlidesOutput,
    WebsiteReviewSlidesStage,
)

pytestmark = pytest.mark.asyncio


class _FakeStorage:
    async def initialize(self) -> None:
        pass

    async def store_stage_artifact(
        self, job_id: str, stage_name: str, data: bytes, filename: str
    ) -> str:
        return f"artifacts/{job_id}/{stage_name}/{filename}"

    async def get_presigned_url(
        self, path: str, expires: int | None = None, bucket: str | None = None
    ) -> str | None:
        return f"https://minio.test/{path}"


@pytest.fixture
def fake_storage():
    tts_storage.set_storage_for_tests(_FakeStorage())
    yield
    tts_storage.set_storage_for_tests(None)


def _deck_entry(line_id: str) -> DeckEntry:
    return (line_id, {"template_id": "tpl-title", "role": "content", "content": {"title": "T"}})


def _working_slide_client(png_count: int) -> AsyncMock:
    fake = AsyncMock()
    fake.resolve_template_id.side_effect = lambda name: f"tpl-{name}"
    fake.start_deck.return_value = "deck-1"
    fake.wait_for_deck.return_value = {"status": "completed", "slide_count": png_count}
    fake.get_slide_png.side_effect = [b"\x89PNG" + bytes([i]) for i in range(png_count)]
    return fake


async def _render(build_entries, client: AsyncMock) -> Any:
    return await render_deck_slide_stage(
        client=client,
        video_client=AsyncMock(),
        build_entries=build_entries,
        anchor="Anchor",
        deck_brand={"name": "B"},
        config_snapshot={},  # no motion_graphics block — stills only
        deliverable={"title": "D"},
        run_id=str(uuid4()),
        stage_name="promo_slides",
        media_id_prefix="promo_slide",
        make_slide=lambda slide_id, line_id, image_url: WebsiteReviewSlide(
            id=slide_id, line_id=line_id, image_url=image_url
        ),
        phase_budget_s=10.0,
    )


def _raiser(exc: BaseException):
    async def _boom() -> list[DeckEntry]:
        raise exc

    return _boom


def _upstream_503():
    """A ``build_entries`` that fails exactly as a slide_service outage does.

    Through ``guarded``, so the resulting error carries the ``failure_class``
    that ``fault_for_status`` stamped from the real status — not one this test
    asserted into existence.
    """
    request = httpx.Request("POST", "https://slide-service.test/api/decks")
    response = httpx.Response(503, request=request, text="deck backend unavailable")
    status_error = httpx.HTTPStatusError("503", request=request, response=response)

    async def _build() -> list[DeckEntry]:
        return await guarded(_raiser(status_error))

    return _build


async def test_a_slide_service_outage_is_recorded_external_not_unknown() -> None:
    """The defect itself: an attributable outage must survive the soft-fail boundary."""
    result = await _render(_upstream_503(), _working_slide_client(0))

    assert result.render_failed is True, "precondition: the render must have failed"
    assert result.failure_class == EXTERNAL, (
        "the exception's stamped attribution died at the soft-fail boundary; the "
        "run records `unknown` for a fault that was knowable at the call site"
    )


class _Slide503Client:
    """A ``SlideServiceClient`` stand-in whose first call is a real guarded 503.

    Not an ``AsyncMock`` with a hand-set attribute: the point is that the class
    under test never sees an attribution this test invented — ``guarded`` stamps
    it from the real ``httpx`` status, exactly as the live client does.
    """

    async def resolve_template_id(self, name: str) -> str:
        request = httpx.Request("POST", "https://slide-service.test/api/decks")
        response = httpx.Response(503, request=request, text="deck backend unavailable")

        async def _boom() -> str:
            raise httpx.HTTPStatusError("503", request=request, response=response)

        return await guarded(_boom)


async def test_the_stage_output_carries_the_fault_to_the_executor() -> None:
    """…and the consumer must COPY it onto its output, or the fault dies one frame later.

    The helper stamping ``failure_class`` is necessary and not sufficient: the
    executor's gate reads the DUMPED stage output, by which point the exception
    object is gone. A consumer that copies only ``slides`` / ``render_failed`` /
    ``notes`` throws the attribution away between the two, which is precisely
    how eight real production failures were persisted ``unknown``.
    """
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {
                "shotlist": [
                    {
                        "kind": "slide",
                        "narration": "A word about the homepage.",
                        "slide": {"template": "title", "title": "Homepage"},
                    }
                ]
            },
            "deliverable": {"title": "D"},
        },
    )
    stage = WebsiteReviewSlidesStage(
        # Structural stand-in: the stage only calls `resolve_template_id`, and a
        # real client here would need a live slide_service.
        client=cast("SlideServiceClient", _Slide503Client()),
        video_client=AsyncMock(),
    )

    output = await stage.handle(WebsiteReviewSlidesInput(), ctx)

    assert output.slides_render_failed is True, "precondition: the render must have failed"
    assert output.failure_class == EXTERNAL, (
        "the consumer dropped the attribution the helper had already computed; "
        "the executor reads the dumped output and can only record `unknown`"
    )


async def test_the_executor_gate_stamps_the_fault_onto_the_raised_error() -> None:
    """The last mile: the dumped class must reach the exception the dispatcher reads.

    ``StageReportedFailureError`` propagates bare out of ``_run_one``, and
    ``attribute_exception`` picks up an attribute named exactly ``failure_class``
    anywhere on the cause chain — so stamping it on the error IS the delivery.
    Asserting both the attribute and ``attribute_exception`` because only the
    second is what the persistence site actually calls.
    """
    dumped: dict[str, Any] = {
        "slides": [],
        "notes": ["slide deck render failed: 503"],
        "slides_render_failed": True,
        "failure_class": EXTERNAL,
    }

    with pytest.raises(Exception) as excinfo:
        DAGExecutor._enforce_failure_flags(
            "website_review_slides", WebsiteReviewSlidesOutput, dumped
        )

    exc = excinfo.value
    assert getattr(exc, "failure_class", None) == EXTERNAL
    assert attribute_exception(exc) == EXTERNAL, (
        "the persistence site reads the attribution through `attribute_exception`; "
        "an attribute under any other name is invisible to it"
    )


async def test_the_executor_gate_refuses_a_value_outside_the_vocabulary() -> None:
    """A persisted output is arbitrary data — it must never widen the enum.

    ``runs.failure_class`` has a DB CHECK in exact parity with the Literal, so a
    class smuggled through a dumped output would 500 every read of the run that
    it did not simply fail to write. The narrowing is by membership, and anything
    unrecognised is ``unknown`` — the honest answer, not a guess.
    """
    dumped: dict[str, Any] = {
        "slides": [],
        "notes": ["slide deck render failed"],
        "slides_render_failed": True,
        "failure_class": "definitely_not_a_class",
    }

    with pytest.raises(Exception) as excinfo:
        DAGExecutor._enforce_failure_flags(
            "website_review_slides", WebsiteReviewSlidesOutput, dumped
        )

    exc = excinfo.value
    assert getattr(exc, "failure_class", None) == UNKNOWN
    assert attribute_exception(exc) == UNKNOWN


async def test_a_programming_error_is_recorded_our_defect() -> None:
    """Control on the value: the boundary must attribute, not emit one constant.

    A helper hardwired to ``external`` would pass the test above. ``TypeError`` is
    in ``_OUR_DEFECT_TYPES`` and carries no stamp, so only a real call into
    ``attribute_exception`` produces this answer.
    """
    result = await _render(
        _raiser(TypeError("None is not subscriptable")), _working_slide_client(0)
    )

    assert result.render_failed is True
    assert result.failure_class == OUR_DEFECT


async def test_an_unattributable_failure_stays_unknown() -> None:
    """A wrong attribution is worse than an absent one — ``ValueError`` is a guess.

    ``models/failure_class.py`` excludes ``ValueError`` deliberately (it is the
    re-prompt signal and pydantic's validation base). The boundary must not
    invent a class for it just because a render failed.
    """
    result = await _render(_raiser(ValueError("deck payload rejected")), _working_slide_client(0))

    assert result.render_failed is True
    assert result.failure_class == UNKNOWN


async def test_a_healthy_render_attributes_nothing(fake_storage) -> None:
    """The no-exception path is unchanged: no failure, and no fault asserted."""
    result = await _render(
        AsyncMock(return_value=[_deck_entry("line-1")]), _working_slide_client(1)
    )

    assert [s.line_id for s in result.slides] == ["line-1"]
    assert result.render_failed is False
    assert result.notes == []
    assert result.failure_class == UNKNOWN, (
        "a deck that rendered must assert no fault; anything else files healthy "
        "runs into a fault bucket"
    )
