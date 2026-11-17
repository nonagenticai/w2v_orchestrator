"""An empty slide deck must never be indistinguishable from a render outage.

This is the ``site_capture`` defect class, copy-propagated. ``site_capture``
returned ``completed`` holding zero screens, the run parked in ``paused``, and
nothing disagreed. The same shape lives in ``shared/slide_stage.py``, whose
``except Exception`` swallows the PNG generation itself and returns
``slides == []`` plus a note — and its own comment says it "mirrors site_capture
/ ai_image_gen soft-fail semantics", so the copy was deliberate.

A note is NOT a fix and these tests deliberately never assert on one: appending
to ``notes`` sets the run's ``degraded`` flag, but the terminal status stays
``completed`` and ``scripts/e2e_validate.py`` never reads that flag. Nothing that
looks at a finished run can see it.

The four sites do NOT get the same fix, because empty is not the same thing at
each of them:

* ``showcase_slides`` / ``website_review_slides`` — empty is a LEGITIMATE
  operating mode (an all-footage demo_video brief authors no cards; a url-only
  or legacy website_review shotlist authors no slide shots). A ``produces``
  contract would mark those correct runs ``degraded`` and ``produces_required``
  would fail them outright. They get the ``action_reel/graphics_design.py``
  treatment instead: ``slides_stubbed`` vs ``slides_render_failed``, so the two
  readings of ``slides == []`` stay tellable apart structurally.
* ``promo_slides`` — no legitimate empty (one slide per script line,
  unconditionally, and the deck is the universal visual fallback), so it gets
  ``produces_required``.
* ``<type>_record`` (audit) — no legitimate empty either; the "no video wanted"
  case is a ``skipped``, not an empty output. It had neither a gate nor a
  ``notes`` channel, so it gets ``produces`` + ``produces_required``.

The gate assertions go through ``DAGExecutor._enforce_produces`` — the real
chokepoint every stage's success passes through — rather than asserting the
``produces_required`` ClassVar. Asserting the ClassVar would pass even if the
executor stopped consulting it.
"""

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.services import tts_storage
from src.services.blueprints.base import DAGBlueprint, StageSpec
from src.services.runs.dag_executor import DAGExecutor
from src.services.runs.run_context import RunContext
from src.services.stages.audit_video.models import AuditRecordOutput
from src.services.stages.audit_video.record import AuditRecordStageBase
from src.services.stages.base import (
    StageProducedNothingError,
    StageReportedFailureError,
    missing_produced_fields,
    raised_failure_flags,
)
from src.services.stages.demo_video.slides import (
    ShowcaseSlide,
    ShowcaseSlidesOutput,
    ShowcaseSlidesStage,
)
from src.services.stages.errors import StagePreconditionError
from src.services.stages.promo.slides import PromoSlide, PromoSlidesOutput, PromoSlidesStage
from src.services.stages.shared.slide_stage import DeckEntry, render_deck_slide_stage
from src.services.stages.website_review.slides import (
    WebsiteReviewSlide,
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


def _ctx(snapshot: dict[str, Any], prior: dict[str, Any] | None = None) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=snapshot,
        prior_run_outputs=prior or {},
    )


def _working_slide_client(png_count: int) -> AsyncMock:
    fake = AsyncMock()
    fake.resolve_template_id.side_effect = lambda name: f"tpl-{name}"
    fake.start_deck.return_value = "deck-1"
    fake.wait_for_deck.return_value = {"status": "completed", "slide_count": png_count}
    fake.get_slide_png.side_effect = [b"\x89PNG" + bytes([i]) for i in range(png_count)]
    return fake


def _broken_slide_client() -> AsyncMock:
    """slide_service is up enough to resolve templates and then dies on the deck.

    This is the shape the soft-fail boundary was written for, and the shape that
    made a total render outage invisible: the exception is raised INSIDE
    ``render_deck_slide_stage``'s ``try``, so it degrades to ``slides == []``.
    """
    fake = AsyncMock()
    fake.resolve_template_id.side_effect = lambda name: f"tpl-{name}"
    fake.start_deck.side_effect = RuntimeError("slide_service 503")
    return fake


# --------------------------------------------------------------------------
# showcase_slides — empty IS a legitimate mode, so the fix is discrimination
# --------------------------------------------------------------------------

_CARD_LINES = [
    {"id": "intro", "text": "This is SpecraLabs.", "card": {"template": "title", "title": "Hi"}},
    {"id": "clip-1", "text": "A clean dashboard."},  # footage line — no card
]
_FOOTAGE_ONLY_LINES = [{"id": "clip-1", "text": "A clean dashboard."}]


async def test_showcase_all_footage_run_is_a_stub_not_a_failure(fake_storage) -> None:
    """The mode the peer agent nearly reddened: an all-footage brief has no cards.

    It must stay empty, stay non-failing, and say WHY it is empty.
    """
    ctx = _ctx({"brief": {"lines": _FOOTAGE_ONLY_LINES}, "deliverable": {"title": "S"}})
    out = await ShowcaseSlidesStage(client=_working_slide_client(0)).run({}, ctx=ctx)

    assert out.slides == []
    assert out.slides_stubbed is True, (
        "an all-footage run authored no cards — that must be reported as a stub, "
        "or it is indistinguishable from a slide_service outage"
    )
    assert out.slides_render_failed is False, (
        "nothing was rendered, so nothing can have failed to render"
    )


async def test_showcase_render_outage_is_flagged_not_just_noted(fake_storage) -> None:
    ctx = _ctx({"brief": {"lines": _CARD_LINES}, "deliverable": {"title": "S"}})
    out = await ShowcaseSlidesStage(client=_broken_slide_client()).run({}, ctx=ctx)

    assert out.slides == []
    assert out.slides_render_failed is True, (
        "authored cards existed and their deck render died — the branded beats are "
        "silently gone from the timeline while demo_video_compile still emits a "
        "valid artifact_url, so this flag is the only machine-readable evidence"
    )
    assert out.slides_stubbed is False, (
        "cards WERE authored; calling this a stub would send an operator hunting "
        "for a missing card block instead of a slide_service outage"
    )


async def test_showcase_two_empty_decks_are_distinguishable(fake_storage) -> None:
    """The whole point, asserted as one proposition.

    Both runs produce ``slides == []``. If the flags did not exist, or were set
    from ``not slides``, these two dicts would be equal and a consumer could not
    tell a correct all-footage run from a total render outage.
    """
    stub = await ShowcaseSlidesStage(client=_working_slide_client(0)).run(
        {}, ctx=_ctx({"brief": {"lines": _FOOTAGE_ONLY_LINES}, "deliverable": {"title": "S"}})
    )
    outage = await ShowcaseSlidesStage(client=_broken_slide_client()).run(
        {}, ctx=_ctx({"brief": {"lines": _CARD_LINES}, "deliverable": {"title": "S"}})
    )

    assert stub.slides == outage.slides == [], "precondition: both decks are empty"
    assert (stub.slides_stubbed, stub.slides_render_failed) != (
        outage.slides_stubbed,
        outage.slides_render_failed,
    ), "two empty decks with the same flags means the flags carry no information"


# --------------------------------------------------------------------------
# website_review_slides — same judgement, same fix
# --------------------------------------------------------------------------

_SLIDE_SHOT_BRIEF = {
    "shotlist": [
        {"kind": "slide", "narration": "Here is the plan.", "slide": {"template": "title"}},
    ]
}
_URL_ONLY_BRIEF = {"shotlist": [{"kind": "url", "url": "https://example.com", "narration": "n"}]}


async def test_website_review_url_only_shotlist_is_empty_not_failed(fake_storage) -> None:
    ctx = _ctx({"brief": _URL_ONLY_BRIEF, "deliverable": {"title": "R"}})
    out = await WebsiteReviewSlidesStage(client=_working_slide_client(0)).run({}, ctx=ctx)

    assert out.slides == []
    # Not `slides_stubbed`: that flag means authored lines carried no slides. The empty
    # deck is recorded by the `slides_not_authored` advisory instead.
    assert out.slides_stubbed is False
    assert out.slides_render_failed is False
    assert [a["code"] for a in out.advisories] == ["slides_not_authored"]


async def test_website_review_render_outage_is_flagged(fake_storage) -> None:
    ctx = _ctx({"brief": _SLIDE_SHOT_BRIEF, "deliverable": {"title": "R"}})
    out = await WebsiteReviewSlidesStage(client=_broken_slide_client()).run({}, ctx=ctx)

    assert out.slides == []
    assert out.slides_render_failed is True, (
        "an authored slide shot existed and its render died; those lines now render "
        "as bare text cards while the compile still succeeds"
    )
    assert out.slides_stubbed is False


# --------------------------------------------------------------------------
# promo_slides — no legitimate empty, so the gate must FAIL, not annotate
# --------------------------------------------------------------------------


async def test_promo_render_outage_fails_the_stage_at_the_gate() -> None:
    """``produces`` alone was not enough: degrade-only is invisible end to end.

    Driven through ``_enforce_produces`` — the executor's single chokepoint — so
    this fails if the contract stops being ``produces_required`` OR if the gate
    stops honouring it.
    """
    empty = PromoSlidesOutput(slides=[]).model_dump()

    with pytest.raises(StageProducedNothingError) as excinfo:
        DAGExecutor._enforce_produces("promo_slides", PromoSlidesOutput, empty)

    assert "slides" in str(excinfo.value)


async def test_promo_deck_that_rendered_passes_the_gate(fake_storage) -> None:
    """Control. Without this the test above passes on a gate that fails everything."""
    ctx = _ctx(
        {"brief": {}, "deliverable": {"title": "P"}},
        prior={"promo_script": {"lines": [{"id": "l1", "text": "One line."}]}},
    )
    out = await PromoSlidesStage(client=_working_slide_client(1)).run({}, ctx=ctx)

    assert len(out.slides) == 1
    DAGExecutor._enforce_produces("promo_slides", PromoSlidesOutput, out.model_dump())


async def test_promo_empty_script_fails_naming_promo_script() -> None:
    """The one benign-looking empty is attributed to the stage that caused it.

    ``promo_script.produces`` is degrade-only, so an empty ``lines`` reaches here.
    Letting the gate above fail instead would name ``promo_slides`` and send the
    operator looking for a slide_service outage that never happened.
    """
    ctx = _ctx({"brief": {}, "deliverable": {"title": "P"}}, prior={"promo_script": {"lines": []}})

    with pytest.raises(StagePreconditionError) as excinfo:
        await PromoSlidesStage(client=_working_slide_client(0)).run({}, ctx=ctx)

    assert "promo_script" in str(excinfo.value)
    assert excinfo.value.missing == ["promo_script.lines"]


# --------------------------------------------------------------------------
# <type>_record (audit) — the site with neither a gate nor a degrade channel
# --------------------------------------------------------------------------


class _SeoAuditRecordStage(AuditRecordStageBase):
    name = "seo_audit_record"
    deck_title = "SEO Audit"


async def test_audit_record_empty_deck_fails_naming_the_slides_stage() -> None:
    """Was a silent ``return AuditRecordOutput(recordings=[])``.

    ``<type>_compile`` then muxed the narration over zero images and the run read
    ``completed``. ``AuditSlidesOutput.produces`` is degrade-only, so an empty
    deck really does get this far.
    """
    ctx = _ctx(
        {"brief": {"generate_video": True}, "deliverable": {"title": "A"}},
        prior={"seo_audit_slides": {"slides": []}},
    )

    with pytest.raises(StagePreconditionError) as excinfo:
        await _SeoAuditRecordStage(client=AsyncMock()).run({}, ctx=ctx)

    assert "seo_audit_slides" in str(excinfo.value)
    assert excinfo.value.missing == ["seo_audit_slides.slides"]


async def test_audit_record_gate_is_the_backstop_for_every_other_empty() -> None:
    """The precondition above covers ONE way to produce nothing.

    ``produces_required`` on the output is what covers the rest, and it is the
    only channel available: ``AuditRecordOutput`` has no ``notes`` field, so a
    degrade-only contract here would be a gate that cannot fire.
    """
    assert not hasattr(AuditRecordOutput, "notes"), (
        "if a notes channel is added, re-read whether produces_required is still "
        "the right half of the contract"
    )
    empty = AuditRecordOutput(recordings=[]).model_dump()

    with pytest.raises(StageProducedNothingError) as excinfo:
        DAGExecutor._enforce_produces("seo_audit_record", AuditRecordOutput, empty)

    assert "recordings" in str(excinfo.value)


async def test_audit_record_with_recordings_passes_the_gate() -> None:
    """Control for the gate test above.

    Without it, the test above would still pass on a gate that rejected
    everything — including a stage that did its job.
    """
    populated = AuditRecordOutput(
        recordings=[{"id": "r1", "slide_id": "s1", "image_url": "https://m/1.png"}]  # type: ignore[list-item]
    ).model_dump()

    assert missing_produced_fields(populated, AuditRecordOutput.produces) == (), (
        "a populated recordings list must satisfy the declared contract"
    )
    DAGExecutor._enforce_produces("seo_audit_record", AuditRecordOutput, populated)
    assert populated["recordings"], "the gate must not mutate a passing output"


# --------------------------------------------------------------------------
# The stub flags must not become a second silent channel
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [ShowcaseSlidesOutput],
    ids=["showcase_slides"],
)
async def test_slide_stubs_default_to_neither_reading(model: type) -> None:
    """A default-constructed output claims neither a stub nor an outage.

    Both flags default False so that a stage which forgets to set one produces an
    unclassified empty deck rather than silently asserting "this is fine".
    """
    out = model()
    assert out.slides == []
    assert out.slides_stubbed is False
    assert out.slides_render_failed is False


# --------------------------------------------------------------------------
# The helper's OWN contract: an empty projection is a failure, not a deck
# --------------------------------------------------------------------------
#
# Every caller above reaches `render_deck_slide_stage` only with a non-empty
# content source and maps over it totally, so no stage-level test can reach the
# empty-entries fold at all — a mutant that deletes it and returns a
# silently-successful empty deck survives all of the tests above. The fold is
# nonetheless the helper's published contract (its docstring tells callers they
# MUST NOT re-derive `render_failed` from `not slides`), and it is what protects
# the next caller whose `build_entries` filters rather than maps. It is tested
# here directly, against the helper, because that is the only level at which it
# is reachable.


def _deck_entry(line_id: str) -> DeckEntry:
    return (line_id, {"template_id": "tpl-title", "role": "content", "content": {"title": "T"}})


async def _render(entries: list[DeckEntry], client: AsyncMock) -> Any:
    return await render_deck_slide_stage(
        client=client,
        video_client=AsyncMock(),
        build_entries=AsyncMock(return_value=entries),
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


async def test_empty_projection_is_a_render_failure_not_an_empty_deck(fake_storage) -> None:
    """A projection that yields zero entries must not read as a rendered deck.

    This is the whole defect in one assertion: `slides == []` with
    `render_failed=False` is the value a completed run holds when the deck it
    shipped does not exist. The helper never submitted a deck — there was nothing
    to submit — so "no slides were produced" here can only mean failure.
    """
    client = _working_slide_client(0)

    result = await _render([], client)

    assert result.slides == []
    assert result.render_failed is True, (
        "an empty projection returned a deck the caller will read as successfully "
        "rendered; showcase/website_review carry this flag straight onto their "
        "output, so the run completes holding a video with zero slides"
    )
    client.start_deck.assert_not_awaited()


async def test_a_real_projection_still_renders_and_clears_the_flag(fake_storage) -> None:
    """Control: `render_failed` must be answering the question, not hardwired.

    Without this the test above would pass on a helper that flagged EVERY render
    as failed — which would redden every healthy slide run instead.
    """
    client = _working_slide_client(1)

    result = await _render([_deck_entry("line-1")], client)

    assert [s.line_id for s in result.slides] == ["line-1"]
    assert result.slides[0].image_url, "a rendered slide must carry a resolvable image url"
    assert result.render_failed is False, "a deck that rendered must not be flagged as failed"
    assert result.notes == []


# --------------------------------------------------------------------------
# ...and the flag must FAIL the stage, not merely be recorded on it
# --------------------------------------------------------------------------
#
# Everything above proves `slides_render_failed` carries the right VALUE. None
# of it proved anyone ACTS on that value, and until `fails_when_flagged` nobody
# did: `git grep slides_render_failed -- src/` found the definitions and the
# assignments and no reader at all. `_enforce_produces` returns early when
# `produces` is empty, so both stages recorded `completed`; `demo_video_compile`
# and `website_review_compile` then built their slide maps by comprehension over
# `[]` with no check and still emitted a valid `artifact_url`. The run shipped —
# and was billed — as a success with every branded card missing.
#
# These tests drive the REAL stages through the REAL executor rather than
# calling the gate helper, because the failure mode is precisely a gate that
# exists and is never consulted. The two halves are asserted as a pair on
# purpose: the naive fix (`produces=("slides",)`) passes the outage half and
# fails the stub half.


def _slide_stage_blueprint(name: str, handler: Any) -> DAGBlueprint:
    return DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[StageSpec(name=name, handler_factory=lambda deps: handler)],
    )


async def _drive(
    name: str,
    handler: Any,
    snapshot: dict[str, Any],
    prior: dict[str, Any] | None = None,
) -> tuple[list[tuple[str, str, str | None]], BaseException | None]:
    """Run one real slide stage through the executor; report rows + what escaped."""
    rows: list[tuple[str, str, str | None]] = []

    async def emit(event: str, **kw: object) -> None:
        return None

    async def persist(
        stage_name: str,
        status: str,
        output: dict[str, object] | None = None,
        error: str | None = None,
    ) -> None:
        rows.append((stage_name, status, error))

    executor = DAGExecutor(_slide_stage_blueprint(name, handler), emit=emit, persist=persist)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=snapshot,
        prior_run_outputs=prior or {},
    )
    raised: BaseException | None = None
    try:
        await executor.run(ctx, pause_after_stages=set())
    except BaseException as exc:  # noqa: BLE001 — the escape IS the assertion
        raised = exc
    return rows, raised


async def test_showcase_render_outage_fails_the_stage_at_the_executor(fake_storage) -> None:
    """The defect, asserted at the only place a run's success is recorded."""
    rows, raised = await _drive(
        "showcase_slides",
        ShowcaseSlidesStage(client=_broken_slide_client()),
        {"brief": {"lines": _CARD_LINES}, "deliverable": {"title": "S"}},
    )

    assert isinstance(raised, StageReportedFailureError), (
        "a total slide_service outage must FAIL the stage — otherwise the run "
        f"completes with every branded card missing (got {raised!r})"
    )
    assert not [r for r in rows if r[1] == "completed"], (
        "the stage must never be recorded as a success; demo_video_compile "
        "depends on it, so a `completed` here is what lets a card-less video ship"
    )
    failed = [r for r in rows if r[1] == "failed"]
    assert failed, "the guilty stage row must be marked failed, not left running"
    message = str(raised)
    assert "showcase_slides" in message, "the operator must be told WHICH stage"
    assert "slides_render_failed" in message, "…and which declared flag fired"
    assert "slide_service 503" in message, (
        "…and the swallowed downstream cause, which is otherwise only in pod logs"
    )


async def test_showcase_all_footage_stub_still_completes_at_the_executor(fake_storage) -> None:
    """The regression guard against the naive fix.

    An all-footage brief authors no cards and produces `slides == []` BY DESIGN.
    Declaring `produces=("slides",)` on this stage — the obvious reading of the
    bug above — reddens exactly this run. The gate must key on the FAILURE flag,
    so this must stay a clean `completed` with no exception at all.
    """
    rows, raised = await _drive(
        "showcase_slides",
        ShowcaseSlidesStage(client=_working_slide_client(0)),
        {"brief": {"lines": _FOOTAGE_ONLY_LINES}, "deliverable": {"title": "S"}},
    )

    assert raised is None, f"a legitimate all-footage run must not fail ({raised!r})"
    completed = [r for r in rows if r[1] == "completed"]
    assert completed, "an all-footage run authored no cards — that is success, not a defect"


async def test_website_review_render_outage_fails_the_stage_at_the_executor(fake_storage) -> None:
    rows, raised = await _drive(
        "website_review_slides",
        WebsiteReviewSlidesStage(client=_broken_slide_client()),
        {"brief": _SLIDE_SHOT_BRIEF, "deliverable": {"title": "R"}, "video": {"enabled": True}},
    )

    assert isinstance(raised, StageReportedFailureError), (
        f"authored slide shots existed and their render died (got {raised!r})"
    )
    assert not [r for r in rows if r[1] == "completed"]
    assert "website_review_slides" in str(raised)
    assert "slide_service 503" in str(raised)


async def test_website_review_url_only_stub_still_completes_at_the_executor(fake_storage) -> None:
    """Second half of the pair: a url-only shotlist is legitimately slide-less."""
    rows, raised = await _drive(
        "website_review_slides",
        WebsiteReviewSlidesStage(client=_working_slide_client(0)),
        {"brief": _URL_ONLY_BRIEF, "deliverable": {"title": "R"}, "video": {"enabled": True}},
    )

    assert raised is None, f"a url-only shotlist must keep completing ({raised!r})"
    assert [r for r in rows if r[1] == "completed"]


async def test_a_healthy_showcase_deck_still_completes_at_the_executor(fake_storage) -> None:
    """Control: the gate must be answering the question, not failing everything.

    Without this, both outage tests above pass on a gate hardwired to raise.
    """
    rows, raised = await _drive(
        "showcase_slides",
        ShowcaseSlidesStage(client=_working_slide_client(1)),
        {"brief": {"lines": _CARD_LINES}, "deliverable": {"title": "S"}},
    )

    assert raised is None, f"a deck that rendered must not fail ({raised!r})"
    assert [r for r in rows if r[1] == "completed"]


async def test_promo_render_outage_trips_the_flag_gate_as_well(fake_storage) -> None:
    """`promo_slides` is belt-and-braces: both gates must fire on the same outage.

    `produces_required` already catches today's total outage. The flag gate is
    what survives a partial deck (some cards rendered, then the render failed),
    which satisfies `produces` and would otherwise ship silently. Asserted at the
    helper level because only the flag distinguishes the two.
    """
    flagged = PromoSlidesOutput(
        slides=[PromoSlide(id="s1", line_id="l1", image_url="https://m/1.png")],
        slides_render_failed=True,
        notes=["slide render failed: slide_service 503"],
    ).model_dump()

    # `produces` is satisfied — a slide IS present — so this gate stays silent…
    DAGExecutor._enforce_produces("promo_slides", PromoSlidesOutput, flagged)
    # …and only the failure flag catches the partial deck.
    with pytest.raises(StageReportedFailureError) as excinfo:
        DAGExecutor._enforce_failure_flags("promo_slides", PromoSlidesOutput, flagged)

    assert "promo_slides" in str(excinfo.value)
    assert "slide_service 503" in str(excinfo.value)


async def test_a_declared_flag_left_false_is_silent() -> None:
    """Control for the gate itself: `False` must never fire it.

    A gate that fired on a present-but-False flag would fail every healthy run of
    all three stages — the exact opposite defect.
    """
    healthy = ShowcaseSlidesOutput(
        slides=[ShowcaseSlide(id="s1", line_id="l1", image_url="https://m/1.png")]
    ).model_dump()
    assert healthy["slides_render_failed"] is False, "precondition: nothing failed"

    assert raised_failure_flags(healthy, ShowcaseSlidesOutput.fails_when_flagged) == (), (
        "a declared-but-False flag must not read as raised, or every healthy run "
        "of all three slide stages fails — the exact opposite defect"
    )
    DAGExecutor._enforce_failure_flags("showcase_slides", ShowcaseSlidesOutput, healthy)
    assert healthy["slides"], "the gate must not mutate a passing output"
