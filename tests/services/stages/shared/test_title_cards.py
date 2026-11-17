"""Tests for the opt-in animated intro/outro title-card shots."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.services import tts_storage
from src.services.runs.run_context import RunContext
from src.services.stages.shared.degradation import (
    TITLE_CARD_DROPPED,
    UNCLASSIFIED_NOTE,
    Degradation,
)
from src.services.stages.shared.title_cards import (
    MAX_TITLE_CARD_SHARE,
    MIN_TITLE_CARD_SECONDS,
    TitleCardPlan,
    build_title_card_shots,
    plan_title_cards,
    splice_title_cards,
    title_card_seconds,
)


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


def _video_fake() -> AsyncMock:
    v = AsyncMock()
    v.record_animation.return_value = "anim-1"
    v.wait_for_content_recording.return_value = {"duration_seconds": 3.0}
    v.get_recording_video.return_value = (b"\x00mp4", "video/mp4", None)
    return v


def _ctx(config_snapshot: dict[str, Any]) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=config_snapshot,
    )


# ── build_title_card_shots ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_block_is_noop() -> None:
    video = _video_fake()
    intro, outro = await build_title_card_shots(_ctx({}), ordered_seconds=None, client=video)
    assert intro is None and outro is None
    video.record_animation.assert_not_awaited()


@pytest.mark.asyncio
async def test_disabled_block_is_noop() -> None:
    video = _video_fake()
    intro, outro = await build_title_card_shots(
        _ctx({"title_card": {"enabled": False}}), ordered_seconds=None, client=video
    )
    assert intro is None and outro is None
    video.record_animation.assert_not_awaited()


@pytest.mark.asyncio
async def test_enabled_renders_intro_and_outro(fake_storage) -> None:
    video = _video_fake()
    ctx = _ctx(
        {
            "title": "My Course",
            "brand": {"name": "Acme", "secondary_color": "#14B8A6"},
            "deliverable": {"format": "16x9"},
            "title_card": {"enabled": True},
            "resolution": "1080p",
        }
    )
    intro, outro = await build_title_card_shots(ctx, ordered_seconds=None, client=video)

    assert intro and outro
    assert intro["line_id"] == "__title_intro__"
    assert outro["line_id"] == "__title_outro__"
    assert intro["video_url"].startswith("https://")
    assert intro["seconds"] == 3.0
    # Two cards rendered at the deliverable's 16:9 dims via animated_title_card.
    assert video.record_animation.await_count == 2
    kwargs = video.record_animation.await_args_list[0].kwargs
    assert kwargs["template_id"] == "animated_title_card"
    assert kwargs["width"] == 1920 and kwargs["height"] == 1080
    # Intro title defaults to the project title, subtitle to the brand name; brand
    # accent flows through motion_brand_params.
    assert kwargs["params"]["title"] == "My Course"
    assert kwargs["params"]["subtitle"] == "Acme"
    assert kwargs["params"]["accent"] == "#14B8A6"


@pytest.mark.asyncio
async def test_unusable_bg_falls_back_to_template_default(fake_storage) -> None:
    """A mid-grey background where neither default text clears contrast → no
    surface override; the template keeps its own readable canvas."""
    video = _video_fake()
    ctx = _ctx(
        {
            "title": "My Course",
            "brand": {"name": "Acme"},
            "theme": {"background_color": "#777777"},
            "deliverable": {"format": "16x9"},
            "title_card": {"enabled": True},
        }
    )
    await build_title_card_shots(ctx, ordered_seconds=None, client=video)
    params = video.record_animation.await_args_list[0].kwargs["params"]
    assert "bg" not in params and "text" not in params


@pytest.mark.asyncio
async def test_intro_only(fake_storage) -> None:
    video = _video_fake()
    ctx = _ctx(
        {
            "title": "T",
            "brand": {"name": "Acme"},
            "deliverable": {"format": "9x16"},
            "title_card": {"enabled": True, "outro": False},
            "resolution": "1080p",
        }
    )
    intro, outro = await build_title_card_shots(ctx, ordered_seconds=None, client=video)
    assert intro and outro is None
    assert video.record_animation.await_count == 1
    kwargs = video.record_animation.await_args.kwargs
    assert kwargs["width"] == 1080 and kwargs["height"] == 1920  # vertical format


@pytest.mark.asyncio
async def test_blank_title_skips_card(fake_storage) -> None:
    """Intro with no title source (no cfg.title, no project title) renders nothing
    rather than a blank card; outro still falls back to the brand name."""
    video = _video_fake()
    ctx = _ctx(
        {
            "title": "",
            "brand": {"name": "Acme"},
            "deliverable": {"format": "16x9"},
            "title_card": {"enabled": True},
        }
    )
    intro, outro = await build_title_card_shots(ctx, ordered_seconds=None, client=video)
    assert intro is None  # nothing to title
    assert outro and outro["video_url"]  # outro uses brand name
    assert video.record_animation.await_count == 1


@pytest.mark.asyncio
async def test_render_error_soft_fails(fake_storage) -> None:
    video = _video_fake()
    video.record_animation.side_effect = RuntimeError("vps down")
    ctx = _ctx(
        {
            "title": "T",
            "brand": {"name": "Acme"},
            "deliverable": {"format": "16x9"},
            "title_card": {"enabled": True},
        }
    )
    intro, outro = await build_title_card_shots(ctx, ordered_seconds=None, client=video)
    # Both sides soft-fail to None; compile still proceeds.
    assert intro is None and outro is None


@pytest.mark.asyncio
async def test_custom_duration_and_outro_text(fake_storage) -> None:
    video = _video_fake()
    ctx = _ctx(
        {
            "title": "T",
            "brand": {"name": "Acme"},
            "deliverable": {"format": "16x9"},
            "title_card": {
                "enabled": True,
                "duration_seconds": 5,
                "outro_title": "Thanks!",
                "outro_subtitle": "see you next time",
            },
        }
    )
    intro, _outro = await build_title_card_shots(ctx, ordered_seconds=None, client=video)
    assert intro is not None
    assert intro["seconds"] == 5.0
    out_kwargs = video.record_animation.await_args_list[1].kwargs
    assert out_kwargs["params"]["title"] == "Thanks!"
    assert out_kwargs["params"]["subtitle"] == "see you next time"
    assert out_kwargs["duration_ms"] == 5000


@pytest.mark.asyncio
async def test_render_error_records_drop_reason_in_notes(fake_storage) -> None:
    """A dropped card must be diagnosable from the persisted stage output, not
    from a log line that dies with the pod. One sentence per dropped card, naming
    WHICH card and the exception CLASS + detail."""
    video = _video_fake()
    video.record_animation.side_effect = RuntimeError("vps down")
    ctx = _ctx(
        {
            "title": "T",
            "brand": {"name": "Acme"},
            "deliverable": {"format": "16x9"},
            "title_card": {"enabled": True},
        }
    )
    notes: list[str] = []
    intro, outro = await build_title_card_shots(
        ctx, ordered_seconds=None, client=video, notes=notes
    )

    assert intro is None and outro is None  # still soft-fail: the compile proceeds
    assert len(notes) == 2, notes
    assert any("title-intro" in n for n in notes)
    assert any("title-outro" in n for n in notes)
    for n in notes:
        assert "render failed, card dropped" in n
        assert "RuntimeError" in n  # class => infra-vs-content discriminator
        assert "vps down" in n  # detail


@pytest.mark.asyncio
async def test_drop_reason_names_a_timeout_distinctly(fake_storage) -> None:
    """An infra timeout must be distinguishable from a content/render error by the
    note alone — that is the whole point of carrying the exception class."""
    video = _video_fake()
    video.wait_for_content_recording.side_effect = TimeoutError("recording poll expired")
    ctx = _ctx(
        {
            "title": "T",
            "brand": {"name": "Acme"},
            "deliverable": {"format": "16x9"},
            "title_card": {"enabled": True, "outro": False},
        }
    )
    notes: list[str] = []
    intro, _outro = await build_title_card_shots(
        ctx, ordered_seconds=None, client=video, notes=notes
    )
    assert intro is None
    assert notes == [
        "title_card: title-intro render failed, card dropped: TimeoutError: recording poll expired"
    ]


@pytest.mark.asyncio
async def test_successful_render_records_no_note(fake_storage) -> None:
    """The notes channel auto-degrades a run (`runs/_shared.py`), so a clean render
    must leave it empty."""
    video = _video_fake()
    ctx = _ctx(
        {
            "title": "T",
            "brand": {"name": "Acme"},
            "deliverable": {"format": "16x9"},
            "title_card": {"enabled": True},
        }
    )
    notes: list[str] = []
    intro, outro = await build_title_card_shots(
        ctx, ordered_seconds=None, client=video, notes=notes
    )
    assert intro and outro
    assert notes == []


@pytest.mark.asyncio
async def test_omitted_notes_sink_is_a_noop(fake_storage) -> None:
    """Backwards compatible: a caller with no notes channel still soft-fails."""
    video = _video_fake()
    video.record_animation.side_effect = RuntimeError("vps down")
    ctx = _ctx(
        {
            "title": "T",
            "brand": {"name": "Acme"},
            "deliverable": {"format": "16x9"},
            "title_card": {"enabled": True},
        }
    )
    intro, outro = await build_title_card_shots(ctx, ordered_seconds=None, client=video)
    assert intro is None and outro is None


# ── degradations sink (defect 3: structured code, not unclassified_note) ────


@pytest.mark.asyncio
async def test_render_error_records_a_structured_degradation_code(fake_storage) -> None:
    """(d) A dropped card must be QUERYABLE, not just readable in prose. Before
    this, a failed title-card render appended only a ``notes`` sentence, which
    the DAG executor stamps the generic ``unclassified_note`` code —
    indistinguishable from every other kind of soft-failed stage note.

    MUTATION PROOF: removing the ``if degradations is not None: ...append(...)``
    block in ``title_cards._render``'s except clause turns this red (an empty
    list instead of one TITLE_CARD_DROPPED entry).
    """
    video = _video_fake()
    video.record_animation.side_effect = RuntimeError("vps down")
    ctx = _ctx(
        {
            "title": "T",
            "brand": {"name": "Acme"},
            "deliverable": {"format": "16x9"},
            "title_card": {"enabled": True},
        }
    )
    notes: list[str] = []
    degradations: list[Degradation] = []
    intro, outro = await build_title_card_shots(
        ctx, ordered_seconds=None, client=video, notes=notes, degradations=degradations
    )

    assert intro is None and outro is None  # still soft-fail: the compile proceeds
    assert len(degradations) == 2, degradations  # one per dropped card (intro + outro)
    codes = [d["code"] for d in degradations]
    assert codes == [TITLE_CARD_DROPPED, TITLE_CARD_DROPPED]
    assert UNCLASSIFIED_NOTE not in codes
    for d in degradations:
        assert "RuntimeError" in d["detail"]
        assert "vps down" in d["detail"]
        assert "render_error" in d["detail"]  # NOT a timeout


@pytest.mark.asyncio
async def test_degradation_detail_distinguishes_timeout_from_render_error(
    fake_storage,
) -> None:
    """The structured code's ``detail`` must say WHICH failure shape it was — a
    poll TIMEOUT (a different owner: a slow/wedged recorder) vs a genuine
    RENDER error (a bad template/params) — exactly as the prose note already
    did before this fix."""
    video = _video_fake()
    video.wait_for_content_recording.side_effect = TimeoutError("recording poll expired")
    ctx = _ctx(
        {
            "title": "T",
            "brand": {"name": "Acme"},
            "deliverable": {"format": "16x9"},
            "title_card": {"enabled": True, "outro": False},
        }
    )
    degradations: list[Degradation] = []
    intro, _outro = await build_title_card_shots(
        ctx, ordered_seconds=None, client=video, degradations=degradations
    )
    assert intro is None
    assert len(degradations) == 1
    assert degradations[0]["code"] == TITLE_CARD_DROPPED
    assert "timeout" in degradations[0]["detail"]
    assert "render_error" not in degradations[0]["detail"]


@pytest.mark.asyncio
async def test_successful_render_records_no_degradation(fake_storage) -> None:
    """A clean render must leave the structured channel empty, same as notes."""
    video = _video_fake()
    ctx = _ctx(
        {
            "title": "T",
            "brand": {"name": "Acme"},
            "deliverable": {"format": "16x9"},
            "title_card": {"enabled": True},
        }
    )
    degradations: list[Degradation] = []
    intro, outro = await build_title_card_shots(
        ctx, ordered_seconds=None, client=video, degradations=degradations
    )
    assert intro and outro
    assert degradations == []


@pytest.mark.asyncio
async def test_omitted_degradations_sink_is_a_noop(fake_storage) -> None:
    """Backwards compatible: a caller with no degradations channel (only the
    prose ``notes`` sink, or neither) still soft-fails without raising."""
    video = _video_fake()
    video.record_animation.side_effect = RuntimeError("vps down")
    ctx = _ctx(
        {
            "title": "T",
            "brand": {"name": "Acme"},
            "deliverable": {"format": "16x9"},
            "title_card": {"enabled": True},
        }
    )
    notes: list[str] = []
    intro, outro = await build_title_card_shots(
        ctx, ordered_seconds=None, client=video, notes=notes
    )
    assert intro is None and outro is None
    assert len(notes) == 2


# ── splice_title_cards ─────────────────────────────────────────────────────


def _shots() -> list[dict[str, Any]]:
    return [{"id": "a", "line_id": "l1"}, {"id": "b", "line_id": "l2"}]


def test_splice_none_returns_original() -> None:
    shots = _shots()
    assert splice_title_cards(shots, None, None) is shots


def test_splice_brackets_in_order() -> None:
    intro = {"id": "i", "line_id": "__title_intro__"}
    outro = {"id": "o", "line_id": "__title_outro__"}
    out = splice_title_cards(_shots(), intro, outro)
    assert [s["line_id"] for s in out] == ["__title_intro__", "l1", "l2", "__title_outro__"]


def test_splice_intro_only() -> None:
    intro = {"id": "i", "line_id": "__title_intro__"}
    out = splice_title_cards(_shots(), intro, None)
    assert [s["line_id"] for s in out] == ["__title_intro__", "l1", "l2"]


def test_splice_outro_only() -> None:
    outro = {"id": "o", "line_id": "__title_outro__"}
    out = splice_title_cards(_shots(), None, outro)
    assert [s["line_id"] for s in out] == ["l1", "l2", "__title_outro__"]


# ── one gate: reserved seconds == rendered seconds ────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "block",
    [None],
)
async def test_reserved_card_seconds_equal_what_the_compile_renders(
    fake_storage, block: dict[str, Any] | None
) -> None:
    """The website_review order floor / script budget reserve ``title_card_seconds``;
    the compile renders ``build_title_card_shots``. Both read ``plan_title_cards``,
    so the reservation is exactly the rendered card time for every block shape —
    including NO block, which renders (and reserves) nothing."""
    from src.services.stages.shared.title_cards import title_card_seconds

    snapshot: dict[str, Any] = {"title": "T", "deliverable": {"format": "16x9"}}
    if block is not None:
        snapshot["title_card"] = block
    intro, outro = await build_title_card_shots(
        _ctx(snapshot), ordered_seconds=None, client=_video_fake()
    )
    rendered = sum(float(shot["seconds"]) for shot in (intro, outro) if shot)
    assert title_card_seconds(block, ordered_seconds=None) == pytest.approx(rendered)


# ── the 20% cap: short orders are not mostly silent card time ─────────────
#
# Measured on 8 live website_review compiles: fixed 2 x 3.0s cards were 6.08s of a
# 9.7s film. Cards may take at most 20% of the order, never run under 1.5s, and the
# outro goes first.

_BOTH = {"enabled": True}


def _cards(plan: TitleCardPlan | None) -> tuple[float, float]:
    """``(intro_seconds, outro_seconds)`` a plan puts on screen; 0.0 = no such card."""
    if plan is None:
        return (0.0, 0.0)
    return (
        plan.seconds_per_card if plan.intro else 0.0,
        plan.seconds_per_card if plan.outro else 0.0,
    )


def test_the_cap_constants_are_the_decided_numbers() -> None:
    assert MAX_TITLE_CARD_SHARE == 0.20
    assert MIN_TITLE_CARD_SECONDS == 1.5


@pytest.mark.parametrize(
    ("ordered", "intro", "outro"),
    [(60, 3.0, 3.0)],
)
def test_worked_examples_of_the_twenty_percent_cap(
    ordered: float, intro: float, outro: float
) -> None:
    plan = plan_title_cards(_BOTH, ordered_seconds=ordered)
    assert _cards(plan) == (intro, outro)
    assert title_card_seconds(_BOTH, ordered_seconds=ordered) == intro + outro
    assert intro + outro <= MAX_TITLE_CARD_SHARE * ordered


@pytest.mark.parametrize("ordered", [30])
def test_an_order_of_thirty_seconds_and_up_plans_exactly_what_it_did(ordered: float) -> None:
    """Orders that already fit must not move: the capped plan EQUALS the uncapped one."""
    for block in (_BOTH, {"enabled": True, "outro": False}, {"enabled": True, "intro": False}):
        assert plan_title_cards(block, ordered_seconds=ordered) == plan_title_cards(
            block, ordered_seconds=None
        )


@pytest.mark.parametrize("ordered", [5])
@pytest.mark.parametrize("block", [None])
def test_cards_not_opted_in_plan_none_at_any_order(block: Any, ordered: float) -> None:
    assert plan_title_cards(block, ordered_seconds=ordered) is None
    assert title_card_seconds(block, ordered_seconds=ordered) == 0.0


@pytest.mark.parametrize("side", ["intro"])
@pytest.mark.parametrize(
    ("ordered", "seconds"),
    [(60, 3.0)],
)
def test_a_single_opted_in_card_obeys_the_same_cap(
    side: str, ordered: float, seconds: float
) -> None:
    """One card: kept at its configured length while it fits, shrunk to the 20% budget,
    and gone once the budget is under 1.5s. An outro-only block keeps its OUTRO — the
    outro is only dropped to make room for an intro."""
    other = "outro" if side == "intro" else "intro"
    block = {"enabled": True, other: False}
    plan = plan_title_cards(block, ordered_seconds=ordered)
    if seconds == 0.0:
        assert plan is None
        return
    assert plan is not None
    assert (getattr(plan, side), getattr(plan, other)) == (True, False)
    assert plan.seconds_per_card == pytest.approx(seconds)


def test_a_shrunk_card_is_never_shorter_than_the_minimum() -> None:
    """Sweep every 0.1s order from 1s to 40s: any card planned is >= 1.5s (or is the
    configured length, which only happens when it already fits) and the total never
    exceeds 20% of the order."""
    for tenths in range(10, 401):
        ordered = tenths / 10
        plan = plan_title_cards(_BOTH, ordered_seconds=ordered)
        if plan is None:
            assert MAX_TITLE_CARD_SHARE * ordered < MIN_TITLE_CARD_SECONDS, ordered
            continue
        assert plan.seconds_per_card >= MIN_TITLE_CARD_SECONDS, ordered
        assert plan.total_seconds <= MAX_TITLE_CARD_SHARE * ordered + 1e-9, ordered
        assert plan.intro, ordered  # the intro is the card that survives


def test_no_resolved_order_plans_the_block_as_configured() -> None:
    """``ordered_seconds=None`` (a product with no order resolver) is today's plan."""
    assert plan_title_cards(_BOTH, ordered_seconds=None) == TitleCardPlan(
        intro=True, outro=True, seconds_per_card=3.0
    )
    assert plan_title_cards(_BOTH, ordered_seconds=0) == TitleCardPlan(
        intro=True, outro=True, seconds_per_card=3.0
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ordered", "intro_s", "outro_s"),
    [(60, 3.0, 3.0)],
)
async def test_the_render_records_the_capped_duration(
    fake_storage, ordered: float, intro_s: float | None, outro_s: float | None
) -> None:
    """What video_production is ASKED to record is the capped length, not 3.0s."""
    video = _video_fake()
    snapshot = {"title": "T", "brand": {"name": "Acme"}, "title_card": _BOTH}
    intro, outro = await build_title_card_shots(
        _ctx(snapshot), ordered_seconds=ordered, client=video
    )
    assert (intro["seconds"] if intro else None) == intro_s
    assert (outro["seconds"] if outro else None) == outro_s
    recorded = [c.kwargs["duration_ms"] for c in video.record_animation.await_args_list]
    assert recorded == [round(s * 1000) for s in (intro_s, outro_s) if s is not None]
