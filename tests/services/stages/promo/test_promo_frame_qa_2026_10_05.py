"""Regression tests for the six defects frame QA found in promo run 01a10d84 (2026-10-05).

1. kinetic cards were placed on the WRONG lines (unkeyed → round-robin pool);
2. kinetic cards were 1080x1080 squares cover-cropped into a 16:9 frame;
3. the film ended on the CTA line's blank gradient — no branded end card;
4. Ken Burns abandoned still-running recordings at one 600s poll (2 of 4 stills frozen);
5. sound design ON could ship with no music bed and no trace;
6. a 60s order delivered 35.4s from a verbatim script, with nothing filling the gap.

Audit follow-ups (review of #1066 head a2dc2a66): card binding is on whole words
with exact matches first and keyed lines reserved; the end card counts toward the
order's +/-10% band; Ken Burns and captions share the 1800s pre-submit deadline
so a congested fleet degrades the render instead of failing it; card binding and
film-frame dims are promo-only; the wiring the earlier tests handed in by hand is
driven through a real ``PromoRenderStage.run``.

The fixtures use the run's real script lines (promo_script of run 01a10d84,
verbatim) and its measured narration lengths.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.services.runs.run_context import RunContext
from src.services.stages.base import ArtifactUrl, StageInput, StageOutputWithNotes
from src.services.stages.generic.motion_graphics_gen import (
    MotionGraphicsGenStage,
    _bind_items_to_script,
)
from src.services.stages.primitives.compile_stage import CompileStageBase, audio_bed_record
from src.services.stages.promo import fill
from src.services.stages.promo.render import PromoRenderStage
from src.services.stages.shared import ken_burns
from src.services.stages.shared.degradation import CAPTIONS_MISSING, SOUND_BED_MISSING
from src.services.stages.shared.ken_burns import apply_ken_burns
from src.services.stages.shared.sound_design import resolve_audio_bed
from src.services.stages.shared.title_cards import render_title_card_plan
from src.services.stages.sound_catalog import SoundLibrary

# The run's seven verbatim lines (promo_script output of run 01a10d84, copied
# exactly) and their measured narration lengths (promo_narrate clip seconds).
_LINES = [
    ("l1", "Go-to-market that runs itself, under your control.", 2.778),
    (
        "l2",
        "Users can set graduated autonomy levels for 30+ capabilities, "
        "ranging from prohibited to automatic.",
        6.951,
    ),
    (
        "l3",
        "Critical financial and consent decisions are made by deterministic code, "
        "not language models.",
        5.982,
    ),
    (
        "l4",
        "The platform prioritizes pull-based demand generation, earning attention "
        "before requesting contact.",
        6.231,
    ),
    (
        "l5",
        "The system uses only public sources and respects robots.txt, avoiding "
        "scraped or bought lists.",
        6.431,
    ),
    (
        "l6",
        "High-ticket and sensitive conversations are automatically routed to a human agent.",
        4.876,
    ),
    ("l7", "Request access at merqi.nl", 2.126),
]
_CTA = "Request access at merqi.nl"


def _script() -> dict[str, Any]:
    return {"lines": [{"id": i, "text": t, "seconds": s} for i, t, s in _LINES]}


def _story(visual: str = "slide") -> dict[str, Any]:
    return {
        "shots": [
            {"id": f"s-{i}", "line_id": i, "seconds": s, "visual_type": visual}
            for i, _t, s in _LINES
        ]
    }


def _narrate() -> dict[str, Any]:
    return {
        "clips": [
            {"id": f"c-{i}", "line_id": i, "seconds": s, "artifact_url": f"https://m/{i}.wav"}
            for i, _t, s in _LINES
        ]
    }


def _ctx(config: dict[str, Any], prior: dict[str, Any] | None = None) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=config,
        prior_run_outputs=prior or {},
    )


def _vp(duration: float = 4.0) -> AsyncMock:
    client = AsyncMock()
    client.record_animation = AsyncMock(return_value="job-1")
    client.wait_for_content_recording = AsyncMock(return_value={"duration_seconds": duration})
    client.get_recording_video = AsyncMock(return_value=(b"\x00mp4", "video/mp4", None))
    return client


@pytest.fixture
def _store(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _stored(**kw: Any) -> str:
        return f"https://minio.test/v/{kw['video_id']}.mp4"

    monkeypatch.setattr(
        "src.services.stages.generic.motion_graphics_gen.store_generated_video", _stored
    )


def _kinetic(*claims: str) -> dict[str, Any]:
    return {
        "template_id": "kinetic_typography",
        "items": [{"params": {"words": claim.split()}} for claim in claims],
    }


# ---------------------------------------------------------------- 1. line binding


@pytest.mark.asyncio
@pytest.mark.usefixtures("_store")
async def test_kinetic_cards_bind_to_the_line_that_says_them_in_script_order() -> None:
    """The run's items arrived as [line 6, line 2, line 1] with no line_id; the
    pool put line 6's card over line 1's narration. Each card must key to the
    line whose text it shows, in script order."""
    cfg = {"motion_graphics": _kinetic(_LINES[5][1], _LINES[1][1], _LINES[0][1])}
    out = await MotionGraphicsGenStage(client=_vp()).run(
        {}, ctx=_ctx(cfg, {"promo_script": _script()})
    )
    assert [c.line_id for c in out.clips] == ["l1", "l2", "l6"]
    assert out.advisories == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("_store")
async def test_a_card_matching_no_line_is_dropped_never_placed_elsewhere() -> None:
    client = _vp()
    cfg = {"motion_graphics": _kinetic("A claim the narration never says", _LINES[0][1])}
    out = await MotionGraphicsGenStage(client=client).run(
        {}, ctx=_ctx(cfg, {"promo_script": _script()})
    )
    assert [c.line_id for c in out.clips] == ["l1"]
    assert client.record_animation.await_count == 1  # the unbound card is never paid for
    assert [a["code"] for a in out.advisories] == ["motion_item_unbound"]


def _bind(
    lines: list[tuple[str, str]], cards: list[str], keyed: list[dict[str, Any]] | None = None
) -> tuple[list[tuple[str, str | None]], list[str]]:
    """Bind card texts against a promo script; ``(card text, line_id)`` + unbound."""
    ctx = _ctx({}, {"promo_script": {"lines": [{"id": i, "text": t} for i, t in lines]}})
    items: list[dict[str, Any]] = [{"params": {"words": c.split()}} for c in cards]
    items += [dict(k) for k in keyed or []]
    unbound: list[str] = []
    out = _bind_items_to_script(items, ctx, unbound)
    shown: list[tuple[str, str | None]] = []
    for o in out:
        params = cast("dict[str, Any]", o["params"])
        line_id = o.get("line_id")
        shown.append(
            (
                " ".join(params.get("words") or []) or str(params),
                str(line_id) if line_id else None,
            )
        )
    return shown, unbound


def test_a_card_never_binds_through_a_word_fragment() -> None:
    """ "ai" is inside "maintain" as CHARACTERS; it is not a word of the card."""
    bound, unbound = _bind(
        [("L1", "AI."), ("L2", "Keep control of every decision.")], ["Maintain control"]
    )
    assert bound == []
    assert len(unbound) == 1


def test_a_short_line_is_not_a_prefix_of_an_unrelated_claim() -> None:
    bound, unbound = _bind(
        [("L1", "Yes."), ("L2", "Pull-based demand.")], ["Yesterday's lists are dead"]
    )
    assert bound == []
    assert len(unbound) == 1


def test_the_cta_card_binds_to_the_cta_line_not_the_brand_line() -> None:
    lines = [
        ("L1", "Merqi."),
        ("L2", "Go-to-market that runs itself."),
        ("L7", "Request access at merqi.nl"),
    ]
    bound, unbound = _bind(lines, ["Request access at merqi.nl"])
    assert bound == [("Request access at merqi.nl", "L7")]
    assert unbound == []


def test_the_full_headline_binds_to_its_exact_line_not_a_shorter_prefix_line() -> None:
    headline = "Go-to-market that runs itself, under your control."
    lines = [("L1", "Go-to-market that runs itself."), ("L5", headline)]
    bound, _unbound = _bind(lines, [headline])
    assert bound == [(headline, "L5")]


def test_one_shared_word_is_not_a_match() -> None:
    """A one-word brand line shares "merqi" with a claim card; that is a
    coincidence, not the line saying the card."""
    bound, unbound = _bind(
        [("L1", "Merqi."), ("L2", "Pull-based demand.")], ["Merqi books the meetings"]
    )
    assert bound == []
    assert len(unbound) == 1


def test_a_one_word_card_is_not_a_quote() -> None:
    """Every word of "Control" is spoken in the line, but one shared content word
    is a coincidence, not the line saying the card."""
    bound, unbound = _bind([("L1", "Keep control of every decision.")], ["Control"])
    assert bound == []
    assert len(unbound) == 1


def test_an_exact_card_is_never_displaced_by_an_earlier_partial_card() -> None:
    """Exact matches are placed before ANY partial one, whatever the item order:
    the abridged card that arrives first must not take the line the second card
    says word for word (and push that card onto a shorter line it only contains)."""
    headline = "Go-to-market that runs itself, under your control."
    lines = [("L1", "Go-to-market that runs itself."), ("L5", headline)]
    bound, unbound = _bind(lines, ["runs itself under your control", headline])
    assert bound == [(headline, "L5")]
    assert len(unbound) == 1


@pytest.mark.parametrize(
    ("line", "card"),
    [("Human approval.", "Human approval is optional for every send")],
)
def test_a_card_never_says_more_or_other_than_its_line(line: str, card: str) -> None:
    bound, unbound = _bind([("L1", line)], [card])
    assert bound == []
    assert len(unbound) == 1


def test_a_negated_card_binds_to_the_line_that_says_the_negation() -> None:
    bound, _unbound = _bind(
        [("L1", "You never lose control of a decision.")], ["Never lose control of a decision"]
    )
    assert [line_id for _text, line_id in bound] == ["L1"]


def test_among_lines_a_card_abridges_the_one_it_covers_most_wins() -> None:
    """Both lines say every word of the card; L2 says little else (overlap 4/5),
    L1 says much more (4/6). Script order would pick L1; the overlap picks L2."""
    lines = [
        ("L1", "Strategy research drafts content and qualifies leads."),
        ("L2", "Strategy research drafts content today."),
    ]
    bound, _unbound = _bind(lines, ["Strategy research drafts content"])
    assert [line_id for _text, line_id in bound] == ["L2"]


def test_an_explicitly_keyed_card_keeps_its_line_against_a_text_card() -> None:
    """The keyed card's line is reserved BEFORE text matching: the text card that
    says the same line is not also keyed to it (the render would overwrite one)."""
    keyed = {"line_id": "L1", "params": {"title": "keyed"}}
    bound, unbound = _bind([("L1", "Hello world, again.")], ["Hello world, again."], [keyed])
    assert [line_id for _text, line_id in bound] == ["L1"]
    assert bound[0][0] == str({"title": "keyed"})
    assert len(unbound) == 1


def test_an_abridged_card_still_binds_to_the_line_it_quotes() -> None:
    """Partial matching survives: every content word of the card is spoken in l6."""
    bound, _unbound = _bind(
        [(i, t) for i, t, _s in _LINES],
        ["HIGH TICKET and sensitive conversations — automatically routed to a human"],
    )
    assert [line_id for _text, line_id in bound] == ["l6"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("_store")
async def test_a_generic_run_is_not_bound_or_resized() -> None:
    """Scope: binding and film-frame dims are PROMO behaviours. A generic run (its
    script is ``script_draft``) keeps its unkeyed pool card and the block's own
    dims (unset → the service default), exactly as before."""
    client = _vp()
    cfg = {
        "motion_graphics": _kinetic("A card the generic draft never says"),
        "deliverable": {"format": "16x9"},
    }
    prior = {"script_draft": {"lines": [{"id": "g1", "text": "Something else entirely."}]}}
    out = await MotionGraphicsGenStage(client=client).run({}, ctx=_ctx(cfg, prior))
    assert [c.line_id for c in out.clips] == [None]
    assert out.advisories == []
    kwargs = client.record_animation.await_args.kwargs
    assert (kwargs["width"], kwargs["height"]) == (None, None)


# ------------------------------------------------------------- 2. aspect / crop


@pytest.mark.asyncio
@pytest.mark.usefixtures("_store")
async def test_cards_render_at_the_films_frame_not_the_services_square_default() -> None:
    client = _vp()
    cfg = {
        "motion_graphics": _kinetic(_LINES[0][1]),
        "deliverable": {"format": "16x9"},
        "resolution": "4k",
    }
    await MotionGraphicsGenStage(client=client).run({}, ctx=_ctx(cfg, {"promo_script": _script()}))
    kwargs = client.record_animation.await_args.kwargs
    assert (kwargs["width"], kwargs["height"]) == (3840, 2160)


# ------------------------------------------------------------- 3. end card


def _promo_cfg(**extra: Any) -> dict[str, Any]:
    return {
        "brand": {"name": "Merqi", "primary_color": "#000000"},
        "brief": {"call_to_action": _CTA, "target_duration_seconds": 60},
        "deliverable": {"format": "16x9"},
        **extra,
    }


def _promo_prior() -> dict[str, Any]:
    return {"promo_script": _script(), "promo_storyboard": _story(), "promo_narrate": _narrate()}


def test_promo_defaults_a_branded_end_card_with_the_cta() -> None:
    cfg = PromoRenderStage().title_card_config(_ctx(_promo_cfg(), _promo_prior()))
    assert isinstance(cfg, dict)
    assert cfg["enabled"] is True and cfg["outro"] is True and cfg["intro"] is False
    assert cfg["outro_title"] == "Merqi"
    assert cfg["outro_subtitle"] == _CTA


def test_an_explicit_title_card_block_wins_even_disabled() -> None:
    ctx = _ctx(_promo_cfg(title_card={"enabled": False}), _promo_prior())
    assert PromoRenderStage().title_card_config(ctx) == {"enabled": False}


@pytest.mark.asyncio
async def test_the_end_card_is_actually_rendered_from_the_promo_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.services.stages.shared.title_cards.store_generated_video",
        AsyncMock(return_value="https://minio.test/v/outro.mp4"),
    )
    client = _vp()
    ctx = _ctx(_promo_cfg(), _promo_prior())
    stage = PromoRenderStage()
    intro, outro = await render_title_card_plan(
        ctx, stage.title_card_plan(ctx), client=client, cfg=stage.title_card_config(ctx)
    )
    assert intro is None and outro is not None
    params = client.record_animation.await_args.kwargs["params"]
    assert (params["title"], params["subtitle"]) == ("Merqi", _CTA)


# ------------------------------------------------------------- 4. Ken Burns


@pytest.mark.asyncio
async def test_a_still_running_recording_is_waited_on_not_abandoned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fetch(url: str, client: Any) -> tuple[bytes, str]:
        return (b"\x89PNG\r\n", "image/png")

    monkeypatch.setattr(ken_burns, "_fetch_own_asset", _fetch)
    monkeypatch.setattr(
        ken_burns, "store_generated_video", AsyncMock(return_value="https://m/kb.mp4")
    )
    monkeypatch.setattr(ken_burns, "find_existing_media", AsyncMock(return_value=None))
    client = _vp()
    # First poll window ends with the job still queued/rendering; the second sees it done.
    client.wait_for_content_recording = AsyncMock(
        side_effect=[TimeoutError("job kb-1 timed out after 600.0s"), {"duration_seconds": 5.0}]
    )
    notes: list[str] = []
    shot = {"id": "s1", "line_id": "l3", "image_url": "https://minio.test/a.png", "seconds": 5.2}
    out = await apply_ken_burns(_ctx({}), [shot], client=client, notes=notes)

    assert out[0].get("video_url") == "https://m/kb.mp4"
    assert notes == []
    assert client.record_animation.await_count == 1  # same job, never re-submitted


def _kb_fakes(monkeypatch: pytest.MonkeyPatch, cached: str | None) -> AsyncMock:
    async def _fetch(url: str, client: Any) -> tuple[bytes, str]:
        return (b"\x89PNG\r\n", "image/png")

    monkeypatch.setattr(ken_burns, "_fetch_own_asset", _fetch)
    monkeypatch.setattr(
        ken_burns, "store_generated_video", AsyncMock(return_value="https://m/kb.mp4")
    )
    monkeypatch.setattr(ken_burns, "find_existing_media", AsyncMock(return_value=cached))
    return _vp()


_STILL = {"id": "s1", "line_id": "l3", "image_url": "https://minio.test/a.png", "seconds": 5.2}


@pytest.mark.asyncio
async def test_with_under_30s_left_ken_burns_records_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recording cannot finish in the seconds left: none is submitted (no bill
    for a clip nobody will wait for), the still stays, and the note says why."""
    client = _kb_fakes(monkeypatch, cached=None)
    notes: list[str] = []
    out = await apply_ken_burns(
        _ctx({}), [dict(_STILL)], client=client, notes=notes, budget_seconds=20.0
    )
    client.record_animation.assert_not_awaited()
    assert not out[0].get("video_url")
    assert len(notes) == 1 and "less than 30s" in notes[0]


@pytest.mark.asyncio
async def test_a_clip_cached_by_an_earlier_attempt_is_used_even_with_no_budget_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resumed compile (pod roll) finds the clip the dead pod stored: one lookup,
    no recording — so a near-exhausted budget must not throw it away."""
    client = _kb_fakes(monkeypatch, cached="https://m/cached-kb.mp4")
    notes: list[str] = []
    hits: list[int] = []
    out = await apply_ken_burns(
        _ctx({}),
        [dict(_STILL)],
        client=client,
        notes=notes,
        budget_seconds=5.0,
        cache_hits=hits,
    )
    assert out[0]["video_url"] == "https://m/cached-kb.mp4"
    assert hits == [0]
    client.record_animation.assert_not_awaited()
    assert notes == []


@pytest.mark.asyncio
async def test_the_caption_timeout_leaves_30s_before_the_aggregate_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The caption build's own timeout must fire (soft: CAPTIONS_MISSING) with a
    margin before the aggregate one (hard: the render fails). Measured on the
    timeouts actually armed, at the production deadline."""
    import asyncio

    from src.services.stages.primitives import compile_stage as cs_mod

    armed: list[float | None] = []
    real_timeout = asyncio.timeout

    def _recording_timeout(delay: float | None) -> Any:
        armed.append(delay)
        return real_timeout(delay)

    monkeypatch.setattr(cs_mod.asyncio, "timeout", _recording_timeout)
    await _run_captioned(
        monkeypatch, _BudgetStage, _ctx({"deliverable": {"captions_mode": "burned_in"}}), 0.0
    )
    outer = cs_mod._PRESUBMIT_DEADLINE_SECONDS
    assert outer in armed
    inner = [d for d in armed if d is not None and d < outer]
    assert inner, armed
    assert outer - 31.0 <= max(inner) <= outer - 30.0


def test_ken_burns_runs_stills_in_parallel_within_the_presubmit_budget() -> None:
    from src.services.stages.primitives import compile_stage

    # Parallel, but never past video_production's GUARANTEED capacity (per-pod cap
    # 3 x HPA min 1 pod): at 4 a one-pod fleet always queued the fourth still, which
    # failed QUEUE_SATURATED on run 01a10fcf (2026-10-06).
    assert ken_burns._CONCURRENCY == ken_burns._RECORDER_PER_POD_MAX == 3
    assert ken_burns._RECORDER_MIN_REPLICAS == 1
    assert ken_burns._PER_ITEM_TIMEOUT_S > ken_burns._PHASE_BUDGET_S
    assert ken_burns._PHASE_BUDGET_S < compile_stage._PRESUBMIT_DEADLINE_SECONDS
    # The share-out the pre-submit comment derives: promo's end card (one 600s
    # recording wait) plus the tail reserve (which must hold one 300s ASR call)
    # leaves Ken Burns a positive budget inside the aggregate deadline.
    from src.services.stages.shared import title_cards

    assert compile_stage._PRESUBMIT_TAIL_RESERVE_SECONDS >= 300.0
    assert (
        title_cards._RECORD_TIMEOUT_S
        + compile_stage._PRESUBMIT_TAIL_RESERVE_SECONDS
        + ken_burns._MIN_PHASE_BUDGET_S
        < compile_stage._PRESUBMIT_DEADLINE_SECONDS
    )


# ------------------------------------------------------------- 5. music bed


def _library(monkeypatch: pytest.MonkeyPatch, library: SoundLibrary) -> None:
    async def _load(*_a: object, **_k: object) -> SoundLibrary:
        return library

    async def _presign(sfx_id: str) -> str:
        return f"https://minio.test/sound/{sfx_id}.mp3"

    monkeypatch.setattr("src.services.stages.shared.sound_design.load_sound_library", _load)
    monkeypatch.setattr("src.services.stages.shared.sound_design.presign_sound_asset", _presign)


@pytest.mark.asyncio
async def test_music_asked_for_but_unresolvable_is_recorded_not_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _library(monkeypatch, SoundLibrary.builtin())
    cfg = {"enabled": True, "music": "not_a_real_track", "ambient": False}
    sink: list[Any] = []
    bed = await resolve_audio_bed(
        _ctx({"sound_effects": cfg}), [{"seconds": 5.0}], cfg, degradations=sink
    )
    assert not (bed or {}).get("music")
    assert [d["code"] for d in sink] == [SOUND_BED_MISSING]


@pytest.mark.asyncio
async def test_music_explicitly_off_records_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    _library(monkeypatch, SoundLibrary.builtin())
    cfg = {"enabled": True, "music": False, "ambient": False}
    sink: list[Any] = []
    await resolve_audio_bed(
        _ctx({"sound_effects": cfg}), [{"seconds": 5.0}], cfg, degradations=sink
    )
    assert sink == []


def test_the_bed_the_compile_sent_is_recorded_positively() -> None:
    record = audio_bed_record(
        {"music": {"url": "https://m/bed.mp3"}, "ambient": [{"url": "https://m/a.mp3"}]},
        {"music_bed": {"applied": True}},
    )
    assert record == {"music_sent": True, "ambient_cues_sent": 1, "compiler": {"applied": True}}
    assert audio_bed_record({}, {})["music_sent"] is False


# ------------------------------------------------------------- 6. duration fill


def test_the_runs_shortfall_is_filled_inside_tolerance() -> None:
    plan = fill.promo_fill(_ctx(_promo_cfg(), _promo_prior()))
    assert plan.narrated_seconds == pytest.approx(35.375, abs=0.01)
    assert plan.end_card_seconds == pytest.approx(8.0)
    assert plan.hold_seconds == pytest.approx(2.0)
    assert plan.projected_seconds >= 60 * 0.9
    assert plan.infeasible_reason is None


def test_a_shortfall_the_fill_cannot_close_is_named_with_its_numbers() -> None:
    cfg = _promo_cfg()
    cfg["brief"]["target_duration_seconds"] = 120
    reason = fill.promo_fill(_ctx(cfg, _promo_prior())).infeasible_reason
    assert reason is not None and "120s ordered" in reason


def test_an_on_target_promo_gets_no_padding() -> None:
    cfg = _promo_cfg()
    cfg["brief"]["target_duration_seconds"] = 38
    plan = fill.promo_fill(_ctx(cfg, _promo_prior()))
    assert plan.hold_seconds == 0.0 and plan.end_card_seconds == pytest.approx(3.0)


def test_the_render_lengthens_every_narrated_shot_by_the_hold() -> None:
    stage = PromoRenderStage()
    ctx = _ctx(_promo_cfg(), _promo_prior())
    shots = {s["line_id"]: s for s in stage.build_shots(ctx)}
    for line_id, _text, seconds in _LINES:
        assert shots[line_id]["seconds"] == pytest.approx(seconds + 2.0, abs=0.01)


def test_a_30s_order_with_31s_of_narration_stays_inside_the_band() -> None:
    """The end card COUNTS toward the order. A 3s default card on 31s of narration
    for a 30s order shipped 34s (+13%); the card shrinks into the room left."""
    plan = fill.plan_promo_fill(30.0, [31.0], end_card=True)
    assert plan.hold_seconds == 0.0
    # 33.0s ceiling - 31s narration - 0.25s encoder headroom.
    assert plan.end_card_seconds == pytest.approx(1.75)
    assert 30 * 0.9 <= plan.projected_seconds <= 30 * 1.1
    assert plan.infeasible_reason is None


def test_the_shrunk_card_is_the_card_that_renders() -> None:
    cfg = _promo_cfg()
    cfg["brief"]["target_duration_seconds"] = 30
    prior = _promo_prior()
    # 31s of narration over the run's seven lines.
    scale = 31.0 / sum(s for _i, _t, s in _LINES)
    for clip in prior["promo_narrate"]["clips"]:
        clip["seconds"] *= scale
    ctx = _ctx(cfg, prior)
    block = PromoRenderStage().title_card_config(ctx)
    assert isinstance(block, dict)
    assert block["duration_seconds"] == fill.promo_fill(ctx).end_card_seconds
    assert 31.0 + block["duration_seconds"] <= 33.0


def test_no_room_for_a_readable_card_drops_it_and_says_why() -> None:
    """30s ordered, 31.6s narrated: 1.4s of room is under the 1.5s readable minimum.
    The card is dropped to stay in the band, and the drop carries its numbers."""
    plan = fill.plan_promo_fill(30.0, [31.6], end_card=True)
    assert plan.end_card_seconds == 0.0
    assert plan.projected_seconds <= 33.0
    assert plan.infeasible_reason is None
    reason = plan.end_card_dropped
    assert reason is not None
    assert "31.6s of narration" in reason and "1.40s under the 33.0s ceiling" in reason
    assert "order >= 31s" in reason
    assert fill.plan_promo_fill(31.0, [31.6], end_card=True).end_card_dropped is None


def test_a_dropped_card_is_never_regrown_into_an_unreadable_sliver() -> None:
    """10s ordered, 9.8s narrated: 1.2s of room drops the card, and the 0.2s still
    short of the order goes to the pause — never back into a 0.2s 'card'."""
    plan = fill.plan_promo_fill(10.0, [9.8], end_card=True)
    assert plan.end_card_seconds == 0.0
    assert plan.end_card_dropped is not None
    assert plan.hold_seconds == pytest.approx(0.2)


def test_narration_already_over_the_ceiling_keeps_the_cta_and_is_flagged() -> None:
    """15s ordered, 20s narrated: the film is off the band whatever the card does,
    so dropping the CTA would buy nothing. Keep it at the minimum; flag the plan."""
    plan = fill.plan_promo_fill(15.0, [20.0], end_card=True)
    assert plan.end_card_seconds == pytest.approx(fill._END_CARD_MIN_S)
    assert plan.end_card_dropped is None
    reason = plan.infeasible_reason
    assert reason is not None and "over the 16.5s ceiling" in reason


def test_pauses_never_cost_the_card_its_room() -> None:
    """A short order where pauses and the card compete for the same seconds: the
    CTA is placed first, at its minimum, and the pauses get only what is left."""
    plan = fill.plan_promo_fill(15.0, [3.7, 3.7, 3.7, 3.7], end_card=True)
    assert plan.end_card_seconds == pytest.approx(fill._END_CARD_MIN_S)
    assert plan.end_card_dropped is None
    assert plan.hold_seconds == 0.0
    assert plan.projected_seconds <= 15.0 * 1.1


@pytest.mark.asyncio
async def test_a_dropped_end_card_is_recorded_on_the_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _promo_cfg()
    cfg["brief"]["target_duration_seconds"] = 30
    scale = 31.6 / sum(s for _i, _t, s in _LINES)
    out, captured, title_vp = await _render(monkeypatch, cfg, 31.6, narration_scale=scale)
    assert captured["shots"][-1]["line_id"] != "__title_outro__"
    title_vp.record_animation.assert_not_awaited()
    dropped = [d for d in out.degradations if d["code"] == "end_card_dropped"]
    assert len(dropped) == 1 and "1.40s under the 33.0s ceiling" in dropped[0]["detail"]


def test_the_end_card_length_comes_from_the_fill_plan() -> None:
    """The run's own numbers stretch the card to 8s; a hardcoded length would not."""
    ctx = _ctx(_promo_cfg(), _promo_prior())
    block = PromoRenderStage().title_card_config(ctx)
    assert isinstance(block, dict)
    assert block["duration_seconds"] == pytest.approx(8.0)
    assert block["duration_seconds"] == fill.promo_fill(ctx).end_card_seconds


# ------------------------------------------------- wiring through a real compile


async def _render(
    monkeypatch: pytest.MonkeyPatch,
    cfg: dict[str, Any],
    delivered: float,
    *,
    narration_scale: float = 1.0,
    prepare: Callable[[RunContext], Awaitable[None]] | None = None,
) -> tuple[Any, dict[str, Any], AsyncMock]:
    """Run PromoRenderStage end to end: real build_shots, real title-card splice,
    real presubmit and build_output; only video_production, storage and the
    compiler are faked. ``prepare`` runs earlier stages on the render's own ctx
    (it adds their outputs to ``prior_run_outputs``). Returns the persisted
    output, the compile kwargs and the title-card VP client."""
    _library(monkeypatch, SoundLibrary.builtin())
    title_vp = _vp()
    monkeypatch.setattr(
        "src.services.stages.shared.title_cards.VideoProductionClient", lambda: title_vp
    )
    monkeypatch.setattr(
        "src.services.stages.shared.title_cards.store_generated_video",
        AsyncMock(return_value="https://minio.test/v/outro.mp4"),
    )
    captured: dict[str, Any] = {}

    async def _compile(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "artifact_url": "http://stub/promo.mp4",
            "duration_seconds": delivered,
            "declared_duration_seconds": delivered,
            "actual_duration_seconds": delivered,
        }

    compiler = AsyncMock()
    compiler.compile = AsyncMock(side_effect=_compile)
    prior = _promo_prior()
    for clip in prior["promo_narrate"]["clips"]:
        clip["seconds"] *= narration_scale
    ctx = _ctx(cfg, prior)
    if prepare is not None:
        await prepare(ctx)
    out = await PromoRenderStage(client=compiler).run({}, ctx=ctx)
    return out, captured, title_vp


@pytest.mark.asyncio
async def test_a_real_promo_compile_ends_on_the_branded_end_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing hands the block in: the compile path itself must ask the promo
    stage for its default card and splice it as the LAST shot."""
    _out, captured, title_vp = await _render(monkeypatch, _promo_cfg(), 57.4)
    last = captured["shots"][-1]
    assert last["line_id"] == "__title_outro__"
    assert last["video_url"] == "https://minio.test/v/outro.mp4"
    assert last["seconds"] == pytest.approx(8.0)
    params = title_vp.record_animation.await_args.kwargs["params"]
    assert (params["title"], params["subtitle"]) == ("Merqi", _CTA)


@pytest.mark.asyncio
async def test_a_real_promo_compile_persists_the_audio_bed_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, _captured, _vp_client = await _render(monkeypatch, _promo_cfg(), 57.4)
    assert out.audio_bed is not None
    assert set(out.audio_bed) == {"music_sent", "ambient_cues_sent", "compiler"}


@pytest.mark.asyncio
async def test_a_real_promo_compile_records_an_infeasible_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _promo_cfg()
    cfg["brief"]["target_duration_seconds"] = 120
    out, _captured, _vp_client = await _render(monkeypatch, cfg, 57.4)
    conformance = out.duration_conformance
    assert conformance is not None
    assert conformance["plan_infeasible"] is True
    assert "120s ordered" in str(conformance["plan_infeasible_reason"])


# ------------------------------------------------- shared pre-submit budget


class _BudgetOutput(StageOutputWithNotes):
    artifact_url: ArtifactUrl
    duration_seconds: float


class _BudgetStage(CompileStageBase[StageInput, _BudgetOutput]):
    """A minimal compile stage: one still shot, burned-in captions, an outro."""

    name = "budget_compile"
    input_model = StageInput
    output_model = _BudgetOutput

    def build_shots(self, ctx: Any) -> list[dict[str, Any]]:
        return [
            {"id": "s1", "line_id": "l1", "seconds": 2.0, "image_url": "https://minio.test/a.png"}
        ]

    def build_output(self, result: dict[str, Any], ctx: Any) -> _BudgetOutput:
        return _BudgetOutput(
            artifact_url=str(result["artifact_url"]),
            duration_seconds=float(result["duration_seconds"]),
        )

    def compile_tags(self, ctx: Any) -> dict[str, str]:
        return {}

    def caption_clip_refs(self, ctx: Any) -> dict[str, Any]:
        return {"l1": object()}


@pytest.mark.asyncio
async def test_a_congested_fleet_degrades_the_render_instead_of_failing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scaled-down congestion: the title card eats most of the pre-submit budget,
    every VP recording is stuck in queue, and the ASR fallback is slow. Before,
    Ken Burns polled to its own 1000s cap and captions ran unbounded, so the
    AGGREGATE deadline fired and hard-failed the whole render. Now Ken Burns gets
    its OWN (bounded) budget — the aggregate deadline is moved out to hold it —
    the stuck still stays a still, and captions time out softly."""
    import asyncio

    from src.services import tts_storage
    from src.services.stages.primitives import compile_stage as cs_mod

    monkeypatch.setattr(cs_mod, "_PRESUBMIT_DEADLINE_SECONDS", 3.0)
    monkeypatch.setattr(cs_mod, "_PRESUBMIT_TAIL_RESERVE_SECONDS", 1.0)
    monkeypatch.setattr(cs_mod, "_PRESUBMIT_FINAL_MARGIN_SECONDS", 0.3)
    monkeypatch.setattr(cs_mod, "_KEN_BURNS_PASS_BUDGET_SECONDS", 1.5)
    monkeypatch.setattr(cs_mod, "_KEN_BURNS_PASS_OVERHEAD_SECONDS", 0.2)
    monkeypatch.setattr(ken_burns, "_MIN_PHASE_BUDGET_S", 0.05)

    async def _slow_cards(*_a: Any, **_k: Any) -> tuple[None, None]:
        await asyncio.sleep(1.0)  # a title card stuck behind the VP queue
        return (None, None)

    async def _stuck(job_id: str, *, timeout: float) -> dict[str, Any]:
        # A recording that never leaves the queue: each poll call runs to its own
        # timeout, exactly as the real client's poll_job does.
        await asyncio.sleep(timeout)
        raise TimeoutError(f"job {job_id} timed out after {timeout}s")

    async def _fetch(url: str, client: Any) -> tuple[bytes, str]:
        return (b"\x89PNG\r\n", "image/png")

    async def _slow_captions(**_k: Any) -> list[dict[str, Any]]:
        await asyncio.sleep(60)  # the ASR fallback on a busy transcription worker
        raise AssertionError("unreachable")  # pragma: no cover

    kb_vp = _vp()
    kb_vp.wait_for_content_recording = AsyncMock(side_effect=_stuck)
    monkeypatch.setattr(ken_burns, "VideoProductionClient", lambda: kb_vp)
    monkeypatch.setattr(ken_burns, "_fetch_own_asset", _fetch)
    monkeypatch.setattr(ken_burns, "_prescale_still", lambda d, c, _w, _h: (d, c))
    monkeypatch.setattr(ken_burns, "find_existing_media", AsyncMock(return_value=None))
    monkeypatch.setattr(cs_mod, "render_title_card_plan", _slow_cards)
    monkeypatch.setattr(cs_mod, "build_shot_captions", _slow_captions)
    monkeypatch.setattr(cs_mod, "TranscriptionWorkerClient", lambda: AsyncMock())

    class _Storage:
        async def resign_presigned_url(self, url: str) -> str:
            return url

    async def _get_storage() -> Any:
        return _Storage()

    monkeypatch.setattr(cs_mod, "_get_storage", _get_storage)
    monkeypatch.setattr(
        cs_mod,
        "get_settings",
        lambda: SimpleNamespace(
            pipeline=SimpleNamespace(
                captions_default_on_showcase=True,
                grade_default_on_showcase=False,
                captions_karaoke=False,
            )
        ),
    )
    captured: dict[str, Any] = {}

    async def _compile(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"artifact_url": "http://m/out.mp4", "duration_seconds": 2.0}

    compiler = AsyncMock()
    compiler.compile = AsyncMock(side_effect=_compile)
    ctx = _ctx({"deliverable": {"captions_mode": "burned_in"}})
    tts_storage.set_storage_for_tests(cast(Any, _Storage()))
    try:
        out = await _BudgetStage(compiler).handle(StageInput(), ctx=ctx)
    finally:
        tts_storage.set_storage_for_tests(None)

    assert captured, "the render never reached the compiler"
    assert not captured["shots"][0].get("video_url")  # the still shipped as a still
    assert captured.get("captions") is None
    assert any("ken_burns" in note for note in out.notes)
    # The subtitles were lost to the timeout, and the record says so.
    assert CAPTIONS_MISSING in [d["code"] for d in out.degradations]


def test_an_explicit_intro_and_outro_both_count_toward_the_order() -> None:
    """An operator's own title cards are fixed length; BOTH sides count, so 28s of
    narration with a 3s intro and a 3s outro for a 30s order (34s) is flagged."""
    cfg = _promo_cfg(title_card={"enabled": True, "intro": True, "outro": True})
    cfg["brief"]["target_duration_seconds"] = 30
    prior = _promo_prior()
    scale = 28.0 / sum(s for _i, _t, s in _LINES)
    for clip in prior["promo_narrate"]["clips"]:
        clip["seconds"] *= scale
    plan = fill.promo_fill(_ctx(cfg, prior))
    assert plan.end_card_seconds == pytest.approx(6.0)
    reason = plan.infeasible_reason
    assert reason is not None and "over the 33.0s ceiling" in reason


async def _run_captioned(
    monkeypatch: pytest.MonkeyPatch,
    stage_cls: type[CompileStageBase[Any, Any]],
    ctx: RunContext,
    captions_sleep: float,
) -> tuple[Any, dict[str, Any]]:
    """Drive ``stage_cls.handle`` through the caption phase with the kill-switch on.

    Title cards and Ken Burns are stubbed out (not under test here); the caption
    build sleeps ``captions_sleep`` — standing in for the ASR fallback on a busy
    transcription worker — then returns no cues.
    """
    import asyncio

    from src.services import tts_storage
    from src.services.stages.primitives import compile_stage as cs_mod

    async def _no_cards(*_a: Any, **_k: Any) -> tuple[None, None]:
        return (None, None)

    async def _no_motion(_ctx: Any, shots: list[dict[str, Any]], **_k: Any) -> list[dict[str, Any]]:
        return shots

    async def _captions(**_k: Any) -> None:
        await asyncio.sleep(captions_sleep)

    class _Storage:
        async def resign_presigned_url(self, url: str) -> str:
            return url

    async def _get_storage() -> Any:
        return _Storage()

    monkeypatch.setattr(cs_mod, "render_title_card_plan", _no_cards)
    monkeypatch.setattr(cs_mod, "apply_ken_burns", _no_motion)
    monkeypatch.setattr(cs_mod, "build_shot_captions", _captions)
    monkeypatch.setattr(cs_mod, "TranscriptionWorkerClient", lambda: AsyncMock())
    monkeypatch.setattr(cs_mod, "_get_storage", _get_storage)
    monkeypatch.setattr(
        cs_mod,
        "get_settings",
        lambda: SimpleNamespace(
            pipeline=SimpleNamespace(
                captions_default_on_showcase=True,
                grade_default_on_showcase=False,
                captions_karaoke=False,
            )
        ),
    )
    _library(monkeypatch, SoundLibrary.builtin())
    captured: dict[str, Any] = {}

    async def _compile(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"artifact_url": "http://m/out.mp4", "duration_seconds": 8.0}

    compiler = AsyncMock()
    compiler.compile = AsyncMock(side_effect=_compile)
    tts_storage.set_storage_for_tests(cast(Any, _Storage()))
    try:
        out = await stage_cls(compiler).handle(StageInput(), ctx=ctx)
    finally:
        tts_storage.set_storage_for_tests(None)
    return out, captured


@pytest.mark.asyncio
async def test_a_slow_caption_build_degrades_a_demo_video_instead_of_failing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-promo scope: the caption timeout lives in the shared compile base, so a
    real ``DemoVideoCompileStage`` takes the same path — the render reaches the
    compiler without subtitles and records CAPTIONS_MISSING, where it used to run
    into the aggregate deadline and fail."""
    from src.services.stages.demo_video.compile import DemoVideoCompileStage
    from src.services.stages.primitives import compile_stage as cs_mod

    monkeypatch.setattr(cs_mod, "_PRESUBMIT_DEADLINE_SECONDS", 2.0)
    monkeypatch.setattr(cs_mod, "_PRESUBMIT_FINAL_MARGIN_SECONDS", 0.5)
    ctx = _ctx(
        {"brief": {}, "deliverable": {"format": "16x9", "captions_mode": "burned_in"}},
        {
            "demo_video_gen": {
                "lines": [
                    {"id": "clip-1", "text": "Requester files a request.", "seconds": 4.0},
                    {"id": "clip-2", "text": "The GM approves it.", "seconds": 4.0},
                ]
            },
            "demo_video_narrate": {
                "clips": [
                    {
                        "id": "n1",
                        "line_id": "clip-1",
                        "seconds": 4.0,
                        "artifact_url": "http://a/1.wav",
                    },
                    {
                        "id": "n2",
                        "line_id": "clip-2",
                        "seconds": 4.0,
                        "artifact_url": "http://a/2.wav",
                    },
                ]
            },
            "demo_video_record": {
                "clips": [
                    {"line_id": "clip-1", "video_url": "http://v/1.mp4"},
                    {"line_id": "clip-2", "video_url": "http://v/2.mp4"},
                ]
            },
        },
    )
    assert DemoVideoCompileStage(client=AsyncMock()).caption_clip_refs(ctx)  # path is armed
    out, captured = await _run_captioned(monkeypatch, DemoVideoCompileStage, ctx, 60.0)
    assert captured, "the demo_video render never reached the compiler"
    assert captured.get("captions") is None
    assert CAPTIONS_MISSING in [d["code"] for d in out.degradations]
