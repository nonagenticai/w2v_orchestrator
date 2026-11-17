"""The closing hold of per-line promo run 01a11295 (2026-10-06, REJECTED at frame QA).

A 60 s order, seven verbatim lines narrated in 35.5 s. Since #1074 a per-line
film absorbs the default end card into the CTA line's own card, and the fill
stretched that card to its 8.0 s maximum to reach the order: the narration ended
at 47.1 s and the closing card held on for 10.5 s with music only (18% of the
film). Rule now, per-line promos only: the closing shot ends at most 3.0 s after
the last narration and the CTA stays on screen at least 2.5 s; the fill never
pads the end card toward the order, and a film the holds cannot bring into the
band is recorded on the render's ``duration_conformance`` as ``plan_infeasible``
(with ``duration_out_of_tolerance`` when the delivery misses the band) instead.

Driven through a real ``PromoRenderStage.run`` after the real motion brief and
card recordings (only video_production, storage and the compiler are faked).
"""

from __future__ import annotations

from typing import Any

import pytest

from src.services.runs.run_context import RunContext
from src.services.stages.promo.fill import (
    CLOSING_TAIL_MAX_SECONDS,
    CLOSING_TAIL_PLANNED_SECONDS,
    CTA_MIN_SECONDS,
    plan_promo_fill,
)
from src.services.stages.shared.degradation import DURATION_OUT_OF_TOLERANCE
from tests.services.stages.promo.test_motion_brief import _brief_and_cards
from tests.services.stages.promo.test_promo_frame_qa_2026_10_05 import (
    _LINES,
    _promo_cfg,
    _render,
)

pytestmark = pytest.mark.usefixtures("_gen_storage")

_NARRATED = {line_id: seconds for line_id, _text, seconds in _LINES}
_HOLD = 2.0  # the fill's pause after each line of this 60 s order


@pytest.fixture
def _gen_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import AsyncMock

    gen = "src.services.stages.generic.motion_graphics_gen"
    monkeypatch.setattr(f"{gen}.find_existing_media", AsyncMock(return_value=None))
    monkeypatch.setattr(
        f"{gen}.store_generated_video",
        AsyncMock(side_effect=lambda **kw: f"https://minio.test/v/{kw['video_id']}.mp4"),
    )


async def _per_line_render(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, list[dict[str, Any]], Any]:
    async def _prepare(ctx: RunContext) -> None:
        await _brief_and_cards(ctx, [])

    planned = sum(_NARRATED.values()) + 7 * _HOLD + (CLOSING_TAIL_PLANNED_SECONDS - _HOLD)
    # The compiler delivers what it was sent: the planned film, not the order.
    out, captured, title_vp = await _render(
        monkeypatch, _promo_cfg(direction={"motion": "per_line"}), planned, prepare=_prepare
    )
    return out, captured["shots"], title_vp


async def test_the_closing_card_ends_three_seconds_after_the_last_word(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _out, shots, title_vp = await _per_line_render(monkeypatch)

    assert [shot["line_id"] for shot in shots] == [i for i, _t, _s in _LINES]
    title_vp.record_animation.assert_not_awaited()  # the closing card IS the end card
    closing = shots[-1]
    tail = float(closing["seconds"]) - _NARRATED["l7"]
    assert tail <= CLOSING_TAIL_MAX_SECONDS + 1e-9  # was 10.0 s (2.0 hold + 8.0 card)
    # planned one 24 fps frame (+1 ms) under the cap, so the compiler's frame
    # rounding cannot push the delivered tail over it (run 01a11511: 3.04 s)
    assert tail == pytest.approx(CLOSING_TAIL_PLANNED_SECONDS)
    assert float(closing["seconds"]) >= CTA_MIN_SECONDS
    assert closing["max_seconds"] == pytest.approx(closing["seconds"])  # the compiler honours it


async def test_the_film_is_not_padded_to_the_order_and_the_shortfall_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, shots, _title_vp = await _per_line_render(monkeypatch)

    total = sum(float(shot["seconds"]) for shot in shots)
    # narration + the ordinary per-line holds + what is left of the closing tail.
    assert total == pytest.approx(
        sum(_NARRATED.values()) + 7 * _HOLD + CLOSING_TAIL_PLANNED_SECONDS - _HOLD
    )
    assert total < 54.0  # under the 60 s order's -10% floor: nothing padded it in
    conformance = out.duration_conformance
    assert conformance is not None
    assert conformance["plan_infeasible"] is True
    reason = conformance["plan_infeasible_reason"] or ""
    assert "never padded" in reason and "60s ordered" in reason
    assert DURATION_OUT_OF_TOLERANCE in {entry["code"] for entry in out.degradations}


def test_a_short_cta_keeps_its_floor_inside_the_tail() -> None:
    """A 0.4 s CTA with no pauses: on screen 2.5 s+, still over by <= 3.0 s."""
    plan = plan_promo_fill(
        None, [10.0, 10.0, 0.4], end_card=True, closing_tail_max=CLOSING_TAIL_MAX_SECONDS
    )
    closing = 0.4 + plan.hold_seconds + plan.cta_floor_seconds + plan.end_card_seconds

    assert closing >= CTA_MIN_SECONDS
    assert closing - 0.4 == pytest.approx(CLOSING_TAIL_PLANNED_SECONDS)


def test_every_other_promo_keeps_the_stretched_end_card() -> None:
    """Without per-line motion the separate end card still takes the shortfall
    first (8.0 s here), exactly as before."""
    narrated = [s for _i, _t, s in _LINES]

    assert plan_promo_fill(60.0, narrated, end_card=True).end_card_seconds == 8.0
    capped = plan_promo_fill(
        60.0, narrated, end_card=True, closing_tail_max=CLOSING_TAIL_MAX_SECONDS
    )
    assert capped.end_card_seconds == pytest.approx(CLOSING_TAIL_PLANNED_SECONDS - _HOLD)
    assert capped.hold_seconds == _HOLD  # the ordinary per-line holds are unchanged


def test_on_a_per_line_film_the_shortfall_goes_to_the_holds_not_the_card() -> None:
    """30 s ordered, 22 s narrated over 7 lines: the 5 s left after the base card
    are pauses (0.71 s each), and the card is what remains of the closing tail —
    not an 8.0 s card that eats the shortfall and leaves no pauses."""
    plan = plan_promo_fill(
        30.0, [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0], end_card=True, closing_tail_max=3.0
    )

    assert plan.hold_seconds == pytest.approx(5.0 / 7, abs=1e-3)
    assert plan.end_card_seconds == pytest.approx(CLOSING_TAIL_PLANNED_SECONDS - 5.0 / 7, abs=1e-3)
