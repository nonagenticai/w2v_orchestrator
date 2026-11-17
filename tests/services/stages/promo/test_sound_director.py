"""promo_sound_director — LLM accent picking with authoritative validation."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

from src.services.runs.run_context import RunContext
from src.services.stages.promo.sound_director import PromoSoundDirectorStage
from src.services.stages.sound_catalog import SoundLibrary


@pytest.fixture(autouse=True)
def _builtin_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate against the built-in library, hermetic from the storage singleton."""

    async def _fake(*_args: Any, **_kwargs: Any) -> SoundLibrary:
        return SoundLibrary.builtin()

    monkeypatch.setattr("src.services.stages.shared.sound_director.load_sound_library", _fake)


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


def _ctx(*, enabled: bool = True, accents: bool | None = None) -> RunContext:
    sound: dict[str, Any] = {}
    if enabled:
        sound["enabled"] = True
    if accents is not None:
        sound["accents"] = accents
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"sound_effects": sound} if sound else {},
        prior_run_outputs={
            "promo_script": {
                "lines": [
                    {"id": "l1", "text": "Meet our new dashboard.", "seconds": 4.0},
                    {"id": "l2", "text": "One click and you're done.", "seconds": 3.0},
                ],
                "total_seconds": 7.0,
            }
        },
    )


async def test_valid_cues_pass_through() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {"cues": [{"line_id": "l2", "sfx_id": "ding_positive", "at_seconds": 1.0, "gain_db": -6.0}]}
    )
    out = await PromoSoundDirectorStage(client=fake).run({}, ctx=_ctx())
    assert len(out.cues) == 1
    assert out.cues[0].line_id == "l2"
    assert out.cues[0].sfx_id == "ding_positive"
    assert out.cues[0].at_seconds == 1.0


async def test_invalid_id_and_unknown_line_dropped() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "cues": [
                {
                    "line_id": "l1",
                    "sfx_id": "corporate_uplift",
                    "at_seconds": 1.0,
                },  # music, not accent
                {"line_id": "l1", "sfx_id": "not_a_sound", "at_seconds": 1.0},  # unknown id
                {"line_id": "nope", "sfx_id": "click_ui", "at_seconds": 0.5},  # unknown line
                {"line_id": "l1", "sfx_id": "click_ui", "at_seconds": 2.0},  # valid
            ]
        }
    )
    out = await PromoSoundDirectorStage(client=fake).run({}, ctx=_ctx())
    assert [(c.line_id, c.sfx_id) for c in out.cues] == [("l1", "click_ui")]


async def test_offset_clamped_into_line_and_capped_per_line() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "cues": [
                {"line_id": "l1", "sfx_id": "click_ui", "at_seconds": 99.0},  # oob → midpoint
                {"line_id": "l1", "sfx_id": "ding_positive", "at_seconds": 1.0},
                {
                    "line_id": "l1",
                    "sfx_id": "chime_soft",
                    "at_seconds": 2.0,
                },  # 3rd → dropped (cap 2)
            ]
        }
    )
    out = await PromoSoundDirectorStage(client=fake).run({}, ctx=_ctx())
    l1 = [c for c in out.cues if c.line_id == "l1"]
    assert len(l1) == 2  # capped at _MAX_ACCENTS_PER_LINE
    # The out-of-range offset is repaired to the line midpoint (4.0s line → 2.0).
    assert l1[0].at_seconds == 2.0
