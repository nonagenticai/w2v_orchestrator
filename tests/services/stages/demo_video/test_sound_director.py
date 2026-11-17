"""demo_video_sound_director — accent picking over gen text + narrate durations."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

from src.services.runs.run_context import RunContext
from src.services.stages.base import StageSkippedError
from src.services.stages.demo_video.sound_director import DemoVideoSoundDirectorStage
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
            # Authored lines carry seconds=0 (timing comes from narration).
            "demo_video_gen": {
                "lines": [
                    {"id": "b1", "text": "Meet our new dashboard.", "seconds": 0.0},
                    {"id": "b2", "text": "Eleven rooms detected.", "seconds": 0.0},
                ]
            },
            # Real per-line durations from narration.
            "demo_video_narrate": {
                "clips": [
                    {"id": "n1", "line_id": "b1", "seconds": 4.0},
                    {"id": "n2", "line_id": "b2", "seconds": 3.0},
                ]
            },
        },
    )


async def test_lines_split_into_bounded_batches() -> None:
    """More narration lines than the batch size ⇒ multiple bounded WM calls, each
    carrying ≤ _CUE_BATCH_SIZE lines. A single unbounded ``cues`` call is the
    small-vLLM timeout / silent-degrade bug this split fixes."""
    from src.services.stages.shared.sound_director import _CUE_BATCH_SIZE

    n = _CUE_BATCH_SIZE * 2 + 1  # 3 batches: sizes 5, 5, 1
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"sound_effects": {"enabled": True, "accents": True}},
        prior_run_outputs={
            "demo_video_gen": {
                "lines": [{"id": f"b{i}", "text": f"Line {i}.", "seconds": 0.0} for i in range(n)]
            },
            "demo_video_narrate": {
                "clips": [{"id": f"n{i}", "line_id": f"b{i}", "seconds": 3.0} for i in range(n)]
            },
        },
    )

    per_call_counts: list[int] = []

    def _batch_resp(*_: Any, **kwargs: Any) -> WMResponse:
        msg = kwargs["message"]
        ids = [i for i in range(n) if f'"line_id":"b{i}"' in msg]
        per_call_counts.append(len(ids))
        return _resp({"cues": []})

    fake = AsyncMock()
    fake.message.side_effect = _batch_resp
    await DemoVideoSoundDirectorStage(client=fake).run({}, ctx=ctx)

    assert fake.message.call_count == 3  # ceil(11 / 5)
    assert per_call_counts == [5, 5, 1]  # each WM call bounded, nothing dropped


async def test_skipped_when_not_opted_in() -> None:
    with pytest.raises(StageSkippedError):
        await DemoVideoSoundDirectorStage(client=AsyncMock()).run({}, ctx=_ctx(enabled=False))


async def test_skipped_when_accents_disabled() -> None:
    with pytest.raises(StageSkippedError):
        await DemoVideoSoundDirectorStage(client=AsyncMock()).run({}, ctx=_ctx(accents=False))


async def test_no_redrive_feedback_message_is_inert() -> None:
    """Without redrive_feedback the built message carries no critic block — the
    normal DAG run is byte-identical to today (feature inert by default)."""
    fake = AsyncMock()
    fake.message.return_value = _resp({"cues": []})
    await DemoVideoSoundDirectorStage(client=fake).run({}, ctx=_ctx())
    message = fake.message.call_args.kwargs["message"]
    assert "CRITIC FEEDBACK" not in message

    # Explicit None threads identically to omitting the field entirely.
    fake_none = AsyncMock()
    fake_none.message.return_value = _resp({"cues": []})
    await DemoVideoSoundDirectorStage(client=fake_none).run({"redrive_feedback": None}, ctx=_ctx())
    assert fake_none.message.call_args.kwargs["message"] == message


async def test_redrive_feedback_appended_to_message() -> None:
    """A non-empty redrive_feedback appends the critic block verbatim so a holistic
    critic stage can re-drive the director from the seams it found."""
    fake = AsyncMock()
    fake.message.return_value = _resp({"cues": []})
    await DemoVideoSoundDirectorStage(client=fake).run(
        {"redrive_feedback": "continuity break at clip-3"}, ctx=_ctx()
    )
    message = fake.message.call_args.kwargs["message"]
    assert "CRITIC FEEDBACK" in message
    assert "continuity break at clip-3" in message


async def test_uses_narrate_durations_and_validates_ids() -> None:
    fake = AsyncMock()
    # b2's real narrated duration is 3.0 → an offset of 99 clamps to the midpoint
    # (1.5), proving the director scored against narrate seconds, not gen's 0.0.
    fake.message.return_value = _resp(
        {
            "cues": [
                {"line_id": "b2", "sfx_id": "ding_positive", "at_seconds": 99.0},
                {"line_id": "b1", "sfx_id": "not_a_sound", "at_seconds": 1.0},  # dropped
                {"line_id": "nope", "sfx_id": "click_ui", "at_seconds": 0.5},  # unknown line
            ]
        }
    )
    out = await DemoVideoSoundDirectorStage(client=fake).run({}, ctx=_ctx())
    assert [(c.line_id, c.sfx_id) for c in out.cues] == [("b2", "ding_positive")]
    assert out.cues[0].at_seconds == 1.5  # 3.0s line midpoint (oob repair)
