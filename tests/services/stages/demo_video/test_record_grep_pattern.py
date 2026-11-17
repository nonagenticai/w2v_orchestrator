"""``demo_video_record`` must not pay FTR to record+narrate a scenario
``demo_video_gen`` already trimmed away.

``gen.lines`` is read AFTER `_apply_target_beats` / `_apply_duration_ceiling` /
`_apply_feasibility_floor` have already dropped beats, but the bdd package this
stage sends to FTR still contains every scenario the ORIGINAL (larger) request
asked bdd to generate. Without a `--grep` filter, FTR downloads, runs, records,
and mux-discards every one of those extra scenarios before `record.py`'s own
`valid_clip_tags` filter (which runs only after the harvest) throws them away —
that is the recording waste this change closes.

`_grep_pattern_for_kept_clips` is tested directly for the anchoring hazard
(`@clip-1` must not substring-match `@clip-10`) and via the real stage to prove
the built payload only selects the KEPT beats' clip tags.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.services.stages.demo_video.record import (
    DemoVideoRecordStage,
)

from .test_stages import _ctx, _passing_analyzer, _qa_settings


def _ftr_client(clips: list[dict[str, Any]]) -> AsyncMock:
    client = AsyncMock()
    client.start_execution.return_value = "ftr-grep"
    client.wait_for_execution.return_value = {}
    client.get_job_result.return_value = {
        "scenario_clips": clips,
        "summary": {"passed": len(clips), "failed": 0},
    }
    client.get_clip_video.return_value = b"mp4"
    return client


@pytest.mark.asyncio
async def test_five_generated_three_kept_pattern_selects_exactly_those_three():
    """The task's own headline scenario: bdd generated 5 scenarios
    (clip-1..clip-5), but demo_video_gen's trimming already kept only 3 lines
    (clip-1, clip-3, clip-5) by the time `record.py` builds the FTR request.
    The sent `grep_pattern` must select exactly those 3 clip tags."""
    kept_clips = [
        {"clip_key": "clip-1", "duration_ms": 1000},
        {"clip_key": "clip-3", "duration_ms": 1000},
        {"clip_key": "clip-5", "duration_ms": 1000},
    ]
    client = _ftr_client(kept_clips)
    gen = {
        "bdd_job_id": "bdd-grep",
        # Only the KEPT lines reach demo_video_record — this is the
        # post-trim spine, not the 5 scenarios bdd originally generated.
        "lines": [{"id": c["clip_key"]} for c in kept_clips],
    }
    stage = DemoVideoRecordStage(client=client, analyzer=_passing_analyzer())
    with (
        patch(
            "src.services.stages.demo_video.record.get_settings",
            return_value=_qa_settings(),
        ),
        patch(
            "src.services.stages.demo_video.record.store_capture_clip",
            new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['clip_id']}.mp4"),
        ),
    ):
        await asyncio.wait_for(
            stage.handle(
                stage.input_model(),
                _ctx(
                    {"target_url": "https://app.example"},
                    prior={"demo_video_gen": gen},
                ),
            ),
            timeout=30.0,
        )

    assert client.start_execution.await_count == 1
    payload = client.start_execution.await_args.args[0]
    grep_pattern = str(payload.get("grep_pattern") or "")
    assert grep_pattern, f"no grep_pattern sent: payload={payload}"

    compiled = re.compile(grep_pattern)
    for kept in ("clip-1", "clip-3", "clip-5"):
        assert compiled.search(f"@{kept} some scenario"), (
            f"kept clip {kept} not selected by {grep_pattern!r}"
        )
    for dropped in ("clip-2", "clip-4"):
        assert not compiled.search(f"@{dropped} some scenario"), (
            f"dropped clip {dropped} WAS selected by {grep_pattern!r}"
        )
