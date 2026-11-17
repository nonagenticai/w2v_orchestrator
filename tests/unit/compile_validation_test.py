"""PR-B: compile-response validation + degenerate-input refusal.

`validate_compile_result` rejects a compositor 200 that did not actually
produce a usable artifact (empty/"None" url, or a zero/non-finite duration —
the 1-frame-placeholder shape), and `StagePreconditionError` is the typed
"there was nothing to compile" signal raised before the compile call.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.services.stages.base import (
    StageDownstreamResponseError,
    StageError,
    StagePreconditionError,
    validate_compile_result,
)


def _ok(**over: Any) -> dict[str, Any]:
    base = {"artifact_url": "https://minio.test/b/final.mp4", "duration_seconds": 12.0}
    base.update(over)
    return base


@pytest.mark.parametrize("bad_url", [""])
def test_missing_or_placeholder_url_rejected(bad_url: Any) -> None:
    with pytest.raises(StageDownstreamResponseError, match="artifact_url"):
        validate_compile_result(_ok(artifact_url=bad_url), stage="video_render")


@pytest.mark.parametrize("bad_dur", [0])
def test_nonpositive_or_nonfinite_duration_rejected(bad_dur: Any) -> None:
    with pytest.raises(StageDownstreamResponseError, match="duration_seconds"):
        validate_compile_result(_ok(duration_seconds=bad_dur), stage="video_render")


def test_error_message_names_the_stage() -> None:
    with pytest.raises(StageDownstreamResponseError, match="course_stitch"):
        validate_compile_result(_ok(artifact_url=""), stage="course_stitch")


def test_precondition_not_met_carries_reason_and_missing() -> None:
    exc = StagePreconditionError("video_render: no shots to compile", missing=["shots"])
    assert exc.reason == "video_render: no shots to compile"
    assert exc.missing == ["shots"]
    # Both new exceptions are StageErrors so the dispatcher's stage-error path owns them.
    assert isinstance(exc, StageError)
    assert isinstance(
        StageDownstreamResponseError("x"),
        StageError,
    )
