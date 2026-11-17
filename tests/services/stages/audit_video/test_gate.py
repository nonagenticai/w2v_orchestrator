from uuid import uuid4

import pytest

from src.services.runs.run_context import RunContext
from src.services.stages.audit_video.gate import audit_video_requested, require_audit_video
from src.services.stages.errors import StageSkippedError


def _ctx(brief: dict | None = None) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": brief} if brief is not None else {},
    )


def test_defaults_off() -> None:
    # No brief, empty brief, and explicit false all read as "video not requested".
    assert audit_video_requested(_ctx()) is False
    assert audit_video_requested(_ctx({})) is False
    assert audit_video_requested(_ctx({"generate_video": False})) is False


def test_on_when_flag_truthy() -> None:
    assert audit_video_requested(_ctx({"generate_video": True})) is True


def test_require_raises_when_off() -> None:
    with pytest.raises(StageSkippedError, match="generate_video"):
        require_audit_video(_ctx())
