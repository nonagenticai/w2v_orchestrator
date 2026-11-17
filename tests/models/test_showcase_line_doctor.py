"""C6 script-doctor per-line override fields on the shared authored-line model.

``AuthoredLine`` (src.models.showcase_line) gained three optional overrides —
``locked`` / ``beat_role`` / ``doctor_note`` — that the script doctor (ORCH-1.2)
writes per line. They inherit onto both variants (``api.schemas.ShowcaseLine`` and
``stages.demo_video.gen.DemoVideoLine``). These tests lock the defaults (absent ⇒
no behavior change) and that they survive a pure-Pydantic + JSON round-trip with no
migration.
"""

from __future__ import annotations

import json

from src.api.schemas import ShowcaseLine
from src.models import AuthoredLine
from src.services.stages.demo_video.gen import DemoVideoLine

_DOCTOR_FIELDS = {"locked", "beat_role", "doctor_note"}


class TestDoctorFieldsExistOnAllVariants:
    def test_all_three_expose_the_new_fields(self) -> None:
        assert set(AuthoredLine.model_fields) >= _DOCTOR_FIELDS
        assert set(ShowcaseLine.model_fields) >= _DOCTOR_FIELDS
        assert set(DemoVideoLine.model_fields) >= _DOCTOR_FIELDS


class TestDoctorFieldDefaults:
    def test_defaults_when_omitted(self) -> None:
        line = AuthoredLine()
        assert line.locked is False
        assert line.beat_role is None
        assert line.doctor_note is None

    def test_defaults_on_stage_twin(self) -> None:
        line = DemoVideoLine(id="clip-1", text="hi", seconds=1.0)
        assert line.locked is False
        assert line.beat_role is None
        assert line.doctor_note is None

    def test_defaults_on_api_door(self) -> None:
        line = ShowcaseLine.model_validate({"text": "hi"})
        assert line.locked is False
        assert line.beat_role is None
        assert line.doctor_note is None


class TestDoctorFieldRoundTrip:
    def test_authored_line_model_dump_revalidate_identical(self) -> None:
        line = AuthoredLine(
            id="clip-1",
            text="hook line",
            locked=True,
            beat_role="hook",
            doctor_note="tighten the open",
        )
        dumped = line.model_dump()
        assert dumped["locked"] is True
        assert dumped["beat_role"] == "hook"
        assert dumped["doctor_note"] == "tighten the open"
        assert AuthoredLine.model_validate(dumped) == line

    def test_showcase_line_json_round_trip_preserves_fields(self) -> None:
        line = ShowcaseLine.model_validate(
            {
                "text": "payoff line",
                "locked": True,
                "beat_role": "payoff",
                "doctor_note": "land the punchline",
            }
        )
        reloaded = ShowcaseLine.model_validate(json.loads(line.model_dump_json()))
        assert reloaded == line
        assert reloaded.locked is True
        assert reloaded.beat_role == "payoff"
        assert reloaded.doctor_note == "land the punchline"

    def test_demo_video_line_json_round_trip_preserves_fields(self) -> None:
        line = DemoVideoLine(
            id="clip-2",
            text="",
            seconds=0.0,
            locked=True,
            beat_role="bridge",
            doctor_note="cover the cut",
        )
        reloaded = DemoVideoLine.model_validate(json.loads(line.model_dump_json()))
        assert reloaded == line
        assert reloaded.locked is True
        assert reloaded.beat_role == "bridge"
        assert reloaded.doctor_note == "cover the cut"
