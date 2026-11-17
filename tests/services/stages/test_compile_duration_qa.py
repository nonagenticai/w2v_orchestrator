"""Auto-QA on compile: declared-vs-actual duration guard (PR-K, orchestrator side)."""

from __future__ import annotations

from src.services.stages.base import StageOutput


class TestStageOutputNotesChannel:
    def test_plain_stageoutput_has_no_notes_field(self):
        # notes must NOT leak onto the plain marker (it mixes into element models
        # like DemoVideoLine whose field set must stay identical to a twin).
        assert "notes" not in StageOutput.model_fields
