"""``showcase_slides`` and ``shot_analysis`` call no LLM and no TTS.

They used to declare ``llm``/``tts`` only because they imported helpers from
LLM-calling modules (``demo_video.gen``, ``action_reel.segment_select``), which
the over-approximating slot scan counted. The requirements manifest then told
tenants these stages need an LLM/TTS provider. The helpers now live in
slot-free modules; ``test_stage_provider_slots`` keeps declared == detected.
"""

from __future__ import annotations

from src.services.stages.action_reel.shot_analysis import ShotAnalysisStage
from src.services.stages.demo_video.slides import ShowcaseSlidesStage
from tests.unit.test_stage_provider_slots import detected_slots


def test_showcase_slides_declares_and_reaches_no_slot() -> None:
    assert ShowcaseSlidesStage.provider_slots == ()
    assert detected_slots(ShowcaseSlidesStage) == set()


def test_shot_analysis_declares_and_reaches_no_slot() -> None:
    assert ShotAnalysisStage.provider_slots == ()
    assert detected_slots(ShotAnalysisStage) == set()
