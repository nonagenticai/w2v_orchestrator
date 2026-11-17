"""Tests for the shared job-config whitelist used by the AI-generation stages.

The point of the shared base whitelist is anti-drift: the parameters common to
both ``ai_image_gen`` and ``ai_video_gen`` are declared ONCE, so a common key
can no longer survive in one stage's job_config while being silently dropped by
the other. These tests lock in the exact per-stage key sets (proving the
refactor changed no behavior) and assert every BASE key survives for BOTH
stages.
"""

from __future__ import annotations

from src.services.stages.generic._job_config import BASE_JOB_CONFIG_KEYS
from src.services.stages.generic.ai_image_gen import AiImageGenStage
from src.services.stages.generic.ai_video_gen import AiVideoGenStage

# The full whitelist each stage emitted before the shared-base refactor. These
# are the behavior contract: build_job_config must reproduce exactly these key
# sets for a fully-populated cfg.
_IMAGE_KEYS = {
    "provider",
    "style",
    "negative_prompt",
    "extra_params",
    "size",
    "quality",
    "reference_image_urls",
    "target_usage",
    "usage_context",
}
_VIDEO_KEYS = {
    "provider",
    "style",
    "negative_prompt",
    "extra_params",
    "duration_s",
    "width",
    "height",
    "image_url",
    "fallback_providers",
    "budget_limit",
    "seed",
}


def _full_cfg(keys: set[str]) -> dict[str, object]:
    # A non-None sentinel per key so nothing is dropped by the None-filter.
    return {key: f"val-{key}" for key in keys}


def test_image_stage_emits_exact_key_set() -> None:
    cfg = _full_cfg(_IMAGE_KEYS)
    out = AiImageGenStage._build_job_config(cfg)
    assert set(out) == _IMAGE_KEYS


def test_video_stage_emits_exact_key_set() -> None:
    cfg = _full_cfg(_VIDEO_KEYS)
    out = AiVideoGenStage._build_job_config(cfg)
    assert set(out) == _VIDEO_KEYS


def test_base_keys_are_the_shared_intersection() -> None:
    # The shared base is exactly the keys common to both stage whitelists —
    # nothing more, nothing less. This guards against a stage-specific key
    # sneaking into the base (which would leak it into the other stage).
    assert set(BASE_JOB_CONFIG_KEYS) == _IMAGE_KEYS & _VIDEO_KEYS


def test_none_valued_keys_are_dropped() -> None:
    cfg = {"provider": None, "style": "editorial", "size": None, "quality": "hd"}
    image_out = AiImageGenStage._build_job_config(cfg)
    assert image_out == {"style": "editorial", "quality": "hd"}

    cfg_v = {"provider": None, "duration_s": 5.0, "width": None}
    video_out = AiVideoGenStage._build_job_config(cfg_v)
    assert video_out == {"duration_s": 5.0}


def test_unknown_keys_are_not_forwarded() -> None:
    cfg = {"provider": "p", "prompt": "not-config", "secret": "x"}
    assert AiImageGenStage._build_job_config(cfg) == {"provider": "p"}
    assert AiVideoGenStage._build_job_config(cfg) == {"provider": "p"}
