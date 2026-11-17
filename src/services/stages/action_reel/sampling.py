"""Sampling constants shared by ``shot_analysis`` and ``segment_select``.

Its own module so ``shot_analysis`` (measure-only, no LLM) can size its frame
budget without importing ``segment_select`` — whose LLM client would otherwise
count against shot_analysis's ``provider_slots``.
"""

from __future__ import annotations

# How many motion samples to show per source. The full curve is one sample per
# analyzed frame — thousands of numbers, most of them redundant — and feeding it
# whole is how a small vLLM times out and silently returns nothing.
MOTION_SAMPLES_PER_SOURCE = 24
