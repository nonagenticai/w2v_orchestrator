"""action_reel stage handlers.

source_ingest -> footage_probe -> shot_analysis -> segment_select ->
cut_plan (joined with music_select) -> graphics_design / narrate -> compile.

The product: the operator uploads raw camera/drone footage, an LLM picks the
segments worth keeping, and those are cut to a music beat with per-clip
time-warp. Every stage in this package emits a REVIEWABLE artifact — which is
why so many of them pause (see ``services/blueprints/action_reel.py``).
"""
