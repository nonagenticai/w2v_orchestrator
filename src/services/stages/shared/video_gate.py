"""Per-output opt-in gates for the website_review pipeline.

A website_review run produces up to three independent artifacts, each toggled by
a brief flag:

* ``generate_critique`` → the critique (findings + scores + PDF). Absent = True.
* ``generate_redesign`` → the redesigned HTML + before/after visuals. Absent =
  False, BUT implied by video (the narrated review IS the before→after of the
  redesign, so a video always needs one).
* ``generate_video``    → the narrated review video (script → TTS → compile →
  growth bundle). Absent = True.

The flags are independent except video⟶redesign. Absent-defaults reproduce the
original full pipeline, so existing projects and the post-promote E2E are
unchanged. Each stage calls the matching ``require_*`` at the top of ``handle``
and skips itself (``StageSkippedError`` → terminal-but-fine) when not requested.

(Filename kept as ``video_gate`` for import stability; it now gates all three
outputs, not just the video.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.services.stages.errors import StageSkippedError

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext


def _brief(ctx: RunContext) -> dict:
    return ctx.config_snapshot.get("brief") or {}


def critique_requested(ctx: RunContext) -> bool:
    """True unless the run explicitly set ``brief.generate_critique`` to False."""
    return bool(_brief(ctx).get("generate_critique", True))


def video_requested(ctx: RunContext) -> bool:
    """True unless the run explicitly set ``brief.generate_video`` to False."""
    return bool(_brief(ctx).get("generate_video", True))


def redesign_requested(ctx: RunContext) -> bool:
    """True when a redesign is asked for directly, or implied by the video.

    ``generate_redesign`` defaults to False (redesign is opt-in on its own), but
    a video always needs the redesign to show, so ``generate_video`` forces it on.
    """
    return bool(_brief(ctx).get("generate_redesign", False)) or video_requested(ctx)


def require_critique(ctx: RunContext) -> None:
    if not critique_requested(ctx):
        raise StageSkippedError(
            "critique not requested (brief.generate_critique is false)",
            missing=["brief.generate_critique"],
        )


def require_redesign(ctx: RunContext) -> None:
    if not redesign_requested(ctx):
        raise StageSkippedError(
            "redesign not requested (brief.generate_redesign / generate_video both false)",
            missing=["brief.generate_redesign"],
        )


def require_video(ctx: RunContext) -> None:
    if not video_requested(ctx):
        raise StageSkippedError(
            "video deliverable not requested (brief.generate_video is false)",
            missing=["brief.generate_video"],
        )
