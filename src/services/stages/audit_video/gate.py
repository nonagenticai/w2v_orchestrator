"""Opt-in gate for an audit type's narrated-video sub-chain.

Every scorecard audit (seo_audit / cro_audit / …) is scan-only by default: the
``<type>_scan`` stage's structured output IS the deliverable. The optional
narrated-video variant (``<type>_script`` → ``<type>_slides`` →
``<type>_narrate`` ‖ ``<type>_record`` → ``<type>_compile``) is gated behind a
single brief flag.

⚠️ Unlike ``shared/video_gate.video_requested`` (which defaults **True** for
website_review), audit video is **off by default** — a plain audit run must never
render a video. So this gate reads ``brief.get("generate_video", False)`` and
every video stage calls :func:`require_audit_video` at the top of ``handle``,
raising ``StageSkippedError`` (→ ``skipped``, a terminal state) when the flag is
absent/false. The scan-only run then completes on the scan output while the whole
video chain skips cleanly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.services.stages.errors import StageSkippedError

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext


def audit_video_requested(ctx: RunContext) -> bool:
    """True only when the run explicitly set ``brief.generate_video`` truthy.

    Defaults **False** — an audit is scan-only unless the operator opts into the
    narrated-video variant.
    """
    brief = ctx.config_snapshot.get("brief") or {}
    return bool(brief.get("generate_video", False))


def require_audit_video(ctx: RunContext) -> None:
    """Skip the calling stage unless the narrated-video variant was requested."""
    if not audit_video_requested(ctx):
        raise StageSkippedError(
            "audit video variant not requested (brief.generate_video is false)",
            missing=["brief.generate_video"],
        )
