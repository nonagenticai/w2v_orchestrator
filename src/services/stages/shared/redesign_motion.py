"""Opt-in animated hero clip for redesign / web_design outputs.

Phase 4 of the motion-graphics feature. A redesign / web_design run produces a
static screenshot of the generated page; when ``brief.redesign_motion`` is set,
this also captures a **deterministic scroll-through clip** of the generated HTML
— a smooth top-to-bottom pan — and attaches it to the hero visual so the
run-view can play the redesign in motion. The still PNG always remains (this is
additive + soft-fail), and the motion is deterministic (a frame-stepped scroll,
not LLM-authored animation), so quality does not depend on the generator.

The clip is produced by feeding the generated HTML to video_production's
``/api/record/animation`` with an injected ``window.render(t)`` that scrolls the
page — reusing the same frame-step renderer the motion templates use, so the
compiler/run-view need no new concept (it is just another ``video_url``).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING
from uuid import uuid7

from src.services.stages.shared.motion_slides import format_dims
from src.services.stages.upstream import PermanentUpstreamError, guarded
from src.services.tts_storage import store_generated_video
from src.utils import coerce_float

if TYPE_CHECKING:
    from src.clients.design_client import DesignClient
    from src.clients.video_production_client import VideoProductionClient

    # Imported under TYPE_CHECKING only: ``website_review.redesign`` imports this
    # module (``attach_motion_hero``) at runtime, so a runtime import here would
    # be a hard cycle.
    from src.services.stages.website_review.redesign import RedesignVisual

log = logging.getLogger(__name__)

_RECORD_TIMEOUT_S = 600.0
_SCROLL_DURATION_MS = 8000

# A deterministic scroll-through: render(t) eases the scroll from top to bottom
# over the timeline, so the frame-stepper captures a smooth pan of the page
# regardless of whether the generated HTML carries any CSS animation.
_SCROLL_SCRIPT = f"""
<script>
window.RENDER_DURATION_MS = {_SCROLL_DURATION_MS};
window.render = function (t) {{
  var doc = document.documentElement;
  var pageH = Math.max(
    document.body ? document.body.scrollHeight : 0,
    doc ? doc.scrollHeight : 0
  );
  var max = Math.max(0, pageH - window.innerHeight);
  var p = window.RENDER_DURATION_MS > 0 ? Math.min(1, t / window.RENDER_DURATION_MS) : 1;
  var eased = p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;  // ease in/out
  window.scrollTo(0, eased * max);
}};
</script>
"""


def _inject_scroll(html: str) -> str:
    """Append the deterministic scroll-through driver to the generated HTML."""
    if "</body>" in html:
        return html.replace("</body>", _SCROLL_SCRIPT + "</body>", 1)
    return html + _SCROLL_SCRIPT


def _hero_index(visuals: list[RedesignVisual]) -> int:
    """Pick the visual to animate: the first 'after'/redesign shot, else the first.

    'before'/'original' roles are the source site — the motion belongs on the
    redesign hero. Falls back to index 0 when no role clearly marks the after."""
    for i, v in enumerate(visuals):
        role = str(getattr(v, "role", "") or "").lower()
        if not role.startswith(("before", "original")):
            return i
    return 0


async def attach_motion_hero(
    visuals: list[RedesignVisual],
    *,
    design_client: DesignClient,
    video_client: VideoProductionClient,
    job_id: str,
    brief: Mapping[str, object],
    deliverable: Mapping[str, object],
    run_id: str,
    stage_name: str,
    tier: str = "1080p",
) -> list[RedesignVisual]:
    """Return ``visuals`` with an animated scroll-through clip on the hero visual.

    Opt-in via ``brief.redesign_motion``; a no-op (returns ``visuals`` unchanged)
    when disabled, when there are no visuals, or — soft-fail — when the capture
    fails for any reason (the still PNGs always remain)."""
    if not brief.get("redesign_motion") or not visuals:
        return visuals
    try:
        html = await guarded(lambda: design_client.get_design_html(job_id))
        if not html or "<" not in html:
            return visuals
        fmt = deliverable.get("format")
        width, height = format_dims(fmt if isinstance(fmt, str) else None, tier)
        anim_job = await video_client.record_animation(
            html=_inject_scroll(html),
            part_id=f"redesign-hero-{job_id}",
            fps=30,
            width=width,
            height=height,
            duration_ms=_SCROLL_DURATION_MS,
            output_format="mp4",
        )
        result = await video_client.wait_for_content_recording(anim_job, timeout=_RECORD_TIMEOUT_S)
        video_bytes, _ctype, _disp = await video_client.get_recording_video(anim_job)
        video_url = await store_generated_video(
            run_id=run_id,
            stage_name=stage_name,
            video_id=str(uuid7()),
            video_bytes=video_bytes,
            video_format="mp4",
        )
    except PermanentUpstreamError as exc:
        # Still a soft-fail — the motion hero is opt-in and the still PNGs remain,
        # so killing the run over it would be a worse bug than the one being fixed.
        # But a permanent upstream failure here is a CONFIGURATION fault (an expired
        # service licence surfaces as html_render 401), not a blip: it will recur on
        # every run until someone acts, and at WARNING alongside genuine transients
        # it was indistinguishable from noise. ERROR is the difference between
        # "swallowed" and "actionable".
        log.error(
            "%s: redesign motion-hero capture failed permanently — needs operator "
            "action, not a retry: %s",
            stage_name,
            exc,
        )
        return visuals
    except Exception as exc:  # noqa: BLE001 -- soft-fail: keep the static visuals
        log.warning("%s: redesign motion-hero capture failed: %s", stage_name, exc)
        return visuals

    duration = result.get("duration_seconds") if isinstance(result, dict) else None
    idx = _hero_index(visuals)
    visuals[idx] = visuals[idx].model_copy(
        update={"video_url": video_url, "video_duration_s": coerce_float(duration)}
    )
    return visuals
