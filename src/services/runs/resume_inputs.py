"""What a run's stages read LIVE, and the fingerprint that makes a resume safe.

A bare Restart reuses the parent's completed stages that no change affects
(``services/runs/restart_plan.py``). The config snapshot alone cannot answer that:
some stages read project-scoped inputs from the DB / object store AT RUN TIME
rather than from the snapshot. ``action_reel``'s ``footage_probe`` selects the
project's ``video_sources`` rows, and ``shot_analysis`` / ``segment_select`` /
``cut_plan`` / ``compile`` consume its output — so a clip uploaded or deleted
after the failure left the snapshot identical and a resume rendered the old
footage.

This module closes that class rather than the one case:

* :data:`STAGE_LIVE_INPUTS` is the sweep — every stage name any registered
  blueprint wires, mapped to the mutable project inputs it reads live. A stage
  absent from the map is UNSWEPT, and a blueprint containing one never resumes.
* :func:`build_resume_inputs` renders those inputs into a stable, cheap
  representation (ids + ``updated_at`` + the fields the reader consumes). It is
  persisted inside the run's config snapshot under :data:`RESUME_INPUTS_KEY` at
  ``start_run`` and re-derived by the resume check, so parent and would-be child
  are compared by the SAME function through ``order_fingerprint``. A parent that
  predates this key compares unequal, i.e. a full re-run.
* :data:`RESUMABLE_BLUEPRINTS` is the explicit allowlist. A test enumerates the
  registry and reds when an allowlisted blueprint gains a stage the sweep does
  not cover, so a new stage forces a conscious update here.

Deliberately NOT inputs (not project-scoped or not mutable in place): the run's own
artifacts and stage rows (``design_job_stash``, ``word_timed_captions``,
``audiobook_qa``), derived caches keyed by content (``source_cache``, the P3b
media cache), the global sound library catalog, and external live services / web
pages (critique, site capture, media discovery). Uploads stored under a
``uuid7`` key (thumbnails, packages, voiceovers, storyboard images) are immutable
and already pinned by their ref in the snapshot.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from sqlalchemy import select

from src.database.models import VideoSource
from src.models.source_spec import parse_sources

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.services.blueprints.base import DAGBlueprint

#: Snapshot key the fingerprint is persisted under (kept by ``order_fingerprint``).
RESUME_INPUTS_KEY = "resume_inputs"

_FINGERPRINT_VERSION = 1


class LiveInput(enum.StrEnum):
    """A project-scoped, mutable input some stage reads at run time."""

    #: ``video_sources`` rows (``footage_probe``): upload / delete / re-probe.
    VIDEO_SOURCES = "video_sources"
    #: ``brief.sources`` of kind ``document`` (``source_ingest`` → ``fetch_document``).
    #: The object key is ``artifacts/documents/{tenant}/upload/{filename}``, so a
    #: re-upload under the same filename overwrites the bytes behind an UNCHANGED
    #: ref, and the storage facade exposes no cheap stat/etag. Represented by
    #: refusing to resume whenever one is present.
    UPLOADED_DOCUMENTS = "uploaded_documents"
    #: ``brief.manuscript_ref`` (``audiobook_ingest``): filename-keyed like documents
    #: and ALWAYS present, so it has no useful representation.
    UPLOADED_MANUSCRIPT = "uploaded_manuscript"
    #: A course's child projects + their deliverables (``course_compile_lessons``
    #: re-reads them and spawns sub-DAG runs from them).
    COURSE_CHILDREN = "course_children"


#: Inputs :func:`build_resume_inputs` represents. A stage reading anything else is
#: not covered, and neither is any blueprint wiring it.
FINGERPRINTED_INPUTS: frozenset[LiveInput] = frozenset(
    {LiveInput.VIDEO_SOURCES, LiveInput.UPLOADED_DOCUMENTS}
)

_NONE: frozenset[LiveInput] = frozenset()

#: The sweep (2026-09-15, every stage of every registered blueprint, transitive
#: imports scanned for DB sessions/repositories and object-store reads). Keyed by
#: stage NAME; a shared name (``narrate``, ``render_manifest``) was checked for
#: every handler wired under it.
STAGE_LIVE_INPUTS: Mapping[str, frozenset[LiveInput]] = {
    # action_reel
    "footage_probe": frozenset({LiveInput.VIDEO_SOURCES}),
    "shot_analysis": _NONE,
    "segment_select": _NONE,
    "music_select": _NONE,
    "cut_plan": _NONE,
    "graphics_design": _NONE,
    "narrate": _NONE,
    "compile": _NONE,
    "render_manifest": _NONE,
    # shared opt-in leaves
    "source_ingest": frozenset({LiveInput.UPLOADED_DOCUMENTS}),
    "media_discovery": _NONE,
    "site_capture": _NONE,
    "growth_bundle": _NONE,
    "ai_image_gen": _NONE,
    "ai_video_gen": _NONE,
    "motion_graphics_gen": _NONE,
    # website_review
    "website_critique": _NONE,
    "website_redesign": _NONE,
    "website_review_script": _NONE,
    "website_review_record": _NONE,
    "website_review_visual_director": _NONE,
    # Reads the critique + script stage outputs of THIS run and calls WM; no
    # project-scoped mutable input.
    "website_review_slide_plan": _NONE,
    "website_review_slides": _NONE,
    "website_review_compile": _NONE,
    # app_review
    "app_inspect": _NONE,
    "app_critique": _NONE,
    "app_review_script": _NONE,
    "app_review_record": _NONE,
    "app_review_compile": _NONE,
    # audit family
    "seo_audit_scan": _NONE,
    "seo_audit_script": _NONE,
    "seo_audit_slides": _NONE,
    "seo_audit_narrate": _NONE,
    "seo_audit_record": _NONE,
    "seo_audit_compile": _NONE,
    "cro_audit_scan": _NONE,
    "cro_audit_script": _NONE,
    "cro_audit_slides": _NONE,
    "cro_audit_narrate": _NONE,
    "cro_audit_record": _NONE,
    "cro_audit_compile": _NONE,
    "marketing_scorecard_scan": _NONE,
    # demo_video / showcase
    "demo_video_gen": _NONE,
    "script_doctor": _NONE,
    "claim_finish": _NONE,
    "retention_plan": _NONE,
    "reconcile_timeline": _NONE,
    "voice_director": _NONE,
    "demo_video_narrate": _NONE,
    "demo_video_record": _NONE,
    "showcase_slides": _NONE,
    "demo_video_sound_director": _NONE,
    "demo_video_footage_direction": _NONE,
    "demo_video_compile": _NONE,
    "demo_video_footage_critique": _NONE,
    "demo_video_encode_master": _NONE,
    "final_critique": _NONE,
    # generic
    "brief_intake": _NONE,
    "script_draft": _NONE,
    "storyboard_draft": _NONE,
    "video_render": _NONE,
    # lesson
    "lesson_script": _NONE,
    "lesson_slides": _NONE,
    "lesson_narrate": _NONE,
    "lesson_record": _NONE,
    "lesson_compile": _NONE,
    # promo
    "promo_analyze": _NONE,
    "promo_script": _NONE,
    "promo_storyboard": _NONE,
    "promo_narrate": _NONE,
    "promo_render": _NONE,
    "promo_footage_critique": _NONE,
    "promo_review": _NONE,
    "promo_image_prompts": _NONE,
    "promo_visual_director": _NONE,
    "promo_slides": _NONE,
    "promo_footage_direction": _NONE,
    "promo_sound_director": _NONE,
    "promo_html": _NONE,
    "promo_motion_brief": _NONE,
    # web_design v1/v2 (the design-job stash reads the run's OWN stage row)
    "web_design": _NONE,
    "brand_extract": _NONE,
    "design": _NONE,
    "verify": _NONE,
    "assemble": _NONE,
    # course
    "course_outline": _NONE,
    "course_compile_lessons": frozenset({LiveInput.COURSE_CHILDREN}),
    "course_stitch": _NONE,
    "course_thumbnail": _NONE,
    # audiobook
    "audiobook_ingest": frozenset({LiveInput.UPLOADED_MANUSCRIPT}),
    "audiobook_plan": _NONE,
    "audiobook_segment": _NONE,
    "audiobook_cast": _NONE,
    "audiobook_direct": _NONE,
    "audiobook_narrate": _NONE,
    "audiobook_qa": _NONE,
    "audiobook_stitch": _NONE,
}

#: Blueprints a failed run of may resume. Every stage of each is swept AND reads
#: only fingerprinted inputs (enforced by tests/services/runs/test_resume_inputs.py).
#: ``course_v1`` (live child projects + sub-DAG runs) and ``audiobook_v1``
#: (filename-keyed manuscript) are out: a Restart of those is a full re-run.
RESUMABLE_BLUEPRINTS: frozenset[str] = frozenset(
    {
        "website_review_v1",
        "action_reel_v1",
        "app_review_v1",
        "demo_video_v1",
        "showcase_v1",
        "promo_v1",
        "lesson_v1",
        "generic_v1",
        "seo_audit_v1",
        "cro_audit_v1",
        "marketing_scorecard_v1",
        "growth_bundle_v1",
        "web_design_v1",
        "web_design_v2",
    }
)


def uncovered_stages(blueprint: DAGBlueprint) -> list[str]:
    """Stage names of ``blueprint`` that are unswept or read an unfingerprinted input."""
    return [
        spec.name
        for spec in blueprint.stages
        if spec.name not in STAGE_LIVE_INPUTS
        or not STAGE_LIVE_INPUTS[spec.name] <= FINGERPRINTED_INPUTS
    ]


async def build_resume_inputs(
    session: AsyncSession,
    *,
    blueprint: DAGBlueprint,
    project_id: UUID,
    config_snapshot: Mapping[str, object],
) -> dict[str, object]:
    """The live inputs ``blueprint``'s stages read, as a JSON-stable mapping.

    Only the inputs the blueprint actually reads are queried (a website_review run
    never touches ``video_sources``). ``unresumable`` lists why a resume from a run
    carrying this fingerprint is refused; it is empty only when every input the
    blueprint reads is represented here.
    """
    needed: set[LiveInput] = set()
    unresumable: list[str] = []
    for name in uncovered_stages(blueprint):
        unresumable.append(f"stage {name!r} reads an input with no fingerprint")
    for spec in blueprint.stages:
        needed |= STAGE_LIVE_INPUTS.get(spec.name, _NONE)

    inputs: dict[str, object] = {"version": _FINGERPRINT_VERSION}
    if LiveInput.VIDEO_SOURCES in needed:
        inputs[LiveInput.VIDEO_SOURCES.value] = await _video_sources(session, project_id)
    if LiveInput.UPLOADED_DOCUMENTS in needed:
        brief = config_snapshot.get("brief")
        sources = parse_sources(dict(brief)) if isinstance(brief, Mapping) else []
        if any(source.kind == "document" for source in sources):
            unresumable.append(
                "an uploaded document is read by a filename-keyed ref a re-upload "
                "overwrites in place"
            )
    inputs["unresumable"] = unresumable
    return inputs


async def _video_sources(session: AsyncSession, project_id: UUID) -> list[list[object]]:
    """``[id, updated_at, object_key, bytes, preview_status]`` per row, upload order.

    ``updated_at`` is ``onupdate=now()``, so a re-probe / derivative / status change
    moves it; an upload adds a row and a delete removes one. Strings, not UUID /
    datetime objects, so the value JSONB-round-trips byte-identical.
    """
    rows = await session.execute(
        select(
            VideoSource.id,
            VideoSource.updated_at,
            VideoSource.object_key,
            VideoSource.bytes,
            VideoSource.preview_status,
        )
        .where(VideoSource.project_id == project_id)
        .order_by(VideoSource.created_at, VideoSource.id)
    )
    return [
        [str(row.id), row.updated_at.isoformat(), row.object_key, row.bytes, row.preview_status]
        for row in rows
    ]


def resume_refusal(snapshot: Mapping[str, object]) -> Sequence[str] | None:
    """Why a run with this persisted ``snapshot`` may not be resumed, or ``None``.

    A snapshot without the key predates the fingerprint: refused.
    """
    inputs = snapshot.get(RESUME_INPUTS_KEY)
    if not isinstance(inputs, Mapping):
        return ["run predates the live-input fingerprint"]
    reasons = inputs.get("unresumable")
    if not isinstance(reasons, list):
        return ["malformed live-input fingerprint"]
    return [str(reason) for reason in reasons] or None
