"""A clip reorder must reach the array the COMPILE stage actually reads.

On ``demo_video_v1`` the only legal reorder spine is ``demo_video_gen``, but four
optional stages re-emit its ``lines`` shape with the same ids, and
``resolve_plan_output`` (``stages/demo_video/spine.py``) ranks the raw spine
LOWEST — ``reconcile_timeline`` wins. ``demo_video_compile`` stitches in the order
of THAT list. So a reorder that touches only the spine produces a green child run
and a byte-identical video.

These tests pin the fix in ``build_reorder_state``: every preserved pre-cut stage
that mirrors the spine array (same key, same id set) is reordered too — and a
stage carrying a DIFFERENT id set is not, which is what keeps the id-set match a
guard rather than a blanket rewrite.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from src.database.models import Deliverable, Project, Run, StageOutput
from src.services.blueprints.registry import get_registry
from src.services.runs.regenerate import build_reorder_state

pytestmark = pytest.mark.asyncio

CLIP_IDS = ["clip-1", "clip-3", "clip-4", "clip-5"]
NEW_ORDER = ["clip-5", "clip-1", "clip-3", "clip-4"]

#: Every stage of ``PLAN_STAGE_PRECEDENCE`` that can carry the spine ``lines``.
#: ``demo_video_gen`` is the spine; the other three are the mirrors the compile
#: stage may actually read instead.
MIRROR_STAGES = ("demo_video_gen", "script_doctor", "retention_plan", "reconcile_timeline")


def _lines() -> list[dict[str, object]]:
    return [{"id": cid, "text": f"line {cid}", "seconds": 4.0} for cid in CLIP_IDS]


async def _seed_demo_video_parent(db_session) -> Run:
    tenant_id, subject_id = uuid4(), uuid4()
    project = Project(
        id=uuid4(),
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        visibility="tenant",
        type="demo_video",
        title="mirror-reorder-parent",
        brief={},
        brand={},
        voice={},
        theme={},
    )
    deliverable = Deliverable(
        id=uuid4(),
        project_id=project.id,
        name="d",
        format="16x9",
        captions_mode="none",
        target_duration_seconds=30,
    )
    parent = Run(
        id=uuid4(),
        deliverable_id=deliverable.id,
        blueprint_id="demo_video_v1",
        status="completed",
        config_snapshot={
            "project_id": str(project.id),
            "tenant_id": str(tenant_id),
            "subject_id": str(subject_id),
            "title": project.title,
            "type": "demo_video",
            "brief": {},
            "brand": {},
            "voice": {},
            "theme": {},
            "deliverable": {
                "id": str(deliverable.id),
                "name": deliverable.name,
                "format": deliverable.format,
                "captions_mode": deliverable.captions_mode,
                "target_duration_seconds": deliverable.target_duration_seconds,
            },
        },
        pause_after_stages=[],
    )
    outputs: dict[str, dict[str, object]] = {name: {"lines": _lines()} for name in MIRROR_STAGES}
    # NEGATIVE CASE: same ``lines`` KEY, a DIFFERENT id set (partially OVERLAPPING
    # the spine, and authored in an order the spine's rank would actively change).
    # Not a mirror ⇒ must survive untouched. The overlap is deliberate: a disjoint
    # id set sorts identically with or without the guard, which would make this
    # assertion unfalsifiable.
    outputs["claim_finish"] = {
        "lines": [
            {"id": "clip-3", "text": "a"},
            {"id": "clip-1", "text": "b"},
            {"id": "extra-9", "text": "c"},
        ]
    }
    # A pre-cut stage with no element array at all: preserved verbatim.
    outputs["demo_video_record"] = {"clips": [{"line_id": cid} for cid in CLIP_IDS]}
    sos = [
        StageOutput(
            id=uuid4(), run_id=parent.id, stage_name=name, status="completed", output=output
        )
        for name, output in outputs.items()
    ]
    db_session.add_all([project, deliverable, parent, *sos])
    await db_session.commit()
    return parent


async def test_reorder_propagates_to_every_mirror_of_the_spine(db_session) -> None:
    parent = await _seed_demo_video_parent(db_session)
    blueprint = get_registry().resolve("demo_video")

    preserved = await build_reorder_state(
        session=db_session,
        parent_run=parent,
        blueprint=blueprint,
        spine_stage="demo_video_gen",
        ordered_ids=NEW_ORDER,
    )

    # The one that matters most FIRST: ``resolve_plan_output`` ranks
    # reconcile_timeline ABOVE the spine, so this is the list
    # ``demo_video_compile`` actually stitches from. Asserted before the others so
    # a regression names the stage that decides the delivered video.
    assert [line["id"] for line in preserved["reconcile_timeline"]["lines"]] == NEW_ORDER
    # The remaining mirrors, asserted EXPLICITLY — a loop that skipped one would
    # still pass a set-wise check.
    assert [line["id"] for line in preserved["retention_plan"]["lines"]] == NEW_ORDER
    assert [line["id"] for line in preserved["script_doctor"]["lines"]] == NEW_ORDER
    # The spine itself.
    assert [line["id"] for line in preserved["demo_video_gen"]["lines"]] == NEW_ORDER

    # Bodies travel with their ids — the reorder must not re-key the content.
    assert [line["text"] for line in preserved["reconcile_timeline"]["lines"]] == [
        f"line {cid}" for cid in NEW_ORDER
    ]


async def test_reorder_leaves_a_different_id_set_untouched(db_session) -> None:
    parent = await _seed_demo_video_parent(db_session)
    blueprint = get_registry().resolve("demo_video")

    preserved = await build_reorder_state(
        session=db_session,
        parent_run=parent,
        blueprint=blueprint,
        spine_stage="demo_video_gen",
        ordered_ids=NEW_ORDER,
    )

    # Same ``lines`` key, different id set ⇒ not a mirror ⇒ preserved verbatim.
    # Drop the id-set guard and the spine's rank would pull clip-1 ahead of clip-3.
    assert [line["id"] for line in preserved["claim_finish"]["lines"]] == [
        "clip-3",
        "clip-1",
        "extra-9",
    ]
    # A stage with no spine-shaped array at all is untouched too.
    assert preserved["demo_video_record"] == {"clips": [{"line_id": cid} for cid in CLIP_IDS]}
