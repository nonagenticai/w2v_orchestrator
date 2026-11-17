"""Stage-output routes: output read, element patch/restamp, course-stitch
chapter/thumbnail overrides, and stage-output annotations."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import func, text, update

from src.api.routes._common import SessionDep, TenantIdDep
from src.api.schemas import (
    AnnotationCreate,
    AnnotationResponse,
    ChapterTitleOverrides,
    StageElementPatch,
    StageElementRestamp,
    StageOutputAnnotations,
    StageOutputResponse,
    ThumbnailOverride,
)
from src.database.models import StageOutput
from src.database.repositories.annotation_repository import AnnotationRepository
from src.database.repositories.stage_output_repository import StageOutputRepository
from src.database.repositories.stage_step_repository import StageStepRepository
from src.database.stage_output_guard import ALLOW_EDIT_SQL
from src.services.artifact_ref_guard import (
    ArtifactRefError,
    assert_refs_owned,
    run_artifact_refs,
)
from src.services.blueprints.registry import get_registry
from src.services.runs.elements import (
    downstream_stage_names,
    element_ids_in,
    restamp_element_in_output,
    stage_supports_element_pinning,
)

from ._shared import (
    _COURSE_STITCH,
    _chapter_key,
    _is_presignable_artifact_url,
    _stage_response,
    _tenant_owns_run,
    router,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

#: A run the executor owns (or is about to): an in-place output edit would race it.
#: Every other status — paused, completed, failed, CANCELLED — is editable, because
#: the point of cancelling is usually to fix something and Restart (restart_plan.py).
_ACTIVE_RUN_STATUSES = frozenset({"pending", "running"})


async def _reject_unowned_refs(
    session: AsyncSession,
    run_id: UUID,
    tenant_id: UUID,
    payload: object,
) -> None:
    """422 unless every MinIO ref in ``payload`` is one this caller already owns.

    Called BEFORE the write, on every route that lets a caller put a value into
    ``StageOutput.output``. Without it the artifact proxy's read gate authorises
    against a record the attacker just authored: patch
    ``…/w2v-audiobooks/artifacts/{victim-run}/audiobook_stitch/book.mp3`` into
    your own run's output and the proxy happily streams the victim's book, every
    other gate satisfied. The ref set is read from the run's *current* rows, so a
    write can never widen its own allowlist.
    """
    try:
        assert_refs_owned(
            payload,
            tenant_id=tenant_id,
            owned_refs=await run_artifact_refs(session, run_id),
        )
    except ArtifactRefError as exc:
        # 422, not 404: the caller owns the run, the request is simply not one we
        # will store. Nothing about another tenant's data is revealed — the detail
        # only echoes the ref the caller itself sent.
        raise HTTPException(422, f"unowned artifact reference in payload: {exc}") from exc


def _shallow_merge_element(output: dict, stage_name: str, element_id: str, patch: dict) -> None:
    """Shallow-merge ``patch`` into the element addressed by ``element_id``,
    in place, across every addressable path of ``stage_name``. The element's
    ``id`` is preserved regardless of ``patch`` content.

    Extracted verbatim from ``patch_stage_element`` (same walk, same merge) to
    keep that handler under the cyclomatic gate once it grew its status guard.
    """
    from src.services.runs.elements import _ELEMENT_PATHS

    for path_key in _ELEMENT_PATHS.get(stage_name, []):
        nodes = output.get(path_key)
        if not isinstance(nodes, list):
            continue
        output[path_key] = [
            {**node, **patch, "id": element_id}
            if isinstance(node, dict) and node.get("id") == element_id
            else node
            for node in nodes
        ]


async def _allow_terminal_output_edit(session: AsyncSession) -> None:
    """Opt THIS TRANSACTION out of the DB's terminal-output write-once trigger.

    ``stage_outputs.output`` is frozen at the DB level once the stage is terminal
    (``trg_stage_outputs_write_once``, migration 0063 /
    ``src.database.stage_output_guard``) — the routes in this module are the ONLY
    deliberate post-terminal writers, so they are the only place this belongs.
    Every one of them has already established that the caller owns the run and
    that no unowned artifact ref is being smuggled in, so the exemption is granted
    on an authorized, validated edit and nothing else.

    ``SET LOCAL``, never a bare ``SET``: the app reaches postgres through
    pgbouncer in TRANSACTION pooling mode, so a session-scoped GUC would stay set
    on the pooled connection and silently exempt whatever unrelated request got it
    next. Transaction scope also means one call covers every statement in the same
    transaction — which is why the restamp's downstream-reset UPDATE (terminal
    rows back to ``pending`` with ``output = NULL``) needs no second opt-in.
    """
    await session.execute(text(ALLOW_EDIT_SQL))


@router.get(
    "/{run_id}/stages/{stage_name}/output",
    response_model=StageOutputResponse,
    operation_id="runs.stages.output",
)
async def get_stage_output(
    run_id: UUID,
    stage_name: str,
    session: SessionDep,
    tenant_id: TenantIdDep,
) -> StageOutputResponse:
    if (await _tenant_owns_run(session, run_id, tenant_id)) is None:
        raise HTTPException(404, "run not found")
    so = await StageOutputRepository(session).get(run_id, stage_name)
    if so is None:
        raise HTTPException(404, "stage output not found")
    return _stage_response(run_id, so)


@router.patch(
    "/{run_id}/stages/{stage_name}/elements/{element_id}",
    response_model=StageOutputResponse,
    operation_id="runs.stages.elements.patch",
)
async def patch_stage_element(
    run_id: UUID,
    stage_name: str,
    element_id: str,
    body: StageElementPatch,
    session: SessionDep,
    tenant_id: TenantIdDep,
) -> StageOutputResponse:
    """Shallow-merge `body.patch` into one element inside StageOutput.output.

    Element addressing reuses `_ELEMENT_PATHS` from `services/runs/elements.py`.
    The element's `id` is preserved regardless of `patch` content. The read takes
    a row lock (`get_for_update`) so a concurrent edit of a DIFFERENT element in
    the same stage output is not silently discarded.

    Allowed on every run the executor does not own — paused, completed, failed and
    CANCELLED (409 on pending/running). Nothing downstream changes in place; the edit
    stamps ``edited_at``, so a later Restart / regenerate keeps this stage's edited
    output and re-runs only the stages depending on it (``restart_plan.py``)."""
    if not stage_supports_element_pinning(stage_name):
        raise HTTPException(422, f"stage '{stage_name}' does not support element pinning")
    run = await _tenant_owns_run(session, run_id, tenant_id)
    if run is None:
        raise HTTPException(404, "run not found")
    # Same 409 `restamp` has, and for a sharper reason. A `running` run's executor
    # owns these rows: a `pending`/`running` stage (a resumable-timeout reset
    # PRESERVES `output`, so the row still looks editable) does not fire the 0063
    # terminal-output trigger, and the executor's `mark_completed` then passes the
    # `_NON_TERMINAL` guard and overwrites the edit outright — HTTP 200, edit
    # silently gone.
    # ``pending`` too: the dispatcher is about to pick the run up and the executor
    # then owns these rows exactly as it does while running.
    if run.status in _ACTIVE_RUN_STATUSES:
        raise HTTPException(
            409, f"cannot edit a {run.status} run; pause it or wait for it to finish"
        )
    so = await StageOutputRepository(session).get_for_update(run_id, stage_name)
    if so is None:
        raise HTTPException(404, "stage output not found")
    output = dict(so.output or {})
    if element_id not in element_ids_in(output, stage_name):
        raise HTTPException(404, "element not found")
    await _reject_unowned_refs(session, run_id, tenant_id, body.patch)
    _shallow_merge_element(output, stage_name, element_id, body.patch)
    await _allow_terminal_output_edit(session)
    await session.execute(
        update(StageOutput)
        .where(StageOutput.id == so.id)
        .values(output=output, edited_at=func.now())
    )
    await session.commit()
    refreshed = await StageOutputRepository(session).get(run_id, stage_name)
    if refreshed is None:
        raise HTTPException(500, "stage output disappeared")
    # Through the SAME helper the GET uses. Three of this file's five returns
    # already did; these two did not, and the difference is not cosmetic: the
    # helper is what rewrites `minio.test` artifact URLs onto the `/api`
    # proxy path (that hostname does not resolve outside the cluster) and what
    # derives `degraded` from the output's notes. No caller renders this body
    # today -- the store keeps no cache and the page refetches -- so nothing is
    # visibly broken, which is exactly why it would have stayed wrong until a
    # future consumer trusted it.
    return _stage_response(run_id, refreshed)


@router.patch(
    "/{run_id}/stages/course_stitch/chapters",
    response_model=StageOutputResponse,
    operation_id="runs.stages.chapters.patch",
)
async def patch_course_chapters(
    run_id: UUID,
    body: ChapterTitleOverrides,
    session: SessionDep,
    tenant_id: TenantIdDep,
) -> StageOutputResponse:
    """Override chapter titles and sub-chapter visibility on a course's
    ``course_stitch`` output.

    Overrides are keyed by the chapter's stable ``chapter_key`` (see
    ``_chapter_key``): the bare ``lesson_project_id`` for a per-lesson (level-0)
    marker, ``f"{lesson_project_id}#{ordinal}"`` for a per-slide (level-1)
    sub-chapter — both stable across re-stitches, unlike sequence or the slide's
    fresh uuid. They are stored as the source of truth in
    ``chapter_title_overrides`` and ALSO folded into ``chapters[].title`` in
    place (a denormalized copy so any consumer reading raw chapters sees the
    edit). Unknown chapter key → 404.

    ``hidden`` (None = leave untouched, full-replacement otherwise) is the set of
    level-1 ``chapter_key``\\ s to drop from the published YouTube chapters + the
    in-app player; it is stored in ``hidden_chapter_keys`` and denormalized as a
    ``chapters[].hidden`` bool. Only level-1 sub-chapters are hideable — a lesson
    (level-0) or unknown key → 404."""
    if (await _tenant_owns_run(session, run_id, tenant_id)) is None:
        raise HTTPException(404, "run not found")
    so = await StageOutputRepository(session).get_for_update(run_id, _COURSE_STITCH)
    if so is None:
        raise HTTPException(404, "stage output not found")
    output = dict(so.output or {})
    chapters = list(output.get("chapters") or [])
    valid_keys = {k for c in chapters if (k := _chapter_key(c))}
    unknown = sorted(set(body.overrides) - valid_keys)
    if unknown:
        raise HTTPException(404, f"unknown chapter key(s): {unknown}")
    if body.hidden is not None:
        hideable = {k for c in chapters if (k := _chapter_key(c)) and int(c.get("level") or 0) == 1}
        bad = sorted(set(body.hidden) - hideable)
        if bad:
            raise HTTPException(404, f"unknown or non-hideable chapter key(s): {bad}")
        output["hidden_chapter_keys"] = list(dict.fromkeys(body.hidden))
    merged = {**(output.get("chapter_title_overrides") or {}), **body.overrides}
    output["chapter_title_overrides"] = merged
    hidden_set = set(output.get("hidden_chapter_keys") or [])
    output["chapters"] = [
        {**c, "title": merged.get(k, c.get("title")), "hidden": k in hidden_set}
        if (k := _chapter_key(c))
        else c
        for c in chapters
    ]
    await _allow_terminal_output_edit(session)
    await session.execute(
        update(StageOutput)
        .where(StageOutput.id == so.id)
        .values(output=output, edited_at=func.now())
    )
    await session.commit()
    refreshed = await StageOutputRepository(session).get(run_id, _COURSE_STITCH)
    if refreshed is None:
        raise HTTPException(500, "stage output disappeared")
    return _stage_response(run_id, refreshed)


@router.patch(
    "/{run_id}/stages/course_stitch/thumbnail",
    response_model=StageOutputResponse,
    operation_id="runs.stages.thumbnail.patch",
)
async def patch_course_thumbnail(
    run_id: UUID,
    body: ThumbnailOverride,
    session: SessionDep,
    tenant_id: TenantIdDep,
) -> StageOutputResponse:
    """Override (or clear) the course video thumbnail / poster.

    A non-null ``thumbnail_url`` must be an artifact URL the orchestrator owns
    (validated by ``_is_presignable_artifact_url``) — it is stored as
    ``thumbnail_override_url`` and preferred over the auto AI cover by both the
    in-app poster and the YouTube passthrough. ``null`` clears the override and
    reverts to the auto cover (``thumbnail_url`` is left untouched throughout, so
    the revert is lossless)."""
    if (await _tenant_owns_run(session, run_id, tenant_id)) is None:
        raise HTTPException(404, "run not found")
    so = await StageOutputRepository(session).get_for_update(run_id, _COURSE_STITCH)
    if so is None:
        raise HTTPException(404, "stage output not found")
    output = dict(so.output or {})
    if body.thumbnail_url is None:
        output.pop("thumbnail_override_url", None)
    elif not _is_presignable_artifact_url(body.thumbnail_url):
        raise HTTPException(422, "thumbnail_url must be an artifact URL the orchestrator owns")
    else:
        # Shape ("an artifact URL on our own host") is not ownership: the run must
        # already reference the object, or it must be this tenant's own staged
        # upload. Otherwise the override both leaks through the YouTube
        # passthrough and lands in the run's outputs, which is what the artifact
        # proxy reads as proof of ownership.
        await _reject_unowned_refs(session, run_id, tenant_id, body.thumbnail_url)
        output["thumbnail_override_url"] = body.thumbnail_url
    await _allow_terminal_output_edit(session)
    await session.execute(
        update(StageOutput)
        .where(StageOutput.id == so.id)
        .values(output=output, edited_at=func.now())
    )
    await session.commit()
    refreshed = await StageOutputRepository(session).get(run_id, _COURSE_STITCH)
    if refreshed is None:
        raise HTTPException(500, "stage output disappeared")
    return _stage_response(run_id, refreshed)


async def _reset_downstream_stages(
    session: AsyncSession, run_id: UUID, downstream: list[str]
) -> None:
    """Send ``downstream`` stage rows back to ``pending`` and forget their sub-steps.

    Caller has already opted the transaction out of the write-once trigger
    (:func:`_allow_terminal_output_edit`) and commits afterwards; both statements
    here ride that one transaction so the reset and the ledger can never disagree.

    The step-ledger delete is not optional. The user asked for these stages to be
    done again; a sub-step whose input hash does not happen to cover the restamped
    content would otherwise replay its stored paid result on resume (same
    ``run_id``) instead of re-running. This is the ONLY code that sends a finished
    stage back to ``pending`` inside a run — pinned, with the delete, by
    ``tests/api/routes/test_restamp_clears_step_ledger.py``.
    """
    if not downstream:
        return
    await session.execute(
        update(StageOutput)
        .where(StageOutput.run_id == run_id, StageOutput.stage_name.in_(downstream))
        .values(status="pending", output=None, error=None, completed_at=None)
    )
    step_repo = StageStepRepository(session)
    for stage_name in downstream:
        await step_repo.delete_for_stage(run_id, stage_name)


@router.post(
    "/{run_id}/stages/{stage_name}/elements/{element_id}/restamp",
    response_model=StageOutputResponse,
    operation_id="runs.stages.elements.restamp",
)
async def restamp_stage_element(
    run_id: UUID,
    stage_name: str,
    element_id: str,
    body: StageElementRestamp,
    session: SessionDep,
    tenant_id: TenantIdDep,
) -> StageOutputResponse:
    """Replace one element's value (id preserved) and reset downstream stages.

    Unlike `runs.stages.elements.patch` (shallow-merge, downstream untouched in
    place), a restamp:

    1. Fully replaces the addressed element with ``body.value`` (the element's
       ``id`` is preserved regardless of ``value`` content).
    2. Sets every downstream stage_output (per the run's blueprint topological
       order) to ``status='pending'`` with ``output=null`` and ``error=null``.

    After a restamp the caller is expected to ``POST /runs/{id}/regenerate``
    to re-execute the reset stages (the restart plan re-runs them and reuses the
    rest). The restamp itself does NOT spawn a new run — same run-id, same
    stage_output row, fresh content. 409 on a pending/running run (the executor owns
    the rows) and on a completed one (it would destroy the delivered output).
    """
    if not stage_supports_element_pinning(stage_name):
        raise HTTPException(422, f"stage '{stage_name}' does not support element pinning")
    run = await _tenant_owns_run(session, run_id, tenant_id)
    if run is None:
        raise HTTPException(404, "run not found")
    if run.status in _ACTIVE_RUN_STATUSES:
        raise HTTPException(
            409, f"cannot restamp on a {run.status} run; pause it or wait for it to finish"
        )
    if run.status == "completed":
        # A restamp resets every DOWNSTREAM stage to pending/output=NULL — the
        # terminal compile stage included — but leaves `runs.status='completed'`.
        # Nothing re-runs a terminal run (the reconciler only picks up `running`
        # orphans), so on a delivered video this destroys the delivered output
        # with no way back. The non-destructive path for a finished run is
        # POST /runs/{id}/regenerate, which spawns a CHILD and leaves this run's
        # artifacts intact.
        raise HTTPException(
            409,
            "cannot restamp a completed run; POST /runs/{run_id}/regenerate to "
            "spawn a child run instead",
        )
    so_repo = StageOutputRepository(session)
    so = await so_repo.get_for_update(run_id, stage_name)
    if so is None:
        raise HTTPException(404, "stage output not found")
    # Same free-form-value hole as the element patch — a restamp replaces the
    # element outright, so it is if anything the easier injection point.
    await _reject_unowned_refs(session, run_id, tenant_id, body.value)
    new_output = restamp_element_in_output(
        dict(so.output or {}), stage_name, element_id, body.value
    )
    if new_output is None:
        raise HTTPException(404, "element not found")
    try:
        blueprint = get_registry().resolve_by_id(run.blueprint_id)
    except KeyError as exc:
        raise HTTPException(500, f"unknown blueprint '{run.blueprint_id}'") from exc
    downstream = downstream_stage_names(blueprint, stage_name)
    # Restamp the addressed stage's output (keep status='completed') and reset the
    # downstream rows — BOTH are post-terminal output writes, and the single
    # transaction-scoped opt-in below covers both.
    await _allow_terminal_output_edit(session)
    await session.execute(
        update(StageOutput)
        .where(StageOutput.id == so.id)
        .values(output=new_output, edited_at=func.now())
    )
    await _reset_downstream_stages(session, run_id, downstream)
    await session.commit()
    refreshed = await so_repo.get(run_id, stage_name)
    if refreshed is None:
        raise HTTPException(500, "stage output disappeared")
    # See the element-PATCH return above: same helper, same two reasons.
    return _stage_response(run_id, refreshed)


@router.get(
    "/{run_id}/stages/{stage_name}/annotations",
    response_model=StageOutputAnnotations,
    operation_id="runs.stages.annotations.list",
)
async def list_stage_annotations(
    run_id: UUID,
    stage_name: str,
    session: SessionDep,
    tenant_id: TenantIdDep,
    status: str | None = None,
) -> StageOutputAnnotations:
    if (await _tenant_owns_run(session, run_id, tenant_id)) is None:
        raise HTTPException(404, "run not found")
    so = await StageOutputRepository(session).get(run_id, stage_name)
    if so is None:
        raise HTTPException(404, "stage output not found")
    filter_status = None if status == "all" else (status or "open")
    items = await AnnotationRepository(session).list_for_stage_output(so.id, status=filter_status)
    return StageOutputAnnotations(
        stage_output_id=so.id,
        items=[AnnotationResponse.model_validate(a, from_attributes=True) for a in items],
    )


@router.post(
    "/{run_id}/stages/{stage_name}/annotations",
    response_model=AnnotationResponse,
    status_code=status.HTTP_201_CREATED,
    operation_id="runs.stages.annotations.create",
)
async def create_stage_annotation(
    run_id: UUID,
    stage_name: str,
    body: AnnotationCreate,
    session: SessionDep,
    tenant_id: TenantIdDep,
) -> AnnotationResponse:
    if (await _tenant_owns_run(session, run_id, tenant_id)) is None:
        raise HTTPException(404, "run not found")
    so = await StageOutputRepository(session).get(run_id, stage_name)
    if so is None:
        raise HTTPException(404, "stage output not found")
    if body.element_id is not None:
        if not stage_supports_element_pinning(stage_name):
            raise HTTPException(400, f"stage '{stage_name}' does not support element pinning")
        if so.output is None:
            raise HTTPException(
                400,
                "stage has no output yet — element_id cannot be validated",
            )
        valid_ids = element_ids_in(so.output, stage_name)
        if body.element_id not in valid_ids:
            raise HTTPException(
                400,
                f"element_id '{body.element_id}' not present in stage output",
            )
    a = await AnnotationRepository(session).create(
        stage_output_id=so.id, text=body.text, element_id=body.element_id
    )
    # Commit BEFORE building the response. `AnnotationRepository.create` only
    # flushes, and `get_db_session` commits in its yield-teardown, which FastAPI
    # runs AFTER the response has been sent — so without this the 201 describes a
    # row no other connection can see yet, and the client's immediate follow-up
    # GET (a different pooled connection) can race it. Every sibling mutating
    # handler in this module already does this (`patch_stage_element`,
    # `patch_course_chapters`, `patch_course_thumbnail`, `restamp_stage_element`);
    # this one was missed by the #804 sweep. Safe here: no HTTPException follows,
    # and there is no deferrable constraint on this insert to map to a 409.
    await session.commit()
    return AnnotationResponse.model_validate(a, from_attributes=True)
