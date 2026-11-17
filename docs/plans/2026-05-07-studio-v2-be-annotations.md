# Studio v2 Backend — Annotations substrate + apply-N-notes regenerate

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Replace the one-shot `feedback_note` parameter on regenerate endpoints with a persistent `annotations` substrate. Notes pin to elements (or stages), queue up `open` until applied, get marked `applied` when consumed by a regenerate, and stamp the resulting `applied_in_draft_id` for traceability.

**Architecture:** New `annotations` table sibling to `stage_drafts`. CRUD endpoints under `/api/pipeline/{job}/stages/{stage}/annotations`. The two regenerate endpoints (`POST .../regenerate` and `POST .../elements/{id}/regenerate`) drop the `feedback_note: str` body field and accept `annotation_ids: list[UUID]` instead. Auto-derived scope rule: all selected annotations on the same `target_element_id` → element regen; mixed targets or any stage-pinned → whole-stage regen.

**Tech Stack:** Python 3.14, FastAPI, psycopg 3.x, Alembic, postgres18.

**Prerequisites:** `2026-05-07-studio-v2-be-collapse.md` should be merged first (this plan extends the unified projects tree). If you sequence them in parallel, the only conflict is migration numbering — pick the next free number when you reach Task 1.

---

## File Structure

```
src/
├── api/
│   ├── routes/
│   │   ├── annotations.py             # CREATE
│   │   ├── regenerate.py              # MODIFY  (annotation_ids replaces feedback_note)
│   │   └── __init__.py                # MODIFY  (mount annotations router)
│   └── schemas/
│       ├── annotations.py             # CREATE
│       └── regenerate.py              # MODIFY
├── database/migrations/versions/
│   └── 0013_annotations.py            # CREATE
├── models/
│   └── annotations.py                 # CREATE
├── repositories/
│   └── annotations.py                 # CREATE
├── services/
│   └── steering/
│       ├── annotations_service.py     # CREATE
│       └── regenerate.py              # MODIFY  (consume annotation_ids)
└── tests/
    ├── api/test_annotations_crud.py   # CREATE
    ├── api/test_regenerate_with_annotations.py  # CREATE
    └── migrations/test_0013.py        # CREATE
```

---

### Task 1: Migration 0013 — `annotations` table

**Files:**
- Create: `src/database/migrations/versions/0013_annotations.py`
- Test: `tests/migrations/test_0013.py`

- [ ] **Step 1: Migration**

```python
# src/database/migrations/versions/0013_annotations.py
"""annotations table — element- or stage-pinned, lifecycle: open|applied|dismissed"""
from alembic import op
import sqlalchemy as sa

revision = "0013"
down_revision = "0012"


def upgrade() -> None:
    op.create_table(
        "annotations",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("job_id", sa.dialects.postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("pipeline_jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("stage_name", sa.Text, nullable=False),
        sa.Column("target_element_id", sa.Text, nullable=True),  # NULL = stage-pinned
        sa.Column("text", sa.Text, nullable=False),
        sa.Column(
            "status",
            sa.Enum("open", "applied", "dismissed", name="annotation_status_enum"),
            nullable=False,
            server_default="open",
        ),
        sa.Column("applied_in_draft_id", sa.dialects.postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("stage_drafts.draft_id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_annotations_job_stage", "annotations", ["job_id", "stage_name"])
    op.create_index(
        "ix_annotations_open",
        "annotations",
        ["job_id", "stage_name"],
        postgresql_where=sa.text("status = 'open'"),
    )


def downgrade() -> None:
    op.drop_index("ix_annotations_open", table_name="annotations")
    op.drop_index("ix_annotations_job_stage", table_name="annotations")
    op.drop_table("annotations")
    op.execute("DROP TYPE IF EXISTS annotation_status_enum")
```

- [ ] **Step 2: Roundtrip test + commit**

```python
# tests/migrations/test_0013.py
@pytest.mark.postgres
def test_0013_upgrade_downgrade(alembic_runner):
    alembic_runner.migrate_up_to("0012")
    alembic_runner.migrate_up_to("0013")
    insp = sa.inspect(alembic_runner.engine)
    assert "annotations" in insp.get_table_names()
    alembic_runner.migrate_down_to("0012")
    assert "annotations" not in sa.inspect(alembic_runner.engine).get_table_names()
```

```bash
uv run pytest tests/migrations/test_0013.py -v
git add src/database/migrations/versions/0013_annotations.py tests/migrations/test_0013.py
git commit -m "feat(db): annotations table — element- or stage-pinned"
```

---

### Task 2: Model + Repository

**Files:**
- Create: `src/models/annotations.py`
- Create: `src/repositories/annotations.py`

- [ ] **Step 1: Model**

```python
# src/models/annotations.py
from datetime import datetime
import uuid
from sqlalchemy import DateTime, Enum, ForeignKey, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base

class Annotation(Base):
    __tablename__ = "annotations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True,
                                           server_default=func.gen_random_uuid())
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("pipeline_jobs.id", ondelete="CASCADE"))
    stage_name: Mapped[str] = mapped_column(Text)
    target_element_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    text: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        Enum("open", "applied", "dismissed", name="annotation_status_enum"),
        default="open",
    )
    applied_in_draft_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("stage_drafts.draft_id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        onupdate=func.now())
```

- [ ] **Step 2: Repository**

```python
# src/repositories/annotations.py
import uuid
from typing import Iterable
import sqlalchemy as sa
from src.models.annotations import Annotation


class AnnotationsRepository:
    def __init__(self, db):
        self.db = db

    async def create(self, *, job_id, stage_name, target_element_id, text) -> Annotation:
        ann = Annotation(
            job_id=job_id, stage_name=stage_name,
            target_element_id=target_element_id, text=text,
        )
        self.db.add(ann)
        await self.db.commit()
        await self.db.refresh(ann)
        return ann

    async def list(self, *, job_id, stage_name, status: str | None = None) -> list[Annotation]:
        stmt = (
            sa.select(Annotation)
            .where(Annotation.job_id == job_id, Annotation.stage_name == stage_name)
            .order_by(Annotation.created_at.asc())
        )
        if status:
            stmt = stmt.where(Annotation.status == status)
        return list((await self.db.execute(stmt)).scalars().all())

    async def get(self, ann_id: uuid.UUID) -> Annotation | None:
        return (await self.db.execute(
            sa.select(Annotation).where(Annotation.id == ann_id)
        )).scalar_one_or_none()

    async def update_text(self, ann_id: uuid.UUID, text: str) -> Annotation:
        ann = await self.get(ann_id)
        if ann is None:
            raise NotFoundError(str(ann_id))
        if ann.status != "open":
            raise ValidationError("can only edit annotations while open")
        ann.text = text
        await self.db.commit()
        return ann

    async def dismiss(self, ann_id: uuid.UUID) -> None:
        ann = await self.get(ann_id)
        if ann is None or ann.status != "open":
            return
        ann.status = "dismissed"
        await self.db.commit()

    async def apply(self, ann_ids: Iterable[uuid.UUID], *, draft_id: uuid.UUID) -> None:
        await self.db.execute(
            sa.update(Annotation)
              .where(Annotation.id.in_(list(ann_ids)),
                     Annotation.status == "open")
              .values(status="applied", applied_in_draft_id=draft_id)
        )
        await self.db.commit()

    async def fetch_for_apply(self, ann_ids: Iterable[uuid.UUID]) -> list[Annotation]:
        return list((await self.db.execute(
            sa.select(Annotation).where(Annotation.id.in_(list(ann_ids)))
        )).scalars().all())
```

- [ ] **Step 3: Test (skipping — hit by api integration tests in Task 4)**
- [ ] **Step 4: Commit**

```bash
git add src/models/annotations.py src/repositories/annotations.py
git commit -m "feat(annotations): model + repository"
```

---

### Task 3: Schemas

**Files:**
- Create: `src/api/schemas/annotations.py`

- [ ] **Step 1: Schemas**

```python
# src/api/schemas/annotations.py
from datetime import datetime
import uuid
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class CreateAnnotationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    target_element_id: str | None = None  # null = stage-pinned


class UpdateAnnotationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


class AnnotationResponse(BaseModel):
    id: uuid.UUID
    job_id: uuid.UUID
    stage_name: str
    target_element_id: str | None
    text: str
    status: Literal["open", "applied", "dismissed"]
    applied_in_draft_id: uuid.UUID | None
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)


class AnnotationListResponse(BaseModel):
    items: list[AnnotationResponse]
```

- [ ] **Step 2: Commit**

```bash
git add src/api/schemas/annotations.py
git commit -m "feat(annotations): pydantic schemas"
```

---

### Task 4: Routes — annotations CRUD

**Files:**
- Create: `src/api/routes/annotations.py`
- Modify: `src/api/routes/__init__.py`
- Test: `tests/api/test_annotations_crud.py`

- [ ] **Step 1: Tests**

```python
# tests/api/test_annotations_crud.py
import pytest

@pytest.mark.postgres
async def test_create_list_dismiss_annotation_lifecycle(client, seed):
    job = await seed.pipeline_job()

    # CREATE
    r = await client.post(
        f"/api/pipeline/{job.id}/stages/storyline/annotations",
        json={"text": "punchier hook", "target_element_id": "scene_1"},
    )
    assert r.status_code == 201
    ann_id = r.json()["id"]
    assert r.json()["status"] == "open"

    # LIST
    r = await client.get(f"/api/pipeline/{job.id}/stages/storyline/annotations")
    assert len(r.json()["items"]) == 1

    # FILTER
    r = await client.get(
        f"/api/pipeline/{job.id}/stages/storyline/annotations?status=open"
    )
    assert len(r.json()["items"]) == 1

    # EDIT
    r = await client.patch(
        f"/api/pipeline/{job.id}/stages/storyline/annotations/{ann_id}",
        json={"text": "much punchier hook"},
    )
    assert r.status_code == 200
    assert r.json()["text"] == "much punchier hook"

    # DISMISS
    r = await client.delete(
        f"/api/pipeline/{job.id}/stages/storyline/annotations/{ann_id}"
    )
    assert r.status_code == 204

    r = await client.get(
        f"/api/pipeline/{job.id}/stages/storyline/annotations?status=open"
    )
    assert len(r.json()["items"]) == 0
```

- [ ] **Step 2: Routes**

```python
# src/api/routes/annotations.py
from fastapi import APIRouter, Depends, Response
import uuid
from src.api.schemas.annotations import (
    AnnotationListResponse, AnnotationResponse,
    CreateAnnotationRequest, UpdateAnnotationRequest,
)
from src.repositories.annotations import AnnotationsRepository

router = APIRouter()


@router.post(
    "/pipeline/{job_id}/stages/{stage_name}/annotations",
    response_model=AnnotationResponse, status_code=201,
)
async def create_annotation(
    job_id: uuid.UUID,
    stage_name: str,
    req: CreateAnnotationRequest,
    repo: AnnotationsRepository = Depends(get_annotations_repo),
):
    ann = await repo.create(
        job_id=job_id, stage_name=stage_name,
        target_element_id=req.target_element_id, text=req.text,
    )
    return ann


@router.get(
    "/pipeline/{job_id}/stages/{stage_name}/annotations",
    response_model=AnnotationListResponse,
)
async def list_annotations(
    job_id: uuid.UUID,
    stage_name: str,
    status: str | None = None,
    repo: AnnotationsRepository = Depends(get_annotations_repo),
):
    return AnnotationListResponse(items=await repo.list(
        job_id=job_id, stage_name=stage_name, status=status,
    ))


@router.patch(
    "/pipeline/{job_id}/stages/{stage_name}/annotations/{ann_id}",
    response_model=AnnotationResponse,
)
async def update_annotation(
    job_id: uuid.UUID, stage_name: str, ann_id: uuid.UUID,
    req: UpdateAnnotationRequest,
    repo: AnnotationsRepository = Depends(get_annotations_repo),
):
    return await repo.update_text(ann_id, req.text)


@router.delete(
    "/pipeline/{job_id}/stages/{stage_name}/annotations/{ann_id}",
    status_code=204,
)
async def dismiss_annotation(
    job_id: uuid.UUID, stage_name: str, ann_id: uuid.UUID,
    repo: AnnotationsRepository = Depends(get_annotations_repo),
):
    await repo.dismiss(ann_id)
    return Response(status_code=204)
```

(Wire `router.include_router(annotations.router)` in `src/api/routes/__init__.py`.)

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/api/test_annotations_crud.py -v
git add src/api/routes/annotations.py src/api/routes/__init__.py tests/api/test_annotations_crud.py
git commit -m "feat(api): annotations CRUD endpoints"
```

---

### Task 5: Regenerate consumes `annotation_ids` instead of `feedback_note`

**Files:**
- Modify: `src/api/schemas/regenerate.py` (or wherever the request body lives)
- Modify: `src/api/routes/regenerate.py`
- Modify: `src/services/steering/regenerate.py`
- Test: `tests/api/test_regenerate_with_annotations.py`

- [ ] **Step 1: Schema**

```python
# src/api/schemas/regenerate.py
import uuid
from pydantic import BaseModel, Field


class RegenerateStageRequest(BaseModel):
    annotation_ids: list[uuid.UUID] = Field(default_factory=list)
    auto_advance: bool = False


class RegenerateElementRequest(BaseModel):
    annotation_ids: list[uuid.UUID] = Field(default_factory=list)
    auto_advance: bool = False
```

(Drop the `feedback_note: str | None` field. Callers wanting "fresh take, no specific feedback" pass `annotation_ids=[]` — the service just re-runs the LLM with no extra notes.)

- [ ] **Step 2: Service — fold notes into prompt, mark applied, stamp draft id**

```python
# src/services/steering/regenerate.py — relevant fragment
class RegenerateService:
    async def regenerate_stage(
        self, *, job_id, stage_name, annotation_ids: list[uuid.UUID], auto_advance: bool
    ):
        notes = await self.annotations_repo.fetch_for_apply(annotation_ids)
        # Defensive: silently skip ids that aren't open or don't match this job/stage.
        valid = [a for a in notes if a.status == "open"
                 and a.job_id == job_id and a.stage_name == stage_name]
        feedback_block = self._format_notes(valid)
        new_draft = await self._run_llm_regen(
            job_id=job_id, stage_name=stage_name,
            extra_prompt=feedback_block,
            source="llm_regen",
        )
        if valid:
            await self.annotations_repo.apply(
                [a.id for a in valid], draft_id=new_draft.draft_id,
            )
        if not auto_advance:
            await self._set_status_awaiting_review(job_id, stage_name)
        return new_draft

    async def regenerate_element(
        self, *, job_id, stage_name, element_id, annotation_ids, auto_advance,
    ):
        notes = await self.annotations_repo.fetch_for_apply(annotation_ids)
        # Element-regen guard: every note must target this element_id (or be stage-pinned;
        # stage-pinned notes can't go through element-regen).
        for n in notes:
            if n.target_element_id is None or n.target_element_id != element_id:
                raise ValidationError(
                    "element regenerate accepts only annotations targeting this element"
                )
        # ... LLM call with element-only context, source="llm_element_regen"
        new_draft = await self._run_llm_element_regen(...)
        if notes:
            await self.annotations_repo.apply(
                [a.id for a in notes], draft_id=new_draft.draft_id,
            )
        return new_draft

    @staticmethod
    def _format_notes(notes: list[Annotation]) -> str:
        if not notes:
            return ""
        lines = ["## Reviewer notes — apply ALL of these:"]
        for n in notes:
            target = f"[on {n.target_element_id}] " if n.target_element_id else "[stage] "
            lines.append(f"- {target}{n.text}")
        return "\n".join(lines)
```

- [ ] **Step 3: Routes (drop feedback_note)**

In `src/api/routes/regenerate.py`, replace the body model on `POST .../regenerate` and `POST .../elements/{id}/regenerate` with `RegenerateStageRequest` / `RegenerateElementRequest`. Pass `annotation_ids` through.

- [ ] **Step 4: Tests**

```python
# tests/api/test_regenerate_with_annotations.py
@pytest.mark.postgres
async def test_stage_regenerate_marks_notes_applied(client, seed, llm_mock):
    job = await seed.pipeline_job(stage="storyline", status="awaiting_review")
    ann1 = await seed.annotation(job_id=job.id, stage="storyline",
                                  target="scene_1", text="punchier")
    ann2 = await seed.annotation(job_id=job.id, stage="storyline",
                                  target=None, text="add a CTA")
    llm_mock.respond({"chapters": [{"chapter_id": "c1", "title": "..."}]})

    r = await client.post(
        f"/api/pipeline/{job.id}/stages/storyline/regenerate",
        json={"annotation_ids": [str(ann1.id), str(ann2.id)]},
    )
    assert r.status_code == 200
    new_draft_id = r.json()["draft_id"]

    list_resp = await client.get(
        f"/api/pipeline/{job.id}/stages/storyline/annotations"
    )
    by_id = {a["id"]: a for a in list_resp.json()["items"]}
    assert by_id[str(ann1.id)]["status"] == "applied"
    assert by_id[str(ann1.id)]["applied_in_draft_id"] == new_draft_id
    assert by_id[str(ann2.id)]["status"] == "applied"

    # The LLM saw both notes in the prompt.
    prompt = llm_mock.last_call_args["system_prompt"]
    assert "punchier" in prompt and "add a CTA" in prompt


@pytest.mark.postgres
async def test_element_regenerate_rejects_mismatched_targets(client, seed):
    job = await seed.pipeline_job(stage="shotlist", status="awaiting_review")
    ann_other = await seed.annotation(job_id=job.id, stage="shotlist",
                                       target="scene_2", text="...")
    r = await client.post(
        f"/api/pipeline/{job.id}/stages/shotlist/elements/scene_1/regenerate",
        json={"annotation_ids": [str(ann_other.id)]},
    )
    assert r.status_code == 422
```

- [ ] **Step 5: Run + commit**

```bash
uv run pytest tests/api/test_regenerate_with_annotations.py -v
git add src/api/schemas/regenerate.py src/api/routes/regenerate.py \
        src/services/steering/regenerate.py tests/api/test_regenerate_with_annotations.py
git commit -m "feat(regen): consume annotation_ids, mark applied, stamp draft id"
```

---

### Task 6: Notification events for annotation lifecycle

**Files:**
- Modify: `src/services/notification_service.py`
- Modify: `src/services/steering/annotations_service.py` (publish on each lifecycle change)

- [ ] **Step 1: Event names**

```python
# src/services/notification_service.py — add to PipelineEvent or sibling enum
class AnnotationEvent(StrEnum):
    CREATED = "annotation_created"
    UPDATED = "annotation_updated"
    DISMISSED = "annotation_dismissed"
    APPLIED = "annotation_applied"
```

Each event publishes on `f"job:{job_id}"` (matches the existing job WS channel). Payload: `{annotation_id, stage_name, target_element_id, status, ...}`.

- [ ] **Step 2: Wire into the routes / service**

After every successful create / update / dismiss / apply, publish the matching event with the post-state annotation as payload.

- [ ] **Step 3: Test**

```python
# tests/api/test_annotation_events.py
@pytest.mark.postgres
async def test_create_emits_annotation_created(client, seed, ws_listener):
    job = await seed.pipeline_job()
    async with ws_listener(f"job:{job.id}") as feed:
        await client.post(
            f"/api/pipeline/{job.id}/stages/storyline/annotations",
            json={"text": "..", "target_element_id": "scene_1"},
        )
        evt = await feed.next()
        assert evt.name == "annotation_created"
```

- [ ] **Step 4: Commit**

```bash
git add src/services/notification_service.py src/services/steering/annotations_service.py tests/api/test_annotation_events.py
git commit -m "feat(annotations): emit lifecycle events on the job channel"
```

---

## Final verification

```bash
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
uv run ty check src/ tests/
uv run pytest -v
```

All green. Open PR titled `feat: studio v2 backend annotations — apply-N-notes regenerate`. After merge, the FE plan `2026-05-07-studio-v2-annotations-and-diff.md` is unblocked.
