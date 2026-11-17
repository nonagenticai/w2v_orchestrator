# Studio v2 Backend — Collapse curricula into projects, recursive children, clone-from

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the separate `curricula` + `curriculum_lessons` substrate with one recursive `projects` tree (`project_type` + `parent_project_id`), add child-project endpoints, and a creation-time `clone_settings_from` shortcut for brand/voice/tone reuse.

**Architecture:** A video-course Project owns lesson Projects as children. Each lesson is itself a full Project that runs the `training_course` DAG. Curriculum-level metadata (audience, syllabus) lives in the parent Project's `brief` JSONB; the dedicated curricula tables go away. Clone-from copies brand/voice/tone (and brief defaults) at creation time — no separate brand-profile entity.

**Tech Stack:** Python 3.14, FastAPI, psycopg 3.x (NOT asyncpg), Alembic, postgres18, uv, ruff, ty.

**Sequencing vs the in-flight 7-phase plan (`~/.claude/plans/right-now-everything-vast-teapot.md`):**
Phase 7 of that plan introduces `curricula` + `curriculum_lessons` (migrations 0010 already on `main`). This plan **replaces** that substrate. Recommended:
- If vast-teapot phase 7 has not yet built curriculum-design DAG handlers in production traffic, **revert migration 0010 in this plan** and never expose curriculum endpoints externally.
- If phase 7 has shipped curriculum-design DAG handlers (`curriculum_audience` / `curriculum_outline` / `curriculum_syllabus`), keep those handlers — they're stage handlers on a Project of `project_type=training_course`. Just delete the curricula *tables and routes*; the DAG is unchanged.

The migration in Task 2 below handles both cases — it drops the tables but is data-preserving (curriculum_lessons rows are upserted into projects.parent_project_id + position).

---

## File Structure

```
src/
├── api/
│   ├── routes/
│   │   ├── projects.py                # MODIFY  (children + clone + ?type filter + SSE)
│   │   ├── curricula.py               # DELETE
│   │   └── __init__.py                # MODIFY  (drop curricula router include)
│   └── schemas/
│       ├── projects.py                # MODIFY  (parent fields, children, clone_settings_from)
│       └── curricula.py               # DELETE
├── database/
│   └── migrations/versions/
│       ├── 0011_project_type_and_recursion.py   # CREATE
│       └── 0012_drop_curricula_tables.py        # CREATE
├── models/
│   └── projects.py                    # MODIFY  (Column rename, parent_project_id, position, lesson_summary_json)
├── repositories/
│   └── projects.py                    # MODIFY  (list-by-type, list-children, reorder, clone helper)
├── services/
│   ├── projects/
│   │   ├── project_service.py         # MODIFY  (clone-from, create-child, reorder)
│   │   └── curriculum/                # DELETE  (whole subdir)
│   ├── notification_service.py        # MODIFY  (child_added / child_reordered / child_status_changed)
│   └── orchestration/registry.py      # MODIFY  (alias 'lesson' to training_course DAG)
└── tests/
    ├── api/test_projects_children.py             # CREATE
    ├── api/test_projects_clone.py                # CREATE
    ├── repositories/test_projects_recursion.py   # CREATE
    └── migrations/test_0011_0012.py              # CREATE
```

---

### Task 1: Migration 0011 — `project_type` rename + recursion fields

**Files:**
- Create: `src/database/migrations/versions/0011_project_type_and_recursion.py`
- Modify: `src/models/projects.py`
- Test: `tests/migrations/test_0011_0012.py`

- [ ] **Step 1: Write the migration**

```python
# src/database/migrations/versions/0011_project_type_and_recursion.py
"""project_type rename + recursion fields

Revision ID: 0011
Revises: 0010
Create Date: 2026-05-07
"""
from alembic import op
import sqlalchemy as sa

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Rename column content_kind -> project_type. Enum values unchanged.
    op.alter_column("projects", "content_kind", new_column_name="project_type")

    # Extend the enum with 'lesson'.
    op.execute("ALTER TYPE content_kind RENAME TO project_type_enum")
    op.execute("ALTER TYPE project_type_enum ADD VALUE IF NOT EXISTS 'lesson'")

    # Recursion fields.
    op.add_column(
        "projects",
        sa.Column(
            "parent_project_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )
    op.add_column("projects", sa.Column("position", sa.Integer(), nullable=True))
    op.add_column(
        "projects",
        sa.Column(
            "lesson_summary_json",
            sa.dialects.postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )

    # Partial unique index: position uniqueness within a parent.
    op.create_index(
        "ix_projects_parent_position",
        "projects",
        ["parent_project_id", "position"],
        unique=True,
        postgresql_where=sa.text("parent_project_id IS NOT NULL"),
    )
    # Lookup index for child listing.
    op.create_index(
        "ix_projects_parent",
        "projects",
        ["parent_project_id"],
        postgresql_where=sa.text("parent_project_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_projects_parent", table_name="projects")
    op.drop_index("ix_projects_parent_position", table_name="projects")
    op.drop_column("projects", "lesson_summary_json")
    op.drop_column("projects", "position")
    op.drop_column("projects", "parent_project_id")
    # Note: removing an enum value in postgres requires a type rebuild; we
    # keep 'lesson' in the enum on downgrade since data may reference it.
    op.execute("ALTER TYPE project_type_enum RENAME TO content_kind")
    op.alter_column("projects", "project_type", new_column_name="content_kind")
```

- [ ] **Step 2: Update the SQLAlchemy model**

```python
# src/models/projects.py — Project class additions
class Project(Base):
    # ... existing columns
    project_type: Mapped[str] = mapped_column(  # was content_kind
        sa.Enum("promo", "app_review", "training_course", "generic", "lesson",
                name="project_type_enum"),
        nullable=False,
    )
    parent_project_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.dialects.postgresql.UUID(as_uuid=True),
        sa.ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    position: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    lesson_summary_json: Mapped[dict | None] = mapped_column(
        sa.dialects.postgresql.JSONB, nullable=True
    )

    # Self-referential relationship for the children collection.
    children: Mapped[list["Project"]] = relationship(
        "Project",
        backref=sa.orm.backref("parent", remote_side="Project.id"),
        order_by="Project.position",
    )
```

Replace every `content_kind` reference in the codebase with `project_type`:

```bash
grep -rn "content_kind" src/ tests/ --include="*.py" -l
```

Each match: rename. Run ruff format after.

- [ ] **Step 3: Migration roundtrip test**

```python
# tests/migrations/test_0011_0012.py
import subprocess
import pytest

@pytest.mark.postgres
def test_0011_upgrade_downgrade(alembic_runner):
    alembic_runner.migrate_up_to("0010")
    alembic_runner.migrate_up_to("0011")
    # After upgrade: column renamed, new columns exist, indexes present.
    assert _column_exists(alembic_runner.engine, "projects", "project_type")
    assert _column_exists(alembic_runner.engine, "projects", "parent_project_id")
    assert _column_exists(alembic_runner.engine, "projects", "position")
    assert _column_exists(alembic_runner.engine, "projects", "lesson_summary_json")
    assert not _column_exists(alembic_runner.engine, "projects", "content_kind")

    alembic_runner.migrate_down_to("0010")
    assert _column_exists(alembic_runner.engine, "projects", "content_kind")
    assert not _column_exists(alembic_runner.engine, "projects", "parent_project_id")

# helper
def _column_exists(engine, table: str, col: str) -> bool:
    insp = sa.inspect(engine)
    return col in {c["name"] for c in insp.get_columns(table)}
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/migrations/test_0011_0012.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/database/migrations/versions/0011_project_type_and_recursion.py \
        src/models/projects.py tests/migrations/test_0011_0012.py
git add -u  # captures all the content_kind→project_type renames
git commit -m "feat(db): rename content_kind→project_type, add recursion fields"
```

---

### Task 2: Migration 0012 — drop curricula, migrate data into projects

**Files:**
- Create: `src/database/migrations/versions/0012_drop_curricula_tables.py`

- [ ] **Step 1: Migration with data preservation**

```python
# src/database/migrations/versions/0012_drop_curricula_tables.py
"""drop curricula tables, hoist lesson links into projects.parent_project_id

Revision ID: 0012
Revises: 0011
Create Date: 2026-05-07

The pre-state (after 0010 + 0011) has:
  - curricula                     (id, tenant_id, ...)
  - curriculum_lessons            (id, curriculum_id, position, project_id, accepted_summary_json, ...)
  - projects.curriculum_id        (added 0010)
  - projects.parent_project_id    (added 0011, currently NULL)
  - projects.position             (added 0011, currently NULL)
  - projects.lesson_summary_json  (added 0011, currently NULL)

Migration semantics:
  Step A: For every curriculum_lessons row whose project_id IS NOT NULL,
          UPDATE the linked project: set parent_project_id = (project of the curriculum)
          ... but the curriculum itself doesn't have a corresponding project row.
          Decision: create a stub project row PER CURRICULUM with
                    project_type='training_course', title=curricula.title,
                    brief=jsonb_build_object('audience', curricula.audience).

  Step B: For each curriculum_lessons row with project_id NOT NULL,
          UPDATE projects SET parent_project_id=<stub_project_id>,
                              position=curriculum_lessons.position,
                              lesson_summary_json=curriculum_lessons.accepted_summary_json,
                              project_type='lesson'
          WHERE id = curriculum_lessons.project_id.

  Step C: Drop curriculum_lessons, curricula, and projects.curriculum_id.
"""
from alembic import op
import sqlalchemy as sa

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    bind.execute(sa.text("""
        WITH stub AS (
            INSERT INTO projects (
                id, tenant_id, subject_id, visibility, title, project_type,
                brief, brand_config, voice_config, theme_config, status,
                created_at, updated_at
            )
            SELECT
                gen_random_uuid(),  -- new course-project id; lessons will FK to it
                c.tenant_id, c.subject_id, c.visibility,
                c.title, 'training_course'::project_type_enum,
                jsonb_build_object('audience', c.audience),
                c.brand_config, c.voice_config, c.theme_config,
                CASE c.status
                    WHEN 'designing' THEN 'drafting'
                    WHEN 'authoring' THEN 'producing'
                    WHEN 'completed' THEN 'completed'
                END::project_status_enum,
                c.created_at, c.updated_at
            FROM curricula c
            RETURNING id, title, tenant_id
        ),
        mapping AS (
            -- Re-discover stub→curriculum by (tenant_id, title) since RETURNING
            -- can't include the source curricula.id directly.
            SELECT s.id AS stub_id, c.id AS curriculum_id
            FROM stub s
            JOIN curricula c
              ON c.tenant_id = s.tenant_id AND c.title = s.title
        )
        UPDATE projects p
           SET parent_project_id     = m.stub_id,
               position              = cl.position,
               lesson_summary_json   = cl.accepted_summary_json,
               project_type          = 'lesson'::project_type_enum
        FROM curriculum_lessons cl
        JOIN mapping m ON m.curriculum_id = cl.curriculum_id
        WHERE p.id = cl.project_id;
    """))
    op.drop_table("curriculum_lessons")
    op.drop_column("projects", "curriculum_id")
    op.drop_table("curricula")


def downgrade() -> None:
    # This is a one-way data migration. The downgrade re-creates empty tables
    # so subsequent migrations can roll back, but the original rows are gone.
    op.create_table(
        "curricula",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("subject_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("visibility", sa.Text, nullable=False),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("audience", sa.dialects.postgresql.JSONB),
        sa.Column("brand_config", sa.dialects.postgresql.JSONB),
        sa.Column("voice_config", sa.dialects.postgresql.JSONB),
        sa.Column("theme_config", sa.dialects.postgresql.JSONB),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("design_pipeline_job_id", sa.dialects.postgresql.UUID(as_uuid=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.add_column(
        "projects",
        sa.Column(
            "curriculum_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            sa.ForeignKey("curricula.id"),
            nullable=True,
        ),
    )
    op.create_table(
        "curriculum_lessons",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("curriculum_id", sa.dialects.postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("curricula.id", ondelete="CASCADE"), nullable=False),
        sa.Column("position", sa.Integer, nullable=False),
        sa.Column("title", sa.Text),
        sa.Column("learning_objective", sa.Text),
        sa.Column("estimated_duration_minutes", sa.Integer),
        sa.Column("prereqs", sa.dialects.postgresql.ARRAY(sa.Integer)),
        sa.Column("project_id", sa.dialects.postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("projects.id")),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("accepted_summary_json", sa.dialects.postgresql.JSONB),
        sa.UniqueConstraint("curriculum_id", "position"),
    )
```

- [ ] **Step 2: Test data preservation against a seeded fixture**

```python
# tests/migrations/test_0011_0012.py — additional case
@pytest.mark.postgres
def test_0012_preserves_lesson_summary(alembic_runner, db_session):
    alembic_runner.migrate_up_to("0010")
    # Seed: 1 curriculum, 1 curriculum_lessons row with a real project_id.
    cur_id = _seed_curriculum(db_session, title="Course X", audience={"role": "PM"})
    proj_id = _seed_project(db_session, title="L1")
    _seed_curriculum_lesson(
        db_session, cur_id, position=1, project_id=proj_id,
        accepted_summary_json={"summary_text": "intro", "key_points": [], "referenced_terms": []}
    )
    db_session.commit()

    alembic_runner.migrate_up_to("0011")
    alembic_runner.migrate_up_to("0012")

    row = db_session.execute(sa.text(
        "SELECT project_type, parent_project_id, position, lesson_summary_json"
        "  FROM projects WHERE id = :pid"
    ), {"pid": proj_id}).mappings().one()
    assert row["project_type"] == "lesson"
    assert row["position"] == 1
    assert row["lesson_summary_json"]["summary_text"] == "intro"
    parent = db_session.execute(sa.text(
        "SELECT title, project_type FROM projects WHERE id = :pid"
    ), {"pid": row["parent_project_id"]}).mappings().one()
    assert parent["title"] == "Course X"
    assert parent["project_type"] == "training_course"
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/migrations/test_0011_0012.py -v
git add src/database/migrations/versions/0012_drop_curricula_tables.py tests/migrations/test_0011_0012.py
git commit -m "feat(db): drop curricula tables, hoist lessons into projects tree"
```

---

### Task 3: Repository — list-by-type, children, reorder, clone

**Files:**
- Modify: `src/repositories/projects.py`
- Test: `tests/repositories/test_projects_recursion.py`

- [ ] **Step 1: Tests**

```python
# tests/repositories/test_projects_recursion.py
import pytest
from src.repositories.projects import ProjectRepository

@pytest.mark.postgres
async def test_list_excludes_lessons_by_default(db_session, seed):
    course = await seed.project(project_type="training_course")
    lesson = await seed.project(project_type="lesson", parent_project_id=course.id, position=1)
    promo = await seed.project(project_type="promo")

    repo = ProjectRepository(db_session)
    items = await repo.list(tenant_id=course.tenant_id)

    ids = {p.id for p in items}
    assert course.id in ids
    assert promo.id in ids
    assert lesson.id not in ids


@pytest.mark.postgres
async def test_list_with_type_lesson_returns_only_lessons(db_session, seed):
    course = await seed.project(project_type="training_course")
    l1 = await seed.project(project_type="lesson", parent_project_id=course.id, position=1)
    l2 = await seed.project(project_type="lesson", parent_project_id=course.id, position=2)

    repo = ProjectRepository(db_session)
    items = await repo.list(tenant_id=course.tenant_id, project_type="lesson")
    assert {p.id for p in items} == {l1.id, l2.id}


@pytest.mark.postgres
async def test_reorder_children_assigns_positions(db_session, seed):
    course = await seed.project(project_type="training_course")
    a = await seed.project(project_type="lesson", parent_project_id=course.id, position=1)
    b = await seed.project(project_type="lesson", parent_project_id=course.id, position=2)
    c = await seed.project(project_type="lesson", parent_project_id=course.id, position=3)

    repo = ProjectRepository(db_session)
    await repo.reorder_children(course.id, [c.id, a.id, b.id])
    refreshed = await repo.list_children(course.id)
    assert [p.position for p in refreshed] == [1, 2, 3]
    assert [p.id for p in refreshed] == [c.id, a.id, b.id]


@pytest.mark.postgres
async def test_clone_settings_copies_brand_voice_theme(db_session, seed):
    src = await seed.project(
        project_type="promo",
        brand_config={"primary": "#abc"},
        voice_config={"narrator": "alex"},
        theme_config={"font": "Inter"},
        brief={"tone": "punchy"},
    )

    repo = ProjectRepository(db_session)
    clone_payload = await repo.fetch_settings_for_clone(src.id)

    assert clone_payload == {
        "brand_config": {"primary": "#abc"},
        "voice_config": {"narrator": "alex"},
        "theme_config": {"font": "Inter"},
        "brief_defaults": {"tone": "punchy"},
    }
```

- [ ] **Step 2: Implement**

```python
# src/repositories/projects.py — additions

class ProjectRepository:
    # ... existing
    async def list(
        self,
        *,
        tenant_id: uuid.UUID,
        project_type: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Project]:
        stmt = sa.select(Project).where(Project.tenant_id == tenant_id)
        if project_type:
            stmt = stmt.where(Project.project_type == project_type)
        else:
            # Hide lessons unless asked for explicitly.
            stmt = stmt.where(Project.parent_project_id.is_(None))
        if status:
            stmt = stmt.where(Project.status == status)
        stmt = stmt.order_by(Project.created_at.desc()).limit(limit).offset(offset)
        return list((await self.db.execute(stmt)).scalars().all())

    async def list_children(self, parent_id: uuid.UUID) -> list[Project]:
        stmt = (
            sa.select(Project)
            .where(Project.parent_project_id == parent_id)
            .order_by(Project.position.asc())
        )
        return list((await self.db.execute(stmt)).scalars().all())

    async def reorder_children(
        self, parent_id: uuid.UUID, order: list[uuid.UUID]
    ) -> None:
        # Two-pass UPDATE to dodge the partial unique index during the swap.
        # Pass 1: park positions out-of-band (negative).
        await self.db.execute(
            sa.text("""
                UPDATE projects SET position = -position
                 WHERE parent_project_id = :pid
            """),
            {"pid": parent_id},
        )
        # Pass 2: assign final 1..N.
        for idx, child_id in enumerate(order, start=1):
            await self.db.execute(
                sa.text("UPDATE projects SET position = :pos WHERE id = :cid AND parent_project_id = :pid"),
                {"pos": idx, "cid": child_id, "pid": parent_id},
            )
        await self.db.commit()

    async def fetch_settings_for_clone(
        self, source_project_id: uuid.UUID
    ) -> dict[str, Any]:
        row = (await self.db.execute(
            sa.select(
                Project.brand_config, Project.voice_config,
                Project.theme_config, Project.brief
            ).where(Project.id == source_project_id)
        )).one()
        return {
            "brand_config": row.brand_config,
            "voice_config": row.voice_config,
            "theme_config": row.theme_config,
            "brief_defaults": row.brief,
        }
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/repositories/test_projects_recursion.py -v
git add src/repositories/projects.py tests/repositories/test_projects_recursion.py
git commit -m "feat(repo): list-by-type, children, reorder, clone helpers"
```

---

### Task 4: Schemas — `parent_project_id`, `children`, `clone_settings_from`

**Files:**
- Modify: `src/api/schemas/projects.py`

- [ ] **Step 1: Update Pydantic schemas**

```python
# src/api/schemas/projects.py — relevant fields

ProjectType = Literal["promo", "app_review", "training_course", "generic", "lesson"]


class ProjectBase(BaseModel):
    title: str
    project_type: ProjectType  # was content_kind
    brief: dict[str, Any] = Field(default_factory=dict)
    brand_config: dict[str, Any] | None = None
    voice_config: dict[str, Any] | None = None
    theme_config: dict[str, Any] | None = None


class CreateProjectRequest(ProjectBase):
    deliverables: list[CreateDeliverableRequest] | None = None
    clone_settings_from: uuid.UUID | None = None


class ProjectResponse(ProjectBase):
    id: uuid.UUID
    tenant_id: uuid.UUID
    status: ProjectStatus
    parent_project_id: uuid.UUID | None
    position: int | None
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)


class LessonSummary(BaseModel):
    summary_text: str
    key_points: list[str]
    referenced_terms: list[str]


class ProjectDetailResponse(ProjectResponse):
    deliverables: list[DeliverableResponse]
    children: list[ProjectResponse]
    lesson_summary_json: LessonSummary | None = None


class CreateChildProjectRequest(BaseModel):
    project_type: ProjectType
    title: str
    position: int | None = None
    brief: dict[str, Any] = Field(default_factory=dict)
    deliverables: list[CreateDeliverableRequest] | None = None


class ReorderChildrenRequest(BaseModel):
    order: list[uuid.UUID]
```

- [ ] **Step 2: Sanity test (validates parsing)**

```python
# tests/api/test_schemas.py — append
def test_create_project_request_with_clone_settings_from():
    req = CreateProjectRequest(
        title="X", project_type="promo", brief={},
        clone_settings_from=uuid.uuid4()
    )
    assert req.clone_settings_from is not None
```

- [ ] **Step 3: Commit**

```bash
git add src/api/schemas/projects.py tests/api/test_schemas.py
git commit -m "feat(schema): clone_settings_from + children + parent fields"
```

---

### Task 5: Service — clone-on-create, create-child, reorder

**Files:**
- Modify: `src/services/projects/project_service.py`
- Test: extends `tests/api/test_projects_clone.py`, `tests/api/test_projects_children.py`

- [ ] **Step 1: Test the clone path**

```python
# tests/api/test_projects_clone.py
import pytest

@pytest.mark.postgres
async def test_create_with_clone_settings_copies_brand_voice_theme(client, seed):
    src = await seed.project(
        brand_config={"primary": "#abc"},
        voice_config={"narrator": "alex"},
        theme_config={"font": "Inter"},
        brief={"tone": "punchy", "audience": "PMs"},
    )

    resp = await client.post(
        "/api/projects",
        json={
            "title": "New",
            "project_type": "promo",
            "brief": {"key_messages": ["A"]},  # caller-supplied wins on collisions
            "clone_settings_from": str(src.id),
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["brand_config"] == {"primary": "#abc"}
    assert body["voice_config"] == {"narrator": "alex"}
    assert body["theme_config"] == {"font": "Inter"}
    # caller's brief overrides on top of the cloned defaults
    assert body["brief"]["key_messages"] == ["A"]
    assert body["brief"]["tone"] == "punchy"
    assert body["brief"]["audience"] == "PMs"


@pytest.mark.postgres
async def test_create_with_clone_settings_404_on_missing_source(client):
    resp = await client.post(
        "/api/projects",
        json={
            "title": "New", "project_type": "promo", "brief": {},
            "clone_settings_from": "00000000-0000-0000-0000-000000000000",
        },
    )
    assert resp.status_code == 404
```

- [ ] **Step 2: Test the children path**

```python
# tests/api/test_projects_children.py
import pytest

@pytest.mark.postgres
async def test_add_child_appends_to_end(client, seed):
    course = await seed.project(project_type="training_course")
    resp = await client.post(
        f"/api/projects/{course.id}/children",
        json={"project_type": "lesson", "title": "L1", "brief": {}},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["parent_project_id"] == str(course.id)
    assert body["position"] == 1


@pytest.mark.postgres
async def test_reorder_children(client, seed):
    course = await seed.project(project_type="training_course")
    a = await seed.project(project_type="lesson", parent_project_id=course.id, position=1)
    b = await seed.project(project_type="lesson", parent_project_id=course.id, position=2)
    c = await seed.project(project_type="lesson", parent_project_id=course.id, position=3)
    resp = await client.patch(
        f"/api/projects/{course.id}/children/reorder",
        json={"order": [str(c.id), str(a.id), str(b.id)]},
    )
    assert resp.status_code == 204

    detail = (await client.get(f"/api/projects/{course.id}")).json()
    assert [c["id"] for c in detail["children"]] == [str(c.id), str(a.id), str(b.id)]
```

- [ ] **Step 3: Implement**

```python
# src/services/projects/project_service.py — additions
class ProjectService:
    async def create(self, req: CreateProjectRequest, *, tenant_id, subject_id, visibility) -> Project:
        cloned: dict[str, Any] = {}
        if req.clone_settings_from:
            cloned = await self.repo.fetch_settings_for_clone(req.clone_settings_from)
            if cloned is None:
                raise NotFoundError(f"source project {req.clone_settings_from} not found")
        merged_brief = {**(cloned.get("brief_defaults") or {}), **req.brief}
        project = await self.repo.create(
            tenant_id=tenant_id,
            subject_id=subject_id,
            visibility=visibility,
            title=req.title,
            project_type=req.project_type,
            brief=merged_brief,
            brand_config=req.brand_config or cloned.get("brand_config"),
            voice_config=req.voice_config or cloned.get("voice_config"),
            theme_config=req.theme_config or cloned.get("theme_config"),
        )
        if req.deliverables:
            for d in req.deliverables:
                await self.repo.create_deliverable(project.id, d)
        return project

    async def add_child(
        self, parent_id: uuid.UUID, req: CreateChildProjectRequest
    ) -> Project:
        parent = await self.repo.get(parent_id)
        if parent is None:
            raise NotFoundError(f"parent project {parent_id} not found")
        position = req.position
        if position is None:
            siblings = await self.repo.list_children(parent_id)
            position = len(siblings) + 1
        child = await self.repo.create(
            tenant_id=parent.tenant_id,
            subject_id=parent.subject_id,
            visibility=parent.visibility,
            title=req.title,
            project_type=req.project_type,
            brief=req.brief,
            brand_config=parent.brand_config,
            voice_config=parent.voice_config,
            theme_config=parent.theme_config,
            parent_project_id=parent_id,
            position=position,
        )
        if req.deliverables:
            for d in req.deliverables:
                await self.repo.create_deliverable(child.id, d)
        await self.notifications.publish(
            channel=f"project:{parent_id}",
            event="child_added",
            payload={"child": ProjectResponse.model_validate(child).model_dump(mode="json")},
        )
        return child

    async def reorder_children(self, parent_id: uuid.UUID, order: list[uuid.UUID]) -> None:
        existing = {p.id for p in await self.repo.list_children(parent_id)}
        if set(order) != existing:
            raise ValidationError("reorder must list every existing child exactly once")
        await self.repo.reorder_children(parent_id, order)
        await self.notifications.publish(
            channel=f"project:{parent_id}",
            event="child_reordered",
            payload={"order": [str(c) for c in order]},
        )
```

- [ ] **Step 4: Run + commit**

```bash
uv run pytest tests/api/test_projects_clone.py tests/api/test_projects_children.py -v
git add src/services/projects/project_service.py tests/api/test_projects_clone.py tests/api/test_projects_children.py
git commit -m "feat(projects): clone-on-create, add-child, reorder-children services"
```

---

### Task 6: Routes — `/projects` `?type=` filter, children endpoints, drop curricula

**Files:**
- Modify: `src/api/routes/projects.py`
- Delete: `src/api/routes/curricula.py`
- Modify: `src/api/routes/__init__.py`

- [ ] **Step 1: Add child + reorder endpoints**

```python
# src/api/routes/projects.py — additions
@router.get("", response_model=ProjectListResponse)
async def list_projects(
    *,
    type: str | None = None,           # NEW filter
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
    auth: AuthContext = Depends(require_auth),
    repo: ProjectRepository = Depends(get_project_repo),
):
    items = await repo.list(
        tenant_id=auth.tenant_id,
        project_type=type,
        status=status,
        limit=limit,
        offset=offset,
    )
    return ProjectListResponse(items=items, total=len(items))


@router.post("/{project_id}/children", response_model=ProjectDetailResponse, status_code=201)
async def add_child(
    project_id: uuid.UUID,
    req: CreateChildProjectRequest,
    auth: AuthContext = Depends(require_auth),
    service: ProjectService = Depends(get_project_service),
):
    child = await service.add_child(project_id, req)
    return await service.get_detail(child.id)


@router.patch("/{project_id}/children/reorder", status_code=204)
async def reorder_children(
    project_id: uuid.UUID,
    req: ReorderChildrenRequest,
    auth: AuthContext = Depends(require_auth),
    service: ProjectService = Depends(get_project_service),
):
    await service.reorder_children(project_id, req.order)
    return Response(status_code=204)
```

- [ ] **Step 2: Drop the curricula router**

```bash
rm src/api/routes/curricula.py
rm src/api/schemas/curricula.py
rm -rf src/services/projects/curriculum  # if it exists from vast-teapot phase 7
```

```python
# src/api/routes/__init__.py — drop curricula include
# from . import curricula           # DELETE
# router.include_router(curricula.router)   # DELETE
```

- [ ] **Step 3: Run + commit**

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run ty check src/ tests/
uv run pytest tests/api -v
git add -A
git commit -m "feat(api): /projects type filter, children endpoints, drop curricula"
```

---

### Task 7: SSE events — `child_added`, `child_reordered`, `child_status_changed`

**Files:**
- Modify: `src/services/notification_service.py`
- Modify: `src/api/routes/projects.py` (SSE endpoint)

- [ ] **Step 1: Add the new event types to PipelineEvent (or ProjectEvent enum)**

```python
# src/services/notification_service.py
class ProjectEvent(StrEnum):
    CHILD_ADDED = "child_added"
    CHILD_REORDERED = "child_reordered"
    CHILD_STATUS_CHANGED = "child_status_changed"
```

(Reuse the existing redis-pub/sub channel pattern. `channel=f"project:{parent_id}"` is already used by the service in Task 5.)

- [ ] **Step 2: Add a project-level SSE endpoint**

```python
# src/api/routes/projects.py — append
@router.get("/{project_id}/events")
async def project_events(
    project_id: uuid.UUID,
    auth: AuthContext = Depends(require_auth),
    pubsub: NotificationService = Depends(get_notifications),
):
    async def gen():
        async for event in pubsub.subscribe(f"project:{project_id}"):
            yield f"event: {event.name}\ndata: {event.json}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
```

- [ ] **Step 3: Wire `child_status_changed` from project completion**

In whatever path flips a Project's status (existing `pipeline_jobs` completion handler), if the project has a `parent_project_id`, additionally publish:

```python
await self.notifications.publish(
    channel=f"project:{project.parent_project_id}",
    event="child_status_changed",
    payload={"child_id": str(project.id), "status": project.status},
)
```

- [ ] **Step 4: Integration test**

```python
# tests/api/test_projects_sse.py
@pytest.mark.postgres
async def test_child_added_emits_event(client, seed):
    course = await seed.project(project_type="training_course")
    async with client.stream("GET", f"/api/projects/{course.id}/events") as stream:
        # Trigger an add in parallel.
        ...
```

- [ ] **Step 5: Commit**

```bash
git add src/services/notification_service.py src/api/routes/projects.py tests/api/test_projects_sse.py
git commit -m "feat(projects): SSE event stream for child lifecycle"
```

---

### Task 8: Registry — alias `lesson` to training_course DAG

**Files:**
- Modify: `src/services/orchestration/registry.py`

- [ ] **Step 1: Add the alias**

```python
# src/services/orchestration/registry.py
REGISTRY["lesson"] = REGISTRY["training_course"]   # same DAG, same defaults
```

(A lesson Project runs the same training_course DAG. The `lesson_summarize` tail still appends because it keys on the parent presence, not the kind.)

- [ ] **Step 2: Test**

```python
def test_lesson_kind_aliases_training_course():
    from src.services.orchestration.registry import REGISTRY
    assert REGISTRY["lesson"].stages == REGISTRY["training_course"].stages
```

- [ ] **Step 3: Commit**

```bash
git add src/services/orchestration/registry.py tests/services/test_registry.py
git commit -m "feat(registry): lesson kind aliases training_course DAG"
```

---

## Final verification

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run ty check src/ tests/
uv run pytest -v
```

All green. Open PR titled `feat: studio v2 backend collapse — projects tree, clone-from, drop curricula`. CI must pass; smoke-test in test-<sha> namespace must succeed before promote.

After merge, the FE plan `2026-05-07-studio-v2-foundation.md` (in `w2v_frontend/docs/plans/`) is unblocked.
