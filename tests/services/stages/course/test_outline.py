from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

from src.config import clear_settings_cache
from src.services.runs.run_context import RunContext
from src.services.stages.base import StageError
from src.services.stages.course.outline import CourseOutlineStage


def _resp(output: dict) -> WMResponse:
    """Build a WMResponse from an ``output`` dict (cuts mock boilerplate)."""
    return WMResponse(
        output=output,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


def _oneshot_two_sections() -> dict:
    """Full one-shot outline payload with two title+objective sections."""
    return {
        "sections": [
            {"title": "Diagnose", "objective": "Find churn drivers"},
            {"title": "Apply", "objective": "Run the playbook"},
        ],
        "intro_summary": "Cut churn.",
    }


def _two_lesson_ctx() -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "title": "Anti-Churn",
            "brief": {"audience": "founders", "outcome": "Cut churn 20%."},
            "lessons": [
                {"lesson_project_id": "p-1", "title": "L1"},
                {"lesson_project_id": "p-2", "title": "L2"},
            ],
        },
    )


@pytest.mark.asyncio
async def test_outline_returns_sections_with_ids_from_llm() -> None:
    fake = AsyncMock()
    fake.message.return_value = WMResponse(
        output={
            "sections": [
                {"title": "Diagnose", "objective": "Find churn drivers"},
                {"title": "Apply", "objective": "Run the playbook"},
            ],
            "intro_summary": "Cut churn.",
        },
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )
    stage = CourseOutlineStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "title": "Anti-Churn",
            "brief": {"audience": "founders", "outcome": "Cut churn 20%."},
            "lessons": [
                {"lesson_project_id": "p-1", "title": "L1"},
                {"lesson_project_id": "p-2", "title": "L2"},
            ],
        },
    )

    out = await stage.run({}, ctx=ctx)

    assert len(out.sections) == 2
    assert all(s.id for s in out.sections)
    assert out.sections[0].lesson_project_id == "p-1"
    assert out.sections[1].lesson_project_id == "p-2"
    assert out.intro_summary == "Cut churn."


def _ok_resp() -> WMResponse:
    return WMResponse(
        output={
            "sections": [{"title": "Diagnose", "objective": "Find churn drivers"}],
            "intro_summary": "Cut churn.",
        },
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


@pytest.mark.asyncio
async def test_outline_grounds_in_source_ingest_when_present() -> None:
    # When the operator grounded the course in extra sources, the outline prompt
    # carries the SOURCE MATERIAL block from source_ingest.
    fake = AsyncMock()
    fake.message.return_value = _ok_resp()
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"title": "x", "brief": {}, "lessons": []},
        prior_run_outputs={
            "source_ingest": {"grounding_text": "Churn drivers: pricing, onboarding."}
        },
    )
    await CourseOutlineStage(client=fake).run({}, ctx=ctx)
    _, kwargs = fake.message.call_args
    assert "SOURCE MATERIAL" in kwargs["message"]
    assert "Churn drivers: pricing, onboarding." in kwargs["message"]


@pytest.mark.asyncio
async def test_outline_prompt_unchanged_without_source_ingest() -> None:
    # Plain-run-safe: no source_ingest output → no grounding block appended.
    fake = AsyncMock()
    fake.message.return_value = _ok_resp()
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"title": "x", "brief": {}, "lessons": []},
    )
    await CourseOutlineStage(client=fake).run({}, ctx=ctx)
    _, kwargs = fake.message.call_args
    assert "SOURCE MATERIAL" not in kwargs["message"]


@pytest.mark.asyncio
async def test_outline_raises_on_malformed_llm() -> None:
    fake = AsyncMock()
    fake.message.return_value = WMResponse(
        output={},
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )
    stage = CourseOutlineStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "title": "x",
            "brief": {"outcome": "fallback intro"},
            "lessons": [
                {
                    "lesson_project_id": "p-1",
                    "title": "L1",
                    "brief": {"learning_objective": "obj1"},
                },
            ],
        },
    )

    with pytest.raises(StageError):
        await stage.run({}, ctx=ctx)


@pytest.mark.asyncio
async def test_oneshot_derives_missing_objective_from_lesson() -> None:
    """Small models emit section titles but often drop the per-section
    'objective'. The one-shot path must derive it from the authored lesson
    content instead of hard-failing a paid render."""
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "sections": [{"title": "Diagnose"}, {"title": "Apply"}],  # no 'objective'
            "intro_summary": "Cut churn.",
        }
    )
    out = await CourseOutlineStage(client=fake).run({}, ctx=_two_lesson_ctx())
    assert [s.title for s in out.sections] == ["Diagnose", "Apply"]
    # No lesson learning_objective set → derived from each lesson's authored title.
    assert [s.objective for s in out.sections] == ["L1", "L2"]
    assert out.intro_summary == "Cut churn."


@pytest.mark.asyncio
async def test_oneshot_tolerates_missing_intro_summary() -> None:
    """A missing intro_summary falls back to the course title, not a hard fail."""
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {"sections": [{"title": "Diagnose", "objective": "o1"}]}  # no intro_summary
    )
    out = await CourseOutlineStage(client=fake).run({}, ctx=_two_lesson_ctx())
    assert out.sections[0].objective == "o1"
    assert out.intro_summary  # non-empty, derived from the course title


@pytest.mark.asyncio
async def test_outline_split_off_byte_identical() -> None:
    # Flag unset (default OFF) → one-shot path: exactly one WM call, full payload.
    clear_settings_cache()
    fake = AsyncMock()
    fake.message.return_value = _resp(_oneshot_two_sections())
    stage = CourseOutlineStage(client=fake)
    ctx = _two_lesson_ctx()

    out = await stage.run({}, ctx=ctx)

    assert fake.message.call_count == 1
    assert len(out.sections) == 2
    assert [s.title for s in out.sections] == ["Diagnose", "Apply"]
    assert [s.objective for s in out.sections] == [
        "Find churn drivers",
        "Run the playbook",
    ]
    assert out.sections[0].lesson_project_id == "p-1"
    assert out.sections[1].lesson_project_id == "p-2"
    assert out.intro_summary == "Cut churn."


@pytest.mark.asyncio
async def test_outline_split_on_stitches_two_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Flag ON → structure call, then objectives call, deterministically stitched.
    monkeypatch.setenv("PIPELINE_COURSE_OUTLINE_SPLIT", "true")
    clear_settings_cache()
    fake = AsyncMock()
    fake.message.side_effect = [
        _resp(
            {
                "sections": [{"title": "Diagnose"}, {"title": "Apply"}],
                "intro_summary": "Cut churn.",
            }
        ),
        _resp({"objectives": ["Find churn drivers", "Run the playbook"]}),
    ]
    stage = CourseOutlineStage(client=fake)
    ctx = _two_lesson_ctx()

    out = await stage.run({}, ctx=ctx)

    assert fake.message.call_count == 2
    assert [s.title for s in out.sections] == ["Diagnose", "Apply"]
    assert [s.objective for s in out.sections] == [
        "Find churn drivers",
        "Run the playbook",
    ]
    assert out.sections[0].lesson_project_id == "p-1"
    assert out.sections[1].lesson_project_id == "p-2"
    assert all(s.id for s in out.sections)
    assert out.intro_summary == "Cut churn."


@pytest.mark.asyncio
async def test_outline_split_empty_structure_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Flag ON, structure well-formed but EMPTY sections → NO retry, immediate
    # one-shot fallback (structure call + one-shot call == 2 calls).
    monkeypatch.setenv("PIPELINE_COURSE_OUTLINE_SPLIT", "true")
    clear_settings_cache()
    fake = AsyncMock()
    fake.message.side_effect = [
        _resp({"sections": [], "intro_summary": "x"}),
        _resp(_oneshot_two_sections()),
    ]
    stage = CourseOutlineStage(client=fake)
    ctx = _two_lesson_ctx()

    out = await stage.run({}, ctx=ctx)

    assert fake.message.call_count == 2
    assert [s.title for s in out.sections] == ["Diagnose", "Apply"]
    assert [s.objective for s in out.sections] == [
        "Find churn drivers",
        "Run the playbook",
    ]
    assert out.intro_summary == "Cut churn."


@pytest.mark.asyncio
async def test_outline_split_malformed_reprompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Flag ON, first structure draw {} MISSING keys → raises ValueError inside
    # the budget → re-prompts (retries=1) → 2nd valid structure, then objectives.
    monkeypatch.setenv("PIPELINE_COURSE_OUTLINE_SPLIT", "true")
    clear_settings_cache()
    fake = AsyncMock()
    fake.message.side_effect = [
        _resp({}),
        _resp({"sections": [{"title": "A"}], "intro_summary": "i"}),
        _resp({"objectives": ["obj-A"]}),
    ]
    stage = CourseOutlineStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "title": "x",
            "brief": {"outcome": "fallback intro"},
            "lessons": [
                {
                    "lesson_project_id": "p-1",
                    "title": "L1",
                    "brief": {"learning_objective": "obj1"},
                },
            ],
        },
    )

    out = await stage.run({}, ctx=ctx)

    assert fake.message.call_count == 3
    assert len(out.sections) == 1
    assert out.sections[0].title == "A"
    assert out.sections[0].objective == "obj-A"


@pytest.mark.asyncio
async def test_outline_split_objectives_mismatch_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Flag ON, structure gives 2 titles but objectives returns 1 → count
    # mismatch → one-shot fallback (structure + objectives + one-shot == 3).
    monkeypatch.setenv("PIPELINE_COURSE_OUTLINE_SPLIT", "true")
    clear_settings_cache()
    fake = AsyncMock()
    fake.message.side_effect = [
        _resp(
            {
                "sections": [{"title": "Diagnose"}, {"title": "Apply"}],
                "intro_summary": "Cut churn.",
            }
        ),
        _resp({"objectives": ["only-one"]}),
        _resp(_oneshot_two_sections()),
    ]
    stage = CourseOutlineStage(client=fake)
    ctx = _two_lesson_ctx()

    out = await stage.run({}, ctx=ctx)

    assert fake.message.call_count == 3
    assert [s.title for s in out.sections] == ["Diagnose", "Apply"]
    assert [s.objective for s in out.sections] == [
        "Find churn drivers",
        "Run the playbook",
    ]
    assert out.intro_summary == "Cut churn."


@pytest.mark.asyncio
async def test_split_objectives_missing_key_falls_back_to_oneshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Flag ON, structure OK, but the objectives draw returns a dict MISSING the
    # 'objectives' key (schema not decoder-enforced on the split path). This must
    # NOT raise — it normalizes to [] → length mismatch → one-shot fallback.
    monkeypatch.setenv("PIPELINE_COURSE_OUTLINE_SPLIT", "true")
    clear_settings_cache()
    fake = AsyncMock()
    fake.message.side_effect = [
        _resp(
            {
                "sections": [{"title": "Diagnose"}, {"title": "Apply"}],
                "intro_summary": "Cut churn.",
            }
        ),
        _resp({"foo": ["not objectives"]}),
        _resp(_oneshot_two_sections()),
    ]
    stage = CourseOutlineStage(client=fake)
    ctx = _two_lesson_ctx()

    out = await stage.run({}, ctx=ctx)

    assert fake.message.call_count == 3
    assert [s.title for s in out.sections] == ["Diagnose", "Apply"]
    assert [s.objective for s in out.sections] == [
        "Find churn drivers",
        "Run the playbook",
    ]
    assert out.intro_summary == "Cut churn."


@pytest.mark.asyncio
async def test_split_objectives_array_root_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Flag ON, objectives draw returns a BARE LIST (array-root) matching the
    # title count → used directly, no one-shot fallback (2 calls total).
    monkeypatch.setenv("PIPELINE_COURSE_OUTLINE_SPLIT", "true")
    clear_settings_cache()
    fake = AsyncMock()
    fake.message.side_effect = [
        _resp(
            {
                "sections": [{"title": "Diagnose"}, {"title": "Apply"}],
                "intro_summary": "Cut churn.",
            }
        ),
        WMResponse(
            output=["Find churn drivers", "Run the playbook"],
            output_type="array",
            output_schema_used=None,
            metadata=None,
            usage_metrics=None,
        ),
    ]
    stage = CourseOutlineStage(client=fake)
    ctx = _two_lesson_ctx()

    out = await stage.run({}, ctx=ctx)

    assert fake.message.call_count == 2
    assert [s.title for s in out.sections] == ["Diagnose", "Apply"]
    assert [s.objective for s in out.sections] == [
        "Find churn drivers",
        "Run the playbook",
    ]
    assert out.intro_summary == "Cut churn."


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [["Diagnose", "Apply"]])
async def test_outline_split_non_object_structure_reprompts(
    monkeypatch: pytest.MonkeyPatch, bad: object
) -> None:
    # Flag ON, first structure draw is NOT an object (array-root / null). It must
    # re-prompt inside the budget like a malformed dict, not AttributeError out.
    monkeypatch.setenv("PIPELINE_COURSE_OUTLINE_SPLIT", "true")
    clear_settings_cache()
    fake = AsyncMock()
    fake.message.side_effect = [
        WMResponse(
            output=bad,
            output_type="object",
            output_schema_used=None,
            metadata=None,
            usage_metrics=None,
        ),
        _resp({"sections": [{"title": "Diagnose"}, {"title": "Apply"}], "intro_summary": "i"}),
        _resp({"objectives": ["Find churn drivers", "Run the playbook"]}),
    ]
    out = await CourseOutlineStage(client=fake).run({}, ctx=_two_lesson_ctx())
    assert fake.message.call_count == 3
    assert [s.title for s in out.sections] == ["Diagnose", "Apply"]
