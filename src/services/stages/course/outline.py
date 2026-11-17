"""course_outline — outline the course flow from the brief + lessons list."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, ClassVar, cast
from uuid import uuid7

from w2v_common.stages import stage_call

from src.services.llm import WMClient
from src.services.stages.base import SlotUse, StageHandler, StageInput, StageOutput
from src.services.stages.generic.grounding import grounding_prompt_block
from src.services.stages.llm_budget import (
    LLM_RETRY_ON,
    course_outline_split,
    downshift_reasoning,
    llm_budget_seconds,
    llm_client_timeout_seconds,
    llm_retries,
    output_token_budget,
)
from src.services.stages.upstream import guarded

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

# WM worker name (regex ^[a-zA-Z0-9_-]+$). WM auto-creates it on first call and
# the per-call system_prompt steers it; the provider/model lives on the WM
# worker config, NOT here — that's the whole point of routing through WM.
_WORKER_NAME = "course-outline"

# --- Per-call OUTPUT token budgets -------------------------------------------
# Omitting ``max_tokens`` leaves WM's 6000 default, which a REASONING model can
# spend entirely on thinking before emitting any JSON -> WM 422 ->
# PermanentUpstreamError -> terminal stage failure. ``output_token_budget`` adds
# the reasoning headroom on top of the ANSWER estimate below and caps at the
# router's chain deadline; the mechanism lives in ``stages/llm_budget.py``.
#
# Sizing input: a course is one section per lesson, and 20 is well above the
# lesson counts this pipeline authors.
_OUTLINE_MAX_SECTIONS = 20
# One ``{"title": ..., "objective": ...}`` entry: a short title (~15) plus a
# one-sentence objective (~40) plus keys/quotes/commas.
_TOKENS_PER_SECTION = 70
# Title-only entry on the split path's structure call.
_TOKENS_PER_TITLE = 30
# A bare objective string in the split path's objectives array.
_TOKENS_PER_OBJECTIVE = 45
# ``intro_summary`` — a short paragraph.
_INTRO_SUMMARY_TOKENS = 150

# One-shot draw: full sections (title + objective) + intro_summary.
_OUTLINE_ANSWER_TOKENS = _OUTLINE_MAX_SECTIONS * _TOKENS_PER_SECTION + _INTRO_SUMMARY_TOKENS
_OUTLINE_MAX_TOKENS = output_token_budget(answer_tokens=_OUTLINE_ANSWER_TOKENS)

# Split path, call 1: titles + intro_summary only — a materially smaller answer.
_STRUCTURE_ANSWER_TOKENS = _OUTLINE_MAX_SECTIONS * _TOKENS_PER_TITLE + _INTRO_SUMMARY_TOKENS
_STRUCTURE_MAX_TOKENS = output_token_budget(answer_tokens=_STRUCTURE_ANSWER_TOKENS)

# Split path, call 2: one objective string per title, nothing else.
_OBJECTIVES_ANSWER_TOKENS = _OUTLINE_MAX_SECTIONS * _TOKENS_PER_OBJECTIVE
_OBJECTIVES_MAX_TOKENS = output_token_budget(answer_tokens=_OBJECTIVES_ANSWER_TOKENS)

# JSON Schema literals live at the WM structured-output boundary — the one place
# ``dict[str, Any]`` is unavoidable (a JSON Schema is not a pydantic model). One
# alias names that boundary and keeps every schema constant typed the same way.
_JsonSchema = dict[str, Any]

_SYSTEM_PROMPT = (
    "You are a course architect. Outline the course flow from the brief and "
    "the lessons list. Return only JSON matching the schema."
)

# Object-rooted structured-output schema (WM convention). WM returns
# ``response.output`` already parsed to this shape.
_OUTPUT_SCHEMA: _JsonSchema = {
    "type": "object",
    "properties": {
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "objective": {"type": "string"},
                },
                "required": ["title", "objective"],
            },
        },
        "intro_summary": {"type": "string"},
    },
    "required": ["sections", "intro_summary"],
}

# --- Pillar-2 vLLM split (PIPELINE_COURSE_OUTLINE_SPLIT) -----------------------
# Two small draws instead of one big one on the single slow GB10 vLLM: a
# structure call (section titles + intro_summary) then a 1:1 objective batch,
# stitched deterministically. Falls back to the one-shot path above on an
# empty/malformed structure or an objectives-vs-titles length mismatch.
_STRUCTURE_PROMPT = (
    "You are a course architect. Propose the section titles for the course "
    "flow from the brief and lessons list. Return only JSON matching the schema."
)
_FILL_PROMPT = (
    "You are a course architect. For each given section title, write a single "
    "concrete learning objective grounded in the brief and that section's "
    "lesson. Return only JSON: an 'objectives' array aligned 1:1 with the "
    "titles in order."
)
_STRUCTURE_SCHEMA: _JsonSchema = {
    "type": "object",
    "properties": {
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
            },
        },
        "intro_summary": {"type": "string"},
    },
    "required": ["sections", "intro_summary"],
}
_OBJECTIVES_SCHEMA: _JsonSchema = {
    "type": "object",
    "properties": {"objectives": {"type": "array", "items": {"type": "string"}}},
    "required": ["objectives"],
}


def _extract_objectives(output: object) -> list[str]:
    """Read the objectives list defensively from a WM objectives-fill reply.

    On the split path the object-root schema is not always decoder-enforced, so
    ``resp.output`` can arrive either as a bare list (array-root — that IS the
    objectives) or as a dict without an ``objectives`` list. Return whatever
    usable list of strings is present, else ``[]``. An empty result is SAFE: the
    caller's ``len(objectives) != len(titles)`` guard then routes to a real
    one-shot WM draw rather than raising (matches the read-side normalization
    idiom in reconcile_timeline / app_review critique).
    """
    if isinstance(output, list):
        return [o for o in output if isinstance(o, str)]
    if isinstance(output, dict):
        found = cast("dict[str, object]", output).get("objectives")
        if isinstance(found, list):
            return [o for o in found if isinstance(o, str)]
    return []


def _derive_objective(title: str, lesson: object) -> str:
    """Grounded fallback objective when the model omits the per-section one.

    Small models reliably emit section titles but often drop ``objective``.
    Rather than hard-fail a paid render, prefer the lesson's authored learning
    objective (the SAME grounding the objectives-fill prompt uses as a hint),
    else the lesson title, else the section title. Never fabricates unrelated
    content — every fallback is real authored/model output already present.
    """
    if isinstance(lesson, dict):
        lesson_d = cast("dict[str, object]", lesson)
        brief = lesson_d.get("brief")
        derived = (
            cast("dict[str, object]", brief).get("learning_objective")
            if isinstance(brief, dict)
            else None
        )
        derived = derived or lesson_d.get("title")
        if isinstance(derived, str) and derived.strip():
            return derived.strip()
    return title


class CourseOutlineInput(StageInput):
    pass


class CourseSection(StageOutput):
    id: str
    lesson_project_id: str | None
    title: str
    objective: str


class CourseOutlineOutput(StageOutput):
    sections: list[CourseSection]
    intro_summary: str


class CourseOutlineStage(StageHandler[CourseOutlineInput, CourseOutlineOutput]):
    name = "course_outline"
    input_model = CourseOutlineInput
    output_model = CourseOutlineOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("llm"),)

    def __init__(self, client: WMClient | None = None) -> None:
        self._llm_timeout = llm_budget_seconds()
        self._llm_retries = llm_retries()
        self._wm = client or WMClient(timeout=llm_client_timeout_seconds())

    async def handle(
        self,
        payload: CourseOutlineInput,
        ctx: RunContext,
    ) -> CourseOutlineOutput:
        cfg = ctx.config_snapshot
        lessons = list(cfg.get("lessons") or [])
        brief = cfg.get("brief") or {}
        prompt = (
            f"Outline the course '{cfg.get('title')}'. "
            f"Audience: {brief.get('audience')}. "
            f"Outcome: {brief.get('outcome')}. "
            f"Lessons: {[lesson.get('title') for lesson in lessons]}. "
            'Output JSON: {"sections":[{"title":"...","objective":"..."}, ...],'
            '"intro_summary":"..."}.' + grounding_prompt_block(ctx)
        )

        async def _oneshot_attempt(enable_reasoning: bool | None) -> CourseOutlineOutput:
            # Shape-validate inside the retry budget so a reply missing a section
            # field is re-prompted, not run-fatal on first draw.
            resp = await self._wm.message(
                enable_reasoning=enable_reasoning,
                worker_name=_WORKER_NAME,
                message=prompt,
                system_prompt=_SYSTEM_PROMPT,
                output_schema=_OUTPUT_SCHEMA,
                max_tokens=_OUTLINE_MAX_TOKENS,
            )
            data = resp.output
            # A titled section is the load-bearing signal; a missing per-section
            # 'objective' is common on small models and is derived from grounded
            # authored content rather than re-prompted forever (which never helps
            # a model that structurally can't emit it). No usable section at all
            # IS re-promptable, so raise a retryable ValueError there.
            if not isinstance(data, dict) or not isinstance(data.get("sections"), list):
                raise ValueError("course_outline: LLM output missing 'sections' list")
            raw = [
                s
                for s in data["sections"]
                if isinstance(s, dict) and str(s.get("title") or "").strip()
            ]
            if not raw:
                raise ValueError("course_outline: LLM output had no usable titled sections")
            sections = [
                CourseSection(
                    id=str(uuid7()),
                    lesson_project_id=(
                        lessons[i].get("lesson_project_id") if i < len(lessons) else None
                    ),
                    title=str(s["title"]).strip(),
                    objective=(
                        str(s.get("objective") or "").strip()
                        or _derive_objective(
                            str(s["title"]).strip(),
                            lessons[i] if i < len(lessons) else None,
                        )
                    ),
                )
                for i, s in enumerate(raw)
            ]
            intro = data.get("intro_summary")
            intro_summary = (
                intro.strip()
                if isinstance(intro, str) and intro.strip()
                else str(cfg.get("title") or "Course overview")
            )
            return CourseOutlineOutput(sections=sections, intro_summary=intro_summary)

        if not course_outline_split():
            return await stage_call(
                # guarded(): a WM 402 "configure your LLM provider" is permanent — fail
                # with that sentence instead of burning the budget on a retry that cannot
                # help. LLM_RETRY_ON keeps the ValueError re-prompt above retryable.
                downshift_reasoning(lambda think: guarded(lambda: _oneshot_attempt(think))),
                timeout_seconds=self._llm_timeout,
                retries=self._llm_retries,
                retry_on=LLM_RETRY_ON,
                stage_name="course_outline",
            )
        return await self._split(prompt, lessons, _oneshot_attempt)

    async def _oneshot(
        self, attempt: Callable[[bool | None], Awaitable[CourseOutlineOutput]]
    ) -> CourseOutlineOutput:
        """Run the one-shot path via a fresh WM draw (never fabricate a result)."""
        return await stage_call(
            downshift_reasoning(lambda think: guarded(lambda: attempt(think))),
            timeout_seconds=self._llm_timeout,
            retries=self._llm_retries,
            retry_on=LLM_RETRY_ON,
            stage_name="course_outline",
        )

    async def _split(
        self,
        prompt: str,
        lessons: list[Any],
        oneshot_attempt: Callable[[bool | None], Awaitable[CourseOutlineOutput]],
    ) -> CourseOutlineOutput:
        """Two small draws (structure, then objective batch) stitched 1:1.

        Each draw runs through its own guarded stage_call so a small malformed
        reply is re-prompted inside the same budget as the one-shot path. Any
        structural degradation (empty structure, length mismatch) falls back to
        a real one-shot WM draw rather than fabricating an outline.
        """

        async def _structure_attempt(enable_reasoning: bool | None) -> tuple[list[str], str]:
            # ``prompt`` already carries grounding_prompt_block(ctx), so grounding
            # is preserved on the structure call.
            resp = await self._wm.message(
                enable_reasoning=enable_reasoning,
                worker_name=_WORKER_NAME,
                message=prompt,
                system_prompt=_STRUCTURE_PROMPT,
                output_schema=_STRUCTURE_SCHEMA,
                max_tokens=_STRUCTURE_MAX_TOKENS,
            )
            data = resp.output
            if not isinstance(data, dict):
                # A non-object reply (bare list / null) would AttributeError on
                # .get below — outside retry_on — instead of re-prompting.
                raise ValueError("course_outline: structure output is not an object")
            if not isinstance(data.get("sections"), list) or not isinstance(
                data.get("intro_summary"), str
            ):
                raise ValueError("course_outline: structure output missing sections/intro_summary")
            titles = [
                str(s["title"]) for s in data["sections"] if isinstance(s, dict) and s.get("title")
            ]
            return titles, str(data["intro_summary"])

        titles, intro_summary = await stage_call(
            downshift_reasoning(lambda think: guarded(lambda: _structure_attempt(think))),
            timeout_seconds=self._llm_timeout,
            retries=self._llm_retries,
            retry_on=LLM_RETRY_ON,
            stage_name="course_outline",
        )

        # Well-formed but empty structure ⇒ re-run the one-shot path (real WM draw).
        if not titles:
            return await self._oneshot(oneshot_attempt)

        async def _objectives_attempt(enable_reasoning: bool | None) -> list[str]:
            lines: list[str] = []
            for i, title in enumerate(titles):
                hint = ""
                if i < len(lessons):
                    hint = (
                        lessons[i].get("brief", {}).get("learning_objective")
                        or lessons[i].get("title")
                        or ""
                    )
                lines.append(f"{i + 1}. {title}" + (f" (lesson: {hint})" if hint else ""))
            fill_msg = (
                "Write one concrete learning objective per section title, in "
                "the same order:\n" + "\n".join(lines)
            )
            resp = await self._wm.message(
                enable_reasoning=enable_reasoning,
                worker_name=_WORKER_NAME,
                message=fill_msg,
                system_prompt=_FILL_PROMPT,
                output_schema=_OBJECTIVES_SCHEMA,
                max_tokens=_OBJECTIVES_MAX_TOKENS,
            )
            return _extract_objectives(resp.output)

        objectives = await stage_call(
            downshift_reasoning(lambda think: guarded(lambda: _objectives_attempt(think))),
            timeout_seconds=self._llm_timeout,
            retries=self._llm_retries,
            retry_on=LLM_RETRY_ON,
            stage_name="course_outline",
        )

        # 1:1 alignment is load-bearing; a mismatch ⇒ one-shot fallback.
        if len(objectives) != len(titles):
            return await self._oneshot(oneshot_attempt)

        sections = [
            CourseSection(
                id=str(uuid7()),
                lesson_project_id=(
                    lessons[i].get("lesson_project_id") if i < len(lessons) else None
                ),
                title=titles[i],
                objective=objectives[i],
            )
            for i, _ in enumerate(titles)
        ]
        return CourseOutlineOutput(sections=sections, intro_summary=intro_summary)
