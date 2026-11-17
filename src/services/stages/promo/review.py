"""promo_review — rubric-scored evaluator.

Emits an ``overall_score`` plus per-criterion ``items`` with stable ids
(PR-6 annotations key off these). The criteria are inlined here rather
than imported from ``src/services/evaluation/rubrics/``: that legacy
per-stage rubric family targeted the old review/critique stages with a
different protocol (``evaluate(artifact_uris, stage_execution) ->
list[MetricResult]``) and has since been retired outright.

The user reviews the score; if low they trigger regenerate via PR-6.
A *degenerate* review (see :func:`reviewer_quality_notes`) is a different
problem from a low one, and is surfaced via the ``notes`` degradation
channel rather than by failing the run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar, cast
from uuid import uuid7

from w2v_common.stages import StageRetriesExhaustedError, stage_call

from src.services.llm import WMClient
from src.services.stages.base import (
    SlotUse,
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
    StagePreconditionError,
)
from src.services.stages.llm_budget import (
    LLM_RETRY_ON,
    MEDIUM_ANSWER_TOKENS,
    downshift_reasoning,
    llm_budget_seconds,
    llm_client_timeout_seconds,
    llm_retries,
    output_token_budget,
)
from src.services.stages.shared.degradation import PROMO_REVIEW_EMPTY, degradation
from src.services.stages.shared.placeholder_text import is_placeholder_text, text_or_empty
from src.services.stages.upstream import guarded

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

# WM worker name (regex ^[a-zA-Z0-9_-]+$); provider/model live on the WM worker
# config, not here. WM returns ``resp.output`` already parsed to the schema.
_WORKER_NAME = "promo-review"

_SYSTEM_PROMPT = (
    "You are a promo-video reviewer. Score the promo on each given criterion "
    "(1-5) with a short rationale and an overall summary. Return only JSON "
    "matching the schema."
)

_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "criterion": {"type": "string"},
                    "score": {"type": "integer"},
                    "rationale": {"type": "string"},
                },
                "required": ["criterion", "score"],
            },
        },
        "summary": {"type": "string"},
        # Structured, NOT prose: a clean review legitimately writes "no 'None'
        # found", so a regex over rationales failed healthy runs. Only this
        # boolean (plus the deterministic narration check) can fail the stage.
        "placeholder_text_visible": {
            "type": "boolean",
            "description": (
                "true ONLY if placeholder text (None, null, undefined, N/A) is actually "
                "spoken in the narration or shown on a slide; false otherwise"
            ),
        },
        "placeholder_text_evidence": {"type": "string"},
    },
    "required": ["items"],
}


# Reasoning-model output budget (see stages/llm_budget.py — a FLOOR, not a cap).
# ``{items[], summary}`` — a FIXED five criteria (PROMO_RUBRIC_CRITERIA), each a
# score plus a short prose rationale, and one overall summary paragraph. A handful
# of objects with prose fields ⇒ MEDIUM.
_REVIEW_MAX_TOKENS = output_token_budget(answer_tokens=MEDIUM_ANSWER_TOKENS)

PROMO_RUBRIC_CRITERIA: tuple[str, ...] = (
    "clarity",
    "hook_strength",
    "cta_presence",
    "brand_voice_fit",
    "pacing",
)


class PromoReviewInput(StageInput):
    pass


class PromoReviewItem(StageOutput):
    id: str
    criterion: str
    score: int  # 1-5
    rationale: str


class PromoReviewOutput(StageOutputWithNotes):
    # `items` is the contract, not `overall_score`: the score is DERIVED from
    # items (`sum(...) / max(1, len(items))`), so an empty review reports a
    # fabricated 0.0 rather than nothing, and the degenerate-reviewer detector
    # has nothing to flag. `runs/elements.py` also pins annotations to item ids.
    produces: ClassVar[tuple[str, ...]] = ("items",)
    # Placeholder text in a customer's video ("None", "null", …) is not a low
    # score, it is a broken deliverable: a run whose last line was a spoken
    # "None" was scored cta_presence 2/5 and published. The flag hard-fails this
    # stage through the executor's one failure-flag gate, which also blocks
    # `growth_bundle` (it depends on promo_review) from publishing it.
    fails_when_flagged: ClassVar[tuple[str, ...]] = ("placeholder_text_found",)

    overall_score: float
    items: list[PromoReviewItem]
    summary: str
    placeholder_text_found: bool = False


# --------------------------------------------------------------------------- #
# Degenerate-reviewer detection
# --------------------------------------------------------------------------- #
#
# Ported from the retired ``evaluation/rubrics/review.py`` (``_score_discrimination``
# + ``_na_check``). The idea worth keeping: a reviewer LLM that hands every
# criterion the same score, or fills every rationale with "N/A", has not actually
# reviewed anything — and that is invisible in ``overall_score`` alone (a uniform
# 4/5 looks like a healthy 4.0). This is a *reviewer-quality* signal, not a
# product-quality one, so it degrades the run rather than failing it: the score
# is untrustworthy, not the promo.

_NA_RATIONALES = frozenset(
    {"", "n/a", "na", "none", "null", "not applicable", "not available", "-"}
)
# Scores are ints on a 1-5 scale, so any genuine differentiation across >=2
# criteria yields a stddev of at least 0.5. Anything under this is flat.
_MIN_SCORE_STDDEV = 0.25
_MAX_NA_RATIONALE_RATIO = 0.5


def _stddev(values: Sequence[int]) -> float:
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def reviewer_quality_notes(items: Sequence[PromoReviewItem]) -> list[str]:
    """Flag a review whose own scores carry no information.

    Returns human-readable degradation notes (empty == healthy review, so the
    run is byte-identical to before when the reviewer behaves).
    """
    notes: list[str] = []
    if not items:
        return notes

    scores = [item.score for item in items]
    if len(scores) >= 2:
        spread = _stddev(scores)
        if spread < _MIN_SCORE_STDDEV:
            notes.append(
                f"promo_review: no score discrimination — all {len(scores)} criteria scored "
                f"{scores[0]} (stddev {spread:.2f} < {_MIN_SCORE_STDDEV}); "
                "the reviewer did not differentiate, treat overall_score as unreliable"
            )

    na_count = sum(1 for item in items if item.rationale.strip().lower() in _NA_RATIONALES)
    if na_count / len(items) > _MAX_NA_RATIONALE_RATIO:
        notes.append(
            f"promo_review: {na_count}/{len(items)} rationales are empty or N/A; "
            "the reviewer returned scores without justification"
        )
    return notes


def placeholder_findings(
    script_lines: Sequence[Mapping[str, object]],
    *,
    reviewer_saw_placeholder: bool = False,
    reviewer_evidence: str = "",
) -> list[str]:
    """Every sign that placeholder text reached the video, as failure notes.

    - PRIMARY, deterministic: a narration line whose text IS a placeholder. The
      script stage drops these at parse time, but an authored-shotlist run
      bypasses that parse, so the review re-checks what is actually shipped.
    - The reviewer's STRUCTURED ``placeholder_text_visible`` boolean — never its
      prose, which on a clean video says things like "no 'None' found".
    """
    notes: list[str] = []
    for index, line in enumerate(script_lines):
        if "text" in line and is_placeholder_text(line.get("text")):
            notes.append(
                f"promo_review: narration line {index + 1} is placeholder text "
                f"{str(line.get('text'))!r} — it would be spoken in the video"
            )
    if reviewer_saw_placeholder:
        evidence = text_or_empty(reviewer_evidence)
        notes.append(
            "promo_review: reviewer reports placeholder text visible in the video"
            + (f": {evidence}" if evidence else "")
        )
    return notes


#: (items, summary, reviewer saw placeholder text, the text it saw)
type _Shaped = tuple[list[PromoReviewItem], str, bool, str]


class _EmptyReviewError(ValueError):
    """The reviewer answered with no scores. A ValueError, so it is re-prompted
    inside the stage's retry budget like any other malformed reply."""


#: What the run page shows when the reviewer never scored the video. Prod run
#: 01a17e6f: the reviewer replied ``{"items": [], "summary": "No promo content
#: provided to review."}`` to a 5-line script, and the stage reported success
#: with a fabricated 0.0 and an "Unclassified note".
_EMPTY_REVIEW_DETAIL = (
    "The reviewer returned no scores for this video, so it was not reviewed. "
    "Rerun this step to review it."
)


def _narration(script: Mapping[str, object]) -> list[str]:
    """The spoken lines the review scores (blank lines dropped)."""
    lines = script.get("lines")
    if not isinstance(lines, list):
        return []
    return [
        text
        for line in lines
        if isinstance(line, Mapping) and (text := str(line.get("text") or "").strip())
    ]


#: The brief fields a reviewer scores against. A WHITELIST, not the whole brief:
#: prod run 01a17e6f sent the entire brief, which put the site sign-in (username,
#: sealed password token, an OIDC URL with state/nonce) into an LLM prompt, and
#: buried the 5-line script under a duplicate shotlist, sources, render settings
#: and a 1.7k-char value proposition. That reviewer answered "No promo content
#: provided" with zero scores.
_REVIEW_BRIEF_FIELDS: tuple[str, ...] = (
    "title",
    "product_name",
    "audience",
    "value_proposition",
    "call_to_action",
    "cta_url",
    "tone",
    "angle",
    "instructions",
    "target_duration_seconds",
)


def _present_fields(brief: Mapping[str, object]) -> dict[str, object]:
    """The review-relevant brief fields, without absent/placeholder values.

    Rendering ``{'call_to_action': None}`` into the prompt invites the reviewer
    to write about a "None" CTA on a brief that simply has none. Never a login.
    """
    return {
        k: v
        for k in _REVIEW_BRIEF_FIELDS
        if k in brief
        and not (
            (v := brief[k]) is None
            or (isinstance(v, str) and is_placeholder_text(v))
            or v in ({}, [])
        )
    }


class PromoReviewStage(StageHandler[PromoReviewInput, PromoReviewOutput]):
    name = "promo_review"
    input_model = PromoReviewInput
    output_model = PromoReviewOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("llm"),)

    def __init__(self, client: WMClient | None = None) -> None:
        self._llm_timeout = llm_budget_seconds()
        self._llm_retries = llm_retries()
        self._wm = client or WMClient(timeout=llm_client_timeout_seconds())

    async def handle(
        self,
        payload: PromoReviewInput,
        ctx: RunContext,
    ) -> PromoReviewOutput:
        render = ctx.prior_output("promo_render")
        script = ctx.prior_output("promo_script")
        if not (render and script):
            raise RuntimeError("render + script required for review")
        if not _narration(script):
            # Nothing to review is a missing input, not a review: never let the
            # reviewer "pass" an empty script.
            raise StagePreconditionError(
                "promo_review: the script has no narration to review — fix the script "
                "step and rerun from there",
                missing=["promo_script.lines"],
            )
        # The delivered artifact is the footage-critique terminal output (best re-render,
        # or promo_render passed through unchanged when critique wasn't requested). Fall
        # back to promo_render's own URL if the critique output is somehow absent.
        critique = ctx.prior_output("promo_footage_critique") or {}
        delivered_url = str(critique.get("artifact_url") or render.get("artifact_url") or "")
        prompt = self._build_prompt(script, ctx.config_snapshot.get("brief") or {}, delivered_url)

        async def _attempt(enable_reasoning: bool | None) -> _Shaped:
            # Shape-validate INSIDE the retry budget so an LLM reply that is
            # missing a per-item field (`criterion`/`score`) is re-prompted
            # rather than killing the whole run on the first draw.
            resp = await self._wm.message(
                enable_reasoning=enable_reasoning,
                worker_name=_WORKER_NAME,
                message=prompt,
                system_prompt=_SYSTEM_PROMPT,
                output_schema=_OUTPUT_SCHEMA,
                max_tokens=_REVIEW_MAX_TOKENS,
            )
            return self._shape(resp.output)

        try:
            items, summary, saw_placeholder, evidence = await stage_call(
                downshift_reasoning(lambda think: guarded(lambda: _attempt(think))),
                timeout_seconds=self._llm_timeout,
                retries=self._llm_retries,
                retry_on=LLM_RETRY_ON,
                stage_name="promo_review",
            )
        except StageRetriesExhaustedError as exc:
            if not isinstance(exc.last_error, _EmptyReviewError):
                raise
            # Every draw came back empty. The video exists and is not broken, so
            # the run keeps it — but the review is FLAGGED with a real code (the run
            # page links "Fix and rerun this step"), never a silent 0.0 pass.
            return PromoReviewOutput(
                overall_score=0.0,
                items=[],
                summary="Not reviewed: the reviewer returned no scores.",
                notes=[f"promo_review: {_EMPTY_REVIEW_DETAIL}"],
                degradations=[degradation(PROMO_REVIEW_EMPTY, _EMPTY_REVIEW_DETAIL)],
                placeholder_text_found=bool(placeholder_findings(script.get("lines") or [])),
            )
        overall = sum(i.score for i in items) / max(1, len(items))
        placeholders = placeholder_findings(
            script.get("lines") or [],
            reviewer_saw_placeholder=saw_placeholder,
            reviewer_evidence=evidence,
        )
        return PromoReviewOutput(
            overall_score=overall,
            items=items,
            summary=summary,
            notes=[*reviewer_quality_notes(items), *placeholders],
            placeholder_text_found=bool(placeholders),
        )

    @staticmethod
    def _build_prompt(
        script: dict,
        brief: dict,
        output_url: str,
    ) -> str:
        return (
            "Score this promo on each criterion (1-5) and emit JSON:\n"
            "The promo's content is its narration script below (you cannot watch the "
            "video; the URL is for reference). Score EVERY criterion — never return an "
            "empty items list.\n"
            '{"items":[{"criterion":"...","score":<1-5>,"rationale":"..."}, ...],'
            '"summary":"..."}\n\n'
            f"Criteria: {list(PROMO_RUBRIC_CRITERIA)}\n"
            "Set placeholder_text_visible to true ONLY if placeholder text (None, null, "
            "undefined, N/A) is actually spoken in the narration or shown on a slide, "
            "and put that text in placeholder_text_evidence.\n"
            "Script (one entry per spoken line, in order):\n"
            + "\n".join(f"{i}. {line}" for i, line in enumerate(_narration(script), 1))
            + f"\nOutput URL: {output_url}\n"
            f"Brief: {_present_fields(brief)}\n"
        )

    @staticmethod
    def _shape(data: Mapping[str, object]) -> _Shaped:
        try:
            items = [
                PromoReviewItem(
                    id=str(uuid7()),
                    criterion=str(item["criterion"]),
                    score=int(cast("int", item["score"])),
                    rationale=str(item.get("rationale", "")),
                )
                for item in cast("Sequence[Mapping[str, object]]", data["items"])
            ]
            if not items:
                raise _EmptyReviewError("promo_review: the reviewer returned no scored criteria")
            summary = str(data.get("summary", "no summary"))
            # Strictly ``is True``: an absent or non-boolean answer is not a finding.
            saw_placeholder = data.get("placeholder_text_visible") is True
            evidence = str(data.get("placeholder_text_evidence") or "")
            return items, summary, saw_placeholder, evidence
        except _EmptyReviewError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"promo_review: LLM output missing expected fields: {exc}") from exc
