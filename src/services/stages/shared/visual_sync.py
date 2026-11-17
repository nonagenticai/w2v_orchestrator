"""Line→visual sync helpers shared by promo and website_review visual directors.

A two-hander narration (asker "a" ↔ answerer "b") walks through a set of discrete
visuals (demo screens, redesign sections). For the video to *follow* the words,
the visible visual must switch exactly when the answerer points at it — not while
the question that precedes it is still being asked. The LLM does the semantic
match (line → best visual by label); these deterministic helpers enforce the
structural rule the LLM can't be trusted with (the same lesson as deriving the
speaker from the question role rather than the LLM's enum).

Lifted verbatim from ``promo/visual_director.py`` so promo and website_review
share one implementation; promo re-imports these under its old private names.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from w2v_common.stages import stage_call

from src.services.stages.llm_budget import (
    LLM_RETRY_ON,
    MEDIUM_ANSWER_TOKENS,
    downshift_reasoning,
    output_token_budget,
)
from src.services.stages.upstream import guarded

if TYPE_CHECKING:
    from src.services.llm import WMClient

log = logging.getLogger(__name__)


# Reasoning-model output budget (see stages/llm_budget.py — a FLOOR, not a cap).
# ``{assignments[]}`` is one two-string pair (line_id + screen_id/section_id) per
# narration line, with no prose anywhere. The list length tracks the script (this
# call is not batched), but each item costs only a few tokens ⇒ MEDIUM.
_ASSIGNMENT_MAX_TOKENS = output_token_budget(answer_tokens=MEDIUM_ANSWER_TOKENS)


def assignments_schema(target_key: str) -> dict[str, Any]:
    """The line→visual assignment output schema, keyed by the visual field name.

    promo maps lines to ``screen_id``, website_review to ``section_id``; both
    return the same ``{assignments: [{line_id, <target_key>}]}`` shape. Builds it
    from the field name so the two directors share one schema definition.
    """
    return {
        "type": "object",
        "properties": {
            "assignments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "line_id": {"type": "string"},
                        target_key: {"type": "string"},
                    },
                    "required": ["line_id", target_key],
                },
            }
        },
        "required": ["assignments"],
    }


async def run_assignment_llm(
    wm: WMClient,
    *,
    worker_name: str,
    message: str,
    system_prompt: str,
    target_key: str,
    valid_lines: set[str],
    timeout_seconds: float,
    retries: int,
    retry_on: tuple[type[BaseException], ...] = LLM_RETRY_ON,
    stage_name: str,
    notes: list[str],
) -> dict[str, str]:
    """Run the line→visual LLM match, returning a ``{line_id: target_id}`` map.

    The shared LLM half of both visual directors: call the worker, keep only
    assignments whose line is a known footage/after line, and soft-fail to an
    empty map (the deterministic ``resolve_assignments`` fill still maps every
    line). ``target_key`` is ``screen_id`` (promo) or ``section_id``
    (website_review). Appends a degradation note on soft-fail.

    The call is routed through ``guarded`` with ``retry_on=LLM_RETRY_ON``: this is a
    soft-fail stage, so a permanent upstream failure (WM 402, token limit) should
    reach the carry-forward fill after ONE call, not after the full retry ladder —
    and the degradation note then carries WM's real sentence rather than "retries
    exhausted".
    """

    async def _attempt(enable_reasoning: bool | None) -> dict[str, str]:
        resp = await wm.message(
            enable_reasoning=enable_reasoning,
            worker_name=worker_name,
            message=message,
            system_prompt=system_prompt,
            output_schema=assignments_schema(target_key),
            max_tokens=_ASSIGNMENT_MAX_TOKENS,
        )
        raw = list((resp.output or {}).get("assignments") or [])
        mapping = {
            str(a["line_id"]): str(a[target_key])
            for a in raw
            if a.get("line_id") and a.get(target_key) and str(a["line_id"]) in valid_lines
        }
        if not mapping:
            raise ValueError(f"{stage_name}: LLM returned no usable assignments")
        return mapping

    try:
        return await stage_call(
            downshift_reasoning(lambda think: guarded(lambda: _attempt(think))),
            timeout_seconds=timeout_seconds,
            retries=retries,
            retry_on=retry_on,
            stage_name=stage_name,
        )
    except Exception as exc:  # noqa: BLE001 -- soft-fail: carry-forward fill still maps every line
        log.warning("%s: LLM mapping failed, using fill: %s", stage_name, exc)
        notes.append(f"LLM mapping failed; used carry-forward fill: {exc}")
        return {}


def is_question(line: dict[str, Any]) -> bool:
    """A line is a question (asker turn) by its derived speaker, or a trailing '?'."""
    if str(line.get("speaker") or "").strip().lower() == "a":
        return True
    return str(line.get("text") or "").rstrip().endswith("?")


def resolve_assignments(
    footage_lines: list[dict[str, Any]],
    screen_ids: list[str],
    llm_map: dict[str, str],
    *,
    spread_on_collapse: bool = False,
) -> dict[str, str]:
    """Combine the LLM line→visual map with the deterministic structural rules.

    Only *answer* lines drive a switch: the answerer is the one pointing at the
    visual ("look at this leaderboard"), so the switch must land on the answer,
    not the question that precedes it. Question lines therefore take no direct
    assignment; the carry-forward fill below holds them on the previously
    resolved, currently-visible visual — the opening question falls back to the
    first visual and consecutive questions all hold the same one.

    ``spread_on_collapse`` adds an anti-collapse safety net (off by default, so
    promo is unaffected): when the LLM piles every answer line onto a single
    visual (or maps none) while two or more visuals exist, the after act would
    hold one still the whole way. In that case the answer lines are walked across
    the visuals in their given (document, top→bottom) order so the video tours
    them instead. Same philosophy as the question/answer rule — the LLM does the
    semantic match, a deterministic rule enforces the variety it can't be trusted
    to produce.
    """
    if not screen_ids:
        return {}
    valid = set(screen_ids)
    answer_lines = [ln for ln in footage_lines if not is_question(ln)]
    # Only answer lines anchor a visual; questions hold the prior one.
    anchors: dict[str, str] = {
        str(ln["line_id"]): llm_map[str(ln["line_id"])]
        for ln in answer_lines
        if llm_map.get(str(ln["line_id"])) in valid
    }

    if (
        spread_on_collapse
        and len(answer_lines) >= 2
        and len(screen_ids) >= 2
        and len(set(anchors.values())) <= 1
    ):
        # Collapsed onto one visual (or none) — distribute the answer lines evenly
        # across the visuals in order so the after act advances top→bottom.
        anchors = {
            str(ln["line_id"]): screen_ids[i * len(screen_ids) // len(answer_lines)]
            for i, ln in enumerate(answer_lines)
        }

    # Carry-forward fill: every line holds the last anchored visual until an
    # answer switches it, so a question shows what the previous answer showed and
    # the visible visual only changes when the answer is spoken.
    resolved: dict[str, str] = dict(anchors)
    last = screen_ids[0]
    for ln in footage_lines:
        lid = str(ln["line_id"])
        if resolved.get(lid):
            last = resolved[lid]
        else:
            resolved[lid] = last
    return resolved
