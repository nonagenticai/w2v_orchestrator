"""An EMPTY script draw is a failure of the script stage — never a result.

Live 2026-10-06, run ``01a10e19-145f-7475-a34a-aa1fc8d3113f`` (website_review,
50s order): the script call answered HTTP 200 with ``{"lines": []}``, the
primitive's default floor of 0 accepted it, the stage persisted 0 lines as a
success, and the run died three stages later in compile with "no shots to
compile". ``app_review``, ``promo`` and ``lesson`` accepted the same draw the
same way.

The floor itself is ``min_elements = 1`` on each stage, applied by the PARSER
inside the retry budget (``LLMScriptStageBase._attempt`` raises
``ScriptDrawUnderFloorError``; ``stage_call`` re-prompts). It cannot live in the
output schema — worker-management strips ``minItems``
(``tests/test_output_schema_bounds.py``). This module is only what the four
stages share AROUND that floor: naming the exhausted failure for what it is, and
refusing a zero-element output however it came about.

Scope: the LLM draw only. A verbatim bypass (``short_circuit`` — an authored
shotlist or ``source_script``) never reaches the parser and is not judged here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from w2v_common.stages import StageRetriesExhaustedError

from src.services.stages.primitives.script_stage import ScriptDrawUnderFloorError

if TYPE_CHECKING:
    from collections.abc import Sized

    from src.services.runs.run_context import RunContext


class EmptyScriptDrawError(ValueError):
    """The script stage holds ZERO narration elements — a failure, never a result.

    Everything after a script stage is a function of its elements: narration
    speaks them, the picture stages map onto them, compile cuts to them. A
    zero-element script therefore cannot be degraded-and-shipped the way a SHORT
    one is — there is nothing to ship.
    """


#: ``SiteCaptureOutput.page_text_status`` values that mean the scrape produced
#: little or nothing to ground a script on.
_UNGROUNDED_PAGE_TEXT: Final = frozenset({"thin", "empty", "fetch_failed", "blocked"})


def thin_capture_clause(ctx: RunContext) -> str:
    """What ``site_capture`` read, when that was thin — else ``""``.

    Read from the capture's OWN output as this run holds it (``page_text_status``
    is the discriminator ``SiteCaptureOutput`` tells consumers to branch on; the
    count is ``len(page_text.strip())``, the number its 200-char floor is applied
    to). The output may not exist (``site_capture`` is optional): absent, or
    readable, says nothing.

    ONLY for a stage whose script prompt actually CONTAINS the capture's
    ``page_text`` — today that is ``promo_script`` alone. ``website_review_script``
    is prompted from the critique and the redesign and never sees the scrape, so
    blaming a thin capture there names a cause the model was never exposed to; it
    reports its own prompt facts instead (``WebsiteReviewScriptStage._draw_facts``).
    """
    capture = ctx.prior_output("site_capture") or {}
    status = str(capture.get("page_text_status") or "")
    if status not in _UNGROUNDED_PAGE_TEXT:
        return ""
    chars = len(str(capture.get("page_text") or "").strip())
    return (
        f"; site capture read {chars} chars (page_text_status={status!r}) — the "
        "model had almost nothing to ground the script on"
    )


def empty_draw_failure(
    exc: StageRetriesExhaustedError, *, stage: str, context: str = ""
) -> StageRetriesExhaustedError | None:
    """``exc`` renamed to what happened, when what happened was an empty draw.

    ``None`` when the retries were exhausted by anything else — the caller
    re-raises that untouched. Otherwise the SAME error type the stage always
    raised (so the executor classifies it exactly as before), whose cause is an
    :class:`EmptyScriptDrawError` naming the empty draw; ``context`` is an optional
    clause saying why the model may have had nothing to write from.
    """
    under = exc.last_error
    if not isinstance(under, ScriptDrawUnderFloorError):
        return None
    return StageRetriesExhaustedError(
        stage,
        exc.attempts,
        EmptyScriptDrawError(
            f"{stage}: empty script draw — the model returned {under.count} narration "
            f"elements on every one of {exc.attempts} attempts (at least "
            f"{under.required} required), so there is no script to narrate or "
            f"compile{context}"
        ),
    )


def require_elements(elements: Sized, *, stage: str, context: str = "") -> None:
    """Refuse a zero-element script outright — see :class:`EmptyScriptDrawError`."""
    if len(elements):
        return
    raise EmptyScriptDrawError(
        f"{stage}: empty script draw — the script holds zero narration elements, so "
        f"there is no script to narrate or compile{context}"
    )
