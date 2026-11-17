"""website_review_slide_plan — author a REAL slide deck from the critique.

``website_review_slides`` was inert by construction: nothing in the blueprint ever
authored ``kind="slide"`` shots, so it found zero of them on EVERY fully-AI run,
returned in ~0.3s and emitted the ``slides_not_authored`` advisory forever. Compile
then fell through to the scroll-capture stills for any line the site capture and
the redesign sections did not cover.

This stage fills that hole. It projects the critique findings + the BEFORE-act
narration lines into one slide per line, so a line that says "three things hurt the
first impression" has a card on screen that NAMES those three things. The deck is
keyed by SCRIPT line id (not the shotlist's positional ``shot-NNN``), which is the
key ``website_review_compile`` binds visuals on.

Contract with the rest of the pipeline:

* It NEVER runs for the authored-shotlist path (A2). If the operator placed
  ``slide`` shots in ``brief.shotlist``, that IS the deck and this stage returns an
  empty plan — the operator's explicit content must not be second-guessed.
* It NEVER changes narration length. Slides are pure visuals; the planned shots
  carry ``seconds=0.0`` and nothing downstream reads them for duration. The ordered
  duration is settled by ``script.py`` / ``order.py``, which this stage does not
  touch.
* Slides do NOT outrank the live site capture. ``compile._key_lines`` puts a
  captured screen first; a planned slide only fills a line that would otherwise
  fall through to the full-page scroll pool.

⛔ It never ships a hollow deck silently. A malformed / empty / failed LLM reply
falls back to a DETERMINISTIC projection of the critique findings and records the
:data:`SLIDE_PLAN_LLM_FALLBACK` advisory, so the fallback is answerable from the run
record. (``html_render_service`` #132 is the same shape failing the other way:
returning ``{}`` on a malformed reply with no trace.)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from w2v_common.stages import stage_call

from src.services.llm import WMClient
from src.services.stages.base import (
    SlotUse,
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithDegradations,
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
from src.services.stages.shared.degradation import SLIDE_PLAN_LLM_FALLBACK, advisory
from src.services.stages.shared.placeholder_text import slide_anchor
from src.services.stages.shared.shotlist import normalize_shotlist, normalize_slide
from src.services.stages.shared.video_gate import require_video
from src.services.stages.upstream import guarded

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

log = logging.getLogger(__name__)

#: Hard ceiling on deck size. A slide per BEFORE-act line is the intent, but the
#: deck is rendered one PNG at a time by slide_service inside the 600s
#: ``website_review_slides`` budget, so an unusually long script must not turn a
#: visual nicety into a stage timeout. Beyond this the remaining lines keep the
#: existing fallbacks (scroll pool), which is exactly today's behaviour.
MAX_PLANNED_SLIDES = 12

#: Reasoning-model output budget. One card per line — an eyebrow, a title and up to
#: three short bullets — so the payload grows with the script but every item is a
#: handful of words. MEDIUM, same class as the visual-director assignments.
_PLAN_MAX_TOKENS = output_token_budget(answer_tokens=MEDIUM_ANSWER_TOKENS)

_WORKER = "website-review-slide-planner"

_SYSTEM_PROMPT = (
    "You are a presentation designer building the on-screen cards for a narrated "
    "website review. You get the reviewer's findings about the site and the "
    "narration lines, in order. For each narration line, write the ONE card that "
    "should be on screen while it is spoken.\n"
    "Rules:\n"
    "- The card must say what THAT line is about. Never repeat the narration "
    "verbatim — the viewer hears it; the card is the visual summary.\n"
    "- Ground the card in the finding the line cites when it cites one: name the "
    "problem and, in the bullets, what to do about it.\n"
    "- Titles are short (at most about eight words). Bullets are at most three per "
    "card, one short phrase each.\n"
    '- Use template "bullets" when the line lists or explains problems, "title" '
    'for a statement or transition, and "cta" only for a closing call to action.\n'
    "- Return one entry per line_id you were given, and use those ids exactly.\n"
    "Return only JSON matching the schema."
)

_PLAN_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "slides": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "line_id": {"type": "string"},
                    "template": {"type": "string", "enum": ["title", "bullets", "cta"]},
                    "eyebrow": {"type": "string"},
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "bullets": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["line_id", "title"],
            },
        }
    },
    "required": ["slides"],
}


class WebsiteReviewSlidePlanInput(StageInput):
    pass


class PlannedSlide(StageOutput):
    """One authored card, keyed to the narration line it plays under."""

    line_id: str
    narration: str
    #: The normalized shotlist ``slide`` sub-tree (``shotlist.normalize_slide``):
    #: ``template`` plus whichever of ``eyebrow``/``title``/``body``/``bullets``/
    #: ``cta_url`` are non-empty. Canonical at the WRITE site so
    #: ``website_review_slides`` can project it with ``_shotlist_deck_slide``
    #: exactly as it projects an operator-authored shot.
    slide: dict[str, object] = {}  # noqa: RUF012 — pydantic copies field defaults per-instance
    #: The critique finding this card was grounded in, when the script named one.
    critique_item_id: str | None = None


class WebsiteReviewSlidePlanOutput(StageOutputWithDegradations):
    # NO `produces` on `slides`, deliberately — the same reasoning as
    # `WebsiteReviewSlidesOutput`. An empty plan is a LEGITIMATE mode: an
    # operator-authored shotlist deck (`source="authored_shotlist"`), a
    # critique-only run, or a run whose script produced no BEFORE-act lines.
    # Declaring `produces` would mark every one of those correct runs degraded.
    slides: list[PlannedSlide] = []  # noqa: RUF012 — pydantic copies field defaults per-instance
    #: How the deck was authored, so an inspector never has to guess:
    #: ``llm`` | ``critique_fallback`` | ``authored_shotlist`` | ``no_lines``.
    #: (``advisories`` comes from StageOutputWithDegradations. The fallback path
    #: is ALWAYS recorded there — a deck that quietly stopped being model-authored
    #: is the failure this stage exists to make visible. ADVISORY, not degradation:
    #: the deterministic deck is a real deck, and nothing is missing from the video.)
    source: str = "no_lines"


def _before_lines(ctx: RunContext) -> list[dict[str, str]]:
    """The BEFORE-act narration lines, in order, with their critique citation."""
    script = ctx.prior_output("website_review_script") or {}
    out: list[dict[str, str]] = []
    for ln in script.get("lines", []):
        if str(ln.get("phase") or "before").strip().lower() != "before":
            continue
        text = str(ln.get("text") or "").strip()
        if not ln.get("id") or not text:
            continue
        out.append(
            {
                "line_id": str(ln["id"]),
                "text": text,
                "critique_item_id": str(ln.get("critique_item_id") or ""),
            }
        )
    return out


def _findings(ctx: RunContext) -> list[dict[str, str]]:
    critique = ctx.prior_output("website_critique") or {}
    out: list[dict[str, str]] = []
    for f in critique.get("findings", []):
        if not isinstance(f, dict) or not str(f.get("title") or "").strip():
            continue
        out.append(
            {
                "id": str(f.get("id") or ""),
                "category": str(f.get("category") or "").strip(),
                "severity": str(f.get("severity") or "").strip(),
                "title": str(f["title"]).strip(),
                "description": str(f.get("description") or "").strip(),
                "recommendation": str(f.get("recommendation") or "").strip(),
            }
        )
    return out


def deterministic_plan(
    lines: list[dict[str, str]],
    findings: list[dict[str, str]],
    anchor: str,
) -> list[PlannedSlide]:
    """Project critique findings onto narration lines with NO model in the loop.

    The floor under the LLM path, not a degraded stub: a line that cites a finding
    gets that finding's own words — ``title`` ← ``finding.title``, the bullets ←
    ``recommendation`` then ``description`` (whichever are non-empty). A line that
    cites nothing gets a title card of its own narration, which is the same
    universal fallback ``_shotlist_deck_slide`` applies to an unmatched shot.
    """
    by_id = {f["id"]: f for f in findings if f["id"]}
    planned: list[PlannedSlide] = []
    for line in lines:
        finding = by_id.get(line["critique_item_id"])
        if finding is None:
            raw: dict[str, object] = {"template": "title", "eyebrow": anchor, "title": line["text"]}
        else:
            bullets = [b for b in (finding["recommendation"], finding["description"]) if b]
            raw = {
                "template": "bullets" if bullets else "title",
                "eyebrow": finding["category"] or anchor,
                "title": finding["title"],
                "bullets": bullets,
            }
        planned.append(
            PlannedSlide(
                line_id=line["line_id"],
                narration=line["text"],
                slide=normalize_slide(raw),
                critique_item_id=line["critique_item_id"] or None,
            )
        )
    return planned


def _slides_from_reply(
    reply: object,
    lines: list[dict[str, str]],
    anchor: str,
) -> list[PlannedSlide]:
    """Keep only entries that name a known line AND carry a non-empty title.

    Everything else is dropped. An empty return is the caller's signal to fall back
    — ``normalize_slide`` would happily turn ``{}`` into a template-only card, which
    is exactly the hollow deck this stage must never ship.
    """
    if not isinstance(reply, dict):
        return []
    raw = reply.get("slides")
    if not isinstance(raw, list):
        return []
    by_line = {line["line_id"]: line for line in lines}
    seen: set[str] = set()
    planned: list[PlannedSlide] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        lid = str(entry.get("line_id") or "")
        line = by_line.get(lid)
        if line is None or lid in seen:
            continue
        if not str(entry.get("title") or "").strip():
            continue
        seen.add(lid)
        slide = normalize_slide({**entry, "eyebrow": entry.get("eyebrow") or anchor})
        planned.append(
            PlannedSlide(
                line_id=lid,
                narration=line["text"],
                slide=slide,
                critique_item_id=line["critique_item_id"] or None,
            )
        )
    # Keep the script's order, not the model's.
    order = {line["line_id"]: i for i, line in enumerate(lines)}
    planned.sort(key=lambda p: order[p.line_id])
    return planned


class WebsiteReviewSlidePlanStage(
    StageHandler[WebsiteReviewSlidePlanInput, WebsiteReviewSlidePlanOutput]
):
    name = "website_review_slide_plan"
    input_model = WebsiteReviewSlidePlanInput
    output_model = WebsiteReviewSlidePlanOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("llm"),)

    def __init__(self, client: WMClient | None = None) -> None:
        self._llm_timeout = llm_budget_seconds()
        self._llm_retries = llm_retries()
        self._wm = client or WMClient(timeout=llm_client_timeout_seconds())

    async def handle(
        self,
        payload: WebsiteReviewSlidePlanInput,
        ctx: RunContext,
    ) -> WebsiteReviewSlidePlanOutput:
        require_video(ctx)  # video-only: a critique-only run has no deck to show

        brief = ctx.config_snapshot.get("brief") or {}
        if any(s.kind == "slide" for s in normalize_shotlist(brief)):
            # The operator authored the deck by hand (A2). That IS the content;
            # website_review_slides renders it and this stage stays out of the way.
            return WebsiteReviewSlidePlanOutput(slides=[], source="authored_shotlist")

        lines = _before_lines(ctx)[:MAX_PLANNED_SLIDES]
        if not lines:
            return WebsiteReviewSlidePlanOutput(slides=[], source="no_lines")

        # Never a hardcoded house brand: an unbranded run renders no eyebrow.
        anchor = slide_anchor(ctx.config_snapshot)
        findings = _findings(ctx)

        reason = ""
        try:
            reply = await self._author(lines, findings)
            planned = _slides_from_reply(reply, lines, anchor)
            if not planned:
                reason = "the reply named no usable slide (no known line_id with a title)"
        except Exception as exc:  # noqa: BLE001 -- soft-fail: the deterministic floor still ships a deck
            planned = []
            reason = f"{type(exc).__name__}: {exc}"

        if planned:
            return WebsiteReviewSlidePlanOutput(slides=planned, source="llm")

        log.warning("%s: LLM plan unusable (%s); using critique projection", self.name, reason)
        return WebsiteReviewSlidePlanOutput(
            slides=deterministic_plan(lines, findings, anchor),
            source="critique_fallback",
            advisories=[
                advisory(
                    SLIDE_PLAN_LLM_FALLBACK,
                    "slide deck authored by the deterministic critique projection, "
                    f"not the model — {reason}",
                )
            ],
        )

    async def _author(
        self,
        lines: list[dict[str, str]],
        findings: list[dict[str, str]],
    ) -> object:
        findings_block = (
            "\n".join(
                f"- finding {f['id']} ({f['category'] or 'general'}"
                f"{'/' + f['severity'] if f['severity'] else ''}): "
                f"{f['title']} — {f['description']} FIX: {f['recommendation']}"
                for f in findings
            )
            or "(the reviewer recorded no itemised findings)"
        )
        lines_block = "\n".join(
            f"- line_id {line['line_id']}"
            + (f" [cites finding {line['critique_item_id']}]" if line["critique_item_id"] else "")
            + f': "{line["text"]}"'
            for line in lines
        )
        message = (
            "Write the on-screen card for each narration line of this website "
            "review.\n\n"
            f"Reviewer findings:\n{findings_block}\n\n"
            f"Narration lines (in order):\n{lines_block}"
        )

        async def _attempt(enable_reasoning: bool | None) -> object:
            resp = await self._wm.message(
                enable_reasoning=enable_reasoning,
                worker_name=_WORKER,
                message=message,
                system_prompt=_SYSTEM_PROMPT,
                output_schema=_PLAN_SCHEMA,
                max_tokens=_PLAN_MAX_TOKENS,
            )
            return resp.output

        return await stage_call(
            downshift_reasoning(lambda think: guarded(lambda: _attempt(think))),
            timeout_seconds=self._llm_timeout,
            retries=self._llm_retries,
            retry_on=LLM_RETRY_ON,
            stage_name=self.name,
        )
