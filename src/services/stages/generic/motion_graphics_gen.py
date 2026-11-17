"""motion_graphics_gen — deterministic branded motion-graphics clips.

A code-driven, on-brand, zero-cost sibling of ``ai_video_gen``. Instead of a
generative provider, it dispatches to video_production_service's
``POST /api/record/animation``, which frame-steps an HTML/CSS/JS motion template
(``window.render(t)`` / CSS ``@keyframes``) into an MP4 — pixel-identical run to
run. The produced clips slot into the per-line compositor exactly like AI video
or recorded footage (``video_url`` keyed by ``line_id``); the compiler needs no
change.

Reads ``motion_graphics`` from ``ctx.config_snapshot``::

    motion_graphics:
      template_id: pipeline_story    # registered template (or use `html`)
      html:                          # optional raw template instead of template_id
      width: 1920
      height: 1080
      fps: 30
      duration_ms:                   # optional; else template's RENDER_DURATION_MS
      params: {...}                  # base params merged into every item
      items_from_stage: motion_brief # optional prior stage → {items:[{line_id,params}]}
      items: [{line_id, params}]     # optional static per-line items

An item may carry its OWN ``template_id`` (and ``duration_ms``): the run-level
values are then only the default, so one run can cut a stat card, a timeline and
a title card. An item that names a template never inherits the block's raw
``html``.

When no per-line ``items``/``items_from_stage`` resolve, a single clip is
produced from the base ``params`` (``line_id`` None → round-robin in render).

Stage is OPT-IN: with no ``motion_graphics`` block the stage is skipped and the
run still completes — a blueprint may wire it at zero cost. The one other way
in is promo's ``promo_motion_brief`` stage: when it ran, its authored per-line
items are rendered even without a ``motion_graphics`` block. Items the request
authored itself (``items`` / ``items_from_stage``) always win over the brief's.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, ClassVar, cast

from w2v_common.stages import stage_gather

from src.clients.video_production_client import VideoProductionClient
from src.services.stages.base import (
    StageDownstreamResponseError,
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
    StagePreconditionError,
    StageSkippedError,
)
from src.services.stages.shared.degradation import MOTION_ITEM_UNBOUND, advisory
from src.services.stages.shared.motion_slides import (
    format_dims,
    motion_per_line_requested,
    quality_tier,
    run_motion_brand_params,
)
from src.services.tts_storage import (
    deterministic_media_id,
    find_existing_media,
    store_generated_video,
)
from src.utils import coerce_float

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

log = logging.getLogger(__name__)

_RECORD_TIMEOUT_S = 600.0
_PER_ITEM_TIMEOUT_SECONDS = 900.0
_CONCURRENCY = 2
# The promo script stage. Its presence in the run's prior outputs is what makes
# this a PROMO run: promo's blueprint wires ``motion_graphics_gen`` after
# ``promo_script``; the generic blueprint wires it after ``script_draft`` and
# never has this output. Both promo-only behaviours below (binding cards to
# script lines, recording at the film's frame) key off it, so a generic run is
# byte-identical to before run 01a10d84's fixes.
_PROMO_SCRIPT_STAGE = "promo_script"
# Promo's per-line motion brief. Its ``items`` are this stage's items when the
# request authored none of its own (see ``_resolve_items``).
_MOTION_BRIEF_STAGE = "promo_motion_brief"
# Item ``params`` keys that carry the text a card SHOWS, in precedence order.
# ``text`` is deliberately absent: ``kinetic_typography`` reads ``text`` as a
# colour token, so it is not on-screen copy.
_TEXT_PARAM_KEYS = ("words", "label", "title", "headline", "quote")
_NON_WORD = re.compile(r"[^\w%+]+")


class MotionGraphicsGenInput(StageInput):
    pass  # All inputs come from ctx.config_snapshot["motion_graphics"].


class MotionClip(StageOutput):
    id: str
    line_id: str | None
    template_id: str | None
    video_url: str | None
    duration_s: float | None
    status: str


class MotionGraphicsGenOutput(StageOutputWithNotes):
    # Both no-op paths skip (no `motion_graphics` block; opted in but no
    # template_id/html), and a produced clip missing `video_url` already raises.
    # So zero clips on a COMPLETED stage is the silent case in between, and
    # `visual_media.collect_motion_clips_by_line` turns it into "no motion at
    # all" without anything disagreeing.
    produces: ClassVar[tuple[str, ...]] = ("clips",)

    clips: list[MotionClip]


class MotionGraphicsGenStage(StageHandler[MotionGraphicsGenInput, MotionGraphicsGenOutput]):
    name = "motion_graphics_gen"
    input_model = MotionGraphicsGenInput
    output_model = MotionGraphicsGenOutput

    def __init__(self, client: VideoProductionClient | None = None) -> None:
        self._client = client or VideoProductionClient()

    async def handle(
        self,
        payload: MotionGraphicsGenInput,
        ctx: RunContext,
    ) -> MotionGraphicsGenOutput:
        cfg = ctx.config_snapshot.get("motion_graphics") or {}
        if not cfg and not _brief_items(ctx):
            raise StageSkippedError(
                "motion_graphics_gen: no motion_graphics config block (opt-in stage not requested)",
                missing=["motion_graphics"],
            )

        default_template = cfg.get("template_id")
        default_html = cfg.get("html")
        resolved = self._resolve_items(cfg, ctx)
        if (
            not default_template
            and not default_html
            and not any(item.get("template_id") for item in resolved)
        ):
            raise StageSkippedError(
                "motion_graphics_gen: opted in but neither template_id nor html provided",
                missing=["motion_graphics.template_id"],
            )

        unbound: list[str] = []
        items = _bind_items_to_script(resolved, ctx, unbound)
        for item in items:
            if not item.get("template_id") and not default_template and not default_html:
                raise StagePreconditionError(
                    f"motion_graphics_gen: item for line {item.get('line_id')!r} names no "
                    "template_id and the motion_graphics block has no default template_id/html",
                    missing=["motion_graphics.items[].template_id"],
                )
        # Brand params from the ONE resolution the end card and the animated
        # slides use, so a promo's motion cards share their palette and font.
        # Only a promo that opted in to per-line motion: every other run — a
        # generic run, and a promo without the flag — keeps sending exactly its
        # block's params. The block's and the item's params win.
        brand_params = (
            run_motion_brand_params(ctx.config_snapshot)
            if _is_promo_run(ctx) and motion_per_line_requested(ctx.config_snapshot)
            else {}
        )
        base_params = {**brand_params, **(cfg.get("params") or {})}
        width, height = _record_dims(cfg, ctx)
        fps = int(cfg.get("fps") or 30)
        default_duration_ms = cfg.get("duration_ms")
        run_id = str(ctx.run_id)

        async def _generate(item: dict[str, object]) -> MotionClip:
            params = {**base_params, **(cast("dict[str, object]", item.get("params")) or {})}
            # The item's own template wins; the block's is the default. An item
            # that names a template is that template — never the block's raw html.
            own_template = cast("str | None", item.get("template_id") or None)
            template_id = own_template or default_template
            html = None if own_template else default_html
            duration_ms = cast("int | None", item.get("duration_ms") or default_duration_ms)
            # Content-address the clip by everything that determines its pixels so a
            # resumed/retried render reuses an already-recorded clip.
            media_id = deterministic_media_id(
                self.name,
                template_id,
                html,
                json.dumps(params, sort_keys=True, default=str),
                width,
                height,
                fps,
                duration_ms,
                item.get("line_id"),
            )
            existing = await find_existing_media(run_id, self.name, media_id, "mp4")
            if existing is not None:
                return MotionClip(
                    id=media_id,
                    line_id=cast("str | None", item.get("line_id")),
                    template_id=template_id,
                    video_url=existing,
                    duration_s=None,
                    status="reused",
                )
            # Idempotency: stage_gather retries the WHOLE callable on a transient
            # wait failure. Without a dedupe key the re-submit would start a
            # SECOND paid recording for the same clip (double-bill). The per-item
            # key (run + stage + content-addressed media_id) makes the service
            # return the ORIGINAL job on the re-submit, so only the wait is
            # actually retried. Mirrors demo_video/record.py's start_execution.
            job_id = await self._client.record_animation(
                template_id=template_id,
                html=html,
                params=params or None,
                part_id=f"motion-{media_id[:8]}",
                fps=fps,
                width=width,
                height=height,
                duration_ms=duration_ms,
                output_format="mp4",
                idempotency_key=f"{run_id}:{self.name}:{media_id}",
            )
            result = await self._client.wait_for_content_recording(
                job_id, timeout=_RECORD_TIMEOUT_S
            )
            video_bytes, _ctype, _disp = await self._client.get_recording_video(job_id)
            video_url = await store_generated_video(
                run_id=run_id,
                stage_name=self.name,
                video_id=media_id,
                video_bytes=video_bytes,
                video_format="mp4",
            )
            return MotionClip(
                id=media_id,
                line_id=cast("str | None", item.get("line_id")),
                template_id=template_id,
                video_url=video_url,
                duration_s=coerce_float(result.get("duration_seconds")),
                status="completed",
            )

        if not items:
            # Every text-bearing item was unbound (none matches a script line):
            # nothing to render, and nothing may be placed on a line it does not
            # belong to.
            return MotionGraphicsGenOutput(
                clips=[],
                advisories=[advisory(MOTION_ITEM_UNBOUND, reason) for reason in unbound],
            )
        clips = await stage_gather(
            items,
            _generate,
            concurrency=_CONCURRENCY,
            per_item_timeout_seconds=_PER_ITEM_TIMEOUT_SECONDS,
            per_item_retries=1,
            stage_name="motion_graphics_gen",
        )
        for clip in clips:
            if not clip.video_url:
                raise StageDownstreamResponseError(
                    f"motion_graphics_gen: no usable video_url for line {clip.line_id!r}"
                )
        return MotionGraphicsGenOutput(
            clips=clips,
            advisories=[advisory(MOTION_ITEM_UNBOUND, reason) for reason in unbound],
        )

    @staticmethod
    def _resolve_items(cfg: dict, ctx: RunContext) -> list[dict[str, object]]:
        """Per-line items from a prior stage or static config; else one base clip.

        ``items_from_stage`` names a prior stage whose output carries
        ``items: [{line_id, params}, ...]`` (mirrors ai_image_gen's
        ``prompts_from_stage``). Failing both, the items promo's motion brief
        authored (each with its own ``template_id``) — so anything the request
        authored itself wins over the brief. A single ``{}`` item means "one
        clip from the base params, unkeyed".
        """
        from_stage = cfg.get("items_from_stage")
        if from_stage:
            prior = ctx.prior_output(str(from_stage)) or {}
            items = [dict(it) for it in (prior.get("items") or []) if isinstance(it, dict)]
            if items:
                return items
        raw = cfg.get("items")
        if isinstance(raw, list) and raw:
            return [dict(it) for it in raw if isinstance(it, dict)]
        return _brief_items(ctx) or [{}]


# Words that carry no claim of their own. A card and a line that share ONLY
# these ("at the", "is a") say different things; see ``_bind_items_to_script``.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "for",
        "by", "with", "from", "as", "is", "are", "was", "were", "be", "been", "it",
        "its", "this", "that", "these", "those", "your", "our", "you", "we", "us",
        "they", "their", "s",
    }
)  # fmt: skip
# Words that flip a claim. They must match on both sides of a partial match.
_NEGATORS = frozenset({"not", "never", "no", "without", "nor", "none", "nothing"})
_CONTRACTED_NOT = re.compile(r"n['\u2019]t\b")
# A partial (non-identical) match must share at least this many content words.
# One shared word is a coincidence ("control", "merqi"); two is a quoted claim.
_MIN_PARTIAL_CONTENT_WORDS = 2


def _brief_items(ctx: RunContext) -> list[dict[str, object]]:
    """The per-line items ``promo_motion_brief`` authored (``[]`` when it skipped)."""
    prior = ctx.prior_output(_MOTION_BRIEF_STAGE) or {}
    return [dict(item) for item in prior.get("items") or [] if isinstance(item, dict)]


def _is_promo_run(ctx: RunContext) -> bool:
    return ctx.prior_output(_PROMO_SCRIPT_STAGE) is not None


def _record_dims(cfg: dict, ctx: RunContext) -> tuple[int | None, int | None]:
    """Record dims: the block's own pins, else (PROMO only) the film's frame.

    A promo clip is cut INTO the film's frame, so it is recorded at the film's
    own format + resolution tier unless the block pins dims. Leaving them unset
    made video_production fall back to its 1080x1080 default, and a square
    kinetic-typography card cover-cropped into a 16:9 frame lost its top and
    bottom lines (run 01a10d84, 2026-10-05). Scoped to promo: a generic run
    keeps sending exactly what its block says (``None`` = the service default),
    so its fleet cost and its clips do not change under it.
    """
    width, height = cfg.get("width"), cfg.get("height")
    if not _is_promo_run(ctx):
        return width, height
    deliverable = ctx.config_snapshot.get("deliverable") or {}
    default_w, default_h = format_dims(deliverable.get("format"), quality_tier(ctx.config_snapshot))
    return int(width or default_w), int(height or default_h)


def _tokens(text: str) -> tuple[str, ...]:
    """Whole words of ``text``, lower-cased; punctuation is a word boundary.

    Contracted negation is spelled out first (``don't`` → ``do not``,
    ``cannot`` → ``can not``) so a negator is always its own token — the
    apostrophe would otherwise split it into a meaningless ``t``.
    """
    lowered = _CONTRACTED_NOT.sub(" not", text.lower().replace("\u2019", "'"))
    lowered = lowered.replace("cannot", "can not")
    return tuple(_NON_WORD.sub(" ", lowered).split())


def _content(tokens: tuple[str, ...]) -> frozenset[str]:
    return frozenset(t for t in tokens if t not in _STOPWORDS)


def _item_text(item: dict[str, object]) -> str:
    """The on-screen text an item's card shows (``""`` when it carries none)."""
    params = item.get("params")
    if not isinstance(params, dict):
        return ""
    for key in _TEXT_PARAM_KEYS:
        value = cast("dict[str, object]", params).get(key)
        if isinstance(value, list):
            value = " ".join(str(v) for v in value)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _script_lines(ctx: RunContext) -> list[tuple[str, tuple[str, ...]]]:
    """``(line_id, tokens)`` of the promo script, in script order."""
    lines = (ctx.prior_output(_PROMO_SCRIPT_STAGE) or {}).get("lines") or []
    return [
        (str(line["id"]), _tokens(str(line.get("text") or "")))
        for line in lines
        if isinstance(line, dict) and line.get("id")
    ]


def _negators(words: frozenset[str]) -> frozenset[str]:
    return words & _NEGATORS


def _partial_score(card: tuple[str, ...], line: tuple[str, ...]) -> float:
    """How strongly a card QUOTES a line; ``0.0`` when it does not.

    One direction only — the card may ABRIDGE its line, never add to it:

    * every content word on the card is spoken in the line, numbers and
      percentages included ("90%" is not said by a line that says "30% in 90
      days"), so a card cannot put an offer or a figure on screen that the
      narration does not say;
    * negators (``not``/``never``/``no``/``without``/``nor``, contractions
      spelled out) match on BOTH sides, so neither a card that adds one ("Agents
      never send…" over "Agents send…") nor one that drops one ("Lose control…"
      over "You never lose control…") binds;
    * the card carries at least ``_MIN_PARTIAL_CONTENT_WORDS`` content words —
      one shared word is a coincidence, not a quote.

    Words, never characters: "maintain" does not contain "AI". The score is the
    Jaccard overlap of the two content-word sets, so among the lines a card
    abridges, the one it covers most wins (ties: the earlier line).
    """
    card_words, line_words = _content(card), _content(line)
    if len(card_words) < _MIN_PARTIAL_CONTENT_WORDS or not card_words <= line_words:
        return 0.0
    if _negators(card_words) != _negators(line_words):
        return 0.0
    return len(card_words) / len(line_words)


def _match_cards(
    cards: list[tuple[str, ...]], lines: list[tuple[str, tuple[str, ...]]], taken: set[str]
) -> dict[int, str]:
    """``{card index: line_id}`` — every exact match first, then partial ones.

    ``taken`` (already holding the explicitly keyed lines) is updated in place.
    """
    order = {line_id: index for index, (line_id, _words) in enumerate(lines)}
    match_by_card: dict[int, str] = {}
    for position, words in enumerate(cards):
        exact = next((lid for lid, line in lines if lid not in taken and line == words), None)
        if exact is not None:
            taken.add(exact)
            match_by_card[position] = exact
    for position, words in enumerate(cards):
        if position in match_by_card:
            continue
        scored = [
            (score, -order[lid], lid)
            for lid, line in lines
            if lid not in taken and (score := _partial_score(words, line)) > 0.0
        ]
        if scored:
            best = max(scored)[2]
            taken.add(best)
            match_by_card[position] = best
    return match_by_card


def _bind_items_to_script(
    items: list[dict[str, object]], ctx: RunContext, unbound: list[str]
) -> list[dict[str, object]]:
    """Key every text-bearing, unkeyed PROMO item to the script line that SAYS it.

    An unkeyed clip joins the render's round-robin pool and lands on whichever
    bare line the spread reaches — so a card reading "High-ticket and sensitive
    conversations…" opened a film whose narration was "Go-to-market that runs
    itself…" (run 01a10d84). Callers cannot key the items themselves (the script
    line ids are minted at run time), so the join happens here, by text.

    Matching is on whole words (punctuation and case ignored), in two passes so
    no card can steal another card's line:

    1. **reserved** — an item that arrives with an explicit ``line_id`` keeps it,
       and that line is taken before any text matching runs;
    2. **exact** — a card whose words ARE a line's words binds to that line,
       wherever it sits in the script (the CTA card binds to the CTA line, never
       to an earlier one-word brand line it happens to contain);
    3. **partial** — only then, a card binds to the free line it quotes most
       strongly (:func:`_partial_score`: word containment one way or the other
       with at least two content words; ties go to the earlier line).

    Items come back in SCRIPT order; one card per line. A text-bearing item that
    matches no free line is DROPPED and its text recorded in ``unbound`` — a card
    is never placed on a line it does not say. Explicitly keyed items, items with
    no text (a base/branding clip), and non-promo runs are returned unchanged
    (the generic blueprint's pool behaviour is not altered by this promo fix).
    """
    lines = _script_lines(ctx)
    if not lines:
        return items
    order = {line_id: index for index, (line_id, _words) in enumerate(lines)}
    taken: set[str] = {str(item["line_id"]) for item in items if item.get("line_id")}
    passthrough: list[dict[str, object]] = []
    pending: list[tuple[dict[str, object], tuple[str, ...]]] = []
    for item in items:
        words = () if item.get("line_id") else _tokens(_item_text(item))
        if words:
            pending.append((item, words))
        else:
            passthrough.append(item)

    match_by_item = _match_cards([words for _item, words in pending], lines, taken)
    bound: list[tuple[int, dict[str, object]]] = []
    for position, (item, _words) in enumerate(pending):
        match = match_by_item.get(position)
        if match is None:
            unbound.append(
                f"motion item {_item_text(item)[:80]!r} matches no free script line; not placed"
            )
            continue
        bound.append((order[match], {**item, "line_id": match}))
    bound.sort(key=lambda pair: pair[0])
    return [item for _index, item in bound] + passthrough
