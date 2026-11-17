"""Shared animated intro/outro title-card shots.

An opt-in branded intro and/or outro card, rendered as a deterministic motion
clip via video_production_service (the same ``animated_title_card`` template the
animated-slide path uses) and injected as a free-standing ``video_url`` shot at
the front/back of the compile shot list. Mirrors the ``motion_graphics_gen``
render sequence (record → poll → fetch → store) and the per-clip soft-fail
discipline used everywhere in this feature family: a card whose render fails is
simply dropped — the video still compiles, just without that card.

The orchestrator compiles through ``video_compile`` (flat ``shots``), so the
intro/outro is produced here as a clip rather than via the VPS title_card part
path (which the main pipeline never calls). Reads ``ctx.config_snapshot``::

    title_card:
      enabled: true
      intro: true            # default true when enabled
      outro: true            # default true when enabled
      title:                 # intro title; defaults to the project title
      subtitle:              # intro subtitle; defaults to the brand name
      outro_title:           # outro title; defaults to the brand name
      outro_subtitle:        # outro subtitle; defaults to ""
      duration_seconds: 3.0  # per card
      width / height / fps   # optional; default from deliverable format

Absent/falsy block → ``(None, None)`` (no cards), so a blueprint can wire this
at zero cost and an un-opted run is byte-for-byte unchanged.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

import httpx

from src.clients.video_production_client import VideoProductionClient
from src.services.stages.shared.degradation import TITLE_CARD_DROPPED, Degradation, degradation
from src.services.stages.shared.motion_slides import (
    format_dims,
    motion_per_line_requested,
    quality_tier,
    run_motion_brand_params,
    theme_motion_params,
)
from src.services.tts_storage import (
    deterministic_media_id,
    find_existing_media,
    store_generated_video,
)

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

log = logging.getLogger(__name__)

_TITLE_TEMPLATE = "animated_title_card"
_RECORD_TIMEOUT_S = 600.0
_DEFAULT_DURATION_S = 3.0
#: Cards may take at most this share of the ORDERED film: 8 live website_review compiles
#: showed fixed 2 x 3.0s cards were 6.08s of a 9.7s film and 58% of the 15.1s ones.
MAX_TITLE_CARD_SHARE = 0.20
#: A shrunk card never runs shorter than this; below it the outro (then every card) is
#: dropped instead — the same 8 compiles: a 10s order has 2.0s to spend, i.e. one card.
MIN_TITLE_CARD_SECONDS = 1.5
#: Stage namespace the cards are stored (and resume-probed) under.
_STAGE_NAME = "title_card"


def title_card_media_id(
    *,
    part_id: str,
    params: Mapping[str, object],
    width: int,
    height: int,
    fps: int,
    duration_ms: int,
) -> str:
    """Content address of ONE rendered card: every input that determines its bytes.

    Resume-stable by construction (no run id, uuid or timestamp), so a pod roll
    mid-compile re-finds the card the dead pod already stored instead of paying
    video_production to record it again. ``params`` (title/subtitle/brand colours)
    is folded as canonical JSON, so a brief edit that changes what the card SAYS or
    LOOKS like addresses a different object and is re-rendered, never served stale.
    ``part_id`` keeps intro and outro distinct even when their text coincides.
    """
    return deterministic_media_id(
        _STAGE_NAME,
        _TITLE_TEMPLATE,
        part_id,
        json.dumps(params, sort_keys=True, separators=(",", ":"), default=str),
        width,
        height,
        fps,
        duration_ms,
    )


def title_card_idempotency_key(run_id: str, media_id: str) -> str:
    """video_production Idempotency-Key for one card record (same shape as ken_burns').

    A re-submit for the same run + content returns the ORIGINAL recording job, so a
    resume that lands while the dead pod's record is still in flight attaches to it
    rather than starting a second paid recording.
    """
    return f"{run_id}:{_STAGE_NAME}:{media_id}"


async def _find_stored_card(
    run_id: str, media_id: str, part_id: str, cache_hits: list[str] | None
) -> str | None:
    """The already-stored card's URL (recording a hit in ``cache_hits``), else ``None``."""
    existing = await find_existing_media(run_id, _STAGE_NAME, media_id, "mp4")
    if existing is not None and cache_hits is not None:
        cache_hits.append(part_id)
    return existing


def _shot_id(media_id: str) -> str:
    """Deterministic UUID-shaped shot id derived from the card's content address."""
    return str(UUID(media_id[:32]))


@dataclass(frozen=True)
class TitleCardPlan:
    """The intro/outro cards a ``title_card`` block makes the compile render."""

    intro: bool
    outro: bool
    seconds_per_card: float

    @property
    def card_count(self) -> int:
        return int(self.intro) + int(self.outro)

    @property
    def total_seconds(self) -> float:
        return self.seconds_per_card * self.card_count

    @property
    def duration_ms(self) -> int:
        """Per-card length as the render (and the cache address) sees it."""
        return round(self.seconds_per_card * 1000)

    @property
    def intro_seconds(self) -> float:
        """Seconds the intro takes — ``0.0`` when this plan has no intro."""
        return self.seconds_per_card if self.intro else 0.0

    @property
    def outro_seconds(self) -> float:
        """Seconds the outro takes — ``0.0`` when this plan has no outro."""
        return self.seconds_per_card if self.outro else 0.0


def _whole_ms(seconds: float) -> float:
    """``seconds`` floored to a whole millisecond (never rounds UP past a budget).

    The epsilon absorbs binary noise (``0.2 * 15 / 2`` is 1.5000000000000002), so the
    budget's seconds and the render's ``duration_ms`` are the same number.
    """
    return math.floor(seconds * 1000 + 1e-6) / 1000


def _fit_to_order(plan: TitleCardPlan, ordered_seconds: float) -> TitleCardPlan | None:
    """``plan`` capped at :data:`MAX_TITLE_CARD_SHARE` of the order; ``None`` = no cards.

    Fits as configured → returned UNCHANGED (the same object). Otherwise the cards
    shrink equally, never below :data:`MIN_TITLE_CARD_SECONDS`; when two cannot both
    fit at that minimum the OUTRO is dropped and the intro takes the budget; when not
    even one minimum-length card fits there are no cards at all.
    """
    budget = MAX_TITLE_CARD_SHARE * ordered_seconds
    while plan.card_count:
        if plan.total_seconds <= budget + 1e-9:
            return plan
        shrunk = _whole_ms(budget / plan.card_count)
        if shrunk >= MIN_TITLE_CARD_SECONDS:
            return TitleCardPlan(intro=plan.intro, outro=plan.outro, seconds_per_card=shrunk)
        if plan.card_count == 1:
            return None
        plan = TitleCardPlan(intro=True, outro=False, seconds_per_card=plan.seconds_per_card)
    return plan


def plan_title_cards(cfg: object, *, ordered_seconds: float | None) -> TitleCardPlan | None:
    """THE gate for title cards — ``None`` when the compile adds none.

    One function read by BOTH :func:`build_title_card_shots` (what the compile
    renders) and :func:`title_card_seconds` (what the website_review order floor and
    script budget reserve), so the reservation cannot drift from the render. Cards are
    OPT-IN: no block, or a block without a truthy ``enabled``, adds zero seconds —
    there are no default cards. ``intro`` / ``outro`` default on inside an enabled
    block; ``duration_seconds`` per card defaults to :data:`_DEFAULT_DURATION_S`.

    ``ordered_seconds`` is the film length the customer ORDERED, as the product's own
    resolver reads it (``website_review/order.website_review_ordered_seconds``). It is
    required-keyword on purpose: every caller must say which order it plans against.
    A product whose SCRIPT reserves the card time must not call this a second time at
    compile: it persists the plan it reserved and the compile renders that record (see
    ``website_review/order.title_card_plan_to_render``) — two calls agree only while
    the code and the snapshot between them stand still. With it, the cards are capped
    by :func:`_fit_to_order` (20% of the order, 1.5s minimum, outro dropped first) — a
    2 x 3.0s block is untouched on an order of 30s and up. ``None`` (a product with no
    resolved order) or a non-positive value plans the block exactly as configured.
    """
    if not isinstance(cfg, dict) or not cfg.get("enabled"):
        return None
    try:
        per_card = float(cfg.get("duration_seconds") or _DEFAULT_DURATION_S)
    except TypeError, ValueError:
        per_card = _DEFAULT_DURATION_S
    plan = TitleCardPlan(
        intro=bool(cfg.get("intro", True)),
        outro=bool(cfg.get("outro", True)),
        seconds_per_card=max(per_card, 0.0),
    )
    if ordered_seconds is None or ordered_seconds <= 0:
        return plan
    return _fit_to_order(plan, ordered_seconds)


def title_card_seconds(cfg: object, *, ordered_seconds: float | None) -> float:
    """Seconds the intro/outro cards add to the film, from a ``title_card`` block.

    :func:`plan_title_cards`' total, so a script stage can reserve the time the
    compile will splice in rather than learn about it from an overrun. A card whose
    render later fails is dropped, so this is the PLANNED spend — the upper bound.
    """
    plan = plan_title_cards(cfg, ordered_seconds=ordered_seconds)
    return plan.total_seconds if plan is not None else 0.0


def _title_params(
    *, title: str, subtitle: str, eyebrow: str, brand_params: dict[str, str]
) -> dict[str, Any]:
    """``animated_title_card`` params — same surface the animated-slide path uses
    (eyebrow + title + subtitle + brand accents; the template keeps its dark
    canvas + light text so contrast holds for any brand)."""
    return {"eyebrow": eyebrow, "title": title, "subtitle": subtitle, **brand_params}


#: Promo's per-line motion brief. When it ran, the promo's END CARD is drawn in
#: the same film style as its cards (see :func:`film_end_card`).
_MOTION_BRIEF_STAGE = "promo_motion_brief"
#: A film-style end card that does NOT continue a closing beat is recorded this
#: much longer and starts this far in: with every mark at 0 its lockup (headline
#: 850 ms, rule +250/700 ms, dial 1200 ms) is complete by then.
FILM_END_CARD_SETTLE_SECONDS = 1.5


@dataclass(frozen=True)
class FilmEndCard:
    """How a promo's end card continues the film its motion brief authored.

    ``lead_seconds`` is the settled lead the card is recorded with (the shot
    starts that far into the clip); ``film_t_ms`` is the film time of the clip's
    FIRST frame, so at the cut the shared ambient layer is exactly where the
    previous shot left it. ``continues_closing_beat``: the card says what the
    closing beat (the call-to-action line's card) says, so it is recorded as
    that same card running on — same params, same clock, the lead is the
    closing beat's whole length — and the cut between the two is invisible.
    """

    lead_seconds: float
    film_t_ms: int
    continues_closing_beat: bool

    def params(
        self, *, title: str, subtitle: str, duration_ms: int, brand_params: Mapping[str, str]
    ) -> dict[str, object]:
        """``animated_title_card`` film-style params, every row settled from the start."""
        return {
            **brand_params,
            "style": "film",
            "eyebrow": "",
            "chapter": "",
            "title": title,
            "subtitle": subtitle,
            # Equal to the title, so the template draws the name once.
            "brand_mark": title,
            "marks": [0, 0, 0, 0],
            "duration_ms": duration_ms,
            "film_t_ms": self.film_t_ms,
        }


def _outro_text(ctx: RunContext, cfg: Mapping[str, object]) -> tuple[str, str]:
    """``(title, subtitle)`` of the outro card of ``cfg`` — one resolution for both styles."""
    brand = ctx.config_snapshot.get("brand") or {}
    brand_name = str(brand.get("name") or "") if isinstance(brand, Mapping) else ""
    project_title = str(ctx.config_snapshot.get("title") or "")
    return (
        str(cfg.get("outro_title") or brand_name or project_title),
        str(cfg.get("outro_subtitle") or ""),
    )


def film_end_card(ctx: RunContext, cfg: object = None) -> FilmEndCard | None:
    """The film-style end card of a promo whose per-line motion brief ran, else ``None``.

    ``None`` — the card keeps the original ``animated_title_card`` look — for
    every other product and for a promo without the brief: a film-style end card
    would not match the rest of an older-style film. PROMO ONLY by construction:
    only promo's blueprint has the ``promo_motion_brief`` stage.
    """
    brief = ctx.prior_output(_MOTION_BRIEF_STAGE) or {}
    if not brief.get("items"):
        return None
    block = _card_block(ctx, cfg)
    closing = brief.get("closing_beat")
    if isinstance(block, Mapping) and isinstance(closing, Mapping):
        shown = closing.get("params")
        beat = shown if isinstance(shown, Mapping) else {}
        said = (str(beat.get("title") or ""), str(beat.get("subtitle") or ""))
        lead_ms = int(closing.get("duration_ms") or 0)
        if lead_ms > 0 and said == _outro_text(ctx, block):
            return FilmEndCard(
                lead_seconds=lead_ms / 1000.0,
                film_t_ms=int(beat.get("film_t_ms") or 0),
                continues_closing_beat=True,
            )
    lead_ms = round(FILM_END_CARD_SETTLE_SECONDS * 1000)
    return FilmEndCard(
        lead_seconds=lead_ms / 1000.0,
        film_t_ms=max(0, int(brief.get("film_ms") or 0) - lead_ms),
        continues_closing_beat=False,
    )


#: ``(intro_shot, outro_shot)`` — each ``None`` when that card is not rendered.
type TitleCardShots = tuple[dict[str, Any] | None, dict[str, Any] | None]


async def build_title_card_shots(
    ctx: RunContext,
    *,
    ordered_seconds: float | None,
    client: VideoProductionClient | None = None,
    notes: list[str] | None = None,
    degradations: list[Degradation] | None = None,
    cache_hits: list[str] | None = None,
) -> TitleCardShots:
    """Plan the snapshot's ``title_card`` block against ``ordered_seconds``, then render.

    :func:`plan_title_cards` + :func:`render_title_card_plan`, for a caller that has
    no plan of its own (``course/stitch``). ``CompileStageBase`` does NOT come through
    here: it resolves ONE plan from its ``title_card_plan`` hook and hands that same
    object to the render and to the cache-hit accounting, so a product whose script
    already RESERVED the card seconds (``website_review``) renders the reservation
    instead of a second, independently computed plan.
    """
    plan = plan_title_cards(ctx.config_snapshot.get("title_card"), ordered_seconds=ordered_seconds)
    return await render_title_card_plan(
        ctx,
        plan,
        client=client,
        notes=notes,
        degradations=degradations,
        cache_hits=cache_hits,
    )


def _card_block(ctx: RunContext, cfg: object) -> object:
    """The ``title_card`` block a card's text comes from: the caller's, else the run's."""
    return ctx.config_snapshot.get("title_card") if cfg is None else cfg


def _card_brand_params(ctx: RunContext, brand: dict[str, object]) -> dict[str, str]:
    """The brand params a title card is recorded with.

    A promo that opted in to per-line motion (``direction.motion == "per_line"``):
    one resolution shared with its motion cards, so the end card is not a second
    design system (frame QA of run 01a11082). Every other run — every other
    product, and a promo without the flag — keeps the params it always sent,
    byte-identical, so its cached cards still address.
    """
    if ctx.prior_output("promo_script") is not None and motion_per_line_requested(
        ctx.config_snapshot
    ):
        return run_motion_brand_params(ctx.config_snapshot)
    return theme_motion_params(brand, ctx.config_snapshot.get("theme") or {})


def _record_card_drop(
    part_id: str,
    kind: str,
    why: str,
    planned_seconds: float,
    notes: list[str] | None,
    degradations: list[Degradation] | None,
) -> None:
    """One dropped PLANNED card ⇒ one prose note + one coded TITLE_CARD_DROPPED.

    ``detail`` carries the planned seconds: the card's time was reserved inside the
    order (``reconcile_timeline`` subtracts ``title_card_seconds``), so the delivered
    film comes out that much shorter than planned.
    """
    reason = f"title_card: {part_id} {why}"
    log.warning("%s", reason)
    if notes is not None:
        notes.append(reason)
    if degradations is not None:
        degradations.append(
            degradation(
                TITLE_CARD_DROPPED,
                f"{part_id} dropped ({kind}, {planned_seconds:.2f}s planned): {why}",
            )
        )


async def render_title_card_plan(
    ctx: RunContext,
    plan: TitleCardPlan | None,
    *,
    client: VideoProductionClient | None = None,
    notes: list[str] | None = None,
    degradations: list[Degradation] | None = None,
    cfg: object = None,
    cache_hits: list[str] | None = None,
    settled_lead_seconds: float = 0.0,
) -> TitleCardShots:
    """Render exactly ``plan``'s intro/outro cards; return ``(intro_shot, outro_shot)``.

    ``settled_lead_seconds`` (default ``0`` = unchanged): the card is RECORDED
    this much longer than planned and the shot starts that far into the clip
    (``clip_lead_seconds``), so the template's build-in is cut off and the title,
    subtitle and brand mark are all on screen from the cut. The shot still lasts
    exactly the planned seconds. Run 01a10fcf (2026-10-06): the end card's brand
    mark faded up 0.75s+ after the cut (``animated_title_card`` brings it in at
    2000-2800ms, the headline at 650-1500ms).

    ``cfg`` is the ``title_card`` block to render from; ``None`` reads it off
    ``ctx.config_snapshot`` (every caller before ``CompileStageBase.title_card_config``
    existed). A product that derives a default card (promo's branded end card)
    passes its resolved block here.

    Each side is independently soft-failed to ``None`` so a render error never
    breaks the compile. ``(None, None)`` when ``plan`` is ``None`` (no cards) or the
    snapshot carries no ``title_card`` block to take the card text from.

    ``plan`` is rendered AS GIVEN — this function never caps, shrinks or re-plans it.
    Whoever resolved it (``plan_title_cards``, or a script stage's persisted
    reservation) is the only authority on how many seconds the cards take.

    ``notes`` is the OPTIONAL prose degrade channel (same ``list[str]`` sink
    ``generic/site_capture`` and ``shared/visual_sync`` use): a caller that owns a
    stage output with a ``notes`` field passes its list in, and a dropped card
    appends ONE sentence naming the card and the exception. That is a durable
    record of the drop for a human — a ``log.warning`` evaporates with the pod —
    but on the wire it is indistinguishable from every other kind of soft-failed
    stage note: the DAG executor stamps it the generic :data:`UNCLASSIFIED_NOTE`
    code, so "how many runs shipped without their branded title card" could not
    be queried.

    ``degradations`` is the OPTIONAL STRUCTURED twin (same ``list[Degradation]``
    sink ``build_audio_bed`` uses): a dropped card appends ONE
    :data:`TITLE_CARD_DROPPED` entry whose ``detail`` names which of the two
    failure shapes it was (a poll TIMEOUT vs a genuine RENDER error — see the
    ``_render`` closure below) so a reader can tell a slow/wedged recorder from a
    bad template/params without re-parsing prose. Behaviour is unchanged either
    way: the card is still dropped, the compile still runs, and ``None`` for
    either sink (no caller wired in) is a no-op — this function never raises on
    a render failure.

    ``cache_hits`` is an OPTIONAL sink: the ``part_id`` of every card served from
    the content-addressed store (see :func:`title_card_media_id`) instead of being
    re-recorded is appended, so a resumed compile can prove on the run record that
    a pod roll did not re-pay for the cards.
    """
    cfg = _card_block(ctx, cfg)
    if plan is None or not plan.card_count or not isinstance(cfg, dict):
        return (None, None)

    brand = ctx.config_snapshot.get("brand") or {}
    brand_name = str(brand.get("name") or "")
    project_title = str(ctx.config_snapshot.get("title") or "")
    deliverable = ctx.config_snapshot.get("deliverable") or {}
    default_w, default_h = format_dims(deliverable.get("format"), quality_tier(ctx.config_snapshot))
    width = int(cfg.get("width") or default_w)
    height = int(cfg.get("height") or default_h)
    fps = int(cfg.get("fps") or 30)
    lead_ms = max(0, round(settled_lead_seconds * 1000))
    duration_ms = plan.duration_ms + lead_ms
    planned_seconds = plan.duration_ms / 1000.0
    brand_params = _card_brand_params(ctx, brand)
    vp_client = client or VideoProductionClient()
    run_id = str(ctx.run_id)

    async def _render(part_id: str, line_id: str, params: dict[str, Any]) -> dict[str, Any] | None:
        if not params.get("title"):
            # Nothing to show — skip this card rather than render a blank. But it was
            # PLANNED: ``reconcile_timeline`` already reserved its seconds out of the
            # order (``title_card_seconds``), so dropping it silently ships a film that
            # much short with no coded reason. Record it like any other dropped card.
            _record_card_drop(
                part_id,
                "no_title",
                "planned but has no title text, card dropped",
                planned_seconds,
                notes,
                degradations,
            )
            return None
        media_id = title_card_media_id(
            part_id=part_id,
            params=params,
            width=width,
            height=height,
            fps=fps,
            duration_ms=duration_ms,
        )
        shot: dict[str, object] = {
            "id": _shot_id(media_id),
            "line_id": line_id,
            "seconds": plan.duration_ms / 1000.0,
            # The compiler seeks a silent clip LONGER than its shot by this much
            # (``clip_lead_seconds`` is honoured up to clip length - shot length).
            **({"clip_lead_seconds": lead_ms / 1000.0} if lead_ms else {}),
        }
        try:
            # Sub-stage checkpoint: a card the previous (rolled) pod already stored
            # is reused, not re-recorded and re-paid. A probe failure returns None
            # (indeterminate) and falls through to a normal render.
            existing = await _find_stored_card(run_id, media_id, part_id, cache_hits)
            if existing is not None:
                return {**shot, "video_url": existing}
            job_id = await vp_client.record_animation(
                template_id=_TITLE_TEMPLATE,
                params=params,
                part_id=part_id,
                fps=fps,
                width=width,
                height=height,
                duration_ms=duration_ms,
                output_format="mp4",
                idempotency_key=title_card_idempotency_key(run_id, media_id),
            )
            await vp_client.wait_for_content_recording(job_id, timeout=_RECORD_TIMEOUT_S)
            video_bytes, _ctype, _disp = await vp_client.get_recording_video(job_id)
            video_url = await store_generated_video(
                run_id=run_id,
                stage_name=_STAGE_NAME,
                video_id=media_id,
                video_bytes=video_bytes,
                video_format="mp4",
            )
            return {
                **shot,
                "video_url": video_url,
            }
        except Exception as exc:  # noqa: BLE001 -- soft-fail: video compiles without this card
            # The exception CLASS is load-bearing: it is what separates an infra
            # timeout (httpx.TimeoutException, or the bare TimeoutError
            # BaseServiceClient.poll_job raises on wall-clock expiry inside
            # wait_for_content_recording) from a content/render error (a 4xx/5xx
            # from video_production, a bad template/params, a storage failure) —
            # different owners, so the structured code's `detail` records which.
            is_timeout = isinstance(exc, (TimeoutError, httpx.TimeoutException))
            _record_card_drop(
                part_id,
                "timeout" if is_timeout else "render_error",
                f"render failed, card dropped: {type(exc).__name__}: {exc}",
                planned_seconds,
                notes,
                degradations,
            )
            return None

    # intro/outro default to on when the block is enabled (opt-in is the block itself).
    intro_shot = None
    if plan.intro:
        intro_shot = await _render(
            "title-intro",
            "__title_intro__",
            _title_params(
                title=str(cfg.get("title") or project_title),
                subtitle=str(cfg.get("subtitle") or brand_name),
                eyebrow="",
                brand_params=brand_params,
            ),
        )
    outro_shot = None
    if plan.outro:
        outro_title, outro_subtitle = _outro_text(ctx, cfg)
        film = film_end_card(ctx, cfg)
        outro_shot = await _render(
            "title-outro",
            "__title_outro__",
            # A promo whose motion brief ran: the end card is the film's own
            # closing lockup (the caller's settled lead is ``film.lead_seconds``).
            film.params(
                title=outro_title,
                subtitle=outro_subtitle,
                duration_ms=duration_ms,
                brand_params=brand_params,
            )
            if film is not None
            else _title_params(
                title=outro_title,
                subtitle=outro_subtitle,
                eyebrow="",
                brand_params=brand_params,
            ),
        )
    return (intro_shot, outro_shot)


def splice_title_cards(
    shots: list[dict[str, Any]],
    intro_shot: dict[str, Any] | None,
    outro_shot: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Prepend the intro shot and append the outro shot to a built shot list.

    Pure (no I/O) so the bracket ordering is unit-testable on its own.
    """
    if not intro_shot and not outro_shot:
        return shots
    spliced: list[dict[str, Any]] = []
    if intro_shot:
        spliced.append(intro_shot)
    spliced.extend(shots)
    if outro_shot:
        spliced.append(outro_shot)
    return spliced
