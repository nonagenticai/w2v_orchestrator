"""promo_render — realise the per-line visual plan and compile the promo.

Joins the storyboard (one shot per script line, each tagged with a
``visual_type``) with ``promo_narrate`` audio (by ``line_id``) and the four
visual producers, routing each line to exactly one source per its plan:

* ``footage``   → a ``site_capture`` per-screen asset: the reveal *clip* on the
                  line that introduces a screen (``video_by_line``), the steady
                  *still* on lines that hold it (``image_by_line``)
* ``html``      → a ``promo_html`` redesigned-HTML screenshot (cycling)
* ``ai_image``  → an ``ai_image_gen`` image (cycling)
* ``slide``     → the line's ``promo_slides`` slide

Every line also gets its ``promo_slides`` slide as a **universal fallback** in
``image_by_line``, so a footage/html/ai asset that fails to materialise degrades
to a designed slide rather than a blank shot (and a missing slide degrades to the
storyboard text card). Burned-in captions are opt-in per deliverable
(``captions_mode=burned_in`` via ``caption_clip_refs`` → the word-timed seam in
``CompileStageBase.handle``); on the default ``captions_mode=none`` no captions
are burned and an accurate soft-CC track is attached at upload time from the
narration timing.

Operator-curated real media (pinned / extracted / discovered, already
re-published to run storage) overlays this plan: **images** fill the
lowest-value image lines (``ai_image`` without a story image, captured-empty
footage, redundant repeat-stills); **video** clips then take the remaining
lowest-value lines (bare slide / round-robin AI-pool) — ``video_url`` wins over
the slide/AI image in the compositor, so a curated clip plays full-frame.
Captured live-demo footage, story-keyed AI images and curated stills are never
overridden by curated video.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import cycle
from typing import TYPE_CHECKING, Any, ClassVar

from src.config import get_settings
from src.services.captions import caption_text_by_line
from src.services.stages.base import (
    ArtifactUrl,
    Degradation,
    StageInput,
    StageOutputWithNotes,
)
from src.services.stages.primitives.compile_stage import (
    CompileStageBase,
    PayloadDict,
    captions_mode,
    output_dimension_fields,
    playable_duration_seconds,
    resolution_tags,
)
from src.services.stages.promo.fill import (
    CTA_MIN_SECONDS,
    apply_holds,
    default_end_card,
    end_card_config,
    extend_shot,
    promo_fill,
)
from src.services.stages.shared.compute_plan import transition_gain_db
from src.services.stages.shared.degradation import (
    CAPTION_KARAOKE_OFF,
    END_CARD_DROPPED,
    MOTION_CLIP_UNUSED,
    advisory,
    degradation,
)
from src.services.stages.shared.duration_conformance import (
    DurationConformance,
    duration_conformance_advisories,
    duration_conformance_degradations,
    observe_duration_conformance,
    picture_truncation_degradations,
    promo_ordered_seconds,
)
from src.services.stages.shared.footage_plans import resolve_footage_plans
from src.services.stages.shared.motion_slides import (
    motion_per_line_requested,
    run_motion_brand_params,
)
from src.services.stages.shared.sound_design import (
    resolve_audio_bed,
    sfx_by_line,
    sound_config,
    transition_sfx_id,
)
from src.services.stages.shared.title_cards import film_end_card
from src.services.stages.shot_assembly import ShotDict, assemble_shots, footage_tracks_by_line
from src.services.tenant_defaults import captions_karaoke as resolve_captions_karaoke
from src.services.visual_media import (
    assign_real_media_videos_by_line,
    collect_ai_video_urls,
    collect_motion_clips_by_line,
    collect_real_media_urls,
    collect_real_media_videos,
    is_fetchable_url,
)
from src.services.word_timed_captions import ClipRef

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

# The compile-fetchability guard lives in ``visual_media`` now (shared with the
# producing stages); re-exported here under its long-standing private name so
# existing call sites / tests keep working.
_fetchable = is_fetchable_url


def _assign_footage(
    footage_line_ids: list[str],
    primary_urls: list[str],
    cutaway_urls: list[str],
) -> dict[str, str]:
    """Map footage ``line_id`` → clip URL with the demo clip as the spine.

    The **primary** clip (the scripted demo walkthrough) carries essentially the
    whole footage timeline — the compiler's per-shot seek-spread then pans across
    its long recording, so each reuse shows a *different* screen rather than the
    same frame. The **cutaway** clips (landing page, book) each appear exactly
    once, at evenly-spread slots, so no static page recurs (the old round-robin
    landed the landing page on ~5 shots).

    Fallbacks: with no primary identified, round-robin across all clips (legacy
    behaviour for promos whose capture had no interactive demo). With no footage
    clips at all, assign nothing (those lines keep their slide fallback).
    """
    if not footage_line_ids:
        return {}
    if not primary_urls:
        pool = cycle(cutaway_urls) if cutaway_urls else None
        if pool is None:
            return {}
        return {lid: next(pool) for lid in footage_line_ids}

    spine = cycle(primary_urls)  # usually one demo clip; round-robin if several
    out = {lid: next(spine) for lid in footage_line_ids}
    n = len(footage_line_ids)
    c = len(cutaway_urls)
    # Only intersperse cutaways when there are clearly enough footage slots for the
    # demo to stay dominant; otherwise keep the timeline all-demo.
    if c and n >= 2 * (c + 1):
        placed: set[int] = set()
        for k, clip in enumerate(cutaway_urls):
            idx = min(n - 1, max(0, round((k + 1) * n / (c + 1))))
            while idx in placed and idx < n - 1:
                idx += 1
            placed.add(idx)
            out[footage_line_ids[idx]] = clip
    return out


def _resolve_ai_image_url(
    line_id: str,
    *,
    keyed_ai: dict[str, str],
    real_media: Iterator[str],
    ai: Iterator[str] | None,
    dedicated: set[str],
    generated: set[str] | None = None,
) -> str | None:
    """Pick an ai_image line's image: story-keyed wins, else a curated real-media
    image (once each), else the round-robin AI pool; ``None`` keeps the slide.

    A story-keyed or curated image is a *dedicated* visual — its line is recorded
    in ``dedicated`` so the curated-video overlay never displaces it; a line that
    falls to the generic AI pool (or keeps its slide) stays free for a clip.
    A story-keyed image is also recorded in ``generated`` (a keyed motion clip
    may still take that line — see :func:`_overlay_motion_clips`).
    """
    keyed = keyed_ai.get(line_id)
    curated = None if keyed else next(real_media, None)
    if keyed or curated:
        dedicated.add(line_id)
    if keyed and generated is not None:
        generated.add(line_id)
    return keyed or curated or (next(ai) if ai is not None else None)


def _resolve_keyed_still(
    line_id: str,
    *,
    keyed_still: dict[str, str],
    repeat_still: set[str],
    real_media: Iterator[str],
    image_by_line: dict[str, str],
    protected_lines: set[str],
    redundant_still_lines: set[str],
) -> None:
    """Place a footage line keyed to a per-screen still, mutating the plan maps.

    A *repeat-still* (re-shows an already-revealed screen) prefers a curated
    real-media image when one is available — that line then becomes *dedicated*
    (``protected_lines``). A repeat-still with no curated image keeps its
    redundant screenshot and is recorded in ``redundant_still_lines`` so the
    curated/AI VIDEO overlay may later take it. A screen's FIRST still (not in
    ``repeat_still``) is genuine demo content — kept and left protected.
    """
    curated = next(real_media, None) if line_id in repeat_still else None
    if curated:
        image_by_line[line_id] = curated
        protected_lines.add(line_id)
    else:
        image_by_line[line_id] = keyed_still[line_id]
        if line_id in repeat_still:
            redundant_still_lines.add(line_id)


#: ``animated_title_card`` (video_production) brings the headline in at
#: 650-1500ms, the subtitle at 1250-2000ms and the brand mark at 2000-2800ms.
#: The promo's end card is recorded this much longer and starts this far in, so
#: all three are on screen from the cut (run 01a10fcf: the mark arrived late).
END_CARD_SETTLED_LEAD_SECONDS = 2.8

# Motion-graphics clips are DESIGNED cards (kinetic text, stat, quote): their
# text runs to the card's edges, so they are letterboxed into the frame, never
# cover-cropped. ``cover`` cut the top and bottom lines off a square kinetic
# card in a 16:9 film (run 01a10d84, 2026-10-05). A clip recorded at the film's
# own aspect (motion_graphics_gen's default) fills the frame either way.
_MOTION_FIT = "contain"


def _carded_lines(
    shots: list[ShotDict], motion_clip_by_line: dict[str, str] | None
) -> dict[str, str]:
    """``line_id -> card url`` for each storyboard line that has a recorded card."""
    line_ids = {str(shot["line_id"]) for shot in shots}
    return {
        line_id: url for line_id, url in (motion_clip_by_line or {}).items() if line_id in line_ids
    }


def _uncarded(shots: list[ShotDict], carded: dict[str, str]) -> list[ShotDict]:
    """The shots whose line has no card (``shots`` itself when nothing is carded)."""
    if not carded:
        return shots
    return [shot for shot in shots if str(shot["line_id"]) not in carded]


def _uncarded_keys(by_line: dict[str, str], carded: dict[str, str]) -> dict[str, str]:
    """``by_line`` without the carded lines (itself when nothing is carded)."""
    if not carded:
        return by_line
    return {line_id: url for line_id, url in by_line.items() if line_id not in carded}


#: Per-shot storyboard fields that frame FOOTAGE (in/out points, anchor, an
#: author's fit). A line whose shot is its motion card carries none of them: the
#: card is recorded for the whole shot at the film's frame.
_FOOTAGE_SHOT_KEYS = ("clip_anchor", "clip_lead_seconds", "clip_out_seconds", "fit")


def _without_footage_framing(
    storyboard_shots: list[ShotDict], carded: dict[str, str]
) -> list[ShotDict]:
    """``storyboard_shots`` with the footage framing dropped from each carded line."""
    if not carded:
        return storyboard_shots
    return [
        {key: value for key, value in shot.items() if key not in _FOOTAGE_SHOT_KEYS}
        if str(shot["line_id"]) in carded
        else shot
        for shot in storyboard_shots
    ]


def _overlay_motion_clips(
    video_by_line: dict[str, str],
    fit_by_line: dict[str, str],
    *,
    shots: list[dict[str, Any]],
    protected: set[str],
    motion_clip_by_line: dict[str, str] | None,
    motion_clip_urls: list[str] | None,
    generated_still_lines: set[str] | None = None,
) -> None:
    """Lay branded motion-graphics clips into ``video_by_line`` in place.

    Keyed clips (from a motion_brief) join 1:1 by ``line_id``; unkeyed clips
    spread first across the lines the storyboard planned as ``motion_graphics``,
    then any surplus across the remaining bare lines. Any line already carrying
    a visual (footage / curated media / ``protected``) is left untouched, so
    motion sits above generative AI video but below real footage and curated
    media. Empty inputs → no change.

    ``generated_still_lines`` are lines whose dedicated visual is a story-keyed
    GENERATED still (``ai_image_gen``). A clip KEYED to such a line takes it:
    the brief ordered motion for exactly that line and the clip was rendered and
    paid for, while the still is a generic illustration. Run 01a10fcf
    (2026-10-06): the brief ordered three kinetic cards for lines 1-3, all three
    rendered, and lines 2 and 3 shipped AI stills because the storyboard had
    typed them ``ai_image`` — two paid clips dropped with no record. Unkeyed
    pool clips still never displace a keyed still.
    """
    keyed_protected = (set(video_by_line) | protected) - (generated_still_lines or set())
    for line_id, url in (motion_clip_by_line or {}).items():
        if line_id not in keyed_protected:
            video_by_line[line_id] = url
            fit_by_line[line_id] = _MOTION_FIT
    pool = list(motion_clip_urls or [])
    # Lines the storyboard planned as ``motion_graphics`` take the unkeyed pool
    # first (spread across them); only clips left over spill onto other bare
    # lines. Without this the pool landed on the title/CTA slides while the
    # planned motion lines fell back to slides.
    planned = [
        shot
        for shot in shots
        if str(shot.get("visual_type") or "") == "motion_graphics"
        and str(shot["line_id"]) not in set(video_by_line) | protected
    ]
    if planned and pool:
        first, pool = pool[: len(planned)], pool[len(planned) :]
        overlay, _fit = assign_real_media_videos_by_line(planned, first, distribute=True)
        video_by_line.update(overlay)
        fit_by_line.update(dict.fromkeys(overlay, _MOTION_FIT))
    pool_protected = set(video_by_line) | protected
    overlay, _fit = assign_real_media_videos_by_line(
        shots, pool, skip_line_ids=pool_protected, distribute=True
    )
    video_by_line.update(overlay)
    fit_by_line.update(dict.fromkeys(overlay, _MOTION_FIT))


def _plan_visuals(
    shots: list[dict[str, Any]],
    *,
    slide_by_line: dict[str, str],
    footage_primary: list[str],
    footage_cutaways: list[str],
    html_urls: list[str],
    ai_urls: list[str],
    ai_image_by_line: dict[str, str] | None = None,
    footage_screenshot_by_line: dict[str, str] | None = None,
    footage_clip_by_line: dict[str, str] | None = None,
    footage_repeat_still_lines: set[str] | None = None,
    real_media_urls: list[str] | None = None,
    real_media_video_urls: list[str] | None = None,
    ai_video_urls: list[str] | None = None,
    motion_clip_by_line: dict[str, str] | None = None,
    motion_clip_urls: list[str] | None = None,
    cards_first: bool = False,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Route each shot to one visual per its ``visual_type``.

    ``cards_first`` (a run that opted in to per-line motion): a line's keyed
    motion card IS its shot — it outranks captured footage, html and curated
    media for that line, whatever the storyboard typed it. Such a line is planned
    out of every other source (it takes no footage slot, no html screenshot, no
    curated image or clip); a line WITHOUT a card is routed exactly as below.
    Default ``False``: the ranking documented below, unchanged.

    Returns ``(image_by_line, video_by_line, fit_by_line)``. Slides seed
    ``image_by_line`` as the universal fallback; footage/html/ai-image lines
    then override (footage via ``video_by_line``, which wins over the slide
    image in the compositor). A type whose source list is empty keeps the
    fallback slide.

    ``real_media_urls`` is the operator-pinned / extracted / discovered real
    imagery (already re-published, compiler-fetchable). It is consumed **once
    each** (no repeats) to fill lines that would otherwise have no real visual:
    ``ai_image`` lines without a story-keyed AI image (before the generic AI
    pool), ``footage`` lines that captured no asset (before the slide
    fallback), and ``footage`` *repeat-still* lines named in
    ``footage_repeat_still_lines`` (which merely re-show an already-revealed
    screen) — so curated media surfaces whether or not the run produced footage,
    including all-footage ``demo_sync`` promos. Captured reveal *clips* and a
    screen's FIRST still are never overridden; only redundant repeats are.

    Footage routing keys per line off the promo_visual_director mapping:

    * a line keyed in ``footage_clip_by_line`` (the first line that *reveals* a
      screen) gets that screen's short interaction *clip* in ``video_by_line``
      — the click landing on the screen, played as motion under the line that
      introduces it;
    * a line keyed in ``footage_screenshot_by_line`` (a line that *holds* an
      already-revealed screen) gets that screen's steady *still* in
      ``image_by_line``;
    * any remaining footage line falls back to the legacy panned-clip spine via
      :func:`_assign_footage` (demo clip = spine, landing/book = single cutaways).

    ``real_media_video_urls`` is the operator-pinned / discovered real **video**
    (already re-published, compiler-fetchable). After the image plan is laid, each
    clip is consumed **once each** over the remaining lowest-value lines — bare
    slide-fallback and round-robin AI-pool lines, plus the **redundant
    repeat-still** lines that kept their screenshot (so curated motion surfaces on
    all-footage ``demo_sync`` promos, the only place it can land) — keyed into
    ``video_by_line`` (``video_url`` wins over the slide/AI/still ``image_url`` in
    the compositor) and marked ``"cover"``. Lines already carrying a dedicated
    visual are protected: captured footage clips/spine, a screen's reveal clip and
    its FIRST still, story-keyed AI images, html redesign screenshots, and lines a
    curated real-media *image* already took (curated image beats curated video on a
    repeat-still) — so a clip never fights an image for the same line. Empty
    ``real_media_video_urls`` → ``video_by_line`` unchanged (regression-safe).

    ``fit_by_line`` marks AI-image, keyed-still and keyed-clip lines ``"cover"``
    so the compositor crops-to-fill them (full-frame, no navy bars); every other
    line keeps the default ``contain`` so slide text is never cropped.
    """
    image_by_line: dict[str, str] = dict(slide_by_line)
    fit_by_line: dict[str, str] = {}
    keyed_ai = ai_image_by_line or {}
    keyed_still = footage_screenshot_by_line or {}
    keyed_clip = footage_clip_by_line or {}
    repeat_still = footage_repeat_still_lines or set()
    html = cycle(html_urls) if html_urls else None
    ai = cycle(ai_urls) if ai_urls else None
    # Real media is used once each (not cycled) so the same stock photo doesn't
    # repeat; when exhausted, lines fall through to the AI pool / slide fallback.
    real_media = iter(real_media_urls or [])
    # Per-line motion: the lines whose shot is their card (see ``cards_first``)
    # are planned out of every other source — ``open_shots`` is the rest.
    carded = _carded_lines(shots, motion_clip_by_line) if cards_first else {}
    open_shots = _uncarded(shots, carded)
    keyed_still = _uncarded_keys(keyed_still, carded)
    keyed_clip = _uncarded_keys(keyed_clip, carded)

    # Only footage lines keyed to neither a per-screen clip nor still fall to the
    # legacy panned-clip spine; keyed ones are placed in the loop below.
    footage_line_ids = [
        str(shot["line_id"])
        for shot in open_shots
        if str(shot.get("visual_type") or "slide") == "footage"
        and str(shot["line_id"]) not in keyed_still
        and str(shot["line_id"]) not in keyed_clip
    ]
    video_by_line = _assign_footage(footage_line_ids, footage_primary, footage_cutaways)
    # Per-screen reveal clips win over the legacy spine (keyed video beats the
    # slide image in the compositor; nothing else writes these line ids).
    video_by_line.update(keyed_clip)
    # A carded line's card is laid before anything else can claim the line; being
    # in ``video_by_line`` protects it from every overlay below.
    video_by_line.update(carded)
    fit_by_line.update(dict.fromkeys(carded, _MOTION_FIT))

    # Lines that received a dedicated visual (real-media image, story-keyed AI
    # image, or html screenshot) — protected from the curated-video overlay below
    # so a clip never displaces an already-curated image. (Captured footage and
    # per-screen stills are covered by ``video_by_line`` / ``keyed_still``.)
    protected_lines: set[str] = set()
    # Repeat-still lines that kept their redundant screenshot (no curated image
    # was available) — the only stills curated/AI VIDEO may overlay; a screen's
    # reveal clip and FIRST still stay protected. On all-footage demo_sync promos
    # these are the sole slots where curated motion can land.
    redundant_still_lines: set[str] = set()
    # Lines holding a story-keyed GENERATED still — a keyed motion clip outranks
    # it (see _overlay_motion_clips).
    generated_still_lines: set[str] = set()

    for shot in open_shots:
        line_id = str(shot["line_id"])
        vtype = str(shot.get("visual_type") or "slide")
        if vtype == "footage":
            if line_id in keyed_clip:
                # Reveal clip already in video_by_line; cover-fit so it fills.
                fit_by_line[line_id] = "cover"
            elif line_id in keyed_still:
                # A repeat-still line re-shows an ALREADY-revealed screen as a
                # static image — the lowest-value slot. A curated real-media image
                # takes it when available (demo_sync promos are all-footage, so
                # this is the only place curated media can land); otherwise the
                # redundant screenshot is kept and recorded so the curated/AI
                # VIDEO overlay below may take it. Reveal clips and a screen's
                # FIRST still are always kept.
                _resolve_keyed_still(
                    line_id,
                    keyed_still=keyed_still,
                    repeat_still=repeat_still,
                    real_media=real_media,
                    image_by_line=image_by_line,
                    protected_lines=protected_lines,
                    redundant_still_lines=redundant_still_lines,
                )
                fit_by_line[line_id] = "cover"
            elif line_id not in video_by_line:
                # No captured asset for this footage line (empty/sparse capture):
                # borrow a real-media image before degrading to the slide.
                url = next(real_media, None)
                if url:
                    image_by_line[line_id] = url
                    fit_by_line[line_id] = "cover"
                    protected_lines.add(line_id)
            continue  # else assigned via _assign_footage (or slide fallback)
        if vtype == "html" and html is not None:
            image_by_line[line_id] = next(html)
            protected_lines.add(line_id)
        elif vtype == "ai_image":
            url = _resolve_ai_image_url(
                line_id,
                keyed_ai=keyed_ai,
                real_media=real_media,
                ai=ai,
                dedicated=protected_lines,
                generated=generated_still_lines,
            )
            if url:
                image_by_line[line_id] = url
                fit_by_line[line_id] = "cover"
        # slide / unmatched: keep the fallback slide already in image_by_line.

    # Curated real-media VIDEO overlays the remaining lowest-value lines (bare
    # slide / round-robin AI-pool) plus the redundant repeat-still lines that
    # kept their screenshot (on all-footage demo_sync promos the only slots for
    # curated motion), once each; video_url wins over the image already there.
    # Protected lines (captured footage, reveal clips, a screen's FIRST still,
    # keyed AI / html / curated-image lines) keep their visual. Empty → no change.
    still_protected = set(keyed_still) - redundant_still_lines
    protected = set(video_by_line) | still_protected | protected_lines
    video_overlay, video_fit = assign_real_media_videos_by_line(
        shots, real_media_video_urls or [], skip_line_ids=protected
    )
    video_by_line.update(video_overlay)
    fit_by_line.update(video_fit)

    # Branded motion-graphics clips (motion_graphics_gen, opt-in) sit BELOW
    # captured footage + operator-curated real media but ABOVE generative AI
    # video: deterministic, on-brand motion is the preferred fill when no real
    # footage exists. (See _overlay_motion_clips.) Empty → no change.
    _overlay_motion_clips(
        video_by_line,
        fit_by_line,
        shots=shots,
        protected=still_protected | protected_lines,
        motion_clip_by_line=motion_clip_by_line,
        motion_clip_urls=motion_clip_urls,
        generated_still_lines=generated_still_lines,
    )

    # AI-generated clips (ai_video_gen, opt-in) are the lowest-priority motion:
    # curated real-media video (just laid above) wins, so they fill only the
    # remaining lowest-value lines (bare slide / round-robin AI-pool / a still-
    # redundant repeat-still) — SPREAD EVENLY across them (``distribute=True``) so
    # generative motion recurs through the timeline rather than clustering up
    # front. Every dedicated visual — captured footage, reveal clips, first
    # stills, keyed AI / html / curated-image lines, and curated-video lines (now
    # in ``video_by_line``) — stays protected. Empty ``ai_video_urls`` → no change
    # (regression-safe).
    ai_protected = set(video_by_line) | still_protected | protected_lines
    ai_video_overlay, ai_video_fit = assign_real_media_videos_by_line(
        shots, ai_video_urls or [], skip_line_ids=ai_protected, distribute=True
    )
    video_by_line.update(ai_video_overlay)
    fit_by_line.update(ai_video_fit)
    return image_by_line, video_by_line, fit_by_line


class PromoRenderInput(StageInput):
    pass


class PromoRenderOutput(StageOutputWithNotes):
    # Declared output contract — enforced once, centrally, in
    # `runs/dag_executor.py::_enforce_produces`. This is a TERMINAL artifact
    # producer: `artifact_url` IS the deliverable, not a hint about it, so an
    # empty one is never an acceptable outcome. Hence `produces_required` — the
    # stage FAILS instead of degrading. Degrade was the wrong default here: it let
    # the run finish while handing the user a reference to nothing, which is the
    # exact defect class this contract exists to close.
    produces: ClassVar[tuple[str, ...]] = ("artifact_url",)
    produces_required: ClassVar[bool] = True

    artifact_url: ArtifactUrl
    duration_seconds: float
    captions_burned: bool
    # Actual encoded output resolution (ffprobed by video_compiler). None on
    # older compiler builds / probe miss. Lets the e2e check assert the tier.
    width: int | None = None
    height: int | None = None
    # Delivered-vs-ORDERED duration, recorded so a deviation is visible in the
    # run record: promo masters measured 129% of their ordered length on
    # 2026-09-10 and nothing anywhere noticed, because nothing compared the two
    # (this stage echoed the encoder's own measurement out unchecked).
    # Observation ONLY — it never fails or degrades the stage, and it is
    # deliberately NOT a `notes` entry (a note flips the whole run to
    # `degraded`). `None` means no usable duration was ordered, which is NOT the
    # same fact as "on target".
    duration_conformance: DurationConformance | None = None
    # The music/ambient bed this compile sent and the compiler's report on it
    # (``compile_stage.audio_bed_record``). ``None`` only on a compile that never
    # reached the request (it failed first).
    audio_bed: PayloadDict | None = None


def _brief_closing_beat(ctx: RunContext) -> tuple[str, float] | None:
    """``(closing line id, end-card seconds)`` the motion BRIEF sized its closing card for.

    Set only when ``promo_motion_brief`` ran, its closing beat absorbed the
    default end card (``ClosingBeat.end_card_seconds``) and that beat is the
    storyboard's last line. This is what the brief PLANNED — whether the film
    does it is :func:`_absorbed_end_card`'s call, from the shots actually laid.
    """
    if ctx.config_snapshot.get("title_card") is not None:
        return None
    closing = (ctx.prior_output("promo_motion_brief") or {}).get("closing_beat")
    if not isinstance(closing, dict):
        return None
    seconds = float(closing.get("end_card_seconds") or 0.0)
    line_id = str(closing.get("line_id") or "")
    story = (ctx.prior_output("promo_storyboard") or {}).get("shots") or []
    if seconds <= 0 or not story or str(story[-1].get("line_id")) != line_id:
        return None
    return line_id, seconds


def _absorbed_end_card(ctx: RunContext, laid_shots: list[ShotDict]) -> tuple[str, float] | None:
    """``(closing line id, end-card seconds)`` when the closing shot IS the end card.

    Decided from the LAID plan, not from the brief's output: the closing shot
    must carry the closing line's own recorded card. A brief that planned the
    closing beat but whose card was not recorded (or was not laid) absorbs
    nothing — the closing shot keeps its own length and the separate end card is
    rendered. ``build_shots`` and ``title_card_config`` both read this one
    answer, so the card is never both dropped and not paid back (or the reverse).
    ``None`` for every run without per-line motion.
    """
    planned = _brief_closing_beat(ctx)
    if planned is None or not laid_shots:
        return None
    line_id, _seconds = planned
    if not motion_per_line_requested(ctx.config_snapshot):
        return None
    keyed, _pool = collect_motion_clips_by_line(ctx.prior_output("motion_graphics_gen"))
    card = keyed.get(line_id)
    closing_shot = laid_shots[-1]
    if not card or str(closing_shot.get("line_id")) != line_id:
        return None
    return planned if closing_shot.get("video_url") == card else None


#: Per-line promo caption box opacity: dense enough to read over a bright card,
#: light enough that the card shows through as one piece.
CAPTION_BOX_OPACITY = 0.85


def _hex6(value: object) -> str | None:
    """``#RRGGBB`` as the compiler accepts it, or ``None`` (the compiler default)."""
    if isinstance(value, str) and len(value) == 7 and value.startswith("#"):
        try:
            int(value[1:], 16)
        except ValueError:
            return None
        return value.upper()
    return None


class PromoRenderStage(CompileStageBase[PromoRenderInput, PromoRenderOutput]):
    name = "promo_render"
    input_model = PromoRenderInput
    output_model = PromoRenderOutput

    def build_shots(self, ctx: RunContext) -> list[dict[str, Any]]:
        story = ctx.prior_output("promo_storyboard")
        script = ctx.prior_output("promo_script")
        if not (story and script):
            raise RuntimeError("storyboard + script required")
        narrate = ctx.prior_output("promo_narrate") or {}
        image_gen = ctx.prior_output("ai_image_gen") or {}
        capture = ctx.prior_output("site_capture") or {}
        director = ctx.prior_output("promo_visual_director") or {}
        slides = ctx.prior_output("promo_slides") or {}
        html = ctx.prior_output("promo_html") or {}
        # Real (non-AI) imagery the operator pinned / we extracted / media-discovery found,
        # already re-published to run storage so the compiler can fetch it. Empty
        # when neither opt-in source stage ran → behaviour unchanged.
        real_media_urls = collect_real_media_urls(
            source_ingest=ctx.prior_output("source_ingest"),
            media_discovery=ctx.prior_output("media_discovery"),
        )
        # Curated/discovered VIDEO clips (pinned or discovered), re-published
        # as run-storage clips by source_ingest / media_discovery. Empty when no
        # opt-in source produced a clip → render unchanged.
        real_media_videos = collect_real_media_videos(
            source_ingest=ctx.prior_output("source_ingest"),
            media_discovery=ctx.prior_output("media_discovery"),
        )
        # AI-generated clips (ai_video_gen, opt-in, already re-published to run
        # storage). Lowest-priority motion — fills lowest-value lines that
        # curated media left bare. Empty when ai_video_gen skipped → unchanged.
        ai_video_urls = collect_ai_video_urls(ctx.prior_output("ai_video_gen"))
        # Branded motion-graphics clips (motion_graphics_gen, opt-in, already
        # re-published to run storage). Keyed clips join 1:1 by line_id; unkeyed
        # clips form a round-robin pool. Empty when the stage was skipped.
        motion_clip_by_line, motion_clip_urls = collect_motion_clips_by_line(
            ctx.prior_output("motion_graphics_gen")
        )
        # Per-line motion (opt-in): a line's recorded card IS its shot and
        # outranks footage, html and curated media for that line. ``carded`` is
        # empty for every run without the flag, and everything below that reads
        # it is then a no-op.
        cards_first = motion_per_line_requested(ctx.config_snapshot)
        carded = _carded_lines(story["shots"], motion_clip_by_line) if cards_first else {}

        slide_by_line = {
            str(s["line_id"]): str(s["image_url"])
            for s in slides.get("slides", [])
            if s.get("image_url")
        }
        # Only feed URLs the compiler can actually fetch (http[s]); a non-fetchable
        # asset would crash the whole compile, so it degrades to the slide fallback.
        # Split footage into the primary demo clip (the scripted walkthrough — the
        # render spine) and the cutaway clips (landing / book — single bookends).
        fetchable_clips = [c for c in capture.get("clips", []) if _fetchable(c.get("video_url"))]
        footage_primary = [c["video_url"] for c in fetchable_clips if c.get("is_primary")]
        footage_cutaways = [c["video_url"] for c in fetchable_clips if not c.get("is_primary")]
        html_urls = [
            s["image_url"] for s in html.get("segments", []) if _fetchable(s.get("image_url"))
        ]
        # Per-line (story-relevant) images key by line_id; legacy/static images
        # without a line_id fall into the round-robin pool.
        ai_image_by_line = {
            str(img["line_id"]): str(img["image_url"])
            for img in image_gen.get("images", [])
            if img.get("line_id") and _fetchable(img.get("image_url"))
        }
        ai_urls = [
            img["image_url"]
            for img in image_gen.get("images", [])
            if not img.get("line_id") and _fetchable(img.get("image_url"))
        ]
        # Frame-synced demo: join the visual-director line→screen map with the
        # per-screen assets from site_capture. Each screen has a steady still
        # and (in clips mode) a short interaction clip showing the click that
        # reveals it. The FIRST footage line to show a screen gets that clip —
        # the reveal, with motion — and later lines holding the same screen get
        # the still. Clip falls back to still (stills-only capture), still to the
        # universal slide fallback.
        still_by_screen = {
            str(s["id"]): str(s["image_url"])
            for s in capture.get("screens", [])
            if s.get("id") and _fetchable(s.get("image_url"))
        }
        clip_by_screen = {
            str(s["id"]): str(s["clip_url"])
            for s in capture.get("screens", [])
            if s.get("id") and _fetchable(s.get("clip_url"))
        }
        screen_by_line = {
            str(a["line_id"]): str(a["screen_id"])
            for a in director.get("assignments", [])
            if a.get("line_id") and str(a.get("screen_id")) in still_by_screen
        }
        footage_clip_by_line: dict[str, str] = {}
        footage_screenshot_by_line: dict[str, str] = {}
        # Repeat-still lines re-show a screen already revealed earlier — the
        # redundant slots curated real media may take (a screen's FIRST still,
        # when it has no clip, is genuine demo content and stays).
        footage_repeat_still_lines: set[str] = set()
        shown: set[str] = set()
        for shot in story["shots"]:
            if str(shot.get("visual_type") or "slide") != "footage":
                continue
            lid = str(shot["line_id"])
            if lid in carded:
                # Its card is the shot: the screen's reveal goes to the first
                # line that actually shows footage.
                continue
            screen = screen_by_line.get(lid)
            if not screen:
                continue
            first_occurrence = screen not in shown
            shown.add(screen)
            clip = clip_by_screen.get(screen)
            if first_occurrence and clip:
                footage_clip_by_line[lid] = clip
            else:
                footage_screenshot_by_line[lid] = still_by_screen[screen]
                if not first_occurrence:
                    footage_repeat_still_lines.add(lid)

        image_by_line, video_by_line, fit_by_line = _plan_visuals(
            story["shots"],
            slide_by_line=slide_by_line,
            footage_primary=footage_primary,
            footage_cutaways=footage_cutaways,
            html_urls=html_urls,
            ai_urls=ai_urls,
            ai_image_by_line=ai_image_by_line,
            footage_screenshot_by_line=footage_screenshot_by_line,
            footage_clip_by_line=footage_clip_by_line,
            footage_repeat_still_lines=footage_repeat_still_lines,
            real_media_urls=real_media_urls,
            real_media_video_urls=real_media_videos,
            ai_video_urls=ai_video_urls,
            motion_clip_by_line=motion_clip_by_line,
            motion_clip_urls=motion_clip_urls,
            cards_first=cards_first,
        )
        # Animated slides (promo_slides' motion variant) are deliberately NOT laid
        # over a promo's slide lines. Their templates build the card in from an
        # empty frame (headline 350-1100ms, kicker/rule after it) while the line's
        # narration starts AT the cut, and the clip is rendered at the template's
        # own length, so the compiler cannot seek past the entrance either (it only
        # seeks a clip LONGER than its shot). Run 01a10fcf (2026-10-06): line 6's
        # card was empty for 0.8s under its narration and the CTA's text was up
        # for ~1.4s. The still PNG carries the same text from frame 0 and gets its
        # motion from Ken Burns instead.
        captions_by_line = (
            caption_text_by_line({"promo_script": script})
            if captions_mode(ctx) == "burned_in"
            else None
        )
        # Sound design (opt-in): per-line accent cues from the sound-director plus
        # a transition whoosh on every cut. Catalog ids here; resolved to presigned
        # URLs in the compile tail (resolve_shot_sound_urls). Empty → shots carry
        # no sound, render unchanged.
        sound_cfg = sound_config(ctx)
        # ORCH-3.2 directed footage: bind the VP-2 focus/cursor sidecar to each
        # footage line via the same screen→line map used above. Gated on
        # `direction.footage`; absent ⇒ empty maps ⇒ no geometry rides any shot.
        footage = (ctx.config_snapshot.get("direction") or {}).get("footage")
        # A carded line shows no footage, so no footage geometry, plan, callout
        # or split may ride its shot — and its card is never another line's split
        # source.
        footage_screen_by_line = {
            lid: screen for lid, screen in screen_by_line.items() if lid not in carded
        }
        focus_track_by_line, cursor_track_by_line, footage_viewport_by_line = (
            footage_tracks_by_line(
                capture.get("screens", []), footage_screen_by_line, footage=footage
            )
        )
        # ORCH-3.3 directed shot-plan (promo_footage_direction): a per-line framing
        # plan turns a zoom OFF (co-removing its focus+cursor tracks — a cursor easing
        # to a no-zoom target is the failure), and adds a callout whose coordinates come
        # ONLY from the captured focus bbox (never the LLM) and a split whose second
        # source is resolved compile-side via `video_by_line`. Absent (stage skipped ⇒
        # `{}`) ⇒ empty `plan_by_line` ⇒ helper returns the input maps untouched and
        # both derived maps `None` ⇒ byte-identical.
        plan_by_line = {
            str(p["line_id"]): p
            for p in (ctx.prior_output("promo_footage_direction") or {}).get("plans", [])
            if str(p["line_id"]) not in carded
        }
        focus_track_by_line, cursor_track_by_line, callouts_by_line, split_by_line = (
            resolve_footage_plans(
                line_ids=[str(shot["line_id"]) for shot in story["shots"]],
                plan_by_line=plan_by_line,
                focus_track_by_line=focus_track_by_line,
                cursor_track_by_line=cursor_track_by_line,
                video_by_line={lid: url for lid, url in video_by_line.items() if lid not in carded},
            )
        )
        shots = assemble_shots(
            storyboard_shots=_without_footage_framing(story["shots"], carded),
            # Fully keyed per the visual plan — no round-robin pools.
            images=[],
            narration_clips=narrate.get("clips", []),
            captions_by_line=captions_by_line,
            video_clips=[],
            image_by_line=image_by_line,
            video_by_line=video_by_line,
            fit_by_line=fit_by_line,
            sfx_by_line=sfx_by_line(ctx, sound_cfg, "promo_sound_director"),
            transition_sfx_id=transition_sfx_id(sound_cfg),
            transition_sfx_gain_db=transition_gain_db(sound_cfg),
            focus_track_by_line=focus_track_by_line,
            cursor_track_by_line=cursor_track_by_line,
            footage_viewport_by_line=footage_viewport_by_line,
            callouts_by_line=callouts_by_line,
            split_by_line=split_by_line,
        )
        shots = apply_holds(shots, promo_fill(ctx).hold_seconds, last_min_seconds=CTA_MIN_SECONDS)
        closing = _absorbed_end_card(ctx, shots)
        if closing is not None:
            # The closing shot carries its card, so it IS the end card: that one
            # recording runs on for the seconds the fill plan gave the card (no
            # separate card, no cut).
            shots = extend_shot(shots, *closing)
        return shots

    def ken_burns_local_fallback(self, ctx: RunContext) -> bool:
        """A promo never ships a frozen still when the recorder fails one."""
        return True

    def caption_look(self, ctx: RunContext) -> dict[str, object] | None:
        """Per-line promos: captions in the cards' type and palette.

        Frame QA of run 01a11295 rejected DejaVu Sans on a black plate under
        Inter cards. From the SAME brand resolution as the cards: Inter when the
        cards set Inter (the only other family the compiler embeds), the cards'
        text colour, and ONE box per cue in the cards' background colour.
        Without the per-line brief: ``None``, byte-identical to before.
        """
        if not motion_per_line_requested(ctx.config_snapshot):
            return None
        brand = run_motion_brand_params(ctx.config_snapshot)
        first_family = brand.get("font_sans", "").split(",")[0].strip().strip("'\"")
        return {
            "caption_font_family": "Inter" if first_family == "Inter" else None,
            "caption_text_colour": _hex6(brand.get("text")),
            "caption_box_colour": _hex6(brand.get("bg")),
            "caption_box_opacity": CAPTION_BOX_OPACITY,
        }

    def captions_karaoke(self, ctx: RunContext) -> bool:
        """One continuous caption box: no per-word karaoke runs on a promo.

        video_compiler draws its caption box (ASS ``BorderStyle=3``, a
        semi-transparent colour) once per karaoke run, i.e. per WORD, and the
        neighbouring boxes overlap in the gaps, where the colour doubles up into
        dark vertical stripes (run 01a10fcf). Without karaoke each line is a
        single run and gets a single box. The cue timing is unchanged.
        """
        return False

    def title_card_settled_lead_seconds(self, ctx: RunContext) -> float:
        """The end card reads from the cut: skip ``animated_title_card``'s build-in.

        When the per-line motion brief ran, the end card is the film-style
        closing lockup and its lead is whatever makes it continue the film
        (``film_end_card``: the closing beat's whole length when it repeats it).
        """
        film = film_end_card(ctx, self.title_card_config(ctx))
        return film.lead_seconds if film is not None else END_CARD_SETTLED_LEAD_SECONDS

    def title_card_config(self, ctx: RunContext) -> object:
        """An explicit ``title_card`` block wins; otherwise the promo's default
        branded end card (brand name + CTA), sized by :func:`promo_fill` — unless
        the closing shot carries the closing beat's card and so IS the end card
        (``None``: no card is spliced, the closing shot carries its seconds — see
        ``build_shots`` / :func:`_absorbed_end_card`)."""
        explicit = ctx.config_snapshot.get("title_card")
        if explicit is not None:
            return explicit
        if _brief_closing_beat(ctx) is not None and self._end_card_absorbed(ctx):
            return None
        card = end_card_config(ctx)
        if card is not None and motion_per_line_requested(ctx.config_snapshot):
            # Per-line film whose closing shot is NOT its card: this separate card
            # is the only place the CTA is shown, and the fill sized it as the
            # remainder of the 3.0 s closing tail (1.0 s after a 2.0 s hold) — too
            # short to read. It keeps the CTA floor instead.
            seconds = float(card["duration_seconds"])
            card = {**card, "duration_seconds": max(seconds, CTA_MIN_SECONDS)}
        return card

    def _end_card_absorbed(self, ctx: RunContext) -> bool:
        """Whether the laid closing shot is the end card.

        Re-plans the shots (pure: prior outputs only, no I/O). Extending the
        closing shot changes its seconds, never which visual it carries, so the
        answer read off ``build_shots`` is the one ``build_shots`` itself used.
        """
        try:
            return _absorbed_end_card(ctx, self.build_shots(ctx)) is not None
        except RuntimeError:
            return False

    def caption_clip_refs(self, ctx: RunContext) -> dict[str, ClipRef]:
        """Per-line narration clips to word-time-caption (ORCH-4.2).

        Mirrors the EXACT text source ``build_shots`` uses for the static
        ``captions_by_line`` map (``promo_script`` via ``caption_text_by_line``)
        joined to the ``promo_narrate`` clips, so the word-timed and static caption
        text can never drift. Inert ({}) unless the deliverable opted into
        ``captions_mode=burned_in``; the kill-switch gate lives one layer up in
        ``CompileStageBase.handle``.
        """
        if captions_mode(ctx) != "burned_in":
            return {}
        text_by_line = (
            caption_text_by_line({"promo_script": ctx.prior_output("promo_script") or {}}) or {}
        )
        narrate = ctx.prior_output("promo_narrate") or {}
        refs: dict[str, ClipRef] = {}
        for clip in narrate.get("clips", []):
            line_id = str(clip["line_id"])
            text = text_by_line.get(line_id)
            if not text:
                continue
            refs[line_id] = ClipRef(
                line_id=line_id,
                text=text,
                narrate_stage="promo_narrate",
                clip_filename=f"{clip['id']}.wav",
                seconds=float(clip.get("seconds") or 0.0),
            )
        return refs

    async def build_audio_bed(
        self,
        ctx: RunContext,
        shots: list[dict[str, Any]],
        *,
        degradations: list[Degradation] | None = None,
    ) -> dict[str, Any] | None:
        """Continuous music + ambient bed for the promo, from the ``sound_effects``
        config (shared resolver). Sizes the ambient cue to the FINAL encoded
        duration (``shots`` is post title-card + ken-burns). ``None`` when sound
        design is off / no bed resolves. The ``degradations`` sink is forwarded so
        a chosen-but-unresolvable bed is recorded rather than silently dropped."""
        return await resolve_audio_bed(ctx, shots, sound_config(ctx), degradations=degradations)

    def compile_tags(self, ctx: RunContext) -> dict[str, str]:
        deliverable = ctx.config_snapshot.get("deliverable") or {}
        return {
            "run_id": str(ctx.run_id),
            "stage": self.name,
            "captions_mode": captions_mode(ctx),
            "format": str(deliverable.get("format", "16x9")),
            **resolution_tags(ctx),
        }

    def build_output(self, result: dict[str, Any], ctx: RunContext) -> PromoRenderOutput:
        delivered = float(result["duration_seconds"])
        # Ordered length resolves exactly as `promo/script.py:267-272` does
        # (`brief.target_duration_seconds` → `deliverable.target_duration_seconds`
        # → 30s) — the same number the script was written against, so the
        # comparison is like-for-like. A breach lands on BOTH the structured
        # field (the numbers) and the `degradations` channel (the verdict that
        # makes the run report `degraded`).
        plan = promo_fill(ctx)
        conformance = observe_duration_conformance(
            self.name,
            promo_ordered_seconds(ctx.config_snapshot),
            delivered,
            playable_seconds=playable_duration_seconds(result),
            # The bounded fill (end card + pauses) could not reach the order: the
            # reason names the numbers and the lever (order less / more script).
            plan_infeasible=plan.infeasible_reason,
        )
        return PromoRenderOutput(
            artifact_url=str(result["artifact_url"]),
            duration_seconds=delivered,
            captions_burned=captions_mode(ctx) == "burned_in",
            duration_conformance=conformance,
            degradations=[
                *duration_conformance_degradations(conformance),
                # A SECOND, independent fact about the same master. The verdict
                # above grades the PLAYABLE length (max of the two streams)
                # against the order; it cannot express a picture that stopped
                # early, because on such a master that max IS the audio. The
                # picture-vs-audio gap therefore gets its own code — one number
                # carrying both facts is the regression this replaces.
                *picture_truncation_degradations(result.get("mastering")),
                # The fill planned the on-screen CTA out to stay inside the band:
                # never a silent trim (see promo/fill.py `_fit_default_card`).
                *(
                    [degradation(END_CARD_DROPPED, plan.end_card_dropped)]
                    if plan.end_card_dropped and default_end_card(ctx)
                    else []
                ),
            ],
            # The unmeasured fact rides the NON-degrading tier: a length nobody
            # probed is an unverified verdict, not a bad artifact.
            advisories=[
                *duration_conformance_advisories(conformance),
                *self._unused_motion_clip_advisories(ctx),
                *self._caption_advisories(ctx),
            ],
            **output_dimension_fields(result),
        )

    def _caption_advisories(self, ctx: RunContext) -> list[Degradation]:
        """Record that this film's captions carry no word highlight, and why."""
        karaoke = resolve_captions_karaoke(
            ctx.config_snapshot, get_settings().pipeline.captions_karaoke
        )
        if captions_mode(ctx) != "burned_in" or not karaoke:
            return []
        return [
            advisory(
                CAPTION_KARAOKE_OFF,
                "captions burned in without per-word karaoke highlight: promo sends one run "
                "per line so video_compiler draws ONE caption box (it boxes each karaoke "
                "run separately, which stripes the box)",
            )
        ]

    def _unused_motion_clip_advisories(self, ctx: RunContext) -> list[Degradation]:
        """One advisory per KEYED motion clip that no shot of the film carries.

        Re-plans the shots (pure: prior outputs only, no I/O) and checks each
        keyed clip URL against what the plan laid — so the record follows the
        actual placement rules instead of a second copy of them.
        """
        keyed, _pool = collect_motion_clips_by_line(ctx.prior_output("motion_graphics_gen"))
        if not keyed:
            return []
        try:
            laid = {str(shot.get("video_url") or "") for shot in self.build_shots(ctx)}
        except RuntimeError:
            return []
        # Per-line motion: a card outranks every other visual of its line, so an
        # unplaced one has no shot at all (the storyboard has none for its line).
        per_line = motion_per_line_requested(ctx.config_snapshot)

        def _why(line_id: str) -> str:
            if per_line:
                return "no shot of the film carries it (the storyboard has no shot for that line)"
            return (
                f"line {line_id} shipped another visual (footage or curated media outranks motion)"
            )

        return [
            advisory(
                MOTION_CLIP_UNUSED,
                f"motion clip keyed to line {line_id} was rendered but {_why(line_id)}",
            )
            for line_id, url in keyed.items()
            if url not in laid
        ]
