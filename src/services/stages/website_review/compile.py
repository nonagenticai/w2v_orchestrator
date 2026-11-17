"""website_review_compile — join script + narration + visuals → MP4.

Builds one storyboard shot per review-script line (the line text is the
text-card fallback), joins narration audio by ``line_id``, and gives each line a
DISCRETE, settled visual — never a tall full-page pan:

* BEFORE act (the critique) — when ``site_capture`` recorded the live site in
  per-screen mode, each line shows one of those discrete screens. The first line
  to reveal a screen gets its short interaction *clip* (motion); lines holding it
  get the steady *still*. Screens advance on answer lines (speaker != "a") and
  hold through questions, so the visible screen changes when the answerer points
  at it. With no live screens it falls back to the redesign "before" screenshots.
* AFTER act (the redesign) — when html_render captured discrete per-section stills
  and ``website_review_visual_director`` mapped each after-line to the section it
  discusses, the line shows that SETTLED section (cover-fit). With no section
  assets / no mapping it falls back to the redesign full-page screenshots,
  scroll-fit (see below).

Three keying sources, two fits: site_capture screens and redesign-section stills
are VIEWPORT-ish so they ``fit="cover"`` (crop-to-fill — this replaced the
unwatchable top→bottom pan over a live SPA / a ~5000px redesign page). Only the
full-page redesign/original screenshots used as the after-act FALLBACK are very
tall (e.g. 1440x5100), so they ``fit="scroll"`` (pan top→bottom) — cover-fit on a
~5000px-tall image shows only a ~1000px sliver and reads as a blank/white frame.
The keyed maps go through the shared ``assemble_shots`` helper (keyed beats
round-robin; ``video_url`` beats ``image_url`` beats the text card).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from src.services.captions import caption_text_by_line
from src.services.stages.base import ArtifactUrl, StageInput, StageOutputWithNotes
from src.services.stages.primitives.compile_stage import (
    CompileStageBase,
    PayloadDict,
    ShotDict,
    captions_mode,
    output_dimension_fields,
    playable_duration_seconds,
)
from src.services.stages.shared.clip_edit_fields import EDITABLE_CLIP_FIELDS
from src.services.stages.shared.degradation import Degradation
from src.services.stages.shared.duration_conformance import (
    DurationConformance,
    duration_conformance_advisories,
    duration_conformance_degradations,
    observe_duration_conformance,
    picture_truncation_degradations,
)
from src.services.stages.shared.shotlist import normalize_shotlist
from src.services.stages.shared.sound_design import resolve_audio_bed, sound_config
from src.services.stages.shared.video_gate import require_video
from src.services.stages.shot_assembly import assemble_shots, footage_tracks_by_line
from src.services.stages.website_review.grounding import rendered_section_visuals
from src.services.stages.website_review.order import (
    title_card_plan_to_render,
    website_review_ordered_seconds,
)
from src.services.visual_media import (
    assign_real_media_by_line,
    assign_real_media_videos_by_line,
    collect_motion_slides_by_line,
    collect_real_media_urls,
    collect_real_media_videos,
    overlay_motion_slides,
)
from src.services.word_timed_captions import ClipRef

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext
    from src.services.stages.shared.title_cards import TitleCardPlan


def _fetchable(url: object) -> bool:
    """True for URLs the video-compile service can actually fetch (http[s] or the
    orchestrator artifact-proxy path); anything else would 500 the compile."""
    return isinstance(url, str) and (
        url.startswith(("http://", "https://")) or url.startswith("/api/runs/")
    )


def _bucket(visual: dict[str, Any]) -> str:
    """Classify a redesign visual into the two-act bucket it backs.

    analysis_service tags visuals with free-form roles (``original_desktop``,
    ``redesign_mobile``, ``comparison_tablet``) and the screenshot-fallback path
    uses literal ``before``/``after``. Map both: "original"/"before" → the OLD
    page (``before``); a side-by-side "comparison" → its own bucket (after-only
    fallback); everything else (redesign / after / variant) → the NEW page."""
    tag = f"{visual.get('role', '')} {visual.get('label', '')}".lower()
    if "original" in tag or "before" in tag:
        return "before"
    if "comparison" in tag:
        return "comparison"
    return "after"


def _prefer_desktop(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only desktop captures when any are present — the widest shot reads
    best cover-fit. Falls back to all items when none are tagged desktop."""
    desktop = [v for v in items if "desktop" in f"{v.get('role', '')} {v.get('label', '')}".lower()]
    return desktop or items


def _key_live_screens(before_lines: list[dict[str, Any]], screen_ids: list[str]) -> dict[str, str]:
    """Assign each BEFORE line a live-site screen, deterministically.

    The screen advances on answer lines (speaker != "a") and holds through
    questions — the same "only the answerer switches the screen" rule the promo
    visual-director enforces, but derived structurally (no LLM) since the review
    script is a short before→after arc. Every before line resolves to a screen.
    """
    if not screen_ids:
        return {}
    out: dict[str, str] = {}
    idx = 0
    for i, line in enumerate(before_lines):
        is_answer = str(line.get("speaker") or "a").strip().lower() != "a"
        if i > 0 and is_answer:
            idx = (idx + 1) % len(screen_ids)
        out[str(line["id"])] = screen_ids[idx]
    return out


def _phase(line: dict[str, Any]) -> str:
    return str(line.get("phase") or "before").strip().lower()


def _key_lines(
    lines: list[dict[str, Any]],
    *,
    screen_by_line: dict[str, str],
    before_stills: dict[str, str],
    before_clips: dict[str, str],
    slide_by_line: dict[str, str],
    section_by_line: dict[str, str],
    section_by_id: dict[str, str],
    before_pool: list[str],
    after_pool: list[str],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Key every line to a visual + fit, in priority order.

    1. BEFORE live-site screen (site_capture) — reveal CLIP on first sight, else
       STILL; viewport-sized → ``cover``.
    2. BEFORE authored SLIDE (website_review_slides, shotlist slide shots) —
       viewport-sized → ``cover``.
    3. AFTER redesign SECTION still (visual director) — settled, viewport-ish →
       ``cover``.
    4. Full-page redesign/original screenshot fallback pools → ``scroll`` (a
       ~5000px page cover-cropped to a sliver reads blank).
    Returns (image_by_line, video_by_line, fit_by_line).
    """
    image_by_line: dict[str, str] = {}
    video_by_line: dict[str, str] = {}
    fit_by_line: dict[str, str] = {}
    shown: set[str] = set()
    before_i = after_i = 0
    for line in lines:
        lid = str(line["id"])
        screen = screen_by_line.get(lid)
        if screen and screen in before_stills:
            clip = before_clips.get(screen)
            if screen not in shown and clip:
                video_by_line[lid] = clip
            else:
                image_by_line[lid] = before_stills[screen]
            shown.add(screen)
            fit_by_line[lid] = "cover"  # viewport-sized screen
            continue
        slide = slide_by_line.get(lid)
        if slide is not None:
            image_by_line[lid] = slide
            fit_by_line[lid] = "cover"  # viewport-sized authored slide
            continue
        sec = section_by_line.get(lid)
        if _phase(line) == "after" and sec is not None and sec in section_by_id:
            image_by_line[lid] = section_by_id[sec]
            fit_by_line[lid] = "cover"  # settled redesign section
            continue
        # Full-page redesign/original screenshots → scroll (never cover).
        if _phase(line) == "after" and after_pool:
            image_by_line[lid] = after_pool[after_i % len(after_pool)]
            after_i += 1
            fit_by_line[lid] = "scroll"
        elif before_pool:
            image_by_line[lid] = before_pool[before_i % len(before_pool)]
            before_i += 1
            fit_by_line[lid] = "scroll"
        elif after_pool:
            image_by_line[lid] = after_pool[after_i % len(after_pool)]
            after_i += 1
            fit_by_line[lid] = "scroll"
    return image_by_line, video_by_line, fit_by_line


class WebsiteReviewCompileInput(StageInput):
    pass


class WebsiteReviewCompileOutput(StageOutputWithNotes):
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
    # Actual encoded output resolution (ffprobed by video_compiler). None on
    # older compiler builds / probe miss. Lets the e2e check assert the tier.
    width: int | None = None
    height: int | None = None
    # Delivered-vs-ORDERED duration, recorded so a deviation is visible in the
    # run record: website_review masters measured 273% of their ordered length
    # on 2026-09-10 — the worst of the four blueprints — and nothing anywhere
    # noticed, because nothing compared the two. Observation ONLY: it never
    # fails or degrades the stage, and it is deliberately NOT a `notes` entry
    # (a note flips the whole run to `degraded`). `None` means no usable
    # duration was ordered, which is NOT the same fact as "on target".
    duration_conformance: DurationConformance | None = None


class WebsiteReviewCompileStage(
    CompileStageBase[WebsiteReviewCompileInput, WebsiteReviewCompileOutput]
):
    name = "website_review_compile"
    input_model = WebsiteReviewCompileInput
    output_model = WebsiteReviewCompileOutput

    async def handle(
        self, payload: WebsiteReviewCompileInput, ctx: RunContext
    ) -> WebsiteReviewCompileOutput:
        # critique-only runs (generate_video=false) skip the compile; the record
        # it muxes was skipped too, so there are no narration clips to render.
        require_video(ctx)
        return await super().handle(payload, ctx)

    def build_shots(self, ctx: RunContext) -> list[dict[str, Any]]:
        script = ctx.prior_output("website_review_script")
        record = ctx.prior_output("website_review_record")
        redesign = ctx.prior_output("website_redesign") or {}
        capture = ctx.prior_output("site_capture") or {}
        director = ctx.prior_output("website_review_visual_director") or {}
        if not (script and record):
            raise RuntimeError("website_review_script + website_review_record required")

        lines = script["lines"]
        storyboard_shots = [
            {
                "id": str(line["id"]),
                "line_id": str(line["id"]),
                "seconds": float(line.get("seconds") or 0.0),
                "description": str(line.get("text") or ""),
                # Studio per-clip edits (apply-edits patches them onto the script
                # spine). Rebuilding the spine dict field-by-field DROPS anything not
                # named, which would turn every edit into a green no-op re-stitch —
                # so carry exactly the editable fields the author actually set.
                **{
                    field: line[field]
                    for field in sorted(EDITABLE_CLIP_FIELDS)
                    if line.get(field) is not None
                },
            }
            for line in lines
        ]

        # --- Discrete live-site screens for the BEFORE act (site_capture per-screen)
        before_stills = {
            str(s["id"]): str(s["image_url"])
            for s in capture.get("screens", [])
            if s.get("id") and _fetchable(s.get("image_url"))
        }
        before_clips = {
            str(s["id"]): str(s["clip_url"])
            for s in capture.get("screens", [])
            if s.get("id") and _fetchable(s.get("clip_url"))
        }
        live_screen_ids = [
            str(s["id"]) for s in capture.get("screens", []) if str(s["id"]) in before_stills
        ]
        before_lines = [ln for ln in lines if _phase(ln) == "before"]
        # Authored shotlist (A2): each url shot was captured under its own line id
        # (site_capture set capture_id == line_id), so bind identity — no
        # round-robin guessing. The fully-AI path has no per-line authorship, so it
        # keeps the structural "answerer switches the screen" round-robin.
        brief = ctx.config_snapshot.get("brief") or {}
        shots = normalize_shotlist(brief)
        if shots:
            before_url_ids = {s.line_id for s in shots if s.kind == "url"}
            screen_by_line = {
                str(ln["id"]): str(ln["id"])
                for ln in before_lines
                if str(ln["id"]) in before_url_ids and str(ln["id"]) in before_stills
            }
        else:
            screen_by_line = _key_live_screens(before_lines, live_screen_ids)

        # --- Authored BEFORE-act slides (website_review_slides; shotlist slide shots).
        # Each is keyed to its shot line_id; viewport-sized → cover in _key_lines.
        slides = ctx.prior_output("website_review_slides") or {}
        slide_by_line = {
            str(s["line_id"]): str(s["image_url"])
            for s in slides.get("slides", [])
            if s.get("line_id") and _fetchable(s.get("image_url"))
        }

        # --- Discrete redesign-section stills for the AFTER act (visual director)
        # Section visuals (role redesign_section:<name>) are viewport-ish and
        # settled, keyed per after-line to the section that line discusses. Held
        # OUT of the before/after scroll pools below — they're a separate, better
        # source for the after act, with scroll as the fallback.
        # Keyed by the grounding INVENTORY id (page order, a repeated visual id
        # renamed ``-2``, ``-3``): keying by the raw visual id collapsed two sections
        # sharing an id onto whichever came last, so ``sec-2`` could never reach its
        # own still. The first occurrence keeps the raw id, so a raw-id assignment
        # resolves to the first section of that id.
        section_by_id = {
            uid: str(v["image_url"])
            for uid, v in rendered_section_visuals(redesign)
            if _fetchable(v.get("image_url"))
        }
        section_by_line = {
            str(a["line_id"]): str(a["section_id"])
            for a in director.get("assignments", [])
            if a.get("line_id") and str(a.get("section_id")) in section_by_id
        }

        # --- Redesign screenshots: before/after fallback pools (full-page scroll)
        visuals = [
            v
            for v in redesign.get("visuals", [])
            if _fetchable(v.get("image_url"))
            and not str(v.get("role") or "").startswith("redesign_section:")
        ]
        all_urls = [v["image_url"] for v in visuals]
        before_pool = [
            v["image_url"] for v in _prefer_desktop([v for v in visuals if _bucket(v) == "before"])
        ] or all_urls
        after_items = _prefer_desktop([v for v in visuals if _bucket(v) == "after"])
        comparison_items = _prefer_desktop([v for v in visuals if _bucket(v) == "comparison"])
        after_pool = (
            [v["image_url"] for v in after_items]
            or [v["image_url"] for v in comparison_items]
            or all_urls
        )

        # --- Key each line to a visual + fit (priority: live screen → redesign
        # section → full-page scroll pools). Two visual kinds need two fits:
        # site_capture screens + redesign sections are VIEWPORT-ish → "cover";
        # only the full-page scroll-pool fallback is very tall → "scroll" (cover on
        # a ~5000px page shows a ~1000px sliver and reads blank). See _key_lines.
        image_by_line, video_by_line, fit_by_line = _key_lines(
            lines,
            screen_by_line=screen_by_line,
            before_stills=before_stills,
            before_clips=before_clips,
            slide_by_line=slide_by_line,
            section_by_line=section_by_line,
            section_by_id=section_by_id,
            before_pool=before_pool,
            after_pool=after_pool,
        )

        # Animated slides (motion graphics opt-in): supersede an authored slide's
        # still PNG with its branded motion clip on that slide's line. Only the
        # slide's own line is upgraded; the still stays the degrade path.
        overlay_motion_slides(
            image_by_line,
            video_by_line,
            fit_by_line,
            slide_by_line=slide_by_line,
            motion_slide_by_line=collect_motion_slides_by_line(slides),
        )

        # When the operator grounded the review in real media (pinned / extracted
        # / discovered images & clips, re-published compiler-safe by
        # ``source_ingest`` / ``media_discovery``), overlay it on the
        # LOWEST-VALUE lines: the full-page SCROLL fallback (fit="scroll") — a
        # ~5000px page panned top→bottom reads near-blank, so a curated real
        # photo/clip is strictly better. The settled VIEWPORT visuals
        # (live-site screens, redesign-section stills → fit="cover") and reveal
        # CLIPS are PROTECTED and never displaced. No sources / no scroll lines →
        # no change.
        real_urls = collect_real_media_urls(
            source_ingest=ctx.prior_output("source_ingest"),
            media_discovery=ctx.prior_output("media_discovery"),
        )
        real_videos = collect_real_media_videos(
            source_ingest=ctx.prior_output("source_ingest"),
            media_discovery=ctx.prior_output("media_discovery"),
        )
        protected = {lid for lid, fit in fit_by_line.items() if fit == "cover"} | set(video_by_line)
        img_overlay, img_fit = assign_real_media_by_line(
            storyboard_shots, real_urls, skip_line_ids=protected
        )
        vid_overlay, vid_fit = assign_real_media_videos_by_line(
            storyboard_shots, real_videos, skip_line_ids=protected | set(img_overlay)
        )
        image_by_line.update(img_overlay)  # real media beats the scroll fallback
        video_by_line.update(vid_overlay)  # video_url then beats any leftover image
        fit_by_line.update(img_fit)
        fit_by_line.update(vid_fit)

        # Burned-in captions (opt-in via deliverable captions_mode=burned_in).
        # Keyed by review-line id — the same value used as each shot's line_id —
        # from website_review_script, the same text source the soft-CC publish
        # track uses. None for the default captions_mode=none → unchanged runs
        # stay byte-identical.
        captions_by_line = (
            caption_text_by_line({"website_review_script": script})
            if captions_mode(ctx) == "burned_in"
            else None
        )

        # ORCH-3.2 directed footage: bind the VP-2 focus/cursor sidecar to each live
        # BEFORE-act screen line via the same screen→line map used above. Gated on
        # `direction.footage`; absent ⇒ empty maps ⇒ no geometry rides any shot.
        footage = (ctx.config_snapshot.get("direction") or {}).get("footage")
        focus_track_by_line, cursor_track_by_line, footage_viewport_by_line = (
            footage_tracks_by_line(capture.get("screens", []), screen_by_line, footage=footage)
        )
        return assemble_shots(
            storyboard_shots=storyboard_shots,
            images=[],  # fully keyed per line; no positional round-robin
            narration_clips=record["clips"],
            captions_by_line=captions_by_line,
            image_by_line=image_by_line,
            video_by_line=video_by_line,
            fit_by_line=fit_by_line,
            focus_track_by_line=focus_track_by_line,
            cursor_track_by_line=cursor_track_by_line,
            footage_viewport_by_line=footage_viewport_by_line,
        )

    async def build_audio_bed(
        self,
        ctx: RunContext,
        shots: list[ShotDict],
        *,
        degradations: list[Degradation] | None = None,
    ) -> PayloadDict | None:
        """Continuous music (+ ambient) bed from the run's ``sound_effects`` config.

        The base hook returns ``None``, and this stage never overrode it — so a
        Studio music edit (``apply-edits`` merges ``music`` / gain / fades /
        ``duck_db`` into ``sound_effects``) re-stitched a child run whose
        ``audio_bed`` phase did nothing in 0.0s and shipped the same silent bed.
        Same shared resolver demo_video and promo use; ``None`` when sound design is
        off / no bed resolves, and the ``degradations`` sink records a chosen bed
        that could not be resolved instead of dropping it silently."""
        return await resolve_audio_bed(ctx, shots, sound_config(ctx), degradations=degradations)

    def caption_clip_refs(self, ctx: RunContext) -> dict[str, ClipRef]:
        """Per-line narration clips to word-time-caption (ORCH-4.2).

        Mirrors the EXACT text source ``build_shots`` uses for the static
        ``captions_by_line`` map (``website_review_script`` via
        ``caption_text_by_line``) joined to the ``website_review_record`` clips
        (TTS lives in the record stage), so the word-timed and static caption text
        can never drift. Inert ({}) unless the deliverable opted into
        ``captions_mode=burned_in``; the kill-switch gate lives one layer up in
        ``CompileStageBase.handle``.
        """
        if captions_mode(ctx) != "burned_in":
            return {}
        text_by_line = (
            caption_text_by_line(
                {"website_review_script": ctx.prior_output("website_review_script") or {}}
            )
            or {}
        )
        narrate = ctx.prior_output("website_review_record") or {}
        refs: dict[str, ClipRef] = {}
        for clip in narrate.get("clips", []):
            line_id = str(clip["line_id"])
            text = text_by_line.get(line_id)
            if not text:
                continue
            refs[line_id] = ClipRef(
                line_id=line_id,
                text=text,
                narrate_stage="website_review_record",
                clip_filename=f"{clip['id']}.wav",
                seconds=float(clip.get("seconds") or 0.0),
            )
        return refs

    def title_card_plan(self, ctx: RunContext) -> TitleCardPlan | None:
        """The cards the inherited SCRIPT reserved — never a fresh plan.

        ``order.title_card_plan_to_render`` reads the reservation recorded on the
        ``website_review_script`` output: the seconds that script subtracted from its
        narration budget. A script with no record (produced before the record existed)
        gets the cards as configured, uncapped — what it budgeted for. The 20% cap is
        applied once, at script time; recomputing it here is how a run scripted before
        the cap and compiled after it shipped a 10s order as a ~6s film.
        """
        return title_card_plan_to_render(
            ctx.config_snapshot, ctx.prior_output("website_review_script")
        )

    def build_output(self, result: dict[str, Any], ctx: RunContext) -> WebsiteReviewCompileOutput:
        delivered = float(result["duration_seconds"])
        # Ordered length resolves through `website_review/order.py` — the SAME
        # resolver the script budgets against (deliverable.target_duration_seconds →
        # brief.duration_seconds → 120) — and is graded against the FULL order: the
        # script subtracts the title cards from ITS budget, the film includes them.
        # A breach lands on BOTH the structured field (the numbers) and the
        # `degradations` channel (the verdict that makes the run report `degraded`).
        conformance = observe_duration_conformance(
            self.name,
            website_review_ordered_seconds(ctx.config_snapshot),
            delivered,
            playable_seconds=playable_duration_seconds(result),
        )
        return WebsiteReviewCompileOutput(
            artifact_url=str(result["artifact_url"]),
            duration_seconds=delivered,
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
            ],
            # The unmeasured fact rides the NON-degrading tier: a length nobody
            # probed is an unverified verdict, not a bad artifact.
            advisories=duration_conformance_advisories(conformance),
            **output_dimension_fields(result),
        )
