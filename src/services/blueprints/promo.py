"""PromoBlueprint — 5-stage promo DAG: analyze → script → storyboard → render → review."""

from __future__ import annotations

from src.services.blueprints._factories import SOURCE_INGEST_FACTORY
from src.services.blueprints.base import DAGBlueprint, StageSpec, handler
from src.services.blueprints.registry import get_registry
from src.services.stages.generic.ai_image_gen import AiImageGenStage
from src.services.stages.generic.ai_video_gen import AiVideoGenStage
from src.services.stages.generic.media_discovery import MediaDiscoveryStage
from src.services.stages.generic.motion_graphics_gen import MotionGraphicsGenStage
from src.services.stages.generic.site_capture import SiteCaptureStage
from src.services.stages.growth_bundle.handler import GrowthBundleStage
from src.services.stages.primitives.compile_stage import COMPILE_STAGE_OUTER_TIMEOUT_SECONDS
from src.services.stages.promo.analyze import PromoAnalyzeStage
from src.services.stages.promo.footage_critique import PromoFootageCritiqueStage
from src.services.stages.promo.footage_direction import PromoFootageDirectionStage
from src.services.stages.promo.html_segments import PromoHtmlStage
from src.services.stages.promo.image_prompts import PromoImagePromptsStage
from src.services.stages.promo.motion_brief import PromoMotionBriefStage
from src.services.stages.promo.narrate import PromoNarrateStage
from src.services.stages.promo.render import PromoRenderStage
from src.services.stages.promo.review import PromoReviewStage
from src.services.stages.promo.script import PromoScriptStage
from src.services.stages.promo.slides import PromoSlidesStage
from src.services.stages.promo.sound_director import PromoSoundDirectorStage
from src.services.stages.promo.storyboard import PromoStoryboardStage
from src.services.stages.promo.visual_director import PromoVisualDirectorStage

PROMO_BLUEPRINT = DAGBlueprint(
    id="promo_v1",
    project_type="promo",
    # The deliverable is the accept-best footage critique's artifact (which passes
    # promo_render through unchanged when critique isn't requested)…
    terminal_artifact_stage="promo_footage_critique",
    # …but surgical timeline edits must re-stitch at the REAL compile stage, not at
    # the post-compile critique — so the reorder/renarrate cut stays promo_render.
    compile_cut_stage="promo_render",
    # ``delivered_artifact_stage`` is deliberately UNSET here, and that is a
    # DERIVED answer, not an omission. Walking promo's own DAG, exactly two
    # stages declare ``artifact_url`` — ``promo_render`` and, directly downstream
    # of it, ``promo_footage_critique`` — and NOTHING downstream of the critique
    # declares it (``promo_review`` → ``growth_bundle`` publish neither). So the
    # most-downstream artifact-bearing stage IS ``terminal_artifact_stage``
    # already, and the inherit-the-terminal default (``delivered_stage()``)
    # resolves to exactly the right stage. Setting it explicitly would only
    # restate that, and would put a second name in the file to keep in sync.
    #
    # ⚠️ Promo is the blueprint most likely to be "corrected" by mistake, because
    # it is the one where ``terminal_artifact_stage != compile_cut_stage``. That
    # split is about WHERE A RE-STITCH CUTS, not about what ships — do not move
    # either name to chase the other. The demo_video/showcase blueprints are the
    # opposite case (three artifact-bearing pass-throughs follow their compile),
    # which is why only those two set the field.
    stages=[
        StageSpec(
            name="promo_analyze",
            # Also depends on source_ingest so analyze can ground pain points /
            # hooks in the operator-provided sources. source_ingest is opt-in
            # (skips → completed-empty when no brief.sources), so this never
            # blocks the common no-sources promo run.
            depends_on=("source_ingest",),
            handler_factory=handler(PromoAnalyzeStage),
        ),
        StageSpec(
            name="promo_script",
            # Also depends on site_capture (scraped site copy) and source_ingest
            # (multi-source grounding) so the script can be grounded in the real
            # product. Both are opt-in: when not requested they skip (counts as
            # completed-empty), so these deps never block a plain promo run.
            depends_on=("promo_analyze", "site_capture", "source_ingest"),
            handler_factory=handler(PromoScriptStage),
            default_pause_after=True,
        ),
        StageSpec(
            name="promo_storyboard",
            depends_on=("promo_script",),
            handler_factory=handler(PromoStoryboardStage),
            default_timeout_seconds=60,
            default_pause_after=True,
        ),
        StageSpec(
            name="promo_narrate",
            depends_on=("promo_script",),
            handler_factory=handler(PromoNarrateStage),
            # Inner stage_gather worst case: 2 attempts x 600s = 1200s per item.
            default_timeout_seconds=1300,
        ),
        StageSpec(
            name="promo_render",
            depends_on=(
                "promo_storyboard",
                "ai_image_gen",
                "promo_narrate",
                "site_capture",
                "promo_visual_director",
                "promo_slides",
                "promo_html",
                # Real-media sources (opt-in, skip-clean): pinned/extracted images
                # from source_ingest and discovered images from media_discovery
                # are laid over image-suitable lines (see render._plan_visuals).
                "source_ingest",
                "media_discovery",
                # ai_video_gen (opt-in, skip-clean): generative clips fill the
                # lowest-value lines curated media left bare (lowest motion tier).
                "ai_video_gen",
                # motion_graphics_gen (opt-in, skip-clean): deterministic branded
                # motion clips, above AI video but below footage/curated media.
                "motion_graphics_gen",
                # promo_sound_director (opt-in, skip-clean): per-line accent SFX
                # cues. Skips → completed-empty when sound_effects isn't requested,
                # so this dep never blocks a run without sound design.
                "promo_sound_director",
                # promo_footage_direction (opt-in, skip-clean): per-footage-line
                # cursor/zoom/callout INTENT plan. Skips → completed-empty when
                # direction.footage isn't requested, so this dep never blocks a run.
                "promo_footage_direction",
            ),
            handler_factory=handler(PromoRenderStage),
            # Outer Timeout must cover the compile-wait backstop below it — see
            # compile_stage.COMPILE_STAGE_OUTER_TIMEOUT_SECONDS.
            default_timeout_seconds=COMPILE_STAGE_OUTER_TIMEOUT_SECONDS,
        ),
        StageSpec(
            # Post-render SOFT critique loop (L4): scores the rendered promo shot-by-shot
            # (video_analyzer measured numbers → the `footage-critic` WM evaluator), nudges
            # the promo_footage_direction plan for weak shots and re-renders via the canonical
            # PromoRenderStage (≤2 passes, accept-best). NEVER hard-blocks. OPT-IN on
            # `direction.footage.critique`; absent ⇒ passes promo_render's artifact through
            # UNCHANGED (this stage is the terminal_artifact_stage, and promo has no
            # render_manifest reader — so it must always emit a valid artifact_url).
            name="promo_footage_critique",
            depends_on=("promo_render",),
            handler_factory=handler(PromoFootageCritiqueStage),
            # Outer must cover a full re-render (promo_render worst case 2500s) plus the
            # per-pass analyzer + LLM critic; the loop is hard-bounded to ≤2 passes.
            default_timeout_seconds=2500,
        ),
        StageSpec(
            name="promo_review",
            depends_on=("promo_footage_critique",),
            handler_factory=handler(PromoReviewStage),
        ),
        StageSpec(
            name="growth_bundle",
            depends_on=("promo_review",),
            handler_factory=handler(GrowthBundleStage),
        ),
        # promo_image_prompts writes one story-grounded image prompt per
        # ai_image line (keyed by line_id) so the AI accents illustrate the
        # narration instead of generic stock. Soft: degrades to ai_image_gen's
        # static prompts on no-op / LLM failure. ai_image_gen opts in via
        # config `ai_image_gen.prompts_from_stage = "promo_image_prompts"`.
        StageSpec(
            name="promo_image_prompts",
            depends_on=("promo_script", "promo_storyboard"),
            handler_factory=handler(PromoImagePromptsStage),
            # Outer must cover the LLM ladder inner budget (660s) — see
            # PipelineSettings.llm_budget_seconds.
        ),
        StageSpec(
            name="ai_image_gen",
            depends_on=("promo_script", "promo_image_prompts"),
            handler_factory=handler(AiImageGenStage),
            # Inner stage_gather worst case: 2 attempts x 600s = 1200s per item.
            default_timeout_seconds=1300,
        ),
        # promo_slides renders one designed slide per script line via
        # slide_service (one deck → per-slide PNG). It is the visual for
        # `slide`-typed lines AND the universal fallback for every other line,
        # so a missing footage/html/ai asset degrades to a slide, never a blank.
        # promo_visual_director maps each footage narration line to the demo
        # screen it describes (over site_capture's per-screen screenshots), so
        # promo_render can frame-sync the video to the narration. Soft: empty
        # map (no screens / no footage / LLM failure) → render keeps its slide
        # fallback. Needs the script (line text), storyboard (which lines are
        # footage) and site_capture (the captured screen list).
        StageSpec(
            name="promo_visual_director",
            depends_on=("promo_script", "promo_storyboard", "site_capture"),
            handler_factory=handler(PromoVisualDirectorStage),
            # Outer must cover the LLM ladder inner budget (660s) — see
            # PipelineSettings.llm_budget_seconds.
        ),
        StageSpec(
            name="promo_slides",
            depends_on=("promo_script",),
            handler_factory=handler(PromoSlidesStage),
            default_timeout_seconds=900,
        ),
        # promo_footage_direction is OPT-IN — the shared base skips (→ completed-
        # empty) unless `direction.footage` is set in the run config snapshot. An
        # LLM plans per-footage-line cursor/zoom/callout INTENT (coordinates come
        # from the captured focus track compile-side, never the model); promo_render
        # threads the plans into the compositor shots. Needs the storyboard (which
        # lines are footage), script (per-line text — transitively ready via
        # storyboard/visual_director, declared explicitly so the read is a real DAG
        # edge), narrate (per-line durations), site_capture (per-screen focus tracks)
        # and visual_director (the line→screen map for has_click).
        StageSpec(
            name="promo_footage_direction",
            depends_on=(
                "promo_storyboard",
                "promo_script",
                "promo_narrate",
                "site_capture",
                "promo_visual_director",
            ),
            handler_factory=handler(PromoFootageDirectionStage),
            # Outer must cover the LLM ladder inner budget (660s) — see
            # PipelineSettings.llm_budget_seconds.
        ),
        # promo_sound_director is OPT-IN — skips unless `sound_effects.enabled`
        # (with accents on) is set in the run config snapshot. An LLM picks
        # per-line accent SFX cues (validated against the loaded sound library); promo_render
        # threads them into the compositor shots. No external API key needed.
        StageSpec(
            name="promo_sound_director",
            depends_on=("promo_script",),
            handler_factory=handler(PromoSoundDirectorStage),
            # Outer must cover the LLM ladder inner budget (660s) — see
            # PipelineSettings.llm_budget_seconds.
        ),
        # promo_html reuses the redesign service to render a brand-styled HTML
        # treatment of the source page and screenshot it. OPT-IN (skips without
        # brief.source_url) and soft-fail (a redesign error degrades to slides),
        # so it never blocks promo_render. Budget covers one ~590s redesign poll
        # plus the LLM HTML generation + screenshot render.
        StageSpec(
            name="promo_html",
            depends_on=("promo_script",),
            handler_factory=handler(PromoHtmlStage),
            default_timeout_seconds=1800,
        ),
        # ai_video_gen is OPT-IN — the stage handler returns an empty result
        # unless `ai_video_gen` is set in the run config snapshot. Wired as a
        # parallel leaf with no downstream consumer; the promo render stage
        # does NOT composite generative video clips today (that is a separate
        # feature). The wiring exists so the user can drop in an external
        # video-gen API key (Runway / Kling / Veo / Sora / Seedance) on ACG
        # and start producing generative video clips on a per-run basis
        # without a code change.
        StageSpec(
            name="ai_video_gen",
            depends_on=("promo_script",),
            handler_factory=handler(AiVideoGenStage),
            # Inner stage_gather worst case: 2 attempts x 1200s = 2400s per item.
            default_timeout_seconds=2500,
        ),
        # promo_motion_brief is OPT-IN — skips unless `direction.motion == "per_line"`.
        # It designs ONE motion card per script line from the film kit
        # (`motion_templates.PARAM_MODELS`: two_beat / levels / two_up / timeline /
        # checklist / handoff / section_divider, and animated_title_card for the
        # closing beat), every word on it taken from that line, and sizes each clip
        # to its whole shot. In promo_render a line's recorded card IS its shot: it
        # outranks footage, html and curated media for that line, and the closing
        # shot absorbs the end card only when it carries its card. Needs the script
        # (lines), narrate (measured seconds) and the storyboard (the fill plan's
        # per-line hold reads it) — so with it wired, motion_graphics_gen starts
        # after narration rather than beside it.
        StageSpec(
            name="promo_motion_brief",
            depends_on=("promo_script", "promo_storyboard", "promo_narrate"),
            handler_factory=handler(PromoMotionBriefStage),
            # Outer must cover the LLM ladder inner budget (660s) — see
            # PipelineSettings.llm_budget_seconds.
        ),
        # motion_graphics_gen is OPT-IN — skips unless `motion_graphics` is set in
        # the run config snapshot. Renders deterministic, on-brand HTML/CSS motion
        # clips (via video_production /api/record/animation) that promo_render lays
        # in `video_by_line` above generative AI video. No external API key needed.
        StageSpec(
            name="motion_graphics_gen",
            # promo_motion_brief (opt-in, skip-clean) authors one designed card
            # per line; when it ran, its items are this stage's items.
            depends_on=("promo_script", "promo_motion_brief"),
            handler_factory=handler(MotionGraphicsGenStage),
            # Inner stage_gather worst case: 2 attempts x 900s = 1800s per item.
            default_timeout_seconds=1900,
        ),
        # site_capture is OPT-IN — the stage handler skips unless the brief
        # carries a `source_url`. As a no-dep leaf it starts at run open
        # (parallel to promo_analyze) so the slow browser recording overlaps
        # the analyze LLM call. Feeds promo_script (scraped copy) + promo_render
        # (real-site footage as background clips, taking precedence over AI
        # images). Soft-fails per clip, so a recording error degrades the run
        # rather than failing it.
        StageSpec(
            name="site_capture",
            handler_factory=handler(SiteCaptureStage),
            # Per-clip record worst case (wait 600s) x up-to-4 clips at
            # concurrency 2 ≈ 1200s; budget covers it.
            default_timeout_seconds=1300,
        ),
        # Wait for site_capture so ingestion can reuse its rendered, authenticated
        # page text; additional sources still use the normal scraper/parser.
        StageSpec(
            name="source_ingest",
            depends_on=("site_capture",),
            handler_factory=SOURCE_INGEST_FACTORY,
            # Scrape/parse worst case ~180s/source over a small fan-out.
            default_timeout_seconds=900,
        ),
        # media_discovery is OPT-IN (skips unless brief.media_discovery is set):
        # it searches media-discovery, records ranked candidate media (the
        # pipeline-auto surface; the wizard's "Find media" panel is the operator-
        # curated surface) AND re-publishes the top images so promo_render can
        # composite them over image-suitable lines.
        StageSpec(
            name="media_discovery",
            depends_on=("promo_analyze",),
            handler_factory=handler(MediaDiscoveryStage),
            default_timeout_seconds=300,
        ),
    ],
)


get_registry().register(PROMO_BLUEPRINT)
