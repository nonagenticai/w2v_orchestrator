"""music_select — pick the bed the reel is cut to, and fetch its beat grid.

Two jobs, in this order, because the second is what makes the first matter:

1. **Pick a track.** The LLM chooses ONE id from the ``music`` category of the
   tenant-resolved sound library (``sound_catalog.load_sound_library`` — the same
   loader ``/api/music/tracks`` renders the FE picker from). Modelled on
   ``promo_sound_director``: the model is advisory, the library is authoritative,
   and an id that is not in the library is refused rather than passed on. An
   author who already pinned ``brief.music.track_id`` skips the LLM entirely —
   an explicit human choice must never be "reconsidered" by a model.
2. **Fetch its beat grid.** ``video_analyzer``'s synchronous
   ``POST /api/analyze/audio`` returns bpm + beats + downbeats + onsets for the
   track. THIS is the payload the whole product hangs on: without a grid there
   is nothing to cut to and ``cut_plan`` degrades to even spacing.

**Why it depends on ``source_ingest`` and not on the footage chain.** Choosing a
bed needs the brief's mood/genre, not the footage — so hanging it off
``source_ingest`` (a no-dep leaf) lets the track pick and the analyzer round-trip
run CONCURRENTLY with the long footage analysis instead of behind it, and
``cut_plan`` joins the two branches. The edge exists at all (rather than making
this a second leaf) so any grounding text the operator supplied is available to
the mood prompt — and it IS read, through ``grounding_prompt_block`` bounded by
``_MUSIC_GROUNDING_MAX_CHARS``. It was not, for as long as this stage existed:
the edge paid for a scrape/parse fan-out whose output nothing in this module ever
touched, which is a dependency that only costs.

**This stage pauses**: music is the single most subjective choice in the product
and the cheapest to change. Approving the bed before the cut is planned costs a
click; discovering the wrong bed after the compile costs the render.

Skip-clean is deliberately NOT offered here. A reel cut to a beat has no
meaningful "no music" mode — that is a different product — so an empty library
is a precondition failure, not a skip.

There is ONE way to get a bedless reel, and it is explicit: a Studio music edit
that turns the bed OFF (``config_snapshot["music"]["enabled"] is False`` — see
``music_override``). That is honoured rather than ignored, but never quietly:
the stage returns an empty grid with a NOTE, ``cut_plan`` then falls back to
even spacing with ``beat_grid_available=False`` / ``solver_stubbed=True``, and
the empty ``track_id`` also trips the central ``produces`` gate — so the run
surfaces as ``degraded``, which is exactly what a beat-cut reel with no beat is.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, ClassVar

from w2v_common.stages import stage_call

from src.clients.schemas.video_analyzer import VideoAnalyzerAudioGrid
from src.clients.video_analyzer_client import VideoAnalyzerClient
from src.services.llm import WMClient
from src.services.stages.action_reel.music_override import MusicOverride, read_music_override
from src.services.stages.base import (
    SlotUse,
    StageHandler,
    StageInput,
    StageOutputWithNotes,
    StagePreconditionError,
)
from src.services.stages.generic.grounding import grounding_prompt_block
from src.services.stages.llm_budget import (
    LLM_RETRY_ON,
    llm_budget_seconds,
    llm_client_timeout_seconds,
    llm_retries,
    output_token_budget,
)
from src.services.stages.sound_catalog import SoundLibrary, load_sound_library
from src.services.stages.upstream import guarded
from src.services.tts_storage import presign_sound_asset

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a music supervisor for a short action-sports reel. You receive the "
    "brief (what the reel is about and the mood it wants) and the list of music "
    "beds you may choose from. Choose EXACTLY ONE track_id from that list — never "
    "invent one — and give a one-sentence rationale. Prefer energy and a clear "
    "pulse: the reel is cut to this track's beat."
)

_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "track_id": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["track_id"],
}

# Worst-case size of the ANSWER alone: one track id plus a one-sentence rationale,
# JSON-escaped. 200 is that with generous headroom. It is deliberately small AND
# deliberately not the budget: the served model is a reasoning model whose thinking
# tokens are charged against the same ``max_tokens``, and this prompt joins the WHOLE
# music catalogue, so a bigger library makes the model think LONGER — exactly the case
# where a budget sized to the answer dies at 422 before emitting any JSON.
# ``output_token_budget`` adds the reasoning headroom (see ``stages/llm_budget.py``).
_MUSIC_SELECT_ANSWER_TOKENS = 200
_MUSIC_SELECT_MAX_TOKENS = output_token_budget(answer_tokens=_MUSIC_SELECT_ANSWER_TOKENS)

# How much of ``source_ingest``'s grounding text this prompt may carry.
#
# The stage's own docstring and the blueprint comment beside the ``source_ingest``
# DAG edge both justify that edge by saying the operator's grounding text is
# available to the mood prompt — and it was not read at all, so a paid scrape
# fan-out was computed and discarded on every run.
#
# 1500 chars (~400 tokens) and not the shared ``_GROUNDING_PROMPT_MAX_CHARS``
# (8000): this prompt ALREADY joins the whole music catalogue, and the served
# model charges its thinking against the same ``max_tokens`` the answer comes out
# of — the failure mode of an over-large preamble here is a 422 before any JSON,
# not a worse track. A supervisor needs the operator's subject and mood, which
# live in the first paragraphs; it does not need the corpus.
_MUSIC_GROUNDING_MAX_CHARS = 1500


class MusicSelectInput(StageInput):
    """No typed inputs — brief + library + analyzer are read at handle time."""


class MusicSelectOutput(StageOutputWithNotes):
    """The chosen bed and its beat grid.

    ``beats`` / ``downbeats`` are absolute seconds from the head of the track.
    They may legitimately be EMPTY: the analyzer degrades on un-beat-trackable
    audio rather than failing, and ``grid_source`` says which happened. The cut
    planner must therefore check the grid before trusting it — treating an empty
    list as a hard error would turn "this bed has no pulse" into a failed run.
    """

    # Declared output contract — enforced once, centrally, in
    # `runs/dag_executor.py::_enforce_produces`. Empty here means the stage
    # reported success having produced nothing, which degrades the run.
    # `cut_plan` carries both forward (`music_track_id` / `music_url`) and `compile`
    # re-presigns from `track_id` to lay the bed. `beats` is deliberately NOT declared:
    # an un-beat-trackable bed legitimately yields an empty grid (see the docstring),
    # whereas an unpresignable `track_url` ships a silent reel.
    produces: ClassVar[tuple[str, ...]] = ("track_id", "track_url")

    track_id: str
    track_key: str
    track_url: str
    rationale: str = ""
    duration_s: float = 0.0
    bpm: float = 0.0
    bpm_confidence: float = 0.0
    beats: list[float] = []  # noqa: RUF012 -- pydantic per-instance default
    downbeats: list[float] = []  # noqa: RUF012
    onsets: list[float] = []  # noqa: RUF012
    meter: str = ""
    grid_source: str = ""


#: What a run says when the operator turned the bed OFF. Spelled out at this
#: length on purpose: it is the one sentence standing between "the reviewer asked
#: for no music" and "why is this reel silent and off the beat?".
MUSIC_OFF_NOTE = (
    "MUSIC OFF: a studio music edit turned the bed off, so this reel renders SILENT "
    "and its shots are spaced evenly instead of being cut to a beat grid — the "
    "product's timing claim does not apply to this version"
)


def _music_off_output() -> MusicSelectOutput:
    """The explicit no-bed output: no track, no grid, one loud note.

    Everything downstream already tolerates this exact shape — ``cut_plan`` reads
    an empty ``beats`` list and takes its ``_even_spacing`` branch (stamping
    ``beat_grid_available=False`` / ``solver_stubbed=True``), and ``compile``'s
    ``_build_music_bed`` reads an empty ``music_track_id`` and lays no bed. So
    honouring "off" needs no new degradation path; it needs the truth said out
    loud, which is what the note (and the resulting ``degraded`` run) is for.
    """
    return MusicSelectOutput(
        track_id="",
        track_key="",
        track_url="",
        rationale="music turned off by a studio music edit",
        notes=[MUSIC_OFF_NOTE],
    )


class MusicSelectStage(StageHandler[MusicSelectInput, MusicSelectOutput]):
    name = "music_select"
    input_model = MusicSelectInput
    output_model = MusicSelectOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("llm"),)
    worker_name = "action-reel-music-supervisor"

    def __init__(
        self,
        client: WMClient | None = None,
        analyzer: VideoAnalyzerClient | None = None,
    ) -> None:
        self._llm_timeout = llm_budget_seconds()
        self._llm_retries = llm_retries()
        self._wm = client or WMClient(timeout=llm_client_timeout_seconds())
        self._analyzer = analyzer or VideoAnalyzerClient()

    async def handle(self, payload: MusicSelectInput, ctx: RunContext) -> MusicSelectOutput:
        override = read_music_override(ctx.config_snapshot)
        if override.is_off:
            # BEFORE the library load, deliberately: a run whose operator asked for
            # no music must not fail on an empty music catalogue.
            return _music_off_output()
        library = await load_sound_library()
        track_ids = library.ids_in("music")
        if not track_ids:
            raise StagePreconditionError(
                "music_select: the sound library exposes no music beds; seed "
                "sound_library/shared/manifest.json before running action_reel",
                missing=["sound_library.music"],
            )

        brief = ctx.config_snapshot.get("brief") or {}
        notes: list[str] = []
        # The reason the ``source_ingest`` edge exists (see the module docstring).
        # Read HERE and passed down rather than read inside ``_build_message``, so
        # the prompt builder stays a pure function of what it is given.
        grounding = grounding_prompt_block(ctx, max_chars=_MUSIC_GROUNDING_MAX_CHARS)
        track_id, rationale = await self._resolve_track(
            brief, library, track_ids, notes, grounding=grounding, override=override
        )

        key = library.key_for(track_id) or ""
        # The analyzer has no credentials for the orchestrator's own bucket, so it
        # gets a presigned GET. Minted through the shared `presign_sound_asset`
        # helper (own-bucket presign by catalog id) rather than by hand, so the
        # library-key resolution stays in one place.
        #
        # This URL is for the ANALYZER, and it is also stored on the output for the
        # reviewer to audition. It is NOT what the render fetches: this presign is
        # minted long before the compile and the request-level music bed is not covered by the
        # compile-time `refresh_shot_asset_urls` pass (that only walks SHOTS), so
        # the compile stage re-presigns from `track_id` at render time.
        url = await presign_sound_asset(track_id) or ""
        grid = await self._beat_grid(url, notes)
        return MusicSelectOutput(
            track_id=track_id,
            track_key=key,
            track_url=url,
            rationale=rationale,
            duration_s=grid.duration_s if grid else 0.0,
            bpm=grid.bpm if grid else 0.0,
            bpm_confidence=grid.bpm_confidence if grid else 0.0,
            beats=list(grid.beats) if grid else [],
            downbeats=list(grid.downbeats) if grid else [],
            onsets=list(grid.onsets) if grid else [],
            meter=grid.meter if grid else "",
            grid_source=grid.grid_source if grid else "",
            notes=notes,
        )

    async def _resolve_track(
        self,
        brief: dict,
        library: SoundLibrary,
        track_ids: list[str],
        notes: list[str],
        *,
        grounding: str = "",
        override: MusicOverride | None = None,
    ) -> tuple[str, str]:
        """The chosen id + rationale. An authored pin wins outright.

        A pinned id is still validated against the library — an id the compiler
        cannot resolve is worse than no pin, because it fails at render time
        rather than here.

        Two pins can be in play, and the per-run one wins: ``brief.music.track_id``
        is what the project was authored with, while the ``music`` override block
        is a Studio music edit made on THIS run — a later, more specific human
        choice. Both are validated the same way, and an invalid one is noted and
        falls through rather than being obeyed into a render-time failure.

        Pins are checked with ``resolves`` (a tenant track deleted mid-run is
        RETIRED, not gone, while the run is active); the model's free choice is
        checked with ``is_valid``, which never admits a retired track.
        """
        from_edit = str((override.track_id if override else None) or "")
        if from_edit:
            if library.resolves(from_edit, category="music"):
                return from_edit, "pinned by a studio music edit"
            notes.append(
                f"the music edit's track {from_edit!r} is not a library music bed; ignoring it"
            )
        pinned = str((brief.get("music") or {}).get("track_id") or "")
        if pinned:
            if library.resolves(pinned, category="music"):
                return pinned, "pinned by the brief"
            notes.append(f"brief.music.track_id {pinned!r} is not a library music bed; ignoring")

        async def _attempt() -> tuple[str, str]:
            resp = await self._wm.message(
                worker_name=self.worker_name,
                message=self._build_message(brief, library, track_ids, grounding=grounding),
                system_prompt=_SYSTEM_PROMPT,
                output_schema=_OUTPUT_SCHEMA,
                max_tokens=_MUSIC_SELECT_MAX_TOKENS,
                # Chain-of-thought OFF. Measured through worker-management on the
                # production path (Job `wmprobe-segsel`, 2026-08-20), same prompt and
                # schema, `max_tokens=8000`: with reasoning ON the call hit the stage's
                # own 950s deadline and returned NOTHING -- twice, on both output
                # schemas. With reasoning OFF the same call answered in 34-67s with
                # real segments, every time. The model spends the whole budget thinking
                # and never emits the answer, which surfaces one layer up as
                # `Exceeded maximum output retries (3)`.
                #
                # This is only sendable because worker-management #188 added the field
                # and `w2v_common` 0.61.0 can carry it -- `MessageRequest` is
                # `extra="forbid"`, so sending it to an older WM would 422 the call.
                # Omit it and w2v's calls take WM's in-memory auto-create branch
                # (`model_provider="vllm"`), which goes DIRECT to vLLM and never
                # touches inference-router -- the only component that would otherwise
                # turn thinking off. See memory `w2v-wm-calls-bypass-inference-router`.
                enable_reasoning=False,
            )
            data = resp.output
            if not isinstance(data, dict):
                raise ValueError("music_select: WM output is not an object")
            chosen = str(data.get("track_id") or "")
            if not library.is_valid(chosen, category="music"):
                # Retryable through stage_call: re-prompt rather than silently
                # substituting a track nobody chose.
                raise ValueError(f"music_select: {chosen!r} is not an allowed music bed")
            return chosen, str(data.get("rationale") or "")

        return await stage_call(
            lambda: guarded(_attempt),
            timeout_seconds=self._llm_timeout,
            retries=self._llm_retries,
            retry_on=LLM_RETRY_ON,
            stage_name=self.name,
        )

    def _build_message(
        self,
        brief: dict,
        library: SoundLibrary,
        track_ids: list[str],
        *,
        grounding: str = "",
    ) -> str:
        """The brief, the operator's grounding text, then the allowed beds.

        ``grounding`` is ``source_ingest``'s scraped corpus, already bounded and
        already carrying its own leading blank line (``grounding_prompt_block``),
        so a run with no ``brief.sources`` — where the helper returns "" — sends
        the byte-identical message this stage has always sent. It sits between the
        brief and the catalogue on purpose: it is context for the mood, and the
        model should have read it before it starts reading track ids.
        """
        catalog = "\n".join(library.describe(tid) for tid in track_ids)
        summary = json.dumps(
            {
                "title": str(brief.get("title") or ""),
                "goal": str(brief.get("goal") or ""),
                "audience": str(brief.get("audience") or ""),
                "tone": str(brief.get("tone") or ""),
            },
            ensure_ascii=False,
        )
        return f"BRIEF:\n{summary}{grounding}\n\nALLOWED MUSIC BEDS:\n{catalog}"

    async def _beat_grid(self, url: str, notes: list[str]) -> VideoAnalyzerAudioGrid | None:
        """The track's beat grid, or ``None`` with a note.

        Soft-fail is deliberate: the grid is the reel's spine, but the analyzer is
        a separate service and losing it should degrade the cut to even spacing
        (``cut_plan`` handles that) rather than throw away an approved selection
        and a chosen track. The note makes the degradation visible at the pause.
        """
        if not url:
            notes.append("music bed could not be presigned; no beat grid fetched")
            return None
        try:
            return await self._analyzer.analyze_audio(audio_url=url)
        except Exception as exc:  # noqa: BLE001 -- degrade to an even-spacing cut
            log.warning("music_select: beat-grid analysis failed: %s", exc)
            notes.append(f"beat-grid analysis failed ({exc}); cut_plan will space evenly")
            return None
