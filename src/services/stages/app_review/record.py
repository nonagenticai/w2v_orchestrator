"""app_review_record — narration (TTS) + per-line visuals tied to fact ids.

Per-line TTS is bounded by ``stage_gather`` so a 30-line review runs up to
4 calls in parallel instead of the prior sequential pattern that turned a
~1m script into a ~5m wait.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar
from uuid import uuid7

from w2v_common.stages import stage_gather

from src.clients.tts_client import TTSClient
from src.services.stages.base import (
    SlotUse,
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
)
from src.services.stages.primitives.narration import synthesize_and_store

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

_PER_ITEM_TIMEOUT_SECONDS = 600.0
_CONCURRENCY = 4


class AppReviewRecordInput(StageInput):
    pass


class NarrationClip(StageOutput):
    id: str
    line_id: str
    artifact_url: str
    seconds: float


class VisualClip(StageOutput):
    id: str
    line_id: str
    artifact_url: str
    seconds: float
    kind: str  # "screenshot" | "fact_card"


class AppReviewRecordOutput(StageOutputWithNotes):
    # `StageOutputWithNotes`, not a plain `StageOutput`: this stage has a real
    # degrade mode (`app_inspect` completing with no screenshot) and previously had
    # NO channel to report it on. `visuals` was still non-empty — every clip just
    # carried `kind="screenshot"` with `artifact_url=""` — so a `produces` contract
    # would have passed too, `app_review_compile` silently filtered them all out
    # (`compile.py:48-51`), and the run shipped an all-text-card video as clean
    # green. Non-empty `notes` renders the run `degraded`
    # (`api/routes/runs/_shared.py`).
    narrations: list[NarrationClip]
    visuals: list[VisualClip]


class AppReviewRecordStage(StageHandler[AppReviewRecordInput, AppReviewRecordOutput]):
    name = "app_review_record"
    input_model = AppReviewRecordInput
    output_model = AppReviewRecordOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("tts"),)

    def __init__(self, tts: TTSClient | None = None) -> None:
        self._tts = tts or TTSClient()

    async def handle(
        self,
        payload: AppReviewRecordInput,
        ctx: RunContext,
    ) -> AppReviewRecordOutput:
        script = ctx.prior_output("app_review_script")
        inspect = ctx.prior_output("app_inspect")
        if not (script and inspect):
            raise RuntimeError("app_review_script + app_inspect required")
        voice = ctx.config_snapshot.get("voice") or {}
        facts_by_id = {f["id"]: f for f in inspect["facts"]}
        screenshot_url = inspect.get("screenshot_url") or ""
        run_id = str(ctx.run_id)

        async def _narrate_line(line: dict) -> tuple[NarrationClip, VisualClip]:
            clip_id, narration_url, seconds = await synthesize_and_store(
                self._tts,
                text=str(line["text"]),
                # Kokoro voice id, NOT the free-text descriptor (which is a tone
                # hint for the LLM); TTS 500s on an unknown voice. None → default.
                voice=voice.get("voice_id"),
                run_id=run_id,
                stage_name="app_review_record",
            )
            narration = NarrationClip(
                id=clip_id,
                line_id=str(line["id"]),
                artifact_url=narration_url,
                seconds=seconds,
            )
            kind = "screenshot"
            artifact_url = str(screenshot_url)
            fid = line.get("fact_id")
            if fid and fid in facts_by_id:
                fact = facts_by_id[fid]
                if fact.get("kind") == "screenshot":
                    artifact_url = str(fact["text"])
                else:
                    kind = "fact_card"
            if kind == "screenshot" and not artifact_url:
                # Tell the truth about what this shot will actually be. A
                # `kind="screenshot"` clip with an empty url is filtered out by
                # `app_review_compile` and rendered as the compositor's text card
                # anyway — labelling it `screenshot` only disguised the emptiness as
                # a populated visuals list. The clip is KEPT (dropping it would
                # remove the line from the spine and lose its narration timing);
                # only its honesty changes.
                kind = "fact_card"
            visual = VisualClip(
                id=str(uuid7()),
                line_id=str(line["id"]),
                artifact_url=artifact_url,
                seconds=float(line["seconds"]),
                kind=kind,
            )
            return narration, visual

        pairs = await stage_gather(
            script["lines"],
            _narrate_line,
            concurrency=_CONCURRENCY,
            per_item_timeout_seconds=_PER_ITEM_TIMEOUT_SECONDS,
            per_item_retries=1,
            stage_name="app_review_record",
        )
        narrations = [p[0] for p in pairs]
        visuals = [p[1] for p in pairs]
        notes: list[str] = []
        if not screenshot_url:
            # Deliberately a NOTE, not a hard fail. `app_inspect` has no `produces`
            # contract and fails ~24% of the time on missing input, so a screenshot
            # is genuinely optional; failing here would turn a degraded-but-watchable
            # review into an outright run failure. A note marks the run `degraded`,
            # which is the signal that was missing entirely.
            fallback_lines = sum(1 for v in visuals if v.kind == "fact_card")
            notes.append(
                "app_review_record: app_inspect returned no screenshot_url; "
                f"{fallback_lines} of {len(visuals)} shots fall back to a text card"
            )
        return AppReviewRecordOutput(narrations=narrations, visuals=visuals, notes=notes)
