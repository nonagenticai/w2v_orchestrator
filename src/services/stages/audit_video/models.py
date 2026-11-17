"""Shared typed models for the audit narrated-video sub-chain.

The video-chain element shapes are identical across audit types (an audit deck is
plain narrated slides regardless of which scorecard produced it), so they live
here once instead of being cloned per type. The per-type SCAN output shapes
(``groups`` / ``subsections`` / ``dimensions``) genuinely differ and stay in each
type's ``scan`` module.
"""

from __future__ import annotations

from typing import ClassVar

from src.services.stages.base import ArtifactUrl, StageInput, StageOutput, StageOutputWithNotes


class AuditScriptInput(StageInput):
    pass


class AuditScriptLine(StageOutput):
    id: str
    text: str
    seconds: float


class AuditScriptOutput(StageOutputWithNotes):
    # `lines` is the audit deck's spine: `<type>_slides` subscripts `script["lines"]`
    # to build the deck and narrate fans out over the same key. `min_elements = 1` is
    # meant to guarantee non-empty inside the retry budget, so an empty list means
    # that guarantee was bypassed. `total_seconds` has no reader in src/ — not declared.
    produces: ClassVar[tuple[str, ...]] = ("lines",)

    lines: list[AuditScriptLine]
    total_seconds: float


class AuditSlidesInput(StageInput):
    pass


class AuditSlide(StageOutput):
    id: str
    line_ids: list[str]
    title: str
    bullets: list[str]


class AuditSlidesOutput(StageOutputWithNotes):
    # `<type>_record` builds the deck payload from these and `<type>_compile` joins
    # `slide_by_line` off them; zero slides makes record short-circuit to zero
    # recordings and every line loses its image. Non-empty by construction whenever
    # the script has lines, so empty is drift, not a mode.
    produces: ClassVar[tuple[str, ...]] = ("slides",)

    slides: list[AuditSlide]


class AuditNarrateInput(StageInput):
    pass


class AuditAudioClip(StageOutput):
    id: str
    line_id: str
    artifact_url: ArtifactUrl
    seconds: float
    # Speed the kept clip was synthesized at (None = backend default) and its
    # spoken word count, filled by NarrationStageBase. Optional so outputs
    # persisted before these fields existed still validate.
    speed: float | None = None
    words: int | None = None


class AuditNarrateOutput(StageOutputWithNotes):
    # `<type>_compile` builds every shot from `narrate["clips"]`; empty means
    # `assemble_shots` gets an empty spine and the compile raises "no narrated shots
    # to compile" — a failure attributed to compile rather than to narrate.
    produces: ClassVar[tuple[str, ...]] = ("clips",)

    clips: list[AuditAudioClip]


class AuditRecordInput(StageInput):
    pass


class AuditRecording(StageOutput):
    id: str
    slide_id: str
    image_url: str


class AuditRecordOutput(StageOutput):
    # `<type>_compile` hands each recording's `image_url` to the video-compile
    # service; with zero recordings every narrated line loses its still and the
    # compile muxes audio over nothing, yet the run reads `completed`.
    #
    # `produces_required`, not degrade-only, and this stage is the one place in
    # the deck sub-chain where that is forced: `AuditRecordOutput` carries no
    # `notes` field at all, so a degrade-only contract here would have NO channel
    # to report through — it would be a gate that cannot fire (exactly what
    # `test_a_declaring_stage_can_actually_degrade` in the ratchet forbids).
    #
    # There is no legitimate empty. The "no video was asked for" case is handled
    # BEFORE any of this by `require_audit_video`, which raises StageSkippedError
    # and marks the stage `skipped`. So if this stage actually RUNS, the deck is
    # the deliverable and an empty one is never shippable.
    produces: ClassVar[tuple[str, ...]] = ("recordings",)
    produces_required: ClassVar[bool] = True

    recordings: list[AuditRecording]


class AuditCompileInput(StageInput):
    pass


class AuditCompileOutput(StageOutputWithNotes):
    # Declared output contract — enforced centrally in
    # `runs/dag_executor.py::_enforce_produces`. `produces_required` for the same
    # reason as every sibling compile stage: this is the audit blueprints'
    # `terminal_artifact_stage`, so `run_finalizer` probes THIS `artifact_url` for
    # the deliverable and an absent one silently disables that probe for the run.
    produces: ClassVar[tuple[str, ...]] = ("artifact_url",)
    produces_required: ClassVar[bool] = True

    artifact_url: ArtifactUrl
    duration_seconds: float
