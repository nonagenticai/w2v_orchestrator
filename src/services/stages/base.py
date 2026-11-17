"""StageHandler ABC — pure async typed (input)→(output) functions.

The error taxonomy, verbatim helpers, and artifact-URL validation were split
into sibling modules (``errors``, ``verbatim``, ``artifacts``) and are
re-exported here so existing ``from src.services.stages.base import ...``
imports keep working unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, cast

from pydantic import BaseModel, Field, ValidationError

from src.models.failure_class import OUR_DEFECT, UNKNOWN, FailureClass

# Re-exported for import-path stability — see module docstring.
from src.services.stages.artifacts import (
    ArtifactUrl,
    _require_artifact_url,  # noqa: F401  # re-exported; imported directly by a contract test
    validate_compile_result,
)
from src.services.stages.errors import (
    StageDownstreamResponseError,
    StageError,
    StagePreconditionError,
    StageSkippedError,
)
from src.services.stages.shared.degradation import Degradation
from src.services.stages.verbatim import (
    assert_verbatim_present,
    find_missing_verbatim,
    items_for_stage,
    normalize_source_script,
    partition_for_stage,
    record_verbatim_drop,
    verbatim_prompt_block,
)

__all__ = [
    "ArtifactUrl",
    "Degradation",
    "StageDownstreamResponseError",
    "StageError",
    "StageHandler",
    "StageInput",
    "StageOutput",
    "StageOutputWithDegradations",
    "StageOutputWithNotes",
    "StagePreconditionError",
    "StageProducedNothingError",
    "StageReportedFailureError",
    "StageSkippedError",
    "assert_verbatim_present",
    "find_missing_verbatim",
    "items_for_stage",
    "missing_produced_fields",
    "normalize_source_script",
    "partition_for_stage",
    "raised_failure_flags",
    "record_verbatim_drop",
    "validate_compile_result",
    "verbatim_prompt_block",
]


class StageInput(BaseModel):
    """Marker base — every stage's typed input subclasses this."""


class StageOutput(BaseModel):
    """Marker base — every stage's typed output subclasses this.

    Stages that expose addressable elements (script lines, storyboard
    shots, …) MUST add `id: str` on those nested elements. PR-6 wires
    annotations against those ids.

    Deliberately field-free: it is used as a mixin on nested element models too
    (e.g. ``DemoVideoLine``), whose field set must stay identical to a non-stage
    twin — so a field added here would leak onto those. The soft-degradation
    ``notes`` channel lives on ``StageOutputWithNotes``, not here.

    ``produces`` is the "did this stage actually do its job?" contract, and it is
    a ClassVar precisely because of the field-free rule above — a ClassVar is not
    a pydantic field, so it cannot leak onto the nested element models.
    """

    # Field names that must be NON-EMPTY for this stage to have produced
    # anything. Empty means None, "", [], {} or () — never 0 or False, which are
    # legitimate values. Declaring this is opt-in per stage; the ratchet in
    # tests/test_stage_produces_ratchet.py is what drives the declared set up.
    produces: ClassVar[tuple[str, ...]] = ()
    # False (default): a violation DEGRADES the run — a note is appended, and
    # api/routes/runs/_shared.py turns non-empty notes into `degraded`.
    # True: a violation FAILS the stage. Only where an empty output is
    # unambiguously a defect: several stages return empty collections BY DESIGN
    # and must keep completing.
    produces_required: ClassVar[bool] = False
    # Boolean field names that mean "this stage did NOT do its job", even though
    # it returned an output object normally. A True one FAILS the stage.
    #
    # This exists because `produces` cannot express the distinction. `produces`
    # asks "is this field empty?", and for a stage with a legitimate empty mode
    # the answer is the same for "empty because nothing was wanted" and "empty
    # because the render broke" — so declaring `produces` on such a stage fails
    # correct runs, and NOT declaring it lets a total outage complete clean. The
    # flag is the failure fact itself, set by the code that swallowed the error,
    # so it is true in exactly one of those two cases.
    #
    # Enforced centrally in `dag_executor._enforce_failure_flags`, beside
    # `_enforce_produces`, so a new stage adopting the pattern needs no executor
    # change. Always hard-fail (no degrade tier): unlike an empty `produces`
    # field, a raised flag is never ambiguous — the stage itself asserted it.
    fails_when_flagged: ClassVar[tuple[str, ...]] = ()


class StageOutputWithDegradations(StageOutput):
    """A stage output that carries the explicit, MACHINE-READABLE ``degradations``
    channel (``shared/degradation.py``).

    Each entry is a stable ``code`` plus a human-readable ``detail``. A
    ``completed`` stage whose persisted output has a non-empty ``degradations``
    list is surfaced as a ``degraded`` run — the same consequence a non-empty
    ``notes`` has, reached without having to write prose.

    Separate from :class:`StageOutputWithNotes` so a stage that has a quality
    signal but NO prose channel (the slide decks, whose outputs are plain
    ``StageOutput`` by design) can raise it. ``StageOutputWithNotes`` inherits
    from this, so every notes-carrying output has both channels.

    Kept OFF the plain ``StageOutput`` marker for the same reason ``notes`` is:
    a field here would leak onto the nested element models (``DemoVideoLine``
    et al.) that must field-match a non-stage twin.
    """

    degradations: list[Degradation] = Field(default_factory=list)
    #: The NON-degrading tier, same entry shape, codes from
    #: ``shared/degradation.py::ADVISORY_CODES``. An entry here NEVER makes a
    #: stage or a run ``degraded``: ``degradation.is_degraded`` and its SQL twin
    #: ``run_repository.degraded_stage_clause`` both read ``degradations`` and
    #: ``notes`` only, and neither may learn to read this field.
    #:
    #: It exists for facts worth QUERYING that are not evidence of a bad
    #: delivery — the first is a pre-audio narration estimate missing its
    #: steering target, which a lesson run raised while delivering within 0.22%
    #: of its order. The real, artifact-measured verdict still lands on
    #: ``degradations`` from the compile stage, so nothing is downgraded here;
    #: what is removed is a SECOND opinion, formed before any audio existed,
    #: flagging films that are in fact conformant.
    advisories: list[Degradation] = Field(default_factory=list)


class MasteringReportFields(BaseModel):
    """The RAW post-mux mastering report the compiler returns on every render.

    Audit/advisory only — none of these three fields can affect a render. They
    are populated centrally in ``primitives/compile_stage.py::handle`` through
    the same ``"<name>" in type(output).model_fields`` guard that already
    carries ``notes`` and ``degradations``, so every render blueprint gets them
    from one place.

    NOT to be confused with the mastering GATE. The gate's prose notes and its
    structured degradations have been shared since ``shared/mastering_gate.py``
    and are NOT duplicated here; the gap this closes is the raw numbers behind
    the verdict (loudness, true peak, colour tags), which only ``demo_video``
    persisted. On ``lesson`` / ``action_reel`` / ``app_review`` /
    ``website_review`` a downstream reader could see "the gate failed" but never
    *by how much*, and ``final_critique`` could not pass-read ``loudness`` at
    all and would have had to re-measure.

    Why this group hangs off :class:`StageOutputWithNotes` and not off a
    compile-only base: the five terminal-compile outputs
    (``DemoVideoCompileOutput``, ``LessonCompileOutput``,
    ``ActionReelCompileOutput``, ``WebsiteReviewCompileOutput``,
    ``AuditCompileOutput``) each subclass ``StageOutputWithNotes`` DIRECTLY —
    there is no intermediate compile base to hang it on, and introducing one
    would mean editing all five per-blueprint models. So their nearest common
    ancestor is the only seam that reaches every render lane while touching zero
    per-blueprint files. The cost is that ~40 non-compile notes-carrying outputs
    inherit three fields that are always ``None`` for them; they are optional
    and default-``None``, so nothing is required to populate them and no
    persisted payload changes shape for a stage that does not.

    The documented hazard for widening this base — leaking fields onto the
    nested ELEMENT models (``DemoVideoLine`` et al.) that must field-match a
    non-stage twin — does not apply: those are plain ``BaseModel`` /
    ``StageOutput`` and never inherit from here. That is exactly why ``notes``
    was kept off the plain ``StageOutput`` marker, and this group respects the
    same line.

    When the per-blueprint compile models are next free to edit, the correct
    refactor is a ``CompileStageOutput(StageOutputWithNotes,
    MasteringReportFields)`` base that the five subclass, and to drop this mixin
    from ``StageOutputWithNotes``. It is split out as its own class so that move
    is a two-line change rather than a field hunt.
    """

    #: The compiler's post-mux mastering report, verbatim. Absent on older
    #: compiler builds ⇒ ``None``.
    mastering: dict[str, object] | None = None
    #: Content hash of the delivered file. Absent on older compiler builds ⇒
    #: ``None``.
    output_sha256: str | None = None
    #: Post-mux MEASURED loudness, read by the advisory ship gate
    #: (``final_critique``) instead of re-measuring. Absent ⇒ ``None``.
    loudness: dict | None = None


class PhaseTimingFields(BaseModel):
    """Per-phase wall-clock seconds for a ``CompileStageBase`` render, plus the
    denominator that makes them interpretable.

    Populated centrally in ``primitives/compile_stage.py::handle`` through the
    same ``"<name>" in type(output).model_fields`` guard that already carries
    ``mastering``/``output_sha256``/``loudness`` (see :class:`MasteringReportFields`)
    — additive only, never overwriting a value a subclass's own
    ``build_output`` already set.

    Exists because ``phase_tracker.phase(...)`` previously only LOGGED elapsed
    time per named phase (``title_cards``, ``presign_1``, ``ken_burns``,
    ``presign_2``, ``sound``, ``audio_bed``, ``captions``, ``stage_call``) — a
    real regression (``lesson_compile`` going 1.59 → 23.6 compile-seconds per
    delivered output-second between June and September) had to be reconstructed
    from job tables a pruner deletes rather than read off the run record. This
    makes it queryable without touching pod logs.

    ``phase_seconds`` keys are whichever phase names actually ran for THIS
    stage (a subclass need not run every phase — e.g. a stage with
    ``title_cards_enabled() == False`` never opens that phase, so it is simply
    absent, not zero). Absent entirely (``None``) only for a CompileStageBase
    caller whose output does not inherit :class:`StageOutputWithNotes` (the
    internal per-clip/per-bed compiles in ``ken_burns.py``/``shot_sound.py`` use
    their own client calls, not this base, so this case does not arise today)
    or for a run that failed before any phase completed.
    """

    phase_seconds: dict[str, float] | None = None
    #: Count of shots eligible for a Ken Burns still→clip job (an ``image_url``
    #: with no ``video_url``) as measured immediately before the ``ken_burns``
    #: phase runs — the denominator ``phase_seconds["ken_burns"]`` needs to read
    #: as a rate rather than a raw total. ``None`` under the same conditions as
    #: ``phase_seconds`` above; ``0`` is a real, distinct value (a footage-only
    #: render that never calls into Ken Burns at all).
    ken_burns_shot_count: int | None = None
    #: Sub-stage checkpoint evidence (``compile_stage._PresubmitAssets``): Ken Burns
    #: clips / title cards served from the content-addressed store instead of being
    #: re-recorded. Non-zero on a compile resumed after a pod roll — the record that
    #: the roll did not re-pay for work the dead pod already stored. ``None`` when
    #: the phase had nothing to do (no stills / no title_card block).
    ken_burns_cache_hits: int | None = None
    title_card_cache_hits: int | None = None


class StageOutputWithNotes(StageOutputWithDegradations, MasteringReportFields, PhaseTimingFields):
    """A stage output that carries the shared soft-degradation ``notes`` channel.

    A ``completed`` stage whose persisted output has non-empty ``notes`` is
    surfaced as a ``degraded`` run (``api/routes/runs/_shared.py``). The
    final-video compile outputs subclass this so the compile-duration auto-QA
    guard can flag a truncated / short-rendered file without failing the run.
    Kept OFF the plain ``StageOutput`` marker so it never leaks onto nested
    element models (``DemoVideoLine`` et al.) that must field-match a non-stage
    twin.

    ``notes`` is PROSE for a human. The structured quality signal lives in
    ``degradations`` (inherited above) — do not encode machine facts here.

    It also carries the RAW post-mux mastering report — see
    :class:`MasteringReportFields` for why that group lives on this base rather
    than on a compile-only one.
    """

    notes: list[str] = Field(default_factory=list)


def _is_empty(value: object) -> bool:
    """Emptiness for the ``produces`` contract.

    ``0`` and ``False`` are REAL values, so only None, empty strings and empty
    containers count. Erring the other way would fail stages that legitimately
    produce a zero.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list | dict | tuple | set):
        return len(value) == 0
    return False


def missing_produced_fields(
    output: Mapping[str, object], produces: Sequence[str]
) -> tuple[str, ...]:
    """Return the declared ``produces`` fields that came back empty.

    Takes the DUMPED mapping rather than the model, so the executor can apply it
    at the one place every stage's success is recorded without re-validating. A
    declared name absent from the dump counts as missing — that is a renamed
    field, which is exactly the drift this exists to catch.
    """
    return tuple(name for name in produces if _is_empty(output.get(name)))


class StageProducedNothingError(StageError):
    """A stage reported success while producing none of what it declared.

    Raised only for ``produces_required`` stages. The default contract degrades
    instead, so adopting ``produces`` on a stage can never turn a working
    pipeline red without someone explicitly asking for it.

    ``failure_class = our_defect``, stamped on the CLASS (the same delivery
    mechanism ``StageReportedFailureError`` and ``MiddlewareTimeoutError`` use)
    because this error is the one structural signal in the estate that matches
    ``failure_class``'s own definition of ``our_defect`` word for word: "a run
    that reported every stage terminal-good and then had no deliverable". It is
    not a guess and it is not prose — no customer input and no upstream service
    can make a stage of ours claim success while producing nothing for a field it
    declared MANDATORY. If a downstream failure is what emptied it, the stage
    swallowed that failure and returned a success-shaped output, and the
    swallowing IS the defect (a stage with a legitimate empty mode is required to
    use ``fails_when_flagged`` + ``StageReportedFailureError``, which carries the
    swallowed cause's OWN attribution instead of this one).

    ⚠️ Raised bare (no ``raise ... from``) at ``dag_executor._enforce_produces``,
    so there is no cause chain for this stamp to outrank. Should a raiser ever
    chain it to a real cause, note that ``attribute_exception``'s first pass
    returns the OUTERMOST non-unknown stamp — this one would then win over the
    cause's, which would be wrong. Chain it and this stamp must move to the
    raise site.
    """

    failure_class: ClassVar[FailureClass] = OUR_DEFECT

    def __init__(self, stage_name: str, empty_fields: Sequence[str]) -> None:
        self.stage_name = stage_name
        self.empty_fields = tuple(empty_fields)
        # The message lands verbatim in Run.error (via the dispatcher's
        # `f"{type(exc).__name__}: {exc}"`) and in the stage row's error, so it is
        # the ONLY thing an operator reading a failed run has. It therefore names
        # the stage, every empty field, and what the operator should conclude —
        # the stage reported success, so the defect is upstream in the stage, not
        # a crash. Without the last clause the natural reading is "the gate is
        # broken", which sends people to read this file instead of the stage.
        joined = ", ".join(self.empty_fields) or "<none declared>"
        super().__init__(
            f"stage {stage_name!r} reported success but produced nothing for its "
            f"required output field(s): {joined}. This stage's `produces` contract "
            f"is mandatory (produces_required=True) because these fields ARE the "
            f"deliverable — an empty one is never shippable, so the run fails here "
            f"rather than completing with an artifact that points at nothing. "
            f"Look at the stage handler for a swallowed downstream error."
        )


def raised_failure_flags(output: Mapping[str, object], flags: Sequence[str]) -> tuple[str, ...]:
    """Return the declared ``fails_when_flagged`` fields that came back True.

    Takes the DUMPED mapping, like :func:`missing_produced_fields`, so the
    executor applies it at the one place every stage's success is recorded.
    Strictly ``is True``: a declared name that is absent, None or non-boolean is
    NOT a raised failure — a renamed field must not start failing every run of
    the stage. The ratchet test is what catches the rename.
    """
    return tuple(name for name in flags if output.get(name) is True)


class StageReportedFailureError(StageError):
    """A stage reported success while flagging its own work as failed.

    The counterpart to :class:`StageProducedNothingError` for stages that have a
    legitimate empty mode. Those stages cannot use ``produces`` — an empty output
    is correct for them — so they set a boolean flag at the point where they
    swallow a downstream error, declare it in ``fails_when_flagged``, and this
    error is what turns that flag into a failed stage instead of a silent
    ``completed``.
    """

    def __init__(
        self,
        stage_name: str,
        flags: Sequence[str],
        notes: Sequence[str] = (),
        *,
        failure_class: FailureClass = UNKNOWN,
    ) -> None:
        self.stage_name = stage_name
        self.flags = tuple(flags)
        self.notes = tuple(str(n) for n in notes)
        # Named EXACTLY ``failure_class`` because that is the attribute
        # ``models.failure_class.attribute_exception`` looks for as it walks the
        # cause chain at the persistence site. This error propagates bare out of
        # ``dag_executor._run_one``, so stamping it here is the whole delivery
        # mechanism — the stage already computed the attribution
        # (``shared/slide_stage.py`` reads it off the swallowed exception) and it
        # died at the consumers, which copied only slides/render_failed/notes.
        # ``UNKNOWN`` default keeps every existing raiser honest: an
        # unattributable failure must stay unattributed (a wrong attribution is
        # worse than an absent one) and ``attribute_exception`` skips ``unknown``
        # stamps rather than letting them veto a definite answer from a cause.
        self.failure_class = failure_class
        # This message is the ONLY thing an operator reading the failed run sees
        # (Run.error + the stage row's error), so it names the stage, the flag
        # that fired, and — critically — the swallowed downstream cause, which
        # the stage recorded in `notes` and which is otherwise reachable only in
        # pod logs that have long rotated away.
        joined = ", ".join(self.flags) or "<none declared>"
        cause = "; ".join(self.notes) or "<no cause recorded in notes>"
        super().__init__(
            f"stage {stage_name!r} reported success but flagged its own output as "
            f"FAILED via: {joined}. Underlying cause: {cause}. The stage swallowed "
            f"a downstream error and returned an empty-but-success-shaped output; "
            f"the run fails here rather than compiling a deliverable that is "
            f"silently missing what this stage was supposed to render."
        )


#: ``SlotUse.when`` for a call the stage makes when it can but DEGRADES without (soft
#: fail — e.g. word-timed captions via ASR in the compile stages). Listed in the
#: requirements manifest, never required by the run-start BYO preflight.
SLOT_USE_OPTIONAL = "optional"


@dataclass(frozen=True, slots=True)
class SlotUse:
    """A provider slot a stage may call (P8 requirements manifest).

    ``slot`` is a registry slot name (``llm``, ``tts``, ``image``, ``video_gen``,
    ``device_farm``, ``publish.youtube``). ``when`` names the opt-in that gates the
    call: ``"always"`` = whenever the stage runs, ``"optional"``
    (:data:`SLOT_USE_OPTIONAL`) = best-effort, the stage degrades without it,
    otherwise the brief/config key (``"brief.ai_image_gen"``) without which the
    stage skips."""

    slot: str
    when: str = "always"


class StageHandler[I: StageInput, O: StageOutput](ABC):
    """Pure async stage. No mutable state. No reaching into other stages.

    The concrete subclass declares:
      - ``name``: stable string identifier (matches DAGBlueprint references and
        the value stored in stage_outputs.stage_name).
      - ``input_model``: Pydantic class.
      - ``output_model``: Pydantic class.
      - ``handle``: async fn (payload, ctx) → output.
      - ``provider_slots``: the provider slots it may call (:class:`SlotUse`);
        ``tests/unit/test_stage_provider_slots.py`` fails when a stage reaches a
        slot client without declaring it.
    """

    name: ClassVar[str]
    input_model: ClassVar[type[StageInput]]
    output_model: ClassVar[type[StageOutput]]
    provider_slots: ClassVar[tuple[SlotUse, ...]] = ()

    @abstractmethod
    async def handle(self, payload: I, ctx) -> O:  # noqa: ANN001
        ...

    async def run(self, raw_payload: dict, ctx) -> O:  # noqa: ANN001
        try:
            payload = self.input_model.model_validate(raw_payload)
        except ValidationError as exc:
            raise ValueError(f"stage '{self.name}' input invalid: {exc}") from exc
        # Invariant: each subclass sets `input_model` to its own `type[I]`; ClassVar
        # cannot carry the TypeVar, so the narrowing is asserted here, not inferred.
        result = await self.handle(cast(I, payload), ctx)
        if not isinstance(result, self.output_model):
            raise TypeError(
                f"stage '{self.name}' returned {type(result).__name__}; "
                f"expected {self.output_model.__name__}"
            )
        return result
