"""Typed responses for bdd_playwright_service (port 6313).

Source of truth (read, not guessed):
- ``GET /api/bdd/{job_id}``        -> ``JobStatusResponse``
  (bdd_playwright_service ``src/api/schemas/common.py``)
- ``GET /api/bdd/{job_id}/result`` -> a **hand-built dict**
  (bdd_playwright_service ``src/api/routes/jobs.py:67-80``). That route declares
  ``-> dict[str, Any]`` and has no ``response_model``, so there is nothing to
  import and nothing to generate from: these models ARE the orchestrator-side
  contract for that payload.

The ``narration_script`` value is a JSON column that the service builds by hand
in ``src/services/job_processor.py:597-623`` (it is NOT serialized through the
service's own ``NarrationScriptOutput``). Two consequences encoded below:
- a job with no narration serializes as ``{}``, so every field must have a
  default that makes the empty object valid;
- ``expected_text`` is set **conditionally** (only when the knowledge pack
  authored one, ``job_processor.py:605-606``), so it is optional — modelling it
  as required would reject every non-authored demo run.

All fields are snake_case on the wire; no aliases are needed.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

#: Job lifecycle states. Mirrors ``JobStatus`` in the service's
#: ``src/api/schemas/common.py``. The ``/result`` route only ever returns
#: ``completed`` or ``failed`` (it 409s otherwise), but the status envelope
#: returned by ``GET /api/bdd/{job_id}`` can carry any of the four.
BddJobState = Literal["pending", "processing", "completed", "failed"]


class BddNarrationPart(BaseModel):
    """One narration segment, keyed by the ``clip-<n>`` join spine.

    ``part_id`` is the load-bearing join key across bdd -> frontend_test_runner
    -> compile; it is the only field the orchestrator treats as required.
    """

    part_id: str
    scenario_name: str = ""
    text: str = ""
    original_gherkin: str = ""
    word_count: int = 0
    estimated_duration_seconds: float = 0.0
    # O1b QA metadata: pack-authored expected on-screen text for this clip.
    # Present only when the knowledge pack authored one.
    expected_text: str | None = None
    #: The GUARANTEED per-scenario recording duration, in seconds — bdd #153.
    #:
    #: ⚠️ NOT ``estimated_duration_seconds``, which is how long the NARRATION takes to
    #: SPEAK. This is how long the CLIP is guaranteed to last. Conflating the two is
    #: exactly how a 4-second narration line came to imply a 4-second clip, and how the
    #: orchestrator ended up carrying its own ``_SECONDS_PER_SCENARIO = 4.0``.
    #:
    #: Owned by ``HERO_DWELL_MS`` in bdd's
    #: ``templates/scaffold/lib/recording_helpers.ts``, surfaced through
    #: ``step_def_generator.GUARANTEED_RECORDING_SECONDS_PER_SCENARIO``. It is deliberately
    #: read off the WIRE and never restated here: a copy of the number in this repo would
    #: silently desync the planner the next time that dwell changes.
    #:
    #: A FLOOR, not a prediction — the scenario's own step actions run before the dwell,
    #: so real clips land at or above it. Sizing a timeline off it under-fills at worst.
    #:
    #: ``None`` on every non-demo_video stream: those packages ship no ``lib/`` and no
    #: pacing hook, so bdd deliberately publishes nothing rather than a fabricated number.
    #: The orchestrator must read that as UNMEASURED, never substitute a default.
    expected_recording_seconds: float | None = None


class BddNarrationScript(BaseModel):
    """The generated narration script. Serializes as ``{}`` when absent."""

    parts: list[BddNarrationPart] = Field(default_factory=list)
    total_words: int = 0
    estimated_total_duration_seconds: float = 0.0
    tone: str = "tutorial"


class BddJobStatus(BaseModel):
    """``GET /api/bdd/{job_id}`` — the poll/status envelope."""

    job_id: str
    status: BddJobState
    # Not consumed by the orchestrator, so deliberately NOT a Literal: a new
    # bdd stream type must not be able to hard-fail a running pipeline.
    stream: str | None = None
    features_generated: int = 0
    step_defs_generated: int = 0
    created_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    download_url: str | None = None


class BddUngroundedGherkinConcept(BaseModel):
    """One demo ``Then`` step naming UI the pre-interaction page snapshot does not show.

    Mirrors bdd's ``UngroundedGherkinConcept`` TypedDict
    (``src/database/models.py``). Warning-only on bdd's side — the step itself
    is kept, this is a flag riding alongside it.
    """

    file: str = ""
    scenario: str = ""
    step: str = ""
    concept: str = ""


class BddUngroundedAssertion(BaseModel):
    """One assertion bdd's grounding pass REMOVED, and why.

    Mirrors bdd's ``UngroundedAssertion`` TypedDict (``src/database/models.py``).
    """

    file: str = ""
    step: str = ""
    selector: str = ""
    claim: str = ""


class BddStepCoverage(BaseModel):
    """``step_coverage`` — how much of a generated suite PASSES WITHOUT ACTING,
    plus the grounding findings bdd's LLM-invention checks produced.

    Mirrors bdd's ``StepCoverage`` TypedDict (``src/database/models.py``),
    minus ``duration_budget`` (kept opaque — the orchestrator does not
    currently read it and a full mirror would duplicate a second large shape
    for no consumer). Every field defaults so a genuinely-reported-but-clean
    suite (``steps_total`` > 0, every ``*_flagged`` count 0) still validates.

    ⚠️ ABSENT vs EMPTY: bdd's route returns ``job.step_coverage or {}`` — an
    empty dict on the wire means the DB column was never populated (pre-field
    job, or a non-demo_video stream that never ran the grounding pass), because
    a populated report always carries ``steps_total``/``degraded``/etc. keys.
    ``BddJobResult.step_coverage`` maps that empty-dict wire value to ``None``
    (see its validator) so ABSENT and "reported, zero findings" never collapse
    into the same zero-valued model.
    """

    steps_total: int = 0
    steps_stubbed: int = 0
    stub_ratio: float = 0.0
    degraded: bool = False
    degraded_threshold: float = 0.0
    stubbed_steps: list[str] = Field(default_factory=list)
    duration_budget: dict[str, object] | None = None
    ungrounded_assertions_dropped: int = 0
    ungrounded_assertions: list[BddUngroundedAssertion] = Field(default_factory=list)
    ungrounded_gherkin_concepts_flagged: int = 0
    ungrounded_gherkin_concepts: list[BddUngroundedGherkinConcept] = Field(default_factory=list)


class BddJobResult(BaseModel):
    """``GET /api/bdd/{job_id}/result`` — the full result envelope."""

    job_id: str
    status: BddJobState
    stream: str | None = None
    features_generated: int = 0
    step_defs_generated: int = 0
    narration_script: BddNarrationScript = Field(default_factory=BddNarrationScript)
    # The grounding report — see ``BddStepCoverage`` for the ABSENT/EMPTY/
    # NON-EMPTY contract. ``None`` is the ABSENT state; pydantic v2's default
    # ``extra="ignore"`` used to drop this field entirely (no ``step_coverage``
    # declared here at all), which is exactly the same failure mode as ABSENT —
    # a run whose bdd job flagged real invented-UI findings surfaced nothing.
    step_coverage: BddStepCoverage | None = None
    generation_time_seconds: float | None = None
    error: str | None = None
    created_at: str | None = None
    completed_at: str | None = None
    # Opaque JSON blobs the orchestrator never reads. ``dict[str, object]``
    # accepts any JSON object while still forcing a consumer to narrow — this is
    # the deliberate alternative to ``dict[str, Any]``.
    artifact_paths: dict[str, object] = Field(default_factory=dict)
    downstream_jobs: dict[str, object] = Field(default_factory=dict)

    @field_validator("step_coverage", mode="before")
    @classmethod
    def _empty_wire_dict_is_absent(cls, value: object) -> object:
        """``jobs.py`` sends ``job.step_coverage or {}`` — collapse the falsy
        wire value back to ``None`` (ABSENT) instead of letting it validate into
        an all-zero-defaults ``BddStepCoverage`` that reads identically to a
        genuinely reported, zero-findings suite."""
        if value in (None, {}):
            return None
        return value
