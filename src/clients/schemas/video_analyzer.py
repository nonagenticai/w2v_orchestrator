"""Typed responses for video_analyzer_service (port 6317).

Source of truth (read, not guessed) — video_analyzer_service
``src/api/schemas.py``:
- ``GET /api/analyze/{job_id}``        -> ``JobStatusResponse``  (:131-145)
- ``GET /api/analyze/{job_id}/result`` -> ``JobResultResponse``  (:148-155)

``JobResultResponse`` **inherits** ``JobStatusResponse``, so the result body
carries the whole status envelope plus five result fields; that inheritance is
mirrored here rather than flattened.

⚠️ The field is ``frame_analyses`` — there is **no** ``frames`` key. The route
(``src/api/routes/analysis.py:266``) declares ``response_model=JobResultResponse``,
so FastAPI **strips** any key not on that model: reading ``frames`` off this body
returned ``None`` on every job forever, silently killing the per-shot OCR signal
in ``stages/shared/footage_critique.py``. That bug is the reason this module
exists — the wire name is the only name.

Per-frame records (``frame_analyses`` items) are declared ``list[dict[str, Any]]``
server-side and stored in a free-form JSONB column, so nothing upstream validates
their shape. They are deliberately NOT modelled as a nested pydantic model here:
mirroring the server's own looseness keeps a legacy/garbled row degrading to an
empty OCR bucket instead of raising a ValidationError that the caller would then
misreport as "analyzer down". For reference, the live serializer
(``src/services/pipeline.py:823-831``) emits
``{index, timestamp, path, description, ocr_text, labels}``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class VideoAnalyzerScreenshotMatch(BaseModel):
    """A reference-image pHash match against a frame — mirrors ``ScreenshotMatch``."""

    name: str
    matched_frame: str | None = None
    timestamp: float | None = None
    distance: int


class VideoAnalyzerVerificationFailure(BaseModel):
    """One failed verification criterion — mirrors ``VerificationFailure``.

    ``expected`` / ``actual`` are ``Any | None`` server-side (they hold whatever
    the criterion compared). ``object`` accepts the same JSON while still forcing
    a consumer to narrow — the deliberate alternative to ``Any``.
    """

    criterion: str
    expected: object | None = None
    actual: object | None = None
    evidence: str | None = None


class VideoAnalyzerVerification(BaseModel):
    """Outcome of a VerificationCriteria run — mirrors ``VerificationResult``."""

    passed: bool
    criteria_total: int
    criteria_passed: int
    failures: list[VideoAnalyzerVerificationFailure] = Field(default_factory=list)
    #: Criteria that could NOT be evaluated because an upstream dependency was
    #: unavailable (the transcription backend down, say) — mirrors the server's
    #: own field and its reasoning: they are EXCLUDED from ``criteria_total`` and
    #: are never failures, because an infra outage must not produce a content
    #: "failed" verdict.
    #:
    #: This model has no ``extra="forbid"`` (deliberately — it deserialises a
    #: SEPARATELY DEPLOYED server, and forbidding extras would turn every future
    #: additive server field into a runtime ValidationError storm), so pydantic's
    #: default ``extra="ignore"`` DROPPED this list silently. A consumer reading
    #: ``criteria_passed == criteria_total`` then saw a clean pass where the
    #: server was reporting that it could not measure at all. The guard against
    #: the next omission is ``tests/contracts/test_video_analyzer_contract.py``,
    #: not a stricter model config.
    indeterminate: list[str] = Field(default_factory=list)
    transcript_similarity: float | None = None
    screenshot_matches: list[VideoAnalyzerScreenshotMatch] = Field(default_factory=list)


class VideoAnalyzerMotionSample(BaseModel):
    """One point on the motion curve — mirrors ``MotionSample``.

    Server truth: video_analyzer_service ``src/domain/analysis.py:43-53``
    (dataclass, serialized via ``asdict``). ``t`` is the timestamp of the LATER
    frame of the compared pair; ``motion`` is the 0..1 frame-delta.
    """

    t: float
    motion: float


class VideoAnalyzerVisualSample(BaseModel):
    """One per-frame visual metric point — mirrors ``VisualSample``.

    Server truth: video_analyzer_service ``src/domain/analysis.py`` (dataclass
    ``VisualSample``, wire model ``VisualSampleResponse``, added in analyzer PR #73).
    The whole-video ``mean_sharpness`` / ``mean_brightness`` / ``dominant_colors`` on
    :class:`VideoAnalyzerQualityMetrics` are aggregates — identical for every shot of a
    film by construction — so a consumer diffing shots against them measures nothing.
    These are the SAME quantities kept PER FRAME, which is what makes shot-to-shot
    continuity / cohesion / colour drift measurable.

    ``t`` is this frame's own timestamp (the join key against a cut time). Non-``t``
    fields default so a partial/newer sample shape cannot hard-fail a running pipeline
    (same contract as ``status`` not being a ``Literal`` elsewhere in this module).
    """

    t: float
    index: int = 0
    sharpness: float = 0.0
    brightness: float = 0.0
    dominant_rgb: list[int] = Field(default_factory=list)


#: Server truth: video_analyzer_service ``src/services/frame_extractor.py:25``.
#: The analyzer's own floor for "enough frames to judge a clip on".
_MIN_FRAMES_FOR_QUALITY = 5


class VideoAnalyzerQualityMetrics(BaseModel):
    """Aggregate visual quality metrics — mirrors ``QualityMetrics``.

    Server truth: video_analyzer_service ``src/domain/analysis.py:56-70``
    (dataclass, serialized via ``asdict``). ``motion_timeline`` / ``visual_timeline``
    are defaulted: rows persisted by an older analyzer predate the fields entirely, and
    the consumer contract is "no curve ⇒ fall back to a duration-derived default / to the
    whole-video means", never a validation failure.
    """

    mean_sharpness: float
    mean_brightness: float
    motion_score: float
    dominant_colors: list[str]
    overall_score: float
    motion_timeline: list[VideoAnalyzerMotionSample] = Field(default_factory=list)
    # Per-frame visual curve (analyzer PR #73). Empty on jobs an older analyzer wrote —
    # the consumer then falls back to the whole-video means, which gestalt correctly reads
    # as zero-dispersion / insufficient_evidence rather than a flattering constant.
    visual_timeline: list[VideoAnalyzerVisualSample] = Field(default_factory=list)
    # Sample-size provenance for every aggregate above (analyzer PR #122). The
    # analyzer added these documented explicitly as "the orchestrator's blank
    # hard-fail gate reads this body" — and this model omitted them, so
    # ``extra="ignore"`` silently dropped both and the producer's fix was INERT.
    #
    # Without them ``mean_sharpness`` is a number with no denominator: a clip whose
    # decode yielded ONE sharp frame is indistinguishable on the wire from a fully
    # clean clip, and a single DARK frame hard-fails a paid render. ``None``, not 0,
    # on a pre-#122 job: absent is UNMEASURED, which a consumer must be able to tell
    # apart from a measured zero.
    frames_planned: int | None = None
    frames_analyzed: int | None = None

    def sample_is_known_insufficient(self) -> bool:
        """Whether these aggregates are KNOWN to rest on too few frames to judge.

        Mirrors the analyzer's OWN sufficiency rule rather than inventing a second
        one: ``verifier.required_quality_sample`` requires
        ``frames_analyzed >= min(MIN_FRAMES_FOR_QUALITY (5), frames_planned)``. The
        ``min`` is the point — a clip legitimately too short to yield five samples
        must not be failed for doing exactly what it was told.

        DELIBERATE BOUNDARY — absent fields return ``False`` (i.e. "not known to be
        insufficient"), preserving the pre-#122 behaviour of trusting
        ``mean_sharpness``. Reading absent as insufficient would be the tidier
        "absent is UNMEASURED" posture, but it would stand the blank hard-fail gate
        DOWN for every clip the moment an analyzer without #122 served the request —
        turning the protection against a 100% black deliverable off fleet-wide as a
        side effect of a consumer-side fix. The defect being closed here is a
        verdict drawn from a sample we can SEE is too small; an unknown denominator
        is a different, pre-existing gap and is not silently widened or narrowed by
        this change.
        """
        if self.frames_planned is None or self.frames_analyzed is None:
            return False
        if self.frames_planned <= 0:
            return False
        return self.frames_analyzed < min(_MIN_FRAMES_FOR_QUALITY, self.frames_planned)


class VideoAnalyzerAudioGrid(BaseModel):
    """``POST /api/analyze/audio`` — the beat grid of one music bed.

    SYNCHRONOUS (unlike ``/api/analyze``, which is submit→poll): beat tracking a
    two-minute bed is seconds of DSP, not a vision pipeline, so the analyzer
    answers on the request. That is why this is a plain ``_request`` rather than
    a ``start``/``poll``/``get_result`` triple.

    Every field except ``duration_s`` is defaulted on purpose. The analyzer
    degrades rather than 500s on un-beat-trackable audio (an ambient bed with no
    percussive onsets), and ``grid_source`` is how it says so: ``"detected"`` for
    a real tracked grid vs a synthetic/assumed one. A consumer that treats an
    empty ``beats`` list as a hard failure would turn "this bed has no beat" into
    a failed run, so the cut planner must check ``beats``/``bpm_confidence``
    itself and fall back to an even-spacing cut.
    """

    duration_s: float
    bpm: float = 0.0
    bpm_confidence: float = 0.0
    #: Absolute seconds from the head of the track, ascending.
    beats: list[float] = Field(default_factory=list)
    #: The subset of ``beats`` that start a bar — the cut points a reel wants.
    downbeats: list[float] = Field(default_factory=list)
    onsets: list[float] = Field(default_factory=list)
    #: Free-form string, never a Literal: a new meter must not be able to
    #: hard-fail a running pipeline (same posture as ``status`` above).
    #:
    #: ⚠️ The analyzer sends an **int** — ``src/api/schemas.py:449`` declares
    #: ``meter: int`` ("Beats per bar used for the downbeat inference"), so the
    #: wire value is ``4``, not ``"4/4"``. Declared ``str``, this field rejected
    #: every real grid with ``Input should be a valid string``, and because
    #: ``music_select`` catches the ValidationError and degrades
    #: (``music_select.py:237``), the failure surfaced only as ``beats: []`` +
    #: a note — so every reel silently fell back to even spacing and NO run was
    #: ever cut to the beat. Coerce instead of widening the annotation: the
    #: orchestrator's own output contract (``music_select.MusicSelectOutput.meter``)
    #: is a string, and nothing parses it — downbeats are matched by TIME, not by
    #: an assumed modulus (``cut_plan._downbeat_indices``).
    meter: str = ""
    grid_source: str = ""

    @field_validator("meter", mode="before")
    @classmethod
    def _coerce_meter(cls, value: object) -> str:
        """Accept whatever the analyzer calls a meter and keep it a string.

        ``None`` becomes ``""`` so an omitted meter reads the same as a missing
        one rather than the literal ``"None"``.
        """
        if value is None:
            return ""
        return value if isinstance(value, str) else str(value)


class VideoAnalyzerJobStatus(BaseModel):
    """``GET /api/analyze/{job_id}`` — mirrors ``JobStatusResponse``.

    ``status`` / ``stage`` are deliberately NOT ``Literal``: a new analyzer stage
    name must not be able to hard-fail a running pipeline.
    """

    job_id: str
    status: str
    stage: str
    progress: float
    error_message: str | None = None
    artifacts: dict[str, str] | None = None
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    has_audio: bool | None = None
    vision_cost_usd: float = 0.0
    verification_passed: bool | None = None


class VideoAnalyzerJobResult(VideoAnalyzerJobStatus):
    """``GET /api/analyze/{job_id}/result`` — mirrors ``JobResultResponse``.

    Every result field is ``| None``: the analyzer only populates the stages the
    request enabled (``compute_quality_metrics``, ``extract_ocr``, ...), so a
    disabled stage legitimately yields ``None`` here.
    """

    transcript_data: dict[str, object] | None = None
    scene_boundaries: list[float] | None = None
    frame_analyses: list[dict[str, object]] | None = None
    quality_metrics: VideoAnalyzerQualityMetrics | None = None
    verification: VideoAnalyzerVerification | None = None
