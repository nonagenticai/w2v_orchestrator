"""demo_video_record — run the generated BDD suite + harvest per-scenario clips.

The novel stage. Hands the ``demo_video_gen`` BDD package straight to the
frontend_test_runner (by ``bdd_job_id`` — FTR downloads it from bdd_playwright),
which runs ``bddgen`` + Playwright with video recording and returns one clip per
scenario keyed by its ``clip-<n>`` id. Each clip is downloaded (license-gated S2S)
and re-published into the orchestrator's own storage so the video-compile service
can fetch it anonymously — the same download + ``store_capture_clip`` pattern
``site_capture`` uses with VPS recordings.

Output ``clips`` are keyed by their ``clip-<n>`` tag. In the default demo_video
path a line's id IS that tag (1:1). In authored/showcase mode a line names the
``clip`` it plays over, so several dialogue lines can share one beat — clips stay
keyed by tag and ``demo_video_compile`` fans them out per line. A scenario clip no
line references (e.g. an auth ``setup`` project's video) is dropped, not muxed.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, cast

from pydantic import BaseModel
from w2v_common.stages import stage_call

from src.clients.frontend_test_runner_client import FrontendTestRunnerClient
from src.clients.video_analyzer_client import VideoAnalyzerClient
from src.config import get_settings
from src.services.auth_secrets import dispatchable_auth, unseal_auth
from src.services.package_refs import (
    PackageRefError,
    brief_has_login,
    parse_package_ref,
    resolve_package_ref,
)
from src.services.stages.base import (
    ArtifactUrl,
    SlotUse,
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithDegradations,
    StageSkippedError,
)
from src.services.stages.demo_video.reconcile_timeline import _resolve_timing_config
from src.services.stages.shared import idle_trim
from src.services.stages.shared.degradation import (
    FAILED_SCENARIO_FOOTAGE,
    Degradation,
    degradation,
)
from src.services.stages.shared.footage_plans import split_second_clip_tags
from src.services.stages.shared.idle_trim import ClipIdleTrim
from src.services.stages.shared.narration_budget import wpm_is_usable
from src.services.stages.shared.time_budget import (
    DEFAULT_WPM_BAND,
    has_authored_clip_window,
    max_content_floor_credit,
)
from src.services.tts_storage import store_capture_clip

if TYPE_CHECKING:
    from src.config import PipelineSettings
    from src.services.runs.run_context import RunContext

log = logging.getLogger(__name__)

# Per-poll wall-clock deadline. BaseServiceClient.poll_job HARD-REJECTS any value
# > 600s, so a single poll is capped at 600. FTR runs bddgen + npm install
# (node_modules cached) + N recorded scenarios. A small demo suite completes in
# one poll; a large suite (many scenarios / cold npm install) can exceed 600s.
_POLL_TIMEOUT_S = 600.0
# Re-poll on timeout so the total wait can exceed the 600s per-poll cap. Each
# attempt re-polls the SAME FTR job (which keeps running server-side), so this
# is idempotent — unlike start_execution, which must run exactly once.
# 1 + _WAIT_RETRIES polls x 600s ~= 30 min of recording headroom.
_WAIT_RETRIES = 2

# Per-test Playwright timeout FTR applies as a HARD --timeout (it overrides the
# package's own playwright.config). Authored packages raise it further via
# brief.timeout_per_test_ms. Clamp to FTR's own accepted bounds (schema:
# ge=1_000, le=600_000) so an out-of-range brief value can't make FTR reject the
# whole job.
#
# 20s -> 45s -> 180s. Each raise answered the failure the previous budget caused:
#   20s  killed the generated nav hook mid-navigation, recording a black clip.
#   45s  was chosen to fail BROKEN generated selectors fast — a sound trade-off
#        while they were broken, and they were: the LLM invented class chains
#        (`li.todo-list`) that matched nothing, so every scenario died early and
#        no scenario ever needed a large budget.
#   180s answers what happens once they are NOT broken. bdd #119 (never invent a
#        selector; target the text the Gherkin supplies) and #121 (drop a matcher
#        whose expected argument is provably undefined) made scenarios actually
#        RUN, and a recording scenario is deliberately SLOW: it types visibly,
#        dwells on the result for video pacing, then asserts. Measured on
#        orchestrator E2E run 32185939947, scenario "Add a new todo item" hit
#        `Test timeout of 45000ms exceeded` while its own steps were resolving
#        correctly — the budget tuned for fast FAILURE had become the binding
#        constraint on SUCCESS, and the next scenario then failed looking for the
#        todo the first one never got to add.
#
# Raising the per-test ceiling does NOT slow a genuinely hung selector down.
# FTR derives independent sub-budgets from this value and caps each one
# (`execution_engine.py`: action min(10s), navigation min(15s), expect min(5s)),
# so a step that will never resolve still fails in seconds. Only the TOTAL a
# succeeding scenario may take grows — which is exactly the knob a recording
# needs. The suite-level wall clock remains the backstop for a pathological run.
_DEFAULT_TEST_TIMEOUT_MS = 180_000
_MIN_TEST_TIMEOUT_MS = 1_000
_MAX_TEST_TIMEOUT_MS = 600_000

# --- Reaped-job re-submission -------------------------------------------------
#
# An FTR pod deleted mid-job (eviction, node drain, OOM) leaves its row
# `processing` with nobody writing to it. FTR's reaper
# (`cleanup_service.py`, `reap_stuck_test_execution_jobs`) flips such a row to
# `failed` after 30 min without a heartbeat and stamps this EXACT message. That
# reaper is the ONLY producer of this string, which is what makes it a safe
# discriminator.
#
# WHY THIS IS NOT A REAL FAILURE, and cannot mask one:
#   * A suite whose TESTS fail does NOT reach a `failed` JOB. FTR returns
#     `completed` with a summary, and `_observe_ftr_verdict` below weighs it.
#     So retrying here can never re-run, or hide, a genuine assertion failure.
#   * The other producer of a `failed` job is `JobManager.mark_failed` from
#     `_process_job`'s except handler, whose message is `"{ExcType}: {exc}"` —
#     a real execution error (bad package, engine crash). It does NOT match this
#     sentinel and is NOT retried; it still fails the stage immediately.
# The match is deliberately on the stable prefix, not the whole sentence, so the
# em-dashed tail can be reworded without silently disarming the retry.
_FTR_REAPED_SENTINEL = "Job stuck (no heartbeat"

# Total submissions, NOT retries after the first: 2 == one re-submit. Each attempt
# can cost a full recording, and reaching this path already cost the reaper's
# 30-min detection latency, so the bound is tight on purpose. An unbounded retry
# against a pool that is evicting pods is worse than failing the run.
_SUBMIT_ATTEMPTS = 2
# The reaper's 30-min threshold already spaces attempts enormously; this is only
# a courtesy pause so a re-submit doesn't land in the same instant as the
# eviction that killed the last one. No exponential backoff to grow.
_RESUBMIT_DELAY_S = 5.0


def _is_reaped_job_error(exc: BaseException) -> bool:
    """True iff ``exc`` is poll_job's RuntimeError for a REAPED (pod-died) job.

    ``BaseServiceClient.poll_job`` raises ``RuntimeError(f"... failed: {detail}")``
    where ``detail`` is the job's ``error`` field — which FTR's status route
    populates verbatim from ``error_message`` (``routes.py``: ``error=job.error_message``).
    So the reaper's sentinel survives into the exception text.

    Note this is deliberately NOT a 404 check. A 404 from ``/report``/``/artifacts``
    is a *separate* storage bug (pod-local `persist_dir` on an emptyDir, so a poll
    landing on another replica misses the files) — a 404 there does NOT mean the
    job is lost, and retrying on it would mask that bug.
    """
    return isinstance(exc, RuntimeError) and _FTR_REAPED_SENTINEL in str(exc)


def _attempt_idempotency_key(run_id: object, stage_name: str, attempt: int) -> str:
    """Deterministic, ATTEMPT-SCOPED Idempotency-Key for one FTR submission.

    Attempt 0 is byte-identical to the historical key, so a resumable-timeout
    re-dispatch of the stage still dedupes onto the original job exactly as
    before — no duplicate paid recording.

    Attempts >0 MUST differ. FTR's dedupe is unconditional: `create_job` upserts
    on the partial unique index `(tenant_id, idempotency_key)` with ON CONFLICT
    DO NOTHING and returns `created=False` for ANY pre-existing row, including a
    reaped `failed` one. Re-submitting the identical key would therefore hand
    back the very corpse we are retrying, and the retry would be a guaranteed
    no-op loop. The suffix is derived, not random, so the key stays idempotent
    WITHIN an attempt — a re-dispatch during attempt 1 still dedupes onto
    attempt 1's job rather than opening a third.
    """
    base = f"{run_id}:{stage_name}"
    return base if attempt == 0 else f"{base}:r{attempt}"


def _resolve_test_timeout_ms(brief: dict[str, Any]) -> int:
    """Per-test timeout (ms) for FTR: brief override, else the 45s default."""
    raw = brief.get("timeout_per_test_ms")
    if raw is None:
        return _DEFAULT_TEST_TIMEOUT_MS
    try:
        value = int(raw)
    except TypeError, ValueError:
        return _DEFAULT_TEST_TIMEOUT_MS
    return max(_MIN_TEST_TIMEOUT_MS, min(_MAX_TEST_TIMEOUT_MS, value))


def _footage_gap_notes(
    valid_clip_tags: set[str], recorded_tags: set[str], any_clip: bool
) -> list[str]:
    """Diagnostics for narration lines whose footage never joined.

    ``any_clip=False`` → a leading summary note (the whole demo lost its footage).
    Then one note per referenced clip tag FTR never recorded — so a single mistyped
    or non-running ref among many doesn't vanish silently into a text card.

    These notes are SOFT (compile still renders narration over text cards), but the
    all-miss case is no longer *reached* softly on a paid capture: when a footage
    SOURCE was present and ZERO clips were harvested, ``handle`` hard-fails first
    (see the total-footage-loss gate) unless ``demo_require_footage`` is off. So in
    practice ``any_clip=False`` here means either the flag was disabled or there was
    never a source at all — the per-tag notes remain the normal partial-harvest path.
    """
    notes: list[str] = []
    if not any_clip:
        notes.append(
            "no recorded scenario clips matched the narration lines; "
            "compile will fall back to narrated text cards"
        )
    for tag in sorted(valid_clip_tags - recorded_tags):
        log.warning(
            "demo_video_record: clip %r referenced by a narration line but no "
            "matching footage was recorded; that beat falls back to a text card",
            tag,
        )
        notes.append(
            f"clip {tag!r} referenced by a narration line but no matching footage "
            "was recorded; that beat falls back to a text card"
        )
    return notes


# --- Per-clip visual QA (O1) ------------------------------------------------
# ffprobe metadata proves a harvested clip is the right *length*, but a white /
# blank / short-rendered frame at the right length is byte-indistinguishable
# from a good one — exactly the v12 failure the compile duration-guard can't
# catch. We run each STORED clip through video_analyzer's quality path (frame
# extract + OpenCV Laplacian-variance sharpness) and surface a flat/blank clip
# through the same `notes` → `degraded` channel the footage gaps use.
_QA_POLL_TIMEOUT_S = 90.0
# Cap extracted frames: blank/flat detection needs only a sample, and a tight
# cap keeps the per-clip QA job cheap. video_analyzer bounds it too (max_frames).
_QA_MAX_FRAMES = 40
# The video_analyzer criterion label for an OCR (wrong-screen) miss. OCR findings
# degrade a run but NEVER hard-fail it under strict mode: a mis-provisioned analyzer
# without tesseract yields empty OCR for every frame → a spurious "text not found".
# Sharpness/quality misses stay strict-eligible.
_OCR_CRITERION = "required_ocr_text"


def _qa_analysis_config(_settings: PipelineSettings, *, with_ocr: bool = False) -> dict[str, Any]:
    """video_analyzer ``AnalysisRequest`` overrides for cheap, self-contained QA.

    Runs frame-extract + OpenCV quality ONLY. ``generate_walkthrough=False`` with
    no transcript criteria makes the analyzer's ``_needs_transcript`` False, so it
    never calls ``asr.test`` (no Whisper); ``extract_frame_descriptions=False``
    + ``vision_provider="stub"`` means no worker_management vision spend. So a QA
    pass costs one ffprobe + N frame decodes + Laplacian variance — no LLM, no
    cross-service GPU dependency.

    ``with_ocr`` (O1b) adds tesseract OCR on the sampled frames — enabled only for a
    clip that carries an ``expected_text`` to assert, so a QA pass with nothing to
    match stays at the cheap quality-only cost.
    """
    return {
        "generate_walkthrough": False,
        "generate_ui_inventory": False,
        "generate_frame_index": False,
        "extract_frame_descriptions": False,
        "extract_ocr": with_ocr,
        "compute_quality_metrics": True,
        "vision_provider": "stub",
        "max_frames": _QA_MAX_FRAMES,
    }


def _qa_verification_criteria(
    settings: PipelineSettings, *, expected_text: str | None = None
) -> dict[str, Any]:
    """Blank/low-quality criteria. ``mean_sharpness`` (raw Laplacian variance) of
    a flat white/black frame ≈ 0; a real UI sits well above the (conservative)
    threshold. ``min_overall_quality`` is an optional, off-by-default secondary
    guard (omitted unless configured >0 so it never counts as a criterion).

    ``expected_text`` (O1b) adds a ``required_ocr_text`` assertion: the clip's frames
    must contain that text, so a clip that recorded the WRONG screen fails. OCR
    findings are surfaced but NEVER hard-fail a run (see ``_qa_clip``), because a
    mis-provisioned analyzer without tesseract yields empty OCR = spurious failure."""
    criteria: dict[str, Any] = {
        "min_sharpness_score": settings.demo_qa_min_sharpness,
        "fail_on_any": True,
    }
    if settings.demo_qa_min_overall_quality > 0:
        criteria["min_overall_quality"] = settings.demo_qa_min_overall_quality
    if expected_text:
        criteria["required_ocr_text"] = [expected_text]
    return criteria


#: Only a bare ``clip-<n>`` (the generator's own shape, no leading ``@``) is
#: safe to fold into a ``--grep`` pattern below — anything else is an
#: authored/showcase id we don't know how to match against a Gherkin tag.
_BARE_CLIP_TAG_RE = re.compile(r"^clip-\d+$")


def _grep_pattern_for_kept_clips(kept_clip_tags: set[str]) -> str | None:
    """Build the ``--grep`` value that selects only the KEPT beats' scenarios.

    Sending this avoids recording+narrating footage `demo_video_gen` already
    trimmed away (see `_apply_target_beats` / `_apply_duration_ceiling` /
    `_apply_feasibility_floor`) — today FTR downloads, runs, and mux-discards
    every scenario the bdd package generated, whether or not any narration
    line still references it (`record.py`'s own `valid_clip_tags` filter runs
    only AFTER the recording is paid for).

    Anchoring matters: Playwright's ``--grep`` is a plain, UNANCHORED
    ``RegExp(pattern, 'gi').test(...)`` against each scenario's
    ``describe-path + title + tags`` (space-joined) — confirmed by reading the
    installed ``playwright@1.50.0`` + ``playwright-bdd@8.5.1`` source
    (`loadUtils.js` / `test.js` / `util.js`). An unescaped ``@clip-1`` would
    also substring-match ``@clip-10``, ``@clip-11``, ... — `(?!\\d)` after the
    number closes that without relying on ``\\b`` (which is fragile around the
    ``-``/digit boundary). Each tag is also `re.escape`d in case a future
    generator ever emits something regex-special.

    The Playwright `auth.setup` project (when `auth_config` is set) is
    UNAFFECTED regardless of this pattern: playwright-bdd/Playwright always
    "prepend[s] dependency projects without filtration" (`loadUtils.js`), so a
    `--grep` here never excludes the login/session-setup step other projects
    `dependencies: ['setup']` on.

    Returns ``None`` (send no filter — today's safe, unfiltered behaviour) if
    ``kept_clip_tags`` is empty or contains anything outside the generator's
    own ``clip-<n>`` shape, since a pattern built from an unrecognised id could
    silently exclude a scenario we don't know how to match.
    """
    if not kept_clip_tags or not all(_BARE_CLIP_TAG_RE.match(t) for t in kept_clip_tags):
        return None
    alternatives = "|".join(re.escape(f"@{tag}") for tag in sorted(kept_clip_tags))
    return f"(?:{alternatives})(?!\\d)"


def _apply_grep_pattern(request: dict[str, object], kept_clip_tags: set[str]) -> None:
    """Set ``request["grep_pattern"]`` in place, if a safe one can be built.

    Split out of ``handle`` (rather than an inline ``if``) to keep that
    method's branch count under the C901 ceiling — this is a mechanical
    single-purpose mutation, not additional control flow worth inlining.
    Typed ``dict[str, object]`` (matching ``_submit_and_wait``'s own
    parameter, which this feeds into), not ``dict[str, Any]`` — the repo caps
    ``dict[str, Any]`` occurrences (``tests/test_any_ratchet.py``).
    """
    grep_pattern = _grep_pattern_for_kept_clips(kept_clip_tags)
    if grep_pattern:
        request["grep_pattern"] = grep_pattern


def _expected_text_by_tag(lines: list[dict[str, Any]]) -> dict[str, str]:
    """Map each footage tag → its expected on-screen text (O1b), for the OCR check.

    Keyed like ``valid_clip_tags`` (``line.clip`` or ``line.id``). A line without
    ``expected_text`` — or one that would blank an already-set entry — is skipped,
    so a beat that shares a footage clip keeps the text of whichever line set it.
    """
    by_tag: dict[str, str] = {}
    for line in lines:
        text = line.get("expected_text")
        if text:
            by_tag.setdefault(str(line.get("clip") or line["id"]), str(text))
    return by_tag


def _format_qa_failures(verification: dict[str, Any]) -> str:
    """Render a VerificationResult's failures into a short note fragment."""
    failures = verification.get("failures") or []
    parts: list[str] = []
    for failure in failures:
        criterion = failure.get("criterion") or "?"
        actual = failure.get("actual")
        parts.append(f"{criterion}={actual}" if actual is not None else str(criterion))
    return "; ".join(parts) or "visual quality below threshold"


class DemoVideoRecordInput(StageInput):
    pass


class MotionSample(BaseModel):
    """One point on a clip's motion curve — ``motion`` (0..1) observed at ``t`` seconds.

    Mirrors video_analyzer's ``QualityMetrics.motion_timeline`` entries 1:1.
    """

    t: float
    motion: float


class _ClipQa(NamedTuple):
    """What one clip's QA pass learned: an optional failure, plus its motion curve.

    ``failure`` is the pre-existing ``(note, strict_eligible)`` pair. ``motion_timeline``
    is a **free by-product** — the analyzer already computed it under the QA config's
    ``compute_quality_metrics``; before this we simply never fetched the result body on
    the happy path. Empty when QA errored, when the analyzer predates the field, or
    when the clip had too few decodable frames to form a pair.
    """

    failure: tuple[str, bool] | None = None
    motion_timeline: list[MotionSample] = []  # noqa: RUF012 -- NamedTuple default is read-only
    # The analyzer ffprobe's the FINALIZED clip, so it carries the real duration even
    # when FTR reported ``duration_ms=0`` (FTR races the Playwright ``recordVideo`` webm
    # before it is finalized). ``None``/``0`` when QA was off/errored or the analyzer is
    # too old to emit it. Used to repair ``DemoVideoClip.seconds`` (``_stamp_durations``).
    duration_seconds: float | None = None
    # True when the clip's measured ``mean_sharpness`` fell below the blank hard-fail
    # threshold (``effective_demo_qa_fail_sharpness``) — a genuinely BLANK recording (the
    # app never loaded). Set on EVERY QA return path, including the analyzer-PASS path,
    # so a blank clip the analyzer's own softer 3.0 criterion happened to pass is still
    # caught. ``False`` when QA was off/errored, the analyzer is too old to emit
    # ``mean_sharpness``, or the gate is disabled (``fail_sharpness == 0``). Consumed by
    # ``_qa_notes`` to hard-fail the run independent of ``demo_qa_strict``.
    blank: bool = False
    # True only when ``is_blank`` was actually COMPUTED for this clip. ``blank=False``
    # alone is ambiguous — it is the value both for "measured, and the clip is fine" and
    # for "never measured", and the blank hard-fail reads exactly that field. So a single
    # analyzer timeout returned ``_ClipQa()`` for every clip and the gate protecting the
    # customer from a 100% black video silently did not run, with no note to say so.
    # ``_qa_notes`` now degrades the run when this is False, which keeps the deliberate
    # "analyzer trouble must not fail the run" contract while making the gap visible.
    blank_gate_ran: bool = False


class _ClipDiag(NamedTuple):
    """What FTR itself said about one recorded clip — the blank-clip post-mortem.

    A blank recording almost always has an UPSTREAM explanation FTR already put on
    the wire (typically the bdd nav hook: ``nav hook: no paint within 12000ms``,
    leaving the page on ``about:blank``), which the orchestrator used to throw away —
    so a killed paid render carried only ``min_sharpness_score=0.0`` and no reason.

    ``ftr_error`` is joined from FTR's ``test_results[].error`` on
    ``(feature, scenario)`` ↔ ``(feature, name)`` (``clip_key`` does NOT exist on a
    test result). Every field is defaulted so an FTR that omits one still renders.
    """

    feature: str = ""
    scenario: str = ""
    ftr_status: str = ""
    ftr_error: str = ""
    duration_ms: float = 0.0
    width: int = 0
    height: int = 0
    size_bytes: int = 0


# Per-clip cap on the rendered diagnostic. The text lands in the raised RuntimeError
# message, which the DAG executor persists verbatim into `StageOutput.error` (Text) —
# N blank clips must not turn one row into a wall of truncated Playwright traces.
_DIAG_MAX_CHARS = 400
# `ftr_status` is a Playwright verdict word ("failed"/"timedOut"), so a long one means
# FTR drifted — clamp it rather than let it eat the error's budget.
_DIAG_MAX_STATUS_CHARS = 32
# The error is the payload; never starve it to nothing however wide the other fields go.
_DIAG_MIN_ERROR_CHARS = 80
# `feature::scenario`, the spec coordinates the clip came from. Bounded like every other
# component so a deeply-nested feature path cannot starve the error's budget.
_DIAG_MAX_ORIGIN_CHARS = 96
# Backstop: whatever the components do, one clip's diagnostic cannot exceed this.
_DIAG_HARD_MAX_CHARS = _DIAG_MAX_CHARS + 128
# How many DISTINCT FTR errors the total-footage-loss raise names. Same bounding
# reason as `_DIAG_MAX_CHARS`: the message is persisted verbatim into
# `StageOutput.error`, and a wholly-failed suite would otherwise paste every
# scenario's Playwright trace into one row.
_FOOTAGE_LOSS_MAX_ERRORS = 3
# FTR test statuses that are NOT a scenario failure. Everything else on a
# `test_results[]` entry (failed / timedOut / interrupted / an unknown future word)
# counts as failed, so a Playwright verdict FTR adds later still gets surfaced
# rather than silently dropped.
_FTR_OK_STATUSES = frozenset({"passed", "skipped", "expected", "flaky"})


def _as_int(value: object, default: int = 0) -> int:
    """Coerce an FTR-supplied number, defaulting instead of raising.

    FTR types these `int`/`float` on the wire, so this is drift insurance: a
    `ValueError` here would abort the stage with the wrong exception type from inside
    the clip loop — in the one code path whose whole purpose is making failures legible.
    """
    return int(_as_float(value, float(default)))


def _as_float(value: object, default: float = 0.0) -> float:
    """Coerce an FTR-supplied float, defaulting instead of raising. See `_as_int`."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


_DEFAULT_CLIP_FORMAT = "mp4"
# Matroska/WebM EBML magic (first 4 bytes) and the ISO-BMFF `ftyp` box tag,
# which sits at offset 4 of every MP4/MOV.
_EBML_MAGIC = b"\x1a\x45\xdf\xa3"
_FTYP_TAG = b"ftyp"


def _derive_clip_format(clip_key: str, video_bytes: bytes) -> str:
    """Derive the stored container extension by sniffing the downloaded bytes.

    FTR converts each recording to MP4 only when ``ffmpeg`` is present in its
    image; when the conversion fails closed it publishes the RAW Playwright
    WebM instead. Nothing on the ``scenario_clips`` entry names the container
    (``ScenarioClip`` carries no filename/format field, and ``video_url`` is an
    extension-less route), and ``get_clip_video`` hands back only the body — so
    the bytes in hand are the most reliable signal we have, and the only one
    that cannot be wrong about what was actually served.

    Storing a WebM under an ``.mp4`` name is what made video_analyzer report
    ``duration_ms=0, 0x0`` and the blank-clip gate hard-fail paid runs.
    Unrecognised/empty bodies keep the historical ``mp4`` default.
    """
    if video_bytes.startswith(_EBML_MAGIC):
        # FTR served a raw recording — store it under its TRUE extension so
        # downstream decoders see the real container, and make the broken
        # conversion visible instead of silently shipping a mislabelled file.
        log.warning(
            "demo_video_record: clip %s served as webm, not mp4 — "
            "frontend_test_runner's mp4 conversion is failing",
            clip_key,
        )
        return "webm"
    if video_bytes[4:8] == _FTYP_TAG:
        return "mp4"
    return _DEFAULT_CLIP_FORMAT


def _ftr_errors_by_scenario(
    result: dict[str, object],
) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    """Index FTR's ``test_results`` errors for the blank-clip diagnostic.

    Returns ``((feature, name) -> error, name -> error)``.

    The pair map is the primary join and is expected to ALWAYS hit: FTR builds both
    sides from the same Playwright spec object and the same immediate-parent suite
    title (``_collect_clip_specs`` sets ``feature``/``scenario``, ``_extract_tests_from_suite``
    sets ``feature``/``name``, recursing identically). A Scenario Outline does NOT
    break it — its examples share one spec, so every test carries the same
    ``name``/``feature`` as the clip.

    The name-only map is therefore purely a defensive fallback for FTR shape drift,
    and it deliberately REFUSES TO GUESS: a name claimed by more than one test result
    is dropped entirely. Two features can legitimately share a scenario title, and
    attaching another feature's traceback to this clip would send the next operator to
    debug the wrong scenario — a confidently-wrong post-mortem is worse than none.

    Defensive: the field may be absent, ``None``, or carry non-dict entries.
    """
    by_pair: dict[tuple[str, str], str] = {}
    by_name: dict[str, str] = {}
    ambiguous: set[str] = set()
    raw = result.get("test_results")
    if not isinstance(raw, list):
        return by_pair, by_name
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        tr = cast("dict[str, object]", entry)
        name = str(tr.get("name") or "")
        error = str(tr.get("error") or "").strip()
        if not (name and error):
            continue
        feature = str(tr.get("feature") or "")
        if (feature, name) not in by_pair:
            by_pair[(feature, name)] = error
        # A second DISTINCT (feature, name) pair claiming this name makes the
        # name-only lookup ambiguous; poison it rather than pick suite order.
        if name in by_name and by_name[name] != error:
            ambiguous.add(name)
        by_name.setdefault(name, error)
    for name in ambiguous:
        del by_name[name]
    return by_pair, by_name


def _blank_diag_text(clip_id: str, diag: _ClipDiag) -> str:
    """Render one clip's FTR-side post-mortem, capped at ``_DIAG_MAX_CHARS``.

    ``ftr_error`` passes through INTACT (``##SOFT_FAIL##`` markers, nav-hook JSON and
    all) up to the cap — it is the whole point of the diagnostic. It absorbs the
    truncation so the cheap structural signals (duration, geometry, bytes) always
    survive.

    Every component is individually clamped and the result is hard-truncated as a
    backstop, so no single pathological field can blow the cap or starve the budget to
    the point where the whole diagnostic collapses to an ellipsis.
    """
    if diag == _ClipDiag():
        return f"no FTR diagnostic for clip {clip_id}"
    status = (diag.ftr_status or "<none>")[:_DIAG_MAX_STATUS_CHARS]
    head = f"ftr_status={status}, ftr_error="
    # `feature`/`scenario` were stamped on every _ClipDiag but never rendered, so the
    # post-mortem named a clip id the operator then had to map back to a spec by hand.
    origin = _clamp_diag(f"{diag.feature}::{diag.scenario}", _DIAG_MAX_ORIGIN_CHARS)
    tail = (
        f", scenario={origin}"
        f", duration_ms={diag.duration_ms:g}, {diag.width}x{diag.height}"
        f", size_bytes={diag.size_bytes}"
    )
    error = diag.ftr_error or "<none>"
    # Always leave the error a usable slice, even if `tail` is unexpectedly wide.
    budget = max(_DIAG_MAX_CHARS - len(head) - len(tail), _DIAG_MIN_ERROR_CHARS)
    return f"{head}{_clamp_diag(error, budget)}{tail}"[:_DIAG_HARD_MAX_CHARS]


def _clamp_diag(text: str, budget: int) -> str:
    """Truncate ONE diagnostic component to ``budget`` chars, ellipsis-marked.

    The single truncation scheme in this module: every diagnostic that lands in a
    persisted ``notes`` entry or a raised ``StageOutput.error`` clamps through here,
    so a pathological Playwright trace can never blow one row into a wall of text.
    """
    if len(text) <= budget:
        return text
    return text[: max(budget - 1, 1)] + "…"


def _footage_loss_summary(ftr_errors_by_pair: dict[tuple[str, str], str]) -> str:
    """Render FTR's own reasons for a capture that returned NO clips at all.

    The blank-clip post-mortem (``_blank_diag_text``) hangs off a ``_ClipDiag`` built
    inside the harvest loop, so when ``scenario_clips`` is empty it never runs — yet
    ``test_results[].error`` still carries the upstream cause (typically the same bdd
    nav-hook timeout, repeated once per scenario).

    Dedupe on the error TEXT, not the ``(feature, name)`` key: the live shape is one
    identical failure across every scenario, and naming it once with the first scenario
    that hit it is the whole signal. Capped at ``_FOOTAGE_LOSS_MAX_ERRORS`` distinct
    reasons, each clamped to ``_DIAG_MAX_CHARS``.
    """
    parts: list[str] = []
    seen: set[str] = set()
    for (feature, name), error in ftr_errors_by_pair.items():
        if error in seen:
            continue
        seen.add(error)
        parts.append(f"{feature or '<no feature>'}/{name}: {error[:_DIAG_MAX_CHARS]}")
        if len(parts) >= _FOOTAGE_LOSS_MAX_ERRORS:
            break
    return "; ".join(parts)


def _failed_scenario_notes(result: Mapping[str, object]) -> list[str]:
    """One note per FAILED entry in FTR's ``test_results`` — scenario named, error clamped.

    ``TestCaseResult.status`` is a Playwright verdict word; FTR demotes a SOFT-failed
    scenario (``resilientClick``/``resilientFill`` returning without throwing) to
    ``"failed"`` before it reports, so this covers the silent-soft-fail shape that
    produced a "clean" 61s static film. Anything not in ``_FTR_OK_STATUSES`` counts.
    """
    raw = result.get("test_results")
    if not isinstance(raw, list):
        return []
    notes: list[str] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        tr = cast("dict[str, object]", entry)
        status = str(tr.get("status") or "")
        if status.lower() in _FTR_OK_STATUSES:
            continue
        name = str(tr.get("name") or "") or "<unnamed>"
        feature = str(tr.get("feature") or "") or "<no feature>"
        error = str(tr.get("error") or "").strip() or "<no error>"
        notes.append(
            f"FTR scenario {feature}/{name} {status or 'failed'}: "
            f"{_clamp_diag(error, _DIAG_MAX_CHARS)}"
        )
    return notes


class _NoScenarioPassedError(RuntimeError):
    """FTR passed ZERO scenarios — raised by the verdict, CAUGHT at the call site.

    Not a plain RuntimeError any more, because "no scenario passed" stopped being
    the same thing as "nothing was recorded". bdd #119 made a navigation
    visibility assert SOFT so the recording survives it, and FTR demotes ANY
    soft-failed test to ``failed`` (its honesty gate). So a scenario that typed,
    clicked and recorded perfectly well still reports ``failed`` if one decorative
    heading assert missed — and this stage was hard-failing that run before it had
    even looked at the footage, which also meant the operator never got to see the
    footage that would have proven it fine.

    The verdict still detects the condition; the CALL SITE now decides, weighing it
    against what the capture actually contains.
    """


def _ftr_verdict_notes(result: Mapping[str, object]) -> list[str]:
    """Propagate frontend_test_runner's OWN pass/fail verdict into the stage (#840).

    Before this, record read ``scenario_clips`` and NEVER looked at ``summary``: a run
    whose every scenario failed at step 1 harvested the (still-recorded) clips, reported
    ``notes == []`` and shipped a 61s static film as a clean success. FTR knew — it
    demotes a soft-failed scenario to ``failed`` and reports the counts — nobody asked.

    Three outcomes, all keyed off ``summary``:

    * ``failed > 0`` → ONE note per failed ``test_results`` entry (``runs/_shared.py``
      turns any note into a ``degraded`` run).
    * ``total > 0`` and ``passed == 0`` → ``RuntimeError``, the SAME mechanism
      ``_require_footage`` uses: the run lands ``status=failed``. Nothing the suite
      claimed to demo actually worked, so whatever was recorded is not a demo. There
      is NO automatic refund — see ``reconcile_timeline.py:1006``: ``grep -rn refund``
      over this repo finds docstrings only and ``run_usage_reconciler`` mirrors usage
      read-only, so failing here FORFEITS the spend of the stages that already ran.
      Raising is still right — shipping a worthless deliverable is worse — but the
      customer is not made whole by it.
    * summary absent / not a mapping / ``total <= 0`` → do NOTHING. An OLDER FTR that
      emits no summary (or a run with no tests) must not be able to fail a run — the
      same "the analyzer must not be able to fail a run" contract the blank gate's
      ``blank_gate_ran`` field encodes.

    This is the DETERMINISTIC signal: it reads an explicit upstream verdict rather than
    inferring capture health from pixels, so it — not the motion heuristic — is the
    authoritative hard-fail for a broken recording.
    """
    summary = result.get("summary")
    if not isinstance(summary, Mapping):
        return []
    counts = cast("Mapping[str, object]", summary)
    total = _as_int(counts.get("total"))
    if total <= 0:
        return []
    passed = _as_int(counts.get("passed"))
    failed = _as_int(counts.get("failed"))
    notes = _failed_scenario_notes(result) if failed > 0 else []
    if passed > 0:
        if notes:
            log.warning("demo_video_record: FTR reported %d/%d scenario(s) failed", failed, total)
        return notes
    log.error("demo_video_record: FTR passed 0 of %d scenario(s) — capture is not a demo", total)
    detail = "; ".join(notes[:_FOOTAGE_LOSS_MAX_ERRORS])
    raise _NoScenarioPassedError(
        f"demo_video_record: frontend_test_runner passed 0 of {total} scenario(s) "
        f"({failed} failed) — every scenario the demo was supposed to show is broken, "
        "so any footage recorded is a static/half-loaded app, not a demo. This is an "
        "EXPECTED capture loss, not a degrade: the run lands status=failed and the "
        "work already done is still billed (there is no automatic refund)."
        + (f" FTR reported: {detail}" if detail else "")
    )


class DemoVideoClip(StageOutput):
    id: str
    line_id: str
    video_url: ArtifactUrl
    seconds: float
    # The per-second motion curve measured off the RECORDED footage (ORCH-1.3c).
    # Populated only when per-clip QA ran (`demo_qa_enabled`) and the analyzer is new
    # enough to emit it; `None`/`[]` otherwise, and every consumer must degrade to its
    # own default rather than assume a curve exists. Appended last + defaulted so a
    # persisted clip from an older run still deserialises.
    motion_timeline: list[MotionSample] | None = None
    # ORCH-3.2 directed footage: the per-clip focus/cursor geometry FTR captured
    # JS-side during the Playwright run (`ScenarioClip.focus_track`/`cursor_track`),
    # plus the recorded source `viewport` ({w,h}) those coords were measured in. The
    # demo_video/showcase compile threads these onto the matching shot so the compiler
    # drives a synthetic eased cursor + auto-zoom. Appended last + defaulted None so a
    # persisted clip from an older run (or an FTR that emitted no track) still
    # deserialises and rides byte-identically.
    focus_track: list[dict[str, object]] | None = None
    cursor_track: list[dict[str, object]] | None = None
    viewport: dict[str, int] | None = None
    # FTR's verdict for the scenario that recorded this clip (``ScenarioClip.status``:
    # ``passed``/``failed``/...). ``None`` on a clip persisted before this field.
    scenario_status: str | None = None
    # The FILE length, set ONLY when ``seconds`` was trimmed below it
    # (``_trim_failed_scenario_tails``). ``seconds`` is the USABLE footage — the
    # length every consumer pins to (``reconcile_timeline._clip_seconds``,
    # ``compile._resolve_clip_anchor``/``_resolve_cold_open``) — so a failed
    # scenario's dead timeout wait is never budgeted into the film. ``None`` means
    # ``seconds`` IS the file length.
    recorded_seconds: float | None = None
    # What the idle pass (``_trim_idle_footage``) measured and did to this clip:
    # frozen seconds/fraction before and after, spans shortened, the coverage floor.
    # ``None`` when the pass did not run (disabled, or a clip persisted before it).
    idle_trim: ClipIdleTrim | None = None


class DemoVideoRecordOutput(StageOutputWithDegradations):
    clips: list[DemoVideoClip]
    notes: list[str] = []  # noqa: RUF012 — pydantic copies field defaults per-instance
    # Total frozen footage the idle pass cut across every clip (seconds). 0.0 when
    # nothing was cut or the pass was disabled.
    idle_removed_seconds: float = 0.0


def _stamp_motion_timelines(clips: list[DemoVideoClip], qa_by_clip: dict[str, _ClipQa]) -> None:
    """Attach each clip's measured motion curve (ORCH-1.3c), in place.

    Left as ``None`` when QA is off / errored / the analyzer is too old to emit one, so
    a run without a curve stays byte-identical to before.
    """
    for clip in clips:
        timeline = qa_by_clip.get(clip.id, _ClipQa()).motion_timeline
        if timeline:
            clip.motion_timeline = timeline


# Motion ceiling below which a clip's curve is indistinguishable from a STILL page.
#
# Calibrated against run 01a00df3-772f-760b-9615-1d2bd78f7278, which shipped a 61.47s
# "successful" demo that is 92% duplicate frames: the Playwright scenario's very first
# step failed (generated selectors did not match the target app), so nothing was ever
# clicked and all four clips recorded the same static landing page. On those clips the
# ONLY sample above the noise floor was the FIRST one — 0.0412, the page's initial paint
# — and every subsequent sample over the remaining 18-30s was <= 0.0033. A clip with real
# interaction (a click, a scroll, a route change) moves whole regions of the viewport and
# scores orders of magnitude higher. 0.01 is ~3x above the measured inert ceiling
# (0.0033) and ~4x below the paint spike (0.0412), so it separates the two populations
# without sitting on either.
_INERT_MOTION_MAX = 0.01


def _clip_inert_verdict(clip: DemoVideoClip) -> bool | None:
    """The tri-state inert verdict for one clip: ``True``/``False``/**unmeasured**.

    ``None`` means the clip could not be judged either way — the same "unmeasured is
    not a value, it is an absent measurement" gap the blank gate's ``blank_gate_ran``
    field exists to make expressible. Previously ``_clip_is_inert`` collapsed this case
    into ``False`` ("not inert"), which let an unmeasured clip masquerade as
    CORROBORATED evidence of real interaction wherever callers read ``not inert`` as
    "this clip proves the capture worked" (see ``_decide_no_scenario_passed``).

    Both conditions must hold for ``True`` (provably inert):

    (a) no interaction was captured at all — FTR emitted neither a ``focus_track`` nor a
        ``cursor_track`` (the JS-side geometry it records per click/focus), and
        ⚠️ **currently VESTIGIAL**: a survey of the last 30 ``demo_video_record``
        outputs found BOTH tracks null on 30 of 30 runs — FTR has never emitted them —
        so this condition is a constant ``True`` in prod and the verdict rests entirely
        on (b). It is KEPT, not deleted: the moment FTR starts emitting tracks it
        becomes meaningful again and correctly exonerates an interactive clip.
    (b) the measured motion curve, EXCLUDING its first sample (the page's initial paint,
        which fires on every clip whether or not the scenario did anything), never
        exceeds ``_INERT_MOTION_MAX``.

    With no motion curve, or a single-sample one (paint only), nothing post-paint was
    measured — return ``None`` rather than guessing. That keeps the guard from ever
    HARD-FAILING a run on absent evidence (the "the analyzer must not be able to fail a
    run" contract), while still making the gap visible to callers that check for it
    explicitly instead of treating "not True" as "corroborated fine".
    """
    if clip.focus_track or clip.cursor_track:
        return False
    post_paint = (clip.motion_timeline or [])[1:]
    if not post_paint:
        return None
    return all(sample.motion <= _INERT_MOTION_MAX for sample in post_paint)


def _clip_is_inert(clip: DemoVideoClip) -> bool:
    """True only for a clip PROVABLY inert. Unmeasured (``None``) reads as False here —
    callers that need to tell "provably active" apart from "unmeasured" must use
    ``_clip_inert_verdict`` directly (see ``_decide_no_scenario_passed``).
    """
    return _clip_inert_verdict(clip) is True


def _unmeasured_clip_ids(clips: list[DemoVideoClip]) -> list[str]:
    """Clip ids whose inert verdict is UNKNOWN — no post-paint motion sample at all."""
    return sorted(clip.id for clip in clips if _clip_inert_verdict(clip) is None)


def _any_clip_corroborated_active(clips: list[DemoVideoClip]) -> bool:
    """True only when at least one clip has POSITIVE evidence of real interaction.

    Distinct from ``not _every_clip_is_inert(clips)``: that also goes true when every
    non-inert clip is merely unmeasured, which is not evidence of anything.
    """
    return any(_clip_inert_verdict(clip) is False for clip in clips)


def _observe_ftr_verdict(result: Mapping[str, object]) -> tuple[list[str], str | None]:
    """Read FTR's verdict WITHOUT acting on it: ``(notes, none_passed_detail)``.

    Split from the decision so the "no scenario passed" case can be judged later,
    against footage that does not exist yet at this point in the stage.
    """
    try:
        return _ftr_verdict_notes(result), None
    except _NoScenarioPassedError as exc:
        return [], str(exc)


def _decide_no_scenario_passed(detail: str | None, clips: list[DemoVideoClip]) -> list[str]:
    """Weigh FTR's "no scenario passed" against what the capture actually contains.

    FTR saying no scenario passed is either "the app never worked" or "every
    scenario was demoted over a soft-failed decorative assert while recording
    fine". Only the footage tells them apart, and it does not exist until the
    clips are harvested — which is why this decision moved after harvesting.

    nothing captured ⇒ RAISE: both signals agree the run is worthless.
    real (CORROBORATED) interaction ⇒ NOTE: the recording is usable and is KEPT; the
    failing assertions ride in the note, which ``runs/_shared.py`` turns into a
    ``degraded`` run. A usable demo is not thrown away over a heading name, and
    the operator still sees exactly what missed.
    neither — every non-inert clip is merely UNMEASURED ⇒ still NOTE, never raise (the
    analyzer must not be able to fail a run), but say so honestly: this branch used to
    claim "the capture DID record interaction" on zero evidence, which is exactly the
    silent-exoneration bug an unmeasured clip must not be able to cause. Logged at
    ERROR so the gap is loud, matching the blank gate's ``blank_gate_ran`` treatment.
    """
    if detail is None:
        return []
    if _every_clip_is_inert(clips):
        raise RuntimeError(detail)
    if _any_clip_corroborated_active(clips):
        return [
            "FTR passed no scenario, but the capture DID record interaction — keeping "
            "the footage and degrading instead of failing. " + detail
        ]
    unmeasured = _unmeasured_clip_ids(clips)
    log.error(
        "demo_video_record: FTR passed no scenario and the inert-capture gate could "
        "not measure %d/%d clip(s): %s — keeping the footage, but interaction is "
        "UNPROVEN, not corroborated",
        len(unmeasured),
        len(clips),
        ", ".join(unmeasured),
    )
    return [
        "FTR passed no scenario, and the inert-capture gate could not measure "
        f"{len(unmeasured)}/{len(clips)} clip(s) ({', '.join(unmeasured)}) — keeping "
        "the footage (an analyzer gap must not fail a run), but interaction is "
        "UNPROVEN, not corroborated. " + detail
    ]


def _every_clip_is_inert(clips: list[DemoVideoClip]) -> bool:
    """True when EVERY clip is PROVABLY inert — the strongest evidence this gate has.

    The corroborating half of the FTR verdict. `_clip_is_inert` requires POSITIVE
    post-paint evidence, so an unmeasured clip is never called inert; therefore an
    empty clip list is inert by definition (nothing was recorded at all) while a
    clip we could not measure keeps this ``False`` and spares the run from the
    raise. Erring that way is deliberate: a false HARD FAIL destroys usable
    footage, a false degrade only adds a note.

    ⚠️ ``False`` here means "not proven all-inert" — it does NOT mean "some clip
    proved active". Callers that need the latter must use
    ``_any_clip_corroborated_active``; treating this function's ``False`` as
    proof of interaction is exactly the amplifier bug an unmeasured clip used to
    cause (fixed in ``_decide_no_scenario_passed``).
    """
    if not clips:
        return True
    return all(_clip_is_inert(clip) for clip in clips)


def _inert_capture_notes(clips: list[DemoVideoClip]) -> list[str]:
    """Note every clip that captured nothing — including when EVERY clip did.

    Before this, the defect above was completely silent: the stage's ``notes`` came back
    ``[]`` and the run reported a clean success, while downstream behaved correctly given
    the (empty) input — ``footage_direction`` reads ``has_click`` off the missing
    ``focus_track``, forces ``zoom="off"``, and ``footage_plans`` drops both tracks, so
    the renderer emits no motion and the customer is charged for a slideshow of one
    screenshot.

    ⚠️ **NOTE-ONLY as of #840 — this guard no longer raises.** It used to hard-fail the
    ALL-inert case. Two measurements retired that:

    * ``_clip_is_inert``'s condition (a) is currently a CONSTANT TRUE — FTR emits no
      ``focus_track``/``cursor_track`` on any run (0 of the last 30), so the verdict is
      a single pixel heuristic, and
    * 15 of the 21 clip-producing runs in that survey sit below ``_INERT_MOTION_MAX``.
      A raise here would therefore fail the majority of live runs on a heuristic.

    The authoritative failure signal is now ``_ftr_verdict_notes``: FTR's own
    ``summary`` says deterministically whether any scenario passed, and THAT raises
    (landing the run at ``status=failed``; no refund follows — ``reconcile_timeline.py:1006``).
    This function stays as the corroborating pixel evidence —
    every inert clip is named in ``notes``, and ``runs/_shared.py`` turns any note into
    a ``degraded`` run, so the defect is loud rather than silent. Same treatment for
    SOME and ALL inert (the ALL case adds one extra summary note), matching
    ``_footage_gap_notes``.
    """
    inert = [clip.id for clip in clips if _clip_is_inert(clip)]
    unmeasured = _unmeasured_clip_ids(clips)
    notes = [
        f"clip {clip_id} captured no interaction: no focus/cursor track and motion "
        f"<= {_INERT_MOTION_MAX} after the first (page-paint) sample — the recording "
        "is a static page"
        for clip_id in inert
    ]
    # Unmeasured is not "fine" — it is a GAP in the same evidence this gate relies on.
    # Silent before this fix: a clip with 0/1 motion samples (the coarse 3.0s fleet
    # sampling interval genuinely produces these on short clips) never appeared in
    # `notes` at all, so a run mixing one unmeasured clip with N-1 inert ones reported
    # only "N-1/N inert" — visually indistinguishable from "the Nth clip is fine".
    if unmeasured:
        log.error(
            "demo_video_record: inert-capture gate could not measure %d/%d clip(s): %s "
            "— no post-paint motion sample (verdict unproven, NOT treated as fine)",
            len(unmeasured),
            len(clips),
            ", ".join(unmeasured),
        )
        notes.append(
            f"inert-capture gate could not measure {len(unmeasured)}/{len(clips)} "
            f"clip(s) ({', '.join(unmeasured)}) — no post-paint motion sample; the "
            "inert/active verdict for these is UNPROVEN, not passing"
        )
    if not inert:
        return notes
    if len(inert) < len(clips):
        log.warning(
            "demo_video_record: %d/%d clip(s) captured nothing: %s",
            len(inert),
            len(clips),
            ", ".join(inert),
        )
        return notes
    log.error(
        "demo_video_record: all %d clip(s) captured nothing: %s",
        len(clips),
        ", ".join(inert),
    )
    notes.append(
        f"inert capture — all {len(clips)} clip(s) recorded a static page with no "
        "interaction (pixel heuristic; FTR's own verdict is the authoritative gate)"
    )
    return notes


def _stamp_durations(clips: list[DemoVideoClip], qa_by_clip: dict[str, _ClipQa]) -> None:
    """Repair a clip whose FTR ``duration_ms`` was 0 with the analyzer's real duration.

    FTR races the Playwright ``recordVideo`` webm and often reports ``duration_ms=0``
    though the finalized clip is a real N-second video; a 0-second clip silently loses
    its footage-pin in ``reconcile_timeline`` (it drops ``seconds<=0`` clips) and its
    auto clip-anchor / cold-open framing in ``compile``. The video_analyzer ffprobe'd
    the finalized clip during QA, so its ``duration_seconds`` is the true length.

    The ``clip.seconds <= 0`` guard is the regression firewall: a clip that already
    carries a real FTR duration is NEVER touched. Best-effort — a missing/zero analyzer
    duration leaves the clip unchanged (no failure, byte-identical to before).
    """
    for clip in clips:
        if clip.seconds > 0:
            continue
        probed = qa_by_clip.get(clip.id, _ClipQa()).duration_seconds
        if probed and probed > 0:
            clip.seconds = float(probed)


# A Playwright WAIT that expired: an ``expect(...)`` poll (``Timed out 5000ms waiting
# for expect(locator)...``) or an action/locator wait (``Timeout 5000ms exceeded.``).
# Both mean the page sat idle for exactly that long at the END of the attempt, because
# a hard failure ends the test. Deliberately does NOT match the whole-test ceiling
# (``Test timeout of 45000ms exceeded``, lowercase ``timeout of``): that is not a wait
# that follows the last action, it is the entire scenario.
_WAIT_TIMEOUT_RE = re.compile(
    r"(?:Timed out (\d+)ms waiting for expect\(|Timeout (\d+)ms exceeded)"
)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# FTR's error for a scenario demoted ONLY for bdd soft-fails
# (frontend_test_runner_service execution_engine.py `_extract_tests_from_suite`:
# `error = "soft-failed selector(s): " + "; ".join(reasons)`).
_SOFT_FAIL_PREFIX = "soft-failed selector(s):"
# FTR's failing verdicts (``_summarize_test_run``). ``flaky`` is NOT one: its kept
# video is the passing retry, while its error text can still describe the failed
# attempt — trimming on it would cut real footage.
_FAILED_SCENARIO_STATUSES = frozenset({"failed", "timedOut", "interrupted"})


def _wait_timeout_seconds(error: str) -> float | None:
    """The expired Playwright wait in ``error``, in seconds, or ``None``.

    ANSI colour codes are stripped first: FTR forwards Playwright's coloured message
    verbatim (``Timed out 5000ms waiting for \x1b[39m\x1b[2mexpect(``), and a pattern
    that happens to match through them today would silently stop matching the day
    the colouring moves. The FIRST match is the failure that ended the attempt.
    """
    text = _ANSI_RE.sub("", error or "")
    if text.lstrip().startswith(_SOFT_FAIL_PREFIX):
        # A bdd soft-fail (`resilientClick`/`resilientFill`) CATCHES its 2.5s miss and
        # the scenario KEEPS RUNNING; FTR only demotes the passed test to `failed` and
        # writes the annotations as the error. Its "Timeout 2500ms exceeded" is a wait
        # in the MIDDLE of a completed scenario, never the tail — cutting it would
        # delete real footage.
        return None
    match = _WAIT_TIMEOUT_RE.search(text)
    if match is None:
        return None
    ms = float(match.group(1) or match.group(2))
    return ms / 1000.0 if ms > 0 else None


def _last_recorded_action_t(clip: DemoVideoClip) -> float:
    """Latest ``t`` of any directed-footage event FTR captured (0.0 when none)."""
    latest = 0.0
    for track in (clip.focus_track, clip.cursor_track):
        for event in track or []:
            t = event.get("t")
            if isinstance(t, (int, float)) and not isinstance(t, bool):
                latest = max(latest, float(t))
    return latest


def _trim_failed_scenario_tails(
    clips: list[DemoVideoClip], diag_by_clip: Mapping[str, _ClipDiag]
) -> list[Degradation]:
    """Cut a FAILED scenario's clip at the start of the wait that failed it, in place.

    WHY. Every beat is pinned to its clip's ``seconds``, and a scenario that fails on
    an expired wait records that whole wait as a motionless tail. Measured on prod run
    ``01a0b007-9c2c-716a-883f-beefd14f4fd5``: clip-1 was 9.64s passing the run before
    and 14.56s failing on ``Timed out 5000ms`` — +4.92s of dead footage the film was
    lengthened by, scored "near-zero motion" by the footage critic.

    THE RULE — FTR sends no per-step timings and no failure timestamp
    (``ScenarioClip``: status/duration_ms/tracks only; ``TestCaseResult``:
    status/duration_ms/error), so the failure point is DERIVED: the timeout is the
    last thing the attempt did, so it started ``timeout`` seconds before the clip
    ended. ``seconds := seconds - timeout``. Teardown after the failure only moves the
    cut LATER, so no action before the failing wait can be cut. It is applied only
    when the tail is PROVEN dead; otherwise the clip is kept whole and the reason is
    recorded:

    * no expired WAIT in the error (an immediate assert mismatch has no idle tail);
    * a recorded focus/cursor action lands INSIDE the would-be tail (the wait was a
      soft, non-terminal one and the scenario kept going);
    * the timeout is not shorter than the clip.

    A clip is NEVER dropped here — a line keeps its only footage either way. Every
    failed clip yields one :data:`FAILED_SCENARIO_FOOTAGE` entry.
    """
    out: list[Degradation] = []
    for clip in clips:
        diag = diag_by_clip.get(clip.id)
        status = diag.ftr_status if diag is not None else (clip.scenario_status or "")
        if status not in _FAILED_SCENARIO_STATUSES or clip.seconds <= 0:
            continue
        where = f"clip {clip.id}"
        if diag is not None and diag.scenario:
            where += f" ({diag.feature}/{diag.scenario})"
        timeout = _wait_timeout_seconds(diag.ftr_error if diag is not None else "")
        # BEST-EFFORT guard: the track recorder's clock is page-relative (it starts at
        # the first page load and resets on a cross-origin navigation), so `t` is not
        # guaranteed to share the video's zero. It can miss a late action; it cannot
        # invent one.
        last_action = _last_recorded_action_t(clip)
        if timeout is None:
            reason = "no expired Playwright wait in its error, so no idle tail to cut"
        elif timeout >= clip.seconds:
            reason = f"its {timeout:.3f}s wait is not shorter than the {clip.seconds:.3f}s clip"
        elif last_action > clip.seconds - timeout:
            reason = (
                f"a recorded action at t={last_action:.3f}s lands inside the "
                f"{timeout:.3f}s would-be tail, so the wait was not terminal"
            )
        else:
            usable = round(clip.seconds - timeout, 3)
            log.warning(
                "demo_video_record: %s failed (%s) on a %.3fs wait — trimming the dead "
                "tail %.3fs -> %.3fs",
                where,
                status,
                timeout,
                clip.seconds,
                usable,
            )
            out.append(
                degradation(
                    FAILED_SCENARIO_FOOTAGE,
                    f"{where} scenario {status}: trimmed its {timeout:.3f}s timeout "
                    f"tail, footage {clip.seconds:.3f}s -> {usable:.3f}s",
                )
            )
            clip.recorded_seconds = clip.seconds
            clip.seconds = usable
            continue
        out.append(
            degradation(
                FAILED_SCENARIO_FOOTAGE,
                f"{where} scenario {status}: kept whole at {clip.seconds:.3f}s — {reason}",
            )
        )
    return out


# Wall-clock ceiling for the whole idle pass (every clip's detect + cut + store). It is
# HARD: every ffmpeg/ffprobe call runs under ``min(its own ceiling, remaining budget)``
# (``idle_trim.DETECT_TIMEOUT_S`` / ``CUT_TIMEOUT_S`` / ``PROBE_TIMEOUT_S``) and the
# store under the remaining budget, so a cut started at 239s is killed at 240s, not
# 180s later. A clip interrupted by it keeps its harvested footage (``cut_failed`` /
# ``detect_failed``); clips not reached are left untouched (``budget_exhausted``).
_IDLE_TRIM_BUDGET_S = 240.0


def _remaining(deadline: float) -> float:
    """Seconds left of the idle pass's wall-clock budget (``<= 0`` once spent)."""
    return deadline - time.monotonic()


def _idle_trim_enabled(ctx: RunContext) -> bool:
    """On by default; ``record.idle_trim: false`` turns the pass off."""
    record_cfg = ctx.config_snapshot.get("record") or {}
    return not (isinstance(record_cfg, Mapping) and record_cfg.get("idle_trim") is False)


def _narration_wpm(gen: Mapping[str, object]) -> float:
    """The rate the coverage floor estimates narration at.

    gen's ``resolved_wpm`` when gen says it is a MEASUREMENT of this run's voice
    (``resolved_wpm_measured`` — a synthesis of the draft's own lines, the calibration
    passage, or the author's ``direction.timing.wpm.natural``) and the value is a
    plausible rate (:func:`wpm_is_usable`). Anything else — an unmeasured fallback
    constant, a pre-Phase-B output with no flag, a non-number — falls back to the
    bottom of the default wpm band: with no measurement, erring slow (a longer
    estimate ⇒ a higher floor ⇒ less footage cut) is the safe direction.

    A measured rate is used as-is, NOT clamped to the band floor: run 01a15b57's voice
    measured 176.6 wpm, the old ``min(measured, 130)`` read every clip's narration ~36%
    long and its floor sat above the clip. The ``NARRATION_MARGIN`` (x1.10) applied in
    :func:`idle_trim.clip_floor_seconds` already covers downstream rewrites.
    """
    resolved = gen.get("resolved_wpm")
    if (
        gen.get("resolved_wpm_measured") is True
        and isinstance(resolved, (int, float))
        and not isinstance(resolved, bool)
        and math.isfinite(resolved)
        and wpm_is_usable(float(resolved))
    ):
        return float(resolved)
    return DEFAULT_WPM_BAND[0]


def _lines_by_clip(lines: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    """Spine lines grouped by the recording they play over (``clip`` or own id)."""
    out: dict[str, list[Mapping[str, Any]]] = {}
    for line in lines:
        if not isinstance(line, Mapping) or line.get("id") is None:
            continue
        out.setdefault(str(line.get("clip") or line["id"]), []).append(line)
    return out


def _positive_seconds(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    return None


def _played_seconds(clip: DemoVideoClip, bound: Sequence[Mapping[str, Any]]) -> float:
    """Seconds of ``clip`` the timeline actually PLAYS — what the film-level floor sums.

    A clip no spine line plays over (a split's second pane, a skipped scenario's
    recording) contributes NOTHING: counting it would inflate the film's footage and
    hand the trim room it does not have, pushing the PLAYED footage under the order.
    A line authoring a window plays only that window: ``clip_out_seconds`` minus an
    honoured ``clip_lead_seconds``; a window whose extent this module cannot compute
    (an unknown ``clip_anchor``) counts 0 — undercounting only shrinks the room, the
    safe direction. Several windowed lines on one clip are capped at the clip length.
    """
    if clip.seconds <= 0 or not bound:
        return 0.0
    if not any(has_authored_clip_window(line) for line in bound):
        return clip.seconds
    played = 0.0
    for line in bound:
        anchor = line.get("clip_anchor")
        if anchor not in (None, "start", "end", "auto"):
            continue
        lead = (
            _positive_seconds(line.get("clip_lead_seconds")) if anchor in (None, "start") else None
        )
        out = _positive_seconds(line.get("clip_out_seconds"))
        hi = min(out, clip.seconds) if out is not None else clip.seconds
        played += max(0.0, hi - (lead or 0.0))
    return min(played, clip.seconds)


def _apply_cuts(clip: DemoVideoClip, cuts: Sequence[tuple[float, float]]) -> None:
    """Remap the clip's time-indexed by-products onto the trimmed axis, in place."""
    clip.focus_track = idle_trim.remap_track(clip.focus_track, cuts)
    clip.cursor_track = idle_trim.remap_track(clip.cursor_track, cuts)
    if clip.motion_timeline:
        clip.motion_timeline = [
            MotionSample(t=round(idle_trim.remap_time(s.t, cuts), 3), motion=s.motion)
            for s in clip.motion_timeline
            if not any(lo < s.t < hi for lo, hi in cuts)
        ]


async def _trim_idle_footage(  # noqa: C901 — three phases (detect, budget, cut) share per-clip state
    clips: list[DemoVideoClip],
    bytes_by_clip: Mapping[str, tuple[bytes, str]],
    *,
    lines: Sequence[Mapping[str, Any]],
    wpm: float,
    ordered_target: float | None,
    run_id: str,
    stage_name: str,
    min_clip_seconds: float = 0.0,
    budget_s: float = _IDLE_TRIM_BUDGET_S,
) -> float:
    """Shorten every frozen span of every eligible clip to a short hold, in place.

    See :mod:`src.services.stages.shared.idle_trim` for the why and the thresholds.
    Returns the total seconds cut. FAIL-OPEN: any error leaves that clip exactly as
    harvested (status recorded on ``clip.idle_trim``, logged) — never raises.

    Eligibility: a clip must have a measured length, its bytes in hand, at least one
    spine line playing over it (a split's second pane has none — its timing belongs to
    another beat), and NO line authoring a file-time window on it (a cut would move
    the window's content; ``has_authored_clip_window`` is the shared definition).

    ``min_clip_seconds`` is an extra per-clip floor — the stage passes
    ``_short_clip_floor`` so a trim never pushes a clip under the short-clip check that
    runs right after it (which would turn a quality pass into a ``degraded`` run).

    Phases: (1) detect frozen spans on every eligible clip; (2) budget the cut per clip
    (narration floor) and per film (ordered body); (3) cut, re-encode, store, remap.
    The budget needs every clip's detection first, hence the split.
    """
    deadline = time.monotonic() + budget_s
    by_clip = _lines_by_clip(lines)
    spans_by_clip: dict[str, list[idle_trim.FrozenSpan]] = {}
    floor_by_clip: dict[str, float] = {}

    with tempfile.TemporaryDirectory(prefix="idle-trim-") as tmp:
        src_path: dict[str, Path] = {}
        # ── (1) detect ──
        for clip in clips:
            bound = by_clip.get(clip.id, [])
            if clip.seconds <= 0 or clip.id not in bytes_by_clip:
                clip.idle_trim = ClipIdleTrim(
                    status=idle_trim.STATUS_SKIPPED, detail="unmeasured clip or no bytes"
                )
                continue
            if not bound:
                clip.idle_trim = ClipIdleTrim(
                    status=idle_trim.STATUS_SKIPPED,
                    seconds_before=clip.seconds,
                    seconds_after=clip.seconds,
                    detail="no spine line plays over this clip",
                )
                continue
            if any(has_authored_clip_window(line) for line in bound):
                clip.idle_trim = ClipIdleTrim(
                    status=idle_trim.STATUS_SKIPPED,
                    seconds_before=clip.seconds,
                    seconds_after=clip.seconds,
                    detail="a line authors a clip window on this recording",
                )
                continue
            if _remaining(deadline) <= 0:
                clip.idle_trim = ClipIdleTrim(
                    status=idle_trim.STATUS_BUDGET_EXHAUSTED,
                    seconds_before=clip.seconds,
                    seconds_after=clip.seconds,
                )
                continue
            data, fmt = bytes_by_clip[clip.id]
            path = Path(tmp) / f"{clip.id}.{fmt}"
            try:
                path.write_bytes(data)
                spans = await idle_trim.detect_frozen_spans(
                    path, usable=clip.seconds, timeout_s=_remaining(deadline)
                )
            except (idle_trim.IdleTrimError, OSError) as exc:
                log.warning("demo_video_record: idle detect failed on %s: %s", clip.id, exc)
                clip.idle_trim = ClipIdleTrim(
                    status=idle_trim.STATUS_DETECT_FAILED,
                    seconds_before=clip.seconds,
                    seconds_after=clip.seconds,
                    detail=str(exc)[:300],
                )
                continue
            src_path[clip.id] = path
            spans_by_clip[clip.id] = spans
            narration = idle_trim.estimated_narration_seconds(
                (str(line.get("text") or "") for line in bound), wpm
            )
            floor_by_clip[clip.id] = max(idle_trim.clip_floor_seconds(narration), min_clip_seconds)

        # ── (2) budget ──
        reducible = {cid: idle_trim.reducible_seconds(s) for cid, s in spans_by_clip.items()}
        room = {
            clip.id: max(0.0, clip.seconds - floor_by_clip[clip.id])
            for clip in clips
            if clip.id in spans_by_clip
        }
        # Only footage the timeline PLAYS counts toward the ordered body (see
        # ``_played_seconds``) — an unplayed clip is not room to cut elsewhere.
        played = sum(_played_seconds(c, by_clip.get(c.id, [])) for c in clips)
        global_room = (played - ordered_target) if ordered_target else None
        budgets = idle_trim.allocate_trim_budgets(reducible, room, global_room)

        # ── (3) cut ──
        removed_total = 0.0
        for clip in clips:
            if clip.id not in spans_by_clip:
                continue
            spans = spans_by_clip[clip.id]
            before = clip.seconds
            frozen_before = sum(s.seconds for s in spans)
            base = ClipIdleTrim(
                status=idle_trim.STATUS_NO_IDLE,
                seconds_before=before,
                seconds_after=before,
                frozen_seconds_before=round(frozen_before, 3),
                frozen_seconds_after=round(frozen_before, 3),
                idle_fraction_before=round(frozen_before / before, 4),
                idle_fraction_after=round(frozen_before / before, 4),
                spans_detected=len(spans),
                floor_seconds=round(floor_by_clip[clip.id], 3),
            )
            cuts, protected = idle_trim.plan_cuts(
                spans,
                budgets.get(clip.id, 0.0),
                protected_times=idle_trim.track_times(clip.focus_track, clip.cursor_track),
            )
            cut_seconds = sum(hi - lo for lo, hi in cuts)
            if not spans or reducible.get(clip.id, 0.0) < idle_trim.MIN_CUT_SECONDS:
                clip.idle_trim = base
                continue
            if cut_seconds < idle_trim.MIN_CLIP_SAVING_SECONDS:
                budget = budgets.get(clip.id, 0.0)
                status = (
                    idle_trim.STATUS_PROTECTED
                    if protected and budget >= idle_trim.MIN_CUT_SECONDS
                    else idle_trim.blocked_status(
                        reducible.get(clip.id, 0.0), room.get(clip.id, 0.0), budget
                    )
                )
                clip.idle_trim = base.model_copy(
                    update={"status": status, "spans_protected": protected}
                )
                # A quality pass declining to cut is not a defect: logged, never a note.
                log.info(
                    "demo_video_record: idle not trimmed on %s: %s (reducible %.3fs, "
                    "clip room %.3fs, budget %.3fs, floor %.3fs, film room %s)",
                    clip.id,
                    status,
                    reducible.get(clip.id, 0.0),
                    room.get(clip.id, 0.0),
                    budget,
                    floor_by_clip[clip.id],
                    "unordered" if global_room is None else f"{global_room:.3f}s",
                )
                continue
            if _remaining(deadline) <= 0:
                clip.idle_trim = base.model_copy(
                    update={
                        "status": idle_trim.STATUS_BUDGET_EXHAUSTED,
                        "spans_protected": protected,
                    }
                )
                continue
            out_path = Path(tmp) / f"{clip.id}-idle.mp4"
            try:
                await idle_trim.cut_segments(
                    src_path[clip.id],
                    out_path,
                    idle_trim.kept_segments(cuts, before),
                    timeout_s=_remaining(deadline),
                )
                probed = await idle_trim.probe_seconds(out_path, timeout_s=_remaining(deadline))
                if _remaining(deadline) <= 0:
                    raise idle_trim.IdleTrimError("idle-trim time budget exhausted")
                video_url = await asyncio.wait_for(
                    store_capture_clip(
                        run_id=run_id,
                        stage_name=stage_name,
                        clip_id=f"{clip.id}-idle",
                        video_bytes=out_path.read_bytes(),
                        video_format="mp4",
                    ),
                    timeout=_remaining(deadline),
                )
            except Exception as exc:  # noqa: BLE001 — fail-open: keep the harvested clip
                log.warning("demo_video_record: idle cut failed on %s: %s", clip.id, exc)
                clip.idle_trim = base.model_copy(
                    update={
                        "status": idle_trim.STATUS_CUT_FAILED,
                        "spans_protected": protected,
                        "detail": str(exc)[:300],
                    }
                )
                continue
            # The encoded file is the truth for every downstream seek; fall back to the
            # arithmetic only when ffprobe could not read it.
            after = round(probed if probed is not None else before - cut_seconds, 3)
            frozen_left = idle_trim.frozen_after(spans, cuts)
            _apply_cuts(clip, cuts)
            clip.video_url = video_url
            clip.seconds = after
            # The new file contains ONLY usable footage (the detect + cut ran over
            # ``[0, seconds]``), so a record-trimmed tail is gone from the file too.
            clip.recorded_seconds = None
            removed_total += cut_seconds
            clip.idle_trim = base.model_copy(
                update={
                    "status": idle_trim.STATUS_TRIMMED,
                    "seconds_after": after,
                    "frozen_seconds_after": round(frozen_left, 3),
                    "idle_fraction_after": round(frozen_left / after, 4) if after > 0 else 0.0,
                    "removed_seconds": round(cut_seconds, 3),
                    "spans_shortened": len(cuts),
                    "spans_protected": protected,
                }
            )
            log.info(
                "demo_video_record: idle-trimmed %s %.3fs -> %.3fs (%d frozen span(s), "
                "%.3fs cut, floor %.3fs)",
                clip.id,
                before,
                after,
                len(cuts),
                cut_seconds,
                floor_by_clip[clip.id],
            )
    return round(removed_total, 3)


def _measured_seconds(clips: list[DemoVideoClip]) -> list[float]:
    """The clips whose length is actually KNOWN, in seconds.

    ``seconds <= 0`` is not "a zero-length clip", it is "nobody measured this one":
    FTR races the Playwright ``recordVideo`` webm and reports ``duration_ms=0``, and
    ``_stamp_durations`` can only repair that when per-clip QA ran and the analyzer
    was new enough to return a probe. ``reconcile_timeline`` drops those clips for
    the same reason. So they are excluded from BOTH the yield measurement and the
    short-clip check — a guard must never fire on absent evidence (the same
    positive-evidence contract ``_clip_is_inert`` and ``blank_gate_ran`` encode).
    """
    return [clip.seconds for clip in clips if clip.seconds > 0]


def _log_scenario_yield(clips: list[DemoVideoClip]) -> None:
    """Log the REALISED seconds-per-scenario of this capture.

    This used to print the mean against ``gen._SECONDS_PER_SCENARIO`` (4.0) — a figure
    extrapolated from two hand-observed samples on one 2026-08 E2E run and matching
    NEITHER mode the pipeline has since been measured in (1.2-3.9s unpaced, 16.7-36.9s
    paced). Comparing a real measurement against a fiction produced a delta that meant
    nothing, so the constant is gone (see ``gen._measured_seconds_per_beat``) and so is
    the comparison. What remains is the measurement itself, logged on EVERY capture, so
    the yield is one grep away instead of an inference from an outage.

    Log-only by design: the DECISION this measurement feeds is now taken at PLAN time
    (``gen._assert_plan_can_reach_floor``), before any of this was recorded. The
    per-clip floor below is what decides whether a specific recording is too short to
    be footage.
    """
    measured = _measured_seconds(clips)
    if not measured:
        log.info(
            "demo_video_record: scenario yield UNMEASURED — 0 of %d clip(s) carry a "
            "duration; this capture contributes no yield evidence",
            len(clips),
        )
        return
    mean = sum(measured) / len(measured)
    log.info(
        "demo_video_record: scenario yield measured mean=%.2fs/clip over %d of %d "
        "clip(s) (total=%.2fs, min=%.2fs, max=%.2fs)",
        mean,
        len(measured),
        len(clips),
        sum(measured),
        min(measured),
        max(measured),
    )


def _ordered_target_seconds(ctx: RunContext) -> float | None:
    """The ordered duration ``reconcile_timeline`` will hold this film to, or ``None``.

    Reuses ``reconcile_timeline._resolve_timing_config`` — the SAME ladder, not a
    re-derivation of it — so the two stages can never disagree about what was
    ordered: ``direction.timing.target_seconds`` wins outright, the deliverable's
    ``target_duration_seconds`` is the fallback. A ``StageSkippedError`` means
    ``reconcile_timeline`` itself has nothing to reconcile to (no ``direction.timing``
    block, ``mode: off``, or neither value present) — in which case there is no
    ordered-duration floor tighter than the derived-scenario-count assumption below.

    A malformed override (``target_seconds`` / ``tolerance_pct`` / a ``wpm`` field
    authored as a non-numeric string, or any shape ``_resolve_timing_config`` cannot
    coerce with ``float()``) raises ``ValueError``/``TypeError`` there, NOT
    ``StageSkippedError`` — this stage is NOTE-only (see ``_short_clip_notes``), so
    that must degrade to "no ordered target resolvable" too, not crash the harvest
    stage over a config typo that ``reconcile_timeline`` itself will report clearly,
    three stages later, when it hits the same bad value.
    """
    try:
        return _resolve_timing_config(ctx).target
    except StageSkippedError:
        return None
    except (ValueError, TypeError) as exc:
        log.warning("demo_video_record: could not resolve the ordered target: %s", exc)
        return None


def _short_clip_floor(
    settings: PipelineSettings, ordered_target: float | None, clip_count: int
) -> float:
    """Seconds below which a harvested clip is materially shorter than was ordered.

    NOT a new magic number. It is the run-level gross-underfill criterion the pipeline
    already ships, applied per clip: ``reconcile_timeline`` refuses a film below
    ``pipeline.demo_timeline_underfull_fail_ratio`` (default 0.5) of the duration that
    was ORDERED (``ordered_target``, resolved via the same ladder reconcile_timeline
    uses — see :func:`_ordered_target_seconds`), split evenly across the harvested
    clips: ``ratio * ordered_target / clip_count``.

    THE OLD FORMULA (``ratio * _SECONDS_PER_SCENARIO``, unconditionally) rested on a
    premise that is FALSE whenever the scenario count is set explicitly rather than
    derived: it assumed the ordered duration is always ``_SECONDS_PER_SCENARIO`` x the
    scenario count, but ``_scenarios_for_target`` short-circuits past that arithmetic
    the moment ``brief.max_scenarios_per_feature`` is given, so the ordered duration
    comes from ``direction.timing.target_seconds`` / the deliverable's target instead
    and ``_SECONDS_PER_SCENARIO`` is never consulted for the count. Measured live on
    ``demo_video_calibration`` (orders 30s, ``max_scenarios_per_feature: 3``): the old
    formula floored a clip at 0.5 x 4.0 = 2.0s while ``reconcile_timeline`` actually
    required 0.5 x 30.0 = 15.0s — accepting captures at harvest that were certain to
    hard-fail three stages later.

    With no resolvable ``ordered_target`` (no ``direction.timing`` block, ``mode:
    off``, or neither ``target_seconds`` nor a deliverable target present) there is
    nothing tighter to fall back to, so the old scenario-count assumption still
    stands. Setting the ratio to 0 disables both this floor and the run-level gate.

    WITH NO RESOLVABLE ``ordered_target`` THERE IS NO FLOOR (returns ``0.0``, i.e. the
    per-clip check does not fire). The old fallback multiplied the ratio by
    ``gen._SECONDS_PER_SCENARIO`` — but that constant was never a measurement of
    anything (two hand-observed samples on one run, against a yield since measured
    across 1.2-36.9s), so the "floor" it produced was a fabricated threshold reported in
    the same words as a real one. An absent ordered target means UNMEASURED, not 2.0s,
    and a guard must never fire on absent evidence — the same positive-evidence contract
    ``_clip_is_inert`` and ``blank_gate_ran`` encode. The run-level gate is likewise
    inert without a target (``reconcile_timeline`` skips), so nothing is left unguarded
    that used to be guarded.
    """
    if ordered_target and clip_count > 0:
        return settings.demo_timeline_underfull_fail_ratio * ordered_target / clip_count
    return 0.0


def _short_clip_notes(
    clips: list[DemoVideoClip], settings: PipelineSettings, ctx: RunContext
) -> list[str]:
    """Attribute every harvested clip materially shorter than was ordered.

    WHY AT HARVEST: nothing in the capture path checked a clip's length at all. A
    capture that came back at 0.8s/clip instead of the assumed 4.0s produced a film 5x
    underfull, and the FIRST thing that noticed was ``reconcile_timeline``'s
    GROSSLY-UNDERFULL raise three stages later — which only fires below
    ``demo_timeline_underfull_fail_ratio`` of the target, so a film 20-49% short shipped
    silently. This names the short recordings at the point they are harvested, while the
    job id, the FTR diagnostic and the spec coordinates are still in hand.

    Also adds a WHOLE-CAPTURE note the moment the total measured footage is already
    below ``ratio * ordered_target`` — the exact number ``reconcile_timeline`` will
    apply — quoting both figures, so a run that cannot possibly clear the run-level
    gate says so here instead of three stages later.

    NOTE-only, deliberately, matching ``_footage_gap_notes`` and ``_inert_capture_notes``
    (#840): ``runs/_shared.py`` turns any note into a ``degraded`` run, so this is loud
    without throwing away usable footage over a length heuristic. The hard-fail for a
    film that is genuinely too short to ship stays where it is armed and verified — the
    ``reconcile_timeline`` gate — which this deliberately does not duplicate or move.
    """
    ordered_target = _ordered_target_seconds(ctx)
    ratio = settings.demo_timeline_underfull_fail_ratio
    floor = _short_clip_floor(settings, ordered_target, len(clips))
    measured = _measured_seconds(clips)
    notes: list[str] = []

    if ordered_target and ratio > 0 and measured:
        total = sum(measured)
        required = ratio * ordered_target
        # Ask the GATE's own arithmetic, do not re-derive it. `reconcile_timeline`
        # credits freeze pad against this floor, capped at `MAX_PAD_CREDIT_RATIO` of
        # target; no pad exists yet at harvest, so assume the most generous pad the cap
        # could ever allow (`max_content_floor_credit`). If even that best case lands
        # under the floor the run PROVABLY cannot satisfy the gate — which is the claim
        # this note makes, so it must be true by construction rather than by a second
        # formula that happens to agree. Two components computing one quantity two ways
        # is how live run 01a059a2 shipped: this note correctly predicted the failure
        # ("capture yielded 1.40s; the 5.00s order needs >= 2.50s") and
        # `reconcile_timeline` then disagreed with its own predictor and passed the run.
        best_case = max_content_floor_credit(total, ordered_target)
        if best_case < required:
            notes.append(
                f"capture yielded {total:.2f}s; the {ordered_target:.2f}s order needs "
                f">= {required:.2f}s at the {ratio:.0%} floor, and even the maximum "
                f"credited freeze pad only reaches {best_case:.2f}s, so this run cannot "
                "satisfy reconcile_timeline"
            )

    if floor > 0:
        short = [clip for clip in clips if 0 < clip.seconds < floor]
        if short:
            # The reference quantity the floor was computed against — the same one
            # `_short_clip_floor` used. With a resolvable `ordered_target` that is
            # `ordered_target / clip_count` (a per-clip share of what was actually
            # ORDERED), never `_SECONDS_PER_SCENARIO`; quoting the scenario constant
            # here when an ordered target drove the floor is exactly the "note quotes
            # the wrong target" defect this PR exists to fix (see #922) — it would
            # also print a negative, nonsensical "short by roughly -0.50s" once the
            # clip is longer than the (unrelated) scenario constant but still below
            # the real floor.
            # `floor > 0` implies a resolvable `ordered_target` and `clips` (see
            # `_short_clip_floor`), so the reference quantity is always the per-clip
            # share of what was actually ORDERED — never a scenario constant.
            assumed = ordered_target / len(clips) if ordered_target and clips else 0.0
            assumed_desc = f"{assumed:.2f}s/clip share of the {ordered_target or 0.0:.2f}s order"
            clip_notes = [
                f"clip {clip.id} recorded {clip.seconds:.2f}s — below the {floor:.2f}s "
                f"floor ({assumed_desc}); the film will be short by roughly "
                f"{assumed - clip.seconds:.2f}s per clip like this"
                for clip in short
            ]
            if len(short) == len(measured):
                clip_notes.insert(
                    0,
                    f"EVERY measured clip ({len(short)}) is below the "
                    f"{floor:.2f}s floor — the capture yields "
                    f"{sum(measured) / len(measured):.2f}s/clip against the "
                    f"{assumed_desc}, so the ordered duration is unreachable from this "
                    "footage",
                )
            notes.extend(clip_notes)

    for note in notes:
        log.warning("demo_video_record: %s", note)
    return notes


async def _trusted_package_url(package_url: str, brief: dict, tenant_id: object) -> str:
    """The URL FTR may fetch the authored package from.

    The package's auth.setup types the tenant's credentials, so a LOGIN run only ever
    runs a package this orchestrator stored for THIS tenant, pinned by sha256
    (package_refs) — re-checked here for briefs stored before the door enforced it.
    A ref becomes a fresh URL only after its hash matches; never falls back."""
    if parse_package_ref(package_url) is not None:
        return await resolve_package_ref(package_url, tenant_id)
    if brief_has_login(brief):
        raise PackageRefError(
            "demo_video_record: a login run needs an uploaded package_ref, not an "
            "external package_url; credentials were not sent"
        )
    return package_url


class DemoVideoRecordStage(StageHandler[DemoVideoRecordInput, DemoVideoRecordOutput]):
    name = "demo_video_record"
    input_model = DemoVideoRecordInput
    output_model = DemoVideoRecordOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (
        SlotUse("llm"),
        SlotUse("tts"),
    )

    def __init__(
        self,
        client: FrontendTestRunnerClient | None = None,
        analyzer: VideoAnalyzerClient | None = None,
    ) -> None:
        self._client = client or FrontendTestRunnerClient()
        self._analyzer = analyzer or VideoAnalyzerClient()

    async def _qa_clip(
        self,
        *,
        clip_key: str,
        video_url: str,
        settings: PipelineSettings,
        expected_text: str | None = None,
    ) -> _ClipQa:
        """QA one stored clip; return its failure (if any) plus its motion curve.

        Best-effort: any analyzer error/timeout is swallowed (logged, returns an
        empty :class:`_ClipQa`) so a QA-service outage degrades nothing. Only a
        definite ``verification_passed is False`` produces a note.

        ``strict_eligible`` is True when a sharpness/quality criterion failed (a
        real blank/short-render) and False when the ONLY failure is the OCR
        wrong-screen check — so ``demo_qa_strict`` never hard-fails a run on OCR,
        which a mis-provisioned (tesseract-less) analyzer would flag spuriously.
        ``with_ocr`` is enabled only for a clip that actually carries an
        ``expected_text`` to assert (O1b), keeping other clips at quality-only cost.
        """
        with_ocr = bool(settings.demo_qa_ocr_enabled and expected_text)
        try:
            job_id = await self._analyzer.start_analysis(
                video_url=video_url,
                verification=_qa_verification_criteria(
                    settings, expected_text=expected_text if with_ocr else None
                ),
                config_overrides=_qa_analysis_config(settings, with_ocr=with_ocr),
            )
            status = await self._analyzer.poll_analysis(job_id, timeout=_QA_POLL_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001 -- best-effort QA: never fail the run on analyzer trouble
            log.warning("demo_video_record: QA skipped for clip %s: %s", clip_key, exc)
            return _ClipQa()
        # Harvest the motion curve AND mean_sharpness on BOTH paths (pass and fail): the
        # QA config already paid for `compute_quality_metrics`, so both are free
        # by-products of a job that has already run. Fetched before the pass/fail branch
        # because the blank gate must see a blank clip even when the analyzer's own softer
        # 3.0 criterion happened to PASS it (the analyzer-pass early-return below).
        motion_timeline, mean_sharpness = await self._fetch_quality(job_id, clip_key)
        # A genuinely blank recording (app never loaded) measures ≈0 variance; a real UI
        # sits >=3. `effective_demo_qa_fail_sharpness` is clamped below the degrade band,
        # so this NEVER fires on a clip the degrade band would pass. `None` sharpness
        # (old analyzer / fetch error) ⇒ no fire (hard-fail needs positive evidence);
        # threshold 0.0 ⇒ gate disabled.
        fail_thr = settings.effective_demo_qa_fail_sharpness
        is_blank = fail_thr > 0 and mean_sharpness is not None and mean_sharpness < fail_thr
        # "The gate reached a verdict" == we had the evidence to decide, i.e. a real
        # `mean_sharpness`. A `None` (old analyzer, or `_fetch_quality` failed) yields
        # `is_blank=False` for LACK of evidence, which is indistinguishable from a
        # measured-and-fine clip — the same ambiguity as the analyzer-outage path above.
        # `fail_thr == 0` deliberately disables the gate by config and is NOT reported as
        # a gap: an explicit operator choice must not nag on every run.
        gate_ran = mean_sharpness is not None
        # The analyzer ffprobe'd the finalized clip → the real duration, even when FTR
        # reported 0. Carried on every return below so `_stamp_durations` can repair a
        # clip whose FTR `duration_ms` was 0. `None`/absent on an older analyzer.
        probe_dur = status.get("duration_seconds") if isinstance(status, dict) else None
        if status.get("verification_passed") is not False:
            # Analyzer PASSED its own criteria — but still hard-flag a blank it let
            # through (independent mean_sharpness read; closes the pass-path false-negative).
            failure = (
                (f"clip {clip_key} is blank (mean_sharpness={mean_sharpness})", True)
                if is_blank
                else None
            )
            return _ClipQa(
                failure=failure,
                motion_timeline=motion_timeline,
                duration_seconds=probe_dur,
                blank=is_blank,
                blank_gate_ran=gate_ran,
            )
        try:
            verification = await self._analyzer.get_verification_result(job_id)
        except Exception as exc:  # noqa: BLE001 -- verdict body optional; we already know it failed
            log.warning("demo_video_record: QA verdict fetch failed for clip %s: %s", clip_key, exc)
            # Without the verdict body we can't tell which criterion failed. If OCR
            # was in play, stay soft (OCR must never hard-fail); otherwise it can
            # only be a quality miss, which is strict-eligible.
            return _ClipQa(
                failure=(
                    f"clip {clip_key} flagged low visual quality (details unavailable)",
                    not with_ocr,
                ),
                motion_timeline=motion_timeline,
                duration_seconds=probe_dur,
                blank=is_blank,
                blank_gate_ran=gate_ran,
            )
        failed_criteria = {(f.get("criterion") or "") for f in verification.get("failures") or []}
        strict_eligible = bool(failed_criteria - {_OCR_CRITERION})
        reasons = _format_qa_failures(verification)
        log.warning("demo_video_record: clip %s failed visual QA: %s", clip_key, reasons)
        return _ClipQa(
            failure=(f"clip {clip_key} failed visual QA: {reasons}", strict_eligible),
            motion_timeline=motion_timeline,
            duration_seconds=probe_dur,
            blank=is_blank,
            blank_gate_ran=gate_ran,
        )

    async def _fetch_quality(
        self, job_id: str, clip_key: str
    ) -> tuple[list[MotionSample], float | None]:
        """Read the finished QA job's motion curve AND its ``mean_sharpness``.

        Both come from the SAME ``get_result`` body (the QA config already paid for
        ``compute_quality_metrics``), so this is one network call, no extra cost. The
        motion curve feeds the cold-open; ``mean_sharpness`` (raw Laplacian variance,
        ≈0 on a blank frame) feeds the blank hard-fail gate.

        Best-effort ⇒ ``([], None)`` on any trouble. A missing/garbled curve is NOT a
        QA failure — the cold-open falls back to its duration-derived default. A
        ``None`` sharpness (older analyzer without the field, a fetch error, or — since
        analyzer #122 — a sample too small to support a verdict) means the blank gate
        simply cannot fire for this clip: hard-failing a run needs positive near-zero
        evidence, never an absence of evidence. ``[]``/``None`` is also the correct
        answer for an analyzer too old to emit either field.

        The motion curve is still returned on the insufficient-sample path: it feeds
        the cold-open, not a hard-fail, and a short curve is a usable curve.
        """
        try:
            result = await self._analyzer.get_result(job_id)
        except Exception as exc:  # noqa: BLE001 -- quality metrics are a bonus, never a gate
            log.warning(
                "demo_video_record: quality metrics unavailable for clip %s: %s", clip_key, exc
            )
            return [], None
        if result.quality_metrics is None:
            return [], None
        timeline = [
            MotionSample(t=sample.t, motion=sample.motion)
            for sample in result.quality_metrics.motion_timeline
        ]
        # SAMPLE SIZE FIRST. ``mean_sharpness`` is an aggregate, and an aggregate
        # without its denominator is not evidence. Analyzer PR #122 started emitting
        # ``frames_planned``/``frames_analyzed`` precisely so this gate could tell a
        # PARTIAL analysis from a complete one; until this change the orchestrator's
        # client schema omitted both and ``extra="ignore"`` dropped them, so:
        #
        #   * a clip whose decode yielded ONE sharp frame passed the blank gate
        #     identically to a fully clean clip, and
        #   * a single DARK frame hard-failed a paid render.
        #
        # Both verdicts were drawn from n=1 and both read as confident. A partial
        # analysis therefore yields ``None`` — the SAME answer as an analyzer outage,
        # which the existing contract already routes to ``blank_gate_ran=False`` and a
        # loud ERROR in ``_qa_notes``. That keeps the measurement gap EXPRESSIBLE and
        # LOUD rather than silently resolving it to "fine" or to "blank".
        quality = result.quality_metrics
        if quality.sample_is_known_insufficient():
            log.warning(
                "demo_video_record: blank gate has no verdict for clip %s — quality rests "
                "on %s of %s planned frames",
                clip_key,
                quality.frames_analyzed,
                quality.frames_planned,
            )
            return timeline, None
        # Only trust a numeric sharpness. The schema types it ``float``, so the real
        # path always is one; a malformed/partial body yields None ⇒ the blank gate
        # simply cannot fire (hard-failing needs positive near-zero evidence, never an
        # absent or garbled field). ``bool`` is excluded so a stray True/False can't read
        # as 1.0/0.0.
        sharp = quality.mean_sharpness
        sharpness = (
            float(sharp)
            if isinstance(sharp, (int, float)) and not isinstance(sharp, bool)
            else None
        )
        return timeline, sharpness

    async def _qa_clips(
        self,
        clips: list[DemoVideoClip],
        settings: PipelineSettings,
        expected_by_tag: dict[str, str] | None = None,
    ) -> dict[str, _ClipQa]:
        """QA every stored clip concurrently; return each clip's QA outcome by clip id.

        Concurrent so N clips don't serialise their polls. Empty when QA is
        disabled or there are no clips — which is also why a consumer of
        ``motion_timeline`` must always have a no-curve fallback: turning QA off
        turns the motion curve off with it. ``expected_by_tag`` maps a clip's footage
        tag → its expected on-screen text (O1b); a clip without an entry runs the
        blank/quality check only.
        """
        if not (settings.demo_qa_enabled and clips):
            return {}
        by_tag = expected_by_tag or {}
        qa_results = await asyncio.gather(
            *(
                self._qa_clip(
                    clip_key=clip.id,
                    video_url=str(clip.video_url),
                    settings=settings,
                    expected_text=by_tag.get(clip.id),
                )
                for clip in clips
            )
        )
        return dict(zip((clip.id for clip in clips), qa_results, strict=True))

    async def _submit_and_wait(self, request: dict[str, object], ctx: RunContext) -> str:
        """Submit the FTR job and wait for it, re-submitting ONLY a REAPED job.

        ``request`` is typed as the EXACT parameter type of the seam it forwards
        to — ``FrontendTestRunnerClient.start_execution(payload: dict[str, object])``
        — rather than ``dict[str, Any]``. This method never indexes the payload;
        it only hands it back to the client, so ``object`` is the honest element
        type and it keeps the ``Any`` ratchet (tests/test_any_ratchet.py) flat.

        Returns the job id that actually finished. Every non-reaped outcome —
        success, a genuine execution failure, a timeout — leaves the loop on its
        first pass, so behaviour outside the pod-died case is unchanged.
        """
        last_job_id = ""
        for attempt in range(_SUBMIT_ATTEMPTS):
            last_job_id = await self._client.start_execution(
                request,
                idempotency_key=_attempt_idempotency_key(ctx.run_id, self.name, attempt),
            )
            # Re-poll the (idempotent) wait on timeout so a large suite that runs past
            # the 600s per-poll cap keeps waiting instead of failing. retry_on=()
            # means ONLY a poll timeout retries (the job is still running); a real FTR
            # job failure — which poll_job raises as a non-TimeoutError — propagates
            # immediately and correctly fails the stage (recording IS core to a demo
            # whose brief asked for it; an expected footage loss lands the run at
            # `failed` and there is no refund for the spend already incurred — see
            # reconcile_timeline.py:1006).
            try:
                await stage_call(
                    # `jid=` binds THIS attempt's id into the factory; the bare loop
                    # variable would be a late-binding hazard (ruff B023).
                    lambda jid=last_job_id: self._client.wait_for_execution(
                        jid, timeout=_POLL_TIMEOUT_S
                    ),
                    timeout_seconds=_POLL_TIMEOUT_S + 60,
                    retries=_WAIT_RETRIES,
                    retry_on=(),
                    stage_name=self.name,
                )
            except RuntimeError as exc:
                # Exhausted, or a genuine failure: propagate UNCHANGED so the stage
                # fails exactly as it does today. Only a reaped job with an attempt
                # remaining is re-submitted.
                if not _is_reaped_job_error(exc) or attempt == _SUBMIT_ATTEMPTS - 1:
                    raise
                log.warning(
                    "demo_video_record: FTR job %s was reaped (producing pod died) — "
                    "re-submitting under a fresh idempotency key (attempt %d/%d): %s",
                    last_job_id,
                    attempt + 2,
                    _SUBMIT_ATTEMPTS,
                    exc,
                )
                await asyncio.sleep(_RESUBMIT_DELAY_S)
                continue
            break
        return last_job_id

    async def handle(self, payload: DemoVideoRecordInput, ctx: RunContext) -> DemoVideoRecordOutput:
        # OPT-OUT, skip-clean: a run can disable the video-capture branch entirely by
        # setting `record.enabled: false` in its config — e.g. a plan-only run that
        # only needs the script/claim-analysis spine (script_doctor → source_ingest →
        # claim_finish) and pauses before the render tail. record is otherwise the ONE
        # UNCONDITIONAL demo_video stage (every sibling is already opt-in skip-clean);
        # because it is a CONCURRENT sibling of the claim branch (both depend only on
        # demo_video_gen), a hard failure here re-raises out of the DAG round and
        # aborts the whole run, stranding the independent claim_finish branch as
        # `pending` — even though claim_finish never needs record's output (its only
        # consumers, reconcile_timeline/footage_direction/compile, all sit after
        # claim_finish and are gated off by pause_after_stages). Skipping clean lets
        # such a run reach claim_finish + its pause deterministically instead of
        # depending on the (downstream, non-deterministic) bddgen/capture path. Guarded
        # so the default (no `record` block, or enabled != false) still records.
        record_cfg = ctx.config_snapshot.get("record") or {}
        if record_cfg.get("enabled") is False:
            raise StageSkippedError(
                "demo_video_record: record.enabled=false — video-capture branch opted out"
            )

        gen = ctx.prior_output("demo_video_gen")
        if not gen:
            raise RuntimeError("demo_video_record: demo_video_gen output required")

        brief = ctx.config_snapshot.get("brief") or {}
        target_url = (
            brief.get("target_url") or brief.get("source_url") or brief.get("url") or ""
        ).strip()
        if not target_url:
            raise RuntimeError("demo_video_record: brief.target_url is required")

        # Keep the FTR clips whose ``@clip-<n>`` key a narration line plays over.
        # A line plays over its ``clip`` ref when set (several dialogue lines may
        # share one beat), else over the clip with its own id — so this is the set
        # of referenced clip tags, NOT the line ids. Clips are output keyed by that
        # tag; demo_video_compile fans them back out to each line via its ``clip``.
        valid_clip_tags = {str(line.get("clip") or line["id"]) for line in gen.get("lines", [])}
        # …plus any footage tag an author-pinned split names as its second pane: no line
        # plays over it, but the compositor needs it recorded and kept.
        valid_clip_tags |= split_second_clip_tags(gen.get("lines", []))
        expected_by_tag = _expected_text_by_tag(gen.get("lines", []))

        # Read BEFORE the request is built: `demo_require_footage` drives BOTH sides of
        # the zero-footage decision (FTR's `expect_recording` below and this stage's own
        # `_require_footage` gate), so it must be resolved once, here.
        settings = get_settings().pipeline

        request: dict[str, Any] = {
            "target_url": target_url,
            # Bound each generated scenario so a step whose selector never matches
            # the live DOM fails fast instead of pinning Playwright's 30s default
            # and overrunning FTR's wall budget. FTR applies this as a HARD per-test
            # --timeout (it OVERRIDES the package's own playwright.config `timeout`)
            # and derives its action/navigation/expect sub-budgets from it (exposed
            # to the suite as PW_*). The 45s default suits generated suites — it must
            # stay comfortably above the generated nav hook's own budget, or the hook
            # is killed mid-navigation and the scenario records a black clip (the 20s
            # default did exactly that) — but an AUTHORED showcase package
            # (brief.package_url) can carry legitimately
            # long steps — e.g. live server-side room detection that runs 60-160s —
            # so let the brief raise the ceiling, clamped to FTR's accepted range.
            "timeout_per_test_ms": _resolve_test_timeout_ms(brief),
            # BOTH zero-footage gates are driven by the SAME setting, deliberately.
            # FTR's `_guard_zero_footage` defaults `expect_recording=True` and FAILS the
            # job on zero clips; because this request never sent the field, that upstream
            # failure fired FIRST — `wait_for_execution` raised, `get_job_result` was
            # never reached, and `_require_footage` (and therefore the
            # PIPELINE_DEMO_REQUIRE_FOOTAGE=false fallback its message advertises) could
            # NEVER run. Sending the setting through means: gate ON → FTR still hard-fails
            # early (cheaper, and it knows why); gate OFF → FTR completes with an empty
            # clip list and THIS stage decides, so the documented narrated-text-card
            # fallback is actually reachable.
            "expect_recording": settings.demo_require_footage,
        }
        # Authored mode (showcase): run a pre-built suite hosted at an http(s) URL.
        # FTR accepts ``package_url`` as a first-class alternative to ``bdd_job_id``
        # (SSRF-validated, no license token attached), so a proven recording suite
        # drives the footage instead of a freshly generated package. The generated
        # path's gen.package_url is a relative ``/api/bdd/...`` and never set here.
        package_url = (brief.get("package_url") or "").strip()
        if package_url:
            package_url = await _trusted_package_url(package_url, brief, ctx.tenant_id)
            request["package_url"] = package_url
        else:
            bdd_job_id = str(gen.get("bdd_job_id") or "")
            if not bdd_job_id:
                # No footage SOURCE: an authored VO spine with neither a
                # brief.package_url nor a demo_video_gen.bdd_job_id has nothing to
                # record. DEGRADE (empty clips + note) rather than hard-failing the
                # run — the note marks the run `degraded` (runs/_shared.py) so the
                # timeline logic (reconcile_timeline) still runs and the deliverable
                # is a best-effort VO/text-card video. This is deliberately NARROW and
                # it is the ONLY degrade path: only "nothing was ever recordable"
                # degrades. Reaching past this branch means a footage SOURCE exists
                # (package_url or bdd_job_id) and the capture is PAID FOR, so from here
                # on every total loss hard-fails — an FTR job that errors (below), and
                # equally an FTR job that reports success but yields ZERO harvestable
                # clips (the total-footage-loss gate after the harvest loop). Both are
                # EXPECTED footage losses, so the run lands `status=failed` instead of
                # being silently degraded and shipped as a finished deliverable. That is
                # NOT a chargeback: there is no refund path (reconcile_timeline.py:1006),
                # and the BDD render already paid for is forfeited. Failing is still the
                # right call — a footage-free "demo" is worth less than nothing — but the
                # customer is not made whole.
                log.warning(
                    "demo_video_record: no package_url or bdd_job_id — nothing to "
                    "record; degrading run"
                )
                return DemoVideoRecordOutput(
                    clips=[],
                    notes=[
                        "no footage source — brief supplied no package_url and "
                        "demo_video_gen produced no bdd_job_id (authored VO spine); "
                        "recording skipped"
                    ],
                )
            request["bdd_job_id"] = bdd_job_id
            # Only for the GENERATED path: an authored `package_url` suite is a
            # proven, pre-built showcase whose titles/tags this stage does not
            # control, so it is never filtered.
            _apply_grep_pattern(request, valid_clip_tags)
        # Unsealed in memory HERE and nowhere else: the brief stores the login secrets
        # sealed (auth_secrets), FTR needs the plaintext to type into the form. A dict
        # is one login; a list is one login per origin (multi-app demo). Never logged.
        # Orchestrator-only keys (signs_into, needs_reentry) are stripped: they are
        # never forwarded to a service that signs in.
        auth = dispatchable_auth(unseal_auth(brief.get("auth_config")))
        if auth:
            request["auth_config"] = auth
            # A Playwright trace records every action's arguments — `fill(password)`
            # included — and FTR uploads trace zips into the shared object store with
            # the HTML report. FTR defaults `trace_on_failure=True` (`--trace
            # on-first-retry`, which overrides the package config), so any retry of a
            # login-gated run would persist the plaintext password. Off whenever a
            # login is present; the CLI flag wins over the package's own config.
            request["trace_on_failure"] = False

        # Stable across a resumable-timeout re-dispatch (run_id + stage name), so a
        # resumed record re-uses the ORIGINAL FTR job instead of spawning a second
        # paid bddgen+Playwright recording. Mirrors the compile idempotency key.
        job_id = await self._submit_and_wait(request, ctx)
        result = await self._client.get_job_result(job_id)

        # FTR's OWN verdict. Observed FIRST, but no longer DECIDED first.
        #
        # "0 scenarios passed" used to mean "nothing was recorded", and hard-failing
        # on it immediately was right. It stopped meaning that: bdd #119 made a
        # navigation visibility assert SOFT so the recording survives it, and FTR
        # demotes ANY soft-failed test to `failed`. A scenario that typed, clicked
        # and recorded perfectly can now report `failed` because one decorative
        # heading assert missed — measured on E2E run 32253408902, where both
        # scenarios were demoted solely for
        # `getByRole('heading', { name: /TodoMVC/ })` (the app's heading is `todos`).
        #
        # Raising here also destroyed the evidence: the stage stored NO output, so
        # nobody could see whether the footage was usable. So the verdict is now
        # weighed against the capture below, once the clips exist.
        verdict_notes, no_scenario_passed = _observe_ftr_verdict(result)

        run_id = str(ctx.run_id)
        clips: list[DemoVideoClip] = []
        notes: list[str] = list(verdict_notes)
        recorded_tags: set[str] = set()
        bytes_by_clip: dict[str, tuple[bytes, str]] = {}
        scenario_clips = cast("list[dict[str, object]]", result.get("scenario_clips") or [])
        # FTR's per-test errors, indexed for the blank-clip post-mortem below. Harvested
        # from the SAME response as the clips — no extra call, no schema change.
        ftr_errors_by_pair, ftr_errors_by_name = _ftr_errors_by_scenario(result)
        diag_by_clip: dict[str, _ClipDiag] = {}
        for sc in scenario_clips:
            clip_key = str(sc.get("clip_key") or "")
            if clip_key not in valid_clip_tags:
                # Aux clips (e.g. the auth `setup` project's video) no narration
                # line plays over — skip rather than mux a silent shot.
                continue
            recorded_tags.add(clip_key)
            feature = str(sc.get("feature") or "")
            scenario = str(sc.get("scenario") or "")
            diag_by_clip[clip_key] = _ClipDiag(
                feature=feature,
                scenario=scenario,
                ftr_status=str(sc.get("status") or ""),
                # Exact (feature, scenario) join first; the name-only map is the
                # Scenario-Outline fallback (per-example titles miss the pair).
                ftr_error=(
                    ftr_errors_by_pair.get((feature, scenario))
                    or ftr_errors_by_name.get(scenario)
                    or ""
                ),
                duration_ms=_as_float(sc.get("duration_ms")),
                width=_as_int(sc.get("width")),
                height=_as_int(sc.get("height")),
                size_bytes=_as_int(sc.get("size_bytes")),
            )
            try:
                video_bytes = await self._client.get_clip_video(job_id, clip_key)
                clip_format = _derive_clip_format(clip_key, video_bytes)
                video_url = await store_capture_clip(
                    run_id=run_id,
                    stage_name=self.name,
                    clip_id=clip_key,
                    video_bytes=video_bytes,
                    video_format=clip_format,
                )
            except Exception as exc:  # noqa: BLE001 -- soft-fail per clip: degrade, don't abort
                log.warning("demo_video_record: clip %s fetch/store failed: %s", clip_key, exc)
                notes.append(f"clip {clip_key} unavailable: {exc}")
                continue
            clips.append(
                DemoVideoClip(
                    id=clip_key,
                    line_id=clip_key,
                    video_url=video_url,
                    seconds=_as_float(sc.get("duration_ms")) / 1000.0,
                    focus_track=cast("list[dict[str, object]] | None", sc.get("focus_track")),
                    cursor_track=cast("list[dict[str, object]] | None", sc.get("cursor_track")),
                    viewport=cast("dict[str, int] | None", sc.get("viewport")),
                    scenario_status=str(sc.get("status") or "") or None,
                )
            )
            # Kept for the idle pass below, which needs the file locally; dropped when
            # the stage returns.
            bytes_by_clip[clip_key] = (video_bytes, clip_format)

        self._require_footage(
            clips,
            settings,
            job_id=job_id,
            referenced_tags=len(valid_clip_tags),
            returned_clips=len(scenario_clips),
            ftr_errors_by_pair=ftr_errors_by_pair,
        )

        # Surface footage gaps: an all-miss summary (if nothing joined) plus a
        # per-line note for each referenced clip tag FTR never recorded.
        notes.extend(_footage_gap_notes(valid_clip_tags, recorded_tags, bool(clips)))

        # Per-clip visual QA (O1 blank/short-render + O1b wrong-screen OCR): flag a
        # clip that recorded at the right length but shows the wrong pixels. Best-effort
        # per clip; a failure adds a note → the run reports `degraded` (soft). Under
        # strict mode ONLY quality misses raise — OCR wrong-screen findings always stay
        # soft (a tesseract-less analyzer would flag them spuriously).
        qa_by_clip = await self._qa_clips(clips, settings, expected_by_tag)
        notes.extend(self._qa_notes(qa_by_clip, diag_by_clip, settings))
        _stamp_motion_timelines(clips, qa_by_clip)
        _stamp_durations(clips, qa_by_clip)
        # A failed scenario's clip ends in the wait that failed it. Cut it HERE — after
        # the duration repair (a 0s FTR duration has nothing to trim until then) and
        # before anything reads a length (yield, short-clip floor, and every downstream
        # stage that pins a beat to `seconds`).
        degradations = _trim_failed_scenario_tails(clips, diag_by_clip)
        # Shorten frozen stretches to a short hold — AFTER the tail trim (whose
        # arithmetic subtracts the whole timeout from ``seconds``; a tail already
        # shortened here would make it cut real footage) and BEFORE anything reads a
        # length, so reconcile_timeline pins beats to the footage that will play.
        idle_removed = await self._idle_pass(clips, bytes_by_clip, gen, ctx)
        bytes_by_clip.clear()
        # Durations are only trustworthy AFTER `_stamp_durations` repairs FTR's
        # duration_ms=0 race, so both the yield measurement and the per-clip floor
        # read the clips here and not in the harvest loop above.
        _log_scenario_yield(clips)
        notes.extend(_short_clip_notes(clips, settings, ctx))
        # Must run AFTER the motion curves are stamped — the guard reads them off the
        # clips. NOTE-only (⇒ `degraded`) in every case since #840; the hard-fail for a
        # broken recording is `_ftr_verdict_notes` above, which reads FTR's own verdict.
        inert_notes = _inert_capture_notes(clips)
        notes.extend(inert_notes)

        # THE VERDICT, DECIDED — now that the capture can speak for itself.
        #
        # FTR said no scenario passed. That is either "the app never worked" or
        # "every scenario was demoted over a soft-failed decorative assert while
        # recording fine". The footage distinguishes them, and only now exists:
        #
        #   nothing captured  -> HARD FAIL. Both signals agree the run is worthless.
        #   real interaction  -> DEGRADE. The recording is usable and is KEPT; the
        #                        failing assertions ride in `notes`, which
        #                        runs/_shared.py turns into a `degraded` run. A
        #                        usable demo is not thrown away over a heading name,
        #                        and the operator can still see exactly what missed.
        notes.extend(_decide_no_scenario_passed(no_scenario_passed, clips))

        return DemoVideoRecordOutput(
            clips=clips,
            notes=notes,
            degradations=degradations,
            idle_removed_seconds=idle_removed,
        )

    async def _idle_pass(
        self,
        clips: list[DemoVideoClip],
        bytes_by_clip: Mapping[str, tuple[bytes, str]],
        gen: Mapping[str, Any],
        ctx: RunContext,
    ) -> float:
        """Run :func:`_trim_idle_footage` unless ``record.idle_trim: false``."""
        if not _idle_trim_enabled(ctx):
            return 0.0
        ordered_target = _ordered_target_seconds(ctx)
        return await _trim_idle_footage(
            clips,
            bytes_by_clip,
            lines=gen.get("lines") or [],
            wpm=_narration_wpm(gen),
            ordered_target=ordered_target,
            run_id=str(ctx.run_id),
            stage_name=self.name,
            min_clip_seconds=_short_clip_floor(
                get_settings().pipeline, ordered_target, len(_measured_seconds(clips))
            ),
        )

    @staticmethod
    def _require_footage(
        clips: list[DemoVideoClip],
        settings: PipelineSettings,
        *,
        job_id: str,
        referenced_tags: int,
        returned_clips: int,
        ftr_errors_by_pair: dict[tuple[str, str], str],
    ) -> None:
        """Hard-fail a TOTAL footage loss on a PAID source.

        What "hard-fail" buys and what it does NOT: the raise lands the run at
        ``status=failed`` so a footage-free video is never shipped as a finished
        deliverable. It does NOT make the customer whole — there is no refund path
        (``reconcile_timeline.py:1006``: ``grep -rn refund`` finds docstrings only,
        ``run_usage_reconciler`` mirrors usage read-only), so the capture already paid
        for is forfeited on top of the failure.

        Being called at all means a footage SOURCE was present: the only
        degrade-to-empty branch (no ``package_url`` AND no ``bdd_job_id``) returns
        before the harvest, so any run reaching it asked for — and is charged for —
        a capture.

        frontend_test_runner returns ``status="completed"`` with an EMPTY
        ``scenario_clips`` when its Playwright run times out and salvages nothing, so
        ``wait_for_execution`` succeeds and we harvest zero clips. Without this gate
        compile renders narrated text cards and the run lands ``completed``+``degraded``
        — billable for no footage. A PARTIAL harvest is NOT a total loss and stays SOFT
        (``_footage_gap_notes``).

        This is the ZERO-CLIP sibling of the blank-clip gate in ``_qa_notes``: that one
        rejects footage FTR *did* return but which recorded nothing visible; this one
        catches FTR returning no footage at all, where there is no clip to hang a
        ``_ClipDiag`` on and so the blank gate never fires. The two are complementary,
        never both applicable to the same clip.

        Blast radius is identical to the blank-clip hard-fail: record is a CONCURRENT
        sibling of the claim branch, so any raise here re-raises out of the DAG round and
        can strand the independent ``claim_finish`` branch as ``pending``. That failure
        class already exists (this introduces NO new stranding class) and orch #696
        sweeps stranded pending/running stage rows to ``cancelled``. DAG behaviour is
        deliberately unchanged.
        """
        # Fire ONLY when FTR itself returned nothing. `clips` can be empty while
        # `returned_clips > 0` for reasons that are NOT a capture failure and that a
        # hard-fail makes unrecoverable:
        #   * a tag-naming mismatch — FTR recorded real footage under keys no
        #     narration line references, so every clip is filtered out by
        #     `valid_clip_tags`. That is a gen/bddgen mapping defect; the footage
        #     exists and the run should still degrade to text cards.
        #   * a storage outage — each clip soft-fails in the download/store loop and
        #     appends a note. Raising there would blame the recorder for a MinIO fault.
        # Both keep today's degrade. Only "the recorder produced nothing at all" is
        # the EXPECTED footage loss this gate exists to fail the run on.
        if returned_clips or clips or not settings.demo_require_footage:
            return
        # FTR's own reason, joined from the SAME response (#719's index). Without it the
        # message carries only a count, and the per-clip `notes` cannot help: reaching
        # here means the harvest loop never ran, so `notes` is provably still empty.
        summary = _footage_loss_summary(ftr_errors_by_pair)
        log.error(
            "demo_video_record: FTR job %s returned no scenario clips — %s",
            job_id,
            summary or "<no FTR test_results error>",
        )
        raise RuntimeError(
            f"demo_video_record: frontend_test_runner job {job_id!r} recorded no "
            f"usable footage — 0 of {referenced_tags} referenced clip tag(s) "
            f"harvested from {returned_clips} scenario clip(s) returned. The "
            "brief supplied a footage source, so this is an EXPECTED footage loss, "
            "not a degrade: the run lands status=failed and the work already done is "
            "still billed (there is no automatic refund). Set "
            "PIPELINE_DEMO_REQUIRE_FOOTAGE=false to fall back to narrated text cards."
            + (f" FTR reported: {summary}" if summary else "")
        )

    def _qa_notes(
        self,
        qa_by_clip: dict[str, _ClipQa],
        diag_by_clip: dict[str, _ClipDiag],
        settings: PipelineSettings,
    ) -> list[str]:
        """The soft notes from a QA sweep — raising instead when a clip is BLANK
        (always, independent of strict) or when strict mode is on and a
        *strict-eligible* (i.e. non-OCR) criterion failed.

        The blank hard-fail is unconditional because a genuinely blank recording (the
        app never loaded, mean_sharpness below the fail threshold) is not a deliverable
        at any quality bar — shipping it `completed`+degraded charges the customer for a
        worthless video. This lands the run at `status=failed`; no refund follows — see
        `reconcile_timeline.py:1006` — so the spend on the stages that already ran is
        forfeited, which is the deliberate trade against shipping a blank video.
        `demo_qa_fail_sharpness=0.0` disables it (degrade-only)."""
        failures = [qa.failure for qa in qa_by_clip.values() if qa.failure]
        # Each blank note carries FTR's own post-mortem in `[...]` so the raised message —
        # which the DAG executor persists into `StageOutput.error` — names the UPSTREAM
        # cause (e.g. the bdd nav hook never seeing a paint) instead of only
        # `min_sharpness_score=0.0` with no reason.
        blank_notes: list[str] = []
        for clip_id, qa in qa_by_clip.items():
            if not qa.blank:
                continue
            base = qa.failure[0] if qa.failure else f"clip {clip_id} is blank"
            diag_text = _blank_diag_text(clip_id, diag_by_clip.get(clip_id, _ClipDiag()))
            # Logged too: the raise is only durable if the stage's DB write lands.
            log.error("demo_video_record: blank clip %s — %s", clip_id, diag_text)
            blank_notes.append(f"{base} [{diag_text}]")
        if blank_notes:
            raise RuntimeError(
                "demo_video_record: blank clip(s) rejected: " + "; ".join(blank_notes)
            )
        # The blank gate is only meaningful for clips it actually measured. An analyzer
        # outage returns `_ClipQa()` for every clip, whose `blank=False` reads identically
        # to "measured and fine" — so the gate quietly did not run and the loop above found
        # nothing to reject. That stays a DEGRADE and not a failure, deliberately (the
        # analyzer must not be able to fail a run), but it must not be SILENT: without a
        # note the run renders as a clean `completed`, which is exactly how a 100% black
        # video ships as a finished deliverable.
        unmeasured = sorted(cid for cid, qa in qa_by_clip.items() if not qa.blank_gate_ran)
        if unmeasured:
            log.error(
                "demo_video_record: blank gate did not run for %d/%d clip(s): %s",
                len(unmeasured),
                len(qa_by_clip),
                ", ".join(unmeasured),
            )
        strict_notes = [note for note, strict_eligible in failures if strict_eligible]
        if settings.demo_qa_strict and strict_notes:
            raise RuntimeError(
                "demo_video_record: strict visual QA failed: " + "; ".join(strict_notes)
            )
        notes = [note for note, _ in failures]
        if unmeasured:
            notes.append(
                f"visual QA unavailable for {len(unmeasured)}/{len(qa_by_clip)} clip(s) "
                f"({', '.join(unmeasured)}) — the blank-recording check did not run"
            )
        return notes
