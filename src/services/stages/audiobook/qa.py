"""audiobook_qa — listen-back QA over the rendered narration (PR-7).

An OPT-IN, SOFT-gate stage that closes the loop on audiobook narration the same
way ``shared/footage_critique`` closes it on directed footage: it *listens back*
to what was actually rendered and judges it against the source manuscript, so a
mispronunciation / wrong-word / dropped-line that the deterministic synth path
cannot see gets caught. Structurally it is a near-clone of ``footage_critique``,
with the vision pipeline swapped for audio:

* VLM shot-critique      → ASR transcription of each narrated chunk + a WM judge
  scoring the transcript against that chunk's intended manuscript text;
* footage re-render      → chunk re-narrate (re-synth of the flagged chunk_ids);
* ``_nudge_plans``       → apply the judge's suggested pronunciation to the
  flagged chunk's synth input.

The loop is driven by the shared ``loop_engine`` and is bounded to ≤2 passes; it
ALWAYS accepts the current best and NEVER hard-blocks the ship. An ASR outage, a
worker error, or a failed re-roll all degrade to *accept the current narration*.
A genuine unresolved mispronunciation surfaces as a ``degraded`` run (a ``notes``
line naming the offending chunk_ids), never as a failure — the run-read layer
(`api/routes/runs/_shared.py`) flags a ``completed`` stage carrying non-empty
``notes`` as ``degraded``.

⚠️ SOFT IS NOT SILENT — the COVERAGE contract. Degrading must never turn into
reporting a clean pass. Every chunk without a judgment (skipped judge batch,
exhausted budget, worker error mid-loop, missing transcript) lands in
``qa.unjudged_chunks``; any such chunk forces ``converged=False``, a distinct
``qa.reason`` (``partial_judge_coverage`` / ``no_judge_coverage``) and a degrade
note naming the count, and with ZERO judgments ``qa.assessed`` is False and
``qa.score`` is pinned to ``_SCORE_MIN`` ("not assessed"). Defaulting an empty
record set to ``_SCORE_MAX`` made a fully-skipped loop converge on ``threshold``
with ``judgments=[] flagged_chunks=[] notes=[]`` — a "no issues found" verdict on
a PAID stage that judged nothing, strictly worse than a visible ``loop_error``.

Opt-in / byte-identity: with no ``brief.audiobook.qa`` block the stage raises
``StageSkippedError`` before issuing a single ASR or WM call, so a run that did
not ask for QA is byte-identical to one that never had this stage (acceptance #4).

DETECTION by default; SURGICAL RE-ROLL is opt-in (``brief.audiobook.qa.reroll``).
With re-roll OFF the stage only DETECTS: it records the judge's suggested
pronunciation per flagged chunk (``reroll_requests``) + a degraded note, the
``optimizer_fn`` is a no-op, and the loop runs exactly ONE scoring pass
(``_MAX_PASSES == 1``) — a second pass would only re-judge byte-identical audio,
doubling WM spend and letting a retained pass-1 draft report a pass-2 record set.

With re-roll ON the ``optimizer_fn`` APPLIES the judge's respelling to each flagged
chunk's text (via ``apply_lexicon`` — the same plain-text mechanism the ingest
pronunciation lexicon uses, NOT ``<phoneme>`` SSML, because the judge emits a
hyphenated respelling that kokoro's IPA-only phoneme path would mis-speak) and
re-synthesizes ONLY that chunk with the SAME ``(voice, speed)``. The corrected text
yields a new ``deterministic_clip_id`` → genuinely different audio, so a second
scoring pass (``_MAX_PASSES_REROLL == 2``) re-transcribes + re-judges the corrected
clip. Retained corrected clips are emitted as ``corrected_clips`` and preferred by
``audiobook_stitch`` (the DAG runs narrate → qa → stitch). SOFT throughout: any
re-synth failure keeps the original clip; the chunk-id spine is invariant, so the
loop engine's ``id_extractor`` (``_chunk_ids``) guard holds either way.

⚠️ EFFICACY of the re-roll (does the respelling actually correct the render and does
live ASR re-score it above threshold) is NOT provable offline — it needs live kokoro
+ ASR with a seeded mispronunciation (post-promote E2E). This module's plumbing is
what the offline tests cover.

⚠️ Whole-book ASR is transcribed ONCE up front (``_transcribe_all`` warms
``transcript_cache`` before the loop), NOT inside the critic. Transcribing an
entire book inside the loop's per-critic ``timeout_seconds`` budget blew it on any
real-length book → SOFT ``loop_error`` → the stage silently no-opped in prod while
passing on tiny fixtures. The timed critic now pays only for the fast WM judge.

LLM calls route through ``worker_management`` (the ``WMClient`` gateway) — never a
provider directly; the model behind the ``audiobook-listenback-judge`` worker is WM
config, and WM auto-creates the worker on first call (no worker_management change).
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import batched
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

import httpx
from pydantic import Field
from w2v_common.stages import StageRetriesExhaustedError, StageTimeoutError, stage_call

from src.clients.gpu_busy import GpuBusyError
from src.clients.transcription_worker_client import TranscriptionWorkerClient
from src.clients.tts_client import TTSClient
from src.services.llm import WMClient, WMResponse
from src.services.stages.audiobook.pronunciation import apply_lexicon
from src.services.stages.base import (
    SlotUse,
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
    StagePreconditionError,
    StageSkippedError,
)
from src.services.stages.llm_budget import (
    MEDIUM_ANSWER_TOKENS,
    attempt_plan,
    downshift_reasoning,
    llm_budget_seconds,
    llm_client_timeout_seconds,
    llm_retries,
    output_token_budget,
)
from src.services.stages.primitives.narration import synthesize_and_store
from src.services.stages.shared.loop_engine import Critique, run_evaluator_optimizer
from src.services.stages.upstream import (
    PermanentUpstreamError,
    TransientUpstreamError,
    guarded,
)
from src.utils import describe_exception

if TYPE_CHECKING:
    from src.clients.schemas.transcription_worker import TranscriptionResult
    from src.services.runs.run_context import RunContext
    from src.services.storage_service import StorageService

log = logging.getLogger(__name__)

# WM worker name (regex ^[a-zA-Z0-9_-]+$). WM auto-creates it on first call; its
# model/provider is WM-side config — that is the whole point of routing through WM.
_WORKER_NAME = "audiobook-listenback-judge"

# Per-chunk match quality lives on a 1-5 scale (higher = the render matches the
# manuscript better), exactly like footage_critique's per-shot score. A chunk
# below ``_THRESHOLD`` is flagged (mispronounced / wrong-word / dropped) and gets a
# re-roll pass; when the WORST chunk clears the bar the loop converges.
_SCORE_MIN = 1.0
_SCORE_MAX = 5.0
_THRESHOLD = 4.0
_EPSILON = 0.25
# DETECTION-ONLY default (re-roll opt-out): the optimizer is a no-op — a re-synth
# with unchanged (text, voice, speed) is content-addressed to the SAME clip — so a
# SECOND scoring pass would only re-judge identical audio, doubling WM judge spend
# (bug #3) and letting a retained pass-1 draft report a pass-2 record set (bug #2).
# Run exactly ONE scoring pass when re-roll is off.
_MAX_PASSES = 1
# Surgical re-roll (opt-in ``brief.audiobook.qa.reroll``): the optimizer applies the
# judge's suggested respelling to a flagged chunk's text and re-synthesizes it as
# PLAIN TEXT (a new deterministic clip id → genuinely different audio), so a second
# scoring pass re-judges the CORRECTED audio. Two passes: score → re-roll → re-score.
_MAX_PASSES_REROLL = 2

# Cap the judge's ``judgments`` output array per WM call so the single small vLLM
# never sees an unbounded schema (it times out and silently returns a partial /
# empty verdict). Sized like footage_critique's ``_CRITIC_BATCH_SIZE``.
_JUDGE_BATCH_SIZE = 5

# ``_OUTPUT_SCHEMA`` answers ONE judge batch: at most ``_JUDGE_BATCH_SIZE`` (5)
# judgment objects, each an ok/score verdict PLUS a free-text ``issue`` bounded
# at ``_MAX_ISSUE_CHARS`` (300) and an optional pronunciation object. A lone
# verdict would be SMALL; five of them with prose attached is MEDIUM.
_JUDGE_MAX_TOKENS = output_token_budget(answer_tokens=MEDIUM_ANSWER_TOKENS)

# Floor for the per-batch judge budget (bug #2). The judge loop is SEQUENTIAL, so
# without a per-batch bound a 100-chunk book issues ~20 sequential WM calls each
# individually allowed the FULL outer budget — one slow batch eats the whole
# critic budget, the outer timeout fires, and the stage degrades to `loop_error`
# with ZERO judgments recorded (a silent no-op). Each batch now gets
# ``judge_phase / n_batches``, floored here so a very long book never hands a batch
# a budget too small for a real WM round-trip. A batch that blows its slice is
# SKIPPED, not fatal: the remaining batches still judge, so partial judgments
# survive.
#
# ⚠️ The floor makes the NOMINAL share overrun by construction:
# ``n * max(floor, phase/n) >= phase`` always, and STRICTLY exceeds it once
# ``phase/n < floor`` (a 100-batch book: 100 x 30s = 3000s against a ~600s phase).
# The nominal share is therefore NOT the bound — :meth:`_next_batch_budget` clamps
# every slice to the time actually remaining against a monotonic deadline, so the
# slices can never sum past the phase budget and a batch that cannot possibly
# finish is never STARTED (its chunks are recorded as unjudged instead).
_MIN_JUDGE_BATCH_SECONDS = 30.0

# The critic's own budget (``llm_budget_seconds``) has to cover THREE things, not
# just the judge calls: the in-critic ``_transcribe_all`` re-run (only the clips
# that failed the warm-up are retried — successes are cached — but a hung ASR box
# is exactly the case where those retries hang), the sequential judge loop, and
# message building / validation. Splitting it explicitly keeps a slow ASR box from
# consuming the whole critic budget before a single WM call is issued (bug #3),
# and leaves headroom so the judge loop finishes INSIDE the outer wait rather than
# being cancelled by it (which would discard the pass).
_CRITIC_ASR_SHARE = 0.2
_JUDGE_PHASE_SHARE = 0.7

# Whole-book ASR budget (bug #3). ``_transcribe_all`` is hoisted out of the
# per-critic timeout and was therefore bounded ONLY by the audiobook blueprint's
# ``default_timeout_seconds=3600`` — which fires as a HARD ``StageTimeoutError``
# from the stage middleware, OUTSIDE this stage's soft-degrade guard, killing a
# paid stage on a hung ASR box. Half the stage budget leaves ample room for the
# judge loop afterwards and keeps a hang INSIDE the soft gate (accept-current).
_ASR_BUDGET_SECONDS = 1800.0

# String hardening for the model's free-text verdict fields.
_MAX_ISSUE_CHARS = 300
_MAX_WORD_CHARS = 120
# How much manuscript / transcript to show the judge per chunk. A chunk is
# long-form (~2000 chars); the head is enough to spot a wrong word / dropped line
# without blowing the worker token budget.
_MAX_TEXT_CHARS = 700

# Where audiobook_narrate persisted each clip, and the on-disk format. The
# non-gateway narrate path stores ``{clip_id}.wav`` under this stage via
# ``store_tts_clip`` (default ``wav``); we retrieve by that key rather than the
# presigned ``artifact_url`` so there is NO SSRF surface. (The gateway narrate
# path stores the clip on the TTS gateway, not in orchestrator storage, so a
# gateway-rendered clip simply reads back as "no bytes" and is skipped — SOFT.)
_NARRATE_STAGE = "audiobook_narrate"
_CLIP_SUFFIX = ".wav"

# Bounded concurrency for the listen-back transcription fan-out (mirrors
# word_timed_captions' _MAX_CONCURRENCY): the ASR box serialises anyway.
_ASR_CONCURRENCY = 4


_SYSTEM_PROMPT = (
    "You are a listen-back QA proofer for an audiobook. For each narration chunk you "
    "receive its intended MANUSCRIPT text and the ASR TRANSCRIPT of the audio that was "
    "actually rendered for it. Judge whether the rendered audio faithfully speaks the "
    "manuscript: catch mispronunciations, wrong or substituted words, and dropped / "
    "skipped text. Minor ASR noise (punctuation, casing, obvious homophones the "
    "transcriber mis-spelled) is NOT a defect — be conservative, flag only genuine "
    "narration errors. For EACH chunk_id return: ok (true when the render is faithful), "
    "a short issue describing the problem when not ok, severity_score (0 = perfect, "
    "higher = worse), and — only for a mispronunciation you can fix — "
    "suggested_pronunciation as {word, ipa_or_phonetic} (else null). Return one entry "
    "per chunk_id you were given."
)

# Object-rooted structured-output schema (WM convention): a required ``judgments``
# array, one per-chunk verdict. WM returns ``response.output`` parsed to this shape.
_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["judgments"],
    "properties": {
        "judgments": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["chunk_id", "ok"],
                "properties": {
                    "chunk_id": {"type": "string"},
                    "ok": {"type": "boolean"},
                    "issue": {"type": "string"},
                    "severity_score": {"type": "number"},
                    "suggested_pronunciation": {
                        "type": ["object", "null"],
                        "properties": {
                            "word": {"type": "string"},
                            "ipa_or_phonetic": {"type": "string"},
                        },
                    },
                },
            },
        }
    },
}


@dataclass(frozen=True)
class _QAClip:
    """One narrated chunk under QA: how to fetch its audio + its intended text.

    ``voice``/``speed`` are the synth provenance (from ``NarratedChunk``) the re-roll
    needs to reproduce the clip with a corrected text; ``speed is None`` marks a clip
    with NO recoverable provenance (a pre-provenance narrate output or a gateway
    clip) — the re-roll skips those. ``manuscript_text`` is the ORIGINAL intended
    text and is NEVER rewritten (the re-roll changes only the transient synth input),
    so the judge keeps comparing the transcript against what the book actually says.
    """

    chunk_id: str
    clip_id: str  # NarratedChunk.id — the storage key stem ({clip_id}.wav)
    artifact_url: str
    manuscript_text: str  # the intended text from the plan/segment spine
    voice: str | None = None
    speed: float | None = None
    seconds: float = 0.0


@dataclass(frozen=True)
class _QADraft:
    """One point in the loop: the set of narrated clips being judged.

    The re-roll (optimizer) may rewrite a flagged clip's audio but must NEVER add,
    drop, or reorder chunks — the ``id_extractor`` (:func:`_chunk_ids`) guards that
    C0 spine invariant inside the loop engine.
    """

    clips: tuple[_QAClip, ...]


async def _noop_optimizer(draft: _QADraft, _critique: Critique) -> _QADraft:
    """Detection-only optimizer seam (re-roll deferred).

    There is nothing to optimize: a re-synth with unchanged ``(text, voice, speed)``
    is content-addressed to the SAME clip. With ``_MAX_PASSES == 1`` the loop never
    invokes this; it is the seam the surgical re-roll fills in (apply the judge's
    ``suggested_pronunciation`` via a ``<phoneme>`` SSML span, then re-synth ONLY the
    flagged chunk). Returning the same draft keeps the chunk-id spine identical for
    the engine's ``id_extractor`` guard."""
    return draft


def _chunk_ids(draft: _QADraft) -> frozenset[str]:
    """The draft's chunk-id spine — the loop-engine ``id_extractor`` guard key."""
    return frozenset(c.chunk_id for c in draft.clips)


# The loop engine's own stop reasons. Only these are overridden by a coverage
# reason; a diagnostic reason the stage already chose (``loop_error``,
# ``asr_timeout``, …) is more specific and must survive.
_LOOP_REASONS = frozenset({"threshold", "plateau", "max_passes", "no_critic"})

# Cap the unjudged-id list rendered into a degrade note: a 500-chunk book must not
# emit a 10 KB note. The COUNT is the load-bearing part; the ids are a sample.
_MAX_NOTE_IDS = 20


def _unjudged_chunks(clips: Sequence[_QAClip], records: Sequence[ChunkJudgment]) -> list[str]:
    """Chunks in the draft that carry NO judgment — the coverage hole.

    Derived, never tracked separately, so it cannot drift from the records that
    were actually produced (a skipped batch, an aborted loop, and a clip with no
    transcript all land here identically)."""
    judged = {r.chunk_id for r in records}
    return [c.chunk_id for c in clips if c.chunk_id not in judged]


def _preview(ids: Sequence[str]) -> str:
    """Bounded ', '-joined id sample for a degrade note."""
    if len(ids) <= _MAX_NOTE_IDS:
        return ", ".join(ids)
    return f"{', '.join(ids[:_MAX_NOTE_IDS])} … (+{len(ids) - _MAX_NOTE_IDS} more)"


def _corrected_clips(
    original: Sequence[_QAClip], final: Sequence[_QAClip]
) -> list[dict[str, object]]:
    """The clips the re-roll re-synthesized and the loop RETAINED — one per chunk_id
    whose ``clip_id`` changed between the initial and the winning draft.

    Emitted in ``NarratedChunk``-compatible shape so ``audiobook_stitch`` can prefer
    them over the narrate clip of the same chunk_id. Empty when no clip changed (the
    detection-only path, a clean run, or a re-roll the loop did not keep)."""
    orig_by_chunk = {c.chunk_id: c.clip_id for c in original}
    return [
        {
            "chunk_id": c.chunk_id,
            "id": c.clip_id,
            "artifact_url": c.artifact_url,
            "seconds": c.seconds,
        }
        for c in final
        if orig_by_chunk.get(c.chunk_id) not in (None, c.clip_id)
    ]


class ChunkJudgment(StageOutput):
    """One chunk's validated listen-back verdict (authoritative-out)."""

    chunk_id: str
    ok: bool
    quality_score: float
    issue: str = ""
    severity_score: float = 0.0
    suggested_word: str = ""
    suggested_pronunciation: str = ""


class TokenUsage(StageOutput):
    """Aggregate WM token spend across every judge call in the loop."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0


class AudiobookQARecord(StageOutput):
    """The structured outcome of the loop for the retained (best) narration.

    ``assessed`` / ``unjudged_chunks`` are the COVERAGE contract: a listen-back
    verdict is only meaningful over the chunks that were actually judged. A skipped
    judge batch (budget exhausted), a WM outage mid-loop, or a clip that never got a
    transcript leaves chunks unjudged — and a result with unjudged chunks must never
    read as a clean pass. ``score`` is the worst JUDGED chunk; with zero judgments it
    is pinned to ``_SCORE_MIN`` ("not assessed"), never ``_SCORE_MAX`` ("perfect").
    """

    score: float
    passes: int
    converged: bool
    reason: str
    flagged_chunks: list[str]
    judgments: list[ChunkJudgment]
    assessed: bool = True
    unjudged_chunks: list[str] = Field(default_factory=list)


class AudiobookQAInput(StageInput):
    """No typed inputs — everything is read from ``ctx`` (prior output + config)."""


class AudiobookQAOutput(StageOutputWithNotes):
    """The listen-back verdict + token spend, plus recorded re-roll intents.

    This stage is NOT the ``terminal_artifact_stage`` (``audiobook_stitch`` is and
    stays so): it delivers no artifact of its own. Its job is to SURFACE quality —
    ``notes`` (from ``StageOutputWithNotes``) turn an unresolved mispronunciation
    into a ``degraded`` run without failing it, and ``reroll_requests`` records the
    exact per-chunk fix intent for a later re-roll consumer (PR-8) to act on.
    ``reroll_requests`` is EMPTY on a clean pass.
    """

    qa: AudiobookQARecord
    token_usage: TokenUsage
    reroll_requests: list[dict[str, object]] = Field(default_factory=list)
    # Clips the surgical re-roll actually re-synthesized and RETAINED (the winning
    # draft), one per corrected chunk_id: ``{chunk_id, id, artifact_url, seconds}``.
    # ``audiobook_stitch`` prefers these over the narrate clip of the same chunk_id.
    # EMPTY unless the run opted into ``brief.audiobook.qa.reroll`` AND a re-synth
    # improved a flagged chunk — so a detection-only / clean run is byte-identical.
    corrected_clips: list[dict[str, object]] = Field(default_factory=list)


def _clamp_score(value: object, *, lo: float, hi: float, default: float) -> float:
    """Coerce a model number into ``[lo, hi]``; junk ⇒ ``default``."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        score = float(value)
    except TypeError, ValueError:
        return default
    if score != score:  # NaN
        return default
    return max(lo, min(hi, score))


def _quality(ok: bool, severity: float) -> float:
    """Convergence quality (1-5, higher = better) from the judge's verdict.

    A not-ok chunk pins to the low end; an ok chunk's quality falls off with the
    reported severity so ``flag ⟺ quality < threshold`` stays self-consistent and
    the WORST chunk drives loop convergence (mirrors footage_critique)."""
    if not ok:
        return _SCORE_MIN
    return max(_SCORE_MIN, min(_SCORE_MAX, _SCORE_MAX - severity))


def _usage_add(acc: dict[str, int], usage: Mapping[str, object] | None) -> None:
    """Best-effort accumulate WM ``usage_metrics`` into ``acc`` (never raises)."""
    acc["calls"] = acc.get("calls", 0) + 1
    if not isinstance(usage, Mapping):
        return
    for field_name, keys in (
        ("prompt_tokens", ("prompt_tokens", "input_tokens")),
        ("completion_tokens", ("completion_tokens", "output_tokens")),
        ("total_tokens", ("total_tokens", "tokens")),
    ):
        for key in keys:
            raw = usage.get(key)
            if isinstance(raw, bool):  # bool is an int subclass — not a token count
                continue
            if isinstance(raw, (int, float)):
                acc[field_name] = acc.get(field_name, 0) + int(raw)
                break


def _build_usage(acc: Mapping[str, int]) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=acc.get("prompt_tokens", 0),
        completion_tokens=acc.get("completion_tokens", 0),
        total_tokens=acc.get("total_tokens", 0),
        calls=acc.get("calls", 0),
    )


def _manuscript_by_chunk(source: Mapping[str, object]) -> dict[str, str]:
    """Map every plan/segment chunk id → its intended text.

    Mirrors ``audiobook_narrate`` / ``audiobook_direct``: chapters → chunks, each
    carrying ``id`` + ``text``. narrate keys its clip's ``chunk_id`` on ``chunk['id']``,
    so this is the exact join back to the intended manuscript text."""
    out: dict[str, str] = {}
    chapters = source.get("chapters")
    if not isinstance(chapters, list):
        return out
    for raw_chapter in chapters:
        if not isinstance(raw_chapter, Mapping):
            continue
        chapter = cast("Mapping[str, object]", raw_chapter)
        chunks = chapter.get("chunks")
        if not isinstance(chunks, list):
            continue
        for raw_chunk in chunks:
            if not isinstance(raw_chunk, Mapping):
                continue
            chunk = cast("Mapping[str, object]", raw_chunk)
            chunk_id = str(chunk.get("id") or "")
            if chunk_id:
                out[chunk_id] = str(chunk.get("text") or "")
    return out


class AudiobookQAStage(StageHandler[AudiobookQAInput, AudiobookQAOutput]):
    """Listen-back QA loop for audiobook narration. Advisory model, authoritative
    stage: every judge value is clamped/coerced, and the loop is SOFT — it always
    accepts the current narration and only ever DEGRADES a run, never fails it."""

    name = "audiobook_qa"
    input_model = AudiobookQAInput
    output_model = AudiobookQAOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (
        SlotUse("llm"),
        SlotUse("tts"),
        SlotUse("asr"),
    )
    worker_name: str = _WORKER_NAME

    def __init__(
        self,
        client: WMClient | None = None,
        asr: TranscriptionWorkerClient | None = None,
        storage: StorageService | None = None,
        tts: TTSClient | None = None,
    ) -> None:
        self._llm_timeout = llm_budget_seconds()
        self._llm_retries = llm_retries()
        self._min_judge_batch_seconds = _MIN_JUDGE_BATCH_SECONDS
        self._asr_timeout = _ASR_BUDGET_SECONDS
        self._wm = client or WMClient(timeout=llm_client_timeout_seconds())
        self._asr = asr or TranscriptionWorkerClient()
        # Storage resolves lazily at handle time (the process-wide singleton is
        # async-initialised); an injected fake short-circuits that.
        self._storage = storage
        # Only touched on the opt-in re-roll path; constructing it does no network.
        self._tts = tts or TTSClient()

    def _judge_phase_seconds(self) -> float:
        """The slice of the critic budget the sequential judge loop may spend.

        Deliberately < the full critic budget: the in-critic ``_transcribe_all``
        re-run and the final validation have to fit in the same outer wait, and a
        judge loop that runs right up to the outer deadline gets CANCELLED — which
        used to discard the whole pass."""
        return self._llm_timeout * _JUDGE_PHASE_SHARE

    def _judge_batch_budget(self, batch_count: int) -> float:
        """NOMINAL per-batch slice of the judge phase (see ``_MIN_JUDGE_BATCH_SECONDS``).

        ``phase / n``, floored at ``_MIN_JUDGE_BATCH_SECONDS`` because a slice too
        small for a real WM round-trip would just fail every batch. The floor means
        ``n * share`` can EXCEED the phase budget — this value is a per-call cap, not
        the loop's bound. :meth:`_next_batch_budget` supplies the actual bound."""
        phase = self._judge_phase_seconds()
        if batch_count <= 1:
            return phase
        return max(self._min_judge_batch_seconds, phase / batch_count)

    def _next_batch_budget(self, share: float, remaining: float) -> float | None:
        """The budget for the next judge batch, or ``None`` when time is up.

        This is the deadline clamp that makes the per-batch slices sum to at most the
        judge phase however hard the ``_MIN_JUDGE_BATCH_SECONDS`` floor overshoots the
        nominal share. ``None`` means STOP the loop cleanly (the remaining chunks are
        recorded as unjudged) rather than starting a batch that cannot finish."""
        if remaining <= 0.0:
            return None
        return min(share, remaining)

    async def _judge_batch_call(
        self,
        batch: Sequence[_QAClip],
        transcripts: Mapping[str, str],
        *,
        enable_reasoning: bool | None = None,
    ) -> WMResponse:
        """ONE judge call, routed through the ONE upstream policy.

        ``guarded`` (``stages/upstream.py``) turns an upstream HTTP/transport failure
        into ``PermanentUpstreamError`` / ``TransientUpstreamError`` BEFORE the caller's
        ``stage_call`` sees it, which is what lets the batch loop above spend retries on
        a 5xx blip / 429 / 422-bad-draw and refuse to spend any on a 402.

        A method rather than a closure so the caller can bind ``batch`` explicitly —
        a lambda that closed over the loop variable would be a late-binding bug
        (ruff B023). ``enable_reasoning`` is threaded per ATTEMPT by
        ``downshift_reasoning``: this batch's retry stops asking the model to think,
        because an identical re-issue was measured to fail identically.
        ``_build_message`` stays INSIDE, so a bug there raises where it always did.
        """
        return await guarded(
            lambda: self._wm.message(
                enable_reasoning=enable_reasoning,
                worker_name=self.worker_name,
                message=self._build_message(batch, transcripts),
                system_prompt=_SYSTEM_PROMPT,
                output_schema=_OUTPUT_SCHEMA,
                max_tokens=_JUDGE_MAX_TOKENS,
            )
        )

    async def _get_storage(self) -> StorageService:
        if self._storage is not None:
            return self._storage
        from src.services.tts_storage import _get_storage

        # _get_storage() is typed to the abstract _StorageLike seam; the live
        # singleton is always the concrete StorageService (carries
        # retrieve_stage_artifact). Same cast the caption path uses. The cast
        # target is the TYPE_CHECKING import (a string forward-ref, never
        # evaluated at runtime), so no runtime StorageService import is needed.
        return cast("StorageService", await _get_storage())

    async def handle(  # noqa: C901 — the critic/optimizer closures share the loop's soft-degrade state (usage, records, cache); splitting hurts clarity
        self,
        payload: AudiobookQAInput,
        ctx: RunContext,
    ) -> AudiobookQAOutput:
        # Opt-in + skip-clean (acceptance #4): the QA sub-flag lives under the
        # audiobook brief block (`brief.audiobook.qa`). Absent/false ⇒
        # StageSkippedError before any ASR/WM traffic, so a run that did not ask
        # for QA is byte-identical to one that never had this stage. Mirrors
        # audiobook_direct's skip-clean on absent `brief.audiobook.direction`.
        brief = ctx.config_snapshot.get("brief") or {}
        audiobook_cfg = brief.get("audiobook") or {}
        qa_cfg = audiobook_cfg.get("qa")
        if not qa_cfg:
            raise StageSkippedError(
                f"{self.name}: no brief.audiobook.qa (opt-in stage)",
                missing=["brief.audiobook.qa"],
            )
        # Surgical re-roll (opt-in WITHIN the already-opt-in qa block). Absent ⇒
        # detection-only: the optimizer is a no-op and the loop runs a single pass,
        # byte-identical to before. On ⇒ the optimizer re-synthesizes flagged chunks
        # with the judge's respelling and the loop runs a second, re-scoring pass.
        reroll_enabled = isinstance(qa_cfg, Mapping) and bool(qa_cfg.get("reroll"))

        # Same source precedence narrate/stitch use: span-mode segment, else plan.
        source = ctx.prior_output("audiobook_segment")
        if not source or not source.get("chapters"):
            source = ctx.prior_output("audiobook_plan")
        if not source or not source.get("chapters"):
            raise StagePreconditionError(f"{self.name}: no plan/segment output")
        narration = ctx.prior_output("audiobook_narrate")
        if not narration or not narration.get("clips"):
            raise StagePreconditionError(f"{self.name}: audiobook_narrate produced no clips")

        manuscript = _manuscript_by_chunk(source)
        clips: list[_QAClip] = []
        for raw in narration["clips"]:
            if not isinstance(raw, Mapping):
                continue
            chunk_id = str(raw.get("chunk_id") or "")
            clip_id = str(raw.get("id") or "")
            if not chunk_id or not clip_id:
                continue
            raw_speed = raw.get("speed")
            raw_voice = raw.get("voice")
            clips.append(
                _QAClip(
                    chunk_id=chunk_id,
                    clip_id=clip_id,
                    artifact_url=str(raw.get("artifact_url") or ""),
                    manuscript_text=manuscript.get(chunk_id, ""),
                    # Re-roll synth provenance. ``speed is None`` (a pre-provenance
                    # narrate output or a gateway clip) marks a clip the re-roll
                    # cannot reproduce — it is skipped there.
                    voice=str(raw_voice) if raw_voice is not None else None,
                    speed=float(raw_speed) if isinstance(raw_speed, (int, float)) else None,
                    seconds=float(raw.get("seconds") or 0.0),
                )
            )

        run_id = str(ctx.run_id)
        usage: dict[str, int] = {}
        # Retained per-draft verdicts, keyed by the draft's OBJECT IDENTITY (not its
        # chunk-id spine, which is invariant across a re-roll). The initial draft and
        # a re-rolled candidate share the same spine but must NOT share a record set:
        # a discarded/tied candidate's pass-2 judgments would otherwise leak into the
        # retained (initial) draft's report — a false-clean on a bad shipped clip. A
        # retried critic call re-uses the SAME draft object, so id-keying still
        # accumulates across retries (the #735 durability property). The re-roll
        # optimizer preseeds the candidate's list with the non-re-rolled records so
        # pass 2 re-judges ONLY the corrected chunks.
        records_by_key: dict[int, list[ChunkJudgment]] = {}
        # WHY a ledger and not just the log: an upstream fault that stops the judge
        # loop shows up in the output ONLY as unjudged chunks, which is the same
        # shape as "no transcript" or "budget exhausted". The coverage note therefore
        # says "judge budget exhausted, worker error, or no transcript" — three very
        # different operator actions behind one sentence. These entries name WHICH
        # one it was, permanent-vs-transient included, and are emitted as stage notes
        # (⇒ a ``degraded`` run) alongside the coverage note.
        judge_faults: list[str] = []
        # ASR is deterministic per clip; memoise across passes so a second scoring
        # round (when the optimizer plateaus) never re-transcribes the same bytes.
        transcript_cache: dict[str, str] = {}

        initial = _QADraft(clips=tuple(clips))

        if not clips:
            # Nothing narrated to listen back to (e.g. every clip came from the
            # gateway path with no orchestrator-stored bytes) ⇒ accept current.
            return self._accept_current(usage, reason="no_narrated_clips", note="")

        async def critic_fn(draft: _QADraft) -> Critique:
            # DURABILITY (bug: partial judgments were lost on an outer cancel).
            # ``records`` is published into ``records_by_key`` BEFORE the first await
            # and then MUTATED in place, so an outer cancellation (the loop engine's
            # per-attempt ``stage_call`` timeout) unwinding this coroutine still
            # leaves every batch judged so far visible to ``handle``. ``setdefault``
            # (not assignment) so a retried attempt ACCUMULATES onto the previous
            # attempt's work instead of throwing it away. Keyed by object identity so
            # the initial and a re-rolled candidate keep SEPARATE record sets (see the
            # ``records_by_key`` declaration) — a retried critic call re-uses the same
            # draft object, so id-keying still accumulates across retries.
            key = id(draft)
            records = records_by_key.setdefault(key, [])

            # DEADLINE (bug: the per-batch slices could sum past the critic budget).
            # One monotonic deadline captured at critic entry bounds the whole pass;
            # every slice below is clamped to what is actually left against it.
            clock = asyncio.get_running_loop()
            started = clock.time()
            judge_deadline = started + self._judge_phase_seconds()

            # The warm-up cached only SUCCESSES, so this re-run retries exactly the
            # clips that hung/failed — on a slow ASR box that is the whole critic
            # budget burnt before a single WM call. Bound it to its own share and
            # degrade to "judge what we have transcripts for".
            transcripts: dict[str, str]
            try:
                async with asyncio.timeout(self._llm_timeout * _CRITIC_ASR_SHARE):
                    transcripts = await self._transcribe_all(run_id, draft.clips, transcript_cache)
            except TimeoutError:
                transcripts = {
                    c.chunk_id: transcript_cache[c.chunk_id]
                    for c in draft.clips
                    if c.chunk_id in transcript_cache
                }
                log.warning(
                    "%s: in-critic re-transcribe exceeded its %.1fs share, "
                    "judging the %d clip(s) already transcribed",
                    self.name,
                    self._llm_timeout * _CRITIC_ASR_SHARE,
                    len(transcripts),
                )

            # No transcript ⇒ unjudgeable (ASR down, or gateway-only clips). Those
            # chunks fall through to the coverage accounting below — SOFT, but NOT a
            # clean pass.
            judgeable = [c for c in draft.clips if c.chunk_id in transcripts]
            batches = list(batched(judgeable, _JUDGE_BATCH_SIZE, strict=False))
            share = self._judge_batch_budget(len(batches))
            for batch in batches:
                judged_ids = {r.chunk_id for r in records}
                batch_ids = {c.chunk_id for c in batch} - judged_ids
                if not batch_ids:  # already judged by a previous attempt
                    continue
                budget = self._next_batch_budget(share, judge_deadline - clock.time())
                if budget is None:
                    # STOP cleanly: starting a batch that provably cannot finish only
                    # trades the outer cancel (which discards the pass) for nothing.
                    log.warning(
                        "%s: judge budget exhausted after %.1fs, stopping the loop early",
                        self.name,
                        clock.time() - started,
                    )
                    break
                # Retries are paid for out of THIS batch's own slice, never out of
                # the shared critic deadline: ``attempt_plan`` shrinks the ladder to
                # fit ``budget``, and the ``asyncio.timeout(budget)`` below is the
                # hard bound that holds however the ladder is sized (a backoff sleep
                # included). A sibling batch's share is therefore untouchable.
                per_attempt, batch_retries = attempt_plan(budget, self._llm_retries)
                try:
                    # Bound EACH batch: one slow batch must not eat the whole critic
                    # budget and take every other batch's judgments down with it.
                    async with asyncio.timeout(budget):
                        resp = await stage_call(
                            # `_batch=batch` binds the loop variable at lambda
                            # creation. Closing over it would be the late-binding
                            # bug `_judge_batch_call`'s docstring already warns about
                            # (ruff B023) — every batch would judge the LAST one.
                            downshift_reasoning(
                                lambda think, _batch=batch: self._judge_batch_call(
                                    _batch, transcripts, enable_reasoning=think
                                )
                            ),
                            timeout_seconds=per_attempt,
                            retries=batch_retries,
                            retry_on=(TransientUpstreamError, httpx.TransportError),
                            stage_name=f"{self.name}:judge_batch",
                        )
                except PermanentUpstreamError as exc:
                    # FAIL FAST + REPORT. No retry can fix a 402 ("configure your LLM
                    # provider"), a rejected schema or a prompt over the worker's token
                    # limit, and every remaining batch would hit the identical wall —
                    # so stop the loop instead of paying for N guaranteed losses on a
                    # paid run. Still SOFT: the stage degrades to partial coverage.
                    # Logged at ERROR and recorded in ``judge_faults`` so the run's
                    # note carries the actionable upstream sentence rather than a
                    # generic "some chunks went unjudged".
                    detail = str(exc)[:300]
                    log.error(
                        "%s: judge worker PERMANENT upstream fault, stopping the judge "
                        "loop (run degraded, NOT failed): %s",
                        self.name,
                        detail,
                    )
                    judge_faults.append(
                        "audiobook_qa degraded: the judge worker returned a PERMANENT "
                        f"upstream fault that no retry can fix — {detail}"
                    )
                    break
                except StageRetriesExhaustedError as exc:
                    # Transient upstream, retried to the end of THIS batch's slice and
                    # still failing ⇒ treat it as an outage: stop the loop, keep every
                    # judgment made so far, and report why. Never a hard failure.
                    detail = describe_exception(exc.last_error)[:300]
                    log.warning(
                        "%s: judge worker transient upstream failure after %d attempt(s), "
                        "stopping the judge loop: %s",
                        self.name,
                        exc.attempts,
                        detail,
                    )
                    judge_faults.append(
                        "audiobook_qa degraded: the judge worker was unreachable after "
                        f"{exc.attempts} attempt(s) within the batch budget — {detail}"
                    )
                    break
                except TimeoutError, StageTimeoutError:
                    # SOFT + PARTIAL: this batch (its own slice expired, or every
                    # attempt timed out) goes unjudged; every other batch still
                    # judges. Do NOT abort the critic — partial judgments survive.
                    # ``StageTimeoutError`` is listed BEFORE the RuntimeError branch
                    # below on purpose: it subclasses ``StageError``/``RuntimeError``
                    # and would otherwise be mis-read as a worker outage and break.
                    log.warning(
                        "%s: judge batch exceeded its %.1fs budget, skipping %d chunk(s): %s",
                        self.name,
                        budget,
                        len(batch),
                        ", ".join(sorted(batch_ids)),
                    )
                    continue
                except (httpx.HTTPError, httpx.InvalidURL, RuntimeError) as exc:
                    # SOFT: a genuine WM worker OUTAGE must not fail the run. A bug in
                    # building the message / parsing (ValueError, TypeError) is NOT
                    # caught here — it rides the loop's outer gate as a degrade.
                    # BREAK (not return): whatever earlier batches judged is retained
                    # and the rest is reported as unjudged coverage, never as a pass.
                    # This branch is now the UNCLASSIFIED remainder: anything ``guarded``
                    # turns into Permanent/Transient is handled above.
                    log.warning(
                        "%s: judge worker error (%s), stopping the judge loop: %s",
                        self.name,
                        type(exc).__name__,
                        exc,
                    )
                    judge_faults.append(
                        f"audiobook_qa degraded: the judge worker errored "
                        f"({type(exc).__name__}) — {str(exc)[:300]}"
                    )
                    break
                _usage_add(usage, resp.usage_metrics)
                batch_raw = (
                    resp.output.get("judgments") if isinstance(resp.output, Mapping) else None
                )
                if isinstance(batch_raw, list):
                    records.extend(self._validate(batch_raw, batch_ids))
            unjudged = _unjudged_chunks(draft.clips, records)
            flagged = [r.chunk_id for r in records if r.quality_score < _THRESHOLD]
            # Convergence = the WORST chunk: any flagged chunk keeps the loop
            # iterating so it actually gets a re-roll pass. With ZERO judgments the
            # score is pinned to _SCORE_MIN ("not assessed") — defaulting to
            # _SCORE_MAX made a fully-skipped loop converge as a perfect verdict.
            score = min(
                (r.quality_score for r in records),
                default=_SCORE_MIN if unjudged else _SCORE_MAX,
            )
            return Critique(score=score, rerun_stages=flagged, directives=records)

        async def reroll_optimizer(draft: _QADraft, critique: Critique) -> _QADraft:
            """Apply the judge's respelling to each flagged chunk and re-synth it.

            The fix is applied as PLAIN TEXT (``apply_lexicon`` — the same mechanism
            the ingest pronunciation lexicon uses), NOT ``<phoneme>`` SSML: the judge
            emits a hyphenated respelling ("Mer-kee"), which kokoro's IPA-only
            phoneme path would mis-speak; fed as plain text, espeak/misaki phonemizes
            it. A re-synth with the corrected text yields a new ``deterministic_clip_id``
            → genuinely different audio. SOFT throughout: any synth failure keeps the
            original clip, never raises. The chunk-id spine is preserved (only a
            clip's audio changes), so the loop engine's ``id_extractor`` guard holds."""
            records = critique.directives if isinstance(critique.directives, list) else []
            fixes: dict[str, tuple[str, str]] = {
                r.chunk_id: (r.suggested_word, r.suggested_pronunciation)
                for r in records
                if isinstance(r, ChunkJudgment)
                and r.quality_score < _THRESHOLD
                and r.suggested_word
                and r.suggested_pronunciation
            }
            if not fixes:
                return draft  # nothing actionable ⇒ no synth, identical spine
            rerolled: set[str] = set()
            new_clips: list[_QAClip] = []
            for clip in draft.clips:
                fix = fixes.get(clip.chunk_id)
                # Not flagged, or no recoverable synth provenance (pre-provenance
                # narrate output / gateway clip) ⇒ keep the original clip.
                if fix is None or clip.speed is None:
                    new_clips.append(clip)
                    continue
                word, respell = fix
                corrected = apply_lexicon(clip.manuscript_text, {word: respell})
                if corrected == clip.manuscript_text:
                    new_clips.append(clip)  # the respelling matched nothing ⇒ no-op
                    continue
                try:
                    new_id, new_url, new_seconds = await synthesize_and_store(
                        self._tts,
                        text=corrected,
                        voice=clip.voice,
                        run_id=run_id,
                        stage_name=_NARRATE_STAGE,
                        speed=clip.speed,
                    )
                except Exception as exc:  # noqa: BLE001 — SOFT: a re-synth failure keeps the original
                    log.warning(
                        "%s: re-roll synth failed for chunk %s (%s), keeping original: %s",
                        self.name,
                        clip.chunk_id,
                        type(exc).__name__,
                        exc,
                    )
                    new_clips.append(clip)
                    continue
                rerolled.add(clip.chunk_id)
                new_clips.append(
                    replace(clip, clip_id=new_id, artifact_url=new_url, seconds=new_seconds)
                )
            if not rerolled:
                # Nothing actually re-synthesized (every fix was a no-op / SOFT-failed)
                # ⇒ return the SAME draft object so the loop does not re-judge identical
                # audio under a new identity.
                return draft
            candidate = _QADraft(clips=tuple(new_clips))
            # Invalidate the stale transcript for each re-rolled chunk so the next
            # critic pass re-transcribes it (the cache is keyed by chunk_id; a new
            # clip_id under the same chunk_id would otherwise HIT the OLD transcript
            # and the re-roll would look inert).
            for cid in rerolled:
                transcript_cache.pop(cid, None)
            # Seed the CANDIDATE's own record set (id-keyed) with the initial draft's
            # judgments for the NON-re-rolled chunks. Pass 2's critic then re-judges
            # ONLY the corrected chunks (it skips already-judged chunk_ids) while the
            # initial draft's record set stays untouched — so a discarded/tied
            # candidate can never leak a "fixed" verdict into the retained report.
            prior = records_by_key.get(id(draft), [])
            records_by_key[id(candidate)] = [r for r in prior if r.chunk_id not in rerolled]
            return candidate

        # Belt-and-suspenders: the critic/optimizer swallow their own outages, but a
        # defect in the loop itself must STILL accept the current narration — the soft
        # gate never lets this stage fail the ship (mirrors footage_critique).
        try:
            # Hoist whole-book ASR OUT of the loop's per-critic timeout (bug #1):
            # transcribing an entire book (~minutes) inside the critic blew the
            # per-attempt LLM budget (llm_budget_seconds) → StageTimeoutError →
            # SOFT loop_error, so QA silently no-opped on every real-length book
            # while passing on 2-3-chunk fixtures. Warm the transcript cache ONCE
            # here under the stage's own (far larger) blueprint budget; the critic
            # then pays only for the fast WM judge calls. SOFT: _transcribe_all
            # skips per-clip failures and never raises.
            # ...but hoisting it out of the critic left it UNBOUNDED (bug #3): the
            # only remaining bound was the blueprint's default_timeout_seconds=3600,
            # which fires as a HARD StageTimeoutError from the stage middleware —
            # OUTSIDE this soft-degrade guard — so a hung ASR box killed a paid
            # stage. Bound it here so a hang degrades SOFT instead.
            try:
                async with asyncio.timeout(self._asr_timeout):
                    await self._transcribe_all(run_id, initial.clips, transcript_cache)
            except TimeoutError:
                log.warning(
                    "%s: whole-book ASR exceeded %.0fs, accepting current",
                    self.name,
                    self._asr_timeout,
                )
                return self._accept_current(
                    usage,
                    reason="asr_timeout",
                    note=(
                        f"audiobook_qa degraded: whole-book listen-back ASR exceeded "
                        f"{self._asr_timeout:.0f}s — accepted current narration unjudged"
                    ),
                    assessed=False,
                    unjudged=[c.chunk_id for c in initial.clips],
                )
            result = await run_evaluator_optimizer(
                initial,
                critic_fn=critic_fn,
                optimizer_fn=reroll_optimizer if reroll_enabled else _noop_optimizer,
                threshold=_THRESHOLD,
                epsilon=_EPSILON,
                max_passes=_MAX_PASSES_REROLL if reroll_enabled else _MAX_PASSES,
                stage_name=self.name,
                timeout_seconds=self._llm_timeout,
                retries=self._llm_retries,
                id_extractor=_chunk_ids,
            )
        except StageSkippedError, StagePreconditionError:
            # Control-flow signals are NOT failures — they must reach the run engine,
            # not be swallowed into a degraded "accept current".
            raise
        except Exception as exc:  # noqa: BLE001 — SOFT gate: accept current on any loop error
            # PARTIAL-SURVIVES: the critic writes each batch's records into
            # ``records_by_key`` as it goes, so an outer cancel (loop-engine timeout)
            # unwinding mid-pass still lands here with everything judged so far —
            # this used to return ZERO judgments, the exact silent no-op the
            # per-batch bound was meant to prevent.
            # NOT named ``partial``: that shadows ``functools.partial`` for the WHOLE
            # of ``handle`` (Python binds locals per function, not per branch), which
            # makes the judge loop's ``partial(...)`` read an unbound local.
            partial_records = records_by_key.get(id(initial), [])
            log.warning(
                "%s: loop error after %d judgment(s), accepting current: %s",
                self.name,
                len(partial_records),
                exc,
            )
            return self._finalize(
                clips=initial.clips,
                records=partial_records,
                usage=usage,
                score=min((r.quality_score for r in partial_records), default=_SCORE_MIN),
                passes=1 if partial_records else 0,
                converged=False,
                reason="loop_error",
                extra_notes=[*judge_faults, f"audiobook_qa degraded: {exc}"],
            )

        best_records = records_by_key.get(id(result.draft), [])
        return self._finalize(
            clips=result.draft.clips,
            records=best_records,
            usage=usage,
            score=result.score,
            passes=result.passes,
            converged=result.converged,
            reason=result.reason,
            extra_notes=list(judge_faults),
            corrected_clips=_corrected_clips(initial.clips, result.draft.clips),
        )

    def _finalize(
        self,
        *,
        clips: Sequence[_QAClip],
        records: Sequence[ChunkJudgment],
        usage: dict[str, int],
        score: float,
        passes: int,
        converged: bool,
        reason: str,
        extra_notes: Sequence[str],
        corrected_clips: Sequence[dict[str, object]] = (),
    ) -> AudiobookQAOutput:
        """Build the stage output, enforcing the COVERAGE contract.

        A verdict is only as good as its coverage: any chunk without a judgment
        (skipped batch, exhausted budget, WM outage mid-loop, missing transcript)
        makes the result PARTIAL, and a partial result must never present as a clean
        converged pass — it carries a distinct ``reason``, ``converged=False``, and a
        degrade note naming how many chunks went unjudged."""
        flagged = [r for r in records if r.quality_score < _THRESHOLD]
        flagged_chunks = [r.chunk_id for r in flagged]
        unjudged = _unjudged_chunks(clips, records)
        if unjudged:
            converged = False
            if reason in _LOOP_REASONS:
                reason = "partial_judge_coverage" if records else "no_judge_coverage"
        # Re-roll intent is recorded from the retained critique's flagged records
        # (a single scoring pass ⇒ the pass ordinal is always 1). A future real
        # re-roll consumes suggested_pronunciation to re-synth exactly these chunks.
        reroll_requests: list[dict[str, object]] = [
            {
                "chunk_id": r.chunk_id,
                "pass": 1,
                "issue": r.issue,
                "suggested_word": r.suggested_word,
                "suggested_pronunciation": r.suggested_pronunciation,
            }
            for r in flagged
        ]
        notes: list[str] = list(extra_notes)
        if unjudged:
            notes.append(
                f"audiobook_qa degraded: listen-back coverage is PARTIAL — "
                f"{len(unjudged)} of {len(clips)} narrated chunk(s) went UNJUDGED "
                f"(judge budget exhausted, worker error, or no transcript), so this "
                f"is NOT a clean pass; unjudged: {_preview(unjudged)}"
            )
        if flagged_chunks:
            notes.append(
                f"audiobook_qa: unresolved narration issue(s) on chunk(s) "
                f"{', '.join(flagged_chunks)} — accepted current narration at score "
                f"{score:.1f} (< {_THRESHOLD:.1f}) after {passes} pass(es); "
                f"re-roll recorded, deferred (PR-8)"
            )
        return AudiobookQAOutput(
            qa=AudiobookQARecord(
                score=score,
                passes=passes,
                converged=converged,
                reason=reason,
                flagged_chunks=flagged_chunks,
                judgments=list(records),
                assessed=bool(records),
                unjudged_chunks=unjudged,
            ),
            token_usage=_build_usage(usage),
            notes=notes,
            reroll_requests=reroll_requests,
            corrected_clips=list(corrected_clips),
        )

    def _accept_current(
        self,
        usage: dict[str, int],
        *,
        reason: str,
        note: str,
        assessed: bool = True,
        unjudged: Sequence[str] = (),
    ) -> AudiobookQAOutput:
        """Emit an accept-current verdict with no re-roll (the soft fallback).

        ``assessed=False`` means nothing was listened back to: the score is pinned to
        ``_SCORE_MIN`` and ``converged`` is False, because "we judged nothing" must
        not serialise identically to "we judged everything and it was perfect"."""
        return AudiobookQAOutput(
            qa=AudiobookQARecord(
                score=_SCORE_MAX if assessed else _SCORE_MIN,
                passes=0,
                converged=assessed,
                reason=reason,
                flagged_chunks=[],
                judgments=[],
                assessed=assessed,
                unjudged_chunks=list(unjudged),
            ),
            token_usage=_build_usage(usage),
            notes=[note] if note else [],
            reroll_requests=[],
        )

    async def _transcribe_all(
        self, run_id: str, clips: Sequence[_QAClip], cache: dict[str, str]
    ) -> dict[str, str]:
        """Transcribe each clip's stored bytes → ``{chunk_id: transcript_text}``.

        Retrieves each clip by storage key (no presigned-URL fetch → no SSRF
        surface). A per-clip failure (missing object, ASR outage) is SOFT-skipped:
        that chunk is simply absent from the result and goes unjudged, never
        raising. Results are memoised in ``cache`` so a second scoring round reuses
        them without re-transcribing."""
        sem = asyncio.Semaphore(_ASR_CONCURRENCY)

        async def _one(clip: _QAClip) -> tuple[str, str] | None:
            if clip.chunk_id in cache:
                return clip.chunk_id, cache[clip.chunk_id]
            async with sem:
                try:
                    text = await self._transcribe_clip(run_id, clip)
                except (
                    httpx.HTTPError,
                    httpx.InvalidURL,
                    RuntimeError,
                    TimeoutError,
                    FileNotFoundError,
                    # The ASR lane stayed busy past its deadline (#1106) — an outage,
                    # soft-skipped exactly like the RuntimeError a busy 503 used to be.
                    # A byo_required refusal (SlotNotConfiguredError) is NOT here: the
                    # tenant must act, so it propagates and names the fix.
                    GpuBusyError,
                ) as exc:
                    log.warning(
                        "%s: transcription failed for chunk %s (%s), skipping: %s",
                        self.name,
                        clip.chunk_id,
                        type(exc).__name__,
                        exc,
                    )
                    return None
            if text is None:
                return None
            cache[clip.chunk_id] = text
            return clip.chunk_id, text

        results = await asyncio.gather(*(_one(c) for c in clips))
        return {chunk_id: text for r in results if r is not None for (chunk_id, text) in (r,)}

    async def _transcribe_clip(self, run_id: str, clip: _QAClip) -> str | None:
        """Download one clip's stored bytes and return its ASR transcript text.

        ``None`` when the object is absent (e.g. a gateway-rendered clip that was
        never stored in orchestrator storage) — the caller SOFT-skips it."""
        storage = await self._get_storage()
        data = await storage.retrieve_stage_artifact(
            run_id, _NARRATE_STAGE, f"{clip.clip_id}{_CLIP_SUFFIX}"
        )
        if not data:
            return None
        with tempfile.TemporaryDirectory() as td:
            with tempfile.NamedTemporaryFile(dir=td, suffix=_CLIP_SUFFIX, delete=False) as fh:
                fh.write(data)
                clip_path = Path(fh.name)
            result: TranscriptionResult = await self._asr.transcribe_file(
                file_path=clip_path,
                word_timestamps=True,
                content_type="audio/wav",
            )
        return result.text

    def _build_message(self, clips: Sequence[_QAClip], transcripts: Mapping[str, str]) -> str:
        rows = [
            json.dumps(
                {
                    "chunk_id": c.chunk_id,
                    "manuscript": c.manuscript_text[:_MAX_TEXT_CHARS],
                    "transcript": transcripts.get(c.chunk_id, "")[:_MAX_TEXT_CHARS],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            for c in clips
        ]
        return (
            "CHUNKS (one JSON object per chunk — the intended manuscript text and the "
            "ASR transcript of the audio actually rendered for it):\n" + "\n".join(rows)
        )

    def _validate(self, raw: Sequence[object], valid_ids: set[str]) -> list[ChunkJudgment]:
        """Keep only verdicts on a KNOWN chunk_id, coerce ``ok``, clamp severity,
        derive the convergence quality, and bound the free-text fields — the model
        is advisory, this stage is authoritative (mirrors footage_critique)."""
        out: list[ChunkJudgment] = []
        seen: set[str] = set()
        for raw_item in raw:
            if not isinstance(raw_item, Mapping):
                continue
            item = cast("Mapping[str, object]", raw_item)
            chunk_id = str(item.get("chunk_id") or "")
            if chunk_id not in valid_ids or chunk_id in seen:
                continue
            seen.add(chunk_id)
            ok = bool(item.get("ok"))
            severity = _clamp_score(item.get("severity_score"), lo=0.0, hi=_SCORE_MAX, default=0.0)
            word, phonetic = _pronunciation(item.get("suggested_pronunciation"))
            out.append(
                ChunkJudgment(
                    chunk_id=chunk_id,
                    ok=ok,
                    quality_score=_quality(ok, severity),
                    issue=str(item.get("issue") or "")[:_MAX_ISSUE_CHARS].strip(),
                    severity_score=severity,
                    suggested_word=word,
                    suggested_pronunciation=phonetic,
                )
            )
        return out


def _pronunciation(value: object) -> tuple[str, str]:
    """Extract ``(word, ipa_or_phonetic)`` from the judge's optional fix object.

    ``null`` / a non-object ⇒ empty strings (no fix). Both fields are bounded."""
    if not isinstance(value, Mapping):
        return "", ""
    item = cast("Mapping[str, object]", value)
    word = str(item.get("word") or "")[:_MAX_WORD_CHARS].strip()
    phonetic = str(item.get("ipa_or_phonetic") or "")[:_MAX_WORD_CHARS].strip()
    return word, phonetic
