"""Live glue that turns narration line clips into the compiler's ``captions`` payload.

This is the async I/O orchestrator that wires together three already-shipped
pieces — it deliberately reimplements none of them:

* the #500 **pure alignment core** in :mod:`src.services.caption_reconcile`
  (:func:`build_asr_words` + :func:`align_script_to_words`) and the cue packer in
  :mod:`src.services.caption_cues` (:func:`group_tokens_into_cues`);
* the existing **ASR client** :class:`~src.clients.transcription_worker_client.TranscriptionWorkerClient`
  (Whisper word-timestamps over HTTP);
* the existing **storage download-by-key**
  :meth:`~src.services.storage_service.StorageService.retrieve_stage_artifact`.

For each shot that carries a narration clip we download the clip bytes, run ASR to
get per-word timings, align the *authored script* line to those timings (the script
is the caption surface — never the raw ASR transcription), pack the timed tokens
into wrapped on-screen cues, and emit the compiler's per-shot cue payload.

Per-line failures (missing object, ASR error, malformed response) are swallowed to a
skip — one bad clip must never fail the whole caption build. When nothing produces
cues the function returns ``None`` so the caller can omit the ``captions`` field
entirely, keeping the compile byte-identical. This path is gated OFF by default
upstream.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypedDict

from src.clients.schemas.transcription_worker import TranscriptionResult
from src.services.caption_cues import group_tokens_into_cues
from src.services.caption_reconcile import align_script_to_words, build_asr_words
from src.services.stages.shared.step_ledger import (
    JsonValue,
    NullStepLedger,
    StepLedger,
    hash_inputs,
)

logger = logging.getLogger(__name__)


class WordPayload(TypedDict):
    """One cue word's shot-local timing, for the compiler's karaoke ``{\\k}`` highlight."""

    text: str
    start: float
    end: float


class CuePayload(TypedDict):
    """One burned-in caption cue in the compiler's wire shape (shot-local seconds).

    ``words`` carries the cue's per-word timings; the compiler consumes them only
    when karaoke is requested (``burned_in``) and otherwise ignores them, so the
    rendered output is unchanged by their presence.
    """

    text: str
    start: float
    end: float
    words: list[WordPayload]


class ShotCaptionsPayload(TypedDict):
    """The compiler's per-shot ``captions`` entry: a shot index + its cues."""

    shot_index: int
    cues: list[CuePayload]


class _ClipStorage(Protocol):
    """The single storage capability this module needs: download a clip by key."""

    async def retrieve_stage_artifact(
        self, job_id: str, stage_name: str, filename: str
    ) -> bytes | None: ...


class _ClipAsr(Protocol):
    """The single ASR capability this module needs: transcribe a clip with word times."""

    async def transcribe_file(
        self,
        *,
        file_path: Path,
        word_timestamps: bool = ...,
        content_type: str = ...,
    ) -> TranscriptionResult: ...


_MAX_CONCURRENCY = 4


class _AsrParams(TypedDict):
    word_timestamps: bool
    content_type: str


#: What the fallback asks ASR for. Sent AND hashed into the step-ledger key, so a
#: change here invalidates stored timings instead of serving them. The language
#: and model are transcription_worker's own defaults — this client sends neither.
_ASR_PARAMS: _AsrParams = {"word_timestamps": True, "content_type": "audio/wav"}


@dataclass(frozen=True)
class ClipRef:
    """A narration line clip: what to fetch, what caption surface to align to it."""

    line_id: str
    text: str  # the authored SCRIPT line (caption surface — never ASR)
    narrate_stage: str  # stage_name the clip was stored under, e.g. "demo_video_narrate"
    clip_filename: str  # stored object filename, e.g. f"{clip_id}.wav"
    seconds: float  # clip duration (clip_end in the shot-local timebase)


async def _sidecar_word_timings(
    storage: _ClipStorage, job_id: str, ref: ClipRef
) -> list[dict[str, object]]:
    """Per-word spans tts_backend measured for this clip, or ``[]``.

    ``store_tts_clip`` writes a ``{clip_id}.json`` sidecar next to the wav and
    adds a ``word_timings`` list when the synthesizing engine could produce one.
    Returned in ``build_asr_words`` input shape (``{word, start, end}``) so the
    two sources converge on one type; ``probability`` is deliberately left off,
    defaulting to 1.0 — a TTS timing is measured, not inferred, so it must never
    be discounted against an ASR guess.

    ``[]`` on ANY doubt — no sidecar, unreadable JSON, key absent, or an entry
    missing a numeric span. The caller reads empty as "fall back to ASR", so a
    malformed sidecar degrades to the old behaviour instead of producing captions
    with holes in them.
    """
    name = f"{Path(ref.clip_filename).stem}.json"
    try:
        raw = await storage.retrieve_stage_artifact(job_id, ref.narrate_stage, name)
        if not raw:
            return []
        entries = json.loads(raw).get("word_timings")
        if not isinstance(entries, list):
            return []
        return [e for e in entries if isinstance(e, dict) and "start" in e and "end" in e]
    except Exception:  # a bad sidecar must degrade to ASR, never raise
        logger.debug("no usable word_timings sidecar for %s", ref.clip_filename, exc_info=True)
        return []


async def build_shot_captions(
    *,
    final_shots: Sequence[Mapping[str, Any]],
    clip_refs: dict[str, ClipRef],
    job_id: str,
    storage: _ClipStorage,
    asr: _ClipAsr,
    ledger: StepLedger | None = None,
) -> list[ShotCaptionsPayload] | None:
    """Build the compiler ``captions`` payload from narration line clips.

    For every shot in ``final_shots`` whose ``line_id`` has a :class:`ClipRef`, the
    clip is downloaded, transcribed for word timings, aligned to the authored script
    line, and packed into cues. Returns a list of ``{"shot_index", "cues"}`` entries
    (sorted by shot index) or ``None`` when nothing produced cues — so the caller can
    omit ``captions`` and keep the compile byte-identical. Never raises for a single
    clip's failure.

    ``ledger`` is the calling stage's step ledger (``ctx.step_ledger(self.name)``).
    Only the ASR fallback uses it; the sidecar path never touches it. Omitted ⇒
    no reuse, every fallback clip is transcribed.
    """
    ledger = ledger if ledger is not None else NullStepLedger()
    with tempfile.TemporaryDirectory() as td:
        sem = asyncio.Semaphore(_MAX_CONCURRENCY)

        async def _build_one(idx: int, line_id: str) -> ShotCaptionsPayload | None:
            async with sem:
                try:
                    ref = clip_refs[line_id]

                    # PREFERRED PATH: timings tts_backend measured when it
                    # synthesized this line. The kokoro duration predictor gives
                    # true per-word spans, so no transcription is needed — and
                    # not needing the audio means no download and no temp file
                    # either. ASR here cost 366-1990s per clip on mini2 against a
                    # 300s client timeout, i.e. it could not complete at all; this
                    # removes the call rather than tuning it.
                    words = build_asr_words(await _sidecar_word_timings(storage, job_id, ref))

                    if not words:
                        # FALLBACK: clips synthesized by the ONNX engine (which
                        # cannot emit timings) and every clip written before this
                        # sidecar key existed. Absent timings mean "ask ASR", never
                        # "this line has no words".
                        data = await storage.retrieve_stage_artifact(
                            job_id, ref.narrate_stage, ref.clip_filename
                        )
                        if not data:
                            return None
                        suffix = Path(ref.clip_filename).suffix or ".wav"

                        async def _transcribe(
                            data: bytes = data, suffix: str = suffix
                        ) -> list[JsonValue]:
                            with tempfile.NamedTemporaryFile(
                                dir=td, suffix=suffix, delete=False
                            ) as fh:
                                fh.write(data)
                                clip_path = Path(fh.name)
                            raw = await asr.transcribe_file(file_path=clip_path, **_ASR_PARAMS)
                            return [
                                w.model_dump(mode="json") for seg in raw.segments for w in seg.words
                            ]

                        # The one paid call on this path, so the only one that
                        # goes through the ledger: a re-entered compile reuses
                        # the word timings of every clip it already transcribed.
                        # Hashed over the clip's BYTES, not just its key — a
                        # re-narrated line reuses its object key, and timings
                        # for the old audio would mis-time every word.
                        timings = await ledger.once(
                            f"asr:{ref.narrate_stage}/{ref.clip_filename}",
                            hash_inputs(ref.narrate_stage, ref.clip_filename, data, _ASR_PARAMS),
                            _transcribe,
                        )
                        words = build_asr_words(
                            [w for w in timings if isinstance(w, dict)]
                            if isinstance(timings, list)
                            else []
                        )
                    tokens = align_script_to_words(
                        ref.text.split(),
                        words,
                        clip_start=0.0,
                        clip_end=ref.seconds,
                    )
                    cues = group_tokens_into_cues(tokens)
                    emitted: list[CuePayload] = []
                    for cue in cues:
                        text = " ".join(cue.lines).strip()
                        if cue.end > cue.start and text:
                            emitted.append(
                                {
                                    "text": text,
                                    "start": cue.start,
                                    "end": cue.end,
                                    # Only words that satisfy the compiler's CaptionWord
                                    # contract (end > start >= 0). Interpolated words across
                                    # abutting ASR timestamps can be zero-width (start == end,
                                    # see caption_reconcile._interpolate_pending); emitting one
                                    # would 422 the compile — which runs OUTSIDE the caption
                                    # soft-fail guard — and hard-fail the render. Dropping it
                                    # just skips that word's karaoke highlight (cue text is
                                    # unaffected; a fully-dropped list falls back to phrase mode).
                                    "words": [
                                        {"text": w, "start": s, "end": e}
                                        for (w, s, e) in cue.words
                                        if e > s and s >= 0
                                    ],
                                }
                            )
                    if emitted:
                        return {"shot_index": idx, "cues": emitted}
                    return None
                except Exception:  # a bad clip must never fail the render
                    logger.warning(
                        "word-timed caption build failed for line %s",
                        line_id,
                        exc_info=True,
                    )
                    return None

        coros = [
            _build_one(idx, line_id)
            for idx, shot in enumerate(final_shots)
            if (line_id := shot.get("line_id")) in clip_refs
        ]
        results = await asyncio.gather(*coros)

    entries = [entry for entry in results if entry is not None]
    entries.sort(key=lambda e: e["shot_index"])
    return entries or None
