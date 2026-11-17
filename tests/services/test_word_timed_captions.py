"""Unit tests for the DB/IO-coupled word-timed caption builder
``src.services.word_timed_captions.build_shot_captions``.

The pure alignment/packing core is covered in ``test_caption_reconcile.py`` +
``test_caption_cues.py``. Here we prove the *whole chain* end-to-end with only
storage + ASR faked: the builder pulls each spoken line's synthesised clip from
storage, transcribes it, then reuses the REAL pure core
(``build_asr_words`` → ``align_script_to_words`` → ``group_tokens_into_cues``)
to mint shot-local, script-faithful cues. The alignment is NOT mocked — the
fakes feed it realistic Whisper word dicts so the emitted cue text is asserted
to be the authored SCRIPT surface, never the ASR transcription.

Fakes are deterministic (no ffmpeg/network): ``FakeStorage`` maps a clip
filename to canned bytes (or ``None`` to simulate a miss); ``FakeAsr`` reads the
temp file the builder wrote and returns a canned segments/words payload keyed by
those exact bytes (or raises, to exercise per-line soft-fail).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.clients.schemas.transcription_worker import TranscriptionResult
from src.services.word_timed_captions import ClipRef, ShotCaptionsPayload, build_shot_captions

pytestmark = pytest.mark.asyncio


# ── fakes ───────────────────────────────────────────────────────────


class FakeStorage:
    """Maps ``clip_filename`` → canned bytes (or ``None`` for a storage miss)."""

    def __init__(self, contents: dict[str, bytes | None]) -> None:
        self._contents = contents
        self.calls: list[tuple[str, str, str]] = []

    async def retrieve_stage_artifact(
        self, job_id: str, stage_name: str, filename: str
    ) -> bytes | None:
        self.calls.append((job_id, stage_name, filename))
        return self._contents.get(filename)


class FakeAsr:
    """Returns a canned Whisper payload keyed by the clip bytes the builder wrote.

    The builder writes the storage bytes to a temp file and hands us its path; we
    read it back and look the payload up by content, so distinct clips get
    distinct word timings. Bytes listed in ``raise_on`` blow up mid-transcribe to
    exercise the per-line soft-fail path.
    """

    def __init__(
        self,
        payloads: dict[bytes, TranscriptionResult],
        *,
        raise_on: set[bytes] | None = None,
    ) -> None:
        self._payloads = payloads
        self._raise_on = raise_on or set()

    async def transcribe_file(
        self,
        *,
        file_path: Path,
        word_timestamps: bool = True,
        return_srt: bool = False,
        content_type: str = "audio/wav",
    ) -> TranscriptionResult:
        data = Path(file_path).read_bytes()
        if data in self._raise_on:
            raise RuntimeError("whisper exploded on this clip")
        return self._payloads[data]


def _word(word: str, start: float, end: float, probability: float = 0.98) -> dict:
    """One Whisper per-word dict."""
    return {"start": start, "end": end, "word": word, "probability": probability}


def _payload(words: list[dict]) -> TranscriptionResult:
    """A single-segment Whisper transcription result."""
    return TranscriptionResult.model_validate(
        {
            "backend": "whisper",
            "model": "test",
            "duration": 0.0,
            "text": "",
            "segments": [{"id": 0, "start": 0.0, "end": 0.0, "text": "", "words": words}],
        }
    )


def _clip_bytes(marker: str) -> bytes:
    """Distinct WAV-ish bytes so ASR can key on which clip it received."""
    return b"RIFF...." + marker.encode()


def _all_text(entry: ShotCaptionsPayload) -> str:
    """Join every cue's text of one emitted shot-caption entry."""
    return " ".join(cue["text"] for cue in entry["cues"])


def _by_index(result: list[ShotCaptionsPayload]) -> dict[int, ShotCaptionsPayload]:
    return {entry["shot_index"]: entry for entry in result}


# ── 1. text is the SCRIPT, never the ASR transcription ──────────────


async def test_cue_text_is_script_not_asr_transcription() -> None:
    """ASR misspells 'model' as 'mahdel'; the cue must show the SCRIPT word.

    A naive echo-ASR builder that emits the transcription would leak 'mahdel'.
    """
    clip = _clip_bytes("L1")
    storage = FakeStorage({"L1.wav": clip})
    asr = FakeAsr(
        {
            clip: _payload(
                [
                    _word("Deploy", 0.0, 0.4),
                    _word("the", 0.4, 0.6),
                    _word("mahdel", 0.6, 1.1),  # ASR substitution
                    _word("today", 1.1, 1.6),
                ]
            )
        }
    )
    final_shots = [{"line_id": "L1", "text": "Deploy the model today"}]
    clip_refs = {
        "L1": ClipRef(
            line_id="L1",
            text="Deploy the model today",
            narrate_stage="promo_narrate",
            clip_filename="L1.wav",
            seconds=2.0,
        )
    }

    result = await build_shot_captions(
        final_shots=final_shots, clip_refs=clip_refs, job_id="j1", storage=storage, asr=asr
    )

    assert result is not None
    text = _all_text(result[0])
    for script_word in ("Deploy", "the", "model", "today"):
        assert script_word in text
    assert "mahdel" not in text  # ASR surface must never leak


# ── 2. shot_index is the post-splice enumerate position ─────────────


async def test_shot_index_is_post_splice_enumerate_position() -> None:
    """Line L1 sits at enumerate index 1 (behind a title card); index must be 1, not 0."""
    clip = _clip_bytes("L1")
    storage = FakeStorage({"L1.wav": clip})
    asr = FakeAsr({clip: _payload([_word("Hello", 0.0, 0.6), _word("world", 0.6, 1.2)])})
    final_shots = [
        {"title_card": "Intro"},  # index 0, no line_id
        {"line_id": "L1", "text": "Hello world"},  # index 1
        {"disclaimer": "Legal"},  # index 2, no line_id
    ]
    clip_refs = {
        "L1": ClipRef(
            line_id="L1",
            text="Hello world",
            narrate_stage="promo_narrate",
            clip_filename="L1.wav",
            seconds=2.0,
        )
    }

    result = await build_shot_captions(
        final_shots=final_shots, clip_refs=clip_refs, job_id="j1", storage=storage, asr=asr
    )

    assert result is not None
    assert len(result) == 1
    assert result[0]["shot_index"] == 1  # not 0 (pre-splice) — the enumerate slot


# ── 3. timebase is shot-local, in [0, seconds] and monotonic ────────


async def test_timebase_is_shot_local_and_monotonic() -> None:
    """All cue times fall within [0, seconds] and are monotonic non-decreasing.

    A global/wall-clock builder would offset times past the clip's own duration.
    """
    clip = _clip_bytes("L1")
    storage = FakeStorage({"L1.wav": clip})
    asr = FakeAsr(
        {
            clip: _payload(
                [
                    _word("one", 0.0, 1.0),
                    _word("two", 1.0, 2.0),
                    _word("three", 2.0, 3.0),
                    _word("four", 3.0, 4.0),
                ]
            )
        }
    )
    seconds = 4.0
    final_shots = [{"line_id": "L1", "text": "one two three four"}]
    clip_refs = {
        "L1": ClipRef(
            line_id="L1",
            text="one two three four",
            narrate_stage="promo_narrate",
            clip_filename="L1.wav",
            seconds=seconds,
        )
    }

    result = await build_shot_captions(
        final_shots=final_shots, clip_refs=clip_refs, job_id="j1", storage=storage, asr=asr
    )

    assert result is not None
    cues = result[0]["cues"]
    assert cues, "expected at least one cue"
    eps = 1e-6
    prev_start = -1.0
    for cue in cues:
        assert cue["start"] >= -eps
        assert cue["start"] <= cue["end"] + eps
        assert cue["end"] <= seconds + eps  # shot-local: never past the clip duration
        assert cue["start"] >= prev_start - eps  # monotonic non-decreasing
        prev_start = cue["start"]


# ── 4. per-line soft-fail: a bad line degrades, siblings survive ────


async def test_storage_miss_drops_only_that_line() -> None:
    """L1's clip is missing in storage → L1 dropped; L2 still produces a cue."""
    clip2 = _clip_bytes("L2")
    storage = FakeStorage({"L1.wav": None, "L2.wav": clip2})  # L1 miss
    asr = FakeAsr({clip2: _payload([_word("Second", 0.0, 0.7), _word("line", 0.7, 1.4)])})
    final_shots = [
        {"line_id": "L1", "text": "First line"},  # index 0
        {"line_id": "L2", "text": "Second line"},  # index 1
    ]
    clip_refs = {
        "L1": ClipRef(
            line_id="L1",
            text="First line",
            narrate_stage="promo_narrate",
            clip_filename="L1.wav",
            seconds=2.0,
        ),
        "L2": ClipRef(
            line_id="L2",
            text="Second line",
            narrate_stage="promo_narrate",
            clip_filename="L2.wav",
            seconds=2.0,
        ),
    }

    result = await build_shot_captions(
        final_shots=final_shots, clip_refs=clip_refs, job_id="j1", storage=storage, asr=asr
    )

    assert result is not None
    indexed = _by_index(result)
    assert 0 not in indexed  # L1 (storage miss) dropped
    assert 1 in indexed  # L2 survived
    assert "Second" in _all_text(indexed[1])


async def test_asr_raise_drops_only_that_line_without_raising() -> None:
    """ASR raises on L1's clip → L1 dropped, L2 survives, builder does NOT raise."""
    clip1 = _clip_bytes("L1")
    clip2 = _clip_bytes("L2")
    storage = FakeStorage({"L1.wav": clip1, "L2.wav": clip2})
    asr = FakeAsr(
        {
            clip1: _payload([_word("First", 0.0, 0.7), _word("line", 0.7, 1.4)]),
            clip2: _payload([_word("Second", 0.0, 0.7), _word("line", 0.7, 1.4)]),
        },
        raise_on={clip1},  # L1 blows up mid-transcribe
    )
    final_shots = [
        {"line_id": "L1", "text": "First line"},  # index 0
        {"line_id": "L2", "text": "Second line"},  # index 1
    ]
    clip_refs = {
        "L1": ClipRef(
            line_id="L1",
            text="First line",
            narrate_stage="promo_narrate",
            clip_filename="L1.wav",
            seconds=2.0,
        ),
        "L2": ClipRef(
            line_id="L2",
            text="Second line",
            narrate_stage="promo_narrate",
            clip_filename="L2.wav",
            seconds=2.0,
        ),
    }

    # Must not raise despite L1's ASR failure.
    result = await build_shot_captions(
        final_shots=final_shots, clip_refs=clip_refs, job_id="j1", storage=storage, asr=asr
    )

    assert result is not None
    indexed = _by_index(result)
    assert 0 not in indexed  # L1 (asr raise) dropped
    assert 1 in indexed  # L2 survived
    assert "Second" in _all_text(indexed[1])


# ── 5. nothing produced ⇒ None (caller omits the field) ─────────────


async def test_empty_clip_refs_returns_none() -> None:
    """No spoken lines at all ⇒ None, so the caller omits captions (byte-identical)."""
    storage = FakeStorage({})
    asr = FakeAsr({})
    final_shots = [{"title_card": "Intro"}, {"disclaimer": "Legal"}]

    result = await build_shot_captions(
        final_shots=final_shots, clip_refs={}, job_id="j1", storage=storage, asr=asr
    )

    assert result is None


async def test_all_lines_fail_returns_none() -> None:
    """Every line misses in storage ⇒ nothing produced ⇒ None."""
    storage = FakeStorage({"L1.wav": None, "L2.wav": None})
    asr = FakeAsr({})
    final_shots = [
        {"line_id": "L1", "text": "First line"},
        {"line_id": "L2", "text": "Second line"},
    ]
    clip_refs = {
        "L1": ClipRef(
            line_id="L1",
            text="First line",
            narrate_stage="promo_narrate",
            clip_filename="L1.wav",
            seconds=2.0,
        ),
        "L2": ClipRef(
            line_id="L2",
            text="Second line",
            narrate_stage="promo_narrate",
            clip_filename="L2.wav",
            seconds=2.0,
        ),
    }

    result = await build_shot_captions(
        final_shots=final_shots, clip_refs=clip_refs, job_id="j1", storage=storage, asr=asr
    )

    assert result is None


# ── 6. multiple lines map to their correct enumerate indices ────────


async def test_multiple_lines_map_to_correct_indices_and_scripts() -> None:
    """Two line-shots interleaved with cards land on their enumerate slots (1 and 3)."""
    clip1 = _clip_bytes("L1")
    clip2 = _clip_bytes("L2")
    storage = FakeStorage({"L1.wav": clip1, "L2.wav": clip2})
    asr = FakeAsr(
        {
            clip1: _payload([_word("Alpha", 0.0, 0.6), _word("bravo", 0.6, 1.2)]),
            clip2: _payload([_word("Charlie", 0.0, 0.6), _word("delta", 0.6, 1.2)]),
        }
    )
    final_shots = [
        {"title_card": "Intro"},  # index 0
        {"line_id": "L1", "text": "Alpha bravo"},  # index 1
        {"transition": "wipe"},  # index 2
        {"line_id": "L2", "text": "Charlie delta"},  # index 3
    ]
    clip_refs = {
        "L1": ClipRef(
            line_id="L1",
            text="Alpha bravo",
            narrate_stage="promo_narrate",
            clip_filename="L1.wav",
            seconds=2.0,
        ),
        "L2": ClipRef(
            line_id="L2",
            text="Charlie delta",
            narrate_stage="promo_narrate",
            clip_filename="L2.wav",
            seconds=2.0,
        ),
    }

    result = await build_shot_captions(
        final_shots=final_shots, clip_refs=clip_refs, job_id="j1", storage=storage, asr=asr
    )

    assert result is not None
    indexed = _by_index(result)
    assert set(indexed) == {1, 3}
    assert "Alpha" in _all_text(indexed[1])
    assert "bravo" in _all_text(indexed[1])
    assert "Charlie" in _all_text(indexed[3])
    assert "delta" in _all_text(indexed[3])


# ── 7. per-word timings ride the payload for karaoke ────────────────


async def test_cue_payload_carries_per_word_timings() -> None:
    """Each emitted cue carries ``words[]`` — the per-word shot-local timings the
    compiler consumes for karaoke ``{\\k}``. Words are the aligned SCRIPT surface
    (never the ASR spelling), in order, inside the clip window and their cue."""
    clip = _clip_bytes("L1")
    storage = FakeStorage({"L1.wav": clip})
    asr = FakeAsr(
        {
            clip: _payload(
                [
                    _word("Deploy", 0.0, 0.4),
                    _word("the", 0.4, 0.6),
                    _word("mahdel", 0.6, 1.1),  # ASR substitution — word text must be SCRIPT
                    _word("today", 1.1, 1.6),
                ]
            )
        }
    )
    final_shots = [{"line_id": "L1", "text": "Deploy the model today"}]
    clip_refs = {
        "L1": ClipRef(
            line_id="L1",
            text="Deploy the model today",
            narrate_stage="promo_narrate",
            clip_filename="L1.wav",
            seconds=2.0,
        )
    }

    result = await build_shot_captions(
        final_shots=final_shots, clip_refs=clip_refs, job_id="j1", storage=storage, asr=asr
    )

    assert result is not None
    all_words = [w for cue in result[0]["cues"] for w in cue["words"]]
    # the SCRIPT words, in order — 'mahdel' (ASR) must never leak into words[]
    assert [w["text"] for w in all_words] == ["Deploy", "the", "model", "today"]
    eps = 1e-6
    for w in all_words:
        assert 0.0 - eps <= w["start"] < w["end"] <= 2.0 + eps  # shot-local, within the clip
    # each cue's words sit inside its own [start, end) window (karaoke can't desync)
    for cue in result[0]["cues"]:
        assert cue["words"], "a cue lost its per-word timings"
        assert cue["words"][0]["start"] >= cue["start"] - eps
        assert cue["words"][-1]["end"] <= cue["end"] + eps


async def test_zero_width_interpolated_word_is_dropped_from_payload() -> None:
    """A script word with no ASR match, wedged between two ABUTTING ASR words,
    interpolates to zero width (start == end). It must NOT reach ``words[]`` — the
    compiler's CaptionWord requires end > start and would 422 the compile (which
    runs OUTSIDE the caption soft-fail guard), hard-failing the whole render.

    The dropped word still appears in the cue TEXT (script surface); only its
    karaoke highlight is skipped."""
    clip = _clip_bytes("L1")
    storage = FakeStorage({"L1.wav": clip})
    # ASR only has "alpha" and "gamma", abutting at 0.5s; script has "beta" between
    # them → beta is un-timed ("pending") and interpolates into t0==t1==0.5 (width 0).
    asr = FakeAsr({clip: _payload([_word("alpha", 0.0, 0.5), _word("gamma", 0.5, 1.0)])})
    final_shots = [{"line_id": "L1", "text": "alpha beta gamma"}]
    clip_refs = {
        "L1": ClipRef(
            line_id="L1",
            text="alpha beta gamma",
            narrate_stage="promo_narrate",
            clip_filename="L1.wav",
            seconds=2.0,
        )
    }

    result = await build_shot_captions(
        final_shots=final_shots, clip_refs=clip_refs, job_id="j1", storage=storage, asr=asr
    )

    assert result is not None
    all_words = [w for cue in result[0]["cues"] for w in cue["words"]]
    # EVERY emitted word satisfies the compiler's CaptionWord contract.
    for w in all_words:
        assert w["end"] > w["start"], f"zero/negative-width word leaked: {w}"
        assert w["start"] >= 0.0
    # the zero-width "beta" was dropped from words[]…
    assert "beta" not in {w["text"] for w in all_words}
    # …but is still present in the caption TEXT (the script surface is intact).
    assert "beta" in " ".join(c["text"] for c in result[0]["cues"])


# ── 8. Whisper's server-nullable ``probability`` must not fail the boundary ──


async def test_null_probability_word_validates_at_the_boundary() -> None:
    """Whisper legitimately emits ``probability: null`` for some words.

    The boundary schema (``TranscriptWord.probability``) must accept ``None`` so
    ``TranscriptionResult.model_validate`` does NOT raise at
    ``transcription_worker_client`` — otherwise ``word_timed_captions`` silently
    downgrades the whole clip from word-timed to phrase captions. REVERT-PROOF:
    with ``probability: float`` (non-nullable) restored, this raises ValidationError.
    """
    result = TranscriptionResult.model_validate(
        {
            "backend": "whisper",
            "model": "test",
            "duration": 0.0,
            "text": "",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "text": "",
                    "words": [
                        {"start": 0.0, "end": 0.5, "word": "Deploy", "probability": None},
                        {"start": 0.5, "end": 1.0, "word": "now", "probability": 0.97},
                    ],
                }
            ],
        }
    )
    words = result.segments[0].words
    assert words[0].probability is None  # the null word validated, not rejected
    assert words[1].probability == 0.97  # a real confidence still rides through


async def test_null_probability_word_still_yields_word_timed_cue() -> None:
    """A null-probability ASR word must still produce a word-timed cue.

    Downstream (``caption_reconcile``) folds a missing probability to 1.0, so the
    clip stays word-timed instead of soft-failing to phrase captions.
    """
    clip = _clip_bytes("L1")
    storage = FakeStorage({"L1.wav": clip})
    asr = FakeAsr(
        {
            clip: _payload(
                [
                    {"start": 0.0, "end": 0.5, "word": "Deploy", "probability": None},
                    {"start": 0.5, "end": 1.0, "word": "now", "probability": 0.97},
                ]
            )
        }
    )
    final_shots = [{"line_id": "L1", "text": "Deploy now"}]
    clip_refs = {
        "L1": ClipRef(
            line_id="L1",
            text="Deploy now",
            narrate_stage="promo_narrate",
            clip_filename="L1.wav",
            seconds=2.0,
        )
    }

    result = await build_shot_captions(
        final_shots=final_shots, clip_refs=clip_refs, job_id="j1", storage=storage, asr=asr
    )

    assert result is not None
    all_words = [w for cue in result[0]["cues"] for w in cue["words"]]
    # word-timed, not phrase-downgraded: the null-probability word carries a cue.
    assert [w["text"] for w in all_words] == ["Deploy", "now"]


# ── 11. tts_backend word timings REPLACE the ASR call ───────────────
#
# The builder used to transcribe every synthesized line to recover word times.
# On mini2 that call returned 200 only after 366s-1990s against a 300s client
# timeout — it could not complete. tts_backend now measures the spans while it
# synthesizes and stores them in the clip's ``{clip_id}.json`` sidecar, so the
# transcription is not tuned, it is GONE. These tests pin that: the fast path
# must not touch ASR at all, and anything wrong with the sidecar must fall back
# rather than emit captions with holes in them.


class ExplodingAsr:
    """Any call is a test failure: the sidecar path must not transcribe."""

    def __init__(self) -> None:
        self.calls = 0

    async def transcribe_file(
        self,
        *,
        file_path: Path,
        word_timestamps: bool = True,
        content_type: str = "audio/wav",
    ) -> TranscriptionResult:
        self.calls += 1
        raise AssertionError("ASR was called even though word_timings were available")


def _sidecar(words: list[tuple[str, float, float]]) -> bytes:
    import json as _json

    return _json.dumps(
        {
            "seconds": 2.0,
            "format": "wav",
            "word_timings": [{"word": w, "start": s, "end": e} for (w, s, e) in words],
        }
    ).encode("utf-8")


async def test_sidecar_word_timings_are_used_and_asr_is_never_called() -> None:
    """With a timings sidecar the builder emits cues without transcribing.

    Also proves it never even downloads the wav: FakeStorage has no ``L1.wav``
    entry, so a builder that still fetched audio would take the storage-miss
    branch and return None.
    """
    storage = FakeStorage(
        {"L1.json": _sidecar([("Deploy", 0.0, 0.4), ("the", 0.4, 0.6), ("model", 0.6, 1.1)])}
    )
    asr = ExplodingAsr()
    final_shots = [{"line_id": "L1", "text": "Deploy the model"}]
    clip_refs = {
        "L1": ClipRef(
            line_id="L1",
            text="Deploy the model",
            narrate_stage="promo_narrate",
            clip_filename="L1.wav",
            seconds=2.0,
        )
    }

    result = await build_shot_captions(
        final_shots=final_shots, clip_refs=clip_refs, job_id="j1", storage=storage, asr=asr
    )

    assert result is not None
    assert asr.calls == 0
    text = _all_text(result[0])
    for script_word in ("Deploy", "the", "model"):
        assert script_word in text
    assert not any(name.endswith(".wav") for (_j, _s, name) in storage.calls)


async def test_malformed_sidecar_falls_back_to_asr() -> None:
    """A sidecar that is not usable JSON must degrade to ASR, not raise or skip.

    The old behaviour has to remain reachable: clips synthesized by the ONNX
    engine, and every clip written before this key existed, have no timings.
    """
    clip = _clip_bytes("L1")
    storage = FakeStorage({"L1.wav": clip, "L1.json": b"{not json"})
    asr = FakeAsr({clip: _payload([_word("Deploy", 0.0, 0.4), _word("now", 0.4, 0.9)])})
    final_shots = [{"line_id": "L1", "text": "Deploy now"}]
    clip_refs = {
        "L1": ClipRef(
            line_id="L1",
            text="Deploy now",
            narrate_stage="promo_narrate",
            clip_filename="L1.wav",
            seconds=1.5,
        )
    }

    result = await build_shot_captions(
        final_shots=final_shots, clip_refs=clip_refs, job_id="j1", storage=storage, asr=asr
    )

    assert result is not None
    assert "Deploy" in _all_text(result[0])


async def test_sidecar_without_timings_key_falls_back_to_asr() -> None:
    """A pre-existing sidecar (seconds/loudness only) must not shortcut the ASR path."""
    import json as _json

    clip = _clip_bytes("L1")
    storage = FakeStorage(
        {"L1.wav": clip, "L1.json": _json.dumps({"seconds": 1.5, "format": "wav"}).encode()}
    )
    asr = FakeAsr({clip: _payload([_word("Deploy", 0.0, 0.4), _word("now", 0.4, 0.9)])})
    final_shots = [{"line_id": "L1", "text": "Deploy now"}]
    clip_refs = {
        "L1": ClipRef(
            line_id="L1",
            text="Deploy now",
            narrate_stage="promo_narrate",
            clip_filename="L1.wav",
            seconds=1.5,
        )
    }

    result = await build_shot_captions(
        final_shots=final_shots, clip_refs=clip_refs, job_id="j1", storage=storage, asr=asr
    )

    assert result is not None
    assert "Deploy" in _all_text(result[0])
