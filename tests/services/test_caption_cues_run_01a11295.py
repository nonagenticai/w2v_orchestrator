"""Burned-in caption cues of promo run 01a11295 (2026-10-06, REJECTED at frame QA).

Two defects, both on the run's real data: the script lines are the run's
``promo_script`` lines verbatim, and the word timings are the run's own
``promo_narrate`` sidecars (``{clip_id}.json`` ``word_timings``, written by
tts_backend), read from MinIO. They go through the live path — the stored
sidecar, :func:`build_shot_captions`, the script/word alignment and the cue
packer — exactly as ``CompileStageBase`` builds the compiler's ``captions``.

1. Cue breaks. The greedy packer filled each cue to its 2 x 42 characters and
   cut wherever the next word did not fit: "…are made by deterministic code,
   not" / "language models." and "…respects robots.txt, avoiding scraped or" /
   "bought lists." A cue now breaks on a clause boundary, never ends on a
   function word and never strands a 1-2 word tail.
2. Cue timing. A cue ends on its last spoken word. Between two cues of one
   sentence it holds at most ``CUE_TAIL_SECONDS`` (never past the next cue's
   start), so the box does not blink off for the 0.125 s pause after "code,";
   the last cue of a line ends on the last word — never in the 2.0 s hold after
   the narration. (Measured on the delivered film: "prohibited to automatic."
   is gone by 11.8 s with the narration ending at 11.73 s — the clip end the
   sidecar's last word runs to. The "still on screen at 12 s" reading was taken
   against Whisper segment times, which lag the audio.)
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import pytest

from src.clients.schemas.transcription_worker import TranscriptionResult
from src.services import caption_cues
from src.services.caption_cues import CUE_TAIL_SECONDS
from src.services.word_timed_captions import ClipRef, build_shot_captions

# (line id, promo_script text verbatim, clip seconds, sidecar word timings).
_L2 = (
    "l2",
    "Users can set graduated autonomy levels for 30+ capabilities, ranging from "
    "prohibited to automatic.",
    6.955,
    [
        ("Users", 0.0247, 0.5497),
        ("can", 0.5497, 0.7622),
        ("set", 0.7622, 1.0372),
        ("graduated", 1.0372, 1.7372),
        ("autonomy", 1.7372, 2.2747),
        ("levels", 2.2747, 2.7247),
        ("for", 2.7247, 2.8997),
        ("thirty+", 2.8997, 3.5247),
        ("capabilities", 3.5247, 4.5497),
        ("ranging", 4.6872, 5.1997),
        ("from", 5.1997, 5.3997),
        ("prohibited", 5.3997, 6.0122),
        ("to", 6.0122, 6.1872),
        ("automatic", 6.1872, 6.955),
    ],
)
_L3 = (
    "l3",
    "Critical financial and consent decisions are made by deterministic code, not language models.",
    5.982333333333333,
    [
        ("Critical", 0.0089, 0.4339),
        ("financial", 0.4339, 1.1214),
        ("and", 1.1214, 1.2589),
        ("consent", 1.2589, 1.6714),
        ("decisions", 1.6714, 2.5089),
        ("are", 2.5089, 2.6839),
        ("made", 2.6839, 2.9089),
        ("by", 2.9089, 3.0839),
        ("deterministic", 3.0839, 3.9464),
        ("code", 3.9464, 4.6339),
        ("not", 4.7589, 4.9839),
        ("language", 4.9839, 5.3839),
        ("models", 5.3839, 5.9823),
    ],
)
_L5 = (
    "l5",
    "The system uses only public sources and respects robots.txt, avoiding scraped or "
    "bought lists.",
    6.579875,
    [
        ("The", 0.0, 0.1034),
        ("system", 0.1034, 0.4659),
        ("uses", 0.4659, 0.8284),
        ("only", 0.8284, 1.1409),
        ("public", 1.1409, 1.5409),
        ("sources", 1.5409, 2.1784),
        ("and", 2.1784, 2.3034),
        ("respects", 2.3034, 2.7909),
        ("robots", 2.7909, 3.3034),
        ("dot", 3.3034, 3.6034),
        ("T", 3.6034, 3.8659),
        ("X", 3.8659, 4.1284),
        ("T", 4.1284, 4.6659),
        ("avoiding", 4.8284, 5.2784),
        ("scraped", 5.2784, 5.7409),
        ("or", 5.7409, 5.8659),
        ("bought", 5.8659, 6.1034),
        ("lists", 6.1034, 6.5799),
    ],
)
_LINES = [_L2, _L3, _L5]
_FUNCTION_WORDS = {"not", "or", "and", "to", "of", "the", "a", "by", "for", "from", "with"}


class _Storage:
    """The run's narrate sidecars, served by key exactly as storage serves them."""

    def __init__(self) -> None:
        self._objects = {
            f"{line_id}.json": json.dumps(
                {
                    "seconds": seconds,
                    "format": "wav",
                    "word_timings": [{"word": w, "start": s, "end": e} for w, s, e in words],
                }
            ).encode()
            for line_id, _text, seconds, words in _LINES
        }

    async def retrieve_stage_artifact(
        self, job_id: str, stage_name: str, filename: str
    ) -> bytes | None:
        return self._objects.get(filename)


class _NoAsr:
    async def transcribe_file(
        self, *, file_path: Path, word_timestamps: bool = True, content_type: str = "audio/wav"
    ) -> TranscriptionResult:
        raise AssertionError("the sidecar timings must be used")


async def _cues() -> dict[str, list[dict[str, object]]]:
    """``line_id -> cues`` as the compile sends them to video_compiler."""
    refs = {
        line_id: ClipRef(
            line_id=line_id,
            text=text,
            narrate_stage="promo_narrate",
            clip_filename=f"{line_id}.wav",
            seconds=seconds,
        )
        for line_id, text, seconds, _words in _LINES
    }
    shots = [{"line_id": line_id} for line_id, *_rest in _LINES]
    built = await build_shot_captions(
        final_shots=shots, clip_refs=refs, job_id="01a11295", storage=_Storage(), asr=_NoAsr()
    )
    assert built is not None
    return {
        str(shots[entry["shot_index"]]["line_id"]): [dict(cue) for cue in entry["cues"]]
        for entry in built
    }


def _texts(cues: list[dict[str, object]]) -> list[str]:
    return [str(cue["text"]) for cue in cues]


async def test_cues_break_on_the_clause_never_mid_clause() -> None:
    """The two sentences QA rejected, and line 2: each cue ends on its comma."""
    cues = await _cues()

    assert _texts(cues["l3"]) == [
        "Critical financial and consent decisions are made by deterministic code,",
        "not language models.",
    ]
    assert _texts(cues["l5"]) == [
        "The system uses only public sources and respects robots.txt,",
        "avoiding scraped or bought lists.",
    ]
    assert _texts(cues["l2"]) == [
        "Users can set graduated autonomy levels for 30+ capabilities,",
        "ranging from prohibited to automatic.",
    ]


async def test_no_cue_ends_on_a_function_word_or_strands_a_short_tail() -> None:
    for line_id, line_cues in (await _cues()).items():
        texts = _texts(line_cues)
        for text in texts[:-1]:
            last = text.split()[-1].strip(",;:.").lower()
            assert last not in _FUNCTION_WORDS, (line_id, text)
        if len(texts) > 1:
            assert len(texts[-1].split()) > 2, (line_id, texts[-1])
        for text in texts:  # still inside the 2 x 42 the compiler wraps to
            assert len(caption_cues._wrap(text.split(), 42)) <= 2, text


async def test_a_cue_ends_on_its_last_word_and_never_overlaps_the_next() -> None:
    """Inside a line: last word + at most the fixed tail, never past the next cue.
    The line's last cue ends on its last word — the clip's end, never the hold."""
    cues = await _cues()
    for line_id, _text, seconds, _words in _LINES:
        line_cues = cues[line_id]
        for cue, nxt in pairwise(line_cues):
            words = cue["words"]
            assert isinstance(words, list)
            last_end = float(words[-1]["end"])
            end = float(str(cue["end"]))
            assert last_end <= end <= last_end + CUE_TAIL_SECONDS + 1e-9
            assert end <= float(str(nxt["start"])) + 1e-9  # no overlap
        closing = line_cues[-1]
        words = closing["words"]
        assert isinstance(words, list)
        assert float(str(closing["end"])) == pytest.approx(float(words[-1]["end"]))
        assert float(str(closing["end"])) <= seconds + 1e-3  # fixture rounded to 0.1 ms


async def test_the_pause_after_code_is_bridged_not_blinked() -> None:
    """The pause after "code," (4.634 s; "not" starts at 4.759 s): the first cue
    holds to the second cue's start instead of leaving 0.125 s with no caption."""
    first, second = (await _cues())["l3"]

    assert first["end"] == pytest.approx(4.7589)
    assert second["start"] == pytest.approx(4.7589)


# ── the two rules on their own (no comma to lean on) ─────────────────────────


def _pack(sentence: str) -> list[str]:
    """One-line cues of 42 characters — the run's data never forces these rules."""
    from src.services.caption_reconcile import TimedToken

    tokens = [
        TimedToken(text=word, start=i * 0.3, end=(i + 1) * 0.3, source="asr")
        for i, word in enumerate(sentence.split())
    ]
    return [" ".join(cue.lines) for cue in caption_cues.group_tokens_into_cues(tokens, max_lines=1)]


def test_a_cue_never_ends_on_a_function_word() -> None:
    """Greedy filling ends the first cue on "and" (it fits); the packer moves it."""
    assert _pack("Teams set autonomy for each capability and the platform enforces it.") == [
        "Teams set autonomy for each capability",
        "and the platform enforces it.",
    ]


def test_a_cue_never_strands_a_two_word_tail() -> None:
    assert _pack("Merqi drafts each campaign brief overnight while sleeping.") == [
        "Merqi drafts each campaign brief",
        "overnight while sleeping.",
    ]
