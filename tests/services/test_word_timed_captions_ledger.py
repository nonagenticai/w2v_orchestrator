"""The ASR FALLBACK of ``build_shot_captions`` is paid for once per clip, not per entry.

The preferred path reads word timings tts_backend already measured (free). The
fallback transcribes the clip — measured at 366-1990s per clip on mini2 — and was
uncached, so a pod death mid-compile re-transcribed every clip the dead attempt
had already finished. The fallback now goes through the step ledger, keyed on the
clip and hashed over the clip's BYTES.

``build_shot_captions`` soft-fails each clip (``except Exception`` → skip), so a
:class:`SimulatedPodDeath` is swallowed and ``run_with_reentry`` cannot drive it.
The two attempts are run by hand over one shared store; the dead attempt's return
value is discarded as a dead pod's would be.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from src.clients.schemas.transcription_worker import TranscriptionResult
from src.services.stages.shared.step_ledger import StepLedger
from src.services.word_timed_captions import ClipRef, build_shot_captions
from tests.services.test_word_timed_captions import (
    FakeStorage,
    _all_text,
    _by_index,
    _clip_bytes,
    _payload,
    _word,
)
from tests.support.reentry import InMemoryStepStore, Outbound

pytestmark = pytest.mark.asyncio

_STAGE = "promo_render"


class _CountingAsr:
    """ASR fake whose transcriptions go through an :class:`Outbound` (count + kill)."""

    def __init__(self, out: Outbound, payloads: dict[bytes, TranscriptionResult]) -> None:
        self.transcribed: list[bytes] = []

        def _run(*, file_path: Path, **_kwargs: object) -> TranscriptionResult:
            data = Path(file_path).read_bytes()
            self.transcribed.append(data)
            return payloads[data]

        self._call = out.fake("asr.transcribe_file", _run)

    async def transcribe_file(
        self,
        *,
        file_path: Path,
        word_timestamps: bool = True,
        content_type: str = "audio/wav",
    ) -> TranscriptionResult:
        result = await self._call(
            file_path=file_path, word_timestamps=word_timestamps, content_type=content_type
        )
        assert isinstance(result, TranscriptionResult)
        return result


def _ref(line_id: str, text: str) -> ClipRef:
    return ClipRef(
        line_id=line_id,
        text=text,
        narrate_stage="promo_narrate",
        clip_filename=f"{line_id}.wav",
        seconds=2.0,
    )


_SHOTS = [{"line_id": "L1"}, {"line_id": "L2"}]
_REFS = {"L1": _ref("L1", "Hello world"), "L2": _ref("L2", "Ship it")}
_L1 = _clip_bytes("L1")
_L2 = _clip_bytes("L2")
_PAYLOADS = {
    _L1: _payload([_word("Hello", 0.0, 0.6), _word("world", 0.6, 1.2)]),
    _L2: _payload([_word("Ship", 0.0, 0.5), _word("it", 0.5, 0.9)]),
}


async def test_fallback_transcribes_each_clip_once_across_death_and_reentry() -> None:
    out = Outbound()
    asr = _CountingAsr(out, _PAYLOADS)
    storage = FakeStorage({"L1.wav": _L1, "L2.wav": _L2})
    store = InMemoryStepStore()
    run_id = uuid4()

    out.arm(1)  # one clip's transcription finishes; the other one kills the pod
    dead = await build_shot_captions(
        final_shots=_SHOTS,
        clip_refs=_REFS,
        job_id="j1",
        storage=storage,
        asr=asr,
        ledger=StepLedger(store, run_id, _STAGE),
    )
    out.disarm()
    assert dead is not None and len(dead) == 1, "exactly one clip should have survived attempt 1"
    assert out.calls["asr.transcribe_file"] == 1
    finished = asr.transcribed[0]

    survivor = await build_shot_captions(
        final_shots=_SHOTS,
        clip_refs=_REFS,
        job_id="j1",
        storage=storage,
        asr=asr,
        ledger=StepLedger(store, run_id, _STAGE),  # a fresh ledger, as on a fresh pod
    )

    assert out.calls["asr.transcribe_file"] == 2, (
        "2 clips ⇒ 2 transcriptions; a third means the re-entry re-transcribed a finished clip"
    )
    assert asr.transcribed.count(finished) == 1
    assert survivor is not None
    by_index = _by_index(survivor)
    assert _all_text(by_index[0]) == "Hello world"
    assert _all_text(by_index[1]) == "Ship it"
    # The reused clip's cues are the ones the first attempt built from live ASR.
    assert dead[0] == by_index[dead[0]["shot_index"]]
    assert store.step_keys(_STAGE) == {"asr:promo_narrate/L1.wav", "asr:promo_narrate/L2.wav"}


async def test_sidecar_path_makes_no_ledger_calls() -> None:
    out = Outbound()
    asr = _CountingAsr(out, _PAYLOADS)
    sidecar = json.dumps(
        {
            "word_timings": [
                {"word": "Hello", "start": 0.0, "end": 0.6},
                {"word": "world", "start": 0.6, "end": 1.2},
            ]
        }
    ).encode()
    storage = FakeStorage({"L1.json": sidecar, "L1.wav": _L1})
    store = InMemoryStepStore()

    result = await build_shot_captions(
        final_shots=[{"line_id": "L1"}],
        clip_refs={"L1": _REFS["L1"]},
        job_id="j1",
        storage=storage,
        asr=asr,
        ledger=StepLedger(store, uuid4(), _STAGE),
    )

    assert result is not None
    assert _all_text(result[0]) == "Hello world"
    assert store.gets == []
    assert store.puts == []
    assert out.total == 0


async def test_different_audio_under_the_same_clip_key_is_transcribed_again() -> None:
    out = Outbound()
    renarrated = _clip_bytes("L1-renarrated")
    asr = _CountingAsr(
        out,
        {
            _L1: _PAYLOADS[_L1],
            renarrated: _payload([_word("Hello", 0.0, 0.9), _word("world", 0.9, 1.8)]),
        },
    )
    store = InMemoryStepStore()
    run_id = uuid4()
    refs = {"L1": _REFS["L1"]}

    first = await build_shot_captions(
        final_shots=[{"line_id": "L1"}],
        clip_refs=refs,
        job_id="j1",
        storage=FakeStorage({"L1.wav": _L1}),
        asr=asr,
        ledger=StepLedger(store, run_id, _STAGE),
    )
    second = await build_shot_captions(
        final_shots=[{"line_id": "L1"}],
        clip_refs=refs,
        job_id="j1",
        storage=FakeStorage({"L1.wav": renarrated}),  # same object key, new audio
        asr=asr,
        ledger=StepLedger(store, run_id, _STAGE),
    )

    assert asr.transcribed == [_L1, renarrated]
    assert first is not None and second is not None
    # The new audio's timings are used, not the stored ones for the old audio.
    assert first[0]["cues"][0]["end"] == pytest.approx(1.2)
    assert second[0]["cues"][0]["end"] == pytest.approx(1.8)
