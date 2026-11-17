"""audiobook_segment — mechanical span split + LLM speaker attribution + casting.

Covers the deterministic core (split_spans, coalescing, monologue split) without
the LLM, and the span-mode stage/cast wiring with a mocked WM worker.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

from w2v_common.clients import WMResponse

from src.services.runs.run_context import RunContext
from src.services.stages.audiobook.cast import AudiobookCastStage
from src.services.stages.audiobook.parsing import split_spans
from src.services.stages.audiobook.segment import AudiobookSegmentStage, coalesce_units
from src.services.stages.audiobook.voice_catalog import VOICE_CATALOG


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


# --- deterministic core ------------------------------------------------------


def test_split_spans_separates_dialogue_and_narration() -> None:
    spans = split_spans('He walked in. "Hello there," said Anna. She left.')
    kinds = [(s.is_dialogue, s.text) for s in spans]
    assert kinds == [
        (False, "He walked in."),
        (True, "Hello there,"),  # quote chars stripped
        (False, "said Anna. She left."),
    ]


def test_split_spans_no_quotes_is_single_narration() -> None:
    spans = split_spans("The sun rose over the quiet hill.")
    assert len(spans) == 1
    assert spans[0].is_dialogue is False


def test_split_spans_handles_curly_quotes() -> None:
    spans = split_spans("She said “Go now” and vanished.")
    assert [(s.is_dialogue, s.text) for s in spans] == [
        (False, "She said"),
        (True, "Go now"),
        (False, "and vanished."),
    ]


def test_coalesce_merges_adjacent_same_speaker_keeps_cross_speaker() -> None:
    units = [
        {"text": "A", "speaker": "Narrator"},
        {"text": "B", "speaker": "Narrator"},
        {"text": "C", "speaker": "Anna"},
        {"text": "D", "speaker": "Narrator"},
    ]
    out = coalesce_units(units)
    assert out == [
        {"text": "A B", "speaker": "Narrator"},  # merged
        {"text": "C", "speaker": "Anna"},  # short cross-speaker preserved
        {"text": "D", "speaker": "Narrator"},
    ]


# --- stage wiring ------------------------------------------------------------


def _ingest_ctx(granularity: str) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "voice": {"voice_id": "af_heart", "speed": 1.0},
            "brief": {"acting_granularity": granularity},
        },
        prior_run_outputs={
            "audiobook_ingest": {
                "chapters": [
                    {
                        "id": "ch1",
                        "title": "One",
                        "sequence": 0,
                        "text": 'The room was quiet. "I quit," said Bob. He stood.',
                    }
                ]
            }
        },
    )


async def test_segment_noop_unless_span() -> None:
    stage = AudiobookSegmentStage(client=AsyncMock())
    out = await stage.run({}, ctx=_ingest_ctx("chunk"))
    assert out.chapters == [] and out.roster == [] and out.total_spans == 0


async def test_segment_span_attributes_dialogue_and_builds_roster() -> None:
    fake = AsyncMock()
    # spans: [0]narration [1]dialogue("I quit,") [2]narration → attribute idx 1
    fake.message.return_value = _resp({"attributions": [{"index": 1, "speaker": "Bob"}]})
    out = await AudiobookSegmentStage(client=fake).run({}, ctx=_ingest_ctx("span"))

    assert out.roster == ["Bob"]
    units = out.chapters[0].chunks
    by_speaker = {u.speaker for u in units}
    assert "Bob" in by_speaker and "Narrator" in by_speaker
    bob = next(u for u in units if u.speaker == "Bob")
    assert bob.text == "I quit,"
    # sequence is monotonic across the chapter
    assert [u.sequence for u in units] == list(range(len(units)))


async def test_segment_attribution_failure_falls_back_to_narrator() -> None:
    fake = AsyncMock()
    fake.message.side_effect = RuntimeError("worker down")
    out = await AudiobookSegmentStage(client=fake).run({}, ctx=_ingest_ctx("span"))
    # every unit narrated (single voice) rather than the stage failing
    assert all(u.speaker == "Narrator" for u in out.chapters[0].chunks)
    assert out.roster == []


# --- cast span mode ----------------------------------------------------------


def _attribute_wm() -> AsyncMock:
    """Fake WM: attributes every DIALOGUE index present in the batch message to Bob,
    so the merged result is verified against the GLOBAL indices the batches carry."""
    wm = AsyncMock()

    def _side(*, message: str, **_kw: Any) -> WMResponse:
        idxs = [int(m) for m in re.findall(r"\[(\d+)\] DIALOGUE", message)]
        return _resp({"attributions": [{"index": i, "speaker": "Bob"} for i in idxs]})

    wm.message.side_effect = _side
    return wm


async def test_attribute_batches_spans_and_merges_global_indices() -> None:
    # 90 dialogue spans > _SPAN_ATTRIBUTION_BATCH (40) ⇒ the WM call is BATCHED
    # (was one unbounded call → the #718 timeout). Global indices merge correctly.
    spans = [SimpleNamespace(is_dialogue=True, text=f"line {i}") for i in range(90)]
    wm = _attribute_wm()
    stage = AudiobookSegmentStage(client=wm)
    stage._llm_retries = 0
    result = await stage._attribute(spans, ["Bob"])
    assert wm.message.call_count == 3  # 40 + 40 + 10
    assert result == ["Bob"] * 90  # every span attributed via its global index


async def test_attribute_batch_failure_degrades_only_that_window() -> None:
    # A batch that fails leaves ONLY its window as Narrator; the others still
    # attribute (partial-survives), instead of the whole book falling back.
    spans = [SimpleNamespace(is_dialogue=True, text=f"line {i}") for i in range(90)]
    wm = AsyncMock()

    def _side(*, message: str, **_kw: Any) -> WMResponse:
        if "[0] " in message:  # the first window (contains global index 0) always fails
            raise RuntimeError("worker down")
        idxs = [int(m) for m in re.findall(r"\[(\d+)\] DIALOGUE", message)]
        return _resp({"attributions": [{"index": i, "speaker": "Bob"} for i in idxs]})

    wm.message.side_effect = _side
    stage = AudiobookSegmentStage(client=wm)
    stage._llm_retries = 0
    result = await stage._attribute(spans, ["Bob"])
    assert result[0] == "Narrator" and result[39] == "Narrator"  # failed first window
    assert result[40] == "Bob" and result[89] == "Bob"  # later windows survived


async def test_cast_ignores_malformed_voice_cast() -> None:
    # brief.voice_cast as a truthy non-dict (a list) must NOT AttributeError out of
    # cast via `.items()`; it degrades to no explicit casting.
    fake = AsyncMock()
    fake.message.return_value = _resp({"casting": [{"speaker": "Bob", "voice_id": "am_adam"}]})
    ctx = _cast_ctx()
    ctx.config_snapshot["brief"]["voice_cast"] = ["Bob", "am_adam"]  # malformed (list)
    out = await AudiobookCastStage(client=fake).run({}, ctx=ctx)  # must NOT raise
    assert out is not None


def _cast_ctx() -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "voice": {"voice_id": "af_heart", "speed": 1.0},
            "brief": {"acting_granularity": "span"},
        },
        prior_run_outputs={
            "audiobook_plan": {
                "chapters": [{"id": "ch1", "title": "One", "sequence": 0, "chunks": []}]
            },
            "audiobook_segment": {
                "roster": ["Bob"],
                "total_spans": 3,
                "chapters": [
                    {
                        "id": "ch1",
                        "title": "One",
                        "sequence": 0,
                        "chunks": [
                            {
                                "id": "u0",
                                "text": "The room was quiet.",
                                "sequence": 0,
                                "speaker": "Narrator",
                            },
                            {"id": "u1", "text": "I quit,", "sequence": 1, "speaker": "Bob"},
                            {"id": "u2", "text": "He stood.", "sequence": 2, "speaker": "Narrator"},
                        ],
                    }
                ],
            },
        },
    )


async def test_cast_span_maps_roster_to_voices_and_covers_every_unit() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp({"casting": [{"speaker": "Bob", "voice_id": "am_adam"}]})
    out = await AudiobookCastStage(client=fake).run({}, ctx=_cast_ctx())

    perf = {p.chunk_id: p for p in out.performances}
    assert set(perf) == {"u0", "u1", "u2"}  # a performance per span
    assert perf["u1"].voice_id == "am_adam" and perf["u1"].speaker == "Bob"
    assert perf["u0"].voice_id == "af_heart"  # Narrator = project voice


async def test_cast_span_deterministic_fallback_when_llm_fails() -> None:
    fake = AsyncMock()
    fake.message.side_effect = RuntimeError("worker down")
    out = await AudiobookCastStage(client=fake).run({}, ctx=_cast_ctx())
    bob = next(p for p in out.performances if p.speaker == "Bob")
    # a real, distinct catalog voice (not the narrator) assigned deterministically
    assert bob.voice_id in VOICE_CATALOG and bob.voice_id != "af_heart"


async def test_total_attribution_outage_degrades_the_run_instead_of_shipping_single_voice() -> None:
    """A WM outage (402 "configure your LLM provider", schema rejection, dead worker)
    makes EVERY attribution batch soft-fail to {}, so `roster` is empty and every
    span is voiced by the Narrator. Before this fix the output carried no degrade
    channel at all, so `runs/_shared.py` reported a clean `completed` run and the
    customer received a fully single-voice "voice-acted" audiobook with no signal
    anywhere. The stage must say so in `notes`, which is what marks the run degraded.
    """
    fake = AsyncMock()
    fake.message.side_effect = RuntimeError("402 configure your LLM provider")
    out = await AudiobookSegmentStage(client=fake).run({}, ctx=_ingest_ctx("span"))

    assert out.roster == []
    assert all(u.speaker == "Narrator" for u in out.chapters[0].chunks)
    assert out.notes, "a total attribution outage must surface on the notes degrade channel"
    joined = " ".join(out.notes).lower()
    assert "narrator voice" in joined
    # notes must survive the dump — that dict is what runs/_shared.py reads.
    assert out.model_dump()["notes"] == out.notes


async def test_partial_attribution_outage_is_noted_but_roster_survives() -> None:
    """One failed window out of three is still a real loss of voice acting for that
    window, so it is noted — but the run is not described as fully single-voice."""
    spans_text = " ".join(f'She spoke. "line {i}," he said.' for i in range(60))
    ctx = _ingest_ctx("span")
    ctx.prior_run_outputs["audiobook_ingest"]["chapters"][0]["text"] = spans_text

    wm = AsyncMock()
    calls = {"n": 0}

    def _side(*, message: str, **_kw: Any) -> WMResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("worker down")
        idxs = [int(m) for m in re.findall(r"\[(\d+)\] DIALOGUE", message)]
        return _resp({"attributions": [{"index": i, "speaker": "Bob"} for i in idxs]})

    wm.message.side_effect = _side
    stage = AudiobookSegmentStage(client=wm)
    stage._llm_retries = 0
    out = await stage.run({}, ctx=ctx)

    assert out.roster == ["Bob"], "later windows still attribute"
    joined = " ".join(out.notes)
    assert "attribution batches failed" in joined
    assert "ENTIRE audiobook" not in joined


async def test_no_dialogue_and_chunk_mode_stay_clean() -> None:
    """The degrade channel must not cry wolf: the off/chunk no-op path and a book
    with no dialogue at all are legitimately single-voice and stay un-degraded."""
    assert (
        await AudiobookSegmentStage(client=AsyncMock()).run({}, ctx=_ingest_ctx("chunk"))
    ).notes == []

    ctx = _ingest_ctx("span")
    ctx.prior_run_outputs["audiobook_ingest"]["chapters"][0]["text"] = "The sun rose. It set."
    wm = AsyncMock()
    out = await AudiobookSegmentStage(client=wm).run({}, ctx=ctx)
    wm.message.assert_not_awaited()
    assert out.roster == [] and out.notes == []
