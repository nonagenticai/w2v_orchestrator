"""audiobook_cast — two-phase LLM voice casting (#718).

Since #718 the stage makes TWO kinds of WM call, not one:
  * attribution — per-chunk ``{speaker, speed}`` in BOUNDED batches of
    ``_PERFORMANCE_BATCH_SIZE`` (``_PERFORMANCE_ONLY_SCHEMA``), so the unbounded
    performances array can't time out the small vLLM; and
  * casting — ONE global ``{speaker, voice_id}`` call over the distinct-speaker
    roster (``_CASTING_ONLY_SCHEMA``), so a character can never be cast to two
    colliding voices across batches.
The fake WM below routes on ``output_schema`` to model both, and records batch
sizes / rosters so the tests can prove the split actually splits.
"""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

from src.services.runs.run_context import RunContext
from src.services.stages.audiobook.cast import (
    _PERFORMANCE_BATCH_SIZE,
    _PERFORMANCE_ONLY_SCHEMA,
    AudiobookCastStage,
)
from src.services.stages.audiobook.voice_catalog import VOICE_CATALOG
from src.services.stages.base import StagePreconditionError


def _resp(obj: Any) -> WMResponse:
    # WM returns the structured output already parsed (output is the dict
    # itself, not a JSON string) when output_schema is passed.
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


_CHUNK_ID_RE = re.compile(r'"chunk_id":"([^"]+)"')


def _ids_in(message: str) -> list[str]:
    """Chunk ids the attribution prompt actually carried (order preserved)."""
    return _CHUNK_ID_RE.findall(message)


def _roster_in(message: str) -> list[str]:
    """Character names the single casting prompt listed."""
    parts = message.split("CHARACTERS (assign one voice each):\n")
    if len(parts) < 2:
        return []
    return [line.strip() for line in parts[1].splitlines() if line.strip()]


def _two_phase_wm(
    *,
    perf_specs: dict[str, dict[str, Any]] | None = None,
    cast_voices: dict[str, str] | None = None,
    record: dict[str, list] | None = None,
) -> AsyncMock:
    """A WM fake that answers BOTH call shapes.

    ``perf_specs`` maps chunk_id -> the per-chunk fields the model returns
    ({"speaker", optional "speed"}); a chunk absent from it is *omitted* by the
    model (exercises coverage-fill). Each attribution call only ever returns rows
    for chunk_ids present in that batch's prompt. ``cast_voices`` maps speaker ->
    the voice the model casts; a speaker absent from it is left to the
    deterministic distinct-voice fallback. ``record`` (if given) collects
    ``perf_batches`` (size per attribution call) and ``cast_rosters``.
    """
    specs: dict[str, dict[str, Any]] = perf_specs or {}
    voices: dict[str, str] = cast_voices or {}
    fake = AsyncMock()

    async def _msg(*, message: str, output_schema: Any, **_: Any) -> WMResponse:
        if output_schema is _PERFORMANCE_ONLY_SCHEMA:
            ids = _ids_in(message)
            if record is not None:
                record.setdefault("perf_batches", []).append(len(ids))
            performances = [{"chunk_id": cid, **specs[cid]} for cid in ids if cid in specs]
            return _resp({"performances": performances})
        # casting call
        roster = _roster_in(message)
        if record is not None:
            record.setdefault("cast_rosters", []).append(roster)
        casting = [{"speaker": s, "voice_id": voices[s]} for s in roster if s in voices]
        return _resp({"casting": casting})

    fake.message.side_effect = _msg
    return fake


def _plan(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "chapters": [{"id": "ch1", "title": "One", "sequence": 0, "chunks": chunks}],
        "total_chunks": len(chunks),
    }


def _ctx(
    *,
    voice_acting: bool = True,
    voice_cast: dict | None = None,
    chunks: list[dict[str, Any]] | None = None,
) -> RunContext:
    brief: dict[str, Any] = {}
    if voice_acting:
        brief["voice_acting"] = True
    if voice_cast is not None:
        brief["voice_cast"] = voice_cast
    if chunks is None:
        chunks = [
            {"id": "k1", "text": '"Hello," said Alice.', "sequence": 0},
            {"id": "k2", "text": "The sun rose over the hill.", "sequence": 1},
        ]
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"voice": {"voice_id": "af_heart", "speed": 1.0}, "brief": brief},
        prior_run_outputs={"audiobook_plan": _plan(chunks)},
    )


async def test_cast_assigns_consistent_voice_per_speaker() -> None:
    fake = _two_phase_wm(
        perf_specs={
            "k1": {"speaker": "Alice", "speed": 1.2},
            "k2": {"speaker": "Narrator", "speed": 1.0},
        },
        cast_voices={"Alice": "af_bella"},
    )
    out = await AudiobookCastStage(client=fake).run({}, ctx=_ctx())

    by_chunk = {p.chunk_id: p for p in out.performances}
    assert by_chunk["k1"].voice_id == "af_bella" and by_chunk["k1"].speaker == "Alice"
    # Narrator keeps the PROJECT voice — the model attributes narration but never
    # re-casts the narrator (matches shipped span mode).
    assert by_chunk["k2"].voice_id == "af_heart"
    assert by_chunk["k1"].speed == 1.2
    assert by_chunk["k1"].chapter_id == "ch1"
    assert out.narrator_voice_id == "af_heart"


async def test_cast_repairs_hallucinated_voice_to_valid_catalog_voice() -> None:
    fake = _two_phase_wm(
        perf_specs={"k1": {"speaker": "Alice"}, "k2": {"speaker": "Narrator"}},
        cast_voices={"Alice": "xx_not_a_voice"},  # hallucinated
    )
    out = await AudiobookCastStage(client=fake).run({}, ctx=_ctx())

    by_chunk = {p.chunk_id: p for p in out.performances}
    # Invalid voice repaired to a VALID distinct catalog voice; the hallucinated id
    # can never reach TTS. (It is no longer collapsed to the narrator — a character
    # keeps a distinct voice via the deterministic fallback.)
    assert by_chunk["k1"].voice_id in VOICE_CATALOG
    assert all(p.voice_id in VOICE_CATALOG for p in out.performances)
    assert all(p.voice_id != "xx_not_a_voice" for p in out.performances)


async def test_cast_coverage_fills_missing_chunks() -> None:
    fake = _two_phase_wm(
        perf_specs={"k1": {"speaker": "Alice"}},  # k2 omitted by the model
        cast_voices={"Alice": "af_bella"},
    )
    out = await AudiobookCastStage(client=fake).run({}, ctx=_ctx())

    by_chunk = {p.chunk_id: p for p in out.performances}
    assert set(by_chunk) == {"k1", "k2"}  # every plan chunk covered
    assert by_chunk["k2"].speaker == "Narrator"
    assert by_chunk["k2"].voice_id == "af_heart"


async def test_cast_off_skips_llm_and_returns_empty() -> None:
    fake = AsyncMock()
    out = await AudiobookCastStage(client=fake).run({}, ctx=_ctx(voice_acting=False))

    fake.message.assert_not_called()
    assert out.performances == []
    assert out.casting == []
    assert out.narrator_voice_id == "af_heart"


# pinned, not the LLM's af_bella


async def test_cast_clamps_out_of_range_speed() -> None:
    fake = _two_phase_wm(
        perf_specs={
            "k1": {"speaker": "Narrator", "speed": 9.0},
            "k2": {"speaker": "Narrator", "speed": 0.1},
        },
    )
    out = await AudiobookCastStage(client=fake).run({}, ctx=_ctx())

    by_chunk = {p.chunk_id: p for p in out.performances}
    assert by_chunk["k1"].speed == 2.0
    assert by_chunk["k2"].speed == 0.5
    # All-Narrator: no distinct characters → no casting call is even made.
    assert all(
        kw.kwargs["output_schema"] is _PERFORMANCE_ONLY_SCHEMA
        for kw in fake.message.await_args_list
    )


async def test_cast_retries_malformed_attribution_then_succeeds() -> None:
    fake = AsyncMock()
    calls = {"perf": 0}

    async def _msg(*, output_schema: Any, **_: Any) -> WMResponse:
        if output_schema is _PERFORMANCE_ONLY_SCHEMA:
            calls["perf"] += 1
            if calls["perf"] == 1:
                return _resp({"casting": []})  # malformed: no performances key
            return _resp(
                {
                    "performances": [
                        {"chunk_id": "k1", "speaker": "Narrator"},
                        {"chunk_id": "k2", "speaker": "Narrator"},
                    ]
                }
            )
        return _resp({"casting": []})

    fake.message.side_effect = _msg
    out = await AudiobookCastStage(client=fake).run({}, ctx=_ctx())

    assert calls["perf"] == 2  # malformed attribution draw was re-prompted
    assert len(out.performances) == 2


async def test_cast_missing_plan_raises() -> None:
    stage = AudiobookCastStage(client=AsyncMock())
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})
    with pytest.raises(StagePreconditionError, match="audiobook_plan"):
        await stage.run({}, ctx=ctx)


# --- #718: the split itself -------------------------------------------------


async def test_attribution_splits_into_bounded_batches() -> None:
    """The unbounded performances array is chunked into ≤5-chunk WM calls, and the
    flat merge still covers every plan chunk exactly once."""
    n = _PERFORMANCE_BATCH_SIZE * 2 + 1  # 11
    chunks = [{"id": f"k{i}", "text": f"line {i}", "sequence": i} for i in range(n)]
    record: dict[str, list] = {}
    fake = _two_phase_wm(
        perf_specs={f"k{i}": {"speaker": "Narrator"} for i in range(n)},
        record=record,
    )

    out = await AudiobookCastStage(client=fake).run({}, ctx=_ctx(chunks=chunks))

    assert record["perf_batches"] == [5, 5, 1]  # bounded per-call, never all 11
    # Full coverage: exactly one performance per plan chunk, in order.
    assert [p.chunk_id for p in out.performances] == [f"k{i}" for i in range(n)]


async def test_single_global_cast_prevents_cross_batch_collision() -> None:
    """A character appearing in two different attribution batches is cast by ONE
    global call → one consistent voice, distinct from other characters. This is
    the collision #718 exists to prevent."""
    n = _PERFORMANCE_BATCH_SIZE * 2 + 1  # 11 → 3 batches
    chunks = [{"id": f"k{i}", "text": f"line {i}", "sequence": i} for i in range(n)]
    # Bob speaks in batch 0 (k0) AND batch 2 (k10); Alice in batch 1 (k5).
    speakers = {f"k{i}": "Narrator" for i in range(n)}
    speakers["k0"] = "Bob"
    speakers["k10"] = "Bob"
    speakers["k5"] = "Alice"
    record: dict[str, list] = {}
    fake = _two_phase_wm(
        perf_specs={cid: {"speaker": sp} for cid, sp in speakers.items()},
        cast_voices={"Alice": "af_bella", "Bob": "bm_george"},
        record=record,
    )

    out = await AudiobookCastStage(client=fake).run({}, ctx=_ctx(chunks=chunks))

    # Exactly ONE casting call, and it saw the full distinct roster at once.
    assert len(record["cast_rosters"]) == 1
    assert set(record["cast_rosters"][0]) == {"Alice", "Bob"}

    by_chunk = {p.chunk_id: p for p in out.performances}
    # Bob is ONE consistent voice across both his batches — no collision.
    assert by_chunk["k0"].voice_id == by_chunk["k10"].voice_id == "bm_george"
    # Distinct characters stay on distinct voices.
    assert by_chunk["k5"].voice_id == "af_bella"
    assert by_chunk["k0"].voice_id != by_chunk["k5"].voice_id
