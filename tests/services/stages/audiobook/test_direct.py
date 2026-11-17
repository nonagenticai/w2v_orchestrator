"""Tests for audiobook_direct — the authoritative clamp (spec acceptance #7) +
coverage-fill + skip-clean byte-identity. The WM call is not exercised (the model
is advisory); the stage's job is to CLAMP whatever the model returns.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

from src.services.runs.run_context import RunContext
from src.services.stages.audiobook.direct import (
    AudiobookDirectInput,
    AudiobookDirectStage,
)
from src.services.stages.base import StageSkippedError


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


def _stage() -> AudiobookDirectStage:
    # A real WMClient is constructed but never used: these tests exercise only the
    # pure clamp (_assemble) and the skip-clean path, which returns before any WM
    # call. No network happens at construction time.
    return AudiobookDirectStage()


# --- pure coercion/clamp helpers -------------------------------------------


# --- authoritative clamp over the assembled directives (acceptance #7) ------


def test_assemble_clamps_every_field() -> None:
    chunks = [{"id": "c1", "text": "one two three four five"}]
    raw: list[object] = [
        {
            "chunk_id": "c1",
            "rate_hint": 9.0,  # -> clamp to 2.0
            "pause_before_ms": 99999,  # -> clamp to 2000
            "emphasis_word_indices": [0, 2, 99],  # 99 is out of range (5 words)
            "delivery_note": "urgent",
        }
    ]
    out = _stage()._assemble(chunks, raw)
    assert len(out) == 1
    d = out[0]
    assert d.chunk_id == "c1"
    assert d.rate_hint == 2.0
    assert d.pause_before_ms == 2000
    assert d.emphasis_word_indices == [0, 2]  # 99 dropped
    assert d.delivery_note == "urgent"


def test_assemble_coverage_fill_neutral_default() -> None:
    # Two plan chunks, model only returned one → the omitted chunk still gets a
    # directive, and it is neutral (no rate opinion, no pause, no emphasis) so it
    # narrates byte-identically to today.
    chunks = [{"id": "c1", "text": "hello"}, {"id": "c2", "text": "world"}]
    raw: list[object] = [{"chunk_id": "c1", "rate_hint": 1.1}]
    out = _stage()._assemble(chunks, raw)
    by_id = {d.chunk_id: d for d in out}
    assert set(by_id) == {"c1", "c2"}
    assert by_id["c1"].rate_hint == 1.1
    # c2 was omitted → neutral: rate_hint None means "keep cast speed".
    assert by_id["c2"].rate_hint is None
    assert by_id["c2"].pause_before_ms == 0
    assert by_id["c2"].emphasis_word_indices == []


def test_assemble_ignores_non_dict_and_unkeyed_entries() -> None:
    chunks = [{"id": "c1", "text": "x y"}]
    out = _stage()._assemble(chunks, ["garbage", {"no_chunk_id": True}, 42])
    # No usable directive matched → c1 gets the neutral default (not dropped).
    assert len(out) == 1
    assert out[0].chunk_id == "c1"
    assert out[0].rate_hint is None


# --- skip-clean: no direction ⇒ no LLM work, behaviour-preserving -----------


async def test_handle_skips_clean_without_direction() -> None:
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": {"audiobook": {"output_format": "mp3"}}},  # no direction
    )
    with pytest.raises(StageSkippedError):
        await _stage().handle(AudiobookDirectInput(), ctx)


# --- WM call is batched into bounded chunks (spine may be arbitrarily long) --


async def test_chunks_split_into_bounded_batches() -> None:
    from src.services.stages.audiobook.direct import _DIRECTIVE_BATCH_SIZE

    n_chunks = _DIRECTIVE_BATCH_SIZE * 2 + 1  # 11 -> batches of 5, 5, 1
    all_ids = [f"c{i}" for i in range(n_chunks)]
    chunks = [{"id": cid, "text": "word " * 3} for cid in all_ids]

    per_call_counts: list[int] = []

    def _batch_resp(*_: Any, **kwargs: Any) -> WMResponse:
        message = kwargs["message"]
        ids_in_call = [cid for cid in all_ids if f'"chunk_id":"{cid}"' in message]
        per_call_counts.append(len(ids_in_call))
        return _resp(
            {"directives": [{"chunk_id": cid, "delivery_note": "hot"} for cid in ids_in_call]}
        )

    fake = AsyncMock()
    fake.message.side_effect = _batch_resp

    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": {"audiobook": {"direction": "Warm and intimate."}}},
        prior_run_outputs={"audiobook_plan": {"chapters": [{"chunks": chunks}]}},
    )

    out = await AudiobookDirectStage(client=fake).handle(AudiobookDirectInput(), ctx)

    assert fake.message.call_count == 3  # ceil(11 / 5)
    assert per_call_counts == [5, 5, 1]  # bounded, order-preserving

    # Full coverage: every plan chunk carries the merged LLM value, not the
    # neutral coverage-fill default.
    by_id = {d.chunk_id: d for d in out.directives}
    assert set(by_id) == set(all_ids)
    assert all(by_id[cid].delivery_note == "hot" for cid in all_ids)
