"""segment_select — the empty draw, driven through the REAL ``WMClient`` wire path.

BACKLOG #21 claimed "an empty draw is legal in the segment grammar, so the stage
proceeds with nothing". Both halves were checked here, end to end through the
Chat Completions body this stage actually POSTs (``LLM_BACKEND=vllm``, the backend
prod runs) and the response shape the vLLM returns — not through an ``AsyncMock`` that
skips ``WMClient.message`` entirely.

* The GRAMMAR half was true until 2026-10-06 and is now closed: ``segments`` is a
  ``non_empty_array`` (``prefixItems: [ITEM]`` + ``items: ITEM``). ``minItems`` is
  useless — worker-management strips it before building the xgrammar grammar
  (``tests/test_output_schema_bounds.py`` bans the keyword) — but WM KEEPS
  ``prefixItems``, so the first element is required by the CFG itself. Measured
  THROUGH worker-management on the script stages (2026-10-06): a plain array
  answered ``[]`` on 6 of 6 calls, the ``prefixItems`` shape returned full lists
  2 of 2 (``w2v/.proof-archive/2026-10-06/round4/r0-replay/``).
* The parser floor stays as the backstop and is what the wire tests pin: an all-empty answer is
  rejected with the stage's existing retryable ``ValueError``, ``stage_call``
  re-prompts the model for EVERY batch (the ledger is reset), and the stage never
  returns an empty selection.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from src.services.llm import WMClient, vllm_direct
from src.services.runs.run_context import RunContext
from src.services.stages.action_reel.segment_select import (
    _OUTPUT_SCHEMA,
    _SELECT_BATCH_SOURCES,
    SegmentSelectInput,
    SegmentSelectStage,
)
from src.services.stages.llm_budget import llm_retries
from tests.support.platform_llm import CHAT_PATH, PlatformChat, posted_schema, user_text

pytestmark = pytest.mark.usefixtures("platform_llm_socket")


def _sources(n: int) -> list[dict[str, Any]]:
    return [
        {
            "id": f"src-{i:02d}",
            "duration_s": 20.0,
            "scene_boundaries": [5.0],
            "motion_timeline": [{"t": 1.0, "motion": 0.9}],
            "overall_score": 0.7,
        }
        for i in range(1, n + 1)
    ]


def _ctx(sources: list[dict[str, Any]]) -> RunContext:
    from uuid import uuid7

    return RunContext(
        run_id=uuid7(),
        deliverable_id=uuid7(),
        config_snapshot={},
        prior_run_outputs={"shot_analysis": {"sources": sources}},
    )


class _WireWM:
    """The platform LLM (Chat Completions) behind an ``httpx2.MockTransport``.

    Answers the first ``empty_calls`` calls with ``{"segments": []}`` — a
    schema-valid, HTTP-200 answer — and every later one with a real window per
    ``source_id`` named in that call's prompt. Records every request body and the
    worker each ``WMClient.message`` call named (not on the vLLM wire).
    """

    def __init__(self, *, empty_calls: int) -> None:
        self.empty_calls = empty_calls
        self.chat = PlatformChat(self.answer)
        self.workers: list[str] = []

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return self.chat.bodies

    @property
    def paths(self) -> list[str]:
        return self.chat.paths

    def answer(self, body: dict[str, Any], n: int) -> dict[str, Any]:
        if n < self.empty_calls:
            return {"segments": []}
        ids = re.findall(r'"source_id": "([^"]+)"', user_text(body))
        return {
            "segments": [
                {
                    "source_id": source_id,
                    "in_seconds": 1.0,
                    "out_seconds": 5.0,
                    "score": 0.5,
                    "rationale": "clean move",
                }
                for source_id in ids
            ]
        }

    def client(self) -> WMClient:
        wire = self

        class _Recording(WMClient):
            async def message(self, worker_name: str, *args: Any, **kwargs: Any) -> Any:
                wire.workers.append(worker_name)
                return await super().message(worker_name, *args, **kwargs)

        # Route the platform socket here; everything above the transport (body
        # building, schema strip, response parsing) is real.
        self.chat.activate()
        return _Recording(timeout=30.0)


def test_the_wire_grammar_requires_at_least_one_segment() -> None:
    """``{"segments": []}`` must be unproducible by the GRAMMAR, via ``prefixItems``.

    Not ``minItems``: WM strips it (``tests/test_output_schema_bounds.py``), so it
    would be a floor nothing applies at generation time.
    """
    props = _OUTPUT_SCHEMA["properties"]
    assert isinstance(props, dict)
    segments = props["segments"]
    assert segments["type"] == "array"
    assert segments["prefixItems"] == [segments["items"]]
    assert segments["items"]["required"] == [
        "source_id",
        "in_seconds",
        "out_seconds",
        "score",
        "rationale",
    ]
    assert not {"minItems", "maxItems"} & set(segments)
    assert _OUTPUT_SCHEMA["required"] == ["segments"]


async def test_an_all_empty_wire_answer_is_rejected_and_re_prompted_never_shipped() -> None:
    batches = 2
    attempts = llm_retries() + 1
    assert attempts >= 2, "this test needs a retry budget to have anything to prove"
    wire = _WireWM(empty_calls=10**6)
    stage = SegmentSelectStage(client=wire.client())

    with pytest.raises(Exception, match="no valid segments survived validation"):
        await stage.handle(SegmentSelectInput(), _ctx(_sources(_SELECT_BATCH_SOURCES * batches)))

    # Every attempt went back to the model for every batch: a re-prompt, not a
    # replay of banked empties, and no empty selection was ever returned.
    assert len(wire.bodies) == batches * attempts
    assert set(wire.paths) == {CHAT_PATH}
    assert set(wire.workers) == {SegmentSelectStage.worker_name}
    # The schema on the wire is exactly the stage's (as the xgrammar strip carries
    # it) — the parser is judging the contract the model was actually given.
    grammar: Any = vllm_direct.output_contract(_OUTPUT_SCHEMA).grammar
    assert grammar is not None
    assert grammar["properties"]["segments"]["prefixItems"]
    assert all(posted_schema(body) == grammar for body in wire.bodies)


async def test_an_empty_first_attempt_recovers_on_the_re_prompt() -> None:
    batches = 2
    sources = _sources(_SELECT_BATCH_SOURCES * batches)
    wire = _WireWM(empty_calls=batches)
    stage = SegmentSelectStage(client=wire.client())

    out = await stage.handle(SegmentSelectInput(), _ctx(sources))

    assert out.segments, "an empty draw must never be what the stage ships"
    assert {seg.source_id for seg in out.segments} == {str(s["id"]) for s in sources}
    # Attempt 1: both batches answered empty. Attempt 2 re-issued BOTH (ledger
    # reset), not zero.
    assert len(wire.bodies) == batches * 2
