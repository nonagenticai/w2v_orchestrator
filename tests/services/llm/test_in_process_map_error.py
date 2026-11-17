"""BYOK (in-process pydantic-ai) error taxonomy — truncation parity with the platform path.

#1126 made a platform ``finish_reason=length`` (``vllm_direct._complete_once``) a
retryable ``TransientUpstreamError`` / ``EXTERNAL``: a truncated reply is a bad DRAW,
never the customer's fault. pydantic-ai surfaces the same condition on a tenant
connection as ``UnexpectedModelBehavior`` / ``IncompleteToolCall`` carrying "Model
token limit (N) exceeded ...", which ``map_error`` used to file as
``PermanentUpstreamError`` / ``CUSTOMER_INPUT``.
"""

from __future__ import annotations

import pytest
from pydantic_ai.exceptions import UnexpectedModelBehavior

from src.models.failure_class import EXTERNAL
from src.services.llm import in_process
from src.services.stages.upstream import TransientUpstreamError

_BEFORE_ANY = (
    "Model token limit (10050) exceeded before any response was generated. Increase the "
    "`max_tokens` model setting, or simplify the prompt to result in a shorter response "
    "that will fit within the limit."
)
_MID_TOOL_CALL = (
    "Model token limit (10050) exceeded while generating a tool call, resulting in "
    "incomplete arguments."
)


@pytest.mark.parametrize(
    "exc",
    [UnexpectedModelBehavior(_BEFORE_ANY)],
    ids=["before_any_response"],
)
def test_byok_truncation_is_transient_external(exc: Exception) -> None:
    err = in_process.map_error(exc, kind="openai", model="gpt", secret=None)
    assert isinstance(err, TransientUpstreamError)
    assert err.failure_class == EXTERNAL


def test_byok_output_retries_exhausted_stays_transient_external() -> None:
    exc = UnexpectedModelBehavior("Exceeded maximum output retries (3)")
    err = in_process.map_error(exc, kind="openai", model="gpt", secret=None)
    assert isinstance(err, TransientUpstreamError)
    assert err.failure_class == EXTERNAL
