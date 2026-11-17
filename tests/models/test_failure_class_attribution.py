"""``attribute_exception`` — the ours-vs-theirs allowlist, and its refusals.

The property under test is not "does it classify things" but **does it refuse to
classify things it cannot see**. A wrong attribution is worse than an absent one:
``our_defect`` on an infrastructure blip sends an engineer hunting a bug that
does not exist, and ``customer_input`` on our own bug tells a paying user to fix
something they did not break. So roughly half of these cases assert ``unknown``.
"""

from __future__ import annotations

import httpx
import pytest

from src.models.failure_class import (
    CUSTOMER_INPUT,
    EXTERNAL,
    OUR_DEFECT,
    UNKNOWN,
    attribute_exception,
)
from src.services.stages.upstream import (
    PermanentUpstreamError,
    TransientUpstreamError,
    classify,
    classify_transport,
    fault_for_status,
)


def _status_error(status: int, body: str = "{}") -> httpx.HTTPStatusError:
    """A real ``HTTPStatusError`` for ``status``, as ``guarded`` would see it."""
    request = httpx.Request("POST", "http://worker-management:8887/workers/x/message")
    response = httpx.Response(status, request=request, text=body)
    return httpx.HTTPStatusError(f"{status}", request=request, response=response)


# ── The three attributed outcomes ────────────────────────────────────────────


def test_a_wm_402_records_customer_input() -> None:
    """WM 402 ``byo_llm_not_configured``: the tenant has no LLM provider.

    Nothing we ship makes this run succeed — only the customer can act. This is
    the single unambiguous customer-fault status on the upstream path.
    """
    error = classify(
        _status_error(
            402,
            '{"error": {"code": "ERROR", "message": {"code": "byo_llm_not_configured", '
            '"message": "Configure your LLM provider in Settings before running agents."},'
            ' "status": 402}}',
        )
    )
    assert isinstance(error, PermanentUpstreamError)
    assert attribute_exception(error) == CUSTOMER_INPUT


def test_a_python_programming_error_records_our_defect() -> None:
    """An ``AttributeError`` in our own process is a line of our Python being wrong.

    No brief a customer can write and no response an upstream can send produces
    this; it is reached only through a defect we can fix by shipping.
    """
    assert attribute_exception(AttributeError("'NoneType' has no attribute 'clips'")) == OUR_DEFECT


def test_an_upstream_5xx_records_external_not_our_defect() -> None:
    """A dependency that 5xx'd is ``external``, and specifically NOT ``our_defect``.

    The second half is the point. Folding infrastructure flake into the defect
    bucket would inflate it with noise until nobody read it — the failure mode
    that makes a taxonomy worthless.
    """
    error = classify(_status_error(503, "upstream connect error"))
    assert isinstance(error, TransientUpstreamError)
    assert attribute_exception(error) == EXTERNAL


def test_a_transport_error_records_external() -> None:
    """A request the server never saw says nothing about our code or their input."""
    error = classify_transport(httpx.ReadError("connection reset"))
    assert attribute_exception(error) == EXTERNAL


# ── The refusals: everything ambiguous must land on `unknown` ────────────────


@pytest.mark.parametrize(
    ("status", "why"),
    [
        (400, "could be OUR malformed request or THEIR malformed brief"),
        (401, "S2S misconfiguration (ours) and an expired licence (theirs) share it"),
        (403, "same two readings as 401"),
        (404, "our wrong URL, or their missing resource"),
        (422, "a real contract fault (ours) and a lost model draw (neither) share it"),
    ],
)
def test_ambiguous_4xx_statuses_are_not_attributed(status: int, why: str) -> None:
    """Each of these can be read two ways, so none of them may be attributed.

    Being ambiguous is a property of the STATUS, so it is asserted against
    ``fault_for_status`` directly as well as end-to-end: a future edit that adds
    one of these to ``_STATUS_FAULT`` "because it is usually ours" fails here.
    """
    assert fault_for_status(status) == UNKNOWN, why
    assert attribute_exception(classify(_status_error(status))) == UNKNOWN, why


def test_an_unrecognised_exception_records_unknown_and_is_not_misattributed() -> None:
    """The default path. A bare ``RuntimeError`` carries no structural signal.

    Asserted as an EQUALITY to ``unknown`` rather than an inequality to
    ``our_defect``, so widening ``_OUR_DEFECT_TYPES`` to something that swallows
    this (``Exception``, say) fails here instead of quietly re-bucketing every
    unattributable run in the estate.
    """
    assert attribute_exception(RuntimeError("stage 'compile' returned no artifact")) == UNKNOWN


def test_a_value_error_is_not_our_defect() -> None:
    """``ValueError`` is pydantic's validation base and the stages' re-prompt signal.

    It is raised BY tenant data as routinely as by our own bugs, so attributing
    it either way would be a guess.
    """
    assert attribute_exception(ValueError("narration_text must be non-empty")) == UNKNOWN


def test_an_unstamped_upstream_error_asserts_nothing() -> None:
    """A ``TransientUpstreamError`` raised outside ``classify`` has no evidence.

    The class default is ``UNKNOWN`` precisely so the several call sites that
    construct these directly (with no HTTP response in hand) cannot accidentally
    inherit an attribution from the ones that do.
    """
    assert attribute_exception(TransientUpstreamError("poll gave up")) == UNKNOWN


# ── Cause-chain traversal ────────────────────────────────────────────────────


def test_attribution_is_read_through_a_wrapping_cause_chain() -> None:
    """``guarded`` raises ``from``, and stage code re-raises — the stamp must survive.

    Without the walk, every attribution would be lost the moment a stage wrapped
    an upstream failure in its own error, which is the normal path, not the
    exotic one.
    """
    inner = classify(_status_error(402))
    outer = RuntimeError("stage 'script_draft' failed")
    outer.__cause__ = inner
    assert attribute_exception(outer) == CUSTOMER_INPUT


def test_an_explicit_stamp_outranks_a_programming_error_type_in_the_chain() -> None:
    """Pass order matters: the stamped attribution wins over a type inference.

    The stamp was made where the HTTP status was in hand; the type inference is
    made here, with less. If the passes ran in the other order, a
    ``KeyError`` raised while *reading* an upstream error body would relabel a
    customer's unconfigured provider as our bug.
    """
    stamped = classify(_status_error(402))
    wrapper = KeyError("choices")
    wrapper.__cause__ = stamped
    assert attribute_exception(wrapper) == CUSTOMER_INPUT


def test_an_unknown_stamp_does_not_veto_a_definite_cause() -> None:
    """A wrapper that could not attribute itself must not suppress its cause.

    ``_AttributedUpstreamError`` defaults to ``UNKNOWN``, so an unstamped wrapper
    carries the attribute; skipping ``UNKNOWN`` values in the first pass is what
    stops it from short-circuiting a real answer underneath.
    """
    outer = TransientUpstreamError("retries exhausted")
    outer.__cause__ = AttributeError("'NoneType' has no attribute 'url'")
    assert attribute_exception(outer) == OUR_DEFECT


def test_a_self_referential_cause_chain_terminates() -> None:
    """A cycle in ``__cause__`` must not hang the run-failure path.

    This walk happens while a run is being terminal-failed; an infinite loop here
    would strand the row in ``running`` and leave the reconciler re-dispatching
    it forever. Cheap to make impossible, catastrophic if it is not.
    """
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a
    assert attribute_exception(a) == UNKNOWN


def test_implicit_context_is_never_walked() -> None:
    """``__context__`` links whatever happened to be in flight — attributing off it
    would manufacture answers.

    Any ``raise`` inside an ``except`` block sets ``__context__``, so a run that
    failed for its own reasons while merely *handling* an unrelated upstream 402
    would be blamed on the customer. Only explicit ``raise ... from`` is evidence.
    """
    unrelated = classify(_status_error(402))
    actual = RuntimeError("compile produced no master")
    actual.__context__ = unrelated
    assert actual.__cause__ is None
    assert attribute_exception(actual) == UNKNOWN
