"""In-process (tenant BYO) usage rows carry who acted + the funding, so per-user BYO
usage is queryable from ``run_usage`` (memory: usage-funding-taxonomy-and-byo)."""

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal
from uuid import UUID, uuid7

import pytest
from w2v_common.auth.identity_context import set_correlation_context, set_identity_context
from w2v_common.clients import WMResponse

from src.database.models import RunUsage
from src.database.repositories.run_usage_repository import IN_PROCESS_USAGE_SOURCE
from src.services.llm.in_process import LLMResult
from src.services.llm.metering import usage_row
from tests.services.llm._fakes import CONN_ID, tenant_slot

TENANT = UUID("01990000-0000-7000-8000-0000000000aa")
SUBJECT = "01990000-0000-7000-8000-0000000000bb"
ACTOR = "01990000-0000-7000-8000-0000000000cc"


@pytest.fixture(autouse=True)
def _clear() -> Iterator[None]:
    yield
    set_identity_context(None, None, None)
    set_correlation_context(None, None)


def _usage() -> RunUsage:
    result = LLMResult(
        response=WMResponse(
            output="x",
            output_type="string",
            output_schema_used=None,
            metadata=None,
            usage_metrics=None,
        ),
        input_tokens=3,
        output_tokens=2,
        model="tenant-model-7",
        provider_kind="openai-compatible",
    )
    return usage_row(
        run_id=uuid7(),
        stage_name="promo_review",
        worker_name="promo-review",
        resolved=tenant_slot(),
        result=result,
    )


def _row() -> dict[str, object]:
    row = _usage()
    assert row.metadata_ is not None
    return row.metadata_


def test_row_sets_the_attribution_columns() -> None:
    """P6 ORCH-3: per-user / per-slot reads group on columns, not metadata."""
    set_identity_context(str(TENANT), None, SUBJECT, actor_id=ACTOR)

    row = _usage()

    assert row.subject_id == UUID(SUBJECT)
    assert row.slot == "llm"
    assert row.funding == "tenant_byo"
    assert row.units == Decimal(5) and row.unit_kind == "tokens"


def test_a_non_uuid_subject_leaves_the_row_unattributed_not_failed() -> None:
    set_identity_context(str(TENANT), None, "svc-account")

    row = _usage()

    assert row.subject_id is None
    assert row.metadata_ is not None and row.metadata_["subject_id"] == "svc-account"


def test_row_records_actor_subject_and_byo_funding() -> None:
    set_identity_context(str(TENANT), None, SUBJECT, actor_id=ACTOR)

    meta = _row()

    assert meta["actor_id"] == ACTOR
    assert meta["subject_id"] == SUBJECT
    assert meta["funding"] == "tenant_byo"
    assert meta["source"] == IN_PROCESS_USAGE_SOURCE


def test_absent_actor_is_omitted_never_substituted_by_subject() -> None:
    set_identity_context(str(TENANT), None, SUBJECT)

    meta = _row()

    assert "actor_id" not in meta, "no actor bound → field absent, not the subject"
    assert meta["subject_id"] == SUBJECT
    assert meta["funding"] == "tenant_byo"


def test_row_records_the_serving_connection_and_version_as_columns() -> None:
    """P7 / R3 (migration 0088): which tenant connection VERSION served the call."""
    row = _usage()
    assert row.connection_id == CONN_ID
    assert row.connection_version == 3
