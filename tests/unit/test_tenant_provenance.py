"""The background-dispatch identity rebind must carry the tenant's PROVENANCE.

This is the defect class the type checker cannot see. ``tenant_source`` is a
keyword-optional argument to ``set_identity_context``, so omitting it is green
under ruff, ty and every mocked-client test — and then w2v-common 0.67.0's
``_apply_outbound_tenant`` refuses the FIRST metered hop of every background
run, because a tenant of unrecorded provenance may not be delegated.

Each test below therefore pairs the fixed path with the pre-fix shape, so the
assertions cannot pass vacuously: the "without provenance" case must RAISE, or
the "with provenance" case is proving nothing.
"""

from __future__ import annotations

import uuid

import httpx
import pytest
from w2v_common.auth.identity_context import (
    _apply_outbound_tenant,
    clear_identity_context,
    get_tenant_source_from_context,
    set_identity_context,
)
from w2v_common.tenant import TenantSource, UnvouchedTenantError

from src.auth.tenant_provenance import (
    TENANT_SOURCE_KEY,
    snapshot_tenant_source,
    tenant_source_from_snapshot,
)

_TENANT = uuid.UUID("6f1a0f2e-0000-7000-8000-00000000ab01")
#: A header VALUE, not a credential — the tests never authenticate with it.
_SA_BEARER = "Bearer sa-license-token"


@pytest.fixture(autouse=True)
def _clean_identity():
    clear_identity_context()
    yield
    clear_identity_context()


def _metered_request(monkeypatch) -> httpx.Request:
    """An outbound hop shaped exactly like a background run's LLM call.

    Service-account Bearer already on ``X-Tenant-Id`` (``WMClient`` bakes it)
    and a metered destination (``worker_management``). That combination is what
    selects the DELEGATED-HOP branch, which is the only branch that consults
    the provenance.
    """
    monkeypatch.setenv("W2V_METERED_DESTINATIONS", "wm.test")
    return httpx.Request(
        "POST",
        "http://wm.test:9000/workers/writer/message",
        headers={"X-Tenant-Id": _SA_BEARER},
    )


class TestTheRebindWithoutProvenanceIsRefused:
    def test_a_metered_hop_raises_when_the_source_was_not_rebound(self, monkeypatch) -> None:
        """The bug, reproduced. This is what shipped before the snapshot existed."""
        set_identity_context(tenant_id=_TENANT, license_id="lic", subject_id="sub")
        with pytest.raises(UnvouchedTenantError):
            _apply_outbound_tenant(_metered_request(monkeypatch))

    def test_the_same_hop_succeeds_once_the_source_is_rebound(self, monkeypatch) -> None:
        """The fix. Positive control for the test above."""
        set_identity_context(
            tenant_id=_TENANT,
            license_id="lic",
            subject_id="sub",
            tenant_source=TenantSource.LICENSE_TOKEN,
        )
        request = _metered_request(monkeypatch)
        _apply_outbound_tenant(request)
        assert request.headers["X-Effective-Tenant-Id"] == str(_TENANT)
        # And the SA token it authenticates with is untouched.
        assert request.headers["X-Tenant-Id"] == _SA_BEARER


class TestSnapshotRoundTrip:
    @pytest.mark.parametrize(
        "source",
        [TenantSource.LICENSE_TOKEN, TenantSource.JWT_CLAIM, TenantSource.UNVOUCHED_HEADER],
    )
    def test_a_bound_source_survives_the_snapshot(self, source: TenantSource) -> None:
        """Request scope -> JSON column -> background task, unchanged.

        Both vouched and unvouched rungs are round-tripped: silently promoting
        an unvouched header to a vouched source would be the fail-open the
        ladder exists to close, so the trip must be lossless in BOTH directions.
        """
        set_identity_context(
            tenant_id=_TENANT, license_id="lic", subject_id="sub", tenant_source=source
        )
        stored = snapshot_tenant_source()
        assert stored == source.value

        clear_identity_context()
        assert get_tenant_source_from_context() is None

        set_identity_context(
            tenant_id=_TENANT,
            license_id="lic",
            subject_id="sub",
            tenant_source=tenant_source_from_snapshot(stored),
        )
        assert get_tenant_source_from_context() is source

    def test_nothing_bound_snapshots_as_the_absent_spelling(self) -> None:
        assert snapshot_tenant_source() == ""


class TestParsingRefusesRatherThanInvents:
    @pytest.mark.parametrize(
        "value",
        [None, "", "   ", "license_token", "LICENSE-TOKEN", "not-a-source", 7, TenantSource],
        ids=["none", "empty", "blank", "underscored", "uppercased", "garbage", "int", "type"],
    )
    def test_every_unusable_value_becomes_none_not_a_guess(self, value: object) -> None:
        """``None`` is the UNVOUCHED end of the ladder, so unusable degrades to refusal.

        A legacy run row (created before this field existed) lands here. It must
        NOT be promoted to ``LICENSE_TOKEN``: there is no evidence for that, and
        minting provenance is precisely the fail-open the ladder closes. The run
        refuses at its first metered hop and an operator restarts it.
        """
        assert tenant_source_from_snapshot(value) is None

    def test_a_legacy_snapshot_still_refuses_the_metered_hop(self, monkeypatch) -> None:
        """End-to-end for the paragraph above — not just the parse, the consequence."""
        legacy_snapshot: dict[str, object] = {"tenant_id": str(_TENANT), "license_id": "lic"}
        set_identity_context(
            tenant_id=_TENANT,
            license_id="lic",
            subject_id="sub",
            tenant_source=tenant_source_from_snapshot(legacy_snapshot.get(TENANT_SOURCE_KEY)),
        )
        with pytest.raises(UnvouchedTenantError):
            _apply_outbound_tenant(_metered_request(monkeypatch))
