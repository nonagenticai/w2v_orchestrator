"""``tenant_scope_key`` refuses the nil uuid BY VALUE, not just ``None``.

Three subsystems key SHARED storage off the ambient tenant (sound-catalog
library cache, cross-run media cache, ``X-Owner-Tenant-Id``). Before
w2v-common 0.67.0 the nil uuid was the configured ``auth.default_tenant_id``
fallback, so it genuinely reached those call sites. A ``None``-only guard
lets it straight through, because the nil uuid is truthy -- which merges every
unattributed caller into one shared bucket.
"""

from __future__ import annotations

import uuid

import pytest

from src.auth.tenant_scope import tenant_scope_key

_REAL = uuid.UUID("11111111-2222-3333-4444-555555555555")


@pytest.mark.parametrize(
    "value",
    [
        # Every spelling of the nil uuid the stdlib will parse. A guard written as
        # `value != "00000000-0000-0000-0000-000000000000"` passes the first and
        # fails all the rest -- which is exactly why this is checked on `.int`.
        "00000000-0000-0000-0000-000000000000",
        "00000000000000000000000000000000",
        "{00000000-0000-0000-0000-000000000000}",
        "urn:uuid:00000000-0000-0000-0000-000000000000",
        uuid.UUID(int=0),
    ],
    ids=["hyphenated", "bare", "braced", "urn", "uuid-object"],
)
def test_nil_uuid_is_refused_in_every_spelling(value) -> None:
    assert tenant_scope_key(value) is None


@pytest.mark.parametrize(
    "value",
    [None, "", "   ", "not-a-uuid", "tenant-1", 12345],
    ids=["none", "empty", "blank", "garbage", "slug", "int"],
)
def test_unusable_tenants_are_refused(value) -> None:
    assert tenant_scope_key(value) is None


def test_a_real_tenant_is_rendered_canonically() -> None:
    """A usable tenant survives, as a canonical lowercase hyphenated string."""
    assert tenant_scope_key(_REAL) == str(_REAL)
    # Accepts the wire form too, and normalises it to the same key -- otherwise
    # the same tenant could occupy two different cache/storage buckets.
    assert tenant_scope_key(str(_REAL).upper()) == str(_REAL)
    assert tenant_scope_key(_REAL.hex) == str(_REAL)


def test_refusal_is_not_a_raise() -> None:
    """These are scoping call sites: 'no usable tenant' degrades to unscoped.

    Request-handling code that must not proceed without a tenant uses
    ``w2v_common.tenant.require_tenant_id`` instead, which raises.
    """
    assert tenant_scope_key("garbage") is None


def test_tenant_errors_are_value_errors_so_a_bare_except_swallows_them() -> None:
    """The trap this helper exists to keep out of call sites.

    ``w2v_common.tenant``'s errors subclass ``ValueError``, and the idiom
    around uuid parsing is ``try: uuid.UUID(v) except ValueError:``. Wrap a
    tenant parse in that and the refusal is CAUGHT — so code that meant to
    fail loud silently takes the fallback branch instead. Two services shipped
    exactly that bug on the day 0.67.0 landed.

    ``tenant_scope_key`` catches the NARROW ``TenantError`` on purpose. That is
    correct here (this seam degrades to unscoped by design), and this test
    exists so that the subclassing is a documented, asserted property rather
    than something a future reader rediscovers from a production incident.
    """
    from w2v_common.tenant import InvalidTenantIdError, MissingTenantError, TenantError

    assert issubclass(TenantError, ValueError)
    assert issubclass(MissingTenantError, ValueError)
    assert issubclass(InvalidTenantIdError, ValueError)

    # Demonstrate the swallow directly, so the claim above is not folklore.
    from w2v_common.tenant import parse_tenant_id

    swallowed = False
    try:
        parse_tenant_id(None, source="pin")
    except ValueError:
        swallowed = True
    assert swallowed, "MissingTenantError stopped being a ValueError — re-read this test"
