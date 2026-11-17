"""Fingerprint / key-derivation for the persistent source-ingest cache.

Pure functions — no DB, so these run in the fast lane (no ``integration`` mark).
The fingerprint is the cache's isolation boundary: tenant, auth identity, and
document CONTENT must each change it; cosmetic URL differences must not.
"""

from __future__ import annotations

from uuid import UUID, uuid7

import pytest

from src.services.source_cache import (
    SourceCacheRepository,
    auth_id_for,
    document_ref_key,
    normalize_url,
    source_fingerprint,
)

_T1 = uuid7()
_T2 = uuid7()


def _fp_url(tenant, url, **kw):
    return source_fingerprint(tenant, "url", normalize_url(url), **kw)


def test_identical_url_same_tenant_collides():
    assert _fp_url(_T1, "https://example.com/a") == _fp_url(_T1, "https://example.com/a")


def test_tenant_isolation_different_fingerprint():
    # The SAME source under a different tenant is a different row → auto-miss.
    assert _fp_url(_T1, "https://example.com/a") != _fp_url(_T2, "https://example.com/a")


def test_url_normalization_folds_cosmetic_differences():
    # Scheme/host casing, default port, and fragment do not change the body.
    base = _fp_url(_T1, "https://example.com/path?q=1")
    assert base == _fp_url(_T1, "HTTPS://Example.COM/path?q=1")
    assert base == _fp_url(_T1, "https://example.com:443/path?q=1")
    assert base == _fp_url(_T1, "https://example.com/path?q=1#section")


def test_url_query_and_path_are_significant():
    # Query and path CAN change the fetched content → distinct fingerprints.
    assert _fp_url(_T1, "https://example.com/a") != _fp_url(_T1, "https://example.com/b")
    assert _fp_url(_T1, "https://example.com/a?x=1") != _fp_url(_T1, "https://example.com/a?x=2")


def test_auth_identity_prevents_cross_serve():
    # A credentialed fetch and an anonymous fetch of the same URL must not share.
    anon = _fp_url(_T1, "https://example.com/a", auth_id=auth_id_for(None))
    creds = _fp_url(_T1, "https://example.com/a", auth_id=auth_id_for({"token": "s3cr3t"}))
    assert anon != creds
    assert auth_id_for(None) == "anon"
    # Same auth payload → same id (order-independent).
    assert auth_id_for({"user": "u", "pass": "p"}) == auth_id_for({"pass": "p", "user": "u"})


def test_opts_token_invalidates():
    assert _fp_url(_T1, "https://example.com/a", opts_token="img=1") != _fp_url(
        _T1, "https://example.com/a", opts_token="img=0"
    )


def test_document_ref_key_is_content_addressed():
    # Same bytes → same key (ref/filename irrelevant); different bytes → different.
    assert document_ref_key(b"hello world") == document_ref_key(b"hello world")
    assert document_ref_key(b"hello world") != document_ref_key(b"HELLO WORLD")
    fp_a = source_fingerprint(_T1, "document", document_ref_key(b"same-pdf-bytes"))
    fp_b = source_fingerprint(_T1, "document", document_ref_key(b"same-pdf-bytes"))
    assert fp_a == fp_b


def test_fingerprint_is_64_hex():
    fp = _fp_url(_T1, "https://example.com/a")
    assert len(fp) == 64
    assert all(c in "0123456789abcdef" for c in fp)


@pytest.mark.asyncio
async def test_safe_release_swallows_a_non_sqlalchemy_failure() -> None:
    """``_safe_release`` must swallow ANY release failure, not just DB-layer ones.

    ⚠️ Lives HERE, not in ``test_source_cache.py``, on purpose: that file carries a
    file-level ``pytest.mark.integration`` and ``addopts`` is
    ``-m 'not integration ...'``, so a test written there NEVER RUNS under the
    default selection CI uses — it would be decorative cover for this guard.
    This one needs no DB, so it runs in the fast lane where it can actually fail.

    The guard was ``except SQLAlchemyError``. ``release()`` also calls ``_as_uuid()``,
    so a malformed ``tenant_id`` raises a plain ``ValueError`` there, which escaped
    and REPLACED the real ``producer()`` exception on ``get_or_produce``'s ``raise``.
    """
    repo = SourceCacheRepository.__new__(SourceCacheRepository)
    calls: list[tuple[str, UUID | str]] = []

    async def boom(fingerprint: str, tenant_id: UUID | str) -> None:
        calls.append((fingerprint, tenant_id))
        raise ValueError("badly formed hexadecimal UUID string")

    repo.release = boom  # ty: ignore[invalid-assignment]

    # Must return normally: the caller re-raises the REAL producer error after this.
    await repo._safe_release("fp-abc", "not-a-uuid")

    # Assert the guarded call actually RAN and was swallowed. Without this the test
    # would pass even if `_safe_release` stopped calling `release` altogether — it
    # would be asserting nothing about the guard, which is what this repo's own
    # vacuous-test guard exists to catch (and did catch, on the first draft).
    assert calls == [("fp-abc", "not-a-uuid")]
