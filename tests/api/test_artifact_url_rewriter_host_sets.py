"""The ownership guard's host set must cover the re-signer's — W3-1a.

Two independently-defined host sets decide whether a client-supplied URL can be
re-signed with this service's own MinIO credentials:

* ``StorageService._is_own_minio_host`` (w2v_common) — the hosts
  ``resign_presigned_url`` will sign for, read from ``STORAGE_MINIO_ENDPOINT``
  and ``STORAGE_MINIO_EXTERNAL_ENDPOINT``.
* ``artifact_url_rewriter``'s matcher — the hosts whose URLs
  ``assert_refs_owned`` collects, and therefore ownership-checks.

Safety rests on the SECOND covering the FIRST: a URL that can be re-signed must
have been ownership-checked first. Until W3-1a the rewriter hardcoded
``minio.test`` while the re-signer read config, so the property held only
because production happened to leave ``STORAGE_MINIO_EXTERNAL_ENDPOINT`` unset.
Setting it — a supported ConfigMap edit, already done in ``docker-compose.yml`` —
widened the re-signer past the literal and armed a real hole with no test
failing. These tests fail if that divergence is ever reintroduced.
"""

from __future__ import annotations

import pytest

from src.api.artifact_url_rewriter import (
    internal_host_pattern,
    internal_minio_ref,
)
from src.config import StorageSettings, get_settings

_INTERNAL = "minio.test:8389"
# A legitimate second endpoint, as docker-compose.yml already configures.
_EXTERNAL = "https://minio-public.example.com:9050"


def _settings(external: str | None) -> StorageSettings:
    return StorageSettings(
        backend="minio",
        minio_endpoint=_INTERNAL,
        minio_access_key="k",
        minio_secret_key="s",
        minio_external_endpoint=external,
    )


def test_the_matcher_reads_live_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """``internal_host_pattern`` is wired to config, not to a literal."""
    monkeypatch.setenv("STORAGE_BACKEND", "minio")
    monkeypatch.setenv("STORAGE_MINIO_ENDPOINT", _INTERNAL)
    monkeypatch.setenv("STORAGE_MINIO_EXTERNAL_ENDPOINT", _EXTERNAL)
    get_settings.cache_clear()
    try:
        assert internal_host_pattern().match("http://minio-public.example.com:9050/b/k?s=1")
        assert internal_minio_ref("http://minio-public.example.com:9050/b/k?s=1") == "b/k"
    finally:
        get_settings.cache_clear()
