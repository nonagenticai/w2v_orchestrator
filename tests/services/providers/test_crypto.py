"""Provider-secret crypto (``src/services/providers/crypto.py``) — DB-free.

Every claim here is a security property, so each test is written to FAIL if the
property is removed: drop the AAD and the swap tests pass decryption; encrypt with
the last key instead of the first and the rotation test sees the wrong kid; let an
unknown kid fall through to "try every key" and the fail-closed test opens it.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
from uuid import uuid7

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from pydantic import ValidationError

from src.config import ProviderSecretSettings, clear_settings_cache, parse_provider_secret_keys
from src.services.providers.crypto import (
    NONCE_BYTES,
    Keyring,
    KeyringNotConfiguredError,
    SecretDecryptError,
    fingerprint,
    get_keyring,
    peek_key_id,
)

SECRET = "sk-live-0123456789abcdef"  # pragma: allowlist secret
T1, T2 = uuid7(), uuid7()
C1, C2 = uuid7(), uuid7()


def _key() -> bytes:
    return os.urandom(32)


def _b64(key: bytes) -> str:
    return base64.b64encode(key).decode()


@pytest.fixture
def ring() -> Keyring:
    return Keyring([("k1", _key())])


def test_round_trip_and_envelope_shape(ring: Keyring) -> None:
    sealed = ring.encrypt(SECRET, tenant_id=T1, connection_id=C1, version=1)
    assert sealed.key_id == "k1"
    assert sealed.ciphertext.startswith(b"v1|k1|")
    assert SECRET.encode() not in sealed.ciphertext
    # v1|kid| + 12-byte nonce + plaintext + 16-byte tag
    assert len(sealed.ciphertext) == len(b"v1|k1|") + NONCE_BYTES + len(SECRET) + 16
    assert peek_key_id(sealed.ciphertext) == "k1"
    assert ring.decrypt(sealed.ciphertext, tenant_id=T1, connection_id=C1, version=1) == SECRET


def test_nonce_is_fresh_per_encryption(ring: Keyring) -> None:
    a = ring.encrypt(SECRET, tenant_id=T1, connection_id=C1, version=1)
    b = ring.encrypt(SECRET, tenant_id=T1, connection_id=C1, version=1)
    assert a.ciphertext != b.ciphertext


@pytest.mark.parametrize(
    ("tenant", "connection", "version"),
    [(T2, C1, 1)],
    ids=["other-tenant"],
)
def test_ciphertext_moved_to_another_row_fails_closed(
    ring: Keyring, tenant, connection, version
) -> None:
    sealed = ring.encrypt(SECRET, tenant_id=T1, connection_id=C1, version=1)
    with pytest.raises(SecretDecryptError) as caught:
        ring.decrypt(sealed.ciphertext, tenant_id=tenant, connection_id=connection, version=version)
    assert SECRET not in str(caught.value)


def test_unknown_key_id_fails_closed_even_if_another_key_would_open_it() -> None:
    key = _key()
    writer = Keyring([("old", key)])
    sealed = writer.encrypt(SECRET, tenant_id=T1, connection_id=C1, version=1)
    # Same key bytes under a DIFFERENT kid: an implementation that ignores the kid
    # and tries every key would open this. It must not.
    reader = Keyring([("new", key)])
    with pytest.raises(SecretDecryptError, match="unknown key id 'old'"):
        reader.decrypt(sealed.ciphertext, tenant_id=T1, connection_id=C1, version=1)
    assert reader.missing_key_ids(["old", "new", None]) == {"old"}


def test_tampered_or_malformed_envelope_fails_closed(ring: Keyring) -> None:
    sealed = ring.encrypt(SECRET, tenant_id=T1, connection_id=C1, version=1)
    flipped = bytearray(sealed.ciphertext)
    flipped[-1] ^= 0x01
    for bad in (bytes(flipped), b"v2" + sealed.ciphertext[2:], b"v1|k1|short", b"garbage"):
        with pytest.raises(SecretDecryptError):
            ring.decrypt(bad, tenant_id=T1, connection_id=C1, version=1)


def test_first_key_encrypts_every_key_decrypts() -> None:
    old_key, new_key = _key(), _key()
    old_ring = Keyring([("old", old_key)])
    legacy = old_ring.encrypt(SECRET, tenant_id=T1, connection_id=C1, version=3)

    rotated = Keyring([("new", new_key), ("old", old_key)])
    assert rotated.primary_key_id == "new"
    assert rotated.decrypt(legacy.ciphertext, tenant_id=T1, connection_id=C1, version=3) == SECRET
    fresh = rotated.encrypt(SECRET, tenant_id=T1, connection_id=C1, version=3)
    assert fresh.key_id == "new"
    assert peek_key_id(fresh.ciphertext) == "new"
    # ...and it really is sealed by the new key alone:
    with pytest.raises(SecretDecryptError):
        old_ring.decrypt(fresh.ciphertext, tenant_id=T1, connection_id=C1, version=3)


def test_rewrap_moves_to_primary_and_keeps_the_row_binding() -> None:
    old_key, new_key = _key(), _key()
    legacy = Keyring([("old", old_key)]).encrypt(SECRET, tenant_id=T1, connection_id=C1, version=2)
    rotated = Keyring([("new", new_key), ("old", old_key)])

    assert rotated.needs_rewrap(legacy.ciphertext)
    sealed = rotated.rewrap(legacy.ciphertext, tenant_id=T1, connection_id=C1, version=2)
    assert sealed is not None and sealed.key_id == "new"
    # The fingerprint is keyed by the PRIMARY → it changes with the primary on rewrap.
    assert sealed.fingerprint == rotated.fingerprint(SECRET) != legacy.fingerprint
    new_only = Keyring([("new", new_key)])
    assert new_only.decrypt(sealed.ciphertext, tenant_id=T1, connection_id=C1, version=2) == SECRET
    with pytest.raises(SecretDecryptError):
        new_only.decrypt(sealed.ciphertext, tenant_id=T2, connection_id=C1, version=2)
    assert rotated.rewrap(sealed.ciphertext, tenant_id=T1, connection_id=C1, version=2) is None


def test_empty_keyring_refuses_to_encrypt_and_cannot_decrypt(ring: Keyring) -> None:
    empty = Keyring([])
    assert not empty.configured
    with pytest.raises(KeyringNotConfiguredError):
        empty.encrypt(SECRET, tenant_id=T1, connection_id=C1, version=1)
    sealed = ring.encrypt(SECRET, tenant_id=T1, connection_id=C1, version=1)
    # A deployment fault (→ 503), never confused with a bad/tampered row.
    with pytest.raises(KeyringNotConfiguredError):
        empty.decrypt(sealed.ciphertext, tenant_id=T1, connection_id=C1, version=1)
    with pytest.raises(KeyringNotConfiguredError):
        empty.fingerprint(SECRET)


def test_fingerprint_is_12_hex_stable_and_distinguishes(ring: Keyring) -> None:
    fp = ring.fingerprint(SECRET)
    assert len(fp) == 12 and int(fp, 16) >= 0
    assert fp == ring.fingerprint(SECRET)
    assert fp != ring.fingerprint(SECRET + "x")


def test_fingerprint_is_an_hmac_keyed_from_the_primary_key() -> None:
    key = _key()
    ring = Keyring([("k1", key), ("k0", _key())])
    fp_key = HKDF(
        algorithm=hashes.SHA256(), length=32, salt=None, info=b"w2v-provider-fp-v1"
    ).derive(key)
    expected = hmac.new(fp_key, SECRET.encode(), hashlib.sha256).hexdigest()[:12]
    assert ring.fingerprint(SECRET) == fingerprint(SECRET, fp_key) == expected
    # Not a bare (or domain-prefixed) hash: unguessable without the key.
    assert ring.fingerprint(SECRET) != hashlib.sha256(SECRET.encode()).hexdigest()[:12]
    assert Keyring([("k1", _key())]).fingerprint(SECRET) != ring.fingerprint(SECRET)


def test_repr_never_contains_key_material() -> None:
    key = _key()
    text = repr(Keyring([("k1", key)]))
    assert "k1" in text and _b64(key) not in text and repr(key) not in text


# ─── keyring config ──────────────────────────────────────────────────────


def test_parse_keeps_order_and_accepts_urlsafe_b64() -> None:
    a, b = _key(), _key()
    raw = f" a:{_b64(a)} , b:{base64.urlsafe_b64encode(b).decode()} ,"
    assert parse_provider_secret_keys(raw) == [("a", a), ("b", b)]
    assert parse_provider_secret_keys("") == []


@pytest.mark.parametrize(
    "raw",
    ["no-colon"],
    ids=["no-colon"],
)
def test_malformed_keyring_fails_config_load_without_echoing_keys(raw: str) -> None:
    with pytest.raises(ValidationError) as caught:
        ProviderSecretSettings(PROVIDER_SECRET_KEYS=raw)
    for part in raw.split(","):
        b64 = part.partition(":")[2]
        if len(b64) > 8:
            assert b64 not in str(caught.value)


def test_get_keyring_reads_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    k_new, k_old = _key(), _key()
    monkeypatch.setenv("PROVIDER_SECRET_KEYS", f"new:{_b64(k_new)},old:{_b64(k_old)}")
    clear_settings_cache()
    ring = get_keyring()
    assert ring.key_ids == ("new", "old")
    sealed = ring.encrypt(SECRET, tenant_id=T1, connection_id=C1, version=1)
    assert (
        Keyring([("new", k_new)]).decrypt(
            sealed.ciphertext, tenant_id=T1, connection_id=C1, version=1
        )
        == SECRET
    )

    monkeypatch.delenv("PROVIDER_SECRET_KEYS")
    clear_settings_cache()
    assert not get_keyring().configured
