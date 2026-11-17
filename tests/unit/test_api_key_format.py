"""``w2v_sk_<prefix>_<secret>`` — generation, strict parsing, hashing, verification."""

from __future__ import annotations

import hashlib

import pytest

from src.auth.api_key_format import (
    KEY_SCHEME,
    format_api_key,
    generate_api_key,
    hash_api_key,
    looks_like_api_key,
    new_prefix,
    parse_api_key,
    verify_api_key,
)

# 43 chars with '_' inside the secret part.
_SECRET_WITH_UNDERSCORES = "ab_cd_ef" + "x" * 35  # pragma: allowlist secret


def test_generated_key_has_the_documented_shape_and_hash() -> None:
    key = generate_api_key()
    assert key.full_key.startswith(KEY_SCHEME)
    assert len(key.prefix) == 8 and key.prefix == key.prefix.lower()
    assert set(key.prefix) <= set("abcdefghijklmnopqrstuvwxyz234567")
    assert len(key.full_key) == len(KEY_SCHEME) + 8 + 1 + 43
    assert key.key_hash == hashlib.sha256(key.full_key.encode()).digest()
    assert len(key.key_hash) == 32
    parsed = parse_api_key(key.full_key)
    assert parsed is not None and parsed.prefix == key.prefix
    assert key.full_key not in repr(key), "repr must not leak the secret"


def test_keys_and_prefixes_are_unique_per_draw() -> None:
    keys = {generate_api_key().full_key for _ in range(200)}
    assert len(keys) == 200
    assert len({new_prefix() for _ in range(200)}) > 190


def test_parse_handles_underscores_inside_the_secret() -> None:
    token = format_api_key("abcdefgh", _SECRET_WITH_UNDERSCORES)
    parsed = parse_api_key(token)
    assert parsed is not None
    assert (parsed.prefix, parsed.secret) == ("abcdefgh", _SECRET_WITH_UNDERSCORES)


@pytest.mark.parametrize(
    "token",
    [
        "",
        "w2v_sk_",
        "Bearer w2v_sk_abcdefgh_" + "x" * 43,
        "w2v_pk_abcdefgh_" + "x" * 43,  # wrong scheme
        "w2v_sk_ABCDEFGH_" + "x" * 43,  # uppercase prefix
        "w2v_sk_abcdefg1_" + "x" * 43,  # '1' is not base32
        "w2v_sk_abcdefg_" + "x" * 44,  # short prefix (total length kept)
        "w2v_sk_abcdefghX" + "x" * 43,  # missing separator
        "w2v_sk_abcdefgh_" + "x" * 42,  # short secret
        "w2v_sk_abcdefgh_" + "x" * 44,  # long secret
        "w2v_sk_abcdefgh_" + "x" * 42 + "=",  # padding is not urlsafe-unpadded
        "w2v_sk_abcdefgh_" + "x" * 42 + " ",
    ],
)
def test_parse_rejects_malformed(token: str) -> None:
    assert parse_api_key(token) is None


def test_verify_is_hash_equality_of_the_full_key() -> None:
    key = generate_api_key()
    assert verify_api_key(key.full_key, key.key_hash)
    assert not verify_api_key(key.full_key + "x", key.key_hash)
    other = generate_api_key()
    assert not verify_api_key(other.full_key, key.key_hash)
    # Same prefix, different secret → different hash.
    forged = format_api_key(key.prefix, "y" * 43)
    assert hash_api_key(forged) != key.key_hash
    assert not verify_api_key(forged, key.key_hash)


def test_looks_like_api_key_is_only_a_scheme_check() -> None:
    assert looks_like_api_key("w2v_sk_anything")
    assert not looks_like_api_key("eyJhbGciOi...")
