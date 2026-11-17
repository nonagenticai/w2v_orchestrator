"""Provider-connection secrets: AES-256-GCM sealed at rest, bound to their row.

A tenant's provider secret (an LLM API key, …) is stored in
``tenant_connection_versions.secret_ciphertext`` as one opaque envelope::

    b"v1" | b"|" | kid | b"|" | nonce (12 bytes) | ciphertext + GCM tag (16 bytes)

* **kid** names the keyring entry that sealed it (mirrored in the row's
  ``key_id`` so the GC / readiness check can scan kids without parsing bytes).
* **AAD** = ``"<tenant_id>|<connection_id>|<version>"``. The ciphertext is bound
  to the exact row it was written for: copying it to another tenant, another
  connection or another version makes the tag check fail — a DB-level row swap
  cannot move a key between tenants.
* **Keyring** (``PROVIDER_SECRET_KEYS="kid:b64,…"``, ``src/config.py``): the FIRST  # pragma: allowlist secret
  key encrypts, every key decrypts. Rotation = prepend a key, run
  ``scripts/rewrap_provider_secrets.py``, drop the old key.

Fail closed, always: an empty keyring refuses to encrypt
(:class:`KeyringNotConfiguredError` → the API answers 503); an unknown kid, a
wrong AAD, a tampered byte or a malformed envelope raise
:class:`SecretDecryptError` — there is no "return None and carry on". No message
in this module ever contains plaintext or key material.

Secrets are write-only towards clients: the API shows ``has_secret`` and
:func:`fingerprint` (12 hex chars), never the value. The fingerprint is an
HMAC-SHA256 keyed by a key DERIVED from the keyring's primary key (HKDF-SHA256,
info ``w2v-provider-fp-v1``) — a plain hash of a low-entropy key would let anyone
reading the DB (or the API) confirm a guessed secret offline. Consequence: the
fingerprint depends on the primary key, so a rotation changes it. ``rewrap``
recomputes it under the new primary and the rewrap script persists it with the
re-sealed envelope; a purged version (no ciphertext left to open) keeps the
fingerprint it was written with.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from uuid import UUID

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from src.config import PROVIDER_KEY_ID_PATTERN, get_settings, parse_provider_secret_keys

ENVELOPE_VERSION = b"v1"
_SEP = b"|"
NONCE_BYTES = 12
TAG_BYTES = 16
FINGERPRINT_HEX = 12
#: HKDF ``info`` label of the fingerprint key (domain separation from the AES use).
FINGERPRINT_KDF_LABEL = b"w2v-provider-fp-v1"


class ProviderSecretError(RuntimeError):
    """Base: a provider secret cannot be sealed or opened. Never carries the secret."""


class KeyringNotConfiguredError(ProviderSecretError):
    """``PROVIDER_SECRET_KEYS`` is empty — secret writes must be refused (503)."""


class SecretDecryptError(ProviderSecretError):
    """The envelope cannot be opened: unknown kid, wrong AAD, tampered, malformed."""


@dataclass(frozen=True, slots=True)
class SealedSecret:
    """What to persist: ``secret_ciphertext`` + ``key_id`` (+ the fingerprint)."""

    ciphertext: bytes
    key_id: str
    fingerprint: str


def aad(tenant_id: UUID, connection_id: UUID, version: int) -> bytes:
    """The associated data binding a ciphertext to its exact row."""
    return f"{tenant_id}|{connection_id}|{int(version)}".encode()


#: Domain label of a tenant's stored LICENCE TOKEN (``tenants.licence_token_ciphertext``).
#: Its AAD can never equal a connection AAD (whose first field is a uuid).
TENANT_LICENCE_AAD_LABEL = "tenant-licence"


def tenant_licence_aad(tenant_id: UUID) -> bytes:
    """AAD ``("tenant-licence", tenant_id)``: a sealed licence token opens only as
    THAT tenant's — copying the ciphertext onto another tenant row fails the tag."""
    return f"{TENANT_LICENCE_AAD_LABEL}|{tenant_id}".encode()


def derive_fingerprint_key(key: bytes) -> bytes:
    """HKDF-SHA256(key, info=``w2v-provider-fp-v1``) → the 32-byte fingerprint HMAC key."""
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=FINGERPRINT_KDF_LABEL).derive(
        key
    )


def fingerprint(secret: str, fingerprint_key: bytes) -> str:
    """12-hex public fingerprint of ``secret`` (lets a UI say "key …ab12cd" changed).

    HMAC-SHA256 under ``fingerprint_key`` (see :func:`derive_fingerprint_key`), never a
    bare hash: without the key a fingerprint cannot confirm a guessed secret."""
    digest = hmac.new(fingerprint_key, secret.encode(), hashlib.sha256).hexdigest()
    return digest[:FINGERPRINT_HEX]


def peek_key_id(ciphertext: bytes) -> str:
    """The kid an envelope was sealed with (no decryption). Raises on a bad envelope."""
    return _split(ciphertext)[0]


def _split(ciphertext: bytes) -> tuple[str, bytes, bytes]:
    head, sep1, rest = bytes(ciphertext).partition(_SEP)
    kid_raw, sep2, body = rest.partition(_SEP)
    if head != ENVELOPE_VERSION or not sep1 or not sep2:
        raise SecretDecryptError("provider secret envelope is malformed")
    try:
        kid = kid_raw.decode("ascii")
    except UnicodeDecodeError:
        raise SecretDecryptError("provider secret envelope is malformed") from None
    if not PROVIDER_KEY_ID_PATTERN.fullmatch(kid) or len(body) < NONCE_BYTES + TAG_BYTES:
        raise SecretDecryptError("provider secret envelope is malformed")
    return kid, body[:NONCE_BYTES], body[NONCE_BYTES:]


class Keyring:
    """An ordered set of AES-256 keys: ``keys[0]`` encrypts, all decrypt."""

    def __init__(self, entries: Sequence[tuple[str, bytes]]) -> None:
        self._order: tuple[str, ...] = tuple(kid for kid, _ in entries)
        if len(set(self._order)) != len(self._order):
            raise ValueError("keyring repeats a key id")
        self._keys: dict[str, AESGCM] = {kid: AESGCM(key) for kid, key in entries}
        self._fingerprint_key: bytes | None = (
            derive_fingerprint_key(entries[0][1]) if entries else None
        )

    @classmethod
    def from_env_value(cls, raw: str) -> Keyring:
        return cls(parse_provider_secret_keys(raw))

    def __repr__(self) -> str:  # never render key material
        return f"Keyring(kids={list(self._order)!r})"

    @property
    def configured(self) -> bool:
        return bool(self._order)

    @property
    def primary_key_id(self) -> str:
        if not self._order:
            raise KeyringNotConfiguredError("PROVIDER_SECRET_KEYS is not configured")
        return self._order[0]

    @property
    def key_ids(self) -> tuple[str, ...]:
        return self._order

    def fingerprint(self, secret: str) -> str:
        """:func:`fingerprint` under the key derived from the PRIMARY key."""
        if self._fingerprint_key is None:
            raise KeyringNotConfiguredError("PROVIDER_SECRET_KEYS is not configured")
        return fingerprint(secret, self._fingerprint_key)

    def missing_key_ids(self, stored: Iterable[str | None]) -> set[str]:
        """Kids referenced by stored rows that this keyring cannot open (``/readyz``)."""
        return {kid for kid in stored if kid and kid not in self._keys}

    def encrypt(
        self, secret: str, *, tenant_id: UUID, connection_id: UUID, version: int
    ) -> SealedSecret:
        envelope, kid = self.encrypt_with_aad(secret, aad(tenant_id, connection_id, version))
        return SealedSecret(ciphertext=envelope, key_id=kid, fingerprint=self.fingerprint(secret))

    def encrypt_with_aad(self, secret: str, associated_data: bytes) -> tuple[bytes, str]:
        """Seal ``secret`` under the PRIMARY key bound to ``associated_data``.
        Returns ``(envelope, kid)``; raises :class:`KeyringNotConfiguredError` when empty."""
        kid = self.primary_key_id
        nonce = os.urandom(NONCE_BYTES)
        sealed = self._keys[kid].encrypt(nonce, secret.encode(), associated_data)
        return _SEP.join((ENVELOPE_VERSION, kid.encode("ascii"), nonce + sealed)), kid

    def decrypt(
        self, ciphertext: bytes, *, tenant_id: UUID, connection_id: UUID, version: int
    ) -> str:
        return self.decrypt_with_aad(ciphertext, aad(tenant_id, connection_id, version))

    def decrypt_with_aad(self, ciphertext: bytes, associated_data: bytes) -> str:
        """Open an envelope given its row-bound AAD (see :func:`aad`).

        An EMPTY keyring is a deployment fault, not a bad row: it raises
        :class:`KeyringNotConfiguredError` (→ 503), never :class:`SecretDecryptError`."""
        if not self._order:
            raise KeyringNotConfiguredError("PROVIDER_SECRET_KEYS is not configured")
        kid, nonce, sealed = _split(ciphertext)
        cipher = self._keys.get(kid)
        if cipher is None:
            raise SecretDecryptError(f"provider secret sealed with unknown key id {kid!r}")
        try:
            plain = cipher.decrypt(nonce, sealed, associated_data)
        except InvalidTag:
            raise SecretDecryptError(
                "provider secret failed authentication (wrong key, row binding or tampered)"
            ) from None
        try:
            return plain.decode()
        except UnicodeDecodeError:  # pragma: no cover - only a sealed non-UTF-8 payload
            raise SecretDecryptError("provider secret is not valid UTF-8") from None

    def needs_rewrap(self, ciphertext: bytes) -> bool:
        return peek_key_id(ciphertext) != self.primary_key_id

    def rewrap(
        self, ciphertext: bytes, *, tenant_id: UUID, connection_id: UUID, version: int
    ) -> SealedSecret | None:
        """Re-seal under the primary key; ``None`` when it already is.

        The returned :attr:`SealedSecret.fingerprint` is recomputed under the NEW
        primary's derived key — persist it with the envelope (``replace_secret``)."""
        if not self.needs_rewrap(ciphertext):
            return None
        secret = self.decrypt(
            ciphertext, tenant_id=tenant_id, connection_id=connection_id, version=version
        )
        return self.encrypt(
            secret, tenant_id=tenant_id, connection_id=connection_id, version=version
        )

    def rewrap_with_aad(
        self, ciphertext: bytes, associated_data: bytes
    ) -> tuple[bytes, str] | None:
        """:meth:`rewrap` for an envelope with a non-connection AAD (the tenant
        licence token). ``None`` when it is already sealed by the primary."""
        if not self.needs_rewrap(ciphertext):
            return None
        return self.encrypt_with_aad(
            self.decrypt_with_aad(ciphertext, associated_data), associated_data
        )


def get_keyring() -> Keyring:
    """The process keyring from settings (parsed per call; settings are cached)."""
    return Keyring(get_settings().provider_secrets.keyring_entries())
