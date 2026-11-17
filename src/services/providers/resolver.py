"""Resolve which provider a run uses for a slot, and pin that choice at run start.

Order (blueprint §2):

1. **Pinned** — the run snapshot carries ``provider_slots[slot]``. A deployment ref
   resolves to that deployment kind. A tenant ref loads exactly that
   ``(connection_id, version)``; it must exist, be unpurged, belong to the SAME tenant
   and slot, and pass policy — otherwise :class:`SlotUnavailableError`. A paused run
   therefore keeps the connection it started with even if the tenant changes it.
2. **Legacy snapshot** (no ``provider_slots``, or no entry for the slot) → live.
3. **Live** — an active tenant connection is used if valid; an INVALID one raises
   (never falls back to ours). No connection → the deployment default.

Provider access (licences ``byo_mode``, ONE policy for every ``byo_governed`` slot —
``providers/access.py``), applied at pin AND resolve:

* ``local_allowed`` — tenant connection if active, else the platform default.
* ``byo_required`` — never the platform: a deployment pin / no connection raises
  :class:`SlotNotConfiguredError` (the run-start preflight already refused the run;
  this is defence in depth), and so does the tenant's OWN connection when its
  ``base_url`` is a platform-internal host admitted only by the operator egress
  allowlist (``private_ok``) — that would be our GPU behind the tenant's slot.
* ``platform_only`` — :func:`pin_slots` and the live path IGNORE a saved tenant
  connection (platform default). A run pinned to a tenant connection BEFORE the
  operator flipped the tenant to ``platform_only`` is refused with
  :class:`SlotUnavailableError` rather than silently re-routed mid-run.

The posture is read only for governed slots; an unreadable one raises
:class:`ByoPostureUnavailableError` (fail closed, transient).

Storage and crypto are WP1's; this module codes against the two small protocols
below so it can be tested with fakes and wired to the real repository later.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol
from uuid import UUID

from pydantic import SecretStr

from src.services.providers.access import platform_allowed, tenant_connection_used
from src.services.providers.byo_posture import ByoMode, ByoPostureReader
from src.services.providers.crypto import KeyringNotConfiguredError
from src.services.providers.crypto import aad as secret_aad
from src.services.providers.errors import (
    ProviderKeyringUnavailableError,
    SlotNotConfiguredError,
    SlotUnavailableError,
)
from src.services.providers.policy import check_egress, check_kind, check_not_self
from src.services.providers.registry import PINNED_SLOT_NAMES, get_slot
from src.services.providers.settings import ProviderPolicySettings, get_provider_policy

SlotSource = Literal["deployment", "tenant"]

#: Snapshot key the run pin is stored under (WP3 adds it to RUN_SCOPED_SNAPSHOT_KEYS).
PROVIDER_SLOTS_KEY = "provider_slots"


class ConnectionVersionRecord(Protocol):
    """One immutable connection version, joined with its connection's tenant + slot."""

    @property
    def tenant_id(self) -> UUID: ...
    @property
    def slot(self) -> str: ...
    @property
    def connection_id(self) -> UUID: ...
    @property
    def version(self) -> int: ...
    @property
    def kind(self) -> str: ...
    @property
    def base_url(self) -> str | None: ...
    @property
    def model(self) -> str | None: ...
    @property
    def options(self) -> Mapping[str, object]: ...
    @property
    def secret_ciphertext(self) -> bytes | None: ...
    @property
    def purged_at(self) -> object | None: ...


class ConnectionRepository(Protocol):
    """Read side of WP1's ``connections_repository`` that the resolver needs."""

    async def get_active(self, tenant_id: UUID, slot: str) -> ConnectionVersionRecord | None:
        """The current version of the tenant's ACTIVE connection on ``slot``, if any."""
        ...

    async def get_version(
        self, connection_id: UUID, version: int
    ) -> ConnectionVersionRecord | None:
        """Exactly that version (any status — a pinned run may outlive a soft delete)."""
        ...


class SecretDecryptor(Protocol):
    """WP1 ``providers/crypto``: AES-GCM open; raises on any failure."""

    def decrypt(self, ciphertext: bytes, aad: bytes) -> str: ...


@dataclass(frozen=True, slots=True)
class SlotRef:
    """What a run pins per slot. JSON-able via :meth:`to_dict` / :meth:`from_dict`."""

    slot: str
    source: SlotSource
    kind: str
    connection_id: UUID | None = None
    version: int | None = None

    def __post_init__(self) -> None:
        if self.source == "tenant" and (self.connection_id is None or self.version is None):
            raise ValueError("a tenant SlotRef needs connection_id and version")
        if self.source == "deployment" and (
            self.connection_id is not None or self.version is not None
        ):
            raise ValueError("a deployment SlotRef carries no connection")

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "source": self.source,
            "kind": self.kind,
            "connection_id": str(self.connection_id) if self.connection_id else None,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, slot: str, data: Mapping[str, object]) -> SlotRef:
        raw_source = data.get("source")
        kind = data.get("kind")
        source: SlotSource
        if raw_source == "deployment":
            source = "deployment"
        elif raw_source == "tenant":
            source = "tenant"
        else:
            raise SlotUnavailableError(slot, "the run's pinned provider reference is malformed")
        if not isinstance(kind, str):
            raise SlotUnavailableError(slot, "the run's pinned provider reference is malformed")
        raw_id = data.get("connection_id")
        raw_version = data.get("version")
        try:
            connection_id = UUID(str(raw_id)) if raw_id is not None else None
            version = int(str(raw_version)) if raw_version is not None else None
            return cls(
                slot=slot, source=source, kind=kind, connection_id=connection_id, version=version
            )
        except ValueError as exc:
            raise SlotUnavailableError(
                slot, "the run's pinned provider reference is malformed"
            ) from exc


@dataclass(frozen=True, slots=True)
class ResolvedSlot:
    """A usable provider for one slot. ``secret`` is a :class:`SecretStr` (never in reprs)."""

    ref: SlotRef
    base_url: str | None = None
    model: str | None = None
    options: Mapping[str, object] = field(default_factory=dict)
    secret: SecretStr | None = None
    #: IP the client must connect to (DNS-rebinding pin); ``None`` = allowlisted / no URL.
    pinned_ip: str | None = None

    @property
    def source(self) -> SlotSource:
        return self.ref.source

    @property
    def kind(self) -> str:
        return self.ref.kind


def _deployment(policy: ProviderPolicySettings, slot: str) -> ResolvedSlot:
    return ResolvedSlot(
        ref=SlotRef(slot=slot, source="deployment", kind=policy.deployment_kind(slot))
    )


def _pinned_ref(run_snapshot: Mapping[str, object] | None, slot: str) -> SlotRef | None:
    if run_snapshot is None:
        return None
    pins = run_snapshot.get(PROVIDER_SLOTS_KEY)
    if pins is None:
        return None  # legacy run (pre-P3 snapshot) → live
    if not isinstance(pins, Mapping):
        raise SlotUnavailableError(slot, "the run's pinned providers are malformed")
    entry = pins.get(slot)
    if entry is None:
        return None  # slot added to the registry after this run pinned → live
    if isinstance(entry, SlotRef):
        return entry
    if not isinstance(entry, Mapping):
        raise SlotUnavailableError(slot, "the run's pinned provider reference is malformed")
    return SlotRef.from_dict(slot, entry)


async def _mode(posture: ByoPostureReader, tenant_id: UUID, slot: str) -> ByoMode | None:
    """The tenant's posture for a ``byo_governed`` slot; ``None`` (not read) otherwise."""
    if not get_slot(slot).byo_governed:
        return None
    return await posture.byo_mode(tenant_id, slot=slot)


async def refuse_platform_if_byo_required(
    posture: ByoPostureReader, tenant_id: UUID, slot: str
) -> None:
    """Raise when a byo_required tenant would reach the platform for ``slot``
    (no-op for a slot the posture does not govern)."""
    mode = await _mode(posture, tenant_id, slot)
    if mode is not None and not platform_allowed(mode, slot):
        raise SlotNotConfiguredError(slot)


_PLATFORM_ONLY_PINNED = (
    "the organisation is now set to platform providers only, so the connection this run "
    "was started with can no longer be used; start a new run"
)


async def _materialise(
    record: ConnectionVersionRecord,
    *,
    tenant_id: UUID,
    slot: str,
    policy: ProviderPolicySettings,
    decryptor: SecretDecryptor,
    mode: ByoMode | None,
) -> ResolvedSlot:
    """Validate a tenant connection version and turn it into a :class:`ResolvedSlot`."""
    if record.tenant_id != tenant_id:
        raise SlotUnavailableError(slot, "the pinned connection belongs to another tenant")
    if record.slot != slot:
        raise SlotUnavailableError(slot, "the pinned connection is for a different provider")
    if record.purged_at is not None:
        raise SlotUnavailableError(slot, "the pinned connection version has been purged")
    check_kind(policy, slot, record.kind)

    spec = get_slot(slot)
    pinned_ip: str | None = None
    if record.base_url:
        check_not_self(slot, record.base_url)
        pinned_ip = await check_egress(policy, slot, record.base_url)
        if pinned_ip is None and mode is not None and not platform_allowed(mode, slot):
            # ``None`` = admitted by the operator allowlist (``private_ok``): a
            # platform-internal host. For a byo_required tenant that is our GPU
            # wearing the tenant's slot — refuse it like the platform default.
            raise SlotNotConfiguredError(slot)
    elif record.kind in spec.base_url_required_kinds:
        raise SlotUnavailableError(slot, f"kind {record.kind!r} requires a base URL")

    secret: SecretStr | None = None
    if record.secret_ciphertext is not None:
        aad = secret_aad(record.tenant_id, record.connection_id, record.version)
        try:
            secret = SecretStr(decryptor.decrypt(record.secret_ciphertext, aad))
        except KeyringNotConfiguredError as exc:  # OUR fault, not the tenant's
            raise ProviderKeyringUnavailableError(slot) from exc
        except Exception as exc:  # any crypto failure fails closed
            raise SlotUnavailableError(slot, "the stored credential could not be read") from exc

    return ResolvedSlot(
        ref=SlotRef(
            slot=slot,
            source="tenant",
            kind=record.kind,
            connection_id=record.connection_id,
            version=record.version,
        ),
        base_url=record.base_url,
        model=record.model,
        options=dict(record.options),
        secret=secret,
        pinned_ip=pinned_ip,
    )


async def resolve_slot(
    repo: ConnectionRepository,
    decryptor: SecretDecryptor,
    tenant_id: UUID,
    slot: str,
    *,
    run_snapshot: Mapping[str, object] | None = None,
    policy: ProviderPolicySettings | None = None,
    posture: ByoPostureReader,
) -> ResolvedSlot:
    """Resolve ``slot`` for ``tenant_id`` — pinned version first, else live (module doc).

    ``posture`` is REQUIRED (no default): a tenant connection on an allowlisted
    platform-internal host must never be accepted by omission."""
    get_slot(slot)  # unknown slot → KeyError (programming error, not a tenant fault)
    policy = policy or get_provider_policy()

    pinned = _pinned_ref(run_snapshot, slot)
    mode = await _mode(posture, tenant_id, slot)
    if pinned is not None:
        if pinned.source == "deployment":
            if mode is not None and not platform_allowed(mode, slot):
                raise SlotNotConfiguredError(slot)
            return ResolvedSlot(ref=pinned)
        if mode is not None and not tenant_connection_used(mode, slot):
            raise SlotUnavailableError(slot, _PLATFORM_ONLY_PINNED)
        assert pinned.connection_id is not None and pinned.version is not None  # noqa: S101
        record = await repo.get_version(pinned.connection_id, pinned.version)
        if record is None:
            raise SlotUnavailableError(slot, "the run's pinned connection version no longer exists")
        return await _materialise(
            record,
            tenant_id=tenant_id,
            slot=slot,
            policy=policy,
            decryptor=decryptor,
            mode=mode,
        )

    if mode is None or tenant_connection_used(mode, slot):
        record = await repo.get_active(tenant_id, slot)
        if record is not None:
            return await _materialise(
                record,
                tenant_id=tenant_id,
                slot=slot,
                policy=policy,
                decryptor=decryptor,
                mode=mode,
            )
    if mode is not None and not platform_allowed(mode, slot):
        raise SlotNotConfiguredError(slot)
    return _deployment(policy, slot)


async def pin_slots(
    repo: ConnectionRepository,
    tenant_id: UUID,
    *,
    policy: ProviderPolicySettings | None = None,
    posture: ByoPostureReader,
) -> dict[str, SlotRef]:
    """Pin every pinned registry slot for a new run: tenant ref if used, else deployment.

    Grant slots (``publish.youtube``) are NOT pinned: the grant lives in
    youtube_upload and publishing happens after the run, against the grant as it
    is then.

    ``platform_only``: a saved tenant connection on a ``byo_governed`` slot is
    IGNORED (pinned to the platform default). The posture is read at most once,
    and only when such a connection exists — a tenant with no connections never
    depends on licences here. ``posture`` is REQUIRED so a platform_only tenant's
    connection can never be pinned by omission.

    Validity and the ``byo_required`` refusal are NOT checked here: the run-start
    preflight (``providers/preflight.py``) refuses a byo_required run that NEEDS a
    slot it has no connection for, naming every missing slot; a run that never
    narrates must not be refused because the tenant has no TTS provider.
    """
    policy = policy or get_provider_policy()
    pins: dict[str, SlotRef] = {}
    mode: ByoMode | None = None
    for slot in PINNED_SLOT_NAMES:
        record = await repo.get_active(tenant_id, slot)
        if record is not None and get_slot(slot).byo_governed:
            if mode is None:
                mode = await posture.byo_mode(tenant_id, slot=slot)
            if not tenant_connection_used(mode, slot):
                record = None
        if record is not None:
            pins[slot] = SlotRef(
                slot=slot,
                source="tenant",
                kind=record.kind,
                connection_id=record.connection_id,
                version=record.version,
            )
        else:
            pins[slot] = _deployment(policy, slot).ref
    return pins


def pins_to_snapshot(pins: Mapping[str, SlotRef]) -> dict[str, dict[str, str | int | None]]:
    """JSON form of :func:`pin_slots` output for ``config_snapshot['provider_slots']``."""
    return {slot: ref.to_dict() for slot, ref in pins.items()}
