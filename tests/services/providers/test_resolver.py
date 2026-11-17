"""P3 WP2 — slot resolver, policy, pinning. Fakes stand in for WP1's repository + crypto."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from src.models.failure_class import CUSTOMER_INPUT
from src.services.providers import (
    PROVIDER_SLOTS_KEY,
    SLOT_NAMES,
    ProviderPolicySettings,
    SlotNotConfiguredError,
    SlotRef,
    SlotUnavailableError,
    pin_slots,
    pins_to_snapshot,
    secret_aad,
)
from src.services.providers import resolve_slot as _resolve_slot
from src.services.providers.byo_posture import ByoPostureReader, StaticByoPosture
from src.services.providers.errors import SlotEgressRefusedError
from src.services.providers.policy import check_egress
from src.services.providers.registry import SLOTS, writable_slots
from src.services.stages.upstream import PermanentUpstreamError

TENANT = UUID("11111111-1111-4111-8111-111111111111")
OTHER_TENANT = UUID("22222222-2222-4222-8222-222222222222")
CONN = UUID("33333333-3333-4333-8333-333333333333")
PUBLIC_URL = "https://1.1.1.1/v1"  # IP literal: validates without DNS
PLAINTEXT = "sk-test-not-real"  # pragma: allowlist secret


@dataclass(frozen=True)
class Rec:
    version: int
    tenant_id: UUID = TENANT
    slot: str = "llm"
    connection_id: UUID = CONN
    kind: str = "openai-compatible"
    base_url: str | None = PUBLIC_URL
    model: str | None = "gpt-x"
    options: Mapping[str, object] = field(default_factory=dict)
    secret_ciphertext: bytes | None = None
    purged_at: datetime | None = None

    def sealed(self) -> Rec:
        """Same record with a ciphertext the fake decryptor accepts only under this row's AAD."""
        aad = secret_aad(self.tenant_id, self.connection_id, self.version)
        return replace(self, secret_ciphertext=aad + b"#" + PLAINTEXT.encode())


class FakeRepo:
    def __init__(self, active: Rec | None = None, versions: list[Rec] | None = None) -> None:
        self.active = active
        self.versions = {(r.connection_id, r.version): r for r in versions or []}
        if active is not None:
            self.versions.setdefault((active.connection_id, active.version), active)
        self.active_calls: list[tuple[UUID, str]] = []

    async def get_active(self, tenant_id: UUID, slot: str) -> Rec | None:
        self.active_calls.append((tenant_id, slot))
        if self.active and self.active.tenant_id == tenant_id and self.active.slot == slot:
            return self.active
        return None

    async def get_version(self, connection_id: UUID, version: int) -> Rec | None:
        return self.versions.get((connection_id, version))


class FakeDecryptor:
    """Accepts ``aad#plaintext``; anything else (wrong AAD, garbage) raises — like AES-GCM."""

    def decrypt(self, ciphertext: bytes, aad: bytes) -> str:
        prefix, sep, rest = ciphertext.partition(b"#")
        if not sep or prefix != aad:
            raise ValueError("auth tag mismatch")
        return rest.decode()


def policy(**kw: object) -> ProviderPolicySettings:
    return ProviderPolicySettings.model_validate(kw)


DEC = FakeDecryptor()
LOCAL = StaticByoPosture("local_allowed")


async def resolve_slot(*args: object, posture: ByoPostureReader = LOCAL, **kwargs: object):
    """``resolve_slot`` with an owner-tenant (``local_allowed``) posture unless overridden."""
    return await _resolve_slot(*args, posture=posture, **kwargs)  # ty: ignore[invalid-argument-type]  # why: a forwarding shim over the real signature; each test passes the real types


def pinned(rec: Rec) -> dict[str, object]:
    ref = SlotRef(
        slot="llm",
        source="tenant",
        kind=rec.kind,
        connection_id=rec.connection_id,
        version=rec.version,
    )
    return {PROVIDER_SLOTS_KEY: {"llm": ref.to_dict()}}


# ── registry ────────────────────────────────────────────────────────────────


def test_registry_lists_all_slots_and_the_p5_writable_set() -> None:
    assert set(SLOT_NAMES) == {
        "llm", "vlm", "tts", "asr", "image", "video_gen", "ocr", "publish.youtube",
        "device_farm", "scrape", "media_search", "doc_parse",
    }  # fmt: skip
    assert writable_slots() == (
        "llm", "vlm", "tts", "asr", "image", "video_gen", "ocr", "device_farm",
    )  # fmt: skip
    assert SLOTS["publish.youtube"].slot_type == "grant"
    assert not SLOTS["publish.youtube"].writable
    assert SLOTS["llm"].tenant_kinds == {"openai-compatible", "anthropic"}


# ── no slot → deployment ref ────────────────────────────────────────────────


@pytest.mark.parametrize("slot", ["llm"])
async def test_no_tenant_slot_resolves_to_deployment_default(slot: str) -> None:
    repo = FakeRepo()
    got = await resolve_slot(repo, DEC, TENANT, slot, policy=policy())
    assert got.source == "deployment"
    assert got.kind == {"llm": "nonagentic-wm", "tts": "nonagentic"}.get(slot, "platform")
    assert got.secret is None and got.ref.connection_id is None
    assert repo.active_calls == [(TENANT, slot)]


# ── broken tenant slot raises, never falls back ─────────────────────────────


@pytest.mark.parametrize(
    ("label", "rec", "pol", "error"),
    [
        (
            "kind not allowed by policy",
            Rec(1, kind="anthropic"),
            {"allowed_kinds": "openai-compatible"},
            SlotUnavailableError,
        )
    ],
)
async def test_invalid_tenant_slot_raises_never_falls_back(
    label: str, rec: Rec, pol: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error) as exc_info:
        await resolve_slot(FakeRepo(active=rec), DEC, TENANT, rec.slot, policy=policy(**pol))
    assert isinstance(exc_info.value, PermanentUpstreamError), label
    assert exc_info.value.failure_class == CUSTOMER_INPUT


async def test_valid_tenant_slot_is_used_with_decrypted_secret() -> None:
    rec = Rec(3, options={"temperature": 0.2}).sealed()
    got = await resolve_slot(FakeRepo(active=rec), DEC, TENANT, "llm", policy=policy())
    assert got.ref == SlotRef("llm", "tenant", "openai-compatible", CONN, 3)
    assert got.secret is not None and got.secret.get_secret_value() == PLAINTEXT
    assert PLAINTEXT not in repr(got)
    assert got.pinned_ip == "1.1.1.1"
    assert got.options == {"temperature": 0.2}


# ── byo_required: the tenant's licences posture (#1106) ─────────────────────
# Whether a byo_required tenant may use the PLATFORM is decided at run start
# (tests/services/providers/test_byo_preflight.py) and backstopped by the gateway's
# 402. The resolver's own job is narrower: never accept a tenant connection that is
# really our GPU (an allowlisted platform-internal host) for such a tenant.

BYO = StaticByoPosture("byo_required")
_PRIVATE_OK = {"ssrf_mode": "private_ok", "egress_allowlist": "tts.test:9202"}
_KOKORO = "http://tts.test:9202/v1"


class _BrokenPosture:
    """Licences unreachable: the reader raises, exactly like the production one."""

    def __init__(self) -> None:
        self.calls = 0

    async def byo_mode(self, tenant_id: UUID, *, slot: str):
        from src.services.providers.errors import ByoPostureUnavailableError

        self.calls += 1
        raise ByoPostureUnavailableError(slot, "licences answered 503")


def _kokoro_rec(slot: str = "tts") -> Rec:
    return Rec(1, slot=slot, kind="openai-compatible", base_url=_KOKORO, options={"voice": "a"})


async def test_byo_required_connection_on_a_platform_host_is_refused() -> None:
    with pytest.raises(SlotNotConfiguredError) as exc_info:
        await resolve_slot(
            FakeRepo(active=_kokoro_rec()),
            DEC,
            TENANT,
            "tts",
            policy=policy(**_PRIVATE_OK),
            posture=BYO,
        )
    err = exc_info.value
    assert isinstance(err, PermanentUpstreamError)
    assert err.failure_class == CUSTOMER_INPUT
    assert err.failure_detail["code"] == "byo_provider_required"
    assert err.failure_detail["modality"] == "tts"


async def test_local_allowed_may_use_an_allowlisted_platform_host() -> None:
    posture = StaticByoPosture("local_allowed")
    got = await resolve_slot(
        FakeRepo(active=_kokoro_rec()),
        DEC,
        TENANT,
        "tts",
        policy=policy(**_PRIVATE_OK),
        posture=posture,
    )
    assert got.source == "tenant" and got.base_url == _KOKORO
    assert posture.calls == [TENANT]


async def test_a_byo_required_tenants_public_connection_is_used() -> None:
    posture = StaticByoPosture("byo_required")
    rec = Rec(1, slot="tts", kind="openai", base_url=PUBLIC_URL, options={"voice": "alloy"})
    got = await resolve_slot(
        FakeRepo(active=rec), DEC, TENANT, "tts", policy=policy(), posture=posture
    )
    assert got.source == "tenant" and posture.calls == [TENANT]  # read ONCE


async def test_byo_required_llm_on_an_allowlisted_platform_host_is_refused() -> None:
    """2026-10-22: llm is governed like tts — our host behind the tenant's slot is ours."""
    posture = StaticByoPosture("byo_required")
    rec = Rec(1, base_url="http://llm.test:8001/v1")
    pol = policy(ssrf_mode="private_ok", egress_allowlist="llm.test:8001")
    with pytest.raises(SlotNotConfiguredError) as info:
        await resolve_slot(FakeRepo(active=rec), DEC, TENANT, "llm", policy=pol, posture=posture)
    assert info.value.slots == ("llm",)


@pytest.mark.parametrize("slot", ["device_farm"])
async def test_byo_posture_is_not_consulted_outside_the_model_slots(slot: str) -> None:
    posture = StaticByoPosture("byo_required")
    got = await resolve_slot(FakeRepo(), DEC, TENANT, slot, policy=policy(), posture=posture)
    assert got.source == "deployment" and posture.calls == []


async def test_an_unreadable_posture_on_a_platform_host_fails_closed() -> None:
    from src.models.failure_class import EXTERNAL
    from src.services.providers.errors import ByoPostureUnavailableError
    from src.services.stages.upstream import TransientUpstreamError

    broken = _BrokenPosture()
    with pytest.raises(ByoPostureUnavailableError) as exc_info:
        await resolve_slot(
            FakeRepo(active=_kokoro_rec()),
            DEC,
            TENANT,
            "tts",
            policy=policy(**_PRIVATE_OK),
            posture=broken,
        )
    assert isinstance(exc_info.value, TransientUpstreamError)
    assert exc_info.value.failure_class == EXTERNAL
    assert broken.calls == 1


async def test_pin_slots_does_not_refuse_a_run_up_front() -> None:
    """Pinning records references only; the BYO refusal is the run-start preflight's."""
    pins = await pin_slots(
        FakeRepo(), TENANT, policy=policy(), posture=StaticByoPosture("byo_required")
    )
    assert pins["tts"].source == "deployment" and pins["image"].source == "deployment"


def test_provider_byo_required_env_no_longer_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROVIDER_BYO_REQUIRED", "tts")
    got = ProviderPolicySettings(_env_file=None)
    assert not hasattr(got, "byo_required")


# ── pin vs live / legacy ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("label", "snapshot", "want_version"),
    [("pinned v1 wins over live v2", pinned(Rec(1)), 1)],
)
async def test_pinned_version_vs_live(
    label: str, snapshot: dict[str, object] | None, want_version: int
) -> None:
    repo = FakeRepo(active=Rec(2), versions=[Rec(1)])
    got = await resolve_slot(repo, DEC, TENANT, "llm", run_snapshot=snapshot, policy=policy())
    assert got.ref.version == want_version, label


async def test_pinned_deployment_ref_stays_deployment_after_tenant_adds_slot() -> None:
    pins = await pin_slots(FakeRepo(), TENANT, posture=StaticByoPosture("local_allowed"))
    snapshot = {PROVIDER_SLOTS_KEY: pins_to_snapshot(pins)}
    repo = FakeRepo(active=Rec(1))  # tenant configured llm after the run pinned
    got = await resolve_slot(repo, DEC, TENANT, "llm", run_snapshot=snapshot, policy=policy())
    assert got.source == "deployment" and got.kind == "nonagentic-wm"
    assert repo.active_calls == []


@pytest.mark.parametrize(
    ("label", "repo"),
    [("pinned version missing", FakeRepo(active=Rec(2)))],
)
async def test_unusable_pinned_version_raises(label: str, repo: FakeRepo) -> None:
    with pytest.raises(SlotUnavailableError):
        await resolve_slot(repo, DEC, TENANT, "llm", run_snapshot=pinned(Rec(1)), policy=policy())


async def test_cross_tenant_pin_refused() -> None:
    foreign = Rec(1, tenant_id=OTHER_TENANT).sealed()
    repo = FakeRepo(versions=[foreign])
    with pytest.raises(SlotUnavailableError, match="another tenant"):
        await resolve_slot(repo, DEC, TENANT, "llm", run_snapshot=pinned(foreign), policy=policy())


async def test_pin_slots_pins_every_slot_and_round_trips() -> None:
    pins = await pin_slots(
        FakeRepo(active=Rec(4)), TENANT, policy=policy(), posture=StaticByoPosture("local_allowed")
    )
    assert set(pins) == set(SLOT_NAMES) - {"publish.youtube"}
    assert pins["llm"] == SlotRef("llm", "tenant", "openai-compatible", CONN, 4)
    assert all(pins[s].source == "deployment" for s in pins if s != "llm")
    raw = pins_to_snapshot(pins)
    assert {s: SlotRef.from_dict(s, d) for s, d in raw.items()} == pins


@pytest.mark.parametrize(
    "entry",
    [{"source": "tenant", "kind": "anthropic"}],
)
async def test_malformed_pin_raises(entry: object) -> None:
    snapshot = {PROVIDER_SLOTS_KEY: {"llm": entry}}
    with pytest.raises(SlotUnavailableError):
        await resolve_slot(FakeRepo(active=Rec(1)), DEC, TENANT, "llm", run_snapshot=snapshot)


# ── SSRF / egress ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("mode", "allow", "url", "allowed"),
    [("hosted", "", PUBLIC_URL, True)],
)
async def test_ssrf_policy(mode: str, allow: str, url: str, allowed: bool) -> None:
    pol = policy(ssrf_mode=mode, egress_allowlist=allow)
    repo = FakeRepo(active=Rec(1, base_url=url))
    if allowed:
        await check_egress(pol, "llm", url)
        got = await resolve_slot(repo, DEC, TENANT, "llm", policy=pol)
        assert got.base_url == url
    else:
        with pytest.raises(SlotEgressRefusedError):
            await check_egress(pol, "llm", url)
        with pytest.raises(SlotEgressRefusedError):
            await resolve_slot(repo, DEC, TENANT, "llm", policy=pol)


# ── settings ────────────────────────────────────────────────────────────────


def test_policy_reads_prefixed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROVIDER_ALLOWED_KINDS", "anthropic, openai-compatible")
    monkeypatch.setenv("PROVIDER_SSRF_MODE", "private_ok")
    monkeypatch.setenv("PROVIDER_EGRESS_ALLOWLIST", "LLM.test,gw.test:8443")
    got = ProviderPolicySettings(_env_file=None)
    assert got.allowed_kinds == {"anthropic", "openai-compatible"}
    assert got.ssrf_mode == "private_ok"
    assert got.egress_allowlist == {"llm.test", "gw.test:8443"}
    assert got.deployment_kind("llm") == "nonagentic-wm"
    assert got.deployment_kind("publish.youtube") == "platform"


@pytest.mark.parametrize(
    "kw",
    [{"allowed_kinds": "no-such-kind"}],
)
def test_policy_rejects_unknown_vocabulary(kw: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        policy(**kw)


# ── review fixes (WP1/WP2) ─────────────────────────────────────────────────


def test_secret_aad_is_the_crypto_modules_aad() -> None:
    """One AAD definition: a drifted copy would make every resolve fail closed."""
    from src.services.providers import crypto

    assert secret_aad is crypto.aad


async def test_check_egress_runs_dns_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    from src.services.providers import policy as policy_mod

    seen: list[str] = []

    def fake_validate(url: str, *, allow_hosts: object = None) -> str:
        seen.append(threading.current_thread().name)
        return "1.1.1.1"

    monkeypatch.setattr(policy_mod, "validate_public_url", fake_validate)
    assert await check_egress(policy(), "llm", PUBLIC_URL) == "1.1.1.1"
    assert seen and seen[0] != threading.main_thread().name


async def test_egress_refusal_never_leaks_userinfo_or_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from w2v_common.net import SSRFError

    from src.services.providers import policy as policy_mod

    url = "https://admin:sk-secret-key@10.0.0.5:8443/v1?token=tok-123"  # pragma: allowlist secret

    def fake_validate(raw: str, *, allow_hosts: object = None) -> str:
        raise SSRFError(
            f"refusing {raw}: private address (user admin:sk-secret-key, q token=tok-123)"
        )

    monkeypatch.setattr(policy_mod, "validate_public_url", fake_validate)
    with pytest.raises(SlotEgressRefusedError) as caught:
        await check_egress(policy(), "llm", url)
    text = str(caught.value)
    for leak in ("sk-secret-key", "admin:", "tok-123"):
        assert leak not in text, text
    assert "https://10.0.0.5:8443/v1" in text
    assert caught.value.__cause__ is None and caught.value.__suppress_context__


async def test_empty_keyring_is_our_fault_not_the_tenants() -> None:
    from src.models.failure_class import OUR_DEFECT
    from src.services.providers import ProviderKeyringUnavailableError
    from src.services.providers.crypto import Keyring

    class RingDecryptor:
        def decrypt(self, ciphertext: bytes, aad: bytes) -> str:
            return Keyring([]).decrypt_with_aad(ciphertext, aad)

    repo = FakeRepo(active=Rec(1).sealed())
    with pytest.raises(ProviderKeyringUnavailableError) as caught:
        await resolve_slot(repo, RingDecryptor(), TENANT, "llm", policy=policy())
    assert caught.value.failure_class == OUR_DEFECT
