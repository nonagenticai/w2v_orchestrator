"""Provider policy — operator-set rules for which providers a tenant may use.

Env-prefixed ``PROVIDER_`` (lists are comma-separated):

(Which tenants must bring their OWN model providers is NOT a deployment
setting: it is the tenant's licences ``byo_mode`` — see ``byo_posture.py``. The old
deployment-wide ``PROVIDER_BYO_REQUIRED`` was deleted in #1106; setting it globally
would also have blocked the owner's tenants.)

* ``PROVIDER_ALLOWED_KINDS`` — tenant kinds permitted (empty ⇒ every registry tenant kind).
* ``PROVIDER_SSRF_MODE`` — ``hosted`` (default): a tenant ``base_url`` must be public,
  always. ``private_ok``: a private/internal host is allowed **only** when listed in
  ``PROVIDER_EGRESS_ALLOWLIST`` (self-hosted installs reaching e.g. ``llm.test``).
* ``PROVIDER_EGRESS_ALLOWLIST`` — ``host`` or ``host:port`` entries; consulted only in
  ``private_ok`` (in ``hosted`` it is ignored on purpose — a hosted deployment must
  never let an operator typo open tenant egress into the estate).
* ``PROVIDER_GC_GRACE_SECONDS`` (default 3600) / ``PROVIDER_GC_INTERVAL_SECONDS``
  (default 3600, ``<= 0`` disables) — superseded-secret GC (``gc.py``).
* ``PROVIDER_TEST_RATE_LIMIT`` (default 10) / ``PROVIDER_TEST_RATE_WINDOW_SECONDS``
  (default 60) — live provider tests (``POST …/{slot}/test`` and the test inside
  ``PUT``) allowed per tenant per fixed window (``rate_limit.py``); ``0`` disables.
* ``PROVIDER_DEFAULT_<SLOT>`` — deployment kind per slot (``.`` in a slot name → ``_``);
  ``llm`` defaults to ``nonagentic-wm``; ``tts`` / ``image`` / ``video_gen`` to
  ``nonagentic`` (today's platform path); ``device_farm`` to ``local`` (no farm);
  every other slot to ``platform``.

Kept out of ``src/config.py`` deliberately (that file is owned by the WP1 keyring change).
Unknown slots/kinds are rejected at load — a typo here must fail the pod, not silently
widen or narrow policy.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from src.services.providers.registry import (
    KIND_LOCAL,
    KIND_NONAGENTIC,
    KIND_NONAGENTIC_WM,
    KIND_PLATFORM,
    SLOT_NAMES,
    SLOTS,
)

SsrfMode = Literal["hosted", "private_ok"]

_ALL_TENANT_KINDS = frozenset(k for spec in SLOTS.values() for k in spec.tenant_kinds)


def _csv(value: object) -> object:
    if isinstance(value, str):
        return frozenset(part.strip() for part in value.split(",") if part.strip())
    return value


def default_field_name(slot: str) -> str:
    """``publish.youtube`` → ``default_publish_youtube``."""
    return "default_" + slot.replace(".", "_")


class ProviderPolicySettings(BaseSettings):
    """Operator policy for provider slots (see module docstring for semantics)."""

    allowed_kinds: Annotated[frozenset[str], NoDecode] = Field(default_factory=frozenset)
    ssrf_mode: SsrfMode = "hosted"
    egress_allowlist: Annotated[frozenset[str], NoDecode] = Field(default_factory=frozenset)

    #: Secret GC: keep versions superseded less than this long ago (uncommitted pins).
    gc_grace_seconds: int = Field(default=3600, ge=0)
    #: Secret GC loop interval; ``<= 0`` disables the loop.
    gc_interval_seconds: int = 3600

    #: Live provider tests per tenant per window (``rate_limit.py``); ``0`` disables.
    test_rate_limit: int = Field(default=10, ge=0)
    #: Fixed-window length for ``test_rate_limit``.
    test_rate_window_seconds: int = Field(default=60, ge=1)

    #: Service-lab generations (``/api/services/*``) per tenant per window; ``0``
    #: disables. A generation is a real GPU/provider spend, so the default is a
    #: full hour rather than the live-test minute.
    service_lab_rate_limit: int = Field(default=60, ge=0)
    #: Fixed-window length for ``service_lab_rate_limit``.
    service_lab_rate_window_seconds: int = Field(default=3600, ge=1)

    default_llm: str = KIND_NONAGENTIC_WM
    default_vlm: str = KIND_PLATFORM
    #: Whether the PLATFORM default behind the ``vlm`` slot can actually see.
    #: That slot has no backend of its own — it rides the platform LLM with an
    #: image content part — so its capability is a property of whichever model
    #: is served. False today: pgx's vLLM serves a text-only model, an image
    #: block is a 400 upstream, and the bench reported "the provider failed"
    #: for what is really a deployment gap (measured 2026-11-02). Flip to true
    #: only when the served model accepts image parts; until then the bench
    #: refuses the platform path with 409 instead of blaming a provider.
    platform_vlm_enabled: bool = False
    default_tts: str = KIND_NONAGENTIC
    default_asr: str = KIND_PLATFORM
    default_image: str = KIND_NONAGENTIC
    default_video_gen: str = KIND_NONAGENTIC
    default_ocr: str = KIND_PLATFORM
    default_publish_youtube: str = KIND_PLATFORM
    default_device_farm: str = KIND_LOCAL
    default_scrape: str = KIND_PLATFORM
    default_media_search: str = KIND_PLATFORM
    default_doc_parse: str = KIND_PLATFORM

    model_config = SettingsConfigDict(
        env_prefix="PROVIDER_", env_file=".env", case_sensitive=False, extra="ignore"
    )

    @field_validator("allowed_kinds", "egress_allowlist", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        return _csv(value)

    @field_validator("egress_allowlist")
    @classmethod
    def _lower_hosts(cls, value: frozenset[str]) -> frozenset[str]:
        return frozenset(h.lower() for h in value)

    @model_validator(mode="after")
    def _check_vocabulary(self) -> ProviderPolicySettings:
        unknown_kinds = self.allowed_kinds - _ALL_TENANT_KINDS
        if unknown_kinds:
            raise ValueError(f"PROVIDER_ALLOWED_KINDS names unknown kinds: {sorted(unknown_kinds)}")
        for slot in SLOT_NAMES:
            kind = self.deployment_kind(slot)
            if kind not in SLOTS[slot].deployment_kinds:
                raise ValueError(
                    f"PROVIDER_{default_field_name(slot).upper()}={kind!r} is not a deployment "
                    f"kind of slot {slot!r} (allowed: {sorted(SLOTS[slot].deployment_kinds)})"
                )
        return self

    def deployment_kind(self, slot: str) -> str:
        """The deployment-default kind configured for ``slot``."""
        value = getattr(self, default_field_name(slot))
        if not isinstance(value, str):  # pragma: no cover - registry/field drift
            raise TypeError(f"no PROVIDER_DEFAULT field for slot {slot!r}")
        return value

    def kind_allowed(self, slot: str, kind: str) -> bool:
        """True when ``kind`` is a tenant kind of ``slot`` and permitted by policy."""
        if kind not in SLOTS[slot].tenant_kinds:
            return False
        return not self.allowed_kinds or kind in self.allowed_kinds


@lru_cache(maxsize=1)
def get_provider_policy() -> ProviderPolicySettings:
    """Process-wide policy (cached; tests pass an explicit instance instead)."""
    return ProviderPolicySettings()
