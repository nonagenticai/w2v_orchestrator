"""Slot resolution errors.

Both subclass :class:`PermanentUpstreamError`: no retry can fix a tenant's broken or
missing provider connection, and ``str(err)`` is the sentence the user sees. Both are
attributed ``customer_input`` — the tenant has to act (fix or configure the slot).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal

from src.models.failure_class import CUSTOMER_INPUT, EXTERNAL, OUR_DEFECT, FailureClass
from src.services.providers.registry import BYO_GOVERNED_SLOTS, slot_short_label
from src.services.stages.upstream import PermanentUpstreamError, TransientUpstreamError


class SlotUnavailableError(PermanentUpstreamError):
    """The tenant HAS a connection for the slot but it cannot be used.

    Pinned version missing / purged / owned by another tenant, kind not allowed,
    egress refused, secret undecryptable. Never answered with the deployment
    default: a tenant that configured its own provider must not silently get ours.
    """

    failure_class: FailureClass = CUSTOMER_INPUT

    def __init__(self, slot: str, reason: str) -> None:
        self.slot = slot
        self.reason = reason
        super().__init__(f"Provider '{slot}' is unavailable: {reason}")


class SlotEgressRefusedError(SlotUnavailableError):
    """The connection's ``base_url`` is refused by the SSRF / egress policy."""


#: Machine code the run API carries for "this tenant must bring its own provider".
BYO_PROVIDER_REQUIRED: Final = "byo_provider_required"

#: The model-provider slots a ``byo_required`` tenant must bring itself — every
#: ``byo_governed`` registry slot (owner decision 2026-10-22: all of them, not
#: only the platform-GPU trio). Kept a ``Literal`` so call sites stay typed; the
#: registry equality is pinned by ``tests/services/providers/test_uniform_byo_mode.py``.
ByoModality = Literal["llm", "vlm", "tts", "asr", "image", "video_gen", "ocr"]
BYO_MODALITIES: Final[frozenset[str]] = frozenset(BYO_GOVERNED_SLOTS)


def _join_labels(labels: Sequence[str]) -> str:
    if len(labels) <= 1:
        return "".join(labels)
    return ", ".join(labels[:-1]) + " and " + labels[-1]


class SlotNotConfiguredError(PermanentUpstreamError):
    """A ``byo_required`` tenant has no provider of its own for model slot(s) it needs.

    Raised by the run-start preflight (naming EVERY missing slot), by the resolver
    (no tenant connection, never the platform fallback) and by the clients when
    modality_gateway answers **402** for the tenant. Carries the structured
    ``failure_detail`` the run API surfaces, so the frontend can branch on
    ``code`` + ``modality`` / ``slots`` instead of parsing the sentence.
    """

    failure_class: FailureClass = CUSTOMER_INPUT
    failure_code: str = BYO_PROVIDER_REQUIRED

    def __init__(
        self,
        modality: str | Sequence[str],
        *,
        upstream_detail: str | None = None,
    ) -> None:
        slots: tuple[str, ...] = (
            (modality,) if isinstance(modality, str) else tuple(dict.fromkeys(modality))
        )
        if not slots:
            raise ValueError("SlotNotConfiguredError needs at least one slot")
        unknown = [s for s in slots if s not in BYO_MODALITIES]
        if unknown:
            raise ValueError(f"not byo-governed provider slots: {unknown}")
        #: Every slot the tenant must configure, in first-use order.
        self.slots: tuple[str, ...] = slots
        self.slot: str = slots[0]
        #: The FIRST missing slot (kept for the single-slot wire field).
        self.modality: str = slots[0]
        #: The gateway's own 402 text, kept for logs — never shown to the user.
        self.upstream_detail = upstream_detail
        labels = [slot_short_label(s) for s in slots]
        noun = "provider" if len(slots) == 1 else "providers"
        super().__init__(f"Configure your own {_join_labels(labels)} {noun} to continue")

    @property
    def failure_detail(self) -> dict[str, str]:
        """Wire shape of ``runs.failure_detail`` (``RunDetail.failure_detail``).

        ``failure_detail`` is ``dict[str, str]`` by contract, so ``slots`` is the
        comma-separated list of every missing slot (``modality`` = its first)."""
        return {
            "code": self.failure_code,
            "modality": self.modality,
            "slots": ",".join(self.slots),
            "message": str(self),
        }


class ProviderKeyringUnavailableError(PermanentUpstreamError):
    """A tenant secret exists but ``PROVIDER_SECRET_KEYS`` is empty on this pod.

    A deployment fault (attributed ``our_defect``), not the tenant's: the API maps
    :class:`~src.services.providers.crypto.KeyringNotConfiguredError` to 503, and a
    run stage fails with this rather than blaming the tenant's connection."""

    failure_class: FailureClass = OUR_DEFECT

    def __init__(self, slot: str) -> None:
        self.slot = slot
        super().__init__(
            f"Provider '{slot}' is temporarily unavailable: provider credentials cannot be "
            "read on this deployment."
        )


class ByoPostureUnavailableError(TransientUpstreamError):
    """The tenant's BYO posture could not be read from licences — FAIL CLOSED.

    Never answered with the platform default: an unknown posture is not
    ``local_allowed``. Transient (licences recovering makes the same call succeed)
    and ``external`` (a dependency was unavailable, not our code, not the tenant).
    """

    failure_class: FailureClass = EXTERNAL

    def __init__(self, slot: str, reason: str) -> None:
        self.slot = slot
        self.reason = reason
        super().__init__(
            f"Provider '{slot}' is temporarily unavailable: the organisation's provider "
            f"policy could not be read ({reason})"
        )
