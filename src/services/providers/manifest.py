"""P8 requirements manifest — generated from the slot registry + stage declarations.

``slots``: every slot a tenant can act on — writable connection slots and grant slots
(``publish.youtube``), in registry order. Platform-only slots (``vlm``, ``asr``,
``scrape``, ``media_search``, ``doc_parse``) have nothing to configure and are left
out. ``kinds`` honours the deployment policy (``allowed_kinds``), exactly like
``GET /api/tenant/connections``. ``used_by`` comes from each stage class's
``provider_slots`` (enforced by ``tests/unit/test_stage_provider_slots.py``).

``dependencies``: webscraper + media-discovery — fixed platform dependencies, not
slots (owner 2026-10-09 ruled the platform scrape/search path not tenant-configurable;
they replaced the retired KAG gateway in 2026-10).
"""

from __future__ import annotations

from src.api.schemas.provider_requirements import (
    KindField,
    KindRequirement,
    PlatformDependency,
    ProviderRequirements,
    SlotRequirement,
    SlotUsage,
)
from src.services.blueprints.base import stage_handler_class
from src.services.blueprints.registry import BlueprintRegistry
from src.services.providers.registry import SLOTS, SlotSpec, kind_label
from src.services.providers.settings import ProviderPolicySettings

WEBSCRAPER_DEPENDENCY = PlatformDependency(
    name="webscraper",
    label="Web scraper",
    description=(
        "Platform service that fetches web pages for source ingest (brief.sources url "
        "entries). Provided by the deployment; not tenant-configurable."
    ),
)
MEDIA_DISCOVERY_DEPENDENCY = PlatformDependency(
    name="media_discovery",
    label="Media discovery",
    description=(
        "Platform service that searches stock media for the Find-media panel and the "
        "media_discovery stage (brief.media_discovery). Provided by the deployment; "
        "not tenant-configurable."
    ),
)


def manifest_slots() -> tuple[str, ...]:
    """Slots the manifest lists (and the only ones a requirements GUI can act on)."""
    return tuple(name for name, spec in SLOTS.items() if spec.writable or spec.slot_type == "grant")


def _usage(registry: BlueprintRegistry) -> dict[str, list[SlotUsage]]:
    out: dict[str, list[SlotUsage]] = {}
    for bp in registry.all():
        for spec in bp.stages:
            cls = stage_handler_class(spec)
            if cls is None:
                continue
            for use in cls.provider_slots:
                out.setdefault(use.slot, []).append(
                    SlotUsage(
                        blueprint=bp.id,
                        project_type=bp.project_type,
                        stage=spec.name,
                        when=use.when,
                    )
                )
    return out


def _kind(spec: SlotSpec, kind: str) -> KindRequirement:
    ks = spec.tenant_kind_specs[kind]
    return KindRequirement(
        label=ks.label or kind_label(kind),
        help=ks.help,
        base_url=ks.base_url,
        default_base_url=ks.default_base_url,
        model_required=ks.model_required,
        secret_required=ks.secret_required,
        secret_label=ks.secret_label,
        required_options=list(ks.required_options),
        fields=[
            KindField(
                key=f.key,
                label=f.label,
                required=ks.field_required(f.key),
                help=f.help,
                format=f.format,
            )
            for f in ks.fields
        ],
    )


def build_manifest(
    policy: ProviderPolicySettings, registry: BlueprintRegistry
) -> ProviderRequirements:
    usage = _usage(registry)
    slots: list[SlotRequirement] = []
    for name in manifest_slots():
        spec = SLOTS[name]
        allowed = sorted(k for k in spec.tenant_kinds if policy.kind_allowed(name, k))
        default_kind = policy.deployment_kind(name)
        slots.append(
            SlotRequirement(
                slot=name,
                label=spec.label or name,
                type=spec.slot_type,
                default_kind=default_kind,
                default_kind_label=kind_label(default_kind),
                allowed_kinds=allowed,
                # Tenant-agnostic manifest: no DEPLOYMENT-wide requirement exists any
                # more (#1106). Whether THIS tenant must bring a provider is its licences
                # ``byo_mode`` — served per tenant by ``GET /api/tenant/connections``.
                byo_required=False,
                kinds={k: _kind(spec, k) for k in allowed},
                used_by=usage.get(name, []),
            )
        )
    return ProviderRequirements(
        slots=slots, dependencies=[WEBSCRAPER_DEPENDENCY, MEDIA_DISCOVERY_DEPENDENCY]
    )
