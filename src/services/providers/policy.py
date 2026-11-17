"""Policy checks applied to a tenant connection: kind allow-list and SSRF / egress."""

from __future__ import annotations

import asyncio
from urllib.parse import urlsplit, urlunsplit

from w2v_common.net import SSRFError, validate_public_url

from src.services.providers.errors import SlotEgressRefusedError, SlotUnavailableError
from src.services.providers.registry import get_slot
from src.services.providers.settings import ProviderPolicySettings


def check_kind(policy: ProviderPolicySettings, slot: str, kind: str) -> None:
    """Raise :class:`SlotUnavailableError` unless ``kind`` is a permitted tenant kind."""
    spec = get_slot(slot)
    if spec.slot_type == "grant":
        raise SlotUnavailableError(slot, "this provider is connected through its own grant flow")
    if not spec.writable:
        raise SlotUnavailableError(slot, "this provider is platform provided in this release")
    if kind not in spec.tenant_kinds:
        raise SlotUnavailableError(slot, f"kind {kind!r} is not supported for this provider")
    if not policy.kind_allowed(slot, kind):
        raise SlotUnavailableError(slot, f"kind {kind!r} is not permitted by deployment policy")


#: w2v's OWN hosts: a tenant connection pointing here would make a satellite (or the
#: orchestrator) call itself through the tenant's slot — a request loop, and the
#: tenant's secret sent to our own ingress. ``api.nonagentic.ai`` (the gateway) stays
#: allowed: its GPU paths are legitimate ``nonagentic_endpoint`` targets.
W2V_SELF_HOSTS: frozenset[str] = frozenset(
    {
        "studio.nonagentic.ai",
        "w2v.nonagentic.ai",
        "w2v-orchestrator",
        "w2v-orchestrator.w2v",
        "w2v-orchestrator.w2v.svc",
    }
)


def points_at_w2v_itself(base_url: str) -> bool:
    """True when ``base_url``'s host is one of :data:`W2V_SELF_HOSTS`."""
    try:
        host = (urlsplit(base_url).hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    return host in W2V_SELF_HOSTS


def check_not_self(slot: str, base_url: str) -> None:
    """Raise :class:`SlotEgressRefusedError` for a base URL pointing at w2v itself."""
    if points_at_w2v_itself(base_url):
        raise SlotEgressRefusedError(
            slot, f"endpoint {redact_url(base_url)} is w2v itself (a request loop)"
        )


def redact_url(url: str) -> str:
    """``url`` without userinfo, query and fragment — safe for users and logs."""
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        if ":" in host:  # IPv6 literal
            host = f"[{host}]"
        port = f":{parts.port}" if parts.port is not None else ""
        return urlunsplit((parts.scheme, host + port, parts.path, "", ""))
    except ValueError:
        return "<unparseable url>"


def _sanitise(message: str, url: str) -> str:
    safe = redact_url(url)
    message = message.replace(url, safe)
    try:
        parts = urlsplit(url)
    except ValueError:
        return message
    for leak in (parts.netloc.rpartition("@")[0] if "@" in parts.netloc else "", parts.query):
        if leak:
            message = message.replace(leak, "<redacted>")
    return message


async def check_egress(policy: ProviderPolicySettings, slot: str, base_url: str) -> str | None:
    """Validate ``base_url`` against the SSRF mode; return the IP to pin (``None`` = allowlisted).

    ``hosted``: :func:`validate_public_url` with NO allowlist — private, internal-suffix
    and metadata hosts are refused unconditionally. ``private_ok``: the egress allowlist
    is passed through, so only explicitly listed internal hosts get past the block.
    Called at save/test, at live resolve, and again by the client at connect.

    ``validate_public_url`` resolves DNS with a BLOCKING ``getaddrinfo``, so it runs in
    a worker thread — never on the event loop. Its error text quotes the raw URL,
    which may carry userinfo (``user:password`` before the host) or a query token: the
    message is sanitised and the original exception is NOT chained, so neither the
    user-facing text nor a logged traceback carries the credential.
    """
    allow = policy.egress_allowlist if policy.ssrf_mode == "private_ok" else None
    try:
        return await asyncio.to_thread(validate_public_url, base_url, allow_hosts=allow)
    except (SSRFError, ValueError) as exc:
        reason = _sanitise(str(exc), base_url)
        raise SlotEgressRefusedError(
            slot, f"endpoint {redact_url(base_url)} refused by egress policy ({reason})"
        ) from None
