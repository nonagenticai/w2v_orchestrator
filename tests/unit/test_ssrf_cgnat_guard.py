"""SSRF guard: RFC 6598 CGNAT (100.64.0.0/10) is THIS estate's Tailnet.

``ipaddress.ip_address("100.64.0.6").is_private`` is **False**, so the old
four-flag predicate (``is_private or is_loopback or is_link_local or
is_reserved``) let every Tailnet host through: postgres ``100.64.0.6``,
registry ``100.64.0.7``, k3s master ``100.64.0.5``, worker1
``100.64.0.8``.

Two independent holes existed and both are covered here:

1. ``visual_media._host_is_blocked`` never resolved DNS at all — it only parsed
   the host as a literal IP and returned ``False`` for any *name*. So
   ``http://<name-that-resolves-to-100.64.0.6>/`` sailed straight through.
2. Both call sites built their client with ``follow_redirects=True``, so even a
   correct first-hop check was escapable with a single ``302``.

A third sink, ``ken_burns._fetch_own_asset``, had BOTH holes and was missed by
the first pass: it fetched ``shot["image_url"]`` with a bare
``client.stream("GET", url)`` on a ``follow_redirects=True`` client and no host
check at all. It is guarded now too — but with ``allow_hosts`` set to the
configured storage endpoints, because the legitimate target is our own presigned
MinIO artifact, whose host resolves INTO the CGNAT range the guard blocks.

All three now route through ``w2v_common.net.safe_get``, which resolves once,
pins the connection to the validated IP, and follows redirects MANUALLY with a
re-validation on every hop.

``SSRFError`` subclasses ``ValueError``, so every existing caller
(``media_discovery._republish_images``, ``source_ingest._pin_media``,
``site_capture._scrape_page_text``, ``ken_burns.apply_ken_burns`` — all
``except Exception``) still soft-fails / degrades exactly as before rather than
500-ing the run.
"""

from __future__ import annotations

import ipaddress
import re
import socket

import httpx
import pytest
from w2v_common.net import SSRFError, safe_fetch

from src.services.stages.generic.site_capture import SiteCaptureStage
from src.services.stages.shared import ken_burns
from src.services.visual_media import fetch_image_bytes, fetch_video_bytes

# Names → addresses for the stubbed resolver. No test may touch real DNS.
_DNS = {
    "example.com": "93.184.216.34",
    "cdn.example.com": "93.184.216.34",
    "evil.example.com": "93.184.216.34",  # public first hop, redirects internal
    # A perfectly ordinary-looking hostname that resolves onto the Tailnet.
    # This is the case the old no-DNS guard could not see AT ALL.
    "assets.attacker.example": "100.64.0.6",  # == postgres.internal
    # Our OWN artifact host also resolves into the CGNAT /10 — which is exactly
    # why ken_burns must allow-list it rather than drop the guard.
    "minio.internal": "100.64.0.5",
}


@pytest.fixture(autouse=True)
def _stub_dns(monkeypatch) -> None:
    def fake_getaddrinfo(host, port, *args, **kwargs):
        addr = _DNS.get(host)
        if addr is None:
            try:
                ipaddress.ip_address(host)  # literal IP resolves to itself
            except ValueError:
                raise socket.gaierror(f"stubbed DNS: unknown host {host!r}") from None
            addr = host
        family = socket.AF_INET6 if ":" in addr else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (addr, port or 0))]

    monkeypatch.setattr(safe_fetch.socket, "getaddrinfo", fake_getaddrinfo)


def _png_client(calls: list[httpx.Request] | None = None) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        return httpx.Response(200, headers={"content-type": "image/png"}, content=b"PNG")

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# --- 1. the no-DNS bypass: a NAME that resolves onto the Tailnet -------------


@pytest.mark.asyncio
async def test_hostname_resolving_to_tailnet_cgnat_is_rejected() -> None:
    """The old guard returned False for any non-literal host — this is the proof."""
    with pytest.raises(SSRFError, match=re.escape("100.64.0.6")):
        await fetch_image_bytes("https://assets.attacker.example/x.png", client=_png_client())


@pytest.mark.asyncio
async def test_ssrf_error_is_a_valueerror_so_callers_still_soft_fail() -> None:
    # media_discovery / source_ingest catch bare `Exception`; the fetchers'
    # documented contract is ValueError. SSRFError must stay inside it.
    assert issubclass(SSRFError, ValueError)
    with pytest.raises(ValueError):
        await fetch_video_bytes("https://assets.attacker.example/clip.mp4", client=_png_client())


# --- 2. CGNAT literals, including the IPv4-mapped IPv6 form ------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://100.64.0.1/x",  # bottom of the /10
        "http://100.64.0.7/x",  # registry.internal (worker3)
        "http://100.64.0.6/x",  # postgres.internal primary
        "http://100.64.0.5/x",  # k3s master
        "http://100.64.0.8/x",  # worker1
        "http://[::ffff:100.64.0.6]/x",  # IPv4-mapped IPv6 — same host, other notation
    ],
)
@pytest.mark.asyncio
async def test_cgnat_literals_are_rejected(url: str) -> None:
    with pytest.raises(SSRFError, match="blocked address"):
        await fetch_image_bytes(url, client=_png_client())


# --- 3. the redirect hole ----------------------------------------------------


@pytest.mark.asyncio
async def test_302_from_public_host_to_cgnat_is_rejected_mid_chain() -> None:
    """A public first hop that 302s to the k3s master must NOT be followed.

    The clients used to run ``follow_redirects=True``, so httpx followed this
    hop internally and the guard never saw the internal target.
    """
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://100.64.0.5/x"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(SSRFError, match=re.escape("100.64.0.5")):
        await fetch_image_bytes("https://evil.example.com/x.png", client=client)

    # Exactly ONE request left the process: the public first hop. The redirect
    # target was rejected at validation, before any connection to it.
    assert len(seen) == 1
    assert "100.64.0.5" not in seen[0]


@pytest.mark.asyncio
async def test_site_capture_scrape_degrades_on_redirect_to_cgnat(monkeypatch) -> None:
    """site_capture's scrape soft-fails to "" + a note — never a hard 500."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://100.64.0.5/x"})

    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient

    def _patched(*args, **kwargs):
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _patched)
    notes: list[str] = []
    text, status = await SiteCaptureStage._scrape_page_text("https://evil.example.com/", notes)

    assert text == ""
    # The first hop was public, so the pre-flight passed and the refusal happened
    # inside the fetch (SSRFError on the redirect hop) → "fetch_failed", the status
    # promo_script treats as a hard grounding failure. It must NOT read as "empty".
    assert status == "fetch_failed"
    assert notes and "page scrape failed" in notes[0]
    assert len(seen) == 1  # the internal redirect target was never contacted


def test_site_capture_guard_rejects_cgnat_name_and_literal() -> None:
    assert SiteCaptureStage._safe_to_fetch("https://assets.attacker.example/") is False
    assert SiteCaptureStage._safe_to_fetch("http://100.64.0.7/") is False
    assert SiteCaptureStage._safe_to_fetch("https://example.com/") is True


# --- 4. the deny must be the /10 — not startswith("100.") --------------------


@pytest.mark.asyncio
async def test_public_url_still_accepted() -> None:
    data, fmt = await fetch_image_bytes("https://example.com/img.png", client=_png_client())
    assert (data, fmt) == (b"PNG", "png")


@pytest.mark.parametrize(
    "ip",
    [
        "100.63.255.255",  # one BELOW 100.64.0.0/10
        "100.128.0.0",  # one ABOVE the /10 (100.64.0.0 - 100.127.255.255)
    ],
)
@pytest.mark.asyncio
async def test_cgnat_boundary_addresses_outside_the_slash10_are_allowed(ip: str) -> None:
    """Proves the block is the /10, not a naive ``startswith("100.")``."""
    assert ipaddress.ip_address(ip) not in ipaddress.ip_network("100.64.0.0/10")
    data, _ = await fetch_image_bytes(f"http://{ip}/x.png", client=_png_client())
    assert data == b"PNG"


# --- 5. ken_burns: the third sink, missed by the original conversion ---------
#
# ``ken_burns._fetch_own_asset`` fetched ``shot["image_url"]`` with a bare
# ``client.stream("GET", url)`` on a ``follow_redirects=True`` client — no host
# check, no redirect re-validation. It is guarded now, but it must keep fetching
# OUR OWN presigned MinIO artifact, whose host resolves into the very CGNAT range
# the guard blocks. Hence ``allow_hosts``, not "no guard".


class _StorageStub:
    minio_endpoint = "minio.internal:8389"
    minio_external_endpoint = None


class _SettingsStub:
    storage = _StorageStub()


@pytest.fixture
def _minio_settings(monkeypatch) -> None:
    """Pin the storage config ken_burns derives its allow-list from."""
    monkeypatch.setattr(ken_burns, "get_settings", _SettingsStub)


@pytest.mark.asyncio
async def test_ken_burns_hostname_resolving_to_tailnet_is_rejected(_minio_settings) -> None:
    """A shot image_url on a name that resolves onto the Tailnet must not be fetched."""
    with pytest.raises(SSRFError, match=re.escape("100.64.0.6")):
        await ken_burns._fetch_own_asset("https://assets.attacker.example/x.png", _png_client())


@pytest.mark.asyncio
async def test_ken_burns_redirect_into_tailnet_is_rejected_mid_chain(_minio_settings) -> None:
    """The 302 hole: a public first hop that redirects to the k3s master."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://100.64.0.5/x"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(SSRFError, match=re.escape("100.64.0.5")):
        await ken_burns._fetch_own_asset("https://evil.example.com/x.png", client)

    assert len(seen) == 1  # the internal redirect target was never contacted
    assert "100.64.0.5" not in seen[0]


@pytest.mark.asyncio
async def test_ken_burns_allow_listed_minio_host_cannot_redirect_into_the_estate(
    _minio_settings,
) -> None:
    """``allow_hosts`` is re-checked per hop — trusting MinIO does not trust its 302."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://100.64.0.6/x"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(SSRFError, match=re.escape("100.64.0.6")):
        await ken_burns._fetch_own_asset("https://minio.internal:8389/w2v/a.png", client)

    assert len(seen) == 1


@pytest.mark.asyncio
async def test_ken_burns_still_fetches_our_own_presigned_minio_artifact(_minio_settings) -> None:
    """The regression guard: the allow-list must not break the legitimate fetch.

    ``minio.internal`` resolves to ``100.64.0.5`` (CGNAT), so without
    ``allow_hosts`` every ken_burns shot would soft-fail and the feature would go
    silently inert. The request must also go out UN-pinned (original hostname), or
    the presigned SigV4 ``Host`` signature would not verify.
    """
    seen: list[httpx.Request] = []
    data, content_type = await ken_burns._fetch_own_asset(
        "https://minio.internal:8389/w2v/a.png?X-Amz-Signature=deadbeef", _png_client(seen)
    )

    assert (data, content_type) == (b"PNG", "image/png")
    assert str(seen[0].url).startswith("https://minio.internal:8389/")
    assert seen[0].headers["host"] == "minio.internal:8389"


def test_ken_burns_allow_hosts_are_only_the_configured_storage_endpoints(_minio_settings) -> None:
    assert ken_burns._artifact_allow_hosts() == frozenset({"minio.internal", "minio.internal:8389"})


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("", set()),
        ("   ", set()),
        (None, set()),
        ("minio.internal", {"minio.internal"}),
        ("minio.internal:8389", {"minio.internal", "minio.internal:8389"}),
        ("https://minio.example.com", {"minio.example.com"}),
        ("http://minio.example.com:9000", {"minio.example.com", "minio.example.com:9000"}),
    ],
)
def test_ken_burns_endpoint_host_keys_accepts_every_configured_shape(
    endpoint: str | None, expected: set[str]
) -> None:
    """StorageSettings permits host, host:port, and scheme://host[:port]."""
    assert ken_burns._endpoint_host_keys(endpoint) == expected
