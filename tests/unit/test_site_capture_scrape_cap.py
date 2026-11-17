"""``site_capture._scrape_page_text`` states its own response-body ceiling.

w2v-common 0.55.0 gave ``safe_get`` a 64 MiB *default* body cap
(``DEFAULT_MAX_FETCH_BYTES``) and raises ``ResponseTooLargeError`` past it.
That exception **subclasses ``ValueError``**, and ``_scrape_page_text`` wraps its
fetch in a bare ``except Exception`` that degrades to ``""``.

So the version bump alone does not make this call site safe — it makes it
*quietly* bounded at a ceiling that was chosen for untrusted **media**, and an
over-cap page turns into empty script grounding with no error anywhere. The fix
is an explicit ``max_fetch_bytes=_SCRAPE_MAX_RESPONSE_BYTES`` at the call site.

These tests pin three things:

* the stated ceiling is actually enforced (and is the *stated* one, not the
  inherited default — the mutation that deletes the kwarg must turn them red);
* the degrade is not **silent**: the note names the byte cap, so an operator can
  tell a too-large page from a network failure;
* the false-positive control — an ordinary page still scrapes, and a body just
  under the ceiling is still fetched in full.

``MockTransport`` is legitimate here: the cap is enforced inside ``safe_fetch``
itself (Content-Length check, then a streaming byte counter), strictly above the
transport. It would NOT be legitimate for the RFC 9110 method guard, where h11 —
the layer that actually rejects a CRLF method on the wire — is absent under a
mock; nothing here depends on that.
"""

from __future__ import annotations

import ipaddress
import socket

import httpx
import pytest
from w2v_common.net import DEFAULT_MAX_FETCH_BYTES, ResponseTooLargeError, safe_fetch

from src.services.stages.generic import site_capture
from src.services.stages.generic.site_capture import (
    _SCRAPE_MAX_RESPONSE_BYTES,
    SiteCaptureStage,
)

_HOST = "example.com"
_URL = f"https://{_HOST}/page"


@pytest.fixture(autouse=True)
def _stub_dns(monkeypatch) -> None:
    """No test may touch real DNS. ``example.com`` resolves public."""

    def fake_getaddrinfo(host, port, *args, **kwargs):
        addr = "93.184.216.34" if host == _HOST else None
        if addr is None:
            try:
                ipaddress.ip_address(host)
            except ValueError:
                raise socket.gaierror(f"stubbed DNS: unknown host {host!r}") from None
            addr = host
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (addr, port or 0))]

    monkeypatch.setattr(safe_fetch.socket, "getaddrinfo", fake_getaddrinfo)


def _serve(monkeypatch, body: bytes) -> list[httpx.Request]:
    """Point the client ``_scrape_page_text`` builds internally at ``body``."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, headers={"content-type": "text/html"}, content=body)

    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient

    def _patched(*args, **kwargs):
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _patched)
    return seen


def _html(filler: int) -> bytes:
    return b"<html><body><p>Ground the script.</p>" + b"x" * filler + b"</body></html>"


# --- the cap is enforced, at the value this module states --------------------


@pytest.mark.asyncio
async def test_body_over_the_stated_cap_is_refused(monkeypatch) -> None:
    """A body past ``_SCRAPE_MAX_RESPONSE_BYTES`` must not be scraped.

    This is the real configured ceiling, not a patched-down stand-in — deleting
    ``max_fetch_bytes=`` from the call site drops it back to the 64 MiB default,
    which this 8 MiB+1 body sails straight through, and the assertion below on
    ``text == ""`` fails on a real value.
    """
    _serve(monkeypatch, _html(_SCRAPE_MAX_RESPONSE_BYTES + 1))
    notes: list[str] = []

    text, status = await SiteCaptureStage._scrape_page_text(_URL, notes)

    assert text == ""
    # The over-cap body raises inside the fetch, so it is a hard scrape failure —
    # NOT "empty" (a 200 with no copy), which downstream is allowed to proceed on.
    assert status == "fetch_failed"
    assert notes, "an over-cap page must leave a note, not vanish"


@pytest.mark.asyncio
async def test_the_degrade_names_the_byte_cap_so_it_is_not_silent(monkeypatch) -> None:
    """The operator must be able to tell 'too large' from 'network failed'."""
    _serve(monkeypatch, _html(_SCRAPE_MAX_RESPONSE_BYTES + 1))
    notes: list[str] = []

    _, status = await SiteCaptureStage._scrape_page_text(_URL, notes)

    assert status == "fetch_failed"
    assert len(notes) == 1
    note = notes[0]
    assert "exceeds" in note, f"note does not explain the refusal: {note!r}"
    assert str(_SCRAPE_MAX_RESPONSE_BYTES) in note, (
        f"note does not name the cap that produced it: {note!r}"
    )


@pytest.mark.asyncio
async def test_the_raiser_is_a_valueerror_the_bare_except_would_swallow() -> None:
    """Why this needed a stated cap at all: the degrade path catches it.

    ``ResponseTooLargeError`` inside ``ValueError`` is exactly what makes the
    over-cap case fail *soft* rather than loud. That is the trap this call site
    fell into, and it is the reason the ceiling is stated next to the degrade.
    """
    assert issubclass(ResponseTooLargeError, ValueError)


def test_the_cap_is_stated_not_inherited() -> None:
    """A ceiling equal to the default would be indistinguishable from no cap."""
    assert _SCRAPE_MAX_RESPONSE_BYTES < DEFAULT_MAX_FETCH_BYTES
    assert site_capture._SCRAPE_MAX_RESPONSE_BYTES == 8 * 1024 * 1024


# --- false-positive controls: legitimate pages still fetch -------------------


@pytest.mark.asyncio
async def test_an_ordinary_page_still_scrapes(monkeypatch) -> None:
    """The control that makes the refusals above mean something.

    The body is a realistic paragraph rather than the one-line stub this used to
    use: that stub extracted to 21 characters, which is genuinely below the
    grounding floor and now classifies as ``"thin"``. Keeping it would have made
    the control assert that a page too short to ground a script is fine to ground
    a script on — see ``test_a_js_shell_page_reports_thin`` for that case.
    """
    body = b"<html><body><h1>Acme</h1><p>We sell anvils. Forged overnight and "
    body += b"delivered before breakfast, every Acme anvil is drop-tested to 400kg "
    body += b"and shipped from a workshop that has run without a missed order since "
    body += b"1974. Blacksmiths, prop houses and cartoon coyotes rely on us.</p>"
    body += b"<script>ignored()</script></body></html>"
    _serve(monkeypatch, body)
    notes: list[str] = []

    text, status = await SiteCaptureStage._scrape_page_text(_URL, notes)

    assert "Acme" in text
    assert "We sell anvils." in text
    assert "ignored()" not in text  # script blocks are stripped
    assert status == "ok"
    assert notes == []


@pytest.mark.asyncio
async def test_a_js_shell_page_reports_thin(monkeypatch) -> None:
    """A JS-only SPA is the case the ``"empty"`` status alone does NOT catch.

    Stripping tags surfaces the ``<title>`` and the ``<noscript>`` line, so the
    extraction is short but non-empty — measured against real shells on
    2026-08-12: web.telegram.org 12 chars, excalidraw.com 79, tldraw.com 136.
    All were classified ``"ok"`` and handed to the model as the site's product
    copy. It must report ``"thin"``, still RETURN the fragment (it is real), and
    leave a note so the run degrades.
    """
    body = b"<html><head><title>Telegram Web</title></head><body>"
    body += b"<noscript>Please enable JavaScript.</noscript>"
    body += b"<script>bootTheApp()</script></body></html>"
    _serve(monkeypatch, body)
    notes: list[str] = []

    text, status = await SiteCaptureStage._scrape_page_text(_URL, notes)

    assert status == "thin"
    assert "Telegram Web" in text, "a real fragment must not be discarded"
    assert len(notes) == 1
    assert "grounding floor" in notes[0], notes[0]


@pytest.mark.asyncio
async def test_a_body_just_under_the_cap_is_fetched_in_full(monkeypatch) -> None:
    """The boundary is a ceiling, not a much lower effective limit."""
    _serve(monkeypatch, _html(_SCRAPE_MAX_RESPONSE_BYTES - 1024))
    notes: list[str] = []

    text, status = await SiteCaptureStage._scrape_page_text(_URL, notes)

    assert "Ground the script." in text
    assert status == "ok"
    assert notes == []


# --- the silent path: 200 OK, nothing extractable ----------------------------


@pytest.mark.asyncio
async def test_script_only_200_reports_empty_and_leaves_a_note(monkeypatch) -> None:
    """A 200 whose body has no readable text is a DEGRADE, not a success.

    This is the defect: ``_extract_text`` strips ``<script>``/``<style>`` blocks,
    so a JS-only shell returns "" — and that path used to append NO note at all.
    The run therefore finished green and *undegraded* while the downstream promo
    script had zero site grounding and the LLM invented the product. The status
    must be ``"empty"`` (distinguishable from ``"fetch_failed"``: the fetch DID
    succeed) and the note must say so.
    """
    body = b"<html><head><style>body{color:red}</style></head>"
    body += b"<body><script>render()</script></body></html>"
    _serve(monkeypatch, body)
    notes: list[str] = []

    text, status = await SiteCaptureStage._scrape_page_text(_URL, notes)

    assert (text, status) == ("", "empty")
    assert len(notes) == 1
    assert "no readable text" in notes[0], f"the empty scrape stayed silent: {notes!r}"
    assert _URL in notes[0], f"note does not name the page it failed to read: {notes[0]!r}"
