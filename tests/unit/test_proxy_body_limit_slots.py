"""Large-body slots on POST /api/critique (P9-7 re-review blocker).

Bodies over the 256 KiB default on the override route each take one of N
process-wide slots for as long as the app holds them (body + upstream call);
none free → 429 + Retry-After before more is read; slots are always released.
Driven against a stub ASGI app that reads the whole body then parks on a gate,
so "in flight" is controlled exactly.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from starlette.types import Receive, Scope, Send

from src.middleware.proxy_body_limit import (
    PROXY_BODY_LIMIT_BYTES,
    PROXY_BODY_LIMIT_OVERRIDES,
    ProxyBodyLimitMiddleware,
)

LARGE = b"x" * (PROXY_BODY_LIMIT_BYTES + 1024)
SMALL = b"x" * 1024


class _Stub:
    def __init__(self) -> None:
        self.gate = asyncio.Event()
        self.fail = False
        self.bodies: list[int] = []

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        body = b""
        while True:
            msg = await receive()
            if msg["type"] == "http.disconnect":
                raise RuntimeError("client disconnected mid-body")
            body += msg.get("body", b"")
            if not msg.get("more_body", False):
                break
        self.bodies.append(len(body))
        await self.gate.wait()
        if self.fail:
            raise RuntimeError("upstream blew up")
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


def _chunked(payload: bytes):
    async def gen():
        for i in range(0, len(payload), 64 * 1024):
            yield payload[i : i + 64 * 1024]

    return gen()


async def _until(pred, timeout: float = 5.0) -> None:
    # Polls state owned by in-flight ASGI tasks; there is no event to await.
    async with asyncio.timeout(timeout):
        for _ in range(int(timeout / 0.005)):
            if pred():
                return
            await asyncio.sleep(0.005)
    raise AssertionError("condition never became true")


@pytest.mark.parametrize("chunked", [False], ids=["declared"])
@pytest.mark.asyncio
async def test_n_plus_one_large_bodies_get_exactly_one_429(chunked: bool) -> None:
    stub = _Stub()
    mw = ProxyBodyLimitMiddleware(stub, large_body_slots=2)
    async with _client(mw) as c:

        def post():
            return c.post("/api/critique", content=_chunked(LARGE) if chunked else LARGE)

        tasks = [asyncio.create_task(post()) for _ in range(3)]
        await _until(lambda: mw.large_body_slots.in_use == 2 and any(t.done() for t in tasks))
        busy = [t.result() for t in tasks if t.done()]
        assert len(busy) == 1 and busy[0].status_code == 429, busy
        assert busy[0].headers["retry-after"] == "30"
        assert busy[0].json()["error"]["code"] == "LARGE_BODY_SLOTS_BUSY"

        # Small critiques never take a slot — they pass while the slots are full.
        small = asyncio.create_task(c.post("/api/critique", content=SMALL))
        await _until(lambda: SMALL.__len__() in stub.bodies)
        assert mw.large_body_slots.in_use == 2

        stub.gate.set()
        results = await asyncio.gather(*tasks, small)
    assert sorted(r.status_code for r in results) == [200, 200, 200, 429]
    assert mw.large_body_slots.in_use == 0, "slots released after completion"
    # A fresh large body gets a slot again.
    async with _client(mw) as c:
        assert (await c.post("/api/critique", content=LARGE)).status_code == 200
    assert mw.large_body_slots.in_use == 0


@pytest.mark.asyncio
async def test_slot_released_on_app_error_and_on_413() -> None:
    stub = _Stub()
    stub.gate.set()
    stub.fail = True
    mw = ProxyBodyLimitMiddleware(stub, large_body_slots=1)
    async with _client(mw) as c:
        with pytest.raises(RuntimeError, match="upstream blew up"):
            await c.post("/api/critique", content=LARGE)
        assert mw.large_body_slots.in_use == 0, "released on exception"

        over = b"x" * (PROXY_BODY_LIMIT_OVERRIDES["/api/critique"] + 1)
        r = await c.post("/api/critique", content=_chunked(over))
        assert r.status_code == 413, r.text
        assert mw.large_body_slots.in_use == 0, "released on 413"


@pytest.mark.asyncio
async def test_default_routes_never_use_slots() -> None:
    """Brand routes stay at 256 KiB (413), and never touch the slot pool."""
    stub = _Stub()
    stub.gate.set()
    mw = ProxyBodyLimitMiddleware(stub, large_body_slots=1)
    async with _client(mw) as c:
        r = await c.post("/api/brand/from-url", content=_chunked(LARGE))
    assert r.status_code == 413
    assert mw.large_body_slots.in_use == 0
