"""The compile-cancel edge must never open a socket from the test suite.

WHY THIS EXISTS (measured on prod, 2026-08-20). ``cancel_compiles_for_run``
constructs a REAL ``VideoCompileClient`` — service discovery + the service-account
license token — and POSTs ``/api/jobs/cancel``. Nothing in the test suite stubbed
it, and the orchestrator's CI test job runs on an IN-CLUSTER self-hosted runner
(worker2) where ``video-compiler.video-compiler`` resolves. So
every CI run fired ~13 authenticated POSTs at the PRODUCTION compiler, one per
test that touches a cancel path, each carrying a run id the test had just minted.

The compiler logged all of them at INFO with ``n_cancelled: 0`` — a uuid7 that is
zero seconds old cannot match a job — so 195 prod writes over 5h40m were invisible
to every error grep. On a developer laptop the DNS name does not resolve, which is
why the escape was silent locally too.

This test asserts the property (no non-loopback name is EVER resolved by the
cancel edge), not the mechanism, so it still holds if the stub in ``conftest``
is reshaped.
"""

from __future__ import annotations

import socket
from uuid import uuid7

import pytest

from src.services.compile_cancel import cancel_compiles_for_run


@pytest.mark.asyncio
async def test_cancel_propagation_resolves_no_remote_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved: list[str] = []
    real = socket.getaddrinfo

    def _spy(host, port, *args, **kwargs):  # type: ignore[no-untyped-def]
        name = host.decode() if isinstance(host, bytes) else str(host)
        if name not in ("127.0.0.1", "localhost", "::1"):
            resolved.append(name)
        return real(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", _spy)

    # The exact call every cancel path makes. It must never raise AND never
    # reach for a name outside this process.
    await cancel_compiles_for_run(uuid7())

    assert resolved == [], (
        "compile-cancel escaped the test process and tried to resolve "
        f"{resolved!r} — in CI that name resolves and the POST lands on the "
        "PRODUCTION video-compiler service"
    )
