"""P7 / R3 boot guard: CELL_AUTH_MODE=enforce refuses to boot without a cell signing key.

Under enforce an in-cell hop drops the shared DOWNSTREAM_LICENSE_TOKEN and relies
on the per-request cell JWT. With no CELL_JWT_SIGNING_KEY nothing is minted, so
every in-cell call would leave with NO credential. The orchestrator must fail at
startup — loudly, naming the missing setting — not one request at a time.

The lifespan tests drive the REAL ``src.asgi.lifespan``: the guard sits before the
engine is opened, so ``get_engine`` is replaced with a sentinel that proves how far
startup got (sentinel raised == the guard let boot continue).
"""

from __future__ import annotations

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import FastAPI
from w2v_common.auth.cell_jwt import reset_cell_state

from src.clients.downstream_credential import CellBootConfigError, assert_cell_boot_config

_CELL_ENV = (
    "CELL_AUTH_MODE",
    "CELL_JWT_SIGNING_KEY",
    "CELL_JWT_KID",
    "CELL_SERVICE_NAME",
    "CELL_DEPLOYMENT_ID",
    "ORCHESTRATOR_SKIP_LIFESPAN",
)


class _EngineReachedError(Exception):
    """Startup got past the guard and tried to open the database."""


@pytest.fixture(autouse=True)
def _clean_cell_env(monkeypatch: pytest.MonkeyPatch):
    for name in _CELL_ENV:
        monkeypatch.delenv(name, raising=False)
    reset_cell_state()
    yield
    reset_cell_state()


def _pem() -> str:
    return (
        Ed25519PrivateKey.generate()
        .private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        .decode()
    )


def _configure_signer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CELL_JWT_SIGNING_KEY", _pem())
    monkeypatch.setenv("CELL_JWT_KID", "orch-test-1")
    monkeypatch.setenv("CELL_SERVICE_NAME", "w2v-orchestrator")
    monkeypatch.setenv("CELL_DEPLOYMENT_ID", "test-deployment")


# --------------------------------------------------------------------------- #
# The guard function
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", [None])
def test_guard_is_noop_outside_enforce_without_a_key(monkeypatch, mode):
    if mode is not None:
        monkeypatch.setenv("CELL_AUTH_MODE", mode)
    assert_cell_boot_config()  # must not raise


def test_guard_refuses_enforce_without_signing_key(monkeypatch, caplog):
    monkeypatch.setenv("CELL_AUTH_MODE", "enforce")
    with (
        caplog.at_level("CRITICAL", logger="src.clients.downstream_credential"),
        pytest.raises(CellBootConfigError, match="CELL_JWT_SIGNING_KEY"),
    ):
        assert_cell_boot_config()
    assert any(
        r.levelname == "CRITICAL" and "CELL_JWT_SIGNING_KEY" in r.getMessage()
        for r in caplog.records
    ), "the refusal must be logged loudly, not only raised"


def test_guard_refuses_enforce_with_half_configured_signer(monkeypatch):
    monkeypatch.setenv("CELL_AUTH_MODE", "enforce")
    monkeypatch.setenv("CELL_JWT_SIGNING_KEY", _pem())
    # CELL_JWT_KID / CELL_SERVICE_NAME / CELL_DEPLOYMENT_ID missing.
    with pytest.raises(CellBootConfigError, match="CELL_JWT_KID"):
        assert_cell_boot_config()


def test_guard_refuses_enforce_with_garbage_key(monkeypatch):
    _configure_signer(monkeypatch)
    monkeypatch.setenv("CELL_AUTH_MODE", "enforce")
    monkeypatch.setenv("CELL_JWT_SIGNING_KEY", "not-a-pem")
    with pytest.raises(CellBootConfigError, match="misconfigured"):
        assert_cell_boot_config()


def test_guard_accepts_enforce_with_full_signer(monkeypatch):
    _configure_signer(monkeypatch)
    monkeypatch.setenv("CELL_AUTH_MODE", "enforce")
    assert_cell_boot_config()  # must not raise


# --------------------------------------------------------------------------- #
# Wired into the real lifespan, before the DB is opened
# --------------------------------------------------------------------------- #


async def _run_lifespan(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.asgi as asgi

    async def _sentinel_engine():
        raise _EngineReachedError

    monkeypatch.setattr(asgi, "get_engine", _sentinel_engine)
    async with asgi.lifespan(FastAPI()):
        pytest.fail("lifespan yielded with get_engine replaced by a raising sentinel")


async def test_lifespan_refuses_to_boot_under_enforce_without_key(monkeypatch):
    monkeypatch.setenv("CELL_AUTH_MODE", "enforce")
    with pytest.raises(CellBootConfigError, match="CELL_JWT_SIGNING_KEY"):
        await _run_lifespan(monkeypatch)


@pytest.mark.parametrize("mode", ["off"])
async def test_lifespan_boots_past_guard_outside_enforce(monkeypatch, mode):
    monkeypatch.setenv("CELL_AUTH_MODE", mode)
    with pytest.raises(_EngineReachedError):
        await _run_lifespan(monkeypatch)


async def test_lifespan_boots_past_guard_under_enforce_with_key(monkeypatch):
    _configure_signer(monkeypatch)
    monkeypatch.setenv("CELL_AUTH_MODE", "enforce")
    with pytest.raises(_EngineReachedError):
        await _run_lifespan(monkeypatch)
