"""Migration 0106 declares ``signs_into`` on existing SSO logins without touching secrets.

Project 01a174a6 (sign-in page on the Keycloak host, app elsewhere) must keep its
saved password and sign in on the next rerun with no action from its owner; only a
USER change to ``signs_into`` afterwards clears the password.
"""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import io
import json
import os
import zipfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from alembic import command
from alembic.config import Config
from cryptography.fernet import Fernet
from pydantic import SecretStr
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.config import get_settings
from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.run_repository import RunRepository
from src.services.auth_secrets import (
    redact_auth_secrets,
    restore_redacted,
    seal_auth_secrets,
)
from src.services.runs.run_context import RunContext
from src.services.stages.generic.site_capture import SiteCaptureStage
from tests.conftest import TEST_DATABASE_URL, _fail_or_skip_without_database
from tests.database.test_migration_0090_api_keys_rls import _drop_probe, _server_engine
from tests.support import run_isolation

_PATH = (
    Path(__file__).resolve().parents[2]
    / "src/database/migrations/versions/0106_backfill_login_signs_into.py"
)
_REV = "0106_backfill_login_signs_into"
_PRE = "0105_run_usage_mirror_unique"
_PROBE_DB = f"login_signs_into_probe_{os.getpid()}"
#: The MIGRATION role: owns the probe DB (so every table alembic creates), and is
#: neither superuser nor BYPASSRLS — the shape the migration's docstring relies on.
_MIGRATOR = f"w2v_mig_probe_{os.getpid()}"
_MIGRATOR_PW = "probe-not-secret"  # pragma: allowlist secret
_PW = "sso-pw-not-real"  # pragma: allowlist secret
_IDP = "https://auth.example.com/realms/app/protocol/openid-connect/auth?client_id=app"
_APP = "https://app.example.com"


def _migration():
    spec = importlib.util.spec_from_file_location("m_0106", _PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        get_settings().auth_secrets,
        "auth_config_secret_keys",
        SecretStr(Fernet.generate_key().decode()),
    )


def _legacy_brief() -> dict:
    """The prod shape BEFORE #1163: SSO login, no signs_into, sealed password."""
    return seal_auth_secrets(
        {
            "source_url": f"{_APP}/",
            "shotlist": [{"kind": "url", "url": f"{_APP}/", "narration": "home"}],
            "auth_info": {
                "type": "basic",
                "username": "u@example.com",
                "password": _PW,
                "login_url": _IDP,
            },
            "sources": [{"ref": f"{_APP}/x", "auth_info": {"username": "s", "login_url": _IDP}}],
        }
    )


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Config, str, str]]:
    """A throwaway DB OWNED by a non-superuser, non-BYPASSRLS migration role.

    Yields ``(alembic config, migrator url, superuser url)``. Alembic runs as the
    migrator, exactly as in prod; the superuser url only plants the legacy rows."""
    import tests.database.test_migration_0090_api_keys_rls as m0090

    monkeypatch.setattr(m0090, "_PROBE_DB", _PROBE_DB)
    server = _server_engine()
    try:
        with server.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError as exc:
        _fail_or_skip_without_database(exc)
    _drop_probe(server)
    with server.connect() as conn:
        conn.execute(text(f"DROP ROLE IF EXISTS {_MIGRATOR}"))
        conn.execute(
            text(f"CREATE ROLE {_MIGRATOR} LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD '{_MIGRATOR_PW}'")
        )
        conn.execute(text(f"CREATE DATABASE {_PROBE_DB} OWNER {_MIGRATOR}"))
    admin_url = make_url(TEST_DATABASE_URL).set(database=_PROBE_DB)
    migrator_url = admin_url.set(username=_MIGRATOR, password=_MIGRATOR_PW)
    cfg = Config()
    cfg.set_main_option("script_location", "src/database/migrations")
    monkeypatch.setenv("DATABASE_URL", migrator_url.render_as_string(hide_password=False))
    if os.environ.get("PGOPTIONS") is not None:
        monkeypatch.setenv("PGOPTIONS", run_isolation.pgoptions("public"))
    try:
        yield (
            cfg,
            migrator_url.render_as_string(hide_password=False),
            admin_url.render_as_string(hide_password=False),
        )
    finally:
        _drop_probe(server)
        with server.connect() as conn:
            conn.execute(text(f"DROP ROLE IF EXISTS {_MIGRATOR}"))
        server.dispose()


async def _plant(admin_url: str, brief: dict, tenant_id: UUID) -> tuple:
    engine = create_async_engine(admin_url)
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as s:
            proj = await ProjectRepository(s).create_with_deliverables(
                tenant_id=tenant_id,
                owner_subject_id=uuid4(),
                type_="promo",
                title="sso",
                brief={},
                brand={},
                voice={},
                theme={},
                visibility="tenant",
                deliverables=[{"name": "d", "format": "16x9"}],
            )
            run = await RunRepository(s).create(
                deliverable_id=proj.deliverables[0].id,
                blueprint_id="promo_v1",
                config_snapshot={},
                pause_after_stages=[],
            )
            # RAW SQL: these rows predate the migration; no ORM listener touches them.
            await s.execute(
                text("UPDATE projects SET brief = CAST(:b AS jsonb) WHERE id = :id"),
                {"b": json.dumps(brief), "id": proj.id},
            )
            await s.execute(
                text("UPDATE runs SET config_snapshot = CAST(:b AS jsonb) WHERE id = :id"),
                {"b": json.dumps({"brief": brief}), "id": run.id},
            )
            await s.commit()
            return proj.id, run.id
    finally:
        await engine.dispose()


def _doc(engine: Engine, table: str, column: str, row_id) -> dict:
    with engine.connect() as conn:
        return conn.execute(
            text(f"SELECT {column} FROM {table} WHERE id = :id"),  # noqa: S608 -- test consts
            {"id": row_id},
        ).scalar_one()


def _rls_shape(engine: Engine) -> dict[str, tuple[bool, bool, str]]:
    """``{table: (RLS enabled, RLS forced, owner)}`` for the two backfilled tables."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT relname, relrowsecurity, relforcerowsecurity, relowner::regrole::text "
                "FROM pg_class WHERE relname IN ('projects', 'runs') "
                "AND relnamespace = 'public'::regnamespace"
            )
        ).all()
    return {r[0]: (bool(r[1]), bool(r[2]), str(r[3])) for r in rows}


@pytest.mark.asyncio
@pytest.mark.usefixtures("key")
async def test_upgrade_as_the_migration_role_backfills_every_tenant_and_keeps_the_password(
    probe,
) -> None:
    cfg, migrator_url, admin_url = probe
    # alembic's env.py drives its own event loop: run it off this one.
    await asyncio.to_thread(command.upgrade, cfg, _PRE)

    legacy = _legacy_brief()
    sealed_before = copy.deepcopy(legacy["auth_info"]["password"])
    untouched = {
        # a login LIST is keyed by its own origins: never backfilled
        "target_url": f"{_APP}/",
        "auth_config": [{"login_url": _IDP, "username": "x"}],
    }
    ambiguous = {
        "source_url": f"{_APP}/",
        "target_url": "https://other.example/",
        "auth_info": {"login_url": _IDP, "username": "y"},
    }
    # Two DIFFERENT tenants: the claim is that the migration sees every tenant's rows.
    proj_id, run_id = await _plant(admin_url, legacy, uuid4())
    other = _legacy_brief()
    other_id, other_run = await _plant(admin_url, other, uuid4())
    list_id, _ = await _plant(admin_url, untouched, uuid4())
    amb_id, _ = await _plant(admin_url, ambiguous, uuid4())

    engine = create_engine(migrator_url)
    try:
        # The docstring's RLS claim, asserted as the role that runs the migration: it
        # is NOT superuser and NOT BYPASSRLS, the tables have RLS ENABLEd but not
        # FORCEd, and it OWNS them — so ownership alone makes every row visible.
        with engine.connect() as conn:
            me = conn.execute(
                text(
                    "SELECT current_user, rolsuper, rolbypassrls FROM pg_roles "
                    "WHERE rolname = current_user"
                )
            ).one()
        assert (me[0], bool(me[1]), bool(me[2])) == (_MIGRATOR, False, False)
        assert _rls_shape(engine) == {
            "projects": (True, False, _MIGRATOR),
            "runs": (True, False, _MIGRATOR),
        }

        await asyncio.to_thread(command.upgrade, cfg, _REV)

        for pid, rid, sealed in (
            (proj_id, run_id, sealed_before),
            (other_id, other_run, other["auth_info"]["password"]),
        ):
            brief = _doc(engine, "projects", "brief", pid)
            assert brief["auth_info"]["signs_into"] == _APP, pid
            assert brief["sources"][0]["auth_info"]["signs_into"] == _APP
            assert brief["auth_info"]["password"] == sealed  # byte-identical, still sealed
            assert _PW not in json.dumps(brief)
            snap = _doc(engine, "runs", "config_snapshot", rid)
            assert snap["brief"]["auth_info"]["signs_into"] == _APP
        assert _doc(engine, "projects", "brief", list_id) == untouched
        assert _doc(engine, "projects", "brief", amb_id) == ambiguous

        # Re-runnable: a second pass over the upgraded rows declares nothing.
        with engine.begin() as conn:
            assert _migration().backfill_rows(conn) == {"projects": 0, "runs": 0}
        brief = _doc(engine, "projects", "brief", proj_id)
        assert brief["auth_info"]["password"] == sealed_before  # byte-identical, sealed
    finally:
        engine.dispose()

    # The next rerun signs in with the SAVED password — no action from the owner.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("shot-000.png", b"png")
    client = AsyncMock()
    client.capture_screens = AsyncMock(return_value="job")
    client.wait_for_content_recording = AsyncMock(return_value={"status": "completed"})
    client.get_recording_video = AsyncMock(return_value=(buf.getvalue(), "application/zip", None))
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={"brief": brief})
    with patch(
        "src.services.stages.generic.site_capture.store_capture_image",
        new=AsyncMock(return_value="https://minio/s.png"),
    ):
        await SiteCaptureStage(client=client).run({}, ctx=ctx)
    call = client.capture_screens.await_args
    assert call is not None
    sent = call.kwargs["auth_info"]
    assert sent["password"] == _PW
    assert sent["login_url"] == _IDP
    assert "signs_into" not in sent  # orchestrator-only, never forwarded

    # An unrelated Brief save keeps it; a USER change to signs_into clears + prompts.
    echoed = redact_auth_secrets(brief)
    echoed["instructions"] = "new copy"
    kept = seal_auth_secrets(restore_redacted(echoed, brief))
    assert kept["auth_info"]["password"] == sealed_before
    assert "needs_reentry" not in kept["auth_info"]
    edited = redact_auth_secrets(brief)
    edited["auth_info"]["signs_into"] = "https://elsewhere.example"
    cleared = seal_auth_secrets(restore_redacted(edited, brief))
    assert "password" not in cleared["auth_info"]
    assert cleared["auth_info"]["needs_reentry"] is True
