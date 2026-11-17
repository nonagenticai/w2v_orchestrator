"""KeycloakAdminClient — the backfill reads + the early-access identity writes."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import httpx
import pytest

from src.clients.keycloak_admin import (
    KeycloakAdminClient,
    KeycloakAdminError,
    KeycloakAdminForbiddenError,
)

pytestmark = pytest.mark.asyncio

_BASE = "/admin/realms/portal"


class _Auth:
    async def token(self) -> str:
        return "tok"


def _kc(handler) -> KeycloakAdminClient:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return KeycloakAdminClient(
        authenticator=_Auth(), keycloak_url="http://kc", realm="portal", client=client
    )


_WRITE_SURFACE = (
    "create_user",
    "find_user_by_email",
    "update_user_attributes",
    "execute_actions_email",
    "delete_user",
    "set_user_state",
)
#: The only modules that may reach the Keycloak user-write surface: the
#: early-access approve's own identity (George 2026-10-20) and the early-access
#: REMOVAL, which retires exactly the login that approve created (deleted, or
#: disabled + marked). The ordinary tenant invite stays on provisioning's
#: identity invite API.
_WRITE_CALLERS = {
    "src/services/signup/identity.py",
    "src/services/signup/core.py",
    "src/services/signup/removal.py",
}


async def test_no_role_or_password_power_exists() -> None:
    """w2v mints a credential-less user and mails its owner; it never sets a
    password or grants a role. The ONE disable is ``set_user_state`` — the
    removal's disable-and-mark of the login it created (on the write surface)."""
    assert "set_user_state" in _WRITE_SURFACE
    for name in (
        "disable_user",
        "set_password",
        "reset_password",
        "send_update_password_email",
        "assign_realm_role",
        "remove_realm_role",
    ):
        assert not hasattr(KeycloakAdminClient, name), name


#: Directories that hold no product code (tests call fakes of this surface).
_NOT_SCANNED = {"tests", "node_modules", "__pycache__"}


def _python_files(root: Path) -> list[Path]:
    """Every Python file of the repo outside tests/ and hidden/virtualenv dirs:
    src/, scripts/, deploy/, … and top-level modules."""
    files = sorted(root.glob("*.py"))
    for top in sorted(root.iterdir()):
        if top.is_dir() and not top.name.startswith(".") and top.name not in _NOT_SCANNED:
            files.extend(sorted(top.rglob("*.py")))
    return files


async def test_the_write_surface_is_reached_only_from_the_signup_identity() -> None:
    root = Path(__file__).resolve().parents[2]
    files = _python_files(root)
    rels = {p.relative_to(root).as_posix() for p in files}
    # Non-vacuity of the scan itself: scripts/ (the backfill, a real admin-client
    # caller) is covered, not only src/. The published source tree ships no
    # deploy/ directory, so there is nothing else to cover.
    assert "scripts/backfill_tenancy.py" in rels
    assert not any(r.startswith("tests/") for r in rels)
    callers: dict[str, set[str]] = {}
    for path in files:
        rel = path.relative_to(root).as_posix()
        if rel == "src/clients/keycloak_admin.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _WRITE_SURFACE
            ):
                callers.setdefault(rel, set()).add(node.func.attr)
    assert set(callers) <= _WRITE_CALLERS, callers
    # Non-vacuity: the guard sees the real callers.
    assert "create_user" in callers.get("src/services/signup/identity.py", set())
    assert "execute_actions_email" in callers.get("src/services/signup/core.py", set())
    assert {"delete_user", "set_user_state"} <= callers.get("src/services/signup/removal.py", set())


async def test_get_user_parses_and_404_is_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.headers["authorization"] == "Bearer tok"
        if request.url.path == f"{_BASE}/users/u1":
            return httpx.Response(
                200,
                json={"id": "u1", "email": "a@b.com", "emailVerified": True, "enabled": True},
            )
        return httpx.Response(404)

    kc = _kc(handler)
    user = await kc.get_user("u1")
    assert user is not None
    assert user.email_verified is True
    assert user.email == "a@b.com"
    assert await kc.get_user("missing") is None


async def test_get_user_effective_realm_roles_sorted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"{_BASE}/users/u1/role-mappings/realm/composite"
        return httpx.Response(200, json=[{"name": "tenant_admin"}, {"name": "member"}, {}])

    assert await _kc(handler).get_user_effective_realm_roles("u1") == ["member", "tenant_admin"]


async def test_403_is_the_forbidden_error_and_5xx_is_not() -> None:
    def forbidden(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": "HTTP 403 Forbidden"})

    with pytest.raises(KeycloakAdminForbiddenError) as info:
        await _kc(forbidden).get_user("u1")
    assert info.value.status_code == 403

    def down(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    with pytest.raises(KeycloakAdminError) as info2:
        await _kc(down).get_user_effective_realm_roles("u1")
    assert not isinstance(info2.value, KeycloakAdminForbiddenError)
    assert info2.value.status_code == 503


def _recording(responses):  # type: ignore[no-untyped-def]
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return responses(request)

    return seen, handler


async def test_create_user_posts_the_representation_and_reads_the_location() -> None:
    rep = {"username": "a@b.test", "email": "a@b.test", "enabled": True}
    seen, handler = _recording(
        lambda r: httpx.Response(201, headers={"location": f"http://kc{_BASE}/users/new-id"})
    )
    assert await _kc(handler).create_user(rep) == "new-id"
    assert seen[0].method == "POST" and seen[0].url.path == f"{_BASE}/users"
    assert json.loads(seen[0].content) == rep


async def test_create_user_409_is_none_and_201_without_location_errors() -> None:
    assert await _kc(lambda r: httpx.Response(409)).create_user({}) is None
    with pytest.raises(KeycloakAdminError):
        await _kc(lambda r: httpx.Response(201)).create_user({})


async def test_find_user_by_email_is_exact_and_refuses_two() -> None:
    seen, handler = _recording(lambda r: httpx.Response(200, json=[{"id": "u1"}]))
    user = await _kc(handler).find_user_by_email("  Ann@B.test ")
    assert user is not None and user.id == "u1"
    params = seen[0].url.params
    assert (params["email"], params["exact"], params["max"]) == ("ann@b.test", "true", "2")
    assert await _kc(lambda r: httpx.Response(200, json=[])).find_user_by_email("x@y") is None
    with pytest.raises(KeycloakAdminError):
        await _kc(
            lambda r: httpx.Response(200, json=[{"id": "a"}, {"id": "b"}])
        ).find_user_by_email("x@y")


async def test_update_user_attributes_is_a_whole_representation_read_modify_write() -> None:
    stored = {
        "id": "u1",
        "email": "a@b.test",
        "firstName": "Ann",
        "requiredActions": ["VERIFY_EMAIL"],
        "attributes": {"tenant_id": ["t1"], "locale": ["en"], "gone": ["x"]},
    }

    def answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=stored) if request.method == "GET" else httpx.Response(204)

    seen, handler = _recording(answer)
    await _kc(handler).update_user_attributes(
        "u1", {"locale": "nl", "signup_app": "w2v", "gone": None}
    )
    assert [r.method for r in seen] == ["GET", "PUT"]
    body = json.loads(seen[1].content)
    assert body["firstName"] == "Ann" and body["requiredActions"] == ["VERIFY_EMAIL"]
    assert body["attributes"] == {"tenant_id": ["t1"], "locale": ["nl"], "signup_app": ["w2v"]}


async def test_execute_actions_email_sends_the_actions_and_params() -> None:
    seen, handler = _recording(lambda r: httpx.Response(204))
    await _kc(handler).execute_actions_email(
        "u 1",
        ["UPDATE_PASSWORD", "VERIFY_EMAIL"],
        client_id="w2v-frontend",
        redirect_uri="https://w2v.test/invite/accept?token=abc",
        lifespan_seconds=1209600,
    )
    [req] = seen
    assert req.method == "PUT"
    assert req.url.raw_path.decode().startswith(f"{_BASE}/users/u%201/execute-actions-email?")
    assert dict(req.url.params) == {
        "client_id": "w2v-frontend",
        "redirect_uri": "https://w2v.test/invite/accept?token=abc",
        "lifespan": "1209600",
    }
    assert json.loads(req.content) == ["UPDATE_PASSWORD", "VERIFY_EMAIL"]


async def test_counts_read_credentials_and_idp_links() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        if request.url.path == f"{_BASE}/users/u1/credentials":
            return httpx.Response(200, json=[{"type": "password"}])
        if request.url.path == f"{_BASE}/users/u1/federated-identity":
            return httpx.Response(200, json=[])
        return httpx.Response(404)

    kc = _kc(handler)
    assert await kc.count_credentials("u1") == 1
    assert await kc.count_federated_identities("u1") == 0
    with pytest.raises(KeycloakAdminError):
        await kc.count_credentials("missing")
    with pytest.raises(KeycloakAdminError):
        await _kc(lambda r: httpx.Response(200, json={})).count_federated_identities("u1")


async def test_set_user_state_disables_and_marks_in_one_rmw_and_reads_back() -> None:
    stored: dict[str, object] = {
        "id": "u1",
        "email": "a@b.test",
        "enabled": True,
        "requiredActions": ["VERIFY_EMAIL"],
        "attributes": {"tenant_id": ["t1"], "signup_app": ["w2v"]},
        "createdTimestamp": 1760000000000,
    }

    def answer(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT":
            stored.update(json.loads(request.content))
            return httpx.Response(204)
        return httpx.Response(200, json=stored)

    seen, handler = _recording(answer)
    user = await _kc(handler).set_user_state(
        "u1", enabled=False, attributes={"signup_app": "w2v_removed:x"}
    )
    assert [r.method for r in seen] == ["GET", "PUT", "GET"]  # read, write, READ BACK
    body = json.loads(seen[1].content)
    assert body["enabled"] is False and body["requiredActions"] == ["VERIFY_EMAIL"]
    assert body["attributes"] == {"tenant_id": ["t1"], "signup_app": ["w2v_removed:x"]}
    assert user is not None and user.enabled is False
    assert user.created_timestamp == 1760000000000
    assert (
        await _kc(lambda r: httpx.Response(404)).set_user_state(
            "gone", enabled=False, attributes={}
        )
        is None
    )
    with pytest.raises(KeycloakAdminForbiddenError):
        await _kc(
            lambda r: httpx.Response(200, json=stored) if r.method == "GET" else httpx.Response(403)
        ).set_user_state("u1", enabled=False, attributes={})


async def test_groups_and_direct_role_mappings() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"{_BASE}/users/u1/groups":
            return httpx.Response(200, json=[{"name": "g"}])
        if request.url.path == f"{_BASE}/users/u1/role-mappings":
            return httpx.Response(
                200,
                json={
                    "realmMappings": [{"name": "default-roles-portal"}, {"name": "super_admin"}],
                    "clientMappings": {
                        "fr": {"client": "fr-backend", "mappings": [{"name": "admin"}]}
                    },
                },
            )
        return httpx.Response(404)

    kc = _kc(handler)
    assert await kc.count_groups("u1") == 1
    roles = await kc.get_user_role_mappings("u1")
    assert roles.realm == ["default-roles-portal", "super_admin"]
    assert roles.clients == {"fr-backend": ["admin"]}
    with pytest.raises(KeycloakAdminError):
        await kc.get_user_role_mappings("missing")


async def test_delete_user_404_is_false() -> None:
    assert await _kc(lambda r: httpx.Response(204)).delete_user("u1") is True
    assert await _kc(lambda r: httpx.Response(404)).delete_user("u1") is False


@pytest.mark.parametrize(
    ("status", "forbidden", "uncertain"),
    [(400, False, False), (504, False, True)],
)
async def test_status_error_mapping(status: int, forbidden: bool, uncertain: bool) -> None:
    with pytest.raises(KeycloakAdminError) as info:
        await _kc(lambda r: httpx.Response(status)).execute_actions_email(
            "u1", [], client_id="c", redirect_uri="r", lifespan_seconds=1
        )
    assert info.value.status_code == status
    assert isinstance(info.value, KeycloakAdminForbiddenError) is forbidden
    assert info.value.uncertain is uncertain


@pytest.mark.parametrize(
    ("exc", "uncertain"),
    [
        (httpx.ConnectError("refused"), False),
        (httpx.ConnectTimeout("slow"), False),
        (httpx.ReadTimeout("slow"), True),
        (httpx.RemoteProtocolError("reset"), True),
    ],
)
async def test_transport_errors_are_wrapped_with_their_certainty(
    exc: Exception, uncertain: bool
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    with pytest.raises(KeycloakAdminError) as info:
        await _kc(handler).execute_actions_email(
            "u1", [], client_id="c", redirect_uri="r", lifespan_seconds=1
        )
    assert info.value.status_code is None and info.value.uncertain is uncertain


async def test_unconfigured_url_refuses_without_a_request() -> None:
    kc = KeycloakAdminClient(authenticator=_Auth(), keycloak_url="", realm="portal")
    with pytest.raises(KeycloakAdminError, match="not configured"):
        await kc.create_user({})
