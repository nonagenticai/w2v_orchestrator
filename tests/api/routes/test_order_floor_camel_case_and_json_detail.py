"""Two holes in the order floor shipped by #1024, found by the acceptance run.

**The camelCase bypass.** ``target_duration_seconds`` was honoured in snake_case only.
``_camel`` (``api/schemas/_shared.py``) is ``populate_by_name=True`` with NO
``alias_generator``, so a camelCase client's ``targetDurationSeconds`` was DROPPED by
pydantic's default ``extra="ignore"`` and validated to ``None``. Measured 2026-09-16: a
9s order sent under the camelCase key returned **201 with a null order** instead of the
typed 400 — the floor (``MINIMUM_ORDER_SECONDS = 10.0``) has nothing to refuse when the
value never arrives. Decision: ACCEPT BOTH SPELLINGS via a per-field
``validation_alias`` (see ``schemas/projects._TARGET_DURATION_ALIASES`` for why not
``alias_generator``).

**The unparseable refusal.** The typed 400 body arrived as a Python ``repr`` STRING
inside ``error.message`` — ``"{'reason': 'order_below_floor', 'ordered_seconds': 9.0,
…}"`` — because ``w2v_common.asgi.error_envelope._http_response`` does
``str(exc.detail)`` for any non-string detail and never fills ``details``. Every number
was there and none of it was parseable, so the frontend (PR #448, which surfaces
``ApiError.detail`` inline) could not branch on ``reason``. Fixed orchestrator-side in
``src/api/error_details.py`` — the library is a different repo.
"""

from __future__ import annotations

from uuid import UUID

import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient
from w2v_common.asgi.error_envelope import register_error_envelope_handlers

from src.api.error_details import register_object_detail_http_handler

pytestmark = pytest.mark.asyncio


# ── the camelCase bypass ────────────────────────────────────────────────────────


def _create_body(key: str, seconds: float) -> dict:
    return {
        "type": "website_review",
        "title": "review",
        "brief": {"source_url": "https://example.com"},
        "deliverables": [{"name": "d", "format": "16x9", key: seconds}],
    }


async def test_a_camelcase_nine_second_order_is_refused_not_nulled(client) -> None:
    r = await client.post("/api/projects", json=_create_body("targetDurationSeconds", 9))

    # NOT a 201-with-null-order. The value arrived, so the floor applied.
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert detail["reason"] == "order_below_floor"
    assert detail["ordered_seconds"] == 9.0
    assert detail["minimum_seconds"] == 10.0


async def test_a_camelcase_order_above_the_floor_is_persisted_not_nulled(client) -> None:
    """Accepting the key is only half of it — the value must reach the row.

    A refusal alone could be produced by rejecting the spelling outright; this pins
    that the honoured branch actually stores 42, not ``None``.
    """
    r = await client.post("/api/projects", json=_create_body("targetDurationSeconds", 42))

    assert r.status_code == 201, r.text
    detail = await client.get(f"/api/projects/{r.json()['id']}")
    assert detail.json()["deliverables"][0]["target_duration_seconds"] == 42


async def test_the_snake_case_spelling_is_unchanged(client) -> None:
    r = await client.post("/api/projects", json=_create_body("target_duration_seconds", 42))

    assert r.status_code == 201, r.text
    detail = await client.get(f"/api/projects/{r.json()['id']}")
    assert detail.json()["deliverables"][0]["target_duration_seconds"] == 42


async def test_the_response_body_still_uses_snake_case(client) -> None:
    """The alias is VALIDATION-only — no serialization alias, so responses are unchanged."""
    r = await client.post("/api/projects", json=_create_body("target_duration_seconds", 42))
    detail = await client.get(f"/api/projects/{UUID(r.json()['id'])}")
    deliverable = detail.json()["deliverables"][0]
    assert "target_duration_seconds" in deliverable
    assert "targetDurationSeconds" not in deliverable


# ── the refusal body must be JSON, through the PRODUCTION error envelope ────────
#
# The `client` fixture's hand-built app registers NO error-envelope handlers (see
# tests/conftest.py), which is exactly why this defect survived: the existing floor
# tests read `r.json()["detail"]` off a raw FastAPI response and never saw the
# stringification the real app applies. These build the production registration
# sequence — create_app's handlers, then the orchestrator's override — and assert on
# the PARSED object.


def _envelope_app(*, with_override: bool) -> FastAPI:
    from src.services.stages.website_review.order import OrderRefusal

    app = FastAPI()

    @app.get("/boom")
    async def boom() -> None:
        refusal = OrderRefusal(ordered_seconds=9.0, minimum_seconds=10.0)
        raise HTTPException(400, detail=refusal.model_dump())

    @app.get("/nameless")
    async def nameless() -> None:
        # A structured detail with no human line of its own — e.g. the shape
        # `deliverables.py` would produce for a body that is a bare list.
        raise HTTPException(409, detail=[{"reason": "conflict"}])

    @app.get("/plain")
    async def plain() -> None:
        raise HTTPException(404, detail="run not found")

    register_error_envelope_handlers(app)
    if with_override:
        register_object_detail_http_handler(app)
    return app


async def _get(app: FastAPI, path: str):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        return await c.get(path)


async def test_the_typed_400_reaches_the_wire_as_a_json_object(client) -> None:
    r = await _get(_envelope_app(with_override=True), "/boom")

    assert r.status_code == 400, r.text
    details = r.json()["error"]["details"]
    # A PARSED object, not a substring of a repr string.
    assert isinstance(details, dict), f"details is {type(details).__name__}: {details!r}"
    assert details["reason"] == "order_below_floor"
    assert details["ordered_seconds"] == 9.0
    assert details["minimum_seconds"] == 10.0
    # ...and `message` is the human line, never the repr.
    message = r.json()["error"]["message"]
    assert isinstance(message, str)
    assert "shortest possible review" in message
    assert "reason" not in message, f"message is still a stringified dict: {message!r}"


async def test_the_library_default_stringifies_it_which_is_the_defect(client) -> None:
    """The unfixed shape, pinned — so this test suite fails if the override is dropped."""
    r = await _get(_envelope_app(with_override=False), "/boom")

    body = r.json()["error"]
    assert "details" not in body
    assert body["message"].startswith("{'reason'")


async def test_a_plain_string_detail_is_unchanged(client) -> None:
    """Only a dict/list detail changes shape; the 99% case keeps the library behaviour."""
    r = await _get(_envelope_app(with_override=True), "/plain")

    body = r.json()["error"]
    assert r.status_code == 404
    assert body["message"] == "run not found"
    assert "details" not in body


async def test_a_structured_detail_without_a_message_gets_a_fixed_string(client) -> None:
    """``message`` must stay a real sentence, never a ``repr`` of the body.

    Covers the ``_STRUCTURED_FALLBACK`` branch: the list/dict still lands under
    ``details`` intact, and ``message`` points at it instead of serialising it.
    """
    r = await _get(_envelope_app(with_override=True), "/nameless")

    body = r.json()["error"]
    assert r.status_code == 409
    assert body["details"] == [{"reason": "conflict"}]
    assert body["message"] == "Request refused; see error.details"


# ── the PRODUCTION app must actually carry the handler ──────────────────────────


async def test_the_real_app_serialises_a_structured_detail_as_json() -> None:
    """BLOCKER 4: every other Defect-3 test builds a synthetic ``FastAPI()``.

    Deleting ``register_object_detail_http_handler(app)`` from ``src/asgi.py`` left all
    of them green — a fix that works only on a synthetic app is not shipped.

    This drives the handler ``src.asgi:app`` ACTUALLY HAS registered for
    ``HTTPException`` and reads the bytes it produces. Going over HTTP instead would
    only prove the auth middleware works (it 401s before routing); the registration is
    the thing under test. Remove the registration line in ``src/asgi.py`` and the
    library's stringifying handler takes its place, and this goes red.
    """
    import inspect
    import json

    from starlette.exceptions import HTTPException as StarletteHTTPException
    from starlette.requests import Request
    from starlette.responses import Response

    from src.asgi import app as real_app
    from src.services.stages.website_review.order import OrderRefusal

    handler = real_app.exception_handlers[StarletteHTTPException]
    request = Request(
        {"type": "http", "method": "GET", "path": "/", "headers": [], "query_string": b""}
    )
    refusal = OrderRefusal(ordered_seconds=9.0, minimum_seconds=10.0)

    result = handler(request, HTTPException(400, detail=refusal.model_dump()))
    # Starlette types a handler as sync-or-async; every handler in this stack is async.
    response = await result if inspect.isawaitable(result) else result
    assert isinstance(response, Response)

    assert response.status_code == 400
    body = json.loads(bytes(response.body))["error"]
    details = body["details"]
    assert isinstance(details, dict), f"the REAL app stringified it: {details!r}"
    assert details["reason"] == "order_below_floor"
    assert details["ordered_seconds"] == 9.0
    assert details["minimum_seconds"] == 10.0
    assert "reason" not in body["message"]


# ── the from-intent door: the FOURTH duration key, and its raise site ───────────


def _intent_body(key: str, seconds: int) -> dict:
    return {
        "intent": {
            "type": "website_review",
            "title": "review",
            "url": "https://example.com",
            key: seconds,
        },
        "format": "16x9",
    }


async def test_from_intent_camelcase_duration_is_refused_not_ignored(client) -> None:
    """BLOCKER 2: ``VideoIntent.duration_seconds`` is the FOURTH order door.

    ``projects.py`` resolves the order as
    ``body.target_duration_seconds or intent.duration_seconds``, so this key sets it
    whenever the outer one is omitted. Under ``extra="allow"`` a camelCase spelling was
    not even dropped — it rode into ``brief["intent"]`` while the order fell back to the
    120s default, and a 9s order returned 201 and STARTED A RUN.
    """
    r = await client.post("/api/projects/from-intent", json=_intent_body("durationSeconds", 9))

    assert r.status_code == 400, r.text


async def test_from_intent_snake_case_duration_is_still_refused(client) -> None:
    r = await client.post("/api/projects/from-intent", json=_intent_body("duration_seconds", 9))

    assert r.status_code == 400, r.text


async def test_the_from_intent_refusal_body_is_a_json_object(client) -> None:
    """BLOCKER 3: this door stringified at the RAISE SITE, where the wrapper cannot help.

    ``except ValueError: HTTPException(400, detail=str(exc))`` swallowed the typed
    refusal into a string before any handler saw it, so Defect 3 was fixed on two doors
    of three.
    """
    r = await client.post("/api/projects/from-intent", json=_intent_body("duration_seconds", 9))

    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert isinstance(detail, dict), f"still an unbranchable string: {detail!r}"
    assert detail["reason"] == "order_below_floor"
    assert detail["ordered_seconds"] == 9.0
    assert detail["minimum_seconds"] == 10.0


async def test_from_intent_above_the_floor_still_starts_a_run(client) -> None:
    """The honoured branch: the alias must not turn a valid order into a refusal."""
    r = await client.post("/api/projects/from-intent", json=_intent_body("durationSeconds", 42))

    assert r.status_code == 201, r.text
    assert r.json()["run_id"]


# ── F2: the FIFTH spelling — inside the free-form brief ────────────────────────
#
# `brief` is a plain dict, so no pydantic alias reaches inside it, and
# `website_review_ordered_seconds` reads `brief["duration_seconds"]` as the fallback
# order (order.py:122; `duration_conformance` reads it too at :896). A brief authored
# as `{"durationSeconds": 9}` matched neither reader and the order fell back to the
# 120s default — 201, run started, the stated duration silently ignored.
#
# It fails OPEN, so it is NOT a floor bypass. It is defect 2's exact shape all the
# same: a user's stated duration dropped without a word. Normalised at the door so
# BOTH readers (and the next one) see the canonical key.


def _brief_body(brief: dict) -> dict:
    return {
        "type": "website_review",
        "title": "review",
        "brief": {"source_url": "https://example.com", **brief},
        "deliverables": [{"name": "d", "format": "16x9"}],
    }


async def test_a_camelcase_brief_duration_is_honoured_not_dropped(client) -> None:
    """9s in the brief, camelCase: the floor must SEE it and refuse."""
    r = await client.post("/api/projects", json=_brief_body({"durationSeconds": 9}))

    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert detail["reason"] == "order_below_floor"
    assert detail["ordered_seconds"] == 9.0


async def test_a_camelcase_brief_duration_persists_under_the_canonical_key(
    client,
) -> None:
    """Honouring it is not enough — both readers look for `duration_seconds`.

    Pins that the value reaches the persisted brief under the canonical spelling, and
    that the camelCase key is gone so no consumer can read a second, contradicting
    order.
    """
    r = await client.post("/api/projects", json=_brief_body({"durationSeconds": 42}))

    assert r.status_code == 201, r.text
    brief = (await client.get(f"/api/projects/{r.json()['id']}")).json()["brief"]
    assert brief["duration_seconds"] == 42
    assert "durationSeconds" not in brief


async def test_an_explicit_snake_case_brief_duration_wins(client) -> None:
    """Both spellings present: the canonical one is the author's real intent."""
    r = await client.post(
        "/api/projects",
        json=_brief_body({"duration_seconds": 42, "durationSeconds": 9}),
    )

    assert r.status_code == 201, r.text
    brief = (await client.get(f"/api/projects/{r.json()['id']}")).json()["brief"]
    assert brief["duration_seconds"] == 42
    assert "durationSeconds" not in brief


async def test_a_patched_brief_duration_is_normalised_too(client) -> None:
    """PATCH /projects/{id} can set the order the same way, so it normalises the same."""
    created = await client.post("/api/projects", json=_brief_body({}))
    assert created.status_code == 201, created.text
    pid = created.json()["id"]

    r = await client.patch(
        f"/api/projects/{pid}",
        json={"brief": {"source_url": "https://example.com", "durationSeconds": 42}},
    )

    assert r.status_code == 200, r.text
    brief = (await client.get(f"/api/projects/{pid}")).json()["brief"]
    assert brief["duration_seconds"] == 42
    assert "durationSeconds" not in brief


# ── F1: WHERE the floor actually lives — the run-start choke point ─────────────
#
# The PR first claimed the floor holds because every writer of the order column is
# aliased. That reasoning is WRONG, and the audit caught it. There are three writers,
# and two of them — `POST /projects/{id}/deliverables` and `PATCH /deliverables/{id}` —
# enforce NO floor at all: `check_website_review_order` has exactly one caller,
# `projects.py:188` (the create door). A 9s order persists happily through either.
#
# The invariant holds for a different and better reason: `run_repo.create` has exactly
# ONE call site — `run_engine.py:451` — and `validate_website_review_order` sits two
# lines above it at `:449`. Nothing can start a run without passing the floor.
#
# So the floor lives at RUN START, not at create. The next person adding a duration
# door needs to know that: aliasing a new door is about honouring the user's number,
# NOT about the floor, which is already covered no matter what any door lets through.


async def test_a_sub_floor_order_persisted_via_patch_is_refused_at_run_start(
    client,
) -> None:
    """The real bypass path: set 9s through a door with no floor, then try to run it."""
    created = await client.post(
        "/api/projects",
        json={
            "type": "website_review",
            "title": "review",
            "brief": {"source_url": "https://example.com"},
            "deliverables": [{"name": "d", "format": "16x9", "target_duration_seconds": 60}],
        },
    )
    assert created.status_code == 201, created.text
    deliv = (await client.get(f"/api/projects/{created.json()['id']}")).json()["deliverables"][0]

    # The PATCH door enforces nothing — 9s is accepted and PERSISTED.
    patched = await client.patch(
        f"/api/deliverables/{deliv['id']}", json={"target_duration_seconds": 9}
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["target_duration_seconds"] == 9

    # ...and the choke point catches it anyway.
    run = await client.post(
        f"/api/deliverables/{deliv['id']}/runs", json={"pause_after_stages": []}
    )

    assert run.status_code == 400, run.text
    assert run.json()["detail"]["reason"] == "order_below_floor"
    assert run.json()["detail"]["ordered_seconds"] == 9.0


async def test_the_camelcase_spelling_also_persists_through_patch_and_is_refused(
    client,
) -> None:
    """Same path in camelCase — the alias makes the number land, the choke point refuses."""
    created = await client.post(
        "/api/projects",
        json={
            "type": "website_review",
            "title": "review",
            "brief": {"source_url": "https://example.com"},
            "deliverables": [{"name": "d", "format": "16x9", "target_duration_seconds": 60}],
        },
    )
    deliv = (await client.get(f"/api/projects/{created.json()['id']}")).json()["deliverables"][0]

    patched = await client.patch(
        f"/api/deliverables/{deliv['id']}", json={"targetDurationSeconds": 9}
    )
    assert patched.status_code == 200, patched.text
    # Without the alias this would still read 60 — the user's 9 silently ignored.
    assert patched.json()["target_duration_seconds"] == 9

    run = await client.post(
        f"/api/deliverables/{deliv['id']}/runs", json={"pause_after_stages": []}
    )
    assert run.status_code == 400, run.text
    assert run.json()["detail"]["reason"] == "order_below_floor"
