"""Early-access Telegram notices (contract C1): payload shape, entities, retry, redaction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from uuid import uuid7

import httpx
import pytest

from src.config import SignupSettings, clear_settings_cache
from src.services.signup import telegram as tg

pytestmark = pytest.mark.asyncio

BOT_TOKEN = "123456:SECRET-BOT-TOKEN"  # pragma: allowlist secret


@pytest.fixture
def cfg(monkeypatch) -> SignupSettings:
    monkeypatch.setenv("W2V_FRONTEND_URL", "https://w2v.example.test/")
    monkeypatch.setenv("W2V_SIGNUP_TELEGRAM_BOT_TOKEN", BOT_TOKEN)
    monkeypatch.setenv("W2V_SIGNUP_TELEGRAM_CHAT_ID", "-1003934173351")
    monkeypatch.setenv("W2V_SIGNUP_TELEGRAM_THREAD_ID", "4242")
    clear_settings_cache()
    return SignupSettings()


def _utf16(s: str) -> int:
    return len(s.encode("utf-16-le")) // 2


def _entity_texts(payload: dict) -> list[str]:
    units = payload["text"].encode("utf-16-le")
    return [
        units[2 * e["offset"] : 2 * (e["offset"] + e["length"])].decode("utf-16-le")
        for e in payload["entities"]
    ]


async def test_callback_data_is_c1_and_fits_64_bytes() -> None:
    sid = str(uuid7())
    assert tg.callback_data(sid, "approve") == f"ea1:w2v:a:{sid}"
    assert tg.callback_data(sid, "reject") == f"ea1:w2v:r:{sid}"
    assert len(tg.callback_data(sid, "approve").encode()) <= 64


async def test_new_signup_payload_shape(cfg: SignupSettings) -> None:
    sid = str(uuid7())
    notice = tg.NewSignupNotice(
        id=sid,
        source="web",
        pending_total=3,
        email="ann@acme.example",
        name="Ann 🎬 https://evil.example",
        company="Acme",
        message="Hello\nworld " + "x" * 400,
        locale="de",
    )
    p: dict[str, Any] = tg.new_signup_payload(notice, cfg)
    assert p["chat_id"] == "-1003934173351" and p["message_thread_id"] == 4242
    assert "parse_mode" not in p and p["link_preview_options"] == {"is_disabled": True}
    assert p["text"].startswith(tg.NEW_SIGNUP_HEADLINE)
    assert "Source: web" in p["text"] and "Language: de" in p["text"]
    assert f"Signup id: {sid}" in p["text"]
    # Every requester-typed value sits in a code entity — exactly those, UTF-16 offsets.
    assert all(e["type"] == "code" for e in p["entities"])
    texts = _entity_texts(p)
    assert texts[:3] == ["Ann 🎬 https://evil.example", "ann@acme.example", "Acme"]
    assert texts[3].startswith("Hello world ") and texts[3].endswith("…")
    assert len(texts[3]) == tg.MESSAGE_EXCERPT_MAX + 1
    [[approve, reject], [open_btn]] = p["reply_markup"]["inline_keyboard"]
    assert approve == {"text": "Approve", "callback_data": f"ea1:w2v:a:{sid}"}
    assert reject == {"text": "Reject", "callback_data": f"ea1:w2v:r:{sid}"}
    assert open_btn == {
        "text": "Open in Studio",
        "url": f"https://w2v.example.test/admin/onboarding?signup={sid}",
    }
    assert all(len(b["callback_data"].encode()) <= 64 for b in (approve, reject))


async def test_no_frontend_url_means_no_url_button(cfg: SignupSettings, monkeypatch) -> None:
    monkeypatch.setenv("W2V_FRONTEND_URL", "")
    clear_settings_cache()
    p: dict[str, Any] = tg.new_signup_payload(
        tg.NewSignupNotice(
            id="x", source="chat", pending_total=None, email="a@b.c", name=None, company=None
        ),
        cfg,
    )
    assert len(p["reply_markup"]["inline_keyboard"]) == 1
    assert "Language: - (en)" in p["text"]


async def test_decision_payload_lists_unfinished_steps(cfg: SignupSettings) -> None:
    p: dict[str, Any] = tg.decision_payload(
        tg.DecisionNotice(
            id="sid",
            outcome="approved",
            email="ann@acme.example",
            name="Ann",
            tenant_id="t-1",
            tenant_name="Acme",
            tenant_slug="acme",
            decided_via="Studio onboarding screen",
            email_state="waiting",
            not_ok=[{"step": "provision", "status": "failed", "code": "licences_down"}],
        ),
        cfg,
    )
    assert p["text"].startswith("Studio signup approved")
    assert "  - provision: failed (licences_down)" in p["text"]
    assert "reply_markup" not in p
    assert _entity_texts(p) == ["Ann", "ann@acme.example", "Acme", "acme"]


async def test_disabled_without_full_config(monkeypatch) -> None:
    monkeypatch.setenv("W2V_SIGNUP_TELEGRAM_BOT_TOKEN", BOT_TOKEN)
    monkeypatch.setenv("W2V_SIGNUP_TELEGRAM_CHAT_ID", "-100")
    monkeypatch.setenv("W2V_SIGNUP_TELEGRAM_THREAD_ID", "not-a-number")
    assert SignupSettings().telegram_enabled is False
    sent: list = []

    async def post(url: str, payload: dict) -> httpx.Response:
        sent.append(payload)
        return httpx.Response(200)

    tg.set_telegram_post_for_tests(post)
    try:
        tg.notify_new_signup(
            tg.NewSignupNotice(
                id="x", source="web", pending_total=1, email="a@b.c", name="A", company=None
            )
        )
        await tg.drain_telegram_for_tests()
    finally:
        tg.set_telegram_post_for_tests(None)
    assert sent == []


async def test_fire_and_forget_retries_429_not_400_and_redacts_the_token(
    cfg: SignupSettings, caplog, monkeypatch
) -> None:
    calls: list[str] = []
    answers = [
        httpx.Response(429, json={"parameters": {"retry_after": 1}}),
        httpx.Response(500),
        httpx.Response(200, json={"ok": True}),
    ]

    async def post(url: str, payload: dict) -> httpx.Response:
        calls.append(url)
        return answers.pop(0)

    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    monkeypatch.setattr(tg, "_sleep", fake_sleep)
    tg.set_telegram_post_for_tests(post)
    try:
        tg.notify_decision(tg.DecisionNotice(id="x", outcome="rejected"))
        await tg.drain_telegram_for_tests()
        assert len(calls) == 3 and calls[0].endswith("/sendMessage")
        assert sleeps == [1.0, 4.0]

        # A 400 (bad chat / thread) is configuration: one attempt, logged, redacted.
        calls.clear()

        async def bad(url: str, payload: dict) -> httpx.Response:
            calls.append(url)
            raise httpx.ConnectError(f"cannot reach {url}")

        tg.set_telegram_post_for_tests(bad)
        with caplog.at_level(logging.WARNING, logger="w2v_orchestrator.signup.telegram"):
            tg.notify_decision(tg.DecisionNotice(id="x", outcome="rejected"))
            await tg.drain_telegram_for_tests()
        assert len(calls) == 3
        assert BOT_TOKEN not in caplog.text and "<redacted>" in caplog.text

        calls.clear()

        async def refused(url: str, payload: dict) -> httpx.Response:
            calls.append(url)
            return httpx.Response(400, json={"description": "chat not found"})

        tg.set_telegram_post_for_tests(refused)
        tg.notify_decision(tg.DecisionNotice(id="x", outcome="rejected"))
        await tg.drain_telegram_for_tests()
        assert len(calls) == 1
    finally:
        tg.set_telegram_post_for_tests(None)


async def test_a_successful_send_logs_kind_and_id_without_pii(cfg: SignupSettings, caplog) -> None:
    """A delivered notice leaves an INFO line (kind + signup id) — and nothing else."""
    sid = str(uuid7())

    async def post(url: str, payload: dict) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    tg.set_telegram_post_for_tests(post)
    try:
        with caplog.at_level(logging.INFO, logger="w2v_orchestrator.signup.telegram"):
            tg.notify_removal(tg.RemovalNotice(id=sid, state="done", identity="deleted"))
            tg.notify_decision(
                tg.DecisionNotice(
                    id=sid, outcome="approved", email="ann@acme.example", name="Ann Secret"
                )
            )
            tg.notify_decision(tg.DecisionNotice(id=sid, outcome="rejected"))
            tg.notify_new_signup(
                tg.NewSignupNotice(
                    id=sid,
                    source="web",
                    pending_total=1,
                    email="ann@acme.example",
                    name="Ann Secret",
                    company="Acme",
                )
            )
            await tg.drain_telegram_for_tests()
    finally:
        tg.set_telegram_post_for_tests(None)
    sent = sorted(
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.INFO and r.getMessage().startswith("signup telegram sent")
    )
    assert sent == sorted(
        f"signup telegram sent kind={k} id={sid} attempt=1"
        for k in ("removal:done", "decision:approved", "decision:rejected", "new_signup")
    )
    for leak in ("ann@acme.example", "Ann Secret", "Acme", BOT_TOKEN):
        assert leak not in caplog.text


async def test_a_failed_send_logs_no_success_line(cfg: SignupSettings, caplog) -> None:
    async def refused(url: str, payload: dict) -> httpx.Response:
        return httpx.Response(400, json={"description": "chat not found"})

    tg.set_telegram_post_for_tests(refused)
    try:
        with caplog.at_level(logging.INFO, logger="w2v_orchestrator.signup.telegram"):
            tg.notify_removal(tg.RemovalNotice(id="sid-1", state="failed", attempts=3))
            await tg.drain_telegram_for_tests()
    finally:
        tg.set_telegram_post_for_tests(None)
    assert "signup telegram sent" not in caplog.text
    assert "signup telegram send failed kind=removal:failed id=sid-1" in caplog.text


async def test_send_only_never_touches_the_webhook() -> None:
    src = Path(tg.__file__).read_text()
    for banned in ("setWebhook", "deleteWebhook", "getUpdates"):
        assert f"/{banned}" not in src and f'"{banned}' not in src


async def test_email_uncertain_payload_shape(cfg: SignupSettings, monkeypatch) -> None:
    sid = str(uuid7())
    n = tg.EmailUncertainNotice(
        id=sid, status="unknown", email="ann@acme.example", name="Ann 🎬 https://evil.example"
    )
    p: dict[str, Any] = tg.email_uncertain_payload(n, cfg)
    link = f"https://w2v.example.test/admin/onboarding?signup={sid}"
    assert p["text"].startswith(
        "Studio early access: the welcome email for Ann 🎬 https://evil.example "
        "ann@acme.example may not have been delivered. Check and retry: " + link
    )
    assert "parse_mode" not in p and p["message_thread_id"] == 4242
    assert _entity_texts(p) == ["Ann 🎬 https://evil.example", "ann@acme.example"]
    assert p["reply_markup"] == {"inline_keyboard": [[{"text": "Open in Studio", "url": link}]]}

    monkeypatch.setenv("W2V_FRONTEND_URL", "")
    clear_settings_cache()
    bare: dict[str, Any] = tg.email_uncertain_payload(n, cfg)
    assert "reply_markup" not in bare and "?signup=" not in bare["text"]
