"""Tell George about w2v early-access signups in Telegram (contract C1).

Port of fr ``backend/src/services/signup/telegram.ts``. Bot @nonagenticaibot
(``W2V_SIGNUP_TELEGRAM_BOT_TOKEN``), chat ``W2V_SIGNUP_TELEGRAM_CHAT_ID``, forum topic
``W2V_SIGNUP_TELEGRAM_THREAD_ID`` (never 167 — external-probe alarms). Two messages:

* a NEW pending signup (insert only — a de-duplicated re-submit sends nothing):
  name, email, company, source, language, a message excerpt, and
  1. CALLBACK buttons Approve / Reject, ``callback_data`` = ``ea1:w2v:a|r:<id>``
     (≤ 64 bytes). Telegram delivers the tap to chat-bridge — the bot's ONLY update
     consumer — which checks the tapper and calls
     ``POST /api/signup-approvals/{id}/telegram-decision`` with its own SA token;
  2. a URL button "Open in Studio" → ``{W2V_FRONTEND_URL}/admin/onboarding?signup=<id>``.
* the same message again (headline "still PENDING" + why) when an approve failed
  before the decision, so a pending row always has buttons;
* a short outcome after an approve / reject on the super_admin routes;
* a notice when an early-access REMOVAL ends ``done`` / ``blocked`` / ``failed``;
* a notice when the welcome email's send is UNCERTAIN (``unknown`` / a resumed
  ``in_flight``): the row parks for the super_admin Retry, so somebody must be
  told. Sent once per send attempt (the caller records it on the step).

Plain text, no ``parse_mode``: every REQUESTER-TYPED value sits inside a ``code``
entity, so a URL typed into a name renders as inert monospace instead of a link.
Entity offsets are UTF-16 code units (Telegram's unit), not Python characters.

⛔ SEND ONLY: never setWebhook / deleteWebhook / getUpdates (chat-bridge owns the
bot's webhook). ⛔ FIRE-AND-FORGET: ``notify_*`` return at once; the send runs
detached with a bounded retry, every failure is logged and never raised. The bot
token is part of the API URL, so every logged error is redacted.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from urllib.parse import quote

import httpx

from src.config import SignupSettings, current_signup_settings, get_settings
from src.services.signup.text import excerpt, single_line

_log = logging.getLogger("w2v_orchestrator.signup.telegram")

#: The product key of contract C1 (``ea1:w2v:a|r:<id>``).
SIGNUP_APP_KEY = "w2v"
CALLBACK_PREFIX = f"ea1:{SIGNUP_APP_KEY}"
MESSAGE_EXCERPT_MAX = 300
FIELD_MAX = 200
NEW_SIGNUP_HEADLINE = "New Studio early-access signup"
#: Re-posted when an approve failed before the decision (the row is still
#: pending): a pending row must always have Approve / Reject buttons somewhere.
STILL_PENDING_HEADLINE = "Studio early-access signup still PENDING"

_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = (1.0, 4.0)
_MAX_RETRY_AFTER_SECONDS = 30.0
_SEND_TIMEOUT_SECONDS = 10.0

PostFn = Callable[[str, dict[str, object]], Awaitable[httpx.Response]]
_post_override: PostFn | None = None
_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
_inflight: set[asyncio.Task[None]] = set()


def set_telegram_post_for_tests(post: PostFn | None) -> None:
    """⛔ TESTS ONLY: replace the HTTP POST (no test may reach api.telegram.org)."""
    global _post_override
    _post_override = post


async def drain_telegram_for_tests() -> None:
    """⛔ TESTS ONLY: await every detached send started so far."""
    while _inflight:
        await asyncio.gather(*list(_inflight), return_exceptions=True)


def callback_data(signup_id: str, action: str) -> str:
    return f"{CALLBACK_PREFIX}:{'a' if action == 'approve' else 'r'}:{signup_id}"


def frontend_url() -> str:
    return get_settings().provisioning.w2v_frontend_url.strip().rstrip("/")


def deep_link(signup_id: str) -> str:
    return f"{frontend_url()}/admin/onboarding?signup={quote(signup_id, safe='')}"


def _utf16_len(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2


@dataclass
class _Message:
    text: str = ""
    entities: list[dict[str, object]] = field(default_factory=list)

    def plain(self, value: str) -> _Message:
        self.text += value
        return self

    def code(self, value: str | None, limit: int = FIELD_MAX) -> _Message:
        v = excerpt(value, limit)
        if not v:
            return self.plain("-")
        self.entities.append(
            {"type": "code", "offset": _utf16_len(self.text), "length": _utf16_len(v)}
        )
        self.text += v
        return self

    def field(self, label: str, value: str | None, limit: int = FIELD_MAX) -> _Message:
        v = excerpt(value, limit)
        self.text += f"{label}: "
        if not v:
            return self.plain("-\n")
        self.entities.append(
            {"type": "code", "offset": _utf16_len(self.text), "length": _utf16_len(v)}
        )
        self.text += v
        return self.plain("\n")


def _base(cfg: SignupSettings, m: _Message) -> dict[str, object]:
    return {
        "chat_id": cfg.telegram_chat_id,
        "message_thread_id": int(cfg.telegram_thread_id),
        "text": m.text.rstrip("\n"),
        "entities": m.entities,
        "link_preview_options": {"is_disabled": True},
    }


@dataclass(frozen=True, slots=True)
class NewSignupNotice:
    id: str
    source: str
    pending_total: int | None
    #: Requester-typed — rendered inside ``code`` entities only.
    email: str
    name: str | None
    company: str | None
    message: str | None = None
    locale: str | None = None
    headline: str = NEW_SIGNUP_HEADLINE
    #: One operator line (ours, not requester-typed), e.g. why it is re-posted.
    note: str | None = None


@dataclass(frozen=True, slots=True)
class DecisionNotice:
    id: str
    outcome: str  # approved | rejected
    email: str | None = None
    name: str | None = None
    tenant_id: str | None = None
    tenant_name: str | None = None
    tenant_slug: str | None = None
    decided_via: str | None = None
    email_state: str | None = None  # sent | waiting
    not_ok: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class EmailUncertainNotice:
    id: str
    status: str  # unknown | in_flight
    #: Requester-typed — rendered inside ``code`` entities only.
    email: str
    name: str | None = None


def email_uncertain_payload(n: EmailUncertainNotice, cfg: SignupSettings) -> dict[str, object]:
    m = _Message().plain("Studio early access: the welcome email for ")
    m.code(n.name).plain(" ").code(n.email)
    m.plain(" may not have been delivered. Check and retry: ")
    m.plain(f"{deep_link(n.id)}\n\n" if frontend_url() else "the onboarding page\n\n")
    m.plain(f"Send status: {n.status}\nSignup id: {n.id}")
    payload = _base(cfg, m)
    # A relative URL makes Telegram refuse the WHOLE message (400): no base URL, no button.
    if frontend_url():
        payload["reply_markup"] = {
            "inline_keyboard": [[{"text": "Open in Studio", "url": deep_link(n.id)}]]
        }
    return payload


@dataclass(frozen=True, slots=True)
class RemovalNotice:
    """The early-access REMOVAL reached ``done``, ``blocked`` or ``failed``.

    It NEVER carries requester data — no name, no address, no company-derived
    slug — in any state: a Telegram message cannot be erased with the row. The
    signup id, the slug TOMBSTONE (``removed-<id prefix>``), the tenant id and
    the Open-in-Studio button identify the workspace."""

    id: str
    state: str  # done | blocked | failed
    #: The slug tombstone (never the real, company-derived slug).
    tenant_slug: str | None = None
    tenant_id: str | None = None
    blockers: list[dict[str, object]] = field(default_factory=list)
    last_error: dict[str, object] | None = None
    attempts: int = 0
    #: The identity step's code (``removal.steps.identity.code``): what really
    #: happened to the login — the ``done`` text is built from it.
    identity: str | None = None


#: The ``done`` notice's login clause per identity-step code. Anything not listed
#: says so plainly instead of claiming the login was retired.
_IDENTITY_TEXT: dict[str, str] = {
    "deleted": "login deleted",
    "disabled": "login disabled and marked",
    "disabled_unmarked": "login disabled (the removal marker did NOT stick: check it)",
    "absent": "login already gone",
    "identity_kept_in_use": "login KEPT (another Studio workspace uses it)",
    "identity_not_owned": "login KEPT (it is not this signup's: never touched)",
    "no_identity": "no login to retire",
}


def removal_identity_text(code: str | None) -> str:
    return _IDENTITY_TEXT.get(code or "", f"login: unknown outcome ({code or 'none'})")


def removal_payload(n: RemovalNotice, cfg: SignupSettings) -> dict[str, object]:
    m = _Message().plain(f"Studio early-access workspace removal {n.state.upper()}\n\n")
    m.plain(f"Workspace: {n.tenant_slug or '-'}\n")
    m.plain(f"Tenant id: {n.tenant_id or '-'}\n")
    if n.state == "done":
        m.plain(
            "Tenant deleted, licences revoked and archived, "
            f"{removal_identity_text(n.identity)}, export archive deleted, "
            "personal data erased. "
            "The address may sign up again.\n"
        )
    elif n.state == "blocked":
        m.plain("Nothing further was removed. Blockers:\n")
        for b in n.blockers:
            m.plain(f"  - {b.get('code')}\n")
        m.plain("Fix them and Retry, or Resolve on the onboarding page.\n")
    else:
        m.plain(f"Gave up after {n.attempts} attempts.\n")
        if n.last_error:
            m.plain(f"Last error: {n.last_error.get('step')} ({n.last_error.get('code')})\n")
        m.plain("Retry or Resolve on the onboarding page.\n")
    m.plain(f"Signup id: {n.id}")
    payload = _base(cfg, m)
    if frontend_url() and n.state != "done":
        payload["reply_markup"] = {
            "inline_keyboard": [[{"text": "Open in Studio", "url": deep_link(n.id)}]]
        }
    return payload


def new_signup_payload(n: NewSignupNotice, cfg: SignupSettings) -> dict[str, object]:
    m = (
        _Message()
        .plain(f"{n.headline}\n\n")
        .field("Name", n.name)
        .field("Email", n.email)
        .field("Company", n.company)
        .plain(f"Source: {n.source}\n")
        .plain(f"Language: {n.locale or '- (en)'}\n")
        .plain("Trusted via: website/chat intake (service account)\n")
    )
    if n.message and single_line(n.message):
        m.field("Message", n.message, MESSAGE_EXCERPT_MAX)
    if n.note:
        m.plain(f"{single_line(n.note)}\n")
    m.plain(f"Pending signups: {n.pending_total if n.pending_total is not None else '-'}\n")
    m.plain(f"Signup id: {n.id}\n\n")
    m.plain(
        "Approve creates the workspace now (default name, local LLM access) and emails the "
        "set-password link. Open in Studio to see the request first."
    )
    keyboard: list[list[dict[str, str]]] = [
        [
            {"text": "Approve", "callback_data": callback_data(n.id, "approve")},
            {"text": "Reject", "callback_data": callback_data(n.id, "reject")},
        ]
    ]
    # A relative URL makes Telegram refuse the WHOLE message (400): no base URL, no button.
    if frontend_url():
        keyboard.append([{"text": "Open in Studio", "url": deep_link(n.id)}])
    return {**_base(cfg, m), "reply_markup": {"inline_keyboard": keyboard}}


def decision_payload(d: DecisionNotice, cfg: SignupSettings) -> dict[str, object]:
    m = _Message().plain(f"Studio signup {d.outcome}\n\n")
    m.field("Name", d.name).field("Email", d.email)
    if d.decided_via:
        m.plain(f"Decided via: {d.decided_via}\n")
    if d.outcome == "approved":
        m.field("Workspace", d.tenant_name).field("Slug", d.tenant_slug)
        m.plain(f"Tenant id: {d.tenant_id or '-'}\n")
        m.plain(
            "Set-password email sent.\n"
            if d.email_state == "sent"
            else "The set-password email follows once every provisioning step is ok "
            "(retried automatically).\n"
        )
        if not d.not_ok:
            m.plain("Provisioning: all steps ok.\n")
        else:
            m.plain("Provisioning steps NOT ok:\n")
            for s in d.not_ok:
                code = f" ({s['code']})" if s.get("code") else ""
                m.plain(f"  - {s['step']}: {s['status']}{code}\n")
    m.plain(f"Signup id: {d.id}")
    return _base(cfg, m)


def notify_new_signup(n: NewSignupNotice) -> None:
    cfg = current_signup_settings()
    if not cfg.telegram_enabled:
        return
    _detach("new_signup", cfg, new_signup_payload(n, cfg), n.id)


def notify_decision(d: DecisionNotice) -> None:
    cfg = current_signup_settings()
    if not cfg.telegram_enabled:
        return
    _detach(f"decision:{d.outcome}", cfg, decision_payload(d, cfg), d.id)


def notify_email_uncertain(n: EmailUncertainNotice) -> bool:
    """Fire-and-forget; False when Telegram is not configured (nothing sent)."""
    cfg = current_signup_settings()
    if not cfg.telegram_enabled:
        return False
    _detach("email_uncertain", cfg, email_uncertain_payload(n, cfg), n.id)
    return True


def notify_removal(n: RemovalNotice) -> bool:
    """Fire-and-forget; False when Telegram is not configured (nothing sent)."""
    cfg = current_signup_settings()
    if not cfg.telegram_enabled:
        return False
    _detach(f"removal:{n.state}", cfg, removal_payload(n, cfg), n.id)
    return True


def log_telegram_status() -> None:
    """One line at boot: on, or off and which setting is missing."""
    cfg = current_signup_settings()
    if cfg.telegram_enabled:
        _log.info(
            "signup telegram enabled chat=%s thread=%s",
            cfg.telegram_chat_id,
            cfg.telegram_thread_id,
        )
        return
    missing = [
        name
        for name, ok in (
            ("W2V_SIGNUP_TELEGRAM_BOT_TOKEN", bool(cfg.telegram_bot_token.get_secret_value())),
            ("W2V_SIGNUP_TELEGRAM_CHAT_ID", bool(cfg.telegram_chat_id)),
            ("W2V_SIGNUP_TELEGRAM_THREAD_ID", cfg.telegram_thread_id.isdigit()),
        )
        if not ok
    ]
    _log.info("signup telegram disabled missing=%s", ",".join(missing))


def _redact(text: str, token: str) -> str:
    return text.replace(token, "<redacted>") if token else text


def _detach(kind: str, cfg: SignupSettings, payload: dict[str, object], signup_id: str) -> None:
    """``kind`` and ``signup_id`` are the ONLY notice facts that reach the log: never
    a name, an address or the token."""
    token = cfg.telegram_bot_token.get_secret_value()

    async def _run() -> None:
        try:
            await _send_with_retry(kind, cfg, payload, signup_id)
        except Exception as exc:  # noqa: BLE001 — fire-and-forget: logged, never raised
            _log.error(
                "signup telegram send crashed kind=%s id=%s: %s",
                kind,
                signup_id,
                _redact(f"{type(exc).__name__}: {exc}", token),
            )

    task = asyncio.create_task(_run(), name=f"signup-telegram-{kind}")
    _inflight.add(task)
    task.add_done_callback(_inflight.discard)


async def _post(url: str, payload: dict[str, object]) -> httpx.Response:
    if _post_override is not None:
        return await _post_override(url, payload)
    async with httpx.AsyncClient(timeout=_SEND_TIMEOUT_SECONDS) as client:
        return await client.post(url, json=payload)


async def _send_with_retry(
    kind: str, cfg: SignupSettings, payload: dict[str, object], signup_id: str
) -> None:
    token = cfg.telegram_bot_token.get_secret_value()
    url = f"{cfg.telegram_api_url.rstrip('/')}/bot{token}/sendMessage"
    last_error = ""
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        wait = _BACKOFF_SECONDS[min(attempt - 1, len(_BACKOFF_SECONDS) - 1)]
        try:
            res = await _post(url, payload)
            if 200 <= res.status_code < 300:
                # A success leaves a trace, so a delivered notice is verifiable from the logs.
                _log.info("signup telegram sent kind=%s id=%s attempt=%d", kind, signup_id, attempt)
                return
            try:
                body = res.json()
            except ValueError:
                body = {}
            body = body if isinstance(body, dict) else {}
            description = body.get("description")
            last_error = f"HTTP {res.status_code}" + (f": {description}" if description else "")
            # 4xx other than 429 is configuration (bad chat/thread, bot not in chat).
            if res.status_code != 429 and res.status_code < 500:
                break
            retry_after = (body.get("parameters") or {}).get("retry_after")
            if res.status_code == 429 and isinstance(retry_after, (int, float)):
                wait = min(float(retry_after), _MAX_RETRY_AFTER_SECONDS)
        except (httpx.HTTPError, OSError) as exc:
            last_error = _redact(f"{type(exc).__name__}: {exc}", token)
        if attempt < _MAX_ATTEMPTS:
            await _sleep(wait)
    _log.warning(
        "signup telegram send failed kind=%s id=%s error=%s (the signup itself is recorded; "
        "check W2V_SIGNUP_TELEGRAM_* and the bot's membership of the chat)",
        kind,
        signup_id,
        _redact(last_error, token),
    )
