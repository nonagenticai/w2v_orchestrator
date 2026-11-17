"""Login secrets inside briefs: sealed at rest, redacted on read, unsealed only to dispatch.

A recording brief carries real tenant credentials: ``brief.auth_config`` (showcase /
demo_video → frontend_test_runner) and ``brief.auth_info`` (promo site capture,
website_review critique, ``brief.sources[].auth_info``). Until this module they were
persisted in plaintext in ``projects.brief`` AND copied into every
``runs.config_snapshot``, and both ``GET /api/projects/{id}`` and
``GET /api/runs/{id}`` returned them verbatim (measured 2026-10-02: 31 projects and
31 runs on the primary).

Three rules, one per verb:

* **seal** — every secret field under an auth block is replaced by
  ``{"$sealed": "<fernet token>"}`` before it reaches the database. Applied by ORM
  ``before_insert``/``before_update`` listeners on ``Project.brief`` and
  ``Run.config_snapshot`` (``src/database/models.py``) so NO write path can skip it,
  and explicitly at the API door so a missing key is a clean 503, not a 500.
* **redact** — every API response that carries a brief/snapshot passes through
  :func:`redact_auth_secrets`, which replaces the secret (sealed OR plaintext) with
  :data:`REDACTED`. A client that echoes ``REDACTED`` back on PATCH keeps the stored
  secret (:func:`restore_redacted`).
* **unseal** — only a stage, in memory, at the moment it hands the login to the
  service that types it (:func:`unseal_auth`). Never logged, never persisted.

An auth block is a dict (one login) or a list of dicts (one login per origin — the
showcase demo signs into Keycloak-fronted marketing, Twenty, the console and the
public site in ONE run). :func:`auth_entries` normalises both shapes.

Several logins may share an origin when each carries a distinct ``label`` (a demo
that signs the SAME app in as a requester and as a manager, side by side): a login
is keyed by ``(origin, label)``. A scenario selects its login by label.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterator
from urllib.parse import urljoin

from cryptography.fernet import Fernet, InvalidToken, MultiFernet

from src.config import get_settings

#: Brief keys whose value is a login block (dict or list of dicts), at any depth.
AUTH_BLOCK_KEYS = frozenset({"auth_config", "auth_info"})
#: Fields inside a login block that are secrets.
SECRET_FIELDS = frozenset(
    {
        "password",
        "token",
        "access_token",
        "refresh_token",
        "client_secret",
        "api_key",
        "secret",
        "authorization",
        "headers",
        # Session cookies ARE credentials: the recorder (UrlAuthInfo type=cookies)
        # replays them as a logged-in session. `cookies` is a LIST of objects.
        "cookie_header",
        "cookies",
        "cookie",
    }
)
#: Marks a sealed value that was not a string (e.g. a `cookies` list) — JSON inside.
SEALED_JSON = "$json"
#: Marker key of a sealed value.
SEALED = "$sealed"
#: What a response shows in place of a secret. A dict, so no real password can collide
#: with it; a client echoing it back on PATCH means "keep what is stored".
REDACTED: dict[str, bool] = {"redacted": True}
#: Upper bound on logins per block — the demo needs four; this is a sanity ceiling.
MAX_AUTH_ENTRIES = 10


class AuthSecretError(RuntimeError):
    """A login secret cannot be sealed or unsealed (no key configured, or a bad token).

    The message never contains the secret."""


def _fernet() -> MultiFernet | None:
    raw = get_settings().auth_secrets.auth_config_secret_keys.get_secret_value()
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        return None
    try:
        return MultiFernet([Fernet(k.encode()) for k in keys])
    except ValueError as exc:  # malformed key — never echo it
        raise AuthSecretError("AUTH_CONFIG_SECRET_KEYS holds a malformed Fernet key") from exc


def _is_sealed(value: object) -> bool:
    return isinstance(value, dict) and isinstance(value.get(SEALED), str)


def _is_redacted(value: object) -> bool:
    return value == REDACTED


def auth_entries(block: object) -> list[dict]:
    """The login dicts of an auth block: a dict is one login, a list is several."""
    if isinstance(block, dict):
        return [block]
    if isinstance(block, list):
        return [e for e in block if isinstance(e, dict)]
    return []


#: Brief keys holding a URL a login is bound to (FTR falls back to ``target_url``;
#: ``sources[].ref`` is what a ``sources[].auth_info`` login is sent to).
_BOUND_URL_KEYS = ("target_url", "source_url", "url", "ref")
_LOGIN_URL_KEYS = ("login_url", "origin")
#: The site a login SIGNS INTO when that is not the sign-in page's own origin —
#: single sign-on, where the form is on an identity provider (``login_url`` on
#: ``auth.example``) and the session is for the app (``signs_into: app.example``).
#: DECLARED by the tenant on the Brief page, never inferred: a login without it
#: signs into its own origin (the sign-in page's), exactly as before.
SIGNS_INTO = "signs_into"
#: Set by :func:`restore_redacted` on a login whose saved secret was dropped because
#: the login changed; returned to the client (it is not a secret) so the Brief page
#: can say "re-enter the password", and cleared when a new secret is typed.
NEEDS_REENTRY = "needs_reentry"
#: Keys only the orchestrator reads; never forwarded to a service that signs in.
_ORCHESTRATOR_ONLY_KEYS = frozenset({SIGNS_INTO, NEEDS_REENTRY})
#: A login's label: the name a scenario selects it by (``requester``, ``gm``). A slug,
#: because FTR writes it into a file keyed by it and a package reads it back.
_LABEL_RE = re.compile(r"[a-z0-9](?:[a-z0-9_-]{0,38}[a-z0-9])?")
_DEFAULT_HTTPS_PORT = 443


#: One DNS label: ASCII letters/digits/hyphen, no leading/trailing hyphen, 1..63.
_LDH_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
#: A label WHATWG reads as an IPv4 number (decimal, 0x-hex, octal all match this).
_NUMERIC_LABEL = re.compile(r"(?:0x[0-9a-f]*|[0-9]+)")


def _strict_host_port(raw: object, field: str) -> tuple[str, int]:
    """Parse ``raw`` into ``(host, port)`` or raise ValueError — FAIL CLOSED.

    The threat is a parser differential: this module and the browser's WHATWG
    parser must reach the same host, or a check approves one host while Chromium
    types the password into another (``https://attacker.example\\@twenty.example/``
    is twenty.example to ``urlsplit``, attacker.example to WHATWG). So nothing is
    interpreted; a URL is accepted only in a form both parsers read identically:

    - the literal lowercase prefix ``https://`` (no ``HTTPS:``, ``https:host``,
      ``https:/\\host``, scheme-relative ``//host``);
    - printable ASCII only: no backslash, whitespace, control, zero-width, soft
      hyphen, full-width or ideographic-dot (``。``) characters anywhere;
    - an authority of exactly ``host[:port]`` — no ``@`` userinfo, no ``%``;
    - a host of plain LDH labels, the last one not numeric or ``0x`` (WHATWG would
      read those as an IPv4 address); no IP literals at all, v4 or ``[v6]``;
    - a port of 1..65535 digits, if any.
    """
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{field} must be a non-empty https URL")
    if "\\" in raw:
        raise ValueError(f"{field} must not contain a backslash")
    if any(not (0x21 <= ord(c) <= 0x7E) for c in raw):
        raise ValueError(f"{field} must be printable ASCII with no whitespace")
    if not raw.startswith("https://"):
        raise ValueError(f"{field} must start with the literal https://")
    authority = re.split(r"[/?#]", raw[len("https://") :], maxsplit=1)[0]
    if "@" in authority or "%" in authority:
        raise ValueError(f"{field} must not carry userinfo or percent-encoding in its host")
    host, sep, port_text = authority.partition(":")
    port = _DEFAULT_HTTPS_PORT
    if sep:
        if not port_text.isdigit() or not 0 < int(port_text) <= 65535:
            raise ValueError(f"{field} has an invalid port")
        port = int(port_text)
    host = host.lower()
    labels = host.split(".")
    if len(host) > 253 or not all(_LDH_LABEL.fullmatch(label) for label in labels):
        raise ValueError(f"{field} must name a host of plain ASCII letters, digits and hyphens")
    if _NUMERIC_LABEL.fullmatch(labels[-1]):
        raise ValueError(f"{field} must not be an IP address")
    return host, port


def strict_login_url(raw: object, field: str) -> str:
    """Return ``raw`` iff it is a URL a credential may be typed into; else ValueError."""
    _strict_host_port(raw, field)
    return str(raw)


def canonical_origin(raw: object) -> str | None:
    """THE origin function: ``https://host[:port]`` of a strict login URL, else None.

    Serialised like the browser's ``new URL(u).origin`` (default port omitted), the
    shape FTR keys AUTH_LOGINS_FILE by. Shared by validation, multi-login keying,
    :func:`auth_for_origin` and :func:`restore_redacted`, so the host a secret is
    approved for is computed exactly one way."""
    try:
        host, port = _strict_host_port(raw, "url")
    except ValueError:
        return None
    return f"https://{host}" if port == _DEFAULT_HTTPS_PORT else f"https://{host}:{port}"


def entry_origin(entry: dict) -> str | None:
    """Origin of the page a login is typed into: ``login_url``, else ``origin``."""
    for key in _LOGIN_URL_KEYS:
        raw = entry.get(key)
        if isinstance(raw, str) and raw:
            return canonical_origin(raw)
    return None


def entry_label(entry: dict) -> str | None:
    """The login's label (``None`` when it has none)."""
    raw = entry.get("label")
    return raw if isinstance(raw, str) and raw else None


def _validate_entry(entry: object) -> str | None:
    if not isinstance(entry, dict):
        raise ValueError("every auth_config list entry must be an object")
    if entry.get("label") is not None and not (
        isinstance(entry["label"], str) and _LABEL_RE.fullmatch(entry["label"])
    ):
        raise ValueError("a login's label must be 1-40 lowercase letters, digits, '-' or '_'")
    origins: set[str | None] = set()
    for key in _LOGIN_URL_KEYS:
        if key in entry:
            strict_login_url(entry[key], key)
            origins.add(canonical_origin(entry[key]))
    if entry.get(SIGNS_INTO) is not None:
        strict_login_url(entry[SIGNS_INTO], SIGNS_INTO)
    if len(origins) > 1:
        raise ValueError("a login's origin and login_url must name the same origin")
    return next(iter(origins), None)


def validate_auth_block(block: object) -> None:
    """Reject a malformed login block (raises ``ValueError`` for a pydantic door).

    Every ``login_url`` / ``origin`` must pass :func:`strict_login_url`, and the two
    must agree. A list must hold logins that each resolve an origin, with no origin
    twice unless told apart by a distinct ``label`` — the recorder picks the credential
    by ``(origin, label)``, so a duplicate pair is ambiguous. An origin holding more
    than one login needs EVERY one of them labelled (an unlabelled login beside a
    labelled one on the same origin could not be selected by anyone).
    """
    if block is None:
        return
    if isinstance(block, dict):
        _validate_entry(block)
        return
    if not isinstance(block, list):
        raise ValueError("auth_config must be an object or a list of objects")
    if not block or len(block) > MAX_AUTH_ENTRIES:
        raise ValueError(f"auth_config list must hold 1..{MAX_AUTH_ENTRIES} logins")
    _validate_login_keys(block)


def _validate_login_keys(block: list) -> None:
    """Every login of a list is keyed ``(origin, label)``; no key twice."""
    seen: set[tuple[str, str | None]] = set()
    labels_by_origin: dict[str, list[str | None]] = {}
    for entry in block:
        origin = _validate_entry(entry)
        if origin is None:
            raise ValueError("every auth_config list entry needs an https origin or login_url")
        label = entry_label(entry)
        if (origin, label) in seen:
            if label is None:
                raise ValueError(
                    f"auth_config lists origin {origin} twice — give each login on it a label"
                )
            raise ValueError(f"auth_config lists origin {origin} with label {label!r} twice")
        seen.add((origin, label))
        labels_by_origin.setdefault(origin, []).append(label)
    for origin, labels in labels_by_origin.items():
        if len(labels) > 1 and None in labels:
            raise ValueError(
                f"auth_config lists origin {origin} more than once: every login on it needs a label"
            )


def _validate_login_context(doc: dict) -> None:
    """One dict that carries a login: its blocks, its bound URLs, its binding."""
    for key in AUTH_BLOCK_KEYS:
        validate_auth_block(doc.get(key) or None)
    for key in _BOUND_URL_KEYS:
        if key in doc and doc[key] is not None:
            strict_login_url(doc[key], key)
    originless = any(
        entry_origin(e) is None for k in AUTH_BLOCK_KEYS for e in auth_entries(doc.get(k))
    )
    if originless and context_origin(doc) is None:
        raise ValueError(
            "a login without its own login_url must sit beside exactly one target "
            "origin (target_url / source_url / url / ref)"
        )


def validate_brief_logins(doc: object) -> None:
    """Validate every login block in a brief AND every URL it is bound to.

    A dict holding a non-empty ``auth_config`` / ``auth_info`` is a login context:
    its blocks pass :func:`validate_auth_block`, and its own ``target_url`` /
    ``source_url`` / ``url`` / ``ref`` — where FTR or the scraper sends that login
    when no ``login_url`` is given — must pass :func:`strict_login_url` too."""
    if isinstance(doc, list):
        for item in doc:
            validate_brief_logins(item)
        return
    if not isinstance(doc, dict):
        return
    if any(doc.get(k) for k in AUTH_BLOCK_KEYS):
        _validate_login_context(doc)
    for key, value in doc.items():
        if key not in AUTH_BLOCK_KEYS:
            validate_brief_logins(value)


def _walk_blocks(doc: object) -> Iterator[dict]:
    """Yield every login dict under an auth block anywhere in ``doc`` (mutable views)."""
    if isinstance(doc, dict):
        for key, value in doc.items():
            if key in AUTH_BLOCK_KEYS:
                yield from auth_entries(value)
            else:
                yield from _walk_blocks(value)
    elif isinstance(doc, list):
        for item in doc:
            yield from _walk_blocks(item)


def _is_plaintext(value: object) -> bool:
    """A secret value still readable: any non-empty value that is not sealed/REDACTED."""
    return value not in (None, "", [], {}) and not _is_sealed(value) and not _is_redacted(value)


def has_plaintext_secret(doc: object) -> bool:
    """True when ``doc`` holds at least one un-sealed, non-empty secret."""
    return any(_is_plaintext(entry.get(f)) for entry in _walk_blocks(doc) for f in SECRET_FIELDS)


def seal_auth_secrets(doc: dict) -> dict:
    """Seal every plaintext secret under an auth block, IN PLACE, and return ``doc``.

    Strings are sealed as-is; any other non-empty value (a ``cookies`` list) is sealed
    as JSON and marked :data:`SEALED_JSON`. Idempotent: a sealed value is left alone.
    A leftover :data:`REDACTED` (a client echoed it with nothing stored to restore) is
    dropped — it is not a secret. Raises :class:`AuthSecretError` when a plaintext
    secret is present and no key is configured: fail closed, never persist it readable.
    """
    fernet: MultiFernet | None = None
    for entry in _walk_blocks(doc):
        for field in SECRET_FIELDS & entry.keys():
            value = entry[field]
            if _is_redacted(value):
                del entry[field]
            elif _is_plaintext(value):
                if fernet is None:
                    fernet = _fernet()
                    if fernet is None:
                        raise AuthSecretError(
                            "login secrets cannot be stored: AUTH_CONFIG_SECRET_KEYS is not "
                            "configured on the orchestrator"
                        )
                if isinstance(value, str):
                    entry[field] = {SEALED: fernet.encrypt(value.encode()).decode()}
                else:
                    blob = json.dumps(value, sort_keys=True).encode()
                    entry[field] = {SEALED: fernet.encrypt(blob).decode(), SEALED_JSON: True}
    return doc


def redact_auth_secrets[T](doc: T) -> T:
    """A deep copy of ``doc`` with every non-empty secret under an auth block → REDACTED."""
    out = copy.deepcopy(doc)
    for entry in _walk_blocks(out):
        for field in SECRET_FIELDS & entry.keys():
            if entry[field] not in (None, "", [], {}):
                entry[field] = dict(REDACTED)
    return out


def restore_redacted(new: dict, old: dict | None) -> dict:
    """PATCH round-trip: put a STORED secret back wherever ``new`` echoes REDACTED.

    A secret is kept ONLY while the login it belongs to is unchanged — its
    :func:`_login_binding`: the login itself minus its secrets (``login_url``,
    ``origin``, ``username``, ``type``, ``label``, ``signs_into``, the selectors that
    decide which field it is typed into), the ONE site its context dict binds it to
    (:func:`context_origin`) plus each of that dict's ``target_url`` / ``source_url``
    / ``url`` / ``ref`` origins, and the context keys that pick the code or the login
    a service receives (``package_url``, ``login_label``). Change any of those and
    every redacted secret of that login is dropped, so it must be re-entered; the
    Brief page says so.

    Editing anything else (the title, the shotlist, a narration line, a
    ``site_capture`` path) keeps the password. That is safe because nothing outside
    the binding decides where a secret goes: dispatch releases a login only to its
    bound origin (:func:`auth_for_origin`, and ``site_capture`` only when EVERY page
    of a recording stays on it), and the password is typed only into ``login_url``.
    The old whole-brief rule dropped the password on ANY edit, so a tenant who
    renamed a shot silently lost the sign-in (George, 2026-10-28: never again).

    Logins are paired by JSON PATH, not walk order, so a reordered login list pairs
    each entry with whatever now sits at its index — a different login there is a
    changed binding, and drops. Mutates and returns ``new``.
    """
    stored = {path: (entry, ctx) for path, entry, ctx in _walk_blocks_ctx(old or {})}
    for path, new_entry, new_ctx in _walk_blocks_ctx(new):
        old_entry, old_ctx = stored.get(path, ({}, {}))
        same_login = bool(old_entry) and _login_binding(new_entry, new_ctx) == _login_binding(
            old_entry, old_ctx
        )
        for field in SECRET_FIELDS & new_entry.keys():
            if _is_redacted(new_entry[field]):
                if same_login and field in old_entry:
                    new_entry[field] = copy.deepcopy(old_entry[field])
                else:
                    del new_entry[field]
        if _has_secret(new_entry):
            new_entry.pop(NEEDS_REENTRY, None)  # a secret is saved: nothing to re-enter
        elif not same_login and _has_secret(old_entry):
            new_entry[NEEDS_REENTRY] = True  # the old secret was for the old login
        elif same_login and old_entry.get(NEEDS_REENTRY):
            new_entry[NEEDS_REENTRY] = True  # still waiting for the re-entry
        else:
            new_entry.pop(NEEDS_REENTRY, None)
    return new


def _has_secret(entry: dict) -> bool:
    return any(entry.get(f) not in (None, "", [], {}) for f in SECRET_FIELDS & entry.keys())


#: Keys of a login's enclosing dict that pick the CODE the unsealed login is handed
#: to, or WHICH login a service receives — so they are part of the binding, compared
#: verbatim. ``package_url``: demo_video_record sends the WHOLE unsealed
#: ``auth_config`` to that (tenant-uploaded) package, so swapping it for one's own
#: package and echoing REDACTED would capture the password. ``login_label``: which
#: login demo_video_gen hands bdd (:func:`auth_for_origin` ``label=``).
_BOUND_CODE_KEYS = ("package_url", "login_label")


def _login_binding(entry: dict, context: dict) -> tuple[str, str | None, str]:
    """What a stored secret is approved for: the login minus secrets, its site, and
    every context key that picks a destination or code receiving it.

    Each URL key's origin is bound ON ITS OWN, not only the collapsed
    :func:`context_origin`: that is None whenever the keys disagree, so two briefs
    ambiguous in DIFFERENT ways (``target_url`` swapped to another host beside an
    unchanged ``source_url``) would otherwise compare equal — and FTR records
    ``target_url``. A value that is not a strict URL is bound verbatim."""
    shape = {k: v for k, v in entry.items() if k not in SECRET_FIELDS and k != NEEDS_REENTRY}
    ctx = context if isinstance(context, dict) else {}
    targets = {k: canonical_origin(ctx[k]) or ctx[k] for k in _BOUND_URL_KEYS if ctx.get(k)}
    targets.update({k: ctx[k] for k in _BOUND_CODE_KEYS if ctx.get(k)})
    return (
        json.dumps(shape, sort_keys=True, default=str),
        context_origin(context),
        json.dumps(targets, sort_keys=True, default=str),
    )


def login_needs_reentry(brief: dict) -> list[str]:
    """JSON paths of logins flagged :data:`NEEDS_REENTRY` (saved secret dropped).

    The flag itself is what the client sees: it is an ordinary, non-secret field of
    the login in every brief the API returns, so the Brief page can say "Login
    address changed — re-enter the password" after a reload too. This helper only
    names the paths, for the PATCH log line."""
    return [
        ".".join(str(p) for p in path)
        for path, entry, _ in _walk_blocks_ctx(brief)
        if entry.get(NEEDS_REENTRY) is True
    ]


def _walk_blocks_ctx(
    doc: object, path: tuple = (), parent: dict | None = None
) -> Iterator[tuple[tuple, dict, dict]]:
    """Every login dict under an auth block: ``(json_path, entry, enclosing_dict)``."""
    if isinstance(doc, dict):
        for key, value in doc.items():
            if key in AUTH_BLOCK_KEYS:
                if isinstance(value, dict):
                    yield (*path, key), value, doc
                elif isinstance(value, list):
                    for i, entry in enumerate(value):
                        if isinstance(entry, dict):
                            yield (*path, key, i), entry, doc
            else:
                yield from _walk_blocks_ctx(value, (*path, key), doc)
    elif isinstance(doc, list):
        for i, item in enumerate(doc):
            yield from _walk_blocks_ctx(item, (*path, i), parent)


def unseal_auth(block: object) -> dict | list[dict] | None:
    """A plaintext copy of an auth block for dispatch — the ONLY place a secret is opened.

    Keeps the input's shape (dict stays dict, list stays list). Raises
    :class:`AuthSecretError` for a sealed value that cannot be opened (no key, or a key
    it was not sealed with) — a login that silently lost its password would record a
    login screen and pass for footage.
    """
    if not isinstance(block, (dict, list)) or not block:
        return None
    out = copy.deepcopy(block)
    fernet: MultiFernet | None = None
    for entry in auth_entries(out):
        for field in SECRET_FIELDS & entry.keys():
            value = entry[field]
            if _is_redacted(value):
                del entry[field]
            elif _is_sealed(value):
                if fernet is None:
                    fernet = _fernet()
                    if fernet is None:
                        raise AuthSecretError(
                            "a sealed login secret cannot be opened: AUTH_CONFIG_SECRET_KEYS "
                            "is not configured on the orchestrator"
                        )
                try:
                    opened = fernet.decrypt(value[SEALED].encode()).decode()
                    entry[field] = json.loads(opened) if value.get(SEALED_JSON) else opened
                except InvalidToken as exc:
                    raise AuthSecretError(
                        f"sealed login {field} does not open with the configured keys"
                    ) from exc
    return out


def context_origin(context: object) -> str | None:
    """The ONE origin a login-carrying dict is bound to: the shared origin of its own
    ``target_url`` / ``source_url`` / ``url`` / ``ref``, else None (none, or several
    distinct ones — ambiguous, so no login is bound)."""
    if not isinstance(context, dict):
        return None
    origins = {canonical_origin(context[k]) for k in _BOUND_URL_KEYS if context.get(k)}
    return origins.pop() if len(origins) == 1 else None


def login_site(entry: dict) -> str | None:
    """The origin a login signs INTO: its declared ``signs_into``, else its own.

    Its own origin is the sign-in page's (``login_url`` / ``origin``). ``None`` when
    the login names neither, i.e. it is bound to its context's site instead."""
    declared = entry.get(SIGNS_INTO)
    if isinstance(declared, str) and declared:
        return canonical_origin(declared)
    return entry_origin(entry)


def _dispatchable(entry: dict) -> dict:
    """The login as a signing-in service receives it: no orchestrator-only keys."""
    return {k: v for k, v in entry.items() if k not in _ORCHESTRATOR_ONLY_KEYS}


def dispatchable_auth(block: dict | list[dict] | None) -> dict | list[dict] | None:
    """A WHOLE auth block as a signing-in service receives it: each login through
    :func:`_dispatchable`, shape kept (dict stays dict, list stays list).

    For a caller that forwards every login at once (demo_video_record → FTR) rather
    than one picked by :func:`auth_for_origin`, which already strips them."""
    if isinstance(block, dict):
        return _dispatchable(block)
    if isinstance(block, list):
        return [_dispatchable(e) for e in auth_entries(block)]
    return None


def auth_for_origin(
    block: object, url: str | None, context: object = None, label: str | None = None
) -> dict | None:
    """The login of ``block`` whose origin is EXACTLY ``url``'s, else None.

    Never "the first login". A login's origin is its own ``login_url``/``origin``;
    a login without one is bound to its context dict's single bound origin
    (:func:`context_origin`) — the target it was entered next to — and to nothing
    if that is absent or ambiguous. A login that DECLARES ``signs_into`` (single sign-on:
    the sign-in page is on an identity provider) is for that site instead
    (:func:`login_site`); nothing is ever inferred from origins differing.

    Several logins may share the origin (told apart by ``label``). With ``label`` the
    login carrying it is returned; without one, the origin's only login — and None
    when more than one is left, because picking either would be a guess."""
    target = canonical_origin(url)
    if target is None:
        return None
    fallback = context_origin(context)
    matches = [e for e in auth_entries(block) if (login_site(e) or fallback) == target]
    if label is not None:
        found = next((e for e in matches if entry_label(e) == label), None)
        return _dispatchable(found) if found is not None else None
    return _dispatchable(matches[0]) if len(matches) == 1 else None


def pages_for_login(pages: object, base_url: object, login: dict) -> tuple[list[str], int]:
    """The ``pages`` that may travel in ONE request beside the unsealed ``login``.

    A page list sent with a login is a list of places that login may be used, so
    it is held to the login's bound origins: ``base_url``'s (the site
    :func:`auth_for_origin` picked the login for) and the login's own sign-in
    page (``login_url`` / ``origin``). ``target_pages`` is NOT part of
    :func:`_login_binding`, so without this a tenant could add their own origin
    to it, keep the saved password, and have the signing-in service carry it
    there (bdd forwards target pages + login to webscraper in one batch).

    Each entry is resolved against ``base_url`` exactly as bdd does (absolute
    ``http(s)://`` kept, anything else ``urljoin``-ed — so ``//evil.example/x``
    resolves off-site) and its origin goes through the strict
    :func:`canonical_origin`; a non-string, an unparseable result or an off-site
    origin is DROPPED (fail closed). Returns ``(kept entries as given, dropped)``."""
    base = canonical_origin(base_url)
    allowed = {o for o in (base, entry_origin(login)) if o is not None}
    if not isinstance(pages, list):
        return [], 0
    kept = [
        page
        for page in pages
        if isinstance(page, str)
        and page
        and base is not None
        and canonical_origin(
            page if page.startswith(("http://", "https://")) else urljoin(str(base_url), page)
        )
        in allowed
    ]
    return kept, len(pages) - len(kept)
