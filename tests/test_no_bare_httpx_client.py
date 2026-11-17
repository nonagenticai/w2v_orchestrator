"""CI-enforced guard: no UNDOCUMENTED bare ``httpx`` client under ``src/``.

Why this exists
---------------
Outbound service-to-service calls must forward the identity triple
``X-Tenant-Id`` / ``X-License-Id`` / ``X-Subject-Id`` (plus ``X-Correlation-Id``
/ ``X-Stage``). Where the convention is used it works and needs no discipline:
``w2v_common.auth.identity_context.get_identity_headers`` is auto-installed as
an httpx ``event_hooks={"request": [...]}`` handler by
``w2v_common.clients.base.BaseServiceClient``, so **every** subclass forwards
correctly without a single line of per-call-site code.

The droppers are therefore never the clients — they are the raw
``httpx.AsyncClient(...)`` / ``httpx.Client(...)`` constructions that bypass the
base class entirely. That is a *class* of defect, not an instance: each one is
individually plausible at review time ("it's just a fetch"), and nothing tells
you the headers silently vanished. Hence a static gate rather than a round of
one-off fixes.

How it decides
--------------
An AST walk over every ``src/**/*.py`` finds each construction of ``httpx.Client``
/ ``httpx.AsyncClient`` (attribute form, and the ``from httpx import AsyncClient``
form). Each hit is keyed by ``(path, enclosing-function-qualname)`` — stable
across line-number churn, unlike a ``file:line`` allowlist. Any key not in
:data:`ALLOWED` fails the test.

An exemption is NOT just a line in a list. Every entry carries:

* a **category**, and
* a **written reason**.

and two of the three categories are *structurally substantiated* — the guard
re-checks the claim against the source of the exempted function, so an exemption
cannot quietly outlive the thing that justified it:

* ``SSRF_SINK`` — the function must still CALL one of the guarded fetch helpers
  (:data:`_SSRF_MARKERS`), checked against AST identifiers rather than source
  text. If someone drops ``safe_get`` for a raw ``client.get``, the exemption
  stops applying and this test reds — and it stays red even though the
  function's docstring still talks about ``safe_get`` at length.
* ``IDENTITY_FORWARDED`` — the function must still reference
  ``get_identity_headers`` / ``forward_headers``. If the manual forwarding is
  deleted, the exemption stops applying and this test reds.
* ``NON_W2V_PEER`` — reason only. These call a service outside the w2v S2S
  convention (Keycloak, licenses, billing) where the triple is not the auth
  channel; there is nothing mechanical left to assert, so the reason must stand
  on its own and is held to a minimum length.

And a **stale entry that matches nothing fails the test** (non-vacuity): when a
call site is deleted or routed through ``BaseServiceClient``, its exemption must
go with it. An allowlist that can accumulate dead entries is how a guard rots
into decoration.

And one guard on the guard: :func:`test_ssrf_markers_actually_reach_a_chokepoint`
recomputes, from the AST, that every declared marker transitively reaches one of
the ``w2v_common.net`` chokepoints in :data:`_SSRF_CHOKEPOINTS`. Adding a
plausible-looking name that fetches raw is the obvious way to defeat the
substantiation check; it is now a red test.

Why so many entries and zero "fixes"
------------------------------------
Every current construction falls into one of the three categories above; none is
a plain S2S call that could be routed through ``BaseServiceClient``. That is the
honest finding, not a capitulation — and for the SSRF sinks it is the *only*
correct answer: those fetch an arbitrary, user- or LLM-supplied URL, so
attaching the orchestrator's license token would hand a live credential to a
third-party host. Forwarding identity there would be a credential leak, not a
fix. The value of this file is the floor it sets for the NEXT one.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import NamedTuple

_SRC = Path(__file__).resolve().parent.parent / "src"

# Names whose presence in an exempted function proves the fetch still goes
# through the SSRF chokepoint: ``w2v_common.net`` itself, or one of the three
# named wrappers around it in this repo — the two media sinks in
# ``src/services/visual_media.py`` and ken_burns' ``_fetch_own_asset``, which is
# where THAT module's ``safe_get`` call (with its storage allow_hosts) lives.
# Keep this list to functions that genuinely terminate in ``safe_get``; adding a
# name that does not is how the substantiation check gets quietly defeated.
_SSRF_MARKERS = (
    "safe_get",
    "safe_request",
    "safe_fetch",
    "validate_public_url",
    "fetch_image_bytes",
    "fetch_video_bytes",
    "_fetch_own_asset",
)

# The TERMINAL screens every marker above must transitively reach. All four are
# real exports of ``w2v_common.net`` -- the estate's SSRF chokepoints, which
# resolve the host once, pin the validated IP, re-validate every redirect hop and
# abort the transfer at ``max_fetch_bytes``. They live OUTSIDE ``src/``, so the
# AST cannot prove anything about their bodies: they are the axioms of the
# reachability closure, and the markers above are what actually gets proved.
# Seed a name here ONLY if it performs the screen itself.
_SSRF_CHOKEPOINTS = (
    "safe_get",
    "safe_request",
    "safe_fetch",
    "validate_public_url",
)

# Names whose presence proves the function still stamps the identity triple by
# hand instead of inheriting the BaseServiceClient event hook.
_IDENTITY_MARKERS = ("get_identity_headers", "forward_headers")

SSRF_SINK = "SSRF_SINK"
IDENTITY_FORWARDED = "IDENTITY_FORWARDED"
NON_W2V_PEER = "NON_W2V_PEER"

_MARKERS = {SSRF_SINK: _SSRF_MARKERS, IDENTITY_FORWARDED: _IDENTITY_MARKERS, NON_W2V_PEER: ()}

# A reason short enough to be a shrug is not a reason.
_MIN_REASON_CHARS = 60


class Exemption(NamedTuple):
    category: str
    reason: str


class Construction(NamedTuple):
    path: str  # posix, relative to the repo root
    qualname: str  # enclosing def/class chain, or "<module>"
    lineno: int
    # Identifiers actually REFERENCED in the enclosing function — names and
    # attribute accesses, collected from the AST. Deliberately not the source
    # text: the first cut of this guard grepped the text and passed a mutation
    # that deleted `safe_get` from site_capture, because the function's own
    # DOCSTRING says "safe_get". A guard that reads prose proves nothing about
    # code (same defect class as the Any-ratchet counting comments).
    identifiers: frozenset[str]


# ── The allowlist ────────────────────────────────────────────────────────────
# Key: (path relative to repo root, enclosing function qualname).
# Adding an entry here is a DECISION, not a formality — read the module docstring
# before you add one. The default answer for a new S2S call is
# "subclass BaseServiceClient", which needs no entry at all.
ALLOWED: dict[tuple[str, str], Exemption] = {
    ("src/services/signup/telegram.py", "_post"): Exemption(
        NON_W2V_PEER,
        "Early-access signup notice: Bot API sendMessage to api.telegram.org (a fixed, "
        "configured third-party host). The bot token is in the URL; the w2v identity "
        "triple must never be sent to Telegram.",
    ),
    # --- SSRF sinks: arbitrary user/LLM-supplied URLs. Forwarding the license
    # token here would post a live credential to a third-party host. ---
    ("src/services/providers/tester.py", "HttpProviderTester._run"): Exemption(
        SSRF_SINK,
        "Live test of a TENANT-supplied provider base_url (P3 provider connections). Every "
        "request goes through w2v_common.net.safe_request (resolve once, pin the validated "
        "IP, re-validate every redirect hop, response capped); the target is the tenant's "
        "own LLM provider, never a w2v peer, so identity headers must not be attached.",
    ),
    ("src/services/providers/tester.py", "HttpProviderTester.list_models"): Exemption(
        SSRF_SINK,
        "Model discovery (GET {base}/models) against a TENANT-supplied provider base_url, "
        "the sibling of HttpProviderTester._run. The single request goes through "
        "w2v_common.net.safe_request (resolve once, pin the validated IP, no redirects, "
        "response capped); the target is the tenant's own provider, never a w2v peer, so "
        "identity headers must not be attached.",
    ),
    ("src/services/visual_media.py", "fetch_image_bytes"): Exemption(
        SSRF_SINK,
        "General image sink for tenant-supplied, KAG-discovered and LLM-proposed URLs. "
        "The client is a transport for w2v_common.net.safe_get (resolve once, pin the "
        "validated IP, re-validate every redirect hop); the target is by definition not "
        "a w2v peer, so attaching X-Tenant-Id: Bearer <license> would leak the credential "
        "to an arbitrary host. Callers may pass their own client, which is why this is a "
        "`client or httpx.AsyncClient(...)` fallback rather than a constructor.",
    ),
    ("src/services/visual_media.py", "fetch_video_bytes"): Exemption(
        SSRF_SINK,
        "Same contract as fetch_image_bytes for video: streamed through safe_get with a "
        "byte cap, fetching an external clip URL. No allow_hosts at all on this one, and "
        "no identity headers — the destination is third-party by construction.",
    ),
    ("src/services/stages/shared/ken_burns_local.py", "crop_cover_stills_to_frame"): Exemption(
        SSRF_SINK,
        "Fetches shot['image_url'] to crop a cover-fit still to the frame aspect before "
        "Ken Burns. Same URL, same reasoning and the same guard as apply_ken_burns below: "
        "the fetch goes through ken_burns._fetch_own_asset (safe_get, allow_hosts scoped to "
        "the storage endpoints); the client built here is only its transport. No identity "
        "headers.",
    ),
    ("src/services/stages/shared/ken_burns_local.py", "rescue_frozen_stills"): Exemption(
        SSRF_SINK,
        "Fetches shot['image_url'] of a still whose Ken Burns recording failed, to animate "
        "it locally. Through ken_burns._fetch_own_asset exactly like apply_ken_burns below; "
        "the client built here is only its transport. No identity headers.",
    ),
    ("src/services/stages/shared/ken_burns.py", "apply_ken_burns"): Exemption(
        SSRF_SINK,
        "Fetches shot['image_url'] to animate a still. That URL is EXPECTED to be our own "
        "presigned MinIO artifact, but it rides on the same shot dict that carries "
        "operator-pinned and real-media URLs, so it is never fetched blind — it goes "
        "through _fetch_own_asset, which calls safe_get with allow_hosts scoped to the "
        "configured storage endpoints (the client built here is merely the transport "
        "passed into it). "
        "The module header explains why it does not reuse fetch_image_bytes; do not "
        "'unify' these without reading it. No identity headers: an attacker-steered "
        "image_url would otherwise exfiltrate the token.",
    ),
    (
        "src/services/stages/generic/site_capture.py",
        "SiteCaptureStage._scrape_page_text",
    ): Exemption(
        SSRF_SINK,
        "Scrapes the user-supplied source_url for script grounding — the most openly "
        "attacker-controlled fetch sink in the tree. Guarded by _safe_to_fetch plus "
        "safe_get with an explicit response-byte cap. Sends only a User-Agent; forwarding "
        "the identity triple would post the tenant's license token to whatever host the "
        "user typed into the brief.",
    ),
    ("src/api/routes/storyboard.py", "_resolve_panel_images"): Exemption(
        SSRF_SINK,
        "Storyboard PDF export: one shared client reused across up to _PDF_IMAGE_MAX_FETCHES "
        "panel frames, each fetched through visual_media.fetch_image_bytes. allow_hosts is "
        "granted per-fetch and ONLY when panel_object_ref proves the URL addresses this "
        "panel's own project/tenant prefix; media_ref is client-writable, so a blanket "
        "allowlist (or a forwarded credential) would let a tenant aim the export at any host.",
    ),
    # --- Identity forwarded by hand (verified: the function still references
    # forward_headers / get_identity_headers). ---
    ("src/api/routes/_proxy.py", "proxy_get"): Exemption(
        IDENTITY_FORWARDED,
        "Thin proxy to analysis_service / html_render. forward_headers() already emits "
        "get_identity_headers() plus the license Bearer and X-Effective-Tenant-Id, so the "
        "triple IS forwarded — it just is not inherited from BaseServiceClient. Kept raw "
        "because these proxies must run follow_redirects=False and relay upstream status "
        "codes verbatim to the SPA.",
    ),
    ("src/api/routes/_proxy.py", "proxy_post"): Exemption(
        IDENTITY_FORWARDED,
        "Same forward_headers() path as proxy_get, for JSON POSTs. The X-Effective-Tenant-Id "
        "side-channel it stamps is what keeps a browser-originated job attributed to the "
        "CUSTOMER rather than to the orchestrator's own service-account tenant.",
    ),
    ("src/api/routes/_proxy.py", "proxy_stream"): Exemption(
        IDENTITY_FORWARDED,
        "Same forward_headers() path, buffering a binary artifact (PDF/HTML) before "
        "re-streaming it to the caller. Needs direct control of the response object, which "
        "the base client's JSON-shaped helpers do not expose.",
    ),
    ("src/api/routes/_proxy.py", "proxy_request"): Exemption(
        IDENTITY_FORWARDED,
        "Same forward_headers() path, for JSON or multipart (merqi recon's ASR upload and "
        "video-analyzer jobs). Kept raw for follow_redirects=False and so upstream failures "
        "are MAPPED (map_upstream_error) rather than echoed.",
    ),
    # --- Peers outside the w2v S2S convention: the triple is not their auth
    # channel, and sending it would be at best noise, at worst misread. ---
    ("src/clients/service_auth.py", "ServiceAuthenticator._refresh"): Exemption(
        NON_W2V_PEER,
        "Keycloak's OAuth2 token endpoint. This is the call that MINTS the credential; it "
        "authenticates with client-credentials form data and there is no w2v identity to "
        "forward to an IdP. A license token on this request would be meaningless.",
    ),
    ("src/clients/keycloak_admin.py", "KeycloakAdminClient._request"): Exemption(
        NON_W2V_PEER,
        "Keycloak Admin REST API (the tenancy backfill's reads + the early-access "
        "approve's own identity: create user / action email), authorized by the "
        "w2v-backend client-credentials Bearer. Keycloak is not a w2v mesh peer and does not "
        "read X-Tenant-Id/X-License-Id; the realm is encoded in the URL. Injectable client "
        "kwarg is the test seam.",
    ),
    ("src/clients/identity_inviter.py", "IdentityInviter.invite"): Exemption(
        NON_W2V_PEER,
        "provisioning's identity invite API, authorized ONLY by this service's Keycloak "
        "client-credentials Bearer (realm role identity-inviter + azp allow-list). It is not "
        "a w2v peer and the invite is not on behalf of any tenant: forwarding the caller's "
        "X-Tenant-Id licence token would attach an unrelated credential. Injectable client "
        "kwarg is the test seam.",
    ),
    ("src/clients/licenses_client.py", "LicensesClient._request"): Exemption(
        NON_W2V_PEER,
        "licenses-service in the secure-apps realm. It authorizes from the Keycloak SA "
        "token's realm_access.roles and requires a BARE-UUID X-Tenant-Id — not the "
        "'X-Tenant-Id: Bearer <license>' w2v form. The base client's event hook sets the "
        "Bearer form, so inheriting it would break authorization here.",
    ),
    ("src/clients/licenses_client.py", "LicensesClient.introspect"): Exemption(
        NON_W2V_PEER,
        "licenses' PUBLIC POST /license/introspect: the token in the BODY is the only "
        "credential. Forwarding the caller's identity headers (the base client's event "
        "hook) would attach an unrelated licence token / tenant to a call that must carry "
        "exactly one — the stored tenant token being verified. No headers on purpose.",
    ),
    ("src/clients/ocr_client.py", "OcrClient.ocr_pdf"): Exemption(
        NON_W2V_PEER,
        "The OCR provider (mlx_api today, a commercial Mistral-OCR-shaped vendor later) on "
        "the provider_contracts /v1/ocr wire. Its ONLY credential is Authorization: Bearer "
        "<W2V_PROVIDER_OCR_API_KEY>; it reads no X-Tenant-Id/X-License-Id, and forwarding "
        "the tenant's licence to a third-party vendor would leak a credential. "
        "Injectable transport kwarg is the test seam.",
    ),
    ("src/clients/billing_client.py", "BillingServiceClient.costs_by_correlation"): Exemption(
        NON_W2V_PEER,
        "billing-service is a cross-namespace platform service (fr parity), not a w2v "
        "in-mesh peer, and is not in w2v_common discovery. Its docstring states the "
        "requirement explicitly: Authorization: Bearer <SA license token> plus a PLAIN "
        "X-Tenant-Id UUID, because billing's require_tenant would misread the w2v Bearer "
        "form as the tenant string.",
    ),
}


def _walk_scoped(
    tree: ast.Module,
) -> tuple[list[tuple[ast.AST, str]], dict[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Return (node, enclosing-qualname) pairs plus a qualname → function-node map.

    A nested ``def`` extends the qualname, so the client built inside ken_burns'
    inner ``_gen`` still reports its outer function — which is the unit a
    reviewer reasons about. The OUTERMOST function of a chain is recorded in the
    map, so the substantiation check sees the whole body (nested helpers
    included) rather than one closure.
    """
    pairs: list[tuple[ast.AST, str]] = []
    scopes: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}

    def visit(node: ast.AST, prefix: str, in_function: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                name = f"{prefix}.{child.name}" if prefix else child.name
                # Nested defs keep reporting (and being checked against) their
                # outermost enclosing function.
                reported = prefix if in_function else name
                if not in_function:
                    scopes[name] = child
                pairs.append((child, reported))
                visit(child, reported, True)
            elif isinstance(child, ast.ClassDef):
                name = f"{prefix}.{child.name}" if prefix else child.name
                pairs.append((child, name))
                visit(child, name, in_function)
            else:
                pairs.append((child, prefix))
                visit(child, prefix, in_function)

    visit(tree, "", False)
    return pairs, scopes


def _is_httpx_client_call(node: ast.Call, imported_names: set[str]) -> bool:
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr in ("Client", "AsyncClient"):
        return isinstance(func.value, ast.Name) and func.value.id == "httpx"
    return isinstance(func, ast.Name) and func.id in imported_names


def _referenced_identifiers(node: ast.AST) -> frozenset[str]:
    """Every name / attribute actually referenced inside ``node``.

    AST-based on purpose — see :class:`Construction`. Docstrings and comments are
    not part of the tree's Name/Attribute nodes, so a marker can only match code.
    """
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            names.add(child.attr)
    return frozenset(names)


def _httpx_imported_names(tree: ast.Module) -> set[str]:
    """Names bound by ``from httpx import Client/AsyncClient [as x]``."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "httpx":
            names.update(
                a.asname or a.name for a in node.names if a.name in ("Client", "AsyncClient")
            )
    return names


def _function_identifiers(root: Path = _SRC) -> dict[tuple[str, str], frozenset[str]]:
    """(path, qualname) -> identifiers referenced, for EVERY top-level function/method."""
    repo = root.parent
    table: dict[tuple[str, str], frozenset[str]] = {}
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(repo).as_posix()
        _, scopes = _walk_scoped(tree)
        for qualname, node in scopes.items():
            table[(rel, qualname)] = _referenced_identifiers(node)
    return table


def find_constructions(root: Path = _SRC) -> list[Construction]:
    """Every bare ``httpx`` client construction under ``root``, keyed for the allowlist."""
    repo = root.parent
    found: list[Construction] = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "httpx" not in text:
            continue
        tree = ast.parse(text, filename=str(path))
        imported = _httpx_imported_names(tree)
        rel = path.relative_to(repo).as_posix()
        pairs, scopes = _walk_scoped(tree)
        for node, qualname in pairs:
            if not isinstance(node, ast.Call) or not _is_httpx_client_call(node, imported):
                continue
            owner = scopes.get(qualname)
            names = _referenced_identifiers(owner) if owner else frozenset()
            found.append(Construction(rel, qualname or "<module>", node.lineno, names))
    return found


# ── The gate ─────────────────────────────────────────────────────────────────


def test_no_undocumented_bare_httpx_client() -> None:
    """Every bare httpx client under src/ must carry a written exemption."""
    offenders = [c for c in find_constructions() if (c.path, c.qualname) not in ALLOWED]
    assert not offenders, (
        "bare httpx client construction(s) with no exemption:\n"
        + "\n".join(f"  {c.path}:{c.lineno} in {c.qualname}()" for c in offenders)
        + "\n\nA raw httpx client bypasses BaseServiceClient's identity event hook, so the "
        "X-Tenant-Id / X-License-Id / X-Subject-Id triple is silently DROPPED on that call. "
        "Subclass w2v_common.clients.base.BaseServiceClient instead. If this is an "
        "SSRF-guarded fetch of an arbitrary URL (where forwarding a license token would "
        "leak a credential to a third party), add an entry to ALLOWED in "
        "tests/test_no_bare_httpx_client.py with a category and a written reason."
    )


def test_allowlist_has_no_stale_entries() -> None:
    """Non-vacuity: an exemption may not outlive the call site that excused it."""
    live = {(c.path, c.qualname) for c in find_constructions()}
    stale = sorted(key for key in ALLOWED if key not in live)
    assert not stale, (
        "ALLOWED entries match nothing in src/ — the call site was moved, renamed or "
        f"fixed, so the exemption must be deleted: {stale}"
    )


def test_every_exemption_states_a_category_and_a_reason() -> None:
    thin = {
        key: exemption
        for key, exemption in ALLOWED.items()
        if exemption.category not in _MARKERS or len(exemption.reason) < _MIN_REASON_CHARS
    }
    assert not thin, (
        "every exemption needs a known category and a reason of at least "
        f"{_MIN_REASON_CHARS} chars explaining why the identity triple must not be "
        f"forwarded here: {sorted(thin)}"
    )


def test_exemption_claims_are_still_true_in_the_source() -> None:
    """A structurally substantiated exemption must keep being substantiated.

    ``SSRF_SINK`` and ``IDENTITY_FORWARDED`` are claims about the code, so they are
    re-verified rather than trusted. Delete ``safe_get`` from a sink, or
    ``forward_headers`` from a proxy, and the exemption stops applying here — which
    is exactly the moment a reviewer needs to be told.
    """
    by_key = {(c.path, c.qualname): c for c in find_constructions()}
    unsubstantiated: list[str] = []
    for key, exemption in ALLOWED.items():
        markers = _MARKERS[exemption.category]
        if not markers:
            continue
        construction = by_key.get(key)
        if construction is None:  # covered by the staleness test
            continue
        if not any(marker in construction.identifiers for marker in markers):
            unsubstantiated.append(
                f"{key[0]}:{key[1]} claims {exemption.category} but references none of {markers}"
            )
    assert not unsubstantiated, "\n".join(unsubstantiated)


def test_ssrf_markers_actually_reach_a_chokepoint() -> None:
    """Guard the guard: a marker that does not reach a chokepoint is a backdoor.

    The substantiation check is only as good as :data:`_SSRF_MARKERS`. The cheap
    way to defeat it is to add a plausible-sounding helper that fetches raw, and
    nothing else in this file would notice. So recompute the reachability from
    the AST: seed with :data:`_SSRF_CHOKEPOINTS` and close over "function F
    references a name already known to reach one". Every declared marker must
    land in that set.
    """
    identifiers = _function_identifiers()
    by_name: dict[str, list[frozenset[str]]] = {}
    for (_path, qualname), names in identifiers.items():
        by_name.setdefault(qualname.rsplit(".", 1)[-1], []).append(names)

    reaching = set(_SSRF_CHOKEPOINTS)
    changed = True
    while changed:
        changed = False
        for name, bodies in by_name.items():
            if name in reaching:
                continue
            if any(ref in reaching for body in bodies for ref in body):
                reaching.add(name)
                changed = True

    unreachable = [m for m in _SSRF_MARKERS if m not in reaching]
    assert not unreachable, (
        f"declared SSRF marker(s) {unreachable} do not transitively reach any of "
        f"{_SSRF_CHOKEPOINTS} anywhere in src/ -- an exemption substantiated by one of "
        "these proves nothing. Remove the marker, or route the helper through a chokepoint."
    )


def test_guard_actually_finds_the_known_constructions() -> None:
    """The scanner must not be silently finding nothing (self-check).

    A guard whose finder is broken passes every other test in this file. Pin a
    floor and two known sites so a scanner regression reds instead of relaxing.
    """
    found = find_constructions()
    keys = {(c.path, c.qualname) for c in found}
    assert len(found) >= 13, f"scanner found only {len(found)} constructions; it is broken"
    assert ("src/services/visual_media.py", "fetch_image_bytes") in keys
    assert ("src/api/routes/_proxy.py", "proxy_get") in keys
