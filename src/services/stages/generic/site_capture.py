"""site_capture — record the live source site as background footage + scrape its copy.

Opt-in (like ``ai_image_gen``): the stage runs only when the brief carries a
``source_url``; otherwise it raises ``StageSkippedError`` and the run proceeds
without it. When active it:

  * records one clip per configured path via video_production_service's
    ``/api/record/url`` (optionally authenticating into the SPA with
    ``brief.auth_info``), re-publishes each clip into the orchestrator's own
    storage so the compiler can fetch it anonymously, and
  * best-effort scrapes the page text so ``promo_script`` can ground the
    commercial copy in the real product.

Failure semantics are soft *for failures the recorder actually answered*: a
recording or scrape error degrades to fewer/zero clips (the render falls back to
AI images / text cards) rather than failing the whole run — capturing the site is
an enhancement, not a precondition. "No ``source_url``" short-circuits as a clean
skip. Every degradation is recorded in ``notes`` so it is visible, not silent.

The one case that is NOT soft is a MEASUREMENT GAP on the per-screen path: if
video_production never answers (job timed out, was reaped, the service is
unreachable) we have learned nothing about the site, and returning a successful
stage holding ``screens: []`` reports a total capture outage as a clean render
decision. That path now raises :class:`SiteCaptureUnavailableError`. Whichever
way it goes, ``SiteCaptureOutput.capture_status`` names it explicitly so no
consumer has to infer it from an empty list.

The other case that is NOT soft is a CONTRACTED screen set. Soft-fail is right
because the ``source_url`` path is an enhancement — but an authored
``brief.shotlist`` is a deliverable: each ``kind: url`` shot contracts a screen
with ``screen.id == shot.line_id``. That path passes ``required_screen_ids`` and
a missing id raises :class:`SiteCaptureIncompleteError` instead of reporting a
``completed`` stage holding no deliverable.

Re-entry (pod death mid-stage): every recording is a PAID video_production job, so
the stage resumes instead of re-recording. Part and clip ids are DERIVED from
``(run_id, url)`` — never minted — and each part is tracked in the step ledger
(``submit:<part_id>`` → job id, ``result:<part_id>`` → the finished clip/screens).
See "Resuming a recording" below the constants.

Config block (``brief.site_capture``), all optional:

  site_capture:
    paths:              list        # relative paths to record (default: [""] = base URL).
                                    # an entry may be a string OR a mapping
                                    # {path, interactions:[{action,selector,hold_ms},...]}
                                    # to click through an interactive demo while recording.
    max_clips:          int         # hard cap on clips (default 4)
    viewport:           {width,height}
    scroll:             {enabled, speed_px_per_second, start_delay_seconds, end_delay_seconds}
    post_load_delay_ms: int         # settle after networkidle before capture
                                    # (default 3500; fixes blank-shell white screens)
    wait_for_selector:  str         # optional CSS selector to wait for before capture
    screens:            list        # per-screen capture mode: discrete demo screens
                                    # ({selector, capture_id|id, label?, settle_ms?})
    motion:             str         # per-screen mode only: "clips" also records a
                                    # short interaction clip (the click that reveals
                                    # each screen) alongside the still; default stills-only
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import zipfile
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urljoin, urlsplit, urlunsplit
from uuid import NAMESPACE_URL, uuid5, uuid7

import httpx
from w2v_common.net import safe_get, validate_public_url

from src.clients.video_production_client import VideoProductionClient
from src.models.failure_class import CUSTOMER_INPUT
from src.services.auth_secrets import (
    SECRET_FIELDS,
    auth_entries,
    auth_for_origin,
    canonical_origin,
    unseal_auth,
)
from src.services.stages.base import (
    StageError,
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithDegradations,
    StageSkippedError,
)
from src.services.stages.shared.degradation import (
    SITE_CAPTURE_CLIP_FAILED,
    SITE_CAPTURE_NO_FOOTAGE,
    SITE_CAPTURE_SCREENS_FAILED,
    SITE_PAGE_TEXT_MISSING,
    SITE_PAGE_TEXT_THIN,
    Degradation,
    degradation,
)
from src.services.stages.shared.motion_slides import quality_tier
from src.services.stages.shared.shotlist import ShotSpec, normalize_shotlist
from src.services.stages.shared.step_ledger import JsonValue, StepLedger, hash_inputs
from src.services.tts_storage import store_capture_clip, store_capture_image

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

log = logging.getLogger(__name__)

# One capture target: (resolved_url, relative_path, interaction_steps|None).
# Interactions present => this is the scripted demo walkthrough (the primary clip).
_Target = tuple[str, str, "list[dict[str, Any]] | None"]


def _shot_to_screen_entry(shot: ShotSpec, base_url: str) -> dict[str, Any]:
    """Map an authored URL shot → a per-screen capture entry.

    Consumed by :meth:`SiteCaptureStage._build_screen_steps`. ``capture_id`` is
    the shot's deterministic line id, so the produced ``screen.id == line_id`` and
    ``promo_visual_director`` binds the screenshot to its narration 1:1. ``hold_ms``
    becomes the settle so the viewer can read the screen.

    The full canonical directive set now flows to the recorder (recorder PR #54):
    - ``click`` => an **in-SPA** navigation: a selector clicked before the shot,
      no full reload (preserves SPA client-side routing).
    - ``shot.url`` **without** a ``click`` => a **full navigation** (``goto``) to
      that page, resolved against ``base_url`` so relative paths work. This is how
      a multi-page (multi-absolute-URL) shotlist is expressed. Omitted when the
      resolved URL is the base (already loaded) so the landing shot doesn't
      re-navigate.
    - ``scroll_to`` / ``wait_for`` => forwarded verbatim (selector or ``"bottom"``;
      content-ready selector).
    - ``viewport`` => always forwarded; the recorder de-dupes consecutive identical
      viewports, so emitting it every shot is what lets a later desktop shot reset
      after an earlier tablet/mobile one.
    """
    directives = shot.directives or {}
    entry: dict[str, Any] = {
        "capture_id": shot.line_id,
        "label": shot.narration,
        "settle_ms": int(directives.get("hold_ms") or _DEFAULT_POST_LOAD_DELAY_MS),
        "viewport": str(directives.get("viewport") or "desktop"),
    }
    click = directives.get("click")
    if click:
        entry["selector"] = str(click)
    elif shot.url:
        target = urljoin(base_url, shot.url.strip())
        if target and target != base_url:
            entry["goto"] = target
    for key in ("scroll_to", "wait_for"):
        val = directives.get(key)
        if val:
            entry[key] = str(val)
    return entry


class _LoginScope:
    """A brief's unsealed login block, released per recording job by origin.

    ``for_urls`` returns the login bound EXACTLY to the first URL's origin, and only
    when every URL the job navigates shares that origin; otherwise ``None`` — an
    off-origin path is recorded with no credential at all (no bearer header, no
    cookie, no basic-auth on a 401). Never "the first login"."""

    def __init__(self, block: object, context: dict) -> None:
        self._block = block
        self._context = context

    def for_urls(self, urls: list[str]) -> dict | None:
        if not urls or not self._block:
            return None
        origin = canonical_origin(urls[0])
        if origin is None or any(canonical_origin(u) != origin for u in urls):
            return None
        return auth_for_origin(self._block, urls[0], self._context)

    @property
    def has_login(self) -> bool:
        """True when the brief carries a sign-in the tenant expects this stage to use."""
        return any(
            e.get("username") or any(e.get(f) not in _EMPTY for f in SECRET_FIELDS & e.keys())
            for e in auth_entries(self._block)
        )

    def unbound_reason(self, urls: list[str]) -> str:
        """Why :meth:`for_urls` released no login for ``urls`` — said for the tenant."""
        if self._block and auth_for_origin(self._block, urls[0], self._context) is not None:
            return (
                "some pages in this capture are on another site, and the sign-in is only "
                f"used when every page stays on {_site_name(urls[0])}"
            )
        sign_in_pages = {
            host
            for e in auth_entries(self._block)
            if (origin := canonical_origin(e.get("login_url")))
            and (host := origin.removeprefix("https://"))
            and not e.get("signs_into")
        }
        if sign_in_pages:
            # Single sign-on is DECLARED, never inferred: a sign-in page on another
            # site signs into that site unless the tenant says which app it opens.
            return (
                f"the saved sign-in page is on {sorted(sign_in_pages)[0]}, and the sign-in is "
                f'not set to sign into {_site_name(urls[0])} — set "Signs into" to '
                f"{_site_name(urls[0])}"
            )
        return "the saved sign-in is for a different site"


#: Code the recorder puts at the start of a job error when the sign-in did not work
#: (video_production_service ``url_recorder.SITE_LOGIN_FAILED``). Matched as a code,
#: never as prose, so the recorder can reword its message freely.
SITE_LOGIN_FAILED_CODE = "site_login_failed"


def _site_name(url: str) -> str:
    """The host a tenant recognises (``app.example.com``), never a full URL."""
    return urlsplit(url).hostname or _redact_userinfo(url)


def _is_login_failure(exc: BaseException) -> bool:
    return SITE_LOGIN_FAILED_CODE in str(exc)


def _require_password(auth: dict | None, url: str) -> None:
    """A username+password sign-in whose password is gone cannot sign in: say so.

    The Brief keeps a saved password across edits, but drops it when the login
    address or username changes (``auth_secrets.restore_redacted``). Recording then
    would type the username alone and record the sign-in page."""
    if (
        auth
        and auth.get("type", "basic") == "basic"
        and auth.get("username")
        and not auth.get("password")
    ):
        raise SiteCaptureLoginError(
            url,
            "no password is saved for this sign-in (the login address or username "
            "changed) — re-enter the password",
        )


def _check_clip_logins(source_url: str, targets: list, auth_info: _LoginScope) -> None:
    """Before recording any clip: a saved sign-in must apply to some path, with a password."""
    if auth_info.has_login and not any(auth_info.for_urls([url]) for url, _, _ in targets):
        # A path on another origin is legitimately recorded signed out, but a
        # sign-in that applies to NO path means every clip is a login screen.
        raise SiteCaptureLoginError(source_url, auth_info.unbound_reason([source_url]))
    for url, _, _ in targets:
        _require_password(auth_info.for_urls([url]), url)


def _raise_on_login_failure(source_url: str, failures: list[BaseException]) -> None:
    """Fail the stage when any recording reported that the sign-in did not work."""
    login_failure = next((f for f in failures if _is_login_failure(f)), None)
    if login_failure is not None:
        raise SiteCaptureLoginError(
            source_url, "the sign-in page did not accept the saved login"
        ) from login_failure


class SiteCaptureLoginError(StageError):
    """The brief carries a sign-in and the capture could not use it → fail the stage.

    WHY this is not a soft degrade: recording a sign-in-only app signed OUT records
    its login screen. The stage then "succeeds" with thin page text, and every
    downstream stage scripts, designs and renders a video of a login form (prod run
    01a17e6f, 2026-10-27: the saved sign-in was never sent, three degraded warnings,
    a garbage video). The tenant can fix this — the message says where.
    """

    failure_class = CUSTOMER_INPUT

    def __init__(self, url: str, why: str) -> None:
        self.site = _site_name(url)
        message = (
            f"Couldn't sign in to {self.site} — {why}. Check the source site sign-in on "
            "this project's Brief page, then rerun this step."
        )
        super().__init__(message)
        # Structured, so the run page names the site and links the fix without
        # parsing prose (``failure_detail_for`` → ``runs.failure_detail``).
        self.reason = why
        self.failure_detail = {
            "code": SITE_LOGIN_FAILED_CODE,
            "site": self.site,
            "reason": why,
            "message": message,
        }


_DEFAULT_MAX_CLIPS = 4

#: ``config_snapshot["type"]`` of the one blueprint whose brief carries ``target_url``.
WEBSITE_REVIEW_TYPE = "website_review"
#: The live-site screens a website_review captures when the brief authors none: the
#: landing viewport, then the foot of the page. Soft-fail like every non-contracted
#: screen pass — a capture miss degrades the before act to the redesign's original
#: screenshots, it never fails the review.
WEBSITE_REVIEW_DEFAULT_SCREENS: tuple[dict[str, str], ...] = (
    {"capture_id": "live-top", "label": "Current homepage — top", "viewport": "desktop"},
    {
        "capture_id": "live-bottom",
        "label": "Current homepage — bottom",
        "viewport": "desktop",
        "scroll_to": "bottom",
    },
)
# Settle after networkidle before recording: data-driven SPAs (e.g. the demo
# dashboard) render their main pane via XHR *after* the network goes idle, so
# without this the capture is a blank shell. On by default so the white-screen
# fix doesn't depend on every brief remembering to set it; it also enables the
# recorder's blank-lead trim.
_DEFAULT_POST_LOAD_DELAY_MS = 3500
_RECORD_TIMEOUT_S = 600.0
_CONCURRENCY = 2
_SCRAPE_TIMEOUT_S = 15.0
_PAGE_TEXT_MAX_CHARS = 8000

#: Floor below which scraped copy cannot ground a script, reported as "thin".
#:
#: The ``"empty"`` status alone is near-unreachable in practice: ``_extract_text``
#: strips tags across the WHOLE document, so a JS-only SPA still yields its
#: ``<title>`` and ``<noscript>`` line. Measured against real shells on
#: 2026-08-12: web.telegram.org → 12 chars ("Telegram Web"), excalidraw.com → 79
#: (mostly "You need to enable JavaScript"), tldraw.com → 136. All three were
#: classified ``"ok"`` and handed to the model as "copy from the live product
#: site" — grounding in name only, and the LLM invents exactly as much as it
#: would from nothing. Real content pages in the same probe: 443 / 523 / 2775.
#: 200 splits that cleanly with room on both sides.
#:
#: Thin copy is still RETURNED, never discarded — the caller says "here is what
#: little we read, do not extrapolate past it", which beats both silence and a
#: false claim of grounding.
_MIN_GROUNDING_CHARS = 200

#: Ceiling on the scraped HTML body, stated rather than inherited.
#:
#: w2v-common 0.55.0 gave ``safe_get`` a 64 MiB *default* cap
#: (``DEFAULT_MAX_FETCH_BYTES``), sized for an untrusted internet-sourced
#: **media** fetch. Inheriting it here would be wrong in both directions:
#:
#: * it is far too generous for markup — this fetch exists only to distil
#:   ``_PAGE_TEXT_MAX_CHARS`` (8000) characters of visible text, so 8 MiB is
#:   already ~1000x the retained output and well above any real HTML document;
#: * and, more importantly, the ``except Exception`` below degrades **any**
#:   failure to ``""``. ``ResponseTooLargeError`` subclasses ``ValueError``, so
#:   an over-cap body does not surface as an error — it silently becomes an
#:   empty page-text grounding for the script. A fail-soft path must not rest on
#:   a default that was chosen for a different kind of fetch; the ceiling that
#:   decides when the script loses its grounding is stated here, deliberately,
#:   where the degrade is visible.
#:
#: The note appended on the degrade names the cap, so an operator can tell a
#: too-large page apart from a network failure.
_SCRAPE_MAX_RESPONSE_BYTES = 8 * 1024 * 1024

# Browser recordings follow the run's resolution tier so URL footage is captured
# sharp at the tier's native viewport (Playwright video ignores
# device_scale_factor — the viewport itself must grow), then composited into the
# matching 2K/4K canvas instead of being upscaled from 1080p. 1080p maps to the
# sharper low-CRF "high_quality" preset rather than "standard".
_TIER_TO_RECORDING_PRESET: dict[str, str] = {
    "1080p": "high_quality",
    "2k": "2k",
    "4k": "4k",
}

# ── capture_status: the measurement-gap discriminator ────────────────────
#: At least one real screen/clip came back. The render gets real footage.
CAPTURE_OK = "ok"
#: The capture RAN and we MEASURED a shortfall: the recorder reported the job
#: failed (bad selector, page never settled), or it returned a bundle holding
#: fewer/zero usable screens. The fallback to slides is an informed choice.
CAPTURE_DEGRADED = "degraded"
#: MEASUREMENT GAP — we do not know what the site would have yielded. The
#: recorder never answered (job timed out / was reaped / the service was
#: unreachable), or its bundle could not be read at all. Never treat this as
#: "the site had nothing to show": it is the absence of a measurement, and a
#: consumer that scores it like ``degraded`` is the fail-open this field exists
#: to make impossible.
CAPTURE_UNKNOWN = "unknown"
#: No capture was attempted (no capturable entries).
CAPTURE_NOT_ATTEMPTED = "not_attempted"

#: Infrastructure failures: the downstream never gave us an ANSWER about the
#: site, so zero screens says nothing about the site itself.
#:
#: * ``TimeoutError`` — ``poll_job`` raises it both for the wall-clock deadline
#:   ("video_production job … timed out after 600.0s", the 2026-09-03 outage)
#:   and for exhausted poll attempts. In both the job's outcome is UNKNOWN.
#: * ``httpx.TransportError`` — connect/read/write failure; ``TimeoutException``
#:   is a subclass, so a socket-level timeout lands here too.
#:
#: Deliberately NOT here: a plain ``RuntimeError``. ``poll_job`` raises that
#: when the recorder itself reports ``status == "failed"`` — i.e. it RAN, looked
#: at the page, and told us the capture did not work. That is a measurement, not
#: a gap, and it stays soft so a bad selector in the brief degrades to slides
#: instead of failing the run.
_INFRA_CAPTURE_ERRORS = (TimeoutError, httpx.TransportError)
#: A job we successfully submitted answering 404/410 means it was reaped/evicted
#: before we could read it — again, no answer about the site.
_JOB_GONE_STATUS_CODES = frozenset({404, 410})


# ── Resuming a recording after a pod death ───────────────────────────────
#
# IDS ARE DERIVED, NOT MINTED. ``part_id`` / ``clip_id`` used to be ``uuid7()`` at
# execution time, so a re-entered handler addressed every part differently and
# re-recorded all of them. They are now a pure function of ``(run_id, url)``, so
# they are the same on every entry, in every pod — WITH OR WITHOUT a ledger (a
# ledger read that fails degrades to "call fn", and the Idempotency-Key below must
# still be the one the dead pod sent). Nothing downstream reads structure out of
# either id: ``part_id`` keeps its ``site-xxxxxxxx`` shape, ``clip_id`` stays a
# UUID string and remains the storage object name, so a repeat overwrites its own
# object instead of orphaning one.
#
# PER PART, three ledger rows:
#   ``submit:<part_id>[:r<n>]`` -> the recorder's job id. A re-entry POLLS it.
#   ``result:<part_id>``        -> the finished clip / screen list. A re-entry
#                                  neither submits nor polls.
#   ``abandoned:<job_id>``      -> the job will never yield (the recorder reported
#                                  it failed, it vanished, or its full wait budget
#                                  ran out). The next entry skips to ``:r<n+1>``.
#
# Every submit also carries ``Idempotency-Key: <run_id>:site_capture:<part_id>``
# (none was sent before). video_production replays the ORIGINAL job for the same
# key + body, which covers the window between the recorder accepting the job and
# the ``submit`` row being written. It replays a FAILED job too — which is why a
# resubmit after an abandoned job needs the ``:r<n>`` suffix on the key as well.
#
# Login secrets ride in the request body and reach NOTHING the ledger persists —
# no payload, and no hash either. ``input_hash`` is stored in the clear, and every
# other field under it is reconstructible from the run, so an unsalted digest of
# the request body would let anyone with DB read test password guesses offline:
# the plaintext-at-rest class migration 0069 removed. ``_login_identity`` is what
# is hashed instead: the login's non-secret fields plus WHICH secrets it carries.
# A login whose secret alone changes inside a run therefore still hits; that is
# accepted (the recording already exists) and is the price of persisting no
# function of the secret.
_SUBMIT_GENERATIONS = 8
#: Secret values that count as absent (mirrors ``auth_secrets._is_plaintext``).
_EMPTY: tuple[object, ...] = (None, "", [], {})


def _login_identity(auth: dict | None) -> dict[str, object] | None:
    """What may be hashed about a login: everything except its secrets' values.

    ``SECRET_FIELDS`` is the same definition ``auth_secrets`` seals at rest, so a
    field that is sealed in ``runs.config_snapshot`` never contributes here.
    """
    if not auth:
        return None
    return {
        "fields": {k: v for k, v in auth.items() if k not in SECRET_FIELDS},
        "secrets": sorted(k for k in SECRET_FIELDS & auth.keys() if auth[k] not in _EMPTY),
    }


class _NoMarkerError(Exception):
    """Raised by the read-only ledger probe when a marker row does not exist."""


async def _present() -> bool:
    return True


async def _absent() -> bool:
    raise _NoMarkerError


def _redact_userinfo(url: str) -> str:
    """*url* with any ``user:pass@`` userinfo removed — for hash inputs and anything
    that is LOGGED or PERSISTED (log lines, degrade notes, error messages).

    A ``source_url`` may embed basic-auth credentials. Hashed raw, the password
    becomes part of a ledger key / part id (a stable fingerprint of a secret, and
    a key that changes when only the password rotates). The URL actually SENT to
    the recorder is never passed through this.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if "@" not in parts.netloc:
        return url
    return urlunsplit(parts._replace(netloc=parts.netloc.rpartition("@")[2]))


#: ``scheme://user[:pass]@`` inside free text. ``[^\s/?#@]*`` keeps the match
#: inside one authority, so an ``@`` in a later path/query is not mistaken for one.
_URL_USERINFO_RE = re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://)[^\s/?#@'\"<>]*@")


def _redact_text(text: object) -> str:
    """``str(text)`` with userinfo stripped from every URL embedded in it.

    Exception text is NOT a bare URL: httpx's ``HTTPStatusError`` reads
    ``"Client error '401 Unauthorized' for url '<url with userinfo>'"`` and
    w2v_common's pinned request keeps that userinfo, so :func:`_redact_userinfo`
    on the URL alone left the password in every note/log that interpolated
    ``{exc}``. Use this for any exception text bound for a log line, a degrade
    note or an error message.
    """
    return _URL_USERINFO_RE.sub(r"\1", str(text))


def _stable_part_id(prefix: str, run_id: str, identity: str) -> str:
    """Recorder ``part_id`` for one capture target — identical on every entry."""
    identity = _redact_userinfo(identity)
    return f"{prefix}-{hash_inputs(run_id, 'site_capture', prefix, identity)[:8]}"


def _stable_clip_id(run_id: str, url: str) -> str:
    """Clip id (and storage object name) for one recorded url — a UUID string."""
    return str(uuid5(NAMESPACE_URL, f"{run_id}:site_capture:clip:{_redact_userinfo(url)}"))


def _degrade(notes: list[str], degradations: list[Degradation], code: str, text: str) -> None:
    """Record one degradation on BOTH channels: the prose note and its stable code."""
    notes.append(text)
    degradations.append(degradation(code, text))


def _is_job_gone(exc: BaseException) -> bool:
    return (
        isinstance(exc, httpx.HTTPStatusError)
        and exc.response.status_code in _JOB_GONE_STATUS_CODES
    )


def _job_will_never_yield(exc: BaseException) -> bool:
    """True when waiting on this job again cannot help, so a retry must resubmit.

    ``poll_job`` raises ``RuntimeError`` when the recorder reports the job failed
    or cancelled and ``TimeoutError`` when the whole wait budget ran out; a 404/410
    means the job was reaped. A transport error or a 5xx is NOT here: the job may
    well be fine, and the next entry polls it again rather than paying twice.
    """
    return isinstance(exc, RuntimeError | TimeoutError) or _is_job_gone(exc)


async def _job_abandoned(ledger: StepLedger, job_id: str) -> bool:
    try:
        return bool(await ledger.once(f"abandoned:{job_id}", "abandoned", _absent))
    except _NoMarkerError:
        return False


async def _resumable_recording(
    client: VideoProductionClient,
    ledger: StepLedger,
    *,
    run_id: str,
    part_id: str,
    body_hash: str,
    submit: Callable[[str], Awaitable[str]],
) -> tuple[dict[str, object], bytes]:
    """Submit one recording — or re-attach to the one already submitted — and fetch it.

    ``submit`` takes the Idempotency-Key and returns the job id. Returns the settled
    job payload and the recorded bytes. Raises exactly what the client raised, so
    the callers' soft-fail / measurement-gap handling is unchanged.
    """
    base_key = f"{run_id}:site_capture:{part_id}"
    while True:
        job_id, reattached = "", False
        for generation in range(_SUBMIT_GENERATIONS):
            suffix = f":r{generation}" if generation else ""
            submitted_now = False

            async def _submit(suffix: str = suffix) -> str:
                nonlocal submitted_now
                submitted_now = True
                return await submit(base_key + suffix)

            job_id = await ledger.once(f"submit:{part_id}{suffix}", body_hash, _submit)
            reattached = not submitted_now
            if submitted_now or not await _job_abandoned(ledger, job_id):
                break
        else:
            # Every tracked generation was abandoned. Never stall on a dead job:
            # submit untracked, under a key no earlier attempt used.
            job_id, reattached = await submit(f"{base_key}:r{uuid7().hex[:12]}"), False
        try:
            result = await client.wait_for_content_recording(job_id, timeout=_RECORD_TIMEOUT_S)
            payload, _ctype, _disp = await client.get_recording_video(job_id)
        except Exception as exc:
            if job_id and _job_will_never_yield(exc):
                await ledger.once(f"abandoned:{job_id}", "abandoned", _present)
                if reattached and _is_job_gone(exc):
                    # The job a dead pod submitted no longer exists (the recorder
                    # rolled too). An uninterrupted run never sees this state, so
                    # there is no behaviour to preserve: record it again, now.
                    log.warning(
                        "site_capture: stored job %s for part %s is gone; resubmitting",
                        job_id,
                        part_id,
                    )
                    continue
            raise
        return result, payload


class _BundleUnreadableError(Exception):
    """The recorder answered but its bundle could not be unpacked / re-published.

    Carries the screens that WERE published before the failure, so the per-screen
    path keeps returning them exactly as it did; raising keeps a partial result out
    of the ledger.
    """

    def __init__(self, screens: list[SiteScreen]) -> None:
        super().__init__("per-screen bundle unreadable")
        self.screens = screens


def _dump_screens(screens: list[SiteScreen]) -> JsonValue:
    return [s.model_dump(mode="json") for s in screens]


def _load_screens(raw: JsonValue) -> list[SiteScreen]:
    if not isinstance(raw, list):
        raise TypeError("stored per-screen result is not a list")
    return [SiteScreen.model_validate(item) for item in raw]


class SiteCaptureUnavailableError(StageError):
    """The capture never produced a measurement → fail the stage, loudly.

    WHY this is not a soft degrade: the soft path returns a SUCCESSFUL stage
    holding ``screens: []``. On 2026-09-03 an infrastructure timeout (a
    video_production recording-slot leak) therefore reported ``completed`` for a
    TOTAL capture outage — the run could not tell "this site has nothing to
    record" from "we never asked it". Only failures where the downstream
    answered stay soft; a missing answer fails here so the gap is visible.
    """


class SiteCaptureIncompleteError(StageError):
    """A CONTRACTED screen is missing → fail the stage, even on a soft failure.

    WHY this exists even though soft-fail is the *correct* contract elsewhere:
    the two entry points into ``_capture_screens`` promise different things.

    * ``brief.source_url`` (+ optional ``site_capture.screens``) is **opt-in
      enhancement**. Nothing downstream contracted a specific screen; if the
      recorder answers "the capture did not work", degrading to slides / AI
      images is an informed, complete render. Zero screens there is a legitimate
      outcome and MUST stay soft — see the module docstring.
    * An authored ``brief.shotlist`` is a **deliverable**. Each ``kind: url``
      shot contracts exactly one screen whose ``screen.id == shot.line_id``;
      ``promo_visual_director`` binds narration to that id 1:1. A missing id is
      not a degraded render, it is a missing deliverable, and the downstream
      stages have nothing to bind to.

    The observed failure: ``site_capture`` reported stage status ``completed``
    while emitting zero screens (the recorder ANSWERED "job failed", so the soft
    path was taken and inherited by the shotlist branch). The dependent stages
    then stayed ``pending`` and the whole run parked in ``paused`` — a missing
    deliverable reported as a clean completion. E2E scenario
    ``promo-shotlist-url`` failed at 95s with
    ``screens[].id has 0 distinct value(s) [], expected >= 2`` while the
    deploy-drift guard certified ``✅ stable``.

    So: this is not "soft-fail was wrong". It is "soft-fail is right for an
    enhancement and wrong for a contract", and the shotlist branch was reusing
    the enhancement's semantics for a contract.
    """


def _clip_capture_status(
    clips: list[SiteClip], failures: list[BaseException], notes: list[str]
) -> str:
    """``capture_status`` for the clip path, appending the gap note when it applies.

    The clip path keeps its soft-fail semantics — per-path capture is a
    best-effort enhancement and one bad path must not abort a multi-path run —
    but zero clips caused by a downstream that never ANSWERED is a measurement
    gap, and saying so is not the same as saying the site had nothing to record.
    """
    if clips:
        return CAPTURE_OK
    gaps = [f for f in failures if _is_capture_infrastructure_failure(f)]
    if not gaps:
        return CAPTURE_DEGRADED
    notes.append(
        "no clips captured AND video_production never answered for "
        f"{len(gaps)} path(s) — capture_status=unknown: this is a "
        "MEASUREMENT GAP, not evidence the site had nothing to record"
    )
    return CAPTURE_UNKNOWN


def _is_capture_infrastructure_failure(exc: BaseException) -> bool:
    """True when *exc* means "we never got an answer", not "the answer was bad"."""
    if isinstance(exc, _INFRA_CAPTURE_ERRORS):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        # 5xx = the dependency is broken; 404/410 = the job vanished under us.
        # A 4xx we caused (422 on a malformed step list) is an ANSWER and stays soft.
        return code >= 500 or code in _JOB_GONE_STATUS_CODES
    return False


def _resolve_recording_quality_preset(ctx: RunContext, cfg: dict) -> str:
    """Recorder ``quality_preset`` for this run's resolution tier.

    Default: follow the run tier (``config_snapshot["resolution"]``). A brief can
    pin a different tier per-deliverable via ``site_capture.recording_tier`` —
    ``"follow"`` (default) tracks the run tier, or ``1080p|2k|4k`` pins it (use
    this when a site reflows badly at a 3840-wide viewport). An explicit
    ``site_capture.viewport`` in the brief still wins downstream: the recorder
    applies viewport overrides over the preset's dimensions.
    """
    override = str(cfg.get("recording_tier") or "follow").strip().lower()
    tier = override if override in _TIER_TO_RECORDING_PRESET else quality_tier(ctx.config_snapshot)
    return _TIER_TO_RECORDING_PRESET[tier]


class SiteCaptureInput(StageInput):
    pass  # All inputs come from ctx.config_snapshot["brief"].


class SiteClip(StageOutput):
    id: str
    video_url: str
    seconds: float
    # The relative path this clip recorded (e.g. "/demo"), and whether it is the
    # primary clip — the scripted interactive walkthrough that promo_render makes
    # the spine of the video. A path with ``interactions`` IS the demo; the others
    # (landing, book) become brief bookend cutaways instead of round-robin repeats.
    path: str = ""
    is_primary: bool = False


class SiteScreen(StageOutput):
    """One discrete demo screen captured as a still (and optionally a clip).

    Produced by the per-screen capture mode (``brief.site_capture.screens``):
    ``id`` is the screen's ``capture_id``, ``label`` is the human description
    the visual-director LLM maps narration lines against, and ``image_url`` is
    the re-published PNG the compiler keys per line (frame-synced demo). In
    ``motion: "clips"`` mode ``clip_url`` is the re-published short interaction
    clip (the click landing on this screen) that the render plays on the line
    that reveals the screen; empty in stills-only mode.
    """

    id: str
    label: str = ""
    image_url: str
    clip_url: str = ""
    page_text: str = ""
    html_content: str = ""
    # ORCH-3.2 directed-footage geometry: the VP-2 ``<id>.track.json`` capture
    # sidecar for this screen (frame-pixel coords). ``focus`` = clicked-control
    # bboxes over time (auto-zoom target), ``cursor`` = the pointer path (synthetic
    # eased cursor), ``viewport`` = the recorded frame size. All default ``None``
    # (append-last, defaulted) so a persisted older screen with no sidecar still
    # deserializes and, absent the ``direction.footage`` brief, they stay unset —
    # nothing rides the shot => byte-identical render.
    focus_track: list[dict[str, object]] | None = None
    cursor_track: list[dict[str, object]] | None = None
    viewport: dict[str, object] | None = None


class SiteCaptureOutput(StageOutputWithDegradations):
    clips: list[SiteClip]
    page_text: str
    # WHY: an empty ``page_text`` used to be indistinguishable between "target
    # refused by the SSRF guard", "fetch blew up" and "200 OK but nothing
    # extractable" (JS-only SPA / image-only page). Consumers MUST branch on this
    # discriminator — ``"ok" | "empty" | "blocked" | "fetch_failed" |
    # "not_attempted"`` — and never parse ``notes`` prose to reach the same answer.
    page_text_status: str = "not_attempted"
    html_content: str = ""
    # WHY (mirrors ``page_text_status``, same rule): an empty ``screens``/``clips``
    # list used to be indistinguishable between "the recorder ran and the site
    # yielded nothing" and "we never got an answer at all". Consumers MUST branch
    # on this discriminator — ``"ok" | "degraded" | "unknown" | "not_attempted"``
    # — and never infer the same answer from list length or ``notes`` prose.
    # ``"unknown"`` is a MEASUREMENT GAP, not a zero.
    capture_status: str = CAPTURE_NOT_ATTEMPTED
    notes: list[str]
    # Discrete per-screen screenshots (per-screen capture mode). Empty in the
    # legacy single-clip mode. Each maps a narration line → the exact demo
    # screen it describes via the promo_visual_director stage.
    screens: list[SiteScreen] = []  # noqa: RUF012 — pydantic copies field defaults per-instance
    # Explicit, unmistakable signal of whether the render gets real recorded
    # footage or falls back to AI images / text cards. Derivable from ``clips``
    # but surfaced as a first-class field so consumers (and honesty checks on
    # the run output) never have to infer it from list length or parse notes.
    used_real_footage: bool = False


class SiteCaptureStage(StageHandler[SiteCaptureInput, SiteCaptureOutput]):
    name = "site_capture"
    input_model = SiteCaptureInput
    output_model = SiteCaptureOutput

    def __init__(self, client: VideoProductionClient | None = None) -> None:
        self._client = client or VideoProductionClient()

    async def handle(
        self,
        payload: SiteCaptureInput,
        ctx: RunContext,
    ) -> SiteCaptureOutput:
        brief = ctx.config_snapshot.get("brief") or {}
        cfg = brief.get("site_capture") or {}
        # Sealed at rest (auth_secrets); unsealed in memory for the capture only, and
        # scoped per recording job by `_scoped_auth` — a login is attached ONLY to a
        # job whose every navigated URL is its bound origin.
        auth_info = _LoginScope(unseal_auth(brief.get("auth_info")), brief)
        # Resolve once: browser recordings follow the run's resolution tier so the
        # footage is captured sharp at 2K/4K rather than upscaled in the compiler.
        quality_preset = _resolve_recording_quality_preset(ctx, cfg)
        # ORCH-3.2 directed-footage gate: the presence-is-opt-in ``direction.footage``
        # block (lifted whole into config_snapshot by run_setup). Absent => the VP-2
        # ``.track.json`` sidecar is not extracted and no ``capture_scale`` bump is
        # requested => byte-identical to today. Truthy ``auto_zoom`` inside it also
        # asks the recorder to capture at 2x so compiler punch-ins crop real pixels.
        footage = (ctx.config_snapshot.get("direction") or {}).get("footage")
        ledger = ctx.step_ledger(self.name)

        # Authored shotlist: synthesize per-screen capture from the URL shots (one
        # screenshot per shot, keyed by the shot's deterministic line id so the
        # visual director binds it 1:1). The shotlist's URL shots win over any
        # legacy site_capture.paths/.screens config. Slide shots need no capture.
        url_shots = [s for s in normalize_shotlist(brief) if s.kind == "url"]
        if url_shots:
            # Capture base: an explicit source_url, else the website_review target
            # (that brief uses target_url, not source_url), else the first shot's
            # URL. Relative shot paths resolve against this base in the recorder.
            base_url = (
                (brief.get("source_url") or "").strip()
                or (brief.get("target_url") or "").strip()
                or url_shots[0].url
                or ""
            )
            screens_cfg = [_shot_to_screen_entry(s, base_url) for s in url_shots]
            return await self._capture_screens(
                base_url,
                cfg,
                screens_cfg,
                auth_info,
                str(ctx.run_id),
                quality_preset,
                footage,
                ledger=ledger,
                # The shotlist CONTRACTS one screen per url shot, keyed by line id.
                # Passing them here is what turns a soft degrade into a hard
                # failure on this path only — see SiteCaptureIncompleteError.
                required_screen_ids=[s.line_id for s in url_shots],
            )

        source_url = (brief.get("source_url") or "").strip()
        screens_cfg = cfg.get("screens")
        if not source_url and ctx.config_snapshot.get("type") == WEBSITE_REVIEW_TYPE:
            # A website_review brief names its site `target_url`, never `source_url`
            # (ReviewBrief), so this stage skipped on EVERY review and the before
            # act never showed the live site (prod run 01a0a054). The review's own
            # site is the capture target; with no authored screens it gets the
            # default per-screen pass, because website_review_compile consumes
            # `screens` only — a scroll CLIP would be recorded and never shown.
            source_url = (brief.get("target_url") or "").strip()
            if source_url and not (isinstance(screens_cfg, list) and screens_cfg):
                screens_cfg = list(WEBSITE_REVIEW_DEFAULT_SCREENS)
        if not source_url:
            raise StageSkippedError(
                "site_capture: no brief.source_url (opt-in stage not requested)",
                missing=["brief.source_url"],
            )

        # Per-screen capture mode (frame-synced demo): when the brief lists
        # discrete demo screens, capture one screenshot each in a single pass
        # instead of one continuous panned clip. Opt-in + soft-fail like the
        # clip path.
        if isinstance(screens_cfg, list) and screens_cfg:
            return await self._capture_screens(
                source_url,
                cfg,
                screens_cfg,
                auth_info,
                str(ctx.run_id),
                quality_preset,
                footage,
                ledger=ledger,
            )

        return await self._capture_clips(
            source_url, cfg, auth_info, str(ctx.run_id), quality_preset, footage, ledger=ledger
        )

    async def _record_clip(
        self,
        ledger: StepLedger,
        run_id: str,
        target: _Target,
        auth: dict | None,
        rec_config: dict[str, object],
        scroll_config: dict | None,
    ) -> SiteClip:
        """Record one path — or reuse / re-attach to what a dead pod already paid for."""
        url, path, interactions = target
        # Stable across re-entry (see "Resuming a recording"): the url is
        # the part's identity — targets are de-duped by resolved url.
        part_id = _stable_part_id("site", run_id, url)
        clip_id = _stable_clip_id(run_id, url)
        # Everything the recorder is sent, EXCEPT the login's secrets: only
        # ``_login_identity(auth)`` is hashed (see "Login secrets" above).
        body_hash = hash_inputs(
            {
                "url": _redact_userinfo(url),
                "part_id": part_id,
                "auth_info": _login_identity(auth),
                "recording_config": rec_config or None,
                "scroll_config": scroll_config,
            }
        )

        async def _submit(idempotency_key: str) -> str:
            return await self._client.record_url(
                url=url,
                part_id=part_id,
                auth_info=auth,
                recording_config=rec_config or None,
                scroll_config=scroll_config,
                idempotency_key=idempotency_key,
            )

        async def _record() -> SiteClip:
            result, video_bytes = await _resumable_recording(
                self._client,
                ledger,
                run_id=run_id,
                part_id=part_id,
                body_hash=body_hash,
                submit=_submit,
            )
            video_url = await store_capture_clip(
                run_id=run_id,
                stage_name="site_capture",
                clip_id=clip_id,
                video_bytes=video_bytes,
                video_format="mp4",
            )
            return SiteClip(
                id=clip_id,
                video_url=video_url,
                seconds=float(cast("float | int | str", result.get("duration_seconds") or 0.0)),
                path=path,
                is_primary=bool(interactions),
            )

        # A raise stores nothing, so a failed part is retried on re-entry.
        return await ledger.once(
            f"result:{part_id}",
            hash_inputs(body_hash, path),
            _record,
            dump=lambda clip: clip.model_dump(mode="json"),
            load=SiteClip.model_validate,
        )

    async def _capture_clips(
        self,
        source_url: str,
        cfg: dict,
        auth_info: _LoginScope,
        run_id: str,
        quality_preset: str | None = None,
        footage: dict | None = None,
        *,
        ledger: StepLedger,
    ) -> SiteCaptureOutput:
        """Legacy single/continuous-clip capture (one clip per configured path).

        The scripted-walkthrough clip becomes the render spine; other paths are
        bookend cutaways. Soft-fails per clip so one bad recording never aborts
        the run.
        """
        targets = self._resolve_targets(source_url, cfg)
        _check_clip_logins(source_url, targets, auth_info)
        base_recording_config = self._build_recording_config(cfg, quality_preset, footage)
        scroll_config = cfg.get("scroll") or None
        notes: list[str] = []
        degradations: list[Degradation] = []
        # Paths whose recording failed because the downstream never answered.
        failures: list[BaseException] = []

        async def _capture(target: _Target) -> SiteClip | None:
            url, _path, interactions = target
            # Per-path: a scripted walkthrough (e.g. clicking a demo's nav
            # sections) overrides the plain scroll for that clip — and marks it as
            # the primary (demo) clip the render makes its spine.
            rec_config = dict(base_recording_config or {})
            if interactions:
                rec_config["interaction_steps"] = interactions
            try:
                return await self._record_clip(
                    ledger,
                    run_id,
                    target,
                    auth_info.for_urls([url]),
                    rec_config,
                    scroll_config,
                )
            except Exception as exc:  # noqa: BLE001 -- soft-fail per clip: record, don't abort the run
                shown = _redact_userinfo(url)
                log.warning("site_capture: recording %s failed: %s", shown, _redact_text(exc))
                _degrade(
                    notes,
                    degradations,
                    SITE_CAPTURE_CLIP_FAILED,
                    f"recording failed for {shown}: {_redact_text(exc)}",
                )
                # Per-clip failure semantics are unchanged (one bad path must not
                # abort a multi-path capture), but the failure is now KEPT so
                # ``capture_status`` can tell "the site gave us nothing" from
                # "we never got an answer for any path".
                failures.append(exc)
                return None

        # Bounded concurrency, absorbing per-clip failures (return_exceptions so one
        # bad clip never cancels its siblings — see the narrow-except gather trap).
        sem = asyncio.Semaphore(_CONCURRENCY)

        async def _guarded(target: _Target) -> SiteClip | None:
            async with sem:
                return await _capture(target)

        results = await asyncio.gather(*(_guarded(t) for t in targets), return_exceptions=True)
        _raise_on_login_failure(source_url, failures)
        clips: list[SiteClip] = []
        for res in results:
            if isinstance(res, SiteClip):
                clips.append(res)
            elif isinstance(res, BaseException):
                _degrade(
                    notes,
                    degradations,
                    SITE_CAPTURE_CLIP_FAILED,
                    f"capture task errored: {_redact_text(res)}",
                )
                failures.append(res)

        page_text, page_text_status = await self._scrape_page_text(source_url, notes, degradations)

        capture_status = _clip_capture_status(clips, failures, notes)
        if not clips:
            _degrade(
                notes,
                degradations,
                SITE_CAPTURE_NO_FOOTAGE,
                "no clips captured; render will fall back to AI images / text cards",
            )
        return SiteCaptureOutput(
            clips=clips,
            page_text=page_text,
            page_text_status=page_text_status,
            capture_status=capture_status,
            notes=notes,
            degradations=degradations,
            used_real_footage=bool(clips),
        )

    async def _publish_screens(
        self,
        zip_bytes: bytes,
        run_id: str,
        label_by_id: dict[str, str],
        *,
        record_clips: bool,
        footage: dict | None,
    ) -> list[SiteScreen]:
        """Unpack the recorder's bundle and re-publish each screen.

        Raises :class:`_BundleUnreadableError` (carrying what WAS published) on any
        failure: the recorder answered, so the failure is ours, and the real screen
        count is left UNMEASURED.
        """
        published: list[SiteScreen] = []
        try:
            page_contents: dict[str, dict] = {}
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as bundle:
                for entry in bundle.infolist():
                    if entry.filename.endswith(".page.json") and entry.file_size <= 250000:
                        try:
                            data = json.loads(bundle.read(entry))
                        except ValueError, UnicodeDecodeError:
                            continue
                        if isinstance(data, dict):
                            page_contents[entry.filename.removesuffix(".page.json")] = data
            clips_by_id = dict(self._unzip_screen_clips(zip_bytes)) if record_clips else {}
            # ORCH-3.2 directed footage: extract the per-screen VP-2
            # ``.track.json`` capture sidecars ONLY when the run opted in via
            # ``direction.footage``. Absent => empty map => screens carry no
            # geometry => byte-identical to today.
            tracks_by_id = dict(self._unzip_screen_tracks(zip_bytes)) if footage else {}
            for screen_id, png in self._unzip_screens(zip_bytes):
                image_url = await store_capture_image(
                    run_id=run_id,
                    stage_name="site_capture",
                    image_id=screen_id,
                    image_bytes=png,
                    image_format="png",
                )
                clip_url = ""
                mp4 = clips_by_id.get(screen_id)
                if mp4:
                    clip_url = await store_capture_clip(
                        run_id=run_id,
                        stage_name="site_capture",
                        clip_id=f"{screen_id}-clip",
                        video_bytes=mp4,
                        video_format="mp4",
                    )
                track = tracks_by_id.get(screen_id) or {}
                published.append(
                    SiteScreen(
                        id=screen_id,
                        label=label_by_id.get(screen_id, screen_id),
                        page_text=str(page_contents.get(screen_id, {}).get("text") or "")[:50000],
                        html_content=str(page_contents.get(screen_id, {}).get("html") or "")[
                            :150000
                        ],
                        image_url=image_url,
                        clip_url=clip_url,
                        # _coerce_track already shape-checked these (lists of
                        # dict keyframes / a viewport dict); cast the opaque
                        # passthrough value into the SiteScreen field type
                        # without re-validating.
                        focus_track=cast("list[dict[str, object]] | None", track.get("focus")),
                        cursor_track=cast("list[dict[str, object]] | None", track.get("cursor")),
                        viewport=cast("dict[str, object] | None", track.get("viewport")),
                    )
                )
        except Exception as exc:
            raise _BundleUnreadableError(published) from exc
        return published

    async def _capture_screens(
        self,
        source_url: str,
        cfg: dict,
        screens_cfg: list,
        auth_info: _LoginScope,
        run_id: str,
        quality_preset: str | None = None,
        footage: dict | None = None,
        required_screen_ids: list[str] | None = None,
        *,
        ledger: StepLedger,
    ) -> SiteCaptureOutput:
        """Capture one screenshot per demo screen in a single navigation pass.

        Maps each ``brief.site_capture.screens`` entry to a recorder step
        (``capture_id`` marks a screen to shoot; an entry without one only
        advances the SPA, e.g. a persona switch), downloads the resulting zip,
        unpacks it, and re-publishes each PNG so the compiler can fetch it
        anonymously. Soft-fails to zero screens (render degrades to slides).

        ``required_screen_ids`` names the screens a CALLER contracted (the
        authored-shotlist path passes one line id per ``kind: url`` shot). When
        given, an exit that would return zero — or merely incomplete — screens
        raises :class:`SiteCaptureIncompleteError` instead of reporting a
        successful stage. ``None`` (the ``brief.source_url`` enhancement path)
        keeps the soft-fail semantics byte for byte.
        """
        notes: list[str] = []
        degradations: list[Degradation] = []
        steps, label_by_id = self._build_screen_steps(screens_cfg)
        if not any(s.get("capture_id") for s in steps):
            notes.append("site_capture.screens has no capturable entries; no screens captured")
            # Routed through _screens_output like the other two empty exits so the
            # contract check below cannot be bypassed by an early return.
            return await self._screens_output(
                source_url,
                notes,
                degradations,
                screens=[],
                # Nothing was asked of the recorder — not a gap, not a degrade.
                capture_status=CAPTURE_NOT_ATTEMPTED,
                required_screen_ids=required_screen_ids,
            )

        rec_config = self._build_recording_config(cfg, quality_preset, footage)
        # motion: "clips" → also record a short interaction clip per screen (the
        # click landing on it) for the render to play on the revealing line.
        record_clips = str(cfg.get("motion") or "").strip().lower() == "clips"
        screens: list[SiteScreen] = []
        # Default to the GAP, not to success: every path below has to earn a
        # narrower value. A future early-return that forgets to set it therefore
        # reports "we do not know", which is the honest answer.
        capture_status = CAPTURE_UNKNOWN
        # One recording job for the whole pass, resumable across a pod death (see
        # "Resuming a recording"). One browser session navigates every `goto`: the
        # login rides along only if ALL of them stay on its bound origin.
        part_id = _stable_part_id("screens", run_id, source_url)
        visited = [
            source_url,
            *(urljoin(source_url, str(st["goto"])) for st in steps if st.get("goto")),
        ]
        auth = auth_info.for_urls(visited)
        if auth is None and auth_info.has_login:
            # Recording a sign-in-only app signed out records its login page; the
            # tenant saved a sign-in precisely so that would not happen.
            raise SiteCaptureLoginError(source_url, auth_info.unbound_reason(visited))
        _require_password(auth, source_url)
        # The login's secrets are sent below and never hashed (see "Login secrets").
        # A `goto` may embed `user:pass@` exactly like source_url: redact it in the
        # HASHED copy only. `_submit` sends `steps` verbatim — the recorder's browser
        # needs the embedded basic-auth credentials to load that page.
        hashed_steps = [
            {**st, "goto": _redact_userinfo(str(st["goto"]))} if st.get("goto") else st
            for st in steps
        ]
        body_hash = hash_inputs(
            {
                "url": _redact_userinfo(source_url),
                "part_id": part_id,
                "steps": hashed_steps,
                "auth_info": _login_identity(auth),
                "recording_config": rec_config,
                "record_clips": record_clips,
            }
        )

        async def _submit(idempotency_key: str) -> str:
            return await self._client.capture_screens(
                url=source_url,
                steps=steps,
                part_id=part_id,
                auth_info=auth,
                recording_config=rec_config,
                record_clips=record_clips,
                idempotency_key=idempotency_key,
            )

        async def _capture() -> list[SiteScreen]:
            # Phase 1 — ask the recorder. An exception here propagates as-is: it
            # either means the recorder never answered (gap => fail loud) or it
            # answered "the capture failed" (measurement => soft, as before).
            _result, zip_bytes = await _resumable_recording(
                self._client,
                ledger,
                run_id=run_id,
                part_id=part_id,
                body_hash=body_hash,
                submit=_submit,
            )
            return await self._publish_screens(
                zip_bytes, run_id, label_by_id, record_clips=record_clips, footage=footage
            )

        try:
            # Stored only when the bundle was read end to end; any raise stores
            # nothing, so the next entry polls the job (or resubmits a failed one).
            screens = await ledger.once(
                f"result:{part_id}",
                hash_inputs(body_hash, label_by_id, bool(footage)),
                _capture,
                dump=_dump_screens,
                load=_load_screens,
            )
        except _BundleUnreadableError as unreadable:
            # soft-fail: degrade to slides, don't abort the run
            cause = unreadable.__cause__
            log.warning("site_capture: per-screen capture failed: %s", _redact_text(cause))
            _degrade(
                notes,
                degradations,
                SITE_CAPTURE_SCREENS_FAILED,
                f"per-screen capture failed: {_redact_text(cause)}",
            )
            screens = unreadable.screens
            capture_status = CAPTURE_UNKNOWN
        except Exception as exc:
            if _is_login_failure(exc):
                raise SiteCaptureLoginError(
                    source_url, "the sign-in page did not accept the saved login"
                ) from exc
            if _is_capture_infrastructure_failure(exc):
                # The measurement gap. Returning a successful stage with
                # ``screens: []`` here is what let a 600s recording-slot timeout be
                # reported as `completed` on 2026-09-03.
                log.error("site_capture: per-screen capture UNAVAILABLE: %s", _redact_text(exc))
                raise SiteCaptureUnavailableError(
                    f"site_capture: the per-screen capture of {_redact_userinfo(source_url)} "
                    f"produced NO "
                    f"measurement — video_production never answered ({type(exc).__name__}: "
                    f"{_redact_text(exc)}). Zero screens here does not mean the site had nothing to show, "
                    f"so the run fails rather than silently rendering slides as if it did. "
                    f"Check video_production's recording slots / reachability."
                ) from exc
            # The downstream ANSWERED (the recorder reported the job failed, or a
            # 4xx we caused): a real, measured degradation. Soft-fail to slides
            # exactly as before — same note prose consumers already read.
            log.warning("site_capture: per-screen capture failed: %s", _redact_text(exc))
            _degrade(
                notes,
                degradations,
                SITE_CAPTURE_SCREENS_FAILED,
                f"per-screen capture failed: {_redact_text(exc)}",
            )
            return await self._screens_output(
                source_url,
                notes,
                degradations,
                screens=[],
                capture_status=CAPTURE_DEGRADED,
                required_screen_ids=required_screen_ids,
            )
        else:
            # The bundle was read end to end, so the screen count is MEASURED.
            capture_status = CAPTURE_OK if screens else CAPTURE_DEGRADED

        return await self._screens_output(
            source_url,
            notes,
            degradations,
            screens=screens,
            capture_status=capture_status,
            required_screen_ids=required_screen_ids,
        )

    async def _screens_output(
        self,
        source_url: str,
        notes: list[str],
        degradations: list[Degradation],
        *,
        screens: list[SiteScreen],
        capture_status: str,
        required_screen_ids: list[str] | None = None,
    ) -> SiteCaptureOutput:
        """Scrape the page copy and assemble the per-screen output.

        One place so every per-screen exit reports the same shape — and so no exit
        can quietly forget the ``capture_status`` discriminator. It is also the
        single choke point where a CONTRACTED screen set is enforced: every empty
        exit of :meth:`_capture_screens` goes through here, so the check cannot be
        skipped by adding another early return.
        """
        # The contract check runs BEFORE the page scrape: if the deliverable is
        # already missing there is nothing to assemble, and spending a network
        # fetch to build an output we are about to discard is pure waste.
        if required_screen_ids:
            captured = {s.id for s in screens}
            missing = [i for i in required_screen_ids if i not in captured]
            if missing:
                raise SiteCaptureIncompleteError(
                    f"site_capture: {len(missing)} of {len(required_screen_ids)} CONTRACTED "
                    f"screen(s) are missing from the capture of {_redact_userinfo(source_url)} "
                    f"(capture_status={capture_status}): {missing}. Each authored url shot "
                    f"contracts a screen with id == its line id, and the downstream stages "
                    f"bind narration to those ids; a shotlist run that renders without them "
                    f"is missing a deliverable, not merely degraded. "
                    + ("; ".join(notes) if notes else "")
                )
        captured_text = "\n\n".join(
            dict.fromkeys(s.page_text.strip() for s in screens if s.page_text.strip())
        )[:50000]
        if captured_text:
            page_text = captured_text
            page_text_status = "ok" if len(page_text) >= _MIN_GROUNDING_CHARS else "thin"
            if page_text_status == "thin":
                _degrade(
                    notes,
                    degradations,
                    SITE_PAGE_TEXT_THIN,
                    "Captured pages contain too little readable text to ground the script",
                )
        else:
            page_text, page_text_status = await self._scrape_page_text(
                source_url, notes, degradations
            )

        if not screens:
            _degrade(
                notes,
                degradations,
                SITE_CAPTURE_NO_FOOTAGE,
                "no screens captured; render will fall back to slides / AI images",
            )
        return SiteCaptureOutput(
            clips=[],
            page_text=page_text,
            page_text_status=page_text_status,
            capture_status=capture_status,
            notes=notes,
            degradations=degradations,
            screens=screens,
            html_content=next((s.html_content for s in screens if s.html_content), ""),
            used_real_footage=bool(screens),
        )

    @staticmethod
    def _build_screen_steps(screens_cfg: list) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Map screen entries → recorder steps + ``{capture_id: label}``.

        Each entry is ``{selector, capture_id?|id?, label?, settle_ms?, action?,
        goto?, scroll_to?, wait_for?, viewport?}``. ``label`` is orchestrator-only
        (fed to the visual director), so it is stripped from the recorder step;
        every other key the recorder understands (recorder PR #54) is forwarded
        when present. Legacy ``brief.site_capture.screens`` entries simply lack
        the new keys, so their steps are unchanged.
        """
        steps: list[dict[str, Any]] = []
        label_by_id: dict[str, str] = {}
        for entry in screens_cfg:
            if not isinstance(entry, dict):
                continue
            step: dict[str, str | int] = {"action": str(entry.get("action") or "click")}
            selector = entry.get("selector")
            if selector:
                step["selector"] = str(selector)
            settle = entry.get("settle_ms")
            if settle is not None:
                step["settle_ms"] = int(settle)
            for key in ("goto", "scroll_to", "wait_for", "viewport"):
                val = entry.get(key)
                if val:
                    step[key] = str(val)
            cid = entry.get("capture_id") or entry.get("id")
            if cid:
                step["capture_id"] = str(cid)
                label_by_id[str(cid)] = str(entry.get("label") or cid)
            steps.append(step)
        return steps, label_by_id

    @staticmethod
    def _unzip_screens(zip_bytes: bytes) -> list[tuple[str, bytes]]:
        """Unpack the self-describing screenshot zip → ``[(screen_id, png), …]``.

        Entry names are ``<capture_id>.png``; the stem is the screen id.
        """
        out: list[tuple[str, bytes]] = []
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".png"):
                    continue
                screen_id = name.rsplit("/", 1)[-1].removesuffix(".png")
                if screen_id:
                    out.append((screen_id, zf.read(name)))
        return out

    @staticmethod
    def _unzip_screen_clips(zip_bytes: bytes) -> list[tuple[str, bytes]]:
        """Unpack the per-screen interaction clips → ``[(screen_id, mp4), …]``.

        The clips-mode zip holds both ``<capture_id>.png`` stills and
        ``<capture_id>.mp4`` clips; this picks out the clips (stem = screen id).
        """
        out: list[tuple[str, bytes]] = []
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".mp4"):
                    continue
                screen_id = name.rsplit("/", 1)[-1].removesuffix(".mp4")
                if screen_id:
                    out.append((screen_id, zf.read(name)))
        return out

    @staticmethod
    def _unzip_screen_tracks(zip_bytes: bytes) -> list[tuple[str, dict[str, object]]]:
        """Unpack the per-screen VP-2 capture sidecars → ``[(screen_id, track), …]``.

        The directed-footage zip carries, alongside each ``<capture_id>.png`` still, a
        ``<capture_id>.track.json`` sidecar (frame-pixel focus/cursor geometry). Each
        is parsed DEFENSIVELY via :meth:`_coerce_track` — a malformed, empty, or
        geometry-less sidecar is dropped rather than raising, because capturing the
        directed-footage track is an enhancement, never a precondition. The stem is
        the screen id (mirrors :meth:`_unzip_screen_clips`).
        """
        out: list[tuple[str, dict[str, object]]] = []
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".track.json"):
                    continue
                screen_id = name.rsplit("/", 1)[-1].removesuffix(".track.json")
                if not screen_id:
                    continue
                track = SiteCaptureStage._coerce_track(zf.read(name))
                if track is not None:
                    out.append((screen_id, track))
        return out

    @staticmethod
    def _coerce_track(raw: bytes) -> dict[str, object] | None:
        """Defensively parse a ``<id>.track.json`` sidecar → normalized
        ``{"focus", "cursor", "viewport"}`` or ``None`` — NEVER raises.

        Bad JSON, a non-object payload, or a track with neither a ``focus`` nor a
        ``cursor`` keyframe list degrades to ``None`` (that screen then carries no
        geometry and renders exactly as today). Only the three consumed keys are
        surfaced, each shape-checked, so a screen that gains stray keys still yields a
        lean, predictable shot payload. ``focus``/``cursor`` are dropped when empty so
        the presence-is-opt-in shot threading never stamps a no-op ``[]``.
        """
        try:
            parsed = json.loads(raw)
        except ValueError, TypeError:
            return None
        if not isinstance(parsed, dict):
            return None
        focus = parsed.get("focus")
        cursor = parsed.get("cursor")
        viewport = parsed.get("viewport")
        focus_list = [k for k in focus if isinstance(k, dict)] if isinstance(focus, list) else []
        cursor_list = [k for k in cursor if isinstance(k, dict)] if isinstance(cursor, list) else []
        if not focus_list and not cursor_list:
            return None
        track: dict[str, object] = {}
        if focus_list:
            track["focus"] = focus_list
        if cursor_list:
            track["cursor"] = cursor_list
        if isinstance(viewport, dict):
            track["viewport"] = viewport
        return track

    @staticmethod
    def _resolve_targets(source_url: str, cfg: dict) -> list[_Target]:
        """Resolve ``(url, path, interactions)`` per recorded path.

        Each ``paths`` entry is either a relative-path string OR a mapping
        ``{"path": "/demo", "interactions": [{"action":"click","selector":...}, ...]}``.
        Interactions drive a scripted walkthrough of that page (the recorder
        clicks through it while recording) instead of a plain scroll, and mark
        that clip as the primary demo. De-duped by resolved URL, then capped at
        ``max_clips``.
        """
        paths = cfg.get("paths")
        max_clips = int(cfg.get("max_clips") or _DEFAULT_MAX_CLIPS)
        entries: list[Any] = paths if (isinstance(paths, list) and paths) else [""]
        out: list[_Target] = []
        seen: set[str] = set()
        for entry in entries:
            if isinstance(entry, dict):
                path = str(entry.get("path") or entry.get("url") or "")
                interactions = entry.get("interactions") or None
            else:
                path, interactions = str(entry), None
            url = urljoin(source_url, path)
            if url in seen:
                continue
            seen.add(url)
            out.append((url, path, interactions))
        return out[: max(1, max_clips)]

    @staticmethod
    def _build_recording_config(
        cfg: dict, quality_preset: str | None = None, footage: dict | None = None
    ) -> dict[str, Any] | None:
        viewport = cfg.get("viewport") or {}
        out: dict[str, Any] = {}
        # Tier-derived recording resolution. The recorder applies the preset's
        # viewport, then lets any explicit viewport_width/height below override it,
        # so a brief that pins a viewport still wins.
        if quality_preset:
            out["quality_preset"] = quality_preset
        if viewport.get("width"):
            out["viewport_width"] = int(viewport["width"])
        if viewport.get("height"):
            out["viewport_height"] = int(viewport["height"])
        # Settle-before-capture: default on (fixes blank-shell white screens),
        # overridable per brief. Setting it also enables the recorder's
        # blank-lead trim so the clip starts on rendered content.
        delay = cfg.get("post_load_delay_ms")
        out["post_load_delay_ms"] = int(delay) if delay is not None else _DEFAULT_POST_LOAD_DELAY_MS
        selector = cfg.get("wait_for_selector")
        if selector:
            out["wait_for_selector"] = str(selector)
        # ORCH-3.2 auto-zoom sharpness: capture at 2x device-scale so the compiler's
        # auto-zoom punch-ins crop real pixels instead of upscaling. Opt-in via
        # ``direction.footage.auto_zoom``; absent/falsy => the key is omitted (recorder
        # default 1.0) => byte-identical recording request.
        if isinstance(footage, dict) and footage.get("auto_zoom"):
            out["capture_scale"] = 2.0
        return out or None

    @classmethod
    async def _scrape_page_text(
        cls, url: str, notes: list[str], degradations: list[Degradation] | None = None
    ) -> tuple[str, str]:
        """Best-effort fetch of the page's visible text to ground the script.

        Soft-fails to "" on any error. Refuses internal targets (loopback /
        private / link-local / **CGNAT**, this estate's Tailnet)
        as an SSRF guard — the user-supplied source_url is otherwise an open
        fetch sink. The fetch itself goes through ``w2v_common.net.safe_get``,
        which pins the connection to the validated IP (closing the DNS-rebinding
        TOCTOU the old separate ``getaddrinfo`` left open) and re-validates every
        redirect hop (the old client ran ``follow_redirects=True``, so a 302 to an
        internal address escaped the guard entirely).

        The response body is capped at :data:`_SCRAPE_MAX_RESPONSE_BYTES`,
        stated explicitly rather than inherited from ``safe_get``'s 64 MiB
        media-sized default. Every degrade path here still yields ``""`` for the
        text, so the caller also receives a **status** — ``"blocked"`` (SSRF guard
        refused the target), ``"fetch_failed"`` (the ``except`` below, which an
        over-cap body lands in), ``"empty"`` (200 OK but nothing extractable),
        ``"thin"`` (readable, but under :data:`_MIN_GROUNDING_CHARS` — the usual
        JS-only-SPA outcome, since stripping tags still surfaces ``<title>`` and
        the ``<noscript>`` line) or ``"ok"`` — and can branch on it without
        parsing note prose. ``"empty"`` and ``"thin"`` are both noted; between
        them they were the **silent** paths, producing a green, undegraded run
        whose script the downstream LLM had to invent. Note that ``"thin"``
        still carries its text: the honest move is to hand over what little was
        read and forbid extrapolating past it, not to discard it. Because that
        silent loss of
        script grounding is what the cap triggers, the ceiling belongs here, next
        to the degrade, not in a shared default chosen for a different fetch.
        """
        # Callers that only read the prose (older unit tests) may omit the list.
        coded: list[Degradation] = degradations if degradations is not None else []
        # Prose (notes, degradations, logs) never carries basic-auth userinfo; the
        # fetch itself still uses ``url`` as given.
        shown = _redact_userinfo(url)
        if not cls._safe_to_fetch(url):
            _degrade(
                notes,
                coded,
                SITE_PAGE_TEXT_MISSING,
                f"page scrape skipped (target not allowed): {shown}",
            )
            return ("", "blocked")
        try:
            async with httpx.AsyncClient(timeout=_SCRAPE_TIMEOUT_S) as client:
                resp = await safe_get(
                    client,
                    url,
                    headers={"User-Agent": "w2v-site-capture/1.0"},
                    max_fetch_bytes=_SCRAPE_MAX_RESPONSE_BYTES,
                )
                resp.raise_for_status()
                text = cls._extract_text(resp.text)
                # A 200 with no readable text is a real degrade, not a success:
                # note it so ``runs/_shared.py`` marks the run degraded.
                if not text.strip():
                    _degrade(
                        notes,
                        coded,
                        SITE_PAGE_TEXT_MISSING,
                        f"page scrape returned no readable text for {shown} "
                        "(empty or script-only page)",
                    )
                    return ("", "empty")
                # A JS-only shell still yields its <title> + <noscript> line, so the
                # empty check above almost never fires on a real SPA. Treating those
                # few characters as "copy from the live product site" is the same
                # falsehood in a costume — keep the text, but say it is not enough.
                if len(text.strip()) < _MIN_GROUNDING_CHARS:
                    _degrade(
                        notes,
                        coded,
                        SITE_PAGE_TEXT_THIN,
                        f"page scrape read only {len(text.strip())} characters from {shown} "
                        f"(under the {_MIN_GROUNDING_CHARS}-character grounding floor; "
                        "likely a JavaScript-rendered page)",
                    )
                    return (text, "thin")
                return (text, "ok")
        except Exception as exc:  # noqa: BLE001 -- best-effort scrape: any failure → "" (degrade)
            log.warning("site_capture: page scrape of %s failed: %s", shown, _redact_text(exc))
            _degrade(
                notes,
                coded,
                SITE_PAGE_TEXT_MISSING,
                f"page scrape failed for {shown}: {_redact_text(exc)}",
            )
            return ("", "fetch_failed")

    @staticmethod
    def _safe_to_fetch(url: str) -> bool:
        """Cheap pre-flight so a blocked target yields the 'skipped' note, not 'failed'.

        Delegates to the shared :func:`validate_public_url` (fail-closed on an
        unresolvable host). ``safe_get`` re-validates on the real fetch — this
        only preserves the distinct note text for an obviously-internal target.
        """
        try:
            validate_public_url(url)
        except ValueError:  # SSRFError subclasses ValueError; bad scheme raises it too
            return False
        return True

    @staticmethod
    def _extract_text(html: str) -> str:
        """Strip tags to a rough text body, capped. Intentionally dependency-free."""
        # Drop script/style blocks, then all tags, then collapse whitespace.
        no_blocks = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
        no_tags = re.sub(r"(?s)<[^>]+>", " ", no_blocks)
        text = re.sub(r"\s+", " ", no_tags).strip()
        return text[:_PAGE_TEXT_MAX_CHARS]
