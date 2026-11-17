"""app_inspect — call AppInspectorClient on target + competitor URLs; emit typed facts.

Facts are derived from what app inspection ACTUALLY produces — the inferred app
purpose, per-screen summaries, an accessibility score, the navigation graph, and
(when available) a screenshot. Earlier this stage read ``title/tagline/features/
pricing_lines`` off the inspection, a website-review shape that exists on NEITHER
server model, so every fact was silently empty on the live wire. See
``src/clients/schemas/app_inspector.py`` and memory ``w2v-app-inspector-dead-signal``.

Competitor inspections run with bounded concurrency via ``stage_gather`` —
a brief listing 6+ competitor URLs would otherwise serialise the inspect
calls and turn a 30-second pass into a 3-minute one. Per-item failures
are logged and dropped (competitor fetches are best-effort), so the
gather uses ``per_item_retries=0`` and catches its own exceptions before
they reach ``stage_gather``'s atomic-fail semantics.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar
from uuid import uuid7

import httpx
from pydantic import ValidationError
from w2v_common.stages import stage_call, stage_gather

from src.clients.app_inspector_client import AppInspectorClient
from src.clients.schemas.app_inspector import InspectionResult, ScreenSummary
from src.services.providers.errors import SlotUnavailableError
from src.services.providers.registry import KIND_BROWSERSTACK
from src.services.providers.slot_payload import build_provider_slot
from src.services.stages.base import (
    SlotUse,
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
)
from src.services.stages.upstream import PermanentUpstreamError, TransientUpstreamError

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

log = logging.getLogger(__name__)

_TARGET_TIMEOUT_SECONDS = 600.0
_COMPETITOR_TIMEOUT_SECONDS = 300.0
_COMPETITOR_CONCURRENCY = 3
_SCREEN_TEXT_MAX_CHARS = 300
_MAX_NAV_EDGES_IN_FACT = 8


async def _device_farm_config() -> dict[str, object] | None:
    """app_inspector ``DeviceFarmConfig`` from the run's pinned TENANT ``device_farm``.

    ``None`` (no farm field sent — today's local inspection) unless the run pinned a
    tenant connection. Contract mapping for kind ``browserstack``:
    ``options.username`` → ``browserstack_username``, ``secret`` →
    ``browserstack_access_key``. The returned dict carries the access key in clear
    for the request body only — never log it."""
    slot = await build_provider_slot("device_farm")
    if slot is None:
        return None
    if slot.get("kind") != KIND_BROWSERSTACK:
        raise SlotUnavailableError("device_farm", f"kind {slot.get('kind')!r} is not supported")
    options = slot.get("options")
    username = options.get("username") if isinstance(options, dict) else None
    secret = slot.get("secret")
    if not isinstance(username, str) or not username or not isinstance(secret, str) or not secret:
        raise SlotUnavailableError(
            "device_farm", "the BrowserStack connection needs a username and an access key"
        )
    return {
        "provider": KIND_BROWSERSTACK,
        "browserstack_username": username,
        "browserstack_access_key": secret,
    }


def _url_shaped(value: str | None) -> bool:
    """True only for an http(s) URL — a pod-local screenshot path is NOT usable.

    Under ``STORAGE_BACKEND=local`` the server returns a pod-local file path for
    ``screenshot_path``; feeding that to the compositor as an artifact_url would
    silently break the shot. Only a presigned URL (``=minio``) is surfaced.
    Scheme match is case-insensitive so a valid ``HTTPS://`` is not silently dropped.
    """
    return bool(value) and value.lower().startswith(("http://", "https://"))


#: The ONLY status that denies an artifact exists. ``"unknown"`` is the server's
#: default and means UNMEASURED — a pre-#92 server omits the field entirely — so it
#: must NOT be treated as a failure or every screenshot vanishes on an old server.
_SCREENSHOT_FAILED = "failed"


def _usable_screenshot(screen: ScreenSummary) -> bool:
    """True when this screen's ``screenshot_path`` may be surfaced as a visual.

    Two independent conditions, and the second is the one app_inspector #92 added:
    a capture the server has EXPLICITLY marked ``failed`` still carries a
    non-empty, perfectly URL-shaped ``screenshot_path`` to a file that was never
    written. ``_url_shaped`` alone cannot tell that apart from a real capture, so
    the reference rode through, ``app_review_record``'s "no screenshot_url" note
    never fired (the url is non-empty!), and ``app_review_compile`` filtered the
    dead clip out — an all-text-card video on a clean green run.
    """
    return _url_shaped(screen.screenshot_path) and screen.screenshot_status != _SCREENSHOT_FAILED


def _screen_fact_text(screen: ScreenSummary) -> str:
    parts: list[str] = []
    label = screen.screen_type or "screen"
    if screen.activity_name:
        label = f"{label} ({screen.activity_name})"
    parts.append(label)
    if screen.headings:
        parts.append("headings: " + "; ".join(screen.headings))
    if screen.text_content:
        excerpt = screen.text_content.strip()[:_SCREEN_TEXT_MAX_CHARS]
        if excerpt:
            parts.append(f"text: {excerpt}")
    # None means the screen was never scored. Rendering "None/100" into an LLM
    # prompt reads as a real measurement of zero-ish quality; say so instead.
    if screen.accessibility_score is None:
        parts.append("accessibility not measured")
    else:
        parts.append(f"accessibility {screen.accessibility_score}/100")
    # Say it in the LLM-visible text too: the critique stage reads these facts,
    # and "this screen has no usable screenshot" is context it should have.
    if screen.screenshot_status == _SCREENSHOT_FAILED:
        parts.append("screenshot capture FAILED")
    return " — ".join(parts)


def _degrade_notes(inspection: InspectionResult, screenshot_url: str) -> list[str]:
    """The visibility half of the fix — a NOTE, never a raise.

    ``artifacts_degraded`` is DERIVED server-side (app_inspector #92) and is the
    only signal that distinguishes "complete in shape" from "actually carries
    artifacts". Before it, an inspection that discovered screens and captured
    nothing looked identical to a good one on every field this stage reads, so
    the run finished clean green and the defect surfaced only downstream as a
    video full of text cards. A note marks the run ``degraded`` and leaves it
    watchable.
    """
    if not inspection.artifacts_degraded:
        return []
    failed = sum(1 for s in inspection.screens if s.screenshot_status == _SCREENSHOT_FAILED)
    detail = (
        f"{failed} screen(s) report a FAILED capture"
        if failed
        else "no screen carries a screenshot"
    )
    return [
        "app_inspect: the inspection is short on artifacts "
        f"({inspection.screenshots_captured} screenshot(s) captured across "
        f"{inspection.screens_discovered} screen(s); {detail}). "
        f"usable screenshot_url={screenshot_url or '<none>'}"
    ]


class AppInspectInput(StageInput):
    pass


class AppFact(StageOutput):
    id: str
    kind: str  # "purpose" | "screen" | "accessibility" | "navigation" | "screenshot"
    text: str
    source_url: str | None = None


class AppInspectOutput(StageOutputWithNotes):
    """``StageOutputWithNotes``, not a plain ``StageOutput``.

    ``app_inspect`` has a real degrade mode and, until the server grew
    ``artifacts_degraded``, no channel to report it on: an inspection that
    discovered screens and captured NOTHING is complete in shape, so every check
    downstream passed and the run finished clean green. A non-empty ``notes``
    renders the run ``degraded`` (``api/routes/runs/_shared.py``) — visible, but
    NOT a failure. Failing here would be worse than the bug: app_inspect
    legitimately degrades on missing input (measured 194 completed / 62 failed on
    the live DB), and a degraded-but-watchable review beats a dead run.
    """

    target_url: str
    facts: list[AppFact]
    screenshot_url: str
    competitors: list[dict]
    #: Mirrored from the server's DERIVED fields so a consumer (and the persisted
    #: stage row) can gate on them without re-deriving from ``facts``.
    screenshots_captured: int = 0
    artifacts_degraded: bool = False


class AppInspectStage(StageHandler[AppInspectInput, AppInspectOutput]):
    name = "app_inspect"
    input_model = AppInspectInput
    output_model = AppInspectOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("device_farm"),)

    def __init__(self, client: AppInspectorClient | None = None) -> None:
        self._client = client or AppInspectorClient()

    @staticmethod
    def _facts_from_inspection(
        inspection: InspectionResult, target: str
    ) -> tuple[list[AppFact], str]:
        facts: list[AppFact] = []

        def add(kind: str, text: str) -> None:
            facts.append(AppFact(id=str(uuid7()), kind=kind, text=text, source_url=target))

        if inspection.estimated_purpose:
            add("purpose", inspection.estimated_purpose.strip())

        for screen in inspection.screens:
            add("screen", _screen_fact_text(screen))

        add(
            "accessibility",
            (
                f"Accessibility not measured across {inspection.screens_discovered} screen(s)"
                if inspection.overall_accessibility_score is None
                else f"Accessibility {inspection.overall_accessibility_score}/100 "
                f"across {inspection.screens_discovered} screen(s)"
            ),
        )

        if inspection.navigation_graph:
            edges = inspection.navigation_graph[:_MAX_NAV_EDGES_IN_FACT]
            summary = ", ".join(f"{e.from_screen}→{e.to_screen}" for e in edges)
            add(
                "navigation",
                f"{len(inspection.navigation_graph)} navigation transition(s): {summary}",
            )

        # Only a URL-shaped screenshot the server has not marked failed is a
        # usable visual (see _usable_screenshot).
        screenshot_url = next(
            (s.screenshot_path for s in inspection.screens if _usable_screenshot(s)),
            None,
        )
        screenshot_url = screenshot_url or ""
        if screenshot_url:
            add("screenshot", screenshot_url)
        return facts, screenshot_url

    async def handle(
        self,
        payload: AppInspectInput,
        ctx: RunContext,
    ) -> AppInspectOutput:
        brief = ctx.config_snapshot.get("brief") or {}
        # Resolve the target from the SAME three keys every other stage accepts —
        # `demo_video/gen.py`, `demo_video/record.py` and `api/routes/projects.py`'s
        # `source_url` projection all read `target_url or source_url or url`. This stage
        # read `target_url` ALONE, which is why an `app_review` project created with the
        # documented legacy `brief.url` (see tests/api/routes/test_projects.py
        # ::test_source_url_projected_from_brief_url) was accepted at the door and then
        # died here — one paid stage deep. Measured 2026-08-05 on the live DB:
        # app_inspect 194 completed / 62 failed, and all 62 carried this RuntimeError.
        target = brief.get("target_url") or brief.get("source_url") or brief.get("url")
        if not target:
            raise RuntimeError("brief requires one of target_url / source_url / url for app_review")
        # Resolved once per stage: target AND competitors run on the tenant's farm.
        device_farm = await _device_farm_config()
        farm_config = {"device_farm": device_farm} if device_farm is not None else None

        async def _inspect(url: str) -> InspectionResult:
            # Without a tenant farm the call is exactly the pre-P5 one (no config).
            if farm_config is None:
                return await self._client.inspect_app(url)
            return await self._client.inspect_app(url, farm_config)

        target_inspection = await stage_call(
            lambda: _inspect(str(target)),
            timeout_seconds=_TARGET_TIMEOUT_SECONDS,
            retries=0,  # non-idempotent submit: a retry would start a duplicate inspection
            stage_name="app_inspect.target",
        )
        facts, screenshot_url = self._facts_from_inspection(target_inspection, str(target))

        async def _inspect_competitor(url: object) -> dict | None:
            try:
                competitor = await _inspect(str(url))
            except (
                httpx.HTTPError,
                OSError,
                TimeoutError,
                RuntimeError,
                ValidationError,
                PermanentUpstreamError,
                TransientUpstreamError,
            ) as exc:
                # Competitors are ADVISORY context: every per-competitor failure must
                # degrade to None so the PAID target inspection above still ships.
                # ``ValidationError`` belongs here because ``_fetch_result`` ends in
                # ``InspectionResult.model_validate(raw)`` (app_inspector_client.py:70)
                # and pydantic's ValidationError derives from ValueError — NOT from
                # HTTPError/OSError/TimeoutError/RuntimeError. Without it, one
                # schema-drifted competitor body escaped stage_gather and failed the
                # whole app_inspect stage after the target had already succeeded.
                # ``Permanent/TransientUpstreamError`` belong here for the SAME reason,
                # added 2026-08-19 alongside routing ``_fetch_result`` through
                # ``classify``: they subclass plain ``Exception`` and match NONE of the
                # types above, so without them one competitor 503 would escape
                # ``stage_gather`` and fail the whole PAID stage after the target had
                # already succeeded — the exact regression ``ValidationError`` above
                # documents, one class later.
                # NOT included: httpx.InvalidURL. It is genuinely outside HTTPError,
                # but it is unreachable here — the competitor URL only ever lands in
                # the JSON body (``payload["app_path"]``) and every request path is a
                # literal, so it never reaches httpx URL parsing.
                log.warning("competitor fetch failed for %s: %s", url, exc)
                return None
            top_screen_types = [s.screen_type for s in competitor.screens if s.screen_type][:5]
            return {
                "url": str(url),
                "purpose": (competitor.estimated_purpose or "").strip(),
                "screen_count": competitor.screens_discovered,
                "top_screen_types": top_screen_types,
                "accessibility": competitor.overall_accessibility_score,
            }

        competitor_urls = list(brief.get("competitor_urls") or [])
        competitor_results = await stage_gather(
            competitor_urls,
            _inspect_competitor,
            concurrency=_COMPETITOR_CONCURRENCY,
            per_item_timeout_seconds=_COMPETITOR_TIMEOUT_SECONDS,
            per_item_retries=0,
            stage_name="app_inspect.competitor",
        )
        competitors = [c for c in competitor_results if c is not None]
        return AppInspectOutput(
            target_url=str(target),
            facts=facts,
            screenshot_url=screenshot_url,
            competitors=competitors,
            screenshots_captured=target_inspection.screenshots_captured,
            artifacts_degraded=target_inspection.artifacts_degraded,
            notes=_degrade_notes(target_inspection, screenshot_url),
        )
