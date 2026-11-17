"""HTTP client for the html_render_service web-design API (``/api/designs``).

Replaces ``RedesignClient`` (which targeted analysis_service's redesign domain).
The web-design capability now lives in its own service, resolved via discovery
name ``html_render``. The canonical surface is ``/api/designs`` with a ``scope``
vocabulary (hero / section / component / landing / full_page / redesign); the
service returns screenshot artifacts as absolute, fetchable URLs.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import ClassVar

import httpx
from w2v_common.auth.identity_context import (
    EFFECTIVE_TENANT_HEADER,
    get_tenant_id_from_context,
    is_tenant_vouched,
)
from w2v_common.clients.base import JobCancelledError, raise_for_status_with_body
from w2v_common.tenant import UnvouchedTenantError

from src.clients.base import LicensedServiceClient
from src.clients.downstream_credential import service_scope
from src.clients.schemas.design import DesignJobResult, DesignJobStatus
from src.models.failure_class import EXTERNAL, FailureClass

log = logging.getLogger(__name__)


class DesignJobFailedError(RuntimeError):
    """html_render_service accepted the design job, ran it, and reported it FAILED.

    Mirrors ``clients.video_compile.VideoCompileJobFailedError`` — same defect,
    same fix. ``BaseServiceClient.poll_job`` (the shared w2v_common helper
    ``wait_for_design`` calls directly, with no per-service wrapper) raises a
    bare ``RuntimeError`` on a terminal ``failed``/``error`` status, carrying no
    structural signal at all. ``upstream.guarded`` deliberately lets that
    ``RuntimeError`` through untouched (see its docstring: "poll_job's terminal-
    job signal"), so it reached ``models.failure_class.attribute_exception``
    unstamped and came back ``unknown`` — the live shape behind
    ``StageProducedNothingError``/``StageRetriesExhaustedError`` messages like
    "generate_full_page: LLM reply is not a storable page (does not end on a
    closed tag (truncated))".

    ``failure_class = external``, stamped on the CLASS, for the identical
    reason as the video-compile sibling: html_render ran the job and told us it
    failed. That is a materially different finding from "the orchestrator has a
    bug", so it must not fold into ``our_defect``.

    Deliberately does NOT catch ``JobCancelledError`` (a ``RuntimeError``
    subclass poll_job raises for a *cancelled* job, not a failed one) — a user
    cancel is not an unattributed failure and must keep propagating as itself.
    """

    failure_class: ClassVar[FailureClass] = EXTERNAL


# The status html_render writes on insert and replaces with "processing" the
# moment a worker claims the job (its ``job_manager`` — the wire value is the
# column verbatim, ``jobs.py:146`` does ``status=job.status``, with no mapping
# layer). Checking for THIS string rather than a list of started-states is the
# fail-safe direction: if the service ever renames it, ``wait_until_started``
# returns immediately and the caller is back to today's behaviour, never worse.
_QUEUED_STATUS = "pending"

# Ceiling on how long a submitted design may sit unclaimed before the stage
# gives up. Generous on purpose: waiting here is FREE (no worker is running),
# whereas expiring the wait throws away a job the service will still complete.
# Only a genuinely stalled queue — every html_render replica wedged or gone —
# should ever reach it.
_QUEUE_TIMEOUT_SECONDS = 1800.0


async def _effective_tenant_event_hook(request: httpx.Request) -> None:
    """Name the RUN's tenant on every html_render hop, beside the SA token.

    WHY THIS EXISTS AT ALL. ``DesignClient`` authenticates with the
    orchestrator's service-account licence in ``X-Tenant-Id: Bearer <token>``,
    which html_render introspects to the SA TENANT
    ``01a1113b-674f-72b0-a84b-dc746d39afca``. Its ``_get_tenant_id`` then stamps
    that onto the ``redesign_jobs`` row it inserts, and scopes every subsequent
    read of that row by it. So every customer's render was written under the
    orchestrator's own tenant: the requesting tenant's fetch of its OWN job
    404s forever (``/api/redesign/jobs/{id}`` is a pure proxy that forwards the
    CALLER's tenant), and the render is unfetchable, unscoreable and unbillable.

    The shared ``identity_event_hook`` already knows how to say "authenticate as
    me, act for THEM" — ``X-Effective-Tenant-Id`` — but it only does so for a
    destination in ``w2v_common.auth.outbound.METERED_DESTINATION_SERVICES``,
    and ``html_render`` is not one of them (same shape as the
    ``video_production`` omission documented there). Classifying html_render as
    metered lives in another repo and would also change its billing semantics;
    the delegation is what is actually needed, so the client asserts it here —
    the same one-liner ``api.routes._proxy.forward_headers`` already uses for
    the read side of exactly these jobs.

    NOT inert: html-render-service runs the shared ``w2v_common`` AuthMiddleware
    and its prod ConfigMap sets ``AUTH_DELEGATING_SUBJECTS`` to the shared w2v
    service-account subject, so the header is honoured and ``request.state.
    tenant_id`` becomes the delegated tenant that the job row is stamped with.

    Fails closed exactly like ``_apply_outbound_tenant``: a tenant whose
    provenance is not vouched is never delegated. A hop with NO tenant bound is
    left alone — the SA-credentialed call is still authenticated and attributable
    to the SA tenant, which is the one tolerated case (startup probes,
    ``design_job_exists`` running outside a run task).
    """
    if request.headers.get(EFFECTIVE_TENANT_HEADER):
        return
    tenant = get_tenant_id_from_context()
    if tenant is None:
        return
    if not is_tenant_vouched():
        raise UnvouchedTenantError(
            f"outbound {request.method} {request.url}: refusing to delegate tenant "
            f"{tenant} to html_render — its provenance is not vouched, and a bare "
            "X-Tenant-Id is settable by any peer that can reach this service."
        )
    request.headers[EFFECTIVE_TENANT_HEADER] = str(tenant)


class DesignClient(LicensedServiceClient):
    """Client for the html_render_service ``/api/designs`` API.

    The service is behind license-authz (only ``/health*`` is exempt), so the
    orchestrator's service-account licence token is baked into
    ``X-Tenant-Id: Bearer <token>`` by ``BaseServiceClient`` — anonymous calls
    get 401. Same pattern as ``TTSClient`` / ``VideoCompileClient``.
    """

    def __init__(
        self,
        base_url: str | None = None,
        license_token: str | None = None,
    ) -> None:
        super().__init__(
            service_name="design",
            discovery_name="html_render",
            base_url=base_url,
            license_token=license_token,
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """The base client, plus the effective-tenant delegation hook.

        Appended rather than passed to the constructor because
        ``BaseServiceClient._get_client`` fixes its own ``event_hooks``; adding
        to the built client keeps the shared hooks (identity, tracing) and their
        order intact. Idempotent — ``_get_client`` is called per request and the
        client is cached, but a reopened client must get the hook again.

        Wires onto a REAL ``httpx.AsyncClient`` only. Existing tests inject an
        ``AsyncMock`` straight into ``self._client``, whose ``event_hooks`` is
        another ``AsyncMock`` — ``setdefault`` on it returns a coroutine and the
        membership test raises ``TypeError``. Those tests assert on the mocked
        ``request()`` call, never on wire headers, so skipping a client that has
        no real hook pipeline loses nothing; the header behaviour is pinned by
        ``tests/clients/test_design_client_effective_tenant.py``, which drives
        the genuine httpx pipeline.
        """
        client = await super()._get_client()
        if isinstance(client, httpx.AsyncClient):
            hooks = client.event_hooks.setdefault("request", [])
            if _effective_tenant_event_hook not in hooks:
                hooks.append(_effective_tenant_event_hook)
        return client

    async def start_design(
        self,
        url: str = "",
        scope: str = "full_page",
        style: str = "modern",
        critique_findings: list[dict[str, object]] | None = None,
        site_analysis: dict[str, object] | None = None,
        html_content: str | None = None,
        llm_config: dict[str, object] | None = None,
        *,
        mode: str | None = None,
        interactivity: str | None = None,
        component_set: str | None = None,
        embeds: list[dict[str, object]] | None = None,
        page_type: str | None = None,
        user_context: dict[str, object] | None = None,
        brand: dict[str, object] | None = None,
        generate_screenshots: bool | None = None,
        screenshot_viewports: list[str] | None = None,
        capture_original: bool | None = None,
        generate_comparison: bool | None = None,
        styles: list[str] | None = None,
        section_screenshots: bool | None = None,
        full_page: bool | None = None,
    ) -> str:
        """Start a design job, returns job_id.

        ``scope`` selects the generation path:
        ``hero`` (fast single-section — the promo fast path),
        ``full_page`` / ``redesign`` (whole-page chunked; redesign is
        critique-driven), ``section`` / ``component`` / ``landing``.

        The keyword params mirror html_render_service's ``RedesignRequest``
        (``src/api/schemas.py:261``). Every optional param defaults to ``None``
        = omit-from-payload, so the SERVER's defaults apply (they differ per
        field: ``generate_screenshots``/``capture_original``/
        ``generate_comparison``/``full_page`` default True there,
        ``section_screenshots`` False, ``mode``="freeform",
        ``interactivity``="static"). ``embeds`` must serialize as a list when
        given — html_render's ``embeds`` field is ``list[EmbedSpec]`` with no
        ``| None``, so an explicit JSON null 422s while an empty list (or an
        absent key) is accepted.
        """
        payload: dict[str, object] = {"url": url, "scope": scope, "style": style}
        if critique_findings:
            payload["critique_findings"] = critique_findings
        if site_analysis:
            payload["site_analysis"] = site_analysis
        if html_content:
            payload["html_content"] = html_content
        if llm_config:
            payload["llm_config"] = llm_config
        optional: dict[str, object | None] = {
            "mode": mode,
            "interactivity": interactivity,
            "component_set": component_set,
            "embeds": embeds,
            "page_type": page_type,
            "user_context": user_context,
            "brand": brand,
            "generate_screenshots": generate_screenshots,
            "screenshot_viewports": screenshot_viewports,
            "capture_original": capture_original,
            "generate_comparison": generate_comparison,
            "styles": styles,
            "section_screenshots": section_screenshots,
            "full_page": full_page,
        }
        payload.update({key: value for key, value in optional.items() if value is not None})
        return await self.submit_job("/api/designs", payload)

    async def get_design_status(self, job_id: str) -> DesignJobStatus:
        """Read a design job's CURRENT status once, without polling.

        Same endpoint ``wait_for_design`` polls, but a single shot. Two callers,
        both in ``web_design``: proving a stashed job is still alive before a
        resumed stage re-attaches to it instead of submitting a new one, and
        naming the last-seen status in a poll-timeout message (``queued`` for
        590s and ``processing`` for 590s are otherwise indistinguishable —
        html_render runs worker_pool_size=1, so a job can burn the caller's whole
        budget without the renderer logging a thing).
        """
        data = await self._request("GET", f"/api/designs/jobs/{job_id}")
        return DesignJobStatus.model_validate(data)

    async def get_design_result(self, job_id: str) -> DesignJobResult:
        """Get the full design artifacts/result for a completed job."""
        data = await self._request("GET", f"/api/designs/jobs/{job_id}/result")
        return DesignJobResult.model_validate(data)

    async def get_design_html(self, job_id: str) -> str:
        """Download generated HTML content for a completed job."""
        client = await self._get_client()
        response = await client.get(f"/api/designs/jobs/{job_id}/html")
        raise_for_status_with_body(response, "design")
        return response.text

    async def get_design_html_as_service(self, job_id: str) -> str:
        """:meth:`get_design_html` sent as the orchestrator SERVICE when no tenant is bound.

        The ONLY explicit service-scope opt-in on this client
        (``downstream_credential.service_scope``): the reconciler-finalize
        existence probe is the orchestrator's own work, so a tenant-less send
        under ``CELL_AUTH_MODE=enforce`` carries a ``scope=service`` cell token
        instead of failing closed. With a tenant bound it is
        :meth:`get_design_html` unchanged.
        """
        with service_scope(self.discovery_name):
            return await self.get_design_html(job_id)

    async def render_pdf(self, html: str) -> bytes:
        """Render a self-contained HTML string to PDF bytes (stateless Chromium print).

        Posts to ``/api/designs/render/pdf`` (no stored job) — used by the audit-report
        stage to turn a scorecard HTML into a downloadable PDF.
        """
        client = await self._get_client()
        response = await client.post("/api/designs/render/pdf", json={"html": html})
        raise_for_status_with_body(response, "design")
        return response.content

    async def start_refinement(
        self,
        parent_job_id: str,
        feedback: str,
        focus_areas: list[str] | None = None,
        generate_screenshots: bool = True,
        generate_concept_images: bool = False,
        llm_config: dict[str, object] | None = None,
    ) -> str:
        """Start a refinement job for an existing design."""
        payload: dict[str, object] = {
            "feedback": feedback,
            "generate_screenshots": generate_screenshots,
            "generate_concept_images": generate_concept_images,
        }
        if focus_areas:
            payload["focus_areas"] = focus_areas
        if llm_config:
            payload["llm_config"] = llm_config
        return await self.submit_job(f"/api/designs/jobs/{parent_job_id}/refine", payload)

    async def wait_for_design(self, job_id: str, timeout: float = 300.0) -> DesignJobStatus:
        """Poll until the design job completes; returns the final status body.

        ``accessibility_score`` rides on THIS status body (server
        ``JobStatusResponse``), not on the ``/result`` payload.

        A terminal ``failed``/``error`` status is re-raised as
        ``DesignJobFailedError`` (stamped ``failure_class = external``) rather
        than left as ``poll_job``'s bare, unattributed ``RuntimeError`` — see
        that class's docstring. ``JobCancelledError`` is a distinct
        ``RuntimeError`` subclass and passes through unchanged.
        """
        try:
            data = await self.poll_job("/api/designs/jobs", job_id, timeout=timeout)
        except JobCancelledError:
            raise
        except RuntimeError as exc:
            raise DesignJobFailedError(str(exc)) from exc
        return DesignJobStatus.model_validate(data)

    async def wait_until_started(
        self, job_id: str, timeout: float = _QUEUE_TIMEOUT_SECONDS, interval: float = 10.0
    ) -> float:
        """Block while the job is still ``pending`` in html_render's claim queue.

        Returns the seconds spent queued (0.0 if a worker had already claimed it).

        WHY this exists: ``wait_for_design``'s budget is a *generation* budget,
        but html_render is a PG claim-queue (``pending`` -> ``processing``, see
        its ``job_manager``) with a small replica count, so a submitted job can
        sit unclaimed for many minutes. Without this pre-phase that queue time is
        charged against generation, and a design that would have finished fails.

        Measured live 2026-08-10, run ``019fea75-98ee-70d5-84bb-044e12fddd49``:
        job ``019fec3f-…`` was submitted 15:17:18Z and not claimed until
        15:40:26Z — **23 minutes queued**. The stage's 8 x 590s budget expired at
        16:38:32Z having allowed only 58 min of generation; the design service
        went on to finish that same job successfully at 16:47:39Z. The run failed
        and a completed design was discarded.

        A job that leaves ``pending`` for any other state (including ``failed``)
        returns immediately — surfacing the real cause is ``wait_for_design``'s
        job, not this one's.
        """
        start = time.monotonic()
        while True:
            data = await self._request("GET", f"/api/designs/jobs/{job_id}")
            if str(data.get("status", "")) != _QUEUED_STATUS:
                queued_for = time.monotonic() - start
                if queued_for >= 60.0:
                    log.info(
                        "design job %s sat %.0fs in html_render's queue before a worker "
                        "claimed it — that time is NOT charged to the generation budget",
                        job_id,
                        queued_for,
                    )
                return queued_for
            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                raise TimeoutError(
                    f"design job {job_id} was never claimed by a worker "
                    f"(still 'pending' after {elapsed:.0f}s)"
                )
            await asyncio.sleep(min(interval, timeout - elapsed))


async def design_job_exists(job_id: str) -> bool | None:
    """Does html_render still serve this design job's HTML? ``True``/``False``/``None``.

    The existence probe for a blueprint whose deliverable is a design job rather
    than a stored object (both web-design blueprints). Same tri-state contract as
    ``StorageService.object_exists`` — only ``False`` (the service answered, and
    the job/HTML is gone) is safe to fail a run on; anything indeterminate
    (transport error, 5xx, auth) is ``None`` and never a verdict.

    It fetches the HTML rather than reading the job row, because the job row
    outliving its artifact is precisely the failure this must catch: the
    user-facing links (``/api/redesign/jobs/{id}/html|pdf``) proxy this exact
    upstream path, so what this call sees is what the download button sees. The
    PDF is minted on demand FROM this HTML, so one probe covers both links.

    ``404``/``410`` ⇒ ``False``. A 4xx that is about US rather than the artifact
    (401/403 — a service-account token problem) must not fail a finished run, so
    it stays indeterminate.
    """
    client = DesignClient()
    try:
        await client.get_design_html_as_service(job_id)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in (404, 410):
            return False
        return None
    except Exception:  # noqa: BLE001 -- transport/DNS/timeout: indeterminate, never a verdict
        return None
    else:
        return True
    finally:
        await client.close()
