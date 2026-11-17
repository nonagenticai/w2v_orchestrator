"""Register all w2v worker names in worker_management with the inference_router provider.

Why this exists
---------------
`worker_management` auto-creates unknown workers on first call with
``AUTO_CREATE_MODEL_PROVIDER=vllm, AUTO_CREATE_MODEL=local-llm`` (the
WM source-code default). The vllm endpoint is not reachable in this
cluster, so every newly-touched w2v worker 500s on its first call until
someone PUT-fixes it. See ``Movies/BUG-REPORT-wm-auto-create-default.md``.

This script PUTs every w2v-known worker name with
``model_provider=inference_router, model=<worker_name>`` so first-call
behavior is correct without depending on WM env tweaks.

Usage
-----
::

    # From inside any w2v pod (orchestrator, etc.):
    python scripts/register_w2v_workers.py

    # Or override the WM URL:
    WORKER_MANAGEMENT_URL=http://wm:8887 python scripts/register_w2v_workers.py

Idempotent — safe to re-run; ``PUT /workers/{name}`` upserts.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

import httpx

logger = logging.getLogger(__name__)


W2V_WORKER_NAMES: list[str] = [
    "bdd-feature-generator",
    "bdd-step-def-generator",
    "bdd-narration-humanizer",
    "content-script-llm",
    "narration-builder",
    "shotlist-enhancer",
    "storyline-generator",
    "course-curriculum-generator",
    "video-frame-analyzer",
    "video-synthesis",
    "slide-bullet-extractor",
    "analysis-site-summary",
    "critique-summary",
    "seo-intent-classify",
    "seo-eeat-depth",
    "seo-content-gap",
    "edit-plan-generator",
    "brand-analysis",
    "redesign-accessibility-scorer",
    "redesign-html",
    "critique-ux-review",
    "review-design_review",
    "review-usability_review",
    "review-feature_review",
    "review-content_quality_review",
    "review-feature_suggestions",
    "cadence-director",
    "motion-critic",
    "script-doctor-critique",
    "script-doctor-revise",
    "retention-critic",
    "hook-doctor",
    # reconcile_timeline C13 relief loop (A22)
    "pacing-critic",
    "timing-director",
    "beat-rewriter",
    "claim-grounder",
    "claim-softener",
    "voice-director",
    # Audiobook great-output DAG — director / pronunciation-proposer /
    # listen-back-judge. Declared here for inventory + SEED_MANAGED below so the
    # register script skips them and WM's seed owns their model + system_prompt.
    "audiobook-director",
    "pronunciation-proposer",
    "audiobook-listenback-judge",
]


# Workers OWNED by worker_management's seed (seed/w2v_workers.yaml): they carry
# a curated model + system_prompt there. This script must NOT re-register them —
# a blind PUT here replaces the seeded config (drops model=deepseek + the
# system_prompt), silently breaking them. We skip them; the seed owns their
# config (including max_tokens). Keep this in sync with the seed's worker_name
# list. (Their max_tokens belongs in the seed, not here.)
SEED_MANAGED: frozenset[str] = frozenset(
    {
        "brand-analysis",
        "analysis-site-summary",
        "course-curriculum-generator",
        "slide-bullet-extractor",
        "edit-plan-generator",
        "script-doctor-critique",
        "script-doctor-revise",
        "retention-critic",
        "hook-doctor",
        # ORCH-4.4 claim_finish (A18) grounder -> softener loop.
        "claim-grounder",
        "claim-softener",
        # html_render_service redesign workers — seeded with curated prompts.
        "redesign-html",
        "redesign-accessibility-scorer",
        # voice-performance director — seeded with curated model + system_prompt.
        "voice-director",
        # Audiobook great-output workers — WM seed owns their curated model
        # (local-llm strong tier) + system_prompt; a blind PUT here would drop
        # the seeded prompt and re-tier them to model=<name>.
        "audiobook-director",
        "pronunciation-proposer",
        "audiobook-listenback-judge",
        # analysis_service critique workers — WM seed pins them to temperature 0 so
        # a re-critique of the same page is repeatable; this script's blind PUT
        # would put them back to 0.1 (same site scored UX 59.3 vs 52.4 in a day).
        "critique-ux-review",
        "critique-summary",
    }
)


# Per-worker output-token cap. Most workers are fine on the provider default,
# but workers that emit a whole document need a much larger budget or the
# response is truncated mid-output. ``redesign-html`` returns a complete
# Tailwind HTML page (~16k tokens); left at the provider default (~4k) it
# truncates the page. WM accepts this on ``WorkerCreateRequest.max_tokens``.
# The per-call ``MessageRequest`` ALSO accepts ``max_tokens`` — worker_management
# ``src/api/workers/models.py:859`` defines the field and
# ``src/api/workers/message_endpoints.py:355-360`` folds it into the worker config
# for THAT message (None ⇒ keep the worker row's value; clamped to 16384 for
# self-hosted backends downstream). So a caller can override per call via
# ``WMClient.message(max_tokens=...)`` — see
# ``src/services/stages/demo_video/retention_plan.py``. The map below is the
# per-worker DEFAULT for callers that pass nothing.
#
# NOTE: ``redesign-html`` is now SEED_MANAGED, so this script no longer PUTs it
# and the cap below is inert for that worker — its ``max_tokens`` must live in
# WM's seed/w2v_workers.yaml. Kept here as the documented value.
WORKER_MAX_TOKENS: dict[str, int] = {
    "redesign-html": 16000,
    "content-script-llm": 8000,
    "bdd-step-def-generator": 6000,
}


async def register_one(client: httpx.AsyncClient, name: str) -> tuple[str, int, str]:
    body: dict[str, object] = {
        "worker_name": name,
        "agent_type": "pydantic_ai",
        "model_provider": "inference_router",
        "model": name,
        "temperature": 0.1,
        "agent_role": f"AI agent for {name}",
    }
    max_tokens = WORKER_MAX_TOKENS.get(name)
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    resp = await client.put(f"/workers/{name}", json=body, timeout=30)
    return name, resp.status_code, resp.text[:200]


async def main() -> int:
    base_url = os.environ.get(
        "WORKER_MANAGEMENT_URL",
        "http://worker-management.workers:8887",
    ).rstrip("/")
    license_token = os.environ.get("DOWNSTREAM_LICENSE_TOKEN", "")
    headers = {"X-Tenant-Id": f"Bearer {license_token}"} if license_token else {}

    # Skip seed-managed workers — re-registering them here clobbers their
    # curated seed config (model + system_prompt).
    to_register = [n for n in W2V_WORKER_NAMES if n not in SEED_MANAGED]
    skipped = [n for n in W2V_WORKER_NAMES if n in SEED_MANAGED]
    if skipped:
        logger.info("skipping %d seed-managed workers: %s", len(skipped), ", ".join(skipped))

    failures: list[tuple[str, int, str]] = []
    async with httpx.AsyncClient(base_url=base_url, headers=headers) as client:
        results = await asyncio.gather(
            *(register_one(client, n) for n in to_register),
            return_exceptions=False,
        )
        for name, status, snippet in results:
            if 200 <= status < 300:
                logger.info("registered %s -> %d", name, status)
            else:
                failures.append((name, status, snippet))
                logger.error("FAILED %s -> %d: %s", name, status, snippet)

    if failures:
        for name, status, snippet in failures:
            print(f"FAILED {name}: HTTP {status}: {snippet}", file=sys.stderr)
        return 1
    print(
        f"OK: registered {len(to_register)} w2v workers at {base_url} (skipped {len(skipped)} seed-managed)"
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(asyncio.run(main()))
