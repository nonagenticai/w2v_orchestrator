from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from w2v_common.clients import WMResponse

from src.services.runs.run_context import RunContext
from src.services.stages.base import StageError
from src.services.stages.growth_bundle.handler import GrowthBundleStage


def _wm_response(output: dict) -> WMResponse:
    return WMResponse(
        output=output,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


def _ctx_with_priors(prior: dict | None = None) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "title": "Anti-Churn",
            "brief": {"call_to_action": "Buy", "cta_url": "https://x"},
            "voice": {"descriptor": "crisp"},
        },
        prior_run_outputs=prior or {"promo_script": {"lines": [{"id": "L1"}]}},
    )


@pytest.mark.asyncio
async def test_growth_bundle_emits_4_channels_when_llm_does() -> None:
    fake = AsyncMock()
    fake.message.return_value = _wm_response(
        {
            "linkedin": {"text": "L", "cta_url": "https://x"},
            "twitter": {"text": "T", "cta_url": None},
            "email": {"text": "E", "cta_url": "https://x"},
            "landing_page": {"text": "P", "cta_url": None},
        }
    )
    stage = GrowthBundleStage(client=fake)

    out = await stage.run({}, ctx=_ctx_with_priors())

    assert {s.channel for s in out.snippets} == {
        "linkedin",
        "twitter",
        "email",
        "landing_page",
    }
    assert all(s.id for s in out.snippets)


@pytest.mark.asyncio
async def test_growth_bundle_uses_patient_budget_and_single_retry(monkeypatch) -> None:
    """Regression: the WM growth-bundle call must be PATIENT (>=600s, aligned
    with the WM/inference-router 600s stack) so it queues on the shared pgx-local
    vLLM instead of cutting itself off at 300s and re-queuing, and retry once (not
    twice) to avoid re-queue thrash under contention. Guards against silently
    reverting to the 300s/retries=2 that stalled promo/website_review."""
    from src.services.stages.growth_bundle import handler as gb
    from src.services.stages.llm_budget import llm_budget_seconds

    assert llm_budget_seconds() >= 600.0

    fake = AsyncMock()
    fake.message.return_value = _wm_response({"linkedin": {"text": "L", "cta_url": None}})
    stage = GrowthBundleStage(client=fake)

    captured: dict = {}

    async def _capture_stage_call(attempt, *, timeout_seconds, retries, stage_name, retry_on=None):
        captured["timeout_seconds"] = timeout_seconds
        captured["retries"] = retries
        captured["retry_on"] = retry_on
        return await attempt()

    monkeypatch.setattr(gb, "stage_call", _capture_stage_call)

    await stage.run({}, ctx=_ctx_with_priors())

    assert captured["timeout_seconds"] >= 600.0
    assert captured["retries"] == 1
    # The WM call now declares the permanent-vs-transient allow-list so a 402/
    # token-limit fails fast instead of burning the single retry.
    assert captured["retry_on"] is gb.LLM_RETRY_ON


@pytest.mark.asyncio
async def test_growth_bundle_raises_when_malformed() -> None:
    fake = AsyncMock()
    fake.message.return_value = _wm_response({})
    stage = GrowthBundleStage(client=fake)

    with pytest.raises(StageError):
        await stage.run({}, ctx=_ctx_with_priors())


@pytest.mark.asyncio
async def test_growth_bundle_drops_unknown_channel_and_raises_when_empty() -> None:
    fake = AsyncMock()
    fake.message.return_value = _wm_response({"tiktok": {"text": "x", "cta_url": None}})
    stage = GrowthBundleStage(client=fake)

    # tiktok is not a channel → nothing usable → StageError (no synthetic fallback).
    with pytest.raises(StageError):
        await stage.run({}, ctx=_ctx_with_priors())


# ─── showcase spike 01a0fd4d-9948: an empty bundle failed a delivered run ─────────

_DELIVERED = "/api/runs/r/artifact/w2v-compiled-videos/r/final.mp4"


def _showcase_ctx() -> RunContext:
    """A showcase run at growth_bundle: the video is delivered, there is no script
    stage, and the brief carries a sealed login block."""
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "type": "showcase",
            "title": "merqi showcase",
            "brief": {
                "brand": {"name": "merqi"},
                "auth_config": [
                    {"type": "form", "username": "u", "password": {"$sealed": "gAAAAsecret"}}
                ],
            },
        },
        prior_run_outputs={
            "demo_video_gen": {
                "lines": [
                    {"id": "l1", "text": "Give merqi a URL and a brief."},
                    {"id": "l3", "text": "And I review every claim before we say it."},
                ]
            },
            "final_critique": {"artifact_url": _DELIVERED, "duration_seconds": 16.6},
        },
    )


def test_output_schema_requires_every_channel_by_name() -> None:
    """The spike's model answered ``{"snippets": []}`` (8 completion tokens, both
    attempts): the array schema allowed an empty bundle and a free-text channel
    name. Required per-channel properties are grammar-enforceable; ``minItems`` is
    stripped by worker-management and would enforce nothing."""
    from src.services.stages.growth_bundle import handler as gb

    schema = gb._OUTPUT_SCHEMA
    assert schema["required"] == ["linkedin", "twitter", "email", "landing_page"]
    assert set(schema["properties"]) == set(schema["required"])
    for block in schema["properties"].values():
        assert block["required"] == ["text"]
    assert "snippets" not in schema["properties"]


@pytest.mark.asyncio
async def test_showcase_prompt_carries_the_spine_and_no_login_block() -> None:
    fake = AsyncMock()
    fake.message.return_value = _wm_response({"linkedin": {"text": "L", "cta_url": None}})
    stage = GrowthBundleStage(client=fake)

    await stage.run({}, ctx=_showcase_ctx())

    prompt = fake.message.call_args.kwargs["message"]
    assert "And I review every claim before we say it." in prompt
    assert "Prior content: {}" not in prompt
    assert "auth_config" not in prompt
    assert "gAAAAsecret" not in prompt


@pytest.mark.asyncio
async def test_no_secret_reaches_the_prompt_at_any_depth() -> None:
    """The prompt is built from an allowlist, so a login nested anywhere in the
    brief (sources[], site_capture, an auth list, a sealed value) cannot leak."""
    ctx = _showcase_ctx()
    ctx.config_snapshot["brief"] = {
        "brand": {"name": "merqi", "api_token": "BRANDTOKEN1"},
        "call_to_action": "Book a demo",
        "audience": ["growth leads"],
        "sources": [
            {
                "url": "https://x",
                "auth_info": {
                    "username": "u",
                    "password": "NESTEDSECRET2",  # pragma: allowlist secret
                },
            }
        ],
        "site_capture": {
            "auth_config": [{"type": "cookie", "cookies": [{"name": "s", "value": "COOKIE3"}]}],
            "headers": {"authorization": "Bearer HEADERTOKEN4"},
        },
        "auth_config": [{"type": "form", "password": {"$sealed": "gAAAASEALED5"}}],
        "description": {"$sealed": "gAAAASEALED6"},
        "value_proposition": "One brief in, a strategy out.",
    }
    fake = AsyncMock()
    fake.message.return_value = _wm_response({"linkedin": {"text": "L", "cta_url": None}})

    await GrowthBundleStage(client=fake).run({}, ctx=ctx)

    prompt = fake.message.call_args.kwargs["message"]
    for leaked in (
        "NESTEDSECRET2",
        "COOKIE3",
        "HEADERTOKEN4",
        "gAAAASEALED5",
        "gAAAASEALED6",
        "BRANDTOKEN1",
        "password",
        "auth_info",
        "authorization",
    ):
        assert leaked not in prompt, leaked
    # ...while the copy-relevant fields still arrive.
    assert "merqi" in prompt and "Book a demo" in prompt and "growth leads" in prompt
    assert "One brief in, a strategy out." in prompt


_CTA_SECRET = "CTASECRET7"  # pragma: allowlist secret


@pytest.mark.parametrize(
    "cta",
    [{"label": "Book", "password": _CTA_SECRET}],
    ids=["dict"],
)
@pytest.mark.asyncio
async def test_structured_cta_reaches_the_prompt_only_as_its_label(cta: object) -> None:
    """``CTA:`` once interpolated ``brief.call_to_action`` raw, so a password inside
    a CTA object (or a list holding one) went straight into the prompt."""
    ctx = _showcase_ctx()
    ctx.config_snapshot["brief"] = {"brand": {"name": "merqi"}, "call_to_action": cta}
    ctx.config_snapshot["voice"] = {"descriptor": {"text": "crisp", "token": _CTA_SECRET}}
    ctx.config_snapshot["title"] = {"text": "merqi showcase", "secret": _CTA_SECRET}
    fake = AsyncMock()
    fake.message.return_value = _wm_response({"linkedin": {"text": "L", "cta_url": None}})

    await GrowthBundleStage(client=fake).run({}, ctx=ctx)

    prompt = fake.message.call_args.kwargs["message"]
    assert _CTA_SECRET not in prompt
    assert "password" not in prompt
    assert 'CTA: "Book"' in prompt
    assert "Voice: crisp" in prompt and "Title: merqi showcase" in prompt


@pytest.mark.asyncio
async def test_prior_script_is_scrubbed_of_secret_keys() -> None:
    ctx = _ctx_with_priors(
        {
            "promo_script": {
                "lines": [{"id": "L1", "text": "Hello"}],
                "auth_info": {"password": _CTA_SECRET},
                "headers": {"authorization": _CTA_SECRET},
                "note": {"$sealed": _CTA_SECRET},
            }
        }
    )
    fake = AsyncMock()
    fake.message.return_value = _wm_response({"linkedin": {"text": "L", "cta_url": None}})

    await GrowthBundleStage(client=fake).run({}, ctx=ctx)

    prompt = fake.message.call_args.kwargs["message"]
    assert _CTA_SECRET not in prompt
    assert "Hello" in prompt


@pytest.mark.asyncio
async def test_transport_failure_after_delivery_still_fails_the_run() -> None:
    """Only an EMPTY reply degrades. A retried transport error that exhausts the
    budget after delivery must still fail, not be recorded as growth_bundle_empty."""
    from w2v_common.stages import StageRetriesExhaustedError

    fake = AsyncMock()
    fake.message.side_effect = httpx.ConnectError("worker-management down")
    stage = GrowthBundleStage(client=fake)

    with pytest.raises(StageRetriesExhaustedError):
        await stage.run({}, ctx=_showcase_ctx())
    assert fake.message.await_count == 2  # it was retried, then failed


@pytest.mark.asyncio
async def test_empty_bundle_after_delivery_degrades_instead_of_failing() -> None:
    """Every draw empty, video already delivered ⇒ the run keeps its video and the
    gap is a ``growth_bundle_empty`` degradation, not a failed run."""
    from src.services.stages.shared.degradation import GROWTH_BUNDLE_EMPTY

    fake = AsyncMock()
    fake.message.return_value = _wm_response({"snippets": []})
    stage = GrowthBundleStage(client=fake)

    out = await stage.run({}, ctx=_showcase_ctx())

    assert out.snippets == []
    assert [d["code"] for d in out.degradations] == [GROWTH_BUNDLE_EMPTY]
    assert "delivered video is unaffected" in out.degradations[0]["detail"]
    assert fake.message.await_count == 2  # the retry budget was spent first


@pytest.mark.asyncio
async def test_empty_bundle_without_a_delivered_video_still_fails() -> None:
    """Standalone growth_bundle_v1: the copy IS the deliverable, so empty fails."""
    fake = AsyncMock()
    fake.message.return_value = _wm_response({"snippets": []})
    stage = GrowthBundleStage(client=fake)

    with pytest.raises(StageError):
        await stage.run({}, ctx=_ctx_with_priors())


@pytest.mark.asyncio
async def test_upstream_failure_after_delivery_is_not_swallowed() -> None:
    """Only an EMPTY reply degrades. A transport/upstream error still fails."""
    fake = AsyncMock()
    fake.message.side_effect = RuntimeError("worker-management unreachable")
    stage = GrowthBundleStage(client=fake)

    with pytest.raises(RuntimeError, match="worker-management unreachable"):
        await stage.run({}, ctx=_showcase_ctx())


def _counter_value(site: str) -> float:
    """Current value of the growth_bundle advisory-error counter for ``site``."""
    from src.observability.metrics import growth_bundle_advisory_errors_total

    return growth_bundle_advisory_errors_total.labels(site=site)._value.get()


@pytest.mark.asyncio
async def test_growth_bundle_retention_failure_is_observable_and_still_ships(
    monkeypatch, caplog
) -> None:
    """A swallowed retention-analysis failure must be OBSERVABLE (warning log +
    counter) yet still ship the bundle — the advisory block never gates."""
    from src.services.stages.growth_bundle import handler as gb

    def _boom(*_args, **_kwargs):
        raise RuntimeError("retention exploded")

    monkeypatch.setattr(gb, "analyze_retention", _boom)

    fake = AsyncMock()
    fake.message.return_value = _wm_response({"linkedin": {"text": "L", "cta_url": None}})
    stage = GrowthBundleStage(client=fake)
    # demo_video_gen carries lines ⇒ the retention block runs (and then raises).
    ctx = _ctx_with_priors(
        {
            "promo_script": {"lines": [{"id": "L1"}]},
            "demo_video_gen": {"lines": [{"id": "L1", "text": "a"}]},
        }
    )

    before = _counter_value("retention")
    with caplog.at_level("WARNING", logger="src.services.stages.growth_bundle.handler"):
        out = await stage.run({}, ctx=ctx)

    # (a) the site logged a warning
    assert any(
        "growth_bundle: retention advisory insight failed" in r.message for r in caplog.records
    )
    # (b) the counter incremented for this site
    assert _counter_value("retention") == before + 1
    # (c) the deliverable still shipped, retention insights left empty
    assert [s.channel for s in out.snippets] == ["linkedin"]
    assert out.retention_insights == []


@pytest.mark.asyncio
async def test_growth_bundle_claim_grounding_failure_is_observable_and_still_ships(
    monkeypatch, caplog
) -> None:
    """A swallowed claim-grounding failure must surface a warning + counter while
    the bundle still ships with empty claim insights."""
    from src.services.stages.growth_bundle import handler as gb

    def _boom(*_args, **_kwargs):
        raise RuntimeError("claim lint exploded")

    monkeypatch.setattr(gb, "analyze_claims", _boom)

    fake = AsyncMock()
    fake.message.return_value = _wm_response({"email": {"text": "E", "cta_url": None}})
    stage = GrowthBundleStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        # A `legal` brief block opts the claim-grounding lint in.
        config_snapshot={
            "title": "Anti-Churn",
            "brief": {"call_to_action": "Buy"},
            "legal": {"require_grounding": True},
        },
        prior_run_outputs={
            "promo_script": {"lines": [{"id": "L1"}]},
            "demo_video_gen": {"lines": [{"id": "L1", "text": "the best app"}]},
        },
    )

    before = _counter_value("claim-grounding")
    with caplog.at_level("WARNING", logger="src.services.stages.growth_bundle.handler"):
        out = await stage.run({}, ctx=ctx)

    assert any(
        "growth_bundle: claim-grounding advisory insight failed" in r.message
        for r in caplog.records
    )
    assert _counter_value("claim-grounding") == before + 1
    assert [s.channel for s in out.snippets] == ["email"]
    assert out.claim_insights == []
