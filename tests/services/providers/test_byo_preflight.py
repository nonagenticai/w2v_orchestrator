"""#1106 — run-start BYO preflight: a byo_required run is refused before any stage runs.

Owner decision 2026-10-22: the rule covers EVERY ``byo_governed`` slot (llm, vlm,
tts, asr, image, video_gen, ocr), and the refusal names every missing slot.

Unit half: which modalities a run needs (from the stage ``provider_slots``
declarations + opt-ins + preserved stages), and the posture decision.
End-to-end half: ``POST /api/deliverables/{id}/runs`` through the REAL
``RunEngine.start_run`` — a byo_required tenant gets 402 ``byo_provider_required``
with NO run row created (so no stage, and no LLM call, can ever run); a licences
outage is a retryable 503; an owner tenant (local_allowed) starts normally.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import func, select

import src.services.blueprints  # noqa: F401  registers every blueprint
from src.database.models import Run
from src.models.failure_class import EXTERNAL
from src.services.blueprints.registry import get_registry
from src.services.providers.byo_posture import StaticByoPosture
from src.services.providers.errors import ByoPostureUnavailableError, SlotNotConfiguredError
from src.services.providers.preflight import preflight_byo, required_byo_modalities
from src.services.providers.resolver import PROVIDER_SLOTS_KEY
from src.services.stages.upstream import TransientUpstreamError

TENANT = UUID("01990000-0000-7000-8000-0000000022aa")


def _bp(project_type: str):
    return get_registry().resolve(project_type)


class _Down:
    def __init__(self) -> None:
        self.calls = 0

    async def byo_mode(self, tenant_id: UUID, *, slot: str):
        self.calls += 1
        raise ByoPostureUnavailableError(slot, "licences answered 503")


def _pins(**sources: str) -> dict[str, Any]:
    return {
        PROVIDER_SLOTS_KEY: {
            slot: {"source": src, "kind": "openai" if src == "tenant" else "nonagentic"}
            for slot, src in sources.items()
        }
    }


# ── which modalities a run needs ────────────────────────────────────────────


def test_needed_modalities_follow_stage_declarations_and_opt_ins() -> None:
    assert required_byo_modalities(_bp("course"), {}) == ["llm", "image", "tts"]
    # ai_image_gen is opt-in (``when="brief.ai_image_gen"``)
    assert required_byo_modalities(_bp("generic"), {}) == ["llm", "tts"]
    assert required_byo_modalities(_bp("generic"), {"ai_image_gen": {"prompt": "x"}}) == [
        "llm",
        "tts",
        "image",
    ]
    assert required_byo_modalities(_bp("audiobook"), {}) == ["llm", "tts", "asr"]
    # source_ingest's OCR is ``optional`` (only a scanned page needs it): never up front.
    assert "ocr" not in required_byo_modalities(_bp("generic"), {"sources": [{"x": 1}]})
    assert required_byo_modalities(_bp("web_design"), {}) == []


def test_preserved_stages_do_not_count() -> None:
    bp = _bp("course")
    image_stages = {
        s.name for s in bp.stages if s.name.endswith("thumbnail") or "thumbnail" in s.name
    }
    assert image_stages, "course must have its image-declaring thumbnail stage"
    assert "image" not in required_byo_modalities(bp, {}, skip_stages=image_stages)


# ── the posture decision ────────────────────────────────────────────────────


async def test_byo_required_without_a_tenant_connection_is_refused() -> None:
    posture = StaticByoPosture("byo_required")
    with pytest.raises(SlotNotConfiguredError) as info:
        await preflight_byo(_bp("course"), _pins(image="deployment"), TENANT, posture=posture)
    # EVERY missing slot is named, in topological first-use order.
    assert info.value.slots == ("llm", "image", "tts")
    assert info.value.modality == "llm"
    assert str(info.value) == "Configure your own LLM, image and TTS providers to continue"
    assert info.value.failure_detail["slots"] == "llm,image,tts"
    assert posture.calls == [TENANT]


async def test_byo_required_with_tenant_connections_for_every_need_passes() -> None:
    posture = StaticByoPosture("byo_required")
    await preflight_byo(
        _bp("course"), _pins(llm="tenant", image="tenant", tts="tenant"), TENANT, posture=posture
    )
    assert posture.calls == []  # nothing missing → licences is not even asked


async def test_byo_required_names_the_remaining_missing_modality() -> None:
    with pytest.raises(SlotNotConfiguredError) as info:
        await preflight_byo(
            _bp("course"),
            _pins(llm="tenant", image="tenant", tts="deployment"),
            TENANT,
            posture=StaticByoPosture("byo_required"),
        )
    assert info.value.modality == "tts"
    assert info.value.slots == ("tts",)
    assert str(info.value) == "Configure your own TTS provider to continue"


@pytest.mark.parametrize("mode", ["local_allowed"])
async def test_owner_postures_pass(mode: str) -> None:
    posture = StaticByoPosture(mode)  # ty: ignore[invalid-argument-type]  # why: parametrized literal
    result = await preflight_byo(_bp("course"), {}, TENANT, posture=posture)
    assert result is None
    assert posture.calls == [TENANT]  # the posture WAS consulted, and it allowed the run


async def test_a_run_needing_no_gpu_modality_never_asks_licences() -> None:
    down = _Down()
    await preflight_byo(_bp("web_design"), {}, TENANT, posture=down)
    assert down.calls == 0


async def test_a_licences_outage_is_retryable_not_a_platform_fallback() -> None:
    with pytest.raises(ByoPostureUnavailableError) as info:
        await preflight_byo(_bp("course"), {}, TENANT, posture=_Down())
    assert isinstance(info.value, TransientUpstreamError)
    assert info.value.failure_class == EXTERNAL


# ── end to end: POST /api/deliverables/{id}/runs ────────────────────────────


async def _deliverable(client) -> str:
    r = await client.post(
        "/api/projects",
        json={
            "type": "generic",
            "title": "byo preflight",
            "brief": {"angle": "growth", "target_duration_seconds": 30},
            "voice": {"descriptor": "crisp"},
            "deliverables": [
                {
                    "name": "d",
                    "format": "16x9",
                    "captions_mode": "none",
                    "target_duration_seconds": 30,
                }
            ],
        },
    )
    assert r.status_code == 201, r.text
    detail = await client.get(f"/api/projects/{r.json()['id']}")
    return detail.json()["deliverables"][0]["id"]


async def _runs_for(db_session_factory, deliverable_id: str) -> int:
    async with db_session_factory() as s:
        return (
            await s.execute(
                select(func.count())
                .select_from(Run)
                .where(Run.deliverable_id == UUID(deliverable_id))
            )
        ).scalar_one()


def _posture(monkeypatch: pytest.MonkeyPatch, reader: object) -> None:
    monkeypatch.setattr("src.services.providers.adapters.get_byo_posture", lambda: reader)


async def test_byo_required_run_is_refused_before_any_stage(
    client, db_session_factory, monkeypatch
) -> None:
    deliv = await _deliverable(client)
    _posture(monkeypatch, StaticByoPosture("byo_required"))
    created: list[Any] = []
    from src.database.repositories.run_repository import RunRepository

    real_create = RunRepository.create

    async def _spy(self, *a: Any, **k: Any):
        created.append(k)
        return await real_create(self, *a, **k)

    monkeypatch.setattr(RunRepository, "create", _spy)

    r = await client.post(f"/api/deliverables/{deliv}/runs", json={"pause_after_stages": []})

    assert r.status_code == 402, r.text
    assert r.json()["detail"] == {
        "code": "byo_provider_required",
        "modality": "llm",
        "slots": "llm,tts,image",
        "message": "Configure your own LLM, TTS and image providers to continue",
    }
    # No run row → nothing to dispatch → no stage (and no LLM call) can ever execute.
    assert created == []
    assert await _runs_for(db_session_factory, deliv) == 0


async def test_licences_outage_at_preflight_is_a_retryable_503(
    client, db_session_factory, monkeypatch
) -> None:
    deliv = await _deliverable(client)
    _posture(monkeypatch, _Down())

    r = await client.post(f"/api/deliverables/{deliv}/runs", json={"pause_after_stages": []})

    assert r.status_code == 503, r.text
    assert r.headers.get("Retry-After") == "30"
    assert await _runs_for(db_session_factory, deliv) == 0


async def test_owner_tenant_run_starts_normally(
    app, client, db_session_factory, monkeypatch
) -> None:
    from unittest.mock import AsyncMock

    deliv = await _deliverable(client)
    posture = StaticByoPosture("local_allowed")
    _posture(monkeypatch, posture)
    # Only the start decision is under test: do not execute the run's stages.
    monkeypatch.setattr(app.state.run_engine, "_dispatch", AsyncMock())

    r = await client.post(f"/api/deliverables/{deliv}/runs", json={"pause_after_stages": []})

    assert r.status_code == 201, r.text
    assert len(posture.calls) == 1
    assert await _runs_for(db_session_factory, deliv) == 1
