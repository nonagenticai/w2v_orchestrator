"""A pod killed mid-compile must not re-pay, on resume, for work it already stored.

Every merge rolls the ``replicas: 1`` orchestrator, so a compile stage is routinely
SIGTERMed half-way through its pre-submit asset block and re-executed from the top
by the reconciler. This simulates exactly that against the REAL ``handle`` →
``_run_presubmit`` → ``build_title_card_shots`` / ``apply_ken_burns`` path:

1. attempt 1 records both title cards and the first N Ken Burns clips, then the
   task is cancelled while clips N+1..M are still recording (the "kill");
2. attempt 2 (same run_id, same storage, a FRESH video_production client — a new
   pod) must record ONLY the M-N missing clips and zero cards, submit the compile,
   and persist the reuse counts on the output;
3. attempt 3 (a second roll after the compile) records nothing and submits the
   compile under the IDENTICAL idempotency key, so the compiler dedupes it.

The storage fake is an in-memory object store whose existence probe answers from
what was really stored — a hit cannot come from a stub that always says yes.
"""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import UUID, uuid7

import pytest

from src.services import tts_storage
from src.services.stages.base import ArtifactUrl, StageInput, StageOutputWithNotes
from src.services.stages.primitives.compile_stage import CompileStageBase

_M = 4  # Ken Burns candidate stills
_N = 2  # clips stored before the kill


class _ObjectStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def initialize(self) -> None:
        pass

    async def store_stage_artifact(
        self, job_id: str, stage_name: str, data: bytes, filename: str
    ) -> str:
        path = f"artifacts/{job_id}/{stage_name}/{filename}"
        self.objects[path] = data
        return path

    async def get_presigned_url(
        self, path: str, expires: int | None = None, bucket: str | None = None
    ) -> str | None:
        return f"https://minio.test/{path}"

    async def stage_artifact_url(self, job_id: str, stage_name: str, filename: str) -> str | None:
        return f"https://minio.test/artifacts/{job_id}/{stage_name}/{filename}"

    async def object_exists(self, url: str) -> bool | None:
        return url.removeprefix("https://minio.test/") in self.objects

    async def resign_presigned_url(self, url: str) -> str:
        return url

    def count(self, stage_name: str) -> int:
        return sum(1 for p in self.objects if f"/{stage_name}/" in p)


class _Input(StageInput):
    pass


class _Output(StageOutputWithNotes):
    artifact_url: ArtifactUrl
    duration_seconds: float


class _StillsStage(CompileStageBase[_Input, _Output]):
    name = "test_resume_compile"
    input_model = _Input
    output_model = _Output

    def ken_burns_local_only(self, ctx: Any) -> bool:
        # Pins the video_production RECORDER path, which the base no longer takes.
        return False

    def build_shots(self, ctx: Any) -> list[dict[str, Any]]:
        return [
            {
                "id": f"s{i}",
                "line_id": f"l{i}",
                "seconds": 4.0,
                "image_url": f"https://minio.test/w2v-orchestrator/still-{i}.png",
            }
            for i in range(_M)
        ]

    def build_output(self, result: dict[str, Any], ctx: Any) -> _Output:
        return _Output(
            artifact_url=str(result["artifact_url"]),
            duration_seconds=float(result["duration_seconds"]),
        )

    def compile_tags(self, ctx: Any) -> dict[str, str]:
        return {}


class _VideoProduction:
    """A recorder that completes the first ``budget`` Ken Burns records, then hangs
    (the in-flight recordings a SIGTERM interrupts). Title cards always complete."""

    def __init__(self, kb_budget: int | None) -> None:
        self.kb_budget = kb_budget
        self.calls: list[str] = []

    async def record_animation(self, **kwargs: Any) -> str:
        part = str(kwargs["part_id"])
        self.calls.append(part)
        over_budget = (
            self.kb_budget is not None
            and sum(1 for c in self.calls if c.startswith("kb-")) > self.kb_budget
        )
        if part.startswith("kb-") and over_budget:
            await asyncio.Event().wait()  # never completes: killed mid-record
        return f"job-{len(self.calls)}"

    async def wait_for_content_recording(self, job_id: str, timeout: float) -> dict[str, Any]:
        return {"duration_seconds": 4.0}

    async def get_recording_video(self, job_id: str) -> tuple[bytes, str, None]:
        return (b"\x00mp4", "video/mp4", None)

    def kb_calls(self) -> int:
        return sum(1 for c in self.calls if c.startswith("kb-"))

    def card_calls(self) -> int:
        return sum(1 for c in self.calls if c.startswith("title-"))


def _compile_client(keys: list[str]) -> AsyncMock:
    async def _compile(**kwargs: Any) -> dict[str, Any]:
        keys.append(kwargs["idempotency_key"])
        return {
            "artifact_url": "http://m/w2v-compiled-videos/out.mp4",
            "duration_seconds": 22.0,
            "actual_duration_seconds": 22.0,
            "mastering": {"rerun_stages": [], "gate_pass": True, "luma_spread": 210.0},
        }

    client = AsyncMock()
    client.compile = AsyncMock(side_effect=_compile)
    return client


@pytest.mark.asyncio
async def test_a_resume_after_a_mid_stage_kill_only_records_what_was_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _png(url: str, client: Any) -> tuple[bytes, str]:
        return (b"\x89PNG\r\n", "image/png")

    monkeypatch.setattr("src.services.stages.shared.ken_burns._fetch_own_asset", _png)

    store = _ObjectStore()
    tts_storage.set_storage_for_tests(cast(Any, store))
    run_id: UUID = uuid7()
    ctx = cast(
        Any,
        SimpleNamespace(
            run_id=run_id,
            render_idempotency_discriminator="",
            config_snapshot={
                "title": "Ridgeline",
                "brand": {"name": "Acme"},
                "title_card": {"enabled": True},
            },
            prior_output=lambda _stage: None,
        ),
    )

    def _use(vp: _VideoProduction) -> None:
        monkeypatch.setattr(
            "src.services.stages.shared.ken_burns.VideoProductionClient", lambda: vp
        )
        monkeypatch.setattr(
            "src.services.stages.shared.title_cards.VideoProductionClient", lambda: vp
        )

    try:
        # ── attempt 1: killed after N clips are stored, M-N still recording ──
        pod1 = _VideoProduction(kb_budget=_N)
        _use(pod1)
        keys1: list[str] = []
        task = asyncio.create_task(_StillsStage(_compile_client(keys1)).handle(_Input(), ctx))
        for _ in range(500):
            if store.count("ken_burns") >= _N and pod1.kb_calls() > _N:
                break
            await asyncio.sleep(0.01)
        assert store.count("ken_burns") == _N, store.objects.keys()
        assert store.count("title_card") == 2
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert keys1 == []  # killed before the compile was ever submitted

        # ── attempt 2: the resumed stage on a new pod ──
        pod2 = _VideoProduction(kb_budget=None)
        _use(pod2)
        keys2: list[str] = []
        out2 = await _StillsStage(_compile_client(keys2)).handle(_Input(), ctx)

        assert pod2.card_calls() == 0, pod2.calls
        assert pod2.kb_calls() == _M - _N, pod2.calls
        assert out2.title_card_cache_hits == 2
        assert out2.ken_burns_cache_hits == _N
        assert out2.ken_burns_shot_count == _M
        assert keys2 == [f"{run_id}:test_resume_compile"]

        # ── attempt 3: a second roll after the compile — nothing re-recorded ──
        pod3 = _VideoProduction(kb_budget=None)
        _use(pod3)
        keys3: list[str] = []
        out3 = await _StillsStage(_compile_client(keys3)).handle(_Input(), ctx)

        assert pod3.calls == []
        assert out3.ken_burns_cache_hits == _M
        assert out3.title_card_cache_hits == 2
        assert keys3 == keys2  # identical key ⇒ the compiler returns the existing job
    finally:
        tts_storage.set_storage_for_tests(None)


@pytest.mark.asyncio
async def test_counters_are_none_when_the_phases_had_nothing_to_do(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No title_card block and no stills ⇒ ``None``, never a ``0`` that would read
    as "ran cold"."""

    class _FootageStage(_StillsStage):
        def build_shots(self, ctx: Any) -> list[dict[str, Any]]:
            return [{"id": "v", "line_id": "l", "seconds": 4.0, "video_url": "https://x/v.mp4"}]

    store = _ObjectStore()
    tts_storage.set_storage_for_tests(cast(Any, store))
    try:
        out = await _FootageStage(_compile_client([])).handle(
            _Input(),
            cast(
                Any,
                SimpleNamespace(
                    run_id=uuid7(),
                    render_idempotency_discriminator="",
                    config_snapshot={"title": "Ridgeline"},
                    prior_output=lambda _stage: None,
                ),
            ),
        )
    finally:
        tts_storage.set_storage_for_tests(None)
    assert out.title_card_cache_hits is None
    assert out.ken_burns_cache_hits is None
