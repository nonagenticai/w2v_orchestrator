"""The local Ken Burns rescue always has budget left (run 01a11082, 2026-10-06).

A recorder job ran the Ken Burns pass to its deadline and the local fallback then
refused with "no pre-submit budget left for a local render": the film shipped a
frozen still. A product that opted in to the fallback now reserves
``min(stills x 90s, 360s)`` for the rescue on top of the RECORDER pass's own
budget (the stage moves its aggregate deadline out to hold both), and each local
render gets at most its share of what remains. Either way the recorder gets its
own full budget.

Driven through ``CompileStageBase.handle`` with the real presubmit and the real
``rescue_frozen_stills``; the recorder pass, ffmpeg, storage and the compiler are
faked. Time is scaled down (seconds, not minutes) by patching the module's
budget constants.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from src.services import tts_storage
from src.services.stages.base import StageInput, StageOutputWithNotes
from src.services.stages.primitives import compile_stage as cs_mod
from src.services.stages.primitives.compile_stage import CompileStageBase
from src.services.stages.shared import ken_burns_local
from src.services.stages.shared.degradation import KEN_BURNS_LOCAL_FALLBACK
from tests.services.stages.promo.test_promo_frame_qa_2026_10_05 import _ctx, _promo_cfg

_STILLS = 3
_DEADLINE = 6.0
_TAIL = 0.5
_PER_STILL = 1.0
_CAP = 2.5  # < _STILLS x _PER_STILL, so the cap is what applies
# The recorder pass's OWN budget. Deliberately larger than what the deadline
# leaves after the reserves (6 - 0.5 - 2.5 = 3.0): the stage must move its
# aggregate deadline out rather than shrink the pass.
_KB_BUDGET = 4.0
_KB_OVERHEAD = 0.2


class _Out(StageOutputWithNotes):
    artifact_url: str
    duration_seconds: float


class _Stage(CompileStageBase[StageInput, _Out]):
    """Three still shots, no captions, no cards — only the Ken Burns phase matters."""

    name = "rescue_budget_compile"
    input_model = StageInput
    output_model = _Out
    opted_in = True

    def build_shots(self, ctx: Any) -> list[dict[str, Any]]:
        return [
            {"id": f"s{i}", "line_id": f"l{i}", "seconds": 3.0, "image_url": f"https://m/{i}.png"}
            for i in range(_STILLS)
        ]

    def build_output(self, result: dict[str, Any], ctx: Any) -> _Out:
        return _Out(
            artifact_url=str(result["artifact_url"]),
            duration_seconds=float(result["duration_seconds"]),
        )

    def compile_tags(self, ctx: Any) -> dict[str, str]:
        return {}

    def ken_burns_local_fallback(self, ctx: Any) -> bool:
        return self.opted_in

    local_only = False

    def ken_burns_local_only(self, ctx: Any) -> bool:
        return self.local_only


class _Storage:
    async def resign_presigned_url(self, url: str) -> str:
        return url


def _fake_local_io(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Fake ffmpeg, the still fetch, storage and re-signing; returns the list the
    local renders' ``timeout_s`` values are appended to."""
    render_timeouts: list[float] = []

    async def _render(_data: bytes, **kw: Any) -> bytes:
        render_timeouts.append(kw["timeout_s"])
        return b"\x00mp4"

    async def _fetch(_url: str, _client: object) -> tuple[bytes, str]:
        return b"\x89PNG", "image/png"

    async def _store_video(**kw: Any) -> str:
        return f"https://minio.test/v/{kw['video_id']}.mp4"

    async def _same(_ctx: Any, shots: list[dict[str, Any]], **_kw: Any) -> list[dict[str, Any]]:
        return shots

    async def _get_storage() -> Any:
        return _Storage()

    monkeypatch.setattr(cs_mod, "crop_cover_stills_to_frame", _same)
    monkeypatch.setattr(cs_mod, "flag_near_static_clips", AsyncMock())
    monkeypatch.setattr(cs_mod, "_get_storage", _get_storage)
    monkeypatch.setattr(ken_burns_local, "render_local_ken_burns", _render)
    monkeypatch.setattr(ken_burns_local, "local_fit", lambda *_a, **_kw: "cover")
    monkeypatch.setattr(ken_burns_local.kb, "_fetch_own_asset", _fetch)
    monkeypatch.setattr(ken_burns_local, "store_generated_video", _store_video)
    return render_timeouts


async def _compile(
    monkeypatch: pytest.MonkeyPatch,
    *,
    opted_in: bool,
    recorder_sleeps: bool,
    local_only: bool = False,
) -> tuple[dict[str, Any], list[float], list[float], _Out]:
    """``(compile kwargs, recorder budgets, local render timeouts, output)``."""
    monkeypatch.setattr(cs_mod, "_PRESUBMIT_DEADLINE_SECONDS", _DEADLINE)
    monkeypatch.setattr(cs_mod, "_PRESUBMIT_TAIL_RESERVE_SECONDS", _TAIL)
    monkeypatch.setattr(cs_mod, "_LOCAL_RESCUE_SECONDS_PER_STILL", _PER_STILL)
    monkeypatch.setattr(cs_mod, "_LOCAL_RESCUE_RESERVE_CAP_SECONDS", _CAP)
    monkeypatch.setattr(cs_mod, "_KEN_BURNS_PASS_BUDGET_SECONDS", _KB_BUDGET)
    monkeypatch.setattr(cs_mod, "_KEN_BURNS_PASS_OVERHEAD_SECONDS", _KB_OVERHEAD)

    recorder_budgets: list[float] = []

    async def _recorder(_ctx: Any, shots: list[dict[str, Any]], **kw: Any) -> list[dict[str, Any]]:
        # Every recording stuck in the queue: the pass runs its WHOLE allotment
        # and returns every still frozen, with the recorder's soft-fail note.
        recorder_budgets.append(kw["budget_seconds"])
        if recorder_sleeps:
            await asyncio.sleep(kw["budget_seconds"])
        for shot in shots:
            kw["notes"].append(
                f"ken_burns: shot {shot['id']} soft-failed, kept as a still: "
                "TimeoutError: job timed out"
            )
        return shots

    monkeypatch.setattr(cs_mod, "apply_ken_burns", _recorder)
    render_timeouts = _fake_local_io(monkeypatch)
    captured: dict[str, Any] = {}

    async def _submit(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"artifact_url": "http://m/out.mp4", "duration_seconds": 9.0}

    compiler = AsyncMock()
    compiler.compile = AsyncMock(side_effect=_submit)
    stage = _Stage(compiler)
    stage.opted_in = opted_in
    stage.local_only = local_only
    tts_storage.set_storage_for_tests(cast(Any, _Storage()))
    try:
        out = await stage.handle(StageInput(), ctx=_ctx(_promo_cfg()))
    finally:
        tts_storage.set_storage_for_tests(None)
    return captured, recorder_budgets, render_timeouts, out


@pytest.mark.asyncio
async def test_a_recorder_pass_that_spends_its_whole_budget_still_leaves_every_still_rescued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured, budgets, timeouts, out = await _compile(
        monkeypatch, opted_in=True, recorder_sleeps=True
    )

    # The recorder got its OWN budget; the rescue's reserve was added to the
    # deadline instead of being carved out of the pass.
    assert budgets == [_KB_BUDGET]
    # It spent all of it, and still every still left the compile moving.
    assert all(shot.get("video_url") for shot in captured["shots"])
    assert len(timeouts) == _STILLS
    assert [a["code"] for a in out.advisories] == [KEN_BURNS_LOCAL_FALLBACK] * _STILLS
    assert not any("frozen" in note or "no pre-submit budget" in note for note in out.notes)
    # Each render is bounded by its SHARE of what was left, never all of it.
    # (The pass's unspent per-item overhead is left over too.)
    assert timeouts[0] <= (_CAP + _KB_OVERHEAD) / _STILLS + 0.01


@pytest.mark.asyncio
async def test_a_product_without_the_local_fallback_gives_the_recorder_the_full_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured, budgets, timeouts, _out = await _compile(
        monkeypatch, opted_in=False, recorder_sleeps=False
    )

    assert budgets == [_KB_BUDGET]
    assert timeouts == []  # nothing is rescued: the stills ship as the recorder left them
    assert not any(shot.get("video_url") for shot in captured["shots"])


@pytest.mark.asyncio
async def test_the_reserve_is_per_still_and_capped() -> None:
    assert cs_mod._local_rescue_reserve(None) == 0.0
    assert cs_mod._local_rescue_reserve(set()) == 0.0
    assert cs_mod._local_rescue_reserve({0, 1}) == 180.0
    assert cs_mod._local_rescue_reserve(set(range(9))) == 360.0


@pytest.mark.asyncio
async def test_a_local_only_product_never_calls_the_recorder_and_animates_every_still(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run 01a156a8-978b (2026-10-20): ONE lesson slide's recorder job executed 977s
    (zero queue wait) of a 1000s pass budget. A local-only product (lesson) skips
    the recorder entirely; the stills are rendered locally inside the BASE deadline
    (no extension), and that planned path is not reported as a fallback."""
    captured, budgets, timeouts, out = await _compile(
        monkeypatch, opted_in=False, recorder_sleeps=True, local_only=True
    )

    assert budgets == [], "the recorder is never called"
    assert len(timeouts) == _STILLS
    assert all(shot.get("video_url") for shot in captured["shots"])
    assert out.advisories == []
    assert not any("ken_burns" in note for note in out.notes)
    # No recorder budget was made room for: each render's share comes out of the
    # base deadline minus the tail reserve.
    assert timeouts[0] <= (_DEADLINE - _TAIL) / _STILLS + 0.01


def test_every_registered_compile_stage_is_local_only() -> None:
    """Owner decision 2026-10-20: Ken Burns is local for EVERY product.

    The recorder costs ~1000s per still (run 01a156a8-978b) and a live generic run
    timed out on it. Walks every production ``CompileStageBase`` subclass reachable
    from the blueprints, so a product that re-introduces a remote override fails here.
    """
    import importlib
    import pkgutil

    import src.services.blueprints as blueprints

    for mod in pkgutil.iter_modules(blueprints.__path__):
        importlib.import_module(f"{blueprints.__name__}.{mod.name}")

    subclasses: list[type[Any]] = []
    pending: list[type[Any]] = [CompileStageBase]
    while pending:
        for sub in pending.pop().__subclasses__():
            subclasses.append(sub)
            pending.append(sub)
    checked = [
        s
        for s in subclasses
        if getattr(s, "output_model", None) is not None
        and s.__module__.startswith("src.services.stages.")
    ]
    assert len(checked) >= 8, f"expected the fleet of compile stages, found {len(checked)}"
    ctx = _ctx(_promo_cfg())
    remote = [s.__name__ for s in checked if s(AsyncMock()).ken_burns_local_only(ctx) is not True]
    assert not remote, f"these compile stages still send stills to the recorder: {remote}"
