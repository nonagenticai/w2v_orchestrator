"""The pre-submit asset block (title cards → captions) has an aggregate deadline.

Before this, that ~120-line block between ``build_shots`` and ``stage_call`` had
NO deadline at all: on 2026-09-03/04 it burned the blueprint's outer 2500s
ceiling three times over (126 minutes total) with one ``logger.warning`` and
zero DB writes, and the eventual failure was an anonymous, zero-argument
``TimeoutError`` naming nothing. See
``src/services/stages/phase_tracker.py`` for the full incident.

Two properties are asserted here, both against the shots/kwargs that actually
reach the compile call (or the raised exception), never against an internal
call count that could pass while the wiring is dead:

* the aggregate deadline fires when the pre-submit block runs long, and the
  raised error NAMES the phase it died in;
* on a normal run, phases are entered in the documented order.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid7

import pytest
from w2v_common.stages import StageTimeoutError

from src.models.failure_class import attribute_exception
from src.services import tts_storage
from src.services.stages.base import ArtifactUrl, StageInput, StageOutput
from src.services.stages.phase_tracker import PhaseDeadlineError
from src.services.stages.primitives import compile_stage as cs_mod
from src.services.stages.primitives.compile_stage import CompileStageBase


def _fake_ctx(config_snapshot: dict[str, Any]) -> Any:
    return SimpleNamespace(
        run_id=uuid7(),
        render_idempotency_discriminator="",
        config_snapshot=config_snapshot,
        prior_output=lambda _stage: None,
    )


class _Input(StageInput):
    pass


class _Output(StageOutput):
    artifact_url: ArtifactUrl
    duration_seconds: float


class _Stage(CompileStageBase[_Input, _Output]):
    name = "test_compile"
    input_model = _Input
    output_model = _Output

    def ken_burns_local_only(self, ctx: Any) -> bool:
        # Pins the video_production RECORDER path, which the base no longer takes.
        return False

    def build_shots(self, ctx: Any) -> list[dict[str, Any]]:
        return [{"id": "s1", "line_id": "l1", "seconds": 1.0}]

    def build_output(self, result: dict[str, Any], ctx: Any) -> _Output:
        return _Output(
            artifact_url=str(result["artifact_url"]),
            duration_seconds=float(result["duration_seconds"]),
        )

    def compile_tags(self, ctx: Any) -> dict[str, str]:
        return {}


class _FakeStorage:
    async def resign_presigned_url(self, url: str) -> str:
        return url


def _capturing_client(captured: dict[str, Any]) -> AsyncMock:
    async def _compile(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"artifact_url": "http://m/out.mp4", "duration_seconds": 1.0}

    client = AsyncMock()
    client.compile = AsyncMock(side_effect=_compile)
    return client


async def _run(stage: CompileStageBase[Any, Any], ctx: Any) -> Any:
    tts_storage.set_storage_for_tests(cast(Any, _FakeStorage()))
    try:
        return await stage.handle(_Input(), ctx=ctx)
    finally:
        tts_storage.set_storage_for_tests(None)


@pytest.mark.asyncio
async def test_presubmit_deadline_fires_and_names_the_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-submit block that runs long is killed by the AGGREGATE deadline —
    not left to burn the outer per-stage ceiling anonymously — and the error
    names the phase (ken_burns) it died in. The Ken Burns entry moves the
    deadline out by at most its own (here shrunk) room, so a pass that overruns
    THAT is still killed by the aggregate deadline."""
    monkeypatch.setattr(cs_mod, "_PRESUBMIT_DEADLINE_SECONDS", 0.05)
    monkeypatch.setattr(cs_mod, "_KEN_BURNS_PASS_BUDGET_SECONDS", 0.05)
    monkeypatch.setattr(cs_mod, "_KEN_BURNS_PASS_OVERHEAD_SECONDS", 0.0)
    monkeypatch.setattr(cs_mod, "_PRESUBMIT_TAIL_RESERVE_SECONDS", 0.0)

    async def _slow_ken_burns(
        ctx: Any, shots: list[dict[str, Any]], **_kw: Any
    ) -> list[dict[str, Any]]:
        await asyncio.sleep(1.0)
        return shots  # pragma: no cover - never reached, deadline fires first

    monkeypatch.setattr(cs_mod, "apply_ken_burns", _slow_ken_burns)

    captured: dict[str, Any] = {}
    stage = _Stage(_capturing_client(captured))

    # PhaseDeadlineError, not the looser TimeoutError: a bare TimeoutError
    # would satisfy `pytest.raises(TimeoutError)` while classifying `unknown`,
    # so the loose form left the raise site effectively UNCOVERED (verified by
    # mutation -- reverting the raise to a bare TimeoutError kept it green).
    with pytest.raises(PhaseDeadlineError) as exc_info:
        await _run(stage, _fake_ctx({}))

    message = str(exc_info.value)
    assert "ken_burns" in message, message
    assert "aggregate deadline" in message, message
    assert attribute_exception(exc_info.value) == "external"
    # The compile submit must NEVER have been reached — the whole point of the
    # deadline is to fail loud BEFORE stage_call, not after it also times out.
    assert captured == {}


@pytest.mark.asyncio
async def test_presubmit_phases_run_in_documented_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a normal run, every named phase is entered, in the documented order,
    ending with stage_call. Recorded by wrapping the real ``phase`` context
    manager rather than counting calls, so a phase that is skipped (not just
    misordered) also fails this."""
    seen: list[str] = []
    real_phase = cs_mod.phase

    def _recording_phase(name: str) -> Any:
        seen.append(name)
        return real_phase(name)

    monkeypatch.setattr(cs_mod, "phase", _recording_phase)

    captured: dict[str, Any] = {}
    stage = _Stage(_capturing_client(captured))

    await _run(stage, _fake_ctx({}))

    assert seen == [
        "title_cards",
        "presign_1",
        "ken_burns",
        "presign_2",
        "sound",
        "audio_bed",
        "captions",
        "stage_call",
    ], seen
    assert captured, "stage_call never reached the compile client"


def test_presubmit_deadline_error_is_attributable_not_unknown() -> None:
    """The deadline error must NOT classify as `unknown`.

    The entire point of this work: a bare TimeoutError is attributed `unknown`
    by `attribute_exception`, which is what made 126 minutes of production
    downtime unclassifiable. A regression to a bare raise here would restore
    exactly that, and every other assertion in this file would still pass.
    """
    from src.models.failure_class import attribute_exception
    from src.services.stages.phase_tracker import PhaseDeadlineError

    err = PhaseDeadlineError("compile: pre-submit exceeded 1800s (phase='ken_burns')")

    assert attribute_exception(err) == "external"
    assert attribute_exception(err) != "unknown"
    # Must stay resumable + catchable exactly as before.
    assert isinstance(err, TimeoutError)
    assert isinstance(err, StageTimeoutError)
    assert str(err)  # never empty, unlike the zero-arg TimeoutError it replaces


@pytest.mark.asyncio
async def test_inner_timeout_is_not_relabelled_as_the_aggregate_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An INNER timeout must propagate untouched, not be relabelled.

    `apply_ken_burns` has its own `_PER_ITEM_TIMEOUT_S=900` budget and a title
    card render caps at 600s. Without the `cm.expired()` guard, any of those
    firing would be caught by the aggregate handler and reported as "exceeded
    1800s aggregate deadline" -- sending the next reader after a budget that
    never fired, and misstating the number by 1200s.
    """
    inner = TimeoutError("ken_burns item 3 exceeded its 900s per-item budget")

    async def _inner_timeout(
        ctx: Any, shots: list[dict[str, Any]], **_kw: Any
    ) -> list[dict[str, Any]]:
        raise inner

    monkeypatch.setattr(cs_mod, "apply_ken_burns", _inner_timeout)

    stage = _Stage(_capturing_client({}))

    with pytest.raises(TimeoutError) as exc_info:
        await _run(stage, _fake_ctx({}))

    assert exc_info.value is inner, "inner timeout was relabelled by the aggregate handler"
    assert not isinstance(exc_info.value, PhaseDeadlineError)
    assert "900s per-item budget" in str(exc_info.value)
    assert "aggregate deadline" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_ken_burns_gets_its_own_budget_after_slow_title_cards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Title cards that eat almost the whole aggregate deadline no longer starve
    the Ken Burns pass: it receives its OWN full budget, and the aggregate
    deadline is moved out to hold it (plus the tail reserve) instead of firing
    mid-pass. Before: the pass was handed ``remaining - tail`` (~0.1s here) and
    shots of the later recorder waves stayed stills.
    """
    monkeypatch.setattr(cs_mod, "_PRESUBMIT_DEADLINE_SECONDS", 1.0)
    monkeypatch.setattr(cs_mod, "_PRESUBMIT_TAIL_RESERVE_SECONDS", 0.1)
    monkeypatch.setattr(cs_mod, "_KEN_BURNS_PASS_BUDGET_SECONDS", 0.6)
    monkeypatch.setattr(cs_mod, "_KEN_BURNS_PASS_OVERHEAD_SECONDS", 0.05)

    async def _slow_cards(*_a: Any, **_kw: Any) -> tuple[None, None]:
        await asyncio.sleep(0.8)  # 0.2s of the 1.0s deadline left afterwards
        return None, None

    monkeypatch.setattr(cs_mod, "render_title_card_plan", _slow_cards)
    monkeypatch.setattr(_Stage, "title_cards_enabled", lambda self, ctx: True)
    monkeypatch.setattr(
        _Stage,
        "build_shots",
        lambda self, ctx: [
            {"id": "s1", "line_id": "l1", "seconds": 1.0, "image_url": "https://m/1.png"}
        ],
    )
    monkeypatch.setattr(_Stage, "title_card_plan", lambda self, ctx: None)
    monkeypatch.setattr(cs_mod, "_title_card_cache_hits", lambda *_a: None)

    budgets: list[float] = []

    async def _kb_spends_its_budget(
        ctx: Any, shots: list[dict[str, Any]], **kw: Any
    ) -> list[dict[str, Any]]:
        budgets.append(kw["budget_seconds"])
        await asyncio.sleep(kw["budget_seconds"])  # the whole pass, e.g. 3 slow waves
        return shots

    monkeypatch.setattr(cs_mod, "apply_ken_burns", _kb_spends_its_budget)

    captured: dict[str, Any] = {}
    await _run(_Stage(_capturing_client(captured)), _fake_ctx({}))

    assert budgets == [0.6], budgets
    assert captured, "the aggregate deadline fired inside the Ken Burns pass"


@pytest.mark.asyncio
async def test_no_still_to_animate_keeps_the_base_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No bare still ⇒ no Ken Burns room is added: the block keeps its base
    deadline, so a slow later phase still hits it."""
    monkeypatch.setattr(cs_mod, "_PRESUBMIT_DEADLINE_SECONDS", 0.3)

    async def _slow_presign(shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
        await asyncio.sleep(0.2)
        return shots

    async def _slow_sound(shots: list[dict[str, Any]], **_kw: Any) -> list[dict[str, Any]]:
        await asyncio.sleep(1.0)
        return shots  # pragma: no cover - the deadline fires first

    monkeypatch.setattr(cs_mod, "refresh_shot_asset_urls", _slow_presign)
    monkeypatch.setattr(cs_mod, "resolve_shot_sound_urls", _slow_sound)

    with pytest.raises(PhaseDeadlineError, match="0s aggregate deadline"):
        await _run(_Stage(_capturing_client({})), _fake_ctx({}))


def test_outer_ceiling_covers_the_ken_burns_extension() -> None:
    """The per-stage outer ceiling is sized from the pre-submit WORST case
    (base deadline + the room the Ken Burns entry may add), so the extension can
    never push the block past the outer wrapper."""
    room = cs_mod._ken_burns_room(set(range(9)))
    assert cs_mod._PRESUBMIT_DEADLINE_SECONDS + room <= cs_mod._PRESUBMIT_WORST_CASE_SECONDS
    assert cs_mod._KEN_BURNS_PASS_BUDGET_SECONDS >= 1000.0
    assert cs_mod.COMPILE_STAGE_OUTER_TIMEOUT_SECONDS >= (
        cs_mod._PRESUBMIT_WORST_CASE_SECONDS
        + cs_mod._COMPILE_TIMEOUT_SECONDS
        + cs_mod._COMPILE_BACKSTOP_MARGIN_SECONDS
    )
