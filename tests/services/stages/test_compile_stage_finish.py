"""CompileStageBase threads the color-finish block to the video-compile client.

Wiring tests for ORCH-0.4 — `finish.py` used to be 100% dead code (zero
importers), so an authored `finish` block never reached the compiler and no brand
letterbox was ever applied. These pin the whole chain:

1. a present `finish` config block reaches the client as the compiler's one-field
   FinishProfile payload (``letterbox_color``, resolved from the brand);
2. an ABSENT block (the common case — the lift is presence-is-opt-in) omits the
   kwarg entirely, so the compile request is byte-identical to before;
3. nothing else leaks into the payload — not ``broadcast_safe`` (the compiler has
   no such field: a pre-encode luma clamp cannot survive libx264 ringing) and not
   the WM grade knobs (not renderable yet).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid7

from src.services import tts_storage
from src.services.stages.base import ArtifactUrl, StageInput, StageOutput
from src.services.stages.primitives.compile_stage import CompileStageBase
from src.services.stages.shared.mastering_plan import default_loudness_target


class _Input(StageInput):
    pass


class _Output(StageOutput):
    artifact_url: ArtifactUrl
    duration_seconds: float


class _Stage(CompileStageBase[_Input, _Output]):
    name = "test_compile_finish"
    input_model = _Input
    output_model = _Output

    def __init__(self, client: Any) -> None:
        super().__init__(client=client)

    def build_shots(self, ctx: Any) -> list[dict[str, Any]]:
        return [{"id": "s1", "line_id": "l1", "seconds": 2.0}]

    def build_output(self, result: dict[str, Any], ctx: Any) -> _Output:
        return _Output(
            artifact_url=str(result["artifact_url"]),
            duration_seconds=float(result["duration_seconds"]),
        )

    def compile_tags(self, ctx: Any) -> dict[str, str]:
        return {}  # avoid touching ctx.run_id in this unit test


class _FakeStorage:
    async def resign_presigned_url(self, url: str) -> str:
        return url


async def _run_stage(config_snapshot: dict[str, Any]) -> dict[str, Any]:
    """Run the compile tail against a capturing client; return the compile kwargs."""
    captured: dict[str, Any] = {}

    async def _compile(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"artifact_url": "http://m/out.mp4", "duration_seconds": 2.0}

    client = AsyncMock()
    client.compile = AsyncMock(side_effect=_compile)

    tts_storage.set_storage_for_tests(cast(Any, _FakeStorage()))
    try:
        await _Stage(client).handle(
            _Input(),
            ctx=cast(
                Any,
                # The real RunContext always exposes prior_output; a fake that omits it
                # is an incomplete fake, not a smaller one. No stage output here ⇒ the
                # compile tail's opt-in reads (disclaimer card, …) all resolve to None.
                SimpleNamespace(
                    run_id=uuid7(),
                    render_idempotency_discriminator="",
                    config_snapshot=config_snapshot,
                    prior_output=lambda _stage: None,
                ),
            ),
        )
    finally:
        tts_storage.set_storage_for_tests(None)
    return captured


async def test_absent_finish_block_sends_none() -> None:
    """No `finish` key ⇒ finish=None ⇒ the client drops it ⇒ byte-identical payload.
    (The mastering path is untouched by this — it stays None here too.)"""
    captured = await _run_stage({})
    assert captured["finish"] is None
    assert captured["mastering"] is None


async def test_finish_payload_carries_only_letterbox_color() -> None:
    """broadcast_safe and the WM grade knobs never ship, even when the author sets
    them: the compiler has no field for either."""
    captured = await _run_stage(
        {"finish": {"broadcast_safe": True, "contrast": 1.3, "vignette": 0.4}}
    )
    assert set(captured["finish"]) == {"letterbox_color"}


async def test_finish_does_not_disturb_mastering() -> None:
    """A finish block must not knock the existing mastering block off the request."""
    captured = await _run_stage(
        {"finish": {}, "mastering": {"crf": 18}, "brand": {"primary_color": "#280983"}}
    )
    assert captured["finish"] == {"letterbox_color": "#280983"}
    assert captured["mastering"]["crf"] == 18
    # CORRECTED 2026-09-16 (was the literal -12.5). This test's subject is that a
    # `finish` block does not knock `mastering` off the request — the loudness value is
    # the DEFAULT this stage injects (the input above authors only `crf`), so pinning a
    # second literal for it here only duplicated the contract and went stale when the
    # target moved. Compiler PR #217 (2026-09-12) abandoned -12.5 because it drove ~5.5dB
    # of gain riding on a -18 LUFS narration bus (the audible "breathing"); the shipped
    # compiler default is now -14.0. Asserted against the source of truth rather than a
    # copy of it, so this can never again go red for a legitimate retune. The literal
    # itself stays strictly pinned, once, in `shared/test_loudness_contract.py`.
    assert captured["mastering"]["loudness_target"] == default_loudness_target()
