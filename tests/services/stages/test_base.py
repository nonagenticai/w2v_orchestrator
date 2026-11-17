import pytest

from src.services.stages.base import StageHandler, StageInput, StageOutput


class _FakeIn(StageInput):
    text: str


class _FakeOut(StageOutput):
    upper: str


class FakeStage(StageHandler[_FakeIn, _FakeOut]):
    name = "fake"
    input_model = _FakeIn
    output_model = _FakeOut

    async def handle(self, payload: _FakeIn, ctx) -> _FakeOut:
        return _FakeOut(upper=payload.text.upper())


@pytest.mark.asyncio
async def test_stage_handler_validates_input_and_output() -> None:
    stage = FakeStage()
    out = await stage.run({"text": "hello"}, ctx=None)
    assert isinstance(out, _FakeOut)
    assert out.upper == "HELLO"


@pytest.mark.asyncio
async def test_stage_handler_rejects_bad_input() -> None:
    stage = FakeStage()
    with pytest.raises(ValueError):
        await stage.run({"wrong_field": "x"}, ctx=None)
