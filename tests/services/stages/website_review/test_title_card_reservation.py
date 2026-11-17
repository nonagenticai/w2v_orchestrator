"""The compile renders the title-card seconds the SCRIPT reserved — it never re-plans.

The 20% cap (``shared/title_cards.plan_title_cards``) used to be computed twice from
the run snapshot: once by the script's budget, once by the compile's render. Two
computations agree only while the code between them stands still. A run SCRIPTED
before the cap deployed (narration budget = order - 6.0s of cards) and COMPILED after
it (a resume, or ``regenerate {"fromStage": "website_review_compile"}``) rendered the
capped cards against the uncapped budget: a 10s order scripted to 4.0s of narration
got one 2.0s intro, i.e. a ~6s film, -40%.

So the script records what it reserved (``reserved_title_cards``) and the compile
reads the record. Every test here drives the REAL ``WebsiteReviewCompileStage`` and
asserts on the shots that reach the compile CALL and the durations the recorder was
asked for — not on a hook's return value.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.services import tts_storage
from src.services.runs.run_context import RunContext
from src.services.stages.website_review.compile import WebsiteReviewCompileStage
from src.services.stages.website_review.order import (
    ReservedTitleCards,
    fixed_non_narration_seconds,
    narration_budget_seconds,
    reserve_title_cards,
    title_card_plan_to_render,
)
from src.services.stages.website_review.script import ReviewLine, WebsiteReviewScriptStage

_CARDS: dict[str, Any] = {"enabled": True}
_INTRO, _OUTRO = "__title_intro__", "__title_outro__"


class _ObjectStore:
    """Enough storage for the card store/probe and the compile tail's re-sign."""

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


class _Recorder:
    """video_production: records the per-card ``duration_ms`` it was asked for."""

    def __init__(self) -> None:
        self.cards: dict[str, int] = {}

    async def record_animation(self, **kwargs: Any) -> str:
        self.cards[str(kwargs["part_id"])] = int(kwargs["duration_ms"])
        return f"job-{len(self.cards)}"

    async def wait_for_content_recording(self, job_id: str, timeout: float) -> dict[str, Any]:
        return {}

    async def get_recording_video(self, job_id: str) -> tuple[bytes, str, None]:
        return (b"\x00mp4", "video/mp4", None)


def _snapshot(ordered: float, title_card: Any = _CARDS) -> dict[str, Any]:
    return {
        "title": "Acme review",
        "brand": {"name": "Acme"},
        "voice": {"descriptor": "warm"},
        "deliverable": {"format": "16x9", "target_duration_seconds": ordered},
        "title_card": title_card,
    }


_LEGACY_SCRIPT: dict[str, Any] = {
    # A script output written BEFORE the record existed: no `reserved_title_cards` key.
    "lines": [
        {"id": "L1", "text": "Problem.", "seconds": 2.0, "phase": "before"},
        {"id": "L2", "text": "Fix.", "seconds": 2.0, "phase": "after"},
    ],
    "total_seconds": 4.0,
}


def _script(reserved: dict[str, float] | None) -> dict[str, Any]:
    return {**_LEGACY_SCRIPT, "reserved_title_cards": reserved}


def _fresh_script(snapshot: dict[str, Any]) -> dict[str, Any]:
    """What the REAL script stage persists for ``snapshot`` (its own ``build_output``)."""
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=snapshot,
        prior_run_outputs={
            "website_critique": {"url": "https://acme.test", "findings": []},
            "website_redesign": {"style": "modern", "visuals": []},
        },
    )
    stage = WebsiteReviewScriptStage(client=None)
    out = stage.build_output(
        [
            ReviewLine(id="L1", text="The hero buries the offer.", seconds=2.0, phase="before"),
            ReviewLine(id="L2", text="Now it leads with it.", seconds=2.0, phase="after"),
        ],
        ctx,
    )
    return out.model_dump(mode="json")


async def _compile(
    monkeypatch: pytest.MonkeyPatch, snapshot: dict[str, Any], script: dict[str, Any]
) -> tuple[dict[str, float], dict[str, int], int | None]:
    """Run the real compile; return (card seconds SUBMITTED by line id, the
    ``duration_ms`` RECORDED by part id, the output's ``title_card_cache_hits``)."""
    recorder = _Recorder()
    monkeypatch.setattr(
        "src.services.stages.shared.title_cards.VideoProductionClient", lambda: recorder
    )
    client = AsyncMock()
    client.compile.return_value = {
        "artifact_url": "http://stub/review.mp4",
        "duration_seconds": 10.0,
        "actual_duration_seconds": 10.0,
        "mastering": {"rerun_stages": [], "gate_pass": True, "luma_spread": 210.0},
    }
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=snapshot,
        prior_run_outputs={
            "website_review_script": script,
            "website_review_record": {
                "clips": [
                    {
                        "id": f"c-{line['id']}",
                        "line_id": line["id"],
                        "artifact_url": f"http://a/{line['id']}.wav",
                        "seconds": 2.0,
                    }
                    for line in script["lines"]
                ]
            },
            "website_redesign": {"visuals": []},
        },
    )
    tts_storage.set_storage_for_tests(cast(Any, _ObjectStore()))
    try:
        out = await WebsiteReviewCompileStage(client=client).run({}, ctx=ctx)
    finally:
        tts_storage.set_storage_for_tests(None)
    shots = client.compile.call_args.kwargs["shots"]
    submitted = {
        str(shot["line_id"]): float(shot["seconds"])
        for shot in shots
        if shot.get("line_id") in (_INTRO, _OUTRO)
    }
    return submitted, recorder.cards, out.title_card_cache_hits


@pytest.mark.asyncio
async def test_a_script_with_no_record_gets_its_cards_as_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LEGACY. A 10s order scripted before the record existed budgeted 10 - 6.0 = 4.0s
    of narration, so its compile must render 3.0 + 3.0 — NOT the capped 2.0s intro,
    which leaves a ~6s film against a 10s order."""
    submitted, recorded, hits = await _compile(monkeypatch, _snapshot(10), _LEGACY_SCRIPT)

    assert submitted == {_INTRO: 3.0, _OUTRO: 3.0}
    assert recorded == {"title-intro": 3000, "title-outro": 3000}
    assert hits == 0  # cards planned and rendered cold — not "no cards planned"


@pytest.mark.asyncio
async def test_a_null_record_is_the_same_as_no_record(monkeypatch: pytest.MonkeyPatch) -> None:
    """A legacy output re-validated through the model dumps the field as ``None``."""
    submitted, _, _ = await _compile(monkeypatch, _snapshot(10), _script(None))
    assert submitted == {_INTRO: 3.0, _OUTRO: 3.0}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ordered", "reserved", "cards"),
    [(10, {"intro_seconds": 2.0, "outro_seconds": 0.0}, {_INTRO: 2.0})],
)
async def test_the_compile_renders_exactly_the_recorded_reservation(
    monkeypatch: pytest.MonkeyPatch,
    ordered: float,
    reserved: dict[str, float],
    cards: dict[str, float],
) -> None:
    submitted, recorded, hits = await _compile(monkeypatch, _snapshot(ordered), _script(reserved))

    assert submitted == cards
    assert sorted(recorded.values()) == sorted(round(s * 1000) for s in cards.values())
    assert hits == 0


@pytest.mark.asyncio
async def test_a_recorded_no_cards_renders_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 7s order has 1.4s to spend — under the 1.5s minimum — so its script reserved
    NO cards and kept all 7s for narration. The compile must not fall back to the
    configured 3.0 + 3.0 just because the recorded total is falsy."""
    snapshot = _snapshot(7)
    reserved = reserve_title_cards(snapshot)
    assert reserved == ReservedTitleCards(intro_seconds=0.0, outro_seconds=0.0)

    submitted, recorded, hits = await _compile(
        monkeypatch, snapshot, _script(reserved.model_dump(mode="json"))
    )

    assert submitted == {}
    assert recorded == {}
    assert hits is None  # "no cards planned", never a 0 that reads "rendered cold"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ordered", "cards"),
    [(10, {_INTRO: 2.0})],
)
async def test_a_fresh_runs_budget_and_rendered_cards_agree(
    monkeypatch: pytest.MonkeyPatch, ordered: float, cards: dict[str, float]
) -> None:
    """THE agreement, end to end: the seconds ``order.py`` subtracts from the budget,
    the seconds the real script stage RECORDS, and the seconds the real compile puts
    on screen are one number. Order minus rendered cards == the narration budget."""
    snapshot = _snapshot(ordered)
    script = _fresh_script(snapshot)
    assert script["reserved_title_cards"] is not None

    submitted, recorded, _ = await _compile(monkeypatch, snapshot, script)

    assert submitted == cards
    assert sum(recorded.values()) == round(sum(cards.values()) * 1000)
    assert sum(submitted.values()) == pytest.approx(fixed_non_narration_seconds(snapshot))
    assert ordered - sum(submitted.values()) == pytest.approx(narration_budget_seconds(snapshot))
    recorded_total = sum(cast(dict[str, float], script["reserved_title_cards"]).values())
    assert recorded_total == pytest.approx(sum(submitted.values()))


@pytest.mark.asyncio
async def test_cards_switched_off_after_the_script_are_not_rendered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reservation is an upper bound, not an instruction: a child snapshot that
    no longer enables the block renders no card."""
    submitted, recorded, hits = await _compile(
        monkeypatch,
        _snapshot(15, title_card={"enabled": False}),
        _script({"intro_seconds": 1.5, "outro_seconds": 1.5}),
    )
    assert (submitted, recorded, hits) == ({}, {}, None)


def test_a_record_with_two_different_card_lengths_is_refused() -> None:
    """Not a shape this code writes, and not one a single-length plan can render
    "exactly" — so it fails loudly instead of silently picking one of the two."""
    with pytest.raises(ValidationError, match="disagree on length"):
        title_card_plan_to_render(
            _snapshot(15), _script({"intro_seconds": 2.0, "outro_seconds": 1.5})
        )
