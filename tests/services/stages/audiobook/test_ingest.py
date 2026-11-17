"""audiobook_ingest stage — manuscript fetched by ref, parsed into chapters."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

from src.services import manuscript_storage
from src.services.runs.run_context import RunContext
from src.services.stages.audiobook.ingest import (
    AudiobookIngestInput,
    AudiobookIngestStage,
)
from src.services.stages.base import StagePreconditionError


class _FakeStorage:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self.fetched: list[str] = []

    async def initialize(self) -> None:
        pass

    async def store_stage_artifact(
        self, job_id: str, stage_name: str, data: bytes, filename: str
    ) -> str:
        return f"{job_id}/{stage_name}/{filename}"

    async def retrieve(self, ref: str) -> bytes | None:
        self.fetched.append(ref)
        return self._data


@pytest.fixture
def fake_manuscript():
    text = b"Chapter 1\nThe opening.\n\nChapter 2\nThe close."
    storage = _FakeStorage(text)
    manuscript_storage.set_manuscript_storage_for_tests(storage)
    yield storage
    manuscript_storage.set_manuscript_storage_for_tests(None)


async def test_ingest_parses_uploaded_manuscript(fake_manuscript: _FakeStorage) -> None:
    stage = AudiobookIngestStage()
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "title": "My Audiobook",
            "brief": {"manuscript_ref": "manuscripts/t/upload/x.txt", "manuscript_name": "x.txt"},
        },
    )
    out = await stage.run({}, ctx=ctx)
    assert fake_manuscript.fetched == ["manuscripts/t/upload/x.txt"]
    assert [c.title for c in out.chapters] == ["Chapter 1", "Chapter 2"]
    assert all(c.id for c in out.chapters)
    assert [c.sequence for c in out.chapters] == [0, 1]


async def test_ingest_single_chapter_mode(fake_manuscript: _FakeStorage) -> None:
    stage = AudiobookIngestStage()
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "title": "Solo",
            "brief": {
                "manuscript_ref": "r",
                "manuscript_name": "x.txt",
                "chapter_mode": "single",
            },
        },
    )
    out = await stage.run({}, ctx=ctx)
    assert len(out.chapters) == 1


async def test_ingest_missing_ref_is_precondition_error() -> None:
    stage = AudiobookIngestStage()
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={"brief": {}})
    with pytest.raises(StagePreconditionError, match="manuscript_ref"):
        await stage.run({}, ctx=ctx)


# --- pronunciation-proposer (PR-8a) ----------------------------------------


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


@pytest.fixture
def manuscript_factory() -> Iterator[Any]:
    """Yield a setter that installs manuscript bytes and resets on teardown."""

    def _set(text: bytes) -> _FakeStorage:
        storage = _FakeStorage(text)
        manuscript_storage.set_manuscript_storage_for_tests(storage)
        return storage

    yield _set
    manuscript_storage.set_manuscript_storage_for_tests(None)


def _ctx(brief: dict[str, Any]) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"title": "Book", "brief": brief},
    )


async def test_proposer_opt_out_no_wm_traffic_byte_identical(manuscript_factory: Any) -> None:
    # Flag absent → ZERO WM traffic and the parse is byte-identical to today.
    manuscript_factory(b"Chapter 1\nMerqi walked to Doorn.")
    fake = AsyncMock()

    baseline = await AudiobookIngestStage().handle(
        AudiobookIngestInput(),
        _ctx({"manuscript_ref": "r", "manuscript_name": "x.txt"}),
    )

    stage = AudiobookIngestStage(client=fake)
    out = await stage.handle(
        AudiobookIngestInput(),
        _ctx({"manuscript_ref": "r", "manuscript_name": "x.txt"}),
    )
    assert fake.message.await_count == 0
    assert out.pronunciations == {}
    # Same parsed text as a run with no proposer at all.
    assert [c.text for c in out.chapters] == [c.text for c in baseline.chapters]


async def test_proposer_opt_out_carries_brief_pronunciations(manuscript_factory: Any) -> None:
    manuscript_factory(b"Chapter 1\nMerqi walked to Doorn.")
    fake = AsyncMock()
    stage = AudiobookIngestStage(client=fake)
    out = await stage.handle(
        AudiobookIngestInput(),
        _ctx(
            {
                "manuscript_ref": "r",
                "manuscript_name": "x.txt",
                "pronunciations": {"Merqi": "Mer-kee"},
            }
        ),
    )
    assert fake.message.await_count == 0
    assert out.pronunciations == {"Merqi": "Mer-kee"}
    # brief lexicon applied to the parsed body.
    assert "Mer-kee" in out.chapters[0].text


async def test_ingest_ignores_non_dict_brief_pronunciations(manuscript_factory: Any) -> None:
    # A malformed brief with `pronunciations` as a LIST must NOT hard-fail a paid
    # ingest: `.get(...) or {}` let it through and `{**proposals, **brief_pron}`
    # then raised TypeError. It is now coerced to "no custom pronunciations".
    manuscript_factory(b"Chapter 1\nMerqi walked to Doorn.")
    stage = AudiobookIngestStage(client=AsyncMock())
    out = await stage.handle(
        AudiobookIngestInput(),
        _ctx(
            {
                "manuscript_ref": "r",
                "manuscript_name": "x.txt",
                "pronunciations": ["Merqi", "Mer-kee"],  # malformed: a list, not a dict
            }
        ),
    )
    assert out.pronunciations == {}  # coerced, not crashed
    assert out.chapters  # parse still succeeded


async def test_proposer_merges_into_lexicon_human_brief_wins(manuscript_factory: Any) -> None:
    manuscript_factory(b"Chapter 1\nMerqi walked to Doorn.")
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "proposals": [
                {"word": "Merqi", "ipa_or_phonetic": "Mer-kee"},
                {"word": "Doorn", "ipa_or_phonetic": "Dorn"},
            ]
        }
    )
    stage = AudiobookIngestStage(client=fake)
    out = await stage.handle(
        AudiobookIngestInput(),
        _ctx(
            {
                "manuscript_ref": "r",
                "manuscript_name": "x.txt",
                "audiobook": {"propose_pronunciations": True},
                # Human brief overrides the proposer entry for the SAME word.
                "pronunciations": {"Merqi": "HUMAN-WINS"},
            }
        ),
    )
    assert fake.message.await_count == 1
    # Merged map = proposals UNDER the brief; the human wins on Merqi.
    assert out.pronunciations == {"Merqi": "HUMAN-WINS", "Doorn": "Dorn"}
    body = out.chapters[0].text
    # The single parse applied the merged lexicon: human respelling + proposal.
    assert "HUMAN-WINS" in body
    assert "Dorn" in body
    assert "Mer-kee" not in body  # proposer entry lost the conflict


async def test_proposer_wm_failure_soft_degrades(manuscript_factory: Any) -> None:
    # WM raises → SOFT: ingest still succeeds, proposals fall back to {}, no raise.
    manuscript_factory(b"Chapter 1\nMerqi walked to Doorn.")
    fake = AsyncMock()
    fake.message.side_effect = TimeoutError("wm down")
    stage = AudiobookIngestStage(client=fake)
    out = await stage.handle(
        AudiobookIngestInput(),
        _ctx(
            {
                "manuscript_ref": "r",
                "manuscript_name": "x.txt",
                "audiobook": {"propose_pronunciations": True},
            }
        ),
    )
    assert fake.message.await_count >= 1
    # No proposals survived, no brief pronunciations → empty applied map, parse OK.
    assert out.pronunciations == {}
    assert [c.title for c in out.chapters] == ["Chapter 1"]
