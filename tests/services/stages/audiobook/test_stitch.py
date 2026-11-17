"""audiobook_stitch stage — per-chapter + combined concat via video_compile."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.services.runs.run_context import RunContext
from src.services.stages.audiobook.stitch import AudiobookStitchStage
from src.services.stages.base import StagePreconditionError
from src.services.tts_storage import set_storage_for_tests


def _ctx() -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": {"audiobook": {"output_format": "mp3", "chapter_gap_ms": 500}}},
        prior_run_outputs={
            "audiobook_plan": {
                "chapters": [
                    {"id": "ch1", "title": "One", "sequence": 0, "chunks": []},
                    {"id": "ch2", "title": "Two", "sequence": 1, "chunks": []},
                ]
            },
            "audiobook_narrate": {
                "clips": [
                    # deliberately out of order to prove stitch sorts by sequence
                    {
                        "id": "a2",
                        "chapter_id": "ch1",
                        "chunk_id": "k2",
                        "chapter_sequence": 0,
                        "chunk_sequence": 1,
                        "artifact_url": "https://m/a2.wav",
                        "seconds": 3.0,
                    },
                    {
                        "id": "a1",
                        "chapter_id": "ch1",
                        "chunk_id": "k1",
                        "chapter_sequence": 0,
                        "chunk_sequence": 0,
                        "artifact_url": "https://m/a1.wav",
                        "seconds": 2.0,
                    },
                    {
                        "id": "b1",
                        "chapter_id": "ch2",
                        "chunk_id": "k3",
                        "chapter_sequence": 1,
                        "chunk_sequence": 0,
                        "artifact_url": "https://m/b1.wav",
                        "seconds": 4.0,
                    },
                ]
            },
        },
    )


async def test_stitch_idempotency_keys_are_per_chapter_and_stable_across_resume() -> None:
    """Each concat needs a DISTINCT key (a bare run_id:stage would dedupe the N+1
    concats to the first chapter's audio), and each must be STABLE across a
    resumable-timeout re-dispatch so the compiler dedupes the re-submit."""
    fake = AsyncMock()
    fake.concat_audio.side_effect = [
        {"artifact_url": "https://m/ch1.mp3", "duration_seconds": 5.0},
        {"artifact_url": "https://m/ch2.mp3", "duration_seconds": 4.0},
        {"artifact_url": "https://m/book.mp3", "duration_seconds": 9.5},
    ] * 2  # two runs' worth (the re-dispatch)
    stage = AudiobookStitchStage(client=fake)
    ctx = _ctx()
    rid = ctx.run_id

    await stage.run({}, ctx=ctx)
    first = [c.kwargs["idempotency_key"] for c in fake.concat_audio.call_args_list]
    assert first == [
        f"{rid}:audiobook_stitch:chapter:ch1",
        f"{rid}:audiobook_stitch:chapter:ch2",
        f"{rid}:audiobook_stitch:combined",
    ]
    assert len(set(first)) == 3  # distinct — no wrong dedup across the 3 concats

    await stage.run({}, ctx=ctx)  # simulate the resume re-dispatch
    second = [c.kwargs["idempotency_key"] for c in fake.concat_audio.call_args_list[3:]]
    assert second == first  # identical keys → compiler dedupes, no duplicate concats


async def test_stitch_prefers_qa_corrected_clip_over_narrate() -> None:
    # When audiobook_qa (the surgical re-roll) emits a corrected clip for a chunk,
    # stitch must feed the compiler the CORRECTED clip (new id/url/seconds) for that
    # chunk_id while keeping every other chunk's original narrate clip + ordering.
    fake = AsyncMock()
    fake.concat_audio.side_effect = [
        {"artifact_url": "https://m/ch1.mp3", "duration_seconds": 5.0},
        {"artifact_url": "https://m/ch2.mp3", "duration_seconds": 4.0},
        {"artifact_url": "https://m/book.mp3", "duration_seconds": 9.5},
    ]
    stage = AudiobookStitchStage(client=fake)
    ctx = _ctx()
    ctx.prior_run_outputs["audiobook_qa"] = {
        "corrected_clips": [
            {
                "chunk_id": "k1",
                "id": "a1fixed",
                "artifact_url": "https://m/a1fixed.wav",
                "seconds": 2.5,
            }
        ]
    }
    await stage.run({}, ctx=ctx)

    ch1_clips = fake.concat_audio.await_args_list[0].kwargs["clips"]
    by_id = {c["id"]: c for c in ch1_clips}
    # k1 now points at the corrected clip; k2 is untouched; order preserved.
    assert [c["id"] for c in ch1_clips] == ["a1fixed", "a2"]
    assert by_id["a1fixed"]["audio_url"] == "https://m/a1fixed.wav"
    assert by_id["a1fixed"]["seconds"] == 2.5


async def test_stitch_ignores_empty_qa_corrected_clips() -> None:
    # An opted-in qa run that corrected nothing (empty corrected_clips) leaves the
    # narrate clips byte-identical — same guarantee as a skipped qa stage.
    fake = AsyncMock()
    fake.concat_audio.side_effect = [
        {"artifact_url": "https://m/ch1.mp3", "duration_seconds": 5.0},
        {"artifact_url": "https://m/ch2.mp3", "duration_seconds": 4.0},
        {"artifact_url": "https://m/book.mp3", "duration_seconds": 9.5},
    ]
    stage = AudiobookStitchStage(client=fake)
    ctx = _ctx()
    ctx.prior_run_outputs["audiobook_qa"] = {"corrected_clips": []}
    await stage.run({}, ctx=ctx)
    ch1_clips = fake.concat_audio.await_args_list[0].kwargs["clips"]
    assert [c["id"] for c in ch1_clips] == ["a1", "a2"]


class _FakeResignStorage:
    """Re-signs every URL by tagging ``&fresh`` (mirrors the shot_presign fake)."""

    async def resign_presigned_url(self, url: str) -> str:
        return f"{url}&fresh"


async def test_stitch_resigns_clip_urls_before_concat() -> None:
    """A run paused past the 1h presign TTL: the per-chapter clip audio_urls AND the
    combined pass's per-chapter audio_urls must be re-minted before each concat POST,
    or video_compiler 403s the dead signature → concat 500s."""
    set_storage_for_tests(cast("Any", _FakeResignStorage()))
    try:
        fake = AsyncMock()
        fake.concat_audio.side_effect = [
            {"artifact_url": "https://m/ch1.mp3", "duration_seconds": 5.0},
            {"artifact_url": "https://m/ch2.mp3", "duration_seconds": 4.0},
            {"artifact_url": "https://m/book.mp3", "duration_seconds": 9.5},
        ]
        stage = AudiobookStitchStage(client=fake)
        await stage.run({}, ctx=_ctx())
    finally:
        set_storage_for_tests(None)

    # Per-chapter passes: the narration clip urls are re-signed.
    ch1_urls = [c["audio_url"] for c in fake.concat_audio.await_args_list[0].kwargs["clips"]]
    assert ch1_urls == ["https://m/a1.wav&fresh", "https://m/a2.wav&fresh"]
    ch2_urls = [c["audio_url"] for c in fake.concat_audio.await_args_list[1].kwargs["clips"]]
    assert ch2_urls == ["https://m/b1.wav&fresh"]
    # Combined pass: the per-chapter stitched urls are re-signed too.
    combined_urls = [c["audio_url"] for c in fake.concat_audio.await_args_list[2].kwargs["clips"]]
    assert combined_urls == ["https://m/ch1.mp3&fresh", "https://m/ch2.mp3&fresh"]


def _ctx_with(audiobook_cfg: dict[str, Any]) -> RunContext:
    ctx = _ctx()
    ctx.config_snapshot["brief"]["audiobook"] = audiobook_cfg  # type: ignore[index]
    return ctx


def _combined_side_effect() -> list[dict[str, Any]]:
    return [
        {"artifact_url": "https://m/ch1.m4b", "duration_seconds": 5.0},
        {"artifact_url": "https://m/ch2.m4b", "duration_seconds": 4.0},
        {
            "artifact_url": "https://m/book.m4b",
            "duration_seconds": 9.5,
            "acx_report": {
                "integrated_lufs": -20.1,
                "true_peak_dbtp": -3.4,
                "noise_floor_db": -70.0,
                "pass": True,
            },
        },
    ]


async def test_stitch_default_brief_sends_no_acx_master_or_chapters() -> None:
    """Byte-identity: a legacy brief (mp3, no master block) sends NEITHER
    acx_master NOR chapters — the combined request is exactly today's."""
    fake = AsyncMock()
    fake.concat_audio.side_effect = [
        {"artifact_url": "https://m/ch1.mp3", "duration_seconds": 5.0},
        {"artifact_url": "https://m/ch2.mp3", "duration_seconds": 4.0},
        {"artifact_url": "https://m/book.mp3", "duration_seconds": 9.5},
    ]
    stage = AudiobookStitchStage(client=fake)
    out = await stage.run({}, ctx=_ctx())
    combined = fake.concat_audio.await_args_list[2].kwargs
    assert combined["acx_master"] is None
    assert combined["chapters"] is None
    assert out.acx_report is None


async def test_stitch_acx_master_m4b_sends_master_and_chapters() -> None:
    """Opt-in brief (m4b + master) ⇒ the combined concat carries acx_master +
    cumulative chapter markers, and the stage surfaces the acx_report."""
    fake = AsyncMock()
    fake.concat_audio.side_effect = _combined_side_effect()
    stage = AudiobookStitchStage(client=fake)
    ctx = _ctx_with({"output_format": "m4b", "chapter_gap_ms": 500, "master": True})
    out = await stage.run({}, ctx=ctx)

    combined = fake.concat_audio.await_args_list[2].kwargs
    assert combined["output_format"] == "m4b"
    assert combined["acx_master"] == {"enabled": True}
    # ch1 at 0; ch2 at 5.0s + 500ms gap = 5500ms.
    assert combined["chapters"] == [
        {"title": "One", "start_ms": 0},
        {"title": "Two", "start_ms": 5500},
    ]
    assert out.acx_report is not None and out.acx_report["pass"] is True
    # Per-chapter passes stay unmastered (master is a whole-book seam).
    assert fake.concat_audio.await_args_list[0].kwargs.get("acx_master") is None


async def test_stitch_missing_narration_raises() -> None:
    fake = AsyncMock()
    stage = AudiobookStitchStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
        prior_run_outputs={"audiobook_plan": {"chapters": [{"id": "c", "sequence": 0}]}},
    )
    with pytest.raises(StagePreconditionError, match="audiobook_narrate"):
        await stage.run({}, ctx=ctx)
    fake.concat_audio.assert_not_called()


# ── an unmeasured clip duration must never become a chapter marker (tts_gateway#38)


async def test_stitch_sends_none_not_zero_for_an_unmeasured_clip() -> None:
    """``AudioClip.seconds`` on the compiler is ADVISORY (the concat is always
    re-probed) and constrained ``gt=0`` — so forwarding a fabricated 0.0 for a
    clip the gateway could not measure 422s the whole paid stitch. An unmeasured
    clip must go over the wire as ``None`` (= "unknown"), never as a number."""
    fake = AsyncMock()
    fake.concat_audio.side_effect = [
        {"artifact_url": "https://m/ch1.mp3", "duration_seconds": 5.0},
        {"artifact_url": "https://m/ch2.mp3", "duration_seconds": 4.0},
        {"artifact_url": "https://m/book.mp3", "duration_seconds": 9.5},
    ]
    ctx = _ctx()
    # k1's duration is unmeasured. 0.0 is what a legacy/resumed narrate output
    # persisted for exactly this case (and what any future re-fabrication would
    # look like) — stitch must refuse to forward it, not just pass a None through.
    for clip in ctx.prior_run_outputs["audiobook_narrate"]["clips"]:
        if clip["chunk_id"] == "k1":
            clip["seconds"] = 0.0
            clip["duration_unmeasured"] = True

    await AudiobookStitchStage(client=fake).run({}, ctx=ctx)

    ch1_clips = fake.concat_audio.call_args_list[0].kwargs["clips"]
    by_id = {c["id"]: c for c in ch1_clips}
    assert by_id["a1"]["seconds"] is None, (
        "an unmeasured clip must not be advertised to the compiler as 0.0"
    )
    assert by_id["a2"]["seconds"] == 3.0  # a measured neighbour is untouched


async def test_stitch_re_polls_gateway_for_fresh_clip_urls(monkeypatch) -> None:
    """Gateway clips live in the gateway's OWN bucket, which refresh_shot_asset_urls
    cannot re-sign — their narrate-time URLs (1h TTL) 403 at concat once stitch runs
    past that hour. Stitch re-polls the batch and substitutes fresh URLs by job id."""
    from src.clients.schemas.tts_gateway import BatchStatusResult
    from src.config import clear_settings_cache

    monkeypatch.setenv("DOWNSTREAM_AUDIOBOOK_NARRATE_VIA_GATEWAY", "true")
    clear_settings_cache()
    ctx = _ctx()
    gw = AsyncMock()
    gw.get_batch.return_value = BatchStatusResult.model_validate(
        {
            "batch_id": f"{ctx.run_id}:audiobook_narrate",
            "jobs": [
                {
                    "idempotency_key": k,
                    "job_id": j,
                    "status": "done",
                    "result_url": f"https://fresh/{j}",
                }
                for k, j in (("x1", "a1"), ("x2", "a2"), ("x3", "b1"))
            ],
        }
    )
    fake = AsyncMock()
    fake.concat_audio.side_effect = [
        {"artifact_url": "https://m/ch1.mp3", "duration_seconds": 5.0},
        {"artifact_url": "https://m/ch2.mp3", "duration_seconds": 4.0},
        {"artifact_url": "https://m/book.mp3", "duration_seconds": 9.5},
    ]
    try:
        await AudiobookStitchStage(client=fake, gateway_factory=lambda: gw).run({}, ctx=ctx)
    finally:
        clear_settings_cache()

    gw.get_batch.assert_awaited_with(f"{ctx.run_id}:audiobook_narrate")
    assert gw.get_batch.await_count == 2  # once per chapter, right before its concat
    chapter_urls = [
        [c["audio_url"] for c in call.kwargs["clips"]]
        for call in fake.concat_audio.call_args_list[:2]
    ]
    assert chapter_urls == [["https://fresh/a1", "https://fresh/a2"], ["https://fresh/b1"]]


async def test_stitch_does_not_poll_gateway_when_narrate_was_direct() -> None:
    gw = AsyncMock()
    fake = AsyncMock()
    fake.concat_audio.side_effect = [
        {"artifact_url": "https://m/ch1.mp3", "duration_seconds": 5.0},
        {"artifact_url": "https://m/ch2.mp3", "duration_seconds": 4.0},
        {"artifact_url": "https://m/book.mp3", "duration_seconds": 9.5},
    ]
    await AudiobookStitchStage(client=fake, gateway_factory=lambda: gw).run({}, ctx=_ctx())
    gw.get_batch.assert_not_awaited()
