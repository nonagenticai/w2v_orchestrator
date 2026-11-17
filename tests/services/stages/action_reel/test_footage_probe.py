"""footage_probe — which observations are allowed to badge a run "with warnings".

The honesty machinery is deliberate: a ``completed`` stage carrying a non-empty
``notes`` list is serialised ``degraded`` (``api/routes/runs/_shared.py``), which
the run page shows as "Completed with warnings". That signal is only worth
anything if it is RARE and TRUE.

A derivative that has not rendered yet (``preview_status='pending'``) is neither.
It is normal in-flight state, the original is playable, and the status is already
reported losslessly per clip on ``sources[].preview_status``. Noting it badged a
15/15 beat-locked run with ``solver_stubbed: false`` and ``shortfall_seconds:
0.0`` as degraded — which teaches reviewers to ignore the badge and so hides the
next REAL degradation.

So these tests pin both halves, because fixing the first by weakening the second
would be the worse bug:

* ``pending`` produces NO note ⇒ the run is not degraded.
* ``failed`` (the derivative worker errored) still produces a note, and an
  unreadable probe blob still produces a note ⇒ the run is still degraded.
"""

from __future__ import annotations

from uuid import UUID, uuid7

import pytest

from src.api.routes.runs._shared import _stage_response
from src.database.models import StageOutput, VideoSource
from src.services.runs.run_context import RunContext
from src.services.stages.action_reel import footage_probe as fp
from src.services.stages.action_reel.footage_probe import (
    FootageProbeInput,
    FootageProbeOutput,
    FootageProbeStage,
)
from src.services.stages.base import StagePreconditionError

# A well-formed ffprobe blob, so the only note a test can produce is the one it
# is actually about (a zero duration emits its own, separate note).
_GOOD_PROBE: dict[str, object] = {
    "format": {"duration": "12.5"},
    "streams": [
        {
            "codec_type": "video",
            "codec_name": "h264",
            "avg_frame_rate": "30000/1001",
            "width": 3840,
            "height": 2160,
        }
    ],
}


# A MALFORMED blob: a broken-timestamp TS/MKV remux whose only duration is a
# NEGATIVE one on the video stream. `format` carries no duration at all, which is
# what forces `_read_probe` down to the stream fallback — put a negative in
# `format.duration` instead and the fallback masks it, so this shape is the one
# that actually reaches the caller with duration_s < 0.
_NEGATIVE_PROBE: dict[str, object] = {
    "streams": [
        {
            "codec_type": "video",
            "codec_name": "h264",
            "duration": "-30.0",
            "avg_frame_rate": "30000/1001",
            "width": 1920,
            "height": 1080,
        }
    ],
}


def _source(
    *,
    preview_status: str,
    error: str | None = None,
    probe: dict[str, object] | None = None,
    filename: str = "drone-a.mp4",
) -> VideoSource:
    """A detached ``video_sources`` row — no DB needed to describe one."""
    return VideoSource(
        id=uuid7(),
        project_id=uuid7(),
        filename=filename,
        object_key="w2v-video-sources/drone-a.mp4",
        content_type="video/mp4",
        bytes=134_000_000,
        preview_status=preview_status,
        preview_key="w2v-video-sources/drone-a.preview.mp4" if error is None else None,
        error=error,
        probe=_GOOD_PROBE if probe is None else probe,
    )


async def _run(
    monkeypatch: pytest.MonkeyPatch, rows: list[VideoSource], *, presignable: bool = True
) -> FootageProbeOutput:
    """Drive the real stage over ``rows``.

    The row LOADER is stubbed (it is a tenant-bound DB read, not what is under
    test) but ``_describe`` — the code that decides what becomes a note — runs
    for real. Presigning is stubbed too: it reaches MinIO and its own failure
    mode is already soft.
    """

    async def _load_rows(
        _factory: object, _project_id: UUID, _tenant_id: UUID | None
    ) -> list[VideoSource]:
        return rows

    async def _presign(key: str | None) -> str:
        # `presignable=False` reproduces the real soft failure exactly: `_presign`
        # swallows a storage error to "" (at log.info), so the caller sees an
        # empty url and nothing else.
        if not key or not presignable:
            return ""
        return f"https://minio.test/{key}"

    monkeypatch.setattr(FootageProbeStage, "_load_rows", staticmethod(_load_rows))
    monkeypatch.setattr(fp, "_presign", _presign)

    project_id = uuid7()
    stage = FootageProbeStage(session_factory=object())
    ctx = RunContext(
        run_id=uuid7(),
        deliverable_id=uuid7(),
        config_snapshot={"project_id": str(project_id)},
        prior_run_outputs={},
    )
    return await stage.handle(FootageProbeInput(), ctx)


def _degraded(output: FootageProbeOutput) -> bool:
    """What the run page would badge, via the real serializer."""
    so = StageOutput(
        id=uuid7(),
        run_id=uuid7(),
        stage_name="footage_probe",
        status="completed",
        output=output.model_dump(mode="json"),
    )
    return _stage_response(uuid7(), so).degraded


async def test_pending_derivatives_do_not_degrade_a_flawless_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE test. Every clip is intact; only the cosmetic proxies are still
    rendering. That is not a warning, and the run must not be badged as one."""
    output = await _run(monkeypatch, [_source(preview_status="pending") for _ in range(3)])

    assert output.notes == [], f"a still-rendering derivative is not a note; got {output.notes}"
    assert _degraded(output) is False, "a run whose only observation is pending previews is clean"


async def test_pending_is_still_reported_per_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dropping the note must not drop the INFORMATION: the reviewer still has to
    be able to see which clips are unrendered, or this is a cover-up, not a fix."""
    output = await _run(monkeypatch, [_source(preview_status="pending")])

    assert [s.preview_status for s in output.sources] == ["pending"]


async def test_a_failed_derivative_still_degrades_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half. ``failed`` means the derivative worker errored — a genuine
    soft-failure an operator should see, so the warning badge must survive."""
    output = await _run(
        monkeypatch,
        [_source(preview_status="failed", error="ffmpeg exit 1"), _source(preview_status="ready")],
    )

    assert len(output.notes) == 1, f"exactly the failed source is noted; got {output.notes}"
    assert "preview_status=failed" in output.notes[0]
    assert "ffmpeg exit 1" in output.notes[0], "the operator-facing error must reach the note"
    assert _degraded(output) is True, "a failed derivative must still badge the run degraded"


async def test_an_unreadable_probe_blob_still_degrades_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pre-existing zero-duration note is untouched: a clip with no measurable
    duration silently plans as 0s downstream, which is exactly worth a warning.

    Paired with a usable sibling on purpose — a run where the unreadable blob is the
    ONLY source has no product left to degrade INTO and is now a hard failure (see
    ``test_all_sources_unusable_is_a_hard_failure``). This test is about the note, so
    it keeps the run alive to observe it."""
    output = await _run(
        monkeypatch,
        [
            _source(preview_status="ready", probe={}, filename="drone-a.mp4"),
            _source(preview_status="ready", filename="drone-b.mp4"),
        ],
    )

    assert len(output.notes) == 1, f"got {output.notes}"
    assert "no probed duration" in output.notes[0]
    assert _degraded(output) is True


async def test_all_sources_unusable_is_a_hard_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """N rows that ALL failed to probe is the same product outcome as zero rows —
    an empty reel — so it earns the same hard failure, not a "degraded" badge."""
    with pytest.raises(StagePreconditionError) as exc:
        await _run(
            monkeypatch,
            [
                _source(preview_status="ready", probe={}, filename="drone-a.mp4"),
                _source(preview_status="ready", probe={}, filename="drone-b.mp4"),
            ],
        )

    message = str(exc.value)
    assert "all 2 video_sources" in message, f"the count must be nameable; got {message}"
    assert "drone-a.mp4" in message and "drone-b.mp4" in message, (
        f"the operator must be able to name the offending clips; got {message}"
    )


async def test_one_usable_source_still_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """THE fence. The hard failure is a WHOLE-SET condition: as long as ONE clip has
    a usable duration there is still a reel to cut, so the run must complete — noted
    and degraded, never failed. Over-firing here would turn a single bad upload into
    a dead run, which is the opposite of the bug being fixed."""
    output = await _run(
        monkeypatch,
        [
            _source(preview_status="ready", probe={}, filename="drone-a.mp4"),
            _source(preview_status="ready", filename="drone-b.mp4"),
        ],
    )

    assert output.total_duration_s == 12.5, "only the usable source contributes duration"
    assert len(output.notes) == 1, f"the unusable source is still noted; got {output.notes}"
    assert "no probed duration" in output.notes[0]
    assert _degraded(output) is True, "some-usable-some-not is degraded, not fatal"


async def test_a_negative_duration_cannot_cancel_out_a_usable_clip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed probe must not be able to SUBTRACT from good footage.

    ``_as_float`` passes through any parseable float, so a broken-timestamp TS/MKV
    remux reporting ``stream.duration = "-30.0"`` used to produce a negative
    ``duration_s``. Summed against a clean 12.5s clip that is -17.5 — i.e. a hard
    failure on a project whose footage is perfectly usable, and, whenever the sum
    happened to stay positive, a ``total_duration_s`` quietly SHORTER than the real
    footage. The clamp in ``_describe`` kills both: the bad source contributes 0s,
    is noted, and the reported total is exactly the clean clip's duration."""
    output = await _run(
        monkeypatch,
        [
            _source(preview_status="ready", probe=_NEGATIVE_PROBE, filename="broken-remux.mkv"),
            _source(preview_status="ready", filename="drone-b.mp4"),
        ],
    )

    assert output.total_duration_s == 12.5, (
        f"the negative must not drag the reported total down; got {output.total_duration_s}"
    )
    assert [s.duration_s for s in output.sources] == [0.0, 12.5], (
        f"the negative is clamped before it enters the output; got {output.sources}"
    )
    assert len(output.notes) == 1, f"the unusable source is still noted; got {output.notes}"
    assert "no probed duration" in output.notes[0], (
        f"a negative is as unusable as a zero and must be reported; got {output.notes[0]}"
    )
    assert _degraded(output) is True, "one bad probe among good footage is degraded, not fatal"


async def test_all_negative_durations_is_still_a_hard_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other side of the clamp: clamping must not turn the whole-set failure into
    a silent empty reel. Every source unusable is the same product outcome whether
    the probes read 0 or -30, so both raise."""
    with pytest.raises(StagePreconditionError) as exc:
        await _run(
            monkeypatch,
            [
                _source(preview_status="ready", probe=_NEGATIVE_PROBE, filename="broken-a.mkv"),
                _source(preview_status="ready", probe=_NEGATIVE_PROBE, filename="broken-b.mkv"),
            ],
        )

    message = str(exc.value)
    assert "all 2 video_sources" in message, f"got {message}"
    assert "broken-a.mkv" in message and "broken-b.mkv" in message, f"got {message}"


async def test_the_error_names_five_sources_and_counts_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``+N more`` cap is what keeps a 40-clip project's failure readable. It has
    never been exercised: every other case here has fewer than six sources, so the
    branch could be inverted or off by one and the suite would stay green."""
    with pytest.raises(StagePreconditionError) as exc:
        await _run(
            monkeypatch,
            [_source(preview_status="ready", probe={}, filename=f"clip-{i}.mp4") for i in range(7)],
        )

    message = str(exc.value)
    assert "all 7 video_sources" in message, f"got {message}"
    named = [f"clip-{i}.mp4" for i in range(5)]
    assert all(name in message for name in named), f"the first five are named; got {message}"
    assert "clip-5.mp4" not in message and "clip-6.mp4" not in message, (
        f"the overflow is counted, not listed; got {message}"
    )
    assert "(+2 more)" in message, f"the overflow count must be exact; got {message}"


async def test_zero_rows_still_raises_the_original_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unchanged behaviour: no uploads at all names ``video_sources``, not the probe."""
    with pytest.raises(StagePreconditionError) as exc:
        await _run(monkeypatch, [])

    assert "no video_sources" in str(exc.value)


async def test_unpresignable_clips_are_counted_and_noted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The footage is intact and every duration is readable — but not one clip
    has a playable url, because `_presign` swallowed the storage error to "" at
    ``log.info``. That reached the reviewer as a green segment gate whose clips
    do not play. It must be a note (so the run badges degraded), never a failure:
    the footage itself is fine."""
    output = await _run(
        monkeypatch,
        [_source(preview_status="ready", filename=f"clip-{i}.mp4") for i in range(2)],
        presignable=False,
    )

    assert [s.original_url for s in output.sources] == ["", ""]
    presign_notes = [note for note in output.notes if "presign" in note]
    assert len(presign_notes) == 1, f"one summary note, counted; got {output.notes}"
    assert "2 of 2 sources" in presign_notes[0], f"the count must be exact; got {presign_notes[0]}"
    assert "clip-0.mp4" in presign_notes[0] and "clip-1.mp4" in presign_notes[0]
    assert _degraded(output) is True, "clips that will not play must badge the run degraded"
