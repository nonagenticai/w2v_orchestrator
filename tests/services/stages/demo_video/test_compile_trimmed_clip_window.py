"""A record-TRIMMED clip must reach the compiler as a closed window.

``demo_video_record`` lowers a failed scenario's ``seconds`` below the file length
(``recorded_seconds``) to drop its timeout tail. The compiler seeks against the FILE
(``video_compiler_service`` ``src/services/renderer.py::_clip_seek_args``), so a shot
without ``clip_out_seconds`` is seeked INTO the frozen tail under ``clip_anchor='end'``
(the cold-open default) or a motion-peak lead. Replay shape: prod run ``01a0b007``,
clip-1 14.56s file trimmed to 9.56s usable.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest

from src.services.stages.demo_video.compile import DemoVideoCompileStage

FILE_SECONDS = 14.56
USABLE_SECONDS = 9.56
NARRATION_SECONDS = 3.0


def _compiler_shown_span(shot: dict[str, Any], file_len: float) -> tuple[float, float]:
    """The SOURCE span the compiler shows — ``renderer._clip_seek_args`` arithmetic.

    Ported from ``video_compiler_service`` origin/main (window mode vs. the
    anchor/lead branch), minus the sub-frame grid snap. The shot plays for
    ``max(narration, shot.seconds)``; a window shorter than that freeze-pads.
    """
    dur = max(float(shot.get("seconds") or 0.0), NARRATION_SECONDS)
    anchor = shot.get("clip_anchor") or "start"
    lead = float(shot.get("clip_lead_seconds") or 0.0)
    out = shot.get("clip_out_seconds")
    if out is not None:
        end = min(float(out), file_len)
        offset = max(end - dur, lead) if anchor == "end" else lead
        offset = offset if offset >= 0.1 else 0.0
        return offset, offset + min(max(end - offset, 0.0), dur)
    seekable = max(file_len - dur, 0.0)
    if anchor == "end":
        offset = max(seekable - 0.10, 0.0)
    elif lead > 0.0:
        offset = min(lead, seekable)
    else:
        offset = 0.0
    offset = offset if offset >= 0.1 else 0.0
    return offset, min(offset + dur, file_len)


def _build(
    lines: list[dict[str, Any]], clips: list[dict[str, Any]], cfg: dict[str, Any] | None = None
) -> dict[str, dict[str, Any]]:
    narrate = {
        "clips": [
            {
                "line_id": str(ln["id"]),
                "artifact_url": f"http://a/{ln['id']}.mp3",
                "seconds": NARRATION_SECONDS,
            }
            for ln in lines
        ]
    }
    outputs: dict[str, Any] = {
        "demo_video_gen": {"lines": copy.deepcopy(lines), "bdd_job_id": "job-1"},
        "demo_video_narrate": narrate,
        "demo_video_record": {"clips": copy.deepcopy(clips)},
    }
    ctx: Any = SimpleNamespace(
        config_snapshot=copy.deepcopy(cfg or {}),
        prior_output=lambda stage: outputs.get(stage),
    )
    stage = DemoVideoCompileStage.__new__(DemoVideoCompileStage)
    return {s["line_id"]: s for s in stage.build_shots(ctx)}


def _trimmed(line_id: str, **extra: Any) -> dict[str, Any]:
    return {
        "line_id": line_id,
        "video_url": f"http://v/{line_id}.mp4",
        "seconds": USABLE_SECONDS,
        "recorded_seconds": FILE_SECONDS,
        **extra,
    }


def _assert_no_tail(shot: dict[str, Any]) -> None:
    # The rendered span first — that is the defect; the field is only the mechanism.
    _start, end = _compiler_shown_span(shot, FILE_SECONDS)
    assert end <= USABLE_SECONDS + 1e-9, f"shows the dead tail up to {end:.3f}s"
    assert shot["clip_out_seconds"] == pytest.approx(USABLE_SECONDS)
    lead = shot.get("clip_lead_seconds")
    if lead is not None:  # the compiler rejects clip_out_seconds <= clip_lead_seconds
        assert lead < shot["clip_out_seconds"]


def test_anchor_end_on_a_trimmed_clip_shows_no_tail() -> None:
    shots = _build(
        [
            {"id": "clip-0", "text": "open"},
            {"id": "clip-1", "text": "add", "seconds": 5.0, "clip_anchor": "end"},
        ],
        [{"line_id": "clip-0", "video_url": "http://v/0.mp4", "seconds": 4.0}, _trimmed("clip-1")],
    )
    _assert_no_tail(shots["clip-1"])
    # The settled tail of the USABLE footage, not of the file.
    assert _compiler_shown_span(shots["clip-1"], FILE_SECONDS)[1] == pytest.approx(USABLE_SECONDS)


def test_motion_peak_auto_offset_on_a_trimmed_clip_shows_no_tail() -> None:
    peak_late = [
        {"t": 2.0, "motion": 0.001},
        {"t": 5.0, "motion": 0.002},
        {"t": 9.0, "motion": 0.40},
    ]
    shots = _build(
        [
            {"id": "clip-0", "text": "open"},
            {"id": "clip-1", "text": "add", "seconds": 5.0, "clip_anchor": "auto"},
        ],
        [
            {"line_id": "clip-0", "video_url": "http://v/0.mp4", "seconds": 4.0},
            _trimmed("clip-1", motion_timeline=peak_late),
        ],
    )
    _assert_no_tail(shots["clip-1"])


def test_an_authored_earlier_out_point_is_kept_and_a_later_one_is_cut() -> None:
    shots = _build(
        [
            {"id": "clip-0", "text": "a", "clip_out_seconds": 6.0},
            {"id": "clip-1", "text": "b", "clip_lead_seconds": 12.0, "clip_out_seconds": 13.5},
        ],
        [_trimmed("clip-0"), _trimmed("clip-1")],
    )
    assert shots["clip-0"]["clip_out_seconds"] == pytest.approx(6.0)
    _assert_no_tail(shots["clip-1"])


def test_an_untrimmed_clip_gets_no_out_point() -> None:
    shots = _build(
        [{"id": "clip-1", "text": "a", "clip_anchor": "end"}],
        [{"line_id": "clip-1", "video_url": "http://v/1.mp4", "seconds": 9.64}],
    )
    assert "clip_out_seconds" not in shots["clip-1"]


def _compiler_accepts_window(shot: dict[str, Any]) -> None:
    """``video_compiler_service`` ``src/api/schemas.py::Shot._clip_window_is_ordered``
    (origin/main): ``clip_out_seconds <= clip_lead_seconds`` is a 422. Ported verbatim —
    an omitted lead is the compiler's default 0.0."""
    out = shot.get("clip_out_seconds")
    lead = float(shot.get("clip_lead_seconds") or 0.0)
    if out is not None and out <= lead:
        raise AssertionError(
            f"422: clip_out_seconds must be greater than clip_lead_seconds ({out} <= {lead})"
        )


def test_a_cold_open_lead_past_the_usable_end_never_reaches_the_compiler_as_out_le_lead() -> None:
    """A hook asks for a 12s lead on a 14.56s file trimmed to 9.56s usable.

    ``cold_open_window`` clamps the lead to the clip's (usable) ``seconds`` — i.e. to
    EXACTLY ``clip_out_seconds``, the one value the compiler rejects. The window bound
    re-expresses it as ``out - shot`` (9.56 - 5.0 = 4.56), the same clamp the compiler
    applies to a lead in seek mode.
    """
    shots = _build(
        [{"id": "clip-1", "text": "open", "seconds": 5.0}],
        [_trimmed("clip-1")],
        cfg={"direction": {"hook": {"cold_open_anchor": "start", "cold_open_lead_seconds": 12.0}}},
    )
    shot = shots["clip-1"]
    assert shot["clip_anchor"] == "start"
    assert shot["clip_out_seconds"] == pytest.approx(USABLE_SECONDS)
    _compiler_accepts_window(shot)
    assert shot["clip_lead_seconds"] < shot["clip_out_seconds"]
    assert shot["clip_lead_seconds"] == pytest.approx(USABLE_SECONDS - 5.0)
    _assert_no_tail(shot)
