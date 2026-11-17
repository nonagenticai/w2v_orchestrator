import pytest

from src.services.stages.shot_assembly import (
    FABRICATED_HOLD_KEY,
    UNDERSPOKEN_HOLD_KEY,
    assemble_shots,
    drain_fabricated_hold_notes,
    footage_tracks_by_line,
    normalize_footage_viewport,
)
from tests.support.geometry_wire import cursor_wire, focus_wire


def _shot(line_id: str, seconds: float = 2.0) -> dict:
    return {
        "id": f"s-{line_id}",
        "line_id": line_id,
        "description": f"d {line_id}",
        "seconds": seconds,
    }


def test_audio_joins_by_line_and_overrides_seconds() -> None:
    shots = assemble_shots(
        storyboard_shots=[_shot("l1", 2.0)],
        images=[],
        narration_clips=[{"line_id": "l1", "artifact_url": "http://a/clip.wav", "seconds": 7.5}],
    )
    assert len(shots) == 1
    assert shots[0]["audio_url"] == "http://a/clip.wav"
    # narration duration wins over the storyboard's requested seconds
    assert shots[0]["seconds"] == 7.5
    assert shots[0]["description"] == "d l1"
    assert "image_url" not in shots[0]


def test_images_distribute_round_robin_across_shots() -> None:
    shots = assemble_shots(
        storyboard_shots=[_shot("l1"), _shot("l2"), _shot("l3")],
        images=[{"image_url": "img0"}, {"image_url": "img1"}],
        narration_clips=[],
    )
    assert [s["image_url"] for s in shots] == ["img0", "img1", "img0"]


def test_image_only_when_no_narration() -> None:
    shots = assemble_shots(
        storyboard_shots=[_shot("l1", 3.0)],
        images=[{"image_url": "only.png"}],
        narration_clips=[],
    )
    assert shots[0]["image_url"] == "only.png"
    assert "audio_url" not in shots[0]
    assert shots[0]["seconds"] == 3.0


def test_no_sfx_keys_when_unset() -> None:
    shots = assemble_shots(
        storyboard_shots=[_shot("l1"), _shot("l2")],
        images=[],
        narration_clips=[],
    )
    assert all("sfx" not in s and "transition_sfx_id" not in s for s in shots)


def test_sfx_and_transition_thread_through() -> None:
    shots = assemble_shots(
        storyboard_shots=[_shot("l1"), _shot("l2"), _shot("l3")],
        images=[],
        narration_clips=[],
        sfx_by_line={"l2": [{"sfx_id": "ding_positive", "at_seconds": 1.0, "gain_db": -8.0}]},
        transition_sfx_id="whoosh_soft",
    )
    # Accent cue lands only on its line.
    assert shots[1]["sfx"] == [{"sfx_id": "ding_positive", "at_seconds": 1.0, "gain_db": -8.0}]
    assert "sfx" not in shots[0] and "sfx" not in shots[2]
    # Transition rides every cut EXCEPT the open (no incoming cut on shot 0).
    assert "transition_sfx_id" not in shots[0]
    assert shots[1]["transition_sfx_id"] == "whoosh_soft"
    assert shots[2]["transition_sfx_id"] == "whoosh_soft"


def test_transition_gain_rides_every_cut_except_the_open() -> None:
    shots = assemble_shots(
        storyboard_shots=[_shot("l1"), _shot("l2"), _shot("l3")],
        images=[],
        narration_clips=[],
        transition_sfx_id="whoosh_soft",
        transition_sfx_gain_db=-4.0,
    )
    # The gain tracks the id exactly: never on the open, on every later cut.
    assert "transition_sfx_gain_db" not in shots[0]
    assert shots[1]["transition_sfx_gain_db"] == -4.0
    assert shots[2]["transition_sfx_gain_db"] == -4.0


def test_transition_gain_absent_when_unset() -> None:
    shots = assemble_shots(
        storyboard_shots=[_shot("l1"), _shot("l2")],
        images=[],
        narration_clips=[],
        transition_sfx_id="whoosh_soft",
    )
    assert all("transition_sfx_gain_db" not in s for s in shots)


def test_transition_gain_needs_an_id_to_ride() -> None:
    # No cue → no orphan gain key on any shot.
    shots = assemble_shots(
        storyboard_shots=[_shot("l1"), _shot("l2")],
        images=[],
        narration_clips=[],
        transition_sfx_gain_db=-4.0,
    )
    assert all("transition_sfx_gain_db" not in s for s in shots)


def test_images_skip_blank_urls() -> None:
    shots = assemble_shots(
        storyboard_shots=[_shot("l1"), _shot("l2")],
        images=[{"image_url": None}, {"image_url": "real"}],
        narration_clips=[],
    )
    # Only the non-blank image is usable, so the pool is size 1: there is no
    # alternative asset, so both shots cycle onto it (the anti-repeat guard
    # only engages when a pool has >=2 assets to choose between).
    assert [s["image_url"] for s in shots] == ["real", "real"]


def test_captions_applied_by_line() -> None:
    shots = assemble_shots(
        storyboard_shots=[_shot("l1"), _shot("l2")],
        images=[],
        narration_clips=[],
        captions_by_line={"l1": "spoken one"},
    )
    assert shots[0]["caption"] == "spoken one"
    assert "caption" not in shots[1]


def test_fit_applied_by_line_only_when_set() -> None:
    """AI-image lines carry fit='cover' (crop-to-fill, no navy bars); every
    other line omits the key so the compositor uses its 'contain' default."""
    shots = assemble_shots(
        storyboard_shots=[_shot("l1"), _shot("l2")],
        images=[],
        narration_clips=[],
        fit_by_line={"l1": "cover"},
    )
    assert shots[0]["fit"] == "cover"
    assert "fit" not in shots[1]


def test_authored_fit_on_shot_beats_the_map() -> None:
    shots = assemble_shots(
        storyboard_shots=[{**_shot("l1"), "fit": "scroll"}, _shot("l2")],
        images=[],
        narration_clips=[],
        fit_by_line={"l1": "cover"},  # smart-auto → overridden by the authored scroll
    )
    assert shots[0]["fit"] == "scroll"
    assert "fit" not in shots[1]


def test_video_clips_distribute_round_robin() -> None:
    shots = assemble_shots(
        storyboard_shots=[_shot("l1"), _shot("l2"), _shot("l3")],
        images=[],
        narration_clips=[],
        video_clips=[{"video_url": "v0.mp4"}, {"video_url": "v1.mp4"}],
    )
    assert [s["video_url"] for s in shots] == ["v0.mp4", "v1.mp4", "v0.mp4"]


def test_video_clips_skip_blank_urls() -> None:
    shots = assemble_shots(
        storyboard_shots=[_shot("l1"), _shot("l2")],
        images=[],
        narration_clips=[],
        video_clips=[{"video_url": None}, {"video_url": "real.mp4"}],
    )
    # Only the non-blank clip is usable, so the pool is size 1: no
    # alternative asset exists, so both shots get it (a held/repeated visual
    # is the lesser evil vs. a wall of text cards).
    assert [s["video_url"] for s in shots] == ["real.mp4", "real.mp4"]
    assert shots[1]["description"] == "d l2"


def test_no_video_clips_keeps_prior_behavior() -> None:
    shots = assemble_shots(
        storyboard_shots=[_shot("l1", 4.0)],
        images=[],
        narration_clips=[],
        video_clips=None,
    )
    assert "video_url" not in shots[0]


def test_description_omitted_when_absent() -> None:
    # A shot built without a storyboard description (lesson/app_review) must
    # omit the key entirely, not send `description: null` on the wire.
    shots = assemble_shots(
        storyboard_shots=[{"id": "s1", "line_id": "l1", "seconds": 3.0}],
        images=[],
        narration_clips=[],
    )
    assert shots[0] == {"id": "s1", "line_id": "l1", "seconds": 3.0}
    assert "description" not in shots[0]


def test_image_by_line_joins_per_line() -> None:
    shots = assemble_shots(
        storyboard_shots=[{"id": "s1", "line_id": "l1", "seconds": 2.0}, _shot("l2")],
        images=[],
        narration_clips=[],
        image_by_line={"l1": "keyed.png"},
    )
    assert shots[0]["image_url"] == "keyed.png"
    assert "image_url" not in shots[1]  # l2 has no keyed image


# falls back to round-robin


def test_clip_framing_attaches_only_to_footage_shots() -> None:
    # A footage shot (has a keyed video) carries the line's clip_anchor/lead;
    # a non-footage (image/text) shot ignores them.
    footage = {**_shot("l1"), "clip_anchor": "end", "clip_lead_seconds": 4}
    carded = {**_shot("l2"), "clip_anchor": "end", "clip_lead_seconds": 4}
    shots = assemble_shots(
        storyboard_shots=[footage, carded],
        images=[],
        narration_clips=[],
        video_by_line={"l1": "http://v/clip.mp4"},
        image_by_line={"l2": "http://i/card.png"},
    )
    by = {s["line_id"]: s for s in shots}
    assert by["l1"]["video_url"] == "http://v/clip.mp4"
    assert by["l1"]["clip_anchor"] == "end"
    assert by["l1"]["clip_lead_seconds"] == 4.0
    # card shot: no footage → framing keys omitted
    assert "clip_anchor" not in by["l2"]
    assert "clip_lead_seconds" not in by["l2"]


def test_clip_look_attaches_grade_and_overlays_when_present() -> None:
    # Studio look edits ride the spine shot: a colour grade + text overlays pass
    # through unchanged onto the built compositor shot (any shot, not just footage).
    adjust = {"brightness": 0.1, "contrast": 1.2, "saturation": 1.0, "gamma": 1.0, "preset": "warm"}
    overlays = [
        {"text": "Chapter One", "position": "bottom-center", "font_size": 48, "color": "white"}
    ]
    graded = {**_shot("l1"), "adjust": adjust, "overlays": overlays}
    plain = _shot("l2")
    shots = assemble_shots(
        storyboard_shots=[graded, plain],
        images=[],
        narration_clips=[],
    )
    by = {s["line_id"]: s for s in shots}
    assert by["l1"]["adjust"] == adjust
    assert by["l1"]["overlays"] == overlays
    # A shot with no look edits omits both keys (additive / byte-identical default).
    assert "adjust" not in by["l2"]
    assert "overlays" not in by["l2"]


def test_empty_budget_and_bound_maps_behave_exactly_like_none() -> None:
    """The compute-plan seam is inert when reconcile_timeline did not run: a compile
    that passes empty maps must produce the SAME shots as one that passes None."""
    storyboard_shots = [_shot("l1", 2.0), _shot("l2", 3.0)]
    narration_clips = [{"line_id": "l1", "artifact_url": "http://a/1.wav", "seconds": 7.5}]

    empty = assemble_shots(
        storyboard_shots=storyboard_shots,
        images=[],
        narration_clips=narration_clips,
        seconds_by_line={},
        min_max_by_line={},
    )
    omitted = assemble_shots(
        storyboard_shots=storyboard_shots,
        images=[],
        narration_clips=narration_clips,
        seconds_by_line=None,
        min_max_by_line=None,
    )
    assert empty == omitted


def test_budget_fills_only_when_no_narration_and_bound_clamps() -> None:
    shots = assemble_shots(
        storyboard_shots=[_shot("l1", 2.0), _shot("l2", 2.0)],
        images=[],
        # l1 is narrated (clip wins over the budget); l2 has no narration (budget fills).
        narration_clips=[{"line_id": "l1", "artifact_url": "http://a/1.wav", "seconds": 7.5}],
        seconds_by_line={"l1": 5.0, "l2": 6.0},
        min_max_by_line={"l1": (3.0, 6.5)},
    )
    by = {s["line_id"]: s for s in shots}
    # Narration (7.5) beat the 5.0 budget, then the (3.0, 6.5) window clamped it.
    assert by["l1"]["seconds"] == 6.5
    # No narration → the planned budget fills the beat; no bound → no clamp.
    assert by["l2"]["seconds"] == 6.0
    # Step B forward: the beat with a bound carries its upper bound (6.5) to the
    # compiler as max_seconds (the freeze-pad ceiling); the beat WITHOUT a bound
    # omits the key entirely — a missing key, not max_seconds: null.
    assert by["l1"]["max_seconds"] == 6.5
    assert "max_seconds" not in by["l2"]


def test_footage_beat_pinned_to_its_measured_length_is_never_trimmed() -> None:
    """C0: reconcile_timeline pins a footage beat to min == max == the MEASURED clip
    length, so the unconditional clamp can only ever restore that length — the beat
    survives unchanged, whether or not a shorter narration clip rides over it."""
    measured = 6.0
    pinned = {"l1": (measured, measured)}
    unnarrated = assemble_shots(
        storyboard_shots=[_shot("l1", measured)],
        images=[],
        narration_clips=[],
        video_by_line={"l1": "http://v/1.mp4"},
        seconds_by_line={"l1": measured},
        min_max_by_line=pinned,
    )
    assert unnarrated[0]["seconds"] == measured

    narrated = assemble_shots(
        storyboard_shots=[_shot("l1", measured)],
        images=[],
        narration_clips=[{"line_id": "l1", "artifact_url": "http://a/1.wav", "seconds": 4.0}],
        video_by_line={"l1": "http://v/1.mp4"},
        seconds_by_line={"l1": measured},
        min_max_by_line=pinned,
    )
    # The shorter VO does NOT shorten the footage beat — it is held at its full length.
    assert narrated[0]["seconds"] == measured
    # The pinned max (== measured here) is forwarded as the compiler's freeze-pad
    # ceiling. In prod an over-narrated footage beat carries max = clip + pad; here
    # min == max == measured, so the ceiling equals the clip (a 0-length pad).
    assert narrated[0]["max_seconds"] == measured


def test_clip_look_omits_empty_overlays() -> None:
    # An empty overlays list is falsy → the key is omitted, byte-identical to a
    # shot that never carried one (matches the presence-is-opt-in framing story).
    shots = assemble_shots(
        storyboard_shots=[{**_shot("l1"), "overlays": []}],
        images=[],
        narration_clips=[],
    )
    assert "overlays" not in shots[0]


# --- per-clip voiceover upload (Studio narration replacement) -------------------

_TTS = {"line_id": "l1", "artifact_url": "http://a/tts.wav", "seconds": 3.0}
_VO = "http://minio.test/w2v-orchestrator/artifacts/voiceovers/t/vo/take.mp3"


def test_voiceover_upload_overrides_the_tts_narration_url() -> None:
    # The edit-supplied audio REPLACES the narration clip's url — otherwise the
    # merged ClipEdit is invisible to the compiler and the shot keeps its TTS take.
    shots = assemble_shots(
        storyboard_shots=[{**_shot("l1", 2.0), "audio_url": _VO, "audio_seconds": 12.4}],
        images=[],
        narration_clips=[_TTS],
    )
    assert shots[0]["audio_url"] == _VO


# --- ORCH-3.2 directed footage: focus/cursor track threading --------------------

_FOCUS: list[dict[str, object]] = [
    {"t": 0.5, "bbox": {"x": 10, "y": 20, "w": 100, "h": 40}, "action": "click"}
]
_CURSOR: list[dict[str, object]] = [
    {"t": 0.0, "x": 5, "y": 5, "click": False},
    {"t": 0.5, "x": 60, "y": 40, "click": True},
]
# What those RECORDER shapes become on the wire: the compiler declares
# ``FocusKey.bbox`` as a 4-tuple (never the recorder's ``{x,y,w,h}`` dict) and does
# not declare ``action``. See tests/contracts/test_directed_footage_geometry_contract.
_FOCUS_WIRE = focus_wire(_FOCUS)
_CURSOR_WIRE = cursor_wire(_CURSOR)


def test_directed_footage_tracks_copied_onto_footage_shots_when_present() -> None:
    # A footage line (keyed video_url) carries its focus/cursor geometry; the
    # compiler drives a synthetic cursor + auto-zoom from it.
    shots = assemble_shots(
        storyboard_shots=[_shot("l1"), _shot("l2")],
        images=[],
        narration_clips=[],
        video_by_line={"l1": "http://v/clip.mp4"},
        image_by_line={"l2": "http://i/card.png"},
        focus_track_by_line={"l1": _FOCUS},
        cursor_track_by_line={"l1": _CURSOR},
    )
    by = {s["line_id"]: s for s in shots}
    assert by["l1"]["focus_track"] == _FOCUS_WIRE
    assert by["l1"]["cursor_track"] == _CURSOR_WIRE
    # l2 was never keyed a track.
    assert "focus_track" not in by["l2"]


def test_directed_footage_tracks_omitted_on_non_footage_shots() -> None:
    # A track keyed to an image/text-card line (no video_url) never rides the shot —
    # the compiler cursor/zoom filters only make sense on real footage.
    shots = assemble_shots(
        storyboard_shots=[_shot("l2")],
        images=[],
        narration_clips=[],
        image_by_line={"l2": "http://i/card.png"},
        focus_track_by_line={"l2": _FOCUS},
        cursor_track_by_line={"l2": _CURSOR},
    )
    assert "focus_track" not in shots[0]
    assert "cursor_track" not in shots[0]


def test_directed_footage_absent_maps_byte_identical() -> None:
    # Presence-is-opt-in: empty/None track maps ⇒ neither key rides any shot ⇒
    # byte-identical to a plain footage render (the no-op guard).
    storyboard_shots = [_shot("l1"), _shot("l2")]
    with_none = assemble_shots(
        storyboard_shots=storyboard_shots,
        images=[],
        narration_clips=[],
        video_by_line={"l1": "http://v/clip.mp4"},
    )
    with_empty = assemble_shots(
        storyboard_shots=storyboard_shots,
        images=[],
        narration_clips=[],
        video_by_line={"l1": "http://v/clip.mp4"},
        focus_track_by_line={},
        cursor_track_by_line={},
    )
    assert with_none == with_empty
    assert all("focus_track" not in s and "cursor_track" not in s for s in with_none)


def test_directed_footage_does_not_disturb_author_clip_anchor() -> None:
    # An author's per-line clip_anchor still WINS (separate key); both coexist.
    footage = {**_shot("l1"), "clip_anchor": "end", "clip_lead_seconds": 4}
    shots = assemble_shots(
        storyboard_shots=[footage],
        images=[],
        narration_clips=[],
        video_by_line={"l1": "http://v/clip.mp4"},
        focus_track_by_line={"l1": _FOCUS},
        cursor_track_by_line={"l1": _CURSOR},
    )
    assert shots[0]["clip_anchor"] == "end"
    assert shots[0]["clip_lead_seconds"] == 4.0
    # Under the `end` anchor the compiler picks the window from the narrated duration,
    # so which recorded moment a focus keyframe lands on is unknown here ⇒ NO zoom
    # (merqi spike 01a0fc55: punch-ins on blank areas).
    assert "focus_track" not in shots[0]
    # The cursor is windowed by the same lead: both keys (0.0s, 0.5s) sit before the
    # 4.0s in-point, i.e. on a screen this shot never shows, so no cursor rides. It
    # used to be copied raw, which drew a click ring at the wrong moment (merqi
    # showcase spike 01a0fd4d-9948, 16s frame).
    assert "cursor_track" not in shots[0]


# ─── punch-ins target ONLY the interacted element (merqi spike 01a0fc55) ──────────
def _focus_for(track: list[dict[str, object]], **shot_extra: object) -> object:
    shots = assemble_shots(
        storyboard_shots=[{**_shot("l1"), **shot_extra}],
        images=[],
        narration_clips=[],
        video_by_line={"l1": "http://v/clip.mp4"},
        focus_track_by_line={"l1": track},
        footage_viewport_by_line={"l1": [1280, 720]},
    )
    return shots[0].get("focus_track")


def test_zoom_never_targets_a_zero_filled_placeholder_bbox() -> None:
    # FTR writes {0,0,0,0} for a keyframe that carried no bbox; the compiler floors it
    # to 1px and punches in at MAX zoom on the empty top-left corner.
    placeholder: dict[str, object] = {
        "t": 0.5,
        "bbox": {"x": 0, "y": 0, "w": 0, "h": 0},
        "action": "click",
    }
    assert _focus_for([placeholder]) is None


def test_zoom_never_targets_a_non_interaction_keyframe() -> None:
    # `reveal` is what FTR fills in when the entry carried no action: nothing was
    # interacted with, so there is no element to punch in on.
    reveal: dict[str, object] = {
        "t": 0.5,
        "bbox": {"x": 10, "y": 20, "w": 100, "h": 40},
        "action": "reveal",
    }
    assert _focus_for([reveal]) is None


def test_zoom_never_targets_an_off_screen_element() -> None:
    below_fold: dict[str, object] = {
        "t": 0.5,
        "bbox": {"x": 10, "y": 900, "w": 100, "h": 40},
        "action": "click",
    }
    assert _focus_for([below_fold]) is None


def test_zoom_keeps_only_the_interacted_keyframes() -> None:
    good: dict[str, object] = {
        "t": 1.0,
        "bbox": {"x": 10, "y": 20, "w": 100, "h": 40},
        "action": "type",
    }
    junk: dict[str, object] = {
        "t": 0.2,
        "bbox": {"x": 0, "y": 0, "w": 0, "h": 0},
        "action": "click",
    }
    assert _focus_for([junk, good]) == [{"t": 1.0, "bbox": (10.0, 20.0, 100.0, 40.0)}]


def test_zoom_keyframes_outside_the_played_window_are_dropped_and_rebased() -> None:
    # The recorder's `t` is from the head of the RECORDING; the compiler reads it
    # shot-locally. A shot playing source 4.0s-7.5s must not punch in on a click at
    # 1.0s (another screen) and must place the 5.0s click at shot-local 1.0s.
    early: dict[str, object] = {
        "t": 1.0,
        "bbox": {"x": 10, "y": 20, "w": 100, "h": 40},
        "action": "click",
    }
    inside: dict[str, object] = {
        "t": 5.0,
        "bbox": {"x": 300, "y": 200, "w": 80, "h": 30},
        "action": "click",
    }
    focus = _focus_for([early, inside], clip_lead_seconds=4.0, clip_out_seconds=7.5)
    assert focus == [{"t": 1.0, "bbox": (300.0, 200.0, 80.0, 30.0)}]


def _directed_for(*, cold_open_plan: tuple[str, float] | None = None, **shot_extra: object) -> dict:
    """One footage shot carrying the spike's clip-2 cursor + callout geometry."""
    shots = assemble_shots(
        storyboard_shots=[{**_shot("l1"), **shot_extra}],
        images=[],
        narration_clips=[],
        video_by_line={"l1": "http://v/clip.mp4"},
        cursor_track_by_line={
            "l1": [
                {"t": 3.075, "x": 212.0, "y": 864.0, "click": True},
                {"t": 7.501, "x": 1352.0, "y": 154.0, "click": True},
            ]
        },
        callouts_by_line={
            "l1": {"type": "badge", "label": "1", "at": 3.075, "bbox": [74, 848, 278, 31]}
        },
        footage_viewport_by_line={"l1": [2560, 1440]},
        cold_open_plan=cold_open_plan,
    )
    return shots[0]


def test_cursor_and_callout_follow_the_played_window() -> None:
    # Same clock as the focus track: the recorder's `t` is from the head of the
    # RECORDING. A shot playing source 6.0s-9.0s must not draw the 3.075s click
    # ring or the "1" badge (that click is on another screen, the spike's "Sales"
    # nav), and must draw the 7.501s click at shot-local 1.501s.
    shot = _directed_for(clip_lead_seconds=6.0, clip_out_seconds=9.0)
    assert shot["cursor_track"] == [
        {"t": pytest.approx(1.501), "x": 1352.0, "y": 154.0, "click": True}
    ]
    assert "callouts" not in shot


def test_cursor_and_callout_inside_the_window_are_rebased() -> None:
    shot = _directed_for(clip_lead_seconds=2.0, clip_out_seconds=5.0)
    assert shot["cursor_track"] == [
        {"t": pytest.approx(1.075), "x": 212.0, "y": 864.0, "click": True}
    ]
    assert [c["at"] for c in shot["callouts"]] == [pytest.approx(1.075)]


def test_cold_open_lead_rebases_the_tracks_it_moves() -> None:
    # The cold open sets the opener's lead AFTER the tracks were windowed at 0.
    shot = _directed_for(cold_open_plan=("start", 5.0))
    assert shot["clip_lead_seconds"] == 5.0
    assert shot["cursor_track"] == [
        {"t": pytest.approx(2.501), "x": 1352.0, "y": 154.0, "click": True}
    ]
    assert "callouts" not in shot


def test_cold_open_after_an_author_lead_rebases_once_onto_the_effective_start() -> None:
    # Author lead 4s (no anchor), then the cold open sets its own 6s lead. Raw
    # clicks at 9.0s and 12.0s must land at 3.0s and 6.0s, i.e. rebased ONCE by 6,
    # not by 4 and then by 6 (which gave [2.0]).
    shots = assemble_shots(
        storyboard_shots=[{**_shot("l1"), "clip_lead_seconds": 4.0}],
        images=[],
        narration_clips=[],
        video_by_line={"l1": "http://v/clip.mp4"},
        cursor_track_by_line={
            "l1": [
                {"t": 9.0, "x": 10.0, "y": 10.0, "click": True},
                {"t": 12.0, "x": 20.0, "y": 20.0, "click": True},
            ]
        },
        cold_open_plan=("start", 6.0),
    )
    assert shots[0]["clip_lead_seconds"] == 6.0
    assert [k["t"] for k in shots[0]["cursor_track"]] == [pytest.approx(3.0), pytest.approx(6.0)]


def test_unwindowed_shot_keeps_raw_track_times() -> None:
    shot = _directed_for()
    assert [k["t"] for k in shot["cursor_track"]] == [3.075, 7.501]
    assert [c["at"] for c in shot["callouts"]] == [3.075]


def test_footage_tracks_by_line_gated_off_without_direction_footage() -> None:
    screens: list[dict[str, object]] = [
        {
            "id": "home",
            "focus_track": _FOCUS,
            "cursor_track": _CURSOR,
            "viewport": {"w": 1920, "h": 1080},
        }
    ]
    focus, cursor, viewport = footage_tracks_by_line(screens, {"l1": "home"}, footage=None)
    assert focus == {}
    assert cursor == {}
    assert viewport == {}


def test_footage_tracks_by_line_binds_screen_geometry_to_its_line() -> None:
    screens: list[dict[str, object]] = [
        {
            "id": "home",
            "focus_track": _FOCUS,
            "cursor_track": _CURSOR,
            "viewport": {"w": 1920, "h": 1080},
        },
        {"id": "gov"},  # no sidecar geometry → no entry
    ]
    screen_by_line = {"l1": "home", "l2": "gov"}
    focus, cursor, viewport = footage_tracks_by_line(
        screens, screen_by_line, footage={"auto_zoom": True}
    )
    assert focus == {"l1": _FOCUS}
    assert cursor == {"l1": _CURSOR}
    # WS-C: the recorded source viewport rides per line so the compiler's
    # viewport-aware transform lands the geometry on a non-default tier.
    assert viewport == {"l1": [1920, 1080]}


# --- WS-C: footage_viewport threading + normalize_footage_viewport --------------


def test_directed_footage_viewport_copied_onto_footage_shots_when_present() -> None:
    shots = assemble_shots(
        storyboard_shots=[_shot("l1"), _shot("l2")],
        images=[],
        narration_clips=[],
        video_by_line={"l1": "http://v/clip.mp4"},
        image_by_line={"l2": "http://i/card.png"},
        focus_track_by_line={"l1": _FOCUS},
        footage_viewport_by_line={"l1": [1920, 1080]},
    )
    by = {s["line_id"]: s for s in shots}
    assert by["l1"]["footage_viewport"] == [1920, 1080]
    assert "footage_viewport" not in by["l2"]


def test_directed_footage_viewport_omitted_on_non_footage_shots() -> None:
    shots = assemble_shots(
        storyboard_shots=[_shot("l2")],
        images=[],
        narration_clips=[],
        image_by_line={"l2": "http://i/card.png"},
        footage_viewport_by_line={"l2": [1920, 1080]},
    )
    assert "footage_viewport" not in shots[0]


def test_directed_footage_viewport_absent_byte_identical() -> None:
    storyboard_shots = [_shot("l1")]
    with_none = assemble_shots(
        storyboard_shots=storyboard_shots,
        images=[],
        narration_clips=[],
        video_by_line={"l1": "http://v/clip.mp4"},
    )
    with_empty = assemble_shots(
        storyboard_shots=storyboard_shots,
        images=[],
        narration_clips=[],
        video_by_line={"l1": "http://v/clip.mp4"},
        footage_viewport_by_line={},
    )
    assert with_none == with_empty
    assert "footage_viewport" not in with_none[0]


# --- ORCH-3.3 directed shot-plan callouts threading ----------------------------

_CALLOUT: dict[str, object] = {
    "type": "badge",
    "label": "Click here",
    "at": 0.0,
    "bbox": {"x": 10, "y": 20, "w": 100, "h": 40},
}


def test_callout_copied_onto_footage_shot_when_present() -> None:
    shots = assemble_shots(
        storyboard_shots=[_shot("l1"), _shot("l2")],
        images=[],
        narration_clips=[],
        video_by_line={"l1": "http://v/clip.mp4"},
        image_by_line={"l2": "http://i/card.png"},
        callouts_by_line={"l1": _CALLOUT},
    )
    by = {s["line_id"]: s for s in shots}
    # Shot.callouts is declared as an ARRAY, and its bbox as a 4-tuple.
    assert by["l1"]["callouts"] == [{**_CALLOUT, "bbox": (10.0, 20.0, 100.0, 40.0)}]
    assert "callouts" not in by["l2"]


def test_callout_omitted_on_non_footage_shot() -> None:
    # A callout keyed to an image/text-card line (no video_url) never rides the shot.
    shots = assemble_shots(
        storyboard_shots=[_shot("l2")],
        images=[],
        narration_clips=[],
        image_by_line={"l2": "http://i/card.png"},
        callouts_by_line={"l2": _CALLOUT},
    )
    assert "callouts" not in shots[0]


def test_callouts_absent_byte_identical() -> None:
    storyboard_shots = [_shot("l1")]
    with_none = assemble_shots(
        storyboard_shots=storyboard_shots,
        images=[],
        narration_clips=[],
        video_by_line={"l1": "http://v/clip.mp4"},
    )
    with_empty = assemble_shots(
        storyboard_shots=storyboard_shots,
        images=[],
        narration_clips=[],
        video_by_line={"l1": "http://v/clip.mp4"},
        callouts_by_line={},
    )
    assert with_none == with_empty
    assert "callouts" not in with_none[0]


# --- T8 split-screen composite threading ---------------------------------------

_SPLIT: dict[str, object] = {"layout": "h", "second_source_url": "http://v/2.mp4"}


def test_split_copied_onto_footage_shot_when_present() -> None:
    shots = assemble_shots(
        storyboard_shots=[_shot("l1"), _shot("l2")],
        images=[],
        narration_clips=[],
        video_by_line={"l1": "http://v/clip.mp4"},
        image_by_line={"l2": "http://i/card.png"},
        split_by_line={"l1": _SPLIT},
    )
    by = {s["line_id"]: s for s in shots}
    assert by["l1"]["split"] == _SPLIT
    assert "split" not in by["l2"]


def test_split_omitted_on_non_footage_shot() -> None:
    # A split keyed to an image/text-card line (no video_url) never rides the shot.
    shots = assemble_shots(
        storyboard_shots=[_shot("l2")],
        images=[],
        narration_clips=[],
        image_by_line={"l2": "http://i/card.png"},
        split_by_line={"l2": _SPLIT},
    )
    assert "split" not in shots[0]


def test_split_absent_byte_identical() -> None:
    storyboard_shots = [_shot("l1")]
    with_none = assemble_shots(
        storyboard_shots=storyboard_shots,
        images=[],
        narration_clips=[],
        video_by_line={"l1": "http://v/clip.mp4"},
    )
    with_empty = assemble_shots(
        storyboard_shots=storyboard_shots,
        images=[],
        narration_clips=[],
        video_by_line={"l1": "http://v/clip.mp4"},
        split_by_line={},
    )
    assert with_none == with_empty
    assert "split" not in with_none[0]


def test_normalize_footage_viewport_accepts_shapes_and_rejects_junk() -> None:
    assert normalize_footage_viewport({"width": 1920, "height": 1080}) == [1920, 1080]
    assert normalize_footage_viewport({"w": 1280, "h": 720}) == [1280, 720]
    assert normalize_footage_viewport([1920, 1080]) == [1920, 1080]
    assert normalize_footage_viewport((1920.0, 1080.0)) == [1920, 1080]
    # junk / malformed ⇒ None (never raises, never rides a shot)
    assert normalize_footage_viewport(None) is None
    assert normalize_footage_viewport({}) is None
    assert normalize_footage_viewport({"w": 0, "h": 1080}) is None
    assert normalize_footage_viewport({"w": True, "h": 1080}) is None
    assert normalize_footage_viewport([1920]) is None
    assert normalize_footage_viewport("1920x1080") is None


def test_a_fabricated_hold_is_marked_and_drained_into_a_note() -> None:
    """A reconciled 0.0 keeps its 1s floor AND records that the duration was
    substituted — the emitted `seconds` is otherwise a claim nothing measured."""
    shots = assemble_shots(
        storyboard_shots=[_shot("l1", 0.0), _shot("l2", 2.0)],
        images=[],
        narration_clips=[],
    )
    assert shots[0]["seconds"] == 1.0
    assert shots[0][FABRICATED_HOLD_KEY] is True
    assert FABRICATED_HOLD_KEY not in shots[1]

    notes = drain_fabricated_hold_notes(shots)
    assert len(notes) == 1
    assert "FABRICATED" in notes[0]
    assert "l1" in notes[0] and "l2" not in notes[0]
    # Drained: the private marker can never ride out to the compiler.
    assert all(FABRICATED_HOLD_KEY not in s for s in shots)
    assert drain_fabricated_hold_notes(shots) == []


def test_a_real_one_second_line_is_never_marked_fabricated() -> None:
    shots = assemble_shots(
        storyboard_shots=[_shot("l1", 1.0)],
        images=[],
        narration_clips=[],
    )
    assert shots[0]["seconds"] == 1.0
    assert FABRICATED_HOLD_KEY not in shots[0]
    assert drain_fabricated_hold_notes(shots) == []


def test_single_image_asset_over_six_beats_repeats_but_notes_the_shortfall() -> None:
    # One image, SIX beats (a realistic length, not the n=2/n=3 cases where a
    # single skip is invisible): there is no alternative asset, so refusing
    # the repeat would produce a wall of five text cards after the first
    # beat — worse for the viewer than a held/repeated visual. All six beats
    # get the one asset; the "not enough assets" fact is surfaced via the
    # shortfall soft note instead of by starving beats of a visual.
    shots = assemble_shots(
        storyboard_shots=[_shot(f"l{n}") for n in range(1, 7)],
        images=[{"image_url": "only.png"}],
        narration_clips=[],
    )
    assert [s["image_url"] for s in shots] == ["only.png"] * 6
    notes = drain_fabricated_hold_notes(shots)
    assert len(notes) == 1
    assert "1 round-robin image asset(s) for 6 beat(s)" in notes[0]


def test_single_video_asset_over_six_beats_repeats_but_notes_the_shortfall() -> None:
    shots = assemble_shots(
        storyboard_shots=[_shot(f"l{n}") for n in range(1, 7)],
        images=[],
        narration_clips=[],
        video_clips=[{"video_url": "only.mp4"}],
    )
    assert [s["video_url"] for s in shots] == ["only.mp4"] * 6
    notes = drain_fabricated_hold_notes(shots)
    assert len(notes) == 1
    assert "1 round-robin video asset(s) for 6 beat(s)" in notes[0]


def test_two_image_assets_over_six_beats_never_repeat_adjacently() -> None:
    # With a genuine alternative (>=2 assets), the anti-repeat guard DOES
    # engage over a realistic beat count: no two adjacent beats share a
    # visual, and no shortfall note is raised (the pool covers every beat
    # via round-robin already).
    shots = assemble_shots(
        storyboard_shots=[_shot(f"l{n}") for n in range(1, 7)],
        images=[{"image_url": "img0"}, {"image_url": "img1"}],
        narration_clips=[],
    )
    urls = [s["image_url"] for s in shots]
    assert all(urls[k] != urls[k + 1] for k in range(len(urls) - 1))
    assert urls == ["img0", "img1"] * 3


def test_plentiful_pool_is_byte_identical_no_adjacent_dedup_kicks_in() -> None:
    # With enough assets to avoid a natural repeat, output must be exactly
    # what it was before this fix (constraint from the brief).
    shots = assemble_shots(
        storyboard_shots=[_shot("l1"), _shot("l2"), _shot("l3")],
        images=[{"image_url": "img0"}, {"image_url": "img1"}],
        narration_clips=[],
    )
    assert [s["image_url"] for s in shots] == ["img0", "img1", "img0"]


def test_visual_reuse_note_when_pool_smaller_than_beats_needing_one() -> None:
    shots = assemble_shots(
        storyboard_shots=[_shot("l1"), _shot("l2"), _shot("l3")],
        images=[{"image_url": "only.png"}],
        narration_clips=[],
    )
    notes = drain_fabricated_hold_notes(shots)
    assert len(notes) == 1
    assert "1 round-robin image asset(s) for 3 beat(s)" in notes[0]
    assert "degrading" in notes[0]
    # Drained: the private marker can never ride out to the compiler.
    assert not any(k.startswith("_visual_reuse") for s in shots for k in s)


def test_no_visual_reuse_note_when_pool_covers_every_beat() -> None:
    shots = assemble_shots(
        storyboard_shots=[_shot("l1"), _shot("l2")],
        images=[{"image_url": "img0"}, {"image_url": "img1"}],
        narration_clips=[],
    )
    assert drain_fabricated_hold_notes(shots) == []


def test_short_narrated_beat_is_raised_to_the_minimum_floor() -> None:
    # A 0.3s narration clip is a flash cut, not a legitimate measurement.
    shots = assemble_shots(
        storyboard_shots=[_shot("l1", 2.0)],
        images=[],
        narration_clips=[{"line_id": "l1", "artifact_url": "http://a/clip.wav", "seconds": 0.3}],
    )
    assert shots[0]["seconds"] == 1.2
    notes = drain_fabricated_hold_notes(shots)
    assert len(notes) == 1
    assert "flash-cut" in notes[0]
    assert "l1" in notes[0]
    assert "degrading" in notes[0]


def test_narrated_beat_at_or_above_floor_is_unaffected() -> None:
    shots = assemble_shots(
        storyboard_shots=[_shot("l1", 2.0)],
        images=[],
        narration_clips=[{"line_id": "l1", "artifact_url": "http://a/clip.wav", "seconds": 1.2}],
    )
    assert shots[0]["seconds"] == 1.2
    assert drain_fabricated_hold_notes(shots) == []


def test_zero_second_beat_still_gets_fabricated_hold_not_the_short_floor() -> None:
    # An exactly-zero reconciled duration is a DIFFERENT, pre-existing failure
    # mode (no measurement at all) and keeps its own 1.0s hold + marker,
    # distinct from the 1.2s short-beat floor for a positive-but-short one.
    shots = assemble_shots(
        storyboard_shots=[_shot("l1", 0.0)],
        images=[],
        narration_clips=[],
    )
    assert shots[0]["seconds"] == 1.0
    assert shots[0][FABRICATED_HOLD_KEY] is True
    notes = drain_fabricated_hold_notes(shots)
    assert len(notes) == 1
    assert "FABRICATED" in notes[0]


def test_underspoken_hold_extends_the_shot_duration() -> None:
    # A narrated beat whose clip runs 2.0s but whose allotment is 3.2s
    # (underspoken by 1.2s, per AuthoredLine.hold_seconds / the sibling
    # lane's time_budget.underspoken_hold): the picture must hold the extra
    # 1.2s on its last frame rather than shipping the beat short.
    shots = assemble_shots(
        storyboard_shots=[_shot("l1", 2.0)],
        images=[],
        narration_clips=[{"line_id": "l1", "artifact_url": "http://a/clip.wav", "seconds": 2.0}],
        hold_seconds_by_line={"l1": 1.2},
    )
    assert shots[0]["seconds"] == 3.2
    notes = drain_fabricated_hold_notes(shots)
    assert len(notes) == 1
    assert "UNDERSPOKEN" in notes[0]
    assert "1.20s total" in notes[0]
    assert "l1" in notes[0]
    assert "degrading" in notes[0]
    # Drained: the private marker can never ride out to the compiler.
    assert UNDERSPOKEN_HOLD_KEY not in shots[0]


def test_no_hold_is_byte_identical_to_today() -> None:
    with_none = assemble_shots(
        storyboard_shots=[_shot("l1", 2.0)],
        images=[],
        narration_clips=[{"line_id": "l1", "artifact_url": "http://a/clip.wav", "seconds": 2.0}],
    )
    with_empty_map = assemble_shots(
        storyboard_shots=[_shot("l1", 2.0)],
        images=[],
        narration_clips=[{"line_id": "l1", "artifact_url": "http://a/clip.wav", "seconds": 2.0}],
        hold_seconds_by_line={},
    )
    with_zero_hold = assemble_shots(
        storyboard_shots=[_shot("l1", 2.0)],
        images=[],
        narration_clips=[{"line_id": "l1", "artifact_url": "http://a/clip.wav", "seconds": 2.0}],
        hold_seconds_by_line={"l1": 0.0},
    )
    assert with_none == with_empty_map == with_zero_hold
    assert with_none[0]["seconds"] == 2.0
    assert drain_fabricated_hold_notes(with_none) == []


def test_hold_and_short_beat_floor_both_apply() -> None:
    # A 0.3s narration is BOTH a flash cut (floored to 1.2s) AND underspoken.
    # The hold starts where the SPEECH ends (0.3s), not where the floored shot does:
    # a 0.5s hold (0.3 + 0.5 = 0.8s) is already inside the 1.2s floor, so no frame
    # is held twice; a 1.4s hold (0.3 + 1.4 = 1.7s) reaches past it and extends.
    covered = assemble_shots(
        storyboard_shots=[_shot("l1", 2.0)],
        images=[],
        narration_clips=[{"line_id": "l1", "artifact_url": "http://a/clip.wav", "seconds": 0.3}],
        hold_seconds_by_line={"l1": 0.5},
    )
    assert covered[0]["seconds"] == 1.2
    covered_notes = drain_fabricated_hold_notes(covered)
    assert len(covered_notes) == 1
    assert "flash-cut" in covered_notes[0]

    shots = assemble_shots(
        storyboard_shots=[_shot("l1", 2.0)],
        images=[],
        narration_clips=[{"line_id": "l1", "artifact_url": "http://a/clip.wav", "seconds": 0.3}],
        hold_seconds_by_line={"l1": 1.4},
    )
    assert shots[0]["seconds"] == pytest.approx(1.7)  # speech end 0.3 + hold 1.4
    notes = drain_fabricated_hold_notes(shots)
    assert len(notes) == 2
    assert any("flash-cut" in n for n in notes)
    assert any("UNDERSPOKEN" in n for n in notes)


def test_hold_is_never_folded_into_freeze_pad_or_max_seconds() -> None:
    # Critical separation (the sibling lane's exact complaint): a hold must
    # NOT ride as max_seconds/freeze_pad_seconds, which narrate re-fits
    # against and would re-synthesize the narration slower.
    shots = assemble_shots(
        storyboard_shots=[_shot("l1", 2.0)],
        images=[],
        narration_clips=[{"line_id": "l1", "artifact_url": "http://a/clip.wav", "seconds": 2.0}],
        hold_seconds_by_line={"l1": 1.2},
        min_max_by_line={"l1": (2.0, 5.0)},
    )
    assert shots[0]["seconds"] == 3.2
    assert shots[0]["max_seconds"] == 5.0  # untouched by the hold
    assert "freeze_pad_seconds" not in shots[0]


def test_negative_or_nonfinite_hold_is_a_no_op() -> None:
    for bad_hold in (-1.0, float("nan"), float("inf")):
        shots = assemble_shots(
            storyboard_shots=[_shot("l1", 2.0)],
            images=[],
            narration_clips=[
                {"line_id": "l1", "artifact_url": "http://a/clip.wav", "seconds": 2.0}
            ],
            hold_seconds_by_line={"l1": bad_hold},
        )
        assert shots[0]["seconds"] == 2.0
        assert drain_fabricated_hold_notes(shots) == []


def test_unkeyed_beats_draw_distinct_unused_assets_when_keyed_beats_interleave() -> None:
    # 5 beats, beats 1-3 keyed, pool of 4. The pre-fix global-index
    # round-robin (`pool[i % len(pool)]`) gave beats 0 and 4 BOTH pool[0]
    # while pool[1..3] never played; a per-pool draw counter gives them two
    # distinct assets.
    pool = ["img0", "img1", "img2", "img3"]
    shots = assemble_shots(
        storyboard_shots=[_shot(f"l{n}") for n in range(5)],
        images=[{"image_url": u} for u in pool],
        narration_clips=[],
        image_by_line={"l1": "k1.png", "l2": "k2.png", "l3": "k3.png"},
    )
    assert [s["image_url"] for s in shots] == ["img0", "k1.png", "k2.png", "k3.png", "img1"]


def test_unkeyed_beats_skip_pool_assets_a_keyed_beat_already_shows() -> None:
    # Keyed beats show img0 and img1 from the same pool: the un-keyed beats
    # must draw the still-unused img2/img3 instead of repeating them.
    shots = assemble_shots(
        storyboard_shots=[_shot(f"l{n}") for n in range(4)],
        images=[{"image_url": u} for u in ["img0", "img1", "img2", "img3"]],
        narration_clips=[],
        image_by_line={"l1": "img0", "l3": "img1"},
    )
    assert [s["image_url"] for s in shots] == ["img2", "img0", "img3", "img1"]


def test_video_pool_uses_a_per_pool_counter_too() -> None:
    shots = assemble_shots(
        storyboard_shots=[_shot(f"l{n}") for n in range(5)],
        images=[],
        narration_clips=[],
        video_clips=[{"video_url": f"v{n}.mp4"} for n in range(4)],
        video_by_line={"l1": "k1.mp4", "l2": "k2.mp4", "l3": "k3.mp4"},
    )
    assert [s["video_url"] for s in shots] == ["v0.mp4", "k1.mp4", "k2.mp4", "k3.mp4", "v1.mp4"]
