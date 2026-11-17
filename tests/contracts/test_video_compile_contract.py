"""Cross-service contract guard for video-compile payloads.

The root cause of the course/lesson E2E whack-a-mole was orchestrator stages
built against *imagined* downstream contracts. This guard makes that class of
bug fail in CI: every stage that calls ``VideoCompileClient.compile`` is run
with representative inputs through a client that checks the emitted payload
against the compositor's request schema.

**The schema is a GENERATED artifact, not a hand mirror.**
``tests/contracts/compile_request.schema.json`` is produced by
``video_compiler_service/scripts/regen_contract_artifact.py`` from
``CompileRequest.model_json_schema()`` (``sort_keys=True``). Refresh it by
COPYING that file over — never by hand-editing it. The previous version of this
module carried hand-typed pydantic mirrors of the compiler's models; they drifted
badly (``_MusicBed`` had 5 of 11 real fields, ``_Shot`` 11 of 25), and a mirror
that has drifted still passes every test, so the guard read green while guarding
nothing. That mirror is gone.

**Staleness is GUARDED, not merely hoped for.** The vendored copy is still a
snapshot, but it can no longer rot unnoticed: the ``contract-freshness`` job in
``.github/workflows/ci-cd.yaml`` runs on every PR, pulls the compiler's promoted
image from ``registry.test:5000``, regenerates the schema inside it with the
same ``json.dumps(..., indent=2, sort_keys=True)`` rendering the canonical
``regen_contract_artifact.py`` uses, and diffs it against this vendored file. Any
difference fails the build with the refresh command in the log.

No GitHub token is required for that, and none should be added. ``video_compiler_service``
is a PRIVATE repo, so reading its contract over the GitHub API would need a
cross-repo PAT — but the two services already share a Docker registry, and the
compiler's own image carries ``src/api/schemas.py``. Reading the contract out of
the image is the same source of truth with no credential to mint, rotate, or leak.

**Residual window, stated honestly:** the guard compares against the compiler's
``:latest`` image, i.e. what is PROMOTED, not what is merged. A compiler schema
change is therefore invisible here for the gap between its merge and its promote.
That window is minutes-to-hours, it closes by itself, and it fails CLOSED in the
direction that matters (once promoted, a stale copy blocks the next PR). Refreshing
this file remains part of landing a compiler schema change.

What is checked here:

* Every emitted payload's dotted leaf set must be a subset of the schema's —
  i.e. the orchestrator never invents a field. This is the same guarantee
  ``extra="forbid"`` gave, with zero hand-maintenance. It matters because the
  compiler's own request model is ``extra="ignore"``: live, an invented field is
  silently DROPPED, not 422'd, so the knob just never applies (that silence was
  the A7-grade-never-applies bug).
* ``shots[].seconds`` must satisfy the schema's own ``exclusiveMinimum`` — the
  zero-duration shots that 422'd live.
* The response fields stages read back are covered by the response shape.

The OTHER drift direction — the compiler grew a field NO producer sets — is
checked by the sibling ``test_producer_coverage.py``, which consumes
``collect_emitted_leaves()`` from this module.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import cast
from uuid import uuid4

import pytest
from pydantic import BaseModel

from src.clients.video_compile import VideoCompileClient, _fold_captions
from src.services.runs.run_context import RunContext
from src.services.stages import shot_sound
from src.services.stages.action_reel.compile import ActionReelCompileStage
from src.services.stages.app_review.compile import AppReviewCompileStage
from src.services.stages.course.stitch import CourseStitchStage
from src.services.stages.generic.video_render import VideoRenderStage
from src.services.stages.lesson.compile import LessonCompileStage
from src.services.stages.promo.render import PromoRenderStage
from src.services.stages.shared import sound_design
from src.services.stages.sound_catalog import SoundLibrary
from tests.contracts.schema_walk import (
    as_mapping,
    load_schema,
    payload_leaves,
    schema_leaves,
    structure_violations,
    unknown_leaves,
)

_SCHEMA: Mapping[str, object] = load_schema()
_SCHEMA_LEAVES = schema_leaves(_SCHEMA)


class _CompileResponse(BaseModel):
    """Response shape (the artifact only covers the REQUEST model).

    Small enough to stay a hand mirror without the drift risk that killed the
    request mirror: four fields, none of which have changed since VC-1.
    """

    artifact_url: str
    duration_seconds: float
    object_key: str
    bucket: str


# Keys every compile-calling stage reads off the response. Locked against the
# response schema so a stage can't read a field the compositor doesn't return.
_RESPONSE_READS = {"artifact_url", "duration_seconds"}


def _shot_seconds_minimum() -> float:
    """The schema's own gt-0 bound for ``Shot.seconds`` (read, never assumed)."""
    # ``as_mapping`` (not a bare isinstance) because ``Mapping`` is invariant in
    # its key type: an isinstance-narrowed value is ``Mapping[Unknown, object]``
    # and its ``.get``/``[]`` do not accept a str literal.
    defs = as_mapping(_SCHEMA["$defs"])
    assert defs is not None
    shot = as_mapping(defs["Shot"])
    assert shot is not None
    props = as_mapping(shot["properties"])
    assert props is not None
    seconds = as_mapping(props["seconds"])
    assert seconds is not None
    bound = seconds["exclusiveMinimum"]
    assert isinstance(bound, int | float)
    return float(bound)


_SECONDS_EXCLUSIVE_MIN = _shot_seconds_minimum()


def assert_contract_valid(payload: Mapping[str, object]) -> frozenset[str]:
    """Assert a compile payload only uses fields the compiler declares.

    Returns the payload's leaf set so callers can accumulate producer coverage.
    """
    walk = payload_leaves(payload)
    unknown = unknown_leaves(walk.leaves, _SCHEMA_LEAVES, scalar_leaves=walk.scalar_leaves)
    # Reported separately from invented fields: a scalar at an object position is
    # a TYPE error, and saying "field the compiler does not declare" about it
    # sends the reader looking for a typo that isn't there.
    structural = structure_violations(walk, _SCHEMA_LEAVES)
    assert not structural, (
        "compile payload puts a SCALAR where the compiler declares STRUCTURE — "
        "the schema declares named sub-fields under each of these paths, so a "
        f"bare str/int/float/bool cannot validate there: {sorted(structural)}"
    )
    invented = unknown - structural
    assert not invented, (
        "compile payload carries fields the compiler does not declare "
        f"(the compiler would SILENTLY DROP them): {sorted(invented)}"
    )
    shots = payload.get("shots")
    assert isinstance(shots, list) and shots, "compile payload must carry >=1 shot"
    for raw_shot in shots:
        shot = as_mapping(raw_shot)
        assert shot is not None
        seconds = shot.get("seconds")
        assert isinstance(seconds, int | float), f"shot {shot.get('id')!r} has no numeric seconds"
        assert seconds > _SECONDS_EXCLUSIVE_MIN, (
            f"shot {shot.get('id')!r} has seconds={seconds}, "
            f"schema requires > {_SECONDS_EXCLUSIVE_MIN} (a zero-duration shot 422s live)"
        )
    return walk.leaves


class _ValidatingCompileClient:
    """Stand-in for VideoCompileClient that enforces the request contract."""

    def __init__(self) -> None:
        self.last: dict[str, object] = {}
        # Every leaf this client has ever seen emitted, across all calls. The
        # union of these across all stage scenarios is the producer coverage set
        # the sibling inert-knob test consumes.
        self.seen: set[str] = set()

    async def compile(
        self,
        *,
        shots: list[dict[str, object]],
        tags: dict[str, str] | None = None,
        music: dict[str, object] | None = None,
        ambient: list[dict[str, object]] | None = None,
        mastering: dict[str, object] | None = None,
        finish: dict[str, object] | None = None,
        captions: list[dict[str, object]] | None = None,
        captions_render: str | None = None,
        caption_mood: str | None = None,
        captions_karaoke: bool = False,
        caption_look: Mapping[str, object] | None = None,
        cadence: int | None = None,
        emit_shot_timeline: bool = False,
        timeout: float = 1200.0,
        # A transport dedup HEADER (Idempotency-Key), NOT a payload field — accept
        # and ignore it so it never leaks into the contract-validated body.
        idempotency_key: str | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "shots": shots,
            "tags": tags,
            "music": music,
            "ambient": ambient or [],
            "mastering": mastering,
            "finish": finish,
            "cadence": cadence,
        }
        # Mirror the real client's gating: captions_render only rides WITH cues, so
        # a none-mode compile (captions_render="none", captions=None) never puts the
        # illegal "none" on the wire. Folded in only when captions are present.
        # The REAL client's fold, so a caption field the client sends is the
        # caption field this contract sees (no hand-kept copy to drift).
        _fold_captions(
            payload, captions, captions_render, caption_mood, captions_karaoke, caption_look
        )
        # Mirror the real client: the opt-in authoritative-timeline flag only
        # rides the wire when requested, so a non-gated compile stays byte-identical.
        if emit_shot_timeline:
            payload["emit_shot_timeline"] = True
        self.seen |= assert_contract_valid(payload)
        self.last = payload
        return {
            "artifact_url": "http://stub/out.mp4",
            "duration_seconds": 9.0,
            "object_key": "k",
            "bucket": "b",
        }


def _client() -> tuple[_ValidatingCompileClient, VideoCompileClient]:
    """Return the validating fake plus its VideoCompileClient-typed view."""
    fake = _ValidatingCompileClient()
    return fake, cast(VideoCompileClient, fake)


def _ctx(
    prior: Mapping[str, Mapping[str, object]],
    config: Mapping[str, object] | None = None,
) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=dict(config or {}),
        prior_run_outputs={key: dict(value) for key, value in prior.items()},
    )


def _mapping(value: object) -> dict[str, object]:
    """Narrow an emitted payload position to a mapping, typed for assertions.

    Hard-fails (assert) rather than returning ``None`` — in a test, a payload
    position that is not a mapping is the bug, not a branch to handle.
    """
    out = as_mapping(value)
    assert out is not None
    return out


def _number(value: object) -> float:
    """Narrow an emitted payload position to a float, typed for arithmetic."""
    assert isinstance(value, int | float)
    return float(value)


def _shots(fake: _ValidatingCompileClient) -> list[dict[str, object]]:
    """The shot dicts off the last emitted payload, typed for assertions."""
    shots = fake.last["shots"]
    assert isinstance(shots, list)
    return [_mapping(shot) for shot in shots]


# --------------------------------------------------------------------------- #
# stage scenarios — each drives one real compile-calling stage with
# representative inputs. Defined as plain coroutines (NOT pytest fixtures) so
# ``collect_emitted_leaves`` can replay them outside a test.
# --------------------------------------------------------------------------- #
async def _emit_promo(client: VideoCompileClient) -> None:
    ctx = _ctx(
        {
            "promo_script": {"lines": [{"id": "l1", "text": "Hook.", "seconds": 3.0}]},
            "promo_storyboard": {
                "shots": [
                    {
                        "id": "s1",
                        "line_id": "l1",
                        "description": "d",
                        "seconds": 3.0,
                        "visual_style": "x",
                        # visual plan routes this line to the AI image.
                        "visual_type": "ai_image",
                    }
                ]
            },
            "promo_narrate": {
                "clips": [
                    {"id": "c1", "line_id": "l1", "artifact_url": "http://a/1.wav", "seconds": 4.0}
                ]
            },
            "ai_image_gen": {"images": [{"id": "i1", "image_url": "http://a/i1.png"}]},
        },
        {"deliverable": {"format": "16x9", "captions_mode": "burned_in"}},
    )
    await PromoRenderStage(client=client).run({}, ctx=ctx)


async def _emit_generic_video_render(client: VideoCompileClient) -> None:
    ctx = _ctx(
        {
            "storyboard_draft": {
                "shots": [{"id": "s1", "line_id": "l1", "description": "d", "seconds": 2.0}]
            },
            "narrate": {
                "clips": [
                    {"id": "c1", "line_id": "l1", "artifact_url": "http://a/1.wav", "seconds": 2.5}
                ]
            },
            "ai_image_gen": {"images": []},
        }
    )
    await VideoRenderStage(client=client).run({}, ctx=ctx)


async def _emit_course_stitch(client: VideoCompileClient) -> None:
    ctx = _ctx(
        {
            "course_compile_lessons": {
                "lessons": [
                    {
                        "lesson_project_id": "p1",
                        "lesson_run_id": "r1",
                        "artifact_url": "http://a/l1.mp4",
                        "duration_seconds": 60.0,
                    }
                ]
            }
        }
    )
    await CourseStitchStage(client=client).run({}, ctx=ctx)


async def _emit_lesson_compile(client: VideoCompileClient) -> None:
    ctx = _ctx(
        {
            "lesson_slides": {"slides": [{"id": "sl1", "paragraph_ids": ["p1"]}]},
            "lesson_narrate": {
                "clips": [
                    {
                        "id": "c1",
                        "paragraph_id": "p1",
                        "artifact_url": "http://a/1.wav",
                        "seconds": 5.0,
                    }
                ]
            },
            "lesson_record": {"recordings": [{"slide_id": "sl1", "image_url": "http://a/sl1.png"}]},
        },
        {"deliverable": {"format": "16x9", "captions_mode": "none"}},
    )
    await LessonCompileStage(client=client).run({}, ctx=ctx)


async def _emit_app_review_compile(client: VideoCompileClient) -> None:
    ctx = _ctx(
        {
            "app_review_record": {
                "narrations": [
                    {"id": "n1", "line_id": "l1", "artifact_url": "http://a/n1.wav", "seconds": 4.0}
                ],
                "visuals": [
                    {
                        "id": "v1",
                        "line_id": "l1",
                        "artifact_url": "http://shot/s.png",
                        "seconds": 4.0,
                        "kind": "screenshot",
                    }
                ],
            }
        },
        {"deliverable": {"format": "16x9", "captions_mode": "none"}},
    )
    await AppReviewCompileStage(client=client).run({}, ctx=ctx)


async def _emit_finish_block(client: VideoCompileClient) -> None:
    ctx = _ctx(
        {
            "storyboard_draft": {
                "shots": [{"id": "s1", "line_id": "l1", "description": "d", "seconds": 2.0}]
            },
            "narrate": {
                "clips": [
                    {"id": "c1", "line_id": "l1", "artifact_url": "http://a/1.wav", "seconds": 2.5}
                ]
            },
            "ai_image_gen": {"images": []},
        },
        {"finish": {"enabled": True}, "brand": {"primary_color": "#280983"}},
    )
    await VideoRenderStage(client=client).run({}, ctx=ctx)


async def _emit_footage_window(client: VideoCompileClient) -> None:
    """A FOOTAGE line with an explicit source window — the only scenario that
    exercises ``shots.clip_anchor`` / ``clip_lead_seconds`` / ``clip_out_seconds``.

    ``_attach_clip_framing`` is footage-guarded (it returns early unless the built
    shot carries a ``video_url``), so none of the three framing knobs can be
    observed from an image/text-card scenario however the line is authored — which
    is why all of them read as "no producer" before this scenario existed. The
    footage here arrives the ordinary generic-render way: an operator-pinned
    real-media VIDEO on ``source_ingest``, which ``collect_real_media_videos`` →
    ``assign_real_media_videos_by_line`` keys onto the line as ``video_url``.

    The window is the compiler's own documented example: lead 4.0 + out 7.5 shows
    exactly 4.0s-7.5s of the source. Deliberately BOTH ends — an out-point without
    a lead would leave ``clip_lead_seconds`` uncovered, and a lead without an
    out-point is the pre-#161 behaviour where the window's end is merely implied by
    ``seconds``.
    """
    ctx = _ctx(
        {
            "storyboard_draft": {
                "shots": [
                    {
                        "id": "s1",
                        "line_id": "l1",
                        "description": "d",
                        "seconds": 3.0,
                        "clip_anchor": "start",
                        "clip_lead_seconds": 4.0,
                        "clip_out_seconds": 7.5,
                    }
                ]
            },
            "narrate": {
                "clips": [
                    {"id": "c1", "line_id": "l1", "artifact_url": "http://a/1.wav", "seconds": 3.0}
                ]
            },
            "ai_image_gen": {"images": []},
            # An operator-pinned clip, already re-published to run storage — the
            # shape ``collect_real_media_videos`` reads (kind="video" + a fetchable
            # ``composite_url``). Anything else lands on the IMAGE list instead and
            # the shot would never become footage.
            "source_ingest": {
                "pinned_media": [{"kind": "video", "composite_url": "http://a/footage.mp4"}]
            },
        }
    )
    await VideoRenderStage(client=client).run({}, ctx=ctx)


async def _emit_mastering(client: VideoCompileClient) -> None:
    """A run with an authored ``mastering`` block — the only scenario that puts the
    ORCH-0.5 deterministic mastering profile on the wire.

    ``build_mastering_profile`` is presence-is-opt-in: WITHOUT a ``mastering`` key
    in the config snapshot it returns ``None`` and ``CompileStageBase.handle``
    sends no ``mastering`` at all, so the whole surface reads as "no producer" for
    the wrong reason (nobody opted in). Every optional field is authored here
    on purpose: ``as_mastering_payload`` always emits ``loudness_target`` (it
    carries the owner-ratified default) but omits every other field it finds unset,
    so a partially-filled block would leave the rest of the surface unobserved.

    Values are ordinary deliverable settings, not magic: ``crf``/``loudness_target``
    sit inside the profile's clamp windows (0..51 and -30..-5) so what is authored
    is what rides the wire.
    """
    ctx = _ctx(
        {
            "storyboard_draft": {
                "shots": [{"id": "s1", "line_id": "l1", "description": "d", "seconds": 2.0}]
            },
            "narrate": {
                "clips": [
                    {"id": "c1", "line_id": "l1", "artifact_url": "http://a/1.wav", "seconds": 2.5}
                ]
            },
            "ai_image_gen": {"images": []},
        },
        {
            "mastering": {
                "colorspace": "bt709",
                "deband": "auto",  # a STRING ("auto"/"on"/"off"), never a bool
                "bitrate_mode": "vbr",
                "maxrate": "12M",
                "crf": 20,
                "loudness_target": -14.0,
                "true_peak": -1.0,
                "platform": "youtube",
            }
        },
    )
    await VideoRenderStage(client=client).run({}, ctx=ctx)


async def _emit_promo_sound_design(client: VideoCompileClient) -> None:
    """Promo render with sound design ON — the ONLY scenario that exercises the
    audio bus.

    Without it ``resolve_audio_bed`` never runs, so the whole ``music.*`` /
    ``ambient.*`` / ``shots.sfx.*`` / ``shots.transition_sfx.*`` surface reads as
    "no producer" for the wrong reason (nobody turned it on), which is precisely
    the kind of false positive that tempts people to write a JUSTIFIED entry for
    a knob that is in fact wired.

    Two shots on purpose: the transition whoosh rides every cut EXCEPT the open,
    so a single-shot promo would never emit ``shots.transition_sfx.*``.
    """
    ctx = _ctx(
        {
            "promo_script": {
                "lines": [
                    {"id": "l1", "text": "Hook.", "seconds": 3.0},
                    {"id": "l2", "text": "Payoff.", "seconds": 3.0},
                ]
            },
            "promo_storyboard": {
                "shots": [
                    {
                        "id": "s1",
                        "line_id": "l1",
                        "description": "d",
                        "seconds": 3.0,
                        "visual_style": "x",
                        "visual_type": "ai_image",
                    },
                    {
                        "id": "s2",
                        "line_id": "l2",
                        "description": "d",
                        "seconds": 3.0,
                        "visual_style": "x",
                        "visual_type": "ai_image",
                    },
                ]
            },
            "promo_narrate": {
                "clips": [
                    {"id": "c1", "line_id": "l1", "artifact_url": "http://a/1.wav", "seconds": 4.0},
                    {"id": "c2", "line_id": "l2", "artifact_url": "http://a/2.wav", "seconds": 4.0},
                ]
            },
            "ai_image_gen": {
                "images": [
                    {"id": "i1", "image_url": "http://a/i1.png"},
                    {"id": "i2", "image_url": "http://a/i2.png"},
                ]
            },
            # Per-line accent cues — the sound director's real output shape.
            "promo_sound_director": {
                "cues": [{"line_id": "l2", "sfx_id": "ding_positive", "at_seconds": 1.0}]
            },
        },
        {
            "deliverable": {"format": "16x9", "captions_mode": "burned_in"},
            # `music` names a real catalog id and `ambient: True` takes the
            # default room tone; `transitions: True` puts the whoosh on the cut.
            "sound_effects": {
                "enabled": True,
                "music": "corporate_uplift",
                "ambient": True,
                "transitions": True,
                "accents": True,
            },
        },
    )
    with _stubbed_sound_io():
        await PromoRenderStage(client=client).run({}, ctx=ctx)


async def _emit_action_reel_overlays(client: VideoCompileClient) -> None:
    """An action_reel cut carrying a rendered graphics card — the ONLY scenario
    that exercises the alpha overlay surface (``shots.overlay_clips.*``).

    Nothing else can: full-frame motion clips ride as an ordinary ``video_url``
    and the burned-in text path is ``shots.overlays``, a different surface. So
    before this scenario every overlay_clips leaf read as "no producer", which
    was TRUE until ``graphics_design`` shipped and is a false positive now.

    Two deliberate details. ``music_track_id`` is empty so ``build_audio_bed``
    returns before touching sound storage — this scenario is about the overlay
    join, and the bed is already covered by ``_emit_promo_sound_design``. And
    the card is authored with a non-default ``x``/``y``/``opacity`` and a real
    ``seconds``, because a card left on every default would leave half the
    surface unobserved for the usual wrong reason (nobody asked for it).
    """
    ctx = _ctx(
        {
            "cut_plan": {
                "shots": [
                    {
                        "id": "sh1",
                        "video_url": "http://a/drone.mp4",
                        "in_seconds": 12.5,
                        "out_seconds": 15.5,
                        "seconds": 3.0,
                        "speed": 1.0,
                    }
                ],
                "music_track_id": "",
            },
            "graphics_design": {
                "cards": [
                    {
                        "id": "card-1",
                        "shot_id": "sh1",
                        "kind": "title",
                        "text": "Ridgeline",
                        "at_seconds": 0.25,
                        "seconds": 2.5,
                        "url": "http://a/title.webm",
                        "x": "(W-w)/2",
                        "y": "H-h-64",
                        "opacity": 0.9,
                    }
                ]
            },
        }
    )
    await ActionReelCompileStage(client=client).run({}, ctx=ctx)


@contextmanager
def _stubbed_sound_io() -> Iterator[None]:
    """Stub the two async I/O calls the sound path makes, as bound in each module.

    ``collect_emitted_leaves`` is sync and fixture-free (it wraps ``asyncio.run``),
    so the pytest ``monkeypatch`` fixture is not available to it — hence plain
    setattr/restore. Both names are patched WHERE THEY ARE BOUND (from-imports),
    which is also why ``presign_sound_asset`` is stubbed twice: ``sound_design``
    signs the music/ambient bed, ``shot_sound`` signs the per-shot accent and
    transition cues. Same fakes as
    ``tests/services/stages/shared/test_sound_design.py`` and
    ``tests/services/stages/demo_video/test_stages.py``.
    """

    async def _fake_library(*_args: object, **_kwargs: object) -> SoundLibrary:
        return SoundLibrary.builtin()

    async def _fake_presign(sfx_id: str) -> str:
        return f"https://example.invalid/{sfx_id}.mp3?sig=x"

    targets = (
        (sound_design, "load_sound_library", _fake_library),
        (sound_design, "presign_sound_asset", _fake_presign),
        (shot_sound, "presign_sound_asset", _fake_presign),
    )
    originals = [(module, name, getattr(module, name)) for module, name, _ in targets]
    for module, name, fake in targets:
        setattr(module, name, fake)
    try:
        yield
    finally:
        for module, name, original in originals:
            setattr(module, name, original)


async def _emit_promo_caption_look(client: VideoCompileClient) -> None:
    """A per-line promo's caption typography, through the real promo resolver.

    The cue build itself is not reachable without live ASR (see ``_CAPTIONS`` in
    test_producer_coverage.py), so this replays the compile call with one cue and
    the look ``PromoRenderStage.caption_look`` resolves for a per-line brief —
    the same client fold the stage uses.
    """
    ctx = _ctx({}, {"brand": {"name": "Merqi"}, "direction": {"motion": "per_line"}})
    look = PromoRenderStage().caption_look(ctx)
    assert look is not None
    await client.compile(
        shots=[{"id": "s1", "seconds": 3.0, "image_url": "http://a/i1.png"}],
        captions=[{"shot_index": 0, "cues": [{"text": "Hook.", "start": 0.0, "end": 1.0}]}],
        captions_render="burned_in",
        caption_look=look,
        idempotency_key="contract:caption-look",
    )


_SCENARIOS: tuple[Callable[[VideoCompileClient], Awaitable[None]], ...] = (
    _emit_promo,
    _emit_generic_video_render,
    _emit_course_stitch,
    _emit_lesson_compile,
    _emit_app_review_compile,
    _emit_finish_block,
    _emit_footage_window,
    _emit_mastering,
    _emit_promo_sound_design,
    _emit_action_reel_overlays,
    _emit_promo_caption_look,
)


def collect_emitted_leaves() -> frozenset[str]:
    """Run every stage scenario and return the union of emitted leaf paths.

    Sync + fixture-free on purpose: ``test_producer_coverage.py`` imports this to
    answer the OTHER drift question — which declared compiler fields no producer
    ever sets (a silently inert knob).
    """

    async def _run_all() -> frozenset[str]:
        seen: set[str] = set()
        for scenario in _SCENARIOS:
            fake, client = _client()
            await scenario(client)
            seen |= fake.seen
        return frozenset(seen)

    return asyncio.run(_run_all())


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_response_reads_are_covered_by_response_schema() -> None:
    assert set(_CompileResponse.model_fields) >= _RESPONSE_READS


def test_schema_artifact_is_walkable() -> None:
    """The vendored artifact must actually resolve into leaves.

    A truncated / wrong-root copy would otherwise make every subset assertion
    below vacuously... loud (empty schema ⇒ everything unknown) or, if it
    resolved to nothing at all, silently permissive. Pin a few known positions.
    """
    assert {"shots.id", "shots.seconds", "music.url", "finish.grade.enabled"} <= _SCHEMA_LEAVES
    # ``tags`` is a free-form dict[str, str]: a LEAF, not an object to descend.
    assert "tags" in _SCHEMA_LEAVES


async def test_promo_render_payload_is_contract_valid() -> None:
    fake, client = _client()
    await _emit_promo(client)
    shot = _shots(fake)[0]
    assert shot["image_url"] == "http://a/i1.png"
    assert shot["audio_url"] == "http://a/1.wav"
    assert shot["caption"] == "Hook."


async def test_generic_video_render_payload_is_contract_valid() -> None:
    fake, client = _client()
    await _emit_generic_video_render(client)
    assert _shots(fake)[0]["audio_url"] == "http://a/1.wav"


async def test_course_stitch_payload_is_contract_valid() -> None:
    fake, client = _client()
    await _emit_course_stitch(client)
    assert _shots(fake)[0]["video_url"] == "http://a/l1.mp4"


async def test_lesson_compile_payload_is_contract_valid() -> None:
    fake, client = _client()
    await _emit_lesson_compile(client)
    assert _shots(fake)[0]["image_url"] == "http://a/sl1.png"


async def test_app_review_compile_payload_is_contract_valid() -> None:
    fake, client = _client()
    await _emit_app_review_compile(client)
    assert _shots(fake)[0]["image_url"] == "http://shot/s.png"


async def test_promo_sound_design_payload_is_contract_valid() -> None:
    """Sound design ON puts a real music + ambient bed and per-shot cues on the
    wire — including ``music_offset_lu`` / ``ambient_offset_lu``, which the old
    hand-mirrored request model did not declare (so no audio scenario could
    exist at all). Against the generated artifact they validate."""
    fake, client = _client()
    await _emit_promo_sound_design(client)
    shots = _shots(fake)
    music = _mapping(fake.last["music"])
    assert str(music["url"]).endswith("corporate_uplift.mp3?sig=x")
    assert "music_offset_lu" in music
    ambient = fake.last["ambient"]
    assert isinstance(ambient, list) and ambient
    first_cue = _mapping(ambient[0])
    assert first_cue["end_seconds"] == sum(_number(shot["seconds"]) for shot in shots)
    assert "ambient_offset_lu" in first_cue
    # Accent cue on its own line's shot; transition whoosh on the cut, not the open.
    assert "transition_sfx" not in shots[0]
    assert any("sfx" in shot for shot in shots)
    assert any("transition_sfx" in shot for shot in shots)


async def test_footage_window_payload_is_contract_valid() -> None:
    """An authored source window rides the wire as all THREE framing keys.

    Asserting the values (not merely the key presence) pins the pass-through: a
    producer that stamped the lead but silently dropped the out-point would still
    emit ``clip_out_seconds`` if it were, say, defaulted — and the coverage test
    would read green over a broken window.
    """
    fake, client = _client()
    await _emit_footage_window(client)
    shot = _shots(fake)[0]
    assert shot["video_url"] == "http://a/footage.mp4"
    assert shot["clip_anchor"] == "start"
    assert shot["clip_lead_seconds"] == 4.0
    assert shot["clip_out_seconds"] == 7.5


async def test_mastering_payload_is_contract_valid() -> None:
    """An authored `mastering` block reaches the compiler as a full MasteringProfile.

    All eight keys, because ``as_mastering_payload`` omits every unset optional —
    a missing key here means an authored knob never reached the encoder.
    """
    fake, client = _client()
    await _emit_mastering(client)
    mastering = _mapping(fake.last["mastering"])
    assert set(mastering) == {
        "bitrate_mode",
        "colorspace",
        "crf",
        "deband",
        "loudness_target",
        "maxrate",
        "platform",
        "true_peak",
    }
    assert mastering["crf"] == 20
    assert mastering["loudness_target"] == -14.0


async def test_finish_block_is_contract_valid() -> None:
    """An authored `finish` block reaches the compiler as a contract-valid
    FinishProfile: the letterbox filled from the brand primary colour."""
    fake, client = _client()
    await _emit_finish_block(client)
    assert fake.last["finish"] == {"letterbox_color": "#280983"}


def test_guard_rejects_grade_knobs_at_finish_top_level() -> None:
    """The grade knobs belong NESTED under ``finish.grade`` (the compiler's
    GradeSpec), not at ``finish`` top level. A FLAT top-level grade knob —
    ``broadcast_safe`` / ``contrast`` / etc. — is an unknown finish field the
    compiler silently IGNORES (its request model is extra="ignore"), so the whole
    grade never applies. That silent no-op is exactly the A7-grade-dropped bug;
    the leaf-subset guard fails it in CI instead — which is why
    as_finish_payload nests the knobs under ``grade``."""
    flat = payload_leaves(
        {"shots": [{"id": "s1", "seconds": 1.0}], "finish": {"broadcast_safe": True}}
    )
    assert unknown_leaves(flat.leaves, _SCHEMA_LEAVES, scalar_leaves=flat.scalar_leaves) == {
        "finish.broadcast_safe"
    }

    flat_contrast = payload_leaves(
        {"shots": [{"id": "s1", "seconds": 1.0}], "finish": {"contrast": 1.2}}
    )
    assert unknown_leaves(
        flat_contrast.leaves, _SCHEMA_LEAVES, scalar_leaves=flat_contrast.scalar_leaves
    ) == {"finish.contrast"}


def test_nested_grade_is_contract_valid() -> None:
    """A whole-film grade nested under ``finish.grade`` (the shape
    ``FinishSpec.as_finish_payload`` now emits) IS contract-valid — the compiler's
    FinishProfile.grade renders it (VC-5). This is the positive proof that the A7
    grade reaches the wire instead of being silently dropped."""
    assert_contract_valid(
        {
            "shots": [{"id": "s1", "seconds": 1.0}],
            "finish": {
                "letterbox_color": "#280983",
                "grade": {
                    "enabled": True,
                    "look": "cinematic",
                    "shadow_tint": "#101820",
                    "highlight_tint": "#fff4e0",
                    "vignette": 0.2,
                    "grain": 0.1,
                    "glow": 0.15,
                    "sharpen": 0.1,
                    "grade_intensity": 0.9,
                },
            },
        }
    )


def test_guard_rejects_imagined_field() -> None:
    with pytest.raises(AssertionError):
        assert_contract_valid({"shots": [{"id": "s1", "seconds": 1.0, "videoUrl": "typo"}]})


def test_guard_rejects_scalar_where_the_schema_declares_an_object() -> None:
    """A SCALAR at an object position must fail — the case the replaced pydantic
    mirror caught and a naive ancestor rule lets through.

    ``shots.audio`` is a ``ShotAudio`` OBJECT (``gain_db``/``mute``/``fade_*``).
    Emitting a bare string there is a type error the compiler cannot honour. The
    "path is an ancestor of a declared leaf" acceptance rule exists for a real
    reason (a producer stopping early at ``music: None``), but applied naively it
    accepts this too: ``shots.audio.gain_db`` starts with ``shots.audio.``, so
    the garbage string rides through unflagged. The payload walk therefore
    records WHICH leaves came from a scalar, and rule 3 refuses them.

    Do NOT "simplify" the rule back to a bare prefix test — that silently
    re-narrows the guard to less than the mirror it replaced.
    """
    with pytest.raises(AssertionError, match=r"shots\.audio"):
        assert_contract_valid({"shots": [{"id": "s1", "seconds": 1.0, "audio": "garbage-string"}]})


def test_guard_accepts_a_producer_stopping_early_at_none() -> None:
    """The legitimate early stop rule 3 exists for — pinned against its abuse.

    A producer that has no music bed emits ``music: None`` even though the schema
    declares ``music.url`` and friends. That is not an invented field and must
    stay valid; only a SCALAR at such a position is a violation (see
    ``test_guard_rejects_scalar_where_the_schema_declares_an_object``). These two
    tests are a pair — a change that makes either pass by breaking the other has
    not fixed anything.
    """
    assert_contract_valid(
        {"shots": [{"id": "s1", "seconds": 1.0, "audio": None}], "music": None, "finish": {}}
    )


def test_guard_rejects_zero_seconds() -> None:
    with pytest.raises(AssertionError):
        assert_contract_valid({"shots": [{"id": "s1", "seconds": 0.0}]})


def test_emit_shot_timeline_flag_is_declared_by_the_schema() -> None:
    """The fake client can set ``emit_shot_timeline``; the old hand mirror neither
    declared it nor allowed extras, so that path was a latent failure masked only
    by no exercised stage enabling it. Against the generated artifact it is real."""
    assert "emit_shot_timeline" in _SCHEMA_LEAVES


async def test_emit_shot_timeline_payload_is_contract_valid() -> None:
    fake = _ValidatingCompileClient()
    await fake.compile(shots=[{"id": "s1", "seconds": 1.0}], emit_shot_timeline=True)
    assert fake.last["emit_shot_timeline"] is True


def test_collect_emitted_leaves_covers_every_scenario() -> None:
    """The producer-coverage helper must actually execute the stages (an empty or
    tiny set would make the sibling inert-knob test report the whole schema)."""
    emitted = collect_emitted_leaves()
    assert {"shots.id", "shots.seconds", "shots.audio_url", "finish.letterbox_color"} <= emitted
    # The sound-design scenario must be in the union too — its whole point is to
    # keep the audio bus off the uncovered list for a real reason.
    assert {"music.url", "ambient.url", "shots.sfx.url", "shots.transition_sfx.url"} <= emitted
    assert not unknown_leaves(emitted, _SCHEMA_LEAVES)
