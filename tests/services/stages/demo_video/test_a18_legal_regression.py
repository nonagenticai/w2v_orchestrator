"""A18 phase 2 make-or-break regression guard (AC-1): the legal seam is INERT.

The whole point of phase 2 is that a run that did not opt into ``legal.mode``
produces the SAME shot list as today. The dangerous case is not the empty brief —
it is the brief shipping in PRODUCTION RIGHT NOW, which carries a ``legal`` block
of ``{"trademarks": [...], "defensibility": "..."}`` with **no ``mode`` key**. That
block was added for other consumers; it must never wake ``claim_finish`` and must
never put a disclaimer card in front of a customer.

So this pins the seam at the shot-list level, not at a helper: the demo_video shot
list is built end-to-end through the real ``DemoVideoCompileStage.build_shots`` +
the real compile-tail disclaimer splice, once with no ``legal`` block at all and
once with the live prod block, and the two must be EQUAL with no ``__disclaimer__``
line anywhere.

Pure + deterministic: no DB, no I/O — a fake ``ctx`` carries a ``config_snapshot``
+ ``prior_output``, and the stage is built via ``__new__`` (``build_shots`` does not
use ``self.__init__``'s client).
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

from src.services.stages.demo_video.compile import DemoVideoCompileStage
from src.services.stages.shared.disclaimer_card import (
    DISCLAIMER_LINE_ID,
    build_disclaimer_shot,
    splice_disclaimer_card,
)

# The exact shape a prod brief carries TODAY: trademarks + defensibility, NO mode.
_PROD_LEGAL: dict[str, Any] = {
    "trademarks": ["Acme®", "SuperWidget"],
    "defensibility": "patent pending, US 12/345,678",
}

_GEN = {
    "lines": [
        {"id": "clip-1", "text": "Acme is the fastest tracker, 40% faster than the rest."},
        {"id": "clip-2", "text": "Only $9 per month."},
    ],
    "bdd_job_id": "job-1",
}
_NARRATE = {
    "clips": [
        {"line_id": "clip-1", "artifact_url": "https://minio.test/a1.mp3", "seconds": 4.2},
        {"line_id": "clip-2", "artifact_url": "https://minio.test/a2.mp3", "seconds": 3.1},
    ]
}
_RECORD = {
    "clips": [
        {"line_id": "clip-1", "video_url": "https://minio.test/v1.mp4", "seconds": 4.0},
        {"line_id": "clip-2", "video_url": "https://minio.test/v2.mp4", "seconds": 3.0},
    ]
}


def _stage() -> DemoVideoCompileStage:
    # build_shots does not use the client the __init__ builds.
    return DemoVideoCompileStage.__new__(DemoVideoCompileStage)


def _ctx(config_snapshot: dict[str, Any], prior: dict[str, Any] | None = None) -> Any:
    outputs: dict[str, Any] = {
        "demo_video_gen": copy.deepcopy(_GEN),
        "demo_video_narrate": copy.deepcopy(_NARRATE),
        "demo_video_record": copy.deepcopy(_RECORD),
        **(prior or {}),
    }
    return SimpleNamespace(
        config_snapshot=config_snapshot,
        prior_output=lambda stage: outputs.get(stage),
    )


def _shot_list(config_snapshot: dict[str, Any], prior: dict[str, Any] | None = None) -> list[dict]:
    """The demo_video shot list through the real compile tail's legal splice.

    Mirrors ``CompileStageBase.handle``: build the shots, then splice the (opt-in)
    disclaimer card. The title-card / ken-burns / presign steps of the tail are all
    independently opt-in and untouched by A18, so they are not in the seam under test.
    """
    ctx = _ctx(config_snapshot, prior)
    shots = _stage().build_shots(ctx)
    return splice_disclaimer_card(shots, build_disclaimer_shot(ctx))


# ── THE guard. ────────────────────────────────────────────────────────────────
def test_ac1_prod_legal_block_without_mode_leaves_shots_byte_identical() -> None:
    baseline = _shot_list({})
    with_prod_legal = _shot_list({"legal": copy.deepcopy(_PROD_LEGAL)})

    assert with_prod_legal == baseline, (
        "a prod `legal` block with no `mode` key must not change the shot list"
    )
    for shots in (baseline, with_prod_legal):
        assert DISCLAIMER_LINE_ID not in [shot.get("line_id") for shot in shots]
    assert baseline, "the fixture must actually produce shots, or this guard proves nothing"


def test_ac1_legal_mode_off_leaves_shots_byte_identical() -> None:
    baseline = _shot_list({})
    off = _shot_list({"legal": {**copy.deepcopy(_PROD_LEGAL), "mode": "off"}})
    assert off == baseline
    assert DISCLAIMER_LINE_ID not in [shot.get("line_id") for shot in off]


def test_ac1_no_corpus_means_claim_finish_stored_nothing_so_no_card() -> None:
    # `legal.mode=auto` but the stage skipped (no ingested corpus) ⇒ stored `{}`.
    baseline = _shot_list({})
    skipped = _shot_list(
        {"legal": {**copy.deepcopy(_PROD_LEGAL), "mode": "auto"}},
        prior={"claim_finish": {}},
    )
    assert skipped == baseline
    assert DISCLAIMER_LINE_ID not in [shot.get("line_id") for shot in skipped]


# ── The positive control: the SAME seam DOES add the card when it should. ─────
def test_ac8_card_appears_at_the_tail_when_claim_finish_mandated_one() -> None:
    baseline = _shot_list({})
    with_card = _shot_list(
        {"legal": {**copy.deepcopy(_PROD_LEGAL), "mode": "auto"}},
        prior={
            "claim_finish": {
                **copy.deepcopy(_GEN),
                "disclaimers": [
                    {
                        "kind": "pricing",
                        "text": "Pricing subject to change.",
                        "line_ids": ["clip-2"],
                    }
                ],
            }
        },
    )
    assert with_card[:-1] == baseline, "the storyboard shots are untouched"
    assert with_card[-1]["line_id"] == DISCLAIMER_LINE_ID
    assert with_card[-1]["description"] == "Pricing subject to change."
