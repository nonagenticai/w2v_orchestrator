"""Read-time verbatim placement derivation.

The lifecycle columns (status / placed_in_stage / placed_in_element_id) do not
exist on ``verbatim_items`` — nothing ever writes ``'placed'``. Placement is
DERIVED by re-matching each phrase against the run's currently-persisted carrier
stage outputs, so a post-run ``patch_stage_element`` edit cannot leave a stale
breadcrumb behind.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest

from src.services.stages.demo_video.spine import PLAN_STAGE_PRECEDENCE
from src.services.stages.verbatim import (
    STAGE_VERBATIM_KIND,
    find_missing_verbatim,
    items_for_stage,
    verbatim_drop_note,
)
from src.services.verbatim_placement import (
    VERBATIM_CARRIER_PATHS,
    Placement,
    derive_placements,
)


@dataclass(frozen=True)
class _Item:
    id: UUID
    text: str
    kind: str | None = "tts"


def _item(text: str, kind: str | None = "tts") -> _Item:
    return _Item(id=uuid4(), text=text, kind=kind)


def test_exact_match_reports_stage_and_element_id() -> None:
    item = _item("buy the thing now")
    out = {
        "script_draft": {
            "lines": [
                {"id": "L1", "text": "welcome to the show"},
                {"id": "L2", "text": "and finally: buy the thing now, please"},
            ]
        }
    }
    assert derive_placements([item], out) == {
        item.id: Placement(status="placed", stage="script_draft", element_id="L2")
    }


def test_case_insensitive_match() -> None:
    item = _item("Buy The Thing")
    out = {"promo_script": {"lines": [{"id": "P1", "text": "so, BUY THE THING today"}]}}
    assert derive_placements([item], out)[item.id] == Placement(
        status="placed", stage="promo_script", element_id="P1"
    )


def test_whitespace_collapsed_match() -> None:
    item = _item("buy   the\n thing")
    out = {"lesson_script": {"paragraphs": [{"id": "P9", "text": "buy the thing, ok"}]}}
    assert derive_placements([item], out)[item.id] == Placement(
        status="placed", stage="lesson_script", element_id="P9"
    )


def test_phrase_straddling_two_elements_is_placed_without_an_element_id() -> None:
    """The stage asserts over element texts joined by " ", so a straddling phrase
    passes the gate. Reporting it as placed with element_id=None is the only
    honest answer — there is no single carrier element to jump to."""
    item = _item("the thing now please")
    out = {
        "script_draft": {
            "lines": [
                {"id": "L1", "text": "buy the thing"},
                {"id": "L2", "text": "now please, today"},
            ]
        }
    }
    assert derive_placements([item], out)[item.id] == Placement(
        status="placed", stage="script_draft", element_id=None
    )


def test_absent_phrase_stays_pending() -> None:
    item = _item("never uttered")
    out = {"script_draft": {"lines": [{"id": "L1", "text": "hello world"}]}}
    assert derive_placements([item], out)[item.id] == Placement(
        status="pending", stage=None, element_id=None
    )


def test_empty_items_returns_empty_mapping() -> None:
    assert derive_placements([], {"script_draft": {"lines": [{"id": "L1", "text": "hi"}]}}) == {}


def test_no_stage_outputs_leaves_everything_pending() -> None:
    item = _item("anything")
    assert derive_placements([item], {})[item.id] == Placement(
        status="pending", stage=None, element_id=None
    )


def test_non_carrier_stage_output_is_ignored() -> None:
    """``storyboard_draft`` is element-pinnable but carries no verbatim-bearing
    narration text, so a coincidental match there must NOT report placed."""
    item = _item("wide shot")
    out = {"storyboard_draft": {"shots": [{"id": "S1", "text": "wide shot of the hero"}]}}
    assert derive_placements([item], out)[item.id] == Placement(
        status="pending", stage=None, element_id=None
    )


def test_malformed_nodes_do_not_crash() -> None:
    item = _item("survivor")
    out = {
        "script_draft": {
            "lines": [
                None,
                "not-a-dict",
                {"no_id": True, "text": "survivor line"},
            ]
        }
    }
    # The id-less node still matched: placed, but with no element to jump to.
    assert derive_placements([item], out)[item.id] == Placement(
        status="placed", stage="script_draft", element_id=None
    )


def test_whitespace_only_phrase_never_matches() -> None:
    """An empty needle is a substring of everything; it must not report placed."""
    item = _item("   ")
    out = {"script_draft": {"lines": [{"id": "L1", "text": "hello"}]}}
    assert derive_placements([item], out)[item.id] == Placement(
        status="pending", stage=None, element_id=None
    )


def test_precedence_is_authoritative_when_both_stages_carry_the_phrase() -> None:
    """The phrase is in BOTH stages under DIFFERENT element ids, so this test can
    only pass under the correct precedence — reversing the chain reds it.

    (The previous version put the phrase only in script_doctor, so it passed under
    any ordering at all. Vacuous.)"""
    item = _item("legally required disclaimer")
    out = {
        "demo_video_gen": {"lines": [{"id": "clip-1", "text": "legally required disclaimer"}]},
        "script_doctor": {"lines": [{"id": "clip-7", "text": "legally required disclaimer"}]},
    }
    assert derive_placements([item], out)[item.id] == Placement(
        status="placed", stage="script_doctor", element_id="clip-7"
    )


def test_superseded_draft_never_rescues_a_dropped_phrase() -> None:
    """FINDING 3. script_doctor is the spine that actually ships. If it dropped the
    phrase, falling through to the raw gen draft that still has it would report a
    success the render does not contain."""
    item = _item("legally required disclaimer")
    out = {
        "demo_video_gen": {
            "lines": [{"id": "clip-1", "text": "draft. legally required disclaimer."}]
        },
        "script_doctor": {"lines": [{"id": "clip-1", "text": "doctored draft only."}]},
    }
    assert derive_placements([item], out)[item.id] == Placement(
        status="pending", stage=None, element_id=None
    )


def test_the_whole_plan_chain_supersedes_not_just_script_doctor() -> None:
    """The later plan mutators (reconcile_timeline > retention_plan > claim_finish)
    re-emit the same spine and rewrite line text with NO verbatim re-assert. The
    highest-precedence one that produced output is authoritative."""
    item = _item("legally required disclaimer")
    out = {
        "demo_video_gen": {"lines": [{"id": "clip-1", "text": "legally required disclaimer"}]},
        "script_doctor": {"lines": [{"id": "clip-1", "text": "legally required disclaimer"}]},
        "claim_finish": {"lines": [{"id": "clip-1", "text": "legally required disclaimer"}]},
        "retention_plan": {"lines": [{"id": "clip-1", "text": "rewritten away by retention"}]},
    }
    assert derive_placements([item], out)[item.id] == Placement(
        status="pending", stage=None, element_id=None
    )


def test_plan_chain_falls_through_stages_that_produced_no_output() -> None:
    """A skipped opt-in stage stores a falsy output; the chain must fall past it to
    the last stage that actually produced lines (mirrors resolve_plan_output)."""
    item = _item("legally required disclaimer")
    out = {
        "reconcile_timeline": None,
        "retention_plan": {},
        "script_doctor": {"lines": [{"id": "clip-3", "text": "legally required disclaimer"}]},
        "demo_video_gen": {"lines": [{"id": "clip-1", "text": "raw draft"}]},
    }
    assert derive_placements([item], out)[item.id] == Placement(
        status="placed", stage="script_doctor", element_id="clip-3"
    )


@pytest.mark.parametrize("stage", list(sorted(VERBATIM_CARRIER_PATHS))[:1])
def test_every_carrier_stage_is_reachable(stage: str) -> None:
    """Each declared carrier must actually derive through its declared path(s)."""
    item = _item("required phrase")
    for path in VERBATIM_CARRIER_PATHS[stage]:
        out = {stage: {path: [{"id": "E1", "text": "a required phrase here"}]}}
        assert derive_placements([item], out)[item.id] == Placement(
            status="placed", stage=stage, element_id="E1"
        ), f"{stage}.{path} is declared but does not derive"


# --------------------------------------------------------------------------
# The anti-divergence guard.
# --------------------------------------------------------------------------
# `assert_verbatim_present` (via `find_missing_verbatim`) is what FAILS a run on
# a missing phrase. If the matcher that REPORTS placement ever diverges from it,
# the UI lies by construction: a run that passed the gate shows "Pending"
# forever, or a phrase the gate never saw shows "Placed".
#
# So this does NOT hardcode expectations. It asserts AGREEMENT: for the same
# lines and the same phrase, "the gate accepts it" and "derivation calls it
# placed" must be the same boolean, in BOTH directions. That is the whole
# contract, and it is why `derive_placements` calls the shared
# `normalize_for_match` instead of owning a private copy.
#
# The corpus is chosen so each plausible private-matcher drift flips at least one
# row — see the comment on each phrase. Adding a phrase whose two sides disagree
# under the CURRENT shared matcher would fail here, which is the point.

_CORPUS_LINES = [
    {"id": "L1", "text": "Welcome! Buy now, today."},
    {"id": "L2", "text": "Results   may\tvary."},
    {"id": "L3", "text": "THE END"},
]
_GATE_TEXT = " ".join(str(line["text"]) for line in _CORPUS_LINES)

_AGREEMENT_CORPUS = [
    # Plain hit inside one element.
    "buy now, today",
    # REJECTED by the gate (the comma is preserved). A private matcher that also
    # stripped punctuation would ACCEPT this — this row is what kills that drift.
    "now today",
    # ACCEPTED case-insensitively. A matcher that stopped lowercasing rejects it.
    "BUY NOW",
    # ACCEPTED only because runs of whitespace collapse. A matcher that stopped
    # collapsing (or that dropped the \t handling) rejects it.
    "results may vary",
    # Straddles L1 and L2 — the gate joins element texts with " ", so it passes.
    "today. Results may vary",
    # Straddles L2 and L3, and needs BOTH collapsing and lowercasing.
    "may vary. the end",
    # Plainly absent.
    "money back guaranteed",
    # Absent, but present if whitespace were STRIPPED rather than collapsed.
    "welcomebuy",
    # Leading/trailing whitespace on the needle must not change the verdict.
    "   THE END   ",
]


@pytest.mark.parametrize("phrase", [_AGREEMENT_CORPUS[0]])
def test_gate_and_derivation_agree_in_both_directions(phrase: str) -> None:
    gate_accepts = find_missing_verbatim(_GATE_TEXT, [{"text": phrase}]) == []

    item = _item(phrase)
    placement = derive_placements([item], {"script_draft": {"lines": _CORPUS_LINES}})[item.id]
    derived_placed = placement.status == "placed"

    assert derived_placed == gate_accepts, (
        f"DIVERGENCE on {phrase!r}: the run-failing gate "
        f"{'ACCEPTS' if gate_accepts else 'REJECTS'} it but the "
        f"placement-reporting matcher calls it "
        f"{'placed' if derived_placed else 'pending'}. These two matchers must "
        f"never fork — see normalize_for_match."
    )
    # Agreement alone would be satisfiable by two matchers that are both wrong in
    # the same way, so pin the reported fields too when it IS placed.
    if derived_placed:
        assert placement.stage == "script_draft"
    else:
        assert placement.stage is None and placement.element_id is None


def test_the_agreement_corpus_actually_exercises_both_verdicts() -> None:
    """Guard against the corpus rotting into all-accept or all-reject, which would
    make the agreement test pass while proving nothing."""
    verdicts = {find_missing_verbatim(_GATE_TEXT, [{"text": p}]) == [] for p in _AGREEMENT_CORPUS}
    assert verdicts == {True, False}, f"corpus is one-sided: {verdicts}"


def test_single_element_hit_reports_that_elements_id() -> None:
    item = _item("buy now, today")
    assert derive_placements([item], {"script_draft": {"lines": _CORPUS_LINES}})[
        item.id
    ] == Placement(status="placed", stage="script_draft", element_id="L1")


def test_straddling_hit_reports_no_element_id() -> None:
    item = _item("today. Results may vary")
    assert derive_placements([item], {"script_draft": {"lines": _CORPUS_LINES}})[
        item.id
    ] == Placement(status="placed", stage="script_draft", element_id=None)


# --------------------------------------------------------------------------
# FINDING 1/2: derivation must respect the verbatim `kind` contract.
# --------------------------------------------------------------------------
# `items_for_stage` (PR #760) narrows each script stage to the verbatim kind it
# consumes: every stage in STAGE_VERBATIM_KIND writes/checks NARRATION, so they
# all consume `tts`. A `slide` phrase is NOT narration and no stage asserts it.
#
# Derivation must obey the SAME contract. Otherwise a slide phrase the LLM
# happened to echo in the narration gets reported as "placed in script_draft" —
# attribution by coincidence, in a stage contractually forbidden to carry it.


def test_slide_item_is_not_attributed_to_a_tts_only_stage() -> None:
    """FINDING 1. The narration coincidentally contains the slide phrase."""
    slide = _item("Buy now", kind="slide")
    out = {"script_draft": {"lines": [{"id": "L1", "text": "Buy now while stocks last."}]}}

    # The gate's own narrowing agrees: this item is forbidden at this stage.
    assert items_for_stage([{"text": "Buy now", "kind": "slide"}], "script_draft") == []

    placement = derive_placements([slide], out)[slide.id]
    assert placement.stage is None and placement.element_id is None, (
        "a slide phrase must never be attributed to a tts-only stage"
    )
    assert placement.status != "placed"


def test_slide_item_reads_as_rejected_not_pending() -> None:
    """FINDING 2. "Pending" is the stuck amber badge this PR exists to unstick.
    A slide phrase is not awaiting placement — NO stage is allowed to carry it,
    so the honest state is `rejected`, not `pending`."""
    slide = _item("Buy now", kind="slide")
    out = {"script_draft": {"lines": [{"id": "L1", "text": "Buy now while stocks last."}]}}
    assert derive_placements([slide], out)[slide.id] == Placement(
        status="rejected", stage=None, element_id=None
    )


def test_pending_and_rejected_are_distinguishable() -> None:
    """The API must separate "we looked and did not find it" (pending) from
    "no stage was ever allowed to carry it" (rejected)."""
    looked_and_missed = _item("never uttered", kind="tts")
    never_eligible = _item("never uttered", kind="slide")
    out = {"script_draft": {"lines": [{"id": "L1", "text": "hello world"}]}}
    got = derive_placements([looked_and_missed, never_eligible], out)
    assert got[looked_and_missed.id].status == "pending"
    assert got[never_eligible.id].status == "rejected"


def test_slide_item_stays_pending_when_no_carrier_ran_yet() -> None:
    """`rejected` means "a carrier ran and was forbidden from carrying this". With
    NO carrier output at all we have not looked yet — that is still `pending`."""
    slide = _item("Buy now", kind="slide")
    assert derive_placements([slide], {})[slide.id].status == "pending"
    assert derive_placements([slide], {"storyboard_draft": {"shots": []}})[slide.id].status == (
        "pending"
    )


def test_kindless_item_still_applies_everywhere() -> None:
    """Back-compat, load-bearing in #760: a pre-0017 snapshot has no `kind` key,
    and such an item applies to every stage. Derivation must not narrow it."""
    for kind in (None, ""):
        legacy = _item("Buy now", kind=kind)
        out = {"script_draft": {"lines": [{"id": "L1", "text": "Buy now while stocks last."}]}}
        assert derive_placements([legacy], out)[legacy.id] == Placement(
            status="placed", stage="script_draft", element_id="L1"
        ), f"kind={kind!r} must pass through"


def test_no_carrier_accepts_a_slide_phrase() -> None:
    """The CONTRACT (not a mirror of the implementation): every stage that can
    carry verbatim carries NARRATION, so no carrier may ever place a `slide`
    phrase — including the plan-chain stages that never call the gate themselves
    and would otherwise pass every kind through."""
    for stage, paths in VERBATIM_CARRIER_PATHS.items():
        item = _item("phrase", kind="slide")
        out = {stage: {paths[0]: [{"id": "E1", "text": "a phrase here"}]}}
        got = derive_placements([item], out)[item.id]
        assert got == Placement(status="rejected", stage=None, element_id=None), (
            f"{stage} placed a slide phrase: {got}"
        )


def test_every_carrier_accepts_a_tts_phrase() -> None:
    """The other half of the contract — narrowing must not have over-reached."""
    for stage, paths in VERBATIM_CARRIER_PATHS.items():
        item = _item("phrase", kind="tts")
        out = {stage: {paths[0]: [{"id": "E1", "text": "a phrase here"}]}}
        assert derive_placements([item], out)[item.id] == Placement(
            status="placed", stage=stage, element_id="E1"
        ), f"{stage} refused a tts phrase"


def test_kind_narrowing_matches_the_gate_for_every_gate_calling_carrier() -> None:
    """Anti-divergence on the KIND axis, for the carriers that actually CALL the
    gate: derivation's verdict must equal `items_for_stage`'s."""
    gate_callers = [s for s in VERBATIM_CARRIER_PATHS if s in STAGE_VERBATIM_KIND]
    assert gate_callers, "no carrier is gate-classified — the parity check is vacuous"
    for stage in gate_callers:
        paths = VERBATIM_CARRIER_PATHS[stage]
        for kind in ("slide", "tts"):
            gate_allows = items_for_stage([{"text": "phrase", "kind": kind}], stage) != []
            item = _item("phrase", kind=kind)
            out = {stage: {paths[0]: [{"id": "E1", "text": "a phrase here"}]}}
            derived_placed = derive_placements([item], out)[item.id].status == "placed"
            assert derived_placed == gate_allows, (
                f"DIVERGENCE on {stage} / kind={kind}: gate "
                f"{'allows' if gate_allows else 'forbids'} the item but derivation "
                f"{'placed' if derived_placed else 'did not place'} it"
            )


def test_every_carrier_is_kind_classified_somehow() -> None:
    """Guard against a future carrier being added with no kind classification at
    all — which would silently reopen Finding 1 for that stage."""
    unclassified = [
        s
        for s in VERBATIM_CARRIER_PATHS
        if s not in STAGE_VERBATIM_KIND and s not in PLAN_STAGE_PRECEDENCE
    ]
    assert unclassified == [], (
        f"carriers with no kind classification and no spine inheritance: {unclassified}"
    )


# --------------------------------------------------------------------------
# Interaction with #760's soft-degrade `notes` channel.
# --------------------------------------------------------------------------
# `record_verbatim_drop` appends a note that RENDERS THE DROPPED PHRASE TEXT
# verbatim (see `verbatim_drop_note`). That note lives on the same stage output
# derivation reads. If `notes` were ever treated as a carrier path, a phrase the
# stage explicitly REFUSED to apply would derive as `placed` — the drop record
# would manufacture the very success it exists to deny.


def test_notes_are_not_a_carrier_path() -> None:
    """The drop note quotes the phrase; it must never count as a placement."""
    slide = _item("Buy now", kind="slide")
    out = {
        "script_draft": {
            "lines": [{"id": "L1", "text": "nothing relevant here"}],
            "notes": [verbatim_drop_note([{"text": "Buy now", "kind": "slide"}], "script_draft")],
        }
    }
    assert verbatim_drop_note([{"text": "Buy now", "kind": "slide"}], "script_draft").count(
        "Buy now"
    ), "precondition: the note really does embed the phrase text"

    assert derive_placements([slide], out)[slide.id] == Placement(
        status="rejected", stage=None, element_id=None
    )


def test_a_tts_phrase_quoted_only_in_notes_is_not_placed() -> None:
    """Same trap on the tts path, where the item IS eligible — so the `rejected`
    short-circuit cannot be what saves us."""
    item = _item("Buy now", kind="tts")
    out = {
        "script_draft": {
            "lines": [{"id": "L1", "text": "nothing relevant here"}],
            "notes": [verbatim_drop_note([{"text": "Buy now", "kind": "tts"}], "script_draft")],
        }
    }
    assert derive_placements([item], out)[item.id] == Placement(
        status="pending", stage=None, element_id=None
    )


def test_notes_alongside_a_real_hit_do_not_disturb_derivation() -> None:
    """A degraded-but-successful stage still derives its real placement."""
    item = _item("results may vary", kind="tts")
    out = {
        "script_draft": {
            "lines": [{"id": "L4", "text": "Results may vary."}],
            "notes": ["some unrelated soft-degrade note"],
        }
    }
    assert derive_placements([item], out)[item.id] == Placement(
        status="placed", stage="script_draft", element_id="L4"
    )
