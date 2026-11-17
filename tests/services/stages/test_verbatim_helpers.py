"""Unit tests for the verbatim helpers in stages.base."""

from __future__ import annotations

import logging

import pytest
from pydantic import BaseModel, Field

from src.services.stages.base import (
    StageError,
    assert_verbatim_present,
    find_missing_verbatim,
    verbatim_prompt_block,
)
from src.services.stages.verbatim import (
    STAGE_VERBATIM_KIND,
    VERBATIM_DROP_LOG_EVENT,
    items_for_stage,
    partition_for_stage,
    record_verbatim_drop,
)


def test_verbatim_prompt_block_empty() -> None:
    assert verbatim_prompt_block([]) == ""


def test_verbatim_prompt_block_contains_phrase() -> None:
    block = verbatim_prompt_block([{"text": "Buy now"}, {"text": "Limited offer"}])
    assert block != ""
    assert "Buy now" in block
    assert "Limited offer" in block
    assert "verbatim" in block.lower()


def test_find_missing_verbatim_present() -> None:
    text = "Welcome. Buy now while you can. Thanks."
    assert find_missing_verbatim(text, [{"text": "Buy now"}]) == []


def test_find_missing_verbatim_case_and_whitespace_insensitive() -> None:
    text = "Welcome.   BUY    NOW  while you can."
    assert find_missing_verbatim(text, [{"text": "buy now"}]) == []


def test_find_missing_verbatim_missing() -> None:
    text = "Welcome to the show."
    assert find_missing_verbatim(text, [{"text": "Buy now"}]) == ["Buy now"]


def test_assert_verbatim_present_raises_on_missing() -> None:
    with pytest.raises(StageError):
        assert_verbatim_present("nothing here", [{"text": "Buy now"}], stage="script_draft")


def test_assert_verbatim_present_ok_when_present() -> None:
    assert assert_verbatim_present("please Buy now today", [{"text": "Buy now"}], stage="x") is None


def test_assert_verbatim_present_ok_when_empty() -> None:
    assert assert_verbatim_present("anything", [], stage="x") is None


# ---------------------------------------------------------------------------
# items_for_stage — stage→kind scoping (the fix for the slide-verbatim hard-fail)
# ---------------------------------------------------------------------------

_TTS = {"text": "Buy now", "kind": "tts"}
_SLIDE = {"text": "Limited offer", "kind": "slide"}
_NO_KIND = {"text": "Legacy phrase"}


def test_items_for_stage_script_stage_keeps_only_tts_and_kindless() -> None:
    got = items_for_stage([_TTS, _SLIDE, _NO_KIND], "script_draft")
    assert got == [_TTS, _NO_KIND]


def test_items_for_stage_kindless_item_applies_everywhere() -> None:
    """Backward compat: a snapshot written before 0017 has no ``kind`` key."""
    for stage in ("script_draft", "promo_script", "script_doctor", "some_unmapped_stage"):
        assert items_for_stage([_NO_KIND], stage) == [_NO_KIND]


def test_items_for_stage_empty_kind_string_applies_everywhere() -> None:
    item = {"text": "x", "kind": ""}
    assert items_for_stage([item], "script_draft") == [item]


def test_items_for_stage_unmapped_stage_applies_everything() -> None:
    """A stage absent from the map keeps today's apply-to-everything behaviour."""
    items = [_TTS, _SLIDE, _NO_KIND]
    assert items_for_stage(items, "promo_slides") == items
    assert items_for_stage(items, "") == items


def test_items_for_stage_empty_input_is_empty() -> None:
    assert items_for_stage([], "script_draft") == []


def test_items_for_stage_returns_a_new_list() -> None:
    items = [_NO_KIND]
    assert items_for_stage(items, "unmapped") is not items


def test_every_mapped_script_stage_consumes_tts() -> None:
    """Guard against a typo'd kind silently disabling enforcement for a stage."""
    assert STAGE_VERBATIM_KIND
    assert set(STAGE_VERBATIM_KIND.values()) == {"tts"}


# ---------------------------------------------------------------------------
# THE SILENT-DROP GUARD
#
# `kind='slide'` is the SPA's DEFAULT for a new verbatim item
# (VerbatimItemList.svelte: `formKind = fixedKind ?? 'slide'`, and the run page's
# add-form passes no fixedKind). No stage consumes it — every slide stage is a
# deterministic projection with no LLM prompt. So narrowing alone turns the
# default user action into silent data loss: green run, phrase absent, nobody
# told. These tests make that impossible to reintroduce.
# ---------------------------------------------------------------------------


class _NotesOutput(BaseModel):
    notes: list[str] = Field(default_factory=list)


class _NoNotesOutput(BaseModel):
    pass


_SLIDE_ITEM = {"id": "11111111-1111-1111-1111-111111111111", "text": "Ship it", "kind": "slide"}


def test_slide_kind_item_is_consumed_by_no_stage() -> None:
    """Pins the premise the guard below rests on. If a slide stage ever starts
    consuming verbatim, this reds — and the guard's rationale must be revisited."""
    consumers = [s for s in STAGE_VERBATIM_KIND if items_for_stage([_SLIDE_ITEM], s)]
    assert consumers == []


def test_dropped_item_is_never_silently_lost() -> None:
    """Every narrowing stage must either CONSUME an item or RECORD dropping it."""
    for stage in STAGE_VERBATIM_KIND:
        kept, dropped = partition_for_stage([_SLIDE_ITEM], stage)
        consumed = bool(kept)
        recorded = bool(record_verbatim_drop(_NotesOutput(), dropped, stage=stage).notes)
        assert consumed or recorded, f"{stage} silently dropped {_SLIDE_ITEM}"


def test_drop_note_names_the_stage_the_text_and_the_kind() -> None:
    note = record_verbatim_drop(_NotesOutput(), [_SLIDE_ITEM], stage="script_draft").notes[0]
    assert "Ship it" in note
    assert "script_draft" in note
    assert "slide" in note


def test_record_verbatim_drop_appends_to_existing_notes() -> None:
    out = record_verbatim_drop(_NotesOutput(notes=["earlier"]), [_SLIDE_ITEM], stage="script_draft")
    assert len(out.notes) == 2
    assert out.notes[0] == "earlier"


def test_record_verbatim_drop_never_raises_without_a_notes_channel() -> None:
    """A demo must not die because an output model lacks the channel."""
    assert record_verbatim_drop(_NoNotesOutput(), [_SLIDE_ITEM], stage="script_draft") is not None


def test_drop_emits_a_greppable_structured_log(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="src.services.stages.verbatim"):
        record_verbatim_drop(_NotesOutput(), [_SLIDE_ITEM], stage="script_draft")
    line = "\n".join(r.getMessage() for r in caplog.records)
    assert VERBATIM_DROP_LOG_EVENT in line
    assert "stage=script_draft" in line
    assert _SLIDE_ITEM["id"] in line
    assert "kinds=slide" in line


def test_every_narrowing_stage_output_can_carry_notes() -> None:
    """Structural guard: a stage that narrows MUST have somewhere to record it.

    Walks the real blueprint registry so adding a stage to STAGE_VERBATIM_KIND
    without a `notes` channel on its output model fails here, not in prod.
    """
    import src.services.blueprints  # noqa: F401 -- registers every blueprint
    from src.services.blueprints.registry import get_registry

    seen: set[str] = set()
    for bp in get_registry().all():
        for spec in bp.stages:
            if spec.name not in STAGE_VERBATIM_KIND:
                continue
            seen.add(spec.name)
            output_model = spec.handler_factory({}).output_model
            assert "notes" in output_model.model_fields, (
                f"{spec.name} narrows verbatim but its output {output_model.__name__} "
                "has no `notes` channel — the drop would be log-only"
            )
    assert seen, "no mapped stage found in any blueprint — the walk is vacuous"


class TestTypographicVariantsAreStillVerbatim:
    """A character the user cannot see must not fail their run.

    `normalize_for_match` folded case and whitespace only. An LLM asked to
    reproduce a phrase types it back with typographic punctuation — a straight
    apostrophe returns as U+2019, a hyphen as an en- or em-dash, a space as
    non-breaking. Containment then failed, `assert_verbatim_present` raised, and
    the whole run FAILED over a phrase that is verbatim by any human reading.

    This is the single matcher behind BOTH the gate and the read-time "placed"
    derivation, so widening it keeps those two in agreement by construction.
    """

    @pytest.mark.parametrize(
        ("required", "produced"),
        [("it's the best", "It\u2019s the best")],
    )
    def test_a_typographic_variant_is_found(self, required: str, produced: str) -> None:
        missing = find_missing_verbatim(f"prefix {produced} suffix", [{"text": required}])

        assert missing == [], (
            f"{required!r} is present as {produced!r} — failing the run over an "
            "invisible character is the defect"
        )

    def test_a_genuinely_absent_phrase_is_still_reported(self) -> None:
        """Widening must not become 'everything matches'."""
        missing = find_missing_verbatim("something else entirely", [{"text": "it's the best"}])

        assert missing == ["it's the best"]

    @pytest.mark.parametrize(
        ("required", "rendered"),
        [("5\u00b2 cases", "52 cases")],
    )
    def test_compatibility_equivalents_are_not_treated_as_the_same_phrase(
        self, required: str, rendered: str
    ) -> None:
        missing = find_missing_verbatim(f"prefix {rendered} suffix", [{"text": required}])

        assert missing == [required], (
            f"{rendered!r} is not {required!r} — accepting it is the gate failing OPEN"
        )

    def test_a_different_phrase_is_not_folded_into_a_match(self) -> None:
        missing = find_missing_verbatim("it\u2019s the worst", [{"text": "it's the best"}])

        assert missing == ["it's the best"]
