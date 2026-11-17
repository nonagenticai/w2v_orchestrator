"""Tests for website_review_visual_director — AFTER-act line→redesign-section mapping."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from src.services.stages.website_review.grounding import rendered_section_visuals
from src.services.stages.website_review.visual_director import (
    WebsiteReviewVisualDirectorInput,
    WebsiteReviewVisualDirectorStage,
)


def _ctx(script: dict, redesign: dict) -> MagicMock:
    ctx = MagicMock()
    ctx.prior_output.side_effect = lambda key: {
        "website_review_script": script,
        "website_redesign": redesign,
    }.get(key)
    return ctx


def _redesign(*sections: tuple[str, str]) -> dict:
    """Build a redesign output with redesign_section:* visuals (id, label)."""
    visuals = [
        {
            "id": sid,
            "label": label,
            "role": f"redesign_section:{sid}",
            "image_url": f"https://x/{sid}.png",
        }
        for sid, label in sections
    ]
    # a non-section visual must be ignored by the director
    visuals.append(
        {
            "id": "fp",
            "label": "full page",
            "role": "redesign_desktop",
            "image_url": "https://x/fp.png",
        }
    )
    return {"visuals": visuals}


def test_after_lines_filters_before_act() -> None:
    script = {
        "lines": [
            {"id": "b1", "text": "old hero is weak", "phase": "before", "speaker": "a"},
            {"id": "a1", "text": "the new hero", "phase": "after", "speaker": "b"},
        ]
    }
    out = WebsiteReviewVisualDirectorStage._after_lines(_ctx(script, {}))
    assert out == [{"line_id": "a1", "text": "the new hero", "speaker": "b", "section_id": ""}]


def test_sections_keeps_only_section_role_visuals() -> None:
    sections = WebsiteReviewVisualDirectorStage._sections(
        _ctx({}, _redesign(("hero", "Hero"), ("pricing", "Pricing")))
    )
    assert sections == [
        {"id": "hero", "label": "Hero"},
        {"id": "pricing", "label": "Pricing"},
    ]


async def test_handle_maps_after_lines_to_sections() -> None:
    script = {
        "lines": [
            {"id": "b1", "text": "old hero", "phase": "before", "speaker": "a"},
            {"id": "a1", "text": "here's the new hero", "phase": "after", "speaker": "b"},
            {"id": "a2", "text": "and the pricing", "phase": "after", "speaker": "b"},
        ]
    }
    redesign = _redesign(("hero", "Hero"), ("pricing", "Pricing"))
    fake = MagicMock()
    fake.message = AsyncMock(
        return_value=SimpleNamespace(
            output={
                "assignments": [
                    {"line_id": "a1", "section_id": "hero"},
                    {"line_id": "a2", "section_id": "pricing"},
                ]
            }
        )
    )
    stage = WebsiteReviewVisualDirectorStage(client=fake)
    out = await stage.handle(WebsiteReviewVisualDirectorInput(), _ctx(script, redesign))
    by_line = {a.line_id: a.section_id for a in out.assignments}
    # only after lines are mapped; before line is absent
    assert by_line == {"a1": "hero", "a2": "pricing"}


async def test_handle_no_sections_returns_empty() -> None:
    script = {"lines": [{"id": "a1", "text": "x", "phase": "after", "speaker": "b"}]}
    stage = WebsiteReviewVisualDirectorStage(client=MagicMock())
    out = await stage.handle(WebsiteReviewVisualDirectorInput(), _ctx(script, {"visuals": []}))
    assert out.assignments == []
    assert out.notes


async def test_handle_llm_failure_spreads_across_sections() -> None:
    script = {
        "lines": [
            {"id": "q1", "text": "show me the redesign?", "phase": "after", "speaker": "a"},
            {"id": "a1", "text": "the hero", "phase": "after", "speaker": "b"},
            {"id": "a2", "text": "the features", "phase": "after", "speaker": "b"},
        ]
    }
    redesign = _redesign(("hero", "Hero"), ("features", "Features"))
    fake = MagicMock()
    fake.message = AsyncMock(side_effect=RuntimeError("LLM down"))
    stage = WebsiteReviewVisualDirectorStage(client=fake)
    out = await stage.handle(WebsiteReviewVisualDirectorInput(), _ctx(script, redesign))
    by_line = {a.line_id: a.section_id for a in out.assignments}
    # soft-fail: with no LLM map the anti-collapse net walks the two answers across
    # the two sections (top→bottom); the opening question holds the first section.
    assert by_line == {"q1": "hero", "a1": "hero", "a2": "features"}
    assert out.notes


async def test_handle_spreads_when_llm_collapses_onto_one_section() -> None:
    # The observed failure: the LLM piles every after line onto a single section,
    # so the after act would hold one still. The director must spread them out.
    script = {
        "lines": [
            {"id": "a1", "text": "take a look at the redesign", "phase": "after", "speaker": "b"},
            {"id": "a2", "text": "the chat widget works now", "phase": "after", "speaker": "b"},
            {"id": "a3", "text": "the project cards are cleaner", "phase": "after", "speaker": "b"},
            {
                "id": "a4",
                "text": "and a search bar in the header",
                "phase": "after",
                "speaker": "b",
            },
        ]
    }
    redesign = _redesign(
        ("hero", "Hero"), ("projects", "Projects"), ("contact", "Contact"), ("footer", "Footer")
    )
    fake = MagicMock()
    fake.message = AsyncMock(
        return_value=SimpleNamespace(
            output={
                "assignments": [
                    {"line_id": lid, "section_id": "projects"} for lid in ("a1", "a2", "a3", "a4")
                ]
            }
        )
    )
    stage = WebsiteReviewVisualDirectorStage(client=fake)
    out = await stage.handle(WebsiteReviewVisualDirectorInput(), _ctx(script, redesign))
    by_line = {a.line_id: a.section_id for a in out.assignments}
    # Collapsed onto "projects" → spread across all four sections top→bottom.
    assert by_line == {"a1": "hero", "a2": "projects", "a3": "contact", "a4": "footer"}
    assert len(set(by_line.values())) == 4


# --- Duplicate section ids (grounding renames repeats ``-2``, ``-3``) -------------
# The script grounds after-act lines on the INVENTORY's ids, where a repeated visual
# id is renamed (``sec`` -> ``sec-2``). The director emits those ids AS IS — the
# rename is the only handle telling two ``sec`` sections apart — and compile keys
# its stills by the same inventory ids (grounding.rendered_section_visuals).
def _dup_redesign() -> dict:
    def v(vid: str, name: str, order: int) -> dict[str, str]:
        return {
            "id": vid,
            "label": name.title(),
            "role": f"redesign_section:{name}",
            "image_url": f"https://minio.test/r/{order:02d}_{name}.png",
        }

    return {"visuals": [v("cta", "cta", 3), v("sec", "hero", 1), v("sec", "diensten", 2)]}


def _compile_section_ids(redesign: dict) -> set[str]:
    """The ids compile.build_shots can resolve (its ``section_by_id`` keys)."""
    return {uid for uid, _ in rendered_section_visuals(redesign)}


def test_sections_use_the_inventory_ids_in_page_order() -> None:
    sections = WebsiteReviewVisualDirectorStage._sections(_ctx({}, _dup_redesign()))
    assert [s["id"] for s in sections] == ["sec", "sec-2", "cta"]


async def test_authored_renamed_section_is_emitted_as_is() -> None:
    script = {
        "lines": [
            {"id": "a1", "text": "new hero", "phase": "after", "speaker": "b", "section_id": "sec"},
            {
                "id": "a2",
                "text": "the services",
                "phase": "after",
                "speaker": "b",
                "section_id": "sec-2",
            },
            {
                "id": "a3",
                "text": "and the cta",
                "phase": "after",
                "speaker": "b",
                "section_id": "cta",
            },
        ]
    }
    redesign = _dup_redesign()
    fake = MagicMock()
    fake.message = AsyncMock()
    out = await WebsiteReviewVisualDirectorStage(client=fake).handle(
        WebsiteReviewVisualDirectorInput(), _ctx(script, redesign)
    )
    # Every line was authored on a rendered section: no LLM re-guess.
    fake.message.assert_not_called()
    got = {a.line_id: a.section_id for a in out.assignments}
    assert got == {"a1": "sec", "a2": "sec-2", "a3": "cta"}
    assert set(got.values()) <= _compile_section_ids(redesign)


async def test_llm_picked_renamed_section_is_emitted_as_is() -> None:
    script = {
        "lines": [
            {"id": "a1", "text": "the services", "phase": "after", "speaker": "b"},
        ]
    }
    redesign = _dup_redesign()
    fake = MagicMock()
    fake.message = AsyncMock(
        return_value=SimpleNamespace(
            output={"assignments": [{"line_id": "a1", "section_id": "sec-2"}]}
        )
    )
    out = await WebsiteReviewVisualDirectorStage(client=fake).handle(
        WebsiteReviewVisualDirectorInput(), _ctx(script, redesign)
    )
    assert [(a.line_id, a.section_id) for a in out.assignments] == [("a1", "sec-2")]
    assert {a.section_id for a in out.assignments} <= _compile_section_ids(redesign)
    prompt = fake.message.call_args.kwargs.get("message") or str(fake.message.call_args)
    assert "section_id sec-2" in prompt
