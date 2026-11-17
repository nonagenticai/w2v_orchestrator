"""Duplicate rendered-section ids must not collapse a section out of the inventory.

``RedesignInventory.section_ids`` is a frozenset: two visuals sharing an id made a
second section indistinguishable from the first, so the script could never ground
a line on it (its copy was silently lost). ``build_inventory`` now suffixes repeats
``-2``, ``-3`` deterministically, in page order, and logs a warning.
"""

from __future__ import annotations

import logging

import pytest

from src.services.stages.website_review.grounding import build_inventory


def _visual(vid: str, name: str, order: int) -> dict[str, str]:
    return {
        "id": vid,
        "role": f"redesign_section:{name}",
        "image_url": f"https://minio.test/r/{order:02d}_{name}.png",
    }


_DUP = {
    "visuals": [
        _visual("sec", "hero", 1),
        _visual("sec", "diensten", 2),
        _visual("cta", "cta", 3),
        _visual("sec", "footer", 4),
    ]
}
_HTML = (
    '<section data-section="hero"><h1>Welkom</h1></section>'
    '<section data-section="diensten"><h2>Diensten</h2></section>'
    '<section data-section="footer"><h2>Contact</h2></section>'
)


@pytest.mark.parametrize("html", [None], ids=["no-html"])
def test_duplicate_ids_are_suffixed_not_collapsed(
    html: str | None, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        inventory = build_inventory(_DUP, html)
    assert [(s.id, s.name) for s in inventory.sections] == [
        ("sec", "hero"),
        ("sec-2", "diensten"),
        ("cta", "cta"),
        ("sec-3", "footer"),
    ]
    assert len(inventory.section_ids) == len(inventory.sections) == 4
    assert "sec" in caplog.text and "duplicate" in caplog.text.lower()


def test_suffix_never_collides_with_a_real_id(caplog: pytest.LogCaptureFixture) -> None:
    redesign = {
        "visuals": [
            _visual("sec", "hero", 1),
            _visual("sec-2", "diensten", 2),
            _visual("sec", "footer", 3),
        ]
    }
    ids = [s.id for s in build_inventory(redesign, None).sections]
    assert len(set(ids)) == 3
    assert ids[:2] == ["sec", "sec-2"]


def test_unique_ids_are_untouched_and_silent(caplog: pytest.LogCaptureFixture) -> None:
    redesign = {"visuals": [_visual("a", "hero", 1), _visual("b", "cta", 2)]}
    with caplog.at_level(logging.WARNING):
        ids = [s.id for s in build_inventory(redesign, None).sections]
    assert ids == ["a", "b"]
    assert not caplog.records


def test_rename_warning_shape_and_visual_back_reference(caplog: pytest.LogCaptureFixture) -> None:
    from src.services.stages.website_review.grounding import rendered_section_visuals

    with caplog.at_level(logging.WARNING):
        build_inventory(_DUP, None)
    assert [r.getMessage() for r in caplog.records] == [
        "website_review grounding: duplicate redesign section id 'sec' (section 'diensten') "
        "renamed to 'sec-2'",
        "website_review grounding: duplicate redesign section id 'sec' (section 'footer') "
        "renamed to 'sec-3'",
    ]
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        pairs = rendered_section_visuals(_DUP)
    # Same ids as the inventory, each pointing at the still it was renamed from;
    # silent, so the rename is warned about exactly once per run.
    assert [(uid, v["id"], v["image_url"].rsplit("/", 1)[-1]) for uid, v in pairs] == [
        ("sec", "sec", "01_hero.png"),
        ("sec-2", "sec", "02_diensten.png"),
        ("cta", "cta", "03_cta.png"),
        ("sec-3", "sec", "04_footer.png"),
    ]
    assert not caplog.records
