"""A band-only redesign inventory must not make an empty script the only compliant answer.

Prod round-3 (r3b-kill-capture, script-prompt-attempt1.txt): the redesign had no
section landmarks, so html_render sliced it into viewport bands ``band-01``/``band-02``.
The prompt listed the bare slugs, said "describe ONLY what is listed above", and
listed everything the page does NOT contain — the model answered ``{"lines": []}``.
"""

from __future__ import annotations

from src.services.stages.website_review.grounding import build_inventory

#: The archived run's two rendered stills, as html_render returns them.
_BAND_REDESIGN = {
    "visuals": [
        {
            "id": "redesign-section-band-01",
            "role": "redesign_section:band-01",
            "image_url": "https://minio.test/r/01_band-01.png",
        },
        {
            "id": "redesign-section-band-02",
            "role": "redesign_section:band-02",
            "image_url": "https://minio.test/r/02_band-02.png",
        },
    ]
}
#: A redesign with no ``data-section`` landmarks (why html_render fell back to bands).
_BAND_HTML = "<html><body><div><h1>Welkom</h1><p>Coaching.</p></div></body></html>"


def test_band_only_inventory_permits_general_visual_terms_not_silence() -> None:
    block = build_inventory(_BAND_REDESIGN, _BAND_HTML).prompt_block()

    assert "general visual terms (layout, hierarchy, typography, clarity)" in block
    assert "an empty script is never correct" in block
    assert "redesign-section-band-01: screen 1 of the page (no named landmarks)" in block
    assert "redesign-section-band-02: screen 2 of the page (no named landmarks)" in block
    # the bare-slug-only form and the "say only the list" rule are gone
    assert "redesign-section-band-01: band-01\n" not in block
    assert "describe ONLY what is listed" not in block
    # the absent-feature list is real grounding and stays
    assert "The redesign does NOT contain:" in block
    assert "a form (contact/sign-up)" in block
    assert "never say security headers" in block


def test_band_only_inventory_without_html_is_grounded_the_same_way() -> None:
    block = build_inventory(_BAND_REDESIGN, None).prompt_block()

    assert "general visual terms" in block
    assert "screen 1 of the page (no named landmarks)" in block
    assert "does NOT contain" not in block


def test_named_section_page_prompt_is_unchanged() -> None:
    redesign = {
        "visuals": [
            {
                "id": "redesign-section-hero",
                "role": "redesign_section:hero",
                "image_url": "https://minio.test/r/01_hero.png",
            },
            {
                "id": "redesign-section-cta",
                "role": "redesign_section:cta",
                "image_url": "https://minio.test/r/02_cta.png",
            },
        ]
    }
    html = (
        '<html><body><section data-section="hero"><h1>Leiders die bewegen.</h1>'
        '<a class="btn" href="/c">Plan een gesprek</a></section>'
        '<section data-section="cta"><h2>Klaar?</h2><form><input type="email"></form>'
        "</section></body></html>"
    )
    block = build_inventory(redesign, html).prompt_block()

    assert block == (
        "REDESIGN CONTENT — the redesigned page consists of EXACTLY these rendered "
        "sections, top to bottom (this is everything the viewer will see):\n"
        '  1. section_id redesign-section-hero: hero — heading "Leiders die bewegen." '
        '— contains button "Plan een gesprek"\n'
        '  2. section_id redesign-section-cta: cta — heading "Klaar?" — contains form, '
        "form fields\n"
        "The redesign does NOT contain: a booking/scheduling widget, a chat widget, "
        "a cookie banner, a search field, a skip-to-content link, an embedded map, "
        "an embedded video, breadcrumb navigation.\n"
        "GROUNDING RULES: describe ONLY what is listed above. Never claim the "
        "redesign has a feature, element or fix that is not listed — a critique "
        "RECOMMENDATION is not something the redesign did. The redesign is a static "
        "page design: it cannot fix server settings, so never say security headers, "
        "HTTPS/TLS or other server configuration are added, fixed or live.\n"
    )


def test_named_sections_without_readable_html_keep_the_strict_rule() -> None:
    redesign = {
        "visuals": [
            {
                "id": f"redesign-section-{name}",
                "role": f"redesign_section:{name}",
                "image_url": f"https://minio.test/r/0{i}_{name}.png",
            }
            for i, name in enumerate(("hero", "diensten"), start=1)
        ]
    }
    block = build_inventory(redesign, None).prompt_block()

    assert "redesign-section-hero: hero\n" in block
    assert "describe ONLY what is listed" in block
    assert "general visual terms" not in block
