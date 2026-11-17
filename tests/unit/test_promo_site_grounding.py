"""``promo_script`` must not launder a failed site scrape into a confident script.

``site_capture._scrape_page_text`` now reports a ``page_text_status`` alongside
the text, because an empty ``page_text`` alone cannot distinguish three very
different facts:

* ``"blocked"`` / ``"fetch_failed"`` — we never read the site. Writing a promo
  anyway means the LLM invents features, metrics and integrations from the brief
  and its own priors, and the run reports ``completed``. That is the defect:
  ``PIPELINE_PROMO_REQUIRE_SITE_GROUNDING`` (default **on**) now fails it loud.
* ``"empty"`` — we DID read the site and it genuinely has no readable copy (a
  JS-only shell). Not a failure, but the absence must be **stated** in the
  prompt, or the model fills the silence exactly as above.
* ``"ok"`` — the grounded path, which must stay byte-identical to today's
  prompt; these tests pin that so the new branching cannot drift it.

The stage's ``build_prompt`` is called DIRECTLY (no LLM, no test client): the
prompt string is the artifact under test, and going through ``stage.run`` would
only re-assert what the existing ``tests/services/stages/promo/test_script.py``
already covers.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.config import PipelineSettings
from src.services.runs.run_context import RunContext
from src.services.stages.errors import StagePreconditionError
from src.services.stages.promo import script as promo_script
from src.services.stages.promo.script import PromoScriptStage

_PAGE_TEXT = "Acme ships anvils in 24 hours. 4,000 workshops rely on it."
_SOURCE_URL = "https://acme.example/product"

# The grounded prompt block as it reads TODAY, byte-identical. Spelled out here
# rather than imported so a silent edit to the wording is a failing test, not a
# tautology that follows the source.
_GROUNDED_BLOCK = (
    "\n\nGround the script in this copy from the live product site "
    "(use it for accurate features/value/problems, do not quote verbatim):\n"
    f"{_PAGE_TEXT}"
)
_NO_COPY_CLAUSE = "Do NOT invent product features"


def _settings(*, require_site_grounding: bool) -> SimpleNamespace:
    """A ``get_settings()`` stand-in exposing a real ``PipelineSettings``."""
    return SimpleNamespace(
        pipeline=PipelineSettings(promo_require_site_grounding=require_site_grounding)
    )


def _gate(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> None:
    monkeypatch.setattr(
        promo_script, "get_settings", lambda: _settings(require_site_grounding=enabled)
    )


def _ctx(*, page_text: str, page_text_status: str) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "title": "Anvils, overnight",
            "brief": {
                "call_to_action": "Order an anvil.",
                "target_duration_seconds": 30,
                "value_proposition": "anvils, fast",
                "source_url": _SOURCE_URL,
            },
            "voice": {"descriptor": "crisp"},
        },
        prior_run_outputs={
            "promo_analyze": {
                "pain_points": [{"id": "p1", "text": "slow delivery"}],
                "hook_candidates": [{"id": "h1", "text": "Anvils by morning."}],
                "tone_hints": ["urgent"],
            },
            "site_capture": {
                "clips": [],
                "page_text": page_text,
                "page_text_status": page_text_status,
                "notes": [],
            },
        },
    )


# --- 1. a scrape that never read the site fails loud when grounding is required


@pytest.mark.parametrize("status", ["fetch_failed", "blocked"])
def test_failed_scrape_fails_loud_when_grounding_required(
    monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    """The defect: this used to produce a green run holding an invented script."""
    _gate(monkeypatch, enabled=True)
    stage = PromoScriptStage()  # no LLM: build_prompt is called directly

    with pytest.raises(StagePreconditionError) as exc:
        stage.build_prompt(_ctx(page_text="", page_text_status=status))

    # The operator must be able to act on it: which status, which target, and the
    # exact switch that permits an ungrounded script anyway.
    reason = exc.value.reason
    assert status in reason, f"reason does not name the status: {reason!r}"
    assert _SOURCE_URL in reason, f"reason does not name the scrape target: {reason!r}"
    assert "PIPELINE_PROMO_REQUIRE_SITE_GROUNDING" in reason


# --- 2. "empty" is not a failure — but the absence must be STATED -------------


def test_empty_scrape_does_not_raise_and_states_the_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A site that really has no copy still gets a script — an HONEST one.

    Silence here is the bug: with no site block the prompt looks identical to a
    run that never attempted a scrape, and the model confabulates the product.
    """
    _gate(monkeypatch, enabled=True)
    stage = PromoScriptStage()  # no LLM: build_prompt is called directly

    prompt = stage.build_prompt(_ctx(page_text="", page_text_status="empty"))

    assert _NO_COPY_CLAUSE in prompt, "the empty scrape was passed over in silence"
    assert "No copy could be read from the live product site." in prompt
    # It must not also claim to be grounded.
    assert "Ground the script in this copy from the live product site" not in prompt


# --- 3. the grounded path is unchanged ---------------------------------------


def test_ok_status_emits_todays_grounding_block_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The false-positive control: the new branching must not touch the happy path."""
    _gate(monkeypatch, enabled=True)
    stage = PromoScriptStage()  # no LLM: build_prompt is called directly

    prompt = stage.build_prompt(_ctx(page_text=_PAGE_TEXT, page_text_status="ok"))

    assert _GROUNDED_BLOCK in prompt
    # A grounded run must NOT also carry the "we read nothing" disclaimer.
    assert _NO_COPY_CLAUSE not in prompt


# --- 4. the gate is a gate: off restores the old (silent) behaviour -----------


def test_failed_scrape_is_permitted_when_the_gate_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``PIPELINE_PROMO_REQUIRE_SITE_GROUNDING=false`` is the documented opt-out.

    Asserting only "it did not raise" would pass on a stage that never had a gate
    at all, so this also pins WHAT it produces: an ungrounded prompt carrying
    neither the grounding block nor the empty-scrape disclaimer — the pre-fix
    behaviour, now reachable only by explicit operator choice.
    """
    _gate(monkeypatch, enabled=False)
    stage = PromoScriptStage()  # no LLM: build_prompt is called directly

    prompt = stage.build_prompt(_ctx(page_text="", page_text_status="fetch_failed"))

    assert "Ground the script in this copy from the live product site" not in prompt
    assert _NO_COPY_CLAUSE not in prompt
    # It is still a usable promo prompt, not an empty string.
    assert "Order an anvil." in prompt


# --- 5. legacy rows: copy present but no status ------------------------------


def test_legacy_capture_without_a_status_still_grounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``site_capture`` row written BEFORE ``page_text_status`` existed must
    keep its grounding.

    Every such row deserialises with the ``"not_attempted"`` default while
    carrying perfectly good ``page_text``. An earlier draft of this fix keyed the
    grounded branch on ``status == "ok"``, which silently dropped the site copy
    for all of them — the exact class of regression this file exists to prevent,
    and a strictly worse one, because it degrades runs that used to work. Copy
    that is present is always groundable; the status only discriminates *absence*.
    """
    _gate(monkeypatch, enabled=True)
    stage = PromoScriptStage()

    prompt = stage.build_prompt(_ctx(page_text=_PAGE_TEXT, page_text_status="not_attempted"))

    assert _GROUNDED_BLOCK in prompt
    assert _NO_COPY_CLAUSE not in prompt


# --- 6. thin copy: a fragment is not grounding -------------------------------

_THIN_CAUTION = "That is the ONLY copy that could be read from the live site"


def test_thin_scrape_keeps_the_fragment_but_caps_what_may_be_built_on_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``"empty"`` fix alone does not close the hole it was written for.

    ``_extract_text`` strips tags across the whole document, so a JS-only SPA
    still yields its ``<title>`` and ``<noscript>`` line — measured against real
    shells on 2026-08-12, web.telegram.org gives 12 characters, excalidraw.com
    79, tldraw.com 136. All three used to be classified ``"ok"`` and handed to
    the model as "copy from the live product site", so the empty branch never
    fired and the model confabulated exactly as before.

    ``"thin"`` keeps the fragment (it is real, and useful for naming and tone)
    but must append the ceiling; dropping either half re-opens the defect.
    """
    _gate(monkeypatch, enabled=True)
    fragment = "Telegram Web"
    stage = PromoScriptStage()

    prompt = stage.build_prompt(_ctx(page_text=fragment, page_text_status="thin"))

    assert fragment in prompt, "the fragment is real copy; do not discard it"
    assert _THIN_CAUTION in prompt, "a fragment presented as the page is the falsehood"
    assert "Do NOT invent product features" in prompt


def test_thin_scrape_degrades_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """A thin-grounded script must not read as a clean one.

    Non-empty ``notes`` is what ``runs/_shared.py`` turns into ``degraded``, so
    this is the whole visibility mechanism — without it the run is green.
    """
    _gate(monkeypatch, enabled=True)
    stage = PromoScriptStage()
    ctx = _ctx(page_text="Telegram Web", page_text_status="thin")

    out = stage.build_output([], ctx)

    assert out.notes, "a thin-grounded run must surface as degraded"
    assert any("thin" in n for n in out.notes), out.notes


def test_ok_status_is_not_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard against over-fixing: real copy must keep today's clean prompt."""
    _gate(monkeypatch, enabled=True)
    stage = PromoScriptStage()

    ctx = _ctx(page_text=_PAGE_TEXT, page_text_status="ok")
    prompt = stage.build_prompt(ctx)

    assert _GROUNDED_BLOCK in prompt
    assert _THIN_CAUTION not in prompt
    assert not stage.build_output([], ctx).notes
