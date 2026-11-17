"""``uncovered_steps``: the recorded action steps a beat does not narrate yet.

Pinned on showcase run 01a158a5-e826 (2026-10-20): clip-4 recorded "Add a todo and
mark it completed" while its narration said only "add a second task", and the
beat-expander, told to elaborate on the beat, never narrated the completion.
"""

from __future__ import annotations

from src.services.stages.shared.beat_grounding import uncovered_steps

_CLIP4_GHERKIN = (
    "Scenario: Add a todo and mark it completed\n"
    '  Given I open the React TodoMVC demo at "https://demo.playwright.dev/todomvc"\n'
    '  And I see the heading "todos"\n'
    '  And I see the text "This is just a demo of TodoMVC for testing, not the real '
    'TodoMVC app."\n'
    "  Given I am on the todos page\n"
    '  When I type "Send email" into the input with placeholder "What needs to be done?"\n'
    '  And I press the "Enter" key\n'
    '  Then I see the text "Send email" on the page\n'
    '  When I click the checkbox next to "Send email"\n'
    '  Then the text "Send email" is styled as completed'
)
_CLIP5_GHERKIN = (
    "Scenario: Verify demo disclaimer and links\n"
    '  Given I open the React TodoMVC demo at "https://demo.playwright.dev/todomvc"\n'
    "  Given I am on the todos page\n"
    '  Then I see the link "real TodoMVC app."\n'
    '  And I see the link "Remo H. Jansen"\n'
    '  And I see the link "TodoMVC"'
)


def test_the_live_beat_never_narrated_its_completion_step() -> None:
    text = 'Return to the input field and enter "Send email" to add a second task.'
    steps = uncovered_steps(text, _CLIP4_GHERKIN)
    assert 'click the checkbox next to "Send email"' in steps
    assert 'the text "Send email" is styled as completed' in steps
    # The shared setup block is never offered: narrating it on every beat is the
    # "Let's start by navigating to the todos page" repetition.
    assert not any("open the React" in s or "heading" in s for s in steps)
    # No URL survives into narration material.
    assert not any("http" in s for s in steps)


def test_a_step_the_beat_already_narrates_is_not_offered() -> None:
    text = (
        'Type "Send email" into the input, press Enter, click the checkbox next to it, '
        "and the text is styled as completed on the page."
    )
    assert uncovered_steps(text, _CLIP4_GHERKIN) == []


def test_a_quoted_object_must_be_named_for_the_step_to_count_as_narrated() -> None:
    """'see the link X' is not covered by a beat that names a DIFFERENT link."""
    text = 'Verify the "real TodoMVC app" link is clearly visible on the screen.'
    assert uncovered_steps(text, _CLIP5_GHERKIN) == ['see the link "Remo H. Jansen"']


def test_no_gherkin_means_no_steps() -> None:
    assert uncovered_steps("anything", None) == []
    assert uncovered_steps("anything", "") == []


def test_a_setup_only_scenario_falls_back_to_every_step() -> None:
    gherkin = 'Scenario: x\n  Given I open the settings page\n  And I see the heading "Billing"'
    assert uncovered_steps("Nothing relevant.", gherkin) == [
        "open the settings page",
        'see the heading "Billing"',
    ]
