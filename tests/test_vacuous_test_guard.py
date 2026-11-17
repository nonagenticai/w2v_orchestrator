"""CI-enforced ratchet: no NEW test may be written without an assertion.

Two tests merged on 2026-08-06 were vacuous and were found only by hand-mutating
the production source — review did not catch them and neither did the gate. The
frontend half of this monorepo already guards this class (preloaded
``expect.hasAssertions()`` + a one-way MAX ratchet); the Python side had nothing.

Ruff does NOT cover this. Its ``PT`` (flake8-pytest-style) family governs the
*shape* of assertions — ``PT009`` unittest-style, ``PT011`` too-broad ``raises``,
``PT015`` always-false, ``PT018`` composite — and there is no rule anywhere in
ruff for "this test function contains zero assertions". So: a repo-local check.

Wiring follows ``tests/test_any_ratchet.py`` — the gate is a pytest test, so it
runs under the archetype's ``uv run pytest tests/`` on every PR. A pre-commit-only
hook would be invisible to CI.

Honest scope: this is STATIC. It proves an assertion is present, never that it
ran or that it means anything. It kills the "N passed, 0 assertions" shape and
ratchets the existing 14 down; it does not catch a weak assertion. That needs
mutation testing.
"""

from __future__ import annotations

from scripts.check_vacuous_tests import (
    ALLOWLIST_PATH,
    VacuousTest,
    find_vacuous_tests,
    has_assertion,
    load_allowlist,
)

# ── Ratchet — LOWER ONLY, never raise ────────────────────────────────────────
# Baseline measured 2026-08-08 across 4408 collected `test_*` functions: 14
# assertion-free, ALL of them the same "calling this must not raise" shape.
# Fixing one means deleting its line from tests/vacuous_test_allowlist.txt and
# lowering this number. `test_ratchet_is_not_stale` forces the two to stay in
# lockstep, so the number can only go down.
MAX_VACUOUS_TESTS = 0


def test_no_new_vacuous_tests() -> None:
    """Every assertion-free ``test_*`` must be a grandfathered allowlist entry."""
    allowed = load_allowlist()
    offenders: list[VacuousTest] = find_vacuous_tests()
    new = [o for o in offenders if o.nodeid not in allowed]
    assert not new, (
        "These test functions contain no assertion — they pass no matter what the "
        "production code does:\n"
        + "\n".join(f"  {o.path}:{o.lineno}  {o.name}" for o in new)
        + "\n\nAssert something observable. Do NOT add them to "
        f"{ALLOWLIST_PATH.name} — that list is a one-way ratchet for pre-existing "
        "debt only."
    )


def test_allowlist_has_no_stale_entries() -> None:
    """A fixed or renamed test must be removed from the allowlist.

    Without this, the allowlist silently accumulates dead entries and stops
    describing reality — and a renamed vacuous test would be caught by
    ``test_no_new_vacuous_tests`` while its old exemption lingered as cover.
    """
    live = {o.nodeid for o in find_vacuous_tests()}
    stale = sorted(load_allowlist() - live)
    assert not stale, (
        "Allowlist entries no longer match any assertion-free test (fixed or renamed) — "
        f"delete them from {ALLOWLIST_PATH.name} and lower MAX_VACUOUS_TESTS:\n"
        + "\n".join(f"  {nodeid}" for nodeid in stale)
    )


def test_ratchet_is_not_stale() -> None:
    """``MAX_VACUOUS_TESTS`` must equal the allowlist size — the one-way lock.

    Equality (not ``<=``) is what makes this a ratchet rather than a budget: you
    cannot bank headroom by fixing a test without lowering the number, and you
    cannot grow the list without raising a number that review will see.
    """
    count = len(load_allowlist())
    assert count == MAX_VACUOUS_TESTS, (
        f"allowlist holds {count} entries but MAX_VACUOUS_TESTS is {MAX_VACUOUS_TESTS}. "
        "Fixed a test? Lower MAX_VACUOUS_TESTS. Raising it is not an option."
    )


def test_detector_flags_a_vacuous_function() -> None:
    """The detector itself must be able to fail — mutation-proof of the guard."""
    import ast

    vacuous = ast.parse("def test_x() -> None:\n    do_something()\n").body[0]
    assert isinstance(vacuous, ast.FunctionDef)
    assert not has_assertion(vacuous)


def test_detector_accepts_each_assertion_form() -> None:
    """Each construct we treat as an assertion is actually recognised."""
    import ast

    forms = [
        "assert x == 1",
        "with pytest.raises(ValueError):\n        f()",
        "with pytest.warns(UserWarning):\n        f()",
        "mock.assert_called_once_with(1)",
        "mock.assert_awaited()",
        "self.assertEqual(a, b)",
        "_assert_panel_rendered(page)",
        "pytest.fail('nope')",
        "def _inner() -> None:\n        assert x\n    _inner()",
    ]
    for body in forms:
        node = ast.parse(f"def test_x() -> None:\n    {body}\n").body[0]
        assert isinstance(node, ast.FunctionDef)
        assert has_assertion(node), f"not recognised as an assertion: {body!r}"
