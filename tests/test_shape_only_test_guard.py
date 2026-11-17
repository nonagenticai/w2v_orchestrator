"""CI-enforced ratchet: no NEW test may assert only the IMPLEMENTATION's shape.

``tests/test_vacuous_test_guard.py`` catches a test with NO assertion. This is
the next rung up, and it is the one this fleet actually keeps writing: a test
whose only assertions are call counts and call positions. Such a test breaks on
any CORRECT change and passes on any WRONG one that preserves the shape — worse
than no test, because it looks like coverage.

The evidence is one afternoon. On 2026-08-24/25 ``w2v-common`` 0.65.0 made
``_create_or_check_database`` issue one extra, entirely correct statement
(``REVOKE CONNECT ON DATABASE <db> FROM PUBLIC`` — the born-LOCKED provisioning
policy). It reddened FIVE tests across FOUR sibling repos:

===============================  ====================================================
repo / fix commit                what the test actually asserted
===============================  ====================================================
analysis_service ``05cdd8a``     ``execute.call_count == 2`` (two tests)
app_inspector_service ``212a388``  ``execute.call_count == 2``
ai_content_generation ``d85f85f``  ``call_count == 2`` + ``call_args_list[1]`` → ``3 == 2``
tts_service ``0676f77``          ``side_effect=[mock_result, None]`` → StopIteration
                                 inside a coroutine → RuntimeError; asserted nothing
===============================  ====================================================

Not one of them asserted anything about the database being created.

Wiring follows ``tests/test_any_ratchet.py`` and ``tests/test_vacuous_test_guard.py``:
the gate is a pytest test, so it runs under the archetype's ``uv run pytest tests/``
on every PR. A pre-commit-only hook is invisible to CI.

Honest scope: STATIC. It cannot see a test that asserts a WRONG value
(``video_compiler_service::test_parse_loudness_handles_inf_silence`` asserted
``== -math.inf``, the value postgres rejects) or an inverted contract
(``assert "x-tenant-id" not in headers``, which a later security fix made false).
Both of those are structurally fine and semantically wrong; catching them needs
mutation testing. ``test_out_of_scope_defects_are_not_false_positived`` pins
that this checker does not pretend otherwise.
"""

from __future__ import annotations

import ast

from scripts.check_vacuous_tests import (
    SHAPE_ALLOWLIST_PATH,
    ShapeOnlyTest,
    classify_assertions,
    find_shape_only_tests,
    has_assertion,
    has_fixed_length_side_effect,
    load_allowlist,
    pins_exact_call_count,
    shape_only_reason,
)

# ── Ratchet — LOWER ONLY, never raise ────────────────────────────────────────
# Baseline measured 2026-08-25 over this repo's tests/: 16 shape-only functions.
# Fleet context, same detector, same day: video_production_service 49,
# analysis_service 51, content_service 16, ai_content_generation 10,
# tts_service 10, bdd_playwright 10, html_render 6, app_inspector 5,
# w2v_common 3, youtube_upload 3, video_analyzer 1, video_compiler 1,
# frontend_test_runner 1, tts_gateway 1, slide_service 0, image_gen_backend 0.
# Fixing one here means deleting its line from tests/shape_only_test_allowlist.txt
# and lowering this number; ``test_shape_ratchet_is_not_stale`` keeps the two in
# lockstep, so it can only go down.
MAX_SHAPE_ONLY_TESTS = 6


def _fn(src: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    node = ast.parse(src).body[0]
    assert isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    return node


# ═════════════════════════════════════════════════════════════════════════════
# The gate
# ═════════════════════════════════════════════════════════════════════════════


def test_no_new_shape_only_tests() -> None:
    """Every shape-only ``test_*`` must be a grandfathered allowlist entry."""
    allowed = load_allowlist(SHAPE_ALLOWLIST_PATH)
    offenders: list[ShapeOnlyTest] = find_shape_only_tests()
    new = [o for o in offenders if o.nodeid not in allowed]
    assert not new, (
        "These tests assert only the SHAPE of the implementation (how many calls, "
        "in what position) and nothing about what it did. They break on any correct "
        "change and pass on any wrong one that keeps the shape:\n"
        + "\n".join(f"  {o.path}:{o.lineno}  {o.name}\n      {o.reason}" for o in new)
        + "\n\nAssert an observable property instead. The canonical fix for the "
        "mocked-statement-sequence case is a side_effect FUNCTION that RECORDS what "
        'was executed, then `assert any("CREATE DATABASE ..." in s for s in executed)` '
        "(tts_service 0676f77). Do NOT add them to "
        f"{SHAPE_ALLOWLIST_PATH.name} — that list is a one-way ratchet for "
        "pre-existing debt only."
    )


def test_shape_allowlist_has_no_stale_entries() -> None:
    """A fixed or renamed test must be removed from the allowlist."""
    live = {o.nodeid for o in find_shape_only_tests()}
    stale = sorted(load_allowlist(SHAPE_ALLOWLIST_PATH) - live)
    assert not stale, (
        "Allowlist entries no longer match any shape-only test (fixed or renamed) — "
        f"delete them from {SHAPE_ALLOWLIST_PATH.name} and lower "
        "MAX_SHAPE_ONLY_TESTS:\n" + "\n".join(f"  {nodeid}" for nodeid in stale)
    )


def test_shape_ratchet_is_not_stale() -> None:
    """``MAX_SHAPE_ONLY_TESTS`` must equal the allowlist size — the one-way lock."""
    count = len(load_allowlist(SHAPE_ALLOWLIST_PATH))
    assert count == MAX_SHAPE_ONLY_TESTS, (
        f"allowlist holds {count} entries but MAX_SHAPE_ONLY_TESTS is "
        f"{MAX_SHAPE_ONLY_TESTS}. Fixed a test? Lower it. Raising it is not an option."
    )


# ═════════════════════════════════════════════════════════════════════════════
# The detector's own tests — the five real regressions, vendored verbatim
# ═════════════════════════════════════════════════════════════════════════════
#
# Vendored as source strings rather than read out of the sibling repos' git
# history: a test that shells out to four other checkouts is a test that is
# green because it silently skipped. Each pair is the BEFORE and AFTER of the
# real fix commit named above, trimmed to the assertion surface (the mock
# plumbing that does not affect classification is elided).

_ANALYSIS_BEFORE = """
async def test_creates_db_when_not_exists(self):
    mock_conn = MagicMock()
    check_result = MagicMock()
    check_result.scalar.return_value = None
    create_result = MagicMock()
    mock_conn.execute = MagicMock(side_effect=[check_result, create_result])
    await conn_mod._ensure_database_exists(max_retries=1)
    assert mock_conn.execute.call_count == 2
"""

_ANALYSIS_AFTER = """
async def test_creates_db_when_not_exists(self):
    executed: list[str] = []

    def _execute(statement, *_args, **_kwargs):
        executed.append(str(statement))
        return check_result if len(executed) == 1 else None

    mock_conn.execute = MagicMock(side_effect=_execute)
    await conn_mod._ensure_database_exists(max_retries=1)
    assert executed, "no statement was executed at all"
    assert "pg_database" in executed[0], executed
    assert any("CREATE DATABASE new_db" in stmt for stmt in executed), executed
"""

_ANALYSIS_BRAND_BEFORE = """
async def test_creates_database(self) -> None:
    mock_conn = MagicMock()
    check_result = MagicMock()
    check_result.scalar.return_value = None
    mock_conn.execute.side_effect = [check_result, None]
    await _ensure_database_exists(max_retries=1)
    assert mock_conn.execute.call_count == 2
    mock_engine.dispose.assert_called_once()
"""

_APP_INSPECTOR_BEFORE = """
async def test_database_created(self):
    mock_conn = MagicMock()
    check_result = MagicMock()
    check_result.scalar.return_value = None
    mock_conn.execute = MagicMock(side_effect=[check_result, None])
    await _ensure_database_exists(max_retries=1)
    assert mock_conn.execute.call_count == 2
"""

_AI_CONTENT_BEFORE = """
async def test_database_created_successfully(self) -> None:
    mock_conn = MagicMock()
    mock_result = MagicMock()
    mock_result.scalar.return_value = None
    mock_conn.execute.return_value = mock_result
    await conn_mod._ensure_database_exists(max_retries=1)
    assert mock_conn.execute.call_count == 2
    select_sql = str(mock_conn.execute.call_args_list[0][0][0])
    assert "pg_database" in select_sql
    create_sql = str(mock_conn.execute.call_args_list[1][0][0])
    assert "CREATE DATABASE new_db" in create_sql
    mock_engine.dispose.assert_called_once()
"""

_AI_CONTENT_AFTER = """
async def test_database_created_successfully(self) -> None:
    executed: list[str] = []

    def _execute(statement, *_args, **_kwargs):
        executed.append(str(statement))
        return mock_result if len(executed) == 1 else None

    mock_conn.execute.side_effect = _execute
    await conn_mod._ensure_database_exists(max_retries=1)
    assert executed, "no statement was executed at all"
    assert "pg_database" in executed[0], executed
    assert any("CREATE DATABASE new_db" in stmt for stmt in executed), executed
    mock_engine.dispose.assert_called_once()
"""

_TTS_BEFORE = """
async def test_database_created(self):
    mock_result = MagicMock()
    mock_result.scalar.return_value = None
    mock_conn.execute.side_effect = [mock_result, None]
    await _ensure_database_exists()
"""

_TTS_AFTER = """
async def test_database_created(self):
    executed: list[str] = []

    def _execute(statement, *_args, **_kwargs):
        executed.append(str(statement))
        return mock_result if len(executed) == 1 else None

    mock_conn.execute.side_effect = _execute
    await _ensure_database_exists()
    assert executed, "no statement was executed at all"
    assert "pg_database" in executed[0], executed
    assert any(f"CREATE DATABASE {conn_mod.DB_NAME}" in stmt for stmt in executed), executed
"""


def test_flags_every_real_regression_from_the_0_65_0_relock() -> None:
    """The four SHAPE-family regressions of 2026-08-24/25 are all flagged."""
    for label, src in (
        ("analysis_service/analyze", _ANALYSIS_BEFORE),
        ("analysis_service/brand", _ANALYSIS_BRAND_BEFORE),
        ("app_inspector_service", _APP_INSPECTOR_BEFORE),
        ("ai_content_generation_service", _AI_CONTENT_BEFORE),
    ):
        assert shape_only_reason(_fn(src)) is not None, f"{label}: NOT flagged"


def test_the_tts_regression_is_caught_by_the_assertion_free_half() -> None:
    """tts_service's pre-fix test asserted nothing at all — Part 1's job, not Part 2.

    Recorded explicitly so nobody 'fixes' Part 2 to also claim this one: a test
    with zero assertions is not shape-only, it is vacuous, and the two halves
    must not overlap or the ratchets double-count.
    """
    before = _fn(_TTS_BEFORE)
    assert not has_assertion(before), "tts pre-fix should be assertion-free"
    assert shape_only_reason(before) is None, "vacuous != shape-only"
    assert has_fixed_length_side_effect(before), "side_effect=[a, b] not recognised"


def test_does_not_flag_the_fixed_versions() -> None:
    """The merged fixes must be clean — a gate that reds the fix is worthless."""
    for label, src in (
        ("analysis_service", _ANALYSIS_AFTER),
        ("ai_content_generation_service", _AI_CONTENT_AFTER),
        ("tts_service", _TTS_AFTER),
    ):
        fn = _fn(src)
        assert has_assertion(fn), f"{label}: fixed version has no assertion"
        assert shape_only_reason(fn) is None, (
            f"{label}: FALSE POSITIVE on the fixed version — {shape_only_reason(fn)}"
        )


def test_out_of_scope_defects_are_not_false_positived() -> None:
    """The two non-shape defects from the same session are structurally fine.

    ``video_compiler_service::test_parse_loudness_handles_inf_silence`` pinned
    ``-math.inf`` (the value postgres rejects for JSONB) and the orchestrator's
    e2e check pinned ``"x-tenant-id" not in headers``, a contract a later
    security fix inverted. Both assert real, observable values — they are just
    the WRONG values. No static checker reaches that, and this one must not
    pretend to by flagging them.
    """
    inf_test = _fn(
        "def test_parse_loudness_handles_inf_silence() -> None:\n"
        '    result = parse_loudness(\'{ "input_i" : "-inf" }\')\n'
        '    assert result["integrated_lufs"] == -math.inf\n'
    )
    header_test = _fn(
        "def test_panel_image_is_anonymously_fetchable() -> None:\n"
        "    anon_headers = _capture()\n"
        '    assert anon_headers and "x-tenant-id" not in anon_headers[0]\n'
    )
    for fn in (inf_test, header_test):
        assert shape_only_reason(fn) is None, f"false positive: {fn.name}"


# ═════════════════════════════════════════════════════════════════════════════
# Classifier unit tests — MUTATION EVIDENCE
# ═════════════════════════════════════════════════════════════════════════════
#
# Each pair below was mutation-checked by hand on 2026-08-25: deleting the
# specific guard in scripts/check_vacuous_tests.py turns the paired assertion
# RED. Recorded per case so a future edit that "simplifies" a guard away is
# caught by a failing test rather than by a fleet-wide red six weeks later.


def test_shape_forms_are_recognised() -> None:
    """Every construct the detector calls SHAPE. Mutation: drop the matching
    branch in ``_classify_assert`` / ``_classify_assertion_call`` and one of
    these flips to behavioural, failing here."""
    for body in (
        "assert m.execute.call_count == 2",
        "assert m.await_count == 3",
        "assert len(m.call_args_list) == 2",
        "assert m.called",
        "m.dispose.assert_called_once()",
        "m.send.assert_awaited_once()",
        "self.assertEqual(m.call_count, 2)",
    ):
        fn = _fn(f"def test_x() -> None:\n    {body}\n")
        assert shape_only_reason(fn) is not None, f"not flagged as shape-only: {body!r}"


def test_behavioural_forms_are_never_flagged() -> None:
    """The false-positive floor. Mutation: widen any guard and one of these trips."""
    for body in (
        "assert result == 42",
        'assert "CREATE DATABASE" in executed[0]',
        'm.execute.assert_called_once_with("SELECT 1")',
        "m.execute.assert_any_call(sql)",
        "m.execute.assert_has_calls([call(a), call(b)])",
        "m.execute.assert_not_called()",
        "assert not m.called",
        "assert m.call_count == 0",
        "with pytest.raises(ValueError):\n        f()",
        # call_args (singular, = the LAST call) is deliberately NOT an index.
        'assert m.execute.call_args[0][0] == "SELECT 1"',
        # [-1] is not a positional pin either.
        'assert m.execute.call_args_list[-1].kwargs["sql"] == "x"',
        # [0] on a single-call mock is the standard payload idiom.
        'assert m.record.await_args_list[0].kwargs["params"]["bg"] == "#0A0A0A"',
        # A count assertion BESIDE a real property is fine.
        'assert m.call_count == 2\n    assert out.title == "Diagnose"',
        # ...even when the sequence is scripted with a fixed-length side_effect,
        # as long as no later call is indexed positionally.
        "m.side_effect = [a, b]\n    assert m.call_count == 2\n    assert out.ok is True",
    ):
        fn = _fn(f"def test_x() -> None:\n    {body}\n")
        assert shape_only_reason(fn) is None, f"FALSE POSITIVE on {body!r}: {shape_only_reason(fn)}"


def test_assertion_free_is_not_reported_as_shape_only() -> None:
    """The two halves must not double-count the same function."""
    fn = _fn("def test_x() -> None:\n    do_something()\n")
    assert classify_assertions(fn) == []
    assert shape_only_reason(fn) is None


def test_exact_call_count_pin_requires_two_or_more() -> None:
    """``call_count == 0`` is a negative claim, not a statement-count pin.

    Mutation: change the ``>= 2`` in ``pins_exact_call_count`` to ``>= 0`` and
    this fails.
    """
    assert pins_exact_call_count(_fn("def test_x() -> None:\n    assert m.call_count == 2\n"))
    assert not pins_exact_call_count(_fn("def test_x() -> None:\n    assert m.call_count == 0\n"))


def test_rule_b_needs_both_halves() -> None:
    """Rule B fires only on count-pin AND positional index of a LATER call.

    Mutation: make ``has_fixed_length_side_effect`` an independent trigger and
    the ``side_effect``-only case below flips to flagged — which is exactly the
    25-test false-positive wave measured on this repo before it was narrowed.
    """
    both = _fn(
        "def test_x() -> None:\n"
        "    assert m.call_count == 2\n"
        '    assert str(m.call_args_list[1][0][0]) == "CREATE DATABASE d"\n'
    )
    assert shape_only_reason(both) is not None

    side_effect_only = _fn(
        "def test_x() -> None:\n"
        "    m.side_effect = [a, b]\n"
        "    assert m.call_count == 2\n"
        '    assert out.sections[0].title == "Diagnose"\n'
    )
    assert has_fixed_length_side_effect(side_effect_only)
    assert pins_exact_call_count(side_effect_only)
    assert shape_only_reason(side_effect_only) is None


def test_shape_derived_locals_are_tracked_one_hop() -> None:
    """``sql = str(m.call_args_list[1]...)`` makes ``sql`` shape-derived.

    Mutation: delete ``_shape_derived_locals`` (return an empty set) and the
    ai_content_generation regression stops being flagged by rule A's path.
    """
    fn = _fn(
        "def test_x() -> None:\n"
        "    create_sql = str(m.execute.call_args_list[1][0][0])\n"
        '    assert "CREATE DATABASE" in create_sql\n'
    )
    assert shape_only_reason(fn) is not None
