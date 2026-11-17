"""Static guard against *vacuous* tests — ``test_*`` functions that assert nothing.

Why this exists
---------------
Two tests merged on 2026-08-06 were vacuous and were caught only by hand-mutating
the production source:

* ``test_panel_without_image_renders_placeholder_path`` proved only that a code
  BRANCH was taken. Stubbing the drawing routine to ``return`` immediately left
  every other test in the file green — the feature would have rendered blank
  space and nothing would have said so.
* Nothing asserted that a route reached the network through its SSRF chokepoint;
  swapping the guarded call for a raw ``httpx.get`` left the suite green, because
  both paths failed identically.

Neither review nor the existing gate caught them. Ruff does **not** cover this:
the ``PT`` (flake8-pytest-style) family governs the *shape* of assertions
(``PT009`` unittest-style, ``PT011`` too-broad ``raises``, ``PT015``
always-false, ``PT018`` composite) — there is no rule anywhere in ruff for
"this test contains zero assertions". Hence a repo-local check.

What counts as an assertion-bearing construct
---------------------------------------------
Inside the body of a ``test_*`` function (nested functions included, since a
locally-defined helper that asserts is legitimate):

* an ``assert`` statement;
* ``pytest.raises`` / ``pytest.warns`` / ``pytest.deprecated_call`` /
  ``pytest.fail`` / ``pytest.xfail``;
* any call whose attribute name starts with ``assert`` — this covers the whole
  ``unittest.mock`` family (``assert_called_once_with``, ``assert_not_awaited``,
  …) and ``unittest.TestCase`` style (``self.assertEqual``);
* any call to a bare name starting with ``assert`` or ``_assert`` — the
  project-local ``_assert_*`` helper convention.

Deliberately NOT counted: ``pytest.approx`` (an argument, not an assertion) and
``pytest.skip`` (the opposite of an assertion).

Limits — read before trusting this
-----------------------------------
This is a *static* check, so it proves only that an assertion is **present**,
never that it **ran** or that it is **meaningful**. Of the two motivating bugs it
would have caught neither directly — both tests contained asserts, they were
just weak ones. What it buys is the floor: the "N passed, 0 assertions" shape
can no longer enter the suite, and the offender count can only fall. Catching
the weak-assert shape needs mutation testing, not a linter.

Part 2 of this module (see the banner below ``find_vacuous_tests``) extends the
same machinery to the next rung up: tests whose ONLY assertions describe the
implementation's SHAPE — call counts, call positions. That class is what broke
five tests across four repos when w2v-common 0.65.0 added one correct statement.

Usage::

    uv run python -m scripts.check_vacuous_tests                # check both (exit 1 on new)
    uv run python -m scripts.check_vacuous_tests --list         # assertion-free offenders
    uv run python -m scripts.check_vacuous_tests --list-shape   # shape-only offenders
    uv run python -m scripts.check_vacuous_tests --report \
        --repo-root ../analysis_service                         # fleet survey, never fails

The gates themselves are ``tests/test_vacuous_test_guard.py`` and
``tests/test_shape_only_test_guard.py`` so they run under the archetype's
``uv run pytest tests/`` — same wiring as ``tests/test_any_ratchet.py``.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"
ALLOWLIST_PATH = TESTS_DIR / "vacuous_test_allowlist.txt"

_ASSERTION_CALL_PREFIX = "assert"
_PYTEST_ASSERTION_HELPERS = frozenset(
    {"raises", "warns", "deprecated_call", "fail", "xfail"},
)


@dataclass(frozen=True, order=True)
class VacuousTest:
    """An assertion-free ``test_*`` function, identified the way pytest identifies it.

    Deliberately NOT named ``Test*`` — pytest would try to collect it as a test
    class and warn.
    """

    path: str
    """Repo-relative posix path of the test module."""

    name: str
    """Dotted name: ``TestClass.test_thing`` or plain ``test_thing``."""

    lineno: int

    @property
    def nodeid(self) -> str:
        return f"{self.path}::{self.name}"


def _is_assertion_call(node: ast.Call) -> bool:
    """True if this call is an assertion-bearing construct."""
    func = node.func
    if isinstance(func, ast.Attribute):
        # mock's assert_called_*/assert_awaited_*, unittest's self.assertEqual, …
        if func.attr.startswith(_ASSERTION_CALL_PREFIX):
            return True
        # pytest.raises(...) / pytest.warns(...) / pytest.fail(...)
        return (
            isinstance(func.value, ast.Name)
            and func.value.id == "pytest"
            and func.attr in _PYTEST_ASSERTION_HELPERS
        )
    if isinstance(func, ast.Name):
        # Bare `raises(...)`/`fail(...)` from `from pytest import raises`, plus the
        # project-local `_assert_*` / `assert_*` helper convention.
        return func.id.lstrip("_").startswith(_ASSERTION_CALL_PREFIX) or (
            func.id in _PYTEST_ASSERTION_HELPERS
        )
    return False


def has_assertion(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True if the function body contains at least one assertion-bearing construct.

    Nested function definitions are walked too: a locally defined ``def _check()``
    that asserts, invoked by the test, is a real assertion.
    """
    for node in ast.walk(func):
        if isinstance(node, ast.Assert):
            return True
        if isinstance(node, ast.Call) and _is_assertion_call(node):
            return True
    return False


def _collect(
    body: list[ast.stmt],
    path: str,
    *,
    class_prefix: str = "",
) -> list[VacuousTest]:
    """Collect ``test_*`` functions at module level and inside ``Test*`` classes."""
    found: list[VacuousTest] = []
    for node in body:
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            found.extend(_collect(node.body, path, class_prefix=f"{node.name}."))
        elif (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name.startswith("test_")
            and not has_assertion(node)
        ):
            found.append(
                VacuousTest(path=path, name=f"{class_prefix}{node.name}", lineno=node.lineno)
            )
    return found


def iter_test_files(tests_dir: Path = TESTS_DIR) -> list[Path]:
    """Both halves of pytest's default ``python_files`` — ``test_*.py`` AND ``*_test.py``.

    The repo uses both (e.g. ``tests/unit/artifact_url_test.py``); globbing only
    the first would silently exempt whole modules.
    """
    return sorted(set(tests_dir.rglob("test_*.py")) | set(tests_dir.rglob("*_test.py")))


def find_vacuous_tests(
    tests_dir: Path = TESTS_DIR,
    repo_root: Path = REPO_ROOT,
) -> list[VacuousTest]:
    """Return every ``test_*`` function in ``tests_dir`` with no assertion."""
    offenders: list[VacuousTest] = []
    for file in iter_test_files(tests_dir):
        rel = file.relative_to(repo_root).as_posix()
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        offenders.extend(_collect(tree.body, rel))
    return sorted(offenders)


def load_allowlist(path: Path = ALLOWLIST_PATH) -> set[str]:
    """Parse the allowlist: one ``path::name`` nodeid per line, ``#`` comments stripped."""
    if not path.exists():
        return set()
    entries: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            entries.add(line)
    return entries


# ═════════════════════════════════════════════════════════════════════════════
# Part 2 — SHAPE-ONLY tests (assertions that pin the IMPLEMENTATION, not the
# behaviour). Added 2026-08-25.
# ═════════════════════════════════════════════════════════════════════════════
#
# Part 1 above catches a test with NO assertion. This part catches the next rung
# up, which is measurably more common in this fleet and strictly worse than no
# test: a test whose ONLY assertions describe the SHAPE of the implementation —
# how many calls it made, in what position — rather than what it did.
#
# Such a test breaks on any CORRECT change and passes on any WRONG one that
# preserves the shape. Measured on 2026-08-24/25, when w2v-common 0.65.0 made
# ``_create_or_check_database`` issue a legitimate THIRD statement (``REVOKE
# CONNECT ON DATABASE <db> FROM PUBLIC`` — the born-LOCKED provisioning policy),
# it reddened FIVE tests across FOUR repos, none of which was asserting anything
# about database creation:
#
#   analysis_service        2 tests, sole assertion ``execute.call_count == 2``
#   app_inspector_service   1 test,  sole assertion ``execute.call_count == 2``
#   ai_content_generation   1 test,  ``call_count == 2`` + ``call_args_list[1]``
#                                    positional indexing → ``assert 3 == 2``
#   tts_service             1 test,  ``side_effect=[mock_result, None]`` — a
#                                    fixed-length list pinning the statement
#                                    count; the third call raised StopIteration
#                                    inside a coroutine → RuntimeError
#
# THE THREE SHAPES FLAGGED
# ------------------------
# ``shape``          — ``x.call_count == N`` / ``len(x.call_args_list) == N`` /
#                      ``x.called`` / ``assert_called_once()`` (the ARGUMENT-LESS
#                      form; ``assert_called_once_with(...)`` asserts arguments
#                      and is behavioural).
# ``shape_derived``  — an assertion whose subject is reached through a
#                      NON-NEGATIVE literal index into a mock call list
#                      (``call_args_list[1][0][0]``). The content it checks may
#                      be real, but the positional index pins call ORDER, which
#                      is exactly what a new upstream statement shifts. Locals
#                      assigned from such an index are tracked one hop
#                      (``sql = str(m.call_args_list[1][0][0])`` → ``sql`` is
#                      shape-derived).
# ``behavioural``    — everything else.
#
# FALSE-POSITIVE DISCIPLINE — read before widening anything here
# --------------------------------------------------------------
# A noisy gate is a gate people learn to skip, and this fleet has that failure
# mode on record. So the bar is: flag ONLY when shape is ALL there is.
#
#   * A test asserting a call count AND a real behavioural property is NOT
#     flagged. Counting calls is legitimate; counting calls *instead of*
#     checking behaviour is not.
#   * ``assert_called_once_with(...)`` / ``assert_any_call`` / ``assert_has_calls``
#     assert ARGUMENTS → behavioural, never flagged.
#   * ``assert_not_called()`` / ``assert not m.called`` — hmm: these are counted
#     as shape, because "N == 0" is still a count. They are, however, usually a
#     genuine claim ("we never touched the DB"), so a test whose ONLY assertion
#     is a negative call check WILL be flagged and belongs on the allowlist with
#     a reason. That is a deliberate, reviewed exemption rather than a silent
#     hole in the detector.
#   * ``call_args`` / ``await_args`` (singular, = the LAST call) are NOT treated
#     as positional indices. They are brittle in the same way, but the idiom is
#     pervasive and legitimate for single-call mocks; flagging it would drown
#     the signal.
#   * ``[-1]`` and non-literal indices are not flagged.
#   * A ``BoolOp`` (``assert a.call_count == 2 and result == "x"``) is treated as
#     behavioural — when in doubt, do not flag.
#   * ⚠️ A fixed-length ``side_effect=[a, b]`` beside ``call_count == 2`` is NOT
#     an independent trigger, even though it is literally the tts_service shape.
#     Measured: on w2v_orchestrator that pair alone flags 25 extra tests, and
#     reading them shows the overwhelming majority are legitimate — a two-call
#     protocol (``test_outline_split_on_stitches_two_calls``, "retry once then
#     succeed") where "exactly 2 calls" IS the behaviour under test and a third
#     call SHOULD fail it. A static checker cannot separate "2 is the contract"
#     from "2 is an implementation detail". So the side_effect list only
#     ENRICHES the reason text when Rule B fires on other grounds. Nothing is
#     lost: the tts_service test was assertion-free and Part 1 already had it,
#     and the analysis_service / app_inspector cases are Rule A.
#
# HONEST SCOPE — what this CANNOT catch
# -------------------------------------
# Two of the eleven bad tests found in the same session are NOT of this family
# and no static checker can reach them:
#   * ``video_compiler_service::test_parse_loudness_handles_inf_silence`` asserted
#     ``== -math.inf`` — a behavioural assertion pinning the WRONG value (the one
#     postgres rejects for JSONB).
#   * ``w2v_orchestrator::assert "x-tenant-id" not in headers`` pinned an auth
#     contract that a later security fix inverted.
# Both are semantically wrong, structurally fine. Catching those needs mutation
# testing or a contract, not a linter. This module deliberately does not pretend
# otherwise, and both are verified NOT to be false-positived (see
# tests/test_shape_only_test_guard.py).

SHAPE_ALLOWLIST_PATH = TESTS_DIR / "shape_only_test_allowlist.txt"

_COUNT_ATTRS = frozenset({"call_count", "await_count"})
_TRUTHY_CALL_ATTRS = frozenset({"called", "call_count", "await_count"})
_CALL_LIST_ATTRS = frozenset(
    {"call_args_list", "await_args_list", "mock_calls", "method_calls"},
)
_SHAPE_MOCK_ASSERTS = frozenset(
    {
        "assert_called",
        "assert_called_once",
        "assert_awaited",
        "assert_awaited_once",
    },
)

SHAPE = "shape"
SHAPE_DERIVED = "shape_derived"
BEHAVIOURAL = "behavioural"


@dataclass(frozen=True, order=True)
class ShapeOnlyTest:
    """A ``test_*`` function whose every assertion describes implementation shape."""

    path: str
    name: str
    lineno: int
    reason: str

    @property
    def nodeid(self) -> str:
        return f"{self.path}::{self.name}"


def _names_in(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _is_zero_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == 0 and not isinstance(node.value, bool)


def _is_count_expr(node: ast.AST) -> bool:
    """``m.call_count`` / ``m.await_count`` / ``len(m.call_args_list)``."""
    if isinstance(node, ast.Attribute) and node.attr in _COUNT_ATTRS:
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "len":
        return any(
            isinstance(inner, ast.Attribute) and inner.attr in _CALL_LIST_ATTRS
            for arg in node.args
            for inner in ast.walk(arg)
        )
    return False


def _has_positional_call_index(node: ast.AST) -> bool:
    """``m.call_args_list[1]`` — a literal index >= 1 into a mock call list.

    ``[0]`` is DELIBERATELY excluded, and that exclusion is what makes this
    usable. Reaching into the FIRST call to inspect its payload
    (``m.record.await_args_list[0].kwargs["params"]``) is the standard idiom for
    a single-call mock and is stable under exactly the change that breaks this
    defect class — a statement APPENDED to the sequence. Measured on
    w2v_orchestrator: counting ``[0]`` produced 68 hits, mostly tests that then
    assert richly on the payload; requiring ``>= 1`` cuts that to the tests that
    really pin call ORDER. ``[-1]`` and computed indices are excluded too.
    """
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Subscript):
            continue
        idx = sub.slice
        if not (
            isinstance(idx, ast.Constant)
            and isinstance(idx.value, int)
            and not isinstance(idx.value, bool)
            and idx.value >= 1
        ):
            continue
        if any(
            isinstance(inner, ast.Attribute) and inner.attr in _CALL_LIST_ATTRS
            for inner in ast.walk(sub.value)
        ):
            return True
    return False


def _shape_derived_locals(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Locals bound to a positionally-indexed mock call, propagated one hop.

    ``select_sql = str(m.call_args_list[0][0][0])`` makes ``select_sql`` a
    shape-derived name, so ``assert "pg_database" in select_sql`` is recognised
    for what it is: a real content check reached through a pinned call index.
    """
    derived: set[str] = set()
    for _ in range(2):  # one extra pass so a re-binding chain is caught
        for node in ast.walk(func):
            if not isinstance(node, ast.Assign):
                continue
            if _has_positional_call_index(node.value) or (_names_in(node.value) & derived):
                derived.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return derived


def _classify_assert(node: ast.expr, derived: set[str]) -> str:
    core = node
    while isinstance(core, ast.UnaryOp) and isinstance(core.op, ast.Not):
        core = core.operand
    if isinstance(core, ast.Compare) and (
        _is_count_expr(core.left) or any(_is_count_expr(c) for c in core.comparators)
    ):
        # `call_count == 0` is a NEGATIVE claim ("we never touched the DB"), which
        # is a real property of behaviour — not a pin on the implementation's
        # statement count. Same reasoning excludes `assert_not_called()`.
        if _is_zero_literal(core.left) or any(_is_zero_literal(c) for c in core.comparators):
            return BEHAVIOURAL
        return SHAPE
    if isinstance(core, ast.Attribute) and core.attr in _TRUTHY_CALL_ATTRS:
        # `assert m.called` is shape; `assert not m.called` is a negative claim.
        return BEHAVIOURAL if core is not node else SHAPE
    if _is_count_expr(core):
        return SHAPE
    if _has_positional_call_index(core) or (_names_in(core) & derived):
        return SHAPE_DERIVED
    return BEHAVIOURAL


def _classify_assertion_call(node: ast.Call, derived: set[str]) -> str:
    func = node.func
    if not isinstance(func, ast.Attribute):
        # `pytest.raises(...)` imported bare, or a project `_assert_*` helper —
        # unknowable statically, so: behavioural.
        return BEHAVIOURAL
    if func.attr in _SHAPE_MOCK_ASSERTS and not node.args and not node.keywords:
        return SHAPE
    if not func.attr.startswith(_ASSERTION_CALL_PREFIX):
        return BEHAVIOURAL  # pytest.raises / pytest.warns / pytest.fail
    # unittest style: self.assertEqual(m.call_count, 2)
    if any(_is_count_expr(a) for a in node.args):
        return SHAPE
    if any(_has_positional_call_index(a) or (_names_in(a) & derived) for a in node.args):
        return SHAPE_DERIVED
    return BEHAVIOURAL


def classify_assertions(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Every assertion in ``func``, labelled shape / shape_derived / behavioural."""
    derived = _shape_derived_locals(func)
    kinds: list[str] = []
    for node in ast.walk(func):
        if isinstance(node, ast.Assert):
            kinds.append(_classify_assert(node.test, derived))
        elif isinstance(node, ast.Call) and _is_assertion_call(node):
            kinds.append(_classify_assertion_call(node, derived))
    return kinds


def has_fixed_length_side_effect(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """``side_effect=[a, b]`` / ``m.side_effect = [a, b]`` — a pinned statement COUNT.

    Length >= 2 only: a single-element list is a one-shot value, not a script of
    the exact call sequence.
    """
    for node in ast.walk(func):
        value: ast.expr | None = None
        if (isinstance(node, ast.keyword) and node.arg == "side_effect") or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Attribute) and t.attr == "side_effect" for t in node.targets)
        ):
            value = node.value
        if isinstance(value, ast.List | ast.Tuple) and len(value.elts) >= 2:
            return True
    return False


def pins_exact_call_count(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """``assert m.call_count == N`` with ``N >= 2`` — the exact statement count."""
    for node in ast.walk(func):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        if not isinstance(node.ops[0], ast.Eq):
            continue
        sides = [node.left, *node.comparators]
        if not any(_is_count_expr(sd) for sd in sides):
            continue
        if any(
            isinstance(sd, ast.Constant) and isinstance(sd.value, int) and sd.value >= 2
            for sd in sides
        ):
            return True
    return False


def _indexes_a_later_call(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        _has_positional_call_index(node) for node in ast.walk(func) if isinstance(node, ast.expr)
    )


def shape_only_reason(func: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """Why this test is shape-only, or ``None`` if it is not."""
    kinds = classify_assertions(func)
    if not kinds:
        return None  # assertion-free — Part 1's business, not this one
    if all(k in {SHAPE, SHAPE_DERIVED} for k in kinds):
        if SHAPE_DERIVED in kinds and SHAPE in kinds:
            detail = "call-count assertions plus positionally-indexed call_args_list"
        elif SHAPE_DERIVED in kinds:
            detail = "only positionally-indexed call_args_list assertions"
        else:
            detail = "only call-count / assert_called* assertions"
        return f"asserts implementation SHAPE, never behaviour ({detail})"
    # Rule B — the OVER-SPECIFIED CALL SEQUENCE. The test does carry a real
    # behavioural assertion, but it ALSO pins both how many calls were made and
    # which position each one sits at. That pair is the exact signature of the
    # 2026-08-24 breakage: a legitimate extra upstream statement shifts every
    # index and blows the count. Requiring BOTH halves is what keeps this quiet —
    # a lone count assertion beside real behaviour is fine and is not flagged.
    if pins_exact_call_count(func) and _indexes_a_later_call(func):
        extra = (
            " (and scripts the sequence with a fixed-length side_effect list)"
            if has_fixed_length_side_effect(func)
            else ""
        )
        return (
            "pins the exact call SEQUENCE: `call_count == N` together with "
            f"positional indexing of a later call{extra}"
        )
    return None


def _collect_shape_only(
    body: list[ast.stmt],
    path: str,
    *,
    class_prefix: str = "",
) -> list[ShapeOnlyTest]:
    found: list[ShapeOnlyTest] = []
    for node in body:
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            found.extend(_collect_shape_only(node.body, path, class_prefix=f"{node.name}."))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith(
            "test_"
        ):
            reason = shape_only_reason(node)
            if reason is not None:
                found.append(
                    ShapeOnlyTest(
                        path=path,
                        name=f"{class_prefix}{node.name}",
                        lineno=node.lineno,
                        reason=reason,
                    )
                )
    return found


def find_shape_only_tests(
    tests_dir: Path = TESTS_DIR,
    repo_root: Path = REPO_ROOT,
) -> list[ShapeOnlyTest]:
    """Return every ``test_*`` function in ``tests_dir`` that only asserts shape."""
    offenders: list[ShapeOnlyTest] = []
    for file in iter_test_files(tests_dir):
        rel = file.relative_to(repo_root).as_posix()
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        offenders.extend(_collect_shape_only(tree.body, rel))
    return sorted(offenders)


def _resolve_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    """Repo root + tests dir — defaults to THIS repo, overridable for fleet reports."""
    root = Path(args.repo_root).resolve() if args.repo_root else REPO_ROOT
    tests = Path(args.tests_dir).resolve() if args.tests_dir else root / "tests"
    return root, tests


def _report_against_allowlist(
    label: str,
    offenders: list[VacuousTest] | list[ShapeOnlyTest],
    allowlist_path: Path,
) -> bool:
    """Print new + stale entries for one ratchet; return True if it should fail."""
    allowed = load_allowlist(allowlist_path)
    new = [o for o in offenders if o.nodeid not in allowed]
    stale = allowed - {o.nodeid for o in offenders}
    for offender in new:
        detail = getattr(offender, "reason", "no assertion")
        print(f"{offender.path}:{offender.lineno}: {label} test `{offender.name}` — {detail}")
    if stale:
        print(f"{len(stale)} stale entries in {allowlist_path.name} — remove them:")
        for nodeid in sorted(stale):
            print(f"  {nodeid}")
    return bool(new or stale)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument("--list", action="store_true", help="print assertion-free offenders")
    parser.add_argument("--list-shape", action="store_true", help="print shape-only offenders")
    parser.add_argument("--repo-root", default=None, help="check another repo (fleet report)")
    parser.add_argument("--tests-dir", default=None, help="override the tests directory")
    parser.add_argument(
        "--report",
        action="store_true",
        help="report counts only, never fail (fleet survey mode)",
    )
    args = parser.parse_args(argv)
    root, tests = _resolve_dirs(args)
    if not tests.is_dir():
        print(f"no tests directory at {tests}")
        return 0

    if args.list:
        for offender in find_vacuous_tests(tests, root):
            print(offender.nodeid)
        return 0
    if args.list_shape:
        for shape in find_shape_only_tests(tests, root):
            print(f"{shape.nodeid}  # {shape.reason}")
        return 0

    vacuous = find_vacuous_tests(tests, root)
    shape_only = find_shape_only_tests(tests, root)
    if args.report:
        print(f"{root.name}: {len(vacuous)} assertion-free, {len(shape_only)} shape-only")
        for offender in shape_only:
            print(f"  {offender.path}:{offender.lineno}  {offender.name}")
        return 0

    failed = _report_against_allowlist("VACUOUS", vacuous, ALLOWLIST_PATH)
    failed |= _report_against_allowlist("SHAPE-ONLY", shape_only, SHAPE_ALLOWLIST_PATH)
    if failed:
        return 1
    print(
        f"OK — {len(vacuous)} known assertion-free, {len(shape_only)} known shape-only, none new."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
