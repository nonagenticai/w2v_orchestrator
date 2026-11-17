"""A marker that CI never re-selects makes its tests run on NO machine.

``pyproject.toml``'s ``addopts`` carries
``-m 'not integration and not cross_service_monorepo'``, and
``.github/workflows/ci-cd.yaml`` passes NO ``integration_test_select`` /
``neo4j_test_select`` to the archetype. So until 2026-09-07 every test wearing
either marker was deselected in PR CI, on main, and nightly — 80 tests across 7
files, including the SEVEN route-boundary tests PR #769 added for the
cross-tenant artifact-adoption defence. Unwiring that guard was a green diff.

This gate makes the hole un-reopenable: a marker listed in ``addopts``'s
deselection expression may only be worn by a file that appears in
``ALLOWED_UNRUN`` below, with a written reason.

Two deliberate design choices, both learned the hard way:

* **AST, not regex.** A previous gate in this fleet used a bare regex for its
  escape hatch and matched its own marker string inside a DOCSTRING, silencing
  itself. ``ast`` cannot see comments or strings — only real attribute
  references — so a test that merely *talks about* ``pytest.mark.integration``
  in prose (``tests/unit/test_preview_proxy.py`` does) is invisible here.
* **The allowlist is a Python dict, not a regex escape hatch, and every reason
  is PRINTED.** The same incident shipped a gate that parsed its waiver reason
  and then DISCARDED it, so every waiver rendered identically and an intentional
  exemption was indistinguishable from an accidental one. Here a waiver is a
  real dict entry (comments and docstrings cannot forge one) and its reason is
  echoed on every run. An entry must say why the test genuinely cannot run in
  CI (a live peer service, real network). "It needs Postgres" is NOT such a
  reason: the archetype provisions a private ephemeral Postgres container per
  job, and Redis is faked.
* **It has a VACUITY FLOOR.** "Nothing is marked wrong" is trivially satisfiable
  by deleting the tests. ``REQUIRED_CI_SELECTED`` names the specific
  security-critical tests that must exist AND be CI-selected, so deleting or
  re-hiding them fails just as loudly as marking them.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TESTS_ROOT = _REPO_ROOT / "tests"

#: Files permitted to wear a never-selected marker, mapped to WHY they cannot
#: run in CI. Adding an entry is a deliberate, reviewable statement that the
#: test needs something CI genuinely does not have — a live peer service or real
#: outbound network. Needing Postgres or Redis is not a qualifying reason.
ALLOWED_UNRUN: dict[str, str] = {}

#: The VACUITY FLOOR. These tests were deselected by `addopts` and re-selected
#: nowhere until 2026-09-07, so the behaviour they cover was unenforced. They
#: must exist, and CI must select them. Without this, `test_no_test_wears_a_
#: marker_ci_never_selects` passes just as happily against an empty tests/ tree.
#:
#: The `test_runs.py` block is PR #769's cross-tenant artifact-ref confinement at
#: the PATCH-element / restamp-element / patch-thumbnail route boundary; the
#: `test_run_lifecycle_guards.py` block is #894's cancelled-run resurrection and
#: billing guards. Do not shrink either list to make this file pass.
REQUIRED_CI_SELECTED: dict[str, tuple[str, ...]] = {
    "tests/api/routes/test_runs.py": (
        "test_patch_element_cannot_inject_a_foreign_runs_artifact",
        "test_patch_element_cannot_bury_a_foreign_artifact_in_a_nested_value",
        "test_restamp_element_cannot_inject_a_foreign_runs_artifact",
        "test_patch_thumbnail_cannot_adopt_a_foreign_runs_artifact",
        "test_patch_thumbnail_cannot_adopt_another_tenants_upload",
        "test_patch_element_may_repoint_at_the_runs_own_artifact",
        "test_patch_element_with_no_artifact_refs_is_unaffected",
    ),
}

#: Minimum number of tests that must live in the files this gate un-hid. The
#: 2026-09-07 measurement was 80 across 7 files; the floor sits below that so
#: ordinary churn does not trip it, but wholesale deletion does.
MIN_UNHIDDEN_TESTS = 70

#: The files whose tests were hidden. Counted against MIN_UNHIDDEN_TESTS.
UNHIDDEN_FILES: tuple[str, ...] = (
    "tests/api/routes/test_runs.py",
    "tests/api/routes/test_run_lifecycle_guards.py",
    "tests/api/routes/test_apply_edits_reel_music.py",
    "tests/services/test_run_claim.py",
    "tests/services/test_source_cache.py",
    "tests/services/test_run_dispatcher.py",
    "tests/services/stages/test_source_ingest_cache.py",
)


def _deselected_markers() -> set[str]:
    """The markers `addopts` excludes, read from pyproject — not hardcoded.

    Reading it back means adding a THIRD marker to the deselection expression
    without allowlisting or wiring it also trips this gate.
    """
    cfg = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    addopts = cfg["tool"]["pytest"]["ini_options"]["addopts"]
    if isinstance(addopts, list):
        addopts = " ".join(addopts)
    # `-m 'not a and not b'` -> {"a", "b"}
    tokens = addopts.replace("'", " ").replace('"', " ").split()
    out: set[str] = set()
    for i, tok in enumerate(tokens):
        if tok == "not" and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt not in {"not", "and", "or"}:
                out.add(nxt)
    return out


def _marker_names_used(path: Path) -> set[str]:
    """Every ``<something>.mark.<name>`` referenced anywhere in the module.

    Covers decorators, module-level ``pytestmark`` (scalar or list), and marks
    applied inside ``pytest.param``. Comments and docstrings are not AST
    attribute nodes, so they cannot trip this.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        parent = node.value
        if isinstance(parent, ast.Attribute) and parent.attr == "mark":
            names.add(node.attr)  # pytest.mark.<name>
        elif isinstance(parent, ast.Name) and parent.id == "mark":
            names.add(node.attr)  # from pytest import mark; mark.<name>
    return names


def test_addopts_deselection_is_covered_by_this_gate() -> None:
    """The expression must be parseable and non-degenerate, or the gate is inert."""
    markers = _deselected_markers()
    assert markers, (
        "Could not parse any deselected marker out of pyproject.toml's "
        "[tool.pytest.ini_options] addopts. This gate reads that expression to "
        "know what to police; if the parse yields nothing the gate silently "
        "polices nothing. Fix the parser in _deselected_markers(), do not "
        "delete this assertion."
    )
    declared = set(
        tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"][
            "markers"
        ]
    )
    declared_names = {m.split(":", 1)[0].strip() for m in declared}
    unknown = markers - declared_names
    assert not unknown, (
        f"addopts deselects marker(s) {sorted(unknown)} that are not declared in "
        "[tool.pytest.ini_options].markers. A typo'd marker name in the "
        "deselection expression deselects nothing and hides the mistake."
    )


def _collected_test_names(path: Path) -> set[str]:
    """Top-level `test_*` functions in a module, via AST (no import side effects)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    return {
        n.name
        for n in tree.body
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name.startswith("test_")
    }


def test_no_test_wears_a_marker_ci_never_selects(capsys: pytest.CaptureFixture[str]) -> None:
    markers = _deselected_markers()
    offenders: dict[str, set[str]] = {}
    waived: list[str] = []

    for path in sorted(_TESTS_ROOT.rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        used = _marker_names_used(path) & markers
        if not used:
            continue
        if rel in ALLOWED_UNRUN:
            reason = ALLOWED_UNRUN[rel].strip()
            assert reason, (
                f"{rel} is in ALLOWED_UNRUN with an EMPTY reason. A waiver whose "
                "reason is blank is indistinguishable from an accident — state "
                "what CI genuinely lacks (a live peer service, real network)."
            )
            # PRINTED, not merely parsed: a fleet gate once collected its waiver
            # reasons and discarded them, so every waiver rendered identically.
            waived.append(f"  {rel} [{','.join(sorted(used))}] — {reason}")
            continue
        offenders[rel] = used

    if waived:
        with capsys.disabled():
            print(
                "\nDELIBERATELY UNRUN IN CI (each waiver and its stated reason):\n"
                + "\n".join(sorted(waived))
            )

    if offenders:
        detail = "\n".join(f"  {f}: {sorted(m)}" for f, m in sorted(offenders.items()))
        pytest.fail(
            "These test files wear a pytest marker that pyproject's `addopts` "
            f"DESELECTS and that CI re-selects NOWHERE — so they execute on no "
            f"machine at all:\n{detail}\n\n"
            "PR #769's cross-tenant artifact-confinement route tests sat here "
            "for weeks: unwiring the guard they cover was a green diff.\n\n"
            "Fix it by REMOVING the marker (the archetype gives every job a "
            "private ephemeral Postgres and Redis is faked, so 'it needs a DB' "
            "is not a reason to hide a test), or — if the test truly needs a "
            "live peer service or real network — add it to ALLOWED_UNRUN in "
            f"{Path(__file__).name} with a written reason, which is a reviewable "
            "act rather than a silent one."
        )


def test_the_security_critical_tests_are_present_and_ci_selected() -> None:
    """VACUITY FLOOR: deleting the tests must fail as loudly as hiding them.

    Without this, `test_no_test_wears_a_marker_ci_never_selects` is satisfied by
    an empty tests/ tree.
    """
    markers = _deselected_markers()
    problems: list[str] = []

    for rel, required in REQUIRED_CI_SELECTED.items():
        path = _REPO_ROOT / rel
        if not path.exists():
            problems.append(f"{rel} is GONE — it held {len(required)} required tests.")
            continue
        present = _collected_test_names(path)
        missing = [n for n in required if n not in present]
        if missing:
            problems.append(f"{rel}: missing required test(s) {missing}")
        hidden = _marker_names_used(path) & markers
        if hidden:
            problems.append(
                f"{rel}: file-level marker(s) {sorted(hidden)} would deselect its "
                "required tests in CI."
            )

    assert not problems, (
        "The security-critical tests this gate exists to keep RUNNING are not all "
        "present and CI-selected:\n  " + "\n  ".join(problems) + "\n\n"
        "These are PR #769's cross-tenant artifact-ref confinement tests at the "
        "PATCH-element / restamp / patch-thumbnail route boundary. They cover the "
        "WIRING (the guard function itself is covered by "
        "tests/services/test_artifact_ref_guard.py); without them, deleting the "
        "confinement call from a route handler is a green diff. Restore them — do "
        "not shorten REQUIRED_CI_SELECTED to make this pass."
    )


def test_the_unhidden_files_still_carry_their_tests() -> None:
    """Second floor: the ~80 un-hidden tests must not quietly evaporate."""
    total = 0
    missing_files: list[str] = []
    for rel in UNHIDDEN_FILES:
        path = _REPO_ROOT / rel
        if not path.exists():
            missing_files.append(rel)
            continue
        total += len(_collected_test_names(path))

    assert not missing_files, (
        f"File(s) un-hidden on 2026-09-07 no longer exist: {missing_files}. If a "
        "file was legitimately split or renamed, update UNHIDDEN_FILES in the same "
        "commit so this floor keeps measuring something."
    )
    assert total >= MIN_UNHIDDEN_TESTS, (
        f"Only {total} tests remain across the files this gate un-hid, below the "
        f"floor of {MIN_UNHIDDEN_TESTS} (80 at the time of measurement). A gate "
        "that only asserts 'nothing is marked wrong' is satisfiable by deleting "
        "the tests; this is the assertion that catches that."
    )


def test_stale_allowlist_entries_are_removed() -> None:
    """A waiver for a file that no longer wears the marker is dead weight."""
    markers = _deselected_markers()
    stale = []
    for rel, reason in ALLOWED_UNRUN.items():
        path = _REPO_ROOT / rel
        if not path.exists() or not (_marker_names_used(path) & markers):
            stale.append(f"{rel} (waived because: {reason})")
    assert not stale, (
        "ALLOWED_UNRUN entries no longer needed (file gone, or marker removed):\n  "
        + "\n  ".join(stale)
        + "\nDelete them so the allowlist keeps meaning what it says."
    )
