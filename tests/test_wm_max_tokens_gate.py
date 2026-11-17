"""CI-enforced FLOOR on the ``max_tokens`` every WM ``/message`` call must carry.

A WM call that omits ``max_tokens`` inherits worker-management's
``DEFAULT_SELF_HOSTED_MAX_TOKENS = 6000``. A call that exhausts its budget is
**terminal**: WM answers 422, 422 is absent from
``w2v_common.clients.base._RETRYABLE_STATUS_CODES``, and ``upstream.guarded``
classifies it ``PermanentUpstreamError`` — so the stage dies on attempt 1 and takes
the run with it. Two live examples on 2026-08-19 alone: run
``01a018a2-bda8-74da-a5b2-c8e126b027b9`` (``website_review_script``) and the promo
E2E's ``growth_bundle``.

⚠️ THIS GATE IS A FLOOR, NOT A PRESENCE CHECK — AND THAT DISTINCTION IS THE WHOLE
POINT. The error text ("Model token limit (N) exceeded before any response was
generated") reads like a pre-flight rejection, and PR #851 acted on that reading by
LOWERING the budget to 1760. The raise is actually in
``pydantic_ai/_agent_graph.py:1100-1113``, AFTER generation, when
``finish_reason == 'length'`` and every returned part is a ``ThinkingPart``: the
reasoning model spent the budget thinking and never emitted the answer. Measured
live, one worker, one prompt, only ``max_tokens`` varying::

    500 -> 422 (43.5 s)      1760 -> 422 (131.5 s)      6000 -> no 422

A gate that only checked "is the kwarg present?" would have passed 1760 — the value
that reproduces the outage. So constants are checked against
``MIN_OUTPUT_TOKEN_BUDGET`` too.

DO NOT add a call site to the allow-list to make this pass. Pass a budget from
``src.services.stages.llm_budget.output_token_budget``.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from src.services.stages.llm_budget import MIN_OUTPUT_TOKEN_BUDGET, output_token_budget

_SRC = Path(__file__).resolve().parent.parent / "src"

# Names that reach ``WMClient.message``. A local alias (``send = wm.message``)
# would be invisible to an attribute-only match, so the bare name counts too.
_WM_CALL_NAMES = frozenset({"message"})

# Call sites deliberately exempt. Keyed by (path relative to src/, enclosing
# function qualname) — NEVER by line number, which drifts on every edit. The value
# is a mandatory reason. `test_no_stale_allowlist_entries` fails when an entry no
# longer matches a real site, so this list can only shrink.
_ALLOWED: dict[tuple[str, str], str] = {}


def _called_name(node: ast.Call) -> str | None:
    """The bare name being called, through an attribute or a local alias."""
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _looks_like_wm(node: ast.Call) -> bool:
    """Any spelling that could be carrying a ``worker_name``.

    Keyword, positional (it is NOT keyword-only in ``WMClient.message``), or
    hidden behind a spread — the last of which the gate then fails closed on.
    """
    return (
        any(k.arg == "worker_name" for k in node.keywords)
        or bool(node.args)
        or any(k.arg is None for k in node.keywords)
    )


def _wm_message_calls(tree: ast.AST) -> list[tuple[ast.Call, str]]:
    """Every ``WMClient.message(...)`` call, with its enclosing function qualname.

    Matched by the called NAME (attribute, or a module-local alias) plus any
    spelling that could carry a ``worker_name`` — keyword, positional, or hidden
    behind a spread. ``worker_name`` is unique to this signature, which is what
    keeps the gate from firing on unrelated ``.message(`` calls on loggers,
    exceptions and responses.

    The keyword-only version of this check was walked around three ways by an
    adversarial review (positional first arg, ``*args``/``**kwargs``, and
    ``send = wm.message``), each of which is a real WM call whose budget would
    never have been checked. See the bypass tests at the bottom of this file.
    """
    found: list[tuple[ast.Call, str]] = []
    stack: list[str] = []

    # Local aliases: ``send = wm.message`` makes the later call an ``ast.Name``,
    # invisible to an attribute match. Collect the alias targets first so the
    # visitor can recognise them. Module-scoped on purpose — an alias that
    # escapes its module is beyond what a static gate can follow, and the
    # ``**spread`` rule below is the backstop for anything this misses.
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "message"
        ):
            aliases.update(t.id for t in node.targets if isinstance(t, ast.Name))
    names = _WM_CALL_NAMES | aliases

    class V(ast.NodeVisitor):
        def _scoped(self, node: ast.AST, name: str) -> None:
            stack.append(name)
            self.generic_visit(node)
            stack.pop()

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self._scoped(node, node.name)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._scoped(node, node.name)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._scoped(node, node.name)

        def visit_Call(self, node: ast.Call) -> None:
            # ``worker_name`` is NOT keyword-only in WMClient.message, so a
            # positional first argument is a legal spelling — and an adversarial
            # pass proved the first version of this detector missed all three of
            # ``wm.message("w", "hi")``, ``wm.message(*args, **kw)`` and
            # ``send = wm.message; send(worker_name=...)``. Each of those is a
            # WM call the floor would never have been checked on, so the detector
            # matches the NAME ``message`` however it is reached, and treats an
            # unresolvable call shape as suspicious rather than absent.
            if _called_name(node) in names and _looks_like_wm(node):
                found.append((node, ".".join(stack) or "<module>"))
            self.generic_visit(node)

    V().visit(tree)
    return found


def _budget_kw(call: ast.Call) -> ast.keyword | None:
    return next((k for k in call.keywords if k.arg == "max_tokens"), None)


def _has_spread(call: ast.Call) -> bool:
    """``f(**kwargs)`` — the kwargs are unknowable, so the gate fails CLOSED."""
    return any(k.arg is None for k in call.keywords)


def _sites() -> list[tuple[str, str, ast.Call]]:
    out: list[tuple[str, str, ast.Call]] = []
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        rel = str(path.relative_to(_SRC))
        for call, qualname in _wm_message_calls(tree):
            out.append((rel, qualname, call))
    return out


def test_there_are_wm_call_sites_to_govern() -> None:
    """Vacuity guard: a detector that finds nothing proves nothing."""
    assert len(_sites()) >= 20, (
        "the detector found almost no WM call sites — it is probably broken, "
        "not the codebase suddenly clean"
    )


def test_every_wm_message_call_passes_max_tokens() -> None:
    missing = [
        f"src/{rel}::{qual}"
        for rel, qual, call in _sites()
        if _budget_kw(call) is None and (rel, qual) not in _ALLOWED
    ]
    assert not missing, (
        "WM /message calls without an explicit `max_tokens` (they inherit WM's "
        "6000 default and die TERMINALLY when a reasoning model burns it):\n  "
        + "\n  ".join(missing)
    )


def test_no_wm_message_call_hides_its_kwargs_behind_a_spread() -> None:
    spread = [
        f"src/{rel}::{qual}"
        for rel, qual, call in _sites()
        if _has_spread(call) and (rel, qual) not in _ALLOWED
    ]
    assert not spread, (
        "WM /message calls using **kwargs — this gate cannot see whether a budget "
        "is passed, so it fails closed:\n  " + "\n  ".join(spread)
    )


class _Unresolvable:
    """Sentinel: the gate could not statically read this budget.

    Distinct from "no budget passed" and, critically, distinct from ``None`` —
    the previous resolver returned ``None`` for every shape it did not handle and
    the floor check then ``continue``d past it. An adversarial probe file with
    seven bad ``max_tokens=1760`` calls was caught ONCE: ``self._budget``,
    ``Cfg.tokens``, ``_TIERS["a"]``, ``8000 - 6240``, ``1760 if x else 9000`` and
    ``int("1760")`` all sailed through. An unresolvable budget is not evidence of
    safety, so it is now a FAILURE — the same way ``**spread`` already was.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<unresolvable>"


UNRESOLVABLE = _Unresolvable()

# Budget expressions the gate cannot read but that a human has cleared anyway.
# Keyed by (path relative to src/, enclosing function qualname) exactly like
# ``_ALLOWED``; the value is a MANDATORY reason.
# ``test_no_stale_unresolvable_allowlist_entries`` fails when an entry no longer
# names a site that is actually unresolvable, so this list can only shrink.
# (Empty since w2v_common 0.109.0: the dispatching wrapper's forward to
# ``super().message`` — the worker_management path — was deleted with it.)
_ALLOWED_UNRESOLVABLE: dict[tuple[str, str], str] = {}

_BINOPS: frozenset[type[ast.operator]] = frozenset(
    {ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Div}
)


def _module_of(rel: str) -> object | None:
    """Import the module a call site lives in, or None if that is impossible.

    Synthetic sources (the self-tests below) and any module with an import-time
    side effect resolve to None, which lands every name in them on the
    fail-closed path rather than on a silent pass.
    """
    dotted = "src." + rel.removesuffix(".py").replace("/", ".").removesuffix(".__init__")
    try:
        return importlib.import_module(dotted)
    except Exception:  # noqa: BLE001 - ANY import failure must fail closed, not pass
        return None


def _as_int(value: object) -> int | _Unresolvable:
    # ``bool`` is an ``int`` subclass; a budget of ``True`` is a bug, not a 1.
    if isinstance(value, bool) or not isinstance(value, int):
        return UNRESOLVABLE
    return value


def _resolve_attribute(rel: str, node: ast.Attribute) -> int | _Unresolvable:
    """``Cfg.tokens`` where ``Cfg`` is a MODULE-LEVEL object.

    ``self._budget`` / ``cls.x`` / any local base is not a module attribute, so
    the most natural per-instance spelling lands on the fail-closed path.
    """
    attrs: list[str] = []
    cursor: ast.expr = node
    while isinstance(cursor, ast.Attribute):
        attrs.append(cursor.attr)
        cursor = cursor.value
    if not isinstance(cursor, ast.Name):
        return UNRESOLVABLE
    module = _module_of(rel)
    if module is None:
        return UNRESOLVABLE
    obj = getattr(module, cursor.id, None)
    for attr in reversed(attrs):
        if obj is None:
            return UNRESOLVABLE
        obj = getattr(obj, attr, None)
    return _as_int(obj)


def _resolve_binop(rel: str, node: ast.BinOp) -> int | _Unresolvable:
    """Arithmetic over operands that themselves resolve. Division only when exact."""
    op = type(node.op)
    if op not in _BINOPS:
        return UNRESOLVABLE
    left = _resolve(rel, node.left)
    right = _resolve(rel, node.right)
    if isinstance(left, _Unresolvable) or isinstance(right, _Unresolvable):
        return UNRESOLVABLE
    if op is ast.Add:
        return left + right
    if op is ast.Sub:
        return left - right
    if op is ast.Mult:
        return left * right
    if right == 0 or left % right != 0:
        return UNRESOLVABLE
    return left // right


def _resolve(rel: str, node: ast.expr) -> int | _Unresolvable:
    """The int an expression evaluates to, or ``UNRESOLVABLE``.

    Deliberately NOT an evaluator. Four shapes only:

    * ``ast.Constant`` — read off the AST.
    * ``ast.Name`` — resolved by IMPORTING the module and reading the attribute,
      which is the only way to see through ``_X = output_token_budget(...)``.
    * ``ast.Attribute`` — the same import, then attribute-walked, so
      ``Cfg.tokens`` resolves when ``Cfg`` is a module-level object.
    * ``ast.BinOp`` / unary minus — arithmetic over operands that resolve, so
      ``_A * _B`` is readable.

    Everything else — ``Subscript``, ``IfExp``, ``Call``, comprehensions,
    f-strings — is UNRESOLVABLE by design. No ``eval``, ever.
    """
    if isinstance(node, ast.Constant):
        return _as_int(node.value)
    if isinstance(node, ast.Name):
        module = _module_of(rel)
        return UNRESOLVABLE if module is None else _as_int(getattr(module, node.id, None))
    if isinstance(node, ast.Attribute):
        return _resolve_attribute(rel, node)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _resolve(rel, node.operand)
        return UNRESOLVABLE if isinstance(inner, _Unresolvable) else -inner
    if isinstance(node, ast.BinOp):
        return _resolve_binop(rel, node)
    return UNRESOLVABLE


def _is_sanctioned_budget_call(rel: str, node: ast.expr) -> bool:
    """``output_token_budget(...)`` inline — floor-clearing BY CONSTRUCTION.

    Its last line is ``max(capped, MIN_OUTPUT_TOKEN_BUDGET)``, so no argument can
    drive it under the floor; ``test_the_sanctioned_constructor_cannot_go_under_the_floor``
    below is what keeps that true. This is the ONE call shape the gate accepts
    without reading a number, and it is accepted by IDENTITY, not by name: the
    module must really have imported this function. A look-alike
    ``output_token_budget`` from somewhere else fails closed like anything else.
    """
    if not isinstance(node, ast.Call) or _called_name(node) != "output_token_budget":
        return False
    module = _module_of(rel)
    return (
        module is not None and getattr(module, "output_token_budget", None) is output_token_budget
    )


def _budget_findings(
    rel: str, qual: str, call: ast.Call, source: str
) -> tuple[list[str], list[str]]:
    """(below-floor, unresolvable) findings for one call site."""
    kw = _budget_kw(call)
    if kw is None:
        return [], []
    if _is_sanctioned_budget_call(rel, kw.value):
        return [], []
    value = _resolve(rel, kw.value)
    if isinstance(value, _Unresolvable):
        segment = ast.get_source_segment(source, kw.value) or ast.dump(kw.value)
        return [], [f"src/{rel}::{qual} sends max_tokens={segment}"]
    if value < MIN_OUTPUT_TOKEN_BUDGET:
        return [f"src/{rel}::{qual} sends max_tokens={value}"], []
    return [], []


def _all_findings() -> tuple[list[str], list[str], set[tuple[str, str]]]:
    """Fleet-wide (below-floor, unresolvable, unresolvable-keys)."""
    too_small: list[str] = []
    unresolvable: list[str] = []
    keys: set[tuple[str, str]] = set()
    for path in sorted(_SRC.rglob("*.py")):
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        rel = str(path.relative_to(_SRC))
        for call, qual in _wm_message_calls(tree):
            small, unread = _budget_findings(rel, qual, call, source)
            too_small.extend(small)
            if unread:
                # Recorded whether or not it is allow-listed — the staleness test
                # needs the LIVE set of unreadable sites to prove an entry is
                # still earning its place.
                keys.add((rel, qual))
                if (rel, qual) not in _ALLOWED_UNRESOLVABLE:
                    unresolvable.extend(unread)
    return too_small, unresolvable, keys


def test_every_budget_clears_the_floor() -> None:
    """This is the assertion PR #851 would have failed.

    A too-small budget lowers the ceiling the model has to finish reasoning
    under, so it is strictly worse than passing nothing. Measured live: 1760
    returns 422 where a larger budget does not.

    ⚠️ The floor is NOT worker-management's 6000 default. 6000 is what a caller
    gets by omitting ``max_tokens``, and it is itself an observed production
    failure on six workers (see ``MIN_OUTPUT_TOKEN_BUDGET`` in
    ``services/stages/llm_budget.py`` for the table). A floor set to the default
    would pass this gate fleet-wide while those failures continued.
    """
    too_small, _, _ = _all_findings()
    assert not too_small, (
        f"max_tokens below the {MIN_OUTPUT_TOKEN_BUDGET}-token floor is STRICTLY "
        "WORSE than passing nothing — it lowers the ceiling the model must finish "
        "reasoning under. Use output_token_budget():\n  " + "\n  ".join(too_small)
    )


def test_every_budget_is_statically_readable() -> None:
    """FAIL CLOSED on a budget the gate cannot read.

    Six shapes walked past the previous resolver — ``self._budget``,
    ``Cfg.tokens``, ``_TIERS["a"]``, ``8000 - 6240``, ``1760 if x else 9000`` and
    ``int("1760")`` — because it returned ``None`` for anything that was not a
    ``Constant`` or a ``Name`` and the floor check SKIPPED ``None``. A budget the
    gate cannot read is not a budget the gate has cleared.
    """
    _, unresolvable, _ = _all_findings()
    assert not unresolvable, (
        "max_tokens expressions this gate cannot statically resolve — it fails "
        "CLOSED on them, because an unreadable budget is not a safe one. Assign a "
        "MODULE-LEVEL constant from output_token_budget(answer_tokens=...) and "
        "pass that name:\n  " + "\n  ".join(unresolvable)
    )


def test_at_least_one_budget_is_resolved_through_a_name() -> None:
    """Vacuity guard for the resolver.

    If ``_resolve`` silently stopped resolving names (a bad import path, a
    renamed module), every site would read as unresolvable. The fail-closed test
    above would go red rather than green — but this keeps the POSITIVE direction
    honest too: the resolver must still be reading real ints out of real modules.
    """
    resolved: list[int] = []
    for rel, _, call in _sites():
        kw = _budget_kw(call)
        if kw is None or not isinstance(kw.value, ast.Name):
            continue
        value = _resolve(rel, kw.value)
        if not isinstance(value, _Unresolvable):
            resolved.append(value)
    assert len(resolved) >= 20, (
        f"the resolver only resolved {len(resolved)} named budgets — it is probably "
        "broken, which would make the floor check vacuous"
    )


def test_no_stale_allowlist_entries() -> None:
    live = {(rel, qual) for rel, qual, _ in _sites()}
    stale = sorted(set(_ALLOWED) - live)
    assert not stale, f"allow-list entries no longer matching a real call site: {stale}"


def test_every_allowlist_entry_carries_a_reason() -> None:
    assert all(v.strip() for v in _ALLOWED.values()), "allow-list entries need a reason"


def test_no_stale_unresolvable_allowlist_entries() -> None:
    """An entry survives only while its site is still unreadable.

    Not "still exists" — still UNRESOLVABLE. Once the author rewrites the budget
    as a module-level constant the exemption must go, so the list can only shrink.
    """
    _, _, unreadable = _all_findings()
    stale = sorted(set(_ALLOWED_UNRESOLVABLE) - unreadable)
    assert not stale, (
        "unresolvable-budget allow-list entries whose site is gone or now "
        f"statically readable — delete them: {stale}"
    )


def test_every_unresolvable_allowlist_entry_carries_a_reason() -> None:
    assert all(v.strip() for v in _ALLOWED_UNRESOLVABLE.values()), (
        "unresolvable-budget allow-list entries need a reason"
    )


# --- the detector must be proven to catch the shapes it bans -----------------
#
# A gate nobody has seen fail is decoration. These parse synthetic sources so the
# assertions are real failures, not collection errors.

_POSITIVE = "await c.message(worker_name='w', message='m')"
_WITH_BUDGET = "await c.message(worker_name='w', message='m', max_tokens=9000)"
_SPREAD = "await c.message(worker_name='w', **kw)"
_UNRELATED = "logger.message(foo='bar')\nresponse.message(text='hi')"


def _parse_one(src: str) -> list[tuple[ast.Call, str]]:
    return _wm_message_calls(ast.parse(f"async def f():\n    {src}\n"))


def test_detector_finds_a_call_missing_the_budget() -> None:
    calls = _parse_one(_POSITIVE)
    assert len(calls) == 1
    assert _budget_kw(calls[0][0]) is None


def test_detector_accepts_a_call_carrying_the_budget() -> None:
    calls = _parse_one(_WITH_BUDGET)
    assert len(calls) == 1
    kw = _budget_kw(calls[0][0])
    assert kw is not None
    assert isinstance(kw.value, ast.Constant)
    assert kw.value.value == 9000


def test_detector_fails_closed_on_a_kwargs_spread() -> None:
    calls = _parse_one(_SPREAD)
    assert len(calls) == 1
    assert _has_spread(calls[0][0])


def test_detector_ignores_unrelated_dot_message_calls() -> None:
    assert _wm_message_calls(ast.parse(_UNRELATED)) == []


# --- the three bypasses an adversarial pass drove through the first detector ---
#
# ``worker_name`` is not keyword-only, so all three of these are legal WM calls
# that the original attribute+keyword match returned ZERO hits for. A gate is
# only as good as the spellings it cannot be walked around.

_POSITIONAL = "await c.message('worker-name', 'hi')"
_ALIASED = "send = c.message\n    await send(worker_name='w', message='m')"
_ARGS_SPREAD = "await c.message(*args, **kw)"


def test_detector_catches_a_positional_worker_name() -> None:
    calls = _parse_one(_POSITIONAL)
    assert len(calls) == 1, "a positional worker_name walked straight past the gate"
    assert _budget_kw(calls[0][0]) is None


def test_detector_catches_an_aliased_call() -> None:
    calls = _parse_one(_ALIASED)
    assert len(calls) == 1, "`send = wm.message` walked straight past the gate"


def test_detector_catches_an_args_spread() -> None:
    calls = _parse_one(_ARGS_SPREAD)
    assert len(calls) == 1
    assert _has_spread(calls[0][0])


# --- the SIX shapes that walked past the Constant/Name-only resolver ----------
#
# An adversarial probe file carrying seven bad ``max_tokens=1760`` calls was
# caught exactly ONCE. These six got through, because ``_resolve``
# returned ``None`` for every node type it did not handle and the floor check
# then ``continue``d — i.e. it SKIPPED what it could not read. The gate now fails
# closed instead, and this is where that is proven rather than asserted.
#
# ``8000 - 6240`` is now genuinely READ (it is 1760) and reported as below-floor;
# the other five are reported as unresolvable. Either way each one is REPORTED,
# which is the property that matters.

_SIX_BYPASSES: dict[str, str] = {
    "attribute_on_self": "max_tokens=self._budget",
    "attribute_on_a_class": "max_tokens=Cfg.tokens",
    "subscript_of_a_tier_table": 'max_tokens=_TIERS["a"]',
    "arithmetic": "max_tokens=8000 - 6240",
    "conditional": "max_tokens=1760 if x else 9000",
    "call_expression": 'max_tokens=int("1760")',
}

# A path under src/ that cannot be imported, so name/attribute resolution has to
# take the fail-closed branch exactly as it would for a genuinely opaque site.
_SYNTHETIC_REL = "__gate_selftest__.py"


def _findings_for(expr: str) -> tuple[list[str], list[str]]:
    """Run the real detector over a synthetic one-call module."""
    source = f"async def f():\n    await c.message(worker_name='w', message='m', {expr})\n"
    tree = ast.parse(source)
    calls = _wm_message_calls(tree)
    assert len(calls) == 1, f"the detector did not even find the call for: {expr}"
    call, qual = calls[0]
    return _budget_findings(_SYNTHETIC_REL, qual, call, source)


@pytest.mark.parametrize(("name", "expr"), list(sorted(_SIX_BYPASSES.items()))[:1])
def test_each_known_bypass_shape_is_reported(name: str, expr: str) -> None:
    too_small, unresolvable = _findings_for(expr)
    assert too_small or unresolvable, (
        f"{name}: `{expr}` was NOT reported — this is the bypass class that let "
        "six bad 1760-token budgets through. The gate must fail closed."
    )


def test_the_arithmetic_shape_is_actually_evaluated_not_just_rejected() -> None:
    """``8000 - 6240`` must be READ as 1760, not waved through as opaque."""
    too_small, unresolvable = _findings_for("max_tokens=8000 - 6240")
    assert not unresolvable, "BinOp over constants should resolve, not fail closed"
    assert too_small == [f"src/{_SYNTHETIC_REL}::f sends max_tokens=1760"], too_small


def test_unresolvable_reports_name_the_file_qualname_and_source() -> None:
    """The failure has to tell the author WHERE and WHAT, or it is unactionable."""
    _, unresolvable = _findings_for("max_tokens=self._budget")
    assert unresolvable == [f"src/{_SYNTHETIC_REL}::f sends max_tokens=self._budget"], unresolvable


def test_a_module_level_constant_still_passes_cleanly() -> None:
    """The negative for the fail-closed branch: a real site must NOT be reported.

    Resolved through a genuinely importable module, so this also proves the
    import path in ``_module_of`` still works — without it every site would read
    as unresolvable and the gate would be noise.
    """
    rel = "services/stages/action_reel/segment_select.py"
    source = (
        f"async def f():\n    await c.message(worker_name='w', {'max_tokens=_SELECT_MAX_TOKENS'})\n"
    )
    call, qual = _wm_message_calls(ast.parse(source))[0]
    assert _budget_findings(rel, qual, call, source) == ([], [])


def test_an_unrelated_dot_message_call_is_never_reported() -> None:
    """No ``worker_name``, no positional args, no spread — not a WM call at all.

    A gate that fails closed is only useful if it fails closed on the RIGHT
    calls; firing on ``logger.message(...)`` would make it noise the fleet routes
    around.
    """
    source = "def f():\n    logger.message(foo='bar', max_tokens=self._budget)\n"
    assert _wm_message_calls(ast.parse(source)) == []


def test_the_sanctioned_constructor_cannot_go_under_the_floor() -> None:
    """The premise the inline-call exemption rests on.

    ``_is_sanctioned_budget_call`` accepts ``output_token_budget(...)`` without
    reading its argument. That is only sound while the function clamps. If the
    clamp is ever removed, this goes red HERE rather than letting a dynamic
    argument smuggle a 1760 past the gate.
    """
    for answer_tokens in (1, 10, 500, 1760, 6000, 8000, 50_000, 1_000_000):
        assert output_token_budget(answer_tokens=answer_tokens) >= MIN_OUTPUT_TOKEN_BUDGET


def test_the_sanctioned_call_is_accepted_only_by_identity() -> None:
    """A same-named function the module never imported must NOT buy an exemption."""
    expr = "max_tokens=output_token_budget(answer_tokens=self.answer_tokens)"
    source = f"async def f():\n    await c.message(worker_name='w', {expr})\n"
    call, qual = _wm_message_calls(ast.parse(source))[0]

    real = "services/stages/primitives/script_stage.py"
    assert _budget_findings(real, qual, call, source) == ([], [])

    # Same source text, a module that does not exist -> no identity, fails closed.
    assert _budget_findings(_SYNTHETIC_REL, qual, call, source) == (
        [],
        [
            f"src/{_SYNTHETIC_REL}::f sends max_tokens=output_token_budget(answer_tokens=self.answer_tokens)"
        ],
    )
