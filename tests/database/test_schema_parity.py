"""create_all ↔ alembic schema-parity gate for the RLS + resolver-function surface.

``src/database/models.py`` hand-attaches three DDL registrars via SQLAlchemy
``after_create`` events (``register_rls_ddl``, ``register_run_functions_ddl``,
``register_admin_functions_ddl``). Their whole job is to make a
``Base.metadata.create_all()`` schema (tests/dev) carry the SAME RLS policies and
SECURITY DEFINER resolver functions that the Alembic migrations build in prod.

That parity is otherwise MANUAL: a policy/function added to a migration but not
its registrar (or vice-versa) silently diverges the test schema from prod, and
nothing catches it until a prod-only failure. This test closes the gap by
comparing the RLS + function surface of both builds.

The two builds themselves come from the session-scoped ``parity_schemas``
fixture (``tests/conftest.py`` + ``tests/support/schema_probe.py``), which is
shared with ``tests/contracts/test_stage_status_check_constraint_parity.py`` —
one chain replay serves both. An unreachable database FAILS via the conftest
policy rather than skipping.

The compare here is on ``pg_policies`` (name, table, cmd, USING/WITH CHECK
predicates) and the app's ``pg_proc`` rows (name, identity args, SECURITY DEFINER
flag, volatility, body) — the drift risk the registrars exist to cover. Schema
names are normalized out of predicates/bodies so only the *shape* is compared.
The CHECK-constraint surface is covered by the contract test named above.
"""

from __future__ import annotations

from tests.support.schema_probe import ParityProbe, snapshot_functions, snapshot_policies


def test_create_all_matches_alembic_rls_and_functions(parity_schemas: ParityProbe) -> None:
    """The RLS policy + resolver-function surface must be identical whether the
    schema is built by ``create_all`` (registrars) or ``alembic upgrade head``.

    This is the anti-drift gate: adding an RLS policy or SECURITY DEFINER
    function to a migration without its ``src.database.rls`` registrar (or the
    reverse) makes these snapshots diverge and fails the test here, instead of
    as a prod-only permission/isolation failure.
    """
    probe = parity_schemas
    with probe.engine.connect() as conn:
        ca_policies = snapshot_policies(conn, probe.create_all)
        al_policies = snapshot_policies(conn, probe.alembic)
        ca_funcs = snapshot_functions(conn, probe.create_all)
        al_funcs = snapshot_functions(conn, probe.alembic)

    # Guard against a silent both-empty pass (e.g. a build that no-oped):
    # the registrars emit real policies + functions, so both must be present.
    assert ca_policies, "create_all produced no RLS policies — build no-oped?"
    assert ca_funcs, "create_all produced no resolver functions — build no-oped?"

    # Presence parity first (the headline requirement): a policy/function in
    # one schema but not the other is drift.
    assert set(ca_policies) == set(al_policies), (
        "RLS policy set diverges between create_all and alembic.\n"
        f"  only in create_all: {sorted(set(ca_policies) - set(al_policies))}\n"
        f"  only in alembic:    {sorted(set(al_policies) - set(ca_policies))}"
    )
    assert set(ca_funcs) == set(al_funcs), (
        "resolver-function set diverges between create_all and alembic.\n"
        f"  only in create_all: {sorted(set(ca_funcs) - set(al_funcs))}\n"
        f"  only in alembic:    {sorted(set(al_funcs) - set(ca_funcs))}"
    )

    # Then predicate/body/attribute parity for every shared object.
    for policy_key in sorted(ca_policies):
        assert ca_policies[policy_key] == al_policies[policy_key], (
            f"RLS policy {policy_key} predicate/cmd differs:\n"
            f"  create_all: {ca_policies[policy_key]}\n"
            f"  alembic:    {al_policies[policy_key]}"
        )
    for func_key in sorted(ca_funcs):
        assert ca_funcs[func_key] == al_funcs[func_key], (
            f"resolver function {func_key} definition differs:\n"
            f"  create_all: {ca_funcs[func_key]}\n"
            f"  alembic:    {al_funcs[func_key]}"
        )
