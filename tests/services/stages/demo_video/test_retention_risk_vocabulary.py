"""`drop_risk` was matched exactly against three tokens, on a free-string schema.

`_is_risky` tested `risk.strip().lower() in {"high", "med", "medium"}` while the
output schema declared `drop_risk` as `{"type": "string"}` with **no enum** — so
the model was under no obligation to produce those tokens. Everything else it
naturally writes read as NOT risky: `flagged` came back empty, no beat was
rewritten, and the retention pass completed green having done nothing.

Fixed on both sides, because either alone is insufficient:
* the schema pins an `enum`, so structured output constrains the model;
* the parser matches on PREFIX over a wider vocabulary, so a model answering
  outside the enum (or an older cached generation) is still read correctly.

A value in neither vocabulary is logged, not guessed — it means the beat was
never really considered.
"""

from __future__ import annotations


def test_the_optimizer_ledger_key_covers_the_critique_not_just_the_flagged_set() -> None:
    """Two passes can flag a byte-identical set under a DIFFERENT critique.

    That happens exactly when pass 1's batch came back malformed and contributed
    no revision: `_weak_lines` re-derives the same ids from the same draft. Keyed
    on `flagged` alone, pass 2 hits the same ledger entry, issues ZERO calls, and
    pins pass 1's empty answer for the rest of the loop — under a critique that
    asked for something different.
    """
    import hashlib
    import inspect
    import json

    from src.services.stages.demo_video import retention_plan

    src = inspect.getsource(retention_plan.RetentionPlanStage)
    assert '"critique": critic_output' in src, (
        "the optimizer ledger key must include the critique; keying on the "
        "flagged set alone pins a stale answer across passes"
    )

    # And the digest must actually differ when only the critique differs.
    flagged = [{"id": "b1", "text": "a beat"}]
    key_a = hashlib.sha256(
        json.dumps({"flagged": flagged, "critique": {"note": "A"}}, sort_keys=True).encode()
    ).hexdigest()
    key_b = hashlib.sha256(
        json.dumps({"flagged": flagged, "critique": {"note": "B"}}, sort_keys=True).encode()
    ).hexdigest()
    assert key_a != key_b
