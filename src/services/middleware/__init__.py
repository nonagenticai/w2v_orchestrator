"""Stage-execution middleware chain (metrics + timeout).

Retry and circuit-breaking are intentionally NOT here: retry lives at the call
boundary (``stage_call``/``stage_gather``/``BaseServiceClient``) where per-call
idempotency is known, and a process-global breaker gave no cross-pod protection.
See ``DAGExecutor._build_chain``.
"""
