"""Single source of truth for the LLM-stage timeout + retry budget.

Every LLM text stage (script, growth-bundle, analyze, cast, segment, outline,
the visual directors, sound director, ...) routes WM -> inference-router -> pgx
and must wait out the same downstream read budget rather than cut a
still-working call off early. These accessors replace a scatter of hardcoded
per-stage ``_LLM_TIMEOUT_SECONDS`` literals so the whole ladder moves together
from one config field (``PipelineSettings.llm_budget_seconds`` /
``llm_retries``).

Read at stage construction (``__init__``), never at import: ``get_settings()``
is an ``lru_cache`` singleton, and reading it lazily lets env / test overrides
applied after import still take effect.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextvars import ContextVar

import httpx

from src.config import LLM_BUDGET_SECONDS as _LLM_BUDGET_SECONDS
from src.config import get_settings
from src.services.stages.upstream import TransientUpstreamError


def llm_budget_seconds() -> float:
    """Per-attempt wall-clock budget for an LLM stage's WMClient + stage_call."""
    return get_settings().pipeline.llm_budget_seconds


def client_timeout_for(effective_timeout_seconds: float) -> float:
    """HTTP read timeout for a ``WMClient`` used at a site with its OWN per-attempt clock.

    ⚠️ THIS is the accessor for a site that SUBDIVIDES the budget. Pass the number that
    site hands ``stage_call`` / ``asyncio.timeout``, not the global budget.

    Why the distinction is load-bearing: the ordering invariant is per-SITE, and a rung
    derived from ``llm_budget_seconds()`` is INERT anywhere the site runs on a smaller
    clock. Worked example — ``audiobook/qa.py`` slices ``llm_budget * 0.7`` across N judge
    batches, so a batch's effective bound is ~665/N s while a globally-derived rung is
    930 s: above the site's own bound, not below it.

    ⚠️ AND THE INVARIANT IS NOT ACHIEVABLE FOR EVERY SITE FROM ONE CLIENT.
    ``WMClient.message`` (w2v_common/clients/wm_client.py) takes NO per-call timeout — the
    read timeout is fixed at construction (``self.timeout``, :106). A stage that shares one
    ``self._wm`` across two different effective timeouts (``audiobook_qa`` does: the
    per-batch judge slice AND the full-budget evaluator-optimizer critic) therefore cannot
    satisfy ``client < effective`` at both. Use this helper wherever a site builds its own
    client for a subdivided slice; where it cannot, the guarantee that has to hold instead
    is that the SITE's own clamp fires first — see
    ``test_the_judge_batch_clamp_fires_before_the_client_rung``.

    Sizing, the ``t/2 <= result < t`` guarantee and the arithmetic behind the delta live on
    ``config.LLM_CLIENT_TIMEOUT_DELTA_S``; the boot-time bounds are enforced by
    ``PipelineSettings._validate_llm_timeout_ladder``.
    """
    return get_settings().pipeline.client_timeout_for(effective_timeout_seconds)


def llm_client_timeout_seconds() -> float:
    """HTTP read timeout for a PLAIN LLM stage — one whose per-attempt timeout is the
    global ``llm_budget_seconds()``.

    Exactly ``client_timeout_for(llm_budget_seconds())``; it exists so the ~27 plain sites
    read as one line instead of repeating the derivation.

    ⚠️ Do NOT pass ``llm_budget_seconds()`` itself to ``WMClient(timeout=...)`` — the
    client and ``stage_call`` would hold the SAME number and the stage clock, which starts
    first, wins every tie. ``tests/services/blueprints/test_llm_timeout_ladder.py`` has an
    AST guard that fails on any such construction.

    ⚠️ AND DO NOT REPEAT THE ORIGINAL OVERCLAIM. The tie did NOT make the in-stage retry
    dead code: ``stage_call`` catches its own ``TimeoutError`` and retries it
    (w2v_common/stages/execution.py:110-122); only the last attempt raises
    ``StageTimeoutError``. "all 2 attempts timed out" is stage_call reporting that it
    retried. This rung buys a typed ``httpx.ReadTimeout`` and a socket bound for unwrapped
    call paths — defence in depth, not a resurrected retry.
    """
    return get_settings().pipeline.llm_client_timeout_seconds


def llm_retries() -> int:
    """stage_call retries for LLM stages (attempts = retries + 1)."""
    return get_settings().pipeline.llm_retries


# The smallest per-ATTEMPT slice worth handing ``stage_call``. Below this a WM
# round-trip cannot plausibly complete, so splitting a slice further just converts
# one real attempt into N guaranteed timeouts. ``stage_call`` itself rejects
# anything under 1.0s (``_MIN_TIMEOUT_SECONDS``), which is the hard floor
# ``attempt_plan`` never returns below.
_MIN_ATTEMPT_SECONDS = 5.0
_STAGE_CALL_MIN_TIMEOUT = 1.0

# ``stage_call``'s backoff defaults, mirrored so the ladder's SLEEPS are budgeted
# too. They are real wall clock inside the caller's slice: a ladder sized only by
# its attempts overruns by 1 + 2 + ... seconds and gets cancelled on the last
# attempt — which is precisely the attempt that would have succeeded. The jitter
# multiplier is the HIGH end (``1 + backoff_jitter``), because the bound has to
# hold on the unlucky draw, not the average one.
_BACKOFF_INITIAL_SECONDS = 1.0
_BACKOFF_FACTOR = 2.0
_BACKOFF_JITTER_HIGH = 1.25


def _backoff_allowance(retries: int) -> float:
    """Worst-case total seconds ``stage_call`` sleeps between ``retries`` retries."""
    nominal = sum(_BACKOFF_INITIAL_SECONDS * _BACKOFF_FACTOR**i for i in range(retries))
    return nominal * _BACKOFF_JITTER_HIGH


def attempt_plan(budget_seconds: float, retries: int) -> tuple[float, int]:
    """Split ONE call's wall-clock slice into ``(per_attempt_seconds, retries)``.

    ``stage_call``'s ``timeout_seconds`` is PER ATTEMPT, not a total: passing the
    whole slice with ``retries=2`` authorises up to 3x the slice plus backoff. That
    is fine for a stage that owns its own clock, and WRONG wherever the slice was
    carved out of a deadline SIBLINGS also draw on (a per-batch share of a critic
    budget, an inner ladder nested inside an outer ``stage_call``) — there, retries
    would be paid for out of someone else's time.

    So the retry ladder is sized to fit INSIDE the slice: the per-attempt budget is
    what remains of the slice once the BACKOFF SLEEPS are paid for, divided by the
    attempts (``attempts * per_attempt + worst-case backoff <= slice``), and the
    retry count is reduced (to zero if need be) rather
    than let an attempt drop under ``_MIN_ATTEMPT_SECONDS``, because three attempts
    too short to finish are worth less than one that can.

    The returned per-attempt value is floored at ``stage_call``'s own 1.0s minimum,
    so on a slice smaller than that the caller gets ONE attempt nominally longer
    than its slice — harmless, because a caller in that position is expected to keep
    its own hard ``asyncio.timeout(slice)`` around the whole ladder. That outer bound
    is what actually honours the shared deadline; this function only decides how the
    slice is spent.
    """
    plan_retries = max(0, retries)
    while plan_retries > 0 and _usable(budget_seconds, plan_retries) < _MIN_ATTEMPT_SECONDS:
        plan_retries -= 1
    per_attempt = max(_STAGE_CALL_MIN_TIMEOUT, _usable(budget_seconds, plan_retries))
    return per_attempt, plan_retries


# Set for the duration of ONE attempt by ``downshift_reasoning``; read by
# ``reasoning_for_attempt()``. A ContextVar rather than an argument because the
# evaluator-optimizer closures cannot change their signature — see both docstrings.
_ATTEMPT_REASONING: ContextVar[bool | None] = ContextVar("llm_attempt_reasoning", default=None)


def downshift_reasoning[R](
    make_attempt: Callable[[bool | None], Awaitable[R]],
) -> Callable[[], Awaitable[R]]:
    """Wrap a WM attempt factory so the FIRST attempt reasons and every RETRY does not.

    ``stage_call`` retries by calling the factory again with the same budget and the
    same request. For the failure mode measured in production on 2026-08-20 that retry
    cannot succeed: with chain-of-thought ON the model spent the whole 950s deadline
    thinking and returned NOTHING — reproduced twice on the production path, on both
    output schemas — while the identical call with reasoning OFF answered in 34-67s
    (see #873/#883). Re-issuing an identical call against an identical budget is a
    retry that is *guaranteed* to fail the same way; it only converts one dead stage
    into a dead stage that took twice as long.

    So the ladder varies the one input that was measured to decide the outcome. The
    first attempt is passed ``None`` — NOT ``True`` — so the happy path is
    byte-identical to not calling this helper at all: WM keeps whatever its own
    default is, and no stage silently gains reasoning it did not have. Only attempt 2
    onward passes ``False``.

    This is deliberately NOT a blanket reasoning-off sweep. #883 turned reasoning off
    outright on the two stages where thinking-off output was MEASURED, and left ~19
    others alone because chain-of-thought plausibly earns its cost on a critique and
    the quality of those stages thinking-off is unmeasured. This helper keeps that
    open question closed on the happy path and only spends the answer when the
    alternative is a dead paid run.

    Usage — the attempt takes the flag and forwards it to ``WMClient.message``::

        async def _attempt(enable_reasoning: bool | None) -> Out:
            resp = await self._wm.message(..., enable_reasoning=enable_reasoning)
            ...

        await stage_call(
            downshift_reasoning(lambda think: guarded(lambda: _attempt(think))),
            timeout_seconds=..., retries=..., retry_on=LLM_RETRY_ON, ...
        )

    The returned factory is single-use per ladder: it carries the attempt counter, so
    build it INSIDE the call that hands it to ``stage_call`` and never cache it on
    ``self`` — a factory reused across two stage invocations would start the second
    one already downshifted.
    """
    attempted = False

    def factory() -> Awaitable[R]:
        nonlocal attempted
        # None on the first attempt, False on every retry. `not attempted` would
        # pass True and change the happy path; that is the bug this avoids.
        enable_reasoning: bool | None = None if not attempted else False
        attempted = True

        async def _run() -> R:
            # Bind the ContextVar for the DURATION of this attempt, so a WM call
            # nested too deep to take the flag as an argument (an evaluator-optimizer
            # critic closure, whose signature is fixed by loop_engine) can read it via
            # `reasoning_for_attempt()`. Reset in `finally` so the flag never outlives
            # its attempt and leak into a sibling stage on the same task.
            token = _ATTEMPT_REASONING.set(enable_reasoning)
            try:
                return await make_attempt(enable_reasoning)
            finally:
                _ATTEMPT_REASONING.reset(token)

        return _run()

    return factory


def reasoning_for_attempt() -> bool | None:
    """The reasoning flag for the attempt currently in flight.

    ``None`` outside a :func:`downshift_reasoning` ladder and on its first attempt;
    ``False`` on every retry. Pass this to ``WMClient.message(enable_reasoning=...)``
    at a call site that cannot take the flag as an argument because its signature is
    owned by someone else — the ``critic_fn`` / ``optimizer_fn`` closures handed to
    ``shared/loop_engine.run_evaluator_optimizer`` are the whole reason it exists.

    Prefer the explicit parameter wherever the site owns its own factory: an argument
    is visible in the signature, a ContextVar is not.
    """
    return _ATTEMPT_REASONING.get()


def _usable(budget_seconds: float, retries: int) -> float:
    """Per-attempt seconds left once the backoff sleeps are paid for."""
    return (budget_seconds - _backoff_allowance(retries)) / (retries + 1)


def course_outline_split() -> bool:
    """Feature flag: split course_outline into structure + objective-batch calls."""
    return get_settings().pipeline.course_outline_split


def timing_director_model_tier() -> str | None:
    """Optional model tier for the reconcile_timeline relief loop's WM calls.

    None (default) ⇒ the timing-director + beat-rewriter workers fall through the
    inference-router default_chain to the small local model. An inference-router
    tier alias (``tier-medium`` / ``tier-high``) routes them to a more capable
    model — flipped in GitOps, not code (see ``PipelineSettings``).
    """
    return get_settings().pipeline.timing_director_model_tier


# The retry_on allow-list for every LLM stage's stage_call. It is an ALLOW-list on
# purpose: PermanentUpstreamError (see stages/upstream.py) is absent, deliberately does
# not subclass RuntimeError/StageError, and therefore can never be swallowed here — a
# WM 402 "configure your LLM provider" fails the stage instantly with that sentence
# instead of burning the budget and reporting "retries exhausted".
LLM_RETRY_ON: tuple[type[BaseException], ...] = (
    # Classified by upstream.guarded(): the retryable half of an upstream HTTP failure
    # (5xx blip, 429, 408, ...).
    TransientUpstreamError,
    # Connect/read/write/pool errors — never reached the app.
    httpx.TransportError,
    # SAFETY NET, not the mechanism. WMClient.message
    # (w2v_common/clients/wm_client.py:189) calls client.post directly and does NOT go
    # through BaseServiceClient._request, so _RETRYABLE_STATUS_CODES' 3x internal retry
    # never applies to WM /message — stage_call's retry_on is the ONLY retry a WM 5xx
    # gets. A factory routed through guarded() re-raises as Transient/Permanent BEFORE
    # stage_call ever sees an HTTPStatusError, so this entry only catches an un-guarded
    # call site and degrades it to today's behaviour rather than failing it instantly.
    httpx.HTTPStatusError,
    # ValueError is LOAD-BEARING: primitives/script_stage.py:169-176 deliberately raises
    # ValueError INSIDE the retry budget to re-prompt a short/malformed LLM draw.
    # pydantic.ValidationError and json.JSONDecodeError both subclass ValueError and
    # ride along here.
    ValueError,
    TypeError,
    KeyError,
    IndexError,
    # RuntimeError is deliberately EXCLUDED: it is poll_job's terminal-job-failure signal
    # (w2v_common/clients/base.py:738). Excluding it is what stops a dead job being
    # re-polled until the budget is gone.
)


# ---------------------------------------------------------------------------
# Output-token budgets for WM calls
# ---------------------------------------------------------------------------
#
# ``WMClient.message(max_tokens=...)`` is NOT an optimisation knob. Omitting it
# makes worker-management substitute ``DEFAULT_SELF_HOSTED_MAX_TOKENS = 6000``
# (worker_management/src/worker/agents/pydantic_ai_agent.py:534), and a call that
# runs out of that budget dies TERMINALLY: WM answers 422, 422 is absent from
# ``w2v_common.clients.base._RETRYABLE_STATUS_CODES``, and ``upstream.guarded``
# classifies it ``PermanentUpstreamError`` — so the stage fails on attempt 1 and
# the whole run fails with it.
#
# ⚠️ THE MECHANISM IS NOT WHAT IT LOOKS LIKE, AND AN EARLIER FIX GOT IT BACKWARDS.
#
# The error text — "Model token limit (N) exceeded before any response was
# generated" — reads like a pre-flight rejection, and PR #851 acted on that
# reading: it LOWERED ``max_tokens`` to 1760 on ``segment_select`` believing WM
# reserved 6000 against a small context window. That is wrong twice over. The
# served model is ``Qwen/Qwen3.8-27B-FP8`` with ``--max-model-len=131072``, so no
# realistic prompt plus 6000 can overflow it; and the raise is in
# ``pydantic_ai/_agent_graph.py:1100-1113``, which runs AFTER generation:
#
#     is_thinking_only = ... all(isinstance(p, ThinkingPart) for p in parts)
#     if is_empty or is_thinking_only:
#         # Don't retry if the token limit was exceeded, possibly during thinking.
#         if self.model_response.finish_reason == 'length':
#             raise UnexpectedModelBehavior('Model token limit (...) exceeded ...')
#
# vLLM runs this model with ``--reasoning-parser qwen3``. Its thinking tokens are
# charged against ``max_tokens`` and arrive as ``ThinkingPart``s. So the failure
# is: THE MODEL SPENT THE WHOLE BUDGET THINKING AND NEVER EMITTED THE ANSWER.
# The budget must cover reasoning FIRST and the answer second.
#
# Measured live on 2026-08-19 against a realistic segment-selection prompt, one
# worker, one prompt, only ``max_tokens`` varying:
#
#     max_tokens=  500 -> 422 after  43.5 s
#     max_tokens= 1760 -> 422 after 131.5 s   <-- PR #851's constant
#     max_tokens= 6000 -> no 422 (ran past a 900 s client cap)
#     max_tokens=10000 -> no 422 (ran past a 400 s client cap)
#
# And against the REAL production worker, on the deployed value, with the real
# ``_OUTPUT_SCHEMA`` and ``_SYSTEM_PROMPT`` and a realistic 2-source batch — i.e.
# exactly what a live action_reel run sends:
#
#     worker action-reel-segment-selector, max_tokens=1760 -> 422 after 153.3 s
#
# That is not a theory about the deployed code. It IS the deployed code, and it
# is why this correction is not a refactor.
#
# ⚠️ Read the last two honestly: clearing the floor removes the 422, but it does
# NOT make a hard prompt fast. That probe was deliberately adversarial (a 6-clip
# constraint-satisfaction ask) and simply kept thinking. Above the floor the
# binding constraint stops being the token budget and becomes WALL CLOCK — which
# is what ``_CHAIN_DEADLINE_TOKENS`` caps, and why bounding the INPUT still
# matters: a budget big enough to dodge the 422 can convert a fast, loud failure
# into a slow one the router abandons.
#
# That is why the direction is UP, and why the gate below is a FLOOR. A budget
# sized to the answer alone is not "tight", it is broken.
#
# ⚠️ WHAT THIS DOES AND DOES NOT BUY — STATE IT HONESTLY.
#
# It does NOT make a slow call fast. Under the fleet load measured on 2026-08-19
# (inference-router latencies of 38-94 s per call) NO budget completes this
# stage's 2-source prompt inside 900 s — a benign, smoothly-structured prompt
# timed out at 6800 just as an adversarial one did. That is the box being
# saturated, not a property of the budget.
#
# What it DOES buy is the difference between terminal and retryable, and that is
# worth having on its own:
#
#   422 "Model token limit ..."  -> PERMANENT_STATUSES (upstream.py:21)
#                                -> PermanentUpstreamError -> stage dies attempt 1
#   500 "Agent ... timed out"    -> matches NO PERMANENT_5XX_MARKERS
#                                -> TransientUpstreamError -> in LLM_RETRY_ON,
#                                   so stage_call retries it
#
# Under the old budget a loaded box killed the run outright. Under this one the
# same conditions produce a retry, which succeeds when load drops. Do not
# overclaim it as "the stage now works" — claim what is measured.
#
# Reasoning headroom. Worker-management raised its own default 4000 -> 6000 on
# 2026-08-09 against three unrelated workers dying at exactly this error
# (pydantic_ai_agent.py:490-506) — i.e. 6000 was chosen as the value that clears
# real thinking. We size headroom to that same number rather than inventing one.
_REASONING_HEADROOM_TOKENS = 6000

# Wall-clock ceiling, not a token ceiling: the largest budget a call can ACTUALLY
# spend before something upstream kills it. Above this, extra budget is not extra
# capability — it is just a slower way to fail.
#
# ⚠️ THREE EARLIER VERSIONS OF THIS CONSTANT WERE WRONG, EVERY ONE OF THEM FOR THE
# SAME REASON: a hardcoded number derived from an input that had gone stale.
#
# (1) 11000, from the "~37 tok/s" in worker_management's comment. Contradicted by
#     ~3x. Three timings on the live box, all calls that ran to their budget:
#         500 tok /  43.5 s = 11.5 tok/s
#        1760 tok / 131.5 s = 13.4 tok/s
#        1760 tok / 153.3 s = 11.5 tok/s  (real action-reel-segment-selector)
#     37 tok/s was measured on an IDLE box, and on a model since retired.
#
# (2) 7760 (PR #854), from that same 37 tok/s against "inference-router's 300 s
#     chain deadline" — a code DEFAULT the cluster overrides to 1200.
#
# (3) 6800, from ``600 s * 11.5``, where 600 was worker-management's
#     ``agent_run_timeout_s``. Stale within hours: gitops ``a873d09c2``
#     (2026-08-19 07:54) raised ``AGENT_RUN_TIMEOUT_S`` to **1200**, verified live
#     with ``kubectl get deploy -o yaml``. Its supporting probe — 7500 returning
#     ``500 "Agent timed out"`` — was measured BEFORE that rollout, so it no
#     longer bounds anything.
#
# So the ceiling is not written down a fourth time. It is DERIVED from the same
# per-call wall clock this stage hands to ``stage_call``:
#
#     tokens = LLM_BUDGET_SECONDS * _MEASURED_TOKENS_PER_SEC = 950 * 11.5 = 10925
#
# ⚠️ ``LLM_BUDGET_SECONDS`` is the CODE DEFAULT (``src/config.py``), not
# ``get_settings().pipeline.llm_budget_seconds``. It has to be: these budgets are
# module-level constants evaluated at IMPORT, and reading settings there would
# freeze whichever env was set first. The consequence is explicit rather than
# glossed: **setting ``PIPELINE_LLM_BUDGET_SECONDS`` in GitOps moves the wall
# clock and leaves this ceiling frozen.** That key is deliberately NOT set in
# GitOps for exactly this reason — the code default is the live value — and
# ``test_the_token_ceiling_tracks_the_code_default_clock`` pins the derivation so
# a future GitOps override cannot silently desynchronise the two. If you ever do
# need to set it there, change this constant in the same commit.
#
# What bounds it from ABOVE, verified live rather than read off a default:
# worker-management ``AGENT_RUN_TIMEOUT_S=1200`` / ``IR_HTTP_READ_TIMEOUT_S=1200``
# and inference-router ``INFERENCE_ROUTER_CHAIN_DEADLINE_SECONDS=1200``. 950 s of
# orchestrator budget sits 250 s under all three, so the ORCHESTRATOR's own clock
# is what fires first — a clean ``StageTimeoutError``, not an upstream abort.
#
# ⚠️ And this bounds ONE call. Stages that loop (``segment_select`` runs
# ceil(sources/2) calls, ``footage_critique`` ceil(shots/5)) spend it PER CALL,
# and although those loops are now bounded-parallel a large enough corpus still
# needs more than one wave. In practice mean generation is ~143 tokens
# (pydantic_ai_agent.py:530-533), so the full budget is the rare runaway tail —
# but the tail is exactly what these numbers govern.
_MEASURED_TOKENS_PER_SEC = 11.5
_CHAIN_DEADLINE_TOKENS = int(_LLM_BUDGET_SECONDS * _MEASURED_TOKENS_PER_SEC)

# Mirrors worker_management's SAFE_MAX_TOKENS (pydantic_ai_agent.py:489), the
# FP4-MoE kernel-crash guard from the 2026-05-13 incident.
#
# ⚠️ CORRECTED: an earlier version of this comment said "WM clamps silently".
# That is true only of the paths that BYPASS the API validator
# (pydantic_ai_agent.py:553 clamps a worker-CONFIG-sourced value). On the
# per-message path this module actually uses, WM REJECTS instead:
# api/workers/models.py:291 raises when `v < 1 or v > 16384`, which reaches the
# caller as a terminal 422 — the same failure class this whole file exists to
# stop. So going over is not "you quietly get 16384", it is "the paid stage
# dies". The cap below is load-bearing, not cosmetic, and 16384 is inclusive.
#
# We cap explicitly anyway so a caller asking for more fails a unit test here
# rather than a live run.
_WM_SAFE_MAX_TOKENS = 16_384

# The smallest budget any WM call may pass.
#
# ⚠️ NOT 6000, and this is the correction that matters most. An earlier draft of
# this file set the floor to ``_REASONING_HEADROOM_TOKENS`` (6000) on the
# reasoning that WM's own default must be survivable. It is not. 6000 is what a
# caller gets by omitting ``max_tokens`` entirely, and it is an OBSERVED
# production failure -- from the orchestrator's own ``stage_outputs``, every
# ``Model token limit (N) exceeded`` on record:
#
#     N=6000  promo_script           x2   newest 2026-08-19 08:28Z
#     N=6000  website_review_script  x1          2026-08-19 06:24Z
#     N=6000  growth_bundle          x1          2026-08-19 04:14Z
#     N=6000  segment_select         x1          2026-08-18 21:49Z
#     N=6000  retention_plan         x1          2026-08-10 18:23Z
#     N=6000  promo_analyze          x1          2026-08-10 06:39Z
#     N=1760  segment_select         x1          2026-08-19 08:03Z   (#851)
#     N=4000  footage_direction, course_outline, website_redesign x5
#
# A floor of 6000 would have prevented ZERO of those thirteen. It would have been
# a gate that passes fleet-wide while the failures continue -- green and inert,
# which is worse than absent because it looks like coverage.
#
# 8000 is the smallest value with NO recorded failure. Note what that is and is
# not: an absence of counter-evidence over 45,605 stage rows, not a proof of
# sufficiency. It is also the value ``demo_video/retention_plan.py`` already
# passes at its one explicit site, which has not 422'd since it was set.
# tests/test_wm_max_tokens_gate.py enforces it as a FLOOR across every call site.
MIN_OUTPUT_TOKEN_BUDGET = 8_000


def output_token_budget(*, answer_tokens: int) -> int:
    """Per-call ``max_tokens`` for a WM call, sized for a REASONING model.

    ``answer_tokens`` is the caller's worst-case estimate of the ANSWER alone —
    the JSON it actually wants back. Reasoning headroom is added on top, because
    thinking is charged against the same budget and is what actually exhausts it.

    The result is clamped to what THIS stage's own per-call wall clock can
    deliver (``LLM_BUDGET_SECONDS x _MEASURED_TOKENS_PER_SEC``) and never exceeds
    WM's own ``SAFE_MAX_TOKENS``. It is deliberately NOT the router's chain
    deadline: that number is a code default of 300 which the cluster overrides to
    1200, and citing it here is how two earlier ceilings were derived from a value
    that was never live.

    ⚠️ This bounds the OUTPUT only. It does nothing about an unbounded PROMPT —
    a bigger prompt makes the model think longer, which spends this budget faster
    and costs wall-clock against the same deadline. Sites whose input grows with
    the corpus must ALSO batch or thin their input; ``segment_select`` and
    ``footage_critique`` are the worked examples.
    """
    if answer_tokens <= 0:
        raise ValueError(f"answer_tokens must be positive, got {answer_tokens}")
    budget = answer_tokens + _REASONING_HEADROOM_TOKENS
    capped = min(budget, _CHAIN_DEADLINE_TOKENS, _WM_SAFE_MAX_TOKENS)
    return max(capped, MIN_OUTPUT_TOKEN_BUDGET)


# Named ANSWER sizes for call sites whose output has no natural derivation.
#
# These are inputs to ``output_token_budget``, not budgets themselves. They exist
# so a site with a small, fixed-shape reply says WHY it is small instead of
# inventing a number: with a 6000-token reasoning floor and an 11000-token
# deadline cap, the honest resolution of this dial is about three positions, and
# pretending to more is false precision. Sites whose answer size IS derivable
# (``segment_select``'s batch * segments * tokens, ``script_stage``'s elements *
# tokens-per-element) derive it and do not use these.
#
# SMALL   — a verdict, an id, a score, a short rationale. One object, few fields.
# MEDIUM  — a modest list: a handful of objects, or one object with prose fields.
# LARGE   — a long list whose length tracks the input (timeline rewrites, plans).
#
# ⚠️ MEASURED, not asserted — and the previous version of this comment had it
# backwards. With floor 8000 and headroom 6000, every answer estimate below 2000
# clamps to the floor, so today:
#
#     SMALL  (500)   -> 8000   |
#     MEDIUM (2000)  -> 8000   |  SAME. The tier is documentation, not a budget.
#     LARGE  (5000)  -> 10925  <- the cap
#
# So SMALL and MEDIUM are indistinguishable at runtime (22 of 29 call sites land
# on exactly 8000), and the honest reading is that this dial has TWO positions,
# not three. That is the box talking, not a modelling choice: a reasoning model
# needs ~6000 tokens of thinking before it says anything, which swamps any
# small answer.
#
# The tiers are kept anyway for two reasons, both narrow: they make a site state
# what it expects back instead of inventing a number, and they start separating
# again the moment the serving rate improves (the cap moves with the clock). What
# they must NOT do is imply per-site tuning that does not exist — if you are
# choosing between SMALL and MEDIUM, know that today it changes nothing.
SMALL_ANSWER_TOKENS = 500
MEDIUM_ANSWER_TOKENS = 2_000
LARGE_ANSWER_TOKENS = 5_000
