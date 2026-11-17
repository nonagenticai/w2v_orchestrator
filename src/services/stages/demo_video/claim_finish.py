"""claim_finish — a bounded WM-LLM evaluator-optimizer that softens unsupportable claims.

An OPT-IN stage (ORCH-4.4 / aspect A18) that stands between the plan stages and
``narrate``: it scores every narration line's *checkable* claims against the
ingested source corpus and rewrites the ones the corpus cannot support, BEFORE
narrate spends TTS on them. Today a hallucinated claim ("the only platform",
"guaranteed 40% faster") sails through narrate and the compiler unchecked and is
only ever caught by a human watching the finished film.

Modelled line-for-line on ``demo_video/retention_plan.py``: a deterministic
evidence pass (:func:`analyze_claims` over
``source_ingest.grounding_sentences``) hands the LLM ``claim-grounder`` the
PRE-COMPUTED flagged rows — the matched term, its grounding verdict, its closest
source sentence and a graded :func:`support_score` — never raw text to
free-associate over. The critic scores the draft 1-5 and names the offending
lines by id; the ``claim-softener`` optimizer then rewrites ONLY those flagged,
unlocked, unpinned lines, and the revision is merged back into the *original
ordered draft by id* — so join ids, line order, ``clip`` bindings, locked lines
and author-pinned claims survive by construction, never by trust. The loop is
hard-bounded by the shared ``loop_engine`` (≤2 passes) and keeps the highest
scoring draft.

ANTI-LAUNDERING: the optimizer's word that it fixed a claim is worth nothing. After
the loop the deterministic lint is RE-RUN over the final draft. A line still
carrying an unsourced term is stamped ``claim_status="unsupported_accepted"`` (never
``"softened"``) and a ``notes`` entry is emitted — which auto-marks the run
``degraded`` (see ``stages/base.py`` ``StageOutputWithNotes``) and names the
``rerun_stages`` suggestion. Swapping "the only" for "the leading" launders nothing:
both are unsourced, and the second verdict is the one that counts.

SOFT GATE: the stage can never hard-block. Even at the worst possible score it
returns a valid :class:`DemoVideoGenOutput` and the run proceeds to compile — an
unsupported claim is *recorded*, not fatal. Legal review is a human's job; this
stage's job is to make sure the human is told.

Presence-is-opt-in: the stage wakes ONLY on an explicit ``legal.mode`` of ``auto``
or ``critique_only``. ``legal`` already ships in prod carrying
``{"trademarks": [...], "defensibility": "..."}`` for other consumers, so a
``legal`` block with no ``mode`` key MUST NOT wake this stage — the added keys
(``mode``, ``posture``, ``claims``) are purely additive. A skipped stage stores
``{}`` and the spine resolver falls straight through, so the run is byte-identical
to today.

LLM calls route through ``worker_management`` (the ``WMClient`` gateway) — never
inference-router / a provider directly; the model behind each worker is config,
not code.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING, ClassVar, cast

from pydantic import Field

from src.services.llm import WMClient
from src.services.stages.base import SlotUse, StageHandler, StageInput, StageSkippedError
from src.services.stages.demo_video.gen import DemoVideoGenOutput
from src.services.stages.demo_video.spine import resolve_plan_output
from src.services.stages.generic.grounding import grounding_sentences
from src.services.stages.llm_budget import (
    LARGE_ANSWER_TOKENS,
    MEDIUM_ANSWER_TOKENS,
    llm_budget_seconds,
    llm_client_timeout_seconds,
    llm_retries,
    output_token_budget,
    reasoning_for_attempt,
)
from src.services.stages.shared.asset_rights import (
    build_asset_ledger,
    defensibility_manifest,
    trademark_flags,
)
from src.services.stages.shared.claim_grounding import (
    analyze_claims,
    disclaimer_requirements,
    find_quantitative,
    support_score,
)
from src.services.stages.shared.loop_engine import (
    ON_LOOP_ERROR_KEEP_BEST,
    Critique,
    run_evaluator_optimizer,
)
from src.utils import clamp_score

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext


def _as_list(value: object) -> list[object]:
    """Read-boundary coercion for LLM/generator output: a list only if it is one."""
    return list(value) if isinstance(value, (list, tuple)) else []


# WM worker names (regex ^[a-zA-Z0-9_-]+$). WM auto-creates them on first call;
# the provider/model lives on the WM worker config, not here.
_CRITIC_WORKER = "claim-grounder"
_OPTIMIZER_WORKER = "claim-softener"

# The only ``legal.mode`` values that wake the stage. Anything else — including a
# `legal` block with NO ``mode`` key (the shape already shipping in prod) — skips.
_ACTIVE_MODES = frozenset({"auto", "critique_only"})

# ``legal.claims.on_unsupported``: soften (rewrite) | flag (critique, never rewrite).
_ON_UNSUPPORTED_FLAG = "flag"

_DEFAULT_ITERATIONS = 2
_DEFAULT_THRESHOLD = 4.0

# The author's escape hatch: a line whose claim is deliberate and defended. Never
# rewritten, never re-stamped — byte-identical out.
_PINNED = "pinned"

# What an unresolved claim needs is a genuinely different SCRIPT, which this stage
# cannot write from inside its own bounded loop — so it suggests the copy stage.
_RERUN_STAGES = ["script_doctor"]

# The claim-grounder verdict schema. ``overall_score`` (1-5) drives convergence;
# ``directives`` name the offending beats + the relief the critic wants.
#
# The 1-5 band is a ``description``, NOT ``minimum``/``maximum``: guided decoding cannot
# enforce a numeric bound, and WM's ``retries=0`` on ``inference_router`` turns the
# post-hoc validation failure into a permanent stage kill on the first out-of-range
# integer. Steered by the seeded ``claim-grounder`` system prompt, enforced by ``clamp_score`` at the read.
# See ``tests/test_output_schema_bounds.py``.
_CRITIC_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["overall_score", "directives"],
    "properties": {
        "overall_score": {"type": "integer", "description": "1-5, where 5 is strongest"},
        # Declared for the model's benefit only — nothing reads this key back (the
        # ``defensibility_manifest`` elsewhere in this stage is the operator-declared
        # ``legal`` block, a different data path). Asking for it shapes the critic's
        # reasoning; it is not a signal this stage consumes.
        "defensibility": {"type": "integer", "description": "1-5, where 5 is strongest"},
        "per_claim": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["line_id", "verdict"],
                "properties": {
                    "line_id": {"type": "string"},
                    "matched_term": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["supported", "needs_softening", "unsupportable"],
                    },
                    "rationale": {"type": "string"},
                },
            },
        },
        "directives": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["op", "target_ids"],
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": ["soften", "cite", "drop_claim", "keep"],
                    },
                    "target_ids": {"type": "array", "items": {"type": "string"}},
                    "payload": {"type": "object"},
                },
            },
        },
    },
}

# The claim-softener rewrite schema — deliberately the SAME shape as
# ``retention_plan._OPTIMIZER_SCHEMA`` so the by-id merge helper stays reusable.
_OPTIMIZER_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["lines"],
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id"],
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "candidates": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    },
}


# MEDIUM answer: ``_CRITIC_SCHEMA`` is ONE verdict object — two 1-5 integers plus a
# ``per_claim`` / ``directives`` pair whose length tracks the DETERMINISTICALLY FLAGGED
# claims (``_evidence_rows``), not the spine, and each entry is a couple of ids, an enum
# and one short rationale.
_CRITIC_MAX_TOKENS = output_token_budget(answer_tokens=MEDIUM_ANSWER_TOKENS)

# LARGE answer: ``_OPTIMIZER_SCHEMA`` returns REWRITTEN NARRATION — one object per
# flagged beat carrying full replacement ``text`` (and optionally several ``candidates``),
# so the answer is prose whose length tracks the flagged beat count. This is the
# "per-line rewrite" tier; a short-list budget would truncate the very output the
# stage exists to produce.
_OPTIMIZER_MAX_TOKENS = output_token_budget(answer_tokens=LARGE_ANSWER_TOKENS)


def _lines_json(lines: list[dict[str, object]]) -> str:
    """Deterministic id+text projection of the spine for a prompt block."""
    return json.dumps(
        [{"id": str(line["id"]), "text": str(line.get("text", ""))} for line in lines],
        indent=2,
        ensure_ascii=False,
    )


def _spine_ids(lines: list[dict[str, object]]) -> frozenset[str]:
    """The C0 invariant key: the ordered spine's id-set (order-independent)."""
    return frozenset(str(line["id"]) for line in lines)


def _protected_ids(draft: list[dict[str, object]]) -> frozenset[str]:
    """Ids no optimizer may ever touch: author-locked lines + pinned claims.

    ``locked`` is the existing script-doctor pin (this beat's narration is the
    author's); ``claim_status == "pinned"`` is the *claims* pin — a deliberate,
    defended claim the author stands behind. Both must come out byte-identical.
    """
    return frozenset(
        str(line["id"])
        for line in draft
        if line.get("locked") or str(line.get("claim_status") or "") == _PINNED
    )


def _evidence_rows(
    draft: list[dict[str, object]],
    sentences: list[str],
    *,
    allow_superlatives: bool,
) -> list[dict[str, object]]:
    """The bounded evidence the critic is allowed to reason over.

    One row per checkable claim the deterministic lint found, carrying its verdict,
    the source sentence that grounds it (if any), a graded :func:`support_score`
    against the corpus, and whether it is a superlative or a numeral. With
    ``allow_superlatives`` the operator has accepted superlative *framing* as a
    house style, so only quantitative claims (an invented number is never framing)
    are put in front of the critic.
    """
    rows: list[dict[str, object]] = []
    for row in analyze_claims(draft, sentences).flagged:
        term = str(row["matched_term"])
        quantitative = bool(find_quantitative(term))
        if allow_superlatives and not quantitative:
            continue
        score, closest = support_score(term, sentences)
        rows.append(
            {
                "line_id": str(row["line_id"]),
                "text": str(row["text"]),
                "matched_term": term,
                "kind": "quantitative" if quantitative else "superlative",
                "claim_status": str(row["claim_status"]),
                "claim_source": str(row["claim_source"]),
                "support_score": round(score, 3),
                "closest_source_sentence": closest,
            }
        )
    return rows


def _unsourced_ids(rows: list[dict[str, object]]) -> frozenset[str]:
    """Line ids the DETERMINISTIC lint says carry an unsupported claim."""
    return frozenset(str(row["line_id"]) for row in rows if row["claim_status"] == "unsourced")


def _softenable_lines(
    draft: list[dict[str, object]],
    rows: list[dict[str, object]],
    critic_output: dict[str, object],
) -> list[dict[str, object]]:
    """The rewrite set: unsourced beats the critic named that the author did not pin.

    Doubly bounded. A beat is a candidate only when BOTH the deterministic lint
    flagged it ``unsourced`` AND the critic named it (via a directive target or a
    non-``supported`` ``per_claim`` verdict) — so a critic that hallucinates an
    objection to grounded copy cannot trigger a rewrite, and neither can a critic
    that stays silent. Hallucinated ids (not in the spine), locked lines and pinned
    claims are then subtracted, so the optimizer can only ever touch real, unlocked,
    unpinned, genuinely-unsourced beats.
    """
    named: set[str] = set()
    for directive in _as_list(critic_output.get("directives")):
        if not isinstance(directive, dict):
            continue
        directive = cast("dict[str, object]", directive)
        if directive.get("op") == "keep":
            continue
        for target in _as_list(directive.get("target_ids")):
            named.add(str(target))
    for claim in _as_list(critic_output.get("per_claim")):
        if not isinstance(claim, dict):
            continue
        claim = cast("dict[str, object]", claim)
        if claim.get("line_id") is None:
            continue
        if str(claim.get("verdict") or "") != "supported":
            named.add(str(claim["line_id"]))

    candidates = frozenset(named) & _unsourced_ids(rows) & _spine_ids(draft)
    allowed = candidates - _protected_ids(draft)
    return [line for line in draft if str(line["id"]) in allowed]


def _apply_by_id(
    draft: list[dict[str, object]],
    revised: list[dict[str, object]],
    allowed_ids: frozenset[str],
) -> list[dict[str, object]]:
    """Merge revised TEXT into the ordered draft, by id.

    Iterates ``draft`` in its original order and patches ONLY the ``text`` of a line
    whose id the worker echoed AND that is in ``allowed_ids`` (unsourced + critic-named
    + unlocked + unpinned). Ids the worker hallucinated are ignored — never added;
    ``id`` / ``clip`` / order are NEVER touched. So the spine id-set + join ids survive
    by construction and a locked or pinned line can never be rewritten. ``text`` falls
    back to the first ``candidates`` entry when the worker returned candidates instead
    of a single line. The claim annotation fields are deliberately NOT patched here —
    they are stamped afterwards from the RE-RUN lint (:func:`_stamp_claims`), never
    from the optimizer's own account of its work.
    """
    patches: dict[str, str] = {}
    for item in revised:
        if not isinstance(item, dict):
            continue
        lid = item.get("id")
        if lid is None or str(lid) not in allowed_ids:
            continue
        text = item.get("text")
        if text is None:
            candidates = item.get("candidates")
            if isinstance(candidates, list) and candidates:
                text = candidates[0]
        if text is not None:
            patches[str(lid)] = str(text)
    return [
        {**line, "text": patches[str(line["id"])]} if str(line["id"]) in patches else line
        for line in draft
    ]


def _stamp_claims(
    draft: list[dict[str, object]],
    sentences: list[str],
    *,
    was_unsourced: frozenset[str],
    protected: frozenset[str],
) -> tuple[list[dict[str, object]], list[str]]:
    """RE-RUN the lint on the final draft and stamp the honest verdict — the anti-launder.

    The optimizer's claim to have fixed a line is never trusted. Whatever it returned,
    the deterministic lint runs again over the FINAL text and decides:

    * still carrying an unsourced term → ``claim_status="unsupported_accepted"`` (NOT
      ``"softened"``) plus a ``notes`` entry, which auto-degrades the run and names the
      ``rerun_stages`` suggestion. Swapping "only" for "leading" lands here.
    * carrying only claims the corpus echoes → ``claim_status="grounded"`` with the
      supporting sentence in ``claim_source``.
    * was unsourced, now carries no claim at all → ``claim_status="softened"``.
    * never carried a claim → left untouched (no annotation churn).

    Protected lines (locked / ``claim_status="pinned"``) are skipped entirely, so a
    pinned line is byte-identical on the way out.
    """
    rows_by_id: dict[str, list[dict[str, object]]] = {}
    for row in analyze_claims(draft, sentences).flagged:
        rows_by_id.setdefault(str(row["line_id"]), []).append(row)

    stamped: list[dict[str, object]] = []
    notes: list[str] = []
    for line in draft:
        line_id = str(line["id"])
        if line_id in protected:
            stamped.append(line)
            continue
        rows = rows_by_id.get(line_id) or []
        unsourced = [row for row in rows if row["claim_status"] == "unsourced"]
        if unsourced:
            terms = ", ".join(sorted({str(row["matched_term"]) for row in unsourced}))
            notes.append(
                f"claim_finish: unsupported claim(s) [{terms}] in line {line_id} survived "
                "the softener — accepted UNVERIFIED and stamped unsupported_accepted; "
                f"no ingested source supports them (suggest rerun: {', '.join(_RERUN_STAGES)})"
            )
            stamped.append({**line, "claim_status": "unsupported_accepted", "claim_source": ""})
        elif rows:
            stamped.append(
                {
                    **line,
                    "claim_status": "grounded",
                    "claim_source": str(rows[0]["claim_source"]),
                }
            )
        elif line_id in was_unsourced:
            stamped.append({**line, "claim_status": "softened", "claim_source": ""})
        else:
            stamped.append(line)
    return stamped, notes


def _critic_msg(
    draft: list[dict[str, object]],
    rows: list[dict[str, object]],
    posture: object,
) -> str:
    """Build the critic prompt: ordered spine + the pre-computed grounding evidence."""
    parts = [
        f"Claims posture: {posture if posture else 'default — claims must be defensible'}",
        "Narration spine (ordered, by id):",
        _lines_json(draft),
        "Deterministic grounding evidence (the ONLY facts you may reason from — every "
        "checkable claim in the copy, its verdict against the ingested source corpus, "
        "and the closest source sentence):",
        json.dumps(rows, indent=2, sort_keys=True, ensure_ascii=False),
        "Score the draft's defensibility and name the offending beats by id. "
        "Preserve every id; do not reorder.",
    ]
    return "\n\n".join(parts)


def _optimizer_msg(
    flagged: list[dict[str, object]],
    rows: list[dict[str, object]],
    critic_output: dict[str, object],
) -> str:
    """Build the rewrite prompt: the unsupported beats + their evidence + critic guidance."""
    flagged_ids = _spine_ids(flagged)
    evidence = [row for row in rows if row["line_id"] in flagged_ids]
    parts = [
        "Soften ONLY these beats — the source material does not support their claims. "
        "Echo each id UNCHANGED.",
        "One entry per input beat, same id — no add/drop/merge/split/reorder, no re-clip.",
        "Beats to soften:",
        _lines_json(flagged),
        "Why each is unsupported (matched term + the closest thing the corpus actually "
        "says):\n" + json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False),
        "Critic guidance:\n"
        + json.dumps(critic_output, indent=2, sort_keys=True, ensure_ascii=False),
    ]
    return "\n\n".join(parts)


def _record_half(
    config_snapshot: dict[str, object],
    lines: list[dict[str, object]],
    sentences: list[str],
    legal: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """The RECORD half (A18 phase 2): disclaimers + asset ledger + manifest.

    Deterministic, LLM-free and total: the disclaimer rule engine, the asset-rights
    walk and the trademark scan all run over the FINAL softened spine and the run's
    ``config_snapshot``, and are content-addressed into one manifest. Nothing here
    can block, and nothing here mutates the spine.

    Returns ``(disclaimers, manifest)``. The disclaimers are emitted regardless of
    ``legal.disclaimers.mode`` — the manifest must record what the copy MANDATED even
    when the operator declined to show a card; the ``off`` gate lives at the render
    seam (``disclaimer_card.build_disclaimer_shot``), which is the thing an operator
    is actually declining.
    """
    final_claims = analyze_claims(lines, sentences).flagged
    disclaimers = [
        {**asdict(req), "line_ids": list(req.line_ids)}
        for req in disclaimer_requirements(final_claims)
    ]
    trademark_cfg = legal.get("trademark")
    known_marks = (
        cast("dict[str, object]", trademark_cfg or {}).get("known_marks")
        if isinstance(trademark_cfg, dict)
        else []
    )
    manifest = defensibility_manifest(
        claims=final_claims,
        ledger=build_asset_ledger(config_snapshot),
        disclaimers=disclaimers,
        trademarks=trademark_flags(
            lines,
            config_snapshot.get("brand") or {},
            cast("list[str]", known_marks) if isinstance(known_marks, list) else [],
        ),
    )
    return disclaimers, manifest


class ClaimFinishInput(StageInput):
    """No typed inputs — everything is read from ``ctx`` (prior output + config)."""


class ClaimFinishOutput(DemoVideoGenOutput):
    """The spine, widened by the A18 record half.

    ``DemoVideoGenOutput`` verbatim (so every downstream consumer — which all read
    ``["lines"]`` — is untouched; pydantic ignores the extra keys) plus the two
    artifacts phase 2 produces. Both persist for free via ``StageOutputRepository``
    (the stage output is stored as JSONB), which is what makes the manifest an
    auditable record of the run rather than a log line, and what lets the compile
    tail read ``disclaimers`` back out to build its card.
    """

    defensibility_manifest: dict[str, object] = Field(default_factory=dict)
    disclaimers: list[dict[str, object]] = Field(default_factory=list)


class ClaimFinishStage(StageHandler[ClaimFinishInput, ClaimFinishOutput]):
    name = "claim_finish"
    input_model = ClaimFinishInput
    output_model = ClaimFinishOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (
        SlotUse("llm"),
        SlotUse("tts"),
    )

    def __init__(self, client: WMClient | None = None) -> None:
        self._llm_timeout = llm_budget_seconds()
        self._llm_retries = llm_retries()
        self._wm = client or WMClient(timeout=llm_client_timeout_seconds())

    async def handle(self, payload: ClaimFinishInput, ctx: RunContext) -> ClaimFinishOutput:
        # ── Guard 1: there must be a spine to check. ──
        gen: dict[str, object] = resolve_plan_output(ctx) or {}
        lines: list[dict[str, object]] = cast("list[dict[str, object]]", _as_list(gen.get("lines")))
        if not lines:
            raise StageSkippedError("claim_finish: no spine lines to check claims on")

        # ── Guard 2: presence-opt-in on an EXPLICIT legal.mode. The `legal` block
        # already ships in prod as {"trademarks": [...], "defensibility": "..."} for
        # other consumers — a block with no `mode` key must not wake this stage. ──
        legal = ctx.config_snapshot.get("legal") or {}
        mode = legal.get("mode") if isinstance(legal, dict) else None
        if mode not in _ACTIVE_MODES:
            raise StageSkippedError("claim_finish: no legal.mode of auto|critique_only")

        # ── Guard 3: grounding needs a corpus. With nothing ingested, EVERY claim
        # would read as unsourced — that is a lie about the copy, not a finding. ──
        sentences = grounding_sentences(ctx)
        if not sentences:
            raise StageSkippedError("claim_finish: no ingested source corpus to ground against")

        claims_cfg = legal.get("claims") or {}
        if not isinstance(claims_cfg, dict):
            claims_cfg = {}
        allow_superlatives = bool(claims_cfg.get("allow_superlatives", False))
        on_unsupported = str(claims_cfg.get("on_unsupported") or "soften")
        iterations = int(claims_cfg.get("iterations", _DEFAULT_ITERATIONS))
        threshold = float(claims_cfg.get("score_threshold", _DEFAULT_THRESHOLD))
        posture = legal.get("posture")

        protected = _protected_ids(lines)
        initial_rows = _evidence_rows(lines, sentences, allow_superlatives=allow_superlatives)
        # Recorded BEFORE the loop: the only ids a "softened" verdict may ever apply to.
        was_unsourced = _unsourced_ids(initial_rows) - protected

        async def critic_fn(draft: list[dict[str, object]]) -> Critique:
            # Recompute the evidence on the CURRENT draft so the critic scores fresh
            # verdicts each pass — a rewrite that laundered nothing must still read as
            # unsourced on pass 2.
            rows = _evidence_rows(draft, sentences, allow_superlatives=allow_superlatives)
            resp = await self._wm.message(
                enable_reasoning=reasoning_for_attempt(),
                worker_name=_CRITIC_WORKER,
                message=_critic_msg(draft, rows, posture),
                output_schema=_CRITIC_SCHEMA,
                max_tokens=_CRITIC_MAX_TOKENS,
            )
            output = resp.output if isinstance(resp.output, dict) else {}
            # Clamped, not trusted: the schema can only ASK for 1-5 (see _CRITIC_SCHEMA).
            score = clamp_score(output.get("overall_score"))
            # Advisory only (soft gate): unresolved claims need a genuinely different
            # script, which this stage cannot write from inside its own loop.
            rerun = list(_RERUN_STAGES) if _unsourced_ids(rows) - protected else []
            return Critique(score=score, rerun_stages=rerun, directives=(output, rows))

        async def optimizer_fn(
            draft: list[dict[str, object]], critique: Critique
        ) -> list[dict[str, object]]:
            critic_output, rows = critique.directives
            flagged = _softenable_lines(draft, rows, critic_output)
            if not flagged:
                return draft
            allowed = _spine_ids(flagged)
            resp = await self._wm.message(
                enable_reasoning=reasoning_for_attempt(),
                worker_name=_OPTIMIZER_WORKER,
                message=_optimizer_msg(flagged, rows, critic_output),
                output_schema=_OPTIMIZER_SCHEMA,
                max_tokens=_OPTIMIZER_MAX_TOKENS,
            )
            revised = resp.output.get("lines", []) if isinstance(resp.output, dict) else []
            return _apply_by_id(draft, revised, allowed)

        if was_unsourced:
            # ``critique_only`` / ``on_unsupported=flag``: score once (advisory) but apply
            # NO rewrite. A single pass means the engine calls the critic once and never
            # the optimizer, so the best draft is the untouched spine — exactly the
            # retention_plan contract.
            advisory_only = mode == "critique_only" or on_unsupported == _ON_UNSUPPORTED_FLAG
            max_passes = 1 if advisory_only else iterations

            result = await run_evaluator_optimizer(
                lines,
                critic_fn=critic_fn,
                optimizer_fn=optimizer_fn,
                threshold=threshold,
                id_extractor=_spine_ids,
                max_passes=max_passes,
                stage_name=self.name,
                timeout_seconds=self._llm_timeout,
                retries=self._llm_retries,
                # ADVISORY by contract (the soft-gate note below): the honest verdict
                # is stamped from the RE-RUN lint, so a dead critic degrades to the
                # untouched spine + notes — it must never block the deliverable.
                on_loop_error=ON_LOOP_ERROR_KEEP_BEST,
            )
            best = result.draft
        else:
            # Nothing the corpus fails to support ⇒ nothing to soften. Skip the loop
            # entirely (zero WM traffic) and go straight to stamping the grounded
            # verdicts — the run pays nothing for copy that was already honest.
            best = lines

        # Soft-gate: the best draft is ALWAYS returned, even at the worst score. The
        # honest verdict is stamped from the RE-RUN lint, and anything still unsupported
        # is recorded in ``notes`` (⇒ degraded run) rather than blocking the deliverable.
        merged, claim_notes = _stamp_claims(
            best, sentences, was_unsourced=was_unsourced, protected=protected
        )
        prior_notes = [str(note) for note in _as_list(gen.get("notes"))]

        # ── RECORD half (phase 2): what the FINAL copy mandates + what it rides on. ──
        disclaimers, manifest = _record_half(ctx.config_snapshot, merged, sentences, legal)

        return ClaimFinishOutput.model_validate(
            {
                **gen,
                "lines": merged,
                "notes": [*prior_notes, *claim_notes],
                "disclaimers": disclaimers,
                "defensibility_manifest": manifest,
            }
        )
