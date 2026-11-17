"""LLMScriptStageBase — the invariant LLM-script tail shared by script stages.

Every script stage does the same thing once its prompt is built: send the prompt
to the stage's WM worker with a structured output schema (``stage_call`` on the
shared LLM ladder — ``llm_budget_seconds()`` / ``llm_retries()``), mint a fresh id
per element, fail loud with ``StageError`` if
the output is missing fields, then assert the required verbatim phrases survived.
Per-stage variance lives in three hooks — ``build_prompt`` (the stage's prompt
text, INCLUDING its ``verbatim_prompt_block``), ``make_element`` (one parsed dict
→ typed element), and ``build_output`` (wrap elements, compute total_seconds, add
extras like lesson's learning_objective). lesson also overrides ``short_circuit``
for its verbatim source-script bypass.

LLM access is through ``worker_management`` (``WMClient`` → the per-subclass WM
worker named by ``worker_name`` — the kebab-cased stage name); the
provider/model lives on the WM worker config, NOT here.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
from abc import abstractmethod
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, ClassVar, cast
from uuid import uuid7

from w2v_common.stages import stage_call

from src.services.llm import WMClient
from src.services.stages.base import (
    StageHandler,
    StageInput,
    StageOutput,
    assert_verbatim_present,
    partition_for_stage,
    record_verbatim_drop,
)
from src.services.stages.llm_budget import (
    LLM_RETRY_ON,
    llm_budget_seconds,
    llm_client_timeout_seconds,
    llm_retries,
    output_token_budget,
)
from src.services.stages.shared.placeholder_text import is_placeholder_text
from src.services.stages.shared.step_ledger import JsonValue, hash_inputs
from src.services.stages.upstream import guarded

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator

    from src.services.runs.run_context import RunContext

# Patience budget for the WM script LLM call comes from the shared LLM ladder
# (``PipelineSettings.llm_budget_seconds`` / ``llm_retries``, read via
# ``llm_budget_seconds()`` / ``llm_retries()``). Script stages route through the
# WM worker → inference-router → shared pgx-local vLLM (~5 sessions + heavy ks-*
# reasoning workers), which QUEUES on a per-provider concurrency semaphore. Under
# contention a heavy script call (e.g. website_review_script, grounded in
# critique + redesign + sources) must WAIT for its slot rather than cut itself
# off early and re-queue — the same stall growth_bundle hit (#366). The default
# budget (660s) sits above the 600s downstream stack (WM ir_http_read_timeout_s +
# inference-router provider timeout) and below the 700s outer stage
# TimeoutMiddleware, so a router-successful response is captured and stage_call's
# own timeout fires before the outer wrapper.

_SYSTEM_PROMPT = (
    "You write the script for a video segment as structured data. Produce the "
    "element list the prompt asks for. Return only JSON matching the schema."
)


log = logging.getLogger(__name__)


#: The request hashes ``_draw`` counted inside the CURRENT ``stage_call`` attempt.
#: Per task (a ContextVar), so an abandoned attempt un-counts only its OWN draws —
#: ``line_fit`` runs several lines' rewrites concurrently on one stage instance.
_ATTEMPT_DRAWS: ContextVar[list[str] | None] = ContextVar("_attempt_draws", default=None)


@dataclasses.dataclass(slots=True)
class _Draw:
    """One model draw as the step ledger stores it — the reply plus what it minted.

    ``output`` and ``usage`` are the two ``WMResponse`` fields this primitive reads;
    ``element_ids`` are the ids minted for the reply's elements, stored WITH the
    reply so a re-entered stage hands downstream the same ids for the same text
    (annotations, storyboard and narrate all join on them). ``built`` is the
    already-mapped element list of a FRESH draw — never stored; a replayed draw
    re-maps ``output`` with the stored ids, which is deterministic.
    """

    output: JsonValue
    usage: JsonValue
    element_ids: list[str]
    built: tuple[list[StageOutput], list[dict]] | None = None

    def dump(self) -> JsonValue:
        ids: list[JsonValue] = list(self.element_ids)
        return {"output": self.output, "usage": self.usage, "element_ids": ids}

    @classmethod
    def load(cls, stored: JsonValue) -> _Draw:
        if not isinstance(stored, dict):
            raise TypeError("stored script draw is not an object")
        ids = stored["element_ids"]
        if not isinstance(ids, list):
            raise TypeError("stored script draw carries no element id list")
        return cls(
            output=stored["output"], usage=stored["usage"], element_ids=[str(i) for i in ids]
        )


class ScriptDrawUnderFloorError(ValueError):
    """A parsed draw held fewer elements than the stage's ``min_elements``.

    A ``ValueError`` so it stays retryable (``LLM_RETRY_ON``) and reads exactly as
    it always did; the TYPE exists so a stage can tell "every attempt drew too
    few" from any other exhausted failure via
    ``StageRetriesExhaustedError.last_error`` without matching message prose.
    """

    def __init__(self, message: str, *, count: int, required: int) -> None:
        super().__init__(message)
        self.count = count
        self.required = required


def _log_under_floor_draw(
    *,
    stage: str,
    worker: str,
    attempt: int,
    count: int,
    required: int,
    prompt: str,
    output: object,
    usage: object,
) -> None:
    """Record the exact ask and the exact answer of an under-floor draw, at WARNING.

    The prompt is persisted nowhere else, so until this line existed an empty draw
    could not be replayed: the error carried a summary of the prompt, never the
    prompt (website_review, 2026-10-06 — ``{"lines": []}`` on every attempt, cause
    undiagnosable from the pod log). ONE line, a JSON object after a fixed prefix,
    because the service's JSON formatter emits only ``message`` and drops
    ``extra=`` fields. It carries the prompt and the parsed completion ONLY — no
    request headers, so no license token can reach it.
    """
    payload = {
        "worker": worker,
        "attempt": attempt,
        "count": count,
        "required": required,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_chars": len(prompt),
        "usage": usage,
        "completion": json.dumps(output, default=str, ensure_ascii=False),
        # ``WMClient`` sends ``f"{system_prompt}\n\n{prompt}"`` as the one message
        # (WM ignores a body ``system_prompt``); the hash and length are of ``prompt``.
        "system_prompt": _SYSTEM_PROMPT,
        "prompt": prompt,
    }
    log.warning(
        "%s: under-floor script draw %s",
        stage,
        json.dumps(payload, default=str, ensure_ascii=False),
    )


def _mint_element_ids(output: object, element_key: str) -> list[str]:
    """A fresh id per element of a reply — ``[]`` when it has no element list.

    One per RAW element (before the placeholder drop), so there are always enough;
    a malformed reply mints none and is rejected by the mapping right after.
    """
    elements = output.get(element_key) if isinstance(output, dict) else None
    return [str(uuid7()) for _ in elements] if isinstance(elements, list) else []


def _drop_placeholder_elements(raw: list[Any], *, stage: str) -> list[Any]:
    """Drop elements whose ``text`` is a placeholder ("None", "null", blank, …).

    Every element on this primitive is narration, so such an element would be
    SPOKEN: a promo whose prompt said ``End with this CTA verbatim: "None"``
    returned a final ``{"text": "None"}`` and the customer's video closed on it.
    The prompt is fixed at the source; this is the parse-side backstop for any
    future prompt that leaks an absent value. Only elements that HAVE a ``text``
    key are judged — a missing key stays a shape error and is re-prompted.
    Dropping can push a draw under ``min_elements``, which re-prompts too.
    """
    kept = [
        r
        for r in raw
        if not (isinstance(r, dict) and "text" in r and is_placeholder_text(r["text"]))
    ]
    if len(kept) != len(raw):
        log.warning(
            "%s: dropped %d placeholder-text element(s) from the LLM draw",
            stage,
            len(raw) - len(kept),
        )
    return kept


# Default per-element shape. EVERY make_element reads raw["text"] and
# float(raw["seconds"]), and the base joins raw["text"] for the verbatim check —
# so both are required. WM's structured output only emits properties NAMED in
# the schema, so an underspecified item schema ({"type": "object"} with no
# properties) yields text-less objects → KeyError("text") → retries exhausted →
# the run fails (E2E regression, 2026-06-01). Subclasses with extra fields
# (critique_item_id, fact_id, …) override ``element_schema``.
_ELEMENT_SCHEMA_DEFAULT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "seconds": {"type": "number"},
    },
    "required": ["text", "seconds"],
}


def non_empty_array(item_schema: dict[str, object]) -> dict[str, object]:
    """An array schema whose GRAMMAR cannot produce ``[]``: ``prefixItems: [ITEM]``.

    WHY not ``minItems: 1``: worker-management strips ``minItems`` before building
    the vLLM grammar (``xgrammar_schema.strip_xgrammar_unsupported``;
    ``tests/test_output_schema_bounds.py`` bans the keyword for that reason), so it
    constrains nothing. With a plain ``{"type": "array", "items": ITEM}`` the local
    vLLM (xgrammar, ``disable_any_whitespace``) is steered from ``[`` straight to
    ``]``: replayed on the live model it answered ``{"lines": []}`` on 6 of 6 calls
    (two prompt versions, named or band sections). The SAME request with
    ``prefixItems: [ITEM]`` + ``items: ITEM`` returned 13 and 11 full lines (2/2).
    Evidence: ``w2v/.proof-archive/2026-10-06/round4/r0-replay/`` (``replay.sh``,
    ``request-old.json`` vs ``request-schemafix.json``). WM's grammar builder keeps
    ``prefixItems``, so the first element is REQUIRED by the CFG itself; ``items``
    still admits any number after it.

    This is the grammar half only. The parser-side floor (``min_elements`` →
    :class:`ScriptDrawUnderFloorError` → ``EmptyScriptDrawError``) stays as defence
    in depth — a provider without grammar enforcement can still answer empty.
    """
    return {"type": "array", "prefixItems": [item_schema], "items": item_schema}


def _script_schema(
    element_key: str, element_schema: dict[str, object], *, non_empty: bool = False
) -> dict[str, object]:
    """Object-rooted schema holding the element list under ``element_key``.

    ``non_empty`` makes the list grammar-level non-empty (see :func:`non_empty_array`);
    set it for every stage whose ``min_elements`` is at least 1.
    """
    array = (
        non_empty_array(element_schema) if non_empty else {"type": "array", "items": element_schema}
    )
    return {
        "type": "object",
        "properties": {element_key: array},
        "required": [element_key],
    }


class LLMScriptStageBase[I: StageInput, O: StageOutput](StageHandler[I, O]):
    """Shared LLM-script tail. Subclasses implement build_prompt/make_element/build_output."""

    #: JSON key holding the element list in the LLM output ("lines" | "paragraphs").
    element_key: ClassVar[str]
    #: JSON schema for ONE element object, used as the array ``items`` schema in
    #: the WM ``output_schema``. Default requires text+seconds (read by every
    #: make_element). Subclasses with extra fields (e.g. critique_item_id) MUST
    #: override this — WM only emits schema-named properties.
    element_schema: ClassVar[dict[str, Any]] = _ELEMENT_SCHEMA_DEFAULT

    @property
    def worker_name(self) -> str:
        """WM worker name = kebab-cased stage name (``promo_script`` → ``promo-script``).

        NOT ``self.name`` directly: the underscore-named workers were auto-created
        in WM during the failed #152 E2E (loose item schema + a degraded pgx
        provider), which pinned per-worker config that returns empty ``{}``
        elements regardless of a corrected per-call output_schema. Verified
        2026-06-02: fresh names populate, the old underscore names do not, and
        WM's name-based DELETE 404s (the stale tenant-scoped state isn't
        reachable). Kebab names are fresh → clean auto-create with the current
        healthy default, and match the direct stages' convention
        (``course-outline``, ``promo-analyze``, …).
        """
        return self.name.replace("_", "-")

    #: Minimum element count. An under-floor draw is re-prompted within the retry
    #: budget, then fails loud (StageError) if still short. Default 0 = no floor.
    #: generic ``script_draft`` sets 1 — an empty list silently flows into
    #: storyboard → zero shots → video-compile 422 (bug-bash B1+B2, 2026-05-09).
    min_elements: ClassVar[int] = 0

    @classmethod
    def script_output_schema(cls) -> dict[str, object]:
        """The exact ``output_schema`` this stage sends to WM.

        A stage with a floor (``min_elements >= 1``) gets a grammar-level non-empty
        element list (:func:`non_empty_array`). Keyed on the stage's floor, not on
        ``_corrective``: a corrective re-ask that answers empty is a failed
        correction, so it gains from the grammar floor too (its PARSE floor stays 0).
        """
        return _script_schema(cls.element_key, cls.element_schema, non_empty=cls.min_elements >= 1)

    #: True only inside :meth:`corrective_draw`. A CORRECTIVE re-ask is parsed with
    #: NO element floor, so an empty answer comes back as a result after ONE model
    #: call — the caller already holds a usable draw and treats zero elements as a
    #: failed correction. The floor's re-prompt is for the FIRST draw, which has
    #: nothing to fall back to; spending it again on every correction multiplied
    #: the calls of a run whose model keeps answering empty (website_review 7 vs 4).
    _corrective: bool = False

    #: How many draws of each exact request THIS stage execution has completed,
    #: keyed by request hash. Reset by :meth:`run`, i.e. once per (re-)entry — NOT
    #: per ``handle``, which a blueprint calls again for every corrective re-ask.
    #: It is the "which draw is this" half of the step key: see :meth:`_draw`.
    _draws_done: dict[str, int] | None = None

    async def run(self, raw_payload: dict, ctx: RunContext) -> O:
        self._draws_done = {}
        return await super().run(raw_payload, ctx)

    @contextlib.contextmanager
    def corrective_draw(self) -> Iterator[None]:
        """Run ``handle`` as a correction of a draw the caller already holds."""
        self._corrective = True
        try:
            yield
        finally:
            self._corrective = False

    # --- output token budget -------------------------------------------------
    #
    # THE regression this pair exists for: on 2026-08-19 06:24Z, run
    # 01a018a2-bda8-74da-a5b2-c8e126b027b9 died HERE — ``website_review_script``,
    # WM 422 "Model token limit (6000) exceeded before any response was
    # generated", terminal (422 is not retryable), taking the whole run with it.
    # The same class killed ``growth_bundle`` in the promo E2E the same night.
    # This is ONE call site inherited by EIGHT concrete stages, so it is fixed
    # once here rather than eight times.
    #
    # These are deliberately DECLARED, not derived from the brief's target
    # duration. Three different duration keys are already in use across the
    # subclasses (``brief.duration_seconds`` in app_review/website_review,
    # ``brief.target_duration_seconds`` in generic, that OR the deliverable's in
    # promo), so deriving would mean eight chances to read the wrong key — and it
    # would buy nothing: ``output_token_budget`` only has a useful range of
    # [6000, 11000], so per-second precision here is false precision.

    #: Worst-case output tokens for ONE element: the element JSON with its keys,
    #: braces, commas and escaping. Sized for ``_ELEMENT_SCHEMA_DEFAULT``
    #: (text + seconds). Stages whose elements are PARAGRAPHS rather than lines,
    #: or whose ``element_schema`` adds fields, override this upward.
    tokens_per_element: ClassVar[int] = 90

    #: Worst-case element count for one draw. 45 lines covers the longest script
    #: any stage asks for today (a 120 s narration at ~3 s/line, with headroom
    #: for a model that over-segments). It bounds the ANSWER estimate only — the
    #: real element ceiling is ``min_elements`` on the floor side and the
    #: stage's own validation on the ceiling side; this number never rejects a
    #: draw, it only sizes the budget.
    max_expected_elements: ClassVar[int] = 45

    @property
    def answer_tokens(self) -> int:
        """Worst-case tokens for the ANSWER alone, before reasoning headroom."""
        return self.tokens_per_element * self.max_expected_elements

    def __init__(self, client: WMClient | None = None) -> None:
        self._llm_timeout = llm_budget_seconds()
        self._llm_retries = llm_retries()
        self._wm = client or WMClient(timeout=llm_client_timeout_seconds())

    def short_circuit(self, ctx: RunContext) -> O | None:
        """Optional pre-LLM path (lesson's verbatim source-script). Default: none."""
        return None

    def under_floor_retry_block(self, error: ScriptDrawUnderFloorError, ctx: RunContext) -> str:
        """The addendum a RETRY carries after the previous attempt drew under the floor.

        Without it every retry POSTs the byte-identical prompt (it is built once,
        outside the attempt), i.e. asks the same question that was just answered
        with nothing. This states what came back and restates the minimum as a
        countable fact. Appended to the SAME prompt, so it costs no extra call, and
        only ever after a :class:`ScriptDrawUnderFloorError` — a first draw, and a
        retry for any other reason (shape error, timeout), are sent unchanged.

        A stage whose prompt also states a WORD minimum overrides this to restate
        it (``website_review``).
        """
        key = self.element_key
        return (
            f"\n\nRETRY — your previous answer was REJECTED: it contained {error.count} "
            f"`{key}`, and at least {error.required} are required. An empty or "
            f"near-empty `{key}` list is never a valid answer to this request. HARD "
            f"MINIMUM: return at least {error.required} `{key}` element(s), each "
            "carrying real spoken text written from the material above. This minimum "
            "OVERRIDES any instruction above to write less. Return the full script "
            "again in the same JSON shape."
        )

    @abstractmethod
    def build_prompt(self, ctx: RunContext) -> str:
        """Return the full user prompt, INCLUDING this stage's verbatim_prompt_block."""

    @abstractmethod
    def make_element(self, raw: dict, *, element_id: str) -> StageOutput:
        """Map one parsed element dict (+ a minted id) to the typed element model."""

    @abstractmethod
    def build_output(self, elements: list[StageOutput], ctx: RunContext) -> O:
        """Wrap elements into the typed output (total_seconds + any extra fields)."""

    def _map_elements(self, draw: _Draw) -> tuple[list[StageOutput], list[dict]]:
        """Map a draw's reply to ``(typed elements, raw element dicts)``.

        Pure in ``(draw.output, draw.element_ids)``, so a replayed draw yields the
        elements — ids included — the original did. Ids are assigned in order to the
        elements that SURVIVE the placeholder drop, as they always were.
        """
        try:
            # The reply is untrusted JSON: a non-object or a missing key raises here
            # and is re-prompted, exactly as the untyped subscript always did.
            reply = cast("dict[str, list[dict]]", draw.output)
            raw = _drop_placeholder_elements(list(reply[self.element_key]), stage=self.name)
            ids = iter(draw.element_ids)
            built = [self.make_element(r, element_id=next(ids, None) or str(uuid7())) for r in raw]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{self.name}: LLM output missing expected fields: {exc}") from exc
        return built, raw

    async def _draw(
        self, ctx: RunContext, request_hash: str, send: Callable[[], Awaitable[_Draw]]
    ) -> _Draw:
        """One model draw through the step ledger: a re-entered stage does not re-pay it.

        Step key ``draw:<request hash, 16 hex>:<n>`` — ``n`` counts the draws of
        this EXACT request this execution has already completed. The request hash
        makes the key independent of how the blueprint orders its ``handle`` calls
        (first draw, grounding re-ask, duration re-ask, …); ``n`` keeps two draws
        of a byte-identical request distinct, so an under-floor draw whose retry
        prompt repeats is still re-asked rather than answered from its own row.
        On re-entry the same sequence replays from the ledger, call for call.

        Stored: a reply that PARSED (including an under-floor one — replaying it
        re-takes the same retry, so the retry budget is spent exactly as in an
        uninterrupted run). Not stored: a reply that failed to parse, a timeout,
        any upstream error — ``send`` raised, the ledger kept nothing.
        """
        if self._draws_done is None:  # ``handle`` called without ``run`` (tests)
            self._draws_done = {}
        done = self._draws_done
        nth = done.get(request_hash, 0)
        draw = await ctx.step_ledger(self.name).once(
            f"draw:{request_hash[:16]}:{nth}",
            request_hash,
            send,
            dump=_Draw.dump,
            load=_Draw.load,
        )
        done[request_hash] = nth + 1
        mine = _ATTEMPT_DRAWS.get()
        if mine is not None:
            mine.append(request_hash)
        return draw

    async def _counting_consumed_draws[T](self, attempt: Callable[[], Awaitable[T]]) -> T:
        """Run one ``stage_call`` attempt; un-count its draw if the attempt was ABANDONED.

        The ``n`` in a draw's step key must be the same in process and on re-entry.
        It is, as long as ``_draws_done`` only ever counts a draw whose verdict the
        control flow consumed: the attempt returned, or raised a ``ValueError`` — a
        deterministic judgement of the reply (under the floor, unmappable) that a
        re-entry replaying the stored reply reaches again, then re-takes the retry.

        Any other exit — the per-attempt timeout (a ``CancelledError`` in here), a
        drain, a transport error — says nothing about the reply and will not recur
        on re-entry. Had it left the draw counted, the in-process retry of the SAME
        request would be keyed ``:n+1``: a second paid call, stored under a key a
        re-entry (which reads ``:n`` and carries on) never asks for. Restoring the
        count makes the retry read ``:n`` — the reply already paid for.

        Un-counted by DECREMENTING this attempt's own draws (``_ATTEMPT_DRAWS``), not
        by restoring a snapshot: concurrent attempts on one instance (``line_fit``
        rewrites lines in parallel) would have their completed draws wiped by it.
        """
        mine: list[str] = []
        token = _ATTEMPT_DRAWS.set(mine)
        try:
            return await attempt()
        except ValueError:
            raise
        except BaseException:
            done = self._draws_done
            if done is not None:
                for request_hash in mine:
                    done[request_hash] = max(0, done.get(request_hash, 0) - 1)
            raise
        finally:
            _ATTEMPT_DRAWS.reset(token)

    async def redraw_elements(self, ctx: RunContext, message: str) -> list[StageOutput]:
        """ONE corrective draw of a FRAGMENT (e.g. a single line), parsed as elements.

        The same WM call, schema, token budget, step ledger and per-attempt timeout
        as :meth:`handle`, minus three things that only make sense for a WHOLE
        script: the element floor (a fragment answering empty is a failed
        correction the caller already holds a fallback for), the verbatim gate
        (a fragment need not contain every phrase — the caller must not hand a
        verbatim-carrying fragment to this), and the retry ladder: exactly ONE
        model call (``retries=0``). A failed fragment costs nothing to give up —
        the caller keeps the take it already holds — while a retry would multiply
        the per-line call bound by ``llm_retries() + 1`` for no extra safety.
        Raises on failure; the caller keeps what it had.
        """
        output_schema = self.script_output_schema()
        max_tokens = output_token_budget(answer_tokens=self.answer_tokens)

        async def _attempt() -> list[StageOutput]:
            async def _send() -> _Draw:
                resp = await self._wm.message(
                    worker_name=self.worker_name,
                    message=message,
                    system_prompt=_SYSTEM_PROMPT,
                    output_schema=output_schema,
                    max_tokens=output_token_budget(answer_tokens=self.answer_tokens),
                    enable_reasoning=False,
                )
                draw = _Draw(
                    output=resp.output,
                    usage=getattr(resp, "usage_metrics", None),
                    element_ids=_mint_element_ids(resp.output, self.element_key),
                )
                draw.built = self._map_elements(draw)
                return draw

            draw = await self._draw(
                ctx,
                hash_inputs(
                    {
                        "worker_name": self.worker_name,
                        "message": message,
                        "system_prompt": _SYSTEM_PROMPT,
                        "output_schema": output_schema,
                        "max_tokens": max_tokens,
                        "enable_reasoning": False,
                    }
                ),
                _send,
            )
            built, _raw = draw.built if draw.built is not None else self._map_elements(draw)
            return built

        return await stage_call(
            lambda: guarded(lambda: self._counting_consumed_draws(_attempt)),
            timeout_seconds=self._llm_timeout,
            retries=0,  # one call: the caller's fallback IS the retry (see docstring)
            retry_on=LLM_RETRY_ON,
            stage_name=self.name,
        )

    async def handle(self, payload: I, ctx: RunContext) -> O:
        # Only the verbatim kinds THIS stage consumes (see STAGE_VERBATIM_KIND): a
        # slide-kind phrase is not narration and must not gate the script. The
        # `dropped` half is NOT discarded — no stage renders slide-kind verbatim
        # today, so silently dropping it would delete the user's phrase from the
        # video on a green run. `record_verbatim_drop` logs it and appends a
        # soft-degrade note, which surfaces the run as `degraded`.
        items, dropped = partition_for_stage(
            ctx.config_snapshot.get("verbatim_items") or [], self.name
        )
        early = self.short_circuit(ctx)
        if early is not None:
            return record_verbatim_drop(early, dropped, stage=self.name)
        prompt = self.build_prompt(ctx)
        #: The under-floor error of the most recent attempt that drew too few, and
        #: how many attempts have been made — both live across ``stage_call``'s
        #: retries of ``_attempt`` so a retry can say what the last answer was.
        under_floor: ScriptDrawUnderFloorError | None = None
        attempts = 0

        async def _attempt() -> tuple[list[StageOutput], list[dict]]:
            nonlocal under_floor, attempts
            attempts += 1
            # A retry that follows an under-floor draw is a DIFFERENT ask: the same
            # prompt plus what came back and the minimum, restated. Still one call.
            message = (
                prompt
                if under_floor is None
                else prompt + self.under_floor_retry_block(under_floor, ctx)
            )
            # Parse + map elements INSIDE the retry budget so a valid-JSON reply
            # missing the element list or a per-element field is re-prompted, not
            # run-fatal on the first draw. An empty/under-floor element list is
            # RE-PROMPTED here too (see below): a bare `[]` from an LLM is usually a
            # transient draw, not a permanent content deficiency. verbatim stays
            # terminal — that IS a real content gate, checked after the retry loop.
            output_schema = self.script_output_schema()
            # Hashed below; the call keeps its own inline ``output_token_budget(...)``
            # because ``tests/test_wm_max_tokens_gate.py`` reads the budget statically
            # AT the call site.
            max_tokens = output_token_budget(answer_tokens=self.answer_tokens)

            async def _send() -> _Draw:
                resp = await self._wm.message(
                    worker_name=self.worker_name,
                    message=message,
                    system_prompt=_SYSTEM_PROMPT,
                    output_schema=output_schema,
                    # Omitting this leaves WM's 6000 default, which a reasoning model
                    # can burn entirely on ThinkingParts before emitting any JSON —
                    # a terminal 422 that killed a live run. See `answer_tokens`
                    # above and `llm_budget.output_token_budget`.
                    max_tokens=output_token_budget(answer_tokens=self.answer_tokens),
                    # Chain-of-thought OFF, for EVERY script stage on this primitive.
                    # Measured through worker-management on the production path (Job
                    # `wmprobe-segsel`, 2026-08-20, same prompt and schema,
                    # `max_tokens=8000`): with reasoning ON the call hit the stage's own
                    # 950s deadline and returned NOTHING -- twice, on both output schemas.
                    # With reasoning OFF the same call answered in 34-67s carrying real
                    # content, every time. The model spends the whole budget thinking and
                    # never emits the answer, surfacing one layer up as `Exceeded maximum
                    # output retries (3)` or `StageTimeoutError: all 2 attempts timed out
                    # (per-attempt budget 950.0s)` -- which is how `website_review_script`
                    # died in production on 2026-08-20. The keyword lives HERE rather than
                    # on the subclass because every stage on this primitive makes the SAME
                    # call (build a structured element list; see the module docstring), and
                    # `website_review_script` and `promo_script` are BOTH already on the
                    # token-limit failure ledger in `llm_budget.py`. There is no script
                    # stage whose output is a reasoning artefact -- the elements are the
                    # answer, and #873 measured thinking-off content as equal or better.
                    #
                    # Only sendable because worker-management #188 added the field and
                    # `w2v_common` >= 0.61.0 carries it -- `MessageRequest` is
                    # `extra="forbid"`, so an older WM would 422 the call. Omit it and
                    # w2v's calls take WM's in-memory auto-create branch
                    # (`model_provider="vllm"`), straight to vLLM, never through
                    # inference-router -- the only component that would otherwise turn
                    # thinking off. See memory `w2v-wm-calls-bypass-inference-router`.
                    enable_reasoning=False,
                )
                draw = _Draw(
                    output=resp.output,
                    usage=getattr(resp, "usage_metrics", None),
                    element_ids=_mint_element_ids(resp.output, self.element_key),
                )
                # Mapped INSIDE the ledgered step: a reply this raises on (missing
                # element list, missing per-element field) is not stored, so the
                # retry — the same prompt, byte for byte — is a fresh paid draw
                # exactly as before, and a re-entry can never be served it.
                draw.built = self._map_elements(draw)
                return draw

            draw = await self._draw(
                ctx,
                # The exact request: everything that reaches WM. A corrective or
                # retry block changes ``message`` and so is a different, new draw.
                hash_inputs(
                    {
                        "worker_name": self.worker_name,
                        "message": message,
                        "system_prompt": _SYSTEM_PROMPT,
                        "output_schema": output_schema,
                        "max_tokens": max_tokens,
                        "enable_reasoning": False,
                    }
                ),
                _send,
            )
            data = draw.output
            built, raw = draw.built if draw.built is not None else self._map_elements(draw)
            floor = 0 if self._corrective else self.min_elements
            if len(built) < floor:
                # Retryable: raising inside _attempt spends the retry budget, so a
                # flaky empty draw gets re-prompted. If every attempt is still under
                # the floor, stage_call raises StageRetriesExhaustedError (a
                # StageError) whose message carries this "too few" text — so an empty
                # result STILL fails loud (never flows into storyboard → zero shots →
                # compile 422, bug-bash B1+B2), just after exhausting retries first.
                under_floor = ScriptDrawUnderFloorError(
                    f"{self.name}: LLM returned too few `{self.element_key}` "
                    f"({len(built)} < required {floor})",
                    count=len(built),
                    required=floor,
                )
                _log_under_floor_draw(
                    stage=self.name,
                    worker=self.worker_name,
                    attempt=attempts,
                    count=len(built),
                    required=floor,
                    prompt=message,
                    output=data,
                    usage=draw.usage,
                )
                raise under_floor
            return built, raw

        elements, raw_elements = await stage_call(
            # guarded() only reclassifies upstream HTTP failures; the deliberate
            # ValueError re-prompts above pass through it untouched and stay
            # retryable (ValueError is in LLM_RETRY_ON). A WM 402 / token-limit
            # failure, which no re-prompt can fix, now fails on attempt 1.
            lambda: guarded(lambda: self._counting_consumed_draws(_attempt)),
            timeout_seconds=self._llm_timeout,
            retries=self._llm_retries,
            retry_on=LLM_RETRY_ON,
            stage_name=self.name,
        )
        # element.text == str(raw["text"]) for every stage; join the raw texts so
        # the base needs no typed view of the concrete element model.
        text = " ".join(str(raw["text"]) for raw in raw_elements)
        assert_verbatim_present(text, items, stage=self.name)
        return record_verbatim_drop(self.build_output(elements, ctx), dropped, stage=self.name)
