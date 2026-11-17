"""audiobook_ingest — fetch the uploaded manuscript and split it into chapters.

The manuscript was staged at upload time (``brief.manuscript_ref``); we read it
back by key (no SSRF) and parse it into ordered chapters. ``default_pause_after``
on this stage lets the user review the detected chapter breakdown before any
TTS spend.

Inline pronunciation-proposer (PR-8a, opt-in): when
``brief.audiobook.propose_pronunciations`` is truthy, a WM worker inspects the
raw manuscript prose and proposes book-specific respellings for proper nouns,
invented, and foreign words a general TTS voice would mispronounce. Those
proposals are merged UNDER the human ``brief.pronunciations`` (the human always
wins) and fed to the SAME single ``build_lexicon`` → ``apply_lexicon`` parse, so
no second pass or ``config_snapshot`` write is needed. The proposer is SOFT: any
failure (WM down, 402/500, timeout, malformed output) logs a warning and falls
back to an empty proposal set, so a proposer error never fails a paid ingest.
Default-OFF: with the flag absent there is ZERO WM traffic and the parse is
byte-identical to a brief-only lexicon.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar, cast
from uuid import uuid7

from pydantic import Field
from w2v_common.stages import stage_call

from src.services.llm import WMClient
from src.services.manuscript_storage import fetch_manuscript
from src.services.stages.audiobook.parsing import ManuscriptParseError, parse_manuscript
from src.services.stages.audiobook.pronunciation import build_lexicon
from src.services.stages.base import (
    SlotUse,
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
    StagePreconditionError,
)
from src.services.stages.llm_budget import (
    LARGE_ANSWER_TOKENS,
    LLM_RETRY_ON,
    downshift_reasoning,
    llm_budget_seconds,
    llm_client_timeout_seconds,
    llm_retries,
    output_token_budget,
)
from src.services.stages.upstream import guarded

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

logger = logging.getLogger(__name__)

# WM worker name (regex ^[a-zA-Z0-9_-]+$). WM auto-creates it on first call and
# the per-call system_prompt steers it; provider/model is WM-side config, NOT
# here — that is the whole point of routing through WM. No worker_management
# change is needed to add this stage.
_PROPOSER_WORKER_NAME = "pronunciation-proposer"

# How much manuscript prose to show the proposer. Enough opening prose to spot
# the book's recurring proper nouns without shipping a whole novel to the LLM
# (and blowing the token budget) — names recur, so the head is representative.
_PROPOSER_EXCERPT_CHARS = 8000

# Defensive bounds on each returned entry: a respelling is short by nature, and
# an unbounded key/value would be a hallucination, not a pronunciation. Cap the
# number of entries so a runaway output cannot balloon the lexicon.
_PROPOSER_MAX_WORD_CHARS = 80
_PROPOSER_MAX_RESPELL_CHARS = 120
_PROPOSER_MAX_PROPOSALS = 200

# ``_PROPOSER_OUTPUT_SCHEMA`` is a ``proposals`` array whose length tracks the
# excerpt: up to ``_PROPOSER_MAX_PROPOSALS`` (200) objects of
# {word, ipa_or_phonetic, reason} — the reason is prose. 200 x ~30 tokens
# already exceeds LARGE, so this is the top tier and clamps at the chain cap.
_PROPOSER_MAX_TOKENS = output_token_budget(answer_tokens=LARGE_ANSWER_TOKENS)

_PROPOSER_SYSTEM_PROMPT = (
    "You are a pronunciation editor for an audiobook. You receive manuscript "
    "excerpts and propose spoken respellings ONLY for genuinely ambiguous, "
    "book-specific words: proper nouns, invented/coined words, and foreign terms "
    "a general TTS voice would mispronounce. Return `ipa_or_phonetic` as a "
    "plain-ASCII hyphenated respelling the narrator can read literally (e.g. "
    "`Mer-kee`), never raw IPA glyphs. Be conservative: propose nothing for "
    "ordinary English words, and NEVER propose a word that is also a common word, "
    "because the respelling is applied case-insensitively to every occurrence. "
    "Prefer fewer, high-confidence entries; if nothing is ambiguous, return an "
    "empty list."
)

# Object-rooted structured-output schema (WM convention). WM returns
# ``response.output`` already parsed to this shape.
_PROPOSER_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "word": {"type": "string"},
                    "ipa_or_phonetic": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["word", "ipa_or_phonetic"],
            },
        }
    },
    "required": ["proposals"],
}


def _build_proposer_message(excerpt: str) -> str:
    return (
        "Propose book-specific pronunciations for the following manuscript "
        "excerpt. Return only genuinely ambiguous proper nouns, invented, or "
        "foreign words a general TTS voice would mispronounce; return an empty "
        f"list if nothing is ambiguous.\n\nMANUSCRIPT EXCERPT:\n{excerpt}"
    )


def _parse_proposals(output: object) -> dict[str, str]:
    """Coerce a WM proposer response into a ``{word: respelling}`` map.

    Advisory model, authoritative stage: every entry is validated and clamped.
    A non-Mapping output, a missing/garbage ``proposals`` list, an item that is
    not a Mapping, or an entry missing/blank ``word`` or ``ipa_or_phonetic`` is
    skipped — never raised — and over-long strings are dropped. The map is
    capped at ``_PROPOSER_MAX_PROPOSALS`` entries.
    """
    if not isinstance(output, Mapping):
        return {}
    raw = cast("Mapping[str, object]", output).get("proposals")
    if not isinstance(raw, list):
        return {}
    proposals: dict[str, str] = {}
    for raw_item in raw:
        if not isinstance(raw_item, Mapping):
            continue
        item = cast("Mapping[str, object]", raw_item)
        word = item.get("word")
        respell = item.get("ipa_or_phonetic")
        if not isinstance(word, str) or not isinstance(respell, str):
            continue
        word = word.strip()
        respell = respell.strip()
        if not word or not respell:
            continue
        if len(word) > _PROPOSER_MAX_WORD_CHARS or len(respell) > _PROPOSER_MAX_RESPELL_CHARS:
            continue
        proposals[word] = respell
        if len(proposals) >= _PROPOSER_MAX_PROPOSALS:
            break
    return proposals


class AudiobookIngestInput(StageInput):
    pass


class IngestedChapter(StageOutput):
    id: str
    title: str
    text: str
    sequence: int


class AudiobookIngestOutput(StageOutputWithNotes):
    # Declared output contract — enforced once, centrally, in
    # `runs/dag_executor.py::_enforce_produces`. Empty here means the stage
    # reported success having produced nothing, which degrades the run.
    # `audiobook_plan` and `audiobook_segment` both read `chapters`; every stage below
    # is a projection of it.
    produces: ClassVar[tuple[str, ...]] = ("chapters",)

    title: str
    author: str
    chapters: list[IngestedChapter]
    # The frozen ``{written: spoken}`` map actually applied to this ingest —
    # proposer output merged UNDER the human brief. Empty when the proposer is
    # off and the brief carries no pronunciations, so a re-run / manifest can
    # read exactly what was applied. Default-factory keeps the shape sane.
    pronunciations: dict[str, str] = Field(default_factory=dict)


class AudiobookIngestStage(StageHandler[AudiobookIngestInput, AudiobookIngestOutput]):
    name = "audiobook_ingest"
    input_model = AudiobookIngestInput
    output_model = AudiobookIngestOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("llm"),)

    def __init__(self, client: WMClient | None = None) -> None:
        # Read the budget at construction (not import): get_settings() is an
        # lru_cache singleton, so a lazy read lets test/env overrides applied
        # after import still take effect. Constructing a WMClient does NO
        # network — the opt-out path never touches it.
        self._llm_timeout = llm_budget_seconds()
        self._llm_retries = llm_retries()
        self._wm = client or WMClient(timeout=llm_client_timeout_seconds())

    async def handle(
        self,
        payload: AudiobookIngestInput,
        ctx: RunContext,
    ) -> AudiobookIngestOutput:
        brief = ctx.config_snapshot.get("brief") or {}
        manuscript_ref = brief.get("manuscript_ref")
        if not manuscript_ref:
            raise StagePreconditionError(
                "audiobook_ingest: brief.manuscript_ref is missing — no manuscript was uploaded",
                missing=["manuscript_ref"],
            )
        filename = str(brief.get("manuscript_name") or "manuscript.txt")
        single = str(brief.get("chapter_mode") or "auto") == "single"
        fallback_title = str(ctx.config_snapshot.get("title") or "Audiobook")

        data = await fetch_manuscript(str(manuscript_ref))

        # Inline pronunciation-proposer (opt-in, default-OFF). Run BEFORE
        # build_lexicon on the RAW manuscript prose (pre-lexicon, pre-normalize)
        # so it spots the book's own names; then merge UNDER the human brief so
        # brief.pronunciations always wins, and feed the SINGLE parse below.
        # Coerce defensively: a malformed brief may carry `audiobook` as a truthy
        # non-dict (a string/list), and `.get(...)` below would AttributeError out of
        # this paid stage — the same guard the adjacent `pronunciations` coercion uses.
        audiobook_cfg = ab if isinstance(ab := brief.get("audiobook"), dict) else {}
        # A malformed brief may carry `pronunciations` as a non-dict (e.g. a list):
        # `.get(...) or {}` would let it through and blow up the `{**proposals,
        # **brief_pron}` spread below with a TypeError, HARD-FAILING a paid ingest.
        # Coerce defensively — an unusable override degrades to "no custom pron".
        raw_pron = brief.get("pronunciations")
        if raw_pron and not isinstance(raw_pron, dict):
            logger.warning(
                "audiobook_ingest run=%s ignoring non-dict brief.pronunciations (%s)",
                ctx.run_id,
                type(raw_pron).__name__,
            )
        brief_pron: dict[str, str] = raw_pron if isinstance(raw_pron, dict) else {}
        proposals: dict[str, str] = {}
        if audiobook_cfg.get("propose_pronunciations"):
            excerpt = data.decode("utf-8", errors="replace")[:_PROPOSER_EXCERPT_CHARS]
            proposals = await self._propose_pronunciations(excerpt, ctx)
        merged_pron = {**proposals, **brief_pron}
        lexicon = build_lexicon(merged_pron)

        logger.info(
            "audiobook_ingest run=%s manuscript=%r bytes=%d chapter_mode=%s "
            "pronunciations=%d (proposed=%d brief=%d)",
            ctx.run_id,
            filename,
            len(data),
            "single" if single else "auto",
            len(lexicon),
            len(proposals),
            len(brief_pron),
        )
        try:
            manuscript = parse_manuscript(
                data,
                filename,
                single_chapter=single,
                fallback_title=fallback_title,
                lexicon=lexicon,
            )
        except ManuscriptParseError as exc:
            logger.error("audiobook_ingest run=%s parse failed: %s", ctx.run_id, exc)
            raise StagePreconditionError(f"audiobook_ingest: {exc}") from exc

        logger.info(
            "audiobook_ingest run=%s parsed title=%r author=%r chapters=%d",
            ctx.run_id,
            manuscript.title,
            manuscript.author,
            len(manuscript.chapters),
        )
        for chapter in manuscript.chapters:
            logger.debug(
                "audiobook_ingest run=%s chapter seq=%d title=%r chars=%d",
                ctx.run_id,
                chapter.sequence,
                chapter.title,
                len(chapter.text),
            )

        return AudiobookIngestOutput(
            title=manuscript.title,
            author=manuscript.author,
            chapters=[
                IngestedChapter(
                    id=str(uuid7()),
                    title=chapter.title,
                    text=chapter.text,
                    sequence=chapter.sequence,
                )
                for chapter in manuscript.chapters
            ],
            pronunciations=merged_pron,
        )

    async def _propose_pronunciations(self, excerpt: str, ctx: RunContext) -> dict[str, str]:
        """Ask the WM proposer for book-specific pronunciations. SOFT-degrade.

        Wrapped in the same ``stage_call(downshift_reasoning(lambda think: guarded(lambda: _attempt(think))), …)`` ladder
        as the other LLM stages, but the whole call is caught: a proposer error
        (WM 402/500, timeout, transport, malformed output) NEVER fails a paid
        ingest — it logs a warning and falls back to no proposals, i.e. the
        brief-only lexicon.
        """

        async def _attempt(enable_reasoning: bool | None) -> dict[str, str]:
            resp = await self._wm.message(
                enable_reasoning=enable_reasoning,
                worker_name=_PROPOSER_WORKER_NAME,
                message=_build_proposer_message(excerpt),
                system_prompt=_PROPOSER_SYSTEM_PROMPT,
                output_schema=_PROPOSER_OUTPUT_SCHEMA,
                max_tokens=_PROPOSER_MAX_TOKENS,
            )
            return _parse_proposals(resp.output)

        try:
            return await stage_call(
                downshift_reasoning(lambda think: guarded(lambda: _attempt(think))),
                timeout_seconds=self._llm_timeout,
                retries=self._llm_retries,
                retry_on=LLM_RETRY_ON,
                stage_name=f"{self.name}:propose_pronunciations",
            )
        except Exception as exc:  # noqa: BLE001 — SOFT: a proposer error must never fail a paid ingest
            logger.warning(
                "audiobook_ingest run=%s pronunciation proposer failed — "
                "falling back to brief-only lexicon: %s",
                ctx.run_id,
                exc,
            )
            return {}
