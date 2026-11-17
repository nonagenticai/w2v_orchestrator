"""Pure readers of this run's ``source_ingest`` output, for the stages that ground on it.

Split out of ``source_ingest.py`` so a stage that only READS the grounding text does
not import the ingest stage itself — and with it the document extractor and the
``ocr`` provider slot (``tests/unit/test_stage_provider_slots.py`` follows imports:
importing the stage module would make every grounded stage "reach" OCR).
``source_ingest`` re-exports both helpers for existing callers.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

__all__ = ["grounding_prompt_block", "grounding_sentences"]

#: How much of that stored text any ONE prompt may carry.
#:
#: ``_GROUNDING_MAX_CHARS`` bounds what we PERSIST; this bounds what we PASTE. The
#: two were the same number, which meant a fully-populated grounding block was
#: ~6 000-6 800 tokens (~3.5-4 chars/token) — on its own equal to or larger than
#: the entire budget of the workers that receive it. WM's answer to that is a 500
#: whose body reads "Model token limit (6000) exceeded before any response was
#: generated", and ``stages/upstream.py`` correctly classifies it PERMANENT, so the
#: stage fails outright rather than retrying.
#:
#: 8 000 chars is ~2 000-2 300 tokens: it still dominates every caller's preamble
#: (the largest, ``course_outline``, is under 3 000 chars) while leaving the
#: smallest worker budget in the fleet (4 000 tokens) room to answer. Callers with
#: an unusually large preamble should pass a smaller ``max_chars`` rather than
#: moving this — and raising a shared worker's budget to absorb one caller's paste
#: is the wrong direction entirely.
_GROUNDING_PROMPT_MAX_CHARS = 8000


def grounding_prompt_block(ctx: RunContext, *, max_chars: int = _GROUNDING_PROMPT_MAX_CHARS) -> str:
    """A prompt block grounding the script in the ingested source content.

    Returns "" when ``source_ingest`` didn't run or found nothing, so callers can
    unconditionally append it. Read from this run's ``source_ingest`` output.

    Bounded by :data:`_GROUNDING_PROMPT_MAX_CHARS`, NOT by the (3x larger) storage
    cap — see that constant for why the two must not be the same number.
    """
    out = ctx.prior_output("source_ingest") or {}
    text = str(out.get("grounding_text") or "").strip()
    if not text:
        return ""
    return (
        "\n\nGround this in the SOURCE MATERIAL the operator provided (use it for "
        "accurate facts/features/claims; synthesize, do not quote verbatim):\n"
        f"{text[:max_chars]}"
    )


def grounding_sentences(ctx: RunContext, *, max_sentences: int = 40) -> list[str]:
    """Sentences derived from the ingested grounding text (advisory claim-lint input).

    Returns ``[]`` when ``source_ingest`` didn't run or found nothing, so callers
    can unconditionally use it. Derived from the ALREADY-AGGREGATED
    ``grounding_text`` (this run's ``source_ingest`` output) — NOT by re-parsing
    the raw sources. Capped at ``max_sentences``. Pure; used by the ORCH-0.7
    claim-grounding lint to check marketing superlatives against source material.
    """
    out = ctx.prior_output("source_ingest") or {}
    text = str(out.get("grounding_text") or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences[:max_sentences]
