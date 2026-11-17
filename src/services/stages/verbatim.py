"""Verbatim-phrase helpers + source-script normalization.

Split out of ``base.py``; re-exported there for import-path stability. Used by
the LLM script stages to (a) instruct the model to include required phrases and
(b) fail the stage loud if any phrase is missing from the generated text.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from pydantic import BaseModel
from w2v_common.stages import StageError

log = logging.getLogger(__name__)

__all__ = [
    "STAGE_VERBATIM_KIND",
    "VERBATIM_DROP_LOG_EVENT",
    "assert_verbatim_present",
    "find_missing_verbatim",
    "items_for_stage",
    "normalize_for_match",
    "normalize_source_script",
    "partition_for_stage",
    "record_verbatim_drop",
    "seconds_for",
    "verbatim_drop_note",
    "verbatim_prompt_block",
]


# Stage name → the verbatim ``kind`` that stage consumes. Mirrors the frontend's
# ``VERBATIM_STAGE_TO_KIND`` (w2v_frontend ``src/lib/types/api.ts``): the UI's
# two-track editor hard-locks a new item to ``slide`` or ``tts``, and every stage
# below writes/checks NARRATION, so they all consume ``tts``.
#
# A phrase the user filed under "Slide verbatim" is NOT narration: before this map
# existed, every script stage asserted it against the generated narration and
# raised StageError → stage failed → whole run failed, on the default UI path.
#
# BACKWARD COMPATIBILITY (load-bearing — see :func:`items_for_stage`): a stage
# ABSENT from this map keeps today's apply-to-everything behaviour, and so does an
# item with no ``kind`` (pre-0017 snapshots). Only a stage listed here narrows, and
# only against items that actually declare a conflicting kind.
STAGE_VERBATIM_KIND: dict[str, str] = {
    "script_draft": "tts",
    "promo_script": "tts",
    "lesson_script": "tts",
    "app_review_script": "tts",
    "website_review_script": "tts",
    "cro_audit_script": "tts",
    "seo_audit_script": "tts",
    "script_doctor": "tts",
}


def partition_for_stage(items: list[dict], stage: str) -> tuple[list[dict], list[dict]]:
    """Split ``items`` into ``(kept, dropped)`` for ``stage``.

    Two deliberate pass-through cases keep every currently-green run green:

    * ``stage`` not in :data:`STAGE_VERBATIM_KIND` ⇒ every item applies (a stage
      nobody has classified behaves exactly as it does today).
    * an item with a missing/empty ``kind`` ⇒ applies to every stage (a snapshot
      written before migration 0017 has no ``kind`` key at all).

    Only an item that explicitly declares a DIFFERENT kind than the stage consumes
    is dropped. That narrowness matters: if this ever returned an empty ``kept``,
    verbatim enforcement would vanish silently and every run would go green.

    The ``dropped`` half is NOT discardable. Today no stage consumes ``kind='slide'``
    (every slide stage is a deterministic projection with no LLM prompt), and
    ``slide`` is the SPA's DEFAULT kind — so dropping it silently would delete the
    user's phrase from the video while the run reports green. Callers MUST feed
    ``dropped`` to :func:`record_verbatim_drop`.
    """
    wanted = STAGE_VERBATIM_KIND.get(stage)
    if wanted is None:
        return list(items), []
    kept: list[dict] = []
    dropped: list[dict] = []
    for it in items:
        (kept if (not it.get("kind") or it.get("kind") == wanted) else dropped).append(it)
    return kept, dropped


def items_for_stage(items: list[dict], stage: str) -> list[dict]:
    """The ``kept`` half of :func:`partition_for_stage` (see it for the contract).

    Prefer :func:`partition_for_stage` in a stage handler: this function discards
    the ``dropped`` half, which must be recorded, never silently thrown away.
    """
    return partition_for_stage(items, stage)[0]


# Greppable log event so an operator can answer "why is my phrase missing from the
# video?" straight from the logs, without a DB query.
VERBATIM_DROP_LOG_EVENT = "verbatim_items_dropped"


def verbatim_drop_note(dropped: list[dict], stage: str) -> str:
    """The human-readable soft-degrade note for verbatim items ``stage`` did not apply."""
    rendered = "; ".join(
        f"{str(it.get('text', ''))!r} (kind={it.get('kind') or 'unset'!s})" for it in dropped
    )
    consumes = STAGE_VERBATIM_KIND.get(stage, "any")
    return (
        f"verbatim: {len(dropped)} item(s) NOT applied by {stage} — it consumes "
        f"kind={consumes!r}. No stage currently renders these, so they are absent "
        f"from the output: {rendered}"
    )


def record_verbatim_drop[M: BaseModel](output: M, dropped: list[dict], *, stage: str) -> M:
    """Record narrowed-away verbatim items on the output's soft-degrade ``notes``.

    Returns ``output`` unchanged when nothing was dropped. Otherwise emits a
    structured :data:`VERBATIM_DROP_LOG_EVENT` log line AND appends a note — a
    ``completed`` stage whose persisted output carries non-empty ``notes`` is
    surfaced as a ``degraded`` run by ``api/routes/runs/_shared.py``, which is this
    codebase's established non-blocking warning channel (``demo_video_record``'s
    clip QA uses the same one).

    Deliberately never raises: a demo must not die because a phrase went unplaced.
    """
    if not dropped:
        return output
    log.warning(
        "%s stage=%s count=%d ids=%s kinds=%s",
        VERBATIM_DROP_LOG_EVENT,
        stage,
        len(dropped),
        ",".join(str(it.get("id") or "?") for it in dropped),
        ",".join(str(it.get("kind") or "unset") for it in dropped),
    )
    if "notes" not in type(output).model_fields:
        # Loud, but non-fatal. `test_every_narrowing_stage_output_can_carry_notes`
        # makes this branch unreachable for every stage in STAGE_VERBATIM_KIND.
        log.error(
            "%s stage=%s output model %s has no `notes` channel — drop is LOG-ONLY",
            VERBATIM_DROP_LOG_EVENT,
            stage,
            type(output).__name__,
        )
        return output
    existing = list(getattr(output, "notes", []))
    return output.model_copy(update={"notes": [*existing, verbatim_drop_note(dropped, stage)]})


def seconds_for(text: str, words_per_second: float = 2.5) -> float:
    """Estimate spoken duration (≥1.0s) of ``text`` at ``words_per_second``.

    The single source of truth for verbatim narration timing — used by both the
    source-script normalizer and the shotlist normalizer so a line's planned
    duration is identical however it was authored. (``assemble_shots`` later
    overrides this with the real narration-clip duration; this only feeds the
    pre-narrate ordering math.)
    """
    return max(1.0, round(len(text.split()) / words_per_second, 2))


def normalize_source_script(
    text: str,
    *,
    split: Literal["line", "paragraph"],
    words_per_second: float = 2.5,
) -> list[tuple[str, float]]:
    """Turn a user-supplied literal script into (text, seconds) segments.

    Verbatim: each segment's text is the stripped source segment, unchanged —
    no rewording. Returns [] for empty/whitespace-only input (caller treats []
    as "no source script supplied" and falls back to LLM generation).

    split="line": one segment per non-blank line (video scripts).
    split="paragraph": one segment per blank-line-separated block (lesson scripts).
    """
    if not text or not text.strip():
        return []
    raw_segments = text.split("\n") if split == "line" else re.split(r"\n\s*\n", text)
    segments: list[tuple[str, float]] = []
    for raw in raw_segments:
        seg = raw.strip()
        if not seg:
            continue
        segments.append((seg, seconds_for(seg, words_per_second)))
    return segments


#: Typographic variants an LLM substitutes freely, folded to their ASCII form
#: before matching. NFKC alone does NOT do this: it leaves U+2019 (') and the
#: dash family untouched, because they are distinct characters rather than
#: compatibility forms.
_TYPOGRAPHIC_FOLD = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u2032": "'",
        "\u2033": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u00a0": " ",
        "\u202f": " ",
        "\u2026": "...",
    }
)


def normalize_for_match(text: str) -> str:
    """Lowercase + fold typographic punctuation + collapse whitespace, stripped.

    PUBLIC on purpose. This is the single matcher behind BOTH the gate that
    FAILS a run on a missing verbatim phrase (``find_missing_verbatim`` ->
    ``assert_verbatim_present``) and the read-time derivation that REPORTS a
    phrase as placed (``src.services.verbatim_placement``). A second,
    private copy on either side would be a lie by construction: the UI would
    show "Pending" for a run that passed the gate, or "Placed" for a phrase the
    gate never saw. Never fork it.

    ⚠️ It used to fold case and whitespace ONLY, and that made the gate reject
    phrases that are verbatim by any human reading. A source phrase typed with a
    straight apostrophe comes back from the model with U+2019; a hyphen comes
    back as an en- or em-dash; a space comes back non-breaking. Containment then
    fails, ``assert_verbatim_present`` raises, and the whole run is FAILED over a
    character the user cannot see. Folding is deliberately one-directional and
    narrow — quotes, the dash family, the two non-breaking spaces, the ellipsis —
    so it can widen what counts as "the same phrase" without ever letting a
    genuinely different phrase through.

    ⚠️ It is an EXPLICIT TABLE and deliberately not ``unicodedata.normalize(
    "NFKC", ...)``. The first version applied NFKC first, and NFKC is a
    COMPATIBILITY fold: it rewrites characters that mean different things.
    Measured — ``"5² cases"`` and ``"52 cases"`` normalise identically, as do
    ``"㎡"``/``"m2"``, ``"™"``/``"TM"``, ``"℃"``/``"°C"``. Since this function
    backs the gate, that made the gate FAIL OPEN: a run that rendered "52 cases"
    where "5² cases" was required would have passed. NFKC also contributed
    nothing the table does not already do for the actual failure — smart quotes
    and dashes are distinct characters, not compatibility forms, so NFKC leaves
    them alone.
    """
    folded = text.translate(_TYPOGRAPHIC_FOLD)
    return re.sub(r"\s+", " ", folded).strip().lower()


def verbatim_prompt_block(items: list[dict]) -> str:
    """Prompt fragment instructing the LLM to include each item's text verbatim.

    Returns "" when there are no items, so callers can unconditionally append it.
    """
    if not items:
        return ""
    joined = "\n".join(f'- "{it["text"]}"' for it in items)
    return (
        "\n\nThese exact phrases MUST appear verbatim, word-for-word, somewhere in "
        f"the output text:\n{joined}"
    )


def find_missing_verbatim(text: str, items: list[dict]) -> list[str]:
    """Return the item texts NOT present in `text` (case- and whitespace-insensitive).

    Comparison normalizes both sides (lowercase, whitespace collapsed) and checks
    substring containment. Empty list ⇒ every required phrase is present.
    """
    haystack = normalize_for_match(text)
    return [it["text"] for it in items if normalize_for_match(str(it["text"])) not in haystack]


def assert_verbatim_present(text: str, items: list[dict], *, stage: str) -> None:
    """Raise StageError if any required verbatim phrase is missing from `text`.

    Fail-loud so the stage → failed → run → failed, rather than silently shipping
    a video that dropped a phrase the user marked as required.
    """
    missing = find_missing_verbatim(text, items)
    if missing:
        rendered = "; ".join(repr(m) for m in missing)
        raise StageError(f"{stage}: required verbatim text missing from output: {rendered}")
