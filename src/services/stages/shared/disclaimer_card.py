"""Compile-tail disclaimer card (A18 phase 2, AC-8).

The RENDER half of the legal gate: when ``claim_finish`` decided a comparative /
pricing / performance / testimonial claim mandates a disclaimer, its fixed
template text is spliced onto the END of the shot list as a solid text card.

No compiler change (spec §8 item 15): the card is an ordinary ``Shot.description``
— the solid-text-card shape video_compiler_service already renders — so nothing
downstream learns a new field.

Pure-splice shape, mirroring ``title_cards.splice_title_cards``: the ``ctx`` read
and the list surgery are separate functions, so the surgery is unit-testable with
no ``ctx`` at all and the AC-1 regression guard can assert byte-identity directly.

INERTNESS CONTRACT (AC-1, the make-or-break). Three independent ways to get no
card, each of them the default:

* ``claim_finish`` did not run (no ``legal.mode``, no corpus, no lines) ⇒ its
  stored output is ``{}`` ⇒ no ``disclaimers`` key ⇒ ``None``. This is the case for
  EVERY brief shipping today: prod briefs carry ``legal: {trademarks,
  defensibility}`` with **no** ``mode`` key, so the stage skips and no card appears.
* ``claim_finish`` ran but the rule engine mandated nothing ⇒ empty list ⇒ ``None``.
* ``legal.disclaimers.mode == "off"`` ⇒ the operator declined ⇒ ``None``.

``splice_disclaimer_card(shots, None)`` returns ``shots`` UNCHANGED (same object),
so an un-opted run's shot list is byte-identical to today.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

#: Stable, deterministic ids — no uuid, so a re-render reproduces the same card.
DISCLAIMER_SHOT_ID = "disclaimer-card"
DISCLAIMER_LINE_ID = "__disclaimer__"

#: On-screen budget for the card: long enough to read a couple of one-liners.
DISCLAIMER_SECONDS = 4.0

#: Hard clamp — no unbounded legal prose ever reaches the compiler (spec §5.3).
DISCLAIMER_MAX_CHARS = 240

_JOINER = " · "


def disclaimer_text(disclaimers: list[dict[str, Any]]) -> str:
    """Join the required disclaimers into one card body, clamped and deduped.

    Order is the order ``disclaimer_requirements`` emitted (a fixed kind order), so
    the body is deterministic. Truncation is on the joiner boundary where possible,
    so a clamped card never shows half a sentence.
    """
    seen: set[str] = set()
    parts: list[str] = []
    for entry in disclaimers or []:
        text = str((entry or {}).get("text") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        candidate = _JOINER.join([*parts, text])
        if len(candidate) > DISCLAIMER_MAX_CHARS:
            break
        parts.append(text)
    body = _JOINER.join(parts)
    if not body and disclaimers:
        # A single requirement longer than the whole budget: hard-truncate rather
        # than drop it — the disclaimer must be shown.
        body = str((disclaimers[0] or {}).get("text") or "").strip()[:DISCLAIMER_MAX_CHARS]
    return body


def build_disclaimer_shot(ctx: RunContext) -> dict[str, Any] | None:
    """The tail disclaimer card for this run, or ``None`` (the default).

    Reads ``claim_finish``'s persisted ``disclaimers`` — a skipped stage stores
    ``{}``, which is exactly why an un-opted run gets no card. Gated on
    ``legal.disclaimers.mode != "off"``.
    """
    finished = ctx.prior_output("claim_finish") or {}
    disclaimers = finished.get("disclaimers") or []
    if not isinstance(disclaimers, list) or not disclaimers:
        return None

    legal = ctx.config_snapshot.get("legal") or {}
    cfg = legal.get("disclaimers") if isinstance(legal, dict) else None
    mode = str((cfg or {}).get("mode", "auto")) if isinstance(cfg, dict) else "auto"
    if mode == "off":
        return None

    body = disclaimer_text(disclaimers)
    if not body:
        return None
    return {
        "id": DISCLAIMER_SHOT_ID,
        "line_id": DISCLAIMER_LINE_ID,
        "seconds": DISCLAIMER_SECONDS,
        "description": body,
    }


def splice_disclaimer_card(
    shots: list[dict[str, Any]], card: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Append the disclaimer card to a built shot list. Pure (no I/O).

    ``card is None`` ⇒ ``shots`` is returned UNCHANGED — the AC-1 regression
    guarantee, asserted directly in the tests.
    """
    if card is None:
        return shots
    return [*shots, card]
