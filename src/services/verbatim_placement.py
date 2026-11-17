"""Read-time derivation of verbatim-item placement.

``verbatim_items`` has NO lifecycle columns. Migration 0009 created
``status`` / ``placed_in_stage`` / ``placed_in_element_id``; the Studio v3 reset
(0015) dropped the whole table CASCADE; 0017 re-added only
``kind``/``hint``/``scope``/``scope_target``. Nothing in the codebase has ever
written ``'placed'``, so the SPA's per-item lifecycle badge was pinned to amber
"Pending" forever and its "jump to the carrier element" affordance was
unreachable — even though the pipeline genuinely enforces verbatim phrases
(``assert_verbatim_present`` fails the run on a miss).

This module closes that gap WITHOUT a schema change, by re-matching each phrase
against the run's CURRENTLY-PERSISTED carrier stage outputs at read time.

Why derive rather than record a breadcrumb at stage time: the studio can rewrite
element text after a run via ``patch_stage_element``
(``src/api/routes/runs/stages.py``). A placement recorded when the stage ran
would silently become a lie the moment an editor deletes the phrase. Re-matching
what is stored right now cannot.

Three rules keep the derived answer honest, each mirroring a pipeline contract
rather than re-deciding it:

1. **Matching** is the shared ``normalize_for_match`` — the exact comparison the
   run-failing gate uses.
2. **Kind** is the shared ``items_for_stage`` — an item is only ever matched
   against a stage that was ALLOWED to carry its ``kind``.
3. **Precedence** is authoritative, not best-effort — within a chain of stages
   that re-emit the same spine, only the latest one to produce output is
   consulted. A superseded draft never rescues a phrase the shipped spine
   dropped.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

from src.api.schemas._shared import VerbatimStatus
from src.services.stages.demo_video.spine import PLAN_STAGE_PRECEDENCE
from src.services.stages.verbatim import items_for_stage, normalize_for_match

__all__ = [
    "VERBATIM_CARRIER_CHAINS",
    "VERBATIM_CARRIER_PATHS",
    "Placement",
    "derive_placements",
]


@dataclass(frozen=True, slots=True)
class Carrier:
    """A stage plus the output keys whose elements carry verbatim-bearing text."""

    stage: str
    paths: tuple[str, ...]


# Carriers grouped into PRECEDENCE CHAINS. Within a chain, the stages all re-emit
# the SAME spine (same element ids, revised text), so only the highest-precedence
# one that actually produced output is authoritative — see ``_chain_placement``.
# Across chains there is no precedence: they belong to different blueprints and
# never co-occur in one run.
#
# The carrier set is derived from the production callers of
# ``assert_verbatim_present`` — the gate that fails a run on a missing phrase —
# because a carrier set narrower than the gate's reach would report "Pending" for
# a run that demonstrably shipped the phrase.
#
# NOT the same set as ``runs.elements._ELEMENT_PATHS``, and deliberately not a
# subset of it. Several carriers here — ``website_review_script``,
# ``seo_audit_script``, ``cro_audit_script`` and the whole plan chain except
# ``demo_video_gen`` — are absent from ``_ELEMENT_PATHS``, so their elements are
# not studio-pinnable today. Their ids are still the real, stable ids their own
# stage emitted, so a derived ``placed_in_element_id`` is truthful; it just has no
# annotation target until ``_ELEMENT_PATHS`` gains those stages (a separate
# concern — widening it here would silently change what the annotation-create
# route accepts).
#
# Conversely most of ``_ELEMENT_PATHS`` is excluded: ``storyboard_draft.shots``,
# ``promo_analyze``, ``app_critique``, ``lesson_slides``, ``course_outline`` etc.
# hold analysis/visual metadata, not the narration the verbatim gate reads. A
# coincidental substring hit there is not a placement.
VERBATIM_CARRIER_CHAINS: tuple[tuple[Carrier, ...], ...] = (
    # Single-stage chains: one script stage per blueprint, no supersession.
    (Carrier("script_draft", ("lines",)),),
    (Carrier("promo_script", ("lines",)),),
    (Carrier("lesson_script", ("paragraphs",)),),
    (Carrier("app_review_script", ("lines",)),),
    (Carrier("website_review_script", ("lines",)),),
    (Carrier("seo_audit_script", ("lines",)),),
    (Carrier("cro_audit_script", ("lines",)),),
    # The demo/showcase spine chain, highest precedence FIRST. Built from
    # ``PLAN_STAGE_PRECEDENCE`` itself rather than a hand-copied list, so it can
    # never drift from the resolution every downstream consumer uses
    # (``spine.resolve_plan_output``). Every stage in it emits ``lines`` with the
    # same ``clip-<n>`` ids.
    #
    # This is what makes the old KNOWN LIMIT go away. ``claim_finish`` /
    # ``retention_plan`` / ``reconcile_timeline`` rewrite line TEXT with no
    # verbatim re-assert; because the chain consults only the LATEST stage with
    # output, a phrase one of them rewrote away now reads as not-placed instead of
    # being rescued by the ``script_doctor`` draft underneath it. (The pipeline
    # still does not FAIL such a run — that needs a re-assert in those stages and
    # is out of scope here — but the badge no longer claims a success the render
    # does not contain.)
    tuple(Carrier(stage, ("lines",)) for stage in PLAN_STAGE_PRECEDENCE),
)

# Flat stage → paths view. The route uses it to decide which stage_outputs rows to
# load; tests use it to enumerate carriers. Order is chain order.
VERBATIM_CARRIER_PATHS: Mapping[str, tuple[str, ...]] = {
    carrier.stage: carrier.paths for chain in VERBATIM_CARRIER_CHAINS for carrier in chain
}

# Stages whose verbatim `kind` classification is INHERITED from another stage.
#
# ``STAGE_VERBATIM_KIND`` classifies the stages that CALL the gate. The rest of
# the plan chain never calls it (``demo_video_gen`` never consults verbatim at
# all; the three mutators only rewrite), so ``items_for_stage`` would pass every
# kind through for them — and a `slide` phrase would get attributed to the demo
# spine purely because the LLM echoed it. They all carry the same narration spine
# that ``script_doctor`` gates, so they inherit its classification. Mapping to an
# existing key rather than declaring a kind keeps ``STAGE_VERBATIM_KIND`` the
# single source of truth for what a kind means.
_KIND_INHERITS_FROM: Mapping[str, str] = {
    stage: "script_doctor" for stage in PLAN_STAGE_PRECEDENCE if stage != "script_doctor"
}


@dataclass(frozen=True, slots=True)
class Placement:
    """Where (if anywhere) a verbatim phrase currently lives in a run's output."""

    status: VerbatimStatus
    stage: str | None
    element_id: str | None


_PENDING = Placement(status="pending", stage=None, element_id=None)
# No carrier that ran was ever ALLOWED to carry this item's kind. Distinct from
# `pending` ("we looked at an eligible stage and did not find it"): nothing is
# going to place this phrase, so leaving it amber would be the same stuck badge
# this module exists to unstick.
_REJECTED = Placement(status="rejected", stage=None, element_id=None)


class VerbatimLike(Protocol):
    """The fields derivation needs — satisfied by the ``VerbatimItem`` ORM row."""

    @property
    def id(self) -> UUID: ...

    @property
    def text(self) -> str: ...

    @property
    def kind(self) -> str | None: ...


def _element_texts(
    output: Mapping[str, Any], paths: tuple[str, ...]
) -> Iterator[tuple[str | None, str]]:
    """Yield ``(element_id, text)`` for every well-formed element at ``paths``.

    Malformed nodes (non-dict, no string ``text``) are skipped rather than
    raising: a degraded row must not 500 a listing. A node with text but no
    string ``id`` yields ``None`` — it matched, but there is nothing to jump to.
    """
    for path in paths:
        nodes = output.get(path)
        if not isinstance(nodes, list):
            continue
        for node in nodes:
            if not isinstance(node, dict):
                continue
            text = node.get("text")
            if not isinstance(text, str) or not text:
                continue
            element_id = node.get("id")
            yield (element_id if isinstance(element_id, str) and element_id else None), text


def _authoritative(
    chain: tuple[Carrier, ...], stage_outputs: Mapping[str, Mapping[str, Any] | None]
) -> tuple[Carrier, Mapping[str, Any]] | None:
    """Return the highest-precedence carrier in ``chain`` that produced output.

    A skipped opt-in stage stores a falsy output, so the chain falls past it —
    mirroring ``spine.resolve_plan_output``. Returns ``None`` when no stage in the
    chain ran, i.e. this chain has nothing to say about the run.
    """
    for carrier in chain:
        output = stage_outputs.get(carrier.stage)
        if output:
            return carrier, output
    return None


def _match_in(needle: str, carrier: Carrier, output: Mapping[str, Any]) -> Placement | None:
    pairs = list(_element_texts(output, carrier.paths))
    if not pairs:
        return None
    for element_id, text in pairs:
        if needle in normalize_for_match(text):
            return Placement(status="placed", stage=carrier.stage, element_id=element_id)
    # No single element carries the whole phrase. The gate joins element texts
    # with " " before checking (every ``assert_verbatim_present`` call site does
    # exactly this), so a phrase straddling two elements PASSES the gate and must
    # report as placed — but with no element id, because none of them actually
    # carries it. Never invent one.
    if needle in normalize_for_match(" ".join(text for _, text in pairs)):
        return Placement(status="placed", stage=carrier.stage, element_id=None)
    return None


def _kind_allows(item_kind: str | None, stage: str) -> bool:
    """Whether an item of ``item_kind`` was allowed to be carried by ``stage``.

    Delegates to the shared :func:`items_for_stage` — the same narrowing the stage
    applies before asserting — so the badge can never claim a placement in a stage
    the gate would have excluded the item from. ``_KIND_INHERITS_FROM`` resolves
    the spine stages that carry narration but never call the gate themselves.
    """
    classified_as = _KIND_INHERITS_FROM.get(stage, stage)
    return bool(items_for_stage([{"text": "", "kind": item_kind}], classified_as))


def _placement_for(
    needle: str, item_kind: str | None, stage_outputs: Mapping[str, Mapping[str, Any] | None]
) -> Placement:
    saw_eligible_carrier = False
    saw_forbidden_carrier = False
    for chain in VERBATIM_CARRIER_CHAINS:
        resolved = _authoritative(chain, stage_outputs)
        if resolved is None:
            continue
        carrier, output = resolved
        if not _kind_allows(item_kind, carrier.stage):
            # This chain ran but was contractually forbidden from carrying this
            # item's kind. Do NOT look inside it: a hit would be coincidence.
            saw_forbidden_carrier = True
            continue
        saw_eligible_carrier = True
        # Only the authoritative stage is consulted. Falling through to an earlier
        # stage in the same chain would let a superseded draft report a success the
        # shipped spine no longer contains.
        placement = _match_in(needle, carrier, output)
        if placement is not None:
            return placement
    if saw_eligible_carrier or not saw_forbidden_carrier:
        return _PENDING
    return _REJECTED


def derive_placements(
    items: Sequence[VerbatimLike],
    stage_outputs: Mapping[str, Mapping[str, Any] | None],
) -> dict[UUID, Placement]:
    """Map each item's id to its placement against ``stage_outputs``.

    ``stage_outputs`` is ``stage_name -> the persisted output dict`` (absent,
    ``None`` and ``{}`` all mean "that stage produced nothing"); only
    :data:`VERBATIM_CARRIER_PATHS` keys are consulted.

    The three resulting states are distinct and all load-bearing:

    * ``placed``   — the phrase is in the authoritative carrier's current output.
    * ``pending``  — a carrier the item was eligible for ran, and the phrase is
      not there; or nothing has run yet. "We looked and did not find it."
    * ``rejected`` — every carrier that ran was forbidden from carrying this
      item's ``kind``. "No stage was ever allowed to carry it." Today this is
      exactly the `slide` case: every stage in ``STAGE_VERBATIM_KIND`` consumes
      ``tts``, so slide verbatim has no consumer anywhere in the pipeline.

    Matching is the SHARED :func:`~src.services.stages.verbatim.normalize_for_match`
    and narrowing is the SHARED :func:`~src.services.stages.verbatim.items_for_stage`
    — byte-for-byte what ``assert_verbatim_present`` uses to FAIL a run. Any
    divergence on either axis would make the badge lie; see the anti-divergence
    tests in ``tests/services/test_verbatim_placement.py``.

    A phrase that normalises to empty (whitespace-only) is left ``pending``: the
    empty string is a substring of everything and would otherwise report placed
    against any output at all.
    """
    placements: dict[UUID, Placement] = {}
    for item in items:
        needle = normalize_for_match(str(item.text))
        placements[item.id] = (
            _placement_for(needle, getattr(item, "kind", None), stage_outputs)
            if needle
            else _PENDING
        )
    return placements
