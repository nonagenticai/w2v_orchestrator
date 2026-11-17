"""promo_storyboard — one shot per script line, brand/theme-conditioned.

Output exposes ``shots[*].id`` so PR-6 can pin annotations to shots, and
``shots[*].line_id`` so the FE can correlate shots back to script lines.

Each shot also carries a ``visual_type`` — the per-line *visual plan* that
``promo_render`` realises by routing each line to one source: real captured
``footage``, a designed ``slide`` (slide_service), a redesigned ``html``
segment (reused redesign mechanics), or an ``ai_image`` accent. The plan is a
deterministic projection over the script (no LLM): it opens on a title slide,
closes on a CTA slide, and leads the interior with real footage sprinkled with
html / ai-image / slide accents. When the brief carries no ``source_url`` (no
footage to capture) ``footage`` slots collapse to ``ai_image`` so the plan never
points at clips that won't exist. When the only footage is ONE plain page
capture (bare ``source_url``), footage is capped at one line in four and the
rest go to the requested ``motion_graphics`` / ``ai_image`` producers (else
slides), so a promo is never one long homepage scroll. promo_render always keeps the per-line slide
as a universal fallback, so a missing asset never blanks a shot.

This stage is a deterministic projection over the prior ``promo_script``
output (no downstream client call), so it does not need ``stage_call``
wrapping; the constructor still accepts a WMClient for forward-compatibility
with a planned LLM-backed storyboarding mode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar
from uuid import uuid7

from src.services.llm import WMClient
from src.services.stages.base import (
    SlotUse,
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
)
from src.services.stages.shared.shotlist import normalize_shotlist

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

# Interior visual rhythm (between the opening title slide and closing CTA
# slide): heavily footage-dominant so a captured product demo IS the video —
# real UI on screen carries the narration, with only a sparse on-brand slide and
# the occasional ai_image accent for punctuation. (Earlier a ~50%-footage pattern
# split the timeline across landing/demo/book evenly, so the static landing page
# recurred ~5x and the demo never dominated.) ~80% footage; promo_render then
# routes nearly all of that footage to the demo clip (see render._plan_visuals).
_INTERIOR_PATTERN = (
    "footage",
    "footage",
    "footage",
    "footage",
    "slide",
    "footage",
    "footage",
    "footage",
    "footage",
    "ai_image",
)


# A bare ``source_url`` with no shotlist, no per-screen ``site_capture.screens``
# and at most one ``site_capture.paths`` entry records exactly ONE clip (the
# homepage scroll). Spreading the ~80%-footage interior above over that single
# clip turned whole promos into one long website scroll (merqi, 2026-10-05). In
# that case footage is capped at one line in ``_SINGLE_CAPTURE_FOOTAGE_EVERY``
# and never lands on two consecutive lines; the demoted lines take the richer
# producers the run actually requested.
_SINGLE_CAPTURE_FOOTAGE_EVERY = 4


def is_single_capture(brief: dict) -> bool:
    """True when the brief's only footage source is one plain page capture.

    ``site_capture.screens`` (frame-synced demo), more than one
    ``site_capture.paths`` entry, and a path with scripted ``interactions`` (a
    click-through demo walkthrough) all yield real product footage, so none
    counts. An authored ``brief.shotlist`` never reaches the pattern planner.
    """
    cfg = brief.get("site_capture") or {}
    if not isinstance(cfg, dict):
        return True
    if cfg.get("screens"):
        return False
    paths = cfg.get("paths")
    if not isinstance(paths, list):
        return True
    if len(paths) > 1:
        return False
    return not any(isinstance(p, dict) and p.get("interactions") for p in paths)


def requested_accent_types(config_snapshot: dict) -> tuple[str, ...]:
    """The visual types a demoted footage line may take, from the opt-in blocks
    the run actually carries: ``motion_graphics`` (with a template or inline
    html) and ``ai_image_gen``. Neither requested -> designed ``slide`` only."""
    accents: list[str] = []
    motion = config_snapshot.get("motion_graphics")
    if isinstance(motion, dict) and (motion.get("template_id") or motion.get("html")):
        accents.append("motion_graphics")
    if config_snapshot.get("ai_image_gen"):
        accents.append("ai_image")
    return tuple(accents) or ("slide",)


def _cap_single_capture_footage(types: list[str], accents: tuple[str, ...]) -> list[str]:
    """Keep at most ``len(types) // _SINGLE_CAPTURE_FOOTAGE_EVERY`` footage lines,
    spread evenly over the interior with no two adjacent; every other footage
    line cycles through ``accents``."""
    n = len(types)
    interior = list(range(1, n - 1))
    budget = n // _SINGLE_CAPTURE_FOOTAGE_EVERY
    keep: set[int] = set()
    if budget and interior:
        step = len(interior) / budget
        for k in range(budget):
            idx = interior[min(len(interior) - 1, int((k + 0.5) * step))]
            if idx - 1 not in keep and idx + 1 not in keep:
                keep.add(idx)
    out = list(types)
    j = 0
    for i, t in enumerate(types):
        if t == "footage" and i not in keep:
            out[i] = accents[j % len(accents)]
            j += 1
        elif t != "footage" and i in keep:
            out[i] = "footage"
    return out


def plan_visual_types(
    n: int,
    *,
    has_footage: bool,
    demo_sync: bool = False,
    single_capture: bool = False,
    accents: tuple[str, ...] = ("slide",),
) -> list[str]:
    """Assign each of ``n`` script lines a visual_type.

    Opens on a title ``slide`` and closes on a CTA ``slide``; the interior
    cycles :data:`_INTERIOR_PATTERN`. With ``has_footage`` False (no capture
    source), ``footage`` slots become ``ai_image`` so the plan never points at
    clips that won't be produced.

    ``demo_sync`` (frame-synced per-screen demo, ``brief.site_capture.screens``
    present): the interior is ALL ``footage`` so every interior line gets a keyed
    demo screenshot synced to its narration — only the opening title and closing
    CTA stay as designed slides. This is the "demo screens only" Phase-1 plan.

    ``single_capture`` (the only footage is one page capture, see
    :func:`is_single_capture`): footage is capped at one line in
    :data:`_SINGLE_CAPTURE_FOOTAGE_EVERY`, never on consecutive lines, and the
    demoted lines cycle through ``accents`` (see :func:`requested_accent_types`).
    """
    if n <= 0:
        return []
    if n == 1:
        return ["slide"]
    types = ["slide"]
    for i in range(1, n - 1):
        types.append(
            "footage" if demo_sync else _INTERIOR_PATTERN[(i - 1) % len(_INTERIOR_PATTERN)]
        )
    types.append("slide")
    if not has_footage:
        return ["ai_image" if t == "footage" else t for t in types]
    if single_capture and not demo_sync:
        return _cap_single_capture_footage(types, accents)
    return types


class PromoStoryboardInput(StageInput):
    pass


class PromoShot(StageOutput):
    id: str
    line_id: str
    description: str
    seconds: float
    visual_style: str
    # Per-line visual plan: footage | slide | html | ai_image | motion_graphics.
    # Realised by promo_render (see module docstring). Defaulted for forward-compat with
    # any persisted pre-visual-plan storyboard output.
    visual_type: str = "slide"


class PromoStoryboardOutput(StageOutputWithNotes):
    # The handler builds one shot per script line unconditionally, so empty is
    # never intended. `promo_render` subscripts `story["shots"]`; empty means
    # `assemble_shots` returns [] and the failure surfaces one stage late as
    # PromoRenderOutput's required-artifact violation, pointing at the wrong stage.
    produces: ClassVar[tuple[str, ...]] = ("shots",)

    shots: list[PromoShot]


class PromoStoryboardStage(StageHandler[PromoStoryboardInput, PromoStoryboardOutput]):
    name = "promo_storyboard"
    input_model = PromoStoryboardInput
    output_model = PromoStoryboardOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("llm"),)

    def __init__(self, client: WMClient | None = None) -> None:
        self._wm = client or WMClient()

    async def handle(
        self,
        payload: PromoStoryboardInput,
        ctx: RunContext,
    ) -> PromoStoryboardOutput:
        script = ctx.prior_output("promo_script")
        if not script:
            raise RuntimeError("promo_script output missing")
        brand = ctx.config_snapshot.get("brand") or {}
        theme = ctx.config_snapshot.get("theme") or {}
        brief = ctx.config_snapshot.get("brief") or {}
        has_footage = bool((brief.get("source_url") or "").strip())
        demo_sync = bool((brief.get("site_capture") or {}).get("screens"))
        style = (
            f"Brand font {brand.get('font', 'sans-serif')} on "
            f"{theme.get('background', '#000')} with accent "
            f"{theme.get('accent', '#fff')}"
        )
        lines = script["lines"]
        # Authored shotlist: honor each shot's explicit kind (url→footage,
        # slide→slide) instead of the heuristic pattern. This is what lets an
        # operator drop a slide ANYWHERE between URL shots — plan_visual_types
        # would otherwise force every interior demo line to footage.
        shots = normalize_shotlist(brief)
        if shots:
            kind_by_line = {s.line_id: s.kind for s in shots}
            visual_types = [
                "footage" if kind_by_line.get(str(line["id"])) == "url" else "slide"
                for line in lines
            ]
        else:
            visual_types = plan_visual_types(
                len(lines),
                has_footage=has_footage,
                demo_sync=demo_sync,
                single_capture=is_single_capture(brief),
                accents=requested_accent_types(ctx.config_snapshot),
            )
        shots = [
            PromoShot(
                id=str(uuid7()),
                line_id=str(line["id"]),
                description=f"Visual for: {line['text']}",
                seconds=float(line["seconds"]),
                visual_style=style,
                visual_type=visual_types[i],
            )
            for i, line in enumerate(lines)
        ]
        return PromoStoryboardOutput(shots=shots)
