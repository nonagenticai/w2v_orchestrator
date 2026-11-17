"""render_manifest — the terminal render/provenance AUDIT stage (A20).

A terminal stage that assembles the run's :class:`RenderManifest` — a flat
override ledger (one record per resolved knob), a completeness score, an
optional reproducibility spine (master seed + input hash), and the delivered
``artifact_url`` / ``duration_seconds`` read back from ``demo_video_compile``.
It is a pure audit artifact: it never mutates the spine, never blocks, and
produces a record of *how the delivered video was configured* rather than
changing what was rendered — so the delivered video is byte-identical whether
or not this stage runs.

The stage ALWAYS emits (so the "how this video was made" ledger lights up on
every run whose ``plan_document.provenance`` was recorded — see
``run_setup._record_provenance_ledger``). The ``reproducibility`` block no
longer gates emission; it only decides whether the reproducibility SPINE
(``master_seed`` + ``input_hash``) is populated: an explicit ``mode`` other
than ``"off"`` opts the spine in, absent / ``"off"`` leaves it ``None`` so the
manifest promises no reproducibility it can't keep. The manifest persists for
free as JSONB via ``StageOutputRepository`` (the stage output is stored whole),
which is what makes it an auditable record of the run rather than a log line.

Pure + LLM-free: the whole computation lives in
``stages/shared/render_manifest.assemble_manifest`` over plain mappings, so
this handler only reads ``mode`` to set ``include_spine`` and forwards
``config_snapshot`` plus the prior ``demo_video_compile`` output.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from src.services.stages.base import StageHandler, StageInput, StageOutput
from src.services.stages.shared.render_manifest import RenderManifest, assemble_manifest

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

# A `reproducibility` block with an explicit non-off mode opts the reproducibility
# spine (master_seed + input_hash) into the manifest. Absent block or `mode ==
# "off"` ⇒ spine stays `None` (the ledger still emits).
_OFF = "off"


class RenderManifestInput(StageInput):
    """No typed inputs — everything is read from ``ctx`` (config + prior output)."""


class RenderManifestOutput(StageOutput):
    """Carries the assembled :class:`RenderManifest`.

    Persists for free as JSONB via ``StageOutputRepository`` (the stage output
    is stored whole), which is what makes the manifest an auditable record of
    the run rather than a log line.
    """

    manifest: RenderManifest


class RenderManifestStage(StageHandler[RenderManifestInput, RenderManifestOutput]):
    name = "render_manifest"
    input_model = RenderManifestInput
    output_model = RenderManifestOutput

    # Where the delivered artifact is read from, in DAG order (first non-empty
    # wins). This is the ONE per-blueprint knob of the stage: a blueprint whose
    # compile stage is named differently subclasses and overrides this tuple
    # rather than forking the handler (``handler()`` takes a class, so a
    # subclass drops straight into a StageSpec). The default is demo_video's
    # chain and must stay in this exact order — see the comment in `handle`.
    artifact_stage_names: tuple[str, ...] = (
        "final_critique",
        "demo_video_encode_master",
        "demo_video_footage_critique",
        "demo_video_compile",
    )

    async def handle(self, payload: RenderManifestInput, ctx: RunContext) -> RenderManifestOutput:
        # ── ALWAYS emit the audit manifest (the override ledger + completeness).
        # The reproducibility block no longer gates emission — it only decides
        # whether the reproducibility SPINE (master_seed + input_hash) is populated:
        # an EXPLICIT non-off `mode` opts the spine in; absent / `"off"` leaves it
        # `None` so the manifest makes no reproducibility promise it can't keep. ──
        repro = ctx.config_snapshot.get("reproducibility")
        mode = repro.get("mode", _OFF) if isinstance(repro, Mapping) else _OFF
        include_spine = mode != _OFF

        # The delivered video's artifact + duration ride into the manifest from the
        # compile stage output (best-effort — absent for a partial run). Walk
        # `artifact_stage_names` in DAG order and take the first non-empty output:
        # for demo_video that is the A11 ship gate (`final_critique`, which itself
        # already read encode_master → footage_critique → compile) passing its
        # accepted-best through unchanged; then the S19 encode/mux master
        # (`demo_video_encode_master`); then the L4 footage-critique loop's
        # re-rendered artifact; then `demo_video_compile`. Each earlier stage's
        # output is falsy `{}` when it skipped-clean, so the chain falls through
        # byte-identically. Keyed under `demo_video_compile` REGARDLESS of which
        # stage it came from, because that is the slot `assemble_manifest` reads.
        priors: list[Mapping[str, object]] = [
            prior for name in self.artifact_stage_names if (prior := ctx.prior_output(name))
        ]
        compile_output: Mapping[str, object] = priors[0] if priors else {}

        # ── Per-field resolution for the MASTERING fields — do NOT "simplify" this
        # back into the first-mapping-wins walk above. The walk is correct for
        # `artifact_url` / `duration_seconds` (it deliberately prefers the
        # accepted-best artifact from the LATEST stage that produced one), but the
        # stage that wins it does not necessarily carry the mastering verdict:
        # `FinalCritiqueOutput` declares only `artifact_url` + `duration_seconds`,
        # so when the A11 ship gate ran it won the walk and `mastering` /
        # `output_sha256` silently resolved to `None` — measured in prod on run
        # 01a00b81-2576-71a5-becc-ffc4a843e6db, where `demo_video_compile` had both.
        # The manifest is the AUDIT record of what shipped; it must not drop the
        # mastering verdict or the content hash just because a later stage passed
        # the artifact through. So resolve these two keys independently: first
        # prior in the SAME DAG order that actually has a non-null value. ──
        # ⚠️ A hash may only describe the artifact it is PAIRED WITH. Adversarial review
        # broke the first cut of this: it took `output_sha256` from the first prior that
        # had one (only `demo_video_compile` emits it), while `artifact_url` came from a
        # later stage — so whenever `encode_master` / `footage_critique` actually
        # RE-RENDER (their entire purpose) the manifest attested the SUPERSEDED file's
        # hash under the new file's name. A wrong hash in an audit record is strictly
        # worse than a missing one: `None` is honest, a stale sha is a false attestation.
        # So a donor only counts if its `artifact_url` is the one being published.
        winning_url = compile_output.get("artifact_url")
        merged = dict(compile_output)
        for key in ("mastering", "output_sha256"):
            if merged.get(key) is None:
                for prior in priors:
                    if prior.get("artifact_url") != winning_url:
                        continue
                    value = prior.get(key)
                    if value is not None:
                        merged[key] = value
                        break
        stage_outputs: dict[str, Mapping[str, object]] = {"demo_video_compile": merged}
        manifest = assemble_manifest(
            ctx.config_snapshot, stage_outputs=stage_outputs, include_spine=include_spine
        )
        return RenderManifestOutput(manifest=manifest)
