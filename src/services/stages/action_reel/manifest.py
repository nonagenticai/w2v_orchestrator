"""render_manifest for action_reel — the shared ledger, pointed at `compile`.

The terminal provenance/override ledger itself is reused VERBATIM: the whole
computation lives in ``stages/shared/render_manifest.assemble_manifest`` and
this product needs none of it changed. The ONE thing that is product-specific is
WHERE the delivered artifact is read from — action_reel's render stage is named
``compile``, and there is no post-compile accept-best loop (it is also the
blueprint's ``terminal_artifact_stage``), so the chain is a single name.

That is expressed by overriding ``artifact_stage_names`` rather than forking the
handler. It lives here, next to the other action_reel stages, instead of in
``stages/demo_video/`` — a demo_video module is the wrong home for an
action_reel class, and the base stage is imported the same way any other
consumer imports it.
"""

from __future__ import annotations

from src.services.stages.demo_video.manifest import RenderManifestStage


class ActionReelRenderManifestStage(RenderManifestStage):
    # `compile` is action_reel's render AND its terminal_artifact_stage — no
    # accept-best loop sits below it, so there is nothing more downstream to
    # prefer. The output is still keyed under `demo_video_compile` inside the
    # base handler because that is the slot `assemble_manifest` reads; the key
    # is the manifest's internal contract, not a claim about the product.
    artifact_stage_names = ("compile",)
