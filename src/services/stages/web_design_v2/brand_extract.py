"""web_design_v2.brand_extract — seed the design with the brand's palette/fonts.

Best-effort: when ``brief.target_url`` is given, extract the site's brand via
analysis_service (``resolve_brand_context``); otherwise fall back to the project's
``brand`` block. Never fails the run — a missing brand just means the assembler
uses its default theme.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.services.stages.base import StageHandler, StageInput, StageOutput
from src.services.stages.brand_context import resolve_brand_context
from src.services.tenant_defaults import authored_brand, brief_brand, fill_tenant_brand

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext


class BrandExtractInput(StageInput):
    pass


class BrandExtractOutput(StageOutput):
    brand: dict | None = None
    site_analysis: dict | None = None
    source: str = "none"  # "url" | "project" | "none"


class BrandExtractStage(StageHandler[BrandExtractInput, BrandExtractOutput]):
    name = "brand_extract"
    input_model = BrandExtractInput
    output_model = BrandExtractOutput

    async def handle(self, payload: BrandExtractInput, ctx: RunContext) -> BrandExtractOutput:
        brief = ctx.config_snapshot.get("brief") or {}
        # Precedence: project/brief brand > the site's extracted brand > the tenant's
        # default brand (P6, gap-filler only — it is merged under ``snapshot.brand`` at
        # run creation, so read the AUTHORED part here and add the tenant's back last).
        project_brand = authored_brand(ctx.config_snapshot) or brief_brand(brief) or None
        ref_url = str(brief.get("target_url") or "").strip()

        brand: dict | None = project_brand
        site_analysis: dict | None = None
        source = "project" if project_brand else "none"

        if ref_url:
            brand_ctx, site_analysis = await resolve_brand_context(ref_url)
            if brand_ctx:
                # Project-provided brand wins on conflicts; URL fills the gaps.
                brand = {**brand_ctx, **(project_brand or {})}
                source = "url"

        filled = fill_tenant_brand(brand, ctx.config_snapshot)
        if brand is None and filled is not None:
            source = "tenant"
        brand = filled

        return BrandExtractOutput(brand=brand, site_analysis=site_analysis, source=source)
