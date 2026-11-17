"""Build the text→image prompt (and the pinned seed) for one storyboard panel.

The image runner behind ACG's ``local_image`` provider is **text→image + seed
only**: it takes a prompt, a size and a seed, and nothing else. There is no
img2img, no ControlNet and no IP-Adapter on that path, so an asset's stored
``reference_images`` CANNOT condition the generation — they are art direction
for a human, not an input to the model.

That leaves exactly two levers for keeping a character/location/prop looking
like itself across a board, and this module owns both:

1. **The canonical descriptor.** Every panel that references an asset folds the
   SAME ``canonical_descriptor`` words into its prompt, in the same place, in a
   deterministic order. Identical wording is the only thing making panel 7's
   "Mara" the same Mara as panel 2's.
2. **The pinned seed.** The referenced assets' ``seed`` is reused verbatim, so
   the sampler starts from the same noise every time. A panel with no seeded
   asset falls back to a seed derived from its own id — still stable, so a
   regenerate of an unchanged panel reproduces the image rather than rolling a
   new one.

Everything here is pure: no IO, no session. The prompt string is the cache key
material for ``deterministic_media_id``, so any change to the wording is a
deliberate cache invalidation.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from src.database.models import StoryboardAsset, StoryboardPanel

# The image runner bounds ``seed`` to **int32**, not uint32:
# ``image_gen_backend/src/models.py`` declares ``seed: int | None = Field(None,
# ge=0, le=2**31 - 1)``, so anything above 2147483647 is rejected by FastAPI
# with a 422 before FLUX is ever reached.
#
# This constant used to be 2**32 with a comment claiming "mflux/FLUX takes a
# uint32 seed". That was false, and because ``_fallback_seed`` takes four bytes
# of a sha256, it put HALF of all panels above the limit — a coin flip per
# panel, live-verified against the cluster (panel 019fe5a3…e1ce derived
# 4100493024 and 422'd).
_SEED_MODULUS = 2**31

# Seed preference order across the roles a panel can reference. The character is
# the subject the eye tracks between panels, so its seed wins; the location is
# the next-strongest continuity anchor; a prop is the weakest.
_SEED_ROLE_ORDER = ("character", "location", "prop")


@dataclass(frozen=True, slots=True)
class PanelPrompt:
    """The complete generative input for one panel — the thing that is cached.

    ``prompt`` + ``seed`` + the job config fully determine the output image, so
    these are exactly the fields folded into ``deterministic_media_id`` and
    written back into the panel's ``gen`` provenance blob.
    """

    prompt: str
    seed: int
    style_token: str | None
    asset_ids: tuple[str, ...]


def _clean(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _referenced_assets(
    panel: StoryboardPanel, assets: Mapping[UUID, StoryboardAsset]
) -> list[tuple[str, StoryboardAsset]]:
    """``(role, asset)`` pairs in a STABLE order: location, characters, props.

    Order is part of the prompt text, so it has to come from the panel's own
    author order (``panel.characters`` / ``panel.props`` are already sorted by
    join-row position) and never from dict/set iteration — an unstable order
    would change the prompt string, and with it the content-addressed media id,
    on every call.

    Ids the caller could not resolve (an asset deleted between the panel read
    and the asset read) are skipped rather than raising: a missing descriptor
    weakens consistency, it does not make the panel ungeneratable.
    """
    out: list[tuple[str, StoryboardAsset]] = []
    location_id = panel.location_asset_id
    if location_id is not None:
        # Split from the `is not None` guard rather than chained with `and`: the
        # walrus would then be conditionally bound and `ty` flags the use below.
        location = assets.get(location_id)
        if location is not None:
            out.append(("location", location))
    for raw in panel.characters:
        if (asset := assets.get(UUID(raw))) is not None:
            out.append(("character", asset))
    for raw in panel.props:
        if (asset := assets.get(UUID(raw))) is not None:
            out.append(("prop", asset))
    return out


def _fallback_seed(panel_id: UUID) -> int:
    """A stable per-panel seed for a panel that references no seeded asset.

    Derived from the panel id, NOT random: a regenerate of an unchanged panel
    must reproduce the same image (and therefore hit the media cache instead of
    paying for a second generation).
    """
    digest = sha256(str(panel_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % _SEED_MODULUS


def resolve_seed(
    panel: StoryboardPanel,
    referenced: Sequence[tuple[str, StoryboardAsset]],
    *,
    override: int | None = None,
) -> int:
    """The seed to pin, honouring ``override`` → asset seed → panel fallback."""
    if override is not None:
        return override % _SEED_MODULUS
    for role in _SEED_ROLE_ORDER:
        for asset_role, asset in referenced:
            if asset_role == role and asset.seed is not None:
                return asset.seed % _SEED_MODULUS
    return _fallback_seed(panel.id)


def _asset_lines(referenced: Sequence[tuple[str, StoryboardAsset]]) -> list[str]:
    """One ``Role: Name — canonical descriptor`` line per referenced asset.

    Identical wording in every panel that references the asset is the whole
    consistency mechanism on a text→image model, so this stays byte-stable.
    """
    lines: list[str] = []
    for role, asset in referenced:
        descriptor = _clean(asset.canonical_descriptor)
        name = _clean(asset.name)
        label = role.capitalize()
        lines.append(f"{label}: {name} — {descriptor}" if descriptor else f"{label}: {name}")
    return lines


def _style_tokens(referenced: Sequence[tuple[str, StoryboardAsset]]) -> list[str]:
    """Distinct ``style_token``s in first-seen order (a shared style anchor)."""
    tokens: list[str] = []
    for _role, asset in referenced:
        token = _clean(asset.style_token)
        if token and token not in tokens:
            tokens.append(token)
    return tokens


def build_panel_prompt(
    panel: StoryboardPanel,
    assets: Mapping[UUID, StoryboardAsset],
    *,
    seed_override: int | None = None,
) -> PanelPrompt:
    """Compose the panel's prompt from its own direction plus its assets' descriptors.

    The panel contributes the shot (``shot_size`` / ``angle``), the action, the
    composition note, the lighting and the art style. Each referenced asset
    contributes its NAME and its ``canonical_descriptor`` — the same words in
    every panel that references it, which is what makes the character read as
    the same character. A ``style_token`` on any referenced asset is appended
    once (deduplicated, first-seen order) as a shared style anchor.
    """
    referenced = _referenced_assets(panel, assets)
    lines: list[str] = []

    if art_style := _clean(panel.art_style):
        lines.append(f"Art style: {art_style}.")

    shot_bits = [b for b in (_clean(panel.shot_size), _clean(panel.angle)) if b]
    if shot_bits:
        lines.append(f"Shot: {' '.join(shot_bits)}.")

    if action := _clean(panel.action_description):
        lines.append(action if action.endswith((".", "!", "?")) else f"{action}.")

    lines.extend(_asset_lines(referenced))

    if composition := _clean(panel.composition_notes):
        lines.append(f"Composition: {composition}")

    if lighting := _clean(panel.lighting_mood):
        lines.append(f"Lighting: {lighting}")

    style_tokens = _style_tokens(referenced)
    if style_tokens:
        lines.append(f"Style: {', '.join(style_tokens)}")

    if not lines:
        # A blank panel would otherwise submit an empty prompt, which ACG 422s.
        # Naming the shot keeps the request valid and the failure legible.
        lines.append(f"Storyboard panel {_clean(panel.shot_key) or panel.id}.")

    return PanelPrompt(
        prompt="\n".join(lines),
        seed=resolve_seed(panel, referenced, override=seed_override),
        style_token=style_tokens[0] if style_tokens else None,
        asset_ids=tuple(str(asset.id) for _role, asset in referenced),
    )
