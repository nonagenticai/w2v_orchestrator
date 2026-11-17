"""Storyboard AI authoring (Phase 1) — LLM-only draft + panel rewrite.

Mirrors the canonical LLM stage (``services.stages.course.outline``): route every
call through Worker Management (``WMClient``) with a per-call ``system_prompt`` +
object-rooted structured ``output_schema``, wrap the shape-validating ``_attempt``
in ``stage_call`` (zero-arg factory, ``timeout_seconds`` + ``retries``), and keep
DB persistence OUTSIDE the retry budget so a re-prompt never double-writes rows.

No GPU, no image/video generation — this stage produces the authored board
(scenes / panels / assets) as text only. Enum-constrained panel fields are
coerced against the shipped ``showcase_line`` StrEnums before insert so an
out-of-vocabulary LLM value drops to ``NULL`` rather than tripping the DB CHECK.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import StrEnum
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from w2v_common.stages import stage_call

from src.api.schemas.storyboard import StoryboardDraftRequest
from src.database.models import StoryboardAsset, StoryboardPanel, StoryboardScene
from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.storyboard_repository import (
    StoryboardAssetRepository,
    StoryboardPanelRepository,
    StoryboardRepository,
    StoryboardSceneRepository,
)
from src.models.showcase_line import (
    AssetKind,
    CameraAngle,
    CameraMovement,
    ShotSize,
    TransitionOut,
    normalize_clip_key,
)
from src.services.llm import WMClient
from src.services.stages.llm_budget import (
    LARGE_ANSWER_TOKENS,
    LLM_RETRY_ON,
    SMALL_ANSWER_TOKENS,
    downshift_reasoning,
    llm_budget_seconds,
    llm_client_timeout_seconds,
    llm_retries,
    output_token_budget,
)
from src.services.stages.upstream import guarded
from src.utils import coerce_float


class DraftConflictError(Exception):
    """Board already has panels and ``replace`` was not set (route → 409)."""


class DraftInputError(Exception):
    """No usable source text to draft from (route → 422)."""


# ── WM worker names (regex ^[a-zA-Z0-9_-]+$; provider/model live on WM config) ──
_DRAFT_WORKER = "storyboard-draft"
_REWRITE_WORKER = "storyboard-rewrite"

_DRAFT_SYSTEM_PROMPT = (
    "You are a professional storyboard artist and script supervisor. Break the "
    "supplied script or idea into scenes and, within each scene, an ordered "
    "sequence of shot panels using standard film vocabulary. Also list the "
    "recurring characters, locations, and props as reusable assets, and reference "
    "them by name from the panels. Return only JSON matching the schema."
)

_REWRITE_SYSTEM_PROMPT = (
    "You are a storyboard editor. Rewrite only the requested fields of one panel "
    "to satisfy the instruction, keeping the rest of the panel intact. Return only "
    "JSON with the requested fields matching the schema; omit a field to leave it "
    "unchanged."
)

# Object-rooted structured-output schema (WM convention). ``resp.output`` is
# returned already parsed to this shape.
_DRAFT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "assets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": [e.value for e in AssetKind]},
                    "name": {"type": "string"},
                    "canonical_descriptor": {"type": "string"},
                },
                "required": ["name"],
            },
        },
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "int_ext": {"type": "string"},
                    "location": {"type": "string"},
                    "time_of_day": {"type": "string"},
                    "description": {"type": "string"},
                    "panels": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action_description": {"type": "string"},
                                "dialogue": {"type": "string"},
                                "vo_text": {"type": "string"},
                                "shot_number": {"type": "string"},
                                "shot_size": {
                                    "type": "string",
                                    "enum": [e.value for e in ShotSize],
                                },
                                "angle": {
                                    "type": "string",
                                    "enum": [e.value for e in CameraAngle],
                                },
                                "movement": {
                                    "type": "string",
                                    "enum": [e.value for e in CameraMovement],
                                },
                                "transition_out": {
                                    "type": "string",
                                    "enum": [e.value for e in TransitionOut],
                                },
                                "duration_seconds": {"type": "number"},
                                "characters": {"type": "array", "items": {"type": "string"}},
                                "props": {"type": "array", "items": {"type": "string"}},
                                "location": {"type": "string"},
                            },
                            "required": ["action_description"],
                        },
                    },
                },
                "required": ["panels"],
            },
        },
    },
    "required": ["scenes"],
}

# Per-field property schemas for the rewrite whitelist; a per-call output schema
# is subset from this by the effective field set.
_REWRITE_SCHEMA: dict[str, dict[str, Any]] = {
    "action_description": {"type": "string"},
    "dialogue": {"type": "string"},
    "vo_text": {"type": "string"},
    "composition_notes": {"type": "string"},
    "lighting_mood": {"type": "string"},
    "shot_size": {"type": "string", "enum": [e.value for e in ShotSize]},
    "angle": {"type": "string", "enum": [e.value for e in CameraAngle]},
    "movement": {"type": "string", "enum": [e.value for e in CameraMovement]},
    "transition_out": {"type": "string", "enum": [e.value for e in TransitionOut]},
    "duration_seconds": {"type": "number"},
}
# Reasoning-model output budgets (see stages/llm_budget.py — a FLOOR, not a cap).
#
# DRAFT: ``_DRAFT_SCHEMA`` is a WHOLE board — ``scenes[]``, each with ``panels[]``,
# each panel carrying an action_description, dialogue, vo_text plus a dozen fields.
# The scene/panel count grows with the author's prompt ⇒ LARGE.
_DRAFT_MAX_TOKENS = output_token_budget(answer_tokens=LARGE_ANSWER_TOKENS)
# REWRITE: ONE panel, and only the whitelisted fields asked for — at most the ten
# keys of ``_REWRITE_SCHEMA``, mostly enums and a number, with a few short prose
# fields. A single small fixed-shape object ⇒ SMALL.
_REWRITE_MAX_TOKENS = output_token_budget(answer_tokens=SMALL_ANSWER_TOKENS)

_REWRITE_ENUM_FIELDS: dict[str, type[StrEnum]] = {
    "shot_size": ShotSize,
    "angle": CameraAngle,
    "movement": CameraMovement,
    "transition_out": TransitionOut,
}
_REWRITE_NUM_FIELDS = frozenset({"duration_seconds"})


def _coerce(value: object, enum_cls: type[StrEnum]) -> str | None:
    """Enum-validate; drop to ``None`` on an out-of-vocabulary value (draft path)."""
    if value is None:
        return None
    try:
        return enum_cls(value).value
    except ValueError:
        return None


def _coerce_kind(value: object) -> str:
    """AssetKind cannot be NULL — default an invalid/absent kind to ``prop``."""
    try:
        return AssetKind(value).value
    except ValueError, TypeError:
        return AssetKind.PROP.value


# Brief keys that carry the creative source, in prompt order. ``source_script``
# first: a pasted script is the strongest signal; the rest are the wizard's
# per-door narrative fields (generic/demo/explainer briefs).
_BRIEF_TEXT_KEYS = (
    "source_script",
    "idea",
    "outcome",
    "angle",
    "value_proposition",
    "summary",
    "description",
    "audience",
    "call_to_action",
)


def _brief_text(brief: Mapping[str, object], *, title: str | None = None) -> str:
    """Pull a reasonable source string out of a project ``brief`` dict.

    ``title`` (the project's title) is prepended as context when the brief
    carries any text; on its own it is not draftable source. The scalar
    fallback keeps only free-text values — numeric knobs such as
    ``target_duration_seconds: 60`` are configuration, not source material.
    """
    if not isinstance(brief, dict):
        return ""
    parts = [str(brief[k]).strip() for k in _BRIEF_TEXT_KEYS if brief.get(k)]
    text = " ".join(p for p in parts if p).strip()
    if not text:
        scalars = [
            f"{k}: {v.strip()}"
            for k, v in brief.items()
            if isinstance(v, str) and v.strip() and not str(k).endswith("_seconds")
        ]
        text = "; ".join(scalars)
    if text and title and title.strip():
        return f"Title: {title.strip()}\n{text}"
    return text


class _AssetResolver:
    """Dedup-by-name asset resolver over one project (case-insensitive).

    Holds the ``name → asset-id`` map so the draft's character/prop/location
    references and the LLM's declared ``assets[]`` collapse onto one row per name,
    and referenced-but-undeclared names are created on the fly.
    """

    def __init__(
        self, repo: StoryboardAssetRepository, project_id: UUID, existing: list[StoryboardAsset]
    ) -> None:
        self._repo = repo
        self._project_id = project_id
        self._by_name: dict[str, str] = {a.name.lower(): str(a.id) for a in existing}
        self.created: list[StoryboardAsset] = []

    async def resolve(self, name: str, kind: str) -> str | None:
        """Return the id of the asset called ``name`` (creating it if new)."""
        clean = name.strip()
        if not clean:
            return None
        existing = self._by_name.get(clean.lower())
        if existing is not None:
            return existing
        return await self._create(clean, kind, None, {})

    async def declare(
        self, name: str, kind: object, descriptor: object, attributes: object
    ) -> None:
        """Persist an LLM-declared asset unless a same-name row already exists."""
        clean = name.strip()
        if not clean or clean.lower() in self._by_name:
            return
        await self._create(clean, kind, descriptor, attributes or {})

    async def _create(self, name: str, kind: object, descriptor: object, attributes: object) -> str:
        asset = await self._repo.create(
            project_id=self._project_id,
            position=None,
            kind=_coerce_kind(kind),
            name=name,
            canonical_descriptor=descriptor,
            attributes=attributes or {},
        )
        self._by_name[name.lower()] = str(asset.id)
        self.created.append(asset)
        return str(asset.id)


class StoryboardAssistService:
    """LLM draft + rewrite over a tenant-scoped storyboard."""

    def __init__(self, session: AsyncSession, *, client: WMClient | None = None) -> None:
        self._session = session
        self._timeout = llm_budget_seconds()
        self._retries = llm_retries()
        self._wm = client or WMClient(timeout=llm_client_timeout_seconds())

    async def draft(
        self, *, project_id: UUID, tenant_id: UUID, req: StoryboardDraftRequest
    ) -> tuple[list[StoryboardScene], list[StoryboardPanel], list[StoryboardAsset]]:
        sb = await StoryboardRepository(self._session).ensure_for_project(
            project_id, tenant_id=tenant_id
        )
        if sb is None:  # route pre-checks the project, but guard anyway.
            raise DraftInputError("project not found")

        project = await ProjectRepository(self._session).get(project_id, tenant_id=tenant_id)
        brief = project.brief if project is not None else {}
        title = project.title if project is not None else None
        text = (req.script_or_idea or _brief_text(brief, title=title)).strip()
        if not text:
            raise DraftInputError("no script or idea to draft from")

        panel_repo = StoryboardPanelRepository(self._session)
        existing = await panel_repo.list_for_project(project_id, tenant_id=tenant_id)
        if existing and not req.replace:
            raise DraftConflictError("storyboard already has panels; pass replace=true")
        if req.replace:
            # Assets are intentionally preserved across a re-draft (the library).
            await panel_repo.delete_for_project(project_id, tenant_id=tenant_id)
            await StoryboardSceneRepository(self._session).delete_for_project(
                project_id, tenant_id=tenant_id
            )

        prompt = self._draft_prompt(text, req)
        board, model = await self._draw_board(prompt, project_id=project_id)
        return await self._persist_board(
            board,
            storyboard_id=sb.id,
            project_id=project_id,
            tenant_id=tenant_id,
            prompt=prompt,
            model=model,
        )

    async def _draw_board(self, prompt: str, *, project_id: UUID) -> tuple[dict[str, Any], object]:
        """Single WM draw of the board, shape-validated inside the retry budget.

        ``project_id`` is the storyboard's w2v project (orchestrator
        ``projects.id``), sent as WM's ``X-Project-Id`` so the LLM spend is
        attributed to it — never an fr project from an inbound header.
        """

        async def _attempt(enable_reasoning: bool | None) -> dict[str, Any]:
            resp = await self._wm.message(
                enable_reasoning=enable_reasoning,
                worker_name=_DRAFT_WORKER,
                message=prompt,
                system_prompt=_DRAFT_SYSTEM_PROMPT,
                output_schema=_DRAFT_SCHEMA,
                max_tokens=_DRAFT_MAX_TOKENS,
                project_id=project_id,
            )
            data = resp.output
            if not isinstance(data, dict) or not isinstance(data.get("scenes"), list):
                raise ValueError("storyboard_draft: LLM output missing a 'scenes' list")
            model = (resp.metadata or {}).get("model") if resp.metadata else None
            return {"data": data, "model": model}

        result = await stage_call(
            # Synchronous request path (POST /storyboard/draft): a 402 here must reach
            # the caller as "Configure your LLM provider", not as "retries exhausted"
            # three LLM budgets later. The ValueError re-draw above stays retryable.
            downshift_reasoning(lambda think: guarded(lambda: _attempt(think))),
            timeout_seconds=self._timeout,
            retries=self._retries,
            retry_on=LLM_RETRY_ON,
            stage_name="storyboard_draft",
        )
        return result["data"], result["model"]

    async def _persist_board(
        self,
        board: dict[str, Any],
        *,
        storyboard_id: UUID,
        project_id: UUID,
        tenant_id: UUID,
        prompt: str,
        model: object,
    ) -> tuple[list[StoryboardScene], list[StoryboardPanel], list[StoryboardAsset]]:
        """Persist the drawn board in the caller's txn (OUTSIDE the retry budget)."""
        asset_repo = StoryboardAssetRepository(self._session)
        scene_repo = StoryboardSceneRepository(self._session)
        panel_repo = StoryboardPanelRepository(self._session)
        existing_assets = await asset_repo.list_for_project(project_id, tenant_id=tenant_id)
        resolver = _AssetResolver(asset_repo, project_id, existing_assets)

        # Assets first — dedup by case-insensitive name against this call + project.
        for item in board.get("assets") or []:
            if isinstance(item, dict):
                await resolver.declare(
                    str(item.get("name") or ""),
                    item.get("kind"),
                    item.get("canonical_descriptor"),
                    item.get("attributes"),
                )

        gen = {
            "worker": _DRAFT_WORKER,
            "prompt": prompt,
            "model": model,
            "schema": "storyboard-draft-v1",
        }
        created_scenes: list[StoryboardScene] = []
        created_panels: list[StoryboardPanel] = []
        counter = 1
        # Scenes in order, then panels with one global clip-<n> counter.
        for i, scene_data in enumerate(s for s in board["scenes"] if isinstance(s, dict)):
            scene = await scene_repo.create(
                project_id=project_id,
                storyboard_id=storyboard_id,
                position=i,
                name=scene_data.get("name"),
                int_ext=scene_data.get("int_ext"),
                location=scene_data.get("location"),
                time_of_day=scene_data.get("time_of_day"),
                description=scene_data.get("description"),
            )
            created_scenes.append(scene)
            for panel_data in scene_data.get("panels") or []:
                if not isinstance(panel_data, dict):
                    continue
                panel = await self._persist_panel(
                    panel_repo,
                    panel_data,
                    storyboard_id=storyboard_id,
                    project_id=project_id,
                    scene_id=scene.id,
                    counter=counter,
                    resolver=resolver,
                    gen=gen,
                )
                created_panels.append(panel)
                counter += 1

        # Return the full current asset library so the FE gets every referenced row.
        all_assets = await asset_repo.list_for_project(project_id, tenant_id=tenant_id)
        return created_scenes, created_panels, all_assets

    @staticmethod
    async def _resolve_names(resolver: _AssetResolver, names: Any, kind: str) -> list[str]:
        out: list[str] = []
        for name in names or []:
            rid = await resolver.resolve(str(name), kind)
            if rid is not None:
                out.append(rid)
        return out

    async def _persist_panel(
        self,
        repo: StoryboardPanelRepository,
        panel_data: dict[str, Any],
        *,
        storyboard_id: UUID,
        project_id: UUID,
        scene_id: UUID,
        counter: int,
        resolver: _AssetResolver,
        gen: dict[str, Any],
    ) -> StoryboardPanel:
        """Resolve a panel's asset references + coerce enums, then insert one row."""
        characters = await self._resolve_names(
            resolver, panel_data.get("characters"), AssetKind.CHARACTER.value
        )
        props = await self._resolve_names(resolver, panel_data.get("props"), AssetKind.PROP.value)
        location_asset_id: UUID | None = None
        loc_name = panel_data.get("location")
        if loc_name:
            lid = await resolver.resolve(str(loc_name), AssetKind.LOCATION.value)
            location_asset_id = UUID(lid) if lid is not None else None
        return await repo.create(
            project_id=project_id,
            storyboard_id=storyboard_id,
            position=counter - 1,
            shot_key=normalize_clip_key(f"clip-{counter}"),
            scene_id=scene_id,
            action_description=panel_data.get("action_description"),
            dialogue=panel_data.get("dialogue"),
            vo_text=panel_data.get("vo_text"),
            shot_number=panel_data.get("shot_number"),
            shot_size=_coerce(panel_data.get("shot_size"), ShotSize),
            angle=_coerce(panel_data.get("angle"), CameraAngle),
            movement=_coerce(panel_data.get("movement"), CameraMovement),
            transition_out=_coerce(panel_data.get("transition_out"), TransitionOut),
            duration_seconds=coerce_float(panel_data.get("duration_seconds")),
            characters=characters,
            props=props,
            location_asset_id=location_asset_id,
            source="ai",
            gen=gen,
        )

    async def rewrite(
        self,
        panel_id: UUID,
        *,
        tenant_id: UUID,
        instruction: str,
        fields: list[str] | None,
    ) -> StoryboardPanel | None:
        whitelist = list(_REWRITE_SCHEMA.keys())
        if fields is None:
            effective = whitelist
        else:
            requested = set(fields)
            effective = [f for f in whitelist if f in requested]

        panel_repo = StoryboardPanelRepository(self._session)
        panel = await panel_repo.get(panel_id, tenant_id=tenant_id)
        if panel is None:
            return None
        if not effective:  # nothing rewritable was asked for.
            return panel

        current = {f: getattr(panel, f) for f in effective}
        prompt = (
            "Rewrite the requested fields of this storyboard panel.\n"
            f"Instruction: {instruction}\n"
            f"Current values (JSON): {json.dumps(current, default=str)}\n"
            "Return ONLY JSON with the requested fields matching the schema. "
            "Omit a field to leave it unchanged."
        )
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {f: _REWRITE_SCHEMA[f] for f in effective},
            "required": [],
        }

        async def _attempt(enable_reasoning: bool | None) -> dict[str, Any]:
            resp = await self._wm.message(
                enable_reasoning=enable_reasoning,
                worker_name=_REWRITE_WORKER,
                message=prompt,
                system_prompt=_REWRITE_SYSTEM_PROMPT,
                output_schema=schema,
                max_tokens=_REWRITE_MAX_TOKENS,
                # The panel's own w2v project, for the same attribution as draft.
                project_id=panel.project_id,
            )
            out = resp.output
            if not isinstance(out, dict):
                raise ValueError("storyboard_rewrite: LLM output is not an object")
            return out

        out = await stage_call(
            downshift_reasoning(lambda think: guarded(lambda: _attempt(think))),
            timeout_seconds=self._timeout,
            retries=self._retries,
            retry_on=LLM_RETRY_ON,
            stage_name="storyboard_rewrite",
        )

        changed = self._coerce_rewrite_fields(effective, out)
        changed["revision"] = panel.revision + 1
        changed["source"] = "ai"
        changed["gen"] = {
            **(panel.gen or {}),
            "rewrite": {
                "worker": _REWRITE_WORKER,
                "prompt": prompt,
                "instruction": instruction,
            },
        }
        return await panel_repo.update(panel_id, tenant_id=tenant_id, fields=changed)

    @staticmethod
    def _coerce_rewrite_fields(effective: list[str], out: dict[str, Any]) -> dict[str, Any]:
        """Coerce the LLM's returned fields; drop an invalid enum/number so the
        panel keeps its current value rather than tripping the DB CHECK."""
        changed: dict[str, Any] = {}
        for field in effective:
            if field not in out:
                continue
            value = out[field]
            if field in _REWRITE_ENUM_FIELDS:
                coerced = _coerce(value, _REWRITE_ENUM_FIELDS[field])
                if coerced is not None:
                    changed[field] = coerced
            elif field in _REWRITE_NUM_FIELDS:
                num = coerce_float(value)
                if num is not None:
                    changed[field] = num
            else:
                changed[field] = str(value) if value is not None else None
        return changed

    @staticmethod
    def _draft_prompt(text: str, req: StoryboardDraftRequest) -> str:
        parts = [f"Source material:\n{text}"]
        if req.instructions:
            parts.append(f"Additional instructions: {req.instructions}")
        if req.target_scene_count:
            parts.append(f"Aim for about {req.target_scene_count} scenes.")
        if req.target_panel_count:
            parts.append(f"Aim for about {req.target_panel_count} panels in total.")
        parts.append(
            "Return JSON with an 'assets' array (each item a character/location/prop "
            "with a 'name' and 'canonical_descriptor') and a 'scenes' array. Each "
            "scene has name, int_ext, location, time_of_day, description and a "
            "'panels' array. Each panel has action_description, dialogue, vo_text, "
            "shot_number, shot_size, angle, movement, transition_out, duration_seconds, "
            "characters (asset names), props (asset names) and location (asset name). "
            "Reference every character/location/prop by its asset name."
        )
        return "\n\n".join(parts)
