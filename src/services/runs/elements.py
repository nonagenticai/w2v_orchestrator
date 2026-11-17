"""Element id resolution across stage outputs.

Each StageHandler that supports element-level pinning emits ids on its typed
output (e.g. ``script.lines[*].id``, ``storyboard.shots[*].id``). This module
walks those known shapes and returns the set of present ids so the
annotation-create route can reject a pin to a stale or non-existent element
with a 400 instead of silently accepting it.

Also hosts ``restamp_element_in_output`` (pure dict transform used by the
per-element restamp route) and ``downstream_stage_names`` (DAG helper).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.services.blueprints.base import DAGBlueprint

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.database.models import StageOutput

__all__ = [
    "compile_cut_stage",
    "downstream_stage_names",
    "element_ids_in",
    "element_in",
    "element_line_id",
    "music_rewind_stage",
    "patch_elements_in_output",
    "populated_element_paths",
    "reorder_elements_in_output",
    "resolve_clip_spine_stage",
    "restamp_element_in_output",
    "stage_supports_element_pinning",
]

# Stage name → list of dict-keys whose values are lists of {id: str, ...}.
_ELEMENT_PATHS: dict[str, list[str]] = {
    # Generic blueprint (PR-2)
    "script_draft": ["lines"],
    "storyboard_draft": ["shots"],
    # Promo blueprint (PR-3)
    "promo_analyze": ["pain_points", "hook_candidates"],
    "promo_script": ["lines"],
    "promo_storyboard": ["shots"],
    "promo_review": ["items"],
    # Demo-video / showcase blueprint — the gen stage's ordered ``lines`` are the
    # pipeline spine (each carries a ``clip-<n>`` join id); listing it here lets
    # the studio pin feedback to a clip and reorder the timeline.
    "demo_video_gen": ["lines"],
    # Lesson blueprint (PR-4)
    "lesson_script": ["paragraphs"],
    "lesson_slides": ["slides"],
    "lesson_narrate": ["clips"],
    "lesson_record": ["recordings"],
    # Course blueprint (PR-4)
    "course_outline": ["sections"],
    "course_compile_lessons": ["lessons"],
    # App-review blueprint (PR-5)
    "app_inspect": ["facts"],
    "app_critique": ["pros", "cons", "recommendations"],
    "app_review_script": ["lines"],
    "app_review_record": ["narrations", "visuals"],
    # Website-review blueprint — the script's ordered ``lines`` are the spine:
    # ``website_review_compile`` builds exactly one shot per line (``line_id`` =
    # ``line.id``). Without this entry the Studio's apply-edits / reorder 422'd
    # ``stage 'website_review_script' does not carry a clip timeline`` on every
    # website_review run (prod run 01a09f82-7179-73a5-a23a-40c15d148b37).
    "website_review_script": ["lines"],
}


def element_ids_in(output: dict[str, Any], stage_name: str) -> set[str]:
    """Return the set of element ids present at addressable paths for this stage.

    Stages not in ``_ELEMENT_PATHS`` support only output-level
    (``element_id IS NULL``) annotations; this returns an empty set.
    """
    out: set[str] = set()
    for p in _ELEMENT_PATHS.get(stage_name, []):
        out |= _ids_at(output, p)
    return out


def populated_element_paths(output: dict[str, object], stage_name: str) -> list[str]:
    """Return the stage's addressable paths that actually carry element ids here.

    Judged by the CONTENT of this row's output, never by
    ``len(_ELEMENT_PATHS[stage_name])`` — a stage declared with several paths
    (``promo_analyze``, ``app_critique``, ``app_review_record``) may well have
    populated only one of them on a given run, and that run DOES have a single
    orderable array.
    """
    return [p for p in _ELEMENT_PATHS.get(stage_name, []) if _ids_at(output, p)]


def _ids_at(output: dict[str, object], path_key: str) -> set[str]:
    nodes = output.get(path_key)
    if not isinstance(nodes, list):
        return set()
    return {
        node["id"] for node in nodes if isinstance(node, dict) and isinstance(node.get("id"), str)
    }


def stage_supports_element_pinning(stage_name: str) -> bool:
    return stage_name in _ELEMENT_PATHS


def element_in(output: dict[str, Any], stage_name: str, element_id: str) -> dict[str, Any] | None:
    """Return the addressed element dict at an addressable path, or ``None``.

    The read-only twin of ``restamp_element_in_output``: callers that must inspect
    an element's current fields (e.g. re-narrate checking for an uploaded
    ``audio_url`` voiceover on the spine clip) get the node itself rather than a
    single derived field.
    """
    for p in _ELEMENT_PATHS.get(stage_name, []):
        for node in output.get(p) or []:
            if isinstance(node, dict) and node.get("id") == element_id:
                return node
    return None


def element_line_id(output: dict[str, Any], stage_name: str, element_id: str) -> str | None:
    """Return the ``line_id`` of the addressed spine element, or ``None``.

    The studio pins feedback to a spine element's ``id`` (``shot.id`` / ``line.id``),
    but the per-clip image stage keys its output by the narration ``line_id``. This
    maps one to the other so a single-clip regenerate can target the right image.
    Returns ``None`` when the element is absent or carries no ``line_id``.
    """
    for p in _ELEMENT_PATHS.get(stage_name, []):
        for node in output.get(p) or []:
            if isinstance(node, dict) and node.get("id") == element_id:
                lid = node.get("line_id")
                return lid if isinstance(lid, str) and lid else None
    return None


def restamp_element_in_output(
    output: dict[str, Any], stage_name: str, element_id: str, value: dict[str, Any]
) -> dict[str, Any] | None:
    """Return a new output dict with the addressed element fully replaced.

    The element's ``id`` is preserved regardless of ``value`` content. Returns
    ``None`` if the element is not found at any addressable path for the
    stage — callers should translate that to a 404.
    """
    paths = _ELEMENT_PATHS.get(stage_name, [])
    new_output = dict(output)
    found = False
    for path_key in paths:
        nodes = new_output.get(path_key)
        if not isinstance(nodes, list):
            continue
        new_nodes: list[Any] = []
        for node in nodes:
            if isinstance(node, dict) and node.get("id") == element_id:
                replaced = {**value, "id": element_id}
                new_nodes.append(replaced)
                found = True
            else:
                new_nodes.append(node)
        new_output[path_key] = new_nodes
    return new_output if found else None


def patch_elements_in_output(
    output: dict[str, Any],
    stage_name: str,
    edits: dict[str, dict[str, Any]],
    *,
    allowed_fields: frozenset[str],
) -> dict[str, Any] | None:
    """Return a new output dict with per-element edit fields shallow-merged in.

    ``edits`` maps element id → a patch dict of field overrides (e.g.
    ``{"clip-1": {"clip_lead_seconds": 1.5, "clip_anchor": "start"}}``). Only keys
    in ``allowed_fields`` are applied; any other key in a patch is ignored (the
    route validates the shape, but this keeps the pure transform defensive). Each
    addressed element keeps its ``id``; unedited elements and every non-addressable
    key are untouched. A field set to ``None`` in a patch is DELETED from the
    element (so an edit can clear a previously-authored override back to default).

    Returns ``None`` when no edited id is present at the stage's addressable
    path(s) — i.e. every id in ``edits`` is stale/absent — so callers translate a
    fully-unmatched edit set to a 400, mirroring ``reorder_elements_in_output``.
    A partially-matched set applies the matched ids and still returns the new
    output (unmatched ids are a no-op, not an error).
    """
    paths = _ELEMENT_PATHS.get(stage_name, [])
    present = element_ids_in(output, stage_name)
    if not present or not any(eid in present for eid in edits):
        return None
    new_output = dict(output)
    for path_key in paths:
        nodes = new_output.get(path_key)
        if not isinstance(nodes, list):
            continue
        new_nodes: list[Any] = []
        for node in nodes:
            # ``.get("id")`` (not ``node["id"]``): a malformed/degraded spine row with
            # no id must be SKIPPED (patch=None → appended unchanged), not crash the
            # whole re-stitch with KeyError. Mirrors every sibling helper here.
            patch = edits.get(node.get("id")) if isinstance(node, dict) else None
            if patch is None:
                new_nodes.append(node)
                continue
            merged = dict(node)
            for field, value in patch.items():
                if field not in allowed_fields:
                    continue
                if value is None:
                    merged.pop(field, None)
                else:
                    merged[field] = value
            new_nodes.append(merged)
        new_output[path_key] = new_nodes
    return new_output


def reorder_elements_in_output(
    output: dict[str, Any], stage_name: str, ordered_ids: list[str]
) -> dict[str, Any] | None:
    """Return a new output dict with the addressed element array reordered.

    ``ordered_ids`` must be a permutation of exactly the ids present at the
    stage's addressable path(s) (the spine array — ``shots`` / ``lines``). The
    array is rebuilt in ``ordered_ids`` order; element bodies are untouched.
    Returns ``None`` if ``ordered_ids`` is not a permutation of the present ids
    (missing, extra, or duplicate) — callers translate that to a 400.

    Only the FIRST non-empty addressable path is treated as the orderable spine;
    a stage with several element arrays (e.g. ``promo_analyze``) is not a
    timeline and should not be reordered through this route.
    """
    paths = _ELEMENT_PATHS.get(stage_name, [])
    present = element_ids_in(output, stage_name)
    if not present:
        return None
    if sorted(ordered_ids) != sorted(present) or len(ordered_ids) != len(set(ordered_ids)):
        return None
    rank = {eid: i for i, eid in enumerate(ordered_ids)}
    new_output = dict(output)
    for path_key in paths:
        nodes = new_output.get(path_key)
        if not isinstance(nodes, list) or not nodes:
            continue
        # Only reorder the array that actually holds the addressed ids; leave
        # any incidental sibling array untouched.
        if not any(isinstance(n, dict) and n.get("id") in rank for n in nodes):
            continue
        new_output[path_key] = sorted(
            nodes,
            key=lambda n: rank.get(n.get("id"), len(rank)) if isinstance(n, dict) else len(rank),
        )
    return new_output


def compile_cut_stage(blueprint: DAGBlueprint, spine_stage: str) -> str | None:
    """Return the stage to re-run when the ``spine_stage`` clip order changes.

    Reordering the timeline reuses every generated clip asset and only needs the
    final video re-stitched, so the cut point is the blueprint's
    ``compile_cut_stage`` — the stage whose output is the compiled MP4 — which
    defaults to ``terminal_artifact_stage`` when not set apart (a blueprint whose
    deliverable is a post-compile stage, e.g. an accept-best footage critique,
    pins ``compile_cut_stage`` at the real compile so the re-stitch lands there).
    Preserving every stage strictly before the cut — with the spine reordered —
    lets the DAG executor re-run only the compile stage onward.

    Returns ``None`` when there is nothing to re-stitch: the blueprint declares
    no terminal artifact stage, or the spine is not strictly upstream of it (so
    a reorder could not feed a recompile).
    """
    cut = blueprint.compile_cut_stage or blueprint.terminal_artifact_stage
    if cut is None:
        return None
    order = blueprint.topological_order()
    if spine_stage not in order or cut not in order:
        return None
    if order.index(spine_stage) >= order.index(cut):
        return None
    return cut


def resolve_clip_spine_stage(
    blueprint: DAGBlueprint, stage_rows: Sequence[StageOutput]
) -> str | None:
    """Return the stage a client may legally pass to ``POST /runs/{id}/reorder-clips``.

    WHY this exists: the studio used to GUESS the clip spine client-side, by JSON
    shape — it picked the last ``stage_output`` in the response that carried a
    ``lines[]`` array. That guess is not decidable from the payload. On
    ``demo_video_v1`` FOUR stages (``demo_video_gen``, ``script_doctor``,
    ``retention_plan``, ``reconcile_timeline``) carry an identical ``lines[]`` of
    the same four clip ids, and the query that feeds the response has no
    ``ORDER BY``, so "the last one" is whatever order Postgres happened to return.
    Three of those four are not reorderable — they are not in ``_ELEMENT_PATHS``
    and/or have no compile cut downstream — so the reorder call 422'd at random.
    The backend owns this decision; the client asks and does not infer.

    The result is INDEPENDENT of the order of ``stage_rows``: the rows are only
    indexed by ``stage_name``, and the choice is made by walking the BLUEPRINT's
    topological order in reverse. Last qualifying stage wins, which preserves the
    old frontend's intent that a storyboard downstream of an earlier line-bearing
    stage is the spine.

    A stage qualifies only when all of these hold:

    * ``stage_supports_element_pinning`` — it declares addressable element paths;
    * a row exists for it that is ``completed`` with a non-empty ``output``;
    * that output actually contains element ids (an empty spine is not orderable);
    * that output carries EXACTLY ONE populated element array — see below;
    * ``compile_cut_stage`` resolves — which is exactly what guarantees the
      reorder endpoint will accept the stage rather than 422 it.

    WHY the single-array rule: a stage with several element arrays is not a
    timeline. ``reorder_elements_in_output``'s own contract says so — "Only the
    FIRST non-empty addressable path is treated as the orderable spine; a stage
    with several element arrays (e.g. ``promo_analyze``) is not a timeline and
    should not be reordered through this route." And ``element_ids_in`` returns
    the UNION of ids across every path, which ``reorder_elements_in_output``
    then demands ``ordered_ids`` be a permutation of — so offering a two-array
    stage would require the client to interleave, say, narration ids and visual
    ids into one flat list, which orders nothing meaningful. Concretely
    ``app_review_v1`` used to resolve to ``app_review_record``
    (``["narrations", "visuals"]``); it now falls through to
    ``app_review_script`` (a single ``lines`` array), which is both genuinely
    reorderable and what the studio renders.

    "Exactly one" is judged by the arrays PRESENT AND POPULATED in this row's
    output (``populated_element_paths``), not by how many paths the stage
    DECLARES. A stage declared with two or three paths that populated only one
    on this run is a valid single-array spine and still qualifies.

    Returns ``None`` when the run has no reorderable timeline (nothing completed
    yet, or the blueprint simply has no spine/compile cut).
    """
    by_name = {row.stage_name: row for row in stage_rows}
    for stage in reversed(blueprint.topological_order()):
        if not stage_supports_element_pinning(stage):
            continue
        row = by_name.get(stage)
        if row is None or row.status != "completed" or not row.output:
            continue
        if len(populated_element_paths(row.output, stage)) != 1:
            continue
        if compile_cut_stage(blueprint, stage) is None:
            continue
        return stage
    return None


#: The stage that OWNS the beat grid a music-cut pipeline times its cut against.
#: Only ``action_reel`` declares it (``services/blueprints/action_reel.py``);
#: ``music_rewind_stage`` returns ``None`` for every other blueprint, which is what
#: keeps the music-edit rewind inert outside this product.
_GRID_OWNER_STAGE = "music_select"


def music_rewind_stage(blueprint: DAGBlueprint) -> str | None:
    """Return the stage a MUSIC change must re-run from, or ``None``.

    For most blueprints a music change is a re-stitch: the bed is laid over a
    timeline that does not depend on it, so re-running the compile cut is enough
    and this returns ``None`` (the caller keeps the ordinary ``sound_effects`` +
    edit-directive path).

    ``action_reel`` is the exception, and it is not a stylistic one. Its cut is
    TIMED to the bed: ``cut_plan`` snaps every shot boundary to the selected
    track's beat grid and stamps each shot ``beat_locked``. Swapping the bed
    without re-deriving the grid renders a reel cut to a different track's beats
    — the product's core claim, silently broken, with every stage green. So the
    child run must rewind to the stage that FETCHES the grid
    (``music_select``); its descendant closure (``cut_plan`` → ``graphics_design``
    / ``narrate`` → ``compile``) re-runs, while the expensive footage branch
    (``source_ingest`` → ``footage_probe`` → ``shot_analysis`` →
    ``segment_select``) is preserved and never re-billed.

    Returns ``None`` unless the blueprint actually has that stage AND it is
    strictly upstream of the compile cut — a blueprint that grew a same-named
    stage downstream of its cut would not be re-derivable this way.
    """
    cut = blueprint.compile_cut_stage or blueprint.terminal_artifact_stage
    if cut is None:
        return None
    order = blueprint.topological_order()
    if _GRID_OWNER_STAGE not in order or cut not in order:
        return None
    if order.index(_GRID_OWNER_STAGE) > order.index(cut):
        return None
    return _GRID_OWNER_STAGE


def downstream_stage_names(blueprint: DAGBlueprint, stage_name: str) -> list[str]:
    """Return stages strictly after ``stage_name`` in the blueprint topological order.

    Stages absent from the topology return ``[]``.
    """
    order = blueprint.topological_order()
    try:
        idx = order.index(stage_name)
    except ValueError:
        return []
    return order[idx + 1 :]
