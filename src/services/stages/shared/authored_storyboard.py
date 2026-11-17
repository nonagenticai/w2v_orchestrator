"""Normalize a persisted/inline storyboard (``brief["storyboard"]``) into the
single-source ``AuthoredLine`` spine the authored short-circuits already consume.

Phase 0 (no GPU/AI): a structured, authored storyboard flows through the EXISTING
authored-lines path — panels become ``AuthoredLine``s keyed by the ``clip-<n>``
spine id (== ``NarrationPart.part_id``), exactly as a hand-authored
``brief["lines"]`` would. Visual/camera fields (image_ref, shot_size, angle, …)
are intentionally NOT consumed here; they are carried by the persistence layer
for later (GPU) phases.

Precedence (see ``run_setup.build_config_snapshot`` seed): a persisted storyboard
is folded into ``brief["storyboard"]`` and WINS over any inline ``brief["storyboard"]``.
Within a consumer, a present storyboard WINS over legacy ``brief["lines"]``.

No I/O, no LLM, no config. Both blueprint short-circuits (generic ``script_draft``,
``demo_video`` ``gen``) import from here so panel→line normalization has a single
home and cannot drift from ``AuthoredLine``.
"""

from __future__ import annotations

from typing import Any

from src.models.showcase_line import AuthoredLine, AuthoredSplit, normalize_clip_key


def _panel_text(raw: dict[str, Any]) -> str:
    """Spoken narration for a panel: ``vo_text``, else ``dialogue``, else the
    ``action_description``. The first non-empty wins (stripped)."""
    for key in ("vo_text", "dialogue", "action_description"):
        value = raw.get(key)
        if value:
            return str(value).strip()
    return ""


def _timeline_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """The A22/C6 per-line timeline slots (``time_lock`` / ``min_seconds`` /
    ``max_seconds`` / ``vo_rate``) plus the author's ``locked`` text pin, coerced and
    carried through verbatim.

    Absent keys map to ``None`` (never a default) — ``reconcile_timeline`` treats
    ``None`` as "compute it" and a present value as AUTHORED, which always wins. Both
    the panel path and the legacy ``lines`` path share this so a board and a
    hand-authored spine cannot drift.
    """
    return {
        "time_lock": bool(raw["time_lock"]) if raw.get("time_lock") is not None else None,
        "min_seconds": float(raw["min_seconds"]) if raw.get("min_seconds") is not None else None,
        "max_seconds": float(raw["max_seconds"]) if raw.get("max_seconds") is not None else None,
        "vo_rate": float(raw["vo_rate"]) if raw.get("vo_rate") is not None else None,
        # The author's verbatim pin. Dropping it here (it was, until the merqi showcase
        # spike 01a0fc55) silently UNLOCKS every authored line: script_doctor /
        # claim_finish / retention_plan / reconcile_timeline all honour
        # ``line["locked"]``, but this field-by-field rebuild is the only door an
        # authored line passes through, so none of them ever saw it.
        "locked": bool(raw.get("locked")),
    }


def _panel_to_line(raw: dict[str, Any], idx: int) -> AuthoredLine:
    """Map one storyboard panel dict onto the shared ``AuthoredLine`` field set.

    ``shot_key`` (the ``clip-<n>`` spine join key) normalizes to ``id`` via
    ``normalize_clip_key`` — the SAME id rule ``demo_video.line_ids.authored_line_id``
    uses — falling back to ``clip-<idx+1>`` when a panel omits it, so the two paths
    derive the same spine. ``id``/``clip`` are accepted as aliases for a hand-inlined
    board that predates the persisted ``shot_key`` column.

    The A22 timeline slots are carried through here too: a panel that pins a beat
    (``time_lock``) must reconcile the same way a ``brief["lines"]`` entry does — a
    board with no such keys simply yields ``None`` for all four, so a Phase-0 board
    is unaffected.
    """
    base = (
        str(raw.get("shot_key") or raw.get("id") or raw.get("clip") or "").strip()
        or f"clip-{idx + 1}"
    )
    return AuthoredLine(
        id=normalize_clip_key(base),
        text=_panel_text(raw),
        seconds=float(raw.get("duration_seconds") or raw.get("seconds") or 0.0),
        scenario_name=str(raw.get("shot_number") or raw.get("scenario_name") or ""),
        speaker=(str(raw["speaker"]).strip() if raw.get("speaker") else None),
        expected_text=(str(raw["expected_text"]).strip() if raw.get("expected_text") else None),
        # Author-pinned two-pane layout. Invalid (bad layout / empty second_clip) raises
        # at the API door's ShowcaseLine; a hand-inlined legacy value is validated here.
        split=(AuthoredSplit.model_validate(raw["split"]) if raw.get("split") else None),
        **_timeline_fields(raw),
    )


def storyboard_panels(brief: dict[str, Any]) -> list[AuthoredLine] | None:
    """Normalize ``brief["storyboard"]`` → ``AuthoredLine``s, or ``None`` if absent.

    ``None`` (not ``[]``) signals "no storyboard authored" so a consumer falls
    through to its legacy path untouched. An explicitly empty list also yields
    ``None`` (an empty board is not an authored spine). Used by ``generic`` — which
    does NOT otherwise honor ``brief["lines"]`` — so generic gains ONLY the new
    storyboard trigger and existing generic projects are unaffected.
    """
    panels = brief.get("storyboard")
    if not panels:
        return None
    return [_panel_to_line(raw, idx) for idx, raw in enumerate(panels)]


def _lines_entry_to_line(raw: dict[str, Any], idx: int) -> AuthoredLine:
    """Legacy ``brief["lines"]`` entry → ``AuthoredLine`` — byte-parity with the old
    inline ``demo_video.gen._authored_output`` mapping.

    Reproduces the same id rule (``id``/``line_id`` → ``clip-<idx+1>`` →
    ``normalize_clip_key``) and the same field-for-field coercions. The ``speaker``
    slot is carried RAW here (stripped only); ``demo_video`` re-applies its own
    ``_normalize_speaker`` alias map (stage policy) so this helper stays free of
    stage-specific behavior.

    Plus the A22/C6 timeline slots (``_timeline_fields``) — this rebuild is
    FIELD-BY-FIELD, so anything not listed here is silently dropped from the
    authored brief.
    """
    base = str(raw.get("id") or raw.get("line_id") or "").strip() or f"clip-{idx + 1}"
    speaker = raw.get("speaker") or raw.get("role")
    return AuthoredLine(
        id=normalize_clip_key(base),
        text=str(raw.get("text") or "").strip(),
        seconds=float(raw.get("seconds") or 0.0),
        scenario_name=str(raw.get("scenario_name") or ""),
        gherkin=str(raw.get("gherkin") or ""),
        speaker=(str(speaker).strip() if speaker else None),
        clip=(str(raw["clip"]).strip() if raw.get("clip") else None),
        clip_anchor=(str(raw["clip_anchor"]).strip() if raw.get("clip_anchor") else None),
        clip_lead_seconds=(
            float(raw["clip_lead_seconds"]) if raw.get("clip_lead_seconds") else None
        ),
        clip_out_seconds=(float(raw["clip_out_seconds"]) if raw.get("clip_out_seconds") else None),
        expected_text=(str(raw["expected_text"]).strip() if raw.get("expected_text") else None),
        # Author-pinned two-pane layout; an invalid one raises here (and at the API door).
        split=(AuthoredSplit.model_validate(raw["split"]) if raw.get("split") else None),
        **_timeline_fields(raw),
    )


def authored_lines(brief: dict[str, Any]) -> list[AuthoredLine] | None:
    """Storyboard-wins-over-``lines`` authored spine, or ``None`` if neither present.

    Used by consumers (``demo_video``) that ALREADY honor legacy ``brief["lines"]``:
    a present storyboard supersedes ``lines``; else legacy ``lines`` are normalized
    through the same ``AuthoredLine`` shape; else ``None`` → generated (bdd) path.

    Legacy invariant: with no storyboard, this returns ``AuthoredLine``s built from
    ``brief["lines"]`` field-for-field, so the caller produces byte-identical output
    to reading ``brief["lines"]`` directly.
    """
    panels = storyboard_panels(brief)
    if panels is not None:
        return panels
    lines = brief.get("lines")
    if not lines:
        return None
    return [_lines_entry_to_line(raw, idx) for idx, raw in enumerate(lines)]
