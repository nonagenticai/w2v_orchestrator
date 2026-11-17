"""Assemble compositor shots from a storyboard + generated images + narration.

The video-compile ``Shot`` is the join point of three upstream stages that do
not share a key uniformly:

* storyboard shots carry ``line_id`` (1 per script line) — the spine.
* narration clips carry ``line_id`` — joined 1:1 to the storyboard shot.
* ai-generated images carry **no** ``line_id`` (they are produced per config
  prompt, not per line), so they are distributed round-robin across shots.
* keyed images (``image_by_line``) DO carry a ``line_id`` and join 1:1 — this
  is how lesson's paragraph→slide→image two-hop and app_review's per-line
  screenshot express through the one helper. A keyed image wins over a
  round-robin AI image.
* site-capture video clips also carry no ``line_id`` (one per recorded path),
  so they too cycle round-robin. The compositor ``Shot`` honours
  ``video_url > image_url > description``, so when a real-site clip is present
  it becomes the shot's background and the AI image is the fallback.

A shot keeps its ``description`` (when the storyboard supplied one) so the
compositor can fall back to a text card when no visual is available;
``seconds`` is reconciled to the narration clip's duration when present (the
compositor stretches the visual to match). A shot with no ``description`` key
(lesson/app_review build from narration/visuals, not a storyboard) omits it
rather than sending ``description: null`` on the wire.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

from src.services.stages.shared.accessibility import (
    choose_backplate,
    text_region_geometry,
)
from src.services.stages.shared.degradation import (
    SHOT_DURATION_FABRICATED,
    SHOT_DURATION_FLOORED,
    UNDERSPOKEN_BEAT_HELD,
    VISUAL_ASSETS_REUSED,
    Degradation,
    degradation,
)

if TYPE_CHECKING:
    from src.services.stages.shared.accessibility import A11yPolicy

# A "shot" dict — either a SPINE shot (the storyboard row plus whatever Studio
# per-clip edits were merged onto it) or a BUILT compositor shot (what this
# module emits on the wire to video-compile). Both are open JSON mappings: the
# key set is presence-is-opt-in by design, so an absent edit omits its key
# rather than sending a null, and it stays a dict rather than a model here.
type ShotDict = dict[str, Any]

# An upstream stage's JSON payload joined onto the spine by ``line_id`` (or
# round-robin): a narration clip, a generated image, a site-capture video clip,
# an sfx cue. Each producer owns its own key set — this module only reads the
# few keys it joins on.
type StagePayload = dict[str, Any]


class _VisualPool:
    """One un-keyed round-robin pool (images OR video clips) with its OWN draw
    counter — the reference shape is ``website_review/compile.py``'s separate
    per-pool counters.

    Indexing the pool by the GLOBAL beat index (the pre-fix
    ``pool[i % len(pool)]``) let keyed beats consume indices: with 5 beats,
    beats 1-3 keyed and a pool of 4, beats 0 and 4 both drew ``pool[0]`` while
    ``pool[1..3]`` never played. Three rules fix that:

    * the counter advances only on an UN-KEYED draw;
    * assets a keyed beat already shows are excluded from the drawable pool
      (falling back to the FULL pool only when that would leave it empty — a
      keyed line may legitimately reuse a pool asset, and a single-asset pool
      must keep supplying it, see :func:`_resolve_visual`);
    * among the drawable assets the LEAST-USED wins (keyed uses seeded in), ties
      broken in round-robin order starting at the counter. With no keyed beats
      this is exactly plain ``pool[k % len(pool)]`` — byte-identical output.
    """

    def __init__(self, urls: list[str], keyed_urls: list[str]) -> None:
        keyed = set(keyed_urls)
        fresh = [u for u in urls if u not in keyed]
        self.urls: list[str] = fresh or list(urls)
        self._uses: dict[str, int] = dict.fromkeys(self.urls, 0)
        for u in keyed_urls:
            if u in self._uses:
                self._uses[u] += 1
        self._draws = 0

    def __bool__(self) -> bool:
        return bool(self.urls)

    def __len__(self) -> int:
        return len(self.urls)

    def draw(self) -> str:
        """The next candidate; advances the per-pool counter. Usage is recorded
        separately via :meth:`record` — a candidate the anti-repeat guard
        refuses was never shown, so it must not count as used."""
        n = len(self.urls)
        start = self._draws % n
        self._draws += 1
        order = [self.urls[(start + k) % n] for k in range(n)]
        least = min(self._uses[u] for u in order)
        return next(u for u in order if self._uses[u] == least)

    def record(self, url: str) -> None:
        self._uses[url] += 1


def _resolve_visual(
    line_id: str,
    *,
    keyed_videos: dict[str, str],
    video_pool: _VisualPool,
    keyed_images: dict[str, str],
    image_pool: _VisualPool,
    prev_video_url: str | None = None,
    prev_image_url: str | None = None,
) -> dict[str, str]:
    """Pick a shot's visual keys: keyed beats round-robin; video beats image.

    Returns the ``video_url`` / ``image_url`` keys to merge (either, both, or
    neither). The compositor honours video_url > image_url > description, so a
    shot may legitimately carry both (video wins) — promo's plan avoids that by
    keying exactly one per line.

    ``prev_video_url`` / ``prev_image_url`` are the round-robin pick that rode
    the PREVIOUS shot's same channel (``None`` once a gap has broken the
    streak — see ``assemble_shots``). The anti-repeat guard below is gated on
    the pool having a genuine ALTERNATIVE (``len(pool) >= 2``): when one
    exists and the pool's next draw (see :class:`_VisualPool`) would replay
    the visual that just played, the key is omitted here instead, so the shot falls through
    to its next-lower-precedence visual (image, then the text-card
    ``description``) rather than reading as a frozen frame.

    A pool of exactly ONE usable asset is a DIFFERENT case, deliberately NOT
    covered by that guard: there is no alternative, so refusing the repeat
    would fall through on every beat after the first and turn a 10-beat video
    into "one image, then nine text cards" — a wall of text cards reads as a
    broken video, which is worse than a held/repeated visual. So a
    single-asset pool keeps assigning that one asset to every beat, exactly
    as before this fix; the underlying "not enough assets" fact is surfaced
    instead via the shortfall soft note (see ``_visual_reuse_shortfall_notes``),
    not by starving the shot of a visual. Keyed visuals are untouched — an
    authored per-line plan may legitimately repeat an asset on purpose.
    """
    out: dict[str, str] = {}
    keyed_video = keyed_videos.get(line_id)
    if keyed_video:
        out["video_url"] = keyed_video
    elif video_pool:
        candidate = video_pool.draw()
        if len(video_pool) < 2 or candidate != prev_video_url:
            video_pool.record(candidate)
            out["video_url"] = candidate
    keyed_image = keyed_images.get(line_id)
    if keyed_image:
        out["image_url"] = keyed_image
    elif image_pool:
        candidate = image_pool.draw()
        if len(image_pool) < 2 or candidate != prev_image_url:
            image_pool.record(candidate)
            out["image_url"] = candidate
    return out


def _next_round_robin_state(
    line_id: str,
    visual: dict[str, str],
    *,
    keyed_videos: dict[str, str],
    keyed_images: dict[str, str],
) -> tuple[str | None, str | None]:
    """The ``(prev_video_url, prev_image_url)`` to carry into the NEXT shot's
    :func:`_resolve_visual` call. Only the un-keyed (round-robin) path feeds
    the anti-repeat streak — an authored per-line plan may legitimately
    repeat an asset, so a keyed hit resets that channel's streak to ``None``.
    """
    next_video = None if keyed_videos.get(line_id) else visual.get("video_url")
    next_image = None if keyed_images.get(line_id) else visual.get("image_url")
    return next_video, next_image


def normalize_footage_viewport(raw: object) -> list[int] | None:
    """Normalize a recorded-source viewport into the ``[w, h]`` int pair the
    compiler's viewport-aware coordinate transform (WS-A ``Shot.footage_viewport``)
    consumes — the RECORDED source WxH the focus/cursor geometry was measured in.

    Accepts the VP-2 sidecar / FTR clip shapes ``{"width","height"}``, ``{"w","h"}``
    or a ``[w, h]`` pair; anything else (missing, non-numeric, non-positive, or a
    bool masquerading as a dimension) ⇒ ``None`` ⇒ no viewport rides the shot ⇒
    byte-identical. NEVER raises: a malformed viewport must not fail a compile."""
    w: object = None
    h: object = None
    if isinstance(raw, dict):
        d = cast("dict[str, object]", raw)
        raw_w = d.get("width")
        raw_h = d.get("height")
        w = raw_w if raw_w is not None else d.get("w")
        h = raw_h if raw_h is not None else d.get("h")
    elif isinstance(raw, (list, tuple)) and len(raw) == 2:
        w, h = raw[0], raw[1]
    # bool is an int subclass — a flag is not a dimension.
    if isinstance(w, bool) or isinstance(h, bool):
        return None
    if not isinstance(w, (int, float)) or not isinstance(h, (int, float)):
        return None
    if w <= 0 or h <= 0:
        return None
    return [int(w), int(h)]


_BBOX_KEYS = ("x", "y", "w", "h")


def _finite(raw: object) -> float | None:
    """A real, finite number or ``None``. ``bool`` is an ``int`` subclass — a flag
    is not a coordinate — and ``inf``/``nan`` are not renderable geometry."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    return value if value == value and value not in (float("inf"), float("-inf")) else None


def normalize_bbox(raw: object) -> tuple[float, float, float, float] | None:
    """Normalize a recorder bbox into the compiler's ``(x, y, w, h)`` 4-tuple.

    The click/focus geometry recorder emits a bbox as a DICT (``{"x","y","w","h"}``),
    but the compiler declares ``FocusKey.bbox`` / ``Callout.bbox`` as
    ``tuple[float, float, float, float]`` — a 4-number ARRAY on the wire (see
    ``tests/contracts/compile_request.schema.json``: ``prefixItems`` x4,
    ``minItems``/``maxItems`` 4). Pydantic does not coerce a mapping into a tuple, so
    passing the recorder shape through verbatim 422s the compile.

    An already-normalized 4-element list/tuple passes through. Anything else — a
    missing key, a non-numeric or non-finite value, a wrong-length sequence — yields
    ``None`` so the CALLER can drop that keyframe rather than emit a malformed one.
    NEVER raises: bad geometry must degrade, not fail a compile."""
    if isinstance(raw, dict):
        d = cast("dict[str, object]", raw)
        values = [_finite(d.get(key)) for key in _BBOX_KEYS]
    elif isinstance(raw, (list, tuple)) and len(raw) == 4:
        values = [_finite(item) for item in raw]
    else:
        return None
    x, y, w, h = values
    if x is None or y is None or w is None or h is None:
        return None
    return (x, y, w, h)


#: Recorder ``action``s that mean "the step INTERACTED with this element" — FTR's
#: track recorder emits ``click`` (mousedown target) and ``type`` (focused editable);
#: VP-2's capture sidecar emits ``click``. Anything else — notably FTR's ``reveal``,
#: the value it fills in when an entry carried NO action — is not an interaction.
_INTERACTION_ACTIONS = frozenset({"click", "type"})

#: Below this many pixels on either side a bbox is not an element the viewer can see —
#: it is a zero-filled placeholder (FTR writes ``{0,0,0,0}`` for a missing bbox) or a
#: collapsed/hidden input. The compiler floors such a bbox to 1px and punches in at its
#: MAX zoom, i.e. straight into the empty top-left corner of the frame.
_MIN_TARGET_PX = 4.0


def interacted_bbox(
    entry: object, viewport: list[int] | None = None
) -> tuple[float, float, float, float] | None:
    """The bbox of the element this keyframe's step INTERACTED with, or ``None``.

    The ONLY coordinates a punch-in may target (merqi showcase spike 01a0fc55: the
    camera zoomed into blank areas). ``None`` — and so no zoom for this keyframe — when
    the entry is not an interaction (``action`` present but not click/type), when its
    bbox will not convert, when it is degenerate (either side under
    :data:`_MIN_TARGET_PX`), or when it lies wholly outside the recorded ``viewport``
    (an element scrolled off-screen: zooming to it lands on whatever is NOT there).
    An entry with no ``action`` key at all is a pre-normalized/legacy keyframe and is
    judged on its geometry alone. NEVER raises."""
    if not isinstance(entry, dict):
        return None
    item = cast("dict[str, object]", entry)
    action = item.get("action")
    if action is not None and action not in _INTERACTION_ACTIONS:
        return None
    bbox = normalize_bbox(item.get("bbox"))
    if bbox is None:
        return None
    x, y, w, h = bbox
    if w < _MIN_TARGET_PX or h < _MIN_TARGET_PX:
        return None
    if viewport:
        vw, vh = viewport
        if x >= vw or y >= vh or x + w <= 0 or y + h <= 0:
            return None
    return bbox


def normalize_focus_track(
    raw: object,
    *,
    viewport: list[int] | None = None,
    window_start: float = 0.0,
    window_end: float | None = None,
) -> list[dict[str, object]] | None:
    """Normalize a recorder focus track into the compiler's ``FocusKey`` list.

    Each keyframe is reduced to exactly the two fields the compiler declares —
    ``t`` (shot-local seconds, ``>= 0``) and ``bbox`` as a 4-tuple. The recorder's
    extra ``action`` key is DROPPED: the compiler's request model is
    ``extra="ignore"``, so an undeclared field is silently discarded live anyway,
    and the repo's leaf-set contract guard reads it as an invented field.

    Every keyframe is a punch-in target, so only an :func:`interacted_bbox` survives:
    a keyframe whose step did not interact with a real, visible element is dropped,
    never zoomed at a placeholder. ``window_start`` / ``window_end`` are the SOURCE
    seconds the shot actually plays (``clip_lead_seconds`` / ``clip_out_seconds``):
    the recorder's ``t`` is measured from the head of the RECORDING, while the
    compiler reads it shot-locally, so a keyframe outside the window points at a
    screen that is not on screen and is dropped, and the rest are rebased.

    A keyframe with a missing/malformed ``t`` is dropped too — one bad keyframe must
    not 422 the whole compile. An empty or all-dropped track yields ``None``, NOT
    ``[]``: an absent ``focus_track`` is the compiler's byte-identical no-op (no zoom),
    and ``[]`` would be a new (if equivalent) wire shape where ``None`` is the
    established default."""
    if not isinstance(raw, (list, tuple)):
        return None
    keys: list[dict[str, object]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        item = cast("dict[str, object]", entry)
        t = _finite(item.get("t"))
        bbox = interacted_bbox(item, viewport)
        if t is None or t < 0 or bbox is None:
            continue
        if t < window_start or (window_end is not None and t > window_end):
            continue
        keys.append({"t": t - window_start, "bbox": bbox})
    return keys or None


def normalize_cursor_track(
    raw: object,
    *,
    window_start: float = 0.0,
    window_end: float | None = None,
) -> list[dict[str, object]] | None:
    """Normalize a recorder cursor track into the compiler's ``CursorKey`` list.

    The recorder's ``{t, x, y, click}`` shape already MATCHES ``CursorKey``, so a
    well-formed keyframe keeps its ``x``/``y``/``click``; this exists so the same
    drop-the-bad-keyframe rule applies on both tracks (a non-numeric ``x`` 422s
    exactly like a dict bbox does). ``click`` is only carried when it is a real
    ``bool`` — the compiler defaults it to ``False``. Empty ⇒ ``None`` (byte-identical
    no-op), never ``[]``.

    ``window_start`` / ``window_end`` are the SOURCE seconds the shot plays
    (``clip_lead_seconds`` / ``clip_out_seconds``), exactly as for
    :func:`normalize_focus_track`: the recorder's ``t`` is measured from the head of
    the RECORDING, the compiler reads it from the shot's in-point. A keyframe outside
    the window is dropped and the rest are rebased. Without this, a trimmed shot
    drew its cursor and click rings at the raw recording times, i.e. at the wrong
    moment and over a screen that was no longer showing."""
    if not isinstance(raw, (list, tuple)):
        return None
    keys: list[dict[str, object]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        item = cast("dict[str, object]", entry)
        t = _finite(item.get("t"))
        x = _finite(item.get("x"))
        y = _finite(item.get("y"))
        if t is None or t < 0 or x is None or y is None:
            continue
        if t < window_start or (window_end is not None and t > window_end):
            continue
        key: dict[str, object] = {"t": t - window_start, "x": x, "y": y}
        click = item.get("click")
        if isinstance(click, bool):
            key["click"] = click
        keys.append(key)
    return keys or None


def normalize_callouts(raw: object) -> list[dict[str, object]] | None:
    """Normalize the directed-shot-plan callout seam into the compiler's
    ``list[Callout]``.

    TWO shape mismatches are corrected here, both latent until the geometry
    recorder actually delivers a focus track:

    * ``Callout.bbox`` carries the SAME ``(x, y, w, h)`` 4-tuple contract as
      ``FocusKey.bbox`` and is copied verbatim off the captured focus track's first
      keyframe upstream (``shared/footage_plans.py``), so it inherits the dict form.
    * ``Shot.callouts`` is an ARRAY (``maxItems`` 6), but the seam threads ONE
      callout dict per line all the way to the wire. A bare mapping where the
      compiler declares a list 422s exactly like a dict bbox does.

    A callout whose bbox will not convert is DROPPED — a callout placed at a guessed
    position is the bug this seam already refuses to ship. Nothing usable ⇒ ``None``
    ⇒ the key is not emitted ⇒ byte-identical."""
    if not raw:
        return None
    entries: list[object] = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    out: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        item = cast("dict[str, object]", entry)
        bbox = normalize_bbox(item.get("bbox"))
        if bbox is None:
            continue
        out.append({**item, "bbox": bbox})
    return out or None


def _shot_focus_keys(
    focus_track: object,
    anchor: object,
    lead: object,
    out_point: object,
    footage_viewport: list[int] | None,
) -> list[dict[str, object]] | None:
    """The punch-in keyframes for ONE footage shot: only interacted elements, on the
    screen this shot actually shows (see :func:`normalize_focus_track`).

    Under ``clip_anchor='end'`` the compiler seeks to ``clip_end - duration`` — a
    duration only the narration fixes — so which recorded moment a keyframe lands on
    is unknowable here; no known target ⇒ no zoom rather than a punch-in on a screen
    that may already have moved on."""
    if str(anchor or "") == "end":
        return None
    return normalize_focus_track(
        focus_track,
        viewport=footage_viewport,
        window_start=float(cast("float", lead)) if lead else 0.0,
        window_end=float(cast("float", out_point)) if out_point else None,
    )


def _window_callouts(
    callouts: list[dict[str, object]] | None,
    window_start: float,
    window_end: float | None,
) -> list[dict[str, object]] | None:
    """Rebase each callout's ``at`` onto the shot's window, dropping any outside it.

    ``at`` is copied from the first focus keyframe's raw recording time
    (``shared/footage_plans.py``), the same clock as the tracks. A callout whose
    onset is not in the played window would badge an element that is not on
    screen, so it is dropped. A callout with no numeric ``at`` keeps the compiler's
    default onset. ``None`` when nothing survives (byte-identical no-op)."""
    if not callouts:
        return None
    out: list[dict[str, object]] = []
    for callout in callouts:
        at = _finite(callout.get("at"))
        if at is None:
            out.append(callout)
            continue
        if at < window_start or (window_end is not None and at > window_end):
            continue
        out.append({**callout, "at": at - window_start})
    return out or None


def _attach_clip_framing(
    built: ShotDict,
    shot: ShotDict,
    *,
    focus_track: list[dict[str, object]] | None = None,
    cursor_track: list[dict[str, object]] | None = None,
    footage_viewport: list[int] | None = None,
    callouts: dict[str, object] | None = None,
    split: dict[str, object] | None = None,
) -> None:
    """Copy per-line footage framing (compositor ``Shot.clip_anchor`` /
    ``clip_lead_seconds`` / ``clip_out_seconds``) onto a footage shot. No-op for
    non-footage shots or when the storyboard line supplied nothing — so a slow app
    load-in / zoom at the clip head is skipped (no leading white) only where the author
    asked for it.

    ``clip_out_seconds`` closes the window ``clip_lead_seconds`` opens: an explicit
    source-timeline OUT-point, so a lead of 4.0 and an out of 7.5 show exactly 4.0s-7.5s
    of the recording — the only way to cut SEVERAL distinct shots out of ONE long
    capture instead of replaying every shot from the same in-point. It is NOT gated on
    the anchor (verified against the compiler's renderer): unlike ``clip_lead_seconds``,
    which the compiler honours only under ``clip_anchor='start'``, the out-point applies
    under both ``start`` and ``end``. Presence-is-opt-in like its siblings.

    ORCH-3.2 directed footage: when the VP-2 capture sidecar (promo/website_review)
    or the FTR scenario clip (demo_video/showcase) carried focus/cursor geometry for
    this footage line, copy it onto the shot the SAME way ``clip_anchor`` is — the
    compiler drives a synthetic eased cursor + auto-zoom-to-the-clicked-control from
    it. ``footage_viewport`` (the recorded source WxH that geometry was measured in)
    rides alongside so the compiler's WS-A viewport-aware transform maps those pixels
    onto a non-default output tier. Presence-is-opt-in and footage-guarded by the
    early return: absent geometry/viewport ⇒ that key is not emitted ⇒ byte-identical.
    An author's per-line ``clip_anchor`` is untouched (separate key) so it still
    wins.

    ORCH-3.3 directed shot-plan: ``callouts`` is the ``demo_video_footage_direction``
    plan's per-line callout (``{type, label, at, bbox}``, the bbox resolved compile-side
    from the captured focus track — NEVER the LLM), stamped onto the footage shot so the
    compiler burns it in. Presence-is-opt-in + footage-guarded by the early return: an
    absent callout omits the key ⇒ byte-identical. Later slice: grade.

    Split-screen (T8): ``split`` is the ``demo_video_footage_direction`` plan's
    per-line ``{layout, second_source_url}`` — the second clip ref resolved to a
    fetchable URL compile-side (never the LLM) — stamped onto the footage shot so the
    compositor renders a side-by-side/stacked composite. Presence-is-opt-in +
    footage-guarded by the early return: an absent split omits the key ⇒ byte-identical."""
    if not built.get("video_url"):
        return
    anchor = shot.get("clip_anchor")
    if anchor:
        built["clip_anchor"] = str(anchor)
    lead = shot.get("clip_lead_seconds")
    if lead:
        built["clip_lead_seconds"] = float(lead)
    out_point = shot.get("clip_out_seconds")
    if out_point:
        built["clip_out_seconds"] = float(out_point)
    # The recorder emits ``bbox`` as a dict; the compiler declares a 4-tuple. This
    # is the single choke point every geometry source (VP-2 capture sidecar and FTR
    # scenario clip alike) funnels through, so the conversion lives here once.
    focus_keys = _shot_focus_keys(focus_track, anchor, lead, out_point, footage_viewport)
    if focus_keys:
        built["focus_track"] = focus_keys
    # Cursor and callout times share the focus track's recording clock, so they are
    # windowed by the same in/out points. Track times are sent relative to the
    # declared in-point (``clip_lead_seconds``); any further seek the compiler
    # chooses itself (shared-clip pan, 'end' tail) it applies on its side.
    window_start = float(cast("float", lead)) if lead else 0.0
    window_end = float(cast("float", out_point)) if out_point else None
    cursor_keys = normalize_cursor_track(
        cursor_track, window_start=window_start, window_end=window_end
    )
    if cursor_keys:
        built["cursor_track"] = cursor_keys
    if footage_viewport:
        built["footage_viewport"] = footage_viewport
    callout_list = _window_callouts(normalize_callouts(callouts), window_start, window_end)
    if callout_list:
        built["callouts"] = callout_list
    if split:
        built["split"] = split


def footage_tracks_by_line(
    screens: list[dict[str, object]],
    screen_by_line: dict[str, str],
    *,
    footage: object,
) -> tuple[
    dict[str, list[dict[str, object]]],
    dict[str, list[dict[str, object]]],
    dict[str, list[int]],
]:
    """Resolve per-line focus/cursor tracks + source viewport from the VP-2 capture
    sidecar (ORCH-3.2).

    ``screens`` is ``site_capture``'s ``screens`` output (each may carry the parsed
    ``focus_track``/``cursor_track``/``viewport`` from ``_unzip_screen_tracks``);
    ``screen_by_line`` is the SAME line→screen map the promo/website_review compile
    already builds to bind a footage line to its captured screen. Returns
    ``(focus_by_line, cursor_by_line, viewport_by_line)`` keyed by ``line_id`` — the
    shape ``assemble_shots`` consumes. ``viewport_by_line`` is the recorded source WxH
    (``[w, h]``) the geometry was measured in, so the compiler's WS-A viewport-aware
    transform lands it on-target at a non-default tier.

    Gated on the presence-is-opt-in ``direction.footage`` block: a falsy ``footage`` ⇒
    empty maps ⇒ no geometry rides any shot ⇒ byte-identical to today. (Extraction is
    already gated in ``site_capture``, so absent-footage screens carry no track anyway;
    this second gate keeps the compile side honestly inert too.)"""
    if not footage:
        return {}, {}, {}
    # screens are opaque JSON dicts; the two track values are lists of keyframe
    # objects (already shape-checked by site_capture._coerce_track). isinstance +
    # cast narrows the passthrough value without re-validating each keyframe.
    focus_by_screen: dict[str, list[dict[str, object]]] = {}
    cursor_by_screen: dict[str, list[dict[str, object]]] = {}
    viewport_by_screen: dict[str, list[int]] = {}
    for s in screens:
        sid = s.get("id")
        if not sid:
            continue
        focus = s.get("focus_track")
        if isinstance(focus, list):
            focus_by_screen[str(sid)] = cast("list[dict[str, object]]", focus)
        cursor = s.get("cursor_track")
        if isinstance(cursor, list):
            cursor_by_screen[str(sid)] = cast("list[dict[str, object]]", cursor)
        viewport = normalize_footage_viewport(s.get("viewport"))
        if viewport is not None:
            viewport_by_screen[str(sid)] = viewport
    focus_by_line = {
        lid: focus_by_screen[sid] for lid, sid in screen_by_line.items() if sid in focus_by_screen
    }
    cursor_by_line = {
        lid: cursor_by_screen[sid] for lid, sid in screen_by_line.items() if sid in cursor_by_screen
    }
    viewport_by_line = {
        lid: viewport_by_screen[sid]
        for lid, sid in screen_by_line.items()
        if sid in viewport_by_screen
    }
    return focus_by_line, cursor_by_line, viewport_by_line


def _rebase_attached_tracks(built: ShotDict, lead: float) -> None:
    """Move tracks already on ``built`` by ``lead`` more seconds of source.

    ``lead`` is the shift from the start the tracks are CURRENTLY measured from to
    the new one, not the new absolute lead.

    ``_attach_clip_framing`` windows cursor/focus/callout times by the author's
    lead; the cold open then sets its own lead on the opener, so without this the
    tracks stay measured from the recording head while the compiler reads them
    from the new in-point. Keys before the lead are dropped; an emptied track is
    removed (byte-identical no-op)."""
    if lead == 0:
        return
    for key in ("cursor_track", "focus_track"):
        track = built.get(key)
        if not isinstance(track, list):
            continue
        rebased = [
            {**k, "t": t - lead}
            for k in cast("list[dict[str, object]]", track)
            if (t := _finite(k.get("t"))) is not None and t >= lead
        ]
        if rebased:
            built[key] = rebased
        else:
            built.pop(key, None)
    callouts = built.get("callouts")
    if isinstance(callouts, list):
        windowed = _window_callouts(cast("list[dict[str, object]]", callouts), lead, None)
        if windowed:
            built["callouts"] = windowed
        else:
            built.pop("callouts", None)


def _attach_cold_open(built: ShotDict, cold_open_plan: tuple[str, float] | None) -> None:
    """Reframe the OPENING footage shot to a cold-open window (ORCH-1.3a).
    Presence-is-opt-in: a None plan is a no-op ⇒ byte-identical. Footage-only,
    and an author's explicit per-line clip_anchor (already applied by
    _attach_clip_framing) WINS (C6) — never touches `seconds` (C-duration) or
    shot order (C0)."""
    if cold_open_plan is None:
        return
    if not built.get("video_url"):
        return
    if built.get("clip_anchor"):  # author's per-line framing already won
        return
    anchor, lead = cold_open_plan
    built["clip_anchor"] = anchor
    if lead:
        # The tracks were already windowed by the author's lead (when one was set
        # without an anchor), so move them only by the DIFFERENCE: they end up
        # rebased once, onto the effective shot start.
        prior = _finite(built.get("clip_lead_seconds")) or 0.0
        built["clip_lead_seconds"] = float(lead)
        _rebase_attached_tracks(built, float(lead) - prior)


def _project_a11y(built: ShotDict, shot: ShotDict, a11y_policy: A11yPolicy | None) -> None:
    """Stamp ORCH-0.8 accessibility keys onto a built shot dict — extracted from
    ``assemble_shots`` to keep it under the C901 gate. Presence-is-opt-in: an
    absent policy omits all keys ⇒ byte-identical. Stacks cleanly with ORCH-0.6's
    ``target_fps``/``retime`` projection onto the same dict. A backplate is added
    only for a region that fails the WCAG floor; the exempt flags are only ever
    ``False`` (never a gate)."""
    if a11y_policy is None:
        return
    region = text_region_geometry(shot)
    backplate = choose_backplate(region.text_color, region.bg_color, a11y_policy)
    if backplate is not None:
        built["contrast_backplate"] = backplate
    built["a11y_exempt"] = False
    built["flash_exempt"] = False


def _voiceover(shot: ShotDict) -> tuple[str, float | None] | None:
    """The Studio per-clip voiceover override on a spine shot, or ``None``.

    An uploaded take (``ClipEdit.audio_url`` + its server-probed
    ``audio_seconds``, merged onto the spine) REPLACES this shot's TTS narration.
    Presence-is-opt-in: a shot without ``audio_url`` returns ``None`` and both the
    narration URL and the duration reconcile behave exactly as before.
    """
    url = shot.get("audio_url")
    if not url:
        return None
    raw = shot.get("audio_seconds")
    return str(url), (float(raw) if raw is not None else None)


#: Minimum length for a MEASURED (narrated) beat — a footage/narration clip
#: whose real duration reconciled to something positive but flash-cut short
#: (e.g. a 0.3s narration line). Distinct from ``_FABRICATED_HOLD_SECONDS``
#: (1.0s), which only fires on an exactly-falsy (zero) duration; this floor
#: fires on any positive-but-short MEASURED duration and is intentionally a
#: modest bump, not the [4.5s, 12s] card-motion clamp `slide_stage.py` uses
#: for a different (unmeasured, animated-card) beat shape. No ceiling is
#: added here — only a floor.
_MIN_MEASURED_SECONDS = 1.2
#: Public name for the floor — the shortest narrated beat the compile will ship.
MIN_NARRATED_BEAT_SECONDS = _MIN_MEASURED_SECONDS


def _reconcile_seconds(
    storyboard_seconds: float,
    clip: StagePayload | None,
    *,
    budget: float | None,
    bound: tuple[float, float] | None,
    voiceover: tuple[str, float | None] | None = None,
) -> tuple[float, bool]:
    """Reconcile a shot's duration down the compute-plan ladder.

    An uploaded voiceover outranks everything: it IS this shot's audio, so its
    probed ``audio_seconds`` declares the length (and counts as narrated, so the
    budget fill stays suppressed). Shipping the TTS clip's length instead would be
    a stale duration that poisons the duration-QA guard and the time-budget
    reconciler. Absent an upload, a narration clip's positive duration wins (the
    compositor stretches the visual to match); absent that, an opt-in computed
    ``budget`` fills the duration; an opt-in planned ``(min, max)`` then clamps it.
    With a clip and no ``voiceover`` / ``budget`` / ``bound`` this returns exactly
    today's reconciled value.

    A narrated (measured) duration that reconciles to something positive but
    shorter than :data:`_MIN_MEASURED_SECONDS` is raised to that floor — a
    flash cut is not a legitimate measurement of anything, it is a beat too
    short to read. Returns ``(seconds, floored)`` so the caller can raise a
    soft degrade note; ``floored`` is always ``False`` when nothing narrated
    this beat (the storyboard/budget path is unaffected — no ceiling either).
    """
    seconds = storyboard_seconds
    narrated = False
    clip_seconds = float(clip.get("seconds") or 0.0) if clip else 0.0
    if voiceover is not None:
        narrated = True
        vo_seconds = voiceover[1]
        if vo_seconds and vo_seconds > 0:
            seconds = float(vo_seconds)
        elif clip_seconds > 0:
            # An upload with no probed duration (never emitted by the upload route,
            # which 400s an unprobeable file) — fall back to the TTS length rather
            # than the storyboard's guess.
            seconds = clip_seconds
    elif clip_seconds > 0:
        seconds = clip_seconds
        narrated = True
    floored = False
    if narrated and 0 < seconds < _MIN_MEASURED_SECONDS:
        seconds = _MIN_MEASURED_SECONDS
        floored = True
    if budget is not None and not narrated:
        seconds = float(budget)
    if bound is not None:
        lo, hi = bound
        seconds = min(max(seconds, float(lo)), float(hi))
    return seconds, floored


def _attach_clip_playback(built: ShotDict, shot: ShotDict) -> None:
    """Carry the Studio per-clip playback edits (speed + entering transition)
    from a spine shot onto the built compositor shot.

    Both are presence-is-opt-in: an unset ``speed`` / absent ``transition`` omits
    the key so the compositor keeps its ``1.0`` / hard-cut defaults, byte-identical
    to before. Speed is orthogonal to the narration-duration reconcile (the
    compositor retimes the visual, ``seconds`` is untouched); ``transition`` already
    rode the spine as a ``{type, duration}`` dict.
    """
    speed = shot.get("speed")
    if speed is not None:
        built["speed"] = float(speed)
    transition = shot.get("transition")
    if transition:
        built["transition"] = transition


def _attach_clip_look(built: ShotDict, shot: ShotDict) -> None:
    """Carry the Studio per-clip look edits (colour grade + text overlays) from a
    spine shot onto the built compositor shot.

    Both are presence-is-opt-in: an absent ``adjust`` / empty ``overlays`` omits
    the key so the compositor keeps its ungraded / no-overlay defaults,
    byte-identical to before. Each already rode the spine as the compiler's
    ``{brightness, contrast, saturation, gamma, preset}`` dict / list of
    ``{text, position, …}`` dicts, so they pass through unchanged.
    """
    adjust = shot.get("adjust")
    if adjust:
        built["adjust"] = adjust
    overlays = shot.get("overlays")
    if overlays:
        built["overlays"] = overlays


def _attach_clip_audio(built: ShotDict, shot: ShotDict) -> None:
    """Carry the Studio per-clip audio-level edit (gain/mute/fade) from a spine
    shot onto the built compositor shot. Presence-is-opt-in: absent/falsy audio
    omits the key so the compositor's unmodified-audio default holds (byte-
    identical). Primary audio only — the music bed stays run-level (MusicEdit)."""
    audio = shot.get("audio")
    if audio:
        built["audio"] = audio


def _attach_transition_cue(built: ShotDict, sfx_id: str | None, gain_db: float | None) -> None:
    """Carry the transition cue (catalog id + its gain) onto a built compositor
    shot. Callers apply the ``i > 0`` guard — a transition rides every cut EXCEPT
    the open. No id omits both keys (byte-identical for callers passing no sound);
    no gain omits just the gain, leaving ``shot_sound``'s neutral -10dB default."""
    if not sfx_id:
        return
    built["transition_sfx_id"] = sfx_id
    # The gain rides alongside its id so shot_sound can emit the full
    # {url, gain_db} SoundCue rather than a gainless (full-level) one.
    if gain_db is not None:
        built["transition_sfx_gain_db"] = gain_db


def _freeze_pad_ceiling_patch(
    bound: tuple[float, float] | None, seconds: float
) -> dict[str, float]:
    """The ``max_seconds`` patch to merge onto a compiler shot: reconcile_timeline's
    upper bound (a footage beat's clip + freeze-pad ceiling) so the compiler can cap
    the freeze-pad DRIFT at the ceiling when the ``freeze_pad_cap`` render tag is on
    (Step B).

    Presence-is-opt-in: a non-empty patch ONLY when a positive bound was stamped — an
    empty dict (⇒ no key, not ``max_seconds: null``) on every other path. The compiler
    ignores the field unless the beat is over-narrated footage AND the tag is set, so
    this is inert by default. Returns a patch (merged via ``.update``) rather than
    mutating the shot, so the caller stays a single statement (C901) and this helper
    needs no ``dict[str, Any]`` in its signature.

    ⛔ THE CEILING MAY NOT SIT BELOW THE SHOT'S OWN LENGTH, and it did. ``bound`` is
    reconcile_timeline's window, computed BEFORE :func:`_apply_underspoken_hold` extends
    an underspoken beat, so ``bound[1] < seconds`` is reachable on exactly the beats the
    hold exists for — a shot declaring a ceiling shorter than the duration it also
    declares. Hence ``max(bound[1], seconds)``: a ceiling is never allowed to contradict
    its own shot.

    ⚠️ WHAT THIS DOES **NOT** DO, checked against the compiler before changing it
    (``video_compiler_service`` ``origin/main``): a low ``max_seconds`` never truncated
    anything. ``Shot.max_seconds`` is ``float | None``, ``gt=0``, with NO cross-field
    validation against ``seconds`` (``api/schemas.py:575``), so the wider value is
    accepted rather than rejected; and ``renderer.py:4464`` passes it as ``pad_cap`` to
    ``_freeze_pad_suffix``, which uses it ONLY as the end of the drift ramp
    (``ramp_end = pad_cap if clip_dur < pad_cap < duration else duration``) while the
    ``tpad`` still clones to the FULL duration. So the defect this fixes is a drift ramp
    that stopped early and an internally inconsistent shot — not a hold that was
    "clamped back off". Widening cannot make the compiler reject the shot: a cap outside
    ``(clip_dur, duration)`` simply falls back to the uncapped ramp, i.e. today's
    behaviour."""
    if bound is not None and bound[1] > 0:
        return {"max_seconds": float(max(bound[1], seconds))}
    return {}


# The hold substituted for a shot whose real duration could NOT be measured.
# `video_compiler_service` declares `Shot.seconds` as `gt=0` and required, so
# emitting the reconciled `0.0` would 422 the WHOLE compile at the terminal,
# already-paid stage. The substitution is therefore correct — what was wrong is
# that it used to be silent.
_FABRICATED_HOLD_SECONDS = 1.0

#: Private, orchestrator-internal marker stamped on a shot whose ``seconds`` was
#: fabricated by the floor above. Underscore-prefixed and DRAINED (popped) by
#: :func:`drain_fabricated_hold_notes` before the shots are submitted, so it can
#: never ride out to the compiler's ``Shot`` schema.
FABRICATED_HOLD_KEY = "_fabricated_hold"

_FABRICATED_HOLD_NOTE = (
    "shot duration: {count} shot(s) had NO measurable duration (a zero-second "
    "storyboard line with no narration clip, budget or bound), so a "
    "{seconds:g}s hold was SUBSTITUTED to keep the compile legal — these shots' "
    "lengths are FABRICATED, not measured (line_ids: {lines}); degrading"
)

#: Private, orchestrator-internal marker stamped on a shot whose measured
#: ``seconds`` was raised by :data:`_MIN_MEASURED_SECONDS`. Drained the same
#: way as :data:`FABRICATED_HOLD_KEY` so it never rides out to the compiler.
SHORT_BEAT_FLOOR_KEY = "_short_beat_floored"

_SHORT_BEAT_FLOOR_NOTE = (
    "shot duration: {count} shot(s) measured a flash-cut narration/footage "
    "length under {seconds:g}s, so it was raised to the {seconds:g}s floor "
    "(line_ids: {lines}); degrading"
)

#: Private, orchestrator-internal marker (a list of note strings, stamped on
#: the FIRST built shot only) recording a round-robin visual pool that had
#: fewer assets than beats needing one. Drained alongside the other markers.
_VISUAL_REUSE_NOTES_KEY = "_visual_reuse_notes"

_VISUAL_REUSE_NOTE = (
    "shot visuals: only {pool} round-robin {kind} asset(s) for {needed} beat(s) "
    "needing one — visuals are being REUSED rather than unique; degrading"
)

#: Private, orchestrator-internal marker stamped on a shot whose duration was
#: EXTENDED by an underspoken-beat hold (``AuthoredLine.hold_seconds`` /
#: ``time_budget.underspoken_hold``): the beat's authored words were too few
#: for its allotment even at the natural-rate floor, so the residual is spent
#: as a held last frame instead of dragging the narration further. Stores the
#: applied hold seconds (a float, not a bool) so the drained note can total
#: it. Drained the same way as the other markers so it never rides out to the
#: compiler's ``Shot`` schema.
UNDERSPOKEN_HOLD_KEY = "_underspoken_hold_seconds"

_UNDERSPOKEN_HOLD_NOTE = (
    "shot duration: {count} shot(s) were UNDERSPOKEN (too few authored words for "
    "their allotment, even at the natural-rate floor), so the picture was held on "
    "the last frame for {total:.2f}s total to cover the residual instead of "
    "dragging the narration further (line_ids: {lines}); degrading"
)


def drain_fabricated_hold_notes(
    shots: list[ShotDict], *, degradations: list[Degradation] | None = None
) -> list[str]:
    """Pop the duration/visual degrade markers off ``shots`` and return notes.

    Mutates in place (every marker here is orchestrator-internal and must not
    reach the compiler) and returns the note strings for the shared
    soft-degradation channel — the same sink shape ``apply_ken_burns`` uses.
    Drains FOUR independent markers (fabricated-hold duration, short-beat
    floor, round-robin visual shortfall, underspoken-beat hold) so every
    caller of ``assemble_shots`` gets all four through this one existing
    choke point. Empty list when nothing needed to degrade, so a healthy
    compile is byte-identical to before.

    ``degradations`` is the STRUCTURED twin (``shared/degradation``): every note
    returned is also appended there as a coded entry whose ``detail`` is that
    note, one code per marker, so the stage output is not left to the DAG
    executor's ``unclassified_note``.
    """
    coded: list[tuple[str, str]] = []
    lines = [
        str(shot.get("line_id") or shot.get("id") or "?")
        for shot in shots
        if shot.pop(FABRICATED_HOLD_KEY, False)
    ]
    if lines:
        coded.append(
            (
                SHOT_DURATION_FABRICATED,
                _FABRICATED_HOLD_NOTE.format(
                    count=len(lines), seconds=_FABRICATED_HOLD_SECONDS, lines=", ".join(lines)
                ),
            )
        )
    floored_lines = [
        str(shot.get("line_id") or shot.get("id") or "?")
        for shot in shots
        if shot.pop(SHORT_BEAT_FLOOR_KEY, False)
    ]
    if floored_lines:
        coded.append(
            (
                SHOT_DURATION_FLOORED,
                _SHORT_BEAT_FLOOR_NOTE.format(
                    count=len(floored_lines),
                    seconds=_MIN_MEASURED_SECONDS,
                    lines=", ".join(floored_lines),
                ),
            )
        )
    for shot in shots:
        reuse_notes = shot.pop(_VISUAL_REUSE_NOTES_KEY, None)
        if reuse_notes:
            coded.extend((VISUAL_ASSETS_REUSED, note) for note in reuse_notes)
    held: list[tuple[str, float]] = [
        (str(shot.get("line_id") or shot.get("id") or "?"), popped)
        for shot in shots
        if (popped := shot.pop(UNDERSPOKEN_HOLD_KEY, 0.0))
    ]
    if held:
        coded.append(
            (
                UNDERSPOKEN_BEAT_HELD,
                _UNDERSPOKEN_HOLD_NOTE.format(
                    count=len(held),
                    total=sum(hold for _, hold in held),
                    lines=", ".join(lid for lid, _ in held),
                ),
            )
        )
    if degradations is not None:
        degradations.extend(degradation(code, note) for code, note in coded)
    return [note for _, note in coded]


def _fabricated_hold_patch(seconds: float) -> ShotDict:
    """Mark a shot whose ``seconds`` was SUBSTITUTED by the 1s floor.

    The floor itself is correct and stays (see :data:`_FABRICATED_HOLD_SECONDS`)
    — what it must not do is look measured. Returns a patch (merged via
    ``.update``, mirroring ``_freeze_pad_ceiling_patch``) so the caller stays a
    single statement and ``assemble_shots`` keeps its C901 budget. Empty for any
    positive reconciled duration, so a line that genuinely asked for one second is
    NEVER flagged.
    """
    if seconds:
        return {}
    return {FABRICATED_HOLD_KEY: True}


def _apply_underspoken_hold(
    seconds: float,
    hold: float | None,
    *,
    spoken: float,
    allotment: float | None,
) -> tuple[float, float]:
    """Hold an underspoken beat's picture until its narration + held-frame residual ends.

    ``hold`` is the beat's ``AuthoredLine.hold_seconds`` — computed and
    ceiling-capped upstream by ``time_budget.underspoken_hold``: the seconds an
    underspoken beat's ALLOTMENT holds that its words cannot fill even at the
    natural-rate floor. It is a slice OF the allotment, not an addition to it:
    ``allotment = spoken + hold (+ uncovered)`` (``time_budget.underspoken_hold`` /
    ``underspoken_uncovered``), ``reconcile_timeline`` totals ``seconds`` with the
    hold already inside it, and ``demo_video_narrate.fit_target_for`` SUBTRACTS it
    from the voice's window.

    ⛔ SO IT IS COUNTED ONCE, HERE, AS A FLOOR — never added on top of the reconciled
    duration. This function used to return ``seconds + hold``. On a footage beat the
    reconciled ``seconds`` is already the clip (``min_seconds`` == the footage) and so
    already contains the hold; adding it again held a dead frame PAST the footage.
    Run ``01a0b007-9c2c`` clip-5: footage 9.56s, narration 7.938s, hold 1.50s ⇒ the
    compiled shot was 11.067s (= 9.56 + 1.51) and the master ended in silence.

    The held picture therefore ends at ``spoken + hold`` — the narration's measured end
    plus the residual — capped at the beat's ``allotment`` when one was planned (the
    hold is a slice of it, so it can never reach past it), and it only ever EXTENDS: a
    reconciled duration already at or past that point (footage, a short-beat floor, a
    plan bound) is left alone. ``spoken`` is ``0.0`` when nothing narrated the beat.

    ⛔ Distinct from :data:`_MIN_MEASURED_SECONDS` (that floor fixes a FLASH CUT) and
    never folded into ``freeze_pad_seconds`` / ``max_seconds`` — see the comment on
    ``AuthoredLine.hold_seconds`` for why.

    Returns ``(seconds, applied)``: ``applied`` is the extension actually added —
    ``0.0`` when ``hold`` is absent, non-positive, non-finite, or already covered — so
    the caller raises the soft note only when a frame was really held.
    """
    if hold is None or not math.isfinite(hold) or hold <= 0:
        return seconds, 0.0
    held_end = spoken + hold
    if allotment is not None and math.isfinite(allotment) and allotment > 0:
        held_end = min(held_end, allotment)
    if held_end <= seconds:
        return seconds, 0.0
    return held_end, held_end - seconds


def _spoken_seconds(clip: StagePayload | None, voiceover: tuple[str, float | None] | None) -> float:
    """The measured length of the audio that actually narrates this beat, or ``0.0``.

    Same precedence as :func:`_reconcile_seconds` (an uploaded voiceover's probed
    duration outranks the TTS clip), read BEFORE any floor/bound, because the
    underspoken hold starts where the SPEECH ends, not where the shot does.
    """
    if voiceover is not None and voiceover[1] and voiceover[1] > 0:
        return float(voiceover[1])
    return float(clip.get("seconds") or 0.0) if clip else 0.0


def _duration_fields(
    seconds: float,
    floored: bool,
    hold: float | None,
    bound: tuple[float, float] | None,
    *,
    spoken: float,
    allotment: float | None,
) -> ShotDict:
    """The ``seconds`` + degrade-marker keys to merge onto a built shot, given
    the already-reconciled duration. Folds :func:`_fabricated_hold_patch`,
    the :data:`SHORT_BEAT_FLOOR_KEY` marker, :func:`_apply_underspoken_hold`,
    and :func:`_freeze_pad_ceiling_patch` into one call so ``assemble_shots``
    stays a single statement here (C901).
    """
    fields: ShotDict = {}
    fields.update(_fabricated_hold_patch(seconds))
    base_seconds = seconds or _FABRICATED_HOLD_SECONDS
    extended_seconds, applied_hold = _apply_underspoken_hold(
        base_seconds, hold, spoken=spoken, allotment=allotment
    )
    fields["seconds"] = extended_seconds
    if floored:
        fields[SHORT_BEAT_FLOOR_KEY] = True
    if applied_hold:
        fields[UNDERSPOKEN_HOLD_KEY] = applied_hold
    fields.update(_freeze_pad_ceiling_patch(bound, extended_seconds))
    return fields


def _visual_reuse_shortfall_notes(
    storyboard_shots: list[ShotDict],
    *,
    keyed_videos: dict[str, str],
    clip_urls: list[str],
    keyed_images: dict[str, str],
    image_urls: list[str],
) -> list[str]:
    """Soft-note strings for a round-robin pool with fewer assets than beats
    that will draw from it. A keyed line never draws from the pool, so it
    does not count toward "needing" one. Gated on a non-empty pool: a beat
    with NO assets at all is the pre-existing (untouched) no-visual path, not
    a reuse shortfall. Extracted from :func:`assemble_shots` to keep it under
    the C901 budget.
    """
    video_needed = sum(1 for s in storyboard_shots if not keyed_videos.get(str(s["line_id"])))
    image_needed = sum(1 for s in storyboard_shots if not keyed_images.get(str(s["line_id"])))
    notes: list[str] = []
    if clip_urls and len(clip_urls) < video_needed:
        notes.append(
            _VISUAL_REUSE_NOTE.format(pool=len(clip_urls), kind="video", needed=video_needed)
        )
    if image_urls and len(image_urls) < image_needed:
        notes.append(
            _VISUAL_REUSE_NOTE.format(pool=len(image_urls), kind="image", needed=image_needed)
        )
    return notes


def _stamp_visual_reuse_notes(out: list[ShotDict], notes: list[str]) -> None:
    """Stamp ``notes`` (if any) onto the FIRST built shot as the private
    :data:`_VISUAL_REUSE_NOTES_KEY` marker, for :func:`drain_fabricated_hold_notes`
    to pick up. Extracted purely to keep ``assemble_shots`` under C901."""
    if notes and out:
        out[0][_VISUAL_REUSE_NOTES_KEY] = notes


def assemble_shots(
    *,
    storyboard_shots: list[ShotDict],
    images: list[StagePayload],
    narration_clips: list[StagePayload],
    captions_by_line: dict[str, str] | None = None,
    video_clips: list[StagePayload] | None = None,
    image_by_line: dict[str, str] | None = None,
    video_by_line: dict[str, str] | None = None,
    fit_by_line: dict[str, str] | None = None,
    sfx_by_line: dict[str, list[StagePayload]] | None = None,
    transition_sfx_id: str | None = None,
    transition_sfx_gain_db: float | None = None,
    seconds_by_line: dict[str, float] | None = None,
    min_max_by_line: dict[str, tuple[float, float]] | None = None,
    hold_seconds_by_line: dict[str, float] | None = None,
    target_fps: int | None = None,
    retime: str | None = None,
    a11y_policy: A11yPolicy | None = None,
    cold_open_plan: tuple[str, float] | None = None,
    focus_track_by_line: dict[str, list[dict[str, object]]] | None = None,
    cursor_track_by_line: dict[str, list[dict[str, object]]] | None = None,
    footage_viewport_by_line: dict[str, list[int]] | None = None,
    callouts_by_line: dict[str, dict[str, object]] | None = None,
    split_by_line: dict[str, dict[str, object]] | None = None,
) -> list[ShotDict]:
    """Build video-compile shot dicts. Audio joins by ``line_id``; round-robin
    images and video clips cycle across shots; ``image_by_line`` /
    ``video_by_line`` join 1:1; ``description`` is a text-card fallback emitted
    only when present.

    A shot's visual precedence mirrors the compositor's: a ``video_url`` (real
    footage — a keyed ``video_by_line`` clip beats a round-robin one) wins over
    an ``image_url`` (a keyed ``image_by_line`` image beats a round-robin AI
    image), which wins over the ``description`` text card. The promo blueprint
    drives a per-line visual plan (footage / slide / html / ai-image) through the
    keyed maps; lesson and app_review keep the round-robin + ``image_by_line``
    paths unchanged.

    ``seconds_by_line`` / ``min_max_by_line`` are the opt-in compute-plan seam: a
    per-line budget FILLS a shot's duration when no narration clip fixed it, and a
    per-line ``(min, max)`` CLAMPS it into the planned window. Both default to
    ``None`` → the duration reconcile is byte-identical to before.

    ``hold_seconds_by_line`` is the opt-in underspoken-beat seam: a positive
    per-line value holds the picture on its last frame until the narration's end
    plus that many seconds (capped at the line's ``seconds_by_line`` allotment, of
    which the hold is a slice) — see :func:`_apply_underspoken_hold`. It is a
    FLOOR applied AFTER the ``min_max_by_line`` clamp, never added on top of a
    duration that already covers it, and is never folded
    into ``freeze_pad_seconds``/``max_seconds`` (a different, opposite-direction
    pad — see the constant's docstring). Default ``None`` ⇒ every hold is 0.0 ⇒
    byte-identical to before.

    ``target_fps`` / ``retime`` are the opt-in cadence seam: a resolved master
    frame rate stamps ``target_fps`` + ``retime`` onto every shot. Both default to
    ``None`` → neither key is emitted → byte-identical to before.

    ``cold_open_plan`` is the opt-in cold-open seam (ORCH-1.3a): an
    ``(anchor, lead)`` reframes ONLY the opening footage shot's window; default
    ``None`` ⇒ no shot is reframed ⇒ byte-identical to before.

    ``focus_track_by_line`` / ``cursor_track_by_line`` are the opt-in directed-footage
    seam (ORCH-3.2): the VP-2 capture sidecar's per-line focus/cursor geometry, stamped
    onto the matching FOOTAGE shot (``video_url`` present) so the compiler can drive a
    synthetic eased cursor + auto-zoom. Both default ``None`` ⇒ no key is emitted ⇒
    byte-identical to before (mirrors the ``clip_anchor`` framing seam).

    ``footage_viewport_by_line`` is the recorded source WxH (``[w, h]``) the geometry
    was measured in — stamped onto the same footage shots so the compiler's WS-A
    viewport-aware transform maps those pixels onto a non-default output tier. Default
    ``None`` ⇒ no key is emitted ⇒ byte-identical.

    ``callouts_by_line`` is the opt-in directed shot-plan callout seam (ORCH-3.3): the
    ``demo_video_footage_direction`` plan's per-line callout (``{type, label, at, bbox}``,
    the bbox resolved from the captured focus track — never the LLM), stamped onto the
    matching FOOTAGE shot. Default ``None`` ⇒ no key is emitted ⇒ byte-identical.

    ``split_by_line`` is the opt-in split-screen seam (T8): the
    ``demo_video_footage_direction`` plan's per-line ``{layout, second_source_url}`` —
    the second clip ref resolved to a fetchable URL compile-side (never the LLM) —
    stamped onto the matching FOOTAGE shot so the compositor renders a side-by-side /
    stacked composite. Default ``None`` ⇒ no key is emitted ⇒ byte-identical.

    A spine shot carrying a Studio voiceover upload (``audio_url`` +
    ``audio_seconds``) has that audio REPLACE its narration clip, and its probed
    duration declares the shot's ``seconds``; a shot without ``audio_url`` takes
    its narration clip exactly as before (presence-is-opt-in)."""
    audio_by_line = {str(c["line_id"]): c for c in narration_clips}
    image_urls = [img["image_url"] for img in images if img.get("image_url")]
    clip_urls = [c["video_url"] for c in (video_clips or []) if c.get("video_url")]
    keyed_images = image_by_line or {}
    keyed_videos = video_by_line or {}
    captions = captions_by_line or {}
    fits = fit_by_line or {}
    sfx_map = sfx_by_line or {}
    budgets = seconds_by_line or {}
    bounds = min_max_by_line or {}
    holds = hold_seconds_by_line or {}
    # ORCH-3.2 directed-footage geometry, keyed per footage line (empty ⇒ inert).
    focus_tracks = focus_track_by_line or {}
    cursor_tracks = cursor_track_by_line or {}
    footage_viewports = footage_viewport_by_line or {}
    # ORCH-3.3 directed shot-plan callout, keyed per footage line (empty ⇒ inert).
    callouts = callouts_by_line or {}
    # T8 split-screen composite, keyed per footage line (empty ⇒ inert).
    splits = split_by_line or {}

    visual_reuse_notes = _visual_reuse_shortfall_notes(
        storyboard_shots,
        keyed_videos=keyed_videos,
        clip_urls=clip_urls,
        keyed_images=keyed_images,
        image_urls=image_urls,
    )

    # Round-robin adjacency state: the visual actually assigned to the
    # PREVIOUS shot from each un-keyed pool, so `_resolve_visual` can refuse
    # an immediate repeat. Reset to None whenever a beat did not draw from a
    # pool (keyed, or the pool was exhausted) — a gap breaks the streak, so
    # the asset is free to be reused once something else has shown.
    prev_video_url: str | None = None
    prev_image_url: str | None = None
    # Per-pool draw state for the un-keyed beats (see `_VisualPool`): the
    # shortfall note above deliberately still sizes the FULL pool.
    line_ids = [str(s["line_id"]) for s in storyboard_shots]
    video_pool = _VisualPool(clip_urls, [keyed_videos[x] for x in line_ids if keyed_videos.get(x)])
    image_pool = _VisualPool(image_urls, [keyed_images[x] for x in line_ids if keyed_images.get(x)])

    out: list[ShotDict] = []
    for i, shot in enumerate(storyboard_shots):
        line_id = str(shot["line_id"])
        clip = audio_by_line.get(line_id)
        # A Studio voiceover upload on the spine shot REPLACES the TTS narration —
        # URL and duration must be decided together (an `_attach_*` helper runs
        # after `seconds` is fixed, so it cannot carry this one).
        voiceover = _voiceover(shot)
        audio_url: str | None = (
            voiceover[0] if voiceover else (str(clip["artifact_url"]) if clip else None)
        )
        seconds, floored = _reconcile_seconds(
            float(shot.get("seconds") or 0.0),
            clip,
            budget=budgets.get(line_id),
            bound=bounds.get(line_id),
            voiceover=voiceover,
        )
        built: ShotDict = {
            "id": str(shot["id"]),
            "line_id": line_id,
        }
        # Shot.seconds must be > 0; a zero-duration storyboard line with no
        # narration falls back to a 1s hold so the compositor never 422s. An
        # underspoken beat's residual (if any) then EXTENDS that duration —
        # see `_apply_underspoken_hold`.
        built.update(
            _duration_fields(
                seconds,
                floored,
                holds.get(line_id),
                bounds.get(line_id),
                spoken=_spoken_seconds(clip, voiceover),
                allotment=budgets.get(line_id),
            )
        )
        # Only carry a description when the storyboard supplied one — a missing
        # key, not `description: null`, for narration/visual-built shots.
        description = shot.get("description")
        if description is not None:
            built["description"] = description
        visual = _resolve_visual(
            line_id,
            keyed_videos=keyed_videos,
            video_pool=video_pool,
            keyed_images=keyed_images,
            image_pool=image_pool,
            prev_video_url=prev_video_url,
            prev_image_url=prev_image_url,
        )
        built.update(visual)
        prev_video_url, prev_image_url = _next_round_robin_state(
            line_id, visual, keyed_videos=keyed_videos, keyed_images=keyed_images
        )
        _attach_clip_framing(
            built,
            shot,
            focus_track=focus_tracks.get(line_id),
            cursor_track=cursor_tracks.get(line_id),
            footage_viewport=footage_viewports.get(line_id),
            callouts=callouts.get(line_id),
            split=splits.get(line_id),
        )
        if i == 0:
            _attach_cold_open(built, cold_open_plan)
        # Opt-in cadence seam: a resolved master frame rate stamps every shot with
        # the compiler's conform target + retime mode. Default None ⇒ both keys are
        # omitted ⇒ byte-identical to before (mirrors the seconds_by_line seam).
        if target_fps is not None:
            built["target_fps"] = target_fps
            built["retime"] = retime
        _project_a11y(built, shot, a11y_policy)
        if audio_url:
            built["audio_url"] = audio_url
        # An author's per-shot caption override (studio edit, carried on the spine
        # shot) WINS over the computed `captions_by_line` map — and emits even when
        # burned-in captions are off (`captions_mode != burned_in` → map is None),
        # so an explicitly-authored caption always burns in.
        caption = shot.get("caption") or captions.get(line_id)
        if caption:
            built["caption"] = caption
        # Fit: an author's per-shot override wins over the smart-auto map (e.g.
        # "cover" for AI-image backgrounds / "scroll" for tall captures). The
        # compositor defaults to "contain", so omit the key when neither is set.
        fit = shot.get("fit") or fits.get(line_id)
        if fit:
            built["fit"] = fit
        _attach_clip_playback(built, shot)
        _attach_clip_look(built, shot)
        _attach_clip_audio(built, shot)
        # Per-line accent SFX cues carry catalog ids here; resolve_shot_sound_urls
        # (compile time) turns them into presigned URLs the compositor mixes in.
        # A transition cue rides every cut EXCEPT the open (no incoming cut on the
        # first shot). Omit both keys when empty (regression-safe for callers that
        # pass no sound).
        cues = sfx_map.get(line_id)
        if cues:
            built["sfx"] = cues
        if i > 0:
            _attach_transition_cue(built, transition_sfx_id, transition_sfx_gain_db)
        out.append(built)
    _stamp_visual_reuse_notes(out, visual_reuse_notes)
    return out
