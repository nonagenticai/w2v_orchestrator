"""footage_probe — turn the project's uploaded ``video_sources`` rows into a stage output.

The action_reel input side already exists and is NOT re-implemented here: the
operator uploads clips through ``POST /api/projects/{id}/video-sources``
(``api/routes/uploads_video.py``), which streams the bytes into MinIO and
commits one ``video_sources`` row per clip with ``preview_status='pending'`` and
a raw ffprobe blob in ``probe``. This stage is the bridge from that TABLE into
the DAG: it reads the rows for the run's project and emits one flat, typed
descriptor per source so every stage below it (shot_analysis, segment_select,
cut_plan, compile) reads footage metadata from a stage output like any other
prior output, instead of each opening its own DB session.

Three things this stage deliberately does NOT do:

* It does not probe anything itself. ffprobe already ran at upload time; the
  blob is authoritative and re-probing a 134 MiB original from a pod sized for
  JSON is exactly the memory mistake ``uploads_video`` exists to avoid.
* It does not wait for the preview/poster/sprite derivatives. Those are filled
  asynchronously and ``preview_status`` is surfaced verbatim per clip on
  ``sources[].preview_status`` so the reviewer can see which clips are still
  rendering — blocking on them would wedge the whole run behind a cosmetic proxy.
* It does not fail on a ``pending`` row, and it does not NOTE one either. A note
  marks the whole run ``degraded``, and a derivative that has not rendered yet is
  informational, not a soft-failure — the structured field already says so. Only
  ``preview_status='failed'`` (the worker errored) earns a note.
* It does not fail on an unreadable probe blob — as long as some OTHER source is
  usable. A source with an empty ``probe`` yields zeroed duration/fps/codec plus a
  note; the LLM below can still be shown the clip, and a per-source hard failure
  would make one slow derivative worker able to kill a run whose footage is
  perfectly fine. The whole SET being unusable is a different matter and does fail
  loud — see the any-usable-source guard in :meth:`FootageProbeStage.handle`. That
  guard asks whether ANY source has a positive duration rather than whether the SUM
  is positive, and ``_describe`` clamps a negative probe duration to 0s, so a
  malformed clip can never subtract from a good one.

DB access: this handler needs a session, which stage handlers do not otherwise
get. It follows the ``source_ingest`` precedent — a blueprint factory
(``make_footage_probe_stage``) lifts ``session_factory`` out of the dispatcher's
deps mapping with ``.get`` (never ``[]``), so constructing the stage from an
EMPTY deps mapping still works. That is load-bearing: the LLM timeout-ladder
guard builds every registered handler with ``{}`` and fails loudly on anything
that raises. Without a factory the stage raises a precondition error at RUN
time, where it is a real, visible failure rather than a silent test exclusion.

The session is tenant-bound with ``bind_session_tenant`` before the first query:
``video_sources`` carries no ``tenant_id`` of its own (its RLS policy is a 1-hop
join up to ``projects.tenant_id``), so an unbound session under enforce reads
zero rows and the stage would report "no footage" for a project full of it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar
from uuid import UUID

from sqlalchemy import select

from src.database.models import VideoSource
from src.database.session import bind_session_tenant
from src.services.stages.base import (
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
    StagePreconditionError,
)
from src.services.tts_storage import _get_storage

if TYPE_CHECKING:
    from collections.abc import Iterator

    from src.services.runs.run_context import RunContext

log = logging.getLogger(__name__)

# ffprobe's stream dict spells the frame rate as a RATIONAL string ("30000/1001"),
# not a float — reading it as a number yields 0 for every real camera file.
_FPS_KEYS = ("avg_frame_rate", "r_frame_rate")


class FootageProbeInput(StageInput):
    """No typed inputs — the sources are read from the DB by project id."""


class ProbedSource(StageOutput):
    """One uploaded clip, flattened out of its ``video_sources`` row.

    ``id`` is the ``video_sources`` row id and is THE join key every stage below
    uses to name a clip (segment_select's ``source_id``, cut_plan's shots).

    ``original_url`` / ``preview_url`` are freshly presigned per run of this
    stage. They are stored on the output anyway (rather than re-minted by each
    consumer) because a reviewer looking at the paused stage needs playable
    links; consumers that render much later (compile) re-sign through the
    shared ``refresh_shot_asset_urls`` pass, so an expired URL here is never
    what reaches the compositor.
    """

    id: str
    filename: str
    bytes: int
    content_type: str = ""
    preview_status: str = "pending"
    duration_s: float = 0.0
    fps: float = 0.0
    codec: str = ""
    width: int = 0
    height: int = 0
    original_url: str = ""
    preview_url: str = ""


class FootageProbeOutput(StageOutputWithNotes):
    # Declared output contract — enforced once, centrally, in
    # `runs/dag_executor.py::_enforce_produces`. Empty here means the stage
    # reported success having produced nothing, which degrades the run.
    # `shot_analysis` reads `sources` to fan out the analyzer and `cut_plan` reads it
    # for the per-source playable URL — no sources, no reel.
    produces: ClassVar[tuple[str, ...]] = ("sources",)

    # `notes` is inherited from StageOutputWithNotes — the house soft-degradation
    # channel (a completed stage with non-empty notes surfaces the run as
    # `degraded`). Used here for a source whose derivative pass FAILED or whose
    # probe blob is unreadable — both are "you may want to look at this", never
    # "fail the run". A still-`pending` derivative is deliberately NOT in there:
    # it is normal in-flight state, it is already reported on
    # `sources[].preview_status`, and noting it would badge a flawless run
    # "completed with warnings".
    sources: list[ProbedSource]
    total_duration_s: float = 0.0


def make_footage_probe_stage(deps: dict) -> StageHandler:
    """Blueprint factory: wire the dispatcher's ``session_factory`` in.

    ``.get`` rather than ``[]`` on purpose — see the module docstring. A stage
    built without a factory raises at run time (where a missing session is a
    genuine, visible failure) instead of at construction time (where it would
    silently drop this stage out of the fleet-wide invariant guards).
    """
    return FootageProbeStage(session_factory=deps.get("session_factory"))


class FootageProbeStage(StageHandler[FootageProbeInput, FootageProbeOutput]):
    name = "footage_probe"
    input_model = FootageProbeInput
    output_model = FootageProbeOutput

    def __init__(self, *, session_factory=None) -> None:  # noqa: ANN001 -- house pattern
        # Untyped like SourceCacheRepository / RunDbClaim: the async sessionmaker's
        # type is not importable without dragging SQLAlchemy generics through every
        # stage module, and annotating it `Any` would spend from the bare-`Any`
        # ratchet budget for zero safety.
        self._session_factory = session_factory

    async def handle(self, payload: FootageProbeInput, ctx: RunContext) -> FootageProbeOutput:
        if self._session_factory is None:
            raise StagePreconditionError(
                "footage_probe: no session_factory in handler deps; the dispatcher "
                "must pass one (see make_footage_probe_stage)",
                missing=["session_factory"],
            )
        project_id = str(ctx.config_snapshot.get("project_id") or "")
        if not project_id:
            raise StagePreconditionError(
                "footage_probe: config_snapshot carries no project_id",
                missing=["project_id"],
            )

        rows = await self._load_rows(self._session_factory, UUID(project_id), ctx.tenant_id)
        if not rows:
            # A HARD failure, not a skip: action_reel is defined by the footage the
            # operator uploaded. A run with none of it cannot produce the product,
            # and "completed with zero clips" is precisely the silent-degradation
            # lie the stage error taxonomy exists to prevent.
            raise StagePreconditionError(
                "footage_probe: project has no video_sources; upload footage via "
                "POST /api/projects/{project_id}/video-sources before running",
                missing=["video_sources"],
            )

        notes: list[str] = []
        sources: list[ProbedSource] = []
        # Sources whose storage key exists but could not be minted into a URL.
        # `_presign` swallows that to "" at log.info, which means the reviewer is
        # shown a segment gate whose clips will not play and the run is green with
        # nothing to read. The footage itself is intact, so this is a note (the
        # run degrades) and never a failure.
        unpresignable: list[ProbedSource] = []
        for row in rows:
            described = await self._describe(row, notes)
            if (row.object_key and not described.original_url) or (
                row.preview_key and not described.preview_url
            ):
                unpresignable.append(described)
            sources.append(described)

        if unpresignable:
            named = ", ".join(f"{s.filename} ({s.id})" for s in unpresignable[:5])
            if len(unpresignable) > 5:
                named += f" (+{len(unpresignable) - 5} more)"
            notes.append(
                f"{len(unpresignable)} of {len(sources)} sources could not be presigned; "
                f"their clips will not play for the reviewer (the footage is intact): {named}"
            )

        # ANY usable source, not "the sum is positive". Those are the same test only
        # while every `duration_s` is non-negative — which `_describe` now guarantees
        # by clamping (see there). Asking the question directly is still the honest
        # form at the call site: the product rule is "one usable clip is enough to cut
        # a reel", and a sum states that only by arithmetic accident. It also means a
        # future edit that re-admits a negative degrades the reported total, not the
        # run — a NEGATIVE clip can never again cancel out a good one and hard-fail a
        # project whose footage is fine.
        if not any(s.duration_s > 0 for s in sources):
            # The SAME hard failure as the zero-rows branch above, for the case that
            # branch does not cover: N rows that are all unusable. `_describe` never
            # raises (deliberately — one bad blob among good footage must not kill a
            # run), so every source with an unreadable probe lands here as 0s, and
            # `segment_select` then drops every one of its segments (`duration <= 0
            # -> None`). The run therefore reaches the operator "completed, degraded"
            # with an EMPTY reel and no named cause — which is exactly the
            # silent-degradation lie the zero-rows branch exists to prevent. Zero rows
            # and N-unusable-rows are the same product outcome, so they get the same
            # taxonomy. Note this is a WHOLE-SET test: one usable source is enough to
            # continue, and the per-source notes still degrade the run in that case.
            named = ", ".join(f"{s.filename} ({s.id})" for s in sources[:5])
            if len(sources) > 5:
                named += f" (+{len(sources) - 5} more)"
            raise StagePreconditionError(
                f"footage_probe: all {len(sources)} video_sources have zero usable "
                f"duration; no ffprobe blob yields one: {named}. Re-upload the footage "
                "via POST /api/projects/{project_id}/video-sources, or wait for the "
                "upload probe to complete, before running",
                # NOT `missing=["video_sources"]` — the rows exist. What is missing is
                # the probed duration that makes them usable, and naming that is what
                # tells the operator to look at the upload probe, not at the uploads.
                missing=["probed_duration"],
            )

        # Safe to sum straight: `_describe` clamps each `duration_s` at 0.0, so no
        # source can DRAG THE TOTAL DOWN. This value is user-visible and is what the
        # planning stages below size the reel against — before the clamp, one
        # malformed probe reporting a negative container duration silently shortened
        # the whole project's reported footage.
        total_duration = sum(s.duration_s for s in sources)

        return FootageProbeOutput(
            sources=sources,
            total_duration_s=round(total_duration, 3),
            notes=notes,
        )

    @staticmethod
    async def _load_rows(factory, project_id: UUID, tenant_id: UUID | None) -> list[VideoSource]:  # noqa: ANN001 -- untyped sessionmaker, see __init__
        """The project's sources in upload order (which IS shoot order).

        The tenant GUC is bound BEFORE the first query: ``video_sources`` has no
        tenant column, so its RLS policy joins up to ``projects.tenant_id`` and
        an unbound session under enforce silently returns zero rows.

        ``factory`` is a PARAMETER rather than read off ``self`` so the ``is None``
        precondition checked in :meth:`handle` actually flows here — a type checker
        cannot carry that narrowing across a method boundary, and re-checking it
        would be a second, divergent copy of the same precondition.
        """
        async with factory() as session:
            await bind_session_tenant(session, tenant_id)
            result = await session.execute(
                select(VideoSource)
                .where(VideoSource.project_id == project_id)
                .order_by(VideoSource.created_at)
            )
            return list(result.scalars().all())

    async def _describe(self, row: VideoSource, notes: list[str]) -> ProbedSource:
        """One row → one flat descriptor. Never raises: a source we cannot fully
        describe still reaches the reviewer (with a note) rather than failing."""
        probe = row.probe or {}
        duration, fps, codec, width, height = _read_probe(probe)
        # `pending` is NOT a note. A note makes the whole run `degraded`
        # (api/routes/runs/_shared.py), and a derivative still rendering is
        # informational — the note text itself said "original is still usable".
        # Badging a flawless run "completed with warnings" trains reviewers to
        # ignore the badge, which hides a REAL degradation later. The status is
        # already carried losslessly per clip in `sources[].preview_status`, so
        # dropping the note costs the reviewer nothing. `failed` IS a genuine
        # soft-failure (the derivative worker errored, `row.error` is set) and
        # keeps its note.
        if row.preview_status == "failed":
            notes.append(
                f"source {row.id} preview_status=failed "
                f"({row.error or 'derivative render failed'}; original is still usable)"
            )
        if duration <= 0:
            # Fires for a NEGATIVE as well as a zero — deliberately. A negative is
            # exactly as unusable as a missing one (you cannot cut a -30s clip), and
            # the reviewer has to be told this source contributed nothing.
            notes.append(f"source {row.id} has no probed duration; treating as 0s")
        # Clamp at the SOURCE, before the value can enter `ProbedSource`. `_as_float`
        # passes through any parseable float and `_read_probe` does not bound it, so a
        # malformed remux (a broken-timestamp TS/MKV whose `stream.duration` reads
        # "-30.0") used to hand a NEGATIVE duration downstream. That was two bugs at
        # once: it dragged `total_duration_s` below the real footage length, and in a
        # sum it could cancel out a perfectly good clip (-30.0 + 12.5 = -17.5) and hard
        # -fail a project with usable footage. Clamping here fixes both in one place —
        # every consumer of `sources[].duration_s` (segment_select, cut_plan, the
        # total) then reads the same honest "unusable = 0s" that the note describes,
        # instead of each having to re-defend against a negative.
        duration = max(duration, 0.0)
        return ProbedSource(
            id=str(row.id),
            filename=row.filename,
            bytes=int(row.bytes or 0),
            content_type=row.content_type or "",
            preview_status=row.preview_status,
            duration_s=duration,
            fps=fps,
            codec=codec,
            width=width,
            height=height,
            original_url=await _presign(row.object_key),
            # Only presign a derivative the worker actually produced — presigning a
            # NULL/absent key mints a URL that 404s, which reads to the FE as a
            # broken clip rather than one still rendering.
            preview_url=await _presign(row.preview_key) if row.preview_key else "",
        )


async def _presign(key: str | None) -> str:
    """Best-effort presigned GET; "" when the key is absent or unpresignable.

    Soft on purpose: a storage backend that cannot presign must degrade this
    stage to metadata-only (the reviewer loses playback), never fail a run whose
    footage is intact.
    """
    if not key:
        return ""
    try:
        storage = await _get_storage()
        return await storage.get_presigned_url(key) or ""
    except Exception as exc:  # noqa: BLE001 -- playback links are an enhancement
        log.info("footage_probe: could not presign %s: %s", key, exc)
        return ""


def _read_probe(probe: dict[str, object]) -> tuple[float, float, str, int, int]:
    """Pull ``(duration_s, fps, codec, width, height)`` out of a raw ffprobe blob.

    Defensive by design. ``probe`` is stored as free-form JSONB precisely because
    ffprobe's shape varies per container, and a ``pending`` row's blob is the
    empty object — so every field is probed independently and a miss yields the
    zero value rather than an exception. ``format.duration`` is preferred over a
    stream's: it is the container duration, which is what a cut is measured
    against.

    ``width``/``height`` are the DISPLAY geometry, not the raw coded geometry.
    A phone-shot portrait clip is almost always stored as a landscape raster
    plus a ±90° display matrix, so the raw stream keys read 3840x2160 for a
    clip that every player — and our own preview transcoder, which lets ffmpeg
    autorotate before ``scale=-2:'min(720,ih)'`` — shows as 2160x3840. Reporting
    the coded pair put a 406x720 proxy next to a "3840x2160" label in the
    reviewer's grid. See :func:`_read_rotation`.
    """
    fmt = _as_obj_dict(probe.get("format"))
    duration = _as_float(fmt.get("duration")) if fmt is not None else 0.0
    stream = _first_video_stream(probe)
    if stream is None:
        return duration, 0.0, "", 0, 0
    if duration <= 0:
        duration = _as_float(stream.get("duration"))
    width = int(_as_float(stream.get("width")))
    height = int(_as_float(stream.get("height")))
    if _read_rotation(stream) in (90, 270):
        width, height = height, width
    return (
        duration,
        _read_fps(stream),
        str(stream.get("codec_name") or ""),
        width,
        height,
    )


def _read_rotation(stream: dict[str, object]) -> int:
    """Display rotation in degrees, normalised to ``[0, 360)``; ``0`` when absent.

    ffprobe spells the same fact two ways depending on ffmpeg version and
    container, so both are read, most-authoritative first:

    * a ``side_data_list`` entry of type ``Display Matrix`` carrying ``rotation``
      — the modern form, a FLOAT and usually NEGATIVE (``-90.0`` for a clip that
      displays rotated 90° clockwise);
    * the legacy ``tags.rotate`` string (``"90"``, ``"270"``).

    The Display Matrix wins when both are present: it is what ffmpeg itself
    applies when autorotating, and it is the one modern muxers keep accurate.
    The sign is discarded because only the axis swap matters to us — 90 and 270
    both transpose the frame, 0 and 180 both leave it alone.

    Tolerant like the rest of this module: anything unparseable (a string, a
    dict, ``None``, a NaN) is skipped, and a blob with nothing readable yields
    ``0`` so the caller keeps the raw coded dimensions. These blobs come from
    user-uploaded files; this must never raise.
    """
    for raw in _rotation_candidates(stream):
        degrees = _as_rotation(raw)
        if degrees is not None:
            return degrees
    return 0


def _rotation_candidates(stream: dict[str, object]) -> Iterator[object]:
    """Every place a rotation may hide, in precedence order."""
    side_data = stream.get("side_data_list")
    if isinstance(side_data, list):
        for entry in side_data:
            narrowed = _as_obj_dict(entry)
            if narrowed is None:
                continue
            # Tolerate a blob that omits the type key — only the display matrix
            # carries `rotation` — but never accept a DIFFERENT named side data.
            # A missing `rotation` yields None, which `_as_rotation` skips, so no
            # separate presence check is needed.
            kind = narrowed.get("side_data_type")
            if kind is not None and str(kind) != "Display Matrix":
                continue
            yield narrowed.get("rotation")
    tags = _as_obj_dict(stream.get("tags"))
    if tags is not None:
        yield tags.get("rotate")


def _as_rotation(value: object) -> int | None:
    """A JSONB scalar → whole degrees in ``[0, 360)``, or ``None`` if unreadable."""
    if not isinstance(value, int | float | str):
        return None
    try:
        degrees = round(float(value))
    except TypeError, ValueError, OverflowError:
        # NaN raises ValueError and ±inf raises OverflowError under `round`.
        return None
    return abs(degrees) % 360


def _as_obj_dict(value: object) -> dict[str, object] | None:
    """Narrow an arbitrary JSONB value to a string-keyed dict, else ``None``.

    A bare ``isinstance(x, dict)`` narrows to an UNPARAMETERISED dict, which does
    not satisfy ``dict[str, object]`` — so every read off it, and every attempt to
    return it, is a type error. Doing the check once here keeps the callers honest
    without scattering casts. Keys are re-stringified rather than assumed: this is
    free-form JSONB decoded from Postgres, and nothing guarantees a producer used
    string keys beyond convention.
    """
    if not isinstance(value, dict):
        return None
    return {str(k): v for k, v in value.items()}


def _first_video_stream(probe: dict[str, object]) -> dict[str, object] | None:
    streams = probe.get("streams")
    if not isinstance(streams, list):
        return None
    for entry in streams:
        narrowed = _as_obj_dict(entry)
        if narrowed is not None and narrowed.get("codec_type") == "video":
            return narrowed
    return None


def _read_fps(stream: dict[str, object]) -> float:
    """Frame rate from ffprobe's RATIONAL string ("30000/1001" → 29.97).

    Reading these keys as plain numbers is the classic ffprobe trap: every real
    camera file reports a fraction, so a float parse returns 0 for all of them.
    """
    for key in _FPS_KEYS:
        raw = str(stream.get(key) or "")
        if "/" not in raw:
            continue
        num, _, den = raw.partition("/")
        numerator, denominator = _as_float(num), _as_float(den)
        if denominator > 0 and numerator > 0:
            return round(numerator / denominator, 3)
    return 0.0


def _as_float(value: object) -> float:
    try:
        return float(value)  # ty: ignore[invalid-argument-type]
    except TypeError, ValueError:
        return 0.0
