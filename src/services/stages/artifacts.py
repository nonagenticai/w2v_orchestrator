"""Artifact-reference validation.

Split out of ``base.py``; re-exported there for import-path stability.

* ``ArtifactUrl`` — a Pydantic field type that rejects empty/``"None"``/malformed
  references at the output boundary, so a stage can't persist a green
  ``completed`` row pointing at nothing.
* ``validate_compile_result`` — names the downstream cause before unpacking a
  compile/render response into the output model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Annotated, Any
from urllib.parse import urlparse

from pydantic import AfterValidator

from src.services.stages.errors import StageDownstreamResponseError

__all__ = ["ArtifactUrl", "validate_compile_result"]


def validate_compile_result(result: Mapping[str, Any], *, stage: str) -> None:
    """Reject a compile/render response that did not produce a usable artifact.

    Complements the ``ArtifactUrl`` output-field gate by checking the response
    *before* it is unpacked into the output model, so the error names the
    downstream cause (``stage: compile returned …``) rather than a bare
    Pydantic field error. Enforces: a truthy, non-``"None"`` ``artifact_url``
    and a finite, strictly-positive ``duration_seconds``.
    """
    url = result.get("artifact_url")
    if not url or str(url).strip().lower() in {"none", "null"}:
        raise StageDownstreamResponseError(
            f"{stage}: compile returned no usable artifact_url (got {url!r})"
        )
    raw = result.get("duration_seconds")
    if raw is None:
        raise StageDownstreamResponseError(f"{stage}: compile returned no duration_seconds")
    try:
        duration = float(raw)
    except TypeError, ValueError:
        raise StageDownstreamResponseError(
            f"{stage}: compile returned non-numeric duration_seconds {raw!r}"
        ) from None
    if not math.isfinite(duration) or duration <= 0:
        raise StageDownstreamResponseError(
            f"{stage}: compile returned non-positive duration_seconds {duration!r}"
        )


def _require_artifact_url(value: str) -> str:
    """Reject empty / placeholder / malformed artifact references at the boundary.

    A stage that returns ``artifact_url=""`` or ``"None"`` (the ``str(None)``
    trap) or a bare word has NOT produced a usable artifact. Raising here makes
    Pydantic reject the output, so the DAG executor marks the stage ``failed``
    (with this message in ``Run.error``) instead of persisting a green
    ``completed`` row that points at nothing.

    Any URI *scheme* is accepted (``http``/``https``/``s3``/``gs``/``file``/…)
    as long as there is a scheme and a host *or* a path: the artifact proxy
    (PR #114) rewrites internal schemes for playback downstream, so the stage's
    job is only to refuse a degenerate reference, not to dictate the scheme. The
    path branch admits the local-backend ``file:///run/clip.wav`` fallback
    (``store_tts_clip`` yields it when MinIO is unconfigured), whose ``netloc``
    is empty but which is still a real reference; ``is_fetchable_url`` filters it
    at compile time. Validity of the *content* behind the URL (reachable,
    non-trivial duration) is enforced separately at the compile-response and
    run-finalizer gates.
    """
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"none", "null"}:
        raise ValueError(f"not a usable artifact URL: {value!r}")
    parsed = urlparse(cleaned)
    if not parsed.scheme or not (parsed.netloc or parsed.path):
        raise ValueError(f"not a usable artifact URL: {value!r}")
    return value


ArtifactUrl = Annotated[str, AfterValidator(_require_artifact_url)]
"""A ``str`` field that must be a non-empty, well-formed artifact reference.

Accepts any URI scheme with a host or path (http/https/s3/gs/file/…); rejects
only degenerate values (empty, ``"None"``/``"null"``, bare words). Stays a plain
``str`` in ``model_dump()`` (unlike ``pydantic.HttpUrl``) so the JSONB / SSE /
SPA contract is unchanged; only the *validity* gate is added.
"""
