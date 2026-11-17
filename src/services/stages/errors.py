"""Stage error taxonomy.

Split out of ``base.py``; re-exported there for import-path stability. The
boundary the DAG executor cares about:

* ``StageSkippedError`` → ``skipped`` (terminal-but-fine; opt-in stage not requested)
* ``StagePreconditionError`` / ``StageDownstreamResponseError`` → ``failed`` (fail loud)
"""

from __future__ import annotations

from w2v_common.stages import StageError

__all__ = [
    "StageDownstreamResponseError",
    "StageError",
    "StagePreconditionError",
    "StageSkippedError",
]


class StagePreconditionError(StageError):
    """A REQUIRED input for the stage is absent → the run must fail loud.

    The inputs needed to even attempt the stage do not exist (e.g. no shots to
    compile). This is a failure, not an opt-out: an empty compile that "succeeds"
    into a 1-frame placeholder is the exact silent-degradation lie this cleanup
    removes. Raised by handlers; the DAG executor lets it fall through to the
    failure path and surfaces ``reason`` in ``Run.error``.

    Do NOT use this for "the run didn't request this optional stage" — that is
    ``StageSkippedError`` (terminal-but-fine), which the executor maps to ``skipped``.
    """

    def __init__(self, reason: str, *, missing: list[str] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.missing = missing or []


class StageSkippedError(StageError):
    """An OPT-IN stage whose activation the run did not request.

    Distinct from ``StagePreconditionError`` (a required input is missing → the
    run fails): this stage is legitimately optional and the run simply didn't ask
    for it (e.g. no ``ai_image_gen`` config block). The DAG executor catches this
    and marks the stage ``skipped`` — a terminal state that counts toward run
    completion — rather than ``failed``. Returning an empty, success-shaped
    output instead was the old silent-degradation lie (a run "completed" holding
    zero images, then a text-only placeholder video).
    """

    def __init__(self, reason: str, *, missing: list[str] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.missing = missing or []


class StageDownstreamResponseError(StageError):
    """A downstream service returned a 2xx whose body is not actually usable.

    The compositor returns ``200`` even for a degenerate render (empty/zero-
    duration artifact); trusting that 200 is how a run reported ``completed``
    while holding nothing. Raise this at the stage boundary so the run fails
    loud with the reason instead.
    """
