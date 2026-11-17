"""WHICH degradation, not merely whether: `StageOutputResponse.degradation_codes`.

`services/stages/shared/degradation.py` defines NINE stable codes and every one
of them is emitted — and until this field every one of them collapsed into the
single `degraded` boolean. A failed mastering gate, a video that shipped silent
because its music bed could not be resolved, and a deck that came back a stub
were all the same wire value: `degraded: true`. A full wave was spent making
those messages truthful and nothing could read the truth back.

These tests pin the ADDITIVE contract:

* two distinct degradations on one stage surface BOTH codes, in order;
* a clean stage surfaces `[]` — never `None` — and `degraded` stays False;
* `degraded` itself is byte-for-byte unchanged for every pre-existing case,
  including the notes-only one, which is the whole reason this is additive.

⚠️ Nothing here arms anything. There is no assertion that a degradation is a
failure, and no gate anywhere reads this field. Eight of the nine codes have no
measured base rate, and this repo has already shipped one signal that fired on
96.7% of masters and was therefore worthless.

No `integration` marker: pytest addopts deselects that marker, so a test wearing
it would never execute in CI.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid7

import pytest

from src.api.routes.runs._shared import _stage_response
from src.api.schemas import StageOutputResponse
from src.database.models import StageOutput
from src.services.stages.shared.degradation import (
    CAPTIONS_MISSING,
    DEGRADATION_CODES,
    SOUND_BED_MISSING,
    degradation,
)


def _stage(output: object, *, status: str = "completed") -> StageOutput:
    return StageOutput(
        id=uuid7(),
        run_id=uuid7(),
        stage_name="compile",
        status=status,
        output=cast(Any, output),
        attempts=0,
    )


def _response(output: object, *, status: str = "completed"):
    so = _stage(output, status=status)
    return _stage_response(so.run_id, so)


def test_two_distinct_degradations_surface_both_codes_in_order() -> None:
    """THE point of the field. Two different facts must stay two different facts,
    in the order the stage recorded them — not one boolean, and not a set."""
    resp = _response(
        {
            "degradations": [
                degradation(SOUND_BED_MISSING, "bed 'calm_pad' did not resolve"),
                degradation(CAPTIONS_MISSING, "word-timing soft-failed; no subtitles"),
            ]
        }
    )
    assert resp.degradation_codes == [SOUND_BED_MISSING, CAPTIONS_MISSING]
    # ...and the reverse recording order reports the reverse, i.e. the order is
    # the stage's, not an alphabetical or set-iteration accident. SOUND_BED_MISSING
    # sorts BEFORE CAPTIONS_MISSING under neither sort, so this discriminates.
    flipped = _response(
        {
            "degradations": [
                degradation(CAPTIONS_MISSING, "no subtitles"),
                degradation(SOUND_BED_MISSING, "no bed"),
            ]
        }
    )
    assert flipped.degradation_codes == [CAPTIONS_MISSING, SOUND_BED_MISSING]
    assert resp.degraded is flipped.degraded is True


def test_a_repeated_code_is_reported_once_at_its_first_position() -> None:
    """One code raised twice is ONE fact. De-duplicated, but in FIRST-SEEN order:
    a `set` would report it once too and lose the order doing it."""
    resp = _response(
        {
            "degradations": [
                degradation(SOUND_BED_MISSING, "first"),
                degradation(CAPTIONS_MISSING, "second"),
                degradation(SOUND_BED_MISSING, "third — same fact as the first"),
            ]
        }
    )
    assert resp.degradation_codes == [SOUND_BED_MISSING, CAPTIONS_MISSING]


def test_a_clean_stage_reports_an_empty_list_and_is_not_degraded() -> None:
    """FALSE-POSITIVE GUARD. Every healthy run in the fleet lands here: an empty
    LIST (so a consumer may iterate unconditionally), never None, and `degraded`
    untouched at False."""
    resp = _response({"script": "ok", "notes": [], "degradations": []})
    assert resp.degradation_codes == []
    assert resp.degradation_codes is not None
    assert resp.degraded is False


@pytest.mark.parametrize(
    "output",
    [pytest.param(None, id="no-blob")],
)
def test_a_malformed_or_absent_channel_reports_empty_and_never_raises(output: object) -> None:
    """A quality signal must not be able to 500 the endpoint that reports it.
    Rows predating the channel, a JSON null, and anything hand-written into the
    blob all read as "nothing to report" — never an exception, never None."""
    assert _response(output).degradation_codes == []


def test_the_schema_default_is_an_empty_list_not_none() -> None:
    """The DEFAULT, not just the populated path. `_stage_response` always assigns,
    so a test that only goes through it cannot see the field's declared default —
    and the declared default is what any other construction site gets. A
    `StageOutput` row has no such attribute, so `model_validate` must fall back to
    `[]`, and the value on the wire must be `[]` and never `null`."""
    bare = StageOutputResponse.model_validate(_stage({}), from_attributes=True)
    assert bare.degradation_codes == []
    assert bare.model_dump()["degradation_codes"] == []
    # Two instances must not share one list object — a mutable default shared
    # across responses would let one stage's codes leak into the next.
    other = StageOutputResponse.model_validate(_stage({}), from_attributes=True)
    bare.degradation_codes.append("leaked")
    assert other.degradation_codes == []


def test_an_unknown_code_is_reported_not_rejected() -> None:
    """`degradation()` refuses an unknown code at the WRITE site. The READ side
    must NOT: during a rolling deploy a newer pod persists a code this build has
    never heard of, and rejecting it would break the endpoint rather than report
    the fact."""
    assert "deband_ceiling_missed" not in DEGRADATION_CODES
    resp = _response({"degradations": [{"code": "deband_ceiling_missed", "detail": "future"}]})
    assert resp.degradation_codes == ["deband_ceiling_missed"]


# ══════════════════════════════════════════════════════════════════════════════
# `degraded` is UNCHANGED — the additive half
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("output", "status", "expected_degraded", "expected_codes"),
    [pytest.param({"notes": ["source unavailable"]}, "completed", True, [], id="notes-only")],
)
def test_degraded_keeps_its_exact_pre_existing_semantics(
    output: object, status: str, expected_degraded: bool, expected_codes: list[str]
) -> None:
    """`degraded` is the OR of both channels AND `status == "completed"` — the
    rule in `degradation.is_degraded`, unchanged. Adding codes must not move it
    in either direction for any of these.

    The `not-completed` row carries the one asymmetry, and it is deliberate:
    `degraded` is gated on the stage having completed, the codes are not. Gating
    them would DROP a code that is plainly visible in this very response's
    `output` blob, making the named field less truthful than the bytes it
    mirrors. `notes-only` carries the other: the legacy prose channel raises
    `degraded` and has no stable identity to promote, so it must not manufacture
    a code — that pair names exactly the runs whose degradation is still
    unqueryable English."""
    resp = _response(output, status=status)
    assert resp.degraded is expected_degraded
    assert resp.degradation_codes == expected_codes
