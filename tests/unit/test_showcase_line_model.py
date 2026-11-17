"""Contract for the shared authored clip↔beat model.

``AuthoredLine`` (src.models.showcase_line) is the single source of truth for the
authored-line field set. Two thin variants layer on it:
  - ``api.schemas.ShowcaseLine`` — the request-door model: strict (non-empty text,
    ``clip_anchor`` admits ``'end'`` + the content-aware ``'auto'``), feeds OpenAPI.
  - ``stages.demo_video.gen.DemoVideoLine`` — the stage-output twin: permissive
    (the stage legitimately builds a footage-only beat with empty text / a free-form
    ``clip_anchor``) and carries the ``StageOutput`` marker.

These tests lock that split and guard against the two variants drifting apart.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas import ShowcaseLine
from src.models import AuthoredLine, normalize_clip_key
from src.services.stages.base import StageOutput
from src.services.stages.demo_video.gen import DemoVideoLine
from src.services.stages.demo_video.line_ids import authored_line_id

# The authored field set, declared once on the base.
_FIELDS = {
    "id",
    "text",
    "seconds",
    "scenario_name",
    "gherkin",
    "speaker",
    "clip",
    "clip_anchor",
    "clip_lead_seconds",
    # Source-timeline OUT-point, the twin of `clip_lead_seconds`. Unlike the
    # lead, the compiler honours it under BOTH the "start" and "end" anchors,
    # so it is not gated on `clip_anchor`. Added when the compiler shipped it
    # (video_compiler_service #160); appended right after its sibling so the
    # OpenAPI property order below stays readable.
    "clip_out_seconds",
    "expected_text",
    "locked",
    # A11 PR-4 holistic ship-gate per-shot overrides. Author-set booleans that win
    # over the film-critic's computed findings; both default False → optional, and
    # `pin` is DISTINCT from the `pin_position` ordinal below.
    "pin",
    "continuity_exempt",
    "beat_role",
    "doctor_note",
    "hook_role",
    "pin_position",
    "time_lock",
    "min_seconds",
    "max_seconds",
    "vo_rate",
    # ORCH-4.4 (A18) claim_finish: the grounding verdict for this line, and the
    # source sentence that supports it. `claim_status="pinned"` is the author's
    # "I stand behind this claim" — claim_finish never rewrites such a line.
    "claim_status",
    "claim_source",
    # ORCH-1.4 voice-performance per-line overrides: all optional → author WINS
    # over the voice_director-computed PerformanceDirective. Appended, so field
    # order is unchanged and `required` stays ["text"].
    "provider",
    "delivery",
    "emphasis",
    "pause_before_ms",
    "rate",
    "ssml",
    "pronunciations",
    # The timeline solver's per-beat freeze pad — the window a FOOTAGE beat's `vo_rate`
    # was chosen against. APPENDED, never inserted beside `vo_rate`, so the OpenAPI
    # property order below (and the FE codegen that reads it) is unchanged.
    "freeze_pad_seconds",
    # The underspoken residual — appended right after its sibling for the same
    # OpenAPI/FE-codegen-order reason.
    "hold_seconds",
    # Author-pinned two-pane split (AuthoredSplit) — appended last, same reason.
    "split",
}


class TestSingleSourceOfTruth:
    def test_all_three_expose_the_same_fields(self) -> None:
        # The whole point of the merge: the field set can't drift because both
        # variants inherit it from AuthoredLine.
        assert set(AuthoredLine.model_fields) == _FIELDS
        assert set(ShowcaseLine.model_fields) == _FIELDS
        # …plus exactly the STAGE-OWNED measured-fit marker reconcile_timeline stamps
        # (narration_fit). Deliberately not on AuthoredLine: a measurement of a clip this
        # run synthesized must never be settable through the API's authored lines.
        assert set(DemoVideoLine.model_fields) == _FIELDS | {
            "fitted_seconds",
            "fitted_slack_seconds",
        }

    def test_variants_inherit_the_base(self) -> None:
        assert issubclass(ShowcaseLine, AuthoredLine)
        assert issubclass(DemoVideoLine, AuthoredLine)
        # The stage twin keeps the StageOutput marker (PR-6 annotations bind ids).
        assert issubclass(DemoVideoLine, StageOutput)


class TestAuthoredLineIsPermissive:
    def test_empty_text_and_free_form_clip_anchor_accepted(self) -> None:
        line = AuthoredLine.model_validate({"text": "", "clip_anchor": "start"})
        assert line.text == ""
        assert line.clip_anchor == "start"

    def test_all_defaults(self) -> None:
        line = AuthoredLine()
        assert line.id == "" and line.seconds == 0.0 and line.clip is None


class TestDemoVideoLineMirrorsTheStage:
    def test_permissive_like_the_base(self) -> None:
        # The stage builds these from raw brief / narration output; a footage-only
        # beat has empty text, and clip_anchor may be any string.
        line = DemoVideoLine(id="clip-1", text="", seconds=0.0, clip_anchor="middle")
        assert line.text == "" and line.clip_anchor == "middle"

    def test_id_is_required(self) -> None:
        with pytest.raises(ValidationError):
            DemoVideoLine.model_validate({"text": "hi", "seconds": 1.0})


class TestShowcaseLineKeepsDoorStrictness:
    def test_empty_text_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ShowcaseLine.model_validate({"text": ""})

    def test_negative_seconds_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ShowcaseLine.model_validate({"text": "hi", "seconds": -1.0})

    def test_clip_anchor_restricted_to_end_or_auto(self) -> None:
        assert (
            ShowcaseLine.model_validate({"text": "hi", "clip_anchor": "end"}).clip_anchor == "end"
        )
        # C4: "auto" is now admitted at the door (resolved in build_shots); "start"
        # and any other free-form value are still rejected here.
        assert (
            ShowcaseLine.model_validate({"text": "hi", "clip_anchor": "auto"}).clip_anchor == "auto"
        )
        for bad in ("start", "middle"):
            with pytest.raises(ValidationError):
                ShowcaseLine.model_validate({"text": "hi", "clip_anchor": bad})

    def test_clip_at_prefix_normalized(self) -> None:
        # A door-submitted '@clip-1' is normalized to the 'clip-1' join key.
        assert ShowcaseLine.model_validate({"text": "hi", "clip": "@clip-1"}).clip == "clip-1"

    def test_openapi_schema_shape_preserved(self) -> None:
        # Guards the FE contract: text stays required, clip_anchor stays an
        # enum of ['end', 'auto'] in the generated OpenAPI (C4 widened it from
        # the single-member ['end']).
        schema = ShowcaseLine.model_json_schema()
        assert schema["required"] == ["text"]
        assert schema["properties"]["clip_anchor"]["anyOf"][0]["enum"] == ["end", "auto"]
        # Field order feeds OpenAPI → FE codegen; keep it stable.
        assert list(schema["properties"]) == [
            "id",
            "text",
            "seconds",
            "scenario_name",
            "gherkin",
            "speaker",
            "clip",
            "clip_anchor",
            "clip_lead_seconds",
            # Declared immediately after its sibling lead, so the FE codegen
            # order stays grouped by concern (clip framing together).
            "clip_out_seconds",
            "expected_text",
            "locked",
            # A11 PR-4: per-shot ship-gate overrides, inserted after `locked`. Both
            # optional so `required` stays ["text"]; distinct from `pin_position`.
            "pin",
            "continuity_exempt",
            "beat_role",
            "doctor_note",
            "hook_role",
            "pin_position",
            "time_lock",
            "min_seconds",
            "max_seconds",
            "vo_rate",
            # ORCH-4.4 (A18): appended, so existing FE codegen field order is
            # unchanged and both are optional — `required` stays ["text"].
            "claim_status",
            "claim_source",
            # ORCH-1.4: voice-performance overrides, appended → field order stable,
            # all optional so `required` stays ["text"].
            "provider",
            "delivery",
            "emphasis",
            "pause_before_ms",
            "rate",
            "ssml",
            "pronunciations",
            # Timeline solver output, APPENDED for the same reason as every block above
            # it: field order feeds OpenAPI → FE codegen, so a new field goes on the end
            # rather than beside its conceptual sibling `vo_rate`.
            "freeze_pad_seconds",
            # Appended right after its sibling for the same OpenAPI/FE-codegen-order
            # reason; see AuthoredLine.hold_seconds for what it carries.
            "hold_seconds",
            # Author-pinned split layout — appended last for the same reason.
            "split",
        ]


class TestTimelineFieldsRoundTrip:
    """A22/C6: the four per-line timeline slots. All optional — absent ⇒ ``None``
    (``reconcile_timeline`` reads ``None`` as "compute it", a present value as
    AUTHORED, which always wins)."""

    def test_default_none_on_every_variant(self) -> None:
        for line in (
            AuthoredLine(),
            DemoVideoLine(id="clip-1"),
            ShowcaseLine.model_validate({"text": "hi"}),
        ):
            assert line.time_lock is None
            assert line.min_seconds is None
            assert line.max_seconds is None
            assert line.vo_rate is None

    def test_authored_values_round_trip(self) -> None:
        raw = {
            "text": "hi",
            "time_lock": True,
            "min_seconds": 2.0,
            "max_seconds": 6.5,
            "vo_rate": 1.15,
        }
        line = ShowcaseLine.model_validate(raw)
        assert line.time_lock is True
        assert (line.min_seconds, line.max_seconds, line.vo_rate) == (2.0, 6.5, 1.15)
        # JSON round-trip (pure-Pydantic, no DB migration) preserves them.
        assert ShowcaseLine.model_validate(line.model_dump()) == line
        # The stage twin carries them too — the spine survives gen → reconcile.
        assert DemoVideoLine.model_validate({"id": "clip-1", **raw}).max_seconds == 6.5

    def test_stay_optional_in_the_openapi_contract(self) -> None:
        # Regression guard: a non-optional new field would flip schema["required"].
        assert ShowcaseLine.model_json_schema()["required"] == ["text"]

    @pytest.mark.parametrize(
        "field,value",
        [("min_seconds", -1.0)],
    )
    def test_door_rejects_out_of_range(self, field: str, value: float) -> None:
        # ShowcaseLine TIGHTENS the inherited fields (ge=0 / gt=0) at the request door.
        with pytest.raises(ValidationError):
            ShowcaseLine.model_validate({"text": "hi", field: value})

    def test_base_stays_permissive(self) -> None:
        # The stage layer builds looser lines; only the door is strict.
        assert AuthoredLine(min_seconds=-1.0).min_seconds == -1.0


class TestVoicePerformanceFieldsRoundTrip:
    """ORCH-1.4: the seven per-line voice-performance overrides. All optional —
    absent ⇒ ``None`` (voice_director computes the PerformanceDirective); a present
    value is AUTHORED and always wins."""

    def test_default_none_on_every_variant(self) -> None:
        for line in (
            AuthoredLine(),
            DemoVideoLine(id="clip-1"),
            ShowcaseLine.model_validate({"text": "hi"}),
        ):
            assert line.provider is None
            assert line.delivery is None
            assert line.emphasis is None
            assert line.pause_before_ms is None
            assert line.rate is None
            assert line.ssml is None
            assert line.pronunciations is None

    def test_authored_values_round_trip(self) -> None:
        raw = {
            "text": "hi",
            "provider": "elevenlabs",
            "delivery": "excited",
            "emphasis": [0, 3],
            "pause_before_ms": 250,
            "rate": 1.1,
            "ssml": "<speak>hi</speak>",
            "pronunciations": {"w2v": "web to video"},
        }
        line = ShowcaseLine.model_validate(raw)
        assert line.provider == "elevenlabs"
        assert line.delivery == "excited"
        assert line.emphasis == [0, 3]
        assert line.pause_before_ms == 250
        assert line.rate == 1.1
        assert line.ssml == "<speak>hi</speak>"
        assert line.pronunciations == {"w2v": "web to video"}
        # JSON round-trip (pure-Pydantic, no DB migration) preserves them.
        assert ShowcaseLine.model_validate(line.model_dump()) == line
        # The stage twin carries them too — the spine survives gen → narrate.
        assert DemoVideoLine.model_validate({"id": "clip-1", **raw}).delivery == "excited"

    def test_stay_optional_in_the_openapi_contract(self) -> None:
        # Regression guard: a non-optional new field would flip schema["required"].
        assert ShowcaseLine.model_json_schema()["required"] == ["text"]


class TestHoldSecondsSurvivesModelConstruction:
    """The underspoken-residual field (`time_budget.Allotment.hold_seconds`'s
    consumer-side twin). It is declared here so a stamped value SURVIVES model
    construction — before this field existed, pydantic's default
    `extra="ignore"` silently DROPPED it, which is the whole reason the hold
    could never reach a compositor shot."""

    def test_default_none_on_every_variant(self) -> None:
        for line in (
            AuthoredLine(),
            DemoVideoLine(id="clip-1"),
            ShowcaseLine.model_validate({"text": "hi"}),
        ):
            assert line.hold_seconds is None

    def test_stamped_value_survives_and_round_trips(self) -> None:
        line = AuthoredLine.model_validate({"text": "hi", "hold_seconds": 0.83})
        assert line.hold_seconds == 0.83
        # JSON round-trip (pure-Pydantic, no DB migration) preserves it.
        assert AuthoredLine.model_validate(line.model_dump()).hold_seconds == 0.83
        # The stage twin carries it too — the spine survives gen → reconcile.
        assert (
            DemoVideoLine.model_validate({"id": "clip-1", "hold_seconds": 0.83}).hold_seconds
            == 0.83
        )

    def test_stay_optional_in_the_openapi_contract(self) -> None:
        assert ShowcaseLine.model_json_schema()["required"] == ["text"]


class TestClipKeyNormalization:
    """A ``@clip-<n>`` tag (what a user sees in the .feature) is accepted on input and
    normalized to the ``clip-<n>`` key FTR records footage under, so the join holds."""

    def test_helper_strips_one_leading_at_and_whitespace(self) -> None:
        assert normalize_clip_key("@clip-1") == "clip-1"
        assert normalize_clip_key("  @clip-2  ") == "clip-2"
        assert normalize_clip_key("clip-3") == "clip-3"  # already normal → no-op
        assert normalize_clip_key("@@x") == "@x"  # only a single leading '@'

    def test_clip_field_normalized_on_every_variant(self) -> None:
        assert AuthoredLine(clip="@clip-1").clip == "clip-1"
        assert DemoVideoLine(id="clip-1", clip="@clip-2").clip == "clip-2"
        assert ShowcaseLine.model_validate({"text": "hi", "clip": "@clip-3"}).clip == "clip-3"

    def test_clip_none_and_empty_pass_through(self) -> None:
        assert AuthoredLine(clip=None).clip is None
        assert AuthoredLine(clip="").clip == ""

    def test_authored_line_id_normalizes_and_defaults(self) -> None:
        # Shared by gen + showcase_slides, so both derive the same normalized id.
        assert authored_line_id({"id": "@clip-1"}, 0) == "clip-1"
        assert authored_line_id({"id": "clip-1"}, 0) == "clip-1"
        assert authored_line_id({}, 2) == "clip-3"  # default, never '@'-prefixed
