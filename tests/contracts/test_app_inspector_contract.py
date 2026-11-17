"""Contract guard for the app_inspector ``/result`` wire model.

This is the consumer half of the app_review dead-signal fix. The whole bug class
is "a consumer reads fields the server never sends, hidden behind mocks that
agree only with each other" — so a mock-only test suite cannot catch it. This
test pins the orchestrator's ``InspectionResult`` model against a VENDORED copy
of the server's real JSON Schema and proves the extractor turns a server-shaped
payload into non-empty facts.

Drift is plain set algebra (via ``tests/contracts/schema_walk.py``):
- ``mine - server`` → we read a field the server does not declare (the phantom
  ``title/tagline/pricing`` bug). This test FAILS on it.
- ``server - mine`` is EXPECTED and fine: we intentionally consume a subset.

Provenance: ``tests/contracts/inspection_result.schema.json`` is a verbatim copy
of ``app_inspector_service/contracts/inspection_result.schema.json`` at server
commit ``dcd9cd2a`` (``InspectionResultResponse.model_json_schema()``). Refresh it
by re-copying that artifact::

    git -C ../app_inspector_service show origin/main:contracts/inspection_result.schema.json \
      > tests/contracts/inspection_result.schema.json

the server's ``test_contract_artifact_fresh`` guarantees the server copy tracks
its model.

⚠️ A STALE vendored copy CANNOT announce itself here. Drift is one-directional by
design: only ``mine - server`` fails, and ``server - mine`` (the server grew a
field) is the EXPECTED case for a deliberate subset — so this suite stayed green
for the whole time the copy predated app_inspector #92 and its
``screenshot_status`` / ``artifacts_degraded`` / ``screenshots_captured``
additions. What makes the re-vendor load-bearing rather than cosmetic is that
the consumer now READS those three fields: against the old copy they are
``mine - server`` phantoms and
``test_consumed_model_is_a_subset_of_the_server_schema`` goes red.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.clients.schemas.app_inspector import InspectionResult
from src.services.stages.app_review.inspect import AppInspectStage
from tests.contracts.schema_walk import payload_leaves, schema_leaves

_VENDORED_SERVER_SCHEMA = Path(__file__).with_name("inspection_result.schema.json")

# A server-shaped /result body (InspectionResultResponse). Two stub screens, the
# first carrying a URL-shaped screenshot, plus the inferred purpose the server
# now surfaces (app_inspector_service #60).
_SCREENS: list[dict[str, object]] = [
    {
        "screen_id": "s1",
        "screen_type": "home",
        "activity_name": "MainActivity",
        "text_content": "Your tasks for today",
        "headings": ["Today", "Upcoming"],
        "accessibility_score": 90,
        "screenshot_path": "https://minio.test/app-inspector/s1.png",
        "screenshot_status": "captured",
        "has_forms": False,
        "has_media": True,
    },
    {
        "screen_id": "s2",
        "screen_type": "settings",
        "activity_name": "SettingsActivity",
        "text_content": "Account and notifications",
        "headings": ["Account"],
        "accessibility_score": 86,
        "screenshot_path": "/var/data/screens/s2.png",  # LOCAL path — not a URL
        "screenshot_status": "captured",
        "has_forms": True,
        "has_media": False,
    },
]

_RESULT_FIXTURE: dict[str, object] = {
    "job_id": "job-abc",
    "status": "completed",
    "platform": "android",
    "app_path": "/tmp/app.apk",
    "screens_discovered": 2,
    "overall_accessibility_score": 88,
    "estimated_purpose": "A task-management app",
    "navigation_graph": [
        {"from_screen": "home", "to_screen": "settings", "action": "tap"},
    ],
    "screenshots_captured": 2,
    "artifacts_degraded": False,
    "screens": _SCREENS,
}


def _server_leaves() -> frozenset[str]:
    schema = json.loads(_VENDORED_SERVER_SCHEMA.read_text(encoding="utf-8"))
    return schema_leaves(schema)


def test_consumed_model_is_a_subset_of_the_server_schema() -> None:
    """Every field InspectionResult declares must EXIST on the server model.

    This is the guard that would have caught the original bug: the old consumer
    read title/tagline/features/pricing_lines/screenshot_url, none of which the
    server declares — those leaves would not be a subset here.
    """
    mine = schema_leaves(InspectionResult.model_json_schema())
    server = _server_leaves()
    phantom = mine - server
    assert not phantom, (
        "InspectionResult reads fields the app_inspector /result model does NOT "
        f"declare (phantom/renamed/removed server-side): {sorted(phantom)}. "
        "Re-vendor tests/contracts/inspection_result.schema.json from the server "
        "and reconcile the model."
    )


def test_fixture_is_server_shaped() -> None:
    """The fixture must not invent fields the server does not declare either."""
    fixture_leaves = payload_leaves(_RESULT_FIXTURE).leaves
    server = _server_leaves()
    invented = fixture_leaves - server
    assert not invented, f"fixture invents non-server fields: {sorted(invented)}"


def test_model_round_trips_the_server_payload() -> None:
    result = InspectionResult.model_validate(_RESULT_FIXTURE)
    assert result.estimated_purpose == "A task-management app"
    assert result.screens_discovered == 2
    assert len(result.screens) == 2
    assert result.screens[0].screenshot_path == "https://minio.test/app-inspector/s1.png"


def test_extractor_produces_non_empty_facts_including_purpose() -> None:
    result = InspectionResult.model_validate(_RESULT_FIXTURE)
    facts, screenshot_url = AppInspectStage._facts_from_inspection(result, "app://target")

    kinds = {f.kind for f in facts}
    assert facts, "extractor produced ZERO facts from a populated inspection"
    assert "purpose" in kinds, "the inferred app purpose was not surfaced as a fact"
    assert "screen" in kinds
    assert "accessibility" in kinds
    assert all(f.id and f.text for f in facts)
    # First screen has a URL screenshot → surfaced; the second's local path must NOT leak.
    assert screenshot_url == "https://minio.test/app-inspector/s1.png"
    assert sum(1 for f in facts if f.kind == "screenshot") == 1


def test_local_only_screenshot_is_never_surfaced() -> None:
    """A result whose ONLY screenshot is a pod-local path yields no screenshot."""
    local_only = {
        **_RESULT_FIXTURE,
        "screens": [{**s, "screenshot_path": "/var/data/screens/local.png"} for s in _SCREENS],
    }
    result = InspectionResult.model_validate(local_only)
    facts, screenshot_url = AppInspectStage._facts_from_inspection(result, "app://target")
    assert screenshot_url == ""
    assert not any(f.kind == "screenshot" for f in facts)
    assert any(f.kind == "screen" for f in facts)  # the screens themselves still surface


def test_uppercase_url_scheme_is_still_a_valid_screenshot() -> None:
    """A valid but upper/mixed-case scheme must NOT be silently dropped."""
    payload = {
        **_RESULT_FIXTURE,
        "screens": [{**_SCREENS[0], "screenshot_path": "HTTPS://minio.test/x.png"}],
    }
    result = InspectionResult.model_validate(payload)
    _, screenshot_url = AppInspectStage._facts_from_inspection(result, "app://target")
    assert screenshot_url == "HTTPS://minio.test/x.png"


# --------------------------------------------------------------------------- #
# app_inspector #92 — a failed capture must be VISIBLE, not silently successful
# --------------------------------------------------------------------------- #
def test_vendored_schema_declares_the_degradation_fields() -> None:
    """Fence the re-vendor itself.

    Everything below asserts on BEHAVIOUR, which a hand-written consumer default
    would satisfy even against the stale artifact. This one asserts the three
    fields are DECLARED BY THE SERVER, so re-copying the artifact is what makes
    the gate real rather than the consumer inventing a field the wire never carries.
    """
    server = _server_leaves()
    for leaf in ("screens.screenshot_status", "screenshots_captured", "artifacts_degraded"):
        assert leaf in server, (
            f"the vendored app_inspector schema does not declare {leaf!r} — it predates "
            "server commit dcd9cd2a. Re-vendor it (see this module's docstring)."
        )


def test_a_failed_capture_is_not_surfaced_as_a_screenshot() -> None:
    """The producer bug, from the consumer side.

    A FAILED capture returned a non-empty, perfectly URL-shaped ``screenshot_path``
    to a file that was never written. ``_url_shaped`` cannot tell that apart from a
    real capture — only ``screenshot_status`` can. If it rides through,
    ``app_review_record``'s "no screenshot_url" note never fires (the url is
    non-empty!) and the compile ships an all-text-card video on a clean green run.
    """
    payload = {
        **_RESULT_FIXTURE,
        "screenshots_captured": 0,
        "artifacts_degraded": True,
        "screens": [
            {
                **_SCREENS[0],
                # URL-shaped and non-empty — the lie the server used to tell.
                "screenshot_path": "https://minio.test/app-inspector/never-written.png",
                "screenshot_status": "failed",
            },
        ],
    }
    result = InspectionResult.model_validate(payload)
    assert result.artifacts_degraded is True, "the derived degrade flag did not reach the consumer"

    facts, screenshot_url = AppInspectStage._facts_from_inspection(result, "app://target")

    assert screenshot_url == "", (
        "a screenshot the server marked status='failed' was surfaced as a usable "
        f"visual: {screenshot_url!r}. The path is URL-shaped but names a file that "
        "was never written."
    )
    assert not any(f.kind == "screenshot" for f in facts)
    # The failure is not merely suppressed — it is SAID, in the text the critique
    # stage reads. Suppressing without saying is how the signal went dead before.
    screen_text = " ".join(f.text for f in facts if f.kind == "screen")
    assert "screenshot capture FAILED" in screen_text


def test_an_unknown_status_still_surfaces_the_screenshot() -> None:
    """Fence the suppression above against over-reach.

    ``unknown`` is the server's DEFAULT and means UNMEASURED, and a pre-#92 server
    omits the field entirely. Treating either as a failure would delete every
    screenshot from every inspection — strictly worse than the bug being fixed.
    """
    for screen in ({**_SCREENS[0], "screenshot_status": "unknown"}, _omit_status(_SCREENS[0])):
        result = InspectionResult.model_validate({**_RESULT_FIXTURE, "screens": [screen]})
        _, screenshot_url = AppInspectStage._facts_from_inspection(result, "app://target")
        assert screenshot_url == "https://minio.test/app-inspector/s1.png", (
            f"an unmeasured capture was wrongly suppressed (screen={screen})"
        )


def _omit_status(screen: dict[str, object]) -> dict[str, object]:
    return {k: v for k, v in screen.items() if k != "screenshot_status"}
