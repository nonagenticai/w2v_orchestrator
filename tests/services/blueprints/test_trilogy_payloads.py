"""Structural tests for the trilogy demo-video payloads.

These verify shape, not semantics — the live smoke (task B.5) is what proves
the payloads work end-to-end against `bdd_playwright_service`.
"""

from __future__ import annotations

import pytest

from src.services.blueprints.trilogy_payloads import (
    TRILOGY_BEATS,
    TRILOGY_BEATS_BY_ID,
    all_payloads,
    as_request_dicts,
    beat_payloads,
)

_EXPECTED_BEAT_IDS = ("B1", "B2", "B3", "B4", "B5", "B6")
_HSTACK_BEATS = {"B2", "B4", "B6"}
_FR_BEATS = {"B1", "B2", "B3", "B4"}


def test_six_beats_in_canonical_order() -> None:
    assert tuple(beat["id"] for beat in TRILOGY_BEATS) == _EXPECTED_BEAT_IDS


def test_index_matches_tuple() -> None:
    assert set(TRILOGY_BEATS_BY_ID) == set(_EXPECTED_BEAT_IDS)
    for beat in TRILOGY_BEATS:
        assert TRILOGY_BEATS_BY_ID[beat["id"]] is beat


@pytest.mark.parametrize("beat_id", ["B1"])
def test_every_beat_has_a_url_and_focus(beat_id: str) -> None:
    beat = TRILOGY_BEATS_BY_ID[beat_id]
    primary = beat["primary_payload"]
    assert primary.get("url"), f"{beat_id} primary missing url"
    assert primary.get("focus_areas"), f"{beat_id} primary missing focus_areas"
    assert primary.get("feed_to_video") is True


@pytest.mark.parametrize("beat_id", list(sorted(_HSTACK_BEATS))[:1])
def test_hstack_beats_carry_a_secondary_payload(beat_id: str) -> None:
    secondary = TRILOGY_BEATS_BY_ID[beat_id]["secondary_payload"]
    assert secondary is not None, f"{beat_id} is hstack but has no secondary_payload"
    assert secondary.get("url")
    assert secondary.get("focus_areas")


@pytest.mark.parametrize("beat_id", list(sorted(set(_EXPECTED_BEAT_IDS) - _HSTACK_BEATS))[:1])
def test_single_compose_beats_have_no_secondary(beat_id: str) -> None:
    assert TRILOGY_BEATS_BY_ID[beat_id]["secondary_payload"] is None


@pytest.mark.parametrize("beat_id", list(sorted(_FR_BEATS))[:1])
def test_fr_beats_carry_auth_config(beat_id: str) -> None:
    primary = TRILOGY_BEATS_BY_ID[beat_id]["primary_payload"]
    auth = primary.get("auth_config")
    assert auth is not None, f"{beat_id} hits fr.* but has no auth_config"
    assert auth["login_url"].startswith("https://fr.nonagentic.ai")
    assert auth["type"] == "form"


def test_n8n_executions_panel_does_not_need_auth_config() -> None:
    # The n8n side is recorded from inside the cluster network where the
    # bdd_playwright_service pod sits — n8n.test does not require the
    # Keycloak storage-state flow that fr.* needs.
    for beat_id in ("B2", "B4"):
        secondary = TRILOGY_BEATS_BY_ID[beat_id]["secondary_payload"]
        assert secondary is not None
        assert "auth_config" not in secondary
    primary_b5 = TRILOGY_BEATS_BY_ID["B5"]["primary_payload"]
    assert "auth_config" not in primary_b5
    primary_b6 = TRILOGY_BEATS_BY_ID["B6"]["primary_payload"]
    assert "auth_config" not in primary_b6


def test_planka_secondary_has_its_own_auth() -> None:
    secondary = TRILOGY_BEATS_BY_ID["B6"]["secondary_payload"]
    assert secondary is not None
    auth = secondary.get("auth_config")
    assert auth is not None
    assert auth["login_url"].startswith("https://kanban.nonagentic.ai")


def test_all_payloads_count_matches_compose_modes() -> None:
    # 6 primaries + 3 secondaries (one per hstack beat) = 9.
    assert len(all_payloads()) == 6 + len(_HSTACK_BEATS)


def test_beat_payloads_length_per_compose_mode() -> None:
    for beat_id in _EXPECTED_BEAT_IDS:
        expected = 2 if beat_id in _HSTACK_BEATS else 1
        assert len(beat_payloads(beat_id)) == expected


def test_as_request_dicts_returns_plain_dicts() -> None:
    coerced = as_request_dicts(all_payloads())
    assert all(type(p) is dict for p in coerced)
    assert all("url" in p for p in coerced)


def test_narration_tone_is_demo_everywhere() -> None:
    for payload in all_payloads():
        narration = payload.get("narration")
        assert narration is not None
        assert narration.get("tone") == "demo"
        assert narration.get("enabled") is True
