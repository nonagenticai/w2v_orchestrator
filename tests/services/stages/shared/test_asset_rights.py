"""Tests for the deterministic asset-rights ledger (ORCH-0.7).

Pure — no DB, no client, no ctx. Mirrors ``test_finish.py``: everything runs
against fixed in-memory config snapshots, and identical inputs must yield
byte-identical outputs. Presence-is-opt-in: absent brand+legal ⇒ ``[]``.
"""

from __future__ import annotations

from src.services.stages.shared.asset_rights import (
    AssetRightsRow,
    build_asset_rights_ledger,
)

_SNAPSHOT = {
    "brand": {"name": "Acme", "logo_url": "https://cdn.example/acme-logo.png"},
    "legal": {
        "trademarks": ["Acme(R)", "SuperWidget"],
        "defensibility": "patent pending, US 12/345,678",
    },
}


def test_full_snapshot_yields_deterministic_rows() -> None:
    rows = build_asset_rights_ledger(_SNAPSHOT)
    assert [r.asset_kind for r in rows] == [
        "name",
        "logo",
        "trademark",
        "trademark",
        "defensibility",
    ]
    # Brand rows source from brand_context; legal rows from the legal block.
    name, logo, tm1, tm2, defens = rows
    assert name == AssetRightsRow(asset_kind="name", value="Acme")
    assert logo.value == "https://cdn.example/acme-logo.png"
    assert logo.source == "brand_context"
    assert logo.defensible is False
    assert tm1.value == "Acme(R)" and tm1.source == "legal" and tm1.defensible is True
    assert tm2.value == "SuperWidget"
    assert defens.asset_kind == "defensibility" and defens.defensible is True
    # Deterministic: same input → identical rows.
    assert build_asset_rights_ledger(_SNAPSHOT) == rows


def test_absent_legal_drops_trademark_and_defensibility_rows() -> None:
    snapshot = {"brand": {"name": "Acme", "logo_url": "https://cdn.example/acme-logo.png"}}
    rows = build_asset_rights_ledger(snapshot)
    assert [r.asset_kind for r in rows] == ["name", "logo"]
    # No legal block ⇒ no legally-defensible rows at all.
    assert all(r.defensible is False for r in rows)


def test_empty_or_absent_snapshot_yields_empty_ledger() -> None:
    assert build_asset_rights_ledger({}) == []
    assert build_asset_rights_ledger({"brand": {}, "legal": {}}) == []
    assert build_asset_rights_ledger({"voice": {"descriptor": "neutral"}}) == []


def test_legal_only_snapshot_yields_legal_rows() -> None:
    snapshot = {"legal": {"trademarks": ["OnlyMark"], "defensibility": "trade secret"}}
    rows = build_asset_rights_ledger(snapshot)
    assert [r.asset_kind for r in rows] == ["trademark", "defensibility"]
