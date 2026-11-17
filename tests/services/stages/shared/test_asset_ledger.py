"""A18 phase 2 — asset ledger, trademark scan, defensibility manifest.

Pure: no DB, no ctx, no client. Everything runs against fixed in-memory config
snapshots and identical inputs must yield byte-identical output.

Pins AC-7 (100% ledger coverage, every row licensed, unverified assets counted),
AC-9 (third-party marks listed, first-party excluded) and the determinism half of
AC-10 (identical resolved inputs ⇒ identical manifest hash, in any key order).
"""

from __future__ import annotations

from typing import Any

from src.services.stages.shared.asset_rights import (
    CATALOG_LICENSE,
    DECLARED_LICENSE,
    UNVERIFIED_LICENSE,
    build_asset_ledger,
    defensibility_manifest,
    trademark_flags,
)
from src.services.stages.sound_catalog import DEFAULT_AMBIENT_ID, DEFAULT_TRANSITION_ID

# Every asset class a run can reference, in one snapshot: brand identity (name /
# logo / two fonts), a full sound plan (music + ambient + transitions + accents),
# two ingested sources, and the operator's own legal declarations.
_SNAPSHOT: dict[str, Any] = {
    "brand": {
        "name": "AcmeFlow",
        "logo_url": "https://cdn.example/acmeflow.svg",
        "display_font": "Satoshi",
        "body_font": "Inter",
        "font_family": "Satoshi",  # duplicate of display_font — must dedupe
    },
    "sound_effects": {
        "enabled": True,
        "music": "uplifting-corporate",
        "ambient": True,
        "sfx": ["pop", {"sfx_id": "click"}, "pop"],
    },
    "brief": {
        "sources": [
            "https://acmeflow.example/product",
            {"url": "https://acmeflow.example/pricing"},
        ]
    },
    "legal": {
        "trademarks": ["AcmeFlow®"],
        "defensibility": "patent pending, US 12/345,678",
    },
}


def test_ac7_ledger_covers_every_snapshot_asset() -> None:
    rows = build_asset_ledger(_SNAPSHOT)
    by_kind: dict[str, list[str]] = {}
    for row in rows:
        by_kind.setdefault(row.kind, []).append(row.ref)

    # Brand identity: name, logo, both distinct fonts (the duplicate collapsed).
    assert by_kind["name"] == ["AcmeFlow"]
    assert by_kind["logo"] == ["https://cdn.example/acmeflow.svg"]
    assert by_kind["font"] == ["Satoshi", "Inter"]
    # Sound: the RESOLVED plan (ambient: true → the catalog default), accents sorted.
    assert by_kind["music"] == ["uplifting-corporate"]
    assert by_kind["ambient"] == [DEFAULT_AMBIENT_ID]
    assert by_kind["transition"] == [DEFAULT_TRANSITION_ID]
    assert by_kind["sfx"] == ["click", "pop"]
    # Every ingested source, deduped, in declared order.
    assert by_kind["source"] == [
        "https://acmeflow.example/product",
        "https://acmeflow.example/pricing",
    ]
    # The operator's own declarations.
    assert by_kind["trademark"] == ["AcmeFlow®"]
    assert by_kind["defensibility"] == ["patent pending, US 12/345,678"]


def test_ac7_every_row_carries_a_non_empty_license() -> None:
    rows = build_asset_ledger(_SNAPSHOT)
    assert rows, "the ledger must not be empty for an asset-bearing snapshot"
    assert all(row.license for row in rows), "100% coverage: no row may lack a licence"


def test_ac7_only_catalog_and_declared_rows_are_verified() -> None:
    rows = build_asset_ledger(_SNAPSHOT)
    for row in rows:
        if row.source == "sound_catalog":
            # The one licence this codebase actually knows (sound_catalog docstring).
            assert row.license == CATALOG_LICENSE
            assert row.verified is True
        elif row.source == "legal":
            assert row.license == DECLARED_LICENSE
            assert row.verified is True
        else:
            # Nothing records terms for a scraped logo / webfont / source page —
            # so we say exactly that rather than inventing a licence.
            assert row.license == UNVERIFIED_LICENSE
            assert row.verified is False


def test_ac7_unverified_assets_counts_the_uncleared_rows() -> None:
    rows = build_asset_ledger(_SNAPSHOT)
    manifest = defensibility_manifest(
        claims=[], ledger=rows, disclaimers=[], trademarks={"first_party": []}
    )
    expected = sum(1 for row in rows if not row.verified)
    # name + logo + 2 fonts + 2 sources = 6 uncleared assets.
    assert expected == 6
    assert manifest["unverified_assets"] == expected


def test_ledger_is_deterministic_and_empty_snapshot_is_empty() -> None:
    assert build_asset_ledger({}) == []
    assert build_asset_ledger(_SNAPSHOT) == build_asset_ledger(_SNAPSHOT)


def test_disabled_sound_block_contributes_nothing() -> None:
    snapshot = {"sound_effects": {"enabled": False, "music": "uplifting-corporate"}}
    assert build_asset_ledger(snapshot) == []


# ------------------------------------------------------------------- AC-9 --
_LINES = [
    {"id": "clip-1", "text": "AcmeFlow keeps your board in sync."},
    {"id": "clip-2", "text": "It imports straight from Jira™ and syncs to GitHub."},
    {"id": "clip-3", "text": "Export to CSV, or push to AWS in one click."},
    {"id": "clip-4", "text": "Teams love the speed."},
]
_BRAND = {"name": "AcmeFlow"}


def test_ac9_third_party_marks_listed_first_party_excluded() -> None:
    flags = trademark_flags(_LINES, _BRAND)
    marks = [row["mark"] for row in flags["third_party_candidates"]]

    # Third-party marks: a ™-marked one, two CamelCase, two ALLCAPS.
    assert "Jira™" in marks
    assert "GitHub" in marks
    assert "CSV" in marks
    assert "AWS" in marks
    # First-party is NEVER a third-party candidate …
    assert flags["first_party"] == ["acmeflow"]
    assert "AcmeFlow" not in marks
    # … and neither are ordinary sentence-case / sentence-initial words.
    assert "Teams" not in marks
    assert "It" not in marks
    assert "Export" not in marks


def test_ac9_candidates_carry_their_line_ids_and_are_sorted() -> None:
    flags = trademark_flags(_LINES, _BRAND)
    candidates = flags["third_party_candidates"]
    assert [row["mark"] for row in candidates] == sorted(row["mark"] for row in candidates)
    by_mark = {row["mark"]: row["line_ids"] for row in candidates}
    assert by_mark["GitHub"] == ["clip-2"]
    assert by_mark["AWS"] == ["clip-3"]


# ------------------------------------------------------------------ AC-10 --
_CLAIMS = [
    {
        "line_id": "clip-1",
        "text": "The fastest onboarding on the market.",
        "claim_status": "grounded",
        "claim_source": "Benchmarks rank it the fastest onboarding.",
        "matched_term": "fastest",
    }
]
_DISCLAIMERS = [
    {
        "kind": "comparative",
        "text": "Comparison based on publicly available information.",
        "line_ids": ["clip-1"],
    }
]


def _manifest(snapshot: dict[str, Any]) -> dict[str, Any]:
    return defensibility_manifest(
        claims=_CLAIMS,
        ledger=build_asset_ledger(snapshot),
        disclaimers=_DISCLAIMERS,
        trademarks=trademark_flags(_LINES, snapshot.get("brand") or {}),
    )


def test_ac10_manifest_hash_is_stable_across_shuffled_keys() -> None:
    # The SAME resolved inputs, with every dict's keys in a different insertion
    # order — a content-addressed manifest must not notice.
    shuffled: dict[str, Any] = {
        "legal": {
            "defensibility": _SNAPSHOT["legal"]["defensibility"],
            "trademarks": _SNAPSHOT["legal"]["trademarks"],
        },
        "brief": dict(_SNAPSHOT["brief"]),
        "sound_effects": {
            "sfx": ["pop", {"sfx_id": "click"}, "pop"],
            "ambient": True,
            "music": "uplifting-corporate",
            "enabled": True,
        },
        "brand": {
            "font_family": "Satoshi",
            "body_font": "Inter",
            "logo_url": _SNAPSHOT["brand"]["logo_url"],
            "display_font": "Satoshi",
            "name": "AcmeFlow",
        },
    }
    assert _manifest(shuffled)["hash"] == _manifest(_SNAPSHOT)["hash"]


def test_ac10_manifest_hash_changes_when_an_input_changes() -> None:
    baseline = _manifest(_SNAPSHOT)
    altered = defensibility_manifest(
        claims=_CLAIMS,
        ledger=build_asset_ledger(_SNAPSHOT),
        disclaimers=[],  # the disclaimer requirement disappeared
        trademarks=trademark_flags(_LINES, _SNAPSHOT["brand"]),
    )
    assert altered["hash"] != baseline["hash"]


def test_ac10_manifest_shape() -> None:
    manifest = _manifest(_SNAPSHOT)
    assert manifest["version"] == 1
    assert manifest["claims"] == _CLAIMS
    assert manifest["disclaimers"] == _DISCLAIMERS
    assert manifest["trademarks"]["first_party"] == ["acmeflow"]
    assert len(manifest["hash"]) == 64  # sha256 hex
    assert all(row["license"] for row in manifest["ledger"])


def test_tenant_uploads_are_tenant_declared_never_royalty_free() -> None:
    """P6 ORCH-4: a ``tenant_`` id is a tenant upload whose rights the tenant only
    ATTESTED. It must not carry the bundled library's royalty-free licence or count
    as verified; a bundled id in the same plan keeps its licence (control)."""
    from src.services.stages.shared.asset_rights import TENANT_DECLARED_LICENSE

    snapshot: dict[str, Any] = {
        "sound_effects": {
            "enabled": True,
            "music": "tenant_our_anthem",
            "ambient": "tenant_office_bed",
            "transitions": False,
            "sfx": ["click_ui", "tenant_ding"],
        }
    }
    rows = {(r.kind, r.ref): r for r in build_asset_ledger(snapshot)}
    for key in (
        ("music", "tenant_our_anthem"),
        ("ambient", "tenant_office_bed"),
        ("sfx", "tenant_ding"),
    ):
        row = rows[key]
        assert row.license == TENANT_DECLARED_LICENSE, key
        assert row.license != CATALOG_LICENSE
        assert row.verified is False
        assert row.source == "tenant_sound_library"
    bundled = rows[("sfx", "click_ui")]
    assert bundled.license == CATALOG_LICENSE and bundled.verified is True

    manifest = defensibility_manifest(
        claims=[], ledger=list(rows.values()), disclaimers=[], trademarks={}
    )
    assert manifest["unverified_assets"] == 3
