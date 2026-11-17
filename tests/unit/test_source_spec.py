"""parse_sources — tolerant extraction of the universal multi-source list."""

from __future__ import annotations

from src.models.source_spec import parse_sources


def test_no_sources_returns_empty():
    assert parse_sources({}) == []
    assert parse_sources({"sources": None}) == []
    assert parse_sources({"sources": "not-a-list"}) == []


def test_parses_valid_entries():
    out = parse_sources(
        {
            "sources": [
                {"kind": "url", "ref": "https://a.com", "label": "Site"},
                {"kind": "document", "ref": "documents/t/x.pdf"},
                {"kind": "media", "ref": "https://img/1.jpg", "label": "hero"},
            ]
        }
    )
    assert [s.kind for s in out] == ["url", "document", "media"]
    assert out[0].label == "Site"
    assert out[1].label == ""


def test_skips_malformed_entries():
    out = parse_sources(
        {
            "sources": [
                {"kind": "url", "ref": "https://ok.com"},
                {"kind": "bogus", "ref": "x"},  # invalid kind
                {"kind": "url", "ref": "   "},  # blank ref
                {"kind": "url"},  # missing ref
                "string-not-dict",
                {"ref": "no-kind"},
            ]
        }
    )
    assert len(out) == 1
    assert out[0].ref == "https://ok.com"


def test_auth_info_only_kept_when_dict():
    out = parse_sources(
        {
            "sources": [
                {"kind": "url", "ref": "https://a.com", "auth_info": {"type": "basic"}},
                {"kind": "url", "ref": "https://b.com", "auth_info": "nope"},
            ]
        }
    )
    assert out[0].auth_info == {"type": "basic"}
    assert out[1].auth_info is None
