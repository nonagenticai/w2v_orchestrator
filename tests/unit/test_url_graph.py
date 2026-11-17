"""Tests for URL normalization and graph link handling."""

from __future__ import annotations

from w2v_common.clients.url_graph import URLGraph


def test_add_page_decodes_bytes_link() -> None:
    """Bytes links should be decoded so urljoin receives consistent text types."""
    graph = URLGraph("example.com")

    new_urls = graph.add_page("https://example.com", [b"/about"])

    assert new_urls == ["https://example.com/about"]
    assert graph.to_dict()["https://example.com/"] == ["https://example.com/about"]


def test_add_page_ignores_non_url_link_values() -> None:
    """Non-string link payload items should be ignored without raising."""
    graph = URLGraph("example.com")

    new_urls = graph.add_page("https://example.com", [None, 123, {"href": "/x"}])

    assert new_urls == []
    assert graph.to_dict()["https://example.com/"] == []


def test_add_page_handles_non_collection_links() -> None:
    """Unexpected non-collection link payloads should not break crawling."""
    graph = URLGraph("example.com")

    new_urls = graph.add_page("https://example.com", None)

    assert new_urls == []
    assert graph.to_dict()["https://example.com/"] == []
