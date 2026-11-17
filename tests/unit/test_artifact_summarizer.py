"""Tests for the artifact summarizer module."""

from __future__ import annotations

from src.services.artifact_summarizer import summarize_stage_result


class TestSummarizeCrawl:
    def test_strips_html_and_text_from_pages(self):
        result = {
            "status": "completed",
            "domain": "example.com",
            "pages": [
                {
                    "url": "https://example.com",
                    "title": "Home",
                    "html": "<html><body>Very long HTML...</body></html>" * 100,
                    "text": "Extracted text content " * 200,
                    "page_id": "page-000",
                    "depth": 0,
                    "word_count": 200,
                    "images": ["img1.png", "img2.png"],
                    "links_to": ["https://example.com/about"],
                },
            ],
            "link_graph": {"nodes": [{"id": "a"}, {"id": "b"}], "edges": []},
            "raw": {"total_pages": 1, "domain": "example.com"},
        }

        summary = summarize_stage_result("crawl", result)

        assert summary["status"] == "completed"
        assert summary["domain"] == "example.com"
        assert summary["page_count"] == 1
        assert summary["link_graph_nodes"] == 2

        page = summary["pages"][0]
        assert page["url"] == "https://example.com"
        assert page["title"] == "Home"
        assert page["page_id"] == "page-000"
        assert page["word_count"] == 200
        assert page["images"] == ["img1.png", "img2.png"]
        assert "html" not in page
        assert "text" not in page
        assert "links_to" not in page

    def test_handles_empty_pages(self):
        result = {"status": "completed", "domain": "", "pages": [], "link_graph": {}, "raw": {}}
        summary = summarize_stage_result("crawl", result)
        assert summary["page_count"] == 0
        assert summary["pages"] == []

    def test_preserves_raw(self):
        result = {
            "status": "completed",
            "domain": "x.com",
            "pages": [],
            "link_graph": {},
            "raw": {"total_pages": 5, "domain": "x.com"},
        }
        summary = summarize_stage_result("crawl", result)
        assert summary["raw"] == {"total_pages": 5, "domain": "x.com"}


class TestSummarizeAnalyze:
    def test_strips_html_from_pages(self):
        result = {
            "status": "completed",
            "job_id": "job-abc",
            "site_analysis": {"score": 85},
            "pages": [
                {
                    "url": "https://example.com",
                    "title": "Home",
                    "html": "<html>" * 2000,
                    "score": 90,
                },
            ],
            "raw": {"total": 1},
        }

        summary = summarize_stage_result("analyze", result)

        assert summary["job_id"] == "job-abc"
        assert summary["site_analysis"] == {"score": 85}
        assert summary["page_count"] == 1
        page = summary["pages"][0]
        assert page["url"] == "https://example.com"
        assert page["score"] == 90
        assert "html" not in page

    def test_keeps_short_strings(self):
        result = {
            "status": "completed",
            "job_id": "j1",
            "site_analysis": {},
            "pages": [{"description": "A short description", "category": "tech"}],
            "raw": {},
        }
        summary = summarize_stage_result("analyze", result)
        page = summary["pages"][0]
        assert page["description"] == "A short description"
        assert page["category"] == "tech"


class TestSummarizeGeneric:
    def test_passthrough_for_small_results(self):
        result = {
            "status": "completed",
            "job_id": "j1",
            "reviews": [{"finding": "looks good"}],
            "raw": {},
        }
        summary = summarize_stage_result("review", result)
        assert summary == result

    def test_strips_html_keys_for_large_results(self):
        result = {
            "status": "completed",
            "html": "<html>" * 10000,
            "data": {"html": "<nested>" * 5000, "value": 42},
        }
        summary = summarize_stage_result("unknown_stage", result)
        assert "html" not in summary
        assert "html" not in summary["data"]
        assert summary["data"]["value"] == 42

    def test_truncates_long_strings(self):
        # Generic fallback only strips when serialized size > 50KB
        long_text = "x" * 20000
        result = {
            "status": "completed",
            "content": long_text,
            "bulk": "y" * 40000,
            "short": "ok",
        }
        summary = summarize_stage_result("unknown_stage", result)
        assert len(summary["content"]) < len(long_text)
        assert "truncated" in summary["content"]
        assert summary["short"] == "ok"


class TestSummarizeIngest:
    def test_strips_large_text_from_documents(self):
        result = {
            "status": "completed",
            "pages": [
                {"url": "https://example.com", "html": "<html>" * 2000, "title": "Home"},
            ],
            "documents": [
                {"name": "doc.pdf", "content": "A" * 10000, "type": "pdf"},
            ],
            "raw": {},
        }
        summary = summarize_stage_result("ingest", result)
        assert summary["page_count"] == 1
        assert summary["document_count"] == 1
        assert "html" not in summary["pages"][0]
        assert summary["pages"][0]["title"] == "Home"
        doc = summary["documents"][0]
        assert doc["name"] == "doc.pdf"
        assert doc["type"] == "pdf"
        # content is long but not an html key, so it should be stripped by _MAX_STRING_LEN check
        assert "content" not in doc or len(doc.get("content", "")) <= 5000
