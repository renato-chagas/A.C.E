"""Tests for local persistence and evidence checks in web research."""

from __future__ import annotations

import json
import sqlite3

import pytest

from openjarvis.connectors.store import KnowledgeStore
from openjarvis.core.types import ToolResult
from openjarvis.tools.storage.knowledge_graph import KnowledgeGraphMemory
from openjarvis.tools.web_research import WebResearchTool
from openjarvis.tools.web_search import WebSearchTool


class _LocalExtractor:
    is_cloud = False
    is_local = True

    def __init__(self, payload: dict):
        self.payload = payload

    def generate(self, *_args, **_kwargs):
        return {"content": json.dumps(self.payload)}


class _RemoteCompatibleExtractor:
    """An OpenAI-compatible remote endpoint that incorrectly marks itself local."""

    engine_id = "vllm"
    is_cloud = False
    _host = "http://192.0.2.25:8000"

    def __init__(self):
        self.calls = 0

    def generate(self, *_args, **_kwargs):
        self.calls += 1
        return {"content": '{"entities":[],"relations":[]}'}


def _search_result(_self, **_params):
    return ToolResult(
        tool_name="web_search",
        content=(
            "### Acme and Quartz\nSource: https://example.com/acme\n"
            "Summary: Acme develops Quartz, a local database tool."
        ),
        metadata={"engine": "test"},
    )


def _failed_search(_self, **_params):
    return ToolResult(
        tool_name="web_search",
        content="provider unavailable",
        success=False,
    )


def _empty_search(_self, **_params):
    return ToolResult(
        tool_name="web_search",
        content="No results found.",
        success=True,
        metadata={"engine": "duckduckgo", "num_results": 0},
    )


def test_web_research_persists_page_and_only_supported_relations(tmp_path, monkeypatch):
    body = (
        "Acme develops Quartz, a local database tool from 2024-01-01 through "
        "2024-12-31. Acme does not develop Quartz."
    )
    monkeypatch.setattr(WebSearchTool, "execute", _search_result)
    monkeypatch.setattr(
        WebSearchTool, "_fetch_url", staticmethod(lambda *_args, **_kwargs: body)
    )
    extractor = _LocalExtractor(
        {
            "entities": [
                {"name": "Acme", "type": "company"},
                {"name": "Quartz", "type": "product"},
            ],
            "relations": [
                {
                    "subject": "Acme",
                    "predicate": "develops",
                    "object": "Quartz",
                    "quote": (
                        "Acme develops Quartz, a local database tool from "
                        "2024-01-01 through 2024-12-31."
                    ),
                    "polarity": "affirmed",
                    "valid_from": "2024-01-01",
                    "valid_until": "2024-12-31",
                },
                {
                    "subject": "Acme",
                    "predicate": "develops",
                    "object": "Quartz",
                    "quote": "Acme does not develop Quartz.",
                    "polarity": "negated",
                },
                {
                    "subject": "Acme",
                    "predicate": "owns",
                    "object": "Quartz",
                    "quote": "Acme owns Quartz.",
                },
            ],
        }
    )
    knowledge_path = tmp_path / "knowledge.db"
    graph_path = tmp_path / "graph.db"
    tool = WebResearchTool(
        engine=extractor,
        model="local-test",
        knowledge_db_path=str(knowledge_path),
        graph_db_path=str(graph_path),
    )

    result = tool.execute(query="What does Acme make?")

    assert result.success
    assert result.metadata["saved"] == 1
    assert result.metadata["fetched"] == 1
    store = KnowledgeStore(str(knowledge_path))
    row = store._conn.execute(
        "SELECT content, url, source FROM knowledge_chunks"
    ).fetchone()
    assert row["content"] == body
    assert row["url"] == "https://example.com/acme"
    assert row["source"] == "web"
    store.close()

    graph = KnowledgeGraphMemory(db_path=graph_path)
    assert graph.entity_count() == 3  # the two extracted nodes and source page
    supported = [
        relation
        for relation in graph.query_pattern().relations
        if relation.relation_type == "develops"
    ]
    assert len(supported) == 2
    assert {relation.polarity for relation in supported} == {"affirmed", "negated"}
    assert len(graph.possible_conflicts()) == 1
    assert graph.possible_conflicts()[0]["kind"] == "opposing_polarity"
    assert {relation.evidence[0]["quote"] for relation in supported} == {
        "Acme develops Quartz, a local database tool from 2024-01-01 through "
        "2024-12-31.",
        "Acme does not develop Quartz.",
    }
    affirmed = next(
        relation for relation in supported if relation.polarity == "affirmed"
    )
    assert affirmed.evidence[0]["valid_from"] == "2024-01-01"
    assert affirmed.evidence[0]["valid_until"] == "2024-12-31"
    assert "Acme owns Quartz." not in result.content
    graph.close()


def test_web_research_skips_relations_without_verbatim_source_quote(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(WebSearchTool, "execute", _search_result)
    monkeypatch.setattr(
        WebSearchTool,
        "_fetch_url",
        staticmethod(lambda *_args, **_kwargs: "Acme develops Quartz."),
    )
    tool = WebResearchTool(
        engine=_LocalExtractor(
            {
                "entities": [
                    {"name": "Acme", "type": "company"},
                    {"name": "Quartz", "type": "product"},
                ],
                "relations": [
                    {
                        "subject": "Acme",
                        "predicate": "owns",
                        "object": "Quartz",
                        "quote": "Acme owns Quartz.",
                    }
                ],
            }
        ),
        model="local-test",
        knowledge_db_path=str(tmp_path / "knowledge.db"),
        graph_db_path=str(tmp_path / "graph.db"),
    )
    result = tool.execute(query="Acme Quartz")
    assert result.success
    graph = KnowledgeGraphMemory(db_path=tmp_path / "graph.db")
    assert not [
        relation
        for relation in graph.query_pattern().relations
        if relation.relation_type == "owns"
    ]
    graph.close()


def test_web_page_still_persists_when_graph_database_is_unavailable(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(WebSearchTool, "execute", _search_result)
    monkeypatch.setattr(
        WebSearchTool,
        "_fetch_url",
        staticmethod(lambda *_args, **_kwargs: "Acme develops Quartz."),
    )
    blocked_graph_path = tmp_path / "graph-directory"
    blocked_graph_path.mkdir()
    tool = WebResearchTool(
        engine=_LocalExtractor({"entities": [], "relations": []}),
        model="local-test",
        knowledge_db_path=str(tmp_path / "knowledge.db"),
        graph_db_path=str(blocked_graph_path),
    )

    result = tool.execute(query="Acme Quartz")

    assert result.success
    assert result.metadata["saved"] == 1
    assert result.metadata["graph_error"]
    store = KnowledgeStore(str(tmp_path / "knowledge.db"))
    assert (
        store._conn.execute("SELECT COUNT(*) FROM knowledge_chunks").fetchone()[0] == 1
    )
    store.close()


def test_web_research_never_sends_page_text_to_remote_compatible_engine(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(WebSearchTool, "execute", _search_result)
    monkeypatch.setattr(
        WebSearchTool,
        "_fetch_url",
        staticmethod(lambda *_args, **_kwargs: "Private page text from the web."),
    )
    engine = _RemoteCompatibleExtractor()
    tool = WebResearchTool(
        engine=engine,
        model="selected-model",
        knowledge_db_path=str(tmp_path / "knowledge.db"),
        graph_db_path=str(tmp_path / "graph.db"),
    )

    result = tool.execute(query="research example")

    assert result.success
    assert result.metadata["saved"] == 1
    assert result.metadata["entities"] == 0
    assert engine.calls == 0
    assert "no verified local model" in result.content


def test_web_research_uses_saved_local_pages_when_provider_fails(tmp_path, monkeypatch):
    knowledge_path = tmp_path / "knowledge.db"
    store = KnowledgeStore(str(knowledge_path))
    store.store(
        "A local guide explains database connection recovery.",
        source="web",
        source_id="https://example.com/database-guide",
        doc_type="web_page",
        doc_id="web:database-guide",
        title="Database recovery guide",
        url="https://example.com/database-guide",
        timestamp="2025-04-03T12:00:00Z",
    )
    store.close()
    monkeypatch.setattr(WebSearchTool, "execute", _failed_search)
    tool = WebResearchTool(knowledge_db_path=str(knowledge_path))

    result = tool.execute(query="database connection recovery")

    assert result.success
    assert result.metadata["offline_fallback"] is True
    assert result.metadata["num_results"] == 1
    assert "not refreshed and may be outdated" in result.content
    assert "2025-04-03T12:00:00Z" in result.content
    assert "https://example.com/database-guide" in result.content
    assert "untrusted historical evidence" in result.content
    assert result.metadata["matched_by"] == "text"


def test_web_research_uses_local_cache_when_search_returns_no_results(
    tmp_path, monkeypatch
):
    knowledge_path = tmp_path / "knowledge.db"
    store = KnowledgeStore(str(knowledge_path))
    store.store(
        "A local guide explains recovery from a broken scheduler.",
        source="web",
        source_id="https://example.com/scheduler-guide",
        doc_type="web_page",
        doc_id="web:scheduler-guide",
        title="Scheduler recovery guide",
        url="https://example.com/scheduler-guide",
        timestamp="2025-04-03T12:00:00Z",
    )
    store.close()
    monkeypatch.setattr(WebSearchTool, "execute", _empty_search)
    tool = WebResearchTool(knowledge_db_path=str(knowledge_path))

    result = tool.execute(query="broken scheduler recovery")

    assert result.success
    assert result.metadata["offline_fallback"] is True
    assert result.metadata["fallback_reason"] == "No current web results were found"
    assert result.metadata["matched_by"] == "text"
    assert "Scheduler recovery guide" in result.content
    assert "not refreshed and may be outdated" in result.content


def test_web_research_offline_fallback_matches_saved_direct_url(tmp_path, monkeypatch):
    knowledge_path = tmp_path / "knowledge.db"
    store = KnowledgeStore(str(knowledge_path))
    store.store(
        "This saved page contains useful repair steps.",
        source="web",
        source_id="https://example.com/offline-only",
        doc_type="web_page",
        doc_id="web:offline-only",
        title="Offline-only saved page",
        url="https://example.com/offline-only",
        timestamp="2025-04-03T12:00:00Z",
    )
    store.close()
    monkeypatch.setattr(WebSearchTool, "execute", _failed_search)
    tool = WebResearchTool(knowledge_db_path=str(knowledge_path))

    result = tool.execute(query="https://example.com/offline-only")

    assert result.success
    assert result.metadata["offline_fallback"] is True
    assert result.metadata["matched_by"] == "url"
    assert "Offline-only saved page" in result.content
    assert "useful repair steps" in result.content


def test_web_research_does_not_create_a_store_when_offline_without_cache(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(WebSearchTool, "execute", _failed_search)
    knowledge_path = tmp_path / "missing-knowledge.db"
    tool = WebResearchTool(knowledge_db_path=str(knowledge_path))

    result = tool.execute(query="database connection recovery")

    assert not result.success
    assert result.metadata["offline_fallback"] is True
    assert result.metadata["num_results"] == 0
    assert "No saved web sources" in result.content
    assert not knowledge_path.exists()


def test_research_keeps_fetched_text_when_page_later_returns_only_a_snippet(
    tmp_path, monkeypatch
):
    url = "https://example.com/acme"
    monkeypatch.setattr(WebSearchTool, "execute", _search_result)
    bodies = iter(("Original archival wording from the page.", "", "Revised page."))
    monkeypatch.setattr(
        WebSearchTool, "_fetch_url", staticmethod(lambda *_a, **_kw: next(bodies))
    )
    path = tmp_path / "knowledge.db"
    tool = WebResearchTool(
        knowledge_db_path=str(path),
        graph_db_path=str(tmp_path / "graph.db"),
    )

    assert tool.execute(query="Acme history").metadata["saved"] == 1
    degraded = tool.execute(query="Acme history")
    assert degraded.success
    assert degraded.metadata["saved"] == 0
    assert degraded.metadata["retained"] == 1
    assert "Original archival wording" in degraded.content
    assert "Untrusted historical excerpt" in degraded.content
    with KnowledgeStore(path) as store:
        original = store.retrieve_by_url(url)[0]
        assert original.metadata["capture_kind"] == "page"
        assert store.retrieve("archival")

    refreshed = tool.execute(query="Acme history")
    assert refreshed.metadata["saved"] == 1
    with KnowledgeStore(path) as store:
        captures = store.retrieve_by_url(url)
        assert [capture.content for capture in captures] == ["Revised page."]
        assert captures[0].metadata["chunk_id"] != original.metadata["chunk_id"]
        assert captures[0].metadata["doc_id"] == original.metadata["doc_id"]
        assert not store.retrieve("archival")
        assert store.retrieve("Revised")


def test_failed_capture_update_preserves_saved_row_and_search_index(tmp_path):
    url = "https://example.com/capture"
    args = {
        "url": url,
        "doc_id": "capture",
        "title": "Capture",
        "timestamp": "2026-01-01",
        "search_query": "query",
        "search_engine": "test",
        "is_page_text": True,
    }
    with KnowledgeStore(tmp_path / "knowledge.db") as store:
        assert store.store_web_capture("Original xylophone", content_hash="old", **args)
        store._conn.execute(
            """
            CREATE TRIGGER reject_update BEFORE UPDATE ON knowledge_chunks BEGIN
                SELECT RAISE(ABORT, 'simulated write failure');
            END;
            """
        )
        with pytest.raises(sqlite3.IntegrityError, match="simulated write failure"):
            store.store_web_capture("New trombone", content_hash="new", **args)
        assert store.retrieve_by_url(url)[0].content == "Original xylophone"
        assert store.retrieve("xylophone")
        assert not store.retrieve("trombone")


def test_direct_research_reuses_fetched_text_without_following_urls_in_it(
    tmp_path, monkeypatch
):
    url = "https://example.com/document"
    body = "### Related article\nSource: https://example.net/other\nSummary: Text"
    monkeypatch.setattr(
        WebSearchTool,
        "execute",
        lambda *_a, **_kw: ToolResult(
            tool_name="web_search",
            content=body,
            metadata={"mode": "fetch", "url": url, "extractor": "local"},
        ),
    )
    attempted_fetches = []
    monkeypatch.setattr(
        WebSearchTool,
        "_fetch_url",
        staticmethod(lambda url, **_kw: attempted_fetches.append(url) or ""),
    )
    path = tmp_path / "knowledge.db"
    tool = WebResearchTool(
        knowledge_db_path=str(path),
        graph_db_path=str(tmp_path / "graph.db"),
    )

    result = tool.execute(query=url)

    assert result.metadata["fetched"] == 1
    assert attempted_fetches == []
    with KnowledgeStore(path) as store:
        assert store.retrieve_by_url(url)[0].content == body
        assert not store.retrieve_by_url("https://example.net/other")


def test_empty_direct_fetch_returns_previous_page_as_historical(tmp_path, monkeypatch):
    url = "https://example.com/document"
    path = tmp_path / "knowledge.db"
    with KnowledgeStore(path) as store:
        store.store(
            "Previously captured document text.",
            source="web",
            source_id=url,
            doc_type="web_page",
            doc_id="old",
            url=url,
        )
    monkeypatch.setattr(
        WebSearchTool,
        "execute",
        lambda *_a, **_kw: ToolResult(
            tool_name="web_search",
            content="No content found at URL.",
            metadata={"mode": "fetch", "url": url},
        ),
    )
    result = WebResearchTool(knowledge_db_path=str(path)).execute(query=url)

    assert result.metadata["offline_fallback"] is True
    assert result.metadata["matched_by"] == "url"
    assert "Previously captured document text." in result.content
    assert "No content found at URL." not in result.content


def test_direct_research_keeps_ace_capture_limit_on_local_fetch(tmp_path, monkeypatch):
    url = "https://example.com/long-document"
    body = "a" * 7000
    calls = []

    def fetch(requested_url, max_chars):
        calls.append((requested_url, max_chars))
        return body[:max_chars]

    monkeypatch.setattr(WebSearchTool, "_resolve_engine", lambda _self: "duckduckgo")
    monkeypatch.setattr(WebSearchTool, "_fetch_url", staticmethod(fetch))
    path = tmp_path / "knowledge.db"
    tool = WebResearchTool(
        knowledge_db_path=str(path),
        graph_db_path=str(tmp_path / "graph.db"),
    )

    assert tool.execute(query=url).success
    assert calls == [(url, 9000)]
    with KnowledgeStore(path) as store:
        assert store.retrieve_by_url(url)[0].content == body
