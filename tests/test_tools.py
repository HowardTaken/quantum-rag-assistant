from langchain_core.documents import Document

from tests.fakes import FakeChroma
from tools import format_docs, make_tools, source_counts


def test_format_docs_empty():
    assert format_docs([]) == "No relevant passages found."


def test_format_docs_includes_source_and_page():
    docs = [Document(page_content="hello", metadata={"source": "paper.pdf", "page": 3})]
    out = format_docs(docs)
    assert "[Source: paper.pdf, page 3]" in out
    assert "hello" in out


def test_search_papers_tool_returns_formatted_hits():
    docs = [Document(page_content="quantum cascade laser", metadata={"source": "a.pdf", "page": 1})]
    db = FakeChroma(docs=docs)
    search_papers, _list_papers = make_tools(db)
    result = search_papers.invoke({"query": "laser"})
    assert "a.pdf" in result
    assert "quantum cascade laser" in result


def test_search_papers_tool_handles_no_hits():
    db = FakeChroma(docs=[])
    search_papers, _list_papers = make_tools(db)
    assert search_papers.invoke({"query": "anything"}) == "No relevant passages found."


def test_list_available_papers_tool_counts_chunks():
    db = FakeChroma(metadatas=[{"source": "a.pdf"}, {"source": "a.pdf"}, {"source": "b.pdf"}])
    _search_papers, list_papers = make_tools(db)
    result = list_papers.invoke({})
    assert "a.pdf (2 chunks)" in result
    assert "b.pdf (1 chunks)" in result


def test_list_available_papers_tool_empty():
    db = FakeChroma(metadatas=[])
    _search_papers, list_papers = make_tools(db)
    assert list_papers.invoke({}) == "No papers are currently ingested."


def test_source_counts():
    db = FakeChroma(metadatas=[{"source": "a.pdf"}, {"source": "b.pdf"}, {"source": "a.pdf"}])
    assert source_counts(db) == {"a.pdf": 2, "b.pdf": 1}
