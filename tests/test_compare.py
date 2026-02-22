from typing import Any

import pytest

from memory_inspector.adapters.base import BaseRetrieverAdapter
from memory_inspector.compare import compare
from memory_inspector.types import ComparisonResult, RetrievalResult


def make_retriever(*items: tuple[str, float, str | None]):
    """Build a retriever from (text, score, id) tuples."""
    def retriever(query: str, top_k: int = 5) -> list[RetrievalResult]:
        return [
            RetrievalResult(text=t, score=s, id=doc_id)
            for t, s, doc_id in items[:top_k]
        ]
    return retriever


def test_compare_returns_comparison_result():
    a = make_retriever(("doc A", 0.9, "a"), ("doc B", 0.8, "b"))
    b = make_retriever(("doc B", 0.85, "b"), ("doc A", 0.7, "a"))
    result = compare("test query", a, b)
    assert isinstance(result, ComparisonResult)
    assert result.query == "test query"
    assert len(result.results_a) == 2
    assert len(result.results_b) == 2


def test_compare_promoted():
    a = make_retriever(("doc A", 0.9, "a"), ("doc B", 0.8, "b"))
    b = make_retriever(("doc B", 0.85, "b"), ("doc A", 0.7, "a"))
    result = compare("test", a, b)
    delta_map = {d.id: d for d in result.deltas}
    assert delta_map["b"].status == "promoted"  # rank 1 → 0
    assert delta_map["a"].status == "demoted"   # rank 0 → 1


def test_compare_dropped():
    a = make_retriever(("doc A", 0.9, "a"), ("doc B", 0.8, "b"))
    b = make_retriever(("doc A", 0.9, "a"),)
    result = compare("test", a, b)
    delta_map = {d.id: d for d in result.deltas}
    assert delta_map["b"].status == "dropped"
    assert delta_map["b"].rank_after is None


def test_compare_new():
    a = make_retriever(("doc A", 0.9, "a"),)
    b = make_retriever(("doc A", 0.9, "a"), ("doc C", 0.7, "c"))
    result = compare("test", a, b)
    delta_map = {d.id: d for d in result.deltas}
    assert delta_map["c"].status == "new"
    assert delta_map["c"].rank_before is None


def test_compare_unchanged():
    a = make_retriever(("doc A", 0.9, "a"), ("doc B", 0.8, "b"))
    b = make_retriever(("doc A", 0.9, "a"), ("doc B", 0.8, "b"))
    result = compare("test", a, b)
    for d in result.deltas:
        assert d.status == "unchanged"


def test_compare_text_fallback_matching():
    """Match by text when id is None."""
    def retriever_a(query: str, top_k: int = 5) -> list[RetrievalResult]:
        return [RetrievalResult(text="hello world", score=0.9)]

    def retriever_b(query: str, top_k: int = 5) -> list[RetrievalResult]:
        return [RetrievalResult(text="hello world", score=0.7)]

    result = compare("test", retriever_a, retriever_b)
    assert len(result.deltas) == 1
    assert result.deltas[0].status == "unchanged"


def test_compare_latency_measured():
    a = make_retriever(("doc A", 0.9, "a"))
    b = make_retriever(("doc A", 0.9, "a"))
    result = compare("test", a, b)
    assert result.latency_a_ms >= 0
    assert result.latency_b_ms >= 0


def test_compare_empty_results():
    def empty(query: str, top_k: int = 5) -> list[RetrievalResult]:
        return []

    result = compare("test", empty, empty)
    assert len(result.results_a) == 0
    assert len(result.results_b) == 0
    assert len(result.deltas) == 0


def test_compare_with_custom_adapter():
    class UppercaseAdapter(BaseRetrieverAdapter):
        def normalize(self, raw_output: Any) -> list[RetrievalResult]:
            return [RetrievalResult(text=s.upper(), id=s) for s in raw_output]

    def str_retriever(query: str, top_k: int = 5) -> list[str]:
        return ["hello", "world"]

    result = compare(
        "test",
        str_retriever,
        str_retriever,
        adapter_a=UppercaseAdapter(),
        adapter_b=UppercaseAdapter(),
    )
    assert result.results_a[0].text == "HELLO"
    assert result.results_a[1].text == "WORLD"


def test_compare_score_deltas():
    a = make_retriever(("doc A", 0.9, "a"))
    b = make_retriever(("doc A", 0.7, "a"))
    result = compare("test", a, b)
    delta = result.deltas[0]
    assert delta.score_before == pytest.approx(0.9)
    assert delta.score_after == pytest.approx(0.7)


def test_compare_repr():
    a = make_retriever(("doc A", 0.9, "a"), ("doc B", 0.8, "b"))
    b = make_retriever(("doc B", 0.85, "b"), ("doc A", 0.7, "a"))
    result = compare("what is RAG?", a, b)
    text = repr(result)
    assert "what is RAG?" in text
    assert "retriever_a" in text
    assert "retriever_b" in text
    assert "Deltas" in text


def test_compare_id_matching_preferred_over_text():
    """When id is available, match by id even if text differs."""
    def retriever_a(query: str, top_k: int = 5) -> list[RetrievalResult]:
        return [RetrievalResult(text="original text", score=0.9, id="doc-1")]

    def retriever_b(query: str, top_k: int = 5) -> list[RetrievalResult]:
        return [RetrievalResult(text="updated text", score=0.7, id="doc-1")]

    result = compare("test", retriever_a, retriever_b)
    # Same id → matched as one entry, not two separate
    assert len(result.deltas) == 1
    assert result.deltas[0].status == "unchanged"  # same rank (0 → 0)
