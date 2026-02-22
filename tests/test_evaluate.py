from __future__ import annotations

import pytest

from memory_inspector import evaluate, EvalSample, EvaluationResult, QueryEvalResult
from memory_inspector.types import RetrievalResult


def make_retriever(*doc_ids: str):
    """Return a retriever that yields results with the given IDs in order."""
    results = [
        RetrievalResult(text=f"doc {doc_id}", score=1.0 - i * 0.1, id=doc_id, rank=i)
        for i, doc_id in enumerate(doc_ids)
    ]

    def retriever(query: str, top_k: int = 5):
        return results[:top_k]

    return retriever


# ---------------------------------------------------------------------------
# Basic metric correctness
# ---------------------------------------------------------------------------

def test_first_result_is_relevant():
    retriever = make_retriever("a", "b", "c")
    dataset = [EvalSample(query="q", relevant_ids=("a",))]
    result = evaluate(retriever, dataset, k=5)

    assert result.mrr == pytest.approx(1.0)
    assert result.recall_at_k == pytest.approx(1.0)


def test_relevant_at_position_3():
    retriever = make_retriever("x", "y", "a", "b")
    dataset = [EvalSample(query="q", relevant_ids=("a",))]
    result = evaluate(retriever, dataset, k=5)

    assert result.mrr == pytest.approx(1 / 3)
    assert result.recall_at_k == pytest.approx(1.0)


def test_no_relevant_in_top_k():
    retriever = make_retriever("x", "y", "z")
    dataset = [EvalSample(query="q", relevant_ids=("a",))]
    result = evaluate(retriever, dataset, k=3)

    assert result.mrr == pytest.approx(0.0)
    assert result.recall_at_k == pytest.approx(0.0)


def test_partial_recall_multiple_relevant():
    # 2 relevant docs, only 1 in top-2
    retriever = make_retriever("a", "x", "b")
    dataset = [EvalSample(query="q", relevant_ids=("a", "b"))]
    result = evaluate(retriever, dataset, k=2)

    assert result.recall_at_k == pytest.approx(0.5)  # 1 of 2 relevant in top-2
    assert result.mrr == pytest.approx(1.0)  # "a" is at position 1


def test_full_recall_multiple_relevant():
    retriever = make_retriever("a", "b", "c")
    dataset = [EvalSample(query="q", relevant_ids=("a", "b"))]
    result = evaluate(retriever, dataset, k=5)

    assert result.recall_at_k == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Aggregation across multiple queries
# ---------------------------------------------------------------------------

def test_mrr_averaged_across_queries():
    # Query 1: relevant at position 1 → RR=1.0
    # Query 2: relevant at position 2 → RR=0.5
    # MRR = (1.0 + 0.5) / 2 = 0.75
    retriever_q1 = make_retriever("a", "b")
    retriever_q2 = make_retriever("x", "a")

    calls = [0]

    def smart_retriever(query: str, top_k: int = 5):
        calls[0] += 1
        if query == "q1":
            return retriever_q1("q1", top_k)
        return retriever_q2("q2", top_k)

    dataset = [
        EvalSample(query="q1", relevant_ids=("a",)),
        EvalSample(query="q2", relevant_ids=("a",)),
    ]
    result = evaluate(smart_retriever, dataset, k=5)

    assert result.mrr == pytest.approx(0.75)


def test_recall_averaged_across_queries():
    def smart_retriever(query: str, top_k: int = 5):
        if query == "q1":
            return [RetrievalResult(text="t", score=1.0, id="a", rank=0)]
        # q2: relevant doc not returned
        return [RetrievalResult(text="t", score=1.0, id="x", rank=0)]

    dataset = [
        EvalSample(query="q1", relevant_ids=("a",)),
        EvalSample(query="q2", relevant_ids=("b",)),
    ]
    result = evaluate(smart_retriever, dataset, k=5)

    # query1 recall=1.0, query2 recall=0.0 → mean=0.5
    assert result.recall_at_k == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_dataset():
    retriever = make_retriever("a")
    result = evaluate(retriever, [], k=5)

    assert result.mrr == pytest.approx(0.0)
    assert result.recall_at_k == pytest.approx(0.0)
    assert result.k == 5
    assert result.per_query == ()


def test_empty_relevant_ids_raises():
    retriever = make_retriever("a")
    with pytest.raises(ValueError, match="empty relevant_ids"):
        evaluate(retriever, [EvalSample(query="q", relevant_ids=())], k=5)


def test_result_without_id_not_counted():
    # Results with no id should not match any relevant_id
    def retriever(query: str, top_k: int = 5):
        return [RetrievalResult(text="some text", score=0.9, id=None, rank=0)]

    dataset = [EvalSample(query="q", relevant_ids=("a",))]
    result = evaluate(retriever, dataset, k=5)

    assert result.mrr == pytest.approx(0.0)
    assert result.recall_at_k == pytest.approx(0.0)


def test_k_limits_retrieval():
    # Relevant doc is at position 4, but k=3 — should not be counted
    retriever = make_retriever("x", "y", "z", "a")
    dataset = [EvalSample(query="q", relevant_ids=("a",))]
    result = evaluate(retriever, dataset, k=3)

    assert result.recall_at_k == pytest.approx(0.0)
    assert result.mrr == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Return type and repr
# ---------------------------------------------------------------------------

def test_return_types():
    retriever = make_retriever("a")
    dataset = [EvalSample(query="q", relevant_ids=("a",))]
    result = evaluate(retriever, dataset, k=5)

    assert isinstance(result, EvaluationResult)
    assert isinstance(result.per_query[0], QueryEvalResult)
    assert result.k == 5


def test_repr_format():
    retriever = make_retriever("a")
    dataset = [EvalSample(query="q", relevant_ids=("a",))]
    result = evaluate(retriever, dataset, k=5)
    r = repr(result)

    assert "queries=1" in r
    assert "k=5" in r
    assert "MRR:" in r
    assert "Recall@5:" in r
    # Check 3 decimal places format
    assert "1.000" in r


def test_repr_three_decimal_places():
    def retriever(query: str, top_k: int = 5):
        # RR = 1/3
        return [
            RetrievalResult(text="x", score=0.9, id="x", rank=0),
            RetrievalResult(text="y", score=0.8, id="y", rank=1),
            RetrievalResult(text="a", score=0.7, id="a", rank=2),
        ]

    dataset = [EvalSample(query="q", relevant_ids=("a",))]
    result = evaluate(retriever, dataset, k=5)
    r = repr(result)

    assert "0.333" in r
