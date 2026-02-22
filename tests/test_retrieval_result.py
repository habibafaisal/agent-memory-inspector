import pytest

from memory_inspector.types import RetrievalResult, ScoredResult


def test_retrieval_result_defaults():
    r = RetrievalResult(text="hello")
    assert r.score is None
    assert r.id is None
    assert r.rank is None
    assert r.metadata == {}


def test_retrieval_result_all_fields():
    r = RetrievalResult(text="hello", score=0.9, id="doc-1", rank=0, metadata={"k": "v"})
    assert r.text == "hello"
    assert r.score == 0.9
    assert r.id == "doc-1"
    assert r.rank == 0
    assert r.metadata == {"k": "v"}


def test_retrieval_result_is_frozen():
    r = RetrievalResult(text="hello", score=0.9)
    with pytest.raises(Exception):
        r.score = 0.5  # type: ignore[misc]


def test_scored_result_to_retrieval_result_full():
    sr = ScoredResult(
        content="hello", score=0.9, rank=1, document_id="doc-1", metadata={"k": "v"}
    )
    rr = sr.to_retrieval_result()
    assert rr.text == "hello"
    assert rr.score == 0.9
    assert rr.rank == 1
    assert rr.id == "doc-1"
    assert rr.metadata == {"k": "v"}


def test_scored_result_to_retrieval_result_minimal():
    sr = ScoredResult(content="text", score=0.5)
    rr = sr.to_retrieval_result()
    assert rr.text == "text"
    assert rr.score == 0.5
    assert rr.id is None
    assert rr.rank is None
    assert rr.metadata == {}


def test_retrieval_result_none_score_repr():
    from memory_inspector.types import RetrievalRecord

    results = [RetrievalResult(text="hello", score=None, rank=0)]
    record = RetrievalRecord.create(query="q", results=results, top_k=1, latency_ms=1.0)
    text = repr(record)
    assert "None" in text
