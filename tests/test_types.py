from datetime import datetime, timezone

from memory_inspector.types import RetrievalRecord, RetrievalResult, ScoredResult


def test_scored_result_defaults():
    r = ScoredResult(content="hello", score=0.9)
    assert r.rank is None
    assert r.metadata == {}
    assert r.document_id is None


def test_scored_result_explicit_rank_zero():
    r = ScoredResult(content="hello", score=0.9, rank=0)
    assert r.rank == 0


def test_scored_result_to_retrieval_result():
    sr = ScoredResult(content="hello", score=0.9, rank=1, document_id="doc-1", metadata={"k": "v"})
    rr = sr.to_retrieval_result()
    assert rr.text == "hello"
    assert rr.score == 0.9
    assert rr.rank == 1
    assert rr.id == "doc-1"
    assert rr.metadata == {"k": "v"}


def test_retrieval_result_defaults():
    r = RetrievalResult(text="hello")
    assert r.score is None
    assert r.id is None
    assert r.rank is None
    assert r.metadata == {}


def test_retrieval_record_create_fills_id_and_timestamp():
    results = [RetrievalResult(text="x", score=0.5, rank=0)]
    record = RetrievalRecord.create(query="test", results=results, top_k=5, latency_ms=10.0)
    assert len(record.id) == 32  # uuid4 hex
    assert isinstance(record.timestamp, datetime)
    assert record.timestamp.tzinfo == timezone.utc


def test_retrieval_record_create_stores_results_as_tuple():
    results = [RetrievalResult(text="a", score=0.1, rank=0)]
    record = RetrievalRecord.create(query="q", results=results, top_k=1, latency_ms=1.0)
    assert isinstance(record.results, tuple)
    assert record.results[0].text == "a"


def test_retrieval_record_repr_format():
    results = [
        RetrievalResult(text="Our pricing starts at $10/mo", score=0.92, rank=0),
        RetrievalResult(text="Enterprise pricing available", score=0.87, rank=1),
    ]
    record = RetrievalRecord.create(query="pricing", results=results, top_k=5, latency_ms=3.2)
    text = repr(record)
    assert "pricing" in text
    assert "3.2ms" in text
    assert "0.920" in text
    assert "0.870" in text
    assert "[0]" in text
    assert "[1]" in text


def test_retrieval_record_repr_truncates_long_content():
    long_text = "x" * 100
    results = [RetrievalResult(text=long_text, score=0.5, rank=0)]
    record = RetrievalRecord.create(query="q", results=results, top_k=1, latency_ms=1.0)
    text = repr(record)
    assert "..." in text


def test_retrieval_record_repr_handles_none_score():
    results = [RetrievalResult(text="hello", score=None, rank=0)]
    record = RetrievalRecord.create(query="q", results=results, top_k=1, latency_ms=1.0)
    text = repr(record)
    assert "None" in text
