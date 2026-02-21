from datetime import datetime, timezone

from memory_inspector.types import RetrievalRecord, ScoredResult


def test_scored_result_defaults():
    r = ScoredResult(content="hello", score=0.9)
    assert r.rank is None
    assert r.metadata == {}
    assert r.document_id is None


def test_scored_result_explicit_rank_zero():
    r = ScoredResult(content="hello", score=0.9, rank=0)
    assert r.rank == 0


def test_retrieval_record_create_fills_id_and_timestamp():
    results = [ScoredResult(content="x", score=0.5, rank=0)]
    record = RetrievalRecord.create(query="test", results=results, top_k=5, latency_ms=10.0)
    assert len(record.id) == 32  # uuid4 hex
    assert isinstance(record.timestamp, datetime)
    assert record.timestamp.tzinfo == timezone.utc


def test_retrieval_record_create_stores_results_as_tuple():
    results = [ScoredResult(content="a", score=0.1, rank=0)]
    record = RetrievalRecord.create(query="q", results=results, top_k=1, latency_ms=1.0)
    assert isinstance(record.results, tuple)
    assert record.results[0].content == "a"


def test_retrieval_record_repr_format():
    results = [
        ScoredResult(content="Our pricing starts at $10/mo", score=0.92, rank=0),
        ScoredResult(content="Enterprise pricing available", score=0.87, rank=1),
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
    long_content = "x" * 100
    results = [ScoredResult(content=long_content, score=0.5, rank=0)]
    record = RetrievalRecord.create(query="q", results=results, top_k=1, latency_ms=1.0)
    text = repr(record)
    assert "..." in text
