import pytest

from memory_inspector import Inspector, ScoredResult
from memory_inspector.stores import InMemoryStore
from memory_inspector.types import RetrievalRecord


def test_query_returns_record(retriever):
    inspector = Inspector(retriever)
    record = inspector.query("pricing policy")
    assert isinstance(record, RetrievalRecord)
    assert record.query == "pricing policy"
    assert record.latency_ms >= 0


def test_query_latency_is_positive(retriever):
    inspector = Inspector(retriever)
    record = inspector.query("test")
    assert record.latency_ms >= 0


def test_last_returns_most_recent(retriever):
    inspector = Inspector(retriever)
    assert inspector.last() is None
    record = inspector.query("first")
    assert inspector.last() is record
    record2 = inspector.query("second")
    assert inspector.last() is record2


def test_rank_auto_assigned_when_none(retriever):
    inspector = Inspector(retriever)
    record = inspector.query("test", top_k=3)
    for i, r in enumerate(record.results):
        assert r.rank == i


def test_rank_preserved_when_explicit():
    def ranked_retriever(query: str, top_k: int = 5) -> list[ScoredResult]:
        return [
            ScoredResult(content="a", score=0.9, rank=0),
            ScoredResult(content="b", score=0.8, rank=0),  # intentional rank=0
        ]

    inspector = Inspector(ranked_retriever)
    record = inspector.query("test")
    assert record.results[0].rank == 0
    assert record.results[1].rank == 0


def test_raises_on_non_list_retriever():
    def bad_retriever(query: str, top_k: int = 5):  # type: ignore[return]
        return "not a list"

    inspector = Inspector(bad_retriever)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="list\\[ScoredResult\\]"):
        inspector.query("test")


def test_raises_on_wrong_item_type():
    def bad_retriever(query: str, top_k: int = 5):  # type: ignore[return]
        return [{"content": "x", "score": 0.5}]

    inspector = Inspector(bad_retriever)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ScoredResult"):
        inspector.query("test")


def test_sampling_always_stores(retriever):
    store = InMemoryStore()
    inspector = Inspector(retriever, store=store, random_fn=lambda: 0.0)
    inspector.query("a")
    inspector.query("b")
    assert store.count() == 2


def test_sampling_never_stores(retriever):
    store = InMemoryStore()
    inspector = Inspector(retriever, store=store, random_fn=lambda: 1.0)
    inspector.query("a")
    inspector.query("b")
    assert store.count() == 0


def test_last_always_set_regardless_of_sampling(retriever):
    store = InMemoryStore()
    inspector = Inspector(retriever, store=store, random_fn=lambda: 1.0)
    record = inspector.query("test")
    assert inspector.last() is record
    assert store.count() == 0


def test_history_returns_recent(retriever):
    store = InMemoryStore()
    inspector = Inspector(retriever, store=store, random_fn=lambda: 0.0)
    inspector.query("first")
    inspector.query("second")
    history = inspector.history(limit=10)
    assert len(history) == 2
    assert history[0].query == "second"  # most recent first
