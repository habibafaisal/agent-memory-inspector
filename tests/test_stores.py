from memory_inspector.stores import InMemoryStore, SQLiteStore
from memory_inspector.types import RetrievalRecord, ScoredResult


def make_record(query: str = "test") -> RetrievalRecord:
    return RetrievalRecord.create(
        query=query,
        results=[ScoredResult(content="result", score=0.9, rank=0)],
        top_k=5,
        latency_ms=1.0,
    )


# --- InMemoryStore ---

def test_inmemory_save_and_count():
    store = InMemoryStore()
    store.save(make_record("a"))
    assert store.count() == 1


def test_inmemory_list_most_recent_first():
    store = InMemoryStore()
    store.save(make_record("first"))
    store.save(make_record("second"))
    records = store.list()
    assert records[0].query == "second"
    assert records[1].query == "first"


def test_inmemory_get_by_id():
    store = InMemoryStore()
    record = make_record()
    store.save(record)
    assert store.get(record.id) is record


def test_inmemory_get_missing():
    store = InMemoryStore()
    assert store.get("nonexistent") is None


def test_inmemory_clear():
    store = InMemoryStore()
    store.save(make_record())
    store.clear()
    assert store.count() == 0


def test_inmemory_ring_buffer_evicts_oldest():
    store = InMemoryStore(max_records=3)
    records = [make_record(f"q{i}") for i in range(5)]
    for r in records:
        store.save(r)
    assert store.count() == 3
    listed = store.list(limit=10)
    queries = [r.query for r in listed]
    assert "q0" not in queries
    assert "q1" not in queries
    assert "q4" in queries


def test_inmemory_list_limit():
    store = InMemoryStore()
    for i in range(10):
        store.save(make_record(f"q{i}"))
    assert len(store.list(limit=3)) == 3


# --- SQLiteStore ---

def test_sqlite_save_and_count(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.db"))
    store.save(make_record("a"))
    assert store.count() == 1
    store.close()


def test_sqlite_list_most_recent_first(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.db"))
    store.save(make_record("first"))
    store.save(make_record("second"))
    records = store.list()
    assert records[0].query == "second"
    store.close()


def test_sqlite_get_by_id(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.db"))
    record = make_record()
    store.save(record)
    fetched = store.get(record.id)
    assert fetched is not None
    assert fetched.id == record.id
    assert fetched.query == record.query
    store.close()


def test_sqlite_get_missing(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.db"))
    assert store.get("nonexistent") is None
    store.close()


def test_sqlite_clear(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.db"))
    store.save(make_record())
    store.clear()
    assert store.count() == 0
    store.close()


def test_sqlite_persists_across_instances(tmp_path):
    db = str(tmp_path / "test.db")
    record = make_record("persistent query")

    store1 = SQLiteStore(db)
    store1.save(record)
    store1.close()

    store2 = SQLiteStore(db)
    assert store2.count() == 1
    fetched = store2.get(record.id)
    assert fetched is not None
    assert fetched.query == "persistent query"
    store2.close()


def test_sqlite_round_trips_scored_result_fields(tmp_path):
    db = str(tmp_path / "test.db")
    record = RetrievalRecord.create(
        query="round trip",
        results=[
            ScoredResult(
                content="hello",
                score=0.75,
                rank=2,
                metadata={"source": "wiki"},
                document_id="doc-123",
            )
        ],
        top_k=10,
        latency_ms=42.5,
        metadata={"session": "abc"},
    )

    store = SQLiteStore(db)
    store.save(record)
    fetched = store.get(record.id)
    assert fetched is not None

    r = fetched.results[0]
    assert r.content == "hello"
    assert r.score == 0.75
    assert r.rank == 2
    assert r.metadata == {"source": "wiki"}
    assert r.document_id == "doc-123"
    assert fetched.metadata == {"session": "abc"}
    store.close()
