# agent-memory-inspector

The missing debugger for vector retrieval. See scores in under 60 seconds.

## Install

```bash
pip install agent-memory-inspector
```

Or in development:

```bash
pip install -e ".[dev]"
```

## Quickstart

```python
from memory_inspector import Inspector, ScoredResult

def my_retriever(query: str, top_k: int = 5) -> list[ScoredResult]:
    # Replace with your real retrieval logic
    return [
        ScoredResult(content="Our pricing starts at $10/mo", score=0.92),
        ScoredResult(content="Enterprise pricing available", score=0.87),
        ScoredResult(content="Contact sales for details", score=0.45),
    ]

inspector = Inspector(my_retriever)
result = inspector.query("pricing policy")
print(result)
```

Output:

```
RetrievalRecord(
  query='pricing policy' | latency=0.1ms | 3 results
  [0] score=0.920  Our pricing starts at $10/mo
  [1] score=0.870  Enterprise pricing available
  [2] score=0.450  Contact sales for details
)
```

## API

### `Inspector(retriever, *, config=None, store=None, random_fn=random.random)`

- `retriever`: any callable with signature `(query: str, top_k: int) -> list[ScoredResult]`
- `config`: `InspectorConfig` (optional)
- `store`: `InMemoryStore` or `SQLiteStore` (optional, defaults to in-memory)
- `random_fn`: override for sampling control (useful in tests)

### `inspector.query(query, top_k=5, **kwargs) -> RetrievalRecord`

Calls the retriever, records latency, stores the trace (subject to `sample_rate`), returns the record.

### `inspector.last() -> RetrievalRecord | None`

The most recent record. Always set, regardless of sampling.

### `inspector.history(limit=20) -> list[RetrievalRecord]`

Recent records from the store, most recent first.

## Config

```python
from memory_inspector import InspectorConfig, Mode

config = InspectorConfig(
    mode=Mode.DEV,       # DEV or PROD
    sample_rate=1.0,     # 1.0 = log all, 0.1 = 10%
    max_records=1000,    # ring buffer cap (InMemoryStore)
)
inspector = Inspector(my_retriever, config=config)
```

## Persistent storage

```python
from memory_inspector import Inspector, SQLiteStore

store = SQLiteStore(".memory_inspector/traces.db")
inspector = Inspector(my_retriever, store=store)
```

## Retriever contract

Your retriever must:
- Accept `(query: str, top_k: int)` as positional/keyword arguments
- Return `list[ScoredResult]`
- Each `ScoredResult` must have a numeric `score`

If `rank` is omitted (`None`), it is auto-assigned from list order.
