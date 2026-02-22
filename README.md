# agent-memory-inspector

The missing debugger for vector retrieval. Inspect scores, compare retrievers, and surface rank shifts in under 60 seconds.

## Install

```bash
pip install agent-memory-inspector
```

Development:

```bash
pip install -e ".[dev]"
```

Optional framework adapters:

```bash
pip install -e ".[langchain]"
pip install -e ".[llamaindex]"
```

---

## 1. Basic retrieval inspection with `RetrievalResult`

```python
from memory_inspector import Inspector, RetrievalResult

def my_retriever(query: str, top_k: int = 5) -> list[RetrievalResult]:
    return [
        RetrievalResult(text="Our pricing starts at $10/mo", score=0.92),
        RetrievalResult(text="Enterprise pricing available on request", score=0.87),
        RetrievalResult(text="Contact sales for custom plans", score=0.45),
    ][:top_k]

inspector = Inspector(my_retriever)
result = inspector.query("pricing policy")
print(result)
```

Output:

```
RetrievalRecord(
  query='pricing policy' | latency=0.1ms | 3 results
  [0] score=0.920  Our pricing starts at $10/mo
  [1] score=0.870  Enterprise pricing available on request
  [2] score=0.450  Contact sales for custom plans
)
```

`ScoredResult` (v0.1) is still supported — the inspector converts it automatically.

---

## 2. Comparing two retrievers with `compare()`

The headline feature of v0.2. Pass any two callables and get a side-by-side breakdown of rank shifts, score deltas, dropped docs, and new candidates.

```python
from memory_inspector import compare, RetrievalResult

def embedding_retriever(query: str, top_k: int = 5) -> list[RetrievalResult]:
    return [
        RetrievalResult(text="RAG uses retrieval + generation", score=0.91, id="doc-a"),
        RetrievalResult(text="Vector databases store embeddings", score=0.85, id="doc-b"),
        RetrievalResult(text="LLMs can hallucinate facts", score=0.72, id="doc-c"),
    ][:top_k]

def reranker_retriever(query: str, top_k: int = 5) -> list[RetrievalResult]:
    return [
        RetrievalResult(text="Vector databases store embeddings", score=0.95, id="doc-b"),
        RetrievalResult(text="RAG uses retrieval + generation", score=0.88, id="doc-a"),
        RetrievalResult(text="Chunking strategy affects recall", score=0.61, id="doc-d"),
    ][:top_k]

result = compare("what is RAG?", embedding_retriever, reranker_retriever)
print(result)
```

Output:

```
ComparisonResult(query='what is RAG?')
  retriever_a: 3 results (0.1ms)
  retriever_b: 3 results (0.2ms)
  Deltas:
    'RAG uses retrieval + generation': rank 0 → 1  score 0.91 → 0.88  (demoted)
    'Vector databases store embeddings': rank 1 → 0  score 0.85 → 0.95  (promoted)
    'LLMs can hallucinate facts': rank 2 → —  (dropped)
    'Chunking strategy affects recall': rank — → 2  (new)
```

Delta statuses: `promoted`, `demoted`, `dropped`, `new`, `unchanged`.

Matching is by `id` first, falling back to exact `text` match.

---

## 3. LangChain adapter

```python
from memory_inspector import compare, LangChainAdapter

# retriever_a returns list[Document], retriever_b returns list[tuple[Document, float]]
result = compare(
    "my query",
    langchain_retriever_a,
    langchain_retriever_b,
    adapter_a=LangChainAdapter(),
    adapter_b=LangChainAdapter(),
)
```

The `LangChainAdapter` handles both `list[Document]` (no scores) and `list[tuple[Document, float]]` (with scores) via duck-typing — no hard dependency on `langchain` unless you install `.[langchain]`.

---

## 4. LlamaIndex adapter

```python
from memory_inspector import compare, LlamaIndexAdapter

result = compare(
    "my query",
    llamaindex_retriever_a,
    llamaindex_retriever_b,
    adapter_a=LlamaIndexAdapter(),
    adapter_b=LlamaIndexAdapter(),
)
```

The `LlamaIndexAdapter` handles `list[NodeWithScore]`, accessing `.node.get_content()`, `.score`, `.node.node_id`, and `.node.metadata` via duck-typing.

---

## 5. Query history with SQLiteStore

```python
from memory_inspector import Inspector, RetrievalResult, SQLiteStore

store = SQLiteStore(".memory_inspector/traces.db")
inspector = Inspector(my_retriever, store=store)

inspector.query("pricing policy")
inspector.query("refund process")

history = inspector.history(limit=10)
for record in history:
    print(record.query, record.latency_ms)
```

---

## API reference

### `Inspector(retriever, *, config=None, store=None)`

- `retriever`: callable with signature `(query: str, top_k: int) -> list[RetrievalResult | ScoredResult]`
- `config`: `InspectorConfig` (optional)
- `store`: `InMemoryStore` or `SQLiteStore` (optional, defaults to in-memory)

### `inspector.query(query, top_k=5) -> RetrievalRecord`

Calls the retriever, records latency, stores the trace (subject to `sample_rate`).

### `compare(query, retriever_a, retriever_b, *, top_k=5, adapter_a=None, adapter_b=None) -> ComparisonResult`

Runs both retrievers, normalizes output, computes rank deltas.

### `RetrievalResult`

```python
@dataclass(frozen=True)
class RetrievalResult:
    text: str
    score: float | None = None
    id: str | None = None
    rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

### `ComparisonResult`

```python
@dataclass(frozen=True)
class ComparisonResult:
    query: str
    results_a: tuple[RetrievalResult, ...]
    results_b: tuple[RetrievalResult, ...]
    latency_a_ms: float
    latency_b_ms: float
    deltas: tuple[RankDelta, ...]
```

### `InspectorConfig`

```python
config = InspectorConfig(
    mode=Mode.DEV,       # DEV or PROD
    sample_rate=1.0,     # 1.0 = log all, 0.1 = log 10%
    store_path=None,     # set to a path to use SQLiteStore by default
    max_records=1000,    # ring buffer cap (InMemoryStore)
)
```
