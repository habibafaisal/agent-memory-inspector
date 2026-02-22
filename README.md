# retric

[![PyPI](https://img.shields.io/pypi/v/retric)](https://pypi.org/project/retric/)

Inspect scores, compare retrievers side-by-side, and measure retrieval quality with MRR and Recall@k — in under 60 seconds.

## Install

```bash
pip install retric
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
from retric import Inspector, RetrievalResult

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

Pass any two callables and get a side-by-side breakdown of rank shifts, score deltas, dropped docs, and new candidates.

```python
from retric import compare, RetrievalResult

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
from retric import compare, LangChainAdapter

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
from retric import compare, LlamaIndexAdapter

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
from retric import Inspector, RetrievalResult, SQLiteStore

store = SQLiteStore(".retric/traces.db")
inspector = Inspector(my_retriever, store=store)

inspector.query("pricing policy")
inspector.query("refund process")

history = inspector.history(limit=10)
for record in history:
    print(record.query, record.latency_ms)
```

---

## 6. Evaluating retrieval quality with `evaluate()`

Stop guessing whether your retriever improved. Give it a labeled dataset and get **MRR** and **Recall@k** in one call.

```python
from retric import evaluate, EvalSample, RetrievalResult

# ── tiny knowledge base ──────────────────────────────────────────────────────
_DOCS = {
    "rag-intro":      "RAG grounds LLM outputs in retrieved context, cutting hallucinations",
    "embed-basics":   "Embeddings map text to dense vectors capturing semantic meaning",
    "chunking":       "Chunking strategy determines retrieval granularity and recall",
    "reranking":      "Rerankers rescore retrieved candidates using cross-encoder models",
    "vector-db":      "Vector databases index high-dimensional embeddings for fast ANN search",
    "hybrid-search":  "Hybrid search combines dense and sparse retrieval for better coverage",
    "eval-metrics":   "Recall@k and MRR are standard metrics for retrieval evaluation",
    "context-window": "Context window size limits how much retrieved text an LLM processes",
}

# simulated retrieval table — swap in your real retriever
_RETRIEVAL_TABLE = {
    "how does RAG reduce hallucination?": [
        ("rag-intro", 0.94), ("context-window", 0.71), ("embed-basics", 0.58),
        ("hybrid-search", 0.41), ("chunking", 0.33),
    ],
    "what is a vector database?": [
        ("vector-db", 0.97), ("embed-basics", 0.82), ("hybrid-search", 0.54),
        ("reranking", 0.39), ("rag-intro", 0.31),
    ],
    "how to improve retrieval precision?": [
        ("context-window", 0.72), ("chunking", 0.68), ("reranking", 0.61),
        ("hybrid-search", 0.49), ("eval-metrics", 0.38),
    ],
    "what metrics evaluate retrieval quality?": [
        ("hybrid-search", 0.65), ("rag-intro", 0.52), ("eval-metrics", 0.48),
        ("chunking", 0.41), ("vector-db", 0.33),
    ],
    "how does hybrid search work?": [
        ("hybrid-search", 0.91), ("vector-db", 0.74), ("embed-basics", 0.63),
        ("reranking", 0.52), ("rag-intro", 0.39),
    ],
}

def demo_retriever(query: str, top_k: int = 5) -> list[RetrievalResult]:
    hits = _RETRIEVAL_TABLE[query]
    return [
        RetrievalResult(text=_DOCS[doc_id], score=score, id=doc_id, rank=i)
        for i, (doc_id, score) in enumerate(hits[:top_k])
    ]

# ── labeled dataset ───────────────────────────────────────────────────────────
dataset = [
    EvalSample(query="how does RAG reduce hallucination?",       relevant_ids=("rag-intro",)),
    EvalSample(query="what is a vector database?",               relevant_ids=("vector-db", "embed-basics")),
    EvalSample(query="how to improve retrieval precision?",      relevant_ids=("chunking", "reranking")),
    EvalSample(query="what metrics evaluate retrieval quality?", relevant_ids=("eval-metrics",)),
    EvalSample(query="how does hybrid search work?",             relevant_ids=("hybrid-search",)),
]

result = evaluate(demo_retriever, dataset, k=5)
print(result)
```

Output:

```
EvaluationResult(
  queries=5 | k=5
  MRR:       0.767
  Recall@5:  1.000
)
```

Drill into per-query breakdowns to find exactly where your retriever loses rank:

```python
for qr in result.per_query:
    print(f"  {qr.query[:45]:<45}  RR={qr.reciprocal_rank:.3f}  Recall={qr.recall_at_k:.3f}")
```

```
  how does RAG reduce hallucination?            RR=1.000  Recall=1.000
  what is a vector database?                    RR=1.000  Recall=1.000
  how to improve retrieval precision?           RR=0.500  Recall=1.000
  what metrics evaluate retrieval quality?      RR=0.333  Recall=1.000
  how does hybrid search work?                  RR=1.000  Recall=1.000
```

Everything is recalled within top-5, but two queries miss rank 1 — a clear signal to tune chunking or reranking for those topics. Swap `demo_retriever` for your real one and use `compare()` to confirm the improvement before shipping.

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

### `evaluate(retriever, dataset, *, k=5, adapter=None) -> EvaluationResult`

Benchmarks a retriever against a labeled dataset and returns MRR and Recall@k.

- `retriever`: same callable signature as `Inspector`
- `dataset`: `list[EvalSample]` — each sample pairs a query with ground-truth document IDs
- `k`: cutoff rank for Recall@k and retrieval depth (default 5)
- `adapter`: optional adapter to normalize raw output (defaults to `DefaultAdapter`)

### `EvalSample`

```python
@dataclass(frozen=True)
class EvalSample:
    query: str
    relevant_ids: tuple[str, ...]   # must match RetrievalResult.id values
```

### `EvaluationResult`

```python
@dataclass(frozen=True)
class EvaluationResult:
    mrr: float                              # mean reciprocal rank across all queries
    recall_at_k: float                      # mean Recall@k across all queries
    k: int
    per_query: tuple[QueryEvalResult, ...]  # per-query breakdown
```

### `QueryEvalResult`

```python
@dataclass(frozen=True)
class QueryEvalResult:
    query: str
    reciprocal_rank: float   # 1/rank_of_first_relevant (1-indexed), 0.0 if none found
    recall_at_k: float       # hits in top-k / total relevant
```

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
