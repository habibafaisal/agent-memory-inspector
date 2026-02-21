from __future__ import annotations

import random as _random
import time
from typing import Any, Callable

from memory_inspector.config import InspectorConfig
from memory_inspector.stores import InMemoryStore, SQLiteStore
from memory_inspector.types import RetrievalRecord, ScoredResult


class Inspector:
    def __init__(
        self,
        retriever: Callable[..., list[ScoredResult]],
        *,
        config: InspectorConfig | None = None,
        store: InMemoryStore | SQLiteStore | None = None,
        random_fn: Callable[[], float] = _random.random,
    ) -> None:
        self._retriever = retriever
        self._config = config or InspectorConfig()
        self._random_fn = random_fn
        self._last: RetrievalRecord | None = None

        if store is not None:
            self._store: InMemoryStore | SQLiteStore = store
        elif self._config.store_path:
            self._store = InMemoryStore(max_records=self._config.max_records)
        else:
            self._store = InMemoryStore(max_records=self._config.max_records)

    def query(self, query: str, top_k: int = 5, **kwargs: Any) -> RetrievalRecord:
        start = time.perf_counter()
        raw = self._retriever(query, top_k=top_k, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        if not isinstance(raw, list):
            raise TypeError(
                f"Retriever must return list[ScoredResult], got {type(raw).__name__}"
            )

        results: list[ScoredResult] = []
        for i, item in enumerate(raw):
            if not isinstance(item, ScoredResult):
                raise TypeError(
                    f"Retriever result[{i}] must be ScoredResult, got {type(item).__name__}"
                )
            if not isinstance(item.score, (int, float)):
                raise ValueError(
                    f"ScoredResult[{i}].score must be numeric, got {type(item.score).__name__}"
                )
            if item.rank is None:
                item = ScoredResult(
                    content=item.content,
                    score=item.score,
                    rank=i,
                    metadata=item.metadata,
                    document_id=item.document_id,
                )
            results.append(item)

        record = RetrievalRecord.create(
            query=query,
            results=results,
            top_k=top_k,
            latency_ms=latency_ms,
        )

        self._last = record

        if self._random_fn() < self._config.sample_rate:
            self._store.save(record)

        return record

    def last(self) -> RetrievalRecord | None:
        return self._last

    def history(self, limit: int = 20) -> list[RetrievalRecord]:
        return self._store.list(limit=limit)
