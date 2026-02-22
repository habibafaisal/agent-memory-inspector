from __future__ import annotations

import random as _random
import time
from typing import Any, Callable

from memory_inspector.config import InspectorConfig
from memory_inspector.stores import InMemoryStore, SQLiteStore
from memory_inspector.types import RetrievalRecord, RetrievalResult, ScoredResult


class Inspector:
    def __init__(
        self,
        retriever: Callable[..., Any],
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
            self._store = SQLiteStore(self._config.store_path)
        else:
            self._store = InMemoryStore(max_records=self._config.max_records)

    def query(self, query: str, top_k: int = 5, **kwargs: Any) -> RetrievalRecord:
        start = time.perf_counter()
        raw = self._retriever(query, top_k=top_k, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        if not isinstance(raw, list):
            raise TypeError(
                f"Retriever must return list[ScoredResult] or list[RetrievalResult],"
                f" got {type(raw).__name__}"
            )

        results: list[RetrievalResult] = []
        for i, item in enumerate(raw):
            if isinstance(item, RetrievalResult):
                if item.rank is None:
                    item = RetrievalResult(
                        text=item.text,
                        score=item.score,
                        id=item.id,
                        rank=i,
                        metadata=item.metadata,
                    )
                results.append(item)
            elif isinstance(item, ScoredResult):
                if not isinstance(item.score, (int, float)):
                    raise ValueError(
                        f"ScoredResult[{i}].score must be numeric,"
                        f" got {type(item.score).__name__}"
                    )
                rank = item.rank if item.rank is not None else i
                results.append(
                    RetrievalResult(
                        text=item.content,
                        score=item.score,
                        id=item.document_id,
                        rank=rank,
                        metadata=item.metadata,
                    )
                )
            else:
                raise TypeError(
                    f"Retriever result[{i}] must be ScoredResult or RetrievalResult,"
                    f" got {type(item).__name__}"
                )

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
