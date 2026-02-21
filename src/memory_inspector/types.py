from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class ScoredResult:
    content: str
    score: float
    rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    document_id: str | None = None


@dataclass(frozen=True)
class RetrievalRecord:
    id: str
    timestamp: datetime
    query: str
    results: tuple[ScoredResult, ...]
    top_k: int
    latency_ms: float
    metadata: dict[str, Any]

    @staticmethod
    def create(
        query: str,
        results: list[ScoredResult],
        top_k: int,
        latency_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> RetrievalRecord:
        return RetrievalRecord(
            id=uuid.uuid4().hex,
            timestamp=datetime.now(tz=timezone.utc),
            query=query,
            results=tuple(results),
            top_k=top_k,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )

    def __repr__(self) -> str:
        lines = [
            "RetrievalRecord(",
            f"  query={self.query!r} | latency={self.latency_ms:.1f}ms | {len(self.results)} results",
        ]
        for r in self.results:
            rank_label = r.rank if r.rank is not None else "?"
            truncated = r.content[:60] + "..." if len(r.content) > 60 else r.content
            lines.append(f"  [{rank_label}] score={r.score:.3f}  {truncated}")
        lines.append(")")
        return "\n".join(lines)
