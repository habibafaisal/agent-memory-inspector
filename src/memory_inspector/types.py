from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class RetrievalResult:
    text: str
    score: float | None = None
    id: str | None = None
    rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredResult:
    content: str
    score: float
    rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    document_id: str | None = None

    def to_retrieval_result(self) -> RetrievalResult:
        return RetrievalResult(
            text=self.content,
            score=self.score,
            id=self.document_id,
            rank=self.rank,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class RetrievalRecord:
    id: str
    timestamp: datetime
    query: str
    results: tuple[RetrievalResult, ...]
    top_k: int
    latency_ms: float
    metadata: dict[str, Any]

    @staticmethod
    def create(
        query: str,
        results: list[RetrievalResult],
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
            f"  query={self.query!r} | latency={self.latency_ms:.3f}ms | {len(self.results)} results",
        ]
        for r in self.results:
            rank_label = r.rank if r.rank is not None else "?"
            truncated = r.text[:60] + "..." if len(r.text) > 60 else r.text
            score_str = f"{r.score:.3f}" if r.score is not None else "None"
            lines.append(f"  [{rank_label}] score={score_str}  {truncated}")
        lines.append(")")
        return "\n".join(lines)


@dataclass(frozen=True)
class RankDelta:
    text: str
    id: str | None
    rank_before: int | None  # None = new in retriever_b
    rank_after: int | None  # None = dropped from retriever_b
    score_before: float | None
    score_after: float | None
    status: str  # "promoted" | "demoted" | "dropped" | "new" | "unchanged"


@dataclass(frozen=True)
class EvalSample:
    query: str
    relevant_ids: tuple[str, ...]


@dataclass(frozen=True)
class QueryEvalResult:
    query: str
    reciprocal_rank: float
    recall_at_k: float


@dataclass(frozen=True)
class EvaluationResult:
    mrr: float
    recall_at_k: float
    k: int
    per_query: tuple[QueryEvalResult, ...]

    def __repr__(self) -> str:
        recall_label = f"Recall@{self.k}:"
        lines = [
            "EvaluationResult(",
            f"  queries={len(self.per_query)} | k={self.k}",
            f"  {'MRR:':<10} {self.mrr:.3f}",
            f"  {recall_label:<10} {self.recall_at_k:.3f}",
        ]
        if self.per_query:
            worst = min(self.per_query, key=lambda q: q.reciprocal_rank)
            lines.append(f"  {'worst:':<10} {worst.query!r} (RR={worst.reciprocal_rank:.3f})")
        lines.append(")")
        return "\n".join(lines)


@dataclass(frozen=True)
class ComparisonResult:
    query: str
    results_a: tuple[RetrievalResult, ...]
    results_b: tuple[RetrievalResult, ...]
    latency_a_ms: float
    latency_b_ms: float
    deltas: tuple[RankDelta, ...]

    def __repr__(self) -> str:
        lines = [
            f"ComparisonResult(query={self.query!r})",
            f"  retriever_a: {len(self.results_a)} results ({self.latency_a_ms:.3f}ms)",
            f"  retriever_b: {len(self.results_b)} results ({self.latency_b_ms:.3f}ms)",
            "  Deltas:",
        ]
        for d in self.deltas:
            truncated = d.text[:40] + "..." if len(d.text) > 40 else d.text
            rank_a = str(d.rank_before) if d.rank_before is not None else "\u2014"
            rank_b = str(d.rank_after) if d.rank_after is not None else "\u2014"
            if d.status in ("dropped", "new"):
                lines.append(f"    {truncated!r}: rank {rank_a} \u2192 {rank_b}  ({d.status})")
            else:
                score_a = f"{d.score_before:.2f}" if d.score_before is not None else "\u2014"
                score_b = f"{d.score_after:.2f}" if d.score_after is not None else "\u2014"
                lines.append(
                    f"    {truncated!r}: rank {rank_a} \u2192 {rank_b}"
                    f"  score {score_a} \u2192 {score_b}  ({d.status})"
                )
        return "\n".join(lines)
