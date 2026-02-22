from __future__ import annotations

import time
from typing import Any, Callable

from memory_inspector.adapters.base import BaseRetrieverAdapter, DefaultAdapter
from memory_inspector.compare import _assign_ranks
from memory_inspector.types import (
    EvalSample,
    EvaluationResult,
    QueryEvalResult,
    RetrievalResult,
)


def _eval_query(
    results: list[RetrievalResult],
    relevant_ids: frozenset[str],
    k: int,
) -> QueryEvalResult:
    # Sort by rank (0-indexed), take top-k
    sorted_results = sorted(
        (r for r in results if r.rank is not None),
        key=lambda r: r.rank,  # type: ignore[arg-type]
    )[:k]

    # Recall@k: hits among top-k / total relevant
    hits = sum(1 for r in sorted_results if r.id is not None and r.id in relevant_ids)
    recall = hits / len(relevant_ids)

    # Reciprocal Rank: 1-indexed position of first relevant result
    rr = 0.0
    for position, r in enumerate(sorted_results, start=1):
        if r.id is not None and r.id in relevant_ids:
            rr = 1.0 / position
            break

    return QueryEvalResult(
        query="",  # filled by caller
        reciprocal_rank=rr,
        recall_at_k=recall,
    )


def evaluate(
    retriever: Callable[..., Any],
    dataset: list[EvalSample],
    *,
    k: int = 5,
    adapter: BaseRetrieverAdapter | None = None,
) -> EvaluationResult:
    """Evaluate a retriever on a labeled dataset using MRR and Recall@k.

    Args:
        retriever: Callable matching the standard retriever signature
                   ``(query: str, top_k: int = 5, **kwargs) -> list[...]``.
        dataset:   List of labeled queries. Each ``EvalSample`` must have at
                   least one entry in ``relevant_ids``.
        k:         Cutoff for Recall@k and for how many results to retrieve.
        adapter:   Optional adapter to normalize raw retriever output. Defaults
                   to ``DefaultAdapter`` (handles ``RetrievalResult`` /
                   ``ScoredResult``).

    Returns:
        ``EvaluationResult`` with aggregate MRR and Recall@k, plus per-query
        breakdowns.

    Raises:
        ValueError: If any ``EvalSample`` has an empty ``relevant_ids``.
    """
    norm = adapter or DefaultAdapter()
    per_query: list[QueryEvalResult] = []

    for sample in dataset:
        if not sample.relevant_ids:
            raise ValueError(
                f"EvalSample for query {sample.query!r} has empty relevant_ids."
                " Provide at least one relevant document ID."
            )

        relevant = frozenset(sample.relevant_ids)
        raw = retriever(sample.query, top_k=k)
        results = _assign_ranks(norm.normalize(raw))

        qr = _eval_query(results, relevant, k)
        per_query.append(
            QueryEvalResult(
                query=sample.query,
                reciprocal_rank=qr.reciprocal_rank,
                recall_at_k=qr.recall_at_k,
            )
        )

    if not per_query:
        return EvaluationResult(mrr=0.0, recall_at_k=0.0, k=k, per_query=())

    mrr = sum(qr.reciprocal_rank for qr in per_query) / len(per_query)
    mean_recall = sum(qr.recall_at_k for qr in per_query) / len(per_query)

    return EvaluationResult(
        mrr=mrr,
        recall_at_k=mean_recall,
        k=k,
        per_query=tuple(per_query),
    )
