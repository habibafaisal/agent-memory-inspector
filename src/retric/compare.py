from __future__ import annotations

import time
from typing import Any, Callable

from retric.adapters.base import BaseRetrieverAdapter, DefaultAdapter
from retric.types import ComparisonResult, RankDelta, RetrievalResult


def _assign_ranks(results: list[RetrievalResult]) -> list[RetrievalResult]:
    """Assign ranks based on list position where missing."""
    return [
        r
        if r.rank is not None
        else RetrievalResult(text=r.text, score=r.score, id=r.id, rank=i, metadata=r.metadata)
        for i, r in enumerate(results)
    ]


def _classify_status(rank_before: int | None, rank_after: int | None) -> str:
    if rank_before is None:
        return "new"
    if rank_after is None:
        return "dropped"
    if rank_before == rank_after:
        return "unchanged"
    if rank_after < rank_before:
        return "promoted"
    return "demoted"


def compare(
    query: str,
    retriever_a: Callable[..., Any],
    retriever_b: Callable[..., Any],
    *,
    top_k: int = 5,
    adapter_a: BaseRetrieverAdapter | None = None,
    adapter_b: BaseRetrieverAdapter | None = None,
) -> ComparisonResult:
    """Compare two retrievers side-by-side on a single query."""
    default = DefaultAdapter()
    norm_a = adapter_a or default
    norm_b = adapter_b or default

    start_a = time.perf_counter()
    raw_a = retriever_a(query, top_k=top_k)
    latency_a_ms = (time.perf_counter() - start_a) * 1000

    start_b = time.perf_counter()
    raw_b = retriever_b(query, top_k=top_k)
    latency_b_ms = (time.perf_counter() - start_b) * 1000

    results_a = _assign_ranks(norm_a.normalize(raw_a))
    results_b = _assign_ranks(norm_b.normalize(raw_b))

    def build_index(results: list[RetrievalResult]) -> dict[str, RetrievalResult]:
        index: dict[str, RetrievalResult] = {}
        for r in results:
            key = r.id if r.id is not None else r.text
            index[key] = r
        return index

    index_a = build_index(results_a)
    index_b = build_index(results_b)

    all_keys = set(index_a) | set(index_b)
    deltas: list[RankDelta] = []
    for key in all_keys:
        ra = index_a.get(key)
        rb = index_b.get(key)
        rank_before = ra.rank if ra is not None else None
        rank_after = rb.rank if rb is not None else None
        score_before = ra.score if ra is not None else None
        score_after = rb.score if rb is not None else None
        text = ra.text if ra is not None else (rb.text if rb is not None else key)
        doc_id = (ra.id if ra is not None else None) or (rb.id if rb is not None else None)
        status = _classify_status(rank_before, rank_after)
        deltas.append(
            RankDelta(
                text=text,
                id=doc_id,
                rank_before=rank_before,
                rank_after=rank_after,
                score_before=score_before,
                score_after=score_after,
                status=status,
            )
        )

    def _delta_sort_key(d: RankDelta) -> tuple[int, int, str]:
        before = d.rank_before if d.rank_before is not None else 9999
        after = d.rank_after if d.rank_after is not None else 9999
        return (min(before, after), max(before, after), d.text)

    deltas.sort(key=_delta_sort_key)

    return ComparisonResult(
        query=query,
        results_a=tuple(results_a),
        results_b=tuple(results_b),
        latency_a_ms=latency_a_ms,
        latency_b_ms=latency_b_ms,
        deltas=tuple(deltas),
    )
