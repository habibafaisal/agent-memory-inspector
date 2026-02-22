from __future__ import annotations

from typing import Any

from retric.adapters.base import BaseRetrieverAdapter
from retric.types import RetrievalResult


class LlamaIndexAdapter(BaseRetrieverAdapter):
    """Handles LlamaIndex NodeWithScore — duck-typed, no hard import."""

    def normalize(self, raw_output: Any) -> list[RetrievalResult]:
        if not isinstance(raw_output, list):
            raise TypeError(f"Expected list, got {type(raw_output).__name__}")

        results: list[RetrievalResult] = []
        for item in raw_output:
            node = item.node
            text = node.get_content()
            score = item.score
            node_id = getattr(node, "node_id", None)
            metadata = dict(getattr(node, "metadata", {}))
            results.append(
                RetrievalResult(
                    text=text,
                    score=float(score) if score is not None else None,
                    id=node_id,
                    metadata=metadata,
                )
            )
        return results
