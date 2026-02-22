from __future__ import annotations

from typing import Any

from memory_inspector.adapters.base import BaseRetrieverAdapter
from memory_inspector.types import RetrievalResult


class LangChainAdapter(BaseRetrieverAdapter):
    """Handles LangChain Document and (Document, score) tuples — duck-typed, no hard import."""

    def normalize(self, raw_output: Any) -> list[RetrievalResult]:
        if not isinstance(raw_output, list):
            raise TypeError(f"Expected list, got {type(raw_output).__name__}")

        results: list[RetrievalResult] = []
        for item in raw_output:
            if isinstance(item, tuple) and len(item) == 2:
                doc, score = item
                text = doc.page_content
                doc_id = getattr(doc, "id", None) or doc.metadata.get("id")
                metadata = dict(doc.metadata)
                results.append(
                    RetrievalResult(
                        text=text,
                        score=float(score),
                        id=doc_id,
                        metadata=metadata,
                    )
                )
            else:
                text = item.page_content
                doc_id = getattr(item, "id", None) or item.metadata.get("id")
                metadata = dict(item.metadata)
                results.append(RetrievalResult(text=text, id=doc_id, metadata=metadata))
        return results
