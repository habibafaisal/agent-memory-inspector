from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from memory_inspector.types import RetrievalResult, ScoredResult


class BaseRetrieverAdapter(ABC):
    @abstractmethod
    def normalize(self, raw_output: Any) -> list[RetrievalResult]: ...


class DefaultAdapter(BaseRetrieverAdapter):
    """Handles list[ScoredResult] and list[RetrievalResult]."""

    def normalize(self, raw_output: Any) -> list[RetrievalResult]:
        if not isinstance(raw_output, list):
            raise TypeError(f"Expected list, got {type(raw_output).__name__}")
        results: list[RetrievalResult] = []
        for item in raw_output:
            if isinstance(item, RetrievalResult):
                results.append(item)
            elif isinstance(item, ScoredResult):
                results.append(item.to_retrieval_result())
            else:
                raise TypeError(
                    f"Expected ScoredResult or RetrievalResult, got {type(item).__name__}"
                )
        return results
