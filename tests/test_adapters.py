import pytest

from memory_inspector.adapters.base import DefaultAdapter
from memory_inspector.types import RetrievalResult, ScoredResult


def test_default_adapter_handles_retrieval_result():
    adapter = DefaultAdapter()
    items = [RetrievalResult(text="hello", score=0.9)]
    result = adapter.normalize(items)
    assert len(result) == 1
    assert result[0].text == "hello"
    assert result[0].score == 0.9


def test_default_adapter_handles_scored_result():
    adapter = DefaultAdapter()
    items = [ScoredResult(content="hello", score=0.9, document_id="doc-1")]
    result = adapter.normalize(items)
    assert len(result) == 1
    assert result[0].text == "hello"
    assert result[0].score == 0.9
    assert result[0].id == "doc-1"


def test_default_adapter_raises_on_non_list():
    adapter = DefaultAdapter()
    with pytest.raises(TypeError, match="Expected list"):
        adapter.normalize("not a list")  # type: ignore[arg-type]


def test_default_adapter_raises_on_wrong_item_type():
    adapter = DefaultAdapter()
    with pytest.raises(TypeError, match="ScoredResult or RetrievalResult"):
        adapter.normalize([{"text": "bad"}])  # type: ignore[list-item]


def test_default_adapter_mixed_types():
    adapter = DefaultAdapter()
    items: list[RetrievalResult | ScoredResult] = [
        RetrievalResult(text="a", score=0.9),
        ScoredResult(content="b", score=0.8),
    ]
    result = adapter.normalize(items)
    assert len(result) == 2
    assert result[0].text == "a"
    assert result[1].text == "b"


def test_default_adapter_empty_list():
    adapter = DefaultAdapter()
    result = adapter.normalize([])
    assert result == []
