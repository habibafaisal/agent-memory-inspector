import pytest

from retric.adapters.llamaindex import LlamaIndexAdapter


class FakeNode:
    def __init__(
        self, content: str, node_id: str = "node-1", metadata: dict | None = None
    ) -> None:
        self.node_id = node_id
        self.metadata = metadata or {}
        self._content = content

    def get_content(self) -> str:
        return self._content


class FakeNodeWithScore:
    def __init__(
        self,
        content: str,
        score: float | None,
        node_id: str = "node-1",
        metadata: dict | None = None,
    ) -> None:
        self.node = FakeNode(content, node_id=node_id, metadata=metadata or {})
        self.score = score


def test_llamaindex_adapter_basic():
    adapter = LlamaIndexAdapter()
    items = [FakeNodeWithScore("hello", score=0.85, node_id="n-1")]
    results = adapter.normalize(items)
    assert len(results) == 1
    assert results[0].text == "hello"
    assert results[0].score == 0.85
    assert results[0].id == "n-1"


def test_llamaindex_adapter_none_score():
    adapter = LlamaIndexAdapter()
    items = [FakeNodeWithScore("text", score=None)]
    results = adapter.normalize(items)
    assert results[0].score is None


def test_llamaindex_adapter_metadata():
    adapter = LlamaIndexAdapter()
    items = [FakeNodeWithScore("text", score=0.9, metadata={"source": "wiki"})]
    results = adapter.normalize(items)
    assert results[0].metadata == {"source": "wiki"}


def test_llamaindex_adapter_multiple_nodes():
    adapter = LlamaIndexAdapter()
    items = [
        FakeNodeWithScore("doc A", score=0.9, node_id="a"),
        FakeNodeWithScore("doc B", score=0.7, node_id="b"),
    ]
    results = adapter.normalize(items)
    assert len(results) == 2
    assert results[0].id == "a"
    assert results[1].id == "b"


def test_llamaindex_adapter_raises_on_non_list():
    adapter = LlamaIndexAdapter()
    with pytest.raises(TypeError, match="Expected list"):
        adapter.normalize("not a list")  # type: ignore[arg-type]


def test_llamaindex_adapter_empty_list():
    adapter = LlamaIndexAdapter()
    assert adapter.normalize([]) == []
