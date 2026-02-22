import pytest

from retric.adapters.langchain import LangChainAdapter


class FakeDoc:
    def __init__(self, content: str, metadata: dict | None = None) -> None:
        self.page_content = content
        self.metadata = metadata or {}


def test_langchain_adapter_plain_documents():
    adapter = LangChainAdapter()
    docs = [FakeDoc("hello world", {"source": "wiki"}), FakeDoc("foo bar")]
    results = adapter.normalize(docs)
    assert len(results) == 2
    assert results[0].text == "hello world"
    assert results[0].score is None
    assert results[0].metadata == {"source": "wiki"}
    assert results[1].text == "foo bar"


def test_langchain_adapter_tuple_with_score():
    adapter = LangChainAdapter()
    docs = [(FakeDoc("hello", {"id": "doc-1"}), 0.95), (FakeDoc("world"), 0.70)]
    results = adapter.normalize(docs)
    assert len(results) == 2
    assert results[0].text == "hello"
    assert results[0].score == 0.95
    assert results[1].score == 0.70


def test_langchain_adapter_extracts_id_from_metadata():
    adapter = LangChainAdapter()
    docs = [FakeDoc("text", {"id": "doc-42"})]
    results = adapter.normalize(docs)
    assert results[0].id == "doc-42"


def test_langchain_adapter_no_id():
    adapter = LangChainAdapter()
    docs = [FakeDoc("text", {})]
    results = adapter.normalize(docs)
    assert results[0].id is None


def test_langchain_adapter_raises_on_non_list():
    adapter = LangChainAdapter()
    with pytest.raises(TypeError, match="Expected list"):
        adapter.normalize("not a list")  # type: ignore[arg-type]


def test_langchain_adapter_empty_list():
    adapter = LangChainAdapter()
    assert adapter.normalize([]) == []
