import pytest

from retric.types import ScoredResult


FIXED_RESULTS = [
    ScoredResult(content="Our pricing starts at $10/mo", score=0.92),
    ScoredResult(content="Enterprise pricing available on request", score=0.87),
    ScoredResult(content="Contact sales for custom plans", score=0.45),
]


def fake_retriever(query: str, top_k: int = 5) -> list[ScoredResult]:
    return FIXED_RESULTS[:top_k]


@pytest.fixture
def retriever():
    return fake_retriever
