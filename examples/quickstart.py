from memory_inspector import Inspector, ScoredResult


def my_retriever(query: str, top_k: int = 5) -> list[ScoredResult]:
    return [
        ScoredResult(content="Our pricing starts at $10/mo", score=0.92),
        ScoredResult(content="Enterprise pricing available on request", score=0.87),
        ScoredResult(content="Contact sales for custom plans", score=0.45),
    ][:top_k]


inspector = Inspector(my_retriever)
result = inspector.query("pricing policy")

print(result)
