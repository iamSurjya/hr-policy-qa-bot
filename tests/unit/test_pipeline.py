from hr_bot.generation.providers import LLMProvider
from hr_bot.retrieval.searcher import SearchResult
from hr_bot.pipeline import run_pipeline

class MockLLM(LLMProvider):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return "Mocked answer"

def test_pipeline_basic(monkeypatch):
    # Mock search
    def mock_search(*args, **kwargs):
        return [
            SearchResult(
                content="Employees get 20 days leave",
                source="policy.md",
                title="Leave Policy",
                chunk_index=0,
                similarity=0.9
            ),
            SearchResult(
                content="Extra policy info",
                source="policy.md",
                title="Leave Policy",
                chunk_index=1,
                similarity=0.85
            )
        ]

    monkeypatch.setattr("hr_bot.pipeline.search", mock_search)

    # Run pipeline
    response = run_pipeline(
        question="What is leave policy?",
        llm=MockLLM()
    )

    # Assertions
    assert response.answer == "Mocked answer"

    #  Important checks
    assert isinstance(response.sources, list)
    assert "policy.md" in response.sources

    # Ensure no duplicates (if your pipeline deduplicates)
    assert len(set(response.sources)) == len(response.sources)

    # chunks_used should match number of results used
    assert response.chunks_used == 2

def test_pipeline_empty_search(monkeypatch):
    # Mock empty search
    def mock_search(*args, **kwargs):
        return []

    monkeypatch.setattr("hr_bot.pipeline.search", mock_search)

    # Run pipeline
    response = run_pipeline(
        question="Random question with no answer",
        llm=MockLLM()
    )

    # Assertions
    assert response.answer is not None
    assert isinstance(response.answer, str)
    assert len(response.sources) == 0 or response.sources == []

def test_pipeline_llm_failure(monkeypatch):
    from hr_bot.retrieval.searcher import SearchResult

    # Mock search (valid results)
    def mock_search(*args, **kwargs):
        return [
            SearchResult(
                content="Employees get 20 days leave",
                source="policy.md",
                title="Leave Policy",
                chunk_index=0,
                similarity=0.9
            )
        ]

    monkeypatch.setattr("hr_bot.pipeline.search", mock_search)

    # Mock LLM that fails
    class FailingLLM(MockLLM):
        def generate(self, system_prompt: str, user_prompt: str) -> str:
            raise RuntimeError("LLM API failure")

    # Run + Assert
    try:
        run_pipeline(
            question="What is leave policy?",
            llm=FailingLLM()
        )
        assert False, "Expected exception not raised"
    except RuntimeError as e:
        assert "LLM API failure" in str(e)