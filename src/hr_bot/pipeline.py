from dataclasses import dataclass
import structlog

from hr_bot.retrieval.searcher import search, SearchResult
from hr_bot.generation.providers import get_llm_provider, LLMProvider
from hr_bot.generation.prompt import PromptBuilder

log = structlog.get_logger()

# Minimum similarity threshold — chunks below this are likely noise
MIN_SIMILARITY = 0.15


@dataclass
class RAGResponse:
    """The full response from the RAG pipeline.

    We return more than just the answer — we return sources too.
    This is critical for an HR bot: employees need to know which
    policy their answer came from so they can verify it themselves.
    Transparency builds trust in the system.
    """
    answer: str
    sources: list[str]        # unique source filenames used
    chunks_used: int          # how many chunks informed the answer
    query: str                # original question echoed back


def format_context(results: list[SearchResult]) -> str:
    """Format retrieved chunks into a single context string for the LLM.

    Each chunk is labeled with its source document so the LLM
    can reference it naturally in the answer.
    """
    formatted = []
    for result in results:
        formatted.append(
            f"[Source: {result.title}]\n{result.content}"
        )
    return "\n\n---\n\n".join(formatted)


def run_pipeline(
    question: str,
    llm: LLMProvider | None = None,
    k: int = 5,
) -> RAGResponse:
    """Run the full RAG pipeline for a user question.

    Args:
        question: the employee's question
        llm: optional LLM provider — if None, creates from config.
             Accepting it as a parameter makes testing easy:
             tests can inject a mock LLM without touching config.
        k: number of chunks to retrieve

    Returns:
        RAGResponse with answer and source attribution
    """
    if llm is None:
        llm = get_llm_provider()

    log.info("pipeline_started", question=question[:80])

    # Step 1 — Retrieve relevant chunks
    results = search(question, k=k)

    # Step 2 — Filter out low-quality results
    filtered = [r for r in results if r.similarity >= MIN_SIMILARITY]

    if not filtered:
        log.warning("no_relevant_chunks_found", question=question[:80])
        return RAGResponse(
            answer=(
                "I could not find relevant information in the policy "
                "documents to answer your question. Please contact HR "
                "directly for clarification."
            ),
            sources=[],
            chunks_used=0,
            query=question,
        )

    # Step 3 — Format context for the LLM
    context = format_context(filtered)

    # Step 4 — Build prompt
    builder = PromptBuilder()
    system, user_prompt = builder.build(question, context)

    # Step 5 — Generate answer
    answer = llm.generate(user_prompt, system)

    # Step 6 — Collect unique sources for attribution
    sources = list(dict.fromkeys(r.source for r in filtered))

    log.info(
        "pipeline_complete",
        question=question[:80],
        chunks_used=len(filtered),
        sources=sources,
        answer_length=len(answer),
    )

    return RAGResponse(
        answer=answer,
        sources=sources,
        chunks_used=len(filtered),
        query=question,
    )
