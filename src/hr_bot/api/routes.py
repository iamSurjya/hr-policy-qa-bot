import structlog
from fastapi import APIRouter, HTTPException

from hr_bot.api.schemas import HealthResponse, QueryRequest, QueryResponse
from hr_bot.config import settings
from hr_bot.pipeline import run_pipeline

logger = structlog.get_logger()

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    logger.info("api_query_received", question=request.question, k=request.k)

    try:
        response = run_pipeline(question=request.question, k=request.k)
    except Exception as e:
        logger.error("api_query_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Pipeline failed. Check logs.")

    return QueryResponse(
        answer=response.answer,
        sources=response.sources,
        chunks_used=response.chunks_used,
        query=response.query,
    )


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        provider=settings.llm_provider,
        environment=settings.environment,
        index_path=settings.chroma_path,
    )