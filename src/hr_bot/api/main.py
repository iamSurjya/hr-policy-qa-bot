# src/hr_bot/api/main.py
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from hr_bot.api.routes import router
from hr_bot.config import settings
from hr_bot.generation.providers import LLMProvider, get_llm_provider

logger = structlog.get_logger()

_llm_provider: LLMProvider | None = None


def get_provider() -> LLMProvider:
    if _llm_provider is None:
        raise RuntimeError("LLM provider not initialized. App did not start correctly.")
    return _llm_provider


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _llm_provider

    logger.info("app_startup_begin", environment=settings.environment)

    try:
        _llm_provider = get_llm_provider()
        logger.info("app_startup_complete", provider=settings.llm_provider)
    except Exception as e:
        logger.error("app_startup_failed", error=str(e))
        raise

    yield

    logger.info("app_shutdown")
    _llm_provider = None


app = FastAPI(
    title="HR Policy Q&A Bot",
    version="2.0.0",
    description="RAG-powered HR policy assistant",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")