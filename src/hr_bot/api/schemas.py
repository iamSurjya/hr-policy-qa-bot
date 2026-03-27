# src/hr_bot/api/schemas.py
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: int
    query: str


class HealthResponse(BaseModel):
    status: str
    provider: str
    environment: str
    index_path: str