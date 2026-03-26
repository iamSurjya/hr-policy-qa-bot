from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # App
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")

    # LLM provider switch — "gemini" or "claude"
    llm_provider: str = Field(default="gemini")

    # Gemini
    gemini_api_key: str = Field(default="")
    gemini_model: str = Field(default="gemini-2.0-flash-lite")

    # Anthropic
    anthropic_api_key: str = Field(default="")
    claude_model: str = Field(default="claude-sonnet-4-6")

    # Groq
    groq_api_key: str = Field(default="")
    groq_model: str = Field(default="llama-3.3-70b-versatile")

    # Paths
    chroma_path: str = Field(default="./data/vector_store")
    policy_docs_path: str = Field(default="./data/policy_docs")

    # Embeddings
    embedding_model: str = Field(default="all-MiniLM-L6-v2")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()