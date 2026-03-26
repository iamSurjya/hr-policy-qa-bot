from abc import ABC, abstractmethod
import google.genai as genai
import anthropic
from anthropic.types import TextBlock
import structlog

from hr_bot.config import settings

log = structlog.get_logger()


class LLMProvider(ABC):
    """Abstract base class for all LLM providers.
    
    Every provider must implement generate() with the same signature.
    The rest of the codebase only ever calls this method — it never
    knows or cares which provider is underneath.
    """

    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> str:
        """Generate a response given a prompt and optional system message."""
        ...

class GeminiProvider(LLMProvider):
    """Google Gemini provider — used in development (free tier)."""

    def __init__(self):
        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file."
            )
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_model
        log.info("llm_provider_initialized", provider="gemini", model=self.model)

    def generate(self, prompt: str, system: str = "") -> str:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        response = self.client.models.generate_content(
            model=self.model,
            contents=full_prompt,
        )
        return response.text or ""

class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider — used in production."""

    def __init__(self):
        if not settings.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file."
            )
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.claude_model
        log.info("llm_provider_initialized", provider="claude", model=self.model)

    def generate(self, prompt: str, system: str = "") -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system or "You are a helpful HR policy assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        block = message.content[0]
        if isinstance(block, TextBlock):
            return block.text or ""
        return ""

class GroqProvider(LLMProvider):
    """Groq provider — free tier, used for development.
    
    Runs Llama 3.3 70B on Groq's custom inference hardware.
    Fast, free, and OpenAI-compatible API.
    """

    def __init__(self):
        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file."
            )
        from groq import Groq
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.groq_model
        log.info("llm_provider_initialized", provider="groq", model=self.model)

    def generate(self, prompt: str, system: str = "") -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system or "You are a helpful HR policy assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
        )
        content = response.choices[0].message.content
        return content or ""

def get_llm_provider() -> LLMProvider:
    """Factory function — returns the right provider based on config.
    
    This is the only place in the codebase that knows about the
    LLM_PROVIDER setting. Everything else just calls .generate().
    """
    providers = {
        "gemini": GeminiProvider,
        "claude": ClaudeProvider,
        "groq": GroqProvider,
    }

    provider_name = settings.llm_provider.lower()

    if provider_name not in providers:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider_name}'. "
            f"Valid options: {list(providers.keys())}"
        )

    return providers[provider_name]()