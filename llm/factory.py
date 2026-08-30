from config.models import LLMConfig

from .base import DisabledLLMClient, LLMClient
from .openai_compatible import OpenAICompatibleClient


def create_llm_client(config: LLMConfig) -> LLMClient:
    if config.provider == "none":
        return DisabledLLMClient(provider="none", model=config.model)
    if config.provider in {"deepseek", "openai"}:
        return OpenAICompatibleClient(config)
    raise ValueError(f"Unsupported LLM provider: {config.provider}")
