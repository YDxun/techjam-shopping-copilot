from config.models import LLMConfig
from .base import DisabledLLMClient, LLMClient
from .deepseek import DeepSeekClient


def create_llm_client(config: LLMConfig) -> LLMClient:
    if config.provider == "none":
        return DisabledLLMClient(provider="none", model=config.model)
    if config.provider == "deepseek":
        return DeepSeekClient(config)
    raise ValueError(f"Unsupported LLM provider: {config.provider}")
