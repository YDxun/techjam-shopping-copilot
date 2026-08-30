from .openai_compatible import OpenAICompatibleClient


class DeepSeekClient(OpenAICompatibleClient):
    """Compatibility wrapper for callers that still import DeepSeekClient."""
