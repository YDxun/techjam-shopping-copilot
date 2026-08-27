from .openai_compatible import FailureDisposition, OpenAICompatibleClient, classify_openai_failure


class DeepSeekClient(OpenAICompatibleClient):
    """Compatibility wrapper for callers that still import DeepSeekClient."""
