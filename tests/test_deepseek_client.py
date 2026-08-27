from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from config.models import LLMConfig, ProviderConfig, ProviderConfigs, SecretValue
from llm.deepseek import DeepSeekClient
from llm.openai_compatible import OpenAICompatibleClient


def deepseek_config() -> LLMConfig:
    return LLMConfig(providers=ProviderConfigs(
        deepseek=ProviderConfig("deepseek-chat", "https://api.deepseek.com", "max_tokens", True, SecretValue("deepseek-key")),
        openai=ProviderConfig("gpt-4o-mini", "https://api.openai.com/v1", "max_completion_tokens", True),
    ))


class DeepSeekCompatibilityTest(unittest.TestCase):
    def test_wrapper_is_an_openai_compatible_client(self) -> None:
        self.assertIsInstance(DeepSeekClient(deepseek_config(), sdk_factory=Mock()), OpenAICompatibleClient)

    def test_wrapper_delegates_profile_aware_requests(self) -> None:
        sdk = Mock()
        sdk.chat.completions.create.side_effect = [
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))], usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1)),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))], usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3)),
        ]
        client = DeepSeekClient(deepseek_config(), sdk_factory=Mock(return_value=sdk))
        client.initialize()
        self.assertTrue(client.chat([{"role": "user", "content": "rank"}], temperature=0.1, max_tokens=9).success)
        kwargs = sdk.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "deepseek-chat")
        self.assertEqual(kwargs["max_tokens"], 9)
        self.assertEqual(kwargs["temperature"], 0.1)


if __name__ == "__main__":
    unittest.main()
