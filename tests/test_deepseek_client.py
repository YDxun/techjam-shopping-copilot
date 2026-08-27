from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from config.models import CircuitBreakerConfig, LLMConfig, ProviderConfig, ProviderConfigs, RetryConfig, SecretValue
from llm.base import LLMErrorCategory, LLMState
from llm.deepseek import DeepSeekClient, FailureDisposition, classify_openai_failure


def completion(content: str = "OK", prompt_tokens: int = 2, completion_tokens: int = 1):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


class StatusError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class DeepSeekClientTest(unittest.TestCase):
    def make_config(self, api_key: str = "test-key", **changes) -> LLMConfig:
        model = changes.pop("model", "deepseek-chat")
        base_url = changes.pop("base_url", "https://api.deepseek.com")
        values = {
            "provider": "deepseek",
            "providers": ProviderConfigs(
                deepseek=ProviderConfig(model, base_url, "max_tokens", True, SecretValue(api_key)),
                openai=ProviderConfig("gpt-4o-mini", "https://api.openai.com/v1", "max_completion_tokens", True),
            ),
            "retry": RetryConfig(max_retries=2, base_delay_seconds=0.0, max_delay_seconds=0.0),
            "circuit_breaker": CircuitBreakerConfig(failure_threshold=2),
        }
        values.update(changes)
        return LLMConfig(**values)

    def test_construction_has_no_sdk_or_network_side_effect(self) -> None:
        factory = Mock()
        client = DeepSeekClient(self.make_config(), sdk_factory=factory)
        factory.assert_not_called()
        self.assertEqual(client.status.state, LLMState.DISABLED)

    def test_missing_key_initializes_disabled_without_sdk(self) -> None:
        factory = Mock()
        client = DeepSeekClient(self.make_config(api_key=""), sdk_factory=factory)
        status = client.initialize()
        self.assertEqual(status.state, LLMState.DISABLED)
        factory.assert_not_called()

    def test_successful_probe_marks_available(self) -> None:
        sdk = Mock()
        sdk.chat.completions.create.return_value = completion()
        client = DeepSeekClient(self.make_config(), sdk_factory=Mock(return_value=sdk))
        status = client.initialize()
        self.assertEqual(status.state, LLMState.AVAILABLE)
        self.assertEqual(status.attempts, 1)
        sdk.chat.completions.create.assert_called_once()

    def test_retryable_probe_failure_retries_until_success(self) -> None:
        sdk = Mock()
        transient = StatusError("server unavailable", 503)
        sdk.chat.completions.create.side_effect = [transient, transient, completion()]
        sleeps: list[float] = []
        client = DeepSeekClient(
            self.make_config(),
            sdk_factory=Mock(return_value=sdk),
            sleep=sleeps.append,
            jitter=lambda: 0.0,
            failure_classifier=lambda _: FailureDisposition(LLMErrorCategory.SERVER, True),
        )
        status = client.initialize()
        self.assertEqual(status.state, LLMState.AVAILABLE)
        self.assertEqual(sdk.chat.completions.create.call_count, 3)
        self.assertEqual(sleeps, [0.0, 0.0])

    def test_authentication_probe_failure_does_not_retry(self) -> None:
        sdk = Mock()
        sdk.chat.completions.create.side_effect = StatusError("unauthorized", 401)
        client = DeepSeekClient(
            self.make_config(),
            sdk_factory=Mock(return_value=sdk),
            failure_classifier=lambda _: FailureDisposition(LLMErrorCategory.AUTHENTICATION, False),
        )
        status = client.initialize()
        self.assertEqual(status.state, LLMState.UNAVAILABLE)
        self.assertEqual(status.error_category, LLMErrorCategory.AUTHENTICATION)
        self.assertEqual(sdk.chat.completions.create.call_count, 1)

    def test_exhausted_transient_probe_returns_unavailable(self) -> None:
        sdk = Mock()
        sdk.chat.completions.create.side_effect = StatusError("temporary", 503)
        client = DeepSeekClient(
            self.make_config(),
            sdk_factory=Mock(return_value=sdk),
            sleep=lambda _: None,
            jitter=lambda: 0.0,
            failure_classifier=lambda _: FailureDisposition(LLMErrorCategory.SERVER, True),
        )
        status = client.initialize()
        self.assertEqual(status.state, LLMState.UNAVAILABLE)
        self.assertEqual(status.error_category, LLMErrorCategory.SERVER)
        self.assertEqual(sdk.chat.completions.create.call_count, 3)


    def test_probe_attempts_are_capped_at_three_when_configured_retries_are_higher(self) -> None:
        sdk = Mock()
        sdk.chat.completions.create.side_effect = StatusError("temporary", 503)
        client = DeepSeekClient(
            self.make_config(retry=RetryConfig(max_retries=10, base_delay_seconds=0.0, max_delay_seconds=0.0)),
            sdk_factory=Mock(return_value=sdk),
            sleep=lambda _: None,
            jitter=lambda: 0.0,
            failure_classifier=lambda _: FailureDisposition(LLMErrorCategory.SERVER, True),
        )
        status = client.initialize()
        self.assertEqual(status.state, LLMState.UNAVAILABLE)
        self.assertEqual(status.attempts, 3)
        self.assertEqual(sdk.chat.completions.create.call_count, 3)

    def test_malformed_probe_response_is_a_structured_unavailable_failure(self) -> None:
        sdk = Mock()
        sdk.chat.completions.create.return_value = SimpleNamespace(choices=[])
        client = DeepSeekClient(self.make_config(), sdk_factory=Mock(return_value=sdk))
        status = client.initialize()
        self.assertEqual(status.state, LLMState.UNAVAILABLE)
        self.assertEqual(status.error_category, LLMErrorCategory.UNKNOWN)

    def test_malformed_chat_response_is_a_structured_runtime_failure(self) -> None:
        sdk = Mock()
        sdk.chat.completions.create.side_effect = [completion(), SimpleNamespace(choices=[])]
        client = DeepSeekClient(self.make_config(), sdk_factory=Mock(return_value=sdk))
        client.initialize()
        result = client.chat([{ "role": "user", "content": "hello" }])
        self.assertFalse(result.success)
        self.assertEqual(result.error_category, LLMErrorCategory.UNKNOWN)

    def test_errors_and_repr_do_not_expose_api_key(self) -> None:
        sdk = Mock()
        sdk.chat.completions.create.side_effect = StatusError("Bearer test-key Authorization: test-key")
        client = DeepSeekClient(
            self.make_config(),
            sdk_factory=Mock(return_value=sdk),
            failure_classifier=lambda _: FailureDisposition(LLMErrorCategory.UNKNOWN, False),
        )
        status = client.initialize()
        self.assertNotIn("test-key", status.error_message)
        self.assertNotIn("test-key", repr(client))

    def test_chat_returns_completion_metadata_and_latency(self) -> None:
        sdk = Mock()
        sdk.chat.completions.create.side_effect = [completion(), completion("answer", 4, 7)]
        ticks = iter([10.0, 10.125])
        client = DeepSeekClient(
            self.make_config(model="deepseek-reasoner"),
            sdk_factory=Mock(return_value=sdk),
            clock=lambda: next(ticks),
        )
        client.initialize()
        result = client.chat([{"role": "user", "content": "hello"}])
        self.assertTrue(result.success)
        self.assertEqual(result.content, "answer")
        self.assertEqual(result.usage.prompt_tokens, 4)
        self.assertEqual(result.usage.completion_tokens, 7)
        self.assertEqual(result.provider, "deepseek")
        self.assertEqual(result.model, "deepseek-reasoner")
        self.assertEqual(result.latency_ms, 125.0)

    def test_runtime_failures_open_circuit_without_another_sdk_call(self) -> None:
        sdk = Mock()
        sdk.chat.completions.create.side_effect = [completion(), StatusError("down"), StatusError("down")]
        client = DeepSeekClient(
            self.make_config(),
            sdk_factory=Mock(return_value=sdk),
            failure_classifier=lambda _: FailureDisposition(LLMErrorCategory.SERVER, False),
        )
        client.initialize()
        self.assertFalse(client.chat([{"role": "user", "content": "one"}]).success)
        self.assertFalse(client.chat([{"role": "user", "content": "two"}]).success)
        result = client.chat([{"role": "user", "content": "three"}])
        self.assertEqual(result.error_category, LLMErrorCategory.CIRCUIT_OPEN)
        self.assertEqual(sdk.chat.completions.create.call_count, 3)

    def test_runtime_success_resets_consecutive_failure_count(self) -> None:
        sdk = Mock()
        sdk.chat.completions.create.side_effect = [
            completion(), StatusError("down"), completion("recovered"), StatusError("down"), StatusError("down"),
        ]
        client = DeepSeekClient(
            self.make_config(),
            sdk_factory=Mock(return_value=sdk),
            failure_classifier=lambda _: FailureDisposition(LLMErrorCategory.SERVER, False),
        )
        client.initialize()
        self.assertFalse(client.chat([{"role": "user", "content": "one"}]).success)
        self.assertTrue(client.chat([{"role": "user", "content": "two"}]).success)
        self.assertFalse(client.chat([{"role": "user", "content": "three"}]).success)
        self.assertFalse(client.chat([{"role": "user", "content": "four"}]).success)
        self.assertEqual(sdk.chat.completions.create.call_count, 5)

    def test_default_exception_classifier_maps_openai_style_errors(self) -> None:
        cases = [
            (StatusError("bad request", 400), LLMErrorCategory.BAD_REQUEST, False),
            (StatusError("unauthorized", 401), LLMErrorCategory.AUTHENTICATION, False),
            (StatusError("missing", 404), LLMErrorCategory.NOT_FOUND, False),
            (StatusError("too many", 429), LLMErrorCategory.RATE_LIMIT, True),
            (StatusError("server", 503), LLMErrorCategory.SERVER, True),
            (TimeoutError("timed out"), LLMErrorCategory.TIMEOUT, True),
            (ConnectionError("connection reset"), LLMErrorCategory.CONNECTION, True),
        ]
        for error, category, retryable in cases:
            with self.subTest(error=error):
                disposition = classify_openai_failure(error)
                self.assertEqual(disposition.category, category)
                self.assertEqual(disposition.retryable, retryable)


if __name__ == "__main__":
    unittest.main()
