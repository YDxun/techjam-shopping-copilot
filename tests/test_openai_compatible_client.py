from __future__ import annotations

import httpx
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import openai

from config.models import CircuitBreakerConfig, LLMConfig, ProviderConfig, ProviderConfigs, RetryConfig, SecretValue
from llm.base import LLMErrorCategory, LLMState
from llm.factory import create_llm_client
from llm.openai_compatible import FailureDisposition, OpenAICompatibleClient, classify_openai_failure


def completion(content: str = "OK", prompt_tokens: int = 2, completion_tokens: int = 1):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class StatusError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def config_with_selected_profile(provider: str = "deepseek", **profile_changes: object) -> LLMConfig:
    profiles = ProviderConfigs(
        deepseek=ProviderConfig("deepseek-chat", "https://api.deepseek.com", "max_tokens", True, SecretValue("deepseek-key")),
        openai=ProviderConfig("gpt-4o-mini", "https://api.openai.com/v1", "max_completion_tokens", True, SecretValue("openai-key")),
    )
    selected = getattr(profiles, provider) if provider != "none" else None
    if selected is not None and profile_changes:
        selected = ProviderConfig(
            profile_changes.get("model", selected.model),
            profile_changes.get("base_url", selected.base_url),
            profile_changes.get("token_limit_parameter", selected.token_limit_parameter),
            profile_changes.get("supports_temperature", selected.supports_temperature),
            profile_changes.get("api_key", selected.api_key),
        )
        profiles = ProviderConfigs(
            deepseek=selected if provider == "deepseek" else profiles.deepseek,
            openai=selected if provider == "openai" else profiles.openai,
        )
    return LLMConfig(
        provider=provider,
        providers=profiles,
        retry=RetryConfig(max_retries=2, base_delay_seconds=0.2, max_delay_seconds=1.0),
        circuit_breaker=CircuitBreakerConfig(failure_threshold=2),
    )


def make_client(provider: str = "deepseek", config: LLMConfig | None = None, **kwargs: object) -> tuple[OpenAICompatibleClient, Mock]:
    sdk = Mock()
    sdk.chat.completions.create.return_value = completion()
    client = OpenAICompatibleClient(config or config_with_selected_profile(provider), sdk_factory=Mock(return_value=sdk), **kwargs)
    return client, sdk


class OpenAICompatibleClientTest(unittest.TestCase):
    def test_deepseek_uses_max_tokens_and_temperature(self) -> None:
        client, sdk = make_client(provider="deepseek")
        client.initialize()
        client.chat([{"role": "user", "content": "rank"}], temperature=0.2, max_tokens=77)
        kwargs = sdk.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["max_tokens"], 77)
        self.assertEqual(kwargs["temperature"], 0.2)
        self.assertNotIn("max_completion_tokens", kwargs)

    def test_openai_capability_uses_max_completion_tokens_and_omits_temperature(self) -> None:
        config = config_with_selected_profile("openai", token_limit_parameter="max_completion_tokens", supports_temperature=False)
        client, sdk = make_client(config=config)
        client.initialize()
        client.chat([{"role": "user", "content": "rank"}], temperature=0.2, max_tokens=77)
        kwargs = sdk.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["max_completion_tokens"], 77)
        self.assertNotIn("max_tokens", kwargs)
        self.assertNotIn("temperature", kwargs)

    def test_selected_profile_controls_sdk_key_base_url_and_model(self) -> None:
        config = config_with_selected_profile("openai", model="selected-model", base_url="https://selected.example/v1")
        factory = Mock(return_value=Mock())
        client = OpenAICompatibleClient(config, sdk_factory=factory)
        client.initialize()
        kwargs = factory.call_args.kwargs
        self.assertEqual(kwargs["api_key"], "openai-key")
        self.assertEqual(kwargs["base_url"], "https://selected.example/v1")
        self.assertEqual(kwargs["max_retries"], 0)
        self.assertIsInstance(kwargs["timeout"], httpx.Timeout)
        self.assertEqual(client.status.model, "selected-model")

    def test_inactive_provider_key_is_never_used_to_construct_sdk(self) -> None:
        config = config_with_selected_profile("openai", api_key=SecretValue("selected-key"))
        factory = Mock(return_value=Mock())
        OpenAICompatibleClient(config, sdk_factory=factory).initialize()
        self.assertEqual(factory.call_args.kwargs["api_key"], "selected-key")

    def test_none_provider_never_constructs_sdk(self) -> None:
        factory = Mock()
        status = OpenAICompatibleClient(config_with_selected_profile("none"), sdk_factory=factory).initialize()
        self.assertEqual(status.state, LLMState.DISABLED)
        factory.assert_not_called()

    def test_missing_selected_key_initializes_disabled_without_sdk(self) -> None:
        factory = Mock()
        config = config_with_selected_profile("openai", api_key=SecretValue())
        status = OpenAICompatibleClient(config, sdk_factory=factory).initialize()
        self.assertEqual(status.state, LLMState.DISABLED)
        factory.assert_not_called()


    def test_sdk_factory_import_error_reports_missing_sdk_without_probe(self) -> None:
        factory = Mock(side_effect=ImportError("openai unavailable"))
        status = OpenAICompatibleClient(config_with_selected_profile(), sdk_factory=factory).initialize()
        self.assertEqual(status.state, LLMState.UNAVAILABLE)
        self.assertEqual(status.attempts, 0)
        self.assertEqual(status.error_category, LLMErrorCategory.SDK_MISSING)
        factory.assert_called_once()

    def test_sdk_factory_exception_is_sanitized_unavailable_without_probe(self) -> None:
        factory = Mock(side_effect=ValueError("invalid credential deepseek-key"))
        status = OpenAICompatibleClient(config_with_selected_profile(), sdk_factory=factory).initialize()
        self.assertEqual(status.state, LLMState.UNAVAILABLE)
        self.assertEqual(status.attempts, 0)
        self.assertEqual(status.error_category, LLMErrorCategory.UNKNOWN)
        self.assertNotIn("deepseek-key", status.error_message)
        factory.assert_called_once()
    def test_factory_selects_one_shared_client_for_each_online_provider(self) -> None:
        for provider in ("deepseek", "openai"):
            with self.subTest(provider=provider):
                self.assertIsInstance(create_llm_client(config_with_selected_profile(provider)), OpenAICompatibleClient)
        self.assertEqual(create_llm_client(config_with_selected_profile("none")).status.state, LLMState.DISABLED)

    def test_successful_probe_marks_available_and_accumulates_usage(self) -> None:
        client, sdk = make_client()
        sdk.chat.completions.create.side_effect = [completion(prompt_tokens=3, completion_tokens=4), completion("answer", 5, 6)]
        self.assertEqual(client.initialize().state, LLMState.AVAILABLE)
        result = client.chat([{"role": "user", "content": "rank"}])
        self.assertTrue(result.success)
        self.assertEqual(result.usage.prompt_tokens, 5)
        self.assertEqual(result.usage.completion_tokens, 6)
        self.assertEqual(client.cumulative_usage.prompt_tokens, 8)
        self.assertEqual(client.cumulative_usage.completion_tokens, 10)

    def test_health_check_disabled_does_not_issue_probe(self) -> None:
        config = LLMConfig(**{**config_with_selected_profile().__dict__, "health_check_enabled": False})
        client, sdk = make_client(config=config)
        self.assertEqual(client.initialize().state, LLMState.AVAILABLE)
        sdk.chat.completions.create.assert_not_called()

    def test_retryable_probe_failure_retries_until_success(self) -> None:
        client, sdk = make_client(sleep=Mock(), jitter=lambda: 0.0, failure_classifier=lambda _: FailureDisposition(LLMErrorCategory.SERVER, True))
        sdk.chat.completions.create.side_effect = [StatusError("server", 503), StatusError("server", 503), completion()]
        self.assertEqual(client.initialize().state, LLMState.AVAILABLE)
        self.assertEqual(sdk.chat.completions.create.call_count, 3)

    def test_probe_attempts_are_capped_at_three(self) -> None:
        config = LLMConfig(**{**config_with_selected_profile().__dict__, "retry": RetryConfig(max_retries=10, base_delay_seconds=0, max_delay_seconds=0)})
        client, sdk = make_client(config=config, sleep=lambda _: None, jitter=lambda: 0.0, failure_classifier=lambda _: FailureDisposition(LLMErrorCategory.SERVER, True))
        sdk.chat.completions.create.side_effect = StatusError("server", 503)
        status = client.initialize()
        self.assertEqual(status.state, LLMState.UNAVAILABLE)
        self.assertEqual(status.attempts, 3)

    def test_malformed_completion_is_structured_failure(self) -> None:
        client, sdk = make_client()
        sdk.chat.completions.create.side_effect = [completion(), SimpleNamespace(choices=[])]
        client.initialize()
        result = client.chat([{"role": "user", "content": "rank"}])
        self.assertFalse(result.success)
        self.assertEqual(result.error_category, LLMErrorCategory.UNKNOWN)

    def test_malformed_probe_usage_returns_unavailable_without_escaping(self) -> None:
        client, sdk = make_client()
        sdk.chat.completions.create.return_value = completion(prompt_tokens="two", completion_tokens=1)
        status = client.initialize()
        self.assertEqual(status.state, LLMState.UNAVAILABLE)
        self.assertEqual(status.attempts, 1)
        self.assertEqual(status.error_category, LLMErrorCategory.UNKNOWN)

    def test_malformed_runtime_usage_returns_failed_result_without_accumulating(self) -> None:
        client, sdk = make_client()
        sdk.chat.completions.create.side_effect = [completion(prompt_tokens=2, completion_tokens=3), completion("answer", prompt_tokens=True, completion_tokens=1)]
        client.initialize()
        result = client.chat([{"role": "user", "content": "rank"}])
        self.assertFalse(result.success)
        self.assertEqual(result.error_category, LLMErrorCategory.UNKNOWN)
        self.assertEqual(client.cumulative_usage.prompt_tokens, 2)
        self.assertEqual(client.cumulative_usage.completion_tokens, 3)

    def test_error_messages_redact_selected_key(self) -> None:
        client, sdk = make_client(failure_classifier=lambda _: FailureDisposition(LLMErrorCategory.UNKNOWN, False))
        sdk.chat.completions.create.side_effect = StatusError("Bearer deepseek-key Authorization: deepseek-key")
        status = client.initialize()
        self.assertNotIn("deepseek-key", status.error_message)
        self.assertNotIn("deepseek-key", repr(client))

    def test_runtime_threshold_immediately_marks_status_circuit_open(self) -> None:
        client, sdk = make_client(failure_classifier=lambda _: FailureDisposition(LLMErrorCategory.SERVER, False))
        sdk.chat.completions.create.side_effect = [completion(), StatusError("down"), StatusError("down")]
        client.initialize()
        client.chat([{"role": "user", "content": "one"}])
        client.chat([{"role": "user", "content": "two"}])
        self.assertEqual(client.status.state, LLMState.UNAVAILABLE)
        self.assertEqual(client.status.error_category, LLMErrorCategory.CIRCUIT_OPEN)
        result = client.chat([{"role": "user", "content": "three"}])
        self.assertEqual(result.error_category, LLMErrorCategory.CIRCUIT_OPEN)
        self.assertEqual(sdk.chat.completions.create.call_count, 3)

    def test_runtime_success_resets_failure_count(self) -> None:
        client, sdk = make_client(failure_classifier=lambda _: FailureDisposition(LLMErrorCategory.SERVER, False))
        sdk.chat.completions.create.side_effect = [completion(), StatusError("down"), completion(), StatusError("down"), StatusError("down")]
        client.initialize()
        self.assertFalse(client.chat([{"role": "user", "content": "one"}]).success)
        self.assertTrue(client.chat([{"role": "user", "content": "two"}]).success)
        client.chat([{"role": "user", "content": "three"}])
        client.chat([{"role": "user", "content": "four"}])
        self.assertEqual(client.status.error_category, LLMErrorCategory.CIRCUIT_OPEN)

    def test_default_jitter_is_bounded_and_uses_injected_sleep(self) -> None:
        sleeps: list[float] = []
        client, sdk = make_client(sleep=sleeps.append, failure_classifier=lambda _: FailureDisposition(LLMErrorCategory.SERVER, True))
        sdk.chat.completions.create.side_effect = [StatusError("down"), completion()]
        with patch("llm.openai_compatible.random.uniform", return_value=0.07) as uniform:
            self.assertEqual(client.initialize().state, LLMState.AVAILABLE)
        uniform.assert_called_once_with(0.0, 0.1)
        self.assertEqual(sleeps, [0.27])

    def test_actual_openai_sdk_exceptions_have_expected_dispositions(self) -> None:
        request = httpx.Request("POST", "https://example.test")
        response = httpx.Response(408, request=request)
        cases = (
            (openai.AuthenticationError("bad key", response=httpx.Response(401, request=request), body=None), LLMErrorCategory.AUTHENTICATION, False),
            (openai.PermissionDeniedError("no", response=httpx.Response(403, request=request), body=None), LLMErrorCategory.AUTHENTICATION, False),
            (openai.APIStatusError("slow", response=response, body=None), LLMErrorCategory.TIMEOUT, True),
            (openai.RateLimitError("limited", response=httpx.Response(429, request=request), body=None), LLMErrorCategory.RATE_LIMIT, True),
            (openai.InternalServerError("broken", response=httpx.Response(503, request=request), body=None), LLMErrorCategory.SERVER, True),
            (openai.APIStatusError("odd", response=httpx.Response(418, request=request), body=None), LLMErrorCategory.UNKNOWN, False),
            (openai.APITimeoutError(request=request), LLMErrorCategory.TIMEOUT, True),
            (openai.APIConnectionError(request=request), LLMErrorCategory.CONNECTION, True),
        )
        for error, category, retryable in cases:
            with self.subTest(error=type(error).__name__):
                disposition = classify_openai_failure(error)
                self.assertEqual((disposition.category, disposition.retryable), (category, retryable))

    def test_status_error_classifier_covers_http_and_builtin_errors(self) -> None:
        cases = (
            (StatusError("bad", 400), LLMErrorCategory.BAD_REQUEST, False),
            (StatusError("missing", 404), LLMErrorCategory.NOT_FOUND, False),
            (TimeoutError("timeout"), LLMErrorCategory.TIMEOUT, True),
            (ConnectionError("connection"), LLMErrorCategory.CONNECTION, True),
        )
        for error, category, retryable in cases:
            with self.subTest(error=type(error).__name__):
                self.assertEqual(classify_openai_failure(error), FailureDisposition(category, retryable))


if __name__ == "__main__":
    unittest.main()
