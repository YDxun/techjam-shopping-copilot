from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import httpx

from config.models import LLMConfig, ProviderConfig
from .base import LLMErrorCategory, LLMResult, LLMState, LLMStatus, LLMUsage


@dataclass(frozen=True)
class FailureDisposition:
    category: LLMErrorCategory
    retryable: bool


def classify_openai_failure(error: Exception) -> FailureDisposition:
    status_code = getattr(error, "status_code", None)
    direct = {
        400: (LLMErrorCategory.BAD_REQUEST, False),
        401: (LLMErrorCategory.AUTHENTICATION, False),
        403: (LLMErrorCategory.AUTHENTICATION, False),
        404: (LLMErrorCategory.NOT_FOUND, False),
        408: (LLMErrorCategory.TIMEOUT, True),
        429: (LLMErrorCategory.RATE_LIMIT, True),
    }
    if status_code in direct:
        category, retryable = direct[status_code]
        return FailureDisposition(category, retryable)
    if isinstance(status_code, int) and 500 <= status_code < 600:
        return FailureDisposition(LLMErrorCategory.SERVER, True)
    if isinstance(error, TimeoutError) or "timeout" in type(error).__name__.lower():
        return FailureDisposition(LLMErrorCategory.TIMEOUT, True)
    if isinstance(error, ConnectionError) or "connection" in type(error).__name__.lower():
        return FailureDisposition(LLMErrorCategory.CONNECTION, True)
    return FailureDisposition(LLMErrorCategory.UNKNOWN, False)


def _openai_sdk_factory(**kwargs: object) -> object:
    from openai import OpenAI

    return OpenAI(**kwargs)


class OpenAICompatibleClient:
    _MAX_ERROR_MESSAGE_LENGTH = 500

    def __init__(
        self,
        config: LLMConfig,
        *,
        sdk_factory: Callable[..., object] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] | None = None,
        clock: Callable[[], float] = time.monotonic,
        failure_classifier: Callable[[Exception], FailureDisposition] | None = None,
    ) -> None:
        self._config = config
        self._profile = config.selected_profile
        self._sdk_factory = sdk_factory or _openai_sdk_factory
        self._sleep = sleep
        self._jitter = jitter or (lambda: random.uniform(0.0, 0.1))
        self._clock = clock
        self._failure_classifier = failure_classifier or classify_openai_failure
        self._sdk: object | None = None
        self._runtime_failures = 0
        self._cumulative_usage = LLMUsage()
        self._status = LLMStatus(LLMState.DISABLED, config.provider, self._model)

    @property
    def _model(self) -> str:
        return self._profile.model if self._profile else ""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(provider={self._config.provider!r}, model={self._model!r}, state={self._status.state.value!r})"

    @property
    def status(self) -> LLMStatus:
        return self._status

    @property
    def cumulative_usage(self) -> LLMUsage:
        return self._cumulative_usage

    def _make_sdk(self) -> object:
        if self._sdk is None:
            profile = self._require_profile()
            self._sdk = self._sdk_factory(
                api_key=profile.api_key.reveal(),
                base_url=profile.base_url,
                timeout=httpx.Timeout(self._config.timeout_seconds, connect=self._config.connect_timeout_seconds),
                max_retries=0,
            )
        return self._sdk

    def _require_profile(self) -> ProviderConfig:
        if self._profile is None:
            raise ValueError("no active LLM provider profile")
        return self._profile

    def _sanitize(self, error: Exception | str) -> str:
        raw_key = self._profile.api_key.reveal() if self._profile else ""
        message = str(error).replace(raw_key, "[redacted]") if raw_key else str(error)
        message = re.sub(r"(?i)bearer\s+[^\s,;]+", "Bearer [redacted]", message)
        message = re.sub(r"(?i)(authorization\s*[:=]\s*)[^\s,;]+", r"\1[redacted]", message)
        return message[:self._MAX_ERROR_MESSAGE_LENGTH]

    def _failure_status(self, attempts: int, error: Exception, disposition: FailureDisposition) -> LLMStatus:
        self._status = LLMStatus(LLMState.UNAVAILABLE, self._config.provider, self._model, attempts, disposition.category, self._sanitize(error))
        return self._status

    def _request_kwargs(self, messages: Sequence[dict[str, str]], temperature: float | None, max_tokens: int | None) -> dict[str, object]:
        profile = self._require_profile()
        requested_max_tokens = self._config.max_tokens if max_tokens is None else max_tokens
        requested_temperature = self._config.temperature if temperature is None else temperature
        kwargs: dict[str, object] = {"model": profile.model, "messages": list(messages)}
        kwargs[profile.token_limit_parameter] = requested_max_tokens
        if profile.supports_temperature and requested_temperature is not None:
            kwargs["temperature"] = requested_temperature
        return kwargs

    def _create(self, messages: Sequence[dict[str, str]], temperature: float | None, max_tokens: int | None) -> object:
        return self._make_sdk().chat.completions.create(**self._request_kwargs(messages, temperature, max_tokens))

    def _attempt(self, messages: Sequence[dict[str, str]], temperature: float | None, max_tokens: int | None, max_retries: int | None = None) -> tuple[object | None, int, Exception | None, FailureDisposition | None]:
        allowed_retries = self._config.retry.max_retries if max_retries is None else max_retries
        attempts = 0
        while True:
            attempts += 1
            try:
                return self._create(messages, temperature, max_tokens), attempts, None, None
            except Exception as error:
                disposition = self._failure_classifier(error)
                if not disposition.retryable or attempts > allowed_retries:
                    return None, attempts, error, disposition
                delay = min(self._config.retry.max_delay_seconds, self._config.retry.base_delay_seconds * 2 ** (attempts - 1)) + self._jitter()
                self._sleep(delay)

    def _decode_completion(self, response: object) -> tuple[str, LLMUsage]:
        try:
            content = response.choices[0].message.content
            usage = response.usage
            prompt_tokens = self._usage_tokens(usage, "prompt_tokens")
            completion_tokens = self._usage_tokens(usage, "completion_tokens")
        except (AttributeError, IndexError, TypeError) as error:
            raise ValueError("malformed completion response") from error
        if content is not None and not isinstance(content, str):
            raise ValueError("malformed completion response")
        return content or "", LLMUsage(prompt_tokens, completion_tokens)

    @staticmethod
    def _usage_tokens(usage: object, field: str) -> int:
        value = getattr(usage, field, None)
        if value is None:
            return 0
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("malformed completion response")
        return value

    def _add_usage(self, usage: LLMUsage) -> None:
        self._cumulative_usage = LLMUsage(
            self._cumulative_usage.prompt_tokens + usage.prompt_tokens,
            self._cumulative_usage.completion_tokens + usage.completion_tokens,
        )

    def initialize(self) -> LLMStatus:
        if self._profile is None or not self._profile.api_key:
            self._status = LLMStatus(LLMState.DISABLED, self._config.provider, self._model, error_category=LLMErrorCategory.DISABLED)
            return self._status
        try:
            self._make_sdk()
        except ImportError as error:
            return self._failure_status(0, error, FailureDisposition(LLMErrorCategory.SDK_MISSING, False))
        except Exception as error:
            return self._failure_status(0, error, self._failure_classifier(error))
        if not self._config.health_check_enabled:
            self._status = LLMStatus(LLMState.AVAILABLE, self._config.provider, self._model)
            return self._status
        response, attempts, error, disposition = self._attempt(
            [{"role": "user", "content": "health check"}], 0.0, 1,
            max_retries=min(self._config.retry.max_retries, 2),
        )
        if error is not None:
            return self._failure_status(attempts, error, disposition)
        try:
            _, usage = self._decode_completion(response)
        except Exception as error:
            return self._failure_status(attempts, error, self._failure_classifier(error))
        self._add_usage(usage)
        self._status = LLMStatus(LLMState.AVAILABLE, self._config.provider, self._model, attempts=attempts)
        return self._status

    def _open_circuit(self) -> None:
        self._status = LLMStatus(
            LLMState.UNAVAILABLE,
            self._config.provider,
            self._model,
            attempts=0,
            error_category=LLMErrorCategory.CIRCUIT_OPEN,
            error_message="circuit open",
        )

    def chat(self, messages: Sequence[dict[str, str]], *, temperature: float | None = None, max_tokens: int | None = None) -> LLMResult:
        if self._runtime_failures >= self._config.circuit_breaker.failure_threshold:
            self._open_circuit()
            return LLMResult(False, self._config.provider, self._model, error_category=LLMErrorCategory.CIRCUIT_OPEN, error_message="circuit open")
        if self._status.state != LLMState.AVAILABLE:
            status = self.initialize()
            if status.state != LLMState.AVAILABLE:
                return LLMResult(False, self._config.provider, self._model, error_category=status.error_category, error_message=status.error_message)
        start = self._clock()
        response, _, error, disposition = self._attempt(messages, temperature, max_tokens)
        latency_ms = (self._clock() - start) * 1000
        if error is not None:
            self._runtime_failures += 1
            if self._runtime_failures >= self._config.circuit_breaker.failure_threshold:
                self._open_circuit()
            return LLMResult(False, self._config.provider, self._model, latency_ms=latency_ms, error_category=disposition.category, error_message=self._sanitize(error))
        try:
            content, usage = self._decode_completion(response)
        except Exception as error:
            self._runtime_failures += 1
            disposition = self._failure_classifier(error)
            if self._runtime_failures >= self._config.circuit_breaker.failure_threshold:
                self._open_circuit()
            return LLMResult(False, self._config.provider, self._model, latency_ms=latency_ms, error_category=disposition.category, error_message=self._sanitize(error))
        self._runtime_failures = 0
        self._add_usage(usage)
        return LLMResult(True, self._config.provider, self._model, content=content, usage=usage, latency_ms=latency_ms)
