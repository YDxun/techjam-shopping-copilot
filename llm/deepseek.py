from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from config.models import LLMConfig
from .base import LLMErrorCategory, LLMResult, LLMState, LLMStatus, LLMUsage


@dataclass(frozen=True)
class FailureDisposition:
    category: LLMErrorCategory
    retryable: bool


def classify_openai_failure(error: Exception) -> FailureDisposition:
    status_code = getattr(error, "status_code", None)
    direct = {400: (LLMErrorCategory.BAD_REQUEST, False), 401: (LLMErrorCategory.AUTHENTICATION, False), 403: (LLMErrorCategory.AUTHENTICATION, False), 404: (LLMErrorCategory.NOT_FOUND, False), 408: (LLMErrorCategory.TIMEOUT, True), 429: (LLMErrorCategory.RATE_LIMIT, True)}
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


def _openai_sdk_factory(**kwargs):
    from openai import OpenAI
    import httpx
    timeout = httpx.Timeout(kwargs.pop("timeout_seconds"), connect=kwargs.pop("connect_timeout_seconds"))
    return OpenAI(timeout=timeout, max_retries=0, **kwargs)


class DeepSeekClient:
    _MAX_ERROR_MESSAGE_LENGTH = 500

    def __init__(self, config: LLMConfig, *, sdk_factory: Callable[..., object] | None = None, sleep: Callable[[float], None] = time.sleep, jitter: Callable[[], float] | None = None, clock: Callable[[], float] = time.monotonic, failure_classifier: Callable[[Exception], FailureDisposition] | None = None) -> None:
        self._config = config
        self._sdk_factory = sdk_factory or _openai_sdk_factory
        self._sleep, self._jitter, self._clock = sleep, jitter or (lambda: 0.0), clock
        self._failure_classifier = failure_classifier or classify_openai_failure
        self._sdk: object | None = None
        self._runtime_failures = 0
        self._status = LLMStatus(LLMState.DISABLED, "deepseek", config.model)

    def __repr__(self) -> str:
        return f"DeepSeekClient(provider='deepseek', model={self._config.model!r}, state={self._status.state.value!r})"

    @property
    def status(self) -> LLMStatus:
        return self._status

    def _make_sdk(self) -> object:
        if self._sdk is None:
            self._sdk = self._sdk_factory(api_key=self._config.api_key, base_url=self._config.base_url, timeout_seconds=self._config.timeout_seconds, connect_timeout_seconds=self._config.connect_timeout_seconds)
        return self._sdk

    def _sanitize(self, error: Exception | str) -> str:
        message = str(error).replace(self._config.api_key, "[redacted]") if self._config.api_key else str(error)
        message = re.sub(r"(?i)bearer\s+[^\s,;]+", "Bearer [redacted]", message)
        message = re.sub(r"(?i)(authorization\s*[:=]\s*)[^\s,;]+", r"\1[redacted]", message)
        return message[:self._MAX_ERROR_MESSAGE_LENGTH]

    def _failure_status(self, attempts: int, error: Exception, disposition: FailureDisposition) -> LLMStatus:
        self._status = LLMStatus(LLMState.UNAVAILABLE, "deepseek", self._config.model, attempts, disposition.category, self._sanitize(error))
        return self._status

    def _create(self, messages: Sequence[dict[str, str]], temperature: float, max_tokens: int):
        return self._make_sdk().chat.completions.create(model=self._config.model, messages=list(messages), temperature=temperature, max_tokens=max_tokens)

    def _attempt(self, messages: Sequence[dict[str, str]], temperature: float, max_tokens: int):
        attempts = 0
        while True:
            attempts += 1
            try:
                return self._create(messages, temperature, max_tokens), attempts, None, None
            except Exception as error:
                disposition = self._failure_classifier(error)
                if not disposition.retryable or attempts > self._config.retry.max_retries:
                    return None, attempts, error, disposition
                delay = min(self._config.retry.max_delay_seconds, self._config.retry.base_delay_seconds * 2 ** (attempts - 1)) + self._jitter()
                self._sleep(delay)

    def initialize(self) -> LLMStatus:
        if not self._config.api_key:
            self._status = LLMStatus(LLMState.DISABLED, "deepseek", self._config.model, error_category=LLMErrorCategory.DISABLED)
            return self._status
        try:
            self._make_sdk()
        except ImportError as error:
            return self._failure_status(0, error, FailureDisposition(LLMErrorCategory.SDK_MISSING, False))
        if not self._config.health_check_enabled:
            self._status = LLMStatus(LLMState.AVAILABLE, "deepseek", self._config.model)
            return self._status
        _, attempts, error, disposition = self._attempt([{"role": "user", "content": "health check"}], 0.0, 1)
        if error is not None:
            return self._failure_status(attempts, error, disposition)
        self._status = LLMStatus(LLMState.AVAILABLE, "deepseek", self._config.model, attempts=attempts)
        return self._status

    def chat(self, messages: Sequence[dict[str, str]], *, temperature: float | None = None, max_tokens: int | None = None) -> LLMResult:
        if self._runtime_failures >= self._config.circuit_breaker.failure_threshold:
            return LLMResult(False, "deepseek", self._config.model, error_category=LLMErrorCategory.CIRCUIT_OPEN, error_message="circuit open")
        if self._status.state != LLMState.AVAILABLE:
            status = self.initialize()
            if status.state != LLMState.AVAILABLE:
                return LLMResult(False, "deepseek", self._config.model, error_category=status.error_category, error_message=status.error_message)
        start = self._clock()
        result, _, error, disposition = self._attempt(messages, self._config.temperature if temperature is None else temperature, self._config.max_tokens if max_tokens is None else max_tokens)
        latency_ms = (self._clock() - start) * 1000
        if error is not None:
            self._runtime_failures += 1
            return LLMResult(False, "deepseek", self._config.model, latency_ms=latency_ms, error_category=disposition.category, error_message=self._sanitize(error))
        self._runtime_failures = 0
        usage = getattr(result, "usage", None)
        return LLMResult(True, "deepseek", self._config.model, content=result.choices[0].message.content or "", usage=LLMUsage(getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0), latency_ms=latency_ms)
