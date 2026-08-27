from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence


class LLMState(str, Enum):
    DISABLED = "disabled"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class LLMErrorCategory(str, Enum):
    DISABLED = "disabled"
    AUTHENTICATION = "authentication"
    BAD_REQUEST = "bad_request"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    RATE_LIMIT = "rate_limit"
    SERVER = "server"
    SDK_MISSING = "sdk_missing"
    CIRCUIT_OPEN = "circuit_open"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True)
class LLMStatus:
    state: LLMState
    provider: str
    model: str
    attempts: int = 0
    error_category: LLMErrorCategory | None = None
    error_message: str = ""


@dataclass(frozen=True)
class LLMResult:
    success: bool
    provider: str
    model: str
    content: str = ""
    usage: LLMUsage = LLMUsage()
    latency_ms: float = 0.0
    error_category: LLMErrorCategory | None = None
    error_message: str = ""


class LLMClient(Protocol):
    @property
    def status(self) -> LLMStatus: ...

    def initialize(self) -> LLMStatus: ...

    def chat(self, messages: Sequence[dict[str, str]], *, temperature: float | None = None, max_tokens: int | None = None) -> LLMResult: ...


class DisabledLLMClient:
    def __init__(self, *, provider: str = "none", model: str = "") -> None:
        self._status = LLMStatus(LLMState.DISABLED, provider, model, error_category=LLMErrorCategory.DISABLED)

    @property
    def status(self) -> LLMStatus:
        return self._status

    def initialize(self) -> LLMStatus:
        return self._status

    def chat(self, messages: Sequence[dict[str, str]], *, temperature: float | None = None, max_tokens: int | None = None) -> LLMResult:
        return LLMResult(False, self._status.provider, self._status.model, error_category=LLMErrorCategory.DISABLED, error_message="disabled")
