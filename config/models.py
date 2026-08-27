from __future__ import annotations

from dataclasses import dataclass, field


class SecretValue:
    __slots__ = ("__value",)

    def __init__(self, value: str = "") -> None:
        self.__value = value

    def reveal(self) -> str:
        return self.__value

    def __bool__(self) -> bool:
        return bool(self.__value)

    def __str__(self) -> str:
        return "<set>" if self else "<unset>"

    __repr__ = __str__

    def __deepcopy__(self, memo: dict[int, object]) -> "SecretValue":
        return SecretValue(self.__value)


@dataclass(frozen=True)
class RetryConfig:
    max_retries: int = 2
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 1.5


@dataclass(frozen=True)
class CircuitBreakerConfig:
    failure_threshold: int = 2


@dataclass(frozen=True)
class ProviderConfig:
    model: str
    base_url: str
    token_limit_parameter: str
    supports_temperature: bool
    api_key: SecretValue = field(default_factory=SecretValue, repr=False)


@dataclass(frozen=True)
class ProviderConfigs:
    deepseek: ProviderConfig
    openai: ProviderConfig


def _default_provider_configs() -> ProviderConfigs:
    return ProviderConfigs(
        deepseek=ProviderConfig(
            "deepseek-chat", "https://api.deepseek.com", "max_tokens", True
        ),
        openai=ProviderConfig(
            "gpt-4o-mini", "https://api.openai.com/v1", "max_completion_tokens", True
        ),
    )


@dataclass(frozen=True, repr=False)
class LLMConfig:
    provider: str = "deepseek"
    rerank_enabled: bool = True
    rerank_candidates: int = 12
    health_check_enabled: bool = True
    connect_timeout_seconds: float = 3.0
    timeout_seconds: float = 8.0
    temperature: float = 0.0
    max_tokens: int = 256
    retry: RetryConfig = field(default_factory=RetryConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    providers: ProviderConfigs = field(default_factory=_default_provider_configs)

    @property
    def selected_profile(self) -> ProviderConfig | None:
        if self.provider == "deepseek":
            return self.providers.deepseek
        if self.provider == "openai":
            return self.providers.openai
        return None

    @property
    def model(self) -> str:
        return self.selected_profile.model if self.selected_profile else ""

    @property
    def base_url(self) -> str:
        return self.selected_profile.base_url if self.selected_profile else ""

    @property
    def api_key(self) -> SecretValue:
        profile = self.selected_profile
        return profile.api_key if profile else SecretValue()

    def __repr__(self) -> str:
        return (
            f"LLMConfig(provider={self.provider!r}, model={self.model!r}, "
            f"api_key={'<set>' if self.api_key else '<unset>'})"
        )


@dataclass(frozen=True)
class AppConfig:
    env_mode: str = "dev"
    retrieval_backend: str = "bm25"
    top_k: int = 10
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    clarify_strategy: str = "other"
    override_erase: bool = False
    skip_data_verify: bool = False
    sample_limit: int | None = None
    output_path: str = "results.json"
    max_constraint_asks: int = 3
    llm: LLMConfig = field(default_factory=LLMConfig)
