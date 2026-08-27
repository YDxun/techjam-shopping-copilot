from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RetryConfig:
    max_retries: int = 2
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 1.5


@dataclass(frozen=True)
class CircuitBreakerConfig:
    failure_threshold: int = 2


@dataclass(frozen=True, repr=False)
class LLMConfig:
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    health_check_enabled: bool = True
    connect_timeout_seconds: float = 3.0
    timeout_seconds: float = 8.0
    temperature: float = 0.0
    max_tokens: int = 256
    retry: RetryConfig = field(default_factory=RetryConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    api_key: str = field(default="", repr=False)

    def __repr__(self) -> str:
        key_state = "<set>" if self.api_key else "<unset>"
        return (
            f"LLMConfig(provider={self.provider!r}, model={self.model!r}, "
            f"base_url={self.base_url!r}, api_key={key_state})"
        )


@dataclass(frozen=True)
class AppConfig:
    env_mode: str = "dev"
    llm_backend: str = "none"
    retrieval_backend: str = "bm25"
    top_k: int = 10
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    clarify_strategy: str = "other"
    llm_rerank: bool = True
    override_erase: bool = False
    skip_data_verify: bool = False
    sample_limit: int | None = None
    output_path: str = "results.json"
    rerank_candidates: int = 300
    max_constraint_asks: int = 3
    llm: LLMConfig = field(default_factory=LLMConfig)
