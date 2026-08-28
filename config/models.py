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
class DialogueUnderstandingConfig:
    mode: str = "cascaded"
    rule_confidence_threshold: float = 0.75
    max_evidence_length: int = 180


@dataclass(frozen=True)
class AskUtilityWeights:
    information_gain: float = 0.30
    constraint_gap: float = 0.25
    answer_probability: float = 0.15
    ambiguity_reduction: float = 0.20
    repeat_penalty: float = 0.40
    no_preference_penalty: float = 0.60
    turn_cost: float = 0.15


@dataclass(frozen=True)
class AskUtilityConfig:
    weights: AskUtilityWeights = field(default_factory=AskUtilityWeights)
    normalization: str = "clamp_0_1"
    minimum_ask_utility: float = 0.20


@dataclass(frozen=True)
class StopUtilityWeights:
    constraint_completeness: float = 0.35
    intent_confidence: float = 0.25
    asked_count: float = 0.15
    turn_pressure: float = 0.25
    unresolved_ambiguity: float = 0.30


@dataclass(frozen=True)
class StopUtilityConfig:
    weights: StopUtilityWeights = field(default_factory=StopUtilityWeights)
    minimum_stop_utility: float = 0.55


@dataclass(frozen=True)
class DecisionConfig:
    max_questions: int = 3
    # 数据验证结论：先问 other 平均每轮把候选从 4930 缩到 307、命中保持 0.99
    # （见 data/analysis/report.md）
    ask_other_first: bool = True
    ask_utility: AskUtilityConfig = field(default_factory=AskUtilityConfig)
    stop_utility: StopUtilityConfig = field(default_factory=StopUtilityConfig)


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
        deepseek=ProviderConfig("deepseek-chat", "https://api.deepseek.com", "max_tokens", True),
        openai=ProviderConfig(
            "gpt-4o-mini", "https://api.openai.com/v1", "max_completion_tokens", True
        ),
    )


@dataclass(frozen=True, repr=False)
class LLMConfig:
    provider: str = "deepseek"
    rerank_enabled: bool = False
    rerank_candidates: int = 12
    # 重排后端：text=阿里云 MaaS qwen3-rerank（默认，替换原 chat JSON 打分）/
    #         chat=旧 LLM 语义重排 / auto=text 可用优先
    rerank_backend: str = "text"
    qwen_rerank_model: str = "qwen3-rerank"
    dashscope_workspace_id: str = ""    # MaaS base_url 子域（https://{ws}.cn-beijing.maas.aliyuncs.com/...）
    qwen_rerank_base_url: str = ""      # 完整 base_url 覆盖（优先于 workspace_id 拼接）
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
    # BLaIR 稠密检索（Pillar I 通道3）：离线预计算商品向量 + 查询编码模型
    # 推理阶段只编码用户查询；商品向量由 scripts/encode_catalog_blair.py 预先生成。
    blair_offline_embedding_path: str = "data/offline_blair_embeds.npy"
    blair_query_encoder_model: str = "hyp1231/blair-roberta-large"
    clarify_strategy: str = "other"
    override_erase: bool = False
    skip_data_verify: bool = False
    sample_limit: int | None = None
    output_path: str = "results.json"
    max_constraint_asks: int = 3
    # 自主能力开关（默认全关）：LLM 用于意图识别 / 澄清决策；
    # 由 runtime_controller 在探测到 LLM 实际可用时才真正启用（环境自适应）。
    llm_intent_enabled: bool = False
    llm_clarify_enabled: bool = False
    # bge-reranker-v2-m3 交叉编码重排（默认关，环境自感知开启时可用才启用）
    reranker_model_enabled: bool = False

    dialogue_understanding: DialogueUnderstandingConfig = field(
        default_factory=DialogueUnderstandingConfig
    )
    decision: DecisionConfig = field(default_factory=DecisionConfig)

    llm: LLMConfig = field(default_factory=LLMConfig)
