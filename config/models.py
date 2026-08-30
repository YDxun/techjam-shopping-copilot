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
    # Data-validated result: asking "other" first shrinks the candidate pool from 4930 to 307 per
    # turn while keeping the hit rate at 0.99
    # (see data/analysis/report.md)
    ask_other_first: bool = True
    # Question-phrasing template selection (Pillar II proactive-clarification copy; does not affect
    # ask_attribute or scoring):
    # random=pseudo-random seeded by (session_id, turn, attribute) (more natural, reproducible
    # demos)
    #   rotation=deterministic rotation by turn (legacy behavior)
    question_template_mode: str = "random"
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
    # Rerank backend: text=Alibaba Cloud MaaS qwen3-rerank (default, replaces the old chat JSON
    # scoring) /
    #         chat=legacy LLM semantic rerank / auto=prefer text when available
    rerank_backend: str = "text"
    qwen_rerank_model: str = "qwen3-rerank"
    dashscope_workspace_id: str = (
        ""  # MaaS base_url subdomain (https://{ws}.cn-beijing.maas.aliyuncs.com/...)
    )
    qwen_rerank_base_url: str = ""  # full base_url override (takes priority over workspace_id concatenation)  # noqa: E501
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
class RetrievalConfig:
    """Retrieval knobs (Step 1: hardcoded values exposed as config; defaults = current values)."""

    bm25_field_weights: tuple[float, ...] = (0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)
    rrf_k: float = 60.0
    rrf_constraint_k: float = 10.0
    dense_weight: float = 0.5
    bm25_limit_mult: int = 2
    recall_limit_mult: int = 3


@dataclass(frozen=True)
class FingerprintConfig:
    """Constraint-combination fingerprint knobs (default off; the smaller the count, the rarer the
        combination)."""

    enable: bool = False
    bonus_unique: float = 1.0
    bonus_ten: float = 0.5
    bonus_fifty: float = 0.2
    max_count: int = 50


def _default_rerank_weights() -> dict[str, float]:
    return {
        "coverage": 0.50,
        "combo": 0.10,
        "category": 0.25,
        "rrf": 0.15,
        "popularity": 0.05,
        "profile": 0.05,
    }


@dataclass(frozen=True)
class RetrievalModeConfig:
    """Pillar III mode-switch thresholds (Part B: moved from hardcoded pipeline values into
        config)."""

    exploit_min_hard: int = 2  # len(state.hard) >= this -> exploit
    exploit_min_constraints: int = 4  # total_constraints >= this -> exploit


@dataclass(frozen=True)
class AppConfig:
    env_mode: str = "dev"
    retrieval_backend: str = "bm25"
    top_k: int = 10
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # BLaIR dense retrieval (Pillar I channel 3): offline pre-computed product vectors + query
    # encoder model
    # At inference only the user query is encoded; product vectors are pre-generated by
    # scripts/encode_catalog_blair.py.
    blair_offline_embedding_path: str = "data/offline_blair_embeds.npy"
    blair_query_encoder_model: str = "hyp1231/blair-roberta-large"
    clarify_strategy: str = "other"
    override_erase: bool = False
    skip_data_verify: bool = False
    sample_limit: int | None = None
    output_path: str = "results.json"
    max_constraint_asks: int = 3
    # Autonomous-capability switches (all off by default): LLM for intent recognition / clarify
    # decisions;
    # truly enabled by runtime_controller only when the LLM is probed available
    # (environment-adaptive).
    llm_intent_enabled: bool = False
    llm_clarify_enabled: bool = False
    # bge-reranker-v2-m3 cross-encoder rerank (off by default; enabled only when available per
    # environment awareness)
    reranker_model_enabled: bool = False
    # Data-asset switches (bundled offline static assets in data/assets/*.json)
    asset_vocab_expand: bool = False  # vocab_v2_clean synonym expansion (constraint-term recall)
    asset_category_expand: bool = False  # category_mapping category-token expansion (first-turn routing)  # noqa: E501
    asset_paraphrase: bool = False  # review_paraphrases review-paraphrase extraction (private-set robustness)  # noqa: E501
    asset_field_map: bool = False  # field_mapping field-aware matching (reserved)
    # Output gating (hold-back): fewer recommendations at low confidence; full capacity only at high
    # confidence / late turn (a hit locks the rank -> raises MRR)
    emit_gate: bool = False  # enable with EMIT_GATE=1
    emit_late_turn: int = 4  # full capacity when turn >= this
    emit_k0: int = 1  # #outputs with 0 constraints
    emit_k1: int = 2  # #outputs with 1 constraint
    emit_k2: int = 10  # #outputs with >=2 constraints before the late turn
    emit_fp_confident: int = 3  # fingerprint uniqueness count <= this -> high confidence, release early  # noqa: E501
    emit_margin_confident: float = 0.10  # top-1 vs top-2 margin >= this -> high confidence, release early  # noqa: E501
    emit_commit_constraints: int = 4  # active constraints >= this -> release full capacity (stop holding back)  # noqa: E501
    # Necessity cue words upgrade to hard (Part A): must/need/require/important/key etc. ->
    # generalized extraction becomes HARD
    hard_cue_enabled: bool = True

    # Retrieval/rerank knobs (Step 1 exposure; defaults equal current values, behavior unchanged;
    # tune harness uses overrides)
    retrieval_pool_size: int = 300
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rerank_weights: dict[str, float] = field(default_factory=_default_rerank_weights)
    fingerprint: FingerprintConfig = field(default_factory=FingerprintConfig)

    dialogue_understanding: DialogueUnderstandingConfig = field(
        default_factory=DialogueUnderstandingConfig
    )
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    retrieval_mode: RetrievalModeConfig = field(default_factory=RetrievalModeConfig)

    llm: LLMConfig = field(default_factory=LLMConfig)
