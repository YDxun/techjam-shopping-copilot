"""自主决策控制器（Agent 的"决策层"）：依据能力探测结果 + 配置，决定各环节执行方式。

原则：
- 全部能力开关默认关（LLM 意图/澄清默认不启用）；
- 配置开启 + 探测可用 → 真正启用（环境自适应）；
- 配置开启但环境不可用 → 自动降级（回退规则 / 回退 BM25），并记录原因；
- retrieval_backend 支持 auto：稠密可用→hybrid，否则 bm25。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from agent.capability_probe import CapabilityProfile
from config.env_config import EnvConfig

logger = logging.getLogger(__name__)


@dataclass
class RuntimeDecisions:
    """每轮/全局生效的执行方式决策。"""

    retrieval_backend: str = "bm25"  # 生效的检索后端（auto 已解析）
    use_dense: bool = False  # 是否启用稠密通道
    use_llm_intent: bool = False  # 意图识别是否用 LLM
    use_llm_clarify: bool = False  # 澄清决策是否用 LLM
    use_llm_rerank: bool = False  # 重排是否用 LLM
    use_reranker_model: bool = False  # 是否可用 bge 交叉编码重排（FlagEmbedding）
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"retrieval={self.retrieval_backend} "
            f"intent={'llm' if self.use_llm_intent else 'rule'} "
            f"clarify={'llm' if self.use_llm_clarify else 'rule'} "
            f"rerank={'llm' if self.use_llm_rerank else 'rule'} "
            f"reranker_model={'yes' if self.use_reranker_model else 'no'}"
        )


class RuntimeController:
    """把 CapabilityProfile 编译成 RuntimeDecisions（每次会话/启动调用一次）。"""

    def __init__(self, env: EnvConfig, profile: CapabilityProfile) -> None:
        self.env = env
        self.profile = profile

    def decide(self) -> RuntimeDecisions:
        d = RuntimeDecisions()

        # ---- 检索后端：auto / 显式配置 + 环境回退 ----
        backend = self.env.retrieval_backend
        if backend == "auto":
            d.retrieval_backend = "hybrid" if self.profile.dense_available else "bm25"
            d.use_dense = self.profile.dense_available
            d.reasons.append(
                "retrieval_backend=auto -> "
                f"{d.retrieval_backend} (dense={'yes' if d.use_dense else 'no'})"
            )
        elif backend in ("hybrid", "dense"):
            d.retrieval_backend = backend if self.profile.dense_available else "bm25"
            d.use_dense = self.profile.dense_available
            if not self.profile.dense_available:
                d.reasons.append(f"retrieval_backend={backend} 但稠密不可用 -> 回退 bm25")
        else:
            d.retrieval_backend = backend
            d.use_dense = False

        # ---- LLM 决策：配置开启 && 探测可用才启用（默认关、环境自适应）----
        llm_ok = self.profile.llm_available
        d.use_llm_intent = self.env.llm_intent_enabled and llm_ok
        d.use_llm_clarify = self.env.llm_clarify_enabled and llm_ok
        d.use_llm_rerank = self.env.llm.rerank_enabled and llm_ok
        if (
            self.env.llm_intent_enabled
            or self.env.llm_clarify_enabled
            or self.env.llm.rerank_enabled
        ) and not llm_ok:
            d.reasons.append(f"LLM 已配置但不可用（state={self.profile.llm_state}）→ 全部回退规则")

        # ---- 交叉编码重排模型（bge-reranker-v2-m3）：配置开启 && 探测可用才启用 ----
        d.use_reranker_model = self.env.reranker_model_enabled and self.profile.reranker_available
        if self.env.reranker_model_enabled and not self.profile.reranker_available:
            d.reasons.append("RERANKER_MODEL_ENABLE=1 但 bge-reranker 不可用 → 回退规则排序")

        logger.info("[runtime] %s", d.summary())
        return d
