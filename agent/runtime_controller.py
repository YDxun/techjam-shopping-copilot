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
from utils import lut as lut_utils

logger = logging.getLogger(__name__)


@dataclass
class RuntimeDecisions:
    """每轮/全局生效的执行方式决策。"""

    retrieval_backend: str = "bm25"  # 生效的检索后端（auto 已解析）
    use_dense: bool = False  # 是否启用稠密通道
    use_llm_intent: bool = False  # 意图识别是否用 LLM
    use_llm_clarify: bool = False  # 澄清决策是否用 LLM
    use_llm_rerank: bool = False  # 重排是否启用（qwen3-rerank text / chat LLM）
    text_rerank_active: bool = False  # 是否走 qwen3-rerank 文本重排
    use_reranker_model: bool = False  # 是否可用 bge 交叉编码重排（FlagEmbedding）
    strategy: str = "bm25_rule"  # 选中的策略标签（环境自适应，见 decide()）
    strategy_lut: str | None = None  # LUT 推荐的最优配置（数据驱动；缺失→None 回退默认）
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        rerank_label = (
            "qwen3" if self.text_rerank_active else "llm" if self.use_llm_rerank else "rule"
        )
        lut_txt = f" lut={self.strategy_lut}" if self.strategy_lut else ""
        return (
            f"strategy={self.strategy}{lut_txt} "
            f"retrieval={self.retrieval_backend} "
            f"intent={'llm' if self.use_llm_intent else 'rule'} "
            f"clarify={'llm' if self.use_llm_clarify else 'rule'} "
            f"rerank={rerank_label} "
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
        # 重排后端决策：text=qwen3-rerank MaaS（替换原 chat JSON 打分）/ chat=旧 LLM /
        # auto=text 可用优先，否则回退 chat；全部失败回退规则排序。
        rr_backend = self.env.llm.rerank_backend
        if self.env.llm.rerank_enabled:
            if rr_backend == "text":
                d.use_llm_rerank = self.profile.text_rerank_available
                d.text_rerank_active = self.profile.text_rerank_available
                if not self.profile.text_rerank_available:
                    d.reasons.append(
                        "LLM_RERANK=1 但 qwen3-rerank 不可用"
                        f"（{self.profile.text_rerank_error or '未配置'}）→ 回退规则排序"
                    )
            elif rr_backend == "chat":
                d.use_llm_rerank = llm_ok
                d.text_rerank_active = False
            else:  # auto
                d.use_llm_rerank = self.profile.text_rerank_available or llm_ok
                d.text_rerank_active = self.profile.text_rerank_available
        if (self.env.llm_intent_enabled or self.env.llm_clarify_enabled) and not llm_ok:
            d.reasons.append(f"LLM 已配置但不可用（state={self.profile.llm_state}）→ 回退规则")

        # ---- 交叉编码重排模型（bge-reranker-v2-m3 / RexReranker）：配置开启 && 探测可用才启用 ----
        d.use_reranker_model = self.env.reranker_model_enabled and self.profile.reranker_available
        if self.env.reranker_model_enabled and not self.profile.reranker_available:
            d.reasons.append(
                f"RERANKER_MODEL_ENABLE=1 但 {self.env.reranker_model} 不可用 → 回退规则排序"
            )

        # ---- 策略标签：环境自适应选出"当前环境最优"配置（默认非永远纯规则）----
        # 公开集 A/B：BLaIR 可用时 hybrid+dense(recover) 0.879 > 纯规则 0.876；
        # 无 BLaIR 时 bm25 规则 0.8757 为环境最优。LLM 可用且开启时级联兜底（安全）。
        parts = [d.retrieval_backend]
        if d.use_llm_intent:
            parts.append("llm_intent")
        if d.use_reranker_model:
            parts.append("rerank_model")
        d.strategy = "_".join(parts) if parts else "rule"
        d.reasons.append(
            f"strategy={d.strategy}（dense={'yes' if d.use_dense else 'no'} "
            f"llm={'yes' if d.use_llm_intent else 'no'}）"
        )

        # Step 3：配置-环境-性能 LUT——按环境指纹推荐最优 config_id（数据驱动启动默认；
        # LUT 缺失 / 环境不在表内 → None，回退上面计算出的默认策略，保底安全）
        try:
            fp = lut_utils.env_fingerprint(
                device=self.profile.device,
                dense=self.profile.dense_available,
                llm=self.profile.llm_available,
                network=self.profile.network_available,
            )
            rec = lut_utils.recommend(fp)
            d.strategy_lut = rec["config_id"] if rec else None
            if rec:
                d.reasons.append(
                    f"LUT[{fp}] -> {rec['config_id']} (ts={rec['technical_score']:.4f})"
                )
        except Exception as exc:  # 任何异常不阻塞启动
            logger.warning("[runtime] LUT 推荐失败（%s）→ 回退默认策略", exc)
            d.strategy_lut = None

        logger.info("[runtime] %s", d.summary())
        return d
