"""主 Agent 入口：融合版 —— 队友对话理解管线 + BLaIR 检索/重排能力。

主流程：
  DialogueUnderstandingPipeline（意图识别/状态归约/提问决策，级联规则+LLM）
    -> RecommendationContext
    -> IntentRouter（双轨）
    -> HybridRetriever（BM25 + 硬约束AND + 品类 + BLaIR 稠密，环境自感知）
    -> Reranker（规则融合 + 可选 LLM/bge 重排）
    -> record_shown（版本化商品反馈）

对外契约（官方接口）：
  reset(session_id, user_profile) / respond(session_id, user_message, turn, top_k)
"""

from __future__ import annotations

import logging
from pathlib import Path

from agent.base_agent import BaseAgent
from agent.capability_probe import CapabilityProbe, CapabilityProfile
from agent.dialogue.pipeline import DialogueUnderstandingPipeline
from agent.intent_router import IntentRouter
from agent.reranker import Reranker
from agent.retriever import HybridRetriever
from agent.runtime_controller import RuntimeController
from config.env_config import EnvConfig
from llm.base import DisabledLLMClient, LLMClient
from utils import data_verify

logger = logging.getLogger(__name__)

# 检索候选池规模现由 config.retrieval_pool_size 控制（Step1 暴露，默认 300）。


class Agent(BaseAgent):
    """TechJam2026 购物副驾 Agent（官方接口兼容，融合对话管线 + BLaIR 检索）。"""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        env: EnvConfig | None = None,
        llm_client: LLMClient | None = None,
        retriever: HybridRetriever | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.env = env or EnvConfig.from_env()
        self.llm_client = llm_client if llm_client is not None else DisabledLLMClient()

        # 环境自感知 + 自主决策：启动时探测能力，决定各环节执行方式
        # 对哨兵/未实现完整协议的 client 健壮：探测失败 → 按 LLM 不可用处理（全部回退规则）
        try:
            self.profile = CapabilityProbe(self.env, self.llm_client).probe()
        except Exception:
            logger.warning(
                "[agent] LLM probe failed on injected client %r; assume unavailable",
                type(self.llm_client).__name__,
            )
            self.profile = CapabilityProfile(
                llm_state="disabled",
                notes=["injected client lacks LLM protocol; capability probe skipped"],
            )
        self.decisions = RuntimeController(self.env, self.profile).decide()
        print(f"[capability] {self.profile.summary()}")
        print(f"[decisions ] {self.decisions.summary()}")

        # 数据集完整性校验（Pillar IV / 硬性约束 3），可 SKIP_DATA_VERIFY=1 跳过
        if not self.env.skip_data_verify:
            data_verify.verify_dataset(skip=False)

        # 组件装配：检索/重排使用已验证的 BLaIR 管线；对话/决策使用队友的对话理解管线
        self.retriever = retriever or HybridRetriever(
            catalog_path=catalog_path,
            env=self.env,
            backend=self.decisions.retrieval_backend,
        )
        self.reranker = reranker or Reranker(env=self.env, llm_client=self.llm_client)
        self.dialogue = DialogueUnderstandingPipeline(
            env=self.env,
            llm_client=self.llm_client,
            products=self.retriever.iter_products(),
            # 自动化控制：LLM 意图识别仅在探测可用且 LLM_INTENT_ENABLE=1 时级联启用，
            # 否则走纯规则识别（离线安全）。澄清决策始终用规则策略（"other-first" 数据验证最优）。
            mode="cascaded" if self.decisions.use_llm_intent else "rule_only",
        )
        self.router = IntentRouter(env=self.env)

    # ------------------------------------------------------------------
    def reset(self, session_id: str, user_profile: dict) -> None:
        """新会话开始：初始化独立会话状态（内存态），注入长期用户画像。"""
        self.dialogue.reset(session_id, user_profile)

    # ------------------------------------------------------------------
    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        """每轮对话主流程（Pillar I~IV 编排）。"""
        try:
            return self._respond_impl(session_id, user_message, turn, top_k)
        except Exception as exc:  # 兜底：任何异常都不向评估器抛错（官方按 miss 计）
            logger.exception("[agent] respond error session=%s turn=%d: %s", session_id, turn, exc)
            return {
                "message": "Let me find the best options for you.",
                "ask_attribute": None,
                "recommendations": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }

    # ------------------------------------------------------------------
    def _respond_impl(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        turn_result = self.dialogue.process_turn(session_id, user_message, turn)
        context = turn_result.recommendation_context

        # 1) 对话管线产出推荐上下文；检索/排序链负责 Top10
        route = self.router.route(context, mode=context.retrieval_mode)

        # 2) 多路由混合召回 → 候选池（Pillar I；BLaIR 稠密 + BM25 + 硬约束 AND + 品类）
        candidates = self.retriever.search(
            route,
            top_k=self.env.retrieval_pool_size,
            mode=context.retrieval_mode,
            shelf=context.category_phrase,
        )

        # 3) 精排（Pillar I/IV）：规则 + 可选 LLM/bge，目标把目标商品推前
        ranked = self.reranker.rerank(
            self.retriever,
            candidates,
            context,
            route,
            top_k=top_k,
            mode=context.retrieval_mode,
            use_reranker_model=self.decisions.use_reranker_model,
            use_llm_rerank=self.decisions.use_llm_rerank,
        )
        decision = turn_result.question_decision

        # 输出门控（捂盘，EMIT_GATE=1）：低置信时少给推荐，高置信/临期/停止提问才满仓。
        # 命中即锁定名次并结束会话 -> 把早期低名次命中推迟到"更确信的 rank1"能大幅提升 MRR。
        emit_k = top_k
        if self.env.emit_gate:
            n_constraints = context.total_constraints()
            if turn >= self.env.emit_late_turn or not decision.should_ask:
                emit_k = top_k
            elif n_constraints == 0:
                emit_k = max(1, min(top_k, self.env.emit_k0))
            elif n_constraints == 1:
                emit_k = max(1, min(top_k, self.env.emit_k1))
            else:
                emit_k = top_k
        shown = ranked[:emit_k]
        self.dialogue.record_shown(session_id, shown, turn)

        message = self.dialogue.message_for(decision, turn_result.state)

        return {
            "message": message,
            "ask_attribute": decision.ask_attribute if decision.should_ask else None,
            "recommendations": [{"parent_asin": asin} for asin in shown],
            "usage": {
                "prompt_tokens": (
                    turn_result.prompt_tokens + self.reranker.last_usage["prompt_tokens"]
                ),
                "completion_tokens": (
                    turn_result.completion_tokens + self.reranker.last_usage["completion_tokens"]
                ),
            },
        }
