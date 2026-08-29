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
from agent.dialogue.models import GuardAction
from agent.dialogue.pipeline import DialogueUnderstandingPipeline
from agent.intent_router import IntentRouter
from agent.reranker import Reranker
from agent.retriever import HybridRetriever
from agent.runtime_controller import RuntimeController
from config.env_config import EnvConfig
from llm.base import DisabledLLMClient, LLMClient
from utils import data_verify

logger = logging.getLogger(__name__)

# 检索候选池规模：与 LLM 重排提交数 llm.rerank_candidates 解耦。
# 队友分支曾把 rerank_candidates 语义改为 LLM 提交数（默认 12），若继续用它当候选池会把池子
# 缩到 30，导致高频约束下目标商品被挤出候选池（HR@10 0.995 -> 0.855）。
RETRIEVAL_POOL_SIZE = 300


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

    def intent_recognition_statistics(self) -> dict[str, object]:
        """Expose local diagnostics without changing the official turn-response contract."""
        return self.dialogue.recognizer.statistics()

    def transition_guard_statistics(self) -> dict[str, object]:
        """Expose aggregate guard diagnostics without changing the response contract."""
        return self.dialogue.transition_guard.statistics()

    def dialogue_decision_statistics(self) -> dict[str, object]:
        """Expose local decision diagnostics without changing the response contract."""
        return self.dialogue.dialogue_decision_statistics()

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
        pending = self.dialogue.interpret_turn(session_id, user_message, turn)
        context = pending.recommendation_context

        # 1) 对话管线产出推荐上下文；检索/排序链负责 Top10
        route = self.router.route(context, mode=context.retrieval_mode)

        # 2) 多路由混合召回 → 候选池（Pillar I；BLaIR 稠密 + BM25 + 硬约束 AND + 品类）
        candidate_config = self.env.decision.candidate_question_value
        pool_size = (
            max(RETRIEVAL_POOL_SIZE, candidate_config.pool_size)
            if candidate_config.enabled
            else RETRIEVAL_POOL_SIZE
        )
        candidates = self.retriever.search(route, top_k=pool_size, mode=context.retrieval_mode)

        candidate_signals = None
        calculator = self.dialogue.candidate_signal_calculator
        if candidate_config.enabled and calculator is not None:
            try:
                candidate_signals = calculator.calculate(
                    candidates,
                    eligible_attributes=self.dialogue.eligible_candidate_attributes(pending.state),
                    remaining_question_budget=max(
                        0,
                        self.env.decision.max_questions - len(pending.state.asked_attributes),
                    ),
                    terminal_eligible=(
                        pending.state.turn < 10 and not pending.state.no_more_preferences
                    ),
                )
            except Exception:
                logger.exception(
                    "[agent] candidate signal calculation failed; using static question policy"
                )
        turn_result = self.dialogue.decide_question(
            pending,
            candidate_signals,
            candidate_count=len(candidates),
        )

        # 3) 精排（Pillar I/IV）：规则 + 可选 LLM/bge，目标把目标商品推前
        ranked = self.reranker.rerank(
            self.retriever,
            candidates,
            turn_result.recommendation_context,
            route,
            top_k=top_k,
            mode=context.retrieval_mode,
            use_reranker_model=self.decisions.use_reranker_model,
            use_llm_rerank=self.decisions.use_llm_rerank,
        )
        shown = ranked[:top_k]
        if turn_result.guard_decision.action not in {GuardAction.CLARIFY, GuardAction.REJECT}:
            self.dialogue.record_shown(
                session_id,
                shown,
                turn,
                expected_session=turn_result.committed_session,
                expected_fingerprint=turn_result.committed_session_fingerprint,
            )

        decision = turn_result.question_decision
        message = self.dialogue.message_for(decision, turn_result.state)
        usage = {
            "prompt_tokens": turn_result.prompt_tokens + self.reranker.last_usage["prompt_tokens"],
            "completion_tokens": (
                turn_result.completion_tokens + self.reranker.last_usage["completion_tokens"]
            ),
        }
        response = {
            "message": message,
            "ask_attribute": decision.ask_attribute if decision.should_ask else None,
            "recommendations": [{"parent_asin": asin} for asin in shown],
            "usage": usage,
        }
        try:
            self.dialogue.record_completed_decision(
                result=turn_result,
                recommendation_count=len(shown),
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
            )
        except Exception:
            logger.warning("[diagnostics] decision trace capture failed")
        return response
