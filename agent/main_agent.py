"""主 Agent 入口：串联全部模块，对外暴露官方要求的 Agent 调用接口。

主流程对应关系：
  dialogue pipeline 负责识别、状态归约、提问决策与推荐上下文
  intent_router + retriever + reranker 继续负责召回、排序与 Top10

对外契约（官方接口）：
  reset(session_id, user_profile) / respond(session_id, user_message, turn, top_k)
"""

from __future__ import annotations

import logging
from pathlib import Path

from agent.base_agent import BaseAgent
from agent.dialogue.pipeline import DialogueUnderstandingPipeline
from agent.intent_router import IntentRouter
from agent.reranker import Reranker
from agent.retriever import HybridRetriever
from config.env_config import EnvConfig
from llm.base import DisabledLLMClient, LLMClient
from utils import data_verify

logger = logging.getLogger(__name__)


class Agent(BaseAgent):
    """TechJam2026 购物副驾 Agent（官方接口兼容，业务逻辑完全替换基线）。"""

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

        # 数据集完整性校验（Pillar IV / 硬性约束 3），可 SKIP_DATA_VERIFY=1 跳过
        if not self.env.skip_data_verify:
            data_verify.verify_dataset(skip=False)

        # 组件装配（Pillar I/II/III）
        self.retriever = retriever or HybridRetriever(catalog_path=catalog_path, env=self.env)
        self.reranker = reranker or Reranker(env=self.env, llm_client=self.llm_client)
        self.dialogue = DialogueUnderstandingPipeline(
            env=self.env,
            llm_client=self.llm_client,
            products=self.retriever.iter_products(),
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
        turn_result = self.dialogue.process_turn(
            session_id,
            user_message,
            turn,
        )
        context = turn_result.recommendation_context

        # 1) 新子系统只提供推荐上下文；现有推荐链继续拥有 Top10。
        route = self.router.route(context, mode=context.retrieval_mode)

        # 2) 多路由混合召回 → 候选池（Pillar I）
        candidates = self.retriever.search(
            route, top_k=max(self.env.rerank_candidates, top_k * 3), mode=context.retrieval_mode
        )

        # 3) 精排（Pillar I/IV）：规则 + 可选 LLM，目标把目标商品推前
        ranked = self.reranker.rerank(
            self.retriever, candidates, context, route, top_k=top_k, mode=context.retrieval_mode
        )
        shown = ranked[:top_k]
        self.dialogue.record_shown(session_id, shown, turn)

        decision = turn_result.question_decision
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
