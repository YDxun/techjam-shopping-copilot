"""主 Agent 入口：串联全部模块，对外暴露官方要求的 Agent 调用接口。

四大支柱对应关系：
  Pillar I   intent_router + retriever(混合检索) + reranker(LLM/规则重排)
  Pillar II  dialogue_state_machine(动态状态机/槽位) + clarifier(主动澄清)
  Pillar III dynamic_context_program(运行时上下文蒸馏 + 自适应编排)
  Pillar IV  推荐按 TOP_K 对齐 HitRate@K；排序目标提升 MRR；澄清策略优化 MTTC

对外契约（官方接口）：
  reset(session_id, user_profile) / respond(session_id, user_message, turn, top_k)
"""
from __future__ import annotations

import logging
from pathlib import Path

from agent.base_agent import BaseAgent
from agent.clarifier import Clarifier
from agent.dialogue_state_machine import DialogueStateMachine
from agent.dynamic_context_program import DynamicContextProgram
from agent.intent_router import IntentRouter
from agent.retriever import HybridRetriever
from agent.reranker import Reranker
from config.env_config import EnvConfig
from llm.base import DisabledLLMClient, LLMClient
from utils import data_verify

logger = logging.getLogger(__name__)


class Agent(BaseAgent):
    """TechJam2026 购物副驾 Agent（官方接口兼容，业务逻辑完全替换基线）。"""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl",
                 env: EnvConfig | None = None,
                 llm_client: LLMClient | None = None) -> None:
        self.env = env or EnvConfig.from_env()
        self.llm_client = llm_client if llm_client is not None else DisabledLLMClient()

        # 数据集完整性校验（Pillar IV / 硬性约束 3），可 SKIP_DATA_VERIFY=1 跳过
        if not self.env.skip_data_verify:
            data_verify.verify_dataset(skip=False)

        # 组件装配（Pillar I/II/III）
        self.retriever = HybridRetriever(catalog_path=catalog_path, env=self.env)
        self.reranker = Reranker(env=self.env, llm_client=self.llm_client)
        self.state_machine = DialogueStateMachine(override_erase=self.env.override_erase)
        self.router = IntentRouter(env=self.env)
        self.clarifier = Clarifier(env=self.env)
        self.dcp = DynamicContextProgram(env=self.env)
        self.sessions: dict[str, object] = {}

    # ------------------------------------------------------------------
    def reset(self, session_id: str, user_profile: dict) -> None:
        """新会话开始：初始化独立会话状态（内存态），注入长期用户画像。"""
        state = self.state_machine.new_state(session_id, user_profile)
        self.sessions[session_id] = state

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
        state = self.sessions.get(session_id)
        if state is None:
            state = self.state_machine.new_state(session_id, {})
            self.sessions[session_id] = state

        # 1) 上下文蒸馏（Pillar II/III）：消息 → 槽位/信号
        self.state_machine.update(state, user_message, turn)

        # 2) 自适应编排（Pillar III）：状态 → 运行模式/路由权重/是否澄清
        program = self.dcp.adapt(state, turn)

        # 3) 意图路由（Pillar I）：双轨判定 + 检索 query 构建
        route = self.router.route(state, mode=program.retrieval_mode)

        # 4) 多路由混合召回 → 候选池（Pillar I）
        candidates = self.retriever.search(route, top_k=max(self.env.rerank_candidates, top_k * 3),
                                           mode=program.retrieval_mode)

        # 5) 精排（Pillar I/IV）：规则 + 可选 LLM，目标把目标商品推前
        ranked = self.reranker.rerank(self.retriever, candidates, state, route,
                                      top_k=top_k, mode=program.retrieval_mode)

        # 6) 澄清决策（Pillar II）：信息不足/候选过载时主动问，减少轮次（MTTC）
        ask_attribute: str | None = None
        if program.clarify_on:
            ask_attribute, message = self.clarifier.decide(
                state, turn, asked_so_far=program.ask_count)
        else:
            message = self.clarifier._wrap_up_message(state)

        # 7) 长期画像吸收（Pillar III：跨会话稳健先验，内存态）
        self.dcp.absorb_profile(state)

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": [{"parent_asin": a} for a in ranked[:top_k]],
            "usage": {
                "prompt_tokens": self.reranker.last_usage["prompt_tokens"],
                "completion_tokens": self.reranker.last_usage["completion_tokens"],
            },
        }
