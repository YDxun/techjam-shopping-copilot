"""Pillar I：双轨意图路由（购买高意图轨道 / 浏览开放式轨道）。

- 规则路径（默认、离线可用）：存在 hard 约束 → 购买轨道（高精度硬约束过滤）；
  无约束/过泛 → 浏览轨道（多样化召回）。
- LLM 路径（可选，默认关，环境探测可用时由 runtime_controller 启用）：
  用统一 LLM 客户端做意图判定 + 结构化槽位补充；失败/非法输出 → 严格回退规则路径。
- 输出 IntentRoute：检索关键词、品类域、hard 约束 token 组、soft 词。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agent.dialogue_state_machine import DialogueState
from agent.llm_intent import llm_analyze_intent
from config.env_config import EnvConfig
from llm.base import LLMClient


@dataclass
class IntentRoute:
    track: str = "browsing"                 # buying / browsing
    confidence: float = 0.5
    category_tokens: list[str] = field(default_factory=list)
    hard_groups: list[tuple[str, ...]] = field(default_factory=list)  # 组内 AND
    soft_terms: list[str] = field(default_factory=list)
    query_terms: list[str] = field(default_factory=list)

    @property
    def buying(self) -> bool:
        return self.track == "buying"


class IntentRouter:
    """意图检测：规则为基座，LLM 为可选增强（失败回退规则）。"""

    def __init__(self, env: EnvConfig | None = None) -> None:
        self.env = env or EnvConfig.from_env()

    # ------------------------------------------------------------------
    def route(self, state: DialogueState, mode: str,
              llm_client: LLMClient | None = None,
              use_llm: bool = False,
              user_message: str = "") -> IntentRoute:
        route = self._route_rules(state, mode)

        # 可选 LLM 意图分析：仅当控制器启用且客户端可用；任何失败回退上面的规则结果
        if use_llm and llm_client is not None:
            analysis = llm_analyze_intent(llm_client, state, user_message)
            if analysis is not None:
                self._merge_llm_analysis(route, state, analysis, mode)
        return route

    # ------------------------------------------------------------------
    def _route_rules(self, state: DialogueState, mode: str) -> IntentRoute:
        hard = state.hard
        soft = state.soft
        route = IntentRoute()
        route.category_tokens = list(state.category_tokens)

        # 硬约束 token 组：每个 hard 约束是一组 AND（覆盖度强信号）
        for c in hard:
            if c.tokens:
                route.hard_groups.append(c.tokens)
        # soft 约束词进入宽松检索
        for c in soft:
            route.soft_terms.extend(c.tokens)

        # 双轨判定（Pillar I）
        if len(hard) >= 1:
            route.track = "buying"
            route.confidence = min(0.95, 0.55 + 0.2 * len(hard))
        elif state.flags.get("vague") or state.total_constraints() == 0:
            route.track = "browsing"
            route.confidence = 0.5
        else:
            route.track = "browsing"
            route.confidence = 0.6

        # 查询词 = 品类词 + 约束词
        route.query_terms = list(dict.fromkeys([*route.category_tokens, *route.soft_terms]))
        for group in route.hard_groups:
            route.query_terms.extend(group)
        route.query_terms = list(dict.fromkeys(route.query_terms))[:40]

        # Pillar III 自适应：RECOVER 模式下把 hard 组降级为 soft（放宽过滤）
        if mode == "recover" and route.hard_groups:
            route.soft_terms.extend(t for g in route.hard_groups for t in g)
            route.hard_groups = []
            route.track = "browsing"
        return route

    # ------------------------------------------------------------------
    @staticmethod
    def _merge_llm_analysis(route: IntentRoute, state: DialogueState,
                            analysis: dict, mode: str) -> None:
        """把 LLM 判定合并进路由（保守：LLM 约束只作 soft 词，不直接生成 hard 组）。"""
        track = analysis.get("intent_track")
        if track in ("buying", "browsing"):
            route.track = track
            route.confidence = analysis.get("confidence", route.confidence)

        # LLM 抽取的结构化约束 → 补充 soft 检索词（防幻觉污染 hard 过滤）
        known = state.disclosed_values()
        for key, value in (analysis.get("constraints") or {}).items():
            if value in (None, "", 0):
                continue
            tokens = [t for t in str(value).lower().split() if len(t) > 1]
            for t in tokens:
                if t not in known and t not in route.soft_terms:
                    route.soft_terms.append(t)
        if route.soft_terms:
            route.query_terms = list(dict.fromkeys([*route.query_terms, *route.soft_terms]))[:40]
