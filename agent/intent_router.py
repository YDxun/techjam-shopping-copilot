"""Pillar I：双轨意图路由（购买高意图轨道 / 浏览开放式轨道）。

- 购买轨道：存在 hard 约束（"key requirement"/"what matters"）→ 高精度硬约束过滤。
- 浏览轨道：无 hard 约束、仍在探索 → 多样化稠密/泛化召回。
- 输出 IntentRoute：检索关键词、品类域、hard 约束 token 组、soft 词，
  下游检索与重排据此动态选择路由权重（Pillar III 自适应编排会改写权重）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agent.dialogue.models import RecommendationContext
from config.env_config import EnvConfig


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
    """基于状态信号的轻量意图检测（无 LLM 也可运行，符合离线约束）。"""

    def __init__(self, env: EnvConfig | None = None) -> None:
        self.env = env or EnvConfig.from_env()

    # ------------------------------------------------------------------
    def route(self, state: RecommendationContext, mode: str) -> IntentRoute:
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
        elif state.buying_or_browsing == "browsing" or state.total_constraints() == 0:
            route.track = "browsing"
            route.confidence = 0.5
        else:
            route.track = "browsing"
            route.confidence = 0.6

        # 查询词 = 品类词 + 约束词（Pillar I 多路由检索的 query 构建）
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
