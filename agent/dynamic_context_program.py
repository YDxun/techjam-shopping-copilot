"""Pillar III：自我进化 · 动态上下文编程（运行时适配 + 自适应编排）。

1) 运行时上下文适配（上下文蒸馏）：
   - 短时会话上下文：每轮把对话历史蒸馏成 ContextProgram（约束、品类、意图轨道、
     模式、路由权重、置信度），检索/澄清/重排模块都读它"重新编译"执行。
   - 长期用户画像：会话级 user_profile（不写磁盘），叠加进程内跨会话统计先验
     （preference_tag → 约束类别频率），只做微小加权，避免过拟合公开集。

2) 自适应编排（运行时工作流重编排）：
   - 根据会话状态动态切换运行模式（Pillar II/IV）：
       probe      : 信息不足 → 问 + 宽召回
       exploit    : 约束充足 → 硬约束过滤 + 精排收敛（优化 MRR / MTTC）
       recover    : 连续未命中/过泛 → 放宽过滤、扩大召回（提升 HitRate@K）
       stop_ask   : 顾客无更多偏好 → 停止澄清，专注推荐
   - 路由权重、是否触发澄清、检索模式全部由本模块运行时决定。
"""
from __future__ import annotations

import logging
from collections import defaultdict, Counter
from dataclasses import dataclass, field

from agent.dialogue_state_machine import DialogueState
from config.env_config import EnvConfig

logger = logging.getLogger(__name__)

MODE_PROBE = "probe"
MODE_EXPLOIT = "exploit"
MODE_RECOVER = "recover"
MODE_STOP_ASK = "stop_ask"


@dataclass
class ContextProgram:
    """一轮运行时"上下文程序"：各模块按它执行（动态上下文编程的编译产物）。"""

    mode: str = MODE_PROBE
    confidence: float = 0.5
    route_buy_weight: float = 0.6      # 购买轨道检索权重
    route_browse_weight: float = 0.4   # 浏览轨道检索权重
    clarify_on: bool = True            # 是否触发澄清
    filter_hard: bool = False          # 是否执行 hard 硬过滤
    retrieval_mode: str = "probe"
    ask_count: int = 0
    notes: list[str] = field(default_factory=list)


class DynamicContextProgram:
    """运行时上下文蒸馏 + 自适应编排器（无需模型训练，纯上下文编程）。"""

    def __init__(self, env: EnvConfig | None = None) -> None:
        self.env = env or EnvConfig.from_env()
        # 跨会话长期画像统计（内存态，不持久化；Pillar III）
        self.profile_prior: dict[str, Counter] = defaultdict(Counter)  # tag -> attr_type counts

    # ------------------------------------------------------------------
    # 运行时适配：把会话状态编译成 ContextProgram
    # ------------------------------------------------------------------
    def adapt(self, state: DialogueState, turn: int) -> ContextProgram:
        prog = ContextProgram()
        n_hard = len(state.hard)
        n_soft = len(state.soft)
        total = state.total_constraints()

        # 置信度：hard 约束越多越可信
        prog.confidence = min(0.95, 0.35 + 0.2 * n_hard + 0.05 * n_soft)

        # 1) 模式选择（自适应编排核心）
        if state.flags.get("no_more_pref"):
            prog.mode = MODE_STOP_ASK
        elif total >= 4 or n_hard >= 2:
            prog.mode = MODE_EXPLOIT
        elif state.flags.get("vague") or (turn >= 6 and total == 0):
            prog.mode = MODE_RECOVER
        else:
            prog.mode = MODE_PROBE

        # 2) 模式 → 检索/澄清行为
        if prog.mode == MODE_EXPLOIT:
            prog.filter_hard = True
            prog.clarify_on = False
            prog.retrieval_mode = "exploit"
            prog.route_buy_weight, prog.route_browse_weight = 0.8, 0.2
        elif prog.mode == MODE_RECOVER:
            prog.filter_hard = False
            prog.clarify_on = True
            prog.retrieval_mode = "recover"
            prog.route_buy_weight, prog.route_browse_weight = 0.3, 0.7
        elif prog.mode == MODE_STOP_ASK:
            prog.clarify_on = False
            prog.filter_hard = True
            prog.retrieval_mode = "exploit"
        else:  # probe
            prog.clarify_on = True
            prog.filter_hard = False
            prog.retrieval_mode = "probe"
            prog.route_buy_weight, prog.route_browse_weight = 0.6, 0.4

        prog.ask_count = len(state.flags.get("asked_attrs", []))
        return prog

    # ------------------------------------------------------------------
    # 长期画像维护（跨会话，只学稳健先验）
    # ------------------------------------------------------------------
    def absorb_profile(self, state: DialogueState) -> None:
        """把会话的 user_profile 标签与最终约束类别映射进长期统计。"""
        tags = [t.lower() for t in (state.user_profile or {}).get("preference_tags", []) if isinstance(t, str)]
        for c in state.constraints:
            for tag in tags:
                self.profile_prior[tag][c.attr_type] += 1

    def attribute_prior(self, tags: list[str]) -> list[str]:
        """给定画像标签，返回信息量排序的属性优先级（Pillar III 策略对齐）。"""
        ranking: Counter = Counter()
        for tag in tags:
            ranking.update(self.profile_prior.get(tag.lower(), Counter()))
        if not ranking:
            return ["material", "feature", "color", "size", "style", "use_case", "budget"]
        return [k for k, _ in ranking.most_common()]

    # ------------------------------------------------------------------
    @staticmethod
    def describe(prog: ContextProgram) -> str:
        return (f"mode={prog.mode} conf={prog.confidence:.2f} "
                f"clarify={prog.clarify_on} filter_hard={prog.filter_hard}")
