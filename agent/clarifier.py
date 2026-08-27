"""Pillar II：主动引导澄清逻辑。

- 当候选池过载 / 用户描述过泛（约束稀疏）→ 主动生成结构化澄清提问，
  通过 ask_attribute 向顾客收敛需求，减少轮次，优化 MTTC。
- 策略（CLARIFY_STRATEGY）：
    * other     ：默认。一次最多蒸馏 2 条任意约束（信息量最大，官方契约允许）。
    * attribute ：按学习到的属性优先级（material > feature > color > size > style > use_case）
                  逐属性提问，更"结构化"但每轮信息量更少。
- 防冗余：已问满 MAX_CONSTRAINT_ASKS 轮 / 顾客表示 no additional preference /
  约束已达 4 条 → 停止提问，专注推荐（STOP-ASK），避免无效对话轮次。
"""
from __future__ import annotations

import itertools
import random

from agent.dialogue_state_machine import DialogueState
from config import constants
from config.env_config import EnvConfig

# 属性级澄清的优先级（基于公开集约束类别频率的先验，Pillar III 可运行时调整）
ATTRIBUTE_PRIORITY = ("material", "feature", "color", "size", "style", "use_case", "budget")
_ATTRIBUTE_QUESTION = {
    "material": "Do you have a material preference (e.g. cotton, leather, polyester)?",
    "feature": "Are there any specific features or details you need?",
    "color": "Do you have a color preference?",
    "size": "What size or fit do you need?",
    "style": "Any style or fit preference?",
    "use_case": "What will you use it for?",
    "budget": "Do you have a budget in mind?",
}


class Clarifier:
    """澄清决策器：决定 ask_attribute 与 message（信息通道是 ask_attribute）。"""

    def __init__(self, env: EnvConfig | None = None) -> None:
        self.env = env or EnvConfig.from_env()
        self._open_iter = itertools.cycle(constants.CLARIFY_OPEN_MESSAGES)

    # ------------------------------------------------------------------
    def decide(self, state: DialogueState, turn: int, pool_quality: float = 0.0,
               asked_so_far: int = 0) -> tuple[str | None, str]:
        """返回 (ask_attribute, message)。ask_attribute=None 表示停止澄清。"""
        # 停止条件（Pillar II：避免无效冗余对话）
        if state.flags.get("no_more_pref"):
            return None, self._wrap_up_message(state)
        if state.total_constraints() >= 4:
            return None, self._wrap_up_message(state)
        if asked_so_far >= self.env.max_constraint_asks or turn >= 9:
            return None, self._wrap_up_message(state)

        # 候选池过载 / 过泛 → 主动澄清（Pillar II）
        if self.env.clarify_strategy == "attribute":
            return self._decide_attribute(state)
        return self._decide_other(state)

    # ------------------------------------------------------------------
    def _decide_other(self, state: DialogueState) -> tuple[str, str]:
        """other：信息量最大的澄清（一次最多蒸馏 2 条任意约束）。"""
        asked = state.flags.get("asked_attrs", [])
        if "other" not in asked:
            asked.append("other")
        state.flags["asked_attrs"] = asked
        msg = next(self._open_iter)
        return "other", msg

    def _decide_attribute(self, state: DialogueState) -> tuple[str, str]:
        """attribute：按先验优先级逐个提问。"""
        asked = set(state.flags.get("asked_attrs", []))
        for attr in ATTRIBUTE_PRIORITY:
            if attr not in asked:
                asked.add(attr)
                state.flags["asked_attrs"] = list(asked)
                return attr, _ATTRIBUTE_QUESTION[attr]
        # 全问过 → 兜底 other
        return "other", next(self._open_iter)

    # ------------------------------------------------------------------
    @staticmethod
    def _wrap_up_message(state: DialogueState) -> str:
        if state.category_phrase:
            return f"Here are my best matches for {state.category_phrase} — please take a look."
        return "Here are my best matches for you — please take a look."
