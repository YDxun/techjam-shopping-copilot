"""Pillar II：主动引导澄清逻辑。

- 规则策略（默认、离线可用）：other 最大信息量 / attribute 按属性优先级；
  停止条件（顾客无更多偏好、约束饱和、提问上限）优先判断，避免无效轮次（优化 MTTC）。
- LLM 策略（可选，默认关，探测可用时由 runtime_controller 启用）：
  由 LLM 决定 ask_attribute + 自然语言问题；失败/非法输出 → 回退规则策略。
"""
from __future__ import annotations

import itertools

from agent.dialogue_state_machine import DialogueState
from agent.llm_intent import llm_decide_clarification
from config import constants
from config.env_config import EnvConfig
from llm.base import LLMClient

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
    """澄清决策器：规则为基座，LLM 为可选增强（失败回退规则）。"""

    def __init__(self, env: EnvConfig | None = None) -> None:
        self.env = env or EnvConfig.from_env()
        self._open_iter = itertools.cycle(constants.CLARIFY_OPEN_MESSAGES)

    # ------------------------------------------------------------------
    def decide(self, state: DialogueState, turn: int, pool_quality: float = 0.0,
               asked_so_far: int = 0,
               llm_client: LLMClient | None = None,
               use_llm: bool = False) -> tuple[str | None, str]:
        """返回 (ask_attribute, message)。ask_attribute=None 表示停止澄清。"""
        # 停止条件（Pillar II：避免无效冗余对话；规则与 LLM 共用）
        if state.flags.get("no_more_pref"):
            return None, self._wrap_up_message(state)
        if state.total_constraints() >= 4:
            return None, self._wrap_up_message(state)
        if asked_so_far >= self.env.max_constraint_asks or turn >= 9:
            return None, self._wrap_up_message(state)

        # 可选 LLM 澄清：仅当控制器启用且客户端可用；失败/非法 → 回退规则
        if use_llm and llm_client is not None:
            llm_out = llm_decide_clarification(llm_client, state, pool_quality, turn)
            if llm_out is not None:
                ask, message = llm_out
                if ask is None:
                    return None, message
                return ask, message

        # 规则回退
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
        return "other", next(self._open_iter)

    # ------------------------------------------------------------------
    @staticmethod
    def _wrap_up_message(state: DialogueState) -> str:
        if state.category_phrase:
            return f"Here are my best matches for {state.category_phrase} — please take a look."
        return "Here are my best matches for you — please take a look."
