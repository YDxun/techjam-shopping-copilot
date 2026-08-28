"""LLM 意图识别 / 澄清决策（可选能力，默认关，失败严格回退规则）。

- 通过统一 LLM 客户端调用；prompt 要求严格 JSON 输出；
- 解析失败 / 校验失败 / 客户端失败 → 返回 None，由调用方回退规则实现；
- 超时/熔断由 llm.openai_compatible 客户端保证（connect 3s / total 8s，熔断 2 次）。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from agent.dialogue_state_machine import DialogueState
from config import constants
from llm.base import LLMClient

logger = logging.getLogger(__name__)

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)
_MAX_TOKENS = 200


def _safe_json(content: str) -> dict[str, Any] | None:
    """从模型输出提取 JSON（容忍代码围栏/前后缀文本）。"""
    if not content:
        return None
    text = content.strip()
    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        start, end = text.index("{"), text.rindex("}")
        parsed = json.loads(text[start:end + 1])
    except (ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _clamp_confidence(value: Any) -> float:
    try:
        conf = float(value)
        return max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        return 0.5


# ---------------------------------------------------------------------------
# 意图识别（Pillar I：LLM 双轨判定 + 槽位补充）
# ---------------------------------------------------------------------------
def llm_analyze_intent(client: LLMClient, state: DialogueState, user_message: str) -> dict[str, Any] | None:
    """分析用户消息 → {intent_track, constraints, override_detected, confidence}；失败返回 None。"""
    known = "; ".join(f"{c.attr_type}:{c.value}" for c in state.active) or "none"
    prompt = (
        "You are the intent analyzer of a shopping copilot. Given the user's latest message "
        "and already-known constraints, decide the shopping intent track and extract structured constraints.\n"
        f"known constraints: {known}\n"
        f"user message: {user_message}\n"
        "Reply ONLY with JSON: "
        '{"intent_track":"buying|browsing","constraints":{"material":"","color":"","size":"","budget_max":0},'
        '"override_detected":true|false,"confidence":0.0} '
        "Use empty strings / 0 when unknown."
    )
    result = client.chat([{"role": "user", "content": prompt}],
                         temperature=0.0, max_tokens=_MAX_TOKENS)
    if not result.success:
        logger.debug("[llm_intent] unavailable: %s", result.error_category)
        return None
    parsed = _safe_json(result.content)
    if parsed is None:
        return None
    track = parsed.get("intent_track")
    if track not in ("buying", "browsing"):
        return None
    constraints = parsed.get("constraints")
    return {
        "intent_track": track,
        "constraints": constraints if isinstance(constraints, dict) else {},
        "override_detected": bool(parsed.get("override_detected")),
        "confidence": _clamp_confidence(parsed.get("confidence")),
    }


# ---------------------------------------------------------------------------
# 澄清决策（Pillar II：主动引导 —— LLM 决定问什么属性）
# ---------------------------------------------------------------------------
def llm_decide_clarification(client: LLMClient, state: DialogueState,
                             pool_quality: float, turn: int) -> tuple[str | None, str] | None:
    """让 LLM 决定 ask_attribute + 自然语言问题；失败/非法返回 None（回退规则）。"""
    known = "; ".join(f"{c.attr_type}:{c.value}" for c in state.active) or "none"
    allowed = ", ".join(sorted(constants.ALLOWED_ASK_ATTRIBUTES)) + ", null"
    prompt = (
        "You are the clarifying-question planner of a shopping copilot. "
        "The user is still exploring and we need ONE more attribute to narrow the search.\n"
        f"known constraints: {known}\n"
        f"candidate-pool quality: {pool_quality:.2f} (lower = too broad)\n"
        f"turn: {turn}\n"
        f"Pick the single most informative ask_attribute from: {allowed} "
        "(null only if nothing more to ask), and write a short natural clarifying question.\n"
        'Reply ONLY with JSON: {"ask_attribute":"...","message":"..."}'
    )
    result = client.chat([{"role": "user", "content": prompt}],
                         temperature=0.0, max_tokens=_MAX_TOKENS)
    if not result.success:
        return None
    parsed = _safe_json(result.content)
    if parsed is None:
        return None
    ask = parsed.get("ask_attribute")
    message = parsed.get("message")
    if ask is not None and ask not in constants.ALLOWED_ASK_ATTRIBUTES:
        return None
    if not isinstance(message, str) or not message.strip():
        return None
    return ask, message.strip()
