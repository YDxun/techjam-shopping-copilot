"""BaseAgent：官方 Agent 契约基类（Pillar IV：兼容官方 Python 接口与 API 契约）。

官方契约（docs/agent_api_contract.json + evaluator/local_evaluator.py 用法）：
    reset(session_id: str, user_profile: dict) -> None
    respond(session_id, user_message, turn, top_k) -> {
        "message": str,
        "ask_attribute": str | None,
        "recommendations": [{"parent_asin": str}],
        "usage": {"prompt_tokens": int, "completion_tokens": int},
    }

注意：不改动官方评估器源码、不改动官方接口定义；子类只重写业务逻辑。
"""
from __future__ import annotations

from typing import Any


class BaseAgent:
    """官方 Agent 接口基类。业务 Agent 继承本类实现逻辑。"""

    def reset(self, session_id: str, user_profile: dict) -> None:
        """每个会话开始前由评估器调用；可在此注入长期用户画像并初始化会话状态。"""
        raise NotImplementedError

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict[str, Any]:
        """每轮对话调用；返回官方契约规定的 dict。"""
        raise NotImplementedError
