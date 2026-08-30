"""BaseAgent: official Agent-contract base class (Pillar IV: compatible with the official Python
    interface and API contract).

Official contract (docs/agent_api_contract.json + how evaluator/local_evaluator.py uses it):
    reset(session_id: str, user_profile: dict) -> None
    respond(session_id, user_message, turn, top_k) -> {
        "message": str,
        "ask_attribute": str | None,
        "recommendations": [{"parent_asin": str}],
        "usage": {"prompt_tokens": int, "completion_tokens": int},
    }

Note: never modify the official evaluator source or interface definitions; subclasses only override
business logic.
"""
from __future__ import annotations

from typing import Any


class BaseAgent:
    """Official Agent-interface base class. Business agents inherit and implement the logic."""

    def reset(self, session_id: str, user_profile: dict) -> None:
        """Called by the evaluator before each session; inject the long-term user profile and
            initialize session state here."""
        raise NotImplementedError

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict[str, Any]:
        """Called per turn; returns the dict required by the official contract."""
        raise NotImplementedError
