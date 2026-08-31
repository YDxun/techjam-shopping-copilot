"""Per-turn LLM usage/cost recording for the web runtime.

Product goal: the evaluation dashboard shows live token/cost as the user chats with
online LLM features enabled. Recording is:

- in-memory by default (process lifetime), thread-safe;
- optionally appended to a JSONL file when WEBAPP_METRICS_LOG is set (survives restarts).

No API keys are ever stored: events only carry provider/model labels, token counts and
derived USD estimates.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Approximate USD per 1M tokens. Product estimate for transparency; not authoritative.
COST_PER_MTOKEN: dict[str, dict[str, float]] = {
    # Conservative peak/cache-miss estimate; actual DeepSeek V4 pricing varies by time/cache.
    "deepseek:deepseek-v4-flash": {"input": 0.44, "output": 1.32},
    "deepseek:deepseek-chat": {"input": 0.27, "output": 1.10},
    "deepseek:deepseek-reasoner": {"input": 0.55, "output": 2.19},
    "openai:gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai:gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "openai:gpt-4o": {"input": 2.50, "output": 10.00},
    "dashscope:qwen3-rerank": {"input": 0.10, "output": 0.0},
}


def estimate_cost_usd(
    provider: str, model: str, prompt_tokens: int, completion_tokens: int
) -> float:
    rates = COST_PER_MTOKEN.get(f"{provider}:{model}")
    if not rates:
        return 0.0
    return prompt_tokens / 1e6 * rates["input"] + completion_tokens / 1e6 * rates["output"]


class UsageRecorder:
    """Thread-safe in-memory usage store with optional append-only JSONL persistence."""

    def __init__(self, log_path: Path | None = None, max_events: int = 5000) -> None:
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []
        self._max_events = max_events
        self._log_path = log_path
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "UsageRecorder":
        raw = os.environ.get("WEBAPP_METRICS_LOG", "").strip()
        return cls(log_path=Path(raw) if raw else None)

    def record(self, event: dict[str, Any]) -> None:
        event = dict(event)
        event.setdefault("ts", time.time())
        with self._lock:
            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events.pop(0)
            if self._log_path is not None:
                try:
                    with self._log_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(event, ensure_ascii=False) + "\n")
                except Exception:
                    logger.warning("metrics JSONL write failed", exc_info=True)

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            events = list(self._events)
        return list(reversed(events[-limit:]))

    def summary(self) -> dict[str, Any]:
        with self._lock:
            events = list(self._events)
        total_prompt = sum(int(e.get("prompt_tokens") or 0) for e in events)
        total_completion = sum(int(e.get("completion_tokens") or 0) for e in events)
        total_cost = sum(float(e.get("cost_usd") or 0.0) for e in events)
        online = sum(1 for e in events if e.get("online"))
        per_provider: dict[str, dict[str, Any]] = {}
        for event in events:
            raw_sources = event.get("usage_sources")
            sources = raw_sources if isinstance(raw_sources, list) and raw_sources else [event]
            for source in sources:
                if not isinstance(source, dict):
                    continue
                provider = source.get("provider") or "none"
                bucket = per_provider.setdefault(
                    provider,
                    {
                        "provider": provider,
                        "turns": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "cost_usd": 0.0,
                    },
                )
                bucket["turns"] += 1
                bucket["prompt_tokens"] += int(source.get("prompt_tokens") or 0)
                bucket["completion_tokens"] += int(source.get("completion_tokens") or 0)
                bucket["cost_usd"] += float(source.get("cost_usd") or 0.0)
        return {
            "total_turns": len(events),
            "online_turns": online,
            "offline_turns": len(events) - online,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "total_cost_usd": round(total_cost, 6),
            "per_provider": sorted(per_provider.values(), key=lambda item: -item["cost_usd"]),
        }
