"""P1 closed-loop degradation: phase-level circuit breaker (in-process; degrades on the spot on
    exceptions/timeouts).

- Each phase (dense / reranker / llm_intent) tracks consecutive failures; when failures >=
failure_threshold
  it trips (tripped=True) and callers take the degraded path (hybrid->bm25 / rerank->rule /
  llm_intent->rule) with no more per-turn retries;
- one success clears the failure streak (auto-recovery: if a caller still tries after a trip and
succeeds, it resets);
- the reason (why it tripped) is always recorded for P2 session logs and auditing.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class PhaseCircuitBreaker:
    """Phase-level circuit breaker (trips only on exceptions; zero impact on the happy path)."""

    def __init__(self, name: str, failure_threshold: int = 2) -> None:
        self.name = name
        self.failure_threshold = max(1, failure_threshold)
        self.consecutive_failures = 0
        self.tripped = False
        self.trip_reason: str = ""
        self.trip_count = 0

    @property
    def open(self) -> bool:
        """Whether the breaker is open (callers should take the degraded path)."""
        return self.tripped

    def record_success(self) -> None:
        """On one success: clear the failure streak; if it succeeds again after a trip, reset
            (auto-recovery)."""
        if self.tripped:
            logger.info("[breaker] %s circuit reset (subsequent call succeeded)", self.name)
        self.consecutive_failures = 0
        self.tripped = False

    def record_failure(self, reason: str) -> bool:
        """Record one failure; trips when the threshold is reached. Returns whether the trip just
            happened."""
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.failure_threshold and not self.tripped:
            self.tripped = True
            self.trip_count += 1
            self.trip_reason = (
                f"{self.name} {self.consecutive_failures} consecutive failures -> circuit degraded ({reason})"  # noqa: E501
            )
            logger.warning("[breaker] %s", self.trip_reason)
            return True
        return False

    def reset(self) -> None:
        self.consecutive_failures = 0
        self.tripped = False
        self.trip_reason = ""
