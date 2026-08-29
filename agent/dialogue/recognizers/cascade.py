from __future__ import annotations

import threading
from collections import Counter

from agent.dialogue.models import DialogueAct, RecognitionRequest, RecognitionResult
from agent.dialogue.recognizers.llm import LLMIntentRecognizer
from agent.dialogue.recognizers.rule_based import RuleBasedRecognizer
from llm.base import LLMUsage


class CascadedIntentRecognizer:
    """Rule-first recognition with whole-result LLM replacement and fallback."""

    def __init__(
        self,
        *,
        rule_recognizer: RuleBasedRecognizer,
        llm_recognizer: LLMIntentRecognizer,
        mode: str,
        rule_confidence_threshold: float,
    ) -> None:
        if mode not in {"rule_only", "cascaded"}:
            raise ValueError("mode must be rule_only or cascaded")
        self.rule_recognizer = rule_recognizer
        self.llm_recognizer = llm_recognizer
        self.mode = mode
        self.rule_confidence_threshold = rule_confidence_threshold
        self._local = threading.local()
        self._total_turns = 0
        self._rule_resolutions = 0
        self._llm_attempts = 0
        self._llm_accepted = 0
        self._llm_fallbacks = 0
        self._fallback_reasons: Counter[str] = Counter()

    def recognize(self, request: RecognitionRequest) -> RecognitionResult:
        self._total_turns += 1
        self.last_usage = LLMUsage()
        self.last_fallback_reason = ""
        rule_result = self.rule_recognizer.recognize(request)
        if self.mode == "rule_only" or not self._should_consult_llm(rule_result):
            self._rule_resolutions += 1
            return rule_result
        self._llm_attempts += 1
        llm_result = self.llm_recognizer.recognize(request)
        self.last_usage = self.llm_recognizer.last_usage
        if llm_result is not None:
            self._llm_accepted += 1
            return llm_result
        self._llm_fallbacks += 1
        reason = self.llm_recognizer.last_failure_reason or "unknown"
        self.last_fallback_reason = reason
        self._fallback_reasons[reason] += 1
        return rule_result

    @property
    def last_usage(self) -> LLMUsage:
        return getattr(self._local, "usage", LLMUsage())

    @last_usage.setter
    def last_usage(self, value: LLMUsage) -> None:
        self._local.usage = value

    @property
    def last_fallback_reason(self) -> str:
        return getattr(self._local, "fallback_reason", "")

    @last_fallback_reason.setter
    def last_fallback_reason(self, value: str) -> None:
        self._local.fallback_reason = value

    def statistics(self) -> dict[str, object]:
        """Return a JSON-safe cumulative snapshot for local evaluation diagnostics."""
        return {
            "total_turns": self._total_turns,
            "rule_resolutions": self._rule_resolutions,
            "llm_attempts": self._llm_attempts,
            "llm_accepted": self._llm_accepted,
            "llm_fallbacks": self._llm_fallbacks,
            "fallback_reasons": dict(sorted(self._fallback_reasons.items())),
        }

    def _should_consult_llm(self, result: RecognitionResult) -> bool:
        if not self.llm_recognizer.available:
            return False
        return (
            result.confidence < self.rule_confidence_threshold
            or bool(result.ambiguities)
            or result.dialogue_act in {DialogueAct.AMBIGUOUS, DialogueAct.REPLACE_CONSTRAINT}
        )
