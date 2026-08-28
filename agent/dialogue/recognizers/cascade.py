from __future__ import annotations

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
        self.last_usage = LLMUsage()

    def recognize(self, request: RecognitionRequest) -> RecognitionResult:
        self.last_usage = LLMUsage()
        rule_result = self.rule_recognizer.recognize(request)
        if self.mode == "rule_only" or not self._should_consult_llm(rule_result):
            return rule_result
        llm_result = self.llm_recognizer.recognize(request)
        self.last_usage = self.llm_recognizer.last_usage
        return llm_result if llm_result is not None else rule_result

    def _should_consult_llm(self, result: RecognitionResult) -> bool:
        if not self.llm_recognizer.available:
            return False
        return (
            result.confidence < self.rule_confidence_threshold
            or bool(result.ambiguities)
            or result.dialogue_act in {DialogueAct.AMBIGUOUS, DialogueAct.REPLACE_CONSTRAINT}
        )
