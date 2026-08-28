from __future__ import annotations

from agent.dialogue.catalog_signals import ATTRIBUTE_ORDER, CatalogQuestionSignals
from agent.dialogue.models import DialogueState, QuestionDecision, RecognitionResult
from config.models import DecisionConfig


QUESTION_MESSAGES = {
    "category": "What type of product are you looking for?",
    "material": "Do you have a material preference?",
    "feature": "Are there any specific features you need?",
    "color": "Do you have a color preference?",
    "size": "What size or fit do you need?",
    "style": "Do you have a preferred style or fit?",
    "use_case": "What will you use it for?",
    "budget": "Do you have a budget in mind?",
    "brand": "Do you have a preferred brand?",
    "other": "What else matters most for your choice?",
}


class QuestionPolicy:
    """Deterministic guardrails plus configurable ask/stop utilities."""

    def __init__(self, config: DecisionConfig) -> None:
        self.config = config
        self.last_components: dict[str, dict[str, float]] = {}

    def decide(
        self,
        state: DialogueState,
        recognition: RecognitionResult,
        signals: CatalogQuestionSignals,
    ) -> QuestionDecision:
        guardrail = self._guardrail(state)
        if guardrail is not None:
            return self._stop(guardrail)

        category_signals = signals.for_category(state.category)
        candidates = [
            attribute
            for attribute in ATTRIBUTE_ORDER
            if attribute in category_signals
            and not (attribute == "category" and state.category)
        ]
        if not candidates:
            return self._stop("no_candidate_attribute")

        scores = {
            attribute: self._ask_utility(
                attribute,
                state,
                recognition,
                signals,
                category_signals[attribute],
            )
            for attribute in candidates
        }
        best = min(candidates, key=lambda attribute: (-scores[attribute], ATTRIBUTE_ORDER.index(attribute)))
        best_score = scores[best]
        stop_score = self._stop_utility(state, recognition)
        ordered_scores = {
            attribute: round(scores[attribute], 6)
            for attribute in ATTRIBUTE_ORDER
            if attribute in scores
        }
        if stop_score >= self.config.stop_utility.minimum_stop_utility:
            return QuestionDecision(False, None, "stop_utility_reached", stop_score, ordered_scores)
        if best_score < self.config.ask_utility.minimum_ask_utility:
            return QuestionDecision(False, None, "ask_utility_too_low", best_score, ordered_scores)
        return QuestionDecision(True, best, "highest_ask_utility", best_score, ordered_scores)

    def message_for(self, decision: QuestionDecision, state: DialogueState) -> str:
        if decision.should_ask and decision.ask_attribute:
            return QUESTION_MESSAGES[decision.ask_attribute]
        if state.category:
            return f"Here are my best matches for {state.category} — please take a look."
        return "Here are my best matches for you — please take a look."

    def _guardrail(self, state: DialogueState) -> str | None:
        if state.no_more_preferences:
            return "user_has_no_more_preferences"
        if len(state.asked_attributes) >= self.config.max_questions:
            return "maximum_questions_reached"
        if state.turn >= 9:
            return "turn_limit_guardrail"
        return None

    def _ask_utility(
        self,
        attribute: str,
        state: DialogueState,
        recognition: RecognitionResult,
        signals: CatalogQuestionSignals,
        signal,
    ) -> float:
        weights = self.config.ask_utility.weights
        constrained = {item.attribute for item in state.active_constraints}
        default_gap = 0.0 if attribute in constrained else 1.0
        if attribute == "other":
            default_gap = max(0.0, 1.0 - len(state.active_constraints) / 4.0)
        components = {
            "information_gain": self._clamp(signal.information_gain),
            "constraint_gap": self._clamp(
                signals.constraint_gap_overrides.get(attribute, default_gap)
            ),
            "answer_probability": self._clamp(signal.answer_probability),
            "ambiguity_reduction": 1.0 if recognition.ambiguities else 0.0,
            "repeat_penalty": 1.0 if attribute in state.asked_attributes else 0.0,
            "no_preference_penalty": (
                1.0 if attribute in state.no_preference_attributes else 0.0
            ),
            "turn_cost": self._clamp(max(0, state.turn - 1) / 9.0),
        }
        self.last_components[attribute] = components
        return (
            weights.information_gain * components["information_gain"]
            + weights.constraint_gap * components["constraint_gap"]
            + weights.answer_probability * components["answer_probability"]
            + weights.ambiguity_reduction * components["ambiguity_reduction"]
            - weights.repeat_penalty * components["repeat_penalty"]
            - weights.no_preference_penalty * components["no_preference_penalty"]
            - weights.turn_cost * components["turn_cost"]
        )

    def _stop_utility(
        self,
        state: DialogueState,
        recognition: RecognitionResult,
    ) -> float:
        weights = self.config.stop_utility.weights
        return (
            weights.constraint_completeness
            * self._clamp(len(state.active_constraints) / 4.0)
            + weights.intent_confidence * self._clamp(recognition.confidence)
            + weights.asked_count
            * self._clamp(len(state.asked_attributes) / self.config.max_questions)
            + weights.turn_pressure * self._clamp(state.turn / 10.0)
            - weights.unresolved_ambiguity * (1.0 if recognition.ambiguities else 0.0)
        )

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _stop(reason_code: str) -> QuestionDecision:
        return QuestionDecision(False, None, reason_code, 0.0, {})
