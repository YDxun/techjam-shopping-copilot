from __future__ import annotations

import math
from types import MappingProxyType
from typing import Mapping

from agent.dialogue.catalog_signals import ATTRIBUTE_ORDER, CatalogQuestionSignals
from agent.dialogue.models import (
    CandidateAttributeSignal,
    CandidateQuestionSignals,
    DialogueAct,
    DialogueState,
    QuestionDecision,
    RecognitionResult,
)
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
    """Deterministic guardrails plus legacy and candidate-aware ask policies."""

    def __init__(self, config: DecisionConfig) -> None:
        self.config = config
        self.last_components: Mapping[str, Mapping[str, float]] = {}

    def decide(
        self,
        state: DialogueState,
        recognition: RecognitionResult,
        signals: CatalogQuestionSignals,
        candidate_signals: CandidateQuestionSignals | None = None,
    ) -> QuestionDecision:
        if (
            candidate_signals is None
            or not self.config.candidate_question_value.enabled
            or self.config.question_termination_mode == "legacy"
        ):
            return self._decide_legacy(state, recognition, signals)
        return self._decide_dynamic(state, recognition, signals, candidate_signals)

    def _decide_legacy(
        self,
        state: DialogueState,
        recognition: RecognitionResult,
        signals: CatalogQuestionSignals,
    ) -> QuestionDecision:
        guardrail = self._guardrail(state)
        if guardrail is not None:
            return self._stop(guardrail)
        # 信息枯竭：顾客对 catch-all 的 "other" 也明确没有额外偏好（非边界措辞）=> 停止提问
        if (
            recognition.dialogue_act == DialogueAct.NO_PREFERENCE
            and not recognition.boundary_signal
            and any(op.attribute == "other" for op in recognition.constraint_operations)
        ):
            return self._stop("no_preference_other")

        stop_score = self._stop_utility(state, recognition)
        if stop_score >= self.config.stop_utility.minimum_stop_utility:
            return QuestionDecision(False, None, "stop_utility_reached", stop_score, {})
        # 数据验证结论：先问 other 平均每轮把候选从 4930 缩到 307、命中保持 0.99
        # （见 data/analysis/report.md）。默认开启，可用 decision.ask_other_first=false 关闭。
        if self.config.ask_other_first:
            return QuestionDecision(True, "other", "ask_other_first", 1.0, {})

        category_signals = signals.for_category(state.category)
        candidates = [
            attribute
            for attribute in ATTRIBUTE_ORDER
            if attribute in category_signals and not (attribute == "category" and state.category)
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
        best = min(
            candidates, key=lambda attribute: (-scores[attribute], ATTRIBUTE_ORDER.index(attribute))
        )
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

    def _decide_dynamic(
        self,
        state: DialogueState,
        recognition: RecognitionResult,
        signals: CatalogQuestionSignals,
        candidate_signals: CandidateQuestionSignals,
    ) -> QuestionDecision:
        if state.no_more_preferences:
            return self._dynamic_stop("user_has_no_more_preferences")
        if state.turn >= 10:
            return self._dynamic_stop("final_turn_no_followup")

        concrete = tuple(
            attribute
            for attribute in ATTRIBUTE_ORDER
            if attribute != "category"
            and attribute != "other"
            and attribute in candidate_signals.by_attribute
            and attribute not in state.no_preference_attributes
        )
        other_legal = (
            candidate_signals.other_signal is not None
            and candidate_signals.best_other_pair is not None
            and "other" not in state.no_preference_attributes
        )
        candidates = concrete + (("other",) if other_legal else ())
        if not candidates:
            return self._dynamic_stop("all_attributes_exhausted")

        finish_gate = self._finish_gate(state, candidate_signals, candidates)
        finish_pressure = self._finish_pressure(state, candidate_signals) if finish_gate else 0.0
        category_signals = signals.for_category(state.category)
        components = {
            attribute: self._dynamic_components(
                attribute,
                state,
                category_signals,
                candidate_signals,
                finish_gate,
                finish_pressure,
            )
            for attribute in candidates
        }
        self.last_components = self._freeze_components(components)
        scores = {attribute: values["utility"] for attribute, values in components.items()}
        best_positive = tuple(attribute for attribute in candidates if scores[attribute] > 0.0)
        if best_positive:
            best = self._best(best_positive, scores)
            reason = "highest_dynamic_utility"
        elif concrete:
            best = self._best(concrete, scores)
            reason = "dynamic_concrete_fallback"
        elif other_legal:
            best = "other"
            reason = "dynamic_other_fallback"
        else:
            return self._dynamic_stop("all_attributes_exhausted")
        ordered_scores = {
            attribute: round(scores[attribute], 6)
            for attribute in ATTRIBUTE_ORDER
            if attribute in scores
        }
        return QuestionDecision(True, best, reason, scores[best], ordered_scores)

    def _dynamic_components(
        self,
        attribute: str,
        state: DialogueState,
        category_signals,
        candidate_signals: CandidateQuestionSignals,
        finish_gate: bool,
        finish_pressure: float,
    ) -> dict[str, float]:
        signal = (
            candidate_signals.other_signal
            if attribute == "other"
            else candidate_signals.by_attribute[attribute]
        )
        assert signal is not None
        weights = self.config.candidate_question_value.weights
        static_signal = category_signals.get(attribute)
        answer_probability = (
            self.config.candidate_question_value.other_answer_probability
            if attribute == "other"
            else self._clamp(getattr(static_signal, "answer_probability", 0.0))
        )
        constrained = {constraint.attribute for constraint in state.active_constraints}
        complementarity = 0.0 if attribute in constrained else 1.0
        redundancy = 1.0 if attribute in constrained else 0.0
        base_exploration = (
            weights.expected_shrink * self._clamp(signal.expected_shrink)
            + weights.coverage * self._clamp(signal.coverage)
            + weights.answer_probability * answer_probability
            - weights.missing_penalty * self._clamp(signal.missing_rate)
        )
        state_gain = (
            weights.complementarity * complementarity
            - weights.redundancy_penalty * redundancy
        )
        vagueness_penalty = (
            self.config.candidate_question_value.other_vagueness_penalty
            if attribute == "other"
            else 0.0
        )
        exploration_gain = (
            answer_probability * base_exploration - vagueness_penalty
            if attribute == "other"
            else base_exploration
        )
        base_finish_gain = self._finish_gain(signal, candidate_signals.candidate_count)
        two_step_finish_gain = (
            self._clamp_nonnegative(signal.two_step_finish_gain)
            if finish_gate and self.config.finish_strategy.lookahead_depth == 2
            else 0.0
        )
        finish_gain = (
            answer_probability * base_finish_gain - vagueness_penalty + two_step_finish_gain
            if attribute == "other"
            else base_finish_gain + two_step_finish_gain
        )
        repeat_penalty = weights.repeat_penalty * float(attribute in state.asked_attributes)
        no_preference_penalty = weights.no_preference_penalty * float(
            attribute in state.no_preference_attributes
        )
        turn_cost = weights.turn_cost * self._turn_cost(state)
        utility = (
            (1.0 - finish_pressure) * exploration_gain
            + finish_pressure * finish_gain
            + state_gain
            - repeat_penalty
            - no_preference_penalty
            - turn_cost
        )
        return {
            "expected_shrink": self._clamp(signal.expected_shrink),
            "coverage": self._clamp(signal.coverage),
            "answer_probability": answer_probability,
            "complementarity": complementarity,
            "missing_penalty": weights.missing_penalty * self._clamp(signal.missing_rate),
            "redundancy_penalty": weights.redundancy_penalty * redundancy,
            "exploration_gain": exploration_gain,
            "state_gain": state_gain,
            "base_finish_gain": base_finish_gain,
            "two_step_finish_gain": two_step_finish_gain,
            "finish_gain": finish_gain,
            "finish_pressure": finish_pressure,
            "repeat_penalty": repeat_penalty,
            "no_preference_penalty": no_preference_penalty,
            "turn_cost": turn_cost,
            "utility": utility,
        }

    def _finish_gate(
        self,
        state: DialogueState,
        signals: CandidateQuestionSignals,
        attributes: tuple[str, ...],
    ) -> bool:
        strategy = self.config.finish_strategy
        remaining = self._remaining_question_budget(state)
        phase_ready = (
            signals.candidate_count > 0
            and (
                signals.candidate_count <= strategy.candidate_threshold
                or remaining <= strategy.remaining_question_threshold
            )
        )
        if not strategy.enabled or not phase_ready:
            return False
        gains = [
            self._finish_gain(
                signals.other_signal if attribute == "other" else signals.by_attribute[attribute],
                signals.candidate_count,
            )
            for attribute in attributes
        ]
        return bool(gains) and max(gains) >= strategy.minimum_finish_gain

    def _finish_pressure(
        self, state: DialogueState, signals: CandidateQuestionSignals
    ) -> float:
        threshold_distance = max(1, self.config.finish_strategy.candidate_threshold - 10)
        candidate_to_top10 = 1.0 - self._clamp(
            (signals.candidate_count - 10) / threshold_distance
        )
        shrink_progress = 0.0
        if signals.previous_candidate_count and signals.previous_candidate_count > 0:
            shrink_progress = self._clamp(
                (signals.previous_candidate_count - signals.candidate_count)
                / signals.previous_candidate_count
            )
        budget_pressure = 1.0 - self._clamp(
            self._remaining_question_budget(state) / max(1, self.config.max_questions)
        )
        turn_pressure = self._clamp((state.turn - 1) / 9.0)
        return (candidate_to_top10 + shrink_progress + budget_pressure + turn_pressure) / 4.0

    def _finish_gain(self, signal: CandidateAttributeSignal, candidate_count: int) -> float:
        if candidate_count <= 0:
            return 0.0
        weights = self.config.finish_strategy.weights
        terminal_progress = self._terminal_progress(signal.expected_remaining, candidate_count)
        return (
            weights.resolve_at_10 * self._clamp(signal.resolve_at_10)
            + weights.resolve_at_3 * self._clamp(signal.resolve_at_3)
            + weights.resolve_at_1 * self._clamp(signal.resolve_at_1)
            + weights.terminal_progress * terminal_progress
            - weights.p90_remaining_penalty
            * self._clamp(signal.p90_remaining / candidate_count)
        )

    @staticmethod
    def _terminal_progress(expected_remaining: float, candidate_count: int) -> float:
        if candidate_count <= 10:
            return 0.0
        initial_distance = math.log1p(candidate_count - 10)
        remaining_distance = math.log1p(max(expected_remaining - 10.0, 0.0))
        return 1.0 - remaining_distance / initial_distance

    def _remaining_question_budget(self, state: DialogueState) -> int:
        return max(0, self.config.max_questions - len(state.asked_attributes))

    def _turn_cost(self, state: DialogueState) -> float:
        budget_pressure = 1.0 - self._clamp(
            self._remaining_question_budget(state) / max(1, self.config.max_questions)
        )
        return (budget_pressure + self._clamp((state.turn - 1) / 9.0)) / 2.0

    @staticmethod
    def _best(attributes: tuple[str, ...], scores: Mapping[str, float]) -> str:
        return min(
            attributes,
            key=lambda attribute: (-scores[attribute], ATTRIBUTE_ORDER.index(attribute)),
        )

    @staticmethod
    def _freeze_components(
        components: Mapping[str, Mapping[str, float]]
    ) -> Mapping[str, Mapping[str, float]]:
        return MappingProxyType(
            {
                attribute: MappingProxyType(dict(values))
                for attribute, values in components.items()
            }
        )

    def _dynamic_stop(self, reason_code: str) -> QuestionDecision:
        self.last_components = self._freeze_components({reason_code: {"utility": 0.0}})
        return self._stop(reason_code)

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
            "no_preference_penalty": (1.0 if attribute in state.no_preference_attributes else 0.0),
            "turn_cost": self._clamp(max(0, state.turn - 1) / 9.0),
        }
        # Legacy diagnostics intentionally remain behavior-compatible mutable mappings.
        if not isinstance(self.last_components, dict):
            self.last_components = {}
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
            weights.constraint_completeness * self._clamp(len(state.active_constraints) / 4.0)
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
    def _clamp_nonnegative(value: float) -> float:
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _stop(reason_code: str) -> QuestionDecision:
        return QuestionDecision(False, None, reason_code, 0.0, {})
