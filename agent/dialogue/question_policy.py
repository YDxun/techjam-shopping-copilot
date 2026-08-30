from __future__ import annotations

import hashlib
import random

from agent.dialogue.catalog_signals import ATTRIBUTE_ORDER, CatalogQuestionSignals
from agent.dialogue.models import DialogueAct, DialogueState, QuestionDecision, RecognitionResult
from config.models import DecisionConfig

# Multi-template question phrasing (productized: 4-5 natural templates per attribute, making
# demos/real dialogs feel more like a product).
# Selection strategy controlled by decision.question_template_mode (config/default.json ->
# decision):
#   - "random" (default): pseudo-random selection seeded by (session_id, turn, attribute) --
# different sessions/turns/attributes get different phrasings (looks random) yet fully reproducible
# (same session replays identically).
# - "rotation": deterministic rotation by turn (turn1->template0, turn2->1, turn3->2...), preserving
# the legacy behavior.
# ask_attribute is unaffected by the phrasing text; official simulator replies and scoring stay
# identical (Pillar II proactive clarification).
QUESTION_MESSAGES = {
    "category": [
        "What type of product are you looking for?",
        "Which product category are you browsing?",
        "What kind of item do you have in mind?",
        "Can you tell me what kind of item you're shopping for?",
    ],
    "material": [
        "Do you have a material preference?",
        "What material are you looking for?",
        "Any fabric or material in mind?",
        "Is there a particular fabric you prefer, like cotton or leather?",
        "What kind of material should I focus on for you?",
    ],
    "feature": [
        "Are there any specific features you need?",
        "What features matter most to you?",
        "Any must-have features I should know about?",
        "Are there any particular features that are essential for you?",
        "What would make the perfect item for you — any key features?",
    ],
    "color": [
        "Do you have a color preference?",
        "What color would you like?",
        "Any particular color in mind?",
        "Is there a specific color you're hoping to find?",
        "What color are you leaning toward?",
    ],
    "size": [
        "What size or fit do you need?",
        "Which size works best for you?",
        "What size are you looking for?",
        "Do you have a preferred size or fit?",
        "What size should I keep in mind for you?",
    ],
    "style": [
        "Do you have a preferred style or fit?",
        "What style or fit do you prefer?",
        "Any style you have in mind?",
        "What kind of look or style are you going for?",
    ],
    "use_case": [
        "What will you use it for?",
        "What is the occasion or use case?",
        "How do you plan to use it?",
        "Where are you planning to wear or use it?",
        "What activity or occasion is this for?",
    ],
    "budget": [
        "Do you have a budget in mind?",
        "What is your price range?",
        "Any budget you would like to stay within?",
        "Roughly how much were you hoping to spend?",
        "Is there a price range I should stay within?",
    ],
    "brand": [
        "Do you have a preferred brand?",
        "Any brand you prefer?",
        "Are you looking for a specific brand?",
        "Is there a particular brand you like?",
    ],
    "other": [
        "What else matters most for your choice?",
        "Is there anything else that's important to you?",
        "Any other requirements I should know about?",
        "What else would help me narrow it down for you?",
        "Anything else you'd like me to consider?",
    ],
}

# Valid template-selection strategies (referenced by config validation to avoid magic-string drift)
TEMPLATE_MODES = ("random", "rotation")


class QuestionPolicy:
    """Deterministic guardrails plus configurable ask/stop utilities."""

    def __init__(self, config: DecisionConfig) -> None:
        self.config = config
        self.last_components: dict[str, dict[str, float]] = {}
        # random mode avoids consecutive repeats: track (session_id, attribute) -> last phrasing,
        # so the same attribute never asks the exact same sentence two turns in a row (more natural
        # demos); in-memory, isolated per session.
        self._last_template: dict[tuple[str, str], str] = {}

    def decide(
        self,
        state: DialogueState,
        recognition: RecognitionResult,
        signals: CatalogQuestionSignals,
    ) -> QuestionDecision:
        guardrail = self._guardrail(state)
        if guardrail is not None:
            return self._stop(guardrail)
        # Information exhausted: the customer explicitly has no extra preference for the catch-all
        # "other" (non-boundary wording) => stop asking
        if (
            recognition.dialogue_act == DialogueAct.NO_PREFERENCE
            and not recognition.boundary_signal
            and any(op.attribute == "other" for op in recognition.constraint_operations)
        ):
            return self._stop("no_preference_other")

        stop_score = self._stop_utility(state, recognition)
        if stop_score >= self.config.stop_utility.minimum_stop_utility:
            return QuestionDecision(False, None, "stop_utility_reached", stop_score, {})
        # Data-validated result: asking "other" first shrinks the candidate pool from 4930 to 307
        # per turn while keeping the hit rate at 0.99
        # (see data/analysis/report.md). On by default; disable with decision.ask_other_first=false.
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

    def message_for(self, decision: QuestionDecision, state: DialogueState) -> str:
        if decision.should_ask and decision.ask_attribute:
            templates = QUESTION_MESSAGES[decision.ask_attribute]
            if isinstance(templates, list):
                return self._select_template(templates, state, decision.ask_attribute)
            return templates
        if state.category:
            return f"Here are my best matches for {state.category} — please take a look."
        return "Here are my best matches for you — please take a look."

    def _select_template(
        self, templates: list[str], state: DialogueState, attribute: str
    ) -> str:
        if self.config.question_template_mode == "rotation":
            # Deterministic rotation: pick a template by session turn (turn1->template0, turn2->1,
            # turn3->2...).
            # Not asked_attributes: record_question dedups per attribute and cannot serve as a
            # rotation counter.
            return templates[(max(0, state.turn - 1)) % len(templates)]
        # "random" (default): pseudo-random selection seeded by (session_id, turn, attribute).
        # No global random state (sessions never interfere, thread-safe), and the same session is
        # reproducible.
        key = f"{state.session_id}|{state.turn}|{attribute}".encode("utf-8")
        chosen = random.Random(hashlib.sha256(key).digest()).choice(templates)
        # Anti-consecutive-repeat: if the same sentence was asked last turn for this attribute,
        # advance to the next one so demos don't feel robotic.
        last = self._last_template.get((state.session_id, attribute))
        if last is not None and len(templates) > 1 and chosen == last:
            chosen = templates[(templates.index(chosen) + 1) % len(templates)]
        self._last_template[(state.session_id, attribute)] = chosen
        return chosen

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
    def _stop(reason_code: str) -> QuestionDecision:
        return QuestionDecision(False, None, reason_code, 0.0, {})
