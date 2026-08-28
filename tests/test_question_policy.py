from __future__ import annotations

import unittest

from agent.dialogue.catalog_signals import (
    AttributeSignal,
    CatalogQuestionSignals,
)
from agent.dialogue.models import (
    DialogueAct,
    DialogueState,
    RecognitionResult,
    RecognitionSource,
)
from agent.dialogue.question_policy import QuestionPolicy
from config.models import (
    AskUtilityConfig,
    AskUtilityWeights,
    DecisionConfig,
    StopUtilityConfig,
    StopUtilityWeights,
)


def parsed(confidence: float = 0.9, ambiguities: tuple[str, ...] = ()) -> RecognitionResult:
    return RecognitionResult(
        dialogue_act=DialogueAct.AMBIGUOUS,
        category=None,
        constraint_operations=(),
        explicit_rejected_asins=(),
        confidence=confidence,
        source=RecognitionSource.RULE,
        ambiguities=ambiguities,
    )


def decision_config(
    *,
    information_gain: float,
    constraint_gap: float,
) -> DecisionConfig:
    return DecisionConfig(
        max_questions=3,
        ask_utility=AskUtilityConfig(
            weights=AskUtilityWeights(
                information_gain=information_gain,
                constraint_gap=constraint_gap,
                answer_probability=0.0,
                ambiguity_reduction=0.0,
                repeat_penalty=1.0,
                no_preference_penalty=1.0,
                turn_cost=0.0,
            ),
            minimum_ask_utility=0.01,
        ),
        stop_utility=StopUtilityConfig(
            weights=StopUtilityWeights(
                constraint_completeness=0.0,
                intent_confidence=0.0,
                asked_count=0.0,
                turn_pressure=0.0,
                unresolved_ambiguity=0.0,
            ),
            minimum_stop_utility=1.0,
        ),
    )


class QuestionPolicyTest(unittest.TestCase):
    def test_no_more_preferences_is_a_hard_stop(self) -> None:
        state = DialogueState(
            session_id="s1",
            user_profile={},
            category="shoes",
            no_more_preferences=True,
            turn=4,
        )
        signals = CatalogQuestionSignals(
            by_category={"shoes": {"material": AttributeSignal(0.8, 0.8, 0.8)}}
        )
        policy = QuestionPolicy(DecisionConfig())

        decision = policy.decide(state, parsed(), signals)

        self.assertFalse(decision.should_ask)
        self.assertIsNone(decision.ask_attribute)
        self.assertEqual(decision.reason_code, "user_has_no_more_preferences")

    def test_weight_change_switches_the_selected_attribute_deterministically(self) -> None:
        state = DialogueState(session_id="s1", user_profile={}, category="shoes", turn=1)
        signals = CatalogQuestionSignals(
            by_category={
                "shoes": {
                    "material": AttributeSignal(coverage=1.0, entropy=0.9, answer_probability=0.9),
                    "size": AttributeSignal(coverage=1.0, entropy=0.2, answer_probability=0.9),
                }
            },
            constraint_gap_overrides={"material": 0.2, "size": 1.0},
        )

        information_policy = QuestionPolicy(
            decision_config(information_gain=1.0, constraint_gap=0.0)
        )
        gap_policy = QuestionPolicy(
            decision_config(information_gain=0.0, constraint_gap=1.0)
        )

        self.assertEqual(
            information_policy.decide(state, parsed(), signals).ask_attribute,
            "material",
        )
        self.assertEqual(gap_policy.decide(state, parsed(), signals).ask_attribute, "size")

    def test_default_policy_uses_catch_all_when_the_user_has_no_constraints(self) -> None:
        state = DialogueState(session_id="s1", user_profile={}, category="shoes", turn=1)
        products = [
            {
                "categories": ["Shoes"],
                "title": f"Shoe {index}",
                "features": ["general purpose"],
                "store": f"Brand {index}",
            }
            for index in range(4)
        ]
        signals = CatalogQuestionSignals.from_products(products)

        decision = QuestionPolicy(DecisionConfig()).decide(state, parsed(), signals)

        self.assertEqual(decision.ask_attribute, "other")

    def test_catalog_coverage_and_entropy_change_information_gain(self) -> None:
        products = [
            {"categories": ["Shoes"], "title": "Cotton shoe", "features": ["cotton"]},
            {"categories": ["Shoes"], "title": "Leather shoe", "features": ["leather"]},
            {"categories": ["Shoes"], "title": "Cotton shoe", "features": ["cotton"]},
            {"categories": ["Shoes"], "title": "Leather blue shoe", "features": ["leather"]},
        ]

        signals = CatalogQuestionSignals.from_products(products)
        material = signals.for_category("shoes")["material"]
        color = signals.for_category("shoes")["color"]

        self.assertEqual(material.coverage, 1.0)
        self.assertEqual(color.coverage, 0.25)
        self.assertGreater(material.information_gain, color.information_gain)


if __name__ == "__main__":
    unittest.main()
