from __future__ import annotations

import unittest
from dataclasses import replace
from types import MappingProxyType

from agent.dialogue.catalog_signals import (
    AttributeSignal,
    CatalogQuestionSignals,
)
from agent.dialogue.models import (
    CandidateAttributeSignal,
    CandidateQuestionSignals,
    ConstraintOperation,
    ConstraintStrength,
    DialogueAct,
    DialogueState,
    OperationKind,
    Polarity,
    RecognitionResult,
    RecognitionSource,
)
from agent.dialogue.question_policy import QuestionPolicy
from config.models import (
    AskUtilityConfig,
    AskUtilityWeights,
    CandidateQuestionValueConfig,
    CandidateQuestionWeights,
    DecisionConfig,
    FinishStrategyConfig,
    FinishWeights,
    StopUtilityConfig,
    StopUtilityWeights,
)


def candidate_signal(
    attribute: str,
    *,
    shrink: float = 0.0,
    resolve10: float = 0.0,
    coverage: float = 1.0,
    p90: float = 20.0,
    missing: float = 0.0,
    confidence: float = 1.0,
    two_step: float = 0.0,
) -> CandidateAttributeSignal:
    return CandidateAttributeSignal(
        attribute=attribute,
        coverage=coverage,
        expected_remaining=20.0 * (1.0 - shrink),
        expected_shrink=shrink,
        resolve_at_10=resolve10,
        resolve_at_3=resolve10,
        resolve_at_1=resolve10,
        p90_remaining=p90,
        worst_case_remaining=20,
        missing_rate=missing,
        extraction_confidence=confidence,
        two_step_finish_gain=two_step,
    )


def dynamic_config(
    *,
    max_questions: int = 3,
    finish_enabled: bool = False,
    lookahead_depth: int = 1,
    candidate_threshold: int = 100,
    remaining_question_threshold: int = 2,
    weights: CandidateQuestionWeights | None = None,
    finish_weights: FinishWeights | None = None,
) -> DecisionConfig:
    return DecisionConfig(
        max_questions=max_questions,
        ask_other_first=True,
        candidate_question_value=CandidateQuestionValueConfig(
            enabled=True,
            other_answer_probability=0.75,
            other_vagueness_penalty=0.10,
            weights=weights or CandidateQuestionWeights(),
        ),
        finish_strategy=FinishStrategyConfig(
            enabled=finish_enabled,
            candidate_threshold=candidate_threshold,
            remaining_question_threshold=remaining_question_threshold,
            lookahead_depth=lookahead_depth,
            weights=finish_weights or FinishWeights(),
        ),
        question_termination_mode="explicit_only",
    )


def dynamic_signals(
    by_attribute: dict[str, CandidateAttributeSignal],
    *,
    count: int = 20,
    other: CandidateAttributeSignal | None = None,
    previous_count: int | None = None,
) -> CandidateQuestionSignals:
    return CandidateQuestionSignals(
        candidate_count=count,
        by_attribute=by_attribute,
        target_probabilities={},
        best_other_pair=("material", "color") if other else None,
        other_signal=other,
        previous_candidate_count=previous_count,
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
        ask_other_first=False,
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
        gap_policy = QuestionPolicy(decision_config(information_gain=0.0, constraint_gap=1.0))

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

    @staticmethod
    def _no_pref_other_recognition(boundary_signal: bool) -> RecognitionResult:
        return RecognitionResult(
            dialogue_act=DialogueAct.NO_PREFERENCE,
            category=None,
            constraint_operations=(
                ConstraintOperation(
                    operation=OperationKind.REMOVE,
                    attribute="other",
                    value="other",
                    polarity=Polarity.INCLUDE,
                    strength=ConstraintStrength.SOFT,
                    evidence="I don't have an additional preference for other.",
                    confidence=0.95,
                ),
            ),
            explicit_rejected_asins=(),
            confidence=0.95,
            source=RecognitionSource.RULE,
            ambiguities=(),
            boundary_signal=boundary_signal,
        )

    def test_no_preference_other_non_boundary_stops(self) -> None:
        policy = QuestionPolicy(decision_config(information_gain=1.0, constraint_gap=0.0))
        state = DialogueState(
            session_id="s1",
            user_profile={},
            category="shoes",
            turn=2,
            no_preference_attributes=frozenset({"other"}),
        )
        decision = policy.decide(
            state, self._no_pref_other_recognition(False), CatalogQuestionSignals.empty()
        )
        self.assertFalse(decision.should_ask)
        self.assertEqual(decision.reason_code, "no_preference_other")

    def test_no_preference_other_boundary_keeps_asking(self) -> None:
        policy = QuestionPolicy(DecisionConfig())  # ask_other_first=True 默认
        state = DialogueState(
            session_id="s1",
            user_profile={},
            category="shoes",
            turn=2,
            no_preference_attributes=frozenset({"other"}),
        )
        decision = policy.decide(
            state, self._no_pref_other_recognition(True), CatalogQuestionSignals.empty()
        )
        self.assertTrue(decision.should_ask)


class DynamicQuestionPolicyTest(unittest.TestCase):
    def test_disabled_or_missing_dynamic_signals_match_legacy_snapshot(self) -> None:
        # Removing the legacy routing branch would change this literal legacy decision.
        state = DialogueState(session_id="s1", user_profile={}, category="shoes", turn=1)
        static = CatalogQuestionSignals(
            by_category={"shoes": {"material": AttributeSignal(1.0, 0.9, 0.9)}}
        )
        expected = QuestionPolicy(DecisionConfig()).decide(state, parsed(), static)
        enabled = dynamic_config()
        disabled = replace(
            enabled,
            candidate_question_value=CandidateQuestionValueConfig(enabled=False),
        )

        self.assertEqual(QuestionPolicy(disabled).decide(state, parsed(), static), expected)
        self.assertEqual(QuestionPolicy(enabled).decide(state, parsed(), static), expected)

    def test_explicit_only_asks_at_turn_nine_and_stops_at_ten(self) -> None:
        # Moving the explicit-only terminal boundary from ten to nine would fail.
        static = CatalogQuestionSignals.empty()
        signals = dynamic_signals({"material": candidate_signal("material", shrink=0.4)})
        policy = QuestionPolicy(dynamic_config())

        ninth = policy.decide(
            DialogueState(session_id="s", user_profile={}, category="shoes", turn=9),
            parsed(),
            static,
            signals,
        )
        tenth = policy.decide(
            DialogueState(session_id="s", user_profile={}, category="shoes", turn=10),
            parsed(),
            static,
            signals,
        )

        self.assertTrue(ninth.should_ask)
        self.assertEqual(ninth.ask_attribute, "material")
        self.assertEqual(tenth.reason_code, "final_turn_no_followup")

    def test_explicit_only_uses_max_questions_as_cost_not_a_stop(self) -> None:
        # Reintroducing the legacy max-question guard would suppress this legal ask.
        state = DialogueState(
            session_id="s",
            user_profile={},
            category="shoes",
            turn=4,
            asked_attributes=("material", "color", "size"),
        )
        decision = QuestionPolicy(dynamic_config(max_questions=1)).decide(
            state,
            parsed(),
            CatalogQuestionSignals.empty(),
            dynamic_signals({"feature": candidate_signal("feature", shrink=0.8)}),
        )

        self.assertTrue(decision.should_ask)
        self.assertEqual(decision.ask_attribute, "feature")

    def test_no_preference_other_uses_a_concrete_legal_attribute(self) -> None:
        # Treating no-preference(other) as a session stop would fail this fallback.
        state = DialogueState(
            session_id="s",
            user_profile={},
            category="shoes",
            turn=2,
            no_preference_attributes=frozenset({"other"}),
        )
        decision = QuestionPolicy(dynamic_config()).decide(
            state,
            QuestionPolicyTest._no_pref_other_recognition(False),
            CatalogQuestionSignals.empty(),
            dynamic_signals(
                {
                    "material": candidate_signal("material", shrink=0.5),
                    "color": candidate_signal("color", shrink=0.2),
                },
                other=candidate_signal("other", shrink=0.9),
            ),
        )

        self.assertTrue(decision.should_ask)
        self.assertEqual(decision.ask_attribute, "material")

    def test_nonpositive_utility_falls_back_to_best_concrete_then_exhausts(self) -> None:
        # Returning no question solely for nonpositive utility would fail this rule.
        weights = CandidateQuestionWeights(
            expected_shrink=0.0,
            coverage=0.0,
            complementarity=0.0,
            answer_probability=0.0,
            missing_penalty=0.0,
            redundancy_penalty=0.0,
            repeat_penalty=0.0,
            no_preference_penalty=1.0,
            turn_cost=0.0,
        )
        policy = QuestionPolicy(dynamic_config(weights=weights))
        state = DialogueState(session_id="s", user_profile={}, category="shoes", turn=2)
        signals = dynamic_signals(
            {
                "material": candidate_signal("material"),
                "color": candidate_signal("color"),
            }
        )
        decision = policy.decide(state, parsed(), CatalogQuestionSignals.empty(), signals)
        exhausted = policy.decide(
            replace(state, no_preference_attributes=frozenset({"material", "color"})),
            parsed(),
            CatalogQuestionSignals.empty(),
            signals,
        )

        self.assertEqual(decision.ask_attribute, "material")
        self.assertEqual(exhausted.reason_code, "all_attributes_exhausted")

    def test_dynamic_components_are_deeply_immutable(self) -> None:
        # Exposing mutable diagnostic mappings would let later callers corrupt a decision trace.
        policy = QuestionPolicy(dynamic_config())
        policy.decide(
            DialogueState(session_id="s", user_profile={}, category="shoes", turn=2),
            parsed(),
            CatalogQuestionSignals.empty(),
            dynamic_signals({"material": candidate_signal("material", shrink=0.5)}),
        )

        self.assertIsInstance(policy.last_components, MappingProxyType)
        with self.assertRaises(TypeError):
            policy.last_components["material"]["utility"] = 0.0


if __name__ == "__main__":
    unittest.main()
