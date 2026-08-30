from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from agent.dialogue.models import (
    CandidateAttributeSignal,
    CandidateQuestionSignals,
    GuardAction,
)
from config.env_config import EnvConfig
from config.loader import ConfigError, load_config


class DialogueConfigTest(unittest.TestCase):
    def test_hybrid_question_defaults_preserve_legacy_behavior(self) -> None:
        decision = load_config(environ={}).decision

        self.assertFalse(decision.hybrid_question_policy.enabled)
        self.assertEqual(decision.hybrid_question_policy.pool_size, 300)
        self.assertEqual(decision.hybrid_question_policy.max_replacements_per_session, 1)
        self.assertTrue(decision.hybrid_question_policy.only_after_other_asked)

    def test_hybrid_question_environment_overrides(self) -> None:
        decision = load_config(
            environ={"SHOPPING_DECISION__HYBRID_QUESTION_POLICY__ENABLED": "1"}
        ).decision

        self.assertTrue(decision.hybrid_question_policy.enabled)

    def test_dynamic_question_defaults_preserve_legacy_behavior(self) -> None:
        decision = EnvConfig.from_env(environ={}).decision

        self.assertFalse(decision.candidate_question_value.enabled)
        self.assertEqual(decision.question_termination_mode, "legacy")
        self.assertFalse(decision.finish_strategy.enabled)
        self.assertEqual(decision.candidate_question_value.pool_size, 300)

    def test_dynamic_question_environment_overrides(self) -> None:
        decision = EnvConfig.from_env(
            environ={
                "SHOPPING_DECISION__QUESTION_TERMINATION_MODE": "explicit_only",
                "SHOPPING_DECISION__CANDIDATE_QUESTION_VALUE__ENABLED": "1",
                "SHOPPING_DECISION__CANDIDATE_QUESTION_VALUE__POOL_SIZE": "500",
                "SHOPPING_DECISION__CANDIDATE_QUESTION_VALUE__PRIOR_ALPHA": "0.5",
                "SHOPPING_DECISION__CANDIDATE_QUESTION_VALUE__PRIOR_TEMPERATURE": "2.0",
                "SHOPPING_DECISION__CANDIDATE_QUESTION_VALUE__OTHER_ANSWER_PROBABILITY": "0.6",
                "SHOPPING_DECISION__CANDIDATE_QUESTION_VALUE__OTHER_VAGUENESS_PENALTY": "0.2",
                "SHOPPING_DECISION__CANDIDATE_QUESTION_VALUE__WEIGHTS__EXPECTED_SHRINK": "0.4",
                "SHOPPING_DECISION__CANDIDATE_QUESTION_VALUE__WEIGHTS__COVERAGE": "0.16",
                "SHOPPING_DECISION__CANDIDATE_QUESTION_VALUE__WEIGHTS__COMPLEMENTARITY": "0.17",
                "SHOPPING_DECISION__CANDIDATE_QUESTION_VALUE__WEIGHTS__ANSWER_PROBABILITY": "0.18",
                "SHOPPING_DECISION__CANDIDATE_QUESTION_VALUE__WEIGHTS__MISSING_PENALTY": "0.19",
                "SHOPPING_DECISION__CANDIDATE_QUESTION_VALUE__WEIGHTS__REDUNDANCY_PENALTY": "0.21",
                "SHOPPING_DECISION__CANDIDATE_QUESTION_VALUE__WEIGHTS__REPEAT_PENALTY": "0.41",
                "SHOPPING_DECISION__CANDIDATE_QUESTION_VALUE__WEIGHTS__NO_PREFERENCE_PENALTY": (
                    "0.61"
                ),
                "SHOPPING_DECISION__CANDIDATE_QUESTION_VALUE__WEIGHTS__TURN_COST": "0.22",
                "SHOPPING_DECISION__FINISH_STRATEGY__ENABLED": "true",
                "SHOPPING_DECISION__FINISH_STRATEGY__CANDIDATE_THRESHOLD": "80",
                "SHOPPING_DECISION__FINISH_STRATEGY__REMAINING_QUESTION_THRESHOLD": "3",
                "SHOPPING_DECISION__FINISH_STRATEGY__LOOKAHEAD_DEPTH": "2",
                "SHOPPING_DECISION__FINISH_STRATEGY__MINIMUM_FINISH_GAIN": "0.1",
                "SHOPPING_DECISION__FINISH_STRATEGY__WEIGHTS__RESOLVE_AT_10": "0.7",
                "SHOPPING_DECISION__FINISH_STRATEGY__WEIGHTS__RESOLVE_AT_3": "0.4",
                "SHOPPING_DECISION__FINISH_STRATEGY__WEIGHTS__RESOLVE_AT_1": "0.2",
                "SHOPPING_DECISION__FINISH_STRATEGY__WEIGHTS__TERMINAL_PROGRESS": "0.35",
                "SHOPPING_DECISION__FINISH_STRATEGY__WEIGHTS__P90_REMAINING_PENALTY": "0.25",
            }
        ).decision

        self.assertEqual(decision.question_termination_mode, "explicit_only")
        self.assertTrue(decision.candidate_question_value.enabled)
        self.assertEqual(decision.candidate_question_value.pool_size, 500)
        self.assertEqual(decision.candidate_question_value.prior_alpha, 0.5)
        self.assertEqual(decision.candidate_question_value.prior_temperature, 2.0)
        self.assertEqual(decision.candidate_question_value.other_answer_probability, 0.6)
        self.assertEqual(decision.candidate_question_value.other_vagueness_penalty, 0.2)
        self.assertEqual(decision.candidate_question_value.weights.expected_shrink, 0.4)
        self.assertEqual(decision.candidate_question_value.weights.coverage, 0.16)
        self.assertEqual(decision.candidate_question_value.weights.complementarity, 0.17)
        self.assertEqual(decision.candidate_question_value.weights.answer_probability, 0.18)
        self.assertEqual(decision.candidate_question_value.weights.missing_penalty, 0.19)
        self.assertEqual(decision.candidate_question_value.weights.redundancy_penalty, 0.21)
        self.assertEqual(decision.candidate_question_value.weights.repeat_penalty, 0.41)
        self.assertEqual(decision.candidate_question_value.weights.no_preference_penalty, 0.61)
        self.assertEqual(decision.candidate_question_value.weights.turn_cost, 0.22)
        self.assertTrue(decision.finish_strategy.enabled)
        self.assertEqual(decision.finish_strategy.candidate_threshold, 80)
        self.assertEqual(decision.finish_strategy.remaining_question_threshold, 3)
        self.assertEqual(decision.finish_strategy.lookahead_depth, 2)
        self.assertEqual(decision.finish_strategy.minimum_finish_gain, 0.1)
        self.assertEqual(decision.finish_strategy.weights.resolve_at_10, 0.7)
        self.assertEqual(decision.finish_strategy.weights.resolve_at_3, 0.4)
        self.assertEqual(decision.finish_strategy.weights.resolve_at_1, 0.2)
        self.assertEqual(decision.finish_strategy.weights.terminal_progress, 0.35)
        self.assertEqual(decision.finish_strategy.weights.p90_remaining_penalty, 0.25)

    def test_candidate_question_signal_mappings_are_immutable(self) -> None:
        color = CandidateAttributeSignal(
            attribute="color",
            coverage=0.8,
            expected_remaining=5.0,
            expected_shrink=0.5,
            resolve_at_10=0.7,
            resolve_at_3=0.3,
            resolve_at_1=0.1,
            p90_remaining=8.0,
            worst_case_remaining=10,
            missing_rate=0.2,
            extraction_confidence=0.9,
        )
        attributes = {"color": color}
        probabilities = {"asin-1": 0.8}
        signals = CandidateQuestionSignals(
            candidate_count=10,
            by_attribute=attributes,
            target_probabilities=probabilities,
        )

        attributes["size"] = color
        probabilities["asin-2"] = 0.2

        self.assertEqual(tuple(signals.by_attribute), ("color",))
        self.assertEqual(tuple(signals.target_probabilities), ("asin-1",))
        with self.assertRaises(FrozenInstanceError):
            color.coverage = 0.1
        with self.assertRaises(TypeError):
            signals.by_attribute["size"] = color
        with self.assertRaises(TypeError):
            signals.target_probabilities["asin-2"] = 0.2

    def test_defaults_enable_cascaded_recognition_with_configurable_utility(self) -> None:
        config = load_config(environ={})

        self.assertEqual(config.dialogue_understanding.mode, "cascaded")
        self.assertEqual(config.dialogue_understanding.rule_confidence_threshold, 0.75)
        self.assertEqual(config.decision.ask_utility.weights.information_gain, 0.30)
        self.assertEqual(config.decision.ask_utility.minimum_ask_utility, 0.20)
        self.assertEqual(config.decision.stop_utility.minimum_stop_utility, 0.65)

    def test_environment_overrides_nested_dialogue_and_decision_values(self) -> None:
        config = load_config(
            environ={
                "SHOPPING_DIALOGUE__MODE": "rule_only",
                "SHOPPING_DIALOGUE__RULE_CONFIDENCE_THRESHOLD": "0.65",
                "SHOPPING_DECISION__MAX_QUESTIONS": "4",
                "SHOPPING_DECISION__ASK_UTILITY__MINIMUM": "0.35",
                "SHOPPING_DECISION__ASK_UTILITY__WEIGHTS__INFORMATION_GAIN": "0.9",
                "SHOPPING_DECISION__STOP_UTILITY__MINIMUM": "0.7",
            }
        )

        self.assertEqual(config.dialogue_understanding.mode, "rule_only")
        self.assertEqual(config.dialogue_understanding.rule_confidence_threshold, 0.65)
        self.assertEqual(config.decision.max_questions, 4)
        self.assertEqual(config.decision.ask_utility.minimum_ask_utility, 0.35)
        self.assertEqual(config.decision.ask_utility.weights.information_gain, 0.9)
        self.assertEqual(config.decision.stop_utility.minimum_stop_utility, 0.7)

    def test_transition_guard_defaults_to_disabled(self) -> None:
        config = EnvConfig.from_env(environ={})
        guard = config.dialogue_understanding.transition_guard

        self.assertFalse(guard.enabled)
        self.assertEqual(guard.low_confidence_add_action, "soften")
        self.assertEqual(guard.destructive_failure_action, "clarify")

    def test_transition_guard_environment_switch(self) -> None:
        config = EnvConfig.from_env(
            environ={"SHOPPING_DIALOGUE__TRANSITION_GUARD__ENABLED": "1"}
        )

        self.assertTrue(config.dialogue_understanding.transition_guard.enabled)

    def test_transition_guard_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(ConfigError, "replace_min_confidence"):
            load_config(
                overrides={
                    "dialogue_understanding": {
                        "transition_guard": {"replace_min_confidence": 1.1}
                    }
                },
                environ={},
            )
        with self.assertRaisesRegex(ConfigError, "low_confidence_add_action"):
            load_config(
                overrides={
                    "dialogue_understanding": {
                        "transition_guard": {"low_confidence_add_action": "reject"}
                    }
                },
                environ={},
            )
        with self.assertRaisesRegex(ConfigError, "destructive_failure_action"):
            load_config(
                overrides={
                    "dialogue_understanding": {
                        "transition_guard": {"destructive_failure_action": "soften"}
                    }
                },
                environ={},
            )

    def test_transition_guard_actions_have_stable_values(self) -> None:
        self.assertEqual(GuardAction.APPLY.value, "apply")
        self.assertEqual(GuardAction.SOFTEN.value, "soften")
        self.assertEqual(GuardAction.CLARIFY.value, "clarify")
        self.assertEqual(GuardAction.REJECT.value, "reject")

    def test_invalid_mode_and_weight_are_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigError, "dialogue_understanding.mode"):
            load_config(environ={"SHOPPING_DIALOGUE__MODE": "legacy"})
        with self.assertRaisesRegex(ConfigError, "non-negative"):
            load_config(
                overrides={"decision": {"ask_utility": {"weights": {"turn_cost": -0.1}}}},
                environ={},
            )

    def test_dialogue_and_decision_models_are_immutable(self) -> None:
        config = load_config(environ={})

        with self.assertRaises(FrozenInstanceError):
            config.decision.max_questions = 9


if __name__ == "__main__":
    unittest.main()
