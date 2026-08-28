from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from config.loader import ConfigError, load_config


class DialogueConfigTest(unittest.TestCase):
    def test_defaults_enable_cascaded_recognition_with_configurable_utility(self) -> None:
        config = load_config(environ={})

        self.assertEqual(config.dialogue_understanding.mode, "cascaded")
        self.assertEqual(config.dialogue_understanding.rule_confidence_threshold, 0.75)
        self.assertEqual(config.decision.ask_utility.weights.information_gain, 0.30)
        self.assertEqual(config.decision.ask_utility.minimum_ask_utility, 0.20)
        self.assertEqual(config.decision.stop_utility.minimum_stop_utility, 0.55)

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

    def test_invalid_mode_and_weight_are_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigError, "dialogue_understanding.mode"):
            load_config(environ={"SHOPPING_DIALOGUE__MODE": "legacy"})
        with self.assertRaisesRegex(ConfigError, "non-negative"):
            load_config(
                overrides={
                    "decision": {
                        "ask_utility": {"weights": {"turn_cost": -0.1}}
                    }
                },
                environ={},
            )

    def test_dialogue_and_decision_models_are_immutable(self) -> None:
        config = load_config(environ={})

        with self.assertRaises(FrozenInstanceError):
            config.decision.max_questions = 9


if __name__ == "__main__":
    unittest.main()
