from __future__ import annotations

import unittest

from agent.dialogue.catalog_signals import CatalogQuestionSignals
from agent.dialogue.models import DialogueState
from agent.dialogue.question_policy import QuestionPolicy
from config.models import FinishWeights
from tests.test_question_policy import (
    candidate_signal,
    dynamic_config,
    dynamic_signals,
    parsed,
)


class FinishStrategyTest(unittest.TestCase):
    def test_exploration_prefers_candidate_shrink_before_finish_gate(self) -> None:
        # Using two-step gain before the finish gate would select color rather than material.
        policy = QuestionPolicy(dynamic_config(finish_enabled=False))
        decision = policy.decide(
            DialogueState(session_id="s", user_profile={}, category="shoes", turn=2),
            parsed(),
            CatalogQuestionSignals.empty(),
            dynamic_signals(
                {
                    "material": candidate_signal("material", shrink=0.8, resolve10=0.1),
                    "color": candidate_signal("color", shrink=0.1, resolve10=1.0, two_step=9.0),
                }
            ),
        )

        self.assertEqual(decision.ask_attribute, "material")
        self.assertEqual(policy.last_components["color"]["two_step_finish_gain"], 0.0)

    def test_depth_one_ignores_two_step_gain_when_finish_active(self) -> None:
        # Including depth-two value at depth one would select color instead of material.
        policy = QuestionPolicy(
            dynamic_config(
                finish_enabled=True,
                lookahead_depth=1,
                candidate_threshold=30,
                finish_weights=FinishWeights(
                    resolve_at_10=0.0,
                    resolve_at_3=0.0,
                    resolve_at_1=0.0,
                    terminal_progress=0.0,
                    p90_remaining_penalty=0.0,
                ),
            )
        )
        decision = policy.decide(
            DialogueState(
                session_id="s",
                user_profile={},
                category="shoes",
                turn=8,
                asked_attributes=("feature", "size"),
            ),
            parsed(),
            CatalogQuestionSignals.empty(),
            dynamic_signals(
                {
                    "material": candidate_signal("material", shrink=0.8),
                    "color": candidate_signal("color", shrink=0.1, two_step=9.0),
                },
                count=20,
                previous_count=80,
            ),
        )

        self.assertEqual(decision.ask_attribute, "material")
        self.assertEqual(policy.last_components["color"]["two_step_finish_gain"], 0.0)

    def test_finish_gate_prefers_resolve_at_ten_and_includes_two_step_gain(self) -> None:
        # Omitting the finish-gated two-step term would select material instead of color.
        policy = QuestionPolicy(
            dynamic_config(
                finish_enabled=True,
                lookahead_depth=2,
                candidate_threshold=30,
                remaining_question_threshold=2,
            )
        )
        decision = policy.decide(
            DialogueState(
                session_id="s",
                user_profile={},
                category="shoes",
                turn=8,
                asked_attributes=("feature", "size"),
            ),
            parsed(),
            CatalogQuestionSignals.empty(),
            dynamic_signals(
                {
                    "material": candidate_signal(
                        "material", shrink=0.8, resolve10=0.1, two_step=0.0
                    ),
                    "color": candidate_signal(
                        "color", shrink=0.1, resolve10=1.0, two_step=1.0
                    ),
                },
                count=20,
                previous_count=80,
            ),
        )

        self.assertEqual(decision.ask_attribute, "color")
        self.assertGreater(policy.last_components["color"]["two_step_finish_gain"], 0.0)

    def test_vague_other_loses_to_concrete_attribute_late(self) -> None:
        # Ignoring the configured other vagueness term would choose other here.
        policy = QuestionPolicy(
            dynamic_config(
                finish_enabled=True,
                candidate_threshold=30,
                remaining_question_threshold=2,
            )
        )
        decision = policy.decide(
            DialogueState(
                session_id="s",
                user_profile={},
                category="shoes",
                turn=8,
                asked_attributes=("feature", "size"),
            ),
            parsed(),
            CatalogQuestionSignals.empty(),
            dynamic_signals(
                {"material": candidate_signal("material", shrink=0.4, resolve10=1.0)},
                count=20,
                other=candidate_signal("other", shrink=0.9, resolve10=0.0),
            ),
        )

        self.assertEqual(decision.ask_attribute, "material")


if __name__ == "__main__":
    unittest.main()
