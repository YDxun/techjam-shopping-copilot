from __future__ import annotations

import json
import math
import unittest
from dataclasses import replace

from agent.dialogue.catalog_signals import AttributeSignal, CatalogQuestionSignals
from agent.dialogue.hybrid_question_policy import HybridQuestionPolicy
from agent.dialogue.models import (
    CandidateAttributeSignal,
    CandidateQuestionSignals,
    Constraint,
    ConstraintStrength,
    DialogueState,
    Polarity,
    QuestionDecision,
)
from config.models import HybridQuestionPolicyConfig


def candidate(
    attribute: str,
    *,
    shrink: float = 0.60,
    resolve10: float = 0.40,
    coverage: float = 0.90,
    missing: float = 0.10,
    confidence: float = 0.80,
) -> CandidateAttributeSignal:
    return CandidateAttributeSignal(
        attribute=attribute,
        coverage=coverage,
        expected_remaining=8.0,
        expected_shrink=shrink,
        resolve_at_10=resolve10,
        resolve_at_3=0.20,
        resolve_at_1=0.10,
        p90_remaining=12.0,
        worst_case_remaining=15,
        missing_rate=missing,
        extraction_confidence=confidence,
    )


def candidate_signals(
    by_attribute: dict[str, CandidateAttributeSignal],
) -> CandidateQuestionSignals:
    return CandidateQuestionSignals(
        candidate_count=20,
        by_attribute=by_attribute,
        target_probabilities={},
    )


def static_signals() -> CatalogQuestionSignals:
    return CatalogQuestionSignals(
        by_category={
            "shoes": {
                "material": AttributeSignal(0.8, 0.7, 0.90),
                "color": AttributeSignal(0.8, 0.7, 0.90),
                "category": AttributeSignal(0.8, 0.7, 0.90),
            }
        }
    )


class HybridQuestionPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = HybridQuestionPolicy(HybridQuestionPolicyConfig(enabled=True))
        self.state = DialogueState(
            session_id="private-session", user_profile={}, category="shoes", turn=2
        )
        self.legacy_other = QuestionDecision(True, "other", "ask_other_first", 1.0, {})

    def test_preserves_legacy_stops_and_concrete_questions(self) -> None:
        # Replacing any decision other than repeated other would let Hybrid own Legacy control flow.
        signals = candidate_signals({"material": candidate("material")})
        stopped = QuestionDecision(False, None, "maximum_questions_reached", 0.0, {})
        concrete = QuestionDecision(True, "color", "highest_ask_utility", 0.7, {})

        self.assertEqual(
            self.policy.consider(self.state, stopped, static_signals(), signals), stopped
        )
        self.assertEqual(
            self.policy.consider(self.state, concrete, static_signals(), signals), concrete
        )

    def test_preserves_first_other_then_replaces_one_repeated_other(self) -> None:
        # Replacing first other would erase the validated Legacy opening question.
        signals = candidate_signals({"material": candidate("material")})

        first = self.policy.consider(
            replace(self.state, asked_attributes=()), self.legacy_other, static_signals(), signals
        )
        replacement = self.policy.consider(
            replace(self.state, asked_attributes=("other",)),
            self.legacy_other,
            static_signals(),
            signals,
        )

        self.assertEqual(first.ask_attribute, "other")
        self.assertEqual(first.reason_code, "hybrid_first_other_preserved")
        self.assertEqual(replacement.ask_attribute, "material")
        self.assertEqual(replacement.reason_code, "hybrid_specific_replacement")
        self.assertAlmostEqual(replacement.utility_score, 0.6088888888888889)

    def test_filters_known_or_already_resolved_attributes(self) -> None:
        # A resolved or already-asked field must not be requested again.
        constrained = Constraint(
            attribute="material",
            value="cotton",
            polarity=Polarity.INCLUDE,
            strength=ConstraintStrength.HARD,
            evidence="cotton",
            source_turn=1,
            tokens=("cotton",),
        )
        signals = candidate_signals(
            {
                "material": candidate("material", shrink=1.0),
                "color": candidate("color", shrink=0.9),
                "category": candidate("category", shrink=0.8),
            }
        )
        state = replace(
            self.state,
            asked_attributes=("other", "color"),
            active_constraints=(constrained,),
        )

        decision = self.policy.consider(state, self.legacy_other, static_signals(), signals)

        self.assertEqual(decision.reason_code, "hybrid_no_eligible_attribute")
        self.assertEqual(decision.ask_attribute, "other")

    def test_filters_no_preference_and_non_finite_rows(self) -> None:
        # Coercing NaN into a score could make malformed candidate work influence a question.
        signals = candidate_signals(
            {
                "material": candidate("material", shrink=math.nan),
                "color": candidate("color", shrink=0.9),
            }
        )
        state = replace(
            self.state,
            asked_attributes=("other",),
            no_preference_attributes=frozenset({"color"}),
        )

        decision = self.policy.consider(state, self.legacy_other, static_signals(), signals)

        self.assertEqual(decision.reason_code, "hybrid_no_eligible_attribute")
        self.assertEqual(decision.ask_attribute, "other")

    def test_preserves_other_when_threshold_or_signals_are_unavailable(self) -> None:
        # A weak or absent candidate signal must not change the Legacy question.
        state = replace(self.state, asked_attributes=("other",))
        weak = candidate_signals({"material": candidate("material", coverage=0.50)})

        unavailable = self.policy.consider(state, self.legacy_other, static_signals(), None)
        threshold = self.policy.consider(state, self.legacy_other, static_signals(), weak)

        self.assertEqual(unavailable.reason_code, "hybrid_signals_unavailable")
        self.assertEqual(threshold.reason_code, "hybrid_threshold_not_met")
        self.assertEqual(unavailable.ask_attribute, "other")
        self.assertEqual(threshold.ask_attribute, "other")

    def test_preserves_other_when_replacement_budget_was_used(self) -> None:
        # Ignoring the persisted counter would permit multiple replacements in one session.
        state = replace(self.state, asked_attributes=("other",), hybrid_replacements_used=1)

        decision = self.policy.consider(
            state,
            self.legacy_other,
            static_signals(),
            candidate_signals({"material": candidate("material")}),
        )

        self.assertEqual(decision.reason_code, "hybrid_replacement_already_used")
        self.assertEqual(decision.ask_attribute, "other")

    def test_breaks_equal_scores_by_project_attribute_order(self) -> None:
        # Sorting by mapping insertion order would make equal-score decisions unstable.
        state = replace(self.state, asked_attributes=("other",))
        signals = candidate_signals(
            {"color": candidate("color"), "material": candidate("material")}
        )

        decision = self.policy.consider(state, self.legacy_other, static_signals(), signals)

        self.assertEqual(decision.ask_attribute, "material")

    def test_statistics_are_bounded_and_private_aggregates(self) -> None:
        # Retaining raw inputs would leak session or catalog data through diagnostics.
        signals = candidate_signals({"material": candidate("material")})
        self.policy.consider(
            replace(self.state, asked_attributes=()), self.legacy_other, static_signals(), signals
        )
        self.policy.consider(
            replace(self.state, asked_attributes=("other",)),
            self.legacy_other,
            static_signals(),
            signals,
        )

        statistics = self.policy.statistics()

        self.assertEqual(statistics["reason_counts"]["hybrid_first_other_preserved"], 1)
        self.assertEqual(statistics["reason_counts"]["hybrid_specific_replacement"], 1)
        self.assertEqual(statistics["selected_attribute_counts"], {"material": 1, "other": 1})
        self.assertEqual(statistics["replacement_count"], 1)
        self.assertTrue(math.isfinite(statistics["decision_latency_ms"]["p50"]))
        self.assertTrue(math.isfinite(statistics["decision_latency_ms"]["p95"]))
        self.assertGreaterEqual(statistics["decision_latency_ms"]["p50"], 0.0)
        self.assertGreaterEqual(statistics["decision_latency_ms"]["p95"], 0.0)
        encoded = json.dumps(statistics, sort_keys=True)
        self.assertNotIn("private-session", encoded)
        self.assertNotIn("cotton", encoded)


if __name__ == "__main__":
    unittest.main()
