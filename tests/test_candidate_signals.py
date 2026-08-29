from __future__ import annotations

import math
import unittest

from agent.dialogue.candidate_signals import CandidateSignalCalculator
from agent.dialogue.catalog_attributes import AttributeProfile, CatalogAttributeCache
from config.models import (
    CandidateQuestionValueConfig,
    CandidateQuestionWeights,
    FinishStrategyConfig,
    FinishWeights,
)


def cache_for(**attributes: dict[str, set[str]]) -> CatalogAttributeCache:
    profiles = {
        asin: AttributeProfile(
            parent_asin=asin,
            values=values,
            confidence={attribute: 1.0 for attribute in values},
            sources={attribute: ("test",) for attribute in values},
        )
        for asin, values in attributes.items()
    }
    return CatalogAttributeCache(profiles, "test-v1", "test-fingerprint")


def calculator(
    cache: CatalogAttributeCache,
    *,
    alpha: float = 0.0,
    temperature: float = 1.0,
    lookahead_depth: int = 1,
    turn_cost: float = 0.15,
    finish_weights: FinishWeights | None = None,
) -> CandidateSignalCalculator:
    return CandidateSignalCalculator(
        cache,
        CandidateQuestionValueConfig(
            prior_alpha=alpha,
            prior_temperature=temperature,
            weights=CandidateQuestionWeights(turn_cost=turn_cost),
        ),
        FinishStrategyConfig(
            lookahead_depth=lookahead_depth,
            weights=finish_weights or FinishWeights(),
        ),
    )


class CandidateSignalCalculatorTest(unittest.TestCase):
    def test_uniform_prior_hand_calculates_split_and_constant_attributes(self) -> None:
        # Changing compatible-set construction to exclude an answer-matching product
        # would make the hand-derived material values below fail.
        cache = cache_for(
            A={"material": {"cotton"}, "color": {"black"}},
            B={"material": {"cotton"}, "color": {"black"}},
            C={"material": {"leather"}, "color": {"black"}},
            D={"material": {"leather"}, "color": {"black"}},
        )

        signals = calculator(cache).calculate(
            [{"parent_asin": asin, "rrf": 1.0} for asin in "ABCD"]
        )

        material = signals.by_attribute["material"]
        self.assertEqual(material.expected_remaining, 2.0)
        self.assertEqual(material.expected_shrink, 0.5)
        self.assertEqual(material.resolve_at_10, 1.0)
        self.assertEqual(signals.by_attribute["color"].expected_shrink, 0.0)

    def test_missing_values_remain_in_every_compatible_set(self) -> None:
        # Treating missing metadata as a negative value would incorrectly shrink
        # this pool from the literal expected size of three.
        cache = cache_for(
            A={"material": {"cotton"}},
            B={"material": {"cotton"}},
            C={"material": {"leather"}},
            D={},
        )

        material = calculator(cache).calculate(
            [{"parent_asin": asin} for asin in "ABCD"]
        ).by_attribute["material"]

        self.assertEqual(material.expected_remaining, 3.0)
        self.assertEqual(material.expected_shrink, 0.25)
        self.assertEqual(material.coverage, 0.75)
        self.assertEqual(material.missing_rate, 0.25)
        self.assertEqual(material.worst_case_remaining, 4)

    def test_rrf_prior_mixes_softmax_with_uniform_prior(self) -> None:
        # Replacing the mixture with raw RRF normalization would fail this exact
        # hand-calculated probability.
        cache = cache_for(A={"material": {"cotton"}}, B={"material": {"leather"}})

        probabilities = calculator(cache, alpha=0.5).calculate(
            [{"parent_asin": "A", "rrf": 0.0}, {"parent_asin": "B", "rrf": 1.0}]
        ).target_probabilities

        self.assertAlmostEqual(probabilities["A"], 0.25 + 0.5 / (1.0 + math.e))
        self.assertAlmostEqual(probabilities["B"], 0.25 + 0.5 * math.e / (1.0 + math.e))

    def test_all_equal_or_nonfinite_rrf_falls_back_to_uniform_prior(self) -> None:
        cache = cache_for(A={"material": {"cotton"}}, B={"material": {"leather"}})

        equal = calculator(cache, alpha=1.0).calculate(
            [{"parent_asin": "A", "rrf": 4.0}, {"parent_asin": "B", "rrf": 4.0}]
        )
        nonfinite = calculator(cache, alpha=1.0).calculate(
            [{"parent_asin": "A", "rrf": float("nan")}, {"parent_asin": "B", "rrf": 2.0}]
        )

        self.assertEqual(dict(equal.target_probabilities), {"A": 0.5, "B": 0.5})
        self.assertEqual(dict(nonfinite.target_probabilities), {"A": 0.5, "B": 0.5})

    def test_rejects_nonpositive_prior_temperature(self) -> None:
        cache = cache_for(A={"material": {"cotton"}})

        with self.assertRaisesRegex(ValueError, "prior_temperature"):
            calculator(cache, temperature=0.0)

    def test_weighted_p90_uses_target_probability_mass(self) -> None:
        # A rank-based percentile would pick one; P90 must include the highly
        # likely two-item branch and therefore be two.
        cache = cache_for(
            A={"material": {"cotton"}},
            B={"material": {"leather"}},
            C={"material": {"wool"}},
            D={"material": {"wool"}},
        )

        material = calculator(cache, alpha=1.0).calculate(
            [
                {"parent_asin": "A", "rrf": 0.0},
                {"parent_asin": "B", "rrf": 0.0},
                {"parent_asin": "C", "rrf": 0.0},
                {"parent_asin": "D", "rrf": 2.0},
            ]
        ).by_attribute["material"]

        self.assertEqual(material.p90_remaining, 2.0)
        self.assertEqual(material.worst_case_remaining, 2)

    def test_multivalue_overlap_keeps_any_shared_value(self) -> None:
        # Requiring exact set equality rather than overlap would make the expected
        # compatible-set size smaller than the literal value below.
        cache = cache_for(
            A={"material": {"cotton", "wool"}},
            B={"material": {"wool"}},
            C={"material": {"leather"}},
            D={},
        )

        material = calculator(cache).calculate(
            [{"parent_asin": asin} for asin in "ABCD"]
        ).by_attribute["material"]

        self.assertEqual(material.expected_remaining, 3.0)

    def test_joint_other_uses_best_concrete_pair_with_stable_tie_order(self) -> None:
        cache = cache_for(
            A={"material": {"cotton"}, "color": {"black"}},
            B={"material": {"cotton"}, "color": {"red"}},
            C={"material": {"leather"}, "color": {"black"}},
            D={"material": {"leather"}, "color": {"red"}},
        )

        signals = calculator(cache).calculate([{"parent_asin": asin} for asin in "ABCD"])

        self.assertEqual(signals.best_other_pair, ("material", "color"))
        self.assertIsNotNone(signals.other_signal)
        self.assertEqual(signals.other_signal.expected_remaining, 1.0)

    def test_eligible_attributes_limit_attributes_and_joint_pairs(self) -> None:
        cache = cache_for(
            A={"material": {"cotton"}, "color": {"black"}, "category": {"shirts"}},
            B={"material": {"leather"}, "color": {"red"}, "category": {"shirts"}},
        )

        signals = calculator(cache).calculate(
            [{"parent_asin": "A"}, {"parent_asin": "B"}],
            eligible_attributes=("category", "other", "material"),
        )

        self.assertEqual(tuple(signals.by_attribute), ("material",))
        self.assertIsNone(signals.best_other_pair)
        self.assertIsNone(signals.other_signal)

    def test_depth_one_has_no_two_step_finish_gain(self) -> None:
        cache = cache_for(
            A={"material": {"cotton"}, "color": {"black"}},
            B={"material": {"cotton"}, "color": {"red"}},
            C={"material": {"leather"}, "color": {"black"}},
            D={"material": {"leather"}, "color": {"red"}},
        )

        signals = calculator(cache, lookahead_depth=1).calculate(
            [{"parent_asin": asin} for asin in "ABCD"]
        )

        self.assertEqual(signals.by_attribute["material"].two_step_finish_gain, 0.0)

    def test_depth_two_weights_best_second_branch_gain_and_subtracts_one_turn_cost(self) -> None:
        # Charging cost once per branch would produce 0.70 instead of 0.85.
        cache = cache_for(
            A={"material": {"cotton"}, "color": {"black"}},
            B={"material": {"cotton"}, "color": {"red"}},
            C={"material": {"leather"}, "color": {"black"}},
            D={"material": {"leather"}, "color": {"red"}},
        )
        signals = calculator(
            cache,
            lookahead_depth=2,
            turn_cost=0.15,
            finish_weights=FinishWeights(
                resolve_at_10=0.0,
                resolve_at_3=0.0,
                resolve_at_1=1.0,
                terminal_progress=0.0,
                p90_remaining_penalty=0.0,
            ),
        ).calculate([{"parent_asin": asin} for asin in "ABCD"])

        self.assertAlmostEqual(signals.by_attribute["material"].two_step_finish_gain, 0.85)

    def test_empty_unknown_and_duplicate_candidates_are_safe_and_deterministic(self) -> None:
        cache = cache_for(A={"material": {"cotton"}})
        candidates = [
            {"parent_asin": "A", "rrf": 1.0},
            {"parent_asin": "missing", "rrf": 9.0},
            {"parent_asin": "A", "rrf": 2.0},
        ]

        first = calculator(cache, alpha=1.0).calculate(candidates)
        second = calculator(cache, alpha=1.0).calculate(list(reversed(candidates)))
        empty = calculator(cache).calculate([])

        self.assertEqual(first.candidate_count, 2)
        self.assertEqual(dict(first.target_probabilities), dict(second.target_probabilities))
        self.assertEqual(first.by_attribute["material"].expected_shrink, 0.0)
        self.assertEqual(empty.candidate_count, 0)
        self.assertEqual(dict(empty.by_attribute), {})


if __name__ == "__main__":
    unittest.main()
