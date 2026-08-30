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
    finish_enabled: bool = False,
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
            enabled=finish_enabled,
            lookahead_depth=lookahead_depth,
            weights=finish_weights or FinishWeights(),
        ),
    )


class CandidateSignalCalculatorTest(unittest.TestCase):
    def test_concrete_only_signals_skip_other_and_keep_depth_one(self) -> None:
        # Hybrid only ranks concrete replacement attributes, so it must not spend
        # work deriving the composite ``other`` signal or two-step branches.
        cache = cache_for(
            A={"material": {"cotton"}, "color": {"black"}},
            B={"material": {"leather"}, "color": {"red"}},
            C={"material": {"cotton"}, "color": {"red"}},
        )
        candidates = [{"parent_asin": asin} for asin in "ABC"]
        signal_calculator = calculator(cache, lookahead_depth=2, finish_enabled=True)

        expected = calculator(cache).calculate(candidates)
        concrete_only = signal_calculator.calculate(candidates, include_other=False)

        self.assertIsNone(concrete_only.best_other_pair)
        self.assertIsNone(concrete_only.other_signal)
        self.assertEqual(concrete_only.lookahead_depth_used, 1)
        self.assertEqual(concrete_only.by_attribute, expected.by_attribute)

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
        # An unweighted nearest-rank P90 is two here (nine two-item branches),
        # while probability mass makes the high-RRF ten-item branch P90.
        cache = cache_for(
            A={"material": {"cotton"}},
            B={"material": {"leather"}},
            C={"material": {"wool"}},
            D={"material": {"silk"}},
            E={"material": {"linen"}},
            F={"material": {"denim"}},
            G={"material": {"suede"}},
            H={"material": {"rayon"}},
            I={"material": {"nylon"}},
            J={},
        )

        material = calculator(cache, alpha=1.0).calculate(
            [
                {"parent_asin": "A", "rrf": 0.0},
                {"parent_asin": "B", "rrf": 0.0},
                {"parent_asin": "C", "rrf": 0.0},
                {"parent_asin": "D", "rrf": 0.0},
                {"parent_asin": "E", "rrf": 0.0},
                {"parent_asin": "F", "rrf": 0.0},
                {"parent_asin": "G", "rrf": 0.0},
                {"parent_asin": "H", "rrf": 0.0},
                {"parent_asin": "I", "rrf": 0.0},
                {"parent_asin": "J", "rrf": 2.0},
            ]
        ).by_attribute["material"]

        self.assertNotEqual(material.p90_remaining, 1.0)
        self.assertEqual(material.p90_remaining, 10.0)
        self.assertEqual(material.worst_case_remaining, 10)

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
        # Color and size carry the same partition, so (material, color) and
        # (material, size) tie at one remaining candidate. (Color, size) leaves
        # groups of two, so canonical order must retain this pair.
        cache = cache_for(
            A={"material": {"cotton"}, "color": {"black"}, "size": {"small"}},
            B={"material": {"cotton"}, "color": {"red"}, "size": {"large"}},
            C={"material": {"leather"}, "color": {"black"}, "size": {"small"}},
            D={"material": {"leather"}, "color": {"red"}, "size": {"large"}},
        )

        signals = calculator(cache).calculate([{"parent_asin": asin} for asin in "ABCD"])

        self.assertEqual(signals.best_other_pair, ("material", "color"))
        self.assertIsNotNone(signals.other_signal)
        self.assertEqual(signals.other_signal.expected_remaining, 1.0)

    def test_joint_signal_intersects_known_answers_when_one_attribute_is_missing(self) -> None:
        # Returning the full pool as soon as any target value is missing would
        # retain B for target A despite B's conflicting known color.
        cache = cache_for(
            A={"material": {"cotton"}, "color": {"black"}},
            B={"color": {"red"}},
        )

        signals = calculator(cache).calculate(
            [{"parent_asin": "A"}, {"parent_asin": "B"}],
            eligible_attributes=("material", "color"),
        )

        self.assertEqual(signals.best_other_pair, ("material", "color"))
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

        self.assertEqual(tuple(signals.by_attribute), ("material", "category"))
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
            finish_enabled=True,
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

    def test_depth_two_is_skipped_until_finish_strategy_and_phase_are_eligible(self) -> None:
        # Eager lookahead would call the branch scorer despite disabled or inactive finish.
        cache = cache_for(
            A={"material": {"cotton"}, "color": {"black"}},
            B={"material": {"leather"}, "color": {"red"}},
        )
        candidates = [{"parent_asin": "A"}, {"parent_asin": "B"}]
        disabled = calculator(cache, lookahead_depth=2, finish_enabled=False)
        inactive = calculator(cache, lookahead_depth=2, finish_enabled=True)

        disabled_calls = 0
        inactive_calls = 0
        disabled_original = disabled._two_step_finish_gain
        inactive_original = inactive._two_step_finish_gain

        def count_disabled(*args):
            nonlocal disabled_calls
            disabled_calls += 1
            return disabled_original(*args)

        def count_inactive(*args):
            nonlocal inactive_calls
            inactive_calls += 1
            return inactive_original(*args)

        disabled._two_step_finish_gain = count_disabled
        inactive._two_step_finish_gain = count_inactive
        disabled.calculate(candidates)
        inactive.calculate(candidates, terminal_eligible=False)

        self.assertEqual(disabled_calls, 0)
        self.assertEqual(inactive_calls, 0)

    def test_depth_two_memoizes_equivalent_all_missing_branches(self) -> None:
        # Recomputing every target/second-attribute branch would make this call count
        # grow with the candidate count instead of the one shared all-missing branch.
        cache = cache_for(**{str(index): {} for index in range(8)})
        signal_calculator = calculator(cache, lookahead_depth=2, finish_enabled=True)
        original = signal_calculator._signal_for_attributes
        calls = 0

        def count(*args):
            nonlocal calls
            calls += 1
            return original(*args)

        signal_calculator._signal_for_attributes = count
        signals = signal_calculator.calculate(
            [{"parent_asin": str(index)} for index in range(8)],
            eligible_attributes=("material", "color", "size"),
        )

        self.assertIn("material", signals.by_attribute)
        self.assertLessEqual(calls, 12)

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
