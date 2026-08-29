from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


def _samples() -> list[dict]:
    rows: list[dict] = []
    scenarios = ("buying", "browsing", "intent_override", "boundary")
    for target_index in range(10):
        for duplicate in range(2):
            rows.append(
                {
                    "sample_id": f"sample-{target_index}-{duplicate}",
                    "scenario_type": scenarios[target_index % len(scenarios)],
                    "coarse_category": f"category-{target_index % 3}",
                    "initial_candidate_bin": "large" if target_index % 2 else "small",
                    "ground_truth": {"parent_asin": f"target-{target_index}"},
                }
            )
    return rows


class DecisionCrossValidationTest(unittest.TestCase):
    def test_grouped_folds_are_deterministic_and_do_not_leak_targets(self) -> None:
        # Assigning one target to two folds would leak its outcome into model selection.
        from experiments.decision_cross_validation import grouped_stratified_folds

        first = grouped_stratified_folds(_samples(), fold_count=3, seed=20260829)
        target_to_fold: dict[str, int] = {}
        for fold_index, fold in enumerate(first):
            for sample in fold:
                target = sample["ground_truth"]["parent_asin"]
                assigned = target_to_fold.setdefault(target, fold_index)
                self.assertEqual(assigned, fold_index)
        self.assertEqual(first, grouped_stratified_folds(_samples(), fold_count=3, seed=20260829))
        self.assertLessEqual(max(map(len, first)) - min(map(len, first)), 4)

    def test_manifest_is_bounded_and_excludes_depth_two_before_evaluation(self) -> None:
        # Evaluating depth-two configs would let the known unsafe gate enter the sweep.
        from experiments.decision_cross_validation import build_search_manifest

        search_space = {
            "transition_guard_profile": [{"enabled": False}],
            "candidate_weight_profile": [{"expected_shrink": 0.3}],
            "finish_weight_profile": [{"resolve_at_10": 0.5}],
            "pool_size": [300, 500, 1000],
            "prior_alpha": [0.25],
            "prior_temperature": [1.0],
            "other_answer_probability": [0.75],
            "other_vagueness_penalty": [0.1],
            "finish_candidate_threshold": [100],
            "remaining_question_threshold": [2],
            "lookahead_depth": [1, 2],
            "termination_mode": ["explicit_only"],
        }
        manifest = build_search_manifest(search_space, seed=11, max_depth_one=2)
        self.assertLessEqual(len(manifest["evaluated"]), 3)  # two plus legacy
        self.assertTrue(
            all(item["kind"] == "legacy" or item["depth"] == 1 for item in manifest["evaluated"])
        )
        self.assertTrue(
            all(
                item["exclusion_reason"] == "known_depth_two_gate_mismatch"
                for item in manifest["excluded"]
            )
        )

    def test_paired_bootstrap_is_deterministic(self) -> None:
        # Changing bootstrap sampling would make uncertainty reports irreproducible.
        from experiments.decision_cross_validation import paired_bootstrap_interval

        candidate = [0.8, 0.4, 0.9, 0.6]
        baseline = [0.7, 0.5, 0.5, 0.6]
        self.assertEqual(
            paired_bootstrap_interval(candidate, baseline, iterations=400, seed=7),
            paired_bootstrap_interval(candidate, baseline, iterations=400, seed=7),
        )

    def test_catalog_latency_and_fold_gates_reject_unstable_configuration(self) -> None:
        # A config exceeding production latency or regressing two of three folds must not promote.
        from experiments.decision_cross_validation import eligibility_reasons

        reasons = eligibility_reasons(
            candidate={
                "outer_fold_hr10_deltas": [-0.01, -0.02, 0.01],
                "scenario_hr10_deltas": {"buying": -0.01},
                "catalog_stability_delta": -0.01,
                "latency_p95_ms": 120.0,
            },
            baseline={"latency_p95_ms": 50.0},
            hr10_tolerance=0.005,
            scenario_tolerance=0.02,
            catalog_tolerance=0.02,
            latency_budget_ms=100.0,
        )
        self.assertIn("outer_fold_hr10_regression", reasons)
        self.assertIn("latency_budget_exceeded", reasons)

    def test_two_of_three_nonregressing_folds_are_eligible(self) -> None:
        # Requiring the old four-of-five rule would reject a valid moderate three-fold run.
        from experiments.decision_cross_validation import eligibility_reasons

        reasons = eligibility_reasons(
            candidate={
                "outer_fold_hr10_deltas": [0.0, -0.001, 0.002],
                "scenario_hr10_deltas": {"buying": 0.0},
                "catalog_stability_delta": 0.0,
                "latency_p95_ms": 50.0,
            },
            baseline={"latency_p95_ms": 50.0},
            hr10_tolerance=0.005,
            scenario_tolerance=0.02,
            catalog_tolerance=0.02,
            latency_budget_ms=100.0,
        )
        self.assertNotIn("outer_fold_hr10_regression", reasons)

    def test_one_standard_error_prefers_simpler_configuration(self) -> None:
        # Selecting the highest point estimate inside the SE band would overfit the public sessions.
        from experiments.decision_cross_validation import select_one_standard_error

        selected = select_one_standard_error(
            [
                {
                    "id": "complex",
                    "mean": 0.72,
                    "standard_error": 0.02,
                    "complexity": 5,
                    "latency_p95_ms": 30.0,
                    "canonical_json": "z",
                },
                {
                    "id": "simple",
                    "mean": 0.71,
                    "standard_error": 0.01,
                    "complexity": 2,
                    "latency_p95_ms": 40.0,
                    "canonical_json": "a",
                },
            ]
        )
        self.assertEqual(selected["id"], "simple")

    def test_complete_recommendation_loads_and_output_collision_is_rejected(self) -> None:
        # A partial overlay or an output alias could fail deployment or replace the public dataset.
        from config.loader import load_config
        from experiments.decision_cross_validation import (
            complete_config_document,
            write_json_atomic,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base.json"
            base.write_text(
                Path("config/default.json").read_text(encoding="utf-8"), encoding="utf-8"
            )
            complete = complete_config_document(
                base, {"decision": {"question_termination_mode": "explicit_only"}}
            )
            output = root / "recommended.json"
            write_json_atomic(complete, output, protected_inputs=(base,))
            self.assertEqual(
                load_config(path=output, environ={}).decision.question_termination_mode,
                "explicit_only",
            )
            with self.assertRaisesRegex(ValueError, "protected"):
                write_json_atomic({"bad": True}, base, protected_inputs=(base,))
