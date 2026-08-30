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
    def test_global_fold_objective_balances_realistic_grouped_public_set(self) -> None:
        # Candidate-fold-only costs can strand a fold and omit common scenarios.
        from experiments.decision_cross_validation import grouped_stratified_folds

        scenarios = ["buying"] * 20 + ["browsing"] * 15 + ["intent_override"] * 9 + ["boundary"] * 6
        samples = [
            {
                "sample_id": f"public-{target}-{duplicate}",
                "scenario_type": scenario,
                "coarse_category": f"category-{target % 5}",
                "initial_candidate_bin": ("small", "medium", "large")[target % 3],
                "ground_truth": {"parent_asin": f"target-{target}"},
            }
            for target, scenario in enumerate(scenarios)
            for duplicate in range(4)
        ]

        folds = grouped_stratified_folds(samples, fold_count=3, seed=20260829)
        target_folds: dict[str, set[int]] = {}
        for index, fold in enumerate(folds):
            for row in fold:
                target_folds.setdefault(row["ground_truth"]["parent_asin"], set()).add(index)
        self.assertEqual(len(target_folds), 50)
        self.assertTrue(all(len(indices) == 1 for indices in target_folds.values()))
        self.assertLessEqual(max(map(len, folds)) - min(map(len, folds)), 4)
        for scenario in {row["scenario_type"] for row in samples}:
            self.assertTrue(
                all(any(row["scenario_type"] == scenario for row in fold) for fold in folds)
            )

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

    def test_catalog_population_proxy_and_marginal_folds_ignore_public_target_frequency(
        self,
    ) -> None:
        # Public-target counts would make a popular catalog category look rare.
        from experiments.decision_cross_validation import annotate_samples, grouped_stratified_folds

        samples = _samples()
        categories = {
            **{f"target-{index}": ["Clothing", "rare"] for index in range(10)},
            **{f"catalog-{index}": ["Clothing", "rare"] for index in range(20)},
            **{f"other-{index}": ["Clothing", "common"] for index in range(30)},
        }
        annotated = annotate_samples(samples, categories)
        self.assertTrue(all(row["initial_candidate_population"] == 30 for row in annotated))
        folds = grouped_stratified_folds(annotated, fold_count=3, seed=20260829)
        for field in ("scenario_type", "coarse_category", "initial_candidate_bin"):
            counts = [
                sum(row[field] == "buying" if field == "scenario_type" else 1 for row in fold)
                for fold in folds
            ]
            self.assertLessEqual(max(counts) - min(counts), 4)

    def test_manifest_is_bounded_and_excludes_depth_two_before_evaluation(self) -> None:
        # Evaluating depth-two configs would let the known unsafe gate enter the sweep.
        from experiments.decision_cross_validation import prepare_manifest_for_evaluation

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
        catalog = {
            "pool_sizes": {
                str(pool): {
                    "latency_ms": {"p95": 1.0},
                    "candidate_deletion_stability": {"choice_agreement": 1.0},
                }
                for pool in (300, 500, 1000)
            }
        }
        manifest = prepare_manifest_for_evaluation(search_space, catalog, seed=11, max_depth_one=2)
        evaluated = [item for item in manifest["configs"] if item["status"].startswith("evaluated")]
        self.assertLessEqual(len(evaluated), 2)
        self.assertTrue(all(item["kind"] == "legacy" or item["depth"] == 1 for item in evaluated))
        self.assertTrue(
            all(
                item["reason"] == "known_depth_two_gate_mismatch"
                for item in manifest["configs"]
                if item["status"] == "excluded_depth_two_known_residual"
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
        # An outer audit can flag regressions but cannot change the separately fit deployment ID.
        from experiments.decision_cross_validation import eligibility_reasons

        reasons = eligibility_reasons(
            candidate={
                "outer_fold_hr10_deltas": [-0.01, -0.02, 0.01],
                "scenario_hr10_deltas": {"buying": -0.01},
                "catalog_stability": 0.79,
                "latency_p95_ms": 120.0,
            },
            hr10_tolerance=0.005,
            scenario_tolerance=0.02,
            latency_budget_ms=100.0,
        )
        self.assertIn("outer_fold_hr10_regression", reasons)
        self.assertIn("latency_budget_exceeded", reasons)

    def test_final_fit_filters_predeclared_training_fold_regressions_before_one_se(self) -> None:
        # Selecting the higher technical score despite 3/3 HR regressions would leak an unsafe fit.
        from experiments.decision_cross_validation import _final_training_cv

        samples = [
            {
                "sample_id": f"selection-{target}-{duplicate}",
                "scenario_type": "buying",
                "coarse_category": "category-0",
                "initial_candidate_bin": "large",
                "ground_truth": {"parent_asin": f"selection-target-{target}"},
            }
            for target in range(9)
            for duplicate in range(4)
        ]
        outcomes = {"legacy": {"sessions": []}, "candidate": {"sessions": []}}
        for row in samples:
            sample_id = row["sample_id"]
            target_row = int(sample_id.rsplit("-", 1)[1]) == 0
            base = {
                "sample_id": sample_id,
                "scenario_type": row["scenario_type"],
                "hit": True,
                "reciprocal_rank": 0.1,
                "first_hit_turn": 10,
            }
            outcomes["legacy"]["sessions"].append(base)
            outcomes["candidate"]["sessions"].append(
                {
                    **base,
                    "hit": not target_row,
                    "reciprocal_rank": 0.0 if target_row else 1.0,
                    "first_hit_turn": None if target_row else 1,
                }
            )
        configs = [
            {
                "id": "legacy",
                "overlay": {},
                "complexity": 0,
                "latency_p95_ms": 0.0,
                "catalog_stability": 1.0,
            },
            {
                "id": "candidate",
                "overlay": {"decision": {"candidate_question_value": {"enabled": True}}},
                "complexity": 1,
                "latency_p95_ms": 10.0,
                "catalog_stability": 1.0,
            },
        ]

        selected, rows, _ = _final_training_cv(configs, outcomes, samples, seed=20260829)
        candidate = next(item for item in rows if item["id"] == "candidate")
        self.assertEqual(selected["id"], "legacy")
        self.assertIn("outer_fold_hr10_regression", candidate["selection_eligibility_reasons"])

    def test_two_of_three_nonregressing_folds_are_eligible(self) -> None:
        # Requiring the old four-of-five rule would reject a valid moderate three-fold run.
        from experiments.decision_cross_validation import eligibility_reasons

        reasons = eligibility_reasons(
            candidate={
                "outer_fold_hr10_deltas": [0.0, -0.001, 0.002],
                "scenario_hr10_deltas": {"buying": 0.0},
                "catalog_stability": 0.80,
                "latency_p95_ms": 50.0,
            },
            hr10_tolerance=0.005,
            scenario_tolerance=0.02,
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

    def test_one_standard_error_uses_canonical_equal_best_reference_independent_of_order(
        self,
    ) -> None:
        # First-arrival tie-breaking changes the one-SE band after a harmless reorder.
        from experiments.decision_cross_validation import select_one_standard_error

        configs = [
            {
                "id": "z-best",
                "mean": 0.72,
                "standard_error": 0.001,
                "complexity": 3,
                "latency_p95_ms": 1.0,
                "canonical_json": "z",
            },
            {
                "id": "a-best",
                "mean": 0.72,
                "standard_error": 0.03,
                "complexity": 3,
                "latency_p95_ms": 1.0,
                "canonical_json": "a",
            },
            {
                "id": "simple",
                "mean": 0.70,
                "standard_error": 0.01,
                "complexity": 1,
                "latency_p95_ms": 1.0,
                "canonical_json": "m",
            },
        ]
        self.assertEqual(select_one_standard_error(configs)["id"], "simple")
        self.assertEqual(select_one_standard_error(list(reversed(configs)))["id"], "simple")

    def test_predeclared_catalog_gates_use_only_production_latency_and_account_every_sample(
        self,
    ) -> None:
        # A derived budget or dropped non-selected config hides capacity risk.
        from experiments.decision_cross_validation import prepare_manifest_for_evaluation

        search_space = {
            "coarse_sample_limit": 4,
            "transition_guard_profile": [{"enabled": False}],
            "candidate_weight_profile": [{"expected_shrink": 0.3}],
            "finish_weight_profile": [{"resolve_at_10": 0.5}],
            "pool_size": [300, 1000],
            "prior_alpha": [0.25],
            "prior_temperature": [1.0],
            "other_answer_probability": [0.75],
            "other_vagueness_penalty": [0.1],
            "finish_candidate_threshold": [100],
            "remaining_question_threshold": [2],
            "lookahead_depth": [1],
            "termination_mode": ["explicit_only"],
        }
        catalog = {
            "pool_sizes": {
                "300": {
                    "latency_ms": {"p95": 2999.0},
                    "analysis_kernel_latency_ms": {"p95": 99999.0},
                    "candidate_deletion_stability": {"choice_agreement": 0.80},
                },
                "1000": {
                    "latency_ms": {"p95": 3001.0},
                    "candidate_deletion_stability": {"choice_agreement": 0.99},
                },
            }
        }
        manifest = prepare_manifest_for_evaluation(search_space, catalog, seed=2, max_depth_one=2)
        statuses = {item["status"] for item in manifest["configs"]}
        self.assertIn("evaluated_coarse", statuses)
        self.assertIn("preexcluded_latency_budget", statuses)
        self.assertEqual(len(manifest["configs"]), manifest["accounted_config_count"])

    def test_recognizer_base_rejects_a_frozen_revision_after_intent_changes(self) -> None:
        # The merged hard-cue/parser upgrade must invalidate an experiment pinned to 80e1480.
        from experiments.decision_cross_validation import resolve_and_verify_recognizer_base_sha

        with self.assertRaisesRegex(ValueError, "differ"):
            resolve_and_verify_recognizer_base_sha("80e1480")
        with self.assertRaisesRegex(ValueError, "required"):
            resolve_and_verify_recognizer_base_sha("HEAD")

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
