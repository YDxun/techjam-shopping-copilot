from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class CatalogQuestionExperimentTest(unittest.TestCase):
    def _catalog(self, directory: Path) -> Path:
        rows = [
            {
                "parent_asin": "A",
                "title": "PRIVATE_TITLE_A",
                "description": ["PRIVATE_DESCRIPTION_A"],
                "categories": ["Clothing", "Tops"],
                "details": {"Material": "Cotton", "Color": "Black"},
                "price": 10,
            },
            {
                "parent_asin": "B",
                "title": "PRIVATE_TITLE_B",
                "description": ["PRIVATE_DESCRIPTION_B"],
                "categories": ["Clothing", "Tops"],
                "details": {"Material": "Leather", "Color": "Red"},
                "price": 20,
            },
            {
                "parent_asin": "C",
                "title": "PRIVATE_TITLE_C",
                "description": ["PRIVATE_DESCRIPTION_C"],
                "categories": ["Clothing", "Shoes"],
                "details": {"Material": "Cotton", "Color": "Red"},
                "price": 30,
            },
            {
                "parent_asin": "D",
                "title": "PRIVATE_TITLE_D",
                "description": ["PRIVATE_DESCRIPTION_D"],
                "categories": ["Clothing", "Shoes"],
                "details": {"Material": "Leather", "Color": "Black"},
                "price": 40,
            },
        ]
        path = directory / "catalog.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def test_reports_aggregate_metrics_without_product_text(self) -> None:
        # Returning a raw catalog row or profile source text would expose either
        # sentinel and make this privacy boundary fail.
        from experiments.catalog_question_value import run_catalog_experiment

        with tempfile.TemporaryDirectory() as temporary:
            report = run_catalog_experiment(
                catalog_path=self._catalog(Path(temporary)),
                pool_sizes=(3, 4),
                sample_count=4,
                seed=17,
            )

        self.assertEqual(report["catalog_count"], 4)
        self.assertEqual(set(report["pool_sizes"]), {"3", "4"})
        self.assertIn("attribute_coverage", report)
        pool = report["pool_sizes"]["3"]
        self.assertIn("latency_ms", pool)
        self.assertEqual(pool["latency_ms"]["sample_count"], 4)
        self.assertIn("analysis_kernel_latency_ms", pool)
        self.assertIn("mean_represented_category_count", pool)
        self.assertIn("mean_catalog_mass_coverage", pool)
        self.assertIn("chosen_attribute_agreement_with_largest_pool", pool)
        self.assertIn("mean_expected_shrink", pool)
        self.assertIn("mean_resolve_at_10", pool)
        self.assertIn("one_step_finish_gain", pool)
        self.assertIn("two_step_incremental_gain", pool)
        self.assertIn("two_step_combined_gain", pool)
        self.assertAlmostEqual(
            pool["two_step_combined_gain"],
            pool["one_step_finish_gain"] + pool["two_step_incremental_gain"],
            places=8,
        )
        self.assertIn("per_attribute_missing_rate", pool)
        self.assertIn("candidate_deletion_stability", pool)
        self.assertEqual(pool["candidate_deletion_stability"]["deleted_count"], 1)
        self.assertEqual(pool["candidate_deletion_stability"]["deleted_fraction"], 0.33333333)
        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn("PRIVATE_TITLE", rendered)
        self.assertNotIn("PRIVATE_DESCRIPTION", rendered)
        self.assertNotIn('"A"', rendered)

    def test_same_seed_has_byte_identical_non_timing_fields(self) -> None:
        # Changing sampling order or relying on process-global randomness would
        # change a structural field after timing values are removed.
        from experiments.catalog_question_value import run_catalog_experiment

        with tempfile.TemporaryDirectory() as temporary:
            catalog = self._catalog(Path(temporary))
            first = run_catalog_experiment(catalog, (3, 4), 4, 17)
            second = run_catalog_experiment(catalog, (3, 4), 4, 17)

        self.assertEqual(_without_timing(first), _without_timing(second))
        self.assertEqual(
            json.dumps(_without_timing(first), sort_keys=True, separators=(",", ":")),
            json.dumps(_without_timing(second), sort_keys=True, separators=(",", ":")),
        )

    def test_rejects_missing_catalog_and_invalid_sampling_arguments(self) -> None:
        # Accepting zero, unsorted pools, or a missing input would make a report
        # silently incomparable or misleading.
        from experiments.catalog_question_value import run_catalog_experiment

        with self.assertRaisesRegex(FileNotFoundError, "catalog"):
            run_catalog_experiment(Path("missing-catalog.jsonl"), (3,), 1, 17)
        with tempfile.TemporaryDirectory() as temporary:
            catalog = self._catalog(Path(temporary))
            invalid_arguments = (((0,), 1, 17), ((4, 3), 1, 17), ((3,), 0, 17), ((3,), 1, -1))
            for pool_sizes, sample_count, seed in invalid_arguments:
                with self.assertRaises(ValueError):
                    run_catalog_experiment(catalog, pool_sizes, sample_count, seed)

    def test_atomic_writer_preserves_existing_output_when_replace_fails(self) -> None:
        # Replacing the target before the complete JSON is durable would corrupt
        # a prior usable experiment result on a write failure.
        from experiments.catalog_question_value import write_report_atomic

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report.json"
            catalog = self._catalog(Path(temporary))
            output.write_text("old-report", encoding="utf-8")
            with patch.object(Path, "replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    write_report_atomic({"safe": "aggregate"}, output, catalog)
            self.assertEqual(output.read_text(encoding="utf-8"), "old-report")
            self.assertEqual(list(Path(temporary).glob(".report.json.*.tmp")), [])

    def test_category_rotation_gives_long_tail_proportional_opportunity(self) -> None:
        # Always giving Hamilton remainder seats to the largest category would
        # keep the rare category out of every pool despite its positive mass.
        from experiments.catalog_question_value import run_catalog_experiment

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rows = [
                {
                    "parent_asin": f"T{index}",
                    "categories": ["Clothing", "Tops"],
                    "details": {"Material": "Cotton"},
                }
                for index in range(8)
            ] + [
                {
                    "parent_asin": "S0",
                    "categories": ["Clothing", "Shoes"],
                    "details": {"Material": "Leather"},
                }
            ]
            catalog = directory / "long-tail.jsonl"
            catalog.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            first = run_catalog_experiment(catalog, (2,), 100, 29)
            second = run_catalog_experiment(catalog, (2,), 100, 29)

        pool = first["pool_sizes"]["2"]
        self.assertGreater(pool["mean_represented_category_count"], 1.0)
        self.assertGreater(pool["mean_catalog_mass_coverage"], 0.89)
        self.assertEqual(_without_timing(first), _without_timing(second))

    def test_master_windows_pair_pool_sizes_independently_of_requested_list(self) -> None:
        # Sampling each pool with shared mutable credits makes the 300-pool
        # result depend on whether a 500-pool result was requested beside it.
        from experiments.catalog_question_value import run_catalog_experiment

        with tempfile.TemporaryDirectory() as temporary:
            catalog = self._catalog(Path(temporary))
            alone = run_catalog_experiment(catalog, (2,), 8, 31)
            paired = run_catalog_experiment(catalog, (2, 4), 8, 31)

        self.assertEqual(
            _without_comparator(alone["pool_sizes"]["2"]),
            _without_comparator(paired["pool_sizes"]["2"]),
        )

    def test_master_window_prefixes_and_seed_are_deterministic(self) -> None:
        # Independent pool draws would not guarantee that every small-pool
        # candidate is also present in the paired larger pool.
        from experiments.catalog_question_value import _MasterWindowSampler

        buckets = {"large": ("a", "b", "c", "d"), "small": ("e", "f")}
        first = _MasterWindowSampler(buckets, 41)
        second = _MasterWindowSampler(buckets, 41)
        for index in range(6):
            small = first.window(index, 2)
            large = first.window(index, 5)
            self.assertEqual(small, large[:2])
            self.assertEqual(small, second.window(index, 2))

    def test_rejects_output_aliases_before_touching_catalog(self) -> None:
        # Writing through a direct, symlink, or hardlink alias could replace the
        # catalog itself instead of merely replacing a separate report.
        from experiments.catalog_question_value import write_report_atomic

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            catalog = self._catalog(directory)
            original = catalog.read_bytes()
            aliases = [
                catalog,
                directory / "catalog-symlink.jsonl",
                directory / "catalog-hardlink.jsonl",
            ]
            os.symlink(catalog.name, aliases[1])
            os.link(catalog, aliases[2])
            for alias in aliases:
                with self.subTest(alias=alias.name):
                    with self.assertRaisesRegex(ValueError, "catalog"):
                        write_report_atomic({"safe": "aggregate"}, alias, catalog)
                    self.assertEqual(catalog.read_bytes(), original)

    def test_combined_two_step_gain_sums_displayed_components(self) -> None:
        # Rounding the raw sum instead of the displayed components can make the
        # report's claimed arithmetic false by one final decimal place.
        from agent.dialogue.candidate_signals import CONCRETE_ATTRIBUTES
        from experiments.catalog_question_value import _aggregate_pool_measurements

        signal = SimpleNamespace(expected_shrink=0.1, resolve_at_10=0.2, missing_rate=0.3)
        deleted = SimpleNamespace(
            choice="material",
            signal=signal,
            one_step=0.1,
            two_step=0.2,
            deleted_count=1,
            original_count=10,
        )
        measurement = SimpleNamespace(
            choice="material",
            signals={attribute: signal for attribute in CONCRETE_ATTRIBUTES},
            latency_ms=1.0,
            largest_pool_choice_match=True,
            represented_category_count=1,
            catalog_mass_coverage=1.0,
            one_step=0.956803674,
            two_step=0.536463674,
            deletion=deleted,
        )
        report = _aggregate_pool_measurements([measurement], [1.0])
        self.assertEqual(
            report["two_step_combined_gain"],
            report["one_step_finish_gain"] + report["two_step_incremental_gain"],
        )


def _without_timing(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_timing(item)
            for key, item in value.items()
            if key not in {"latency_ms", "analysis_kernel_latency_ms"}
        }
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    return value


def _without_comparator(value: object) -> object:
    report = _without_timing(value)
    assert isinstance(report, dict)
    return {
        key: item
        for key, item in report.items()
        if key != "chosen_attribute_agreement_with_largest_pool"
    }
