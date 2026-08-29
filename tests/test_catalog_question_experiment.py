from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
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
        self.assertIn("chosen_attribute_agreement_with_largest_pool", pool)
        self.assertIn("mean_expected_shrink", pool)
        self.assertIn("mean_resolve_at_10", pool)
        self.assertIn("one_step_vs_two_step_finish_gain", pool)
        self.assertIn("per_attribute_missing_rate", pool)
        self.assertIn("candidate_deletion_stability", pool)
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
            output.write_text("old-report", encoding="utf-8")
            with patch.object(Path, "replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    write_report_atomic({"safe": "aggregate"}, output)
            self.assertEqual(output.read_text(encoding="utf-8"), "old-report")
            self.assertEqual(list(Path(temporary).glob(".report.json.*.tmp")), [])


def _without_timing(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_timing(item)
            for key, item in value.items()
            if key != "latency_ms"
        }
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    return value
