"""Privacy-safe catalog-scale diagnostic for candidate question value.

This module intentionally does not alter production question-policy scoring.  It
uses the same immutable attribute profiles and depth-one signal definitions, but
evaluates them through a bitset index so a 50k catalog / 1,000 sample experiment
does not repeatedly normalize candidates or rebuild the attribute cache.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from agent.dialogue.candidate_signals import CONCRETE_ATTRIBUTES, CandidateSignalCalculator
from agent.dialogue.catalog_attributes import (
    AttributeProfile,
    CatalogAttributeCache,
    RuleVocabularyExtractor,
)
from config.models import CandidateQuestionValueConfig, FinishStrategyConfig

_DEPTH_TWO_STATUS = "diagnostic_non_promotion_known_gate_mismatch"
_ROUND_DIGITS = 8


def run_catalog_experiment(
    catalog_path: Path | str,
    pool_sizes: Sequence[int] = (300, 500, 1000),
    sample_count: int = 1000,
    seed: int = 20260829,
) -> dict[str, object]:
    """Run a deterministic aggregate-only question-value experiment.

    The catalog is read once and converted to exactly one ``CatalogAttributeCache``.
    Subsequent measurements reuse those profiles and deterministic sampled IDs.
    Neither IDs nor free-text product fields enter the returned report.
    """
    catalog = _validate_catalog_path(catalog_path)
    normalized_pools = _validate_arguments(pool_sizes, sample_count, seed)
    products = _read_catalog_jsonl(catalog)
    cache = CatalogAttributeCache.from_products(products, RuleVocabularyExtractor())
    if not cache.profiles:
        raise ValueError("catalog contains no products with a parent_asin")
    if normalized_pools[-1] > len(cache.profiles):
        raise ValueError("largest pool size cannot exceed the unique catalog product count")

    # Construct the production calculator once to validate the exact baseline
    # configuration.  The batched index below preserves its conservative missing
    # value semantics while avoiding its per-sample candidate normalization cost.
    candidate_config = CandidateQuestionValueConfig()
    finish_config = FinishStrategyConfig()
    CandidateSignalCalculator(cache, candidate_config, finish_config)

    categories = _category_buckets(cache)
    category_by_asin = {
        asin: category for category, asins in categories.items() for asin in asins
    }
    category_mass = {
        category: len(asins) / len(cache.profiles) for category, asins in categories.items()
    }
    coverage = _catalog_attribute_coverage(cache.profiles.values())
    measurements: dict[int, list[_PoolMeasurement]] = {pool: [] for pool in normalized_pools}
    sampler = _MasterWindowSampler(categories, seed)
    latency_samples: dict[int, list[float]] = {pool: [] for pool in normalized_pools}
    latency_indices = _latency_sample_indices(sample_count)

    for sample_index in range(sample_count):
        window = sampler.window(sample_index, normalized_pools[-1])
        sampled_for_pool = {
            pool: tuple(cache.profiles[asin] for asin in window[:pool])
            for pool in normalized_pools
        }
        baseline = {
            pool: _measure_pool(
                sampled_for_pool[pool],
                random.Random(_deletion_seed(seed, sample_index, pool)),
                candidate_config,
                finish_config,
            )
            for pool in normalized_pools
        }
        largest_choice = baseline[normalized_pools[-1]].choice
        for pool, measurement in baseline.items():
            represented = {
                category_by_asin[profile.parent_asin] for profile in sampled_for_pool[pool]
            }
            measurement.represented_category_count = len(represented)
            measurement.catalog_mass_coverage = sum(
                category_mass[category] for category in represented
            )
            measurement.largest_pool_choice_match = measurement.choice == largest_choice
            measurements[pool].append(measurement)
            if sample_index in latency_indices:
                candidates = [
                    {"parent_asin": profile.parent_asin} for profile in sampled_for_pool[pool]
                ]
                start = time.perf_counter()
                CandidateSignalCalculator(cache, candidate_config, finish_config).calculate(
                    candidates
                )
                latency_samples[pool].append((time.perf_counter() - start) * 1000.0)

    return {
        "schema_version": 1,
        "catalog_count": len(cache.profiles),
        "catalog_fingerprint": cache.catalog_fingerprint,
        "vocabulary_version": cache.vocabulary_version,
        "sample_count": sample_count,
        "seed": seed,
        "pool_sizes": {
            str(pool): _aggregate_pool_measurements(measurements[pool], latency_samples[pool])
            for pool in normalized_pools
        },
        "attribute_coverage": coverage,
        "depth_two_measurement": {
            "status": _DEPTH_TWO_STATUS,
            "reason": "known depth-two gate mismatch; diagnostic only and not promotion-ready",
        },
    }


def write_report_atomic(
    report: Mapping[str, object], output_path: Path | str, catalog_path: Path | str
) -> None:
    """Durably replace an output only after complete JSON has been written."""
    output = Path(output_path)
    catalog = _validate_catalog_path(catalog_path)
    if output.resolve() == catalog.resolve() or (
        output.exists() and os.path.samefile(output, catalog)
    ):
        raise ValueError("output path must not resolve to the catalog")
    if not output.parent.exists() or not output.parent.is_dir():
        raise ValueError("output directory does not exist")
    payload = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(output)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _validate_catalog_path(catalog_path: Path | str) -> Path:
    catalog = Path(catalog_path)
    if not catalog.exists() or not catalog.is_file():
        raise FileNotFoundError(f"catalog does not exist or is not a file: {catalog}")
    return catalog


def _validate_arguments(
    pool_sizes: Sequence[int], sample_count: int, seed: int
) -> tuple[int, ...]:
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    normalized = tuple(pool_sizes)
    valid_pool_sizes = normalized and all(
        not isinstance(size, bool) and isinstance(size, int) and size > 0
        for size in normalized
    )
    if not valid_pool_sizes:
        raise ValueError("pool_sizes must contain positive integers")
    if tuple(sorted(normalized)) != normalized or len(set(normalized)) != len(normalized):
        raise ValueError("pool_sizes must be unique and sorted ascending")
    return normalized


def _read_catalog_jsonl(catalog: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with catalog.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL catalog row at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"catalog row at line {line_number} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError("catalog contains no JSONL objects")
    return rows


def _category_buckets(cache: CatalogAttributeCache) -> dict[str, tuple[str, ...]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for asin, profile in cache.profiles.items():
        category_values = profile.values.get("category", frozenset())
        category = min(category_values) if category_values else "__all__"
        buckets[category].append(asin)
    return {category: tuple(sorted(asins)) for category, asins in sorted(buckets.items())}


class _MasterWindowSampler:
    """One proportional order, so all pool sizes share paired candidate windows."""

    def __init__(self, buckets: Mapping[str, Sequence[str]], seed: int) -> None:
        randomizer = random.Random(seed)
        total = sum(len(values) for values in buckets.values())
        scheduled: list[tuple[float, int, str]] = []
        for tie, category in enumerate(sorted(buckets)):
            values = list(buckets[category])
            randomizer.shuffle(values)
            offset = randomizer.random()
            scheduled.extend(
                ((position + offset) / len(values), tie, asin)
                for position, asin in enumerate(values)
            )
        self._master = tuple(asin for _, _, asin in sorted(scheduled))
        self._stride = _coprime_stride(seed, total)

    def window(self, sample_index: int, count: int) -> tuple[str, ...]:
        start = (sample_index * self._stride) % len(self._master)
        return tuple(self._master[(start + offset) % len(self._master)] for offset in range(count))


def _coprime_stride(seed: int, total: int) -> int:
    stride = seed % max(1, total - 1) + 1
    while math.gcd(stride, total) != 1:
        stride = stride % total + 1
    return stride


def _deletion_seed(seed: int, sample_index: int, pool_size: int) -> int:
    return (seed * 1_000_003 + sample_index * 9_973 + pool_size * 101) & ((1 << 63) - 1)


def _latency_sample_indices(sample_count: int) -> frozenset[int]:
    count = min(5, sample_count)
    return frozenset(index * sample_count // count for index in range(count))


def _catalog_attribute_coverage(profiles: Iterable[AttributeProfile]) -> dict[str, float]:
    rows = tuple(profiles)
    return {
        attribute: _round(sum(bool(row.values.get(attribute)) for row in rows) / len(rows))
        for attribute in CONCRETE_ATTRIBUTES
    }


class _PoolIndex:
    def __init__(self, profiles: Sequence[AttributeProfile]) -> None:
        self.profiles = tuple(profiles)
        self.count = len(self.profiles)
        self._attribute_values: dict[str, tuple[frozenset[str], ...]] = {}
        self._confidence: dict[str, tuple[float, ...]] = {}
        self._value_masks: dict[str, dict[str, int]] = {}
        self._missing_masks: dict[str, int] = {}
        for attribute in CONCRETE_ATTRIBUTES:
            values_by_row: list[frozenset[str]] = []
            confidence: list[float] = []
            value_masks: dict[str, int] = {}
            missing_mask = 0
            for index, profile in enumerate(self.profiles):
                values = frozenset(profile.values.get(attribute, frozenset()))
                values_by_row.append(values)
                value = profile.confidence.get(attribute, 0.0)
                confidence.append(_bounded(value))
                if not values:
                    missing_mask |= 1 << index
                for item in values:
                    value_masks[item] = value_masks.get(item, 0) | (1 << index)
            self._attribute_values[attribute] = tuple(values_by_row)
            self._confidence[attribute] = tuple(confidence)
            self._value_masks[attribute] = value_masks
            self._missing_masks[attribute] = missing_mask

    def signal(self, attribute: str) -> _Signal:
        remaining = self._remaining_counts(attribute)
        count = self.count
        return _Signal(
            expected_shrink=_bounded(1.0 - sum(remaining) / (count * count)),
            resolve_at_10=_bounded(sum(value <= 10 for value in remaining) / count),
            resolve_at_3=_bounded(sum(value <= 3 for value in remaining) / count),
            resolve_at_1=_bounded(sum(value <= 1 for value in remaining) / count),
            missing_rate=_bounded(
                sum(not values for values in self._attribute_values[attribute]) / count
            ),
            p90_remaining=float(sorted(remaining)[math.ceil(0.9 * count) - 1]),
            expected_remaining=sum(remaining) / count,
        )

    def branch_mask(self, attribute: str, index: int) -> int:
        values = self._attribute_values[attribute][index]
        if not values:
            return (1 << self.count) - 1
        mask = self._missing_masks[attribute]
        for value in values:
            mask |= self._value_masks[attribute].get(value, 0)
        return mask

    def _remaining_counts(self, attribute: str) -> tuple[int, ...]:
        return tuple(self.branch_mask(attribute, index).bit_count() for index in range(self.count))


class _Signal:
    def __init__(
        self,
        *,
        expected_shrink: float,
        resolve_at_10: float,
        resolve_at_3: float,
        resolve_at_1: float,
        missing_rate: float,
        p90_remaining: float,
        expected_remaining: float,
    ) -> None:
        self.expected_shrink = expected_shrink
        self.resolve_at_10 = resolve_at_10
        self.resolve_at_3 = resolve_at_3
        self.resolve_at_1 = resolve_at_1
        self.missing_rate = missing_rate
        self.p90_remaining = p90_remaining
        self.expected_remaining = expected_remaining


class _PoolMeasurement:
    def __init__(
        nself,
        choice: str,
        signals: Mapping[str, _Signal],
        latency_ms: float,
        one_step: float,
        two_step: float,
        deletion: "_DeletedMeasurement",
    ) -> None:
        nself.choice = choice
        nself.signals = dict(signals)
        nself.latency_ms = latency_ms
        nself.one_step = one_step
        nself.two_step = two_step
        nself.deletion = deletion
        nself.largest_pool_choice_match = False
        nself.represented_category_count = 0
        nself.catalog_mass_coverage = 0.0


class _DeletedMeasurement:
    def __init__(
        nself,
        choice: str,
        signal: _Signal,
        one_step: float,
        two_step: float,
        original_count: int,
        deleted_count: int,
    ) -> None:
        nself.choice = choice
        nself.signal = signal
        nself.one_step = one_step
        nself.two_step = two_step
        nself.original_count = original_count
        nself.deleted_count = deleted_count


def _measure_pool(
    profiles: Sequence[AttributeProfile],
    randomizer: random.Random,
    candidate_config: CandidateQuestionValueConfig,
    finish_config: FinishStrategyConfig,
) -> _PoolMeasurement:
    index = _PoolIndex(profiles)
    start = time.perf_counter()
    signals = {attribute: index.signal(attribute) for attribute in CONCRETE_ATTRIBUTES}
    latency_ms = (time.perf_counter() - start) * 1000.0
    choice = _choose_attribute(signals)
    one_step = _finish_gain(signals[choice], index.count, finish_config)
    two_step = _diagnostic_two_step_gain(index, choice, candidate_config, finish_config)
    deleted_profiles, deleted_count = _delete_candidates(profiles, randomizer)
    deleted_index = _PoolIndex(deleted_profiles)
    deleted_signals = {
        attribute: deleted_index.signal(attribute) for attribute in CONCRETE_ATTRIBUTES
    }
    deleted_choice = _choose_attribute(deleted_signals)
    deleted_one_step = _finish_gain(
        deleted_signals[deleted_choice], deleted_index.count, finish_config
    )
    deleted_two_step = _diagnostic_two_step_gain(
        deleted_index, deleted_choice, candidate_config, finish_config
    )
    return _PoolMeasurement(
        choice,
        signals,
        latency_ms,
        one_step,
        two_step,
        _DeletedMeasurement(
            deleted_choice,
            deleted_signals[deleted_choice],
            deleted_one_step,
            deleted_two_step,
            len(profiles),
            deleted_count,
        ),
    )


def _delete_candidates(
    profiles: Sequence[AttributeProfile], randomizer: random.Random
) -> tuple[tuple[AttributeProfile, ...], int]:
    delete_count = max(1, round(len(profiles) * 0.10)) if len(profiles) > 1 else 0
    deleted = set(randomizer.sample(range(len(profiles)), delete_count))
    remaining = tuple(profile for index, profile in enumerate(profiles) if index not in deleted)
    return remaining, delete_count


def _choose_attribute(signals: Mapping[str, _Signal]) -> str:
    return min(
        signals,
        key=lambda attribute: (-signals[attribute].expected_shrink, attribute),
    )


def _finish_gain(signal: _Signal, count: int, config: FinishStrategyConfig) -> float:
    weights = config.weights
    terminal_progress = _terminal_progress(signal.expected_remaining, count)
    return (
        weights.resolve_at_10 * signal.resolve_at_10
        + weights.resolve_at_3 * signal.resolve_at_3
        + weights.resolve_at_1 * signal.resolve_at_1
        + weights.terminal_progress * terminal_progress
        - weights.p90_remaining_penalty * signal.p90_remaining / count
    )


def _diagnostic_two_step_gain(
    index: _PoolIndex,
    first_attribute: str,
    candidate_config: CandidateQuestionValueConfig,
    finish_config: FinishStrategyConfig,
) -> float:
    """Measure the production depth-two formula without enabling it in policy."""
    branch_gains: dict[int, float] = {}
    weighted_gain = 0.0
    for row_index in range(index.count):
        mask = index.branch_mask(first_attribute, row_index)
        gain = branch_gains.get(mask)
        if gain is None:
            branch_profiles = tuple(
                profile for offset, profile in enumerate(index.profiles) if mask & (1 << offset)
            )
            branch_index = _PoolIndex(branch_profiles)
            branch_signals = {
                attribute: branch_index.signal(attribute)
                for attribute in CONCRETE_ATTRIBUTES
                if attribute != first_attribute
            }
            gain = max(
                _finish_gain(signal, branch_index.count, finish_config)
                for signal in branch_signals.values()
            )
            branch_gains[mask] = gain
        weighted_gain += gain / index.count
    return max(0.0, weighted_gain - candidate_config.weights.turn_cost)


def _terminal_progress(expected_remaining: float, count: int) -> float:
    if count <= 10:
        return 0.0
    initial = math.log1p(count - 10)
    remaining = math.log1p(max(expected_remaining - 10.0, 0.0))
    return 1.0 - remaining / initial


def _aggregate_pool_measurements(
    measurements: Sequence[_PoolMeasurement], production_latency_ms: Sequence[float]
) -> dict[str, object]:
    chosen_signals = [measurement.signals[measurement.choice] for measurement in measurements]
    one_step_finish_gain = _round(_mean(item.one_step for item in measurements))
    two_step_incremental_gain = _round(_mean(item.two_step for item in measurements))
    return {
        "sample_count": len(measurements),
        "latency_ms": {
            "measurement": "production_candidate_signal_calculator_calculate",
            "sample_count": len(production_latency_ms),
            "p50": _round(_percentile(production_latency_ms, 0.50)),
            "p95": _round(_percentile(production_latency_ms, 0.95)),
        },
        "analysis_kernel_latency_ms": {
            "measurement": "diagnostic_optimized_depth_one_analysis_kernel",
            "sample_count": len(measurements),
            "p50": _round(_percentile([item.latency_ms for item in measurements], 0.50)),
            "p95": _round(_percentile([item.latency_ms for item in measurements], 0.95)),
        },
        "mean_represented_category_count": _round(
            _mean(item.represented_category_count for item in measurements)
        ),
        "mean_catalog_mass_coverage": _round(
            _mean(item.catalog_mass_coverage for item in measurements)
        ),
        "chosen_attribute_agreement_with_largest_pool": _round(
            sum(item.largest_pool_choice_match for item in measurements) / len(measurements)
        ),
        "mean_expected_shrink": _round(_mean(item.expected_shrink for item in chosen_signals)),
        "mean_resolve_at_10": _round(_mean(item.resolve_at_10 for item in chosen_signals)),
        "one_step_finish_gain": one_step_finish_gain,
        "two_step_incremental_gain": two_step_incremental_gain,
        "two_step_combined_gain": one_step_finish_gain + two_step_incremental_gain,
        "depth_two_status": _DEPTH_TWO_STATUS,
        "per_attribute_missing_rate": {
            attribute: _round(_mean(item.signals[attribute].missing_rate for item in measurements))
            for attribute in CONCRETE_ATTRIBUTES
        },
        "candidate_deletion_stability": {
            "deleted_count": measurements[0].deletion.deleted_count,
            "deleted_fraction": _round(
                measurements[0].deletion.deleted_count / measurements[0].deletion.original_count
            ),
            "choice_agreement": _round(
                sum(item.choice == item.deletion.choice for item in measurements)
                / len(measurements)
            ),
            "mean_expected_shrink_delta": _round(
                _mean(
                    item.deletion.signal.expected_shrink
                    - item.signals[item.choice].expected_shrink
                    for item in measurements
                )
            ),
            "mean_resolve_at_10_delta": _round(
                _mean(
                    item.deletion.signal.resolve_at_10
                    - item.signals[item.choice].resolve_at_10
                    for item in measurements
                )
            ),
            "mean_one_step_finish_gain_delta": _round(
                _mean(item.deletion.one_step - item.one_step for item in measurements)
            ),
            "mean_two_step_finish_gain_delta": _round(
                _mean(item.deletion.two_step - item.two_step for item in measurements)
            ),
        },
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(values)
    return sum(materialized) / len(materialized)


def _bounded(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, numeric)) if math.isfinite(numeric) else 0.0


def _round(value: float) -> float:
    return round(value, _ROUND_DIGITS)


def _parse_pool_sizes(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(piece) for piece in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pool sizes must be comma-separated integers") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pool-sizes", default=(300, 500, 1000), type=_parse_pool_sizes)
    parser.add_argument("--sample-count", default=1000, type=int)
    parser.add_argument("--seed", default=20260829, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        report = run_catalog_experiment(
            arguments.catalog,
            arguments.pool_sizes,
            arguments.sample_count,
            arguments.seed,
        )
        write_report_atomic(report, arguments.output, arguments.catalog)
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
