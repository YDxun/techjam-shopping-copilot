"""Bounded, offline comparison of Legacy and Hybrid question policies.

The comparison intentionally calls the official evaluator unchanged.  It shares
only immutable catalog-derived work (the SQLite retriever and dialogue resource
bundle); each configuration receives a separate Agent and therefore separate
dialogue, reranker, and diagnostic state.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import signal
import sys
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from agent.dialogue.catalog_resources import DialogueCatalogResources
from agent.main_agent import Agent
from agent.retriever import HybridRetriever
from config.env_config import EnvConfig
from evaluator.local_evaluator import evaluate, load_jsonl

STRATUM_COUNTS = {
    "buying": 8,
    "browsing": 8,
    "intent_override": 3,
    "boundary": 1,
}
_METRIC_KEYS = (
    "sample_count",
    "hit_rate_at_10",
    "mrr",
    "mttc",
    "efficiency",
    "recommended_technical_score",
    "reported_token_usage",
    "scenario_metrics",
)
_FIXED_HYBRID_WEIGHTS = {
    "expected_shrink": 0.40,
    "resolve_at_10": 0.25,
    "coverage": 0.15,
    "answer_probability": 0.10,
    "extraction_confidence": 0.10,
    "missing_penalty": 0.25,
    "turn_cost": 0.10,
}
_GATES = {
    "hybrid_conservative": (0.70, 0.30, 0.35, 0.10, 0.35),
    "hybrid_balanced": (0.60, 0.40, 0.25, 0.05, 0.25),
    "hybrid_permissive": (0.50, 0.50, 0.20, 0.00, 0.15),
}


class TimeBudgetExceeded(TimeoutError):
    """Raised when the process-wide comparison deadline expires."""


def select_stratified_public_samples(samples: Sequence[dict], seed: int) -> list[dict]:
    """Return the deterministic 8/8/3/1 public-set screen without leakage."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    strata: dict[str, list[dict]] = defaultdict(list)
    seen_ids: set[str] = set()
    for row in samples:
        if not isinstance(row, Mapping):
            raise ValueError("public dataset rows must be objects")
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError("public dataset rows require a non-empty sample_id")
        if sample_id in seen_ids:
            raise ValueError("public dataset contains duplicate sample_id values")
        seen_ids.add(sample_id)
        scenario_type = str(row.get("scenario_type", "")).strip()
        if scenario_type in STRATUM_COUNTS:
            strata[scenario_type].append(dict(row))

    selected: list[dict] = []
    randomizer = random.Random(seed)
    for scenario_type, required_count in STRATUM_COUNTS.items():
        source = sorted(strata[scenario_type], key=lambda row: str(row["sample_id"]))
        if len(source) < required_count:
            raise ValueError(
                f"public dataset needs at least {required_count} {scenario_type!r} samples"
            )
        randomizer.shuffle(source)
        selected.extend(source[:required_count])
    return selected


def comparison_configurations() -> tuple[dict[str, object], ...]:
    """Return the four predeclared, fixed offline configuration overlays."""
    configurations = [
        {
            "name": "legacy",
            "decision_overlay": {},
            "overlay": _offline_overlay({}),
        }
    ]
    for name, gates in _GATES.items():
        (
            minimum_coverage,
            maximum_missing_rate,
            minimum_expected_shrink,
            minimum_resolve_at_10,
            minimum_gain,
        ) = gates
        decision_overlay = {
            "hybrid_question_policy": {
                "enabled": True,
                "max_replacements_per_session": 1,
                "only_after_other_asked": True,
                "pool_size": 300,
                "prior_alpha": 0.25,
                "prior_temperature": 1.0,
                "minimum_coverage": minimum_coverage,
                "maximum_missing_rate": maximum_missing_rate,
                "minimum_expected_shrink": minimum_expected_shrink,
                "minimum_resolve_at_10": minimum_resolve_at_10,
                "minimum_gain": minimum_gain,
                "weights": dict(_FIXED_HYBRID_WEIGHTS),
            }
        }
        configurations.append(
            {
                "name": name,
                "decision_overlay": decision_overlay,
                "overlay": _offline_overlay(decision_overlay),
            }
        )
    return tuple(configurations)


def run_comparison(
    catalog_path: str | Path,
    dataset_path: str | Path,
    seed: int,
    time_budget_seconds: float,
    *,
    output_path: str | Path | None = None,
    retriever_factory: Callable[..., object] | None = None,
    resource_factory: Callable[..., object] | None = None,
    agent_factory: Callable[..., object] | None = None,
    evaluator: Callable[..., Mapping[str, object]] | None = None,
    monotonic: Callable[[], float] | None = None,
) -> dict[str, object]:
    """Evaluate Legacy and three Hybrid gate settings under one hard deadline.

    Optional factories make the isolation contract testable without altering
    production evaluation, and are deliberately keyword-only so the public
    runner interface stays compact.
    """
    budget = _validate_time_budget(time_budget_seconds)
    clock = monotonic or time.monotonic
    started_at = clock()
    catalog = _validate_input_path(catalog_path, "catalog")
    dataset = _validate_input_path(dataset_path, "dataset")
    retriever_builder = retriever_factory or HybridRetriever
    resources_builder = resource_factory or DialogueCatalogResources.from_products
    build_agent = agent_factory or Agent
    evaluate_sessions = evaluator or evaluate
    configurations: list[dict[str, object]] = []
    selected: list[dict] = []
    initialization_elapsed = 0.0
    timeout_configuration: str | None = None
    enforcement_mode = _deadline_enforcement_mode()
    status = "complete"

    try:
        with _ProcessDeadline(budget, clock) as deadline:
            samples = load_jsonl(dataset)
            selected = select_stratified_public_samples(samples, seed)
            deadline.check()

            # One retriever builds the SQLite index and owns the single product
            # snapshot used for evaluator data plus catalog-derived resources.
            base_env = EnvConfig.from_env(overrides=_offline_overlay({}))
            retriever = retriever_builder(
                catalog,
                env=base_env,
                backend=base_env.retrieval_backend,
            )
            product_snapshot = tuple(retriever.iter_products())
            catalog_ids, categories, products = _evaluator_catalog_snapshot(product_snapshot)
            resources = resources_builder(product_snapshot, include_attribute_cache=True)
            initialization_elapsed = clock() - started_at
            deadline.check()

            for configuration in comparison_configurations():
                configuration_name = str(configuration["name"])
                deadline.check()
                configuration_started = clock()
                env = EnvConfig.from_env(overrides=configuration["overlay"])
                agent = build_agent(
                    catalog_path=catalog,
                    env=env,
                    retriever=retriever,
                    dialogue_catalog_resources=resources,
                )
                official_result = evaluate_sessions(
                    agent, selected, catalog_ids, categories, products
                )
                deadline.check()
                configurations.append(
                    {
                        "name": configuration_name,
                        "overlay": configuration["overlay"],
                        "decision_overlay": configuration["decision_overlay"],
                        "elapsed_seconds": _elapsed(clock, configuration_started),
                        "official_metrics": _official_metrics(official_result),
                        "hybrid_statistics": _hybrid_statistics(agent, len(selected)),
                    }
                )
    except TimeBudgetExceeded:
        status = "time_budget_exceeded"
        if len(configurations) < len(comparison_configurations()):
            timeout_configuration = str(comparison_configurations()[len(configurations)]["name"])

    total_elapsed = _elapsed(clock, started_at)
    if status == "complete" and len(configurations) != len(comparison_configurations()):
        raise RuntimeError("comparison ended without evaluating every fixed configuration")
    report = _comparison_report(
        seed=seed,
        selected=selected,
        configurations=configurations,
        status=status,
        time_budget_seconds=budget,
        enforcement_mode=enforcement_mode,
        initialization_elapsed=initialization_elapsed,
        total_elapsed=total_elapsed,
        timeout_configuration=timeout_configuration,
    )
    if output_path is not None:
        write_report_atomic(report, output_path, catalog, dataset)
    return report


def write_report_atomic(
    report: Mapping[str, object],
    output_path: str | Path,
    catalog_path: str | Path,
    dataset_path: str | Path,
) -> None:
    """Atomically persist a privacy-safe report without replacing either input."""
    output = Path(output_path)
    catalog = _validate_input_path(catalog_path, "catalog")
    dataset = _validate_input_path(dataset_path, "dataset")
    if not output.parent.is_dir():
        raise ValueError("output directory does not exist")
    if _same_file_or_path(output, catalog) or _same_file_or_path(output, dataset):
        raise ValueError("output path must not resolve to an experiment input")
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


def _offline_overlay(decision_overlay: Mapping[str, object]) -> dict[str, object]:
    overlay: dict[str, object] = {
        "llm": {"provider": "none", "rerank_enabled": False},
        "llm_intent_enabled": False,
        "llm_clarify_enabled": False,
        "reranker_model_enabled": False,
        "skip_data_verify": True,
        "dialogue_understanding": {"mode": "rule_only"},
        "diagnostics": {"decision_trace": {"enabled": False}},
        "decision": {
            "candidate_question_value": {"enabled": False},
            "question_termination_mode": "legacy",
            "finish_strategy": {"enabled": False, "lookahead_depth": 1},
        },
    }
    _deep_merge(overlay, decision_overlay, root="decision")
    return overlay


def _deep_merge(
    target: dict[str, object], source: Mapping[str, object], *, root: str | None = None
) -> None:
    current: dict[str, object] = target
    if root is not None:
        existing = target.get(root)
        if not isinstance(existing, dict):
            existing = {}
            target[root] = existing
        current = existing
    for key, value in source.items():
        existing = current.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            _deep_merge(existing, value)
        else:
            current[key] = copy.deepcopy(value)


def _evaluator_catalog_snapshot(
    products: Sequence[dict],
) -> tuple[set[str], dict[str, list[str]], dict[str, dict]]:
    catalog_ids: set[str] = set()
    categories: dict[str, list[str]] = {}
    by_asin: dict[str, dict] = {}
    for product in products:
        if not isinstance(product, dict) or "parent_asin" not in product:
            raise ValueError("retriever product snapshot requires parent_asin objects")
        asin = str(product["parent_asin"]).strip()
        if not asin or asin in catalog_ids:
            raise ValueError("retriever product snapshot has invalid or duplicate parent_asin")
        catalog_ids.add(asin)
        raw_categories = product.get("categories") or []
        categories[asin] = (
            [str(value) for value in raw_categories] if isinstance(raw_categories, list) else []
        )
        by_asin[asin] = product
    if not catalog_ids:
        raise ValueError("retriever product snapshot is empty")
    return catalog_ids, categories, by_asin


def _official_metrics(result: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(result, Mapping):
        raise ValueError("official evaluator must return a mapping")
    return {
        key: _json_safe(result.get(key))
        for key in _METRIC_KEYS
        if key in result
    }


def _hybrid_statistics(agent: object, sample_count: int) -> dict[str, object]:
    getter = getattr(agent, "hybrid_question_statistics", None)
    raw = getter() if callable(getter) else {}
    if not isinstance(raw, Mapping):
        raw = {}
    latency = raw.get("decision_latency_ms")
    latency = latency if isinstance(latency, Mapping) else {}
    replacement_count = _nonnegative_int(raw.get("replacement_count"))
    return {
        "reason_counts": _count_mapping(raw.get("reason_counts")),
        "selected_attribute_counts": _count_mapping(raw.get("selected_attribute_counts")),
        "replacement_count": replacement_count,
        "replacement_rate": round(replacement_count / sample_count, 8) if sample_count else 0.0,
        "decision_latency_ms": {
            "count": _nonnegative_int(latency.get("count")),
            "p50": _finite_number(latency.get("p50")),
            "p95": _finite_number(latency.get("p95")),
        },
    }


def _comparison_report(
    *,
    seed: int,
    selected: Sequence[dict],
    configurations: Sequence[Mapping[str, object]],
    status: str,
    time_budget_seconds: float,
    enforcement_mode: str,
    initialization_elapsed: float,
    total_elapsed: float,
    timeout_configuration: str | None,
) -> dict[str, object]:
    report: dict[str, object] = {
        "schema_version": 1,
        "experiment": "legacy_vs_hybrid_question_policy",
        "status": status,
        "seed": seed,
        "sample_count": len(selected),
        "selected_sample_id_hashes": [_sample_hash(row) for row in selected],
        "stratum_counts": dict(Counter(str(row["scenario_type"]) for row in selected)),
        "time_budget_seconds": time_budget_seconds,
        "deadline_enforcement": {"mode": enforcement_mode},
        "timing_seconds": {
            "initialization": round(max(0.0, initialization_elapsed), 6),
            "total": round(max(0.0, total_elapsed), 6),
        },
        "completed_configuration_count": len(configurations),
        "completed_sample_count": len(configurations) * len(selected),
        "configurations": list(configurations),
    }
    if timeout_configuration is not None:
        report["timeout"] = {"during_configuration": timeout_configuration}
    if status == "complete":
        report["pairwise_vs_legacy"] = _pairwise_vs_legacy(configurations)
    return report


def _pairwise_vs_legacy(configurations: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not configurations:
        return {}
    legacy_metrics = configurations[0].get("official_metrics")
    if not isinstance(legacy_metrics, Mapping):
        return {}
    comparisons: dict[str, object] = {}
    for configuration in configurations[1:]:
        metrics = configuration.get("official_metrics")
        if not isinstance(metrics, Mapping):
            continue
        name = str(configuration.get("name", "hybrid"))
        deltas = {
            key: _metric_delta(metrics.get(key), legacy_metrics.get(key))
            for key in (
                "hit_rate_at_10",
                "mrr",
                "mttc",
                "efficiency",
                "recommended_technical_score",
            )
        }
        scenario_deltas = _scenario_deltas(
            metrics.get("scenario_metrics"), legacy_metrics.get("scenario_metrics")
        )
        hr_delta = deltas["hit_rate_at_10"]
        technical_delta = deltas["recommended_technical_score"]
        mrr_delta = deltas["mrr"]
        mttc_delta = deltas["mttc"]
        screening_holds = (
            isinstance(hr_delta, float)
            and hr_delta >= 0.0
            and (
                (isinstance(technical_delta, float) and technical_delta > 0.0)
                or (isinstance(mrr_delta, float) and mrr_delta > 0.0)
                or (isinstance(mttc_delta, float) and mttc_delta < 0.0)
            )
        )
        comparisons[name] = {
            "overall_metric_deltas": deltas,
            "scenario_metric_deltas": scenario_deltas,
            "predeclared_screening_condition_holds": screening_holds,
        }
    return comparisons


def _scenario_deltas(current: object, legacy: object) -> dict[str, object]:
    if not isinstance(current, Mapping) or not isinstance(legacy, Mapping):
        return {}
    result: dict[str, object] = {}
    for scenario in sorted(set(current) & set(legacy)):
        current_metrics = current[scenario]
        legacy_metrics = legacy[scenario]
        if not isinstance(current_metrics, Mapping) or not isinstance(legacy_metrics, Mapping):
            continue
        result[str(scenario)] = {
            key: _metric_delta(current_metrics.get(key), legacy_metrics.get(key))
            for key in ("hit_rate_at_10", "mrr", "mttc")
        }
    return result


def _metric_delta(current: object, baseline: object) -> float | None:
    current_number = _finite_number_or_none(current)
    baseline_number = _finite_number_or_none(baseline)
    if current_number is None or baseline_number is None:
        return None
    return round(current_number - baseline_number, 8)


def _count_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): _nonnegative_int(item)
        for key, item in sorted(value.items(), key=lambda item: str(item[0]))
    }


def _nonnegative_int(value: object) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, number)


def _finite_number(value: object) -> float:
    number = _finite_number_or_none(value)
    return 0.0 if number is None else round(number, 8)


def _finite_number_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _sample_hash(sample: Mapping[str, object]) -> str:
    sample_id = str(sample.get("sample_id", ""))
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()


def _validate_time_budget(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("time budget must be a positive finite number")
    try:
        budget = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("time budget must be a positive finite number") from error
    if not math.isfinite(budget) or budget <= 0.0:
        raise ValueError("time budget must be a positive finite number")
    return budget


def _validate_input_path(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {path}")
    return path


def _same_file_or_path(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    return left.exists() and os.path.samefile(left, right)


def _elapsed(clock: Callable[[], float], started_at: float) -> float:
    return max(0.0, clock() - started_at)


def _deadline_enforcement_mode() -> str:
    if (
        hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
        and hasattr(signal, "ITIMER_REAL")
    ):
        return "process_sigalrm"
    return "external_watchdog_required"


class _ProcessDeadline:
    def __init__(self, seconds: float, clock: Callable[[], float]) -> None:
        self._seconds = seconds
        self._clock = clock
        self._started_at = 0.0
        self._previous_handler: Any = None
        self._previous_timer: tuple[float, float] | None = None
        self._alarm_enabled = False

    def __enter__(self) -> "_ProcessDeadline":
        self._started_at = self._clock()
        if _deadline_enforcement_mode() == "process_sigalrm":
            self._previous_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, self._on_alarm)
            self._previous_timer = signal.setitimer(signal.ITIMER_REAL, self._seconds)
            self._alarm_enabled = True
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        if self._alarm_enabled:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, self._previous_handler)
            if self._previous_timer is not None:
                signal.setitimer(signal.ITIMER_REAL, *self._previous_timer)
        return False

    def check(self) -> None:
        if self._clock() - self._started_at >= self._seconds:
            raise TimeBudgetExceeded("comparison time budget exceeded")

    @staticmethod
    def _on_alarm(signum: int, frame: object) -> None:
        del signum, frame
        raise TimeBudgetExceeded("comparison time budget exceeded")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bounded Legacy-versus-Hybrid question-policy comparison"
    )
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--time-budget-seconds", type=float, default=1200.0)
    args = parser.parse_args(argv)
    report = run_comparison(
        args.catalog,
        args.dataset,
        args.seed,
        args.time_budget_seconds,
        output_path=args.output,
    )
    print(json.dumps({"status": report["status"], "output": str(args.output)}, sort_keys=True))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
