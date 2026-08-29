"""Bounded, grouped nested validation for global dialogue decision settings.

The official evaluator is intentionally left untouched.  Each candidate is run
once over the public sessions and the documented per-session technical-score
formula is then aggregated into deterministic grouped folds.  This is both much
less expensive than re-running every inner fold and preserves nesting: an outer
test fold is never included in that outer iteration's inner selection.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import statistics
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agent.main_agent import Agent
from config.env_config import EnvConfig
from config.loader import load_config
from evaluator.local_evaluator import catalog_index, coarse_category, evaluate, load_jsonl

LEGACY_ID = "legacy"
DEPTH_TWO_EXCLUSION = "known_depth_two_gate_mismatch"


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _deep_merge(base: Mapping[str, Any], changes: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = dict(base)
    for key, value in changes.items():
        current = result.get(key)
        result[key] = (
            _deep_merge(current, value)
            if isinstance(current, Mapping) and isinstance(value, Mapping)
            else value
        )
    return result


def _target(sample: Mapping[str, Any]) -> str:
    return str(sample["ground_truth"]["parent_asin"])


def _stratum(sample: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(sample.get("scenario_type", "unknown")),
        str(sample.get("coarse_category", "unknown")),
        str(sample.get("initial_candidate_bin", "unknown")),
    )


def grouped_stratified_folds(
    samples: Sequence[dict[str, Any]], fold_count: int, seed: int
) -> list[list[dict[str, Any]]]:
    """Assign complete target-ASIN groups with deterministic greedy balancing."""
    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[_target(sample)].append(sample)
    if len(grouped) < fold_count:
        raise ValueError("fold_count cannot exceed unique target ASIN groups")
    total_strata = Counter(_stratum(sample) for sample in samples)
    group_records: list[tuple[str, list[dict[str, Any]], Counter[tuple[str, str, str]]]] = [
        (target, rows, Counter(_stratum(row) for row in rows)) for target, rows in grouped.items()
    ]
    rng = random.Random(seed)
    tie = {target: rng.random() for target, _, _ in group_records}
    group_records.sort(
        key=lambda item: (
            -len(item[1]),
            min(total_strata[stratum] for stratum in item[2]),
            tie[item[0]],
            item[0],
        )
    )
    folds: list[list[dict[str, Any]]] = [[] for _ in range(fold_count)]
    fold_sizes = [0] * fold_count
    fold_strata: list[Counter[tuple[str, str, str]]] = [Counter() for _ in range(fold_count)]
    for _, rows, strata in group_records:

        def cost(
            index: int,
            group_rows: Sequence[dict[str, Any]] = rows,
            group_strata: Counter[tuple[str, str, str]] = strata,
        ) -> tuple[float, int]:
            # During greedy construction, penalize already-full folds rather than
            # distance from the final target; the latter ties every empty fold.
            size_cost = (fold_sizes[index] + len(group_rows)) / max(len(samples) / fold_count, 1)
            stratum_cost = sum(
                ((fold_strata[index][key] + count) / max(total_strata[key] / fold_count, 1)) ** 2
                for key, count in group_strata.items()
            )
            return (4.0 * stratum_cost + size_cost, index)

        chosen = min(range(fold_count), key=cost)
        folds[chosen].extend(rows)
        fold_sizes[chosen] += len(rows)
        fold_strata[chosen].update(strata)
    return folds


def annotate_samples(
    samples: Sequence[dict[str, Any]], categories: Mapping[str, list[str]]
) -> list[dict[str, Any]]:
    """Add public grouping proxies without inspecting evaluator outcomes."""
    category_by_target = {
        _target(row): coarse_category(categories.get(_target(row), [])) for row in samples
    }
    population = Counter(category_by_target.values())
    annotated: list[dict[str, Any]] = []
    for sample in samples:
        category = category_by_target[_target(sample)]
        count = population[category]
        bin_name = "small" if count <= 3 else "medium" if count <= 10 else "large"
        annotated.append({**sample, "coarse_category": category, "initial_candidate_bin": bin_name})
    return annotated


def _expand_overlay(values: Mapping[str, Any]) -> dict[str, Any]:
    guard = values["transition_guard_profile"]
    candidate_weights = values["candidate_weight_profile"]
    finish_weights = values["finish_weight_profile"]
    return {
        "dialogue_understanding": {
            "mode": "rule_only",
            "transition_guard": {
                "enabled": bool(guard["enabled"]),
                "add_min_confidence": guard.get("add", 0.65),
                "replace_min_confidence": guard.get("replace", 0.90),
                "remove_min_confidence": guard.get("remove", 0.90),
                "reject_products_min_confidence": guard.get("reject_products", 0.90),
                "no_preference_min_confidence": guard.get("no_preference", 0.85),
                "no_more_preferences_min_confidence": guard.get("no_more_preferences", 0.95),
            },
        },
        "decision": {
            "candidate_question_value": {
                "enabled": True,
                "pool_size": values["pool_size"],
                "prior_alpha": values["prior_alpha"],
                "prior_temperature": values["prior_temperature"],
                "other_answer_probability": values["other_answer_probability"],
                "other_vagueness_penalty": values["other_vagueness_penalty"],
                "weights": candidate_weights,
            },
            "finish_strategy": {
                "enabled": True,
                "candidate_threshold": values["finish_candidate_threshold"],
                "remaining_question_threshold": values["remaining_question_threshold"],
                "lookahead_depth": values["lookahead_depth"],
                "weights": finish_weights,
            },
            "question_termination_mode": values["termination_mode"],
        },
        "llm": {"provider": "none"},
    }


def build_search_manifest(
    search_space: Mapping[str, Any], seed: int, max_depth_one: int = 8
) -> dict[str, Any]:
    """Sample a recorded coarse space; depth two remains recorded, never run."""
    if max_depth_one < 0:
        raise ValueError("max_depth_one must be non-negative")
    keys = (
        "transition_guard_profile",
        "candidate_weight_profile",
        "finish_weight_profile",
        "pool_size",
        "prior_alpha",
        "prior_temperature",
        "other_answer_probability",
        "other_vagueness_penalty",
        "finish_candidate_threshold",
        "remaining_question_threshold",
        "lookahead_depth",
        "termination_mode",
    )
    for key in keys:
        if not isinstance(search_space.get(key), list) or not search_space[key]:
            raise ValueError(f"search space requires non-empty {key}")
    source = [
        dict(zip(keys, row, strict=True))
        for row in itertools.product(*(search_space[key] for key in keys))
    ]
    sample_limit = int(search_space.get("coarse_sample_limit", 50))
    if sample_limit < 1 or sample_limit > 50:
        raise ValueError("coarse_sample_limit must be between 1 and 50")
    sampled = (
        source if len(source) <= sample_limit else random.Random(seed).sample(source, sample_limit)
    )
    sampled.sort(key=canonical_json)
    excluded: list[dict[str, Any]] = []
    runnable: list[dict[str, Any]] = []
    for values in sampled:
        overlay = _expand_overlay(values)
        item = {
            "id": hashlib.sha256(canonical_json(overlay).encode()).hexdigest()[:12],
            "kind": "coarse",
            "overlay": overlay,
            "depth": values["lookahead_depth"],
        }
        if values["lookahead_depth"] == 2:
            excluded.append({**item, "exclusion_reason": DEPTH_TWO_EXCLUSION})
        else:
            runnable.append(item)
    rng = random.Random(seed ^ 0xC0FFEE)
    rng.shuffle(runnable)
    evaluated = [
        {"id": LEGACY_ID, "kind": "legacy", "overlay": {}, "depth": 1},
        *runnable[:max_depth_one],
    ]
    return {
        "sampling_seed": seed,
        "source_space": dict(search_space),
        "source_combination_count": len(source),
        "coarse_sample_count": len(sampled),
        "evaluated": evaluated,
        "excluded": excluded,
    }


def paired_bootstrap_interval(
    candidate: Sequence[float], baseline: Sequence[float], iterations: int, seed: int
) -> dict[str, float]:
    if len(candidate) != len(baseline) or not candidate or iterations < 1:
        raise ValueError(
            "paired contributions must be non-empty, equal length, and use iterations >= 1"
        )
    deltas = [left - right for left, right in zip(candidate, baseline, strict=True)]
    rng = random.Random(seed)
    means = sorted(
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(iterations)
    )

    def quantile(p: float) -> float:
        return means[min(len(means) - 1, max(0, round((len(means) - 1) * p)))]

    return {
        "mean_delta": sum(deltas) / len(deltas),
        "lower_95": quantile(0.025),
        "upper_95": quantile(0.975),
    }


def eligibility_reasons(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    hr10_tolerance: float,
    scenario_tolerance: float,
    catalog_tolerance: float,
    latency_budget_ms: float,
) -> list[str]:
    reasons: list[str] = []
    regressions = sum(delta < -hr10_tolerance for delta in candidate["outer_fold_hr10_deltas"])
    allowed = max(1, len(candidate["outer_fold_hr10_deltas"]) - 2)
    if regressions > allowed:
        reasons.append("outer_fold_hr10_regression")
    if any(delta < -scenario_tolerance for delta in candidate["scenario_hr10_deltas"].values()):
        reasons.append("scenario_collapse")
    if float(candidate["catalog_stability_delta"]) < -catalog_tolerance:
        reasons.append("catalog_stability_regression")
    if float(candidate["latency_p95_ms"]) > latency_budget_ms:
        reasons.append("latency_budget_exceeded")
    return reasons


def select_one_standard_error(configs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not configs:
        raise ValueError("one-standard-error selection needs an eligible configuration")
    best = max(configs, key=lambda item: float(item["mean"]))
    cutoff = float(best["mean"]) - float(best["standard_error"])
    within = [item for item in configs if float(item["mean"]) >= cutoff]
    return min(
        within,
        key=lambda item: (
            int(item["complexity"]),
            float(item["latency_p95_ms"]),
            str(item["canonical_json"]),
        ),
    )


def complete_config_document(
    base_config_path: Path | str, overlay: Mapping[str, Any]
) -> dict[str, Any]:
    base = json.loads(Path(base_config_path).read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ValueError("base config must be a JSON object")
    return _deep_merge(base, overlay)


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except FileNotFoundError:
        return left.resolve() == right.resolve()


def write_json_atomic(
    payload: Mapping[str, Any], output: Path | str, *, protected_inputs: Sequence[Path | str] = ()
) -> None:
    target = Path(output)
    if any(_same_file(target, Path(protected)) for protected in protected_inputs):
        raise ValueError("output must not collide with a protected input")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(target)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _score_contribution(session: Mapping[str, Any]) -> float:
    first_hit_turn = session.get("first_hit_turn")
    turn = 11 if first_hit_turn is None else int(first_hit_turn)
    return (
        0.50 * float(bool(session.get("hit")))
        + 0.30 * float(session["reciprocal_rank"])
        + 0.20 * ((11.0 - turn) / 10.0)
    )


def _session_summary(sessions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not sessions:
        return {"mean": 0.0, "hr10": 0.0, "scenario_hr10": {}, "contributions": {}}
    by_scenario: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for session in sessions:
        by_scenario[str(session["scenario_type"])].append(session)
    return {
        "mean": statistics.fmean(_score_contribution(session) for session in sessions),
        "hr10": statistics.fmean(float(bool(session["hit"])) for session in sessions),
        "scenario_hr10": {
            name: statistics.fmean(float(bool(item["hit"])) for item in rows)
            for name, rows in sorted(by_scenario.items())
        },
        "contributions": {
            str(session["sample_id"]): _score_contribution(session) for session in sessions
        },
    }


def _catalog_gate(catalog_report: Mapping[str, Any], pool_size: int) -> tuple[float, float, float]:
    pool = catalog_report.get("pool_sizes", {}).get(str(pool_size), {})
    latency = float(pool.get("latency_ms", {}).get("p95", math.inf))
    stability = float(pool.get("candidate_deletion_stability", {}).get("choice_agreement", 0.0))
    reference_pool = str(min(int(key) for key in catalog_report.get("pool_sizes", {"300": {}})))
    budget = float(catalog_report["pool_sizes"][reference_pool]["latency_ms"]["p95"])
    return latency, stability, budget


def _complexity(overlay: Mapping[str, Any]) -> int:
    if not overlay:
        return 0
    decision = overlay.get("decision", {})
    return (
        int(bool(decision.get("candidate_question_value", {}).get("enabled")))
        + int(bool(decision.get("finish_strategy", {}).get("enabled")))
        + int(
            bool(
                overlay.get("dialogue_understanding", {}).get("transition_guard", {}).get("enabled")
            )
        )
    )


def _evaluate_once(
    catalog: Path,
    samples: Sequence[dict[str, Any]],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    overlay: Mapping[str, Any],
) -> dict[str, Any]:
    env = EnvConfig.from_env(
        overrides=_deep_merge({"llm": {"provider": "none"}, "skip_data_verify": True}, overlay)
    )
    result = evaluate(
        Agent(catalog_path=catalog, env=env), list(samples), catalog_ids, categories, products
    )
    return {
        "official": {key: value for key, value in result.items() if key != "sessions"},
        "sessions": result["sessions"],
    }


def _run_nested(
    configs: Sequence[dict[str, Any]],
    outcomes: Mapping[str, Mapping[str, Any]],
    folds: Sequence[Sequence[dict[str, Any]]],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {
        identifier: {str(row["sample_id"]): row for row in value["sessions"]}
        for identifier, value in outcomes.items()
    }
    outer_reports: list[dict[str, Any]] = []
    selected_ids: list[str] = []
    for outer_index, test_rows in enumerate(folds):
        test_ids = {str(row["sample_id"]) for row in test_rows}
        train_rows = [row for fold in folds if fold is not test_rows for row in fold]
        inner_folds = grouped_stratified_folds(train_rows, 2, seed + outer_index + 1)
        train_ids = {str(row["sample_id"]) for fold in inner_folds for row in fold}
        selection_rows: list[dict[str, Any]] = []
        for config in configs:
            sessions = [by_id[config["id"]][sample_id] for sample_id in train_ids]
            summary = _session_summary(sessions)
            fold_means = [
                _session_summary([by_id[config["id"]][str(row["sample_id"])] for row in fold])[
                    "mean"
                ]
                for fold in inner_folds
            ]
            selection_rows.append(
                {
                    **config,
                    **summary,
                    "standard_error": statistics.stdev(fold_means) / math.sqrt(len(fold_means))
                    if len(fold_means) > 1
                    else 0.0,
                    "canonical_json": canonical_json(config["overlay"]),
                }
            )
        chosen = select_one_standard_error(selection_rows)
        selected_ids.append(str(chosen["id"]))
        candidate_test = _session_summary(
            [by_id[chosen["id"]][sample_id] for sample_id in test_ids]
        )
        legacy_test = _session_summary([by_id[LEGACY_ID][sample_id] for sample_id in test_ids])
        outer_reports.append(
            {
                "outer_fold": outer_index,
                "selected_config_id": chosen["id"],
                "test": candidate_test,
                "legacy_test": legacy_test,
            }
        )
    final_rows: list[dict[str, Any]] = []
    all_ids = set(by_id[LEGACY_ID])
    for config in configs:
        summary = _session_summary([by_id[config["id"]][sample_id] for sample_id in all_ids])
        fold_means = [
            _session_summary([by_id[config["id"]][str(row["sample_id"])] for row in fold])["mean"]
            for fold in folds
        ]
        final_rows.append(
            {
                **config,
                **summary,
                "standard_error": statistics.stdev(fold_means) / math.sqrt(len(fold_means))
                if len(fold_means) > 1
                else 0.0,
                "canonical_json": canonical_json(config["overlay"]),
            }
        )
    return outer_reports, {"selected_ids": selected_ids, "full_training_candidates": final_rows}


def _commit_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def run_cross_validation(
    *,
    catalog: Path,
    dataset: Path,
    search_space: Path,
    catalog_report: Path,
    seed: int,
    outer_folds: int = 3,
    inner_folds: int = 2,
    max_depth_one: int = 8,
    max_refinements: int = 2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if outer_folds != 3 or inner_folds != 2:
        raise ValueError("moderate policy requires exactly 3 outer and 2 inner folds")
    if max_depth_one > 8 or max_refinements > 2:
        raise ValueError("moderate policy allows at most 8 depth-one and 2 refinement evaluations")
    space = json.loads(search_space.read_text(encoding="utf-8"))
    report = json.loads(catalog_report.read_text(encoding="utf-8"))
    manifest = build_search_manifest(space, seed, max_depth_one=max_depth_one - 1)
    samples = load_jsonl(dataset)
    catalog_ids, categories, products = catalog_index(catalog)
    annotated = annotate_samples(samples, categories)
    folds = grouped_stratified_folds(annotated, outer_folds, seed)
    latency_budget = _catalog_gate(report, 300)[2]
    runnable: list[dict[str, Any]] = []
    pre_rejected: list[dict[str, Any]] = []
    for config in manifest["evaluated"]:
        if config["id"] == LEGACY_ID:
            runnable.append(
                {**config, "latency_p95_ms": 0.0, "catalog_stability": 1.0, "complexity": 0}
            )
            continue
        pool_size = int(config["overlay"]["decision"]["candidate_question_value"]["pool_size"])
        latency, stability, _ = _catalog_gate(report, pool_size)
        prepared = {
            **config,
            "latency_p95_ms": latency,
            "catalog_stability": stability,
            "complexity": _complexity(config["overlay"]),
        }
        if latency > latency_budget:
            pre_rejected.append({**prepared, "exclusion_reason": "latency_budget_exceeded"})
        else:
            runnable.append(prepared)
    outcomes: dict[str, dict[str, Any]] = {}
    for config in runnable:
        outcomes[config["id"]] = _evaluate_once(
            catalog, annotated, catalog_ids, categories, products, config["overlay"]
        )
    # Refinements are intentionally derived from source order, not test metrics;
    # this preserves the invariant that outer test outcomes cannot grow the sweep.
    refinements: list[dict[str, Any]] = []
    for config in runnable[1 : 1 + max_refinements]:
        overlay = json.loads(json.dumps(config["overlay"]))
        alpha = float(overlay["decision"]["candidate_question_value"]["prior_alpha"])
        overlay["decision"]["candidate_question_value"]["prior_alpha"] = round(
            (alpha + 0.5) / 2.0, 6
        )
        item = {
            "id": hashlib.sha256(canonical_json(overlay).encode()).hexdigest()[:12],
            "kind": "adjacent_refinement",
            "overlay": overlay,
            "depth": 1,
            "latency_p95_ms": config["latency_p95_ms"],
            "catalog_stability": config["catalog_stability"],
            "complexity": _complexity(overlay),
        }
        if item["id"] not in outcomes:
            refinements.append(item)
            outcomes[item["id"]] = _evaluate_once(
                catalog, annotated, catalog_ids, categories, products, overlay
            )
    all_configs = [*runnable, *refinements]
    outer_reports, nested = _run_nested(all_configs, outcomes, folds, seed)
    baseline_by_id = {
        str(session["sample_id"]): session for session in outcomes[LEGACY_ID]["sessions"]
    }
    eligible: list[dict[str, Any]] = []
    candidate_reports: list[dict[str, Any]] = []
    for config in nested["full_training_candidates"]:
        candidate_by_id = {
            str(session["sample_id"]): session for session in outcomes[config["id"]]["sessions"]
        }
        outer_deltas = []
        scenario_deltas: dict[str, float] = {}
        fold_metrics: list[dict[str, float | int]] = []
        for fold_index, fold in enumerate(folds):
            candidate_summary = _session_summary(
                [candidate_by_id[str(row["sample_id"])] for row in fold]
            )
            baseline_summary = _session_summary(
                [baseline_by_id[str(row["sample_id"])] for row in fold]
            )
            outer_deltas.append(candidate_summary["hr10"] - baseline_summary["hr10"])
            fold_metrics.append(
                {
                    "fold": fold_index,
                    "technical_score": candidate_summary["mean"],
                    "hit_rate_at_10": candidate_summary["hr10"],
                    "legacy_technical_score": baseline_summary["mean"],
                    "legacy_hit_rate_at_10": baseline_summary["hr10"],
                }
            )
            for name, value in candidate_summary["scenario_hr10"].items():
                scenario_deltas.setdefault(name, 0.0)
                scenario_deltas[name] += value - baseline_summary["scenario_hr10"].get(name, 0.0)
        scenario_deltas = {name: value / outer_folds for name, value in scenario_deltas.items()}
        audit = {
            "outer_fold_hr10_deltas": outer_deltas,
            "scenario_hr10_deltas": scenario_deltas,
            "catalog_stability_delta": float(config["catalog_stability"]) - 0.80,
            "latency_p95_ms": config["latency_p95_ms"],
        }
        reasons = eligibility_reasons(
            audit,
            {"latency_p95_ms": 0.0},
            hr10_tolerance=0.005,
            scenario_tolerance=0.05,
            catalog_tolerance=0.0,
            latency_budget_ms=latency_budget,
        )
        candidate_report = {
            **config,
            "fold_metrics": fold_metrics,
            "outer_audit": audit,
            "eligibility_reasons": reasons,
        }
        candidate_reports.append(candidate_report)
        # The audit is reported but does not affect the full-training selector:
        # using outer tests for that would violate the nesting promised above.
        if not reasons or config["id"] == LEGACY_ID:
            eligible.append(config)
    recommended = select_one_standard_error(eligible)
    base_config = Path("config/default.json")
    complete = complete_config_document(base_config, recommended["overlay"])
    load_config(
        path=base_config, overrides=recommended["overlay"], environ={"LLM_PROVIDER": "none"}
    )
    report_payload = {
        "methodology": {
            "outer_folds": outer_folds,
            "inner_folds": inner_folds,
            "group_key": "ground_truth.parent_asin",
            "strata": ["scenario_type", "coarse_category", "initial_candidate_bin"],
            "outcome_reuse": (
                "Each config/session official evaluate() outcome is computed once; fold aggregates "
                "only partition those outcomes. Inner selection excludes the corresponding "
                "outer test fold."
            ),
            "recognizer": {"llm_provider": "none", "version": _commit_sha()},
        },
        "fold_manifest": [[str(row["sample_id"]) for row in fold] for fold in folds],
        "search_manifest": {
            **manifest,
            "pre_rejected": pre_rejected,
            "refinements": refinements,
            "actually_evaluated_count": len(all_configs),
        },
        "outer_evaluation": outer_reports,
        "configurations": candidate_reports,
        "paired_bootstrap": {
            config["id"]: paired_bootstrap_interval(
                list(config["contributions"].values()),
                [
                    baseline_by_id[sample_id] and _score_contribution(baseline_by_id[sample_id])
                    for sample_id in config["contributions"]
                ],
                2000,
                seed,
            )
            for config in candidate_reports
        },
        "recommendation": {
            "config_id": recommended["id"],
            "overlay": recommended["overlay"],
            "selection": "one_standard_error_then_complexity_then_latency_then_canonical_json",
        },
    }
    return report_payload, complete


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bounded grouped nested validation for dialogue decisions"
    )
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--search-space", default="experiments/decision_search_space.json")
    parser.add_argument("--catalog-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--recommended-config-output", required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--outer-folds", type=int, default=3)
    parser.add_argument("--inner-folds", type=int, default=2)
    parser.add_argument("--max-depth-one", type=int, default=8)
    parser.add_argument("--max-refinements", type=int, default=2)
    args = parser.parse_args()
    report, recommended = run_cross_validation(
        catalog=Path(args.catalog),
        dataset=Path(args.dataset),
        search_space=Path(args.search_space),
        catalog_report=Path(args.catalog_report),
        seed=args.seed,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        max_depth_one=args.max_depth_one,
        max_refinements=args.max_refinements,
    )
    protected = (args.catalog, args.dataset, args.search_space, args.catalog_report)
    write_json_atomic(
        report, args.output, protected_inputs=(*protected, args.recommended_config_output)
    )
    write_json_atomic(
        recommended, args.recommended_config_output, protected_inputs=(*protected, args.output)
    )
    print(
        json.dumps(
            {
                "recommended_config_id": report["recommendation"]["config_id"],
                "actually_evaluated_count": report["search_manifest"]["actually_evaluated_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
