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
MODERATE_LATENCY_BUDGET_MS = 3000.0
CATALOG_STABILITY_GATE_VERSION = "catalog_stability_minimum_v1"
CATALOG_STABILITY_MINIMUM = 0.80
REPO_ROOT = Path(__file__).resolve().parents[1]
PINNED_BASE_CONFIG = REPO_ROOT / "config" / "default.json"
RECOGNIZER_PATH = "agent/dialogue/recognizers"
EVALUATION_ENV = {"LLM_PROVIDER": "none", "SKIP_DATA_VERIFY": "1"}
REQUIRED_RECOGNIZER_BASE_SHA = "80e148029c95c55f59d69f3dcc57f582e582b304"


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
    marginal_totals = {
        field: Counter(str(sample[field]) for sample in samples)
        for field in ("scenario_type", "coarse_category", "initial_candidate_bin")
    }
    group_records = [(target, rows) for target, rows in grouped.items()]
    rng = random.Random(seed)
    tie = {target: rng.random() for target, _ in group_records}
    group_records.sort(
        key=lambda item: (
            -len(item[1]),
            min(
                marginal_totals[field][str(row[field])]
                for field in marginal_totals
                for row in item[1]
            ),
            tie[item[0]],
            item[0],
        )
    )
    folds: list[list[dict[str, Any]]] = [[] for _ in range(fold_count)]
    fold_sizes = [0] * fold_count
    fold_marginals = {field: [Counter() for _ in range(fold_count)] for field in marginal_totals}
    for _, rows in group_records:
        group_marginals = {
            field: Counter(str(row[field]) for row in rows) for field in marginal_totals
        }

        def cost(
            index: int,
            group_rows: Sequence[dict[str, Any]] = rows,
            group_counts: Mapping[str, Counter[str]] = group_marginals,
        ) -> tuple[float, int]:
            # Score the entire hypothetical assignment, not just the candidate
            # fold.  Candidate-fold-only distance can keep selecting a locally
            # attractive fold and strand another fold empty.
            hypothetical_sizes = list(fold_sizes)
            hypothetical_sizes[index] += len(group_rows)
            size_target = len(samples) / fold_count
            size_cost = sum(
                ((size - size_target) / max(size_target, 1)) ** 2
                for size in hypothetical_sizes
            )
            marginal_cost = 0.0
            for field, weight in (
                ("scenario_type", 5.0),
                ("coarse_category", 2.0),
                ("initial_candidate_bin", 1.0),
            ):
                marginal_cost += weight * sum(
                    (
                        (
                            fold_marginals[field][fold_index][value]
                            + (group_counts[field][value] if fold_index == index else 0)
                            - marginal_totals[field][value] / fold_count
                        )
                        / max(marginal_totals[field][value] / fold_count, 1)
                    )
                    ** 2
                    for fold_index in range(fold_count)
                    for value in marginal_totals[field]
                )
            return (marginal_cost + size_cost, index)

        chosen = min(range(fold_count), key=cost)
        folds[chosen].extend(rows)
        fold_sizes[chosen] += len(rows)
        for field, counts in group_marginals.items():
            fold_marginals[field][chosen].update(counts)
    return folds


def annotate_samples(
    samples: Sequence[dict[str, Any]], categories: Mapping[str, list[str]]
) -> list[dict[str, Any]]:
    """Add catalog-population candidate proxies without inspecting outcomes."""
    category_by_target = {
        _target(row): coarse_category(categories.get(_target(row), [])) for row in samples
    }
    population = Counter(coarse_category(values) for values in categories.values())
    annotated: list[dict[str, Any]] = []
    for sample in samples:
        category = category_by_target[_target(sample)]
        count = population[category]
        bin_name = "small" if count <= 10 else "medium" if count <= 100 else "large"
        annotated.append(
            {
                **sample,
                "coarse_category": category,
                "initial_candidate_population": count,
                "initial_candidate_bin": bin_name,
            }
        )
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


def build_search_manifest(search_space: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """Sample the source space and account for every sampled configuration."""
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
    configs: list[dict[str, Any]] = [
        {
            "id": LEGACY_ID,
            "kind": "legacy",
            "overlay": {},
            "depth": 1,
            "status": "pending_catalog_gate",
            "reason": "legacy_baseline",
        }
    ]
    for sampling_rank, values in enumerate(sampled):
        overlay = _expand_overlay(values)
        item = {
            "id": hashlib.sha256(canonical_json(overlay).encode()).hexdigest()[:12],
            "kind": "coarse",
            "overlay": overlay,
            "depth": values["lookahead_depth"],
        }
        configs.append(
            {
                **item,
                "sampling_rank": sampling_rank,
                "status": (
                    "excluded_depth_two_known_residual"
                    if values["lookahead_depth"] == 2
                    else "pending_catalog_gate"
                ),
                "reason": DEPTH_TWO_EXCLUSION if values["lookahead_depth"] == 2 else None,
            }
        )
    return {
        "sampling_seed": seed,
        "source_space": dict(search_space),
        "source_combination_count": len(source),
        "coarse_sample_count": len(sampled),
        "configs": configs,
        "accounted_config_count": len(configs),
    }


def _catalog_measurements(catalog_report: Mapping[str, Any], pool_size: int) -> tuple[float, float]:
    pool = catalog_report.get("pool_sizes", {}).get(str(pool_size), {})
    return (
        float(pool.get("latency_ms", {}).get("p95", math.inf)),
        float(pool.get("candidate_deletion_stability", {}).get("choice_agreement", 0.0)),
    )


def prepare_manifest_for_evaluation(
    search_space: Mapping[str, Any],
    catalog_report: Mapping[str, Any],
    *,
    seed: int,
    max_depth_one: int,
) -> dict[str, Any]:
    """Apply only predeclared catalog gates and the moderate deterministic cap."""
    if max_depth_one < 1 or max_depth_one > 8:
        raise ValueError("max_depth_one must be between 1 and 8 including legacy")
    manifest = build_search_manifest(search_space, seed)
    eligible: list[dict[str, Any]] = []
    for item in manifest["configs"]:
        if item["id"] == LEGACY_ID:
            item.update(
                status="evaluated_legacy",
                reason="legacy_baseline",
                latency_p95_ms=0.0,
                catalog_stability=1.0,
                complexity=0,
            )
            continue
        if item["status"] != "pending_catalog_gate":
            continue
        pool_size = int(item["overlay"]["decision"]["candidate_question_value"]["pool_size"])
        latency, stability = _catalog_measurements(catalog_report, pool_size)
        item.update(
            latency_p95_ms=latency,
            catalog_stability=stability,
            complexity=_complexity(item["overlay"]),
        )
        if latency > MODERATE_LATENCY_BUDGET_MS:
            item.update(
                status="preexcluded_latency_budget",
                reason="production_latency_p95_exceeds_3000ms",
            )
        elif stability < CATALOG_STABILITY_MINIMUM:
            item.update(
                status="preexcluded_catalog_stability",
                reason=f"{CATALOG_STABILITY_GATE_VERSION}_below_0.80",
            )
        else:
            eligible.append(item)
    eligible.sort(key=lambda item: int(item["sampling_rank"]))
    for index, item in enumerate(eligible):
        item.update(
            status="evaluated_coarse"
            if index < max_depth_one - 1
            else "moderate_budget_not_selected",
            reason="moderate_global_depth_one_cap" if index >= max_depth_one - 1 else "selected",
        )
    manifest["accounted_config_count"] = len(manifest["configs"])
    return manifest


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
    *,
    hr10_tolerance: float,
    scenario_tolerance: float,
    latency_budget_ms: float,
) -> list[str]:
    reasons: list[str] = []
    regressions = sum(delta < -hr10_tolerance for delta in candidate["outer_fold_hr10_deltas"])
    allowed = max(1, len(candidate["outer_fold_hr10_deltas"]) - 2)
    if regressions > allowed:
        reasons.append("outer_fold_hr10_regression")
    if any(delta < -scenario_tolerance for delta in candidate["scenario_hr10_deltas"].values()):
        reasons.append("scenario_collapse")
    if float(candidate["catalog_stability"]) < CATALOG_STABILITY_MINIMUM:
        reasons.append("catalog_stability_minimum_not_met")
    if float(candidate["latency_p95_ms"]) > latency_budget_ms:
        reasons.append("latency_budget_exceeded")
    return reasons


def select_one_standard_error(configs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not configs:
        raise ValueError("one-standard-error selection needs an eligible configuration")
    best_mean = max(float(item["mean"]) for item in configs)
    best = min(
        (item for item in configs if float(item["mean"]) == best_mean),
        key=lambda item: str(item["canonical_json"]),
    )
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
        return {
            "mean": 0.0,
            "hr10": 0.0,
            "scenario_hr10": {},
            "contributions": {},
            "official_components": {
                "hit_rate_at_10": 0.0,
                "mrr": 0.0,
                "mttc": None,
                "efficiency": 0.0,
                "recommended_technical_score": 0.0,
            },
        }
    by_scenario: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for session in sessions:
        by_scenario[str(session["scenario_type"])].append(session)
    hr10 = round(statistics.fmean(float(bool(session["hit"])) for session in sessions), 6)
    mrr = round(statistics.fmean(float(session["reciprocal_rank"]) for session in sessions), 6)
    mttc = round(
        statistics.fmean(
            float(session["first_hit_turn"] if session["first_hit_turn"] is not None else 11)
            for session in sessions
        ),
        6,
    )
    efficiency = round(max(0.0, min(1.0, (11.0 - mttc) / 10.0)), 6)
    technical = round(0.50 * hr10 + 0.30 * mrr + 0.20 * efficiency, 6)
    return {
        "mean": technical,
        "hr10": hr10,
        "scenario_hr10": {
            name: statistics.fmean(float(bool(item["hit"])) for item in rows)
            for name, rows in sorted(by_scenario.items())
        },
        "contributions": {
            str(session["sample_id"]): _score_contribution(session) for session in sessions
        },
        "official_components": {
            "hit_rate_at_10": hr10,
            "mrr": mrr,
            "mttc": mttc,
            "efficiency": efficiency,
            "recommended_technical_score": technical,
        },
    }


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
    evaluation_overlay = _deep_merge(
        {"llm": {"provider": "none"}, "skip_data_verify": True}, overlay
    )
    effective_document = complete_config_document(PINNED_BASE_CONFIG, evaluation_overlay)
    env = EnvConfig.from_env(
        path=PINNED_BASE_CONFIG,
        overrides=evaluation_overlay,
        environ=EVALUATION_ENV,
    )
    result = evaluate(
        Agent(catalog_path=catalog, env=env), list(samples), catalog_ids, categories, products
    )
    return {
        "official": {key: value for key, value in result.items() if key != "sessions"},
        "sessions": result["sessions"],
        "effective_config_sha256": hashlib.sha256(
            canonical_json(effective_document).encode("utf-8")
        ).hexdigest(),
    }


def _run_nested_procedure_audit(
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
    return outer_reports, {"selected_ids": selected_ids}


def _final_training_cv(
    configs: Sequence[dict[str, Any]],
    outcomes: Mapping[str, Mapping[str, Any]],
    samples: Sequence[dict[str, Any]],
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """Choose the deployment ID using all public rows, never outer-test outcomes."""
    folds = grouped_stratified_folds(samples, 3, seed ^ 0xF17)
    by_id = {
        identifier: {str(row["sample_id"]): row for row in value["sessions"]}
        for identifier, value in outcomes.items()
    }
    legacy_by_id = by_id[LEGACY_ID]
    legacy_summary = _session_summary(list(legacy_by_id.values()))
    rows: list[dict[str, Any]] = []
    for config in configs:
        sessions = list(by_id[config["id"]].values())
        summary = _session_summary(sessions)
        fold_summaries = [
            _session_summary([by_id[config["id"]][str(row["sample_id"])] for row in fold])
            for fold in folds
        ]
        legacy_fold_summaries = [
            _session_summary([legacy_by_id[str(row["sample_id"])] for row in fold])
            for fold in folds
        ]
        fold_means = [summary["mean"] for summary in fold_summaries]
        scenario_hr10_deltas = {
            name: value - legacy_summary["scenario_hr10"].get(name, 0.0)
            for name, value in summary["scenario_hr10"].items()
        }
        row = {
            **config,
            **summary,
            "training_cv_fold_metrics": [
                {
                    "fold": index,
                    "hr10_delta_vs_legacy": fold_summary["hr10"] - legacy_summary["hr10"],
                    **fold_summary["official_components"],
                }
                for index, (fold_summary, legacy_summary) in enumerate(
                    zip(fold_summaries, legacy_fold_summaries, strict=True)
                )
            ],
            "training_cv_fold_hr10_deltas": [
                fold_summary["hr10"] - legacy_summary["hr10"]
                for fold_summary, legacy_summary in zip(
                    fold_summaries, legacy_fold_summaries, strict=True
                )
            ],
            "scenario_hr10_deltas": scenario_hr10_deltas,
            "standard_error": statistics.stdev(fold_means) / math.sqrt(len(fold_means)),
            "canonical_json": canonical_json(config["overlay"]),
        }
        row["selection_eligibility_reasons"] = eligibility_reasons(
            {
                **row,
                "outer_fold_hr10_deltas": row["training_cv_fold_hr10_deltas"],
            },
            hr10_tolerance=0.005,
            scenario_tolerance=0.02,
            latency_budget_ms=MODERATE_LATENCY_BUDGET_MS,
        )
        row["selection_eligible"] = not row["selection_eligibility_reasons"]
        rows.append(row)
    eligible_rows = [row for row in rows if row["selection_eligible"]]
    return select_one_standard_error(eligible_rows), rows, folds


def _outer_audit_eligibility(
    configs: Sequence[dict[str, Any]],
    outcomes: Mapping[str, Mapping[str, Any]],
    folds: Sequence[Sequence[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Report outer eligibility after final selection has already been frozen."""
    by_id = {
        identifier: {str(row["sample_id"]): row for row in value["sessions"]}
        for identifier, value in outcomes.items()
    }
    legacy_by_id = by_id[LEGACY_ID]
    legacy_summary = _session_summary(list(legacy_by_id.values()))
    reports: list[dict[str, Any]] = []
    for config in configs:
        candidate_by_id = by_id[config["id"]]
        fold_deltas = [
            _session_summary([candidate_by_id[str(row["sample_id"])] for row in fold])["hr10"]
            - _session_summary([legacy_by_id[str(row["sample_id"])] for row in fold])["hr10"]
            for fold in folds
        ]
        candidate_summary = _session_summary(list(candidate_by_id.values()))
        candidate = {
            **config,
            "outer_fold_hr10_deltas": fold_deltas,
            "scenario_hr10_deltas": {
                name: value - legacy_summary["scenario_hr10"].get(name, 0.0)
                for name, value in candidate_summary["scenario_hr10"].items()
            },
        }
        reasons = eligibility_reasons(
            candidate,
            hr10_tolerance=0.005,
            scenario_tolerance=0.02,
            latency_budget_ms=MODERATE_LATENCY_BUDGET_MS,
        )
        reports.append(
            {
                "id": config["id"],
                "outer_fold_hr10_deltas": fold_deltas,
                "scenario_hr10_deltas_session_weighted": candidate["scenario_hr10_deltas"],
                "eligibility_reasons": reasons,
                "eligible": not reasons,
            }
        )
    return reports


def _commit_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _is_dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        return True


def resolve_and_verify_recognizer_base_sha(value: str) -> str:
    try:
        base = subprocess.check_output(
            ["git", "rev-parse", "--verify", f"{value}^{{commit}}"], text=True
        ).strip()
        if base != REQUIRED_RECOGNIZER_BASE_SHA:
            raise ValueError("required recognizer base SHA is 80e1480")
        subprocess.run(["git", "diff", "--quiet", base, "--", RECOGNIZER_PATH], check=True)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        raise ValueError("recognizer files differ from required recognizer base SHA") from error
    return base


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
    recognizer_base_sha: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if outer_folds != 3 or inner_folds != 2:
        raise ValueError("moderate policy requires exactly 3 outer and 2 inner folds")
    if max_depth_one > 8 or max_refinements > 2:
        raise ValueError("moderate policy allows at most 8 depth-one and 2 refinement evaluations")
    recognizer_base = resolve_and_verify_recognizer_base_sha(recognizer_base_sha)
    producer = {"commit": _commit_sha(), "dirty": _is_dirty()}
    space = json.loads(search_space.read_text(encoding="utf-8"))
    report = json.loads(catalog_report.read_text(encoding="utf-8"))
    manifest = prepare_manifest_for_evaluation(
        space, report, seed=seed, max_depth_one=max_depth_one
    )
    samples = load_jsonl(dataset)
    catalog_ids, categories, products = catalog_index(catalog)
    annotated = annotate_samples(samples, categories)
    outer_grouped_folds = grouped_stratified_folds(annotated, outer_folds, seed)
    runnable = [
        config
        for config in manifest["configs"]
        if config["status"] in {"evaluated_legacy", "evaluated_coarse"}
    ]
    outcomes: dict[str, dict[str, Any]] = {}
    for config in runnable:
        outcomes[config["id"]] = _evaluate_once(
            catalog, annotated, catalog_ids, categories, products, config["overlay"]
        )
        config["effective_config_sha256"] = outcomes[config["id"]]["effective_config_sha256"]
    # Refinements are predeclared from catalog-gated coarse source order, never
    # derived from outer tests or from their held-out outcomes.
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
            item.update(status="evaluated_refinement", reason="predeclared_adjacent_prior_alpha")
            manifest["configs"].append(item)
            outcomes[item["id"]] = _evaluate_once(
                catalog, annotated, catalog_ids, categories, products, overlay
            )
            item["effective_config_sha256"] = outcomes[item["id"]]["effective_config_sha256"]
            runnable.append(item)
        else:
            item.update(status="refinement_duplicate", reason="same_canonical_config_as_coarse")
            manifest["configs"].append(item)
    all_configs = runnable
    outer_reports, nested = _run_nested_procedure_audit(
        all_configs, outcomes, outer_grouped_folds, seed
    )
    final_config, candidate_reports, final_fit_folds = _final_training_cv(
        all_configs, outcomes, annotated, seed
    )
    # The final ID is frozen by all-public training CV before any outer-test
    # eligibility evidence is computed or reported.
    outer_audit_eligibility = _outer_audit_eligibility(
        all_configs, outcomes, outer_grouped_folds
    )
    baseline_by_id = {
        str(session["sample_id"]): session for session in outcomes[LEGACY_ID]["sessions"]
    }
    baseline_summary = _session_summary(list(baseline_by_id.values()))
    for candidate_report in candidate_reports:
        candidate_report["effective_config_sha256"] = outcomes[candidate_report["id"]][
            "effective_config_sha256"
        ]
        candidate_report["scenario_hr10_deltas_session_weighted"] = {
            name: value - baseline_summary["scenario_hr10"].get(name, 0.0)
            for name, value in candidate_report["scenario_hr10"].items()
        }
    complete = complete_config_document(PINNED_BASE_CONFIG, final_config["overlay"])
    load_config(
        path=PINNED_BASE_CONFIG,
        overrides=final_config["overlay"],
        environ=EVALUATION_ENV,
    )
    manifest["accounted_config_count"] = len(manifest["configs"])
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
            "fixed_production_latency_budget_ms": MODERATE_LATENCY_BUDGET_MS,
            "catalog_stability_gate": {
                "version": CATALOG_STABILITY_GATE_VERSION,
                "minimum": CATALOG_STABILITY_MINIMUM,
            },
            "recognizer": {
                "llm_provider": "none",
                "base_sha": recognizer_base,
                "files_unchanged_since_base": True,
            },
            "producer": producer,
            "pinned_base_config": str(PINNED_BASE_CONFIG),
            "evaluation_environ": EVALUATION_ENV,
        },
        "procedure_audit": {
            "purpose": (
                "nested-procedure held-out evidence only; audit selections frozen before "
                "final fit and per-candidate eligibility reported read-only after final ID freeze"
            ),
            "outer_fold_manifest": [
                [str(row["sample_id"]) for row in fold] for fold in outer_grouped_folds
            ],
            "outer_results": outer_reports,
            "selected_ids_by_outer_fold": nested["selected_ids"],
            "final_fit_selected_config_id_frozen": final_config["id"],
            "per_candidate_eligibility_read_only": outer_audit_eligibility,
        },
        "final_fit": {
            "purpose": (
                "all-public-data deterministic grouped training-CV selection with "
                "predeclared final-fit eligibility gates"
            ),
            "fold_manifest": [[str(row["sample_id"]) for row in fold] for fold in final_fit_folds],
            "configurations": candidate_reports,
            "selected_config_id": final_config["id"],
        },
        "search_manifest": {
            **manifest,
            "actually_evaluated_count": len(all_configs),
        },
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
            "config_id": final_config["id"],
            "overlay": final_config["overlay"],
            "selection": (
                "final_fit_predeclared_eligibility_then_one_standard_error_then_complexity_"
                "then_latency_then_canonical_json"
            ),
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
    parser.add_argument("--recognizer-base-sha", required=True)
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
        recognizer_base_sha=args.recognizer_base_sha,
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
