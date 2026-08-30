from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path


def _samples() -> list[dict]:
    rows: list[dict] = []
    for scenario_type, count in (
        ("buying", 10),
        ("browsing", 10),
        ("intent_override", 5),
        ("boundary", 2),
    ):
        rows.extend(
            {
                "sample_id": f"{scenario_type}-{index}",
                "scenario_type": scenario_type,
                "user_profile": {},
                "ground_truth": {"parent_asin": "A"},
            }
            for index in range(count)
        )
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_selects_exact_deterministic_public_strata() -> None:
    from experiments.hybrid_question_comparison import select_stratified_public_samples

    samples = _samples()
    selected = select_stratified_public_samples(samples, seed=20260830)

    assert [row["sample_id"] for row in selected] == [
        row["sample_id"] for row in select_stratified_public_samples(samples, seed=20260830)
    ]
    assert len(selected) == 20
    assert Counter(row["scenario_type"] for row in selected) == {
        "buying": 8,
        "browsing": 8,
        "intent_override": 3,
        "boundary": 1,
    }


def test_sampling_rejects_missing_duplicate_and_undersized_strata() -> None:
    from experiments.hybrid_question_comparison import select_stratified_public_samples

    samples = _samples()
    for invalid in (
        [row for row in samples if row["scenario_type"] != "boundary"],
        [*samples, {**samples[0]}],
        [
            row
            for row in samples
            if row["scenario_type"] != "buying" or row["sample_id"].endswith("0")
        ],
    ):
        try:
            select_stratified_public_samples(invalid, seed=20260830)
        except ValueError:
            continue
        raise AssertionError("invalid sample population was accepted")


def test_comparison_reuses_one_catalog_resource_and_isolates_agents() -> None:
    from experiments.hybrid_question_comparison import run_comparison

    calls: dict[str, list] = {"retrievers": [], "resources": [], "agents": [], "evaluations": []}

    class FakeRetriever:
        def iter_products(self) -> tuple[dict, ...]:
            return (
                {"parent_asin": "A", "categories": ["Clothing", "Tops"]},
                {"parent_asin": "B", "categories": ["Clothing", "Shoes"]},
            )

    class FakeAgent:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def hybrid_question_statistics(self) -> dict[str, object]:
            return {
                "reason_counts": {"hybrid_specific_replacement": 1},
                "selected_attribute_counts": {"material": 1},
                "replacement_count": 1,
                "decision_latency_ms": {"count": 1, "p50": 1.0, "p95": 1.0},
            }

    def retriever_factory(*args: object, **kwargs: object) -> FakeRetriever:
        calls["retrievers"].append((args, kwargs))
        return FakeRetriever()

    def resource_factory(products: tuple[dict, ...], *, include_attribute_cache: bool) -> object:
        resource = object()
        calls["resources"].append((products, include_attribute_cache, resource))
        return resource

    def agent_factory(**kwargs: object) -> FakeAgent:
        agent = FakeAgent(**kwargs)
        calls["agents"].append(agent)
        return agent

    def evaluator(agent: FakeAgent, samples: list[dict], *args: object) -> dict:
        calls["evaluations"].append((agent, [row["sample_id"] for row in samples], args))
        return {
            "sample_count": len(samples),
            "hit_rate_at_10": 1.0,
            "mrr": 0.5,
            "mttc": 2.0,
            "efficiency": 0.9,
            "recommended_technical_score": 0.83,
            "reported_token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "scenario_metrics": {},
            "sessions": [{"sample_id": "must-not-leak"}],
        }

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        dataset = directory / "public.jsonl"
        catalog = directory / "catalog.jsonl"
        _write_jsonl(dataset, _samples())
        _write_jsonl(catalog, [{"parent_asin": "A"}])
        report = run_comparison(
            catalog,
            dataset,
            seed=20260830,
            time_budget_seconds=60,
            retriever_factory=retriever_factory,
            resource_factory=resource_factory,
            agent_factory=agent_factory,
            evaluator=evaluator,
        )

    assert report["status"] == "complete"
    assert len(calls["retrievers"]) == 1
    assert len(calls["resources"]) == 1
    assert len(calls["agents"]) == 4
    assert len({id(agent.kwargs["retriever"]) for agent in calls["agents"]}) == 1
    assert len({id(agent.kwargs["dialogue_catalog_resources"]) for agent in calls["agents"]}) == 1
    assert len({tuple(ids) for _, ids, _ in calls["evaluations"]}) == 1
    assert "must-not-leak" not in json.dumps(report, sort_keys=True)
    assert "winner" not in report
    assert "recommendation" not in report
    for configuration in report["configurations"]:
        overlay = configuration["overlay"]
        assert overlay["llm"]["provider"] == "none"
        assert overlay["dialogue_understanding"]["mode"] == "rule_only"
        assert overlay["decision"]["finish_strategy"] == {
            "enabled": False,
            "lookahead_depth": 1,
        }
        assert overlay["diagnostics"]["decision_trace"]["enabled"] is False


def test_timeout_writes_only_completed_configurations_without_a_winner() -> None:
    from experiments.hybrid_question_comparison import TimeBudgetExceeded, run_comparison

    completed: list[str] = []

    class FakeRetriever:
        def iter_products(self) -> tuple[dict, ...]:
            return ({"parent_asin": "A", "categories": ["Clothing", "Tops"]},)

    class FakeAgent:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def hybrid_question_statistics(self) -> dict[str, object]:
            return {}

    def evaluator(agent: FakeAgent, samples: list[dict], *args: object) -> dict:
        if completed:
            raise TimeBudgetExceeded("simulated deadline")
        completed.append("legacy")
        return {
            "sample_count": len(samples),
            "hit_rate_at_10": 1.0,
            "mrr": 0.5,
            "mttc": 2.0,
            "efficiency": 0.9,
            "recommended_technical_score": 0.83,
            "reported_token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "scenario_metrics": {},
        }

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        dataset = directory / "public.jsonl"
        catalog = directory / "catalog.jsonl"
        output = directory / "comparison.json"
        _write_jsonl(dataset, _samples())
        _write_jsonl(catalog, [{"parent_asin": "A"}])
        report = run_comparison(
            catalog,
            dataset,
            seed=20260830,
            time_budget_seconds=60,
            output_path=output,
            retriever_factory=lambda *args, **kwargs: FakeRetriever(),
            resource_factory=lambda *args, **kwargs: object(),
            agent_factory=lambda **kwargs: FakeAgent(**kwargs),
            evaluator=evaluator,
        )
        persisted = json.loads(output.read_text(encoding="utf-8"))

    assert report["status"] == "time_budget_exceeded"
    assert [item["name"] for item in report["configurations"]] == ["legacy"]
    assert persisted == report
    assert "winner" not in report
    assert "recommendation" not in report
