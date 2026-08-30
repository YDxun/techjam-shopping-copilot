"""Tuning harness (Step 2): 160 tuning + 40 validation sessions, grid/joint search, full logging.

- Objective: official-evaluator TechnicalScore on public (HR/MRR/MTTC recorded, per-scenario too);
- samples[:160] are fixed for tuning; samples[160:] are used only for final validation (never in the
search);
- each round moves only one knob group (single-variable grid first; `--joint` for small joint
searches);
- shares the retriever (no FTS index rebuild); each config runs the full official evaluate() once;
- every run is logged to logs/tune_YYYYMMDD.json.

Usage:
  python scripts/tune_knobs.py --group rerank        # rerank_weights + combo + fingerprint
  python scripts/tune_knobs.py --group retrieval     # bm25 weights / rrf_k / limits / dense_weight
  python scripts/tune_knobs.py --group strategy   # decision / retrieval_mode / hard_cue / rule_conf
  python scripts/tune_knobs.py --group joint         # small joint search over the best single
  variables
  python scripts/tune_knobs.py --custom '{"rerank_weights":{"coverage":0.55}}'   # custom single
  point
  python scripts/tune_knobs.py --group rerank --samples 40   # quick smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.main_agent import Agent  # noqa: E402
from agent.reranker import Reranker  # noqa: E402
from agent.retriever import HybridRetriever  # noqa: E402
from config.env_config import EnvConfig  # noqa: E402
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from llm.base import DisabledLLMClient  # noqa: E402

LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# knob-group definitions: label -> overrides snippet (deep-merged by
# EnvConfig.from_env(overrides=...))
# ---------------------------------------------------------------------------
BASELINE_OVERRIDES = {"skip_data_verify": True}


def _rerank_grid() -> list[tuple[str, dict]]:
    out = []
    for cov in (0.40, 0.45, 0.55, 0.60):
        out.append((f"coverage={cov}", {"rerank_weights": {"coverage": cov}}))
    for combo in (0.05, 0.15, 0.20, 0.25):
        out.append((f"combo={combo}", {"rerank_weights": {"combo": combo}}))
    for cat in (0.15, 0.20, 0.30, 0.35):
        out.append((f"category={cat}", {"rerank_weights": {"category": cat}}))
    for rrf in (0.05, 0.10, 0.20, 0.25):
        out.append((f"rrf={rrf}", {"rerank_weights": {"rrf": rrf}}))
    for pop in (0.0, 0.10):
        out.append((f"popularity={pop}", {"rerank_weights": {"popularity": pop}}))
    # fingerprint on + different bonuses (A/B already shows saturation; still in the grid)
    out.append(("fp_on_u1", {"fingerprint": {"enable": True, "bonus_unique": 1.0}}))
    out.append(("fp_on_u0_5", {"fingerprint": {"enable": True, "bonus_unique": 0.5}}))
    return out


def _retrieval_grid() -> list[tuple[str, dict]]:
    out = []
    base = [0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0]
    for title_w in (4.0, 8.0):
        w = list(base)
        w[1] = title_w
        out.append((f"bm25_title={title_w}", {"retrieval": {"bm25_field_weights": w}}))
    for feat_w in (3.0, 5.0):
        w = list(base)
        w[2] = feat_w
        out.append((f"bm25_feat={feat_w}", {"retrieval": {"bm25_field_weights": w}}))
    for desc_w in (0.5, 1.5):
        w = list(base)
        w[6] = desc_w
        out.append((f"bm25_desc={desc_w}", {"retrieval": {"bm25_field_weights": w}}))
    for rrf_k in (40.0, 80.0, 100.0):
        out.append((f"rrf_k={rrf_k}", {"retrieval": {"rrf_k": rrf_k}}))
    for ck in (5.0, 15.0):
        out.append((f"rrf_constraint_k={ck}", {"retrieval": {"rrf_constraint_k": ck}}))
    for dw in (0.3, 0.8):
        out.append((f"dense_weight={dw}", {"retrieval": {"dense_weight": dw}}))
    return out


def _strategy_grid() -> list[tuple[str, dict]]:
    out = []
    for mh in (1, 3):
        out.append((f"exploit_min_hard={mh}", {"retrieval_mode": {"exploit_min_hard": mh}}))
    for mc in (3, 5):
        out.append(
            (f"exploit_min_constraints={mc}",
             {"retrieval_mode": {"exploit_min_constraints": mc}})
        )
    for mq in (2, 4):
        out.append((f"max_questions={mq}", {"decision": {"max_questions": mq}}))
    for ig in (0.20, 0.40):
        out.append(
            (f"ask_ig={ig}",
             {"decision": {"ask_utility": {"weights": {"information_gain": ig}}}})
        )
    for th in (0.60, 0.90):
        out.append(
            (f"rule_conf={th}",
             {"dialogue_understanding": {"rule_confidence_threshold": th}})
        )
    out.append(("hard_cue_off", {"hard_cue_enabled": False}))
    return out


GROUPS = {
    "rerank": _rerank_grid,
    "retrieval": _retrieval_grid,
    "strategy": _strategy_grid,
}


# ---------------------------------------------------------------------------
def metric_summary(result: dict) -> dict:
    return {
        "ts": result["recommended_technical_score"],
        "hr": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "scenarios": {
            k: (m["hit_rate_at_10"], m["mrr"], m["mttc"])
            for k, m in result["scenario_metrics"].items()
        },
    }


def run_eval(env: EnvConfig, agent: Agent, samples: list[dict],
             catalog_ids, categories, products) -> dict:
    result = evaluate(agent, samples, catalog_ids, categories, products)
    return metric_summary(result)


def build_agent(env: EnvConfig, retriever: HybridRetriever | None) -> Agent:
    llm_client = DisabledLLMClient()
    reranker = Reranker(env=env, llm_client=llm_client)
    return Agent(env=env, retriever=retriever, reranker=reranker, llm_client=llm_client)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=list(GROUPS) + ["joint"], default="rerank")
    ap.add_argument("--custom", default="", help="single custom overrides JSON")
    ap.add_argument("--samples", type=int, default=160, help="tuning subset size (<=160)")
    ap.add_argument("--top", type=int, default=3, help="take the top-N single variables for joint")
    args = ap.parse_args()

    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    tune_samples = samples[: args.samples]        # 160 tuning (never touches the 40 validation)
    valid_samples = samples[160:]                 # 40 validation (never appears in the search)
    catalog_ids, categories, products = catalog_index(ROOT / "data" / "catalog.jsonl")

    # shared retriever: FTS index built once (retrieval knobs apply via
    # _retrieval_cfg/_bm25_weights_sql)
    base_env = EnvConfig.from_env(overrides=dict(BASELINE_OVERRIDES))
    retriever = HybridRetriever(catalog_path=ROOT / "data" / "catalog.jsonl", env=base_env,
                                backend=base_env.retrieval_backend)
    # base agent built once (Agent.__init__'s CatalogQuestionSignals scans 50k, ~70s; must be
    # reused)
    base_agent = build_agent(base_env, retriever)

    def _apply_retrieval(env: EnvConfig) -> None:
        cfg = env.retrieval
        retriever._retrieval_cfg = cfg
        retriever._bm25_weights_sql = (
            "bm25(products, " + ", ".join(str(float(w)) for w in cfg.bm25_field_weights) + ")"
        )

    # knobs affecting dialogue/question_policy require rebuilding the agent;
    # rerank/retrieval knobs only swap the reranker + retriever cfg
    DIALOGUE_KEYS = {
        "decision", "retrieval_mode", "dialogue_understanding",
        "hard_cue_enabled", "llm_intent_enabled", "llm_clarify_enabled",
    }

    def eval_overrides(overrides: dict, label: str) -> dict:
        env = EnvConfig.from_env(overrides={**BASELINE_OVERRIDES, **overrides})
        _apply_retrieval(env)
        if any(k in DIALOGUE_KEYS for k in overrides):
            agent = build_agent(env, retriever)
        else:
            agent = base_agent
            agent.reranker = Reranker(env=env, llm_client=DisabledLLMClient())
            agent.last_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        return {
            "label": label,
            "overrides": overrides,
            "tune": run_eval(env, agent, tune_samples, catalog_ids, categories, products),
        }

    log: dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "group": args.group,
        "n_tune": len(tune_samples),
        "runs": [],
    }

    if args.custom:
        overrides = json.loads(args.custom)
        entry = eval_overrides(overrides, "custom")
        log["runs"].append(entry)
        print(json.dumps(entry, ensure_ascii=False, indent=2))
    else:
        if args.group == "joint":
            # first run the rerank group's single-variable grid, then jointly combine the best
            grid = _rerank_grid()
        else:
            grid = GROUPS[args.group]()
        results = []
        for label, overrides in grid:
            entry = eval_overrides(overrides, label)
            results.append(entry)
            log["runs"].append(entry)
            print(f"[tune] {label:22s} ts={entry['tune']['ts']:.4f} hr={entry['tune']['hr']:.3f} "
                  f"mrr={entry['tune']['mrr']:.4f} mttc={entry['tune']['mttc']:.3f}")
        results.sort(key=lambda e: -e["tune"]["ts"])
        print("\n--- 160-tuning Top-5 ---")
        for e in results[:5]:
            print(f"  {e['label']:22s} ts={e['tune']['ts']:.4f} mrr={e['tune']['mrr']:.4f}")

        if args.group == "joint":
            top = results[: args.top]
            # pairwise joint: combine the top overrides (shallow-merge dict keys)
            combos: list[tuple[str, dict]] = []
            for i in range(len(top)):
                for j in range(i + 1, len(top)):
                    merged = {**top[i]["overrides"], **top[j]["overrides"]}
                    # deep-merge nested dicts
                    for key in set(top[i]["overrides"]) & set(top[j]["overrides"]):
                        a, b = top[i]["overrides"][key], top[j]["overrides"][key]
                        if isinstance(a, dict) and isinstance(b, dict):
                            merged[key] = {**a, **b}
                    combos.append((f"joint_{top[i]['label']}+{top[j]['label']}", merged))
            for label, overrides in combos:
                entry = eval_overrides(overrides, label)
                log["runs"].append(entry)
                print(
                    f"[tune] {label:22s} ts={entry['tune']['ts']:.4f} "
                    f"mrr={entry['tune']['mrr']:.4f}"
                )

        # best single variable -> 40-session validation (validation only, never searched)
        best = results[0]
        best_env = EnvConfig.from_env(overrides={**BASELINE_OVERRIDES, **best["overrides"]})
        _apply_retrieval(best_env)
        best_agent = build_agent(best_env, retriever)
        best["validation"] = run_eval(
            best_env, best_agent, valid_samples, catalog_ids, categories, products
        )
        log["best"] = best
        print("\n--- 40-session validation (best single variable) ---")
        print(json.dumps(best, ensure_ascii=False, indent=2))

    log_path = LOGS / f"tune_{time.strftime('%Y%m%d_%H%M%S')}.json"
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nLog written: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
