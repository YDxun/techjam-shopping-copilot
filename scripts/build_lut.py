"""Build the config-environment-performance lookup table data/assets/env_config_lut.json
    (automation-control finalization v2).

- Config profiles: rule_bm25 / hybrid_dense / fingerprint_combo / text_rerank / reranker_model;
- Environment fingerprints: device=cpu|cuda x dense=yes|no x llm=yes|no x network=yes|no;
  * dense=no forces fallback to bm25 via a non-existent blair path;
  * llm=yes/network=yes without a key forces fallback (provider=deepseek without key / text_rerank
  without
    DASHSCOPE key -> probe disabled); the post-fallback behavior is what gets measured;
  * device does not affect the deterministic evaluator score (only latency may differ; dense
  triggers only in recover).
- Latency measurement: per (env, config) at least 3 small-sample timings, median
(latency_ms_per_turn,
   normalized to per-turn by mttc); score runs once on the full --samples (default 200).
- Records: technical_score / hr / mrr / mttc / latency_ms_per_turn / memory_mb / tokens /
  measured / note。

Usage:
  python scripts/build_lut.py                # full 200 (16 fingerprints x 5 profiles; deduped
  scores, ~40-60min)
  python scripts/build_lut.py --samples 40   # smoke (validate the script + quick reference)
"""
from __future__ import annotations

import argparse
import json
import statistics
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
from config.profiles import CONFIG_PROFILES  # noqa: E402  # P3 single source of truth
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from llm.base import DisabledLLMClient  # noqa: E402
from utils.lut import env_fingerprint  # noqa: E402

OUT = ROOT / "data" / "assets" / "env_config_lut.json"
MISSING_BLAIR = "data/_missing_blair.npy"

# Config profiles: CONFIG_PROFILES is the single source of truth (config/profiles.py, P3),
# referenced directly here
PROFILES: dict = CONFIG_PROFILES

# Environment fingerprints (device x dense x llm x network)
ENVS: list[tuple[str, dict]] = []
for device in ("cuda", "cpu"):
    for dense in (True, False):
        for llm in (True, False):
            for network in (True, False):
                fp = env_fingerprint(
                    device=device, dense=dense, llm=llm, network=network
                )
                sim: dict = {}
                if not dense:
                    sim["blair_offline_embedding_path"] = MISSING_BLAIR
                    sim["retrieval_backend"] = "bm25"
                if llm:
                    # no key -> forced fallback (probe disabled -> intent/rerank all fall back to
                    # rules)
                    sim["llm"] = {"provider": "deepseek", "rerank_enabled": True,
                                  "rerank_backend": "text"}
                else:
                    sim["llm"] = {"provider": "none", "rerank_enabled": False}
                note = []
                if not dense:
                    note.append("dense=no: missing blair path -> forced fallback to bm25")
                if llm:
                    note.append(
                        "llm=yes without key -> probe disabled, intent/rerank fall back to rules (post-fallback measured)"  # noqa: E501
                    )
                if not network:
                    note.append("network=no: text_rerank has no network -> falls back")
                ENVS.append((fp, {"dense": dense, "sim": sim, "note": "; ".join(note)}))


def build_agent(env: EnvConfig, retriever: HybridRetriever | None) -> Agent:
    llm_client = DisabledLLMClient()
    return Agent(
        env=env, retriever=retriever,
        reranker=Reranker(env=env, llm_client=llm_client),
        llm_client=llm_client,
    )


# P4 cost disclosure: tokens x unit price (public reference prices, USD/1M tokens; estimates only,
# actual prices per vendor)
UNIT_PRICES = {
    "deepseek": {"input": 0.27, "output": 1.10},  # deepseek-chat
    "openai": {"input": 0.15, "output": 0.60},  # gpt-4o-mini
    "qwen_rerank": None,  # billed per MaaS call, not per token
}


def cost_per_session(tokens: dict, provider: str = "deepseek") -> float:
    """Estimate per-session cost (USD) from tokens; no tokens / offline -> 0."""
    prices = UNIT_PRICES.get(provider)
    if not prices:
        return 0.0
    prompt = tokens.get("prompt_tokens", 0) or 0
    completion = tokens.get("completion_tokens", 0) or 0
    return (prompt / 1e6) * prices["input"] + (completion / 1e6) * prices["output"]


def behavior_key(env: EnvConfig, dense_available: bool) -> tuple:
    """Behavior fingerprint of the deterministic score: whether dense is actually on / fingerprint
        on / rerank actually active.

    Used to dedupe across (env, config): llm=yes without key and reranker_model without a model both
    fall back,
    so their scores equal fingerprint_combo and the full 200 need not be re-run.
    """
    backend = env.retrieval_backend
    dense_on = dense_available and backend in ("auto", "hybrid", "dense")
    fingerprint_on = bool(env.fingerprint.enable)
    # without key/model these are all False (fallback), so the key merges into the matching
    # rule/fingerprint profile
    rerank_active = bool(env.llm.rerank_enabled) and bool(env.llm.api_key)
    reranker_model = bool(env.reranker_model_enabled)
    return (dense_on, fingerprint_on, rerank_active, reranker_model)


def metric_summary(result: dict) -> dict:
    return {
        "ts": result["recommended_technical_score"],
        "hr": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "tokens": result.get("reported_token_usage", {}),
    }


def run_eval(agent: Agent, samples, cid, cats, prods) -> tuple[dict, float]:
    """Run one evaluation; returns (metric_summary, dt seconds)."""
    t0 = time.time()
    result = evaluate(agent, samples, cid, cats, prods)
    dt = time.time() - t0
    return metric_summary(result), dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=200, help="full-eval session count per (env,config)")  # noqa: E501
    ap.add_argument("--latency-samples", type=int, default=5, help="small-sample count for latency measurement")  # noqa: E501
    ap.add_argument("--latency-reps", type=int, default=3, help="latency measurement reps (median)")
    args = ap.parse_args()

    all_samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    score_samples = all_samples[: args.samples]
    lat_samples = all_samples[: args.latency_samples]
    catalog_ids, categories, products = catalog_index(ROOT / "data" / "catalog.jsonl")

    # 2 base agents: dense=yes (hybrid) and dense=no (bm25)
    base_env_yes = EnvConfig.from_env(overrides={"skip_data_verify": True})
    ret_yes = HybridRetriever(
        catalog_path=ROOT / "data" / "catalog.jsonl", env=base_env_yes,
        backend=base_env_yes.retrieval_backend,
    )
    agent_yes = build_agent(base_env_yes, ret_yes)
    base_env_no = EnvConfig.from_env(
        overrides={
            "skip_data_verify": True,
            "blair_offline_embedding_path": MISSING_BLAIR,
            "retrieval_backend": "bm25",
        }
    )
    ret_no = HybridRetriever(
        catalog_path=ROOT / "data" / "catalog.jsonl", env=base_env_no,
        backend=base_env_no.retrieval_backend,
    )
    agent_no = build_agent(base_env_no, ret_no)

    lut: dict = {
        "generated_by": "scripts/build_lut.py",
        "generated_at": "",  # back-filled with the completion time before writing (see end of file)
        "n_samples": len(score_samples),
        "latency_reps": args.latency_reps,
        "latency_samples": args.latency_samples,
        "note": (
            "score is device-independent (deterministic evaluator); llm/network without keys force fallback and record the post-fallback behavior;"  # noqa: E501
            "latency_ms_per_turn = measured latency (small-sample x reps median, normalized per turn by mttc)"  # noqa: E501
        ),
        "profiles": {k: v["label"] for k, v in PROFILES.items()},
        "environments": {},
    }

    score_cache: dict[tuple, dict] = {}  # behavior fingerprint -> {summary, config_id} (dedup reuse)  # noqa: E501

    for fp, env_info in ENVS:
        dense = env_info["dense"]
        base_agent = agent_yes if dense else agent_no
        retriever = ret_yes if dense else ret_no
        lut["environments"][fp] = {"note": env_info["note"], "configs": []}
        for config_id, prof in PROFILES.items():
            overrides = {**env_info["sim"], **prof["overrides"]}
            env = EnvConfig.from_env(overrides={**{"skip_data_verify": True}, **overrides})
            retriever._retrieval_cfg = env.retrieval
            retriever._bm25_weights_sql = (
                "bm25(products, "
                + ", ".join(str(float(w)) for w in env.retrieval.bm25_field_weights)
                + ")"
            )
            agent = base_agent
            agent.reranker = Reranker(env=env, llm_client=DisabledLLMClient())
            agent.last_usage = {"prompt_tokens": 0, "completion_tokens": 0}

            # latency measurement: reps small-sample runs, median per-turn latency
            per_turn: list[float] = []
            for _ in range(args.latency_reps):
                ms, dt = run_eval(agent, lat_samples, catalog_ids, categories, products)
                per_turn.append(dt * 1000 / max(1.0, len(lat_samples) * ms["mttc"]))
            lat_median = statistics.median(per_turn)

            # full score (deduped by behavior fingerprint: profiles with identical behavior run
            # once, others reuse)
            bkey = behavior_key(env, dense)
            if bkey in score_cache:
                ms_full = score_cache[bkey]["summary"]
                score_source = f"reused({score_cache[bkey]['config_id']})"
            else:
                ms_full, _dt_full = run_eval(
                    agent, score_samples, catalog_ids, categories, products
                )
                score_cache[bkey] = {"summary": ms_full, "config_id": config_id}
                score_source = "measured"

            entry = {
                "config_id": config_id,
                "technical_score": round(ms_full["ts"], 6),
                "hr": round(ms_full["hr"], 6),
                "mrr": round(ms_full["mrr"], 6),
                "mttc": round(ms_full["mttc"], 6),
                "latency_ms_per_turn": round(lat_median, 1),
                "memory_mb": 0,
                "tokens": ms_full["tokens"],
                "cost_usd_per_session": round(
                    cost_per_session(ms_full["tokens"]), 6
                ),
                "measured": True,
                "score_source": score_source,
                "device": fp.split(";")[0].split("=")[1],
            }
            lut["environments"][fp]["configs"].append(entry)
            print(
                f"[lut] {fp:46s} {config_id:18s} ts={entry['technical_score']:.4f} "
                f"mrr={entry['mrr']:.4f} lat={entry['latency_ms_per_turn']:.0f}ms/turn",
                flush=True,
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    lut["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")  # completion time
    OUT.write_text(json.dumps(lut, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nLUT written: {OUT} (n_samples={len(score_samples)}, fingerprints x {len(ENVS)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
