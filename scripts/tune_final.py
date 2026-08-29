"""最终候选联合对比：160 调参 + 40 验证（基线 vs 最优单变量 vs 联合）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.main_agent import Agent
from agent.reranker import Reranker
from agent.retriever import HybridRetriever
from config.env_config import EnvConfig
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from llm.base import DisabledLLMClient

ROOT = Path(__file__).resolve().parent.parent
samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
tune = samples[:160]
valid = samples[160:]
cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")

base_env = EnvConfig.from_env(overrides={"skip_data_verify": True})
retriever = HybridRetriever(catalog_path=ROOT / "data" / "catalog.jsonl", env=base_env,
                            backend=base_env.retrieval_backend)
base_agent = Agent(env=base_env, retriever=retriever,
                   reranker=Reranker(env=base_env, llm_client=DisabledLLMClient()),
                   llm_client=DisabledLLMClient())

def apply(env):
    retriever._retrieval_cfg = env.retrieval
    retriever._bm25_weights_sql = (
        "bm25(products, " + ", ".join(str(float(w)) for w in env.retrieval.bm25_field_weights) + ")"
    )

def ev(overrides, subset):
    env = EnvConfig.from_env(overrides={**{"skip_data_verify": True}, **overrides})
    apply(env)
    base_agent.reranker = Reranker(env=env, llm_client=DisabledLLMClient())
    base_agent.last_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    r = evaluate(base_agent, subset, cid, cats, prods)
    return {k: r[k] for k in ["hit_rate_at_10", "mrr", "mttc", "recommended_technical_score"]}

CANDIDATES = [
    ("baseline", {}),
    ("rrf=0.05", {"rerank_weights": {"rrf": 0.05}}),
    ("rrf_k=100", {"retrieval": {"rrf_k": 100.0}}),
    ("rrf0.05+rrf_k100", {"rerank_weights": {"rrf": 0.05}, "retrieval": {"rrf_k": 100.0}}),
    ("rrf0.05+fp", {"rerank_weights": {"rrf": 0.05}, "fingerprint": {"enable": True}}),
]

out = []
for label, ov in CANDIDATES:
    t = ev(ov, tune)
    v = ev(ov, valid)
    out.append({"label": label, "overrides": ov, "tune160": t, "valid40": v})
    print(f"{label:18s} tune160 ts={t['recommended_technical_score']:.4f} mrr={t['mrr']:.4f} "
          f"| valid40 ts={v['recommended_technical_score']:.4f} mrr={v['mrr']:.4f}", flush=True)

(ROOT / "logs" / "tune_final.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
)
print("写入 logs/tune_final.json")
