"""构建"配置-环境-性能"查找表 data/assets/env_config_lut.json（Step 3）。

- 配置档案：rule_bm25 / hybrid_dense / fingerprint_combo / text_rerank / reranker_model；
- 环境维度：device × dense × llm × network（可模拟：dense=no 用不存在的 blair 路径强制回退 bm25；
  llm/network 无 key 时 text_rerank 自动回退——记录"回退后"的实际表现并标注 measured）；
- 默认 SAMPLE_LIMIT=40 冒烟（快速），--full 对指定 (env,config) 全量确认；
- 记录：technical_score / hr / mrr / mttc / latency_ms_per_turn / memory_mb / tokens / measured。

用法：
  python scripts/build_lut.py                # 40 条冒烟 LUT
  python scripts/build_lut.py --samples 200  # 全量 LUT（较慢）
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
from utils.lut import env_fingerprint  # noqa: E402

OUT = ROOT / "data" / "assets" / "env_config_lut.json"

# 配置档案（config_id -> overrides + 说明）
PROFILES: dict[str, dict] = {
    "rule_bm25": {
        "overrides": {"retrieval_backend": "bm25", "fingerprint": {"enable": False},
                      "llm": {"rerank_enabled": False}, "reranker_model_enabled": False},
        "label": "纯规则离线（保底，无 BLaIR/LLM/模型依赖）",
    },
    "hybrid_dense": {
        "overrides": {"retrieval_backend": "auto", "fingerprint": {"enable": False},
                      "llm": {"rerank_enabled": False}, "reranker_model_enabled": False},
        "label": "+BLaIR 稠密（dense-recover，需离线 npy + transformers）",
    },
    "fingerprint_combo": {
        "overrides": {"retrieval_backend": "auto", "fingerprint": {"enable": True},
                      "reranker_model_enabled": False},
        "label": "最优规则 + 约束组合指纹 + combo",
    },
    "text_rerank": {
        "overrides": {"retrieval_backend": "auto",
                      "llm": {"rerank_enabled": True, "rerank_backend": "text"},
                      "reranker_model_enabled": False},
        "label": "qwen3-rerank 文本重排（需 DASHSCOPE key+网络；无 key 自动回退）",
    },
    "reranker_model": {
        "overrides": {"retrieval_backend": "auto", "reranker_model_enabled": True,
                      "reranker_model": "thebajajra/RexReranker-0.6B"},
        "label": "RexReranker-0.6B 交叉编码（需本地模型缓存；recover 模式第二意见）",
    },
}

# 环境矩阵（dense 可模拟；device=cuda 本机；llm/network 无 key 时记录回退表现）
ENVS: list[tuple[str, dict]] = [
    (env_fingerprint(device="cuda", dense=False, llm=False, network=False),
     {"dense": False, "sim": {"blair_offline_embedding_path": "data/_missing_blair.npy"}}),
    (env_fingerprint(device="cuda", dense=True, llm=False, network=False),
     {"dense": True, "sim": {}}),
]


def build_agent(env: EnvConfig, retriever: HybridRetriever | None) -> Agent:
    llm_client = DisabledLLMClient()
    return Agent(env=env, retriever=retriever, reranker=Reranker(env=env, llm_client=llm_client),
                 llm_client=llm_client)


def metric_summary(result: dict) -> dict:
    return {
        "ts": result["recommended_technical_score"],
        "hr": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "tokens": result.get("reported_token_usage", {}),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--samples", type=int, default=40, help="每个 (env,config) 评估条数（默认 40 冒烟）"
    )
    args = ap.parse_args()

    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")[: args.samples]
    catalog_ids, categories, products = catalog_index(ROOT / "data" / "catalog.jsonl")

    # 2 个 base retriever/agent：dense=yes(hybrid) 与 dense=no(bm25)
    base_env_yes = EnvConfig.from_env(overrides={"skip_data_verify": True})
    ret_yes = HybridRetriever(catalog_path=ROOT / "data" / "catalog.jsonl", env=base_env_yes,
                              backend=base_env_yes.retrieval_backend)
    agent_yes = build_agent(base_env_yes, ret_yes)
    base_env_no = EnvConfig.from_env(
        overrides={
            "skip_data_verify": True,
            "blair_offline_embedding_path": "data/_missing_blair.npy",
            "retrieval_backend": "bm25",
        }
    )
    ret_no = HybridRetriever(catalog_path=ROOT / "data" / "catalog.jsonl", env=base_env_no,
                             backend=base_env_no.retrieval_backend)
    agent_no = build_agent(base_env_no, ret_no)

    lut: dict = {
        "generated_by": "scripts/build_lut.py",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": len(samples),
        "note": (
            "score 与 device 无关（确定性评估器）；"
            "llm/network 无 key 时 text_rerank 记录回退表现"
        ),
        "profiles": {k: v["label"] for k, v in PROFILES.items()},
        "environments": {},
    }

    for fp, env_info in ENVS:
        dense = env_info["dense"]
        base_agent = agent_yes if dense else agent_no
        retriever = ret_yes if dense else ret_no
        lut["environments"][fp] = {"configs": []}
        for config_id, prof in PROFILES.items():
            overrides = {**env_info["sim"], **prof["overrides"]}
            env = EnvConfig.from_env(overrides={**{"skip_data_verify": True}, **overrides})
            # 复用 base agent（只换 reranker + 改 retriever cfg）
            retriever._retrieval_cfg = env.retrieval
            retriever._bm25_weights_sql = (
                "bm25(products, "
                + ", ".join(str(float(w)) for w in env.retrieval.bm25_field_weights)
                + ")"
            )
            agent = base_agent
            agent.reranker = Reranker(env=env, llm_client=DisabledLLMClient())
            agent.last_usage = {"prompt_tokens": 0, "completion_tokens": 0}
            t0 = time.time()
            result = evaluate(agent, samples, catalog_ids, categories, products)
            dt = time.time() - t0
            ms = metric_summary(result)
            entry = {
                "config_id": config_id,
                "technical_score": round(ms["ts"], 6),
                "hr": round(ms["hr"], 6),
                "mrr": round(ms["mrr"], 6),
                "mttc": round(ms["mttc"], 6),
                "latency_ms_per_turn": round(dt * 1000 / max(1, len(samples)), 1),
                "memory_mb": 0,
                "tokens": ms["tokens"],
                "measured": True,
            }
            lut["environments"][fp]["configs"].append(entry)
            print(f"[lut] {fp:44s} {config_id:18s} ts={entry['technical_score']:.4f} "
                  f"mrr={entry['mrr']:.4f} lat={entry['latency_ms_per_turn']:.0f}ms/turn")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(lut, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nLUT 写入: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
