"""本地开发/提交模拟 评估启动脚本（调用官方评估器，不修改它）。

用法：
    python run_local_eval.py                        # ENV_MODE=dev 默认
    $env:ENV_MODE="dev";   python run_local_eval.py
    $env:ENV_MODE="submit"; python run_local_eval.py  # 提交模拟：强制离线约束检查
    $env:SAMPLE_LIMIT="10"; python run_local_eval.py  # 冒烟测试

兼容官方评估器：直接复用 evaluator.local_evaluator 的
load_jsonl / catalog_index / evaluate 函数，不改一行评估器源码。
输出：results.json（每会话 + 总体 + 场景指标）。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# 保证从仓库根目录可导入
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import constants
from config.env_config import EnvConfig
from agent.main_agent import Agent
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # 官方评估器（只调用，不修改）
from llm import create_llm_client
from llm.base import LLMClient


def initialize_llm(env: EnvConfig) -> LLMClient:
    """Initialize the optional LLM and report its sanitized availability."""
    client = create_llm_client(env.llm)
    status = client.initialize()
    details = (
        f"provider={status.provider} model={status.model} "
        f"state={status.state.value} attempts={status.attempts}"
    )
    if status.error_category is not None:
        details += f" error={status.error_category.value}"
    print(f"    LLM: {details}")
    return client


def main() -> int:
    parser = argparse.ArgumentParser(description="TechJam2026 购物副驾 Agent 本地评估")
    parser.add_argument("--catalog", default=str(constants.CATALOG_PATH))
    parser.add_argument("--dataset", default=str(constants.PUBLIC_SET_PATH))
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    env = EnvConfig.from_env()
    output = args.output or env.output_path
    print(f"\n=== TechJam2026 购物副驾 Agent 本地评估 ===")
    print(f"    {env.summary()}")
    if env.env_mode == "submit":
        print("    [submit] 提交模拟模式：强制执行离线约束检查（优先设置 LLM_PROVIDER=none；兼容 LLM_BACKEND=none/local）。")
        assert env.offline, "submit 模式禁止依赖外部付费 API（请优先设置 LLM_PROVIDER=none；兼容 LLM_BACKEND=none 或 local）"

    llm_client = initialize_llm(env)

    # 数据集完整性校验（Pillar IV / 硬性约束 3）
    if not env.skip_data_verify:
        from utils import data_verify
        data_verify.verify_dataset(skip=False)

    # 官方评估器数据加载
    t0 = time.time()
    samples = load_jsonl(args.dataset)
    if env.sample_limit:
        samples = samples[: env.sample_limit]
        print(f"    [dev] SAMPLE_LIMIT={env.sample_limit}（冒烟测试子集）")
    catalog_ids, categories, products = catalog_index(args.catalog)
    print(f"    数据加载完成：{len(samples)} 会话 / {len(catalog_ids)} 商品 "
          f"({time.time() - t0:.1f}s)")

    # 实例化业务 Agent（Pillar I~IV）
    agent = Agent(catalog_path=args.catalog, env=env, llm_client=llm_client)

    # 调用官方评估器 evaluate()（唯一评分入口，未修改）
    t0 = time.time()
    result = evaluate(agent, samples, catalog_ids, categories, products)
    print(f"    评估完成：{time.time() - t0:.1f}s")

    Path(output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    summary = {k: v for k, v in result.items() if k != "sessions"}
    print("\n--- 总体指标（官方评估器计算）---")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n结果已写入: {Path(output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
