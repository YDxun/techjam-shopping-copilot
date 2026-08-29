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

from agent.main_agent import Agent  # noqa: E402
from config import constants  # noqa: E402
from config.env_config import EnvConfig  # noqa: E402
from evaluator.local_evaluator import (  # noqa: E402
    catalog_index,
    evaluate,
    load_jsonl,
)  # 官方评估器（只调用，不修改）
from llm import create_llm_client  # noqa: E402
from llm.base import LLMClient  # noqa: E402

logger = logging.getLogger(__name__)


def resolve_trace_output_path(
    output_path: str | Path,
    catalog_path: str | Path,
    dataset_path: str | Path,
    repo_root: Path = ROOT,
    evaluation_output_path: str | Path | None = None,
) -> Path:
    """Resolve a local diagnostics file without permitting protected-input overwrite."""
    lexical_output = Path(output_path)
    if not lexical_output.is_absolute():
        lexical_output = repo_root / lexical_output
    _reject_symlink_components(lexical_output)

    def resolve_from_repo(value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = repo_root / path
        return path.resolve()

    output = resolve_from_repo(output_path)
    protected = {resolve_from_repo(catalog_path), resolve_from_repo(dataset_path)}
    if evaluation_output_path is not None:
        protected.add(Path(evaluation_output_path).resolve())
    if output in protected:
        raise ValueError(
            "decision trace output must not overwrite catalog or dataset or evaluation output"
        )
    return output


def _reject_symlink_components(path: Path) -> None:
    """Reject lexical aliases, including dangling leaf aliases, before resolution."""
    system_aliases = ((Path("/var"), Path("/private/var")), (Path("/tmp"), Path("/private/tmp")))
    for alias, physical in system_aliases:
        try:
            path = physical / path.relative_to(alias)
        except ValueError:
            continue
        break
    current = Path("/") if path.is_absolute() else Path.cwd()
    for component in path.parts[1:] if path.is_absolute() else path.parts:
        if component in {"", "."}:
            continue
        current /= component
        if current.is_symlink():
            raise ValueError(
                "decision trace output must not use symlink components or catalog or dataset"
            )


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
    catalog_path = Path(args.catalog).resolve()
    dataset_path = Path(args.dataset).resolve()
    evaluation_output = Path(output).resolve()
    print("\n=== TechJam2026 购物副驾 Agent 本地评估 ===")
    print(f"    {env.summary()}")
    if env.env_mode == "submit":
        print(
            "    [submit] 提交模拟模式：强制执行离线约束检查\n"
            "    （优先设置 LLM_PROVIDER=none；兼容 LLM_BACKEND=none/local）。"
        )
        assert env.offline, (
            "submit 模式禁止依赖外部付费 API（请优先设置 LLM_PROVIDER=none；\n"
            "兼容 LLM_BACKEND=none 或 local）"
        )

    llm_client = initialize_llm(env)

    # 数据集完整性校验（Pillar IV / 硬性约束 3）
    if not env.skip_data_verify:
        from utils import data_verify

        data_verify.verify_dataset(skip=False)

    # 官方评估器数据加载
    t0 = time.time()
    samples = load_jsonl(dataset_path)
    if env.sample_limit:
        samples = samples[: env.sample_limit]
        print(f"    [dev] SAMPLE_LIMIT={env.sample_limit}（冒烟测试子集）")
    catalog_ids, categories, products = catalog_index(catalog_path)
    print(
        f"    数据加载完成：{len(samples)} 会话 / {len(catalog_ids)} 商品 ({time.time() - t0:.1f}s)"
    )

    # 实例化业务 Agent（Pillar I~IV）
    agent = Agent(catalog_path=catalog_path, env=env, llm_client=llm_client)

    # 调用官方评估器 evaluate()（唯一评分入口，未修改）
    t0 = time.time()
    result = evaluate(agent, samples, catalog_ids, categories, products)
    print(f"    评估完成：{time.time() - t0:.1f}s")

    # 本地诊断附加在最终评测文件，不改变官方每轮 Agent 响应契约或官方评估器。
    result["intent_recognition_statistics"] = agent.intent_recognition_statistics()
    result["transition_guard_statistics"] = agent.transition_guard_statistics()
    result["dialogue_decision_statistics"] = agent.dialogue_decision_statistics()

    evaluation_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if env.diagnostics.decision_trace.enabled:
        try:
            trace_output = resolve_trace_output_path(
                env.diagnostics.decision_trace.output_path,
                catalog_path,
                dataset_path,
                evaluation_output_path=evaluation_output,
            )
            agent.dialogue.decision_trace_recorder.export_jsonl(trace_output)
        except Exception as exc:
            logger.exception("[diagnostics] trace export failed")
            print(f"    ERROR: decision trace export failed after results were written: {exc}")
            return 1

    summary = {k: v for k, v in result.items() if k != "sessions"}
    print("\n--- 总体指标（官方评估器计算）---")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n结果已写入: {Path(output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
