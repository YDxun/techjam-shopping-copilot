"""检索管线演示：构造模拟 session_state，跑通第4-6步完整链路，打印 Top-10 asin。

运行：python retrieval_pipeline/test_pipeline.py
（默认使用竞赛冻结目录 data/catalog.jsonl；离线 npy 不存在时稠密通道自动禁用；
 FlagEmbedding 未安装时重排自动降级 fused 排序——演示链路不依赖任何付费 API。）
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval_pipeline.models import PipelineOutput, SessionState, StrategyConfig
from retrieval_pipeline.pipeline import RetrievalPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def show(session: SessionState, pipeline: RetrievalPipeline, label: str) -> None:
    print("\n" + "=" * 90)
    print(f"[{label}]")
    print(f"  constraints    : {session.constraints}")
    print(f"  recovery_mode  : {session.recovery_mode}")
    print(f"  raw_query      : {session.user_raw_query!r}")
    out: PipelineOutput = pipeline.run(session)
    print(f"  raw_fused n    : {len(out.raw_fused_candidates)}")
    for asin, score in out.raw_fused_candidates[:8]:
        title = (pipeline.catalog.get(asin) or {}).get("title", "")[:60]
        print(f"    fused {score:+.4f}  {asin}  {title}")
    print(f"  reranked_top10 : {out.reranked_top10}")


def main() -> None:
    pipeline = RetrievalPipeline()   # 复用同一实例（索引只建一次）

    # 场景A：普通购买会话（硬过滤）
    show(SessionState(
        constraints={"material": "cotton", "color": "black"},
        recovery_mode=False,
        strategy_config=StrategyConfig(retrieval_pool_size=50),
        user_raw_query="I'm looking for T-Shirts. A key requirement is: cotton.",
    ), pipeline, "普通模式（第5步-通道1 硬过滤）")

    # 场景B：RECOVER 模式（连续 miss>=2）→ 惩罚打分 / 同义词 / 变体 / 大候选池
    show(SessionState(
        constraints={"material": "cotton", "color": "black", "budget_max": 40},
        recovery_mode=True,
        strategy_config=StrategyConfig(
            rrf_alpha=0.8,
            retrieval_pool_size=100,
            enable_query_variant=True,
            enable_synonym=True,
        ),
        user_raw_query="I'm looking for a cotton jumper under $50.",
    ), pipeline, "RECOVER 模式（惩罚打分+同义词+变体+池100）")

    # 场景C：override 已清空约束 → 只用 user_raw_query
    show(SessionState(
        constraints={},
        recovery_mode=False,
        strategy_config=StrategyConfig(),
        user_raw_query="Actually, ignore my earlier preference. What I need is: leather.",
    ), pipeline, "override 清空约束（仅原始 query）")


if __name__ == "__main__":
    main()
