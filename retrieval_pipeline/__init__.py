"""检索管线包：赛题第4-6步（查询构建 / 三通道检索 / 重排序）。

本包只做检索管线，不实现 Agent respond/reset、不实现状态机、不修改评测器。
上层（意图识别+对话状态机）传入 session_state，本包返回 PipelineOutput。
"""

from retrieval_pipeline.models import (
    PipelineOutput,
    QueryBundle,
    SessionState,
    StrategyConfig,
)
from retrieval_pipeline.pipeline import RetrievalPipeline, run_pipeline

__all__ = [
    "SessionState",
    "StrategyConfig",
    "QueryBundle",
    "PipelineOutput",
    "RetrievalPipeline",
    "run_pipeline",
]
