"""pydantic 数据类：检索管线输入/输出契约（上层状态机与本管线的边界）。"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StrategyConfig(BaseModel):
    """上层传入的策略配置（第5/6步）。"""
    rrf_alpha: float = Field(default=0.8, ge=0.0, le=2.0, description="稠密通道权重系数 α")
    retrieval_pool_size: int = Field(
        default=50, ge=10, le=500, description="候选池大小：普通50 / RECOVER=100"
    )
    enable_query_variant: bool = Field(default=False, description="是否生成查询变体")
    enable_synonym: bool = Field(default=False, description="是否开启同义词扩展")


class SessionState(BaseModel):
    """上层（意图识别+对话状态机）传入的会话状态快照。"""
    constraints: dict[str, Any] = Field(
        default_factory=dict, description="结构化约束，上层已处理 override 清空"
    )
    recovery_mode: bool = Field(default=False, description="连续 miss>=2 触发的 RECOVER 模式标记")
    strategy_config: StrategyConfig = Field(default_factory=StrategyConfig)
    user_raw_query: str = Field(default="", description="用户本轮原始输入文本")


class QueryBundle(BaseModel):
    """第4步产物：主查询 + 变体 + 结构化过滤条件 + 同义词标记。"""
    main_query: str = ""
    variant_queries: list[str] = Field(default_factory=list)
    structured_filters: dict[str, Any] = Field(default_factory=dict)
    synonym_expanded: bool = False


class PipelineOutput(BaseModel):
    """检索管线输出：融合候选 + 最终 Top-10（唯一 parent_asin）。"""
    raw_fused_candidates: list[tuple[str, float]] = Field(
        default_factory=list, description="融合后未重排 (parent_asin, fused_score)"
    )
    reranked_top10: list[str] = Field(
        default_factory=list, description="重排完成 Top-10，唯一不重复"
    )
