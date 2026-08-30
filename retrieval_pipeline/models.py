"""Pydantic data classes: retrieval-pipeline input/output contract (the boundary between the upper
    state machine and this pipeline)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StrategyConfig(BaseModel):
    """Strategy config passed from the upper layer (steps 5/6)."""
    rrf_alpha: float = Field(default=0.8, ge=0.0, le=2.0, description="dense-channel weight alpha")
    retrieval_pool_size: int = Field(
        default=50, ge=10, le=500, description="candidate-pool size: normal 50 / RECOVER 100"
    )
    enable_query_variant: bool = Field(default=False, description="whether to generate query variants")  # noqa: E501
    enable_synonym: bool = Field(default=False, description="whether to enable synonym expansion")


class SessionState(BaseModel):
    """Session-state snapshot passed in by the upper layer (intent recognition + dialogue state
        machine)."""
    constraints: dict[str, Any] = Field(
        default_factory=dict, description="structured constraints; override clears them upstream"
    )
    recovery_mode: bool = Field(default=False, description="RECOVER-mode flag triggered by a miss streak >= 2")  # noqa: E501
    strategy_config: StrategyConfig = Field(default_factory=StrategyConfig)
    user_raw_query: str = Field(default="", description="the user's raw input text this turn")


class QueryBundle(BaseModel):
    """Step-4 output: main query + variants + structured filters + synonym flag."""
    main_query: str = ""
    variant_queries: list[str] = Field(default_factory=list)
    structured_filters: dict[str, Any] = Field(default_factory=dict)
    synonym_expanded: bool = False


class PipelineOutput(BaseModel):
    """Retrieval-pipeline output: fused candidates + final Top-10 (unique parent_asins)."""
    raw_fused_candidates: list[tuple[str, float]] = Field(
        default_factory=list, description="fused, not-yet-reranked (parent_asin, fused_score)"
    )
    reranked_top10: list[str] = Field(
        default_factory=list, description="reranked Top-10, unique and non-repeating"
    )
