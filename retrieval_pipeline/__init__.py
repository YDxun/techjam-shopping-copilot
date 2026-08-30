"""Retrieval-pipeline package: task steps 4-6 (query building / three-channel retrieval /
    reranking).

This package only implements the retrieval pipeline; it never implements Agent respond/reset, never
implements a state machine, and never modifies the evaluator.
The upper layer (intent recognition + dialogue state machine) passes session_state; this package
returns PipelineOutput.
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
