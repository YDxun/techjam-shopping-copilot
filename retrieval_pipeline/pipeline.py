"""检索管线编排入口（赛题第4步→第5步→第6步）。

仅做检索管线；不实现 Agent respond/reset、不实现状态机、不修改评测器。
上层拿到 reranked_top10 填入 respond() 的 recommendations 字段。
"""
from __future__ import annotations

import logging
from pathlib import Path

from retrieval_pipeline import config
from retrieval_pipeline.data_access import BlairEmbeddingStore, CatalogStore, load_catalog
from retrieval_pipeline.models import PipelineOutput, SessionState
from retrieval_pipeline.query_builder import QueryBuilder
from retrieval_pipeline.reranker_module import RerankerModule
from retrieval_pipeline.retriever_pipeline import RetrieverPipeline

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """第4-6步完整链路：QueryBuilder → RetrieverPipeline → RerankerModule。"""

    def __init__(self, catalog_path: str | Path | None = None,
                 blair_path: str | Path | None = None) -> None:
        catalog_path = Path(catalog_path or config.PRODUCT_CATALOG_PATH)
        blair_path = Path(blair_path or config.BLAIR_OFFLINE_EMBEDDING_PATH)

        self.catalog: CatalogStore = load_catalog(catalog_path)
        self.blair: BlairEmbeddingStore | None = BlairEmbeddingStore.load(blair_path)

        self.query_builder = QueryBuilder()
        self.retriever = RetrieverPipeline(self.catalog, self.blair)
        self.reranker = RerankerModule(self.catalog)

    # ------------------------------------------------------------------
    def run(self, session_state: SessionState) -> PipelineOutput:
        """执行第4-6步，返回 PipelineOutput。"""
        # 第4步：构建查询
        bundle = self.query_builder.build(session_state)

        # 第5步：三通道检索 + RRF 融合
        raw_fused = self.retriever.retrieve(bundle, session_state)

        # 第6步：重排（bge-reranker-v2-m3，失败降级 fused 排序）
        reranked = self.reranker.rerank(raw_fused, bundle.main_query)

        return PipelineOutput(
            raw_fused_candidates=raw_fused,
            reranked_top10=reranked,
        )


def run_pipeline(session_state: SessionState,
                 catalog_path: str | Path | None = None,
                 blair_path: str | Path | None = None) -> PipelineOutput:
    """便捷入口：一次性构建并运行（测试/上层可复用同一实例以省去重复建索引）。"""
    return RetrievalPipeline(catalog_path, blair_path).run(session_state)
