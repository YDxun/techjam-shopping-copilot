"""Retrieval-pipeline orchestration entrypoint (task steps 4 -> 5 -> 6).

Pipeline only; no Agent respond/reset, no state machine, no evaluator changes.
The upper layer fills reranked_top10 into the recommendations field of respond().
"""
from __future__ import annotationsimport loggingfrom pathlib import Pathfrom retrieval_pipeline import configfrom retrieval_pipeline.data_access import BlairEmbeddingStore, CatalogStore, load_catalogfrom retrieval_pipeline.models import PipelineOutput, SessionStatefrom retrieval_pipeline.query_builder import QueryBuilderfrom retrieval_pipeline.reranker_module import RerankerModulefrom retrieval_pipeline.retriever_pipeline import RetrieverPipelinelogger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Full steps 4-6 chain: QueryBuilder -> RetrieverPipeline -> RerankerModule."""

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
        """Run steps 4-6 and return PipelineOutput."""
        # step 4: build the query
        bundle = self.query_builder.build(session_state)

        # step 5: three-channel retrieval + RRF fusion
        raw_fused = self.retriever.retrieve(bundle, session_state)

        # step 6: rerank (bge-reranker-v2-m3; on failure degrade to fused-order ranking)
        reranked = self.reranker.rerank(raw_fused, bundle.main_query)

        return PipelineOutput(
            raw_fused_candidates=raw_fused,
            reranked_top10=reranked,
        )


def run_pipeline(session_state: SessionState,
                 catalog_path: str | Path | None = None,
                 blair_path: str | Path | None = None) -> PipelineOutput:
    """Convenience entry: build and run in one call (tests/upper layer can reuse one instance to
        avoid rebuilding the index)."""
    return RetrievalPipeline(catalog_path, blair_path).run(session_state)
