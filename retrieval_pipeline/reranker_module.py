"""Module 3: reranking (task step 6): BAAI/bge-reranker-v2-m3 cross-encoder rerank.

- prefers loading bge-reranker-v2-m3 via FlagEmbedding (auto-downloaded from HuggingFace);
- device auto-selected cuda / cpu (DEVICE=auto/cpu/cuda);
- OOM / model-load failures -> directly degrade to fused_score ordering (hard constraint 3);
- outputs Top-10 unique parent_asins (hard constraint 8: catalog IDs only).
"""
from __future__ import annotations

import logging

from retrieval_pipeline import config
from retrieval_pipeline.data_access import CatalogStore

logger = logging.getLogger(__name__)


class RerankerModule:
    """Step 6: cross-encoder fine ranking (auto-degrade on failure)."""

    def __init__(self, catalog: CatalogStore, model_name: str | None = None,
                 device: str | None = None) -> None:
        self.catalog = catalog
        self.model_name = model_name or config.RERANKER_MODEL_NAME
        self.device = device or config.DEVICE
        self._model = None          # None=not loaded, False=load failed, otherwise=model instance
        self._load_model()

    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """Load bge-reranker-v2-m3 via FlagEmbedding; on failure set False (degrade)."""
        try:
            from FlagEmbedding import FlagReranker
            device = self._resolve_device()
            use_fp16 = (device == "cuda")
            self._model = FlagReranker(self.model_name, use_fp16=use_fp16, device=device)
            logger.info("[reranker] loaded %s on %s", self.model_name, device)
        except ImportError:
            logger.warning("[reranker] FlagEmbedding not installed -> degrade to fused_score ordering")  # noqa: E501
            self._model = False
        except Exception as exc:
            logger.warning("[reranker] model load failed (%s) -> degrade to fused_score ordering", exc)  # noqa: E501
            self._model = False

    def _resolve_device(self) -> str:
        if self.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        return self.device

    # ------------------------------------------------------------------
    def rerank(self, raw_candidates: list[tuple[str, float]],
               query_text: str) -> list[str]:
        """Fused candidates -> fine ranking -> Top-10 unique parent_asins."""
        if not raw_candidates:
            return []
        # fallback order: descending fused score (used directly when the model is unavailable or
        # rerank fails)
        fallback_order = [a for a, _ in
                          sorted(raw_candidates, key=lambda x: x[1], reverse=True)]

        if self._model is False or not query_text:
            return self._dedup_top10(fallback_order)

        # build (query, product_text) pairs
        pairs: list[tuple[str, str]] = []
        for asin, _ in raw_candidates:
            product = self.catalog.get(asin)
            if product is None:
                continue
            text = self._product_text(product)
            pairs.append((query_text, text))
        if not pairs:
            return self._dedup_top10(fallback_order)

        try:
            scores = self._score_pairs(pairs)          # cross-encoder scoring
            ordered = [a for a, _ in
                       sorted(zip([a for a, _ in raw_candidates], scores, strict=True),
                              key=lambda x: x[1], reverse=True)]
            return self._dedup_top10(ordered)
        except Exception as exc:
            # OOM / other exceptions -> degrade
            logger.warning("[reranker] rerank failed (%s) -> degrade to fused_score ordering", exc)
            return self._dedup_top10(fallback_order)

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Call FlagReranker.compute_score in batches (CPU automatically when no GPU)."""
        scores: list[float] = []
        batch: list[tuple[str, str]] = []
        for pair in pairs:
            batch.append(pair)
            if len(batch) >= config.RERANK_BATCH_SIZE:
                scores.extend(self._call_model(batch))
                batch = []
        if batch:
            scores.extend(self._call_model(batch))
        return scores

    def _call_model(self, batch: list[tuple[str, str]]) -> list[float]:
        result = self._model.compute_score(batch, normalize=True)
        if isinstance(result, float):
            return [result]
        return [float(x) for x in result]

    # ------------------------------------------------------------------
    @staticmethod
    def _product_text(product: dict) -> str:
        parts = [str(product.get("title") or "")]
        features = product.get("features") or []
        if isinstance(features, list):
            parts.extend(str(f) for f in features[:5])
        else:
            parts.append(str(features))
        return " | ".join(p for p in parts if p)

    @staticmethod
    def _dedup_top10(ordered: list[str]) -> list[str]:
        """Dedup + catalog validation + Top-10."""
        seen: list[str] = []
        for a in ordered:
            if a not in seen:
                seen.append(a)
            if len(seen) >= config.RERANK_TOP_K:
                break
        return seen
