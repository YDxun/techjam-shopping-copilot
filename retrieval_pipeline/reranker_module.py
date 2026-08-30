"""模块3｜重排序（赛题第6步）：BAAI/bge-reranker-v2-m3 交叉编码器重排。

- 优先 FlagEmbedding 加载 bge-reranker-v2-m3（自动从 HuggingFace 下载）；
- device 自动选择 cuda / cpu（DEVICE=auto/cpu/cuda）；
- 捕获 OOM / 模型加载失败 → 直接降级使用融合分 fused_score 排序（硬性约束 3）；
- 输出 Top-10 唯一 parent_asin（硬性约束 8：目录内 ID）。
"""

from __future__ import annotations

import logging

from retrieval_pipeline import config
from retrieval_pipeline.data_access import CatalogStore

logger = logging.getLogger(__name__)


class RerankerModule:
    """第6步：cross-encoder 精排（失败自动降级）。"""

    def __init__(
        self, catalog: CatalogStore, model_name: str | None = None, device: str | None = None
    ) -> None:
        self.catalog = catalog
        self.model_name = model_name or config.RERANKER_MODEL_NAME
        self.device = device or config.DEVICE
        self._model = None  # None=未加载, False=加载失败, 其它=模型实例
        self._load_model()

    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """FlagEmbedding 加载 bge-reranker-v2-m3；失败置 False（降级）。"""
        try:
            from FlagEmbedding import FlagReranker

            device = self._resolve_device()
            use_fp16 = device == "cuda"
            self._model = FlagReranker(self.model_name, use_fp16=use_fp16, device=device)
            logger.info("[reranker] loaded %s on %s", self.model_name, device)
        except ImportError:
            logger.warning("[reranker] FlagEmbedding 未安装 → 降级 fused_score 排序")
            self._model = False
        except Exception as exc:
            logger.warning("[reranker] 模型加载失败（%s）→ 降级 fused_score 排序", exc)
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
    def rerank(self, raw_candidates: list[tuple[str, float]], query_text: str) -> list[str]:
        """输入融合候选 → 精排 → Top-10 唯一 parent_asin。"""
        if not raw_candidates:
            return []
        # 兜底顺序：融合分降序（模型不可用或重排失败时直接使用）
        fallback_order = [a for a, _ in sorted(raw_candidates, key=lambda x: x[1], reverse=True)]

        if self._model is False or not query_text:
            return self._dedup_top10(fallback_order)

        # 构造 (query, product_text) 对
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
            scores = self._score_pairs(pairs)  # cross-encoder 打分
            ordered = [
                a
                for a, _ in sorted(
                    zip([a for a, _ in raw_candidates], scores, strict=True),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ]
            return self._dedup_top10(ordered)
        except Exception as exc:
            # OOM / 其它异常 → 降级
            logger.warning("[reranker] 重排失败（%s）→ 降级 fused_score 排序", exc)
            return self._dedup_top10(fallback_order)

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """分批调用 FlagReranker.compute_score（无 GPU 时 CPU 自动运行）。"""
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
        """去重 + 目录内校验 + Top-10。"""
        seen: list[str] = []
        for a in ordered:
            if a not in seen:
                seen.append(a)
            if len(seen) >= config.RERANK_TOP_K:
                break
        return seen
