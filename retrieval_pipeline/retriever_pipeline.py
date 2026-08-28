"""模块2｜三通道检索 + RRF 融合（赛题第5步）。

通道1：结构化约束匹配（material/color/size/budget…）
   - 普通模式：硬过滤，不满足约束直接筛除；
   - RECOVER 模式：不筛除，对不满足约束的候选施加分数惩罚（放宽优先级 budget>size>material）。
通道2：加权 BM25 词法检索（rank-bm25，title/features 高权重，支持查询变体）。
通道3：BLaIR 稠密语义检索（商品向量来自离线 npy，推理只编码用户查询文本，点积召回）。
多路融合：标准 RRF（k=60），稠密通道带 α 权重；去重 → retrieval_pool_size 截断。
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict

import numpy as np

from retrieval_pipeline import config
from retrieval_pipeline.data_access import BlairEmbeddingStore, CatalogStore
from retrieval_pipeline.models import QueryBundle, SessionState

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9%]+", re.IGNORECASE)
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "the", "this", "to",
    "with", "you", "want", "looking", "im", "i'm", "still", "exploring",
})

# 结构化约束字段 → 通道1文本匹配
_TEXT_FILTER_FIELDS = (
    "material", "color", "size", "style", "brand", "feature", "use_case", "category"
)
_NUMERIC_FILTER_FIELDS = ("budget_max", "budget_min")


def _tokens(text: str) -> list[str]:
    return [
        t.lower()
        for t in _TOKEN_RE.findall(text or "")
        if len(t) > 1 and t.lower() not in _STOPWORDS
    ]


def _field_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{k} {v}" for k, v in value.items() if v not in (None, ""))
    if isinstance(value, list):
        return " ".join(str(x) for x in value if x not in (None, ""))
    return str(value)


class _BM25OkapiFallback:
    """rank-bm25 缺失时的等价 BM25Okapi 实现（零依赖兜底）。"""

    def __init__(self, corpus: list[list[str]]) -> None:
        self.corpus = corpus
        self.doc_freqs: list[dict[str, int]] = []
        self.idf: dict[str, float] = {}
        self.doc_len = np.array([len(d) for d in corpus], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if self.doc_len.size else 0.0
        self.k1, self.b = 1.5, 0.75
        df: dict[str, int] = defaultdict(int)
        for doc in corpus:
            freqs: dict[str, int] = defaultdict(int)
            for t in doc:
                freqs[t] += 1
            self.doc_freqs.append(freqs)
            for t in freqs:
                df[t] += 1
        for t, n in df.items():
            self.idf[t] = float(np.log(1 + (len(corpus) - n + 0.5) / (n + 0.5)))

    def get_scores(self, query: list[str]) -> np.ndarray:
        scores = np.zeros(len(self.corpus), dtype=np.float64)
        q_terms = set(query)
        avgdl = self.avgdl or 1.0
        for i, (freqs, dl) in enumerate(zip(self.doc_freqs, self.doc_len, strict=True)):
            denom = self.k1 * (1 - self.b + self.b * dl / avgdl)
            total = 0.0
            for t in q_terms:
                f = freqs.get(t, 0)
                if f:
                    total += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + denom)
            scores[i] = total
        return scores


class _WeightedBM25Index:
    """多字段加权 BM25（第5步-通道2）：每字段一个 BM25 模型，查询时按权重合并。"""

    def __init__(self, catalog: CatalogStore, weights: dict[str, float]) -> None:
        self.catalog = catalog
        self.weights = weights
        self.fields = list(weights.keys())
        self.models: dict[str, object] = {}
        try:
            from rank_bm25 import BM25Okapi  # 首选库
            self._cls = BM25Okapi
        except ImportError:
            logger.warning("[retriever] rank_bm25 未安装，使用内置 BM25Okapi 等价实现")
            self._cls = None
        for field in self.fields:
            corpus = [
                _tokens(_field_text(catalog.products[asin].get(field)))
                for asin in catalog.ids
            ]
            self.models[field] = (self._cls(corpus) if self._cls is not None
                                  else _BM25OkapiFallback(corpus))
        logger.info("[retriever] weighted BM25 index built (fields=%s)", self.fields)

    def score(self, query: str) -> dict[str, float]:
        """返回 {parent_asin: 加权 BM25 分}。"""
        q_tokens = _tokens(query)
        if not q_tokens:
            return {}
        out: dict[str, float] = defaultdict(float)
        for field in self.fields:
            w = self.weights.get(field, 0.0)
            if w <= 0:
                continue
            scores = self.models[field].get_scores(q_tokens)
            for idx, asin in enumerate(self.catalog.ids):
                out[asin] += w * float(scores[idx])
        return dict(out)


class _QueryEncoder:
    """BLaIR 查询编码器：只编码用户查询文本，不处理商品（商品向量来自离线 npy）。

    编码规范与离线脚本 scripts/encode_catalog_blair.py 完全一致（官方 generate_emb.py）：
      - CLS pooling：last_hidden_state[:, 0]；
      - L2 归一化，检索用点积。
    加载顺序：transformers AutoModel（BLaIR 规范用法）→ sentence-transformers 兜底。
    两者都不可用/加载失败 → 返回 None，稠密通道自动禁用（不阻塞主流程，环境自感知）。
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None          # None=未加载, False=加载失败, 其它=编码器实例
        self._max_length = 512      # 官方示例 max_length=512（查询文本通常很短）

    def _ensure(self):
        if self._model is not None:
            return self._model
        # 首选：transformers AutoModel（BLaIR CLS 规范用法）
        try:
            import os

            import torch
            from transformers import AutoModel, AutoTokenizer
            torch.set_num_threads(max(1, os.cpu_count() or 8))
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name)
            model.eval()
            self._model = {"tokenizer": tokenizer, "model": model}
            logger.info(
                "[retriever] BLaIR query encoder loaded (transformers): %s", self.model_name
            )
            return self._model
        except Exception as exc:
            logger.warning(
                "[retriever] transformers BLaIR 加载失败（%s）→ 尝试 sentence-transformers",
                exc,
            )
        # 兜底：sentence-transformers（部分环境只装了它）
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info(
                "[retriever] BLaIR query encoder loaded (sentence-transformers): %s",
                self.model_name,
            )
            return self._model
        except Exception as exc:
            logger.warning("[retriever] BLaIR 查询编码器不可用（%s）→ 稠密通道禁用", exc)
            self._model = False
        return self._model

    def encode(self, text: str) -> np.ndarray | None:
        model = self._ensure()
        if model is False or not text:
            return None
        try:
            if isinstance(model, dict):
                import torch
                inputs = model["tokenizer"](
                    [text], padding=True, truncation=True, max_length=self._max_length,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    last_hidden = model["model"](**inputs, return_dict=True).last_hidden_state
                vec = last_hidden[:, 0]                                  # CLS pooling
                vec = torch.nn.functional.normalize(vec, p=2, dim=1)[0]  # L2 归一化
                return vec.detach().cpu().numpy().astype(np.float32)
            vec = model.encode([text], normalize_embeddings=True)[0]
            return np.asarray(vec, dtype=np.float32)
        except Exception as exc:
            logger.warning("[retriever] 查询编码失败: %s", exc)
            return None


class RetrieverPipeline:
    """第5步：三通道检索 + RRF 融合。"""

    def __init__(self, catalog: CatalogStore, blair_store: BlairEmbeddingStore | None = None,
                 query_encoder: _QueryEncoder | None = None) -> None:
        self.catalog = catalog
        self.blair = blair_store
        self.bm25 = _WeightedBM25Index(catalog, config.BM25_FIELD_WEIGHTS)
        self.encoder = query_encoder or _QueryEncoder(config.BLAIR_QUERY_ENCODER_MODEL)

    # ------------------------------------------------------------------
    def retrieve(self, bundle: QueryBundle, state: SessionState) -> list[tuple[str, float]]:
        """三通道召回 + RRF 融合 → 去重候选池（按 strategy_config 截断）。"""
        recovery = state.recovery_mode
        alpha = state.strategy_config.rrf_alpha

        # 通道1：结构化约束匹配
        struct_ranked = self._channel_structured(bundle.structured_filters, recovery)
        # 通道2：加权 BM25（主查询 + 变体，取每个 asin 的最高分）
        bm25_scores = self._channel_bm25(bundle)
        # 通道3：BLaIR 稠密（只编码主查询）
        dense_ranked = self._channel_dense(bundle.main_query)

        # RRF 融合（标准公式：score = Σ_channel 1/(k + rank)）
        fused: dict[str, float] = defaultdict(float)
        for asin, rank in struct_ranked:
            fused[asin] += 1.0 / (config.RRF_K + rank)
        bm25_ranked = self._rank_dict(bm25_scores)
        for asin, rank in bm25_ranked:
            fused[asin] += 1.0 / (config.RRF_K + rank)
        if dense_ranked:
            for asin, rank in dense_ranked:
                fused[asin] += alpha * 1.0 / (config.RRF_K + rank)

        pool_size = state.strategy_config.retrieval_pool_size
        candidates = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:pool_size]
        # 硬性约束 8：过滤不在目录中的 ID（双保险）
        candidates = [(a, s) for a, s in candidates if self.catalog.valid_asin(a)]
        logger.info("[retriever] fused candidates: %d (recovery=%s alpha=%.2f)",
                    len(candidates), recovery, alpha)
        return candidates

    # ------------------------------------------------------------------
    # 通道1：结构化约束匹配（第5步-通道1）
    # ------------------------------------------------------------------
    def _channel_structured(self, filters: dict, recovery: bool) -> list[tuple[str, int]]:
        """返回 [(parent_asin, rank)]，按结构分排序。普通模式硬过滤；RECOVER 改为惩罚。"""
        if not filters:
            return []
        scored: list[tuple[str, float]] = []
        for asin in self.catalog.ids:
            product = self.catalog.products[asin]
            text = self.catalog.searchable(asin)
            unmet: list[str] = []
            for key in _TEXT_FILTER_FIELDS:
                if key not in filters:
                    continue
                value = filters[key]
                values = value if isinstance(value, (list, tuple)) else [value]
                if not any(str(v).lower() in text for v in values):
                    unmet.append(key)
            for key in _NUMERIC_FILTER_FIELDS:
                if key not in filters:
                    continue
                price = product.get("price")
                if not isinstance(price, (int, float)):
                    unmet.append("budget")
                    continue
                limit = float(filters[key])
                if key == "budget_max" and price > limit:
                    unmet.append("budget")
                if key == "budget_min" and price < limit:
                    unmet.append("budget")

            if recovery:
                # 惩罚打分：budget 最先放宽（惩罚最小）→ material 最后放宽（惩罚最大）
                penalty = sum(
                    config.STRUCT_UNMET_PENALTY.get(u, config.STRUCT_UNMET_PENALTY["other"])
                    for u in unmet
                )
                scored.append((asin, config.STRUCT_BASE_SCORE - penalty))
            else:
                if not unmet:
                    scored.append((asin, config.STRUCT_BASE_SCORE))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(a, i) for i, (a, _) in enumerate(scored)]

    # ------------------------------------------------------------------
    # 通道2：加权 BM25（第5步-通道2）
    # ------------------------------------------------------------------
    def _channel_bm25(self, bundle: QueryBundle) -> dict[str, float]:
        queries = [bundle.main_query, *bundle.variant_queries]
        best: dict[str, float] = {}
        for q in queries:
            for asin, score in self.bm25.score(q).items():
                if score > best.get(asin, float("-inf")):
                    best[asin] = score
        return dict(sorted(best.items(), key=lambda x: x[1], reverse=True)[: config.BM25_TOP_N])

    # ------------------------------------------------------------------
    # 通道3：BLaIR 稠密（第5步-通道3）
    # ------------------------------------------------------------------
    def _channel_dense(self, query: str) -> list[tuple[str, int]]:
        if self.blair is None or not self.blair.available:
            return []
        qv = self.encoder.encode(query)
        if qv is None:
            return []
        try:
            sims = self.blair.matrix @ qv          # 点积相似度（向量已归一化）
            top_n = min(len(self.blair.asins), config.BM25_TOP_N)
            order = np.argsort(-sims)[:top_n]
            return [(self.blair.asins[int(i)], rank)
                    for rank, i in enumerate(order)]
        except Exception as exc:
            logger.warning("[retriever] 稠密通道失败: %s", exc)
            return []

    # ------------------------------------------------------------------
    @staticmethod
    def _rank_dict(score_map: dict[str, float]) -> list[tuple[str, int]]:
        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        return [(a, i) for i, (a, _) in enumerate(ranked)]
