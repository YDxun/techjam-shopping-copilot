"""Module 2: three-channel retrieval + RRF fusion (task step 5).

Channel 1: structured-constraint matching (material/color/size/budget...)
   - normal mode: hard filter, dropping candidates that violate a constraint;
   - RECOVER mode: no dropping; candidates violating a constraint get a score penalty (relaxation
   priority budget>size>material).
Channel 2: weighted BM25 lexical retrieval (rank-bm25; title/features high weight; supports query
variants).
Channel 3: BLaIR dense semantic retrieval (product vectors from the offline npy; at inference only
the user query is encoded; dot-product recall).
Fusion: standard RRF (k=60) with a dense-channel alpha weight; dedup -> truncate to
retrieval_pool_size.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict

import numpy as np

from retrieval_pipeline import config
from retrieval_pipeline.data_access import BlairEmbeddingStore, CatalogStore
from retrieval_pipeline.models import QueryBundle, SessionState
from utils import field_mapping as fm

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9%]+", re.IGNORECASE)
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "the", "this", "to",
    "with", "you", "want", "looking", "im", "i'm", "still", "exploring",
})

# structured-constraint fields -> channel-1 text matching
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
    """Equivalent BM25Okapi implementation used when rank-bm25 is missing (zero-dependency
        fallback)."""

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
    """Multi-field weighted BM25 (step 5 channel 2): one BM25 model per field, merged by weight at
        query time."""

    def __init__(self, catalog: CatalogStore, weights: dict[str, float]) -> None:
        self.catalog = catalog
        self.weights = weights
        self.fields = list(weights.keys())
        self.models: dict[str, object] = {}
        try:
            from rank_bm25 import BM25Okapi  # preferred library
            self._cls = BM25Okapi
        except ImportError:
            logger.warning("[retriever] rank_bm25 not installed; using the built-in BM25Okapi equivalent")  # noqa: E501
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
        """Return {parent_asin: weighted BM25 score}."""
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
    """BLaIR query encoder: encodes only the user query text, never products (product vectors come
        from the offline npy).

    Encoding convention matches scripts/encode_catalog_blair.py exactly (official generate_emb.py):
      - CLS pooling：last_hidden_state[:, 0]；
      - L2 normalization; retrieval uses dot product.
    Load order: transformers AutoModel (canonical BLaIR usage) -> sentence-transformers fallback.
    If neither loads / both fail -> returns None and the dense channel auto-disables (never blocks
    the main flow; environment-aware).
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None          # None=not loaded, False=load failed, otherwise=encoder instance
        self._max_length = 512      # official example max_length=512 (query texts are usually short)  # noqa: E501

    def _ensure(self):
        if self._model is not None:
            return self._model
        # preferred: transformers AutoModel (canonical BLaIR CLS usage)
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
                "[retriever] transformers BLaIR load failed (%s) -> trying sentence-transformers",
                exc,
            )
        # fallback: sentence-transformers (some environments only have it)
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info(
                "[retriever] BLaIR query encoder loaded (sentence-transformers): %s",
                self.model_name,
            )
            return self._model
        except Exception as exc:
            logger.warning("[retriever] BLaIR query encoder unavailable (%s) -> dense channel disabled", exc)  # noqa: E501
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
                vec = torch.nn.functional.normalize(vec, p=2, dim=1)[0]  # L2 normalization
                return vec.detach().cpu().numpy().astype(np.float32)
            vec = model.encode([text], normalize_embeddings=True)[0]
            return np.asarray(vec, dtype=np.float32)
        except Exception as exc:
            logger.warning("[retriever] query encoding failed: %s", exc)
            return None


class RetrieverPipeline:
    """Step 5: three-channel retrieval + RRF fusion."""

    def __init__(self, catalog: CatalogStore, blair_store: BlairEmbeddingStore | None = None,
                 query_encoder: _QueryEncoder | None = None) -> None:
        self.catalog = catalog
        self.blair = blair_store
        self.bm25 = _WeightedBM25Index(catalog, config.BM25_FIELD_WEIGHTS)
        self.encoder = query_encoder or _QueryEncoder(config.BLAIR_QUERY_ENCODER_MODEL)

    # ------------------------------------------------------------------
    def retrieve(self, bundle: QueryBundle, state: SessionState) -> list[tuple[str, float]]:
        """Three-channel recall + RRF fusion -> deduplicated candidate pool (truncated per
            strategy_config)."""
        recovery = state.recovery_mode
        alpha = state.strategy_config.rrf_alpha

        # channel 1: structured-constraint matching
        struct_ranked = self._channel_structured(bundle.structured_filters, recovery)
        # channel 2: weighted BM25 (main query + variants; keep each asin's best score)
        bm25_scores = self._channel_bm25(bundle)
        # channel 3: BLaIR dense (encodes only the main query)
        dense_ranked = self._channel_dense(bundle.main_query)

        # RRF fusion (standard formula: score = sum over channels of 1/(k + rank))
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
        # hard constraint 8: filter IDs not in the catalog (double safety)
        candidates = [(a, s) for a, s in candidates if self.catalog.valid_asin(a)]
        logger.info("[retriever] fused candidates: %d (recovery=%s alpha=%.2f)",
                    len(candidates), recovery, alpha)
        return candidates

    # ------------------------------------------------------------------
    # channel 1: structured-constraint matching (step 5 channel 1)
    # ------------------------------------------------------------------
    def _channel_structured(self, filters: dict, recovery: bool) -> list[tuple[str, int]]:
        """Return [(parent_asin, rank)] sorted by structure score. Hard filter in normal mode;
            penalties in RECOVER.

        Field-aware (field_mapping.json; Pillar I structured-filter precision):
          - text constraints only search the mapped lookup_fields (material ->
          details.Material/features/title/...),
            never scanning all text blindly; budget -> numeric price check (missing price passes
            through; 79% have no price, so no hard filter);
          - brand -> store with missing pass-through (missing_policy=pass).
        """
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
                ok = False
                for v in values:
                    if fm.constraint_hit(key, str(v), None, product=product, text=text) > 0:
                        ok = True
                        break
                if not ok:
                    unmet.append(key)
            for key in _NUMERIC_FILTER_FIELDS:
                if key not in filters:
                    continue
                price = product.get("price")
                if not isinstance(price, (int, float)):
                    continue  # missing price -> pass through (field_mapping budget missing_policy=pass)  # noqa: E501
                limit = float(filters[key])
                if key == "budget_max" and price > limit:
                    unmet.append("budget")
                if key == "budget_min" and price < limit:
                    unmet.append("budget")

            if recovery:
                # penalty scoring: budget relaxes first (smallest penalty) -> material last (largest
                # penalty)
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
    # channel 2: weighted BM25 (step 5 channel 2)
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
    # channel 3: BLaIR dense (step 5 channel 3)
    # ------------------------------------------------------------------
    def _channel_dense(self, query: str) -> list[tuple[str, int]]:
        if self.blair is None or not self.blair.available:
            return []
        qv = self.encoder.encode(query)
        if qv is None:
            return []
        try:
            sims = self.blair.matrix @ qv          # dot-product similarity (vectors already normalized)  # noqa: E501
            top_n = min(len(self.blair.asins), config.BM25_TOP_N)
            order = np.argsort(-sims)[:top_n]
            return [(self.blair.asins[int(i)], rank)
                    for rank, i in enumerate(order)]
        except Exception as exc:
            logger.warning("[retriever] dense channel failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    @staticmethod
    def _rank_dict(score_map: dict[str, float]) -> list[tuple[str, int]]:
        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        return [(a, i) for i, (a, _) in enumerate(ranked)]
