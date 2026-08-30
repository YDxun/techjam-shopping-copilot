"""Pillar I: multi-route hybrid retrieval (BM25 / category filter / hard-constraint AND / dense
    vectors) with RRF fusion.

- BM25 route: SQLite FTS5 with multi-field weighting (title/features get higher weight).
- Category route: filters by category tokens appearing in the category field.
- Hard-constraint route: FTS "AND" query over the hard-constraint token groups (index-level
intersection),
   guaranteeing must-hit candidates enter the pool (raises HitRate@K).
- Dense route: BLaIR (hyp1231/blair-roberta-large) offline pre-computed product vectors (npy) +
   at inference only the user query is encoded (utils/blair.py, CLS pooling + L2 + dot product);
   missing offline npy / unavailable encoder / any exception -> auto-fallback to BM25
   (environment-aware).
- Fusion: Reciprocal Rank Fusion (RRF) over the union candidate pool, handed to the reranker for
fine ranking.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from pathlib import Path
from typing import Any

from agent.intent_router import IntentRoute
from config.env_config import EnvConfig
from utils import blair as blair_utils
from utils import session_utils as su
from utils import shelf as shelf_utils
from utils.circuit_breaker import PhaseCircuitBreaker

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever: builds the index once, then per-turn multi-route recall + RRF fusion."""

    def __init__(
        self, catalog_path: str | Path, env: EnvConfig | None = None, backend: str | None = None
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.env = env or EnvConfig.from_env()
        # prefer the explicitly passed backend (runtime_controller already resolved auto ->
        # hybrid/bm25)
        self.backend = backend or self.env.retrieval_backend
        if self.backend == "auto":
            self.backend = "hybrid" if self._dense_backend_available() else "bm25"
        elif self.backend in ("dense", "hybrid") and not self._dense_backend_available():
            self.backend = "bm25"  # environment-aware: auto-fallback to BM25 when BLaIR dense is unavailable  # noqa: E501
        self._conn = sqlite3.connect(":memory:")
        self._products: dict[str, dict] = {}
        self._text_lower: dict[str, str] = {}
        self._cat_lower: dict[str, str] = {}
        self._shelf_of: dict[str, str] = {}
        self._by_shelf: dict[str, list[str]] = {}
        self._dense = None  # lazily loaded dense model
        self._dense_breaker = PhaseCircuitBreaker(
            "dense", failure_threshold=2
        )  # P1: trips after consecutive failures -> hybrid -> bm25
        self._dense_matrix = None
        self._retrieval_cfg = self.env.retrieval  # Step 1: retrieval knobs moved into config
        self._bm25_weights_sql = (
            "bm25(products, "
            + ", ".join(str(float(w)) for w in self._retrieval_cfg.bm25_field_weights)
            + ")"
        )
        self._build_index()

    # ------------------------------------------------------------------
    def _build_index(self) -> None:
        """FTS5 full-text index (multi-field weighted); caches product text for coverage."""
        cur = self._conn.cursor()
        cur.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, features, details, categories, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                p = json.loads(line)
                asin = str(p["parent_asin"])
                title = self._text(p.get("title"))
                features = self._text(p.get("features"))
                details = self._text(p.get("details"))
                categories = self._text(p.get("categories"))
                store = self._text(p.get("store"))
                description = self._text(p.get("description"))
                batch.append((asin, title, features, details, categories, store, description))
                self._products[asin] = p
                self._text_lower[asin] = " ".join(
                    [title, features, details, categories, store, description]
                ).lower()
                self._cat_lower[asin] = categories.lower()
                if len(batch) >= 1000:
                    cur.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)", batch)
                    batch.clear()
        if batch:
            cur.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)", batch)
        self._conn.commit()
        # Shelf index: turn-1 category -> candidate shelf (task mechanic: the target is always
        # inside the shelf, so filtering costs zero recall)
        self._shelf_of, self._by_shelf = shelf_utils.build_shelf_index(self._products.values())
        logger.info(
            "[retriever] indexed %d products (backend=%s)", len(self._products), self.backend
        )

    @staticmethod
    def _spec_available(name: str) -> bool:
        import importlib.util

        return importlib.util.find_spec(name) is not None

    def _dense_backend_available(self) -> bool:
        """Environment awareness: whether the BLaIR dense channel is truly usable (encoder
            importable + offline npy present).

        The query encoder is importable (transformers or sentence-transformers) and the offline
        product vectors
        file exists -> dense channel usable; otherwise fall back to BM25. No real model loading here
        (avoids startup cost;
        actual load failures are still caught by _ensure_dense).
        """
        enc_ok = self._spec_available("transformers") or self._spec_available(
            "sentence_transformers"
        )
        if not enc_ok:
            return False
        path = Path(self.env.blair_offline_embedding_path)
        emb = path if path.suffix == ".npy" else path.with_suffix(".npy")
        return emb.exists()

    @staticmethod
    def _text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, dict):
            return " ".join(f"{k} {v}" for k, v in value.items() if v not in (None, ""))
        if isinstance(value, list):
            return " ".join(str(v) for v in value if v not in (None, ""))
        return str(value)

    # ------------------------------------------------------------------
    def search(
        self,
        route: IntentRoute,
        top_k: int = 300,
        mode: str = "probe",
        shelf: str | None = None,
    ) -> list[dict]:
        """Multi-route recall + RRF fusion; returns candidates (parent_asin + fusion score +
            route-hit flags)."""
        pool: dict[str, dict] = {}
        self._route_bm25(route, pool, top_k=top_k * self._retrieval_cfg.bm25_limit_mult)
        self._route_category(route, pool, top_k=top_k)
        self._route_constraints(route, pool, top_k=top_k)  # hard-constraint AND (guarantees hits)
        if self.backend in ("dense", "hybrid"):
            self._route_dense(route, pool, top_k=top_k, mode=mode)

        # Shelf hard filter (optional): keep only products inside the category shelf (target is
        # always inside, zero recall loss).
        # On any matching failure we must fall back to "no filter" -- no exception may miss the
        # whole session.
        if shelf:
            try:
                shelf_key = shelf_utils.match_shelf(shelf, self._by_shelf)
                if shelf_key is not None:
                    allowed = set(self._by_shelf[shelf_key])
                    pool = {asin: entry for asin, entry in pool.items() if asin in allowed}
            except Exception:
                logger.warning("[retriever] shelf filter failed, skip filtering")
        ranked = sorted(pool.values(), key=lambda x: x["rrf"], reverse=True)
        return ranked[:top_k]

    # -- Route 1: weighted multi-field BM25 ------------------------------------------
    def _route_bm25(self, route: IntentRoute, pool: dict, top_k: int) -> None:
        terms = route.query_terms
        if not terms:
            return
        expr = " OR ".join(f'"{t}"' for t in terms[:24])
        try:
            rows = self._conn.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                f"ORDER BY {self._bm25_weights_sql} LIMIT ?",
                (expr, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            return
        for rank, (asin,) in enumerate(rows, start=1):
            self._accumulate(pool, str(asin), 1.0 / (self._retrieval_cfg.rrf_k + rank), "bm25")

    # -- Route 2: category filter (domain hit) ------------------------------------
    def _route_category(self, route: IntentRoute, pool: dict, top_k: int) -> None:
        if not route.category_tokens:
            return
        hits: list[tuple[str, float]] = []
        for asin, cat in self._cat_lower.items():
            frac = sum(1 for t in route.category_tokens if t in cat) / len(route.category_tokens)
            if frac > 0.5:
                rating = su.safe_float(self._products[asin].get("rating_number"), 0.0)
                hits.append((asin, frac + math.log1p(rating) * 0.01))
        hits.sort(key=lambda x: x[1], reverse=True)
        for rank, (asin, _) in enumerate(hits[:top_k], start=1):
            self._accumulate(pool, asin, 1.0 / (self._retrieval_cfg.rrf_k + rank), "category")

    # -- Route 3: hard-constraint AND (guarantees must-hit candidates in pool; Pillar I precision
    # filtering) ----------
    def _route_constraints(self, route: IntentRoute, pool: dict, top_k: int) -> None:
        for group in route.hard_groups:
            if not group:
                continue
            expr = " AND ".join(f'"{t}"' for t in group[:6])
            try:
                rows = self._conn.execute(
                    "SELECT parent_asin FROM products WHERE products MATCH ? LIMIT ?",
                    (expr, top_k * self._retrieval_cfg.bm25_limit_mult),
                ).fetchall()
            except sqlite3.OperationalError:
                continue
            for rank, (asin,) in enumerate(rows, start=1):
                asin = str(asin)
                # cross-filter with the category domain (narrows the pool and raises precision)
                if route.category_tokens:
                    frac = sum(
                        1 for t in route.category_tokens if t in self._cat_lower.get(asin, "")
                    )
                    if frac / len(route.category_tokens) <= 0.5:
                        continue
                self._accumulate(
                    pool,
                    asin,
                    1.0 / (self._retrieval_cfg.rrf_constraint_k + rank) + 0.1,
                    "constraint",
                )
        # Low-weight "recall top-up" route: '(group) AND (cat1 OR cat2)' with a low RRF weight,
        # it only tops up strongly relevant candidates that were pushed out of LIMIT (Pillar I
        # recall guarantee) without disturbing the main order
        self._route_constraint_recall(route, pool, top_k)

    def _route_constraint_recall(self, route: IntentRoute, pool: dict, top_k: int) -> None:
        """Hard-constraint recall with the category folded into SQL: '(group) AND (cat1 OR cat2)',
            ORDER BY bm25.

        Solves high-frequency constraint words (e.g. water resistant) matching too much and the
        target being pushed out of the pool by LIMIT;
        the low RRF weight (1/(60+rank)) ensures it only tops up misses and never dominates.
        """
        if not route.category_tokens:
            return
        for group in route.hard_groups:
            if not group:
                continue
            cat_expr = " OR ".join(f'"{t}"' for t in route.category_tokens[:4])
            group_expr = " AND ".join(f'"{t}"' for t in group[:6])
            expr = f"({group_expr}) AND ({cat_expr})"
            try:
                rows = self._conn.execute(
                    "SELECT parent_asin FROM products WHERE products MATCH ? "
                    f"ORDER BY {self._bm25_weights_sql} LIMIT ?",
                    (expr, top_k * self._retrieval_cfg.recall_limit_mult),
                ).fetchall()
            except sqlite3.OperationalError:
                continue
            for rank, (asin,) in enumerate(rows, start=1):
                self._accumulate(
                    pool,
                    str(asin),
                    1.0 / (self._retrieval_cfg.rrf_k + rank) + 0.02,
                    "constraint_recall",
                )

    # -- Route 4: dense vectors (optional, offline local embeddings) -----------------------
    def _route_dense(self, route: IntentRoute, pool: dict, top_k: int, mode: str = "probe") -> None:
        """BLaIR dense semantic recall (mode-adaptive + hard-constraint re-check).

        - Enabled only in recover (miss streak >= 2, needs broader recall): under probe/exploit,
        semantic candidates disturb
          the aligned rule order (A/B: boundary/browsing MRR drops when dense is always on);
        - Hard-constraint coverage re-check: dense candidates violating any hard group are skipped
        (blocks semantic noise);
        - weight 0.5x makes it a pure "recall supplement", never dominant.
        """
        if mode != "recover":
            return
        if self._dense_breaker.open:  # P1: already tripped -> skip dense now (in-process hybrid -> bm25)  # noqa: E501
            return
        encoder, store = self._ensure_dense()
        if encoder is None or store is None:
            return
        query = " ".join([*route.category_tokens, *route.query_terms]) or "clothing"
        try:
            qv = encoder.encode(query)
            if qv is None:
                return
            import numpy as np  # local import so the core path has no hard dependency

            sims = store.matrix @ qv
            order = np.argsort(-sims)[:top_k]
            for rank, idx in enumerate(order, start=1):
                asin = store.asins[int(idx)]
                if route.hard_groups and not self._dense_passes_hard(route.hard_groups, asin):
                    continue  # hard-constraint re-check: skip when not satisfied
                self._accumulate(
                    pool,
                    asin,
                    self._retrieval_cfg.dense_weight
                    * float(sims[idx])
                    / (self._retrieval_cfg.rrf_k + rank),
                    "dense",
                )
            self._dense_breaker.record_success()  # P1: one success clears the failure streak
        except Exception as exc:  # any dense-route exception never affects the main flow (environment-aware fallback)  # noqa: E501
            logger.warning("[retriever] dense route failed, fallback to bm25: %s", exc)
            # P1: consecutive failures trip the breaker -> disable dense in-process (hybrid ->
            # bm25), no more per-turn retries
            if self._dense_breaker.record_failure(str(exc)):
                self._dense = (None, None)

    def _dense_passes_hard(self, hard_groups: list[tuple[str, ...]], asin: str) -> bool:
        """Hard-constraint coverage re-check for dense candidates: every hard group (AND within
            group) must fully hit."""
        text = self._text_lower.get(asin, "")
        if not text:
            return False
        return all(all(tok in text for tok in group) for group in hard_groups)

    def _ensure_dense(self):
        """Lazily load the BLaIR query encoder + offline product-vector npy; on failure return
            (None, None).

        Product vectors are the offline output of scripts/encode_catalog_blair.py; at inference only
        the query text is encoded;
        missing file / unavailable model -> (None, None), so the dense route auto-skips (fallback to
        BM25).
        """
        if self._dense is not None:
            return self._dense
        store = blair_utils.BlairEmbeddingStore.load(self.env.blair_offline_embedding_path)
        encoder = None
        if store is not None:
            encoder = blair_utils.BlairQueryEncoder(self.env.blair_query_encoder_model)
            if not encoder.ready:
                logger.warning("[retriever] BLaIR query encoder unavailable; dense channel disabled (fallback to BM25)")  # noqa: E501
                encoder = None
        if encoder is None or store is None:
            self._dense = (None, None)
        else:
            self._dense = (encoder, store)
            logger.info(
                "[retriever] dense route ready: %s (%d dims)",
                self.env.blair_query_encoder_model,
                store.dim,
            )
        return self._dense

    @staticmethod
    def _accumulate(pool: dict, asin: str, score: float, source: str) -> None:
        entry = pool.setdefault(asin, {"parent_asin": asin, "rrf": 0.0, "routes": set()})
        entry["rrf"] += score
        entry["routes"].add(source)

    def product(self, asin: str) -> dict | None:
        return self._products.get(asin)

    def iter_products(self) -> tuple[dict, ...]:
        """Expose a read-only snapshot for dialogue question statistics."""
        return tuple(self._products.values())

    def text_lower(self, asin: str) -> str:
        return self._text_lower.get(asin, "")

    def close(self) -> None:
        self._conn.close()
