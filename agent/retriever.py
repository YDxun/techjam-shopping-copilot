"""Pillar I：多路由混合检索（BM25 / 类别过滤 / 硬约束 AND / 稠密向量），RRF 融合。

- BM25 路由：SQLite FTS5 + 多字段加权（title/features 高权重）。
- 类别路由：品类词在 category 字段中的命中过滤。
- 硬约束路由：对 hard 约束 token 组做 FTS "AND" 查询（索引级交集），
  保证"必中"候选一定进入池子（提升 HitRate@K）。
- 稠密路由：可选 sentence-transformers 本地 embedding（CPU 可跑）；
  未安装/不可用时自动降级为 BM25，保证脱离外部 API 可运行。
- 融合：Reciprocal Rank Fusion（RRF）+ 并集候选池，交由重排模块精排。
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
from utils import session_utils as su

logger = logging.getLogger(__name__)


class HybridRetriever:
    """混合检索器：索引构建一次，每轮多路由召回 + RRF 融合。"""

    def __init__(self, catalog_path: str | Path, env: EnvConfig | None = None) -> None:
        self.catalog_path = Path(catalog_path)
        self.env = env or EnvConfig.from_env()
        self.backend = self.env.retrieval_backend
        self._conn = sqlite3.connect(":memory:")
        self._products: dict[str, dict] = {}
        self._text_lower: dict[str, str] = {}
        self._cat_lower: dict[str, str] = {}
        self._dense = None          # 惰性加载的稠密模型
        self._dense_matrix = None
        self._build_index()

    # ------------------------------------------------------------------
    def _build_index(self) -> None:
        """FTS5 全文索引（多字段加权），并缓存商品文本用于覆盖度匹配。"""
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
                self._text_lower[asin] = " ".join([title, features, details, categories, store, description]).lower()
                self._cat_lower[asin] = categories.lower()
                if len(batch) >= 1000:
                    cur.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)", batch)
                    batch.clear()
        if batch:
            cur.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)", batch)
        self._conn.commit()
        logger.info("[retriever] indexed %d products (backend=%s)", len(self._products), self.backend)

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
    def search(self, route: IntentRoute, top_k: int = 300, mode: str = "probe") -> list[dict]:
        """多路由召回 + RRF 融合，返回候选（parent_asin + 融合分 + 路由命中标记）。"""
        pool: dict[str, dict] = {}
        self._route_bm25(route, pool, top_k=top_k * 2)
        self._route_category(route, pool, top_k=top_k)
        self._route_constraints(route, pool, top_k=top_k)     # 硬约束 AND（保命中）
        if self.backend in ("dense", "hybrid"):
            self._route_dense(route, pool, top_k=top_k)

        ranked = sorted(pool.values(), key=lambda x: x["rrf"], reverse=True)
        return ranked[:top_k]

    # -- 路由 1：BM25 多字段加权 ------------------------------------------
    def _route_bm25(self, route: IntentRoute, pool: dict, top_k: int) -> None:
        terms = route.query_terms
        if not terms:
            return
        expr = " OR ".join(f'"{t}"' for t in terms[:24])
        try:
            rows = self._conn.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (expr, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            return
        for rank, (asin,) in enumerate(rows, start=1):
            self._accumulate(pool, str(asin), 1.0 / (60.0 + rank), "bm25")

    # -- 路由 2：类别过滤（品类域命中） ------------------------------------
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
            self._accumulate(pool, asin, 1.0 / (60.0 + rank), "category")

    # -- 路由 3：硬约束 AND（保证必中候选进池，Pillar I 高精度过滤） ----------
    def _route_constraints(self, route: IntentRoute, pool: dict, top_k: int) -> None:
        for group in route.hard_groups:
            if not group:
                continue
            expr = " AND ".join(f'"{t}"' for t in group[:6])
            try:
                rows = self._conn.execute(
                    "SELECT parent_asin FROM products WHERE products MATCH ? LIMIT ?",
                    (expr, top_k * 2),
                ).fetchall()
            except sqlite3.OperationalError:
                continue
            for rank, (asin,) in enumerate(rows, start=1):
                asin = str(asin)
                # 与品类域交叉过滤（缩小 + 提升命中精度）
                if route.category_tokens:
                    frac = sum(1 for t in route.category_tokens if t in self._cat_lower.get(asin, ""))
                    if frac / len(route.category_tokens) <= 0.5:
                        continue
                self._accumulate(pool, asin, 1.0 / (10.0 + rank) + 0.1, "constraint")
        # 轻权重"召回补齐"路由：'（group）AND （cat1 OR cat2）'，低 RRF 权重，
        # 只把此前被挤出 LIMIT 的强相关候选补进池（Pillar I 召回保障），不扰动主流排序
        self._route_constraint_recall(route, pool, top_k)

    def _route_constraint_recall(self, route: IntentRoute, pool: dict, top_k: int) -> None:
        """品类折进 SQL 的硬约束召回：'(group) AND (cat1 OR cat2)'，ORDER BY bm25。

        解决高频约束词（如 water resistant）命中过多、目标被 LIMIT 挤出池的问题；
        低 RRF 权重（1/(60+rank)）保证只补漏、不反客为主。
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
                    "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                    (expr, top_k * 3),
                ).fetchall()
            except sqlite3.OperationalError:
                continue
            for rank, (asin,) in enumerate(rows, start=1):
                self._accumulate(pool, str(asin), 1.0 / (60.0 + rank) + 0.02, "constraint_recall")

    # -- 路由 4：稠密向量（可选，离线本地 embedding） -----------------------
    def _route_dense(self, route: IntentRoute, pool: dict, top_k: int) -> None:
        model, matrix, ids = self._ensure_dense()
        if model is None:
            return
        query = " ".join([*route.category_tokens, *route.query_terms]) or "clothing"
        try:
            qv = model.encode([query], normalize_embeddings=True)[0]
            import numpy as np  # 局部导入，避免核心路径依赖
            sims = matrix @ qv
            order = np.argsort(-sims)[:top_k]
            for rank, idx in enumerate(order, start=1):
                self._accumulate(pool, ids[int(idx)], float(sims[idx]) / (60.0 + rank), "dense")
        except Exception as exc:  # 稠密路由任何异常都不影响主流程
            logger.warning("[retriever] dense route failed, fallback to bm25: %s", exc)

    def _ensure_dense(self):
        """惰性加载 embedding 模型 + 全目录向量矩阵；失败返回 (None,None,None)。"""
        if self._dense is not None:
            return self._dense
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            model = SentenceTransformer(self.env.embedding_model)
            ids = list(self._products.keys())
            texts = [self._text_lower[a][:512] for a in ids]
            matrix = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
            matrix = np.asarray(matrix, dtype=np.float32)
            self._dense = (model, matrix, ids)
            logger.info("[retriever] dense route ready: %s (%d dims)", self.env.embedding_model, matrix.shape[1])
        except Exception as exc:
            logger.warning("[retriever] dense backend unavailable (%s); using bm25/category only", exc)
            self._dense = (None, None, None)
        return self._dense

    # -- 融合工具 ----------------------------------------------------------
    @staticmethod
    def _accumulate(pool: dict, asin: str, score: float, source: str) -> None:
        entry = pool.setdefault(asin, {"parent_asin": asin, "rrf": 0.0, "routes": set()})
        entry["rrf"] += score
        entry["routes"].add(source)

    def product(self, asin: str) -> dict | None:
        return self._products.get(asin)

    def text_lower(self, asin: str) -> str:
        return self._text_lower.get(asin, "")

    def close(self) -> None:
        self._conn.close()

