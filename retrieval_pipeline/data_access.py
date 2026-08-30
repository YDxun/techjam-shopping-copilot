"""Data access: only loads offline pre-computed data (product catalog + BLaIR product-vector npy).

Important:
- BLaIR product vectors are produced by a separate "offline preprocessing script" (this module never
implements full product vectorization);
- this file only provides "load" interfaces; at inference only the user query text is embedded (see
retriever_pipeline).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class CatalogStore:
    """In-memory store of the frozen competition catalog (parent_asin -> metadata + retrieval
        text)."""

    def __init__(self, products: dict[str, dict], search_text: dict[str, str]) -> None:
        self.products = products
        self.search_text = search_text          # asin -> lowercase retrieval text (title/features/...)  # noqa: E501
        self.ids = list(products.keys())
        self.id_set = set(products.keys())

    def valid_asin(self, asin: str) -> bool:
        """Hard constraint 8: only real parent_asins from the catalog are allowed."""
        return asin in self.id_set

    def get(self, asin: str) -> dict | None:
        return self.products.get(asin)

    def searchable(self, asin: str) -> str:
        return self.search_text.get(asin, "")


def load_catalog(path: str | Path) -> CatalogStore:
    """Load the frozen competition catalog (read-only; never modified)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"catalog does not exist: {path} (use PRODUCT_CATALOG_PATH to point elsewhere)")  # noqa: E501
    products: dict[str, dict] = {}
    search_text: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            p = json.loads(line)
            asin = str(p["parent_asin"])
            products[asin] = p
            search_text[asin] = _build_search_text(p).lower()
    logger.info("[data_access] catalog loaded: %d products from %s", len(products), path)
    return CatalogStore(products, search_text)


def _build_search_text(p: dict) -> str:
    parts: list[str] = []
    for key in ("title", "features", "description", "categories", "details", "store"):
        v = p.get(key)
        if isinstance(v, dict):
            parts.append(" ".join(f"{k} {x}" for k, x in v.items() if x not in (None, "")))
        elif isinstance(v, list):
            parts.append(" ".join(str(x) for x in v if x not in (None, "")))
        elif v is not None:
            parts.append(str(v))
    return " ".join(parts)


class BlairEmbeddingStore:
    """BLaIR product-vector store: only loads the offline pre-computed npy; never generates vectors.

    npy file-format convention (produced by the offline preprocessing script):
      - embeds.npy : float32 [N, dim], row order matches asins.npy
      - asins.npy  : object/str array [N]
    """

    def __init__(self, matrix: np.ndarray, asins: list[str], asin_index: dict[str, int]) -> None:
        self.matrix = matrix                    # [N, dim]
        self.asins = asins                      # list[str]
        self.asin_index = asin_index            # asin -> row index
        self.dim = matrix.shape[1] if matrix.ndim == 2 else 0

    @property
    def available(self) -> bool:
        return self.matrix is not None and self.matrix.size > 0

    @classmethod
    def load(cls, path: str | Path) -> "BlairEmbeddingStore | None":
        """Load the offline product vectors; missing/malformed -> None (dense channel
            auto-disabled)."""
        path = Path(path)
        emb_path = path if path.suffix == ".npy" else path.with_suffix(".npy")
        asin_path = emb_path.with_name(emb_path.stem + "_asins.npy")
        if not emb_path.exists():
            logger.warning("[data_access] offline product vectors missing: %s (dense channel disabled)", emb_path)  # noqa: E501
            return None
        try:
            matrix = np.load(emb_path, mmap_mode=None)
            if not asin_path.exists():
                logger.warning("[data_access] asins mapping missing: %s (dense channel disabled)", asin_path)  # noqa: E501
                return None
            asins_raw = np.load(asin_path, allow_pickle=True)
            asins = [str(a) for a in asins_raw.tolist()]
            asin_index = {a: i for i, a in enumerate(asins)}
            if matrix.ndim != 2 or matrix.shape[0] != len(asins):
                logger.warning("[data_access] vector matrix row count does not match asins (dense channel disabled)")  # noqa: E501
                return None
            logger.info("[data_access] BLaIR embeds loaded: %d x %d", *matrix.shape)
            return cls(matrix.astype(np.float32), asins, asin_index)
        except Exception as exc:
            logger.warning("[data_access] failed to load offline vectors: %s (dense channel disabled)", exc)  # noqa: E501
            return None
