"""数据访问：仅加载离线预计算数据（商品目录 + BLaIR 商品向量 npy）。

重点提醒：
- BLaIR 商品向量由后续"离线预处理脚本"生成（本模块不写商品全量向量化逻辑）；
- 本文件只提供"加载"接口；推理阶段只对用户查询文本做 embedding 编码（见 retriever_pipeline）。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class CatalogStore:
    """竞赛冻结产品目录内存态存储（parent_asin → 元数据 + 检索文本）。"""

    def __init__(self, products: dict[str, dict], search_text: dict[str, str]) -> None:
        self.products = products
        self.search_text = search_text          # asin -> 小写检索文本（title/features/...）
        self.ids = list(products.keys())
        self.id_set = set(products.keys())

    def valid_asin(self, asin: str) -> bool:
        """硬性约束 8：只允许目录内真实 parent_asin。"""
        return asin in self.id_set

    def get(self, asin: str) -> dict | None:
        return self.products.get(asin)

    def searchable(self, asin: str) -> str:
        return self.search_text.get(asin, "")


def load_catalog(path: str | Path) -> CatalogStore:
    """加载竞赛冻结产品目录（只读，不修改）。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"产品目录不存在: {path}（可用 PRODUCT_CATALOG_PATH 指定）")
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
    """BLaIR 商品向量存储：只加载离线预计算 npy，不生成向量。

    npy 文件格式约定（由离线预处理脚本生成）：
      - embeds.npy : float32 [N, dim]，行序与 asins.npy 一致
      - asins.npy  : object/str 数组 [N]
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
        """加载离线商品向量；文件缺失/格式错误返回 None（稠密通道自动禁用）。"""
        path = Path(path)
        emb_path = path if path.suffix == ".npy" else path.with_suffix(".npy")
        asin_path = emb_path.with_name(emb_path.stem + "_asins.npy")
        if not emb_path.exists():
            logger.warning("[data_access] 离线商品向量不存在: %s（稠密通道禁用）", emb_path)
            return None
        try:
            matrix = np.load(emb_path, mmap_mode=None)
            if not asin_path.exists():
                logger.warning("[data_access] asins 映射缺失: %s（稠密通道禁用）", asin_path)
                return None
            asins_raw = np.load(asin_path, allow_pickle=True)
            asins = [str(a) for a in asins_raw.tolist()]
            asin_index = {a: i for i, a in enumerate(asins)}
            if matrix.ndim != 2 or matrix.shape[0] != len(asins):
                logger.warning("[data_access] 向量矩阵与 asins 行数不一致（稠密通道禁用）")
                return None
            logger.info("[data_access] BLaIR embeds loaded: %d x %d", *matrix.shape)
            return cls(matrix.astype(np.float32), asins, asin_index)
        except Exception as exc:
            logger.warning("[data_access] 加载离线向量失败: %s（稠密通道禁用）", exc)
            return None
