"""BLaIR 稠密检索共享组件（Pillar I 通道3，环境自感知）。

分工（对齐赛题"离线预计算 + 推理只编码查询"）：
- scripts/encode_catalog_blair.py  离线把 50k 商品文本编码成 npy（CLS pooling + L2）；
- 本模块只做两件事：
    1) BlairEmbeddingStore.load()     加载离线商品向量（npy + asins 映射）；
    2) BlairQueryEncoder.encode()     推理阶段只编码用户查询文本。
- 任何环节不可用（文件缺失 / 模型未安装 / 下载失败）→ 返回 None，
  上层稠密通道自动禁用并回退 BM25（不阻塞主流程，鲁棒性由环境自感知保证）。

编码规范（与官方 hyp1231/AmazonReviews2023 generate_emb.py 一致）：
    CLS pooling（last_hidden_state[:, 0]）+ L2 归一化，检索用点积。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class BlairEmbeddingStore:
    """离线 BLaIR 商品向量存储（只加载 npy，不生成向量）。"""

    def __init__(self, matrix: np.ndarray, asins: list[str], asin_index: dict[str, int]) -> None:
        self.matrix = matrix
        self.asins = asins
        self.asin_index = asin_index
        self.dim = matrix.shape[1] if matrix.ndim == 2 else 0

    @property
    def available(self) -> bool:
        return self.matrix is not None and self.matrix.size > 0

    @classmethod
    def load(cls, path: str | Path) -> "BlairEmbeddingStore | None":
        """加载离线向量；缺失/格式错误返回 None（稠密通道自动禁用）。"""
        path = Path(path)
        emb_path = path if path.suffix == ".npy" else path.with_suffix(".npy")
        asin_path = emb_path.with_name(emb_path.stem + "_asins.npy")
        if not emb_path.exists():
            logger.warning("[blair] 离线商品向量不存在: %s（稠密通道禁用）", emb_path)
            return None
        try:
            matrix = np.load(emb_path, mmap_mode=None)
            if not asin_path.exists():
                logger.warning("[blair] asins 映射缺失: %s（稠密通道禁用）", asin_path)
                return None
            asins = [str(a) for a in np.load(asin_path, allow_pickle=True).tolist()]
            asin_index = {a: i for i, a in enumerate(asins)}
            if matrix.ndim != 2 or matrix.shape[0] != len(asins):
                logger.warning("[blair] 向量矩阵与 asins 行数不一致（稠密通道禁用）")
                return None
            logger.info("[blair] BLaIR embeds loaded: %d x %d (%s)", *matrix.shape, emb_path.name)
            return cls(matrix.astype(np.float32), asins, asin_index)
        except Exception as exc:
            logger.warning("[blair] 加载离线向量失败: %s（稠密通道禁用）", exc)
            return None


class BlairQueryEncoder:
    """推理阶段查询编码器：只编码用户查询文本（与离线编码规范一致）。"""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None          # None=未加载, False=加载失败, 其它=编码器
        self._max_length = 512

    @property
    def ready(self) -> bool:
        return self._ensure() not in (None, False)

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
            logger.info("[blair] query encoder loaded (transformers): %s", self.model_name)
            return self._model
        except Exception as exc:
            logger.warning("[blair] transformers 加载失败（%s）→ 尝试 sentence-transformers", exc)
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info("[blair] query encoder loaded (sentence-transformers): %s", self.model_name)
            return self._model
        except Exception as exc:
            logger.warning("[blair] 查询编码器不可用（%s）→ 稠密通道禁用", exc)
            self._model = False
        return self._model

    def encode(self, text: str) -> np.ndarray | None:
        text = (text or "").strip()
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
            logger.warning("[blair] 查询编码失败: %s", exc)
            return None
