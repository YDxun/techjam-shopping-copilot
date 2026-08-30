"""Shared BLaIR dense-retrieval components (Pillar I channel 3, environment-aware).

Division of labor (aligned with the task's "offline pre-computation + encode only the query at
inference"):
- scripts/encode_catalog_blair.py  encodes 50k product texts offline into npy (CLS pooling + L2);
- this module does only two things:
    1) BlairEmbeddingStore.load()     loads the offline product vectors (npy + asins mapping);
    2) BlairQueryEncoder.encode()     encodes only the user query text at inference.
- If any piece is unavailable (missing file / model not installed / download failure) -> return
None,
   and the upstream dense channel auto-disables and falls back to BM25 (never blocks the main flow;
   robustness via environment awareness).

Encoding convention (matches official hyp1231/AmazonReviews2023 generate_emb.py):
    CLS pooling (last_hidden_state[:, 0]) + L2 normalization; retrieval uses dot product.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class BlairEmbeddingStore:
    """Offline BLaIR product-vector store (only loads npy; never generates vectors)."""

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
        """Load offline vectors; missing/malformed -> None (dense channel auto-disabled)."""
        path = Path(path)
        emb_path = path if path.suffix == ".npy" else path.with_suffix(".npy")
        asin_path = emb_path.with_name(emb_path.stem + "_asins.npy")
        if not emb_path.exists():
            logger.warning("[blair] offline product vectors missing: %s (dense channel disabled)", emb_path)  # noqa: E501
            return None
        try:
            matrix = np.load(emb_path, mmap_mode=None)
            if not asin_path.exists():
                logger.warning("[blair] asins mapping missing: %s (dense channel disabled)", asin_path)  # noqa: E501
                return None
            asins = [str(a) for a in np.load(asin_path, allow_pickle=True).tolist()]
            asin_index = {a: i for i, a in enumerate(asins)}
            if matrix.ndim != 2 or matrix.shape[0] != len(asins):
                logger.warning("[blair] vector matrix row count does not match asins (dense channel disabled)")  # noqa: E501
                return None
            logger.info("[blair] BLaIR embeds loaded: %d x %d (%s)", *matrix.shape, emb_path.name)
            return cls(matrix.astype(np.float32), asins, asin_index)
        except Exception as exc:
            logger.warning("[blair] failed to load offline vectors: %s (dense channel disabled)", exc)  # noqa: E501
            return None


class BlairQueryEncoder:
    """Inference-time query encoder: encodes only the user query text (matches the offline encoding
        convention)."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None          # None=not loaded, False=load failed, otherwise=encoder
        self._max_length = 512

    @property
    def ready(self) -> bool:
        return self._ensure() not in (None, False)

    def _ensure(self):
        if self._model is not None:
            return self._model
        # Preferred: transformers AutoModel (the canonical BLaIR CLS usage)
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
            logger.warning("[blair] transformers load failed (%s) -> trying sentence-transformers", exc)  # noqa: E501
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info("[blair] query encoder loaded (sentence-transformers): %s", self.model_name)
            return self._model
        except Exception as exc:
            logger.warning("[blair] query encoder unavailable (%s) -> dense channel disabled", exc)
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
                vec = torch.nn.functional.normalize(vec, p=2, dim=1)[0]  # L2 normalization
                return vec.detach().cpu().numpy().astype(np.float32)
            vec = model.encode([text], normalize_embeddings=True)[0]
            return np.asarray(vec, dtype=np.float32)
        except Exception as exc:
            logger.warning("[blair] query encoding failed: %s", exc)
            return None
