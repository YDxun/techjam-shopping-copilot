"""Retrieval-pipeline constants and environment-variable config (shared across steps 4-6).

Hard constraint 7: all hyperparameters (RRF k, BM25 field weights, penalty coefficients) are fixed
as constants in this file,
never trained or tuned at runtime; only recovery_mode selects a branch.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# paths and model environment variables
# ---------------------------------------------------------------------------
PRODUCT_CATALOG_PATH = Path(os.environ.get(
    "PRODUCT_CATALOG_PATH", "data/catalog.jsonl"))
BLAIR_OFFLINE_EMBEDDING_PATH = Path(os.environ.get(
    "BLAIR_OFFLINE_EMBEDDING_PATH", "data/offline_blair_embeds.npy"))
RERANKER_MODEL_NAME = os.environ.get(
    "RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
BLAIR_QUERY_ENCODER_MODEL = os.environ.get(
    "BLAIR_QUERY_ENCODER_MODEL", "hyp1231/blair-roberta-large")
DEVICE = os.environ.get("DEVICE", "auto").strip().lower()          # auto/cpu/cuda
QUERY_REWRITE_ENABLE = os.environ.get(
    "QUERY_REWRITE_ENABLE", "false"
).strip().lower() in {"1", "true", "yes", "on"}

# ---------------------------------------------------------------------------
# step 5 channel 2: weighted BM25 field weights (title highest, description lowest)
# ---------------------------------------------------------------------------
BM25_FIELD_WEIGHTS = {
    "title": 6.0,
    "features": 4.0,
    "categories": 2.5,
    "details": 2.0,
    "store": 1.5,
    "description": 1.0,
}
BM25_TOP_N = 200                 # per-route BM25 recall cap for channel 2

# ---------------------------------------------------------------------------
# step 5 fusion: RRF parameter (k) and the dense-channel weight alpha
# ---------------------------------------------------------------------------
RRF_K = 60                       # standard RRF constant
RRF_ALPHA_DEFAULT = 0.8          # dense-channel weight alpha (overridable via strategy_config)

# ---------------------------------------------------------------------------
# step 5 channel 1: structured-constraint penalty coefficients (RECOVER mode)
# relaxation priority: budget first (smallest penalty) -> size -> material last (largest penalty)
# ---------------------------------------------------------------------------
STRUCT_BASE_SCORE = 1.0
STRUCT_UNMET_PENALTY = {
    "budget": 0.15,     # relaxed first
    "color": 0.25,
    "size": 0.30,
    "feature": 0.35,
    "material": 0.45,   # relaxed last
    "other": 0.20,
}

# ---------------------------------------------------------------------------
# step 6: rerank candidate pool size
# ---------------------------------------------------------------------------
RERANK_CANDIDATES_NORMAL = 50
RERANK_CANDIDATES_RECOVER = 100
RERANK_TOP_K = 10
RERANK_BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# query building
# ---------------------------------------------------------------------------
MAX_VARIANTS = 3
