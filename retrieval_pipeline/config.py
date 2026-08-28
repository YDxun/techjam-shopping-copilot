"""检索管线常量与环境变量配置（第4-6步共用）。

硬性约束 7：所有超参（RRF k 值、BM25 字段权重、惩罚系数）写死在本文件常量，
运行时不训练、不调参；仅根据 recovery_mode 走分支逻辑。
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径与模型环境变量
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
QUERY_REWRITE_ENABLE = os.environ.get("QUERY_REWRITE_ENABLE", "false").strip().lower() in {"1", "true", "yes", "on"}

# ---------------------------------------------------------------------------
# 第5步-通道2：加权 BM25 字段权重（title 最高，description 最低）
# ---------------------------------------------------------------------------
BM25_FIELD_WEIGHTS = {
    "title": 6.0,
    "features": 4.0,
    "categories": 2.5,
    "details": 2.0,
    "store": 1.5,
    "description": 1.0,
}
BM25_TOP_N = 200                 # 通道2每路 BM25 召回上限

# ---------------------------------------------------------------------------
# 第5步-多路融合：RRF 参数（k 值）与稠密通道权重 α
# ---------------------------------------------------------------------------
RRF_K = 60                       # 标准 RRF 常数
RRF_ALPHA_DEFAULT = 0.8          # 稠密通道权重系数 α（strategy_config 可覆盖）

# ---------------------------------------------------------------------------
# 第5步-通道1：结构化约束惩罚系数（RECOVER 模式）
# 放宽优先级：budget 最先放宽（惩罚最小）→ size → material 最后放宽（惩罚最大）
# ---------------------------------------------------------------------------
STRUCT_BASE_SCORE = 1.0
STRUCT_UNMET_PENALTY = {
    "budget": 0.15,     # 最先放宽
    "color": 0.25,
    "size": 0.30,
    "feature": 0.35,
    "material": 0.45,   # 最后放宽
    "other": 0.20,
}

# ---------------------------------------------------------------------------
# 第6步：重排候选规模
# ---------------------------------------------------------------------------
RERANK_CANDIDATES_NORMAL = 50
RERANK_CANDIDATES_RECOVER = 100
RERANK_TOP_K = 10
RERANK_BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# 查询构建
# ---------------------------------------------------------------------------
MAX_VARIANTS = 3
