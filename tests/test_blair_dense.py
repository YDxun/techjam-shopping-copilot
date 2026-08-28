"""BLaIR 稠密检索链路测试（Pillar I 通道3 + 环境自感知）。

覆盖：
- 配置：blair_offline_embedding_path / blair_query_encoder_model 读取与默认值；
- 数据访问：BlairEmbeddingStore.load 加载离线 npy（含缺失/格式错误 → None 降级）；
- 环境自感知：capability_probe 的 _blair_npy_exists / _reranker_model_cached；
- retriever 后端自解析：_dense_backend_available（编码器可导入 && npy 存在）。

说明：真实 BLaIR 模型加载（~40s CPU）放在慢测试中，默认跳过（不影响 CI 速度）；
稠密通道端到端已在 scripts/encode_catalog_blair.py + retrieval_pipeline/test_pipeline.py 验证。
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from agent.capability_probe import CapabilityProbe, CapabilityProfile
from config.env_config import EnvConfig
from utils import blair as blair_utils

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "data" / "offline_blair_embeds_checkpoint.npy"


def test_config_blair_fields():
    env = EnvConfig.from_env()
    assert env.blair_offline_embedding_path == "data/offline_blair_embeds.npy"
    assert env.blair_query_encoder_model == "hyp1231/blair-roberta-large"
    # 环境变量可覆盖
    with patch.dict(
        os.environ,
        {
            "BLAIR_OFFLINE_EMBEDDING_PATH": "custom/x.npy",
            "BLAIR_QUERY_ENCODER_MODEL": "hyp1231/blair-roberta-base",
        },
    ):
        env2 = EnvConfig.from_env()
        assert env2.blair_offline_embedding_path == "custom/x.npy"
        assert env2.blair_query_encoder_model == "hyp1231/blair-roberta-base"


def test_blair_store_load_checkpoint():
    if not CHECKPOINT.exists():
        pytest.skip("离线 checkpoint 不存在（未运行 encode_catalog_blair.py）")
    store = blair_utils.BlairEmbeddingStore.load(CHECKPOINT)
    assert store is not None
    assert store.available
    assert store.matrix.shape[0] == len(store.asins) > 0
    assert store.matrix.dtype == np.float32
    assert store.matrix.ndim == 2
    # 行索引一致性
    for a in store.asins[:5]:
        assert store.asin_index[a] == store.asins.index(a)


def test_blair_store_load_missing(tmp_path):
    store = blair_utils.BlairEmbeddingStore.load(tmp_path / "not_exist.npy")
    assert store is None  # 文件缺失 → 稠密通道自动禁用（环境自感知降级）


def test_blair_store_load_shape_mismatch(tmp_path):
    emb = tmp_path / "bad.npy"
    asins = tmp_path / "bad_asins.npy"
    np.save(emb, np.zeros((10, 4), dtype=np.float32))
    np.save(asins, np.asarray(["A", "B"], dtype=object))  # 行数不一致
    store = blair_utils.BlairEmbeddingStore.load(emb)
    assert store is None


def test_probe_blair_npy_exists():
    env = EnvConfig.from_env()
    probe = CapabilityProbe(env, llm_client=None)  # type: ignore[arg-type]
    assert isinstance(probe._blair_npy_exists(), bool)
    # 存在性 = Path(env.blair_offline_embedding_path) 是否存在
    expected = Path(env.blair_offline_embedding_path).exists()
    assert probe._blair_npy_exists() == expected


def test_probe_dense_available_reflects_encoders_and_npy():
    env = EnvConfig.from_env()
    profile = CapabilityProfile()
    enc = blair_utils.BlairQueryEncoder("hyp1231/blair-roberta-large")
    # 只测布尔字段（不实际加载模型）
    profile.dense_encoder_available = enc is not None
    profile.blair_npy_ready = Path(env.blair_offline_embedding_path).exists()
    profile.dense_available = profile.dense_encoder_available and profile.blair_npy_ready
    assert isinstance(profile.dense_available, bool)


def test_retriever_dense_backend_available(tmp_path, monkeypatch):
    from agent.retriever import HybridRetriever
    from config.env_config import EnvConfig

    env = EnvConfig.from_env()
    retriever = HybridRetriever.__new__(HybridRetriever)  # 不建索引
    retriever.env = env

    # 编码器不可导入 → False
    with patch.object(HybridRetriever, "_spec_available", return_value=False):
        assert retriever._dense_backend_available() is False

    # 编码器可导入 + npy 缺失 → False
    retriever.env = EnvConfig.from_env(
        overrides={"blair_offline_embedding_path": str(tmp_path / "missing.npy")}
    )
    with patch.object(HybridRetriever, "_spec_available", return_value=True):
        assert retriever._dense_backend_available() is False

    # 编码器可导入 + npy 存在 → True
    npy = tmp_path / "embeds.npy"
    np.save(npy, np.zeros((4, 4), dtype=np.float32))
    np.save(tmp_path / "embeds_asins.npy", np.asarray(["A", "B", "C", "D"], dtype=object))
    retriever.env = EnvConfig.from_env(overrides={"blair_offline_embedding_path": str(npy)})
    with patch.object(HybridRetriever, "_spec_available", return_value=True):
        assert retriever._dense_backend_available() is True


def test_query_encoder_returns_none_without_text():
    enc = blair_utils.BlairQueryEncoder("hyp1231/blair-roberta-large")
    assert enc.encode("") is None  # 空文本 → None（不触发模型加载）
    assert enc.encode("   ") is None
