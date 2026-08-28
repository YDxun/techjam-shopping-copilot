"""RexReranker-0.6B 接入测试（生成式重排 vs bge 交叉编码分发）。

覆盖：
- is_generation_reranker 识别（Rex/Qwen3-Reranker 为生成式，bge 为交叉编码）
- Reranker._ensure_reranker_model 按模型名分发（mock，不加载真实模型）
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from agent.reranker import Reranker
from utils.rex_reranker import is_generation_reranker


def test_is_generation_reranker_detection():
    assert is_generation_reranker("thebajajra/RexReranker-0.6B") is True
    assert is_generation_reranker("thebajajra/RexReranker-large") is True
    assert is_generation_reranker("Qwen/Qwen3-Reranker-0.6B") is True
    assert is_generation_reranker("BAAI/bge-reranker-v2-m3") is False
    assert is_generation_reranker("") is False


def _env(reranker_model: str) -> object:
    return SimpleNamespace(
        reranker_model=reranker_model,
        rerank_candidates=12,
        llm=SimpleNamespace(rerank_backend="text", qwen_rerank_model="qwen3-rerank",
                            dashscope_workspace_id="", qwen_rerank_base_url=""),
    )


def test_ensure_reranker_model_dispatches_generation(monkeypatch):
    reranker = Reranker.__new__(Reranker)
    reranker.env = _env("thebajajra/RexReranker-0.6B")
    reranker._bge = None
    reranker._rerank_client = None
    with patch("agent.reranker.RexRerankerScorer") as mock_scorer:
        mock_scorer.return_value = object()
        model = reranker._ensure_reranker_model()
        mock_scorer.assert_called_once()
        assert model is not None


def test_ensure_reranker_model_dispatches_cross_encoder(monkeypatch):
    reranker = Reranker.__new__(Reranker)
    reranker.env = _env("BAAI/bge-reranker-v2-m3")
    reranker._bge = None
    reranker._rerank_client = None
    with patch("FlagEmbedding.FlagReranker") as mock_flag:
        mock_flag.return_value = object()
        model = reranker._ensure_reranker_model()
        mock_flag.assert_called_once()
        assert model is not None
