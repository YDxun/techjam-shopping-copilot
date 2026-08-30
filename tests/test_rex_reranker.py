"""RexReranker-0.6B integration tests (generative rerank vs bge cross-encoder dispatch).

Covers:
- is_generation_reranker recognition (Rex/Qwen3-Reranker are generative; bge is a cross-encoder)
- Reranker._ensure_reranker_model dispatches by model name (mocked; no real model loaded)
"""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

from agent.reranker import Reranker
from utils.rex_reranker import is_generation_reranker


def test_generation_reranker_helpers_import_without_optional_model_runtime():
    """Rule-only startup must not require the optional torch/Rex model stack."""
    module = importlib.import_module("utils.rex_reranker")
    assert module.is_generation_reranker("RexReranker-0.6B")


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
        llm=SimpleNamespace(
            rerank_backend="text",
            qwen_rerank_model="qwen3-rerank",
            dashscope_workspace_id="",
            qwen_rerank_base_url="",
        ),
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
    mock_flag = patch("agent.reranker.FlagReranker", create=True)
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    fake_embedding = types.ModuleType("FlagEmbedding")
    with mock_flag as unused_mock_flag:
        unused_mock_flag.return_value = object()
        fake_embedding.FlagReranker = unused_mock_flag
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "FlagEmbedding", fake_embedding)
        model = reranker._ensure_reranker_model()
    assert model is not None
