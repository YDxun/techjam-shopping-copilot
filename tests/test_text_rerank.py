"""qwen3-rerank 文本重排链路测试（替换 LLM 语义重排分支）。

覆盖：
- RerankClient 无配置 → DISABLED（不发网络）
- RerankClient 解析 /reranks 响应（mock OpenAI.post）
- runtime_controller：text 后端可用→启用 / 不可用→回退规则
- capability_probe：text_rerank 字段存在且未配置时为 False
"""

from __future__ import annotations

import json
import sys

from agent.capability_probe import CapabilityProfile
from agent.runtime_controller import RuntimeController
from config.env_config import EnvConfig
from llm.rerank import RerankClient, RerankState


def test_rerank_client_disabled_without_config():
    rc = RerankClient()  # 无 DASHSCOPE_API_KEY / base_url
    st = rc.initialize()
    assert st.state == RerankState.DISABLED
    assert rc.available is False
    assert rc.rerank("q", ["a"]) is None  # 不可用直接返回 None


def test_rerank_client_parses_results_without_requests_dependency(monkeypatch):
    class FakeResponse:
        def read(self):
            return json.dumps(
                {
                    "results": [
                        {"index": 1, "relevance_score": 0.91},
                        {"index": 0, "relevance_score": 0.32},
                    ]
                }
            ).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    def fake_urlopen(request, timeout=None):
        assert request.full_url.endswith("/reranks")
        payload = json.loads(request.data.decode("utf-8"))
        assert payload["model"] == "qwen3-rerank"
        assert "query" in payload and "documents" in payload
        assert timeout == 10.0
        return FakeResponse()

    # The text reranker is optional, but selecting it must not require an
    # undeclared third-party HTTP package. Exercise the real request path with
    # its standard-library transport and ensure no external request is made.
    monkeypatch.setitem(sys.modules, "requests", None)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    rc = RerankClient(
        api_key="sk-test", base_url="https://dashscope-intl.aliyuncs.com/compatible-api/v1"
    )
    rc.initialize()
    assert rc.available is True
    results = rc.rerank("black cotton t-shirt", ["docA", "docB"], top_n=2)
    assert results is not None
    # 按分数降序：index 1 优先
    assert [r.index for r in results] == [1, 0]
    assert abs(results[0].score - 0.91) < 1e-6


def test_rerank_client_base_url_resolution(monkeypatch):
    # workspace_id 拼子域
    rc = RerankClient(api_key="k", workspace_id="ws-abc")
    assert "ws-abc.cn-beijing.maas.aliyuncs.com" in rc._base_url
    # 完整 base_url 优先
    rc2 = RerankClient(api_key="k", workspace_id="ws-abc", base_url="https://custom.example/v1")
    assert rc2._base_url == "https://custom.example/v1"


def test_runtime_controller_text_rerank_decision():
    # text 后端 + 探测可用 → use_llm_rerank=True, text_rerank_active=True
    prof_ok = CapabilityProfile(text_rerank_available=True, rerank_backend="text")
    env2 = EnvConfig.from_env(overrides={"llm": {"rerank_enabled": True, "rerank_backend": "text"}})
    d = RuntimeController(env2, prof_ok).decide()
    assert d.use_llm_rerank is True
    assert d.text_rerank_active is True
    # 探测不可用 → 回退规则
    prof_no = CapabilityProfile(text_rerank_available=False, rerank_backend="text")
    d2 = RuntimeController(env2, prof_no).decide()
    assert d2.use_llm_rerank is False
    assert d2.text_rerank_active is False
    # chat 后端 → 走 LLM 可用性
    env3 = EnvConfig.from_env(overrides={"llm": {"rerank_enabled": True, "rerank_backend": "chat"}})
    prof_llm_no = CapabilityProfile(llm_state="disabled")
    d3 = RuntimeController(env3, prof_llm_no).decide()
    assert d3.use_llm_rerank is False
    # 未启用 rerank → False
    env4 = EnvConfig.from_env(overrides={"llm": {"rerank_enabled": False}})
    d4 = RuntimeController(env4, prof_ok).decide()
    assert d4.use_llm_rerank is False


def test_capability_profile_has_text_rerank_fields():
    profile = CapabilityProfile()
    assert hasattr(profile, "text_rerank_available")
    assert hasattr(profile, "rerank_backend")
    assert hasattr(profile, "text_rerank_error")
