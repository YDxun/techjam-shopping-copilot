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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace

from agent.capability_probe import CapabilityProfile
from agent.reranker import Reranker
from agent.runtime_controller import RuntimeController
from config.env_config import EnvConfig
from llm.rerank import RerankClient, RerankResult, RerankState


def test_rerank_client_disabled_without_config():
    rc = RerankClient()  # 无 DASHSCOPE_API_KEY / base_url
    st = rc.initialize()
    assert st.state == RerankState.DISABLED
    assert rc.available is False
    assert rc.rerank("q", ["a"]) is None  # 不可用直接返回 None


def test_rerank_client_parses_results_without_requests_dependency(monkeypatch):
    request_count = 0
    authorization_seen = False

    class SuccessHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            nonlocal request_count, authorization_seen
            request_count += 1
            authorization_seen = authorization_seen or ("Authorization" in self.headers)
            payload = json.loads(
                self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8")
            )
            assert self.path == "/reranks"
            assert payload["model"] == "qwen3-rerank"
            assert "query" in payload and "documents" in payload
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                b'{"results": [{"index": 1, "relevance_score": 0.91}, '
                b'{"index": 0, "relevance_score": 0.32}], '
                b'"usage": {"total_tokens": 37}}'
            )

        def log_message(self, format, *args):
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), SuccessHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # The text reranker is optional, but selecting it must not require an
    # undeclared third-party HTTP package. Exercise the real request path with
    # its standard-library transport and ensure no external request is made.
    monkeypatch.setitem(sys.modules, "requests", None)
    try:
        rc = RerankClient(
            api_key="sk-test", base_url=f"http://127.0.0.1:{server.server_port}"
        )
        rc.initialize()
        assert rc.available is True
        results = rc.rerank("black cotton t-shirt", ["docA", "docB"], top_n=2)
        assert results is not None
        # 按分数降序：index 1 优先
        assert [r.index for r in results] == [1, 0]
        assert abs(results[0].score - 0.91) < 1e-6
        assert rc.last_usage == {"prompt_tokens": 37, "completion_tokens": 0}
        assert request_count == 2
        assert authorization_seen is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_rerank_client_rejects_redirect_before_other_origin_is_reached():
    target_requests = 0
    authorization_forwarded = False

    class TargetHandler(BaseHTTPRequestHandler):
        def _record_request(self):
            nonlocal target_requests, authorization_forwarded
            target_requests += 1
            authorization_forwarded = authorization_forwarded or (
                "Authorization" in self.headers
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"results": [{"index": 0, "relevance_score": 1.0}]}')

        do_GET = _record_request
        do_POST = _record_request

        def log_message(self, format, *args):
            return None

    target_server = ThreadingHTTPServer(("127.0.0.1", 0), TargetHandler)
    target_thread = Thread(target=target_server.serve_forever, daemon=True)
    target_thread.start()

    class RedirectHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(302)
            self.send_header(
                "Location",
                f"http://127.0.0.1:{target_server.server_port}/redirect-target",
            )
            self.end_headers()

        def log_message(self, format, *args):
            return None

    redirect_server = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
    redirect_thread = Thread(target=redirect_server.serve_forever, daemon=True)
    redirect_thread.start()
    try:
        client = RerankClient(
            api_key="sk-test",
            base_url=f"http://127.0.0.1:{redirect_server.server_port}",
            timeout_seconds=1.0,
        )
        status = client.initialize()

        assert status.state == RerankState.UNAVAILABLE
        assert client.available is False
        assert client.rerank("query", ["document"]) is None
        assert target_requests == 0
        assert authorization_forwarded is False
    finally:
        redirect_server.shutdown()
        redirect_server.server_close()
        redirect_thread.join()
        target_server.shutdown()
        target_server.server_close()
        target_thread.join()


def test_rerank_client_base_url_resolution(monkeypatch):
    # workspace_id 拼子域
    rc = RerankClient(api_key="k", workspace_id="ws-abc")
    assert "ws-abc.cn-beijing.maas.aliyuncs.com" in rc._base_url
    # 完整 base_url 优先
    rc2 = RerankClient(api_key="k", workspace_id="ws-abc", base_url="https://custom.example/v1")
    assert rc2._base_url == "https://custom.example/v1"


def test_text_reranker_propagates_qwen_usage_to_agent_response() -> None:
    class FakeTextClient:
        available = True
        last_usage = {"prompt_tokens": 37, "completion_tokens": 0}
        status = SimpleNamespace(model="qwen3-rerank")

        def rerank(self, query, documents, top_n=None):
            return [RerankResult(index=1, score=0.9), RerankResult(index=0, score=0.5)]

    class FakeRetriever:
        products = {
            "A": {"title": "Alpha", "features": []},
            "B": {"title": "Beta", "features": []},
        }

        def product(self, asin):
            return self.products.get(asin)

    env = EnvConfig.from_env(
        overrides={"llm": {"rerank_enabled": True, "rerank_backend": "text"}},
        environ={},
    )
    reranker = Reranker(env=env)
    reranker._rerank_client = FakeTextClient()

    ranked = reranker._text_rerank(
        ["A", "B"],
        FakeRetriever(),
        SimpleNamespace(active=[]),
        SimpleNamespace(query_terms=[]),
    )

    assert ranked == ["B", "A"]
    assert reranker.last_usage == {"prompt_tokens": 37, "completion_tokens": 0}
    assert reranker.last_usage_sources == [
        {
            "provider": "dashscope",
            "model": "qwen3-rerank",
            "prompt_tokens": 37,
            "completion_tokens": 0,
            "online": True,
        }
    ]


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
