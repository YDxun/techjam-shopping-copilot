"""qwen3-rerank text-rerank chain tests (replaces the LLM semantic-rerank branch).

Covers:
- RerankClient without config -> DISABLED (no network)
- RerankClient parses /reranks responses (mock OpenAI.post)
- runtime_controller: text backend available -> enabled / unavailable -> rule fallback
- capability_probe: text_rerank field exists and is False when unconfigured
"""
from __future__ import annotations

from agent.capability_probe import CapabilityProfile
from agent.runtime_controller import RuntimeController
from config.env_config import EnvConfig
from llm.rerank import RerankClient, RerankState


def test_rerank_client_disabled_without_config():
    rc = RerankClient()  # no DASHSCOPE_API_KEY / base_url
    st = rc.initialize()
    assert st.state == RerankState.DISABLED
    assert rc.available is False
    assert rc.rerank("q", ["a"]) is None  # unavailable -> returns None directly


def test_rerank_client_parses_results(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": [
                {"index": 1, "relevance_score": 0.91},
                {"index": 0, "relevance_score": 0.32},
            ]}

    def fake_post(url, headers=None, json=None, timeout=None):
        assert url.endswith("/reranks")
        assert json["model"] == "qwen3-rerank"
        assert "query" in json and "documents" in json
        return FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)
    rc = RerankClient(api_key="sk-test", base_url="https://dashscope-intl.aliyuncs.com/compatible-api/v1")
    rc.initialize()
    assert rc.available is True
    results = rc.rerank("black cotton t-shirt", ["docA", "docB"], top_n=2)
    assert results is not None
    # descending by score: index 1 first
    assert [r.index for r in results] == [1, 0]
    assert abs(results[0].score - 0.91) < 1e-6


def test_rerank_client_base_url_resolution(monkeypatch):
    # workspace_id builds the subdomain
    rc = RerankClient(api_key="k", workspace_id="ws-abc")
    assert "ws-abc.cn-beijing.maas.aliyuncs.com" in rc._base_url
    # full base_url takes priority
    rc2 = RerankClient(api_key="k", workspace_id="ws-abc", base_url="https://custom.example/v1")
    assert rc2._base_url == "https://custom.example/v1"


def test_runtime_controller_text_rerank_decision():
    # text backend + probe available -> use_llm_rerank=True, text_rerank_active=True
    prof_ok = CapabilityProfile(text_rerank_available=True, rerank_backend="text")
    env2 = EnvConfig.from_env(overrides={"llm": {"rerank_enabled": True, "rerank_backend": "text"}})
    d = RuntimeController(env2, prof_ok).decide()
    assert d.use_llm_rerank is True
    assert d.text_rerank_active is True
    # probe unavailable -> rule fallback
    prof_no = CapabilityProfile(text_rerank_available=False, rerank_backend="text")
    d2 = RuntimeController(env2, prof_no).decide()
    assert d2.use_llm_rerank is False
    assert d2.text_rerank_active is False
    # chat backend -> driven by LLM availability
    env3 = EnvConfig.from_env(overrides={"llm": {"rerank_enabled": True, "rerank_backend": "chat"}})
    prof_llm_no = CapabilityProfile(llm_state="disabled")
    d3 = RuntimeController(env3, prof_llm_no).decide()
    assert d3.use_llm_rerank is False
    # rerank not enabled -> False
    env4 = EnvConfig.from_env(overrides={"llm": {"rerank_enabled": False}})
    d4 = RuntimeController(env4, prof_ok).decide()
    assert d4.use_llm_rerank is False


def test_capability_profile_has_text_rerank_fields():
    profile = CapabilityProfile()
    assert hasattr(profile, "text_rerank_available")
    assert hasattr(profile, "rerank_backend")
    assert hasattr(profile, "text_rerank_error")
