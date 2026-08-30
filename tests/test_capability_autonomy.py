"""Unit tests for autonomous capability (environment awareness + adaptive decisions + LLM
    fallback)."""

from __future__ import annotations

import unittest

from agent.capability_probe import CapabilityProbe
from agent.runtime_controller import RuntimeController
from config.env_config import EnvConfig
from llm.base import DisabledLLMClient, LLMResult, LLMState, LLMStatus


class _FakeLLM:
    """Programmable fake LLM client: controls state and returned content; no network."""

    def __init__(self, state: LLMState = LLMState.AVAILABLE, reply: str = "") -> None:
        self._state = LLMStatus(state, "deepseek", "deepseek-chat")
        self.reply = reply
        self.calls = 0
        self.cumulative_usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()

    @property
    def status(self) -> LLMStatus:
        return self._state

    def initialize(self) -> LLMStatus:
        return self._state

    def chat(self, messages, *, temperature=None, max_tokens=None, request_options=None):
        self.calls += 1
        if not self.reply:
            return LLMResult(
                False, "deepseek", "deepseek-chat", error_category=None, error_message="no reply"
            )
        return LLMResult(True, "deepseek", "deepseek-chat", content=self.reply)


class CapabilityProbeTest(unittest.TestCase):
    def test_disabled_client_reports_disabled(self) -> None:
        env = EnvConfig.from_env(environ={})
        profile = CapabilityProbe(env, DisabledLLMClient()).probe()
        self.assertEqual(profile.llm_state, "disabled")
        self.assertFalse(profile.llm_available)

    def test_available_client_reports_available(self) -> None:
        env = EnvConfig.from_env(environ={})
        profile = CapabilityProbe(env, _FakeLLM(LLMState.AVAILABLE)).probe()
        self.assertEqual(profile.llm_state, "available")
        self.assertTrue(profile.llm_available)


class RuntimeControllerTest(unittest.TestCase):
    def _profile(self, llm_state: str = "disabled", dense: bool = False) -> object:
        from agent.capability_probe import CapabilityProfile

        return CapabilityProfile(
            llm_state=llm_state, dense_available=dense, reranker_available=False, device="cpu"
        )

    def test_llm_auto_disabled_when_unavailable(self) -> None:
        env = EnvConfig.from_env(
            environ={
                "LLM_INTENT_ENABLE": "true",
                "LLM_CLARIFY_ENABLE": "true",
                "LLM_RERANK": "true",
            }
        )
        d = RuntimeController(env, self._profile("unavailable")).decide()
        self.assertFalse(d.use_llm_intent)
        self.assertFalse(d.use_llm_clarify)
        self.assertFalse(d.use_llm_rerank)

    def test_llm_enabled_when_available(self) -> None:
        env = EnvConfig.from_env(environ={"LLM_INTENT_ENABLE": "true"})
        d = RuntimeController(env, self._profile("available")).decide()
        self.assertTrue(d.use_llm_intent)
        self.assertFalse(d.use_llm_clarify)  # not enabled

    def test_retrieval_auto_falls_back_to_bm25(self) -> None:
        env = EnvConfig.from_env(environ={"RETRIEVAL_BACKEND": "auto"})
        d = RuntimeController(env, self._profile("disabled", dense=False)).decide()
        self.assertEqual(d.retrieval_backend, "bm25")
        self.assertFalse(d.use_dense)

    def test_retrieval_auto_uses_hybrid_when_dense_available(self) -> None:
        env = EnvConfig.from_env(environ={"RETRIEVAL_BACKEND": "auto"})
        d = RuntimeController(env, self._profile("disabled", dense=True)).decide()
        self.assertEqual(d.retrieval_backend, "hybrid")
        self.assertTrue(d.use_dense)

    def test_retrieval_hybrid_falls_back_without_dense(self) -> None:
        env = EnvConfig.from_env(environ={"RETRIEVAL_BACKEND": "hybrid"})
        d = RuntimeController(env, self._profile("disabled", dense=False)).decide()
        self.assertEqual(d.retrieval_backend, "bm25")
