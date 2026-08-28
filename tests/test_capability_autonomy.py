"""自主能力（环境自感知 + 自适应决策 + LLM 回退）单元测试。"""
from __future__ import annotations

import unittest

from agent.capability_probe import CapabilityProbe
from agent.clarifier import Clarifier
from agent.dialogue_state_machine import DialogueStateMachine
from agent.intent_router import IntentRouter
from agent.runtime_controller import RuntimeController
from config.env_config import EnvConfig
from llm.base import DisabledLLMClient, LLMResult, LLMState, LLMStatus


class _FakeLLM:
    """可编程的假 LLM 客户端：控制状态与返回内容，无网络。"""

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

    def chat(self, messages, *, temperature=None, max_tokens=None):
        self.calls += 1
        if not self.reply:
            return LLMResult(False, "deepseek", "deepseek-chat", error_category=None, error_message="no reply")
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
        return CapabilityProfile(llm_state=llm_state, dense_available=dense,
                                 reranker_available=False, device="cpu")

    def test_llm_auto_disabled_when_unavailable(self) -> None:
        env = EnvConfig.from_env(environ={"LLM_INTENT_ENABLE": "true", "LLM_CLARIFY_ENABLE": "true", "LLM_RERANK": "true"})
        d = RuntimeController(env, self._profile("unavailable")).decide()
        self.assertFalse(d.use_llm_intent)
        self.assertFalse(d.use_llm_clarify)
        self.assertFalse(d.use_llm_rerank)

    def test_llm_enabled_when_available(self) -> None:
        env = EnvConfig.from_env(environ={"LLM_INTENT_ENABLE": "true"})
        d = RuntimeController(env, self._profile("available")).decide()
        self.assertTrue(d.use_llm_intent)
        self.assertFalse(d.use_llm_clarify)   # 未开启

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


class IntentRouterLLMTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = EnvConfig.from_env(environ={"SKIP_DATA_VERIFY": "1"})
        self.sm = DialogueStateMachine()
        self.router = IntentRouter(self.env)

    def test_llm_bad_reply_falls_back_to_rules(self) -> None:
        state = self.sm.new_state("s1", {})
        self.sm.update(state, "I'm looking for T-Shirts. A key requirement is: cotton.", 1)
        bad = _FakeLLM(LLMState.AVAILABLE, reply="not json at all")
        route = self.router.route(state, "probe", llm_client=bad, use_llm=True, user_message="hi")
        self.assertEqual(route.track, "buying")   # 规则路径：有 hard 约束
        self.assertGreaterEqual(len(route.hard_groups), 1)

    def test_llm_valid_reply_used(self) -> None:
        state = self.sm.new_state("s2", {})
        self.sm.update(state, "I'm looking for shoes, still exploring.", 1)
        good = _FakeLLM(LLMState.AVAILABLE,
                        reply='{"intent_track":"buying","constraints":{"material":"leather"},"override_detected":false,"confidence":0.9}')
        route = self.router.route(state, "probe", llm_client=good, use_llm=True, user_message="hi")
        self.assertEqual(route.track, "buying")
        self.assertIn("leather", route.soft_terms)


class ClarifierLLMTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = EnvConfig.from_env(environ={"SKIP_DATA_VERIFY": "1", "CLARIFY_STRATEGY": "other"})
        self.sm = DialogueStateMachine()
        self.clarifier = Clarifier(self.env)

    def test_llm_invalid_attribute_falls_back_to_rules(self) -> None:
        state = self.sm.new_state("s3", {})
        bad = _FakeLLM(LLMState.AVAILABLE, reply='{"ask_attribute":"not_allowed","message":"hi"}')
        ask, _ = self.clarifier.decide(state, 2, llm_client=bad, use_llm=True)
        self.assertEqual(ask, "other")   # 规则回退

    def test_llm_valid_attribute_used(self) -> None:
        state = self.sm.new_state("s4", {})
        good = _FakeLLM(LLMState.AVAILABLE, reply='{"ask_attribute":"material","message":"Any material preference?"}')
        ask, msg = self.clarifier.decide(state, 2, llm_client=good, use_llm=True)
        self.assertEqual(ask, "material")
        self.assertIn("material", msg.lower())


if __name__ == "__main__":
    unittest.main()
