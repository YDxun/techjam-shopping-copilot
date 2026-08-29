"""Part A（P0）hard 约束提取鲁棒性 + Part B（P1）模式阈值配置 验收测试。

验收（赛题要求）：
- "The most important thing is cotton" → cotton 为 HARD；
- "I need waterproof" → waterproof 为 HARD（feature）；
- override 消息仍走 REPLACE（优先级不变）；
- browsing "still exploring" 不变（无约束）；
- 官方模板 "A key requirement is: X" 行为与现在一致（HARD）；
- hard_cue_enabled=False 时泛化提取不升级；
- 级联：命中线索词 / turn>=2 出现新约束 → 咨询 LLM（失败回退规则）；
- Part B：retrieval_mode.exploit_min_hard/exploit_min_constraints 默认 2/4 且可环境覆盖。
"""
from __future__ import annotations

import unittest

from agent.dialogue.models import (
    ConstraintStrength,
    DialogueAct,
    RecognitionRequest,
    RecognitionSource,
)
from agent.dialogue.recognizers.cascade import CascadedIntentRecognizer
from agent.dialogue.recognizers.llm import LLMIntentRecognizer
from agent.dialogue.recognizers.rule_based import RuleBasedRecognizer
from agent.dialogue.reducer import StateReducer
from config.env_config import EnvConfig
from llm.base import LLMResult, LLMState, LLMStatus, LLMUsage


class FakeLLMClient:
    def __init__(self, result: LLMResult, state: LLMState = LLMState.AVAILABLE) -> None:
        self.result = result
        self.calls: list = []
        self._status = LLMStatus(state=state, provider="deepseek", model="deepseek-chat")

    @property
    def status(self) -> LLMStatus:
        return self._status

    @property
    def cumulative_usage(self) -> LLMUsage:
        return LLMUsage()

    def initialize(self) -> LLMStatus:
        return self._status

    def chat(self, messages, *, temperature=None, max_tokens=None) -> LLMResult:
        self.calls.append((messages, temperature, max_tokens))
        return self.result


def successful_result(content: str) -> LLMResult:
    return LLMResult(True, "deepseek", "deepseek-chat", content=content,
                     usage=LLMUsage(prompt_tokens=11, completion_tokens=7))


class HardCueTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = StateReducer().new_state("s1", {})
        self.rules = RuleBasedRecognizer(max_evidence_length=180, hard_cue_enabled=True)

    def request(self, message: str, turn: int = 1) -> RecognitionRequest:
        return RecognitionRequest(user_message=message, turn=turn, state=self.state)

    def test_most_important_thing_is_cotton_hard(self) -> None:
        r = self.rules.recognize(self.request("I'm looking for T-Shirts. The most important thing is cotton."))
        hard = [op for op in r.constraint_operations if op.strength == ConstraintStrength.HARD]
        self.assertTrue(any(op.value == "cotton" for op in hard))

    def test_i_need_waterproof_hard(self) -> None:
        r = self.rules.recognize(self.request("I need waterproof"))
        hard = [op for op in r.constraint_operations if op.strength == ConstraintStrength.HARD]
        self.assertTrue(any(op.value == "waterproof" for op in hard))
        self.assertEqual(r.dialogue_act, DialogueAct.ADD_CONSTRAINT)

    def test_override_still_replace(self) -> None:
        r = self.rules.recognize(
            self.request("Actually, ignore my earlier preference. What I need is: leather.")
        )
        self.assertEqual(r.dialogue_act, DialogueAct.REPLACE_CONSTRAINT)
        self.assertEqual(r.constraint_operations[0].operation.value, "replace")
        self.assertEqual(r.constraint_operations[0].strength, ConstraintStrength.HARD)

    def test_browsing_unchanged(self) -> None:
        r = self.rules.recognize(self.request("I'm looking for shoes, but I'm still exploring."))
        self.assertEqual(r.dialogue_act, DialogueAct.NEW_SEARCH)
        self.assertEqual(len(r.constraint_operations), 0)

    def test_official_key_requirement_unchanged(self) -> None:
        r = self.rules.recognize(
            self.request("I'm looking for shoes. A key requirement is: waterproof.")
        )
        self.assertEqual(r.constraint_operations[0].strength, ConstraintStrength.HARD)
        self.assertEqual(r.constraint_operations[0].value, "waterproof")

    def test_hard_cue_disabled_keeps_soft(self) -> None:
        rec = RuleBasedRecognizer(max_evidence_length=180, hard_cue_enabled=False)
        r = rec.recognize(self.request("The most important thing is cotton"))
        hard = [op for op in r.constraint_operations if op.strength == ConstraintStrength.HARD]
        self.assertEqual(hard, [])

    def test_cascade_consults_llm_on_hard_cue(self) -> None:
        client = FakeLLMClient(successful_result("{}"))
        cascade = CascadedIntentRecognizer(
            rule_recognizer=self.rules,
            llm_recognizer=LLMIntentRecognizer(client, max_evidence_length=180),
            mode="cascaded",
            rule_confidence_threshold=0.75,
        )
        cascade.recognize(self.request("I need waterproof"))
        self.assertTrue(client.calls)  # 命中线索词 → 咨询 LLM

    def test_cascade_consults_llm_on_turn2_new_constraint(self) -> None:
        client = FakeLLMClient(successful_result("{}"))
        cascade = CascadedIntentRecognizer(
            rule_recognizer=self.rules,
            llm_recognizer=LLMIntentRecognizer(client, max_evidence_length=180),
            mode="cascaded",
            rule_confidence_threshold=0.75,
        )
        # turn>=2 且规则提取到新约束 → 咨询 LLM
        cascade.recognize(self.request("For that, what matters is: cotton.", turn=2))
        self.assertTrue(client.calls)

    def test_cascade_no_consult_when_rule_confident_turn1_no_cue(self) -> None:
        client = FakeLLMClient(successful_result("{}"))
        cascade = CascadedIntentRecognizer(
            rule_recognizer=self.rules,
            llm_recognizer=LLMIntentRecognizer(client, max_evidence_length=180),
            mode="cascaded",
            rule_confidence_threshold=0.75,
        )
        # 无线索词 + 规则高置信 + turn=1 → 不咨询
        cascade.recognize(self.request("For that, what matters is: cotton.", turn=1))
        self.assertEqual(client.calls, [])

    def test_cascade_consults_llm_on_official_key_requirement(self) -> None:
        # 官方 buying 含 "key" 线索词 → 级联也会咨询（LLM 失败回退规则；默认无 key 环境不触发）
        client = FakeLLMClient(successful_result("{}"))
        cascade = CascadedIntentRecognizer(
            rule_recognizer=self.rules,
            llm_recognizer=LLMIntentRecognizer(client, max_evidence_length=180),
            mode="cascaded",
            rule_confidence_threshold=0.75,
        )
        cascade.recognize(self.request("I'm looking for shoes. A key requirement is: waterproof."))
        self.assertTrue(client.calls)


class RetrievalModeConfigTest(unittest.TestCase):
    def test_defaults(self) -> None:
        env = EnvConfig.from_env()
        self.assertEqual(env.retrieval_mode.exploit_min_hard, 2)
        self.assertEqual(env.retrieval_mode.exploit_min_constraints, 4)
        self.assertTrue(env.hard_cue_enabled)


if __name__ == "__main__":
    unittest.main()
