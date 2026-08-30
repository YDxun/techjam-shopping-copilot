"""Acceptance tests for Part A (P0) hard-constraint extraction robustness + Part B (P1) mode-threshold config.

Acceptance (task requirement):
- "The most important thing is cotton" -> cotton is HARD;
- "I need waterproof" -> waterproof is HARD (feature);
- override messages still go through REPLACE (priority unchanged);
- browsing "still exploring" is unchanged (no constraints);
- official template "A key requirement is: X" behaves as before (HARD);
- with hard_cue_enabled=False, generalized extraction does not upgrade;
- cascade: cue-word hit / new constraint at turn>=2 -> consult the LLM (fallback to rules on failure);
- Part B: retrieval_mode.exploit_min_hard/exploit_min_constraints default to 2/4 and are env-overridable.
"""

from __future__ import annotations

import unittest

from agent.dialogue.models import (
    ConstraintStrength,
    DialogueAct,
    RecognitionRequest,
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
    return LLMResult(
        True,
        "deepseek",
        "deepseek-chat",
        content=content,
        usage=LLMUsage(prompt_tokens=11, completion_tokens=7),
    )


class HardCueTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = StateReducer().new_state("s1", {})
        self.rules = RuleBasedRecognizer(max_evidence_length=180, hard_cue_enabled=True)

    def request(self, message: str, turn: int = 1) -> RecognitionRequest:
        return RecognitionRequest(user_message=message, turn=turn, state=self.state)

    def test_most_important_thing_is_cotton_hard(self) -> None:
        r = self.rules.recognize(
            self.request("I'm looking for T-Shirts. The most important thing is cotton.")
        )
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
        self.assertTrue(client.calls)  # cue word hit -> consults the LLM

    def test_cascade_consults_llm_on_turn2_new_constraint(self) -> None:
        client = FakeLLMClient(successful_result("{}"))
        cascade = CascadedIntentRecognizer(
            rule_recognizer=self.rules,
            llm_recognizer=LLMIntentRecognizer(client, max_evidence_length=180),
            mode="cascaded",
            rule_confidence_threshold=0.75,
        )
        # turn>=2 with a new rule-extracted constraint -> consults the LLM
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
        # no cue word + high rule confidence + turn=1 -> no consultation
        cascade.recognize(self.request("For that, what matters is: cotton.", turn=1))
        self.assertEqual(client.calls, [])

    def test_cascade_consults_llm_on_official_key_requirement(self) -> None:
        # official buying contains the "key" cue -> cascade also consults (LLM failure falls back to rules; not triggered in default no-key env)
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
