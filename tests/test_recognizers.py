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
from llm.base import LLMResult, LLMState, LLMStatus, LLMUsage


class FakeLLMClient:
    def __init__(self, result: LLMResult, state: LLMState = LLMState.AVAILABLE) -> None:
        self.result = result
        self.calls: list[tuple] = []
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
        success=True,
        provider="deepseek",
        model="deepseek-chat",
        content=content,
        usage=LLMUsage(prompt_tokens=11, completion_tokens=7),
    )


class RecognizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = StateReducer().new_state("s1", {})
        self.rules = RuleBasedRecognizer(max_evidence_length=180)

    def request(self, message: str, shown: tuple[str, ...] = ()) -> RecognitionRequest:
        return RecognitionRequest(
            user_message=message,
            turn=1,
            state=self.state,
            recently_shown_asins=shown,
        )

    def test_rule_recognizer_extracts_official_category_and_hard_requirement(self) -> None:
        result = self.rules.recognize(
            self.request("I'm looking for running shoes. A key requirement is: cotton.")
        )

        self.assertEqual(result.source, RecognitionSource.RULE)
        self.assertEqual(result.category, "running shoes")
        self.assertEqual(result.dialogue_act, DialogueAct.NEW_SEARCH)
        self.assertEqual(len(result.constraint_operations), 1)
        self.assertEqual(result.constraint_operations[0].value, "cotton")
        self.assertEqual(result.constraint_operations[0].strength, ConstraintStrength.HARD)

    def test_rule_only_mode_never_calls_available_llm(self) -> None:
        client = FakeLLMClient(successful_result("{}"))
        cascade = CascadedIntentRecognizer(
            rule_recognizer=self.rules,
            llm_recognizer=LLMIntentRecognizer(client, max_evidence_length=180),
            mode="rule_only",
            rule_confidence_threshold=0.75,
        )

        result = cascade.recognize(self.request("I might want something different."))

        self.assertEqual(result.source, RecognitionSource.RULE)
        self.assertEqual(client.calls, [])

    def test_attribute_specific_no_additional_preference_is_not_global_stop(self) -> None:
        result = self.rules.recognize(
            self.request("I don't have an additional preference for brand.")
        )

        self.assertEqual(result.dialogue_act, DialogueAct.NO_PREFERENCE)
        self.assertEqual(result.constraint_operations[0].attribute, "brand")

    def test_valid_llm_json_replaces_the_complete_rule_result(self) -> None:
        client = FakeLLMClient(
            successful_result(
                '{"dialogue_act":"add_constraint","category":"shoes",'
                '"constraint_operations":[{"operation":"add","attribute":"material",'
                '"value":"cotton","polarity":"include","strength":"hard",'
                '"evidence":"cotton rather than wool","confidence":0.95}],'
                '"explicit_rejected_asins":[],"confidence":0.95,"ambiguities":[]}'
            )
        )
        cascade = CascadedIntentRecognizer(
            rule_recognizer=self.rules,
            llm_recognizer=LLMIntentRecognizer(client, max_evidence_length=180),
            mode="cascaded",
            rule_confidence_threshold=0.75,
        )

        result = cascade.recognize(self.request("I want that sort, but cotton rather than wool."))

        self.assertEqual(result.source, RecognitionSource.LLM)
        self.assertEqual(result.category, "shoes")
        self.assertEqual(result.constraint_operations[0].value, "cotton")
        self.assertEqual(cascade.last_usage, LLMUsage(prompt_tokens=11, completion_tokens=7))

    def test_malformed_or_out_of_scope_llm_output_falls_back_to_exact_rule_result(self) -> None:
        request = self.request("The previous sort is not what I meant.", shown=("A",))
        rule_result = self.rules.recognize(request)
        responses = (
            "not-json",
            '{"dialogue_act":"reject_products","category":null,'
            '"constraint_operations":[],"explicit_rejected_asins":["UNKNOWN"],'
            '"confidence":0.9,"ambiguities":[]}',
        )
        for response in responses:
            with self.subTest(response=response):
                client = FakeLLMClient(successful_result(response))
                cascade = CascadedIntentRecognizer(
                    rule_recognizer=self.rules,
                    llm_recognizer=LLMIntentRecognizer(client, max_evidence_length=180),
                    mode="cascaded",
                    rule_confidence_threshold=0.75,
                )

                self.assertEqual(cascade.recognize(request), rule_result)


if __name__ == "__main__":
    unittest.main()
