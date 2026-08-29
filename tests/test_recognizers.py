from __future__ import annotations

import json
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

    def test_llm_prompt_declares_the_complete_response_contract(self) -> None:
        recognizer = LLMIntentRecognizer(
            FakeLLMClient(successful_result("{}")),
            max_evidence_length=180,
        )

        system_prompt = recognizer._messages(self.request("I want cotton shirts."))[0]["content"]

        schema_text = system_prompt.split(
            "Return exactly one JSON object matching this JSON Schema:\n",
            1,
        )[1].split("\nInclude every required field", 1)[0]
        schema = json.loads(schema_text)
        operation_schema = schema["properties"]["constraint_operations"]["items"]

        self.assertEqual(
            set(schema["required"]),
            {
                "dialogue_act",
                "category",
                "constraint_operations",
                "explicit_rejected_asins",
                "confidence",
                "ambiguities",
            },
        )
        self.assertEqual(
            set(operation_schema["required"]),
            {
                "operation",
                "attribute",
                "value",
                "polarity",
                "strength",
                "evidence",
                "confidence",
            },
        )
        self.assertEqual(
            set(schema["properties"]["dialogue_act"]["enum"]),
            {
                "new_search",
                "add_constraint",
                "replace_constraint",
                "remove_constraint",
                "reject_products",
                "no_preference",
                "no_more_preferences",
                "ambiguous",
            },
        )
        self.assertEqual(
            schema["properties"]["category"],
            {
                "anyOf": [
                    {"type": "string", "minLength": 1},
                    {"type": "null"},
                ]
            },
        )
        self.assertEqual(
            set(operation_schema["properties"]["operation"]["enum"]),
            {"add", "replace", "remove"},
        )
        self.assertEqual(
            set(operation_schema["properties"]["polarity"]["enum"]),
            {"include", "exclude"},
        )
        self.assertEqual(
            set(operation_schema["properties"]["strength"]["enum"]),
            {"hard", "soft"},
        )
        self.assertEqual(
            operation_schema["properties"]["evidence"],
            {"type": "string", "minLength": 1, "maxLength": 180},
        )
        self.assertIn('"operation":"replace"', system_prompt)
        self.assertIn("shortest exact span", system_prompt)

    def test_cascade_statistics_count_an_accepted_llm_result(self) -> None:
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

        cascade.recognize(self.request("I want that sort, but cotton rather than wool."))

        self.assertEqual(
            cascade.statistics(),
            {
                "total_turns": 1,
                "rule_resolutions": 0,
                "llm_attempts": 1,
                "llm_accepted": 1,
                "llm_fallbacks": 0,
                "fallback_reasons": {},
            },
        )

    def test_cascade_statistics_classify_invalid_json_fallback(self) -> None:
        cascade = CascadedIntentRecognizer(
            rule_recognizer=self.rules,
            llm_recognizer=LLMIntentRecognizer(
                FakeLLMClient(successful_result("not-json")),
                max_evidence_length=180,
            ),
            mode="cascaded",
            rule_confidence_threshold=0.75,
        )

        result = cascade.recognize(self.request("I want that sort, but cotton rather than wool."))

        self.assertEqual(result.source, RecognitionSource.RULE)
        self.assertEqual(cascade.statistics()["llm_fallbacks"], 1)
        self.assertEqual(cascade.statistics()["fallback_reasons"], {"invalid_json": 1})

    def test_cascade_statistics_classify_evidence_length_fallback(self) -> None:
        response = (
            '{"dialogue_act":"replace_constraint","category":null,'
            '"constraint_operations":[{"operation":"replace","attribute":"feature",'
            '"value":"cotton","polarity":"include","strength":"hard",'
            f'"evidence":"{"x" * 181}","confidence":0.95}}],'
            '"explicit_rejected_asins":[],"confidence":0.95,"ambiguities":[]}'
        )
        cascade = CascadedIntentRecognizer(
            rule_recognizer=self.rules,
            llm_recognizer=LLMIntentRecognizer(
                FakeLLMClient(successful_result(response)),
                max_evidence_length=180,
            ),
            mode="cascaded",
            rule_confidence_threshold=0.75,
        )

        cascade.recognize(self.request("Actually, ignore my earlier preference."))

        self.assertEqual(
            cascade.statistics()["fallback_reasons"],
            {"evidence_too_long": 1},
        )

    def test_empty_llm_evidence_falls_back_and_is_counted(self) -> None:
        response = (
            '{"dialogue_act":"add_constraint","category":null,'
            '"constraint_operations":[{"operation":"add","attribute":"material",'
            '"value":"cotton","polarity":"include","strength":"hard",'
            '"evidence":"","confidence":0.95}],'
            '"explicit_rejected_asins":[],"confidence":0.95,"ambiguities":[]}'
        )
        cascade = CascadedIntentRecognizer(
            rule_recognizer=self.rules,
            llm_recognizer=LLMIntentRecognizer(
                FakeLLMClient(successful_result(response)),
                max_evidence_length=180,
            ),
            mode="cascaded",
            rule_confidence_threshold=0.75,
        )

        result = cascade.recognize(self.request("I want that sort, but cotton instead."))

        self.assertEqual(result.source, RecognitionSource.RULE)
        self.assertEqual(
            cascade.statistics()["fallback_reasons"],
            {"invalid_evidence": 1},
        )

    def test_ungrounded_llm_evidence_falls_back_and_is_counted(self) -> None:
        response = (
            '{"dialogue_act":"add_constraint","category":null,'
            '"constraint_operations":[{"operation":"add","attribute":"material",'
            '"value":"cotton","polarity":"include","strength":"hard",'
            '"evidence":"invented span","confidence":0.95}],'
            '"explicit_rejected_asins":[],"confidence":0.95,"ambiguities":[]}'
        )
        cascade = CascadedIntentRecognizer(
            rule_recognizer=self.rules,
            llm_recognizer=LLMIntentRecognizer(
                FakeLLMClient(successful_result(response)),
                max_evidence_length=180,
            ),
            mode="cascaded",
            rule_confidence_threshold=0.75,
        )

        result = cascade.recognize(self.request("I want that sort, but cotton instead."))

        self.assertEqual(result.source, RecognitionSource.RULE)
        self.assertEqual(
            cascade.statistics()["fallback_reasons"],
            {"evidence_not_grounded": 1},
        )

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

    def test_turn1_override_tail_captured_as_soft_constraint(self) -> None:
        rec = RuleBasedRecognizer()
        request = self.request(
            "I'm looking for Tops & Tees Tanks & Camis. Long torso camisole for extra long torso."
        )
        result = rec.recognize(request)
        values = [op.value for op in result.constraint_operations]
        self.assertIn("long torso camisole for extra long torso", " ".join(values).lower())
        self.assertTrue(
            all(op.strength == ConstraintStrength.SOFT for op in result.constraint_operations)
        )

    def test_turn1_browsing_tail_not_captured(self) -> None:
        rec = RuleBasedRecognizer()
        request = self.request("I'm looking for running shoes, but I'm still exploring.")
        result = rec.recognize(request)
        self.assertEqual(result.constraint_operations, ())

    def test_additional_preference_for_other_is_no_preference_without_boundary(self) -> None:
        rec = RuleBasedRecognizer()
        result = rec.recognize(self.request("I don't have an additional preference for other."))
        self.assertEqual(result.dialogue_act, DialogueAct.NO_PREFERENCE)
        self.assertEqual(result.constraint_operations[0].attribute, "other")
        self.assertFalse(result.boundary_signal)

    def test_boundary_a_preference_is_no_preference_with_boundary_signal(self) -> None:
        rec = RuleBasedRecognizer()
        result = rec.recognize(
            self.request("I don't have a preference for other; please use your judgment.")
        )
        self.assertEqual(result.dialogue_act, DialogueAct.NO_PREFERENCE)
        self.assertTrue(result.boundary_signal)


if __name__ == "__main__":
    unittest.main()
