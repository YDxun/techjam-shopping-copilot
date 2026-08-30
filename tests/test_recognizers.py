from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

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

    def chat(
        self,
        messages,
        *,
        temperature=None,
        max_tokens=None,
        request_options=None,
    ) -> LLMResult:
        self.calls.append((messages, temperature, max_tokens, request_options))
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

    def test_pure_english_prompt_does_not_include_chinese_guidance(self) -> None:
        recognizer = LLMIntentRecognizer(
            FakeLLMClient(successful_result("{}")),
            max_evidence_length=180,
        )

        system_prompt = recognizer._messages(
            self.request("I need a waterproof hiking jacket.")
        )[0]["content"]

        self.assertNotIn("Chinese-language input mode", system_prompt)

    def test_chinese_and_mixed_inputs_select_english_normalization_guidance(self) -> None:
        recognizer = LLMIntentRecognizer(
            FakeLLMClient(successful_result("{}")),
            max_evidence_length=180,
        )

        messages = (
            "我想买一件必须防水的徒步外套。",
            "我想買一件必須防水的徒步外套。",
            "我想要 waterproof 的外套，预算不超过 100 USD。",
        )
        for message in messages:
            with self.subTest(message=message):
                system_prompt = recognizer._messages(self.request(message))[0]["content"]
                self.assertIn("Chinese-language input mode", system_prompt)
                self.assertIn(
                    "category, constraint values, and ambiguity descriptions in English",
                    system_prompt,
                )
                self.assertIn("Evidence is the only language exception", system_prompt)
                self.assertIn("必须防水", system_prompt)
                self.assertIn("改成深蓝色", system_prompt)
                self.assertIn('"category":"jacket"', system_prompt)
                self.assertIn('"attribute":"use_case","value":"hiking"', system_prompt)
                self.assertNotIn('"category":"hiking jackets"', system_prompt)

    def test_chinese_prompt_includes_only_catalog_supported_canonical_values(self) -> None:
        vocabulary = SimpleNamespace(
            allowed_values={
                "category": ("jacket",),
                "color": ("navy",),
            },
            canonicalize=lambda attribute, value: value,
        )
        try:
            recognizer = LLMIntentRecognizer(
                FakeLLMClient(successful_result("{}")),
                max_evidence_length=180,
                normalization_vocabulary=vocabulary,
            )
        except TypeError as error:
            self.fail(f"recognizer rejected normalization vocabulary: {error}")

        system_prompt = recognizer._messages(
            self.request("我想买一件深蓝色外套。")
        )[0]["content"]

        self.assertIn('"category":["jacket"]', system_prompt)
        self.assertIn('"color":["navy"]', system_prompt)
        self.assertIn("use the exact canonical value", system_prompt)

    def test_chinese_llm_values_are_canonicalized_after_parsing(self) -> None:
        aliases = {
            ("category", "windbreaker"): "jacket",
            ("color", "navy blue"): "navy",
        }
        vocabulary = SimpleNamespace(
            allowed_values={"category": ("jacket",), "color": ("navy",)},
            canonicalize=lambda attribute, value: aliases.get((attribute, value), value),
        )
        try:
            recognizer = LLMIntentRecognizer(
                FakeLLMClient(successful_result("{}")),
                max_evidence_length=180,
                normalization_vocabulary=vocabulary,
            )
        except TypeError as error:
            self.fail(f"recognizer rejected normalization vocabulary: {error}")
        response = (
            '{"dialogue_act":"new_search","category":"windbreaker",'
            '"constraint_operations":[{"operation":"add","attribute":"color",'
            '"value":"navy blue","polarity":"include","strength":"soft",'
            '"evidence":"深蓝色","confidence":0.95}],'
            '"explicit_rejected_asins":[],"confidence":0.95,"ambiguities":[]}'
        )

        result = recognizer._parse(response, (), "我想买一件深蓝色外套。")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.category, "jacket")
        self.assertEqual(result.constraint_operations[0].value, "navy")

    def test_intent_request_uses_configured_structured_output_options(self) -> None:
        client = FakeLLMClient(
            successful_result(
                '{"dialogue_act":"ambiguous","category":null,'
                '"constraint_operations":[],"explicit_rejected_asins":[],'
                '"confidence":0.5,"ambiguities":[]}'
            )
        )
        request_options = SimpleNamespace(json_output=True, thinking_mode="disabled")
        try:
            recognizer = LLMIntentRecognizer(
                client,
                max_evidence_length=180,
                request_options=request_options,
            )
        except TypeError as error:
            self.fail(f"recognizer rejected structured request options: {error}")

        recognizer.recognize(self.request("我想看看别的选择。"))

        self.assertIs(client.calls[0][3], request_options)

    def test_chinese_evidence_parses_with_english_framework_values(self) -> None:
        recognizer = LLMIntentRecognizer(
            FakeLLMClient(successful_result("{}")),
            max_evidence_length=180,
        )
        response = (
            '{"dialogue_act":"new_search","category":"hiking jackets",'
            '"constraint_operations":[{"operation":"add","attribute":"feature",'
            '"value":"waterproof","polarity":"include","strength":"hard",'
            '"evidence":"必须防水","confidence":0.97}],'
            '"explicit_rejected_asins":[],"confidence":0.97,"ambiguities":[]}'
        )

        result = recognizer._parse(
            response,
            (),
            "我想买一件徒步外套，必须防水。",
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.dialogue_act, DialogueAct.NEW_SEARCH)
        self.assertEqual(result.category, "hiking jackets")
        self.assertEqual(result.constraint_operations[0].value, "waterproof")
        self.assertEqual(result.constraint_operations[0].evidence, "必须防水")

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

    def test_transition_guard_switch_controls_task_four_rule_generalizations(self) -> None:
        legacy = RuleBasedRecognizer()
        guarded = RuleBasedRecognizer(transition_guard_enabled=True)

        replacement_request = self.request("Switch the material to cotton.")
        legacy_replacement = legacy.recognize(replacement_request)
        guarded_replacement = guarded.recognize(replacement_request)

        self.assertEqual(legacy_replacement.dialogue_act, DialogueAct.ADD_CONSTRAINT)
        self.assertEqual(legacy_replacement.constraint_operations[0].operation.value, "add")
        self.assertEqual(guarded_replacement.dialogue_act, DialogueAct.REPLACE_CONSTRAINT)
        self.assertEqual(guarded_replacement.constraint_operations[0].operation.value, "replace")

        noisy_stop_request = self.request("No more prefernces.")
        self.assertEqual(legacy.recognize(noisy_stop_request).dialogue_act, DialogueAct.AMBIGUOUS)
        guarded_stop = guarded.recognize(noisy_stop_request)
        self.assertEqual(guarded_stop.dialogue_act, DialogueAct.NO_MORE_PREFERENCES)
        self.assertTrue(guarded_stop.explicit_no_more_preferences)

    def test_rule_no_more_signal_requires_an_explicit_user_phrase(self) -> None:
        result = RuleBasedRecognizer().recognize(self.request("No more preferences."))

        self.assertEqual(result.dialogue_act, DialogueAct.NO_MORE_PREFERENCES)
        self.assertTrue(result.explicit_no_more_preferences)

    def test_llm_no_more_signal_is_derived_from_the_user_message(self) -> None:
        recognizer = LLMIntentRecognizer(
            FakeLLMClient(successful_result("{}")), max_evidence_length=180
        )
        response = (
            '{"dialogue_act":"no_more_preferences","category":null,'
            '"constraint_operations":[],"explicit_rejected_asins":[],'
            '"confidence":0.99,"ambiguities":[]}'
        )

        grounded = recognizer._parse(response, (), "No more preferences.")
        ungrounded = recognizer._parse(response, (), "Please show me blue options.")

        self.assertIsNotNone(grounded)
        self.assertIsNotNone(ungrounded)
        assert grounded is not None
        assert ungrounded is not None
        self.assertTrue(grounded.explicit_no_more_preferences)
        self.assertFalse(ungrounded.explicit_no_more_preferences)

    def test_llm_chinese_no_more_signal_is_derived_from_the_user_message(self) -> None:
        recognizer = LLMIntentRecognizer(
            FakeLLMClient(successful_result("{}")), max_evidence_length=180
        )
        response = (
            '{"dialogue_act":"no_more_preferences","category":null,'
            '"constraint_operations":[],"explicit_rejected_asins":[],'
            '"confidence":0.99,"ambiguities":[]}'
        )

        for message in ("没有其他要求了。", "沒有其他要求了。"):
            with self.subTest(message=message):
                result = recognizer._parse(response, (), message)
                self.assertIsNotNone(result)
                assert result is not None
                self.assertTrue(result.explicit_no_more_preferences)


if __name__ == "__main__":
    unittest.main()
