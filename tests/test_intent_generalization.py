from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from agent.dialogue.models import (
    Constraint,
    ConstraintStrength,
    DialogueState,
    GuardAction,
    Polarity,
    RecognitionRequest,
    RecognitionResult,
)
from agent.dialogue.recognizers.llm import LLMIntentRecognizer
from agent.dialogue.recognizers.rule_based import RuleBasedRecognizer
from agent.dialogue.reducer import StateReducer
from agent.dialogue.transition_guard import TransitionGuard
from config import load_config
from config.models import TransitionGuardConfig
from llm import create_llm_client
from llm.base import LLMState

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "intent" / "generalization.jsonl"
DESTRUCTIVE_ACTS = frozenset({"replace_constraint", "remove_constraint", "reject_products"})


def load_corpus() -> list[dict[str, object]]:
    with FIXTURE_PATH.open(encoding="utf-8") as fixture_file:
        return [json.loads(line) for line in fixture_file if line.strip()]


def state_from_fixture(row: dict[str, object]) -> DialogueState:
    declared = row["state"]
    assert isinstance(declared, dict)
    constraints = declared.get("constraints", [])
    assert isinstance(constraints, list)
    active = tuple(
        Constraint(
            attribute=item["attribute"],
            value=item["value"],
            polarity=Polarity.INCLUDE,
            strength=ConstraintStrength(item["strength"]),
            evidence=item["value"],
            source_turn=1,
            tokens=(item["value"],),
        )
        for item in constraints
    )
    return DialogueState(
        session_id="fixture",
        user_profile={},
        category=str(declared.get("category", "")),
        active_constraints=active,
    )


def request_from_fixture(row: dict[str, object]) -> RecognitionRequest:
    declared = row["state"]
    assert isinstance(declared, dict)
    return RecognitionRequest(
        user_message=str(row["message"]),
        turn=2,
        state=state_from_fixture(row),
        recently_shown_asins=tuple(declared.get("recently_shown_asins", [])),
    )


def live_destructive_match(expected: dict[str, object], result: RecognitionResult) -> bool:
    if result.dialogue_act.value != expected["dialogue_act"]:
        return False
    if result.dialogue_act.value == "reject_products":
        expected_rejections = expected.get("explicit_rejected_asins")
        return isinstance(expected_rejections, list) and tuple(expected_rejections) == (
            result.explicit_rejected_asins
        )
    operation = result.constraint_operations[0] if result.constraint_operations else None
    return (
        operation is not None
        and operation.operation.value == expected.get("operation")
        and operation.attribute == expected.get("attribute")
        and operation.value == expected.get("value")
    )


class IntentGeneralizationTest(unittest.TestCase):
    def test_fixture_corpus_has_reviewed_coverage(self) -> None:
        rows = load_corpus()
        self.assertTrue(rows)
        self.assertEqual(
            {row["expected"]["dialogue_act"] for row in rows},
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
            {
                row["expected"]["operation"]
                for row in rows
                if "operation" in row["expected"]
            },
            {"add", "replace", "remove"},
        )
        all_tags = {tag for row in rows for tag in row["tags"]}
        self.assertTrue(
            {"negation", "correction", "context_reference", "noisy_spelling"} <= all_tags
        )

    def test_rule_recognizer_matches_literal_reviewed_expectations(self) -> None:
        recognizer = RuleBasedRecognizer(transition_guard_enabled=True)
        for row in load_corpus():
            with self.subTest(fixture_id=row["id"]):
                result = recognizer.recognize(request_from_fixture(row))
                expected = row["expected"]
                self.assertEqual(result.dialogue_act.value, expected["dialogue_act"])
                if "category" in expected:
                    self.assertEqual(result.category, expected["category"])
                if "explicit_rejected_asins" in expected:
                    self.assertEqual(
                        list(result.explicit_rejected_asins), expected["explicit_rejected_asins"]
                    )
                if "operation_count" in expected:
                    self.assertEqual(len(result.constraint_operations), expected["operation_count"])
                if "operation" in expected:
                    self.assertTrue(result.constraint_operations)
                    operation = result.constraint_operations[0]
                    self.assertEqual(operation.operation.value, expected["operation"])
                    for field in ("attribute", "value", "polarity"):
                        if field in expected:
                            actual = (
                                operation.polarity.value
                                if field == "polarity"
                                else getattr(operation, field)
                            )
                            self.assertEqual(actual, expected[field])

    def test_live_destructive_metric_counts_explicit_product_rejection(self) -> None:
        row = next(row for row in load_corpus() if row["id"] == "reject_shown_01")
        result = RuleBasedRecognizer(transition_guard_enabled=True).recognize(
            request_from_fixture(row)
        )

        self.assertTrue(live_destructive_match(row["expected"], result))

    def test_turn_one_negated_destructive_tail_does_not_mutate_state(self) -> None:
        rows = {row["id"]: row for row in load_corpus()}
        recognizer = RuleBasedRecognizer(transition_guard_enabled=True)
        reducer = StateReducer()
        guard = TransitionGuard(TransitionGuardConfig(enabled=True))

        for fixture_id in ("negated_replace_01", "negated_remove_01"):
            with self.subTest(fixture_id=fixture_id):
                row = rows[fixture_id]
                state = DialogueState(session_id="turn-one", user_profile={})
                request = RecognitionRequest(
                    user_message=f"I am looking for shirts. {row['message']}",
                    turn=1,
                    state=state,
                )

                recognition = recognizer.recognize(request)
                decision = guard.evaluate(state, recognition)
                reduced = reducer.reduce(state, decision.recognition, turn=1)

                self.assertEqual(recognition.constraint_operations, ())
                self.assertEqual(decision.action, GuardAction.APPLY)
                self.assertTrue(reduced.applied)
                self.assertEqual(reduced.state.category, "shirts")
                self.assertEqual(reduced.state.active_constraints, ())


@unittest.skipUnless(os.environ.get("RUN_LIVE_LLM") == "1", "live LLM disabled")
class LiveIntentGeneralizationTest(unittest.TestCase):
    def test_live_corpus_reports_aggregate_quality(self) -> None:
        config = load_config()
        client = create_llm_client(config.llm)
        status = client.initialize()
        if config.llm.provider != "deepseek" or status.state != LLMState.AVAILABLE:
            self.skipTest("DeepSeek is not available for live intent evaluation")

        recognizer = LLMIntentRecognizer(client, max_evidence_length=180)
        rows = load_corpus()
        schema_valid = 0
        fallback_count = 0
        predicted_destructive = 0
        precise_destructive = 0

        for row in rows:
            result = recognizer.recognize(request_from_fixture(row))
            if result is None:
                fallback_count += 1
                continue
            schema_valid += 1
            if result.dialogue_act.value in DESTRUCTIVE_ACTS:
                predicted_destructive += 1
                expected = row["expected"]
                if live_destructive_match(expected, result):
                    precise_destructive += 1

        total = len(rows)
        destructive_precision = (
            precise_destructive / predicted_destructive if predicted_destructive else 0.0
        )
        print(
            "live_intent_quality "
            f"schema_valid_rate={schema_valid / total:.3f} "
            f"destructive_precision={destructive_precision:.3f} "
            f"fallback_rate={fallback_count / total:.3f}"
        )
        self.assertGreater(total, 0)


if __name__ == "__main__":
    unittest.main()
