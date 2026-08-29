from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from agent.dialogue.models import (
    Constraint,
    ConstraintStrength,
    DialogueState,
    Polarity,
    RecognitionRequest,
)
from agent.dialogue.recognizers.llm import LLMIntentRecognizer
from agent.dialogue.recognizers.rule_based import RuleBasedRecognizer
from config import load_config
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
        recognizer = RuleBasedRecognizer()
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
                actual_operation = (
                    result.constraint_operations[0] if result.constraint_operations else None
                )
                if (
                    result.dialogue_act.value == expected["dialogue_act"]
                    and actual_operation is not None
                    and actual_operation.operation.value == expected.get("operation")
                    and actual_operation.attribute == expected.get("attribute")
                    and actual_operation.value == expected.get("value")
                ):
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
