from __future__ import annotations

import unittest

from agent.dialogue.models import (
    ConstraintOperation,
    ConstraintStrength,
    DialogueAct,
    OperationKind,
    Polarity,
    RecognitionResult,
    RecognitionSource,
)
from agent.dialogue.reducer import StateReducer


def recognition(
    dialogue_act: DialogueAct,
    *operations: ConstraintOperation,
    category: str | None = None,
) -> RecognitionResult:
    return RecognitionResult(
        dialogue_act=dialogue_act,
        category=category,
        constraint_operations=tuple(operations),
        explicit_rejected_asins=(),
        confidence=0.9,
        source=RecognitionSource.RULE,
        ambiguities=(),
    )


def operation(
    kind: OperationKind,
    attribute: str,
    value: str,
    *,
    strength: ConstraintStrength = ConstraintStrength.SOFT,
) -> ConstraintOperation:
    return ConstraintOperation(
        operation=kind,
        attribute=attribute,
        value=value,
        polarity=Polarity.INCLUDE,
        strength=strength,
        evidence=value,
        confidence=0.9,
    )


class StateReducerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.reducer = StateReducer(max_evidence_length=180)

    def test_invalid_operation_returns_the_original_state(self) -> None:
        state = self.reducer.new_state("s1", {"summary": "profile"})
        invalid = recognition(
            DialogueAct.REMOVE_CONSTRAINT,
            operation(OperationKind.REMOVE, "material", ""),
        )

        result = self.reducer.reduce(state, invalid, turn=1)

        self.assertIs(result.state, state)
        self.assertFalse(result.applied)
        self.assertEqual(result.reason_code, "invalid_constraint_operation")
        self.assertEqual(state.active_constraints, ())

    def test_add_constraint_returns_a_new_state_without_changing_version(self) -> None:
        state = self.reducer.new_state("s1", {})
        parsed = recognition(
            DialogueAct.ADD_CONSTRAINT,
            operation(
                OperationKind.ADD,
                "material",
                "cotton",
                strength=ConstraintStrength.HARD,
            ),
            category="running shoes",
        )

        result = self.reducer.reduce(state, parsed, turn=1)

        self.assertTrue(result.applied)
        self.assertEqual(result.state.intent_version, 1)
        self.assertEqual(result.state.category, "running shoes")
        self.assertEqual([item.value for item in result.state.active_constraints], ["cotton"])
        self.assertEqual(state.active_constraints, ())

    def test_replace_intent_increments_version_and_archives_old_constraints(self) -> None:
        initial = self.reducer.new_state("s1", {})
        with_old_preference = self.reducer.reduce(
            initial,
            recognition(
                DialogueAct.ADD_CONSTRAINT,
                operation(OperationKind.ADD, "style", "casual"),
                category="shoes",
            ),
            turn=1,
        ).state
        replacement = recognition(
            DialogueAct.REPLACE_CONSTRAINT,
            operation(
                OperationKind.REPLACE,
                "material",
                "cotton",
                strength=ConstraintStrength.HARD,
            ),
        )

        result = self.reducer.reduce(with_old_preference, replacement, turn=3)

        self.assertTrue(result.applied)
        self.assertEqual(result.state.intent_version, 2)
        self.assertEqual(result.state.category, "shoes")
        self.assertEqual([item.value for item in result.state.active_constraints], ["cotton"])
        self.assertEqual([item.value for item in result.state.removed_constraints], ["casual"])

    def test_no_more_preferences_is_recorded_without_constraint_mutation(self) -> None:
        state = self.reducer.new_state("s1", {})

        result = self.reducer.reduce(
            state,
            recognition(DialogueAct.NO_MORE_PREFERENCES),
            turn=4,
        )

        self.assertTrue(result.state.no_more_preferences)
        self.assertEqual(result.state.active_constraints, ())
        self.assertEqual(result.state.turn, 4)


if __name__ == "__main__":
    unittest.main()
