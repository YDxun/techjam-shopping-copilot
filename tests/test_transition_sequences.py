from __future__ import annotations

import unittest

from agent.dialogue.models import (
    ConstraintOperation,
    ConstraintStrength,
    DialogueAct,
    DialogueState,
    GuardAction,
    OperationKind,
    Polarity,
    RecognitionResult,
    RecognitionSource,
)
from agent.dialogue.reducer import StateReducer
from agent.dialogue.transition_guard import TransitionGuard
from config.models import TransitionGuardConfig


def recognition(
    dialogue_act: DialogueAct,
    *operations: ConstraintOperation,
    category: str | None = None,
    confidence: float = 0.95,
) -> RecognitionResult:
    return RecognitionResult(
        dialogue_act=dialogue_act,
        category=category,
        constraint_operations=tuple(operations),
        explicit_rejected_asins=(),
        confidence=confidence,
        source=RecognitionSource.RULE,
        ambiguities=(),
    )


def operation(
    kind: OperationKind,
    attribute: str,
    value: str,
    *,
    confidence: float = 0.95,
) -> ConstraintOperation:
    return ConstraintOperation(
        operation=kind,
        attribute=attribute,
        value=value,
        polarity=Polarity.INCLUDE,
        strength=ConstraintStrength.HARD,
        evidence=value,
        confidence=confidence,
    )


def constraint_summary(state: DialogueState) -> list[tuple[str, str, str]]:
    return [
        (item.attribute, item.value, item.strength.value) for item in state.active_constraints
    ]


class TransitionSequenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.reducer = StateReducer()
        self.guard = TransitionGuard(TransitionGuardConfig(enabled=True))
        self.initial = DialogueState(session_id="sequence", user_profile={})

    def apply(self, state: DialogueState, result: RecognitionResult, turn: int) -> DialogueState:
        decision = self.guard.evaluate(state, result)
        if decision.action in {GuardAction.APPLY, GuardAction.SOFTEN}:
            reduced = self.reducer.reduce(state, decision.recognition, turn)
            self.assertTrue(reduced.applied)
            return reduced.state
        return state

    def test_add_replace_remove_has_literal_final_state(self) -> None:
        state = self.apply(
            self.initial,
            recognition(
                DialogueAct.ADD_CONSTRAINT,
                operation(OperationKind.ADD, "style", "casual"),
                category="shirts",
            ),
            1,
        )
        state = self.apply(
            state,
            recognition(
                DialogueAct.REPLACE_CONSTRAINT,
                operation(OperationKind.REPLACE, "material", "cotton"),
            ),
            2,
        )
        final = self.apply(
            state,
            recognition(
                DialogueAct.REMOVE_CONSTRAINT,
                operation(OperationKind.REMOVE, "material", "cotton"),
            ),
            3,
        )

        self.assertEqual(final.intent_version, 2)
        self.assertEqual(
            constraint_summary(final),
            [("style", "casual", "soft")],
        )
        self.assertEqual(final.no_preference_attributes, frozenset())
        self.assertFalse(final.no_more_preferences)

    def test_generic_rejection_then_intent_override_preserves_soft_demotion_path(self) -> None:
        state = self.apply(
            self.initial,
            recognition(
                DialogueAct.ADD_CONSTRAINT,
                operation(OperationKind.ADD, "style", "casual"),
                category="shirts",
            ),
            1,
        )
        rejection = recognition(DialogueAct.REJECT_PRODUCTS, confidence=0.65)
        decision = self.guard.evaluate(state, rejection)
        self.assertEqual(decision.action, GuardAction.APPLY)
        self.assertEqual(decision.reason_code, "generic_rejection_soft_demote")
        state = self.reducer.reduce(state, decision.recognition, 2).state
        final = self.apply(
            state,
            recognition(
                DialogueAct.REPLACE_CONSTRAINT,
                operation(OperationKind.REPLACE, "material", "cotton"),
            ),
            3,
        )

        self.assertEqual(final.intent_version, 2)
        self.assertEqual(
            constraint_summary(final),
            [("style", "casual", "soft"), ("material", "cotton", "hard")],
        )
        self.assertEqual(final.no_preference_attributes, frozenset())
        self.assertFalse(final.no_more_preferences)

    def test_no_more_then_new_intent_clears_stop_state(self) -> None:
        state = self.apply(
            self.initial,
            recognition(
                DialogueAct.ADD_CONSTRAINT,
                operation(OperationKind.ADD, "style", "casual"),
                category="shirts",
            ),
            1,
        )
        state = self.apply(
            state,
            RecognitionResult(
                dialogue_act=DialogueAct.NO_MORE_PREFERENCES,
                category=None,
                constraint_operations=(),
                explicit_rejected_asins=(),
                confidence=0.95,
                source=RecognitionSource.RULE,
                ambiguities=(),
                explicit_no_more_preferences=True,
            ),
            2,
        )
        final = self.apply(
            state,
            recognition(
                DialogueAct.REPLACE_CONSTRAINT,
                operation(OperationKind.REPLACE, "material", "cotton"),
                category="running shoes",
            ),
            3,
        )

        self.assertEqual(final.intent_version, 2)
        self.assertEqual(final.category, "running shoes")
        self.assertEqual(
            constraint_summary(final),
            [("style", "casual", "soft"), ("material", "cotton", "hard")],
        )
        self.assertEqual(final.no_preference_attributes, frozenset())
        self.assertFalse(final.no_more_preferences)

    def test_blocked_destructive_decision_keeps_the_prior_state_exactly(self) -> None:
        state = self.apply(
            self.initial,
            recognition(
                DialogueAct.ADD_CONSTRAINT,
                operation(OperationKind.ADD, "style", "casual"),
                category="shirts",
            ),
            1,
        )
        blocked = recognition(
            DialogueAct.REPLACE_CONSTRAINT,
            operation(OperationKind.REPLACE, "material", "cotton", confidence=0.70),
            confidence=0.70,
        )

        decision = self.guard.evaluate(state, blocked)
        final = self.apply(state, blocked, 2)

        self.assertEqual(decision.action, GuardAction.CLARIFY)
        self.assertEqual(final, state)


if __name__ == "__main__":
    unittest.main()
