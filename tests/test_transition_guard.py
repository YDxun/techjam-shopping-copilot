from __future__ import annotations

import unittest

from agent.dialogue.models import (
    Constraint,
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
from agent.dialogue.transition_guard import TransitionGuard
from config.models import TransitionGuardConfig


def operation(
    kind: OperationKind,
    attribute: str,
    value: str,
    *,
    confidence: float = 0.95,
    evidence: str | None = None,
) -> ConstraintOperation:
    return ConstraintOperation(
        operation=kind,
        attribute=attribute,
        value=value,
        polarity=Polarity.INCLUDE,
        strength=ConstraintStrength.HARD,
        evidence=value if evidence is None else evidence,
        confidence=confidence,
    )


def recognition(
    act: DialogueAct,
    *operations: ConstraintOperation,
    confidence: float = 0.95,
    rejected_asins: tuple[str, ...] = (),
    source: RecognitionSource = RecognitionSource.RULE,
) -> RecognitionResult:
    return RecognitionResult(
        dialogue_act=act,
        category=None,
        constraint_operations=tuple(operations),
        explicit_rejected_asins=rejected_asins,
        confidence=confidence,
        source=source,
        ambiguities=(),
    )


def empty_state() -> DialogueState:
    return DialogueState(session_id="s", user_profile={})


def state_with_style() -> DialogueState:
    return DialogueState(
        session_id="s",
        user_profile={},
        active_constraints=(
            Constraint(
                attribute="style",
                value="casual",
                polarity=Polarity.INCLUDE,
                strength=ConstraintStrength.HARD,
                evidence="casual",
                source_turn=1,
                tokens=("casual",),
            ),
        ),
    )


class TransitionGuardTest(unittest.TestCase):
    def test_disabled_guard_is_exact_passthrough(self) -> None:
        result = recognition(DialogueAct.REPLACE_CONSTRAINT, confidence=0.2)

        decision = TransitionGuard(TransitionGuardConfig(enabled=False)).evaluate(
            empty_state(), result
        )

        self.assertEqual(decision.action, GuardAction.APPLY)
        self.assertIs(decision.recognition, result)
        self.assertEqual(decision.reason_code, "guard_disabled")

    def test_low_confidence_add_is_softened_without_mutating_recognition(self) -> None:
        result = recognition(
            DialogueAct.ADD_CONSTRAINT,
            operation(OperationKind.ADD, "material", "cotton", confidence=0.60),
            confidence=0.60,
        )

        decision = TransitionGuard(TransitionGuardConfig(enabled=True)).evaluate(empty_state(), result)

        self.assertEqual(decision.action, GuardAction.SOFTEN)
        self.assertEqual(decision.recognition.constraint_operations[0].strength, ConstraintStrength.SOFT)
        self.assertEqual(result.constraint_operations[0].strength, ConstraintStrength.HARD)

    def test_low_confidence_replace_requests_attribute_clarification(self) -> None:
        result = recognition(
            DialogueAct.REPLACE_CONSTRAINT,
            operation(OperationKind.REPLACE, "material", "cotton", confidence=0.70),
            confidence=0.70,
        )

        decision = TransitionGuard(TransitionGuardConfig(enabled=True)).evaluate(
            state_with_style(), result
        )

        self.assertEqual(decision.action, GuardAction.CLARIFY)
        self.assertEqual(decision.clarify_attribute, "material")
        self.assertEqual(decision.reason_code, "replace_confidence_below_threshold")

    def test_replace_without_evidence_uses_destructive_failure_action(self) -> None:
        result = recognition(
            DialogueAct.REPLACE_CONSTRAINT,
            operation(OperationKind.REPLACE, "material", "cotton", evidence=""),
        )

        decision = TransitionGuard(TransitionGuardConfig(enabled=True)).evaluate(empty_state(), result)

        self.assertEqual(decision.action, GuardAction.CLARIFY)
        self.assertEqual(decision.reason_code, "replace_missing_explicit_evidence")

    def test_remove_without_an_active_normalized_target_requests_clarification(self) -> None:
        result = recognition(
            DialogueAct.REMOVE_CONSTRAINT,
            operation(OperationKind.REMOVE, "material", "  COTTON "),
        )

        decision = TransitionGuard(TransitionGuardConfig(enabled=True)).evaluate(
            state_with_style(), result
        )

        self.assertEqual(decision.action, GuardAction.CLARIFY)
        self.assertEqual(decision.reason_code, "remove_target_absent")
        self.assertEqual(decision.clarify_attribute, "material")

    def test_remove_with_active_normalized_target_applies(self) -> None:
        state = DialogueState(
            session_id="s",
            user_profile={},
            active_constraints=(
                Constraint(
                    attribute="material",
                    value="cotton",
                    polarity=Polarity.INCLUDE,
                    strength=ConstraintStrength.HARD,
                    evidence="cotton",
                    source_turn=1,
                    tokens=("cotton",),
                ),
            ),
        )
        result = recognition(
            DialogueAct.REMOVE_CONSTRAINT,
            operation(OperationKind.REMOVE, "material", " COTTON "),
        )

        decision = TransitionGuard(TransitionGuardConfig(enabled=True)).evaluate(state, result)

        self.assertEqual(decision.action, GuardAction.APPLY)
        self.assertEqual(decision.reason_code, "guard_passed")

    def test_explicit_product_rejection_at_threshold_applies(self) -> None:
        result = recognition(
            DialogueAct.REJECT_PRODUCTS,
            confidence=0.90,
            rejected_asins=("B001",),
        )

        decision = TransitionGuard(TransitionGuardConfig(enabled=True)).evaluate(empty_state(), result)

        self.assertEqual(decision.action, GuardAction.APPLY)
        self.assertEqual(decision.reason_code, "guard_passed")

    def test_generic_product_rejection_above_add_threshold_applies_soft_demotion_path(self) -> None:
        result = recognition(DialogueAct.REJECT_PRODUCTS, confidence=0.65)

        decision = TransitionGuard(TransitionGuardConfig(enabled=True)).evaluate(empty_state(), result)

        self.assertEqual(decision.action, GuardAction.APPLY)
        self.assertEqual(decision.reason_code, "generic_rejection_soft_demote")

    def test_low_confidence_generic_product_rejection_uses_destructive_failure_action(self) -> None:
        result = recognition(DialogueAct.REJECT_PRODUCTS, confidence=0.64)

        decision = TransitionGuard(TransitionGuardConfig(enabled=True)).evaluate(empty_state(), result)

        self.assertEqual(decision.action, GuardAction.CLARIFY)
        self.assertEqual(decision.reason_code, "generic_rejection_confidence_below_add_threshold")

    def test_no_preference_without_explicit_attribute_requests_other_clarification(self) -> None:
        result = recognition(DialogueAct.NO_PREFERENCE, confidence=0.95)

        decision = TransitionGuard(TransitionGuardConfig(enabled=True)).evaluate(empty_state(), result)

        self.assertEqual(decision.action, GuardAction.CLARIFY)
        self.assertEqual(decision.reason_code, "no_preference_attribute_unclear")
        self.assertEqual(decision.clarify_attribute, "other")

    def test_low_confidence_no_more_preferences_requests_clarification(self) -> None:
        result = recognition(DialogueAct.NO_MORE_PREFERENCES, confidence=0.94)

        decision = TransitionGuard(TransitionGuardConfig(enabled=True)).evaluate(empty_state(), result)

        self.assertEqual(decision.action, GuardAction.CLARIFY)
        self.assertEqual(decision.reason_code, "no_more_preferences_confidence_below_threshold")
        self.assertEqual(decision.clarify_attribute, "other")

    def test_statistics_are_aggregate_sorted_and_exclude_evidence(self) -> None:
        guard = TransitionGuard(TransitionGuardConfig(enabled=True))
        guard.evaluate(empty_state(), recognition(DialogueAct.ADD_CONSTRAINT, confidence=0.60))
        guard.evaluate(
            empty_state(),
            recognition(DialogueAct.NO_MORE_PREFERENCES, confidence=0.94, source=RecognitionSource.LLM),
        )

        self.assertEqual(
            guard.statistics(),
            {
                "enabled": True,
                "total": 2,
                "actions": {"clarify": 1, "soften": 1},
                "reasons": {
                    "add_confidence_below_threshold": 1,
                    "no_more_preferences_confidence_below_threshold": 1,
                },
                "dialogue_acts": {"add_constraint": 1, "no_more_preferences": 1},
                "recognition_sources": {"llm": 1, "rule": 1},
            },
        )


if __name__ == "__main__":
    unittest.main()
