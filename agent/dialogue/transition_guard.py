from __future__ import annotations

from collections import Counter
from dataclasses import replace

from agent.dialogue.models import (
    ConstraintStrength,
    DialogueAct,
    DialogueState,
    GuardAction,
    GuardDecision,
    OperationKind,
    RecognitionResult,
)
from config.models import TransitionGuardConfig
from utils import session_utils as su


class TransitionGuard:
    """Evaluates whether a recognized transition is safe to apply.

    Evaluation preserves its inputs; only aggregate diagnostic counters change.
    """

    def __init__(self, config: TransitionGuardConfig) -> None:
        self.config = config
        self._actions: Counter[str] = Counter()
        self._reasons: Counter[str] = Counter()
        self._dialogue_acts: Counter[str] = Counter()
        self._sources: Counter[str] = Counter()

    def evaluate(
        self,
        state: DialogueState,
        recognition: RecognitionResult,
    ) -> GuardDecision:
        if not self.config.enabled:
            return self._decision(GuardAction.APPLY, recognition, "guard_disabled")

        confidence = min(
            recognition.confidence,
            min(
                (item.confidence for item in recognition.constraint_operations),
                default=recognition.confidence,
            ),
        )
        act = recognition.dialogue_act

        if act == DialogueAct.ADD_CONSTRAINT and confidence < self.config.add_min_confidence:
            return self._decision(
                GuardAction.SOFTEN,
                replace(
                    recognition,
                    constraint_operations=tuple(
                        replace(item, strength=ConstraintStrength.SOFT)
                        for item in recognition.constraint_operations
                    ),
                ),
                "add_confidence_below_threshold",
            )

        if act == DialogueAct.REJECT_PRODUCTS and not recognition.explicit_rejected_asins:
            if confidence >= self.config.add_min_confidence:
                return self._decision(
                    GuardAction.APPLY,
                    recognition,
                    "generic_rejection_soft_demote",
                )
            return self._destructive_failure(
                recognition,
                "generic_rejection_confidence_below_add_threshold",
            )

        destructive_thresholds = {
            DialogueAct.REPLACE_CONSTRAINT: self.config.replace_min_confidence,
            DialogueAct.REMOVE_CONSTRAINT: self.config.remove_min_confidence,
            DialogueAct.REJECT_PRODUCTS: self.config.reject_products_min_confidence,
        }
        if act in destructive_thresholds and confidence < destructive_thresholds[act]:
            reason = {
                DialogueAct.REPLACE_CONSTRAINT: "replace_confidence_below_threshold",
                DialogueAct.REMOVE_CONSTRAINT: "remove_confidence_below_threshold",
                DialogueAct.REJECT_PRODUCTS: "reject_products_confidence_below_threshold",
            }[act]
            return self._destructive_failure(recognition, reason)

        if act == DialogueAct.REPLACE_CONSTRAINT:
            valid = bool(recognition.constraint_operations) and all(
                item.operation == OperationKind.REPLACE
                and item.value.strip()
                and item.evidence.strip()
                for item in recognition.constraint_operations
            )
            if not valid:
                return self._destructive_failure(recognition, "replace_missing_explicit_evidence")

        if act == DialogueAct.REMOVE_CONSTRAINT:
            active_keys = {
                (item.attribute, su.constraint_key(item.value))
                for item in state.active_constraints
            }
            targets = {
                (item.attribute, su.constraint_key(item.value))
                for item in recognition.constraint_operations
                if item.operation == OperationKind.REMOVE
            }
            if not targets or not targets.issubset(active_keys):
                return self._destructive_failure(recognition, "remove_target_absent")

        if act == DialogueAct.NO_PREFERENCE:
            explicit_attributes = {
                item.attribute
                for item in recognition.constraint_operations
                if item.operation == OperationKind.REMOVE
            }
            if confidence < self.config.no_preference_min_confidence or not explicit_attributes:
                return self._decision(
                    GuardAction.CLARIFY,
                    recognition,
                    "no_preference_attribute_unclear",
                    min(explicit_attributes) if explicit_attributes else "other",
                )

        if act == DialogueAct.NO_MORE_PREFERENCES:
            if not recognition.explicit_no_more_preferences:
                return self._decision(
                    GuardAction.CLARIFY,
                    recognition,
                    "no_more_preferences_not_grounded",
                    "other",
                )
            if confidence < self.config.no_more_preferences_min_confidence:
                return self._decision(
                    GuardAction.CLARIFY,
                    recognition,
                    "no_more_preferences_confidence_below_threshold",
                    "other",
                )

        return self._decision(GuardAction.APPLY, recognition, "guard_passed")

    def _destructive_failure(
        self,
        recognition: RecognitionResult,
        reason: str,
    ) -> GuardDecision:
        attribute = (
            recognition.constraint_operations[0].attribute
            if recognition.constraint_operations
            else "other"
        )
        return self._decision(
            GuardAction(self.config.destructive_failure_action),
            recognition,
            reason,
            attribute,
        )

    def _decision(
        self,
        action: GuardAction,
        recognition: RecognitionResult,
        reason: str,
        clarify_attribute: str | None = None,
    ) -> GuardDecision:
        self._actions[action.value] += 1
        self._reasons[reason] += 1
        self._dialogue_acts[recognition.dialogue_act.value] += 1
        self._sources[recognition.source.value] += 1
        return GuardDecision(action, recognition, reason, clarify_attribute)

    def statistics(self) -> dict[str, object]:
        return {
            "enabled": self.config.enabled,
            "total": sum(self._actions.values()),
            "actions": dict(sorted(self._actions.items())),
            "reasons": dict(sorted(self._reasons.items())),
            "dialogue_acts": dict(sorted(self._dialogue_acts.items())),
            "recognition_sources": dict(sorted(self._sources.items())),
        }
