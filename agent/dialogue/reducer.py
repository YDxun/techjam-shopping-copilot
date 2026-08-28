from __future__ import annotations

import math
from dataclasses import replace

from agent.dialogue.models import (
    ALLOWED_ATTRIBUTES,
    Constraint,
    ConstraintOperation,
    ConstraintStrength,
    DialogueAct,
    DialogueState,
    OperationKind,
    Polarity,
    RecognitionResult,
    RecognitionSource,
    ReduceResult,
)
from utils import session_utils as su


class StateReducer:
    """The only component allowed to produce a changed DialogueState."""

    def __init__(self, max_evidence_length: int = 180) -> None:
        if max_evidence_length <= 0:
            raise ValueError("max_evidence_length must be > 0")
        self.max_evidence_length = max_evidence_length

    def new_state(self, session_id: str, user_profile: dict | None) -> DialogueState:
        return DialogueState(session_id=session_id, user_profile=dict(user_profile or {}))

    def reduce(
        self,
        state: DialogueState,
        recognition: RecognitionResult,
        turn: int,
    ) -> ReduceResult:
        reason = self._validate(recognition, turn)
        if reason is not None:
            return ReduceResult(state=state, applied=False, reason_code=reason)

        active = list(state.active_constraints)
        removed = list(state.removed_constraints)
        intent_version = state.intent_version
        category = state.category
        asked_attributes = state.asked_attributes
        no_preference_attributes = state.no_preference_attributes
        no_more_preferences = state.no_more_preferences

        is_new_intent = recognition.dialogue_act in {
            DialogueAct.NEW_SEARCH,
            DialogueAct.REPLACE_CONSTRAINT,
        } and bool(state.category or state.active_constraints)
        if is_new_intent:
            removed.extend(active)
            active = []
            intent_version += 1
            asked_attributes = ()
            no_preference_attributes = frozenset()
            no_more_preferences = False

        if recognition.category is not None:
            category = su.normalize(recognition.category)

        for operation in recognition.constraint_operations:
            self._apply_operation(active, removed, operation, turn)

        if recognition.dialogue_act == DialogueAct.NO_MORE_PREFERENCES:
            no_more_preferences = True
        if recognition.dialogue_act == DialogueAct.NO_PREFERENCE:
            no_preference_attributes = frozenset(
                {*no_preference_attributes, *(op.attribute for op in recognition.constraint_operations)}
            )

        new_state = replace(
            state,
            intent_version=intent_version,
            category=category,
            active_constraints=tuple(active),
            removed_constraints=tuple(removed),
            asked_attributes=asked_attributes,
            no_preference_attributes=no_preference_attributes,
            no_more_preferences=no_more_preferences,
            last_dialogue_act=recognition.dialogue_act,
            turn=turn,
        )
        return ReduceResult(state=new_state, applied=True, reason_code="applied")

    @staticmethod
    def record_question(state: DialogueState, attribute: str | None) -> DialogueState:
        if not attribute or attribute in state.asked_attributes:
            return state
        if attribute not in ALLOWED_ATTRIBUTES:
            raise ValueError("question attribute is not allowed")
        return replace(
            state,
            asked_attributes=(*state.asked_attributes, attribute),
        )

    def _validate(self, recognition: RecognitionResult, turn: int) -> str | None:
        if turn <= 0:
            return "invalid_turn"
        if not isinstance(recognition.dialogue_act, DialogueAct):
            return "invalid_dialogue_act"
        if not isinstance(recognition.source, RecognitionSource):
            return "invalid_recognition_source"
        if not math.isfinite(recognition.confidence) or not 0 <= recognition.confidence <= 1:
            return "invalid_recognition_confidence"
        if recognition.category is not None and not su.normalize(recognition.category):
            return "invalid_category"
        for operation in recognition.constraint_operations:
            if not self._valid_operation(operation):
                return "invalid_constraint_operation"
        return None

    def _valid_operation(self, operation: ConstraintOperation) -> bool:
        if not isinstance(operation.operation, OperationKind):
            return False
        if not isinstance(operation.polarity, Polarity):
            return False
        if not isinstance(operation.strength, ConstraintStrength):
            return False
        if operation.attribute not in ALLOWED_ATTRIBUTES:
            return False
        if not su.constraint_key(operation.value):
            return False
        if len(operation.evidence) > self.max_evidence_length:
            return False
        return math.isfinite(operation.confidence) and 0 <= operation.confidence <= 1

    @staticmethod
    def _apply_operation(
        active: list[Constraint],
        removed: list[Constraint],
        operation: ConstraintOperation,
        turn: int,
    ) -> None:
        key = (operation.attribute, su.constraint_key(operation.value), operation.polarity)
        if operation.operation in {OperationKind.REMOVE, OperationKind.REPLACE}:
            retained: list[Constraint] = []
            for existing in active:
                matches = (
                    existing.attribute == operation.attribute
                    if operation.operation == OperationKind.REPLACE
                    else existing.key == key
                )
                if matches:
                    removed.append(existing)
                else:
                    retained.append(existing)
            active[:] = retained

        if operation.operation == OperationKind.REMOVE:
            return

        candidate = Constraint(
            attribute=operation.attribute,
            value=operation.value.strip()[:180],
            polarity=operation.polarity,
            strength=operation.strength,
            evidence=operation.evidence[:180],
            source_turn=turn,
            tokens=su.group_tokens(operation.value),
        )
        for index, existing in enumerate(active):
            if existing.key == candidate.key:
                if candidate.hardness >= existing.hardness:
                    active[index] = candidate
                return
        active.append(candidate)
