from __future__ import annotations

import re

from agent.dialogue.models import (
    ALLOWED_ATTRIBUTES,
    ConstraintOperation,
    ConstraintStrength,
    DialogueAct,
    OperationKind,
    Polarity,
    RecognitionRequest,
    RecognitionResult,
    RecognitionSource,
)
from config import constants
from utils import session_utils as su


RE_LOOKING_FOR = re.compile(r"looking for\s+(.+?)(?=,|\.\s|;|$)", re.I | re.S)
RE_KEY_REQUIREMENT = re.compile(
    r"key requirement is\s*[:：]?\s*(.+?)(?:\.\s*)?$", re.I | re.S
)
RE_WHAT_MATTERS = re.compile(
    r"what matters is\s*[:：]?\s*(.+?)(?:\.\s*)?$", re.I | re.S
)
RE_OVERRIDE = re.compile(
    r"ignore my earlier preference.*?what i need is\s*[:：]?\s*(.+?)(?:\.\s*)?$",
    re.I | re.S,
)
RE_NO_PREFERENCE = re.compile(r"(?:don't|do not) have a preference for\s+([a-z_]+)", re.I)
RE_NO_MORE = re.compile(
    r"(?:no additional preference|don.t have an additional preference)", re.I
)
RE_NOT_RIGHT = re.compile(r"(?:not quite right|not what i meant|reject)", re.I)
RE_ASIN = re.compile(r"\bB0[A-Z0-9]{8}\b", re.I)
RE_COMPLEX = re.compile(
    r"\b(?:rather than|instead of|except|not the|previous|former|latter|that sort)\b",
    re.I,
)


class RuleBasedRecognizer:
    """Pure deterministic parser for official phrases and bounded variants."""

    def __init__(self, max_evidence_length: int = 180) -> None:
        self.max_evidence_length = max_evidence_length

    def recognize(self, request: RecognitionRequest) -> RecognitionResult:
        text = request.user_message or ""
        category = self._category(text)
        operations: list[ConstraintOperation] = []
        ambiguities: list[str] = []
        dialogue_act = DialogueAct.AMBIGUOUS
        confidence = 0.35

        override = RE_OVERRIDE.search(text)
        no_more = RE_NO_MORE.search(text)
        no_preference = RE_NO_PREFERENCE.search(text)
        not_right = RE_NOT_RIGHT.search(text)

        if override:
            operations.append(
                self._operation(
                    OperationKind.REPLACE,
                    override.group(1),
                    ConstraintStrength.HARD,
                )
            )
            dialogue_act = DialogueAct.REPLACE_CONSTRAINT
            confidence = 0.95
        elif no_more:
            dialogue_act = DialogueAct.NO_MORE_PREFERENCES
            confidence = 0.98
        elif no_preference:
            attribute = no_preference.group(1).lower()
            if attribute not in ALLOWED_ATTRIBUTES:
                attribute = "other"
            operations.append(
                ConstraintOperation(
                    operation=OperationKind.REMOVE,
                    attribute=attribute,
                    value=attribute,
                    polarity=Polarity.INCLUDE,
                    strength=ConstraintStrength.SOFT,
                    evidence=no_preference.group(0)[: self.max_evidence_length],
                    confidence=0.95,
                )
            )
            dialogue_act = DialogueAct.NO_PREFERENCE
            confidence = 0.95
        else:
            for pattern, strength in (
                (RE_KEY_REQUIREMENT, ConstraintStrength.HARD),
                (RE_WHAT_MATTERS, ConstraintStrength.SOFT),
            ):
                match = pattern.search(text)
                if match:
                    for value in su.split_values(match.group(1))[:2]:
                        operations.append(self._operation(OperationKind.ADD, value, strength))
            if category is not None:
                dialogue_act = DialogueAct.NEW_SEARCH
                confidence = 0.95 if operations else 0.85
            elif operations:
                dialogue_act = DialogueAct.ADD_CONSTRAINT
                confidence = 0.92
            elif not_right:
                dialogue_act = DialogueAct.REJECT_PRODUCTS
                confidence = 0.85

        if not operations and dialogue_act not in {
            DialogueAct.NO_MORE_PREFERENCES,
            DialogueAct.NO_PREFERENCE,
        }:
            operations.extend(self._generic_operations(text))
            if operations and dialogue_act == DialogueAct.AMBIGUOUS:
                dialogue_act = DialogueAct.ADD_CONSTRAINT
                confidence = 0.75

        if RE_COMPLEX.search(text):
            ambiguities.append("complex_reference")
            confidence = min(confidence, 0.60)

        rejected = tuple(
            asin
            for asin in dict.fromkeys(match.upper() for match in RE_ASIN.findall(text))
            if asin in request.recently_shown_asins
        )
        if rejected:
            dialogue_act = DialogueAct.REJECT_PRODUCTS
            confidence = max(confidence, 0.95)

        return RecognitionResult(
            dialogue_act=dialogue_act,
            category=category,
            constraint_operations=self._deduplicate(operations),
            explicit_rejected_asins=rejected,
            confidence=confidence,
            source=RecognitionSource.RULE,
            ambiguities=tuple(ambiguities),
        )

    @staticmethod
    def _category(text: str) -> str | None:
        match = RE_LOOKING_FOR.search(text)
        if not match:
            return None
        value = su.normalize(match.group(1))
        return value or None

    def _operation(
        self,
        kind: OperationKind,
        value: str,
        strength: ConstraintStrength,
    ) -> ConstraintOperation:
        cleaned = value.strip()[: self.max_evidence_length]
        return ConstraintOperation(
            operation=kind,
            attribute=su.classify_attribute(cleaned),
            value=cleaned,
            polarity=Polarity.INCLUDE,
            strength=strength,
            evidence=cleaned,
            confidence=0.95,
        )

    def _generic_operations(self, text: str) -> list[ConstraintOperation]:
        lowered = text.lower()
        values: list[str] = []
        values.extend(material for material in constants.MATERIALS if material in lowered)
        color_match = re.search(
            r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b",
            lowered,
        )
        if color_match:
            values.append(f"color: {color_match.group(1)}")
        return [
            self._operation(OperationKind.ADD, value, ConstraintStrength.SOFT)
            for value in values[:2]
        ]

    @staticmethod
    def _deduplicate(
        operations: list[ConstraintOperation],
    ) -> tuple[ConstraintOperation, ...]:
        result: list[ConstraintOperation] = []
        seen: set[tuple[str, str, Polarity]] = set()
        for operation in operations:
            key = (
                operation.attribute,
                su.constraint_key(operation.value),
                operation.polarity,
            )
            if key not in seen:
                seen.add(key)
                result.append(operation)
        return tuple(result)
