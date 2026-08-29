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
RE_KEY_REQUIREMENT = re.compile(r"key requirement is\s*[:：]?\s*(.+?)(?:\.\s*)?$", re.I | re.S)
RE_WHAT_MATTERS = re.compile(r"what matters is\s*[:：]?\s*(.+?)(?:\.\s*)?$", re.I | re.S)
RE_OVERRIDE = re.compile(
    r"ignore my earlier preference.*?what i need is\s*[:：]?\s*(.+?)(?:\.\s*)?$",
    re.I | re.S,
)
RE_NO_PREFERENCE = re.compile(
    r"(?:don't|do not) have (?:an additional|a) preference for\s+([a-z_]+)",
    re.I,
)
RE_NO_MORE = re.compile(r"(?:no more preferences|no additional preferences)", re.I)
RE_NO_MORE_GENERALIZATION = re.compile(r"no more prefer(?:e)?nces", re.I)
RE_BOUNDARY = re.compile(
    r"don't have a preference for\s+([a-z_]+)[^.]*please use your judgment",
    re.I | re.S,
)
RE_NOT_RIGHT = re.compile(r"(?:not quite right|not what i meant|reject)", re.I)
RE_ASIN = re.compile(r"\bB0[A-Z0-9]{8}\b", re.I)
RE_REPLACE_CONSTRAINT = re.compile(
    r"\b(?:switch|change|replace)\s+(?:the\s+)?"
    r"(?P<attribute>material|color|size|style|brand|budget|feature|use_case)\s+"
    r"(?:to|for|with)\s+(?P<value>[a-z][a-z0-9 -]{0,80}?)(?=[.!?;,]|$)",
    re.I,
)
RE_REMOVE_VALUE_ATTRIBUTE = re.compile(
    r"\b(?:remove|drop)\s+(?:the\s+)?(?P<value>[a-z][a-z0-9 -]{0,80}?)\s+"
    r"(?P<attribute>material|color|size|style|brand|budget|feature|use_case)\b",
    re.I,
)
RE_NEGATED_REJECTION = re.compile(
    r"\b(?:don't|do not) mean reject all of (?:them|these)\b.*?"
    r"\b(?:only\s+)?(?:dislike|don't like)\s+the\s+"
    r"(?P<color>black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\s+"
    r"(?:one|ones)\b",
    re.I | re.S,
)
RE_NEGATED_DESTRUCTIVE = re.compile(
    r"\b(?:do not|don't|never)\s+(?:switch|change|replace|remove|drop)\b",
    re.I,
)
RE_NEGATED_EXPLICIT_REJECTION = re.compile(
    r"\b(?:do not|don't|never)\s+reject\s+(?:the\s+|asin\s+)?B0[A-Z0-9]{8}\b",
    re.I,
)
RE_COMPLEX = re.compile(
    r"\b(?:rather than|instead of|except|not the|previous|former|latter|that sort)\b",
    re.I,
)


class RuleBasedRecognizer:
    """Pure deterministic parser for official phrases and bounded variants."""

    def __init__(
        self,
        max_evidence_length: int = 180,
        *,
        transition_guard_enabled: bool = False,
    ) -> None:
        self.max_evidence_length = max_evidence_length
        self.transition_guard_enabled = transition_guard_enabled

    def recognize(self, request: RecognitionRequest) -> RecognitionResult:
        text = request.user_message or ""
        category = self._category(text)
        operations: list[ConstraintOperation] = []
        ambiguities: list[str] = []
        dialogue_act = DialogueAct.AMBIGUOUS
        confidence = 0.35

        override = RE_OVERRIDE.search(text)
        no_more = RE_NO_MORE.search(text)
        if no_more is None and self.transition_guard_enabled:
            no_more = RE_NO_MORE_GENERALIZATION.search(text)
        no_preference = RE_NO_PREFERENCE.search(text)
        not_right = RE_NOT_RIGHT.search(text)
        replacement = (
            RE_REPLACE_CONSTRAINT.search(text) if self.transition_guard_enabled else None
        )
        removal = (
            RE_REMOVE_VALUE_ATTRIBUTE.search(text) if self.transition_guard_enabled else None
        )
        negated_rejection = (
            RE_NEGATED_REJECTION.search(text) if self.transition_guard_enabled else None
        )
        negated_destructive = (
            RE_NEGATED_DESTRUCTIVE.search(text) if self.transition_guard_enabled else None
        )
        negated_explicit_rejection = (
            RE_NEGATED_EXPLICIT_REJECTION.search(text)
            if self.transition_guard_enabled
            else None
        )
        if negated_explicit_rejection:
            not_right = None

        if negated_destructive:
            ambiguities.append("negated_destructive_instruction")
        elif negated_explicit_rejection:
            ambiguities.append("negated_explicit_product_rejection")
        elif negated_rejection:
            color = negated_rejection.group("color").lower()
            operations.append(
                ConstraintOperation(
                    operation=OperationKind.ADD,
                    attribute="color",
                    value=color,
                    polarity=Polarity.EXCLUDE,
                    strength=ConstraintStrength.HARD,
                    evidence=negated_rejection.group(0)[: self.max_evidence_length],
                    confidence=0.95,
                )
            )
            dialogue_act = DialogueAct.ADD_CONSTRAINT
            confidence = 0.95
        elif replacement:
            operations.append(
                self._operation(
                    OperationKind.REPLACE,
                    replacement.group("value"),
                    ConstraintStrength.HARD,
                    attribute=replacement.group("attribute").lower(),
                )
            )
            dialogue_act = DialogueAct.REPLACE_CONSTRAINT
            confidence = 0.95
        elif removal:
            operations.append(
                self._operation(
                    OperationKind.REMOVE,
                    removal.group("value"),
                    ConstraintStrength.HARD,
                    attribute=removal.group("attribute").lower(),
                )
            )
            dialogue_act = DialogueAct.REMOVE_CONSTRAINT
            confidence = 0.95
        elif override:
            operations.append(
                self._operation(
                    OperationKind.REPLACE,
                    override.group(1),
                    ConstraintStrength.HARD,
                )
            )
            dialogue_act = DialogueAct.REPLACE_CONSTRAINT
            confidence = 0.95
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
        elif no_more:
            dialogue_act = DialogueAct.NO_MORE_PREFERENCES
            confidence = 0.98
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

        if (
            not negated_destructive
            and not negated_explicit_rejection
            and not operations
            and dialogue_act not in {
            DialogueAct.NO_MORE_PREFERENCES,
            DialogueAct.NO_PREFERENCE,
            }
        ):
            operations.extend(self._generic_operations(text))
            if operations and dialogue_act == DialogueAct.AMBIGUOUS:
                dialogue_act = DialogueAct.ADD_CONSTRAINT
                confidence = 0.75

        # turn-1 尾部旧偏好捕获（与 0.995 HR 基线行为一致）：
        # intent_override 首条消息为 "I'm looking for {cat}. {old_value}"，
        # 把 {old_value} 作为 soft 约束立刻获得排序信号；buying/browsing 尾部含
        # "key requirement"/"still exploring" 等标记时自动跳过（避免重复/噪声）。
        if not negated_destructive and not negated_explicit_rejection:
            operations.extend(self._turn1_tail_operations(text, request.turn))

        if RE_COMPLEX.search(text):
            ambiguities.append("complex_reference")
            confidence = min(confidence, 0.60)

        rejected = ()
        if not negated_explicit_rejection:
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
            boundary_signal=bool(RE_BOUNDARY.search(text)),
            explicit_no_more_preferences=bool(no_more),
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
        attribute: str | None = None,
    ) -> ConstraintOperation:
        cleaned = value.strip()[: self.max_evidence_length]
        return ConstraintOperation(
            operation=kind,
            attribute=attribute or su.classify_attribute(cleaned),
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

    def _turn1_tail_operations(self, text: str, turn: int) -> list[ConstraintOperation]:
        """turn 1：'looking for X. <tail>' 的 <tail> 若不含标记，捕获为 soft 约束。"""
        if turn != 1:
            return []
        match = RE_LOOKING_FOR.search(text)
        if not match:
            return []
        tail = text[match.end() :].strip(" .;, -\t\n")  # noqa: B005
        low = tail.lower()
        if not tail or any(
            marker in low for marker in ("key requirement", "still exploring", "what matters")
        ):
            return []
        return [
            self._operation(
                OperationKind.ADD,
                tail[: self.max_evidence_length],
                ConstraintStrength.SOFT,
            )
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
