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
RE_BOUNDARY = re.compile(
    r"don't have a preference for\s+([a-z_]+)[^.]*please use your judgment",
    re.I | re.S,
)
RE_NOT_RIGHT = re.compile(r"(?:not quite right|not what i meant|reject)", re.I)
RE_ASIN = re.compile(r"\bB0[A-Z0-9]{8}\b", re.I)
RE_COMPLEX = re.compile(
    r"\b(?:rather than|instead of|except|not the|previous|former|latter|that sort)\b",
    re.I,
)

# Part A (P0): necessity cue-word list -- hitting any word upgrades constraints extracted on the
# generalized ADD path to HARD.
# Official templates ("A key requirement is" / "what matters is" / override) are handled by earlier
# branches and never go through the generalized path.
_HARD_CUES = (
    "must", "need", "needs", "has to", "have to", "require", "requires",
    "important", "crucial", "essential", "key", "the most important thing",
)
_HARD_CUE_PHRASES = ("has to", "have to", "the most important thing")
_HARD_CUE_WORDS = (
    "must", "need", "needs", "require", "requires", "important", "crucial", "essential", "key"
)

# Value capture after a cue word: "I need waterproof" -> waterproof (covers attribute values beyond
# MATERIALS)
RE_CUE_VALUE = re.compile(
    r"(?:must|need|needs|has to|have to|require|requires|important|crucial|essential|"
    r"key|the most important thing)\s*(?:is|be|to be|that is)?\s*[:：]?\s*"
    r"([a-z][a-z0-9%\- ]{2,60}?)(?=[,.;]|$)",
    re.I | re.S,
)


class RuleBasedRecognizer:
    """Pure deterministic parser for official phrases and bounded variants."""

    def __init__(
        self,
        max_evidence_length: int = 180,
        paraphrase_enabled: bool = False,
        hard_cue_enabled: bool = True,
    ) -> None:
        self.max_evidence_length = max_evidence_length
        self.paraphrase_enabled = paraphrase_enabled
        self.hard_cue_enabled = hard_cue_enabled
        self._paraphrase_patterns: list[tuple[re.Pattern, str]] = []
        if paraphrase_enabled:
            try:
                from utils.data_assets import load_assets

                self._paraphrase_patterns = load_assets().paraphrase_patterns()
            except Exception:
                self._paraphrase_patterns = []

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

        if not operations and dialogue_act not in {
            DialogueAct.NO_MORE_PREFERENCES,
            DialogueAct.NO_PREFERENCE,
        }:
            # Part A: necessity cue -> upgrade constraints extracted on the generalized ADD path to
            # HARD (this path only; official template branches take priority)
            hard_cue = self.hard_cue_enabled and self._hard_cue_present(text)
            operations.extend(self._generic_operations(text, hard=hard_cue))
            if operations and dialogue_act == DialogueAct.AMBIGUOUS:
                dialogue_act = DialogueAct.ADD_CONSTRAINT
                confidence = 0.85 if hard_cue else 0.75

        # Review-paraphrase extraction (ASSET_PARAPHRASE): private-set paraphrase robustness
        if self.paraphrase_enabled:
            operations.extend(self._paraphrase_operations(text, operations))

        # Turn-1 tail old-preference capture (matches the 0.995 HR baseline behavior):
        # the intent_override first message is "I'm looking for {cat}. {old_value}",
        # so {old_value} becomes a soft constraint to gain an immediate ranking signal;
        # buying/browsing tails containing
        # markers like "key requirement"/"still exploring" are auto-skipped (avoiding
        # duplication/noise).
        operations.extend(self._turn1_tail_operations(text, request.turn))

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
            boundary_signal=bool(RE_BOUNDARY.search(text)),
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

    def _hard_cue_present(self, text: str) -> bool:
        """Whether a necessity cue word is hit (Part A; always False when the switch is off)."""
        if not self.hard_cue_enabled:
            return False
        lowered = text.lower()
        if any(phrase in lowered for phrase in _HARD_CUE_PHRASES):
            return True
        return any(
            re.search(rf"\b{re.escape(word)}\b", lowered) for word in _HARD_CUE_WORDS
        )

    def _generic_operations(self, text: str, hard: bool = False) -> list[ConstraintOperation]:
        lowered = text.lower()
        strength = ConstraintStrength.HARD if hard else ConstraintStrength.SOFT
        values: list[str] = []
        values.extend(material for material in constants.MATERIALS if material in lowered)
        color_match = re.search(
            r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b",
            lowered,
        )
        if color_match:
            values.append(f"color: {color_match.group(1)}")
        if hard:
            # value capture after a cue word ("I need waterproof" -> waterproof; covers attribute
            # values beyond MATERIALS)
            cue_match = RE_CUE_VALUE.search(text)
            if cue_match:
                value = re.sub(r"^(?:a|an|the)\s+", "", cue_match.group(1).strip(), flags=re.I)
                if value and len(value) >= 2 and value not in values:
                    values.append(value)
        return [
            self._operation(OperationKind.ADD, value, strength)
            for value in values[:2]
        ]

    def _paraphrase_operations(
        self, text: str, existing: list[ConstraintOperation]
    ) -> list[ConstraintOperation]:
        """Extract soft constraints using review_paraphrases patterns (dedup: skip when an existing
            constraint has the same attribute)."""
        lowered = text.lower()
        existing_attrs = {op.attribute for op in existing}
        result: list[ConstraintOperation] = []
        seen: set[str] = set()
        for pattern, attr_type in self._paraphrase_patterns:
            if attr_type in existing_attrs or attr_type in seen:
                continue
            match = pattern.search(lowered)
            if not match:
                continue
            # Negation guard: skip when a negation word appears within the previous 12 characters
            # (e.g. "not made of leather")
            start = max(0, match.start() - 12)
            window = lowered[start : match.start()]
            if re.search(r"(?:not|no|don't|doesn't|dont|never|without|instead of)\b", window):
                continue
            value = match.group(1) if match.groups() else match.group(0)
            value = (value or "").strip()
            if not value or len(value) < 2:
                continue
            result.append(
                self._operation(
                    OperationKind.ADD, value[: self.max_evidence_length], ConstraintStrength.SOFT
                )
            )
            seen.add(attr_type)
            if len(result) >= 2:
                break
        return result

    def _turn1_tail_operations(self, text: str, turn: int) -> list[ConstraintOperation]:
        """Turn 1: if the <tail> of 'looking for X. <tail>' contains no markers, capture it as a
            soft constraint."""
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
