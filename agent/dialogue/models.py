from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from utils import session_utils as su


ALLOWED_ATTRIBUTES = frozenset(
    {
        "category",
        "material",
        "color",
        "size",
        "style",
        "brand",
        "budget",
        "feature",
        "use_case",
        "other",
    }
)


class DialogueAct(str, Enum):
    NEW_SEARCH = "new_search"
    ADD_CONSTRAINT = "add_constraint"
    REPLACE_CONSTRAINT = "replace_constraint"
    REMOVE_CONSTRAINT = "remove_constraint"
    REJECT_PRODUCTS = "reject_products"
    NO_PREFERENCE = "no_preference"
    NO_MORE_PREFERENCES = "no_more_preferences"
    AMBIGUOUS = "ambiguous"


class OperationKind(str, Enum):
    ADD = "add"
    REPLACE = "replace"
    REMOVE = "remove"


class Polarity(str, Enum):
    INCLUDE = "include"
    EXCLUDE = "exclude"


class ConstraintStrength(str, Enum):
    HARD = "hard"
    SOFT = "soft"


class RecognitionSource(str, Enum):
    RULE = "rule"
    LLM = "llm"


@dataclass(frozen=True)
class ConstraintOperation:
    operation: OperationKind
    attribute: str
    value: str
    polarity: Polarity
    strength: ConstraintStrength
    evidence: str
    confidence: float


@dataclass(frozen=True)
class Constraint:
    attribute: str
    value: str
    polarity: Polarity
    strength: ConstraintStrength
    evidence: str
    source_turn: int
    tokens: tuple[str, ...]

    @property
    def key(self) -> tuple[str, str, Polarity]:
        return self.attribute, su.constraint_key(self.value), self.polarity

    @property
    def hardness(self) -> int:
        return 2 if self.strength == ConstraintStrength.HARD else 1


@dataclass(frozen=True)
class RecognitionResult:
    dialogue_act: DialogueAct
    category: str | None
    constraint_operations: tuple[ConstraintOperation, ...]
    explicit_rejected_asins: tuple[str, ...]
    confidence: float
    source: RecognitionSource
    ambiguities: tuple[str, ...]


@dataclass(frozen=True)
class DialogueState:
    session_id: str
    user_profile: dict[str, object]
    intent_version: int = 1
    category: str = ""
    active_constraints: tuple[Constraint, ...] = ()
    removed_constraints: tuple[Constraint, ...] = ()
    asked_attributes: tuple[str, ...] = ()
    no_preference_attributes: frozenset[str] = frozenset()
    no_more_preferences: bool = False
    last_dialogue_act: DialogueAct = DialogueAct.AMBIGUOUS
    turn: int = 0

    @property
    def hard(self) -> tuple[Constraint, ...]:
        return tuple(item for item in self.active_constraints if item.hardness == 2)

    @property
    def soft(self) -> tuple[Constraint, ...]:
        return tuple(item for item in self.active_constraints if item.hardness == 1)

    @property
    def category_phrase(self) -> str:
        return self.category

    @property
    def category_tokens(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(su.tokenize(self.category)))

    def total_constraints(self) -> int:
        return len(self.active_constraints)


@dataclass(frozen=True)
class RecognitionRequest:
    user_message: str
    turn: int
    state: DialogueState
    recently_shown_asins: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReduceResult:
    state: DialogueState
    applied: bool
    reason_code: str
