from __future__ import annotations

from dataclasses import dataclass, field
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


class ProductFeedback(str, Enum):
    NONE = "none"
    SOFT_DEMOTED = "soft_demoted"
    HARD_REJECTED = "hard_rejected"


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
    # Boundary wording "I don't have a preference for X; please use your judgment.":
    # only means no preference for X, not information exhaustion; the policy uses this to decide
    # whether to stop asking.
    boundary_signal: bool = False


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
class ShownProductState:
    asin: str
    intent_version: int
    shown_turns: tuple[int, ...]
    shown_count: int
    evaluation_eliminated: bool = False
    feedback: ProductFeedback = ProductFeedback.NONE
    feedback_evidence: str = ""


@dataclass(frozen=True)
class ProductContextLists:
    evaluation_excluded_asins: tuple[str, ...]
    hard_rejected_asins: tuple[str, ...]
    soft_demoted_asins: tuple[str, ...]


@dataclass(frozen=True)
class QuestionDecision:
    should_ask: bool
    ask_attribute: str | None
    reason_code: str
    utility_score: float
    alternative_scores: dict[str, float]


@dataclass(frozen=True)
class RecommendationContext:
    intent_version: int
    category: str
    active_constraints: tuple[Constraint, ...]
    buying_or_browsing: str
    retrieval_mode: str
    evaluation_excluded_asins: tuple[str, ...]
    hard_rejected_asins: tuple[str, ...]
    soft_demoted_asins: tuple[str, ...]
    asked_attributes: tuple[str, ...]
    no_more_preferences: bool
    user_profile: dict[str, object] = field(repr=False, compare=False)

    @property
    def hard(self) -> tuple[Constraint, ...]:
        return tuple(item for item in self.active_constraints if item.hardness == 2)

    @property
    def soft(self) -> tuple[Constraint, ...]:
        return tuple(item for item in self.active_constraints if item.hardness == 1)

    @property
    def active(self) -> tuple[Constraint, ...]:
        return self.active_constraints

    @property
    def category_phrase(self) -> str:
        return self.category

    @property
    def category_tokens(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(su.tokenize(self.category)))

    def total_constraints(self) -> int:
        return len(self.active_constraints)

    def to_dict(self) -> dict[str, object]:
        return {
            "intent_version": self.intent_version,
            "category": self.category,
            "active_constraints": [
                {
                    "attribute": item.attribute,
                    "value": item.value,
                    "polarity": item.polarity.value,
                    "strength": item.strength.value,
                    "source_turn": item.source_turn,
                }
                for item in self.active_constraints
            ],
            "buying_or_browsing": self.buying_or_browsing,
            "retrieval_mode": self.retrieval_mode,
            "evaluation_excluded_asins": list(self.evaluation_excluded_asins),
            "hard_rejected_asins": list(self.hard_rejected_asins),
            "soft_demoted_asins": list(self.soft_demoted_asins),
            "asked_attributes": list(self.asked_attributes),
            "no_more_preferences": self.no_more_preferences,
        }


@dataclass(frozen=True)
class DialogueTurnResult:
    state: DialogueState
    recognition: RecognitionResult
    recommendation_context: RecommendationContext
    question_decision: QuestionDecision
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class ReduceResult:
    state: DialogueState
    applied: bool
    reason_code: str
