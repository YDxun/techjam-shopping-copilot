from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping

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


class GuardAction(str, Enum):
    APPLY = "apply"
    SOFTEN = "soften"
    CLARIFY = "clarify"
    REJECT = "reject"


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
    # 边界措辞 "I don't have a preference for X; please use your judgment."：
    # 仅表示对 X 无偏好，不表示信息枯竭，policy 据此区分是否停止提问。
    boundary_signal: bool = False
    # 只由输入消息中已验证的明确结束措辞派生；不接受模型自报的结束标记。
    explicit_no_more_preferences: bool = False


@dataclass(frozen=True)
class GuardDecision:
    action: GuardAction
    recognition: RecognitionResult
    reason_code: str
    clarify_attribute: str | None = None


@dataclass(frozen=True)
class DialogueState:
    session_id: str
    user_profile: dict[str, object]
    intent_version: int = 1
    category: str = ""
    active_constraints: tuple[Constraint, ...] = ()
    removed_constraints: tuple[Constraint, ...] = ()
    asked_attributes: tuple[str, ...] = ()
    hybrid_replacements_used: int = 0
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
    attribute_components: Mapping[str, Mapping[str, float]] = field(
        default_factory=dict, repr=False, compare=False
    )


@dataclass(frozen=True)
class CandidateAttributeSignal:
    attribute: str
    coverage: float
    expected_remaining: float
    expected_shrink: float
    resolve_at_10: float
    resolve_at_3: float
    resolve_at_1: float
    p90_remaining: float
    worst_case_remaining: int
    missing_rate: float
    extraction_confidence: float
    two_step_finish_gain: float = 0.0


@dataclass(frozen=True)
class CandidateQuestionSignals:
    candidate_count: int
    by_attribute: Mapping[str, CandidateAttributeSignal]
    target_probabilities: Mapping[str, float]
    best_other_pair: tuple[str, str] | None = None
    other_signal: CandidateAttributeSignal | None = None
    previous_candidate_count: int | None = None
    source: str = "dynamic"
    lookahead_depth_used: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_attribute", MappingProxyType(dict(self.by_attribute)))
        object.__setattr__(
            self, "target_probabilities", MappingProxyType(dict(self.target_probabilities))
        )


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
    guard_decision: GuardDecision
    recommendation_context: RecommendationContext
    question_decision: QuestionDecision
    prompt_tokens: int
    completion_tokens: int
    committed_session: object | None = field(default=None, repr=False, compare=False)
    committed_session_fingerprint: str | None = field(default=None, repr=False, compare=False)
    trace_inputs: object | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class ReduceResult:
    state: DialogueState
    applied: bool
    reason_code: str
