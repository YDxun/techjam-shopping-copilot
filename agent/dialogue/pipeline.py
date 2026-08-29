from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from typing import Iterable

from agent.dialogue.candidate_signals import CONCRETE_ATTRIBUTES, CandidateSignalCalculator
from agent.dialogue.catalog_attributes import CatalogAttributeCache, RuleVocabularyExtractor
from agent.dialogue.catalog_signals import CatalogQuestionSignals
from agent.dialogue.models import (
    CandidateQuestionSignals,
    DialogueAct,
    DialogueState,
    DialogueTurnResult,
    GuardAction,
    GuardDecision,
    QuestionDecision,
    RecognitionRequest,
    RecognitionResult,
    RecommendationContext,
)
from agent.dialogue.product_history import ProductHistory
from agent.dialogue.question_policy import QuestionPolicy
from agent.dialogue.recognizers.cascade import CascadedIntentRecognizer
from agent.dialogue.recognizers.llm import LLMIntentRecognizer
from agent.dialogue.recognizers.rule_based import RuleBasedRecognizer
from agent.dialogue.reducer import StateReducer
from agent.dialogue.transition_guard import TransitionGuard
from config.env_config import EnvConfig
from llm.base import LLMClient

logger = logging.getLogger(__name__)


class StalePendingTurnError(RuntimeError):
    """Raised when an ordinary pending turn no longer matches its base session."""


def _session_fingerprint(session: "SessionState") -> str:
    payload = json.dumps(
        _canonical_session_value(session),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_session_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": [
                [item.name, _canonical_session_value(getattr(value, item.name))]
                for item in fields(value)
            ],
        }
    if isinstance(value, Enum):
        return {
            "enum": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": value.value,
        }
    if isinstance(value, Mapping):
        pairs = [
            [_canonical_session_value(key), _canonical_session_value(item)]
            for key, item in value.items()
        ]
        return {"mapping": sorted(pairs, key=_canonical_sort_key)}
    if isinstance(value, (list, tuple)):
        return {"sequence": [_canonical_session_value(item) for item in value]}
    if isinstance(value, (set, frozenset)):
        return {
            "set": sorted(
                (_canonical_session_value(item) for item in value),
                key=_canonical_sort_key,
            )
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"object": f"{type(value).__module__}.{type(value).__qualname__}", "repr": repr(value)}


def _canonical_sort_key(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True)
class SessionState:
    dialogue: DialogueState
    products: ProductHistory = ProductHistory()
    candidate_counts: tuple[int, ...] = ()


@dataclass(frozen=True)
class PendingDialogueTurn:
    """Immutable interpretation that waits for one retrieved candidate pool."""

    session_id: str
    base_session: SessionState
    base_session_fingerprint: str
    turn: int
    state: DialogueState
    recognition: RecognitionResult
    guard_decision: GuardDecision
    recommendation_context: RecommendationContext
    products: ProductHistory
    prompt_tokens: int
    completion_tokens: int


class DialogueUnderstandingPipeline:
    """Own dialogue interpretation and question decisions, but never Top10 generation."""

    def __init__(
        self,
        *,
        env: EnvConfig,
        llm_client: LLMClient,
        products: Iterable[dict],
        mode: str | None = None,
    ) -> None:
        dialogue_config = env.dialogue_understanding
        self._recognition_mode = mode or dialogue_config.mode
        self.reducer = StateReducer(
            dialogue_config.max_evidence_length,
            override_erase=env.override_erase,
        )
        self.recognizer = CascadedIntentRecognizer(
            rule_recognizer=RuleBasedRecognizer(
                dialogue_config.max_evidence_length,
                transition_guard_enabled=dialogue_config.transition_guard.enabled,
            ),
            llm_recognizer=LLMIntentRecognizer(
                llm_client,
                max_evidence_length=dialogue_config.max_evidence_length,
                max_tokens=env.llm.max_tokens,
            ),
            mode=self._recognition_mode,
            rule_confidence_threshold=dialogue_config.rule_confidence_threshold,
        )
        product_rows = tuple(products)
        self.question_policy = QuestionPolicy(env.decision)
        self.transition_guard = TransitionGuard(dialogue_config.transition_guard)
        self.catalog_signals = CatalogQuestionSignals.from_products(product_rows)
        self.candidate_signal_calculator: CandidateSignalCalculator | None = None
        if env.decision.candidate_question_value.enabled:
            try:
                self.candidate_signal_calculator = CandidateSignalCalculator(
                    CatalogAttributeCache.from_products(product_rows, RuleVocabularyExtractor()),
                    env.decision.candidate_question_value,
                    env.decision.finish_strategy,
                )
            except Exception:
                logger.exception(
                    "[dialogue] dynamic catalog setup failed; using static question policy"
                )
        self._sessions: dict[str, SessionState] = {}
        self._session_locks: dict[str, threading.RLock] = {}
        self._session_locks_guard = threading.Lock()

    def _session_lock(self, session_id: str) -> threading.RLock:
        with self._session_locks_guard:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = threading.RLock()
                self._session_locks[session_id] = lock
            return lock

    def reset(self, session_id: str, user_profile: dict) -> None:
        with self._session_lock(session_id):
            self._sessions[session_id] = SessionState(
                dialogue=self.reducer.new_state(session_id, user_profile)
            )

    def session(self, session_id: str) -> SessionState:
        return self._sessions[session_id]

    def session_token(self, session_id: str) -> tuple[SessionState, str]:
        """Return the current compare-and-set token for explicit shown-result recording."""
        with self._session_lock(session_id):
            session = self._sessions[session_id]
            return session, _session_fingerprint(session)

    def process_turn(
        self,
        session_id: str,
        user_message: str,
        turn: int,
    ) -> DialogueTurnResult:
        pending = self.interpret_turn(session_id, user_message, turn)
        return self.decide_question(pending, None)

    def interpret_turn(
        self,
        session_id: str,
        user_message: str,
        turn: int,
    ) -> PendingDialogueTurn:
        """Interpret a turn once, without recording a follow-up question."""
        with self._session_lock(session_id):
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionState(
                    dialogue=self.reducer.new_state(session_id, {})
                )
            session = self._sessions[session_id]
            base_session_fingerprint = _session_fingerprint(session)
        version_at_start = session.dialogue.intent_version
        request = RecognitionRequest(
            user_message=user_message,
            turn=turn,
            state=session.dialogue,
            recently_shown_asins=session.products.pending_batch,
        )
        recognition = self.recognizer.recognize(request)
        guard_decision = self.transition_guard.evaluate(session.dialogue, recognition)

        if guard_decision.action in {GuardAction.CLARIFY, GuardAction.REJECT}:
            dialogue = session.dialogue
            products = session.products
        else:
            recognition = guard_decision.recognition
            products = session.products.settle_previous_turn(version_at_start)
            products = products.apply_feedback(version_at_start, recognition)
            reduction = self.reducer.reduce(session.dialogue, recognition, turn)

            if reduction.applied:
                dialogue = reduction.state
            else:
                dialogue = session.dialogue

        context = self._build_context(dialogue, recognition, products)
        usage = self.recognizer.last_usage
        return PendingDialogueTurn(
            session_id=session_id,
            base_session=session,
            base_session_fingerprint=base_session_fingerprint,
            turn=turn,
            state=dialogue,
            recognition=recognition,
            guard_decision=guard_decision,
            recommendation_context=context,
            products=products,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )

    def decide_question(
        self,
        pending: PendingDialogueTurn,
        candidate_signals: CandidateQuestionSignals | None,
        *,
        candidate_count: int | None = None,
    ) -> DialogueTurnResult:
        """Choose and commit exactly one ordinary question after candidate retrieval."""
        if pending.guard_decision.action in {GuardAction.CLARIFY, GuardAction.REJECT}:
            decision = QuestionDecision(
                should_ask=True,
                ask_attribute=pending.guard_decision.clarify_attribute or "other",
                reason_code=pending.guard_decision.reason_code,
                utility_score=0.0,
                alternative_scores={},
            )
            return DialogueTurnResult(
                state=pending.state,
                recognition=pending.recognition,
                guard_decision=pending.guard_decision,
                recommendation_context=pending.recommendation_context,
                question_decision=decision,
                prompt_tokens=pending.prompt_tokens,
                completion_tokens=pending.completion_tokens,
            )

        with self._session_lock(pending.session_id):
            session = self._sessions[pending.session_id]
            if (
                session is not pending.base_session
                or _session_fingerprint(session) != pending.base_session_fingerprint
            ):
                raise StalePendingTurnError(
                    "pending dialogue turn no longer matches the current session"
                )

            if candidate_signals is not None:
                candidate_signals = replace(
                    candidate_signals,
                    previous_candidate_count=(
                        session.candidate_counts[-1] if session.candidate_counts else None
                    ),
                )
                candidate_count = candidate_signals.candidate_count
            if pending.state is session.dialogue:
                decision = QuestionDecision(
                    should_ask=True,
                    ask_attribute="other",
                    reason_code="state_update_rejected",
                    utility_score=0.0,
                    alternative_scores={},
                )
            else:
                decision = self.question_policy.decide(
                    pending.state,
                    pending.recognition,
                    self.catalog_signals,
                    candidate_signals,
                )
            dialogue = pending.state
            if decision.should_ask:
                dialogue = self.reducer.record_question(dialogue, decision.ask_attribute)
            context = self._build_context(dialogue, pending.recognition, pending.products)
            counts = session.candidate_counts
            if candidate_count is not None:
                counts = counts + (candidate_count,)
            committed_session = SessionState(
                dialogue=dialogue,
                products=pending.products,
                candidate_counts=counts,
            )
            self._sessions[pending.session_id] = committed_session
            committed_session_fingerprint = _session_fingerprint(committed_session)
        logger.info(
            "[dialogue] session=%s turn=%d intent_version=%d source=%s decision=%s score=%.4f",
            hashlib.sha256(pending.session_id.encode("utf-8")).hexdigest()[:12],
            pending.turn,
            dialogue.intent_version,
            pending.recognition.source.value,
            decision.reason_code,
            decision.utility_score,
        )
        logger.debug(
            "[dialogue.utility] session=%s turn=%d components=%s",
            hashlib.sha256(pending.session_id.encode("utf-8")).hexdigest()[:12],
            pending.turn,
            self.question_policy.last_components,
        )
        return DialogueTurnResult(
            state=dialogue,
            recognition=pending.recognition,
            guard_decision=pending.guard_decision,
            recommendation_context=context,
            question_decision=decision,
            prompt_tokens=pending.prompt_tokens,
            completion_tokens=pending.completion_tokens,
            committed_session=committed_session,
            committed_session_fingerprint=committed_session_fingerprint,
        )

    @staticmethod
    def eligible_candidate_attributes(state: DialogueState) -> tuple[str, ...]:
        return tuple(
            attribute
            for attribute in CONCRETE_ATTRIBUTES
            if attribute not in state.no_preference_attributes
            and (attribute != "category" or not state.category)
        )

    def record_shown(
        self,
        session_id: str,
        asins: tuple[str, ...] | list[str],
        turn: int,
        *,
        expected_session: SessionState | None = None,
        expected_fingerprint: str | None = None,
    ) -> None:
        with self._session_lock(session_id):
            session = self._sessions[session_id]
            if (
                expected_session is None
                or expected_fingerprint is None
                or session is not expected_session
                or _session_fingerprint(session) != expected_fingerprint
            ):
                raise StalePendingTurnError(
                    "shown results no longer match the committed dialogue session"
                )
            products = session.products.record_shown(
                asins,
                intent_version=session.dialogue.intent_version,
                turn=turn,
            )
            self._sessions[session_id] = replace(session, products=products)

    def message_for(self, decision: QuestionDecision, state: DialogueState) -> str:
        return self.question_policy.message_for(decision, state)

    @staticmethod
    def _build_context(
        state: DialogueState,
        recognition,
        products: ProductHistory,
    ) -> RecommendationContext:
        product_lists = products.context_lists(state.intent_version)
        track = "buying" if state.hard else "browsing"
        if recognition.dialogue_act == DialogueAct.REJECT_PRODUCTS:
            retrieval_mode = "recover"
        elif state.no_more_preferences or len(state.hard) >= 2 or state.total_constraints() >= 4:
            retrieval_mode = "exploit"
        else:
            retrieval_mode = "probe"
        return RecommendationContext(
            intent_version=state.intent_version,
            category=state.category,
            active_constraints=state.active_constraints,
            buying_or_browsing=track,
            retrieval_mode=retrieval_mode,
            evaluation_excluded_asins=product_lists.evaluation_excluded_asins,
            hard_rejected_asins=product_lists.hard_rejected_asins,
            soft_demoted_asins=product_lists.soft_demoted_asins,
            asked_attributes=state.asked_attributes,
            no_more_preferences=state.no_more_preferences,
            user_profile=state.user_profile,
        )
