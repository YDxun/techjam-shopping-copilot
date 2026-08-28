from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, replace
from typing import Iterable

from agent.dialogue.catalog_signals import CatalogQuestionSignals
from agent.dialogue.models import (
    DialogueAct,
    DialogueState,
    DialogueTurnResult,
    QuestionDecision,
    RecognitionRequest,
    RecommendationContext,
)
from agent.dialogue.product_history import ProductHistory
from agent.dialogue.question_policy import QuestionPolicy
from agent.dialogue.recognizers.cascade import CascadedIntentRecognizer
from agent.dialogue.recognizers.llm import LLMIntentRecognizer
from agent.dialogue.recognizers.rule_based import RuleBasedRecognizer
from agent.dialogue.reducer import StateReducer
from config.env_config import EnvConfig
from llm.base import LLMClient


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionState:
    dialogue: DialogueState
    products: ProductHistory = ProductHistory()


class DialogueUnderstandingPipeline:
    """Own dialogue interpretation and question decisions, but never Top10 generation."""

    def __init__(
        self,
        *,
        env: EnvConfig,
        llm_client: LLMClient,
        products: Iterable[dict],
    ) -> None:
        dialogue_config = env.dialogue_understanding
        self.reducer = StateReducer(dialogue_config.max_evidence_length)
        self.recognizer = CascadedIntentRecognizer(
            rule_recognizer=RuleBasedRecognizer(dialogue_config.max_evidence_length),
            llm_recognizer=LLMIntentRecognizer(
                llm_client,
                max_evidence_length=dialogue_config.max_evidence_length,
                max_tokens=env.llm.max_tokens,
            ),
            mode=dialogue_config.mode,
            rule_confidence_threshold=dialogue_config.rule_confidence_threshold,
        )
        self.question_policy = QuestionPolicy(env.decision)
        self.catalog_signals = CatalogQuestionSignals.from_products(products)
        self._sessions: dict[str, SessionState] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = SessionState(
            dialogue=self.reducer.new_state(session_id, user_profile)
        )

    def session(self, session_id: str) -> SessionState:
        return self._sessions[session_id]

    def process_turn(
        self,
        session_id: str,
        user_message: str,
        turn: int,
    ) -> DialogueTurnResult:
        if session_id not in self._sessions:
            self.reset(session_id, {})
        session = self._sessions[session_id]
        version_at_start = session.dialogue.intent_version
        products = session.products.settle_previous_turn(version_at_start)
        request = RecognitionRequest(
            user_message=user_message,
            turn=turn,
            state=session.dialogue,
            recently_shown_asins=products.pending_batch,
        )
        recognition = self.recognizer.recognize(request)
        products = products.apply_feedback(version_at_start, recognition)
        reduction = self.reducer.reduce(session.dialogue, recognition, turn)

        if reduction.applied:
            dialogue = reduction.state
            decision = self.question_policy.decide(
                dialogue,
                recognition,
                self.catalog_signals,
            )
        else:
            dialogue = session.dialogue
            decision = QuestionDecision(
                should_ask=True,
                ask_attribute="other",
                reason_code="state_update_rejected",
                utility_score=0.0,
                alternative_scores={},
            )
        if decision.should_ask:
            dialogue = self.reducer.record_question(dialogue, decision.ask_attribute)

        context = self._build_context(dialogue, recognition, products)
        self._sessions[session_id] = SessionState(dialogue=dialogue, products=products)
        logger.info(
            "[dialogue] session=%s turn=%d intent_version=%d source=%s decision=%s "
            "score=%.4f components=%s",
            hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12],
            turn,
            dialogue.intent_version,
            recognition.source.value,
            decision.reason_code,
            decision.utility_score,
            self.question_policy.last_components,
        )
        usage = self.recognizer.last_usage
        return DialogueTurnResult(
            state=dialogue,
            recognition=recognition,
            recommendation_context=context,
            question_decision=decision,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )

    def record_shown(
        self,
        session_id: str,
        asins: tuple[str, ...] | list[str],
        turn: int,
    ) -> None:
        session = self._sessions[session_id]
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
