from __future__ import annotations

from dataclasses import dataclass, replace

from agent.dialogue.models import (
    DialogueAct,
    ProductContextLists,
    ProductFeedback,
    RecognitionResult,
    ShownProductState,
)


@dataclass(frozen=True)
class ProductHistory:
    """Immutable display and feedback history, isolated by intent version."""

    observations: tuple[ShownProductState, ...] = ()
    pending_batch: tuple[str, ...] = ()
    pending_intent_version: int | None = None

    def record_shown(
        self,
        asins: tuple[str, ...] | list[str],
        intent_version: int,
        turn: int,
    ) -> ProductHistory:
        ordered = tuple(dict.fromkeys(asin for asin in asins if isinstance(asin, str) and asin))
        observations = list(self.observations)
        index_by_key = {
            (item.asin, item.intent_version): index
            for index, item in enumerate(observations)
        }
        for asin in ordered:
            key = (asin, intent_version)
            index = index_by_key.get(key)
            if index is None:
                index_by_key[key] = len(observations)
                observations.append(
                    ShownProductState(
                        asin=asin,
                        intent_version=intent_version,
                        shown_turns=(turn,),
                        shown_count=1,
                    )
                )
                continue
            current = observations[index]
            observations[index] = replace(
                current,
                shown_turns=(*current.shown_turns, turn),
                shown_count=current.shown_count + 1,
            )
        return ProductHistory(
            observations=tuple(observations),
            pending_batch=ordered,
            pending_intent_version=intent_version,
        )

    def settle_previous_turn(self, intent_version: int) -> ProductHistory:
        if self.pending_intent_version != intent_version or not self.pending_batch:
            return self
        pending = set(self.pending_batch)
        observations = tuple(
            replace(item, evaluation_eliminated=True)
            if item.intent_version == intent_version and item.asin in pending
            else item
            for item in self.observations
        )
        return replace(self, observations=observations)

    def apply_feedback(
        self,
        intent_version: int,
        recognition: RecognitionResult,
    ) -> ProductHistory:
        observations = self.observations
        if (
            self.pending_intent_version == intent_version
            and recognition.dialogue_act == DialogueAct.REJECT_PRODUCTS
        ):
            pending = set(self.pending_batch)
            explicit = set(recognition.explicit_rejected_asins) & pending
            feedback = (
                ProductFeedback.HARD_REJECTED
                if explicit
                else ProductFeedback.SOFT_DEMOTED
            )
            targets = explicit or pending
            evidence = (
                "explicit_product_rejection"
                if explicit
                else "generic_negative_feedback"
            )
            observations = tuple(
                self._with_feedback(item, feedback, evidence)
                if item.intent_version == intent_version and item.asin in targets
                else item
                for item in observations
            )
        return ProductHistory(observations=observations)

    def context_lists(self, intent_version: int) -> ProductContextLists:
        current = tuple(
            item for item in self.observations if item.intent_version == intent_version
        )
        return ProductContextLists(
            evaluation_excluded_asins=tuple(
                item.asin for item in current if item.evaluation_eliminated
            ),
            hard_rejected_asins=tuple(
                item.asin for item in current
                if item.feedback == ProductFeedback.HARD_REJECTED
            ),
            soft_demoted_asins=tuple(
                item.asin for item in current
                if item.feedback == ProductFeedback.SOFT_DEMOTED
            ),
        )

    @staticmethod
    def _with_feedback(
        item: ShownProductState,
        feedback: ProductFeedback,
        evidence: str,
    ) -> ShownProductState:
        if item.feedback == ProductFeedback.HARD_REJECTED:
            return item
        return replace(item, feedback=feedback, feedback_evidence=evidence)
