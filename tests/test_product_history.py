from __future__ import annotations

import unittest

from agent.dialogue.models import (
    DialogueAct,
    ProductFeedback,
    RecognitionResult,
    RecognitionSource,
)
from agent.dialogue.product_history import ProductHistory


def feedback(*asins: str) -> RecognitionResult:
    return RecognitionResult(
        dialogue_act=DialogueAct.REJECT_PRODUCTS,
        category=None,
        constraint_operations=(),
        explicit_rejected_asins=tuple(asins),
        confidence=0.9,
        source=RecognitionSource.RULE,
        ambiguities=(),
    )


class ProductHistoryTest(unittest.TestCase):
    def test_next_turn_marks_previous_batch_eliminated_and_generic_feedback_soft(self) -> None:
        history = ProductHistory().record_shown(("A", "B"), intent_version=1, turn=1)

        settled = history.settle_previous_turn(intent_version=1)
        updated = settled.apply_feedback(intent_version=1, recognition=feedback())

        context = updated.context_lists(intent_version=1)
        self.assertEqual(context.evaluation_excluded_asins, ("A", "B"))
        self.assertEqual(context.soft_demoted_asins, ("A", "B"))
        self.assertEqual(context.hard_rejected_asins, ())
        self.assertEqual(history.context_lists(1).evaluation_excluded_asins, ())

    def test_explicit_feedback_hard_rejects_only_the_named_shown_product(self) -> None:
        history = ProductHistory().record_shown(("A", "B"), intent_version=1, turn=1)

        updated = history.settle_previous_turn(1).apply_feedback(1, feedback("B"))

        context = updated.context_lists(1)
        self.assertEqual(context.hard_rejected_asins, ("B",))
        self.assertEqual(context.soft_demoted_asins, ())
        product_b = next(item for item in updated.observations if item.asin == "B")
        self.assertEqual(product_b.feedback, ProductFeedback.HARD_REJECTED)

    def test_old_version_feedback_does_not_leak_into_new_intent(self) -> None:
        history = ProductHistory().record_shown(("A", "B"), intent_version=1, turn=1)
        updated = history.settle_previous_turn(1).apply_feedback(1, feedback())

        self.assertEqual(updated.context_lists(2).evaluation_excluded_asins, ())
        self.assertEqual(updated.context_lists(2).soft_demoted_asins, ())
        self.assertEqual(updated.context_lists(2).hard_rejected_asins, ())

    def test_repeated_display_updates_count_without_duplicate_context_entries(self) -> None:
        history = ProductHistory().record_shown(("A",), intent_version=1, turn=1)
        history = history.settle_previous_turn(1).apply_feedback(1, feedback())
        history = history.record_shown(("A",), intent_version=1, turn=2)

        product = history.observations[0]
        self.assertEqual(product.shown_turns, (1, 2))
        self.assertEqual(product.shown_count, 2)
        self.assertEqual(history.context_lists(1).soft_demoted_asins, ("A",))


if __name__ == "__main__":
    unittest.main()
