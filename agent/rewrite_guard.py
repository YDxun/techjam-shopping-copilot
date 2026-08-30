"""P1 runtime adaptation: rewrite detection -> dynamic upgrade (LLM intent / refined generalized
    rules).

Background: rules are optimal on the public set's templated wording, but rule recognition degrades
when the private set introduces paraphrases (tools/paraphrase_eval
A/B: rules 0.38 vs LLM 0.89). This guard detects "rule-health signals" mid-session and decides
whether to upgrade to LLM
intent (cascaded) on the spot, or stay on rules (built-in loose matching / paraphrase refinement
already backs up).

Health signals (any one triggers):
- >=2 consecutive turns with "turn>=2 yet zero new constraints recognized" (disclosure present but
rules missed it -> suspected paraphrase);
- high AMBIGUOUS ratio (>=0.4 across >=3 turns);
- missing category at turn 1 (first-turn paraphrase made the "looking for" template miss).

Trigger decision:
- LLM available (probed available) -> upgrade to cascaded recognition (LLM intent backs up
paraphrases);
- otherwise -> stay on rules and record the reason (built-in refinement already backs up).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class RewriteGuard:
    def __init__(self, llm_available: bool) -> None:
        self.llm_available = llm_available
        self.no_new_constraint_rounds = 0
        self.ambiguous_count = 0
        self.total_rounds = 0
        self.turn1_no_category = False
        self.upgraded_to_llm = False
        self.reasons: list[str] = []

    def observe(self, recognition, turn: int, had_category: bool) -> None:
        """Feed each turn's recognition result and update the health signals."""
        self.total_rounds += 1
        if turn == 1 and not had_category:
            self.turn1_no_category = True
        act = getattr(recognition, "dialogue_act", None)
        if act is not None and getattr(act, "value", "") == "ambiguous":
            self.ambiguous_count += 1
        ops = getattr(recognition, "constraint_operations", ())
        if turn >= 2 and not ops:
            self.no_new_constraint_rounds += 1
        else:
            self.no_new_constraint_rounds = 0

    def should_upgrade(self) -> str | None:
        """Whether to trigger an upgrade; returns the triggering signal name (None = keep the
            status quo)."""
        if self.no_new_constraint_rounds >= 2:
            return "no_new_constraint"
        if self.total_rounds >= 3 and self.ambiguous_count / self.total_rounds >= 0.4:
            return "ambiguous_ratio"
        if self.turn1_no_category:
            return "turn1_no_category"
        return None

    def decide_upgrade(self) -> bool:
        """Decide and act: LLM available -> upgrade to cascaded; otherwise stay on rules (record
            reason)."""
        signal = self.should_upgrade()
        if signal is None:
            return False
        if self.llm_available and not self.upgraded_to_llm:
            self.upgraded_to_llm = True
            self.reasons.append(f"rewrite_guard[{signal}] -> upgrade to LLM intent (cascaded)")
            logger.info("[rewrite_guard] %s -> upgrade to LLM intent", signal)
            return True
        self.reasons.append(f"rewrite_guard[{signal}] -> LLM unavailable, stay on rules (built-in refinement fallback)")  # noqa: E501
        return False
