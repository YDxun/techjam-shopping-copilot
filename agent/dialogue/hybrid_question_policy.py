"""Conservative replacement for a repeated Legacy ``other`` question."""

from __future__ import annotations

import math
import threading
import time
from collections import Counter, deque
from dataclasses import replace
from types import MappingProxyType
from typing import Mapping

from agent.dialogue.catalog_signals import ATTRIBUTE_ORDER, CatalogQuestionSignals
from agent.dialogue.models import (
    CandidateAttributeSignal,
    CandidateQuestionSignals,
    DialogueState,
    QuestionDecision,
)
from config.models import HybridQuestionPolicyConfig


class HybridQuestionPolicy:
    """Keep Legacy in control, replacing at most one repeated ``other`` ask."""

    _LATENCY_SAMPLE_LIMIT = 512

    def __init__(self, config: HybridQuestionPolicyConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._reason_counts: Counter[str] = Counter()
        self._selected_attribute_counts: Counter[str] = Counter()
        self._replacement_count = 0
        self._latency_ms: deque[float] = deque(maxlen=self._LATENCY_SAMPLE_LIMIT)

    def consider(
        self,
        state: DialogueState,
        legacy_decision: QuestionDecision,
        catalog_signals: CatalogQuestionSignals,
        candidate_signals: CandidateQuestionSignals | None,
    ) -> QuestionDecision:
        """Return a repeated-``other`` replacement only when every guard passes."""
        if not self.config.enabled:
            return legacy_decision

        started = time.perf_counter()
        if not legacy_decision.should_ask or legacy_decision.ask_attribute != "other":
            decision = legacy_decision
        elif "other" not in state.asked_attributes:
            decision = replace(legacy_decision, reason_code="hybrid_first_other_preserved")
        elif state.hybrid_replacements_used >= self.config.max_replacements_per_session:
            decision = replace(legacy_decision, reason_code="hybrid_replacement_already_used")
        elif candidate_signals is None:
            decision = replace(legacy_decision, reason_code="hybrid_signals_unavailable")
        else:
            decision = self._consider_replacement(
                state, legacy_decision, catalog_signals, candidate_signals
            )
        self._record(decision, (time.perf_counter() - started) * 1000.0)
        return decision

    def _consider_replacement(
        self,
        state: DialogueState,
        legacy_decision: QuestionDecision,
        catalog_signals: CatalogQuestionSignals,
        candidate_signals: CandidateQuestionSignals,
    ) -> QuestionDecision:
        category_signals = catalog_signals.for_category(state.category)
        constrained = {constraint.attribute for constraint in state.active_constraints}
        components: dict[str, Mapping[str, float]] = {}
        scores: dict[str, float] = {}
        for attribute in ATTRIBUTE_ORDER:
            if not self._is_legal_attribute(
                attribute, state, constrained, candidate_signals, category_signals
            ):
                continue
            signal = candidate_signals.by_attribute[attribute]
            answer_probability = getattr(category_signals[attribute], "answer_probability", None)
            values = self._components(signal, answer_probability, state.turn)
            if values is None:
                continue
            if not self._passes_thresholds(values):
                continue
            components[attribute] = MappingProxyType(values)
            scores[attribute] = values["hybrid_gain"]

        if not scores:
            has_legal = any(
                self._is_legal_attribute(
                    attribute, state, constrained, candidate_signals, category_signals
                )
                for attribute in ATTRIBUTE_ORDER
            )
            reason = "hybrid_threshold_not_met" if has_legal else "hybrid_no_eligible_attribute"
            return replace(legacy_decision, reason_code=reason)

        best = min(
            scores,
            key=lambda attribute: (-scores[attribute], ATTRIBUTE_ORDER.index(attribute)),
        )
        ordered_scores = {
            attribute: round(scores[attribute], 6)
            for attribute in ATTRIBUTE_ORDER
            if attribute in scores
        }
        return QuestionDecision(
            should_ask=True,
            ask_attribute=best,
            reason_code="hybrid_specific_replacement",
            utility_score=scores[best],
            alternative_scores=ordered_scores,
            attribute_components=MappingProxyType(dict(components)),
        )

    def _is_legal_attribute(
        self,
        attribute: str,
        state: DialogueState,
        constrained: set[str],
        candidate_signals: CandidateQuestionSignals,
        category_signals: Mapping[str, object],
    ) -> bool:
        return (
            attribute != "other"
            and attribute not in state.asked_attributes
            and attribute not in state.no_preference_attributes
            and attribute not in constrained
            and (attribute != "category" or not state.category)
            and attribute in candidate_signals.by_attribute
            and attribute in category_signals
            and candidate_signals.by_attribute[attribute].attribute == attribute
            and self._finite_signal(candidate_signals.by_attribute[attribute])
            and self._finite_float(
                getattr(category_signals[attribute], "answer_probability", None)
            )
            is not None
        )

    def _components(
        self,
        signal: CandidateAttributeSignal,
        answer_probability: object,
        turn: int,
    ) -> dict[str, float] | None:
        if signal.attribute not in ATTRIBUTE_ORDER or not self._finite_signal(signal):
            return None
        finite_answer_probability = self._finite_float(answer_probability)
        if finite_answer_probability is None:
            return None
        expected_shrink = self._clamp(signal.expected_shrink)
        resolve_at_10 = self._clamp(signal.resolve_at_10)
        coverage = self._clamp(signal.coverage)
        missing_rate = self._clamp(signal.missing_rate)
        extraction_confidence = self._clamp(signal.extraction_confidence)
        turn_pressure = self._clamp((turn - 1) / 9.0)
        answer_probability = self._clamp(finite_answer_probability)
        weights = self.config.weights
        hybrid_gain = (
            weights.expected_shrink * expected_shrink
            + weights.resolve_at_10 * resolve_at_10
            + weights.coverage * coverage
            + weights.answer_probability * answer_probability
            + weights.extraction_confidence * extraction_confidence
            - weights.missing_penalty * missing_rate
            - weights.turn_cost * turn_pressure
        )
        return {
            "expected_shrink": expected_shrink,
            "resolve_at_10": resolve_at_10,
            "coverage": coverage,
            "answer_probability": answer_probability,
            "extraction_confidence": extraction_confidence,
            "missing_rate": missing_rate,
            "turn_pressure": turn_pressure,
            "hybrid_gain": hybrid_gain,
        }

    def _passes_thresholds(self, components: Mapping[str, float]) -> bool:
        return (
            components["coverage"] >= self.config.minimum_coverage
            and components["missing_rate"] <= self.config.maximum_missing_rate
            and components["expected_shrink"] >= self.config.minimum_expected_shrink
            and components["resolve_at_10"] >= self.config.minimum_resolve_at_10
            and components["hybrid_gain"] >= self.config.minimum_gain
        )

    @staticmethod
    def _finite_signal(signal: CandidateAttributeSignal) -> bool:
        return all(
            HybridQuestionPolicy._finite_float(value) is not None
            for value in (
                signal.coverage,
                signal.expected_remaining,
                signal.expected_shrink,
                signal.resolve_at_10,
                signal.resolve_at_3,
                signal.resolve_at_1,
                signal.p90_remaining,
                signal.worst_case_remaining,
                signal.missing_rate,
                signal.extraction_confidence,
                signal.two_step_finish_gain,
            )
        )

    @staticmethod
    def _finite_float(value: object) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) else None

    @staticmethod
    def _clamp(value: object) -> float:
        numeric = HybridQuestionPolicy._finite_float(value)
        assert numeric is not None
        return max(0.0, min(1.0, numeric))

    def _record(self, decision: QuestionDecision, latency_ms: float) -> None:
        with self._lock:
            self._reason_counts[decision.reason_code] += 1
            if decision.should_ask and decision.ask_attribute:
                self._selected_attribute_counts[decision.ask_attribute] += 1
            if decision.reason_code == "hybrid_specific_replacement":
                self._replacement_count += 1
            self._latency_ms.append(max(0.0, latency_ms))

    def statistics(self) -> dict[str, object]:
        """Return aggregate diagnostics without retaining request or catalog data."""
        with self._lock:
            samples = sorted(self._latency_ms)
            return {
                "enabled": self.config.enabled,
                "reason_counts": dict(sorted(self._reason_counts.items())),
                "selected_attribute_counts": dict(sorted(self._selected_attribute_counts.items())),
                "replacement_count": self._replacement_count,
                "decision_latency_ms": {
                    "count": len(samples),
                    "p50": self._percentile(samples, 0.50),
                    "p95": self._percentile(samples, 0.95),
                },
            }

    @staticmethod
    def _percentile(samples: list[float], percentile: float) -> float:
        if not samples:
            return 0.0
        index = min(len(samples) - 1, max(0, math.ceil(percentile * len(samples)) - 1))
        return samples[index]
