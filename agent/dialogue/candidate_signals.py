"""Deterministic value-of-information signals for a retrieved candidate pool."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations

from agent.dialogue.catalog_attributes import (
    ATTRIBUTE_NAMES,
    AttributeProfile,
    CatalogAttributeCache,
)
from agent.dialogue.models import CandidateAttributeSignal, CandidateQuestionSignals
from config.models import CandidateQuestionValueConfig, FinishStrategyConfig

# ``other`` is a composite action rather than a profile attribute.  Category is
# a normal first-question attribute until the dialogue state has learned it.
CONCRETE_ATTRIBUTES = ATTRIBUTE_NAMES


@dataclass(frozen=True)
class _Candidate:
    asin: str
    profile: AttributeProfile | None
    rrf: float | None


class CandidateSignalCalculator:
    """Compute conservative question signals from immutable catalog profiles.

    A profile unavailable from the cache is deliberately treated as missing for
    every attribute.  It therefore stays in every answer-compatible set instead
    of being discarded or treated as a negative match.
    """

    def __init__(
        self,
        cache: CatalogAttributeCache,
        config: CandidateQuestionValueConfig | None = None,
        finish_strategy: FinishStrategyConfig | None = None,
        *,
        candidate_config: CandidateQuestionValueConfig | None = None,
    ) -> None:
        if config is not None and candidate_config is not None:
            raise ValueError("provide either config or candidate_config, not both")
        self._cache = cache
        self._config = config or candidate_config or CandidateQuestionValueConfig()
        self._finish_strategy = finish_strategy or FinishStrategyConfig()
        self._validate_configuration()

    def calculate(
        self,
        candidates: Iterable[object],
        eligible_attributes: Iterable[str] | None = None,
        *,
        remaining_question_budget: int | None = None,
        terminal_eligible: bool = True,
    ) -> CandidateQuestionSignals:
        """Return signals for unique candidates and unresolved legal attributes.

        Candidate identity is the ``parent_asin``.  Duplicate IDs are collapsed
        by retaining their greatest finite RRF score, independent of input order.
        """
        rows = self._normalize_candidates(candidates)
        if not rows:
            return CandidateQuestionSignals(
                candidate_count=0,
                by_attribute={},
                target_probabilities={},
            )

        attributes = self._eligible_attributes(eligible_attributes)
        probabilities = self._target_probabilities(rows)
        by_attribute = {
            attribute: self._signal_for_attributes(rows, probabilities, (attribute,))
            for attribute in attributes
        }

        best_other_pair, other_signal = self._best_other_pair(rows, probabilities, attributes)
        lookahead_depth_used = 1
        if self._lookahead_enabled(
            rows,
            by_attribute,
            other_signal,
            remaining_question_budget=remaining_question_budget,
            terminal_eligible=terminal_eligible,
        ):
            lookahead_depth_used = 2
            branch_signal_cache: dict[
                tuple[tuple[str, ...], str], CandidateAttributeSignal
            ] = {}
            by_attribute = {
                attribute: replace(
                    signal,
                    two_step_finish_gain=self._two_step_finish_gain(
                        rows,
                        probabilities,
                        attribute,
                        attributes,
                        branch_signal_cache,
                    ),
                )
                for attribute, signal in by_attribute.items()
            }

        return CandidateQuestionSignals(
            candidate_count=len(rows),
            by_attribute=by_attribute,
            target_probabilities={
                asin: self._bounded_ratio(probability)
                for asin, probability in probabilities.items()
            },
            best_other_pair=best_other_pair,
            other_signal=other_signal,
            lookahead_depth_used=lookahead_depth_used,
        )

    def _validate_configuration(self) -> None:
        alpha = self._config.prior_alpha
        temperature = self._config.prior_temperature
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("prior_alpha must be finite and within [0, 1]")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("prior_temperature must be finite and positive")
        if self._finish_strategy.lookahead_depth not in (1, 2):
            raise ValueError("lookahead_depth must be 1 or 2")

    def _normalize_candidates(self, candidates: Iterable[object]) -> tuple[_Candidate, ...]:
        scores: dict[str, float | None] = {}
        for candidate in candidates:
            asin, score = self._candidate_identity_and_score(candidate)
            if not asin:
                continue
            existing = scores.get(asin)
            if score is not None and (existing is None or score > existing):
                scores[asin] = score
            elif asin not in scores:
                scores[asin] = score
        return tuple(
            _Candidate(asin=asin, profile=self._cache.for_asin(asin), rrf=scores[asin])
            for asin in sorted(scores)
        )

    @staticmethod
    def _candidate_identity_and_score(candidate: object) -> tuple[str, float | None]:
        if isinstance(candidate, Mapping):
            asin_value = candidate.get("parent_asin")
            score_value = candidate.get("rrf")
        elif isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
            asin_value = candidate[0] if candidate else None
            score_value = candidate[1] if len(candidate) > 1 else None
        else:
            asin_value = getattr(candidate, "parent_asin", None)
            score_value = getattr(candidate, "rrf", None)
        asin = str(asin_value).strip() if asin_value is not None else ""
        try:
            score = float(score_value)
        except (TypeError, ValueError):
            return asin, None
        return asin, score if math.isfinite(score) else None

    def _eligible_attributes(self, eligible_attributes: Iterable[str] | None) -> tuple[str, ...]:
        if eligible_attributes is None:
            return CONCRETE_ATTRIBUTES
        requested = frozenset(str(attribute) for attribute in eligible_attributes)
        return tuple(attribute for attribute in CONCRETE_ATTRIBUTES if attribute in requested)

    def _target_probabilities(self, rows: Sequence[_Candidate]) -> dict[str, float]:
        count = len(rows)
        uniform = 1.0 / count
        scores = [row.rrf for row in rows]
        usable_scores = all(score is not None for score in scores)
        finite_scores = [float(score) for score in scores if score is not None]
        if not usable_scores or min(finite_scores) == max(finite_scores):
            return {row.asin: uniform for row in rows}

        scaled = [score / self._config.prior_temperature for score in finite_scores]
        maximum = max(scaled)
        exponents = [math.exp(score - maximum) for score in scaled]
        total = sum(exponents)
        if not math.isfinite(total) or total <= 0.0:
            return {row.asin: uniform for row in rows}
        softmax = [value / total for value in exponents]
        alpha = self._config.prior_alpha
        return {
            row.asin: (1.0 - alpha) * uniform + alpha * softmax[index]
            for index, row in enumerate(rows)
        }

    def _signal_for_attributes(
        self,
        rows: Sequence[_Candidate],
        probabilities: Mapping[str, float],
        attributes: tuple[str, ...],
    ) -> CandidateAttributeSignal:
        count = len(rows)
        remaining_by_asin = {
            row.asin: len(self._compatible_rows(rows, row, attributes)) for row in rows
        }
        expected_remaining = sum(
            probabilities[row.asin] * remaining_by_asin[row.asin] for row in rows
        )
        coverage = sum(
            probabilities[row.asin]
            for row in rows
            if all(self._values(row, attribute) for attribute in attributes)
        )
        extraction_confidence = sum(
            probabilities[row.asin]
            * min(self._confidence(row, attribute) for attribute in attributes)
            for row in rows
        )
        remaining_values = sorted(
            (remaining_by_asin[row.asin], probabilities[row.asin]) for row in rows
        )
        p90_remaining = self._weighted_quantile(remaining_values, 0.90)
        return CandidateAttributeSignal(
            attribute=attributes[0],
            coverage=self._bounded_ratio(coverage),
            expected_remaining=expected_remaining,
            expected_shrink=self._bounded_ratio(1.0 - expected_remaining / count),
            resolve_at_10=self._bounded_ratio(
                sum(
                    probabilities[row.asin]
                    for row in rows
                    if remaining_by_asin[row.asin] <= 10
                )
            ),
            resolve_at_3=self._bounded_ratio(
                sum(
                    probabilities[row.asin]
                    for row in rows
                    if remaining_by_asin[row.asin] <= 3
                )
            ),
            resolve_at_1=self._bounded_ratio(
                sum(
                    probabilities[row.asin]
                    for row in rows
                    if remaining_by_asin[row.asin] <= 1
                )
            ),
            p90_remaining=p90_remaining,
            worst_case_remaining=max(remaining_by_asin.values()),
            missing_rate=self._bounded_ratio(1.0 - coverage),
            extraction_confidence=self._bounded_ratio(extraction_confidence),
        )

    def _best_other_pair(
        self,
        rows: Sequence[_Candidate],
        probabilities: Mapping[str, float],
        attributes: tuple[str, ...],
    ) -> tuple[tuple[str, str] | None, CandidateAttributeSignal | None]:
        best_pair: tuple[str, str] | None = None
        best_signal: CandidateAttributeSignal | None = None
        other_attributes = tuple(attribute for attribute in attributes if attribute != "category")
        for pair in combinations(other_attributes, 2):
            signal = self._signal_for_attributes(rows, probabilities, pair)
            if best_signal is None or signal.expected_shrink > best_signal.expected_shrink:
                best_pair, best_signal = pair, signal
        if best_signal is None:
            return None, None
        return best_pair, replace(best_signal, attribute="other")

    def _two_step_finish_gain(
        self,
        rows: Sequence[_Candidate],
        probabilities: Mapping[str, float],
        first_attribute: str,
        attributes: tuple[str, ...],
        branch_signal_cache: dict[tuple[tuple[str, ...], str], CandidateAttributeSignal],
    ) -> float:
        second_attributes = tuple(
            attribute for attribute in attributes if attribute != first_attribute
        )
        if not second_attributes:
            return 0.0

        expected_finish_gain = 0.0
        for target in rows:
            branch = self._compatible_rows(rows, target, (first_attribute,))
            branch_mass = sum(probabilities[row.asin] for row in branch)
            if branch_mass <= 0.0:
                continue
            conditional_probabilities = {
                row.asin: probabilities[row.asin] / branch_mass for row in branch
            }
            branch_asins = tuple(row.asin for row in branch)
            gains = []
            for attribute in second_attributes:
                key = (branch_asins, attribute)
                signal = branch_signal_cache.get(key)
                if signal is None:
                    signal = self._signal_for_attributes(
                        branch, conditional_probabilities, (attribute,)
                    )
                    branch_signal_cache[key] = signal
                gains.append(self._finish_gain(signal, len(branch)))
            best_second_gain = max(gains)
            expected_finish_gain += probabilities[target.asin] * best_second_gain
        return max(0.0, expected_finish_gain - self._config.weights.turn_cost)

    def _lookahead_enabled(
        self,
        rows: Sequence[_Candidate],
        by_attribute: Mapping[str, CandidateAttributeSignal],
        other_signal: CandidateAttributeSignal | None,
        *,
        remaining_question_budget: int | None,
        terminal_eligible: bool,
    ) -> bool:
        strategy = self._finish_strategy
        phase_ready = len(rows) <= strategy.candidate_threshold or (
            remaining_question_budget is not None
            and remaining_question_budget <= strategy.remaining_question_threshold
        )
        if not (
            strategy.enabled
            and strategy.lookahead_depth == 2
            and terminal_eligible
            and phase_ready
        ):
            return False
        one_step_signals = tuple(by_attribute.values()) + (
            (other_signal,) if other_signal is not None else ()
        )
        return bool(one_step_signals) and max(
            self._finish_gain(signal, len(rows)) for signal in one_step_signals
        ) >= strategy.minimum_finish_gain

    def _finish_gain(self, signal: CandidateAttributeSignal, candidate_count: int) -> float:
        weights = self._finish_strategy.weights
        terminal_progress = self._terminal_progress(signal.expected_remaining, candidate_count)
        p90_fraction = signal.p90_remaining / candidate_count
        return (
            weights.resolve_at_10 * signal.resolve_at_10
            + weights.resolve_at_3 * signal.resolve_at_3
            + weights.resolve_at_1 * signal.resolve_at_1
            + weights.terminal_progress * terminal_progress
            - weights.p90_remaining_penalty * p90_fraction
        )

    @staticmethod
    def _terminal_progress(expected_remaining: float, candidate_count: int) -> float:
        if candidate_count <= 10:
            return 0.0
        initial_distance = math.log1p(candidate_count - 10)
        remaining_distance = math.log1p(max(expected_remaining - 10.0, 0.0))
        return 1.0 - remaining_distance / initial_distance

    def _compatible_rows(
        self,
        rows: Sequence[_Candidate],
        target: _Candidate,
        attributes: tuple[str, ...],
    ) -> tuple[_Candidate, ...]:
        target_values = {attribute: self._values(target, attribute) for attribute in attributes}
        return tuple(
            row
            for row in rows
            if all(
                not target_values[attribute]
                or not self._values(row, attribute)
                or self._values(row, attribute) & target_values[attribute]
                for attribute in attributes
            )
        )

    @staticmethod
    def _values(row: _Candidate, attribute: str) -> frozenset[str]:
        if row.profile is None:
            return frozenset()
        values = row.profile.values.get(attribute, frozenset())
        return frozenset(values)

    @staticmethod
    def _confidence(row: _Candidate, attribute: str) -> float:
        if row.profile is None:
            return 0.0
        value = row.profile.confidence.get(attribute, 0.0)
        try:
            return min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _weighted_quantile(values: Sequence[tuple[int, float]], quantile: float) -> float:
        cumulative = 0.0
        for remaining, probability in values:
            cumulative += probability
            if cumulative >= quantile:
                return float(remaining)
        return float(values[-1][0])

    @staticmethod
    def _bounded_ratio(value: float) -> float:
        if not math.isfinite(value):
            return 0.0
        return min(1.0, max(0.0, value))
