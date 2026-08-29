"""Privacy-safe, local-only diagnostics for dialogue decisions.

This module deliberately stores a small, enumerated record schema.  It never
accepts message, evidence, product, error, LLM-response, or secret fields.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from types import MappingProxyType

from agent.dialogue.models import (
    ALLOWED_ATTRIBUTES,
    ConstraintStrength,
    DialogueAct,
    GuardAction,
    RecognitionSource,
)
from config.models import DecisionTraceConfig
from llm.base import LLMErrorCategory
from utils import session_utils as su

_ATTRIBUTE_SCORE_FIELDS = frozenset(
    {
        "expected_shrink",
        "resolve_at_10",
        "resolve_at_3",
        "resolve_at_1",
        "p90_remaining",
        "exploration_gain",
        "finish_gain",
        "utility",
    }
)
_SCORE_SUMMARY_FIELDS = frozenset(
    {
        "selected_utility",
        "maximum_utility",
        "minimum_utility",
        "mean_utility",
        "candidate_expected_shrink",
        "candidate_coverage",
        "candidate_missing_rate",
    }
)
_ROUND_DIGITS = 6
_UNKNOWN = "unknown"
_REDACTED = "<redacted>"
_ASIN_LIKE_RE = re.compile(r"(?i)(?<![a-z0-9])[a-z0-9]{10}(?![a-z0-9])")
_PRODUCT_IDENTIFIER_ATTRIBUTES = frozenset(
    {"asin", "parent_asin", "product", "product_id", "product_identifier", "sku"}
)
_RECOGNITION_SOURCES = frozenset(item.value for item in RecognitionSource)
_DIALOGUE_ACTS = frozenset(item.value for item in DialogueAct)
_GUARD_ACTIONS = frozenset(item.value for item in GuardAction)
_CONSTRAINT_STRENGTHS = frozenset(item.value for item in ConstraintStrength)
_GUARD_REASONS = frozenset(
    {
        "guard_disabled",
        "guard_passed",
        "add_confidence_below_threshold",
        "generic_rejection_soft_demote",
        "generic_rejection_confidence_below_add_threshold",
        "replace_confidence_below_threshold",
        "remove_confidence_below_threshold",
        "reject_products_confidence_below_threshold",
        "replace_missing_explicit_evidence",
        "remove_target_absent",
        "no_preference_attribute_unclear",
        "no_more_preferences_not_grounded",
        "no_more_preferences_confidence_below_threshold",
    }
)
_DECISION_REASONS = frozenset(
    {
        "stop_utility_reached",
        "ask_other_first",
        "no_candidate_attribute",
        "ask_utility_too_low",
        "highest_ask_utility",
        "no_preference_other",
        "user_has_no_more_preferences",
        "maximum_questions_reached",
        "turn_limit_guardrail",
        "final_turn_no_followup",
        "all_attributes_exhausted",
        "highest_dynamic_utility",
        "dynamic_concrete_fallback",
        "dynamic_other_fallback",
        "state_update_rejected",
    }
) | _GUARD_REASONS
_FALLBACK_REASONS = frozenset(
    {
        "not_available",
        "invalid_json",
        "invalid_top_level_schema",
        "invalid_field_value",
        "invalid_category",
        "rejected_asin_out_of_scope",
        "invalid_evidence",
        "evidence_too_long",
        "evidence_not_grounded",
    }
)
_REQUEST_FAILURE_CATEGORIES = frozenset(item.value for item in LLMErrorCategory)


def _text(value: object) -> str:
    raw = getattr(value, "value", value)
    return str(raw).strip().casefold()


def _category(value: object, allowed: frozenset[str]) -> str:
    normalized = _text(value)
    return normalized if normalized in allowed else _UNKNOWN


def _fallback_reason(value: object) -> str:
    normalized = _text(value)
    if normalized in _FALLBACK_REASONS:
        return normalized
    prefix, separator, category = normalized.partition(":")
    if prefix == "request_failed" and separator:
        category = category.split(":", 1)[0]
        if category in _REQUEST_FAILURE_CATEGORIES:
            return f"request_failed:{category}"
    return _UNKNOWN


def _finite_number(value: object) -> tuple[float, int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0, 1
    number = float(value)
    if not math.isfinite(number):
        return 0.0, 1
    return number, 0


def _safe_nonnegative_int(value: object) -> tuple[int, int]:
    number, sanitizations = _finite_number(value)
    if sanitizations:
        return 0, sanitizations
    return max(0, int(number)), 0


def _normalized_constraints(
    values: Sequence[Sequence[object]],
) -> tuple[tuple[str, str, str], ...]:
    normalized = {_normalized_constraint(item) for item in values if len(item) == 3}
    return tuple(sorted(normalized))


def _normalized_constraint(item: Sequence[object]) -> tuple[str, str, str]:
    return _redact_constraint(_private_canonical_constraint(item))


def _private_canonical_constraint(item: Sequence[object]) -> tuple[str, str, str]:
    attribute = _text(item[0])
    value = su.constraint_key(str(item[1]))
    strength = _category(item[2], _CONSTRAINT_STRENGTHS)
    return attribute, value, strength


def _redact_constraint(
    canonical: tuple[str, str, str],
) -> tuple[str, str, str]:
    attribute, value, strength = canonical
    if attribute in _PRODUCT_IDENTIFIER_ATTRIBUTES or _ASIN_LIKE_RE.search(value):
        return _REDACTED, _REDACTED, strength
    if attribute not in ALLOWED_ATTRIBUTES:
        return _REDACTED, _REDACTED, strength
    return attribute, value, strength


def _frozen_numeric_mapping(
    values: Mapping[str, object], allowed: frozenset[str] | None = None
) -> tuple[Mapping[str, float], int]:
    result: dict[str, float] = {}
    sanitizations = 0
    for raw_key in sorted(values, key=str):
        key = _text(raw_key)
        if allowed is not None and key not in allowed:
            continue
        number, count = _finite_number(values[raw_key])
        result[key] = number
        sanitizations += count
    return MappingProxyType(result), sanitizations


def _frozen_attribute_scores(
    values: Mapping[str, Mapping[str, object]],
) -> tuple[Mapping[str, Mapping[str, float]], int]:
    result: dict[str, Mapping[str, float]] = {}
    sanitizations = 0
    for raw_attribute in sorted(values, key=str):
        attribute = _text(raw_attribute)
        if attribute not in ALLOWED_ATTRIBUTES:
            continue
        score_map = values[raw_attribute]
        if not isinstance(score_map, Mapping):
            continue
        frozen, count = _frozen_numeric_mapping(score_map, _ATTRIBUTE_SCORE_FIELDS)
        result[attribute] = frozen
        sanitizations += count
    return MappingProxyType(result), sanitizations


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, float):
        return round(value, _ROUND_DIGITS)
    return value


@dataclass(frozen=True)
class DialogueDecisionTrace:
    """An immutable allow-list of local decision facts."""

    session_id: InitVar[str]
    turn: int
    recognition_source: str = ""
    dialogue_act: str = ""
    recognition_confidence: float = 0.0
    ambiguities: tuple[str, ...] = ()
    fallback_reason: str = ""
    guard_action: str = ""
    guard_reason: str = ""
    intent_version: int = 0
    added_constraints: tuple[tuple[str, str, str], ...] = ()
    removed_constraints: tuple[tuple[str, str, str], ...] = ()
    candidate_count: int = 0
    score_summary: Mapping[str, object] = field(default_factory=dict)
    missing_rates: Mapping[str, object] = field(default_factory=dict)
    attribute_scores: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    selected_attribute: str | None = None
    decision_reason: str = ""
    finish_pressure: float = 0.0
    lookahead_depth: int = 0
    recommendation_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    session_hash: str = field(init=False)
    sanitizations: int = field(init=False, repr=False, compare=False)

    def __post_init__(self, session_id: str) -> None:
        object.__setattr__(
            self,
            "session_hash",
            hashlib.sha256(str(session_id).encode("utf-8")).hexdigest(),
        )
        confidence, confidence_count = _finite_number(self.recognition_confidence)
        pressure, pressure_count = _finite_number(self.finish_pressure)
        turn, turn_count = _safe_nonnegative_int(self.turn)
        intent_version, intent_version_count = _safe_nonnegative_int(self.intent_version)
        candidate_count, candidate_count_count = _safe_nonnegative_int(self.candidate_count)
        lookahead_depth, lookahead_depth_count = _safe_nonnegative_int(self.lookahead_depth)
        recommendation_count, recommendation_count_count = _safe_nonnegative_int(
            self.recommendation_count
        )
        prompt_tokens, prompt_tokens_count = _safe_nonnegative_int(self.prompt_tokens)
        completion_tokens, completion_tokens_count = _safe_nonnegative_int(self.completion_tokens)
        summary, summary_count = _frozen_numeric_mapping(self.score_summary, _SCORE_SUMMARY_FIELDS)
        missing, missing_count = _frozen_numeric_mapping(self.missing_rates, ALLOWED_ATTRIBUTES)
        attribute_scores, attribute_count = _frozen_attribute_scores(self.attribute_scores)
        object.__setattr__(self, "recognition_confidence", confidence)
        object.__setattr__(self, "finish_pressure", pressure)
        object.__setattr__(self, "turn", turn)
        object.__setattr__(self, "intent_version", intent_version)
        object.__setattr__(self, "candidate_count", candidate_count)
        object.__setattr__(self, "lookahead_depth", lookahead_depth)
        object.__setattr__(self, "recommendation_count", recommendation_count)
        object.__setattr__(self, "prompt_tokens", prompt_tokens)
        object.__setattr__(self, "completion_tokens", completion_tokens)
        ambiguities = tuple(
            sorted(
                normalized
                for value in self.ambiguities
                if (normalized := _text(value)) in ALLOWED_ATTRIBUTES
            )
        )
        object.__setattr__(
            self,
            "ambiguities",
            ambiguities,
        )
        object.__setattr__(
            self, "added_constraints", _normalized_constraints(self.added_constraints)
        )
        object.__setattr__(
            self, "removed_constraints", _normalized_constraints(self.removed_constraints)
        )
        object.__setattr__(self, "score_summary", summary)
        object.__setattr__(self, "missing_rates", missing)
        object.__setattr__(self, "attribute_scores", attribute_scores)
        object.__setattr__(
            self,
            "recognition_source",
            _category(self.recognition_source, _RECOGNITION_SOURCES),
        )
        object.__setattr__(self, "dialogue_act", _category(self.dialogue_act, _DIALOGUE_ACTS))
        object.__setattr__(self, "fallback_reason", _fallback_reason(self.fallback_reason))
        object.__setattr__(self, "guard_action", _category(self.guard_action, _GUARD_ACTIONS))
        object.__setattr__(self, "guard_reason", _category(self.guard_reason, _GUARD_REASONS))
        object.__setattr__(
            self,
            "decision_reason",
            _category(self.decision_reason, _DECISION_REASONS),
        )
        if (
            self.selected_attribute is not None
            and _text(self.selected_attribute) not in ALLOWED_ATTRIBUTES
        ):
            object.__setattr__(self, "selected_attribute", None)
        object.__setattr__(
            self,
            "sanitizations",
            confidence_count
            + pressure_count
            + turn_count
            + intent_version_count
            + candidate_count_count
            + lookahead_depth_count
            + recommendation_count_count
            + prompt_tokens_count
            + completion_tokens_count
            + summary_count
            + missing_count
            + attribute_count,
        )

    @classmethod
    def from_turn(
        cls,
        *,
        before_state: object,
        after_state: object,
        recognition: object,
        guard_decision: object,
        candidate_signals: object | None,
        question_decision: object,
        attribute_components: Mapping[str, Mapping[str, object]],
        recommendation_count: int,
        prompt_tokens: int,
        completion_tokens: int,
        lookahead_depth: int,
        fallback_reason: str = "",
        candidate_count: int | None = None,
        turn: int | None = None,
    ) -> "DialogueDecisionTrace":
        """Build a trace from dialogue models without reading their raw text fields."""
        before = _private_state_constraints(before_state, "active_constraints")
        after = _private_state_constraints(after_state, "active_constraints")
        previously_removed = _private_state_constraints(before_state, "removed_constraints")
        currently_removed = _private_state_constraints(after_state, "removed_constraints")
        added = tuple(_redact_constraint(item) for item in sorted(after - before))
        removed = tuple(
            _redact_constraint(item)
            for item in sorted((before - after) | (currently_removed - previously_removed))
        )
        attribute_scores, missing_rates, signal_candidate_count = _candidate_trace_values(
            candidate_signals, attribute_components
        )
        component_scores = getattr(question_decision, "alternative_scores", {})
        score_summary = _score_summary(
            component_scores,
            getattr(question_decision, "utility_score", 0.0),
            candidate_signals,
        )
        selected_attribute = getattr(question_decision, "ask_attribute", None)
        selected_components = (
            attribute_components.get(selected_attribute, {}) if selected_attribute else {}
        )
        return cls(
            session_id=str(getattr(after_state, "session_id", "")),
            turn=getattr(after_state, "turn", 0) if turn is None else turn,
            recognition_source=getattr(recognition, "source", ""),
            dialogue_act=getattr(recognition, "dialogue_act", ""),
            recognition_confidence=getattr(recognition, "confidence", 0.0),
            ambiguities=tuple(getattr(recognition, "ambiguities", ())),
            fallback_reason=fallback_reason,
            guard_action=getattr(guard_decision, "action", ""),
            guard_reason=getattr(guard_decision, "reason_code", ""),
            intent_version=getattr(after_state, "intent_version", 0),
            added_constraints=added,
            removed_constraints=removed,
            candidate_count=(
                signal_candidate_count if candidate_count is None else candidate_count
            ),
            score_summary=score_summary,
            missing_rates=missing_rates,
            attribute_scores=attribute_scores,
            selected_attribute=selected_attribute,
            decision_reason=getattr(question_decision, "reason_code", ""),
            finish_pressure=selected_components.get("finish_pressure", 0.0),
            lookahead_depth=lookahead_depth,
            recommendation_count=recommendation_count,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def to_dict(
        self,
        *,
        include_attribute_scores: bool = True,
        include_state_diff: bool = True,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "session_hash": self.session_hash,
            "turn": self.turn,
            "recognition_source": self.recognition_source,
            "dialogue_act": self.dialogue_act,
            "recognition_confidence": self.recognition_confidence,
            "ambiguities": list(self.ambiguities),
            "fallback_reason": self.fallback_reason,
            "guard_action": self.guard_action,
            "guard_reason": self.guard_reason,
            "intent_version": self.intent_version,
            "candidate_count": self.candidate_count,
            "score_summary": self.score_summary,
            "missing_rates": self.missing_rates,
            "selected_attribute": (
                _text(self.selected_attribute) if self.selected_attribute is not None else None
            ),
            "decision_reason": self.decision_reason,
            "finish_pressure": self.finish_pressure,
            "lookahead_depth": self.lookahead_depth,
            "recommendation_count": self.recommendation_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }
        if include_state_diff:
            payload["added_constraints"] = self.added_constraints
            payload["removed_constraints"] = self.removed_constraints
        if include_attribute_scores:
            payload["attribute_scores"] = self.attribute_scores
        return _json_value(payload)  # type: ignore[return-value]


def _private_state_constraints(state: object, name: str) -> set[tuple[str, str, str]]:
    return {
        _private_canonical_constraint(
            (
                getattr(constraint, "attribute", ""),
                getattr(constraint, "value", ""),
                getattr(constraint, "strength", ""),
            )
        )
        for constraint in getattr(state, name, ())
    }


def _candidate_trace_values(
    candidate_signals: object | None,
    attribute_components: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, dict[str, object]], dict[str, object], object]:
    if candidate_signals is None:
        return {}, {}, 0
    source_signals = dict(getattr(candidate_signals, "by_attribute", {}))
    other_signal = getattr(candidate_signals, "other_signal", None)
    if other_signal is not None:
        source_signals["other"] = other_signal
    scores: dict[str, dict[str, object]] = {}
    missing_rates: dict[str, object] = {}
    for attribute in sorted(ALLOWED_ATTRIBUTES & set(source_signals)):
        signal = source_signals[attribute]
        scores[attribute] = {
            "expected_shrink": getattr(signal, "expected_shrink", 0.0),
            "resolve_at_10": getattr(signal, "resolve_at_10", 0.0),
            "resolve_at_3": getattr(signal, "resolve_at_3", 0.0),
            "resolve_at_1": getattr(signal, "resolve_at_1", 0.0),
            "p90_remaining": getattr(signal, "p90_remaining", 0.0),
        }
        scores[attribute].update(attribute_components.get(attribute, {}))
        missing_rates[attribute] = getattr(signal, "missing_rate", 0.0)
    return scores, missing_rates, getattr(candidate_signals, "candidate_count", 0)


def _score_summary(
    alternative_scores: object,
    selected_utility: object,
    candidate_signals: object | None,
) -> dict[str, object]:
    values = (
        [value for value in alternative_scores.values() if isinstance(value, (int, float))]
        if isinstance(alternative_scores, Mapping)
        else []
    )
    summary: dict[str, object] = {"selected_utility": selected_utility}
    if values:
        summary.update(
            {
                "maximum_utility": max(values),
                "minimum_utility": min(values),
                "mean_utility": sum(values) / len(values),
            }
        )
    if candidate_signals is not None:
        signals = list(getattr(candidate_signals, "by_attribute", {}).values())
        if signals:
            summary.update(
                {
                    "candidate_expected_shrink": sum(
                        getattr(signal, "expected_shrink", 0.0) for signal in signals
                    )
                    / len(signals),
                    "candidate_coverage": sum(
                        getattr(signal, "coverage", 0.0) for signal in signals
                    )
                    / len(signals),
                    "candidate_missing_rate": sum(
                        getattr(signal, "missing_rate", 0.0) for signal in signals
                    )
                    / len(signals),
                }
            )
    return summary


class DecisionTraceRecorder:
    """Bounded trace retention with aggregate counters that survive the cap."""

    def __init__(self, config: DecisionTraceConfig) -> None:
        self.config = config
        self._records: list[DialogueDecisionTrace] = []
        self._total_seen = 0
        self._decision_reasons: Counter[str] = Counter()
        self._guard_actions: Counter[str] = Counter()
        self._sanitizations = 0
        self._lock = threading.Lock()

    def record(self, trace: DialogueDecisionTrace) -> None:
        if not self.config.enabled:
            return
        with self._lock:
            self._total_seen += 1
            self._decision_reasons[_text(trace.decision_reason)] += 1
            self._guard_actions[_text(trace.guard_action)] += 1
            self._sanitizations += trace.sanitizations
            if len(self._records) < self.config.max_traces:
                self._records.append(trace)

    def records(self) -> tuple[DialogueDecisionTrace, ...]:
        with self._lock:
            return tuple(self._records)

    def summary(self) -> dict[str, object]:
        with self._lock:
            return {
                "enabled": self.config.enabled,
                "recorded": len(self._records),
                "total_seen": self._total_seen,
                "decision_reasons": dict(sorted(self._decision_reasons.items())),
                "guard_actions": dict(sorted(self._guard_actions.items())),
                "sanitizations": self._sanitizations,
            }

    def export_jsonl(self, path: str | Path) -> None:
        if not self.config.enabled:
            return
        output = Path(path)
        with self._lock:
            records = tuple(self._records)
        payloads = [
            (
                trace,
                trace.to_dict(
                    include_attribute_scores=self.config.include_attribute_scores,
                    include_state_diff=self.config.include_state_diff,
                ),
            )
            for trace in records
        ]
        payloads.sort(
            key=lambda item: (
                item[0].session_hash,
                item[0].turn,
                json.dumps(
                    item[1],
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            )
        )
        lines = [
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            for _, payload in payloads
        ]
        directory_fd, leaf = _open_pinned_parent(output)
        descriptor: int | None = None
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(leaf, flags, 0o600, dir_fd=directory_fd)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                descriptor = None
                handle.write("\n".join(lines) + ("\n" if lines else ""))
        except Exception:
            if descriptor is not None:
                os.close(descriptor)
            try:
                os.unlink(leaf, dir_fd=directory_fd)
            except OSError:
                pass
            raise
        finally:
            os.close(directory_fd)


def _open_pinned_parent(output: Path) -> tuple[int, str]:
    """Open every parent component without following symlinks, returning a pinned fd."""
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise OSError("secure decision trace export requires directory no-follow support")
    absolute = output if output.is_absolute() else Path.cwd() / output
    # macOS exposes temporary paths through system aliases; canonicalize only
    # those fixed OS entry points, never caller-controlled path components.
    system_aliases = ((Path("/var"), Path("/private/var")), (Path("/tmp"), Path("/private/tmp")))
    for alias, physical in system_aliases:
        try:
            relative = absolute.relative_to(alias)
        except ValueError:
            continue
        absolute = physical / relative
        break
    if not absolute.name or absolute.name in {".", ".."}:
        raise ValueError("decision trace output requires a file name")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open("/", flags)
    try:
        for component in absolute.parent.parts[1:]:
            if component in {"", "."}:
                continue
            next_fd = os.open(component, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        return directory_fd, absolute.name
    except Exception:
        os.close(directory_fd)
        raise
