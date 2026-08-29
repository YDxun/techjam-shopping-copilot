from __future__ import annotations

import json
import math
import re
import threading
from copy import deepcopy

from agent.dialogue.models import (
    ALLOWED_ATTRIBUTES,
    ConstraintOperation,
    ConstraintStrength,
    DialogueAct,
    OperationKind,
    Polarity,
    RecognitionRequest,
    RecognitionResult,
    RecognitionSource,
)
from llm.base import LLMClient, LLMState, LLMUsage

TOP_LEVEL_FIELDS = {
    "dialogue_act",
    "category",
    "constraint_operations",
    "explicit_rejected_asins",
    "confidence",
    "ambiguities",
}
OPERATION_FIELDS = {
    "operation",
    "attribute",
    "value",
    "polarity",
    "strength",
    "evidence",
    "confidence",
}
RE_EXPLICIT_NO_MORE_PREFERENCES = re.compile(
    r"\b(?:no more preferences|no additional preferences)\b",
    re.I,
)

INTENT_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": sorted(TOP_LEVEL_FIELDS),
    "properties": {
        "dialogue_act": {
            "type": "string",
            "enum": [item.value for item in DialogueAct],
        },
        "category": {
            "anyOf": [
                {"type": "string", "minLength": 1},
                {"type": "null"},
            ]
        },
        "constraint_operations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": sorted(OPERATION_FIELDS),
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [item.value for item in OperationKind],
                    },
                    "attribute": {
                        "type": "string",
                        "enum": sorted(ALLOWED_ATTRIBUTES),
                    },
                    "value": {"type": "string", "minLength": 1},
                    "polarity": {
                        "type": "string",
                        "enum": [item.value for item in Polarity],
                    },
                    "strength": {
                        "type": "string",
                        "enum": [item.value for item in ConstraintStrength],
                    },
                    "evidence": {"type": "string", "minLength": 1},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
        "explicit_rejected_asins": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "ambiguities": {"type": "array", "items": {"type": "string"}},
    },
}


class LLMIntentRecognizer:
    """Strict JSON adapter over the already-initialized shared LLM client."""

    def __init__(
        self,
        client: LLMClient,
        *,
        max_evidence_length: int,
        max_tokens: int = 256,
    ) -> None:
        self.client = client
        self.max_evidence_length = max_evidence_length
        self.max_tokens = max_tokens
        self._local = threading.local()

    @property
    def last_usage(self) -> LLMUsage:
        return getattr(self._local, "usage", LLMUsage())

    @last_usage.setter
    def last_usage(self, value: LLMUsage) -> None:
        self._local.usage = value

    @property
    def last_failure_reason(self) -> str | None:
        return getattr(self._local, "failure_reason", None)

    @last_failure_reason.setter
    def last_failure_reason(self, value: str | None) -> None:
        self._local.failure_reason = value

    @property
    def available(self) -> bool:
        return self.client.status.state == LLMState.AVAILABLE

    def recognize(self, request: RecognitionRequest) -> RecognitionResult | None:
        self.last_usage = LLMUsage()
        self.last_failure_reason = None
        if not self.available:
            self.last_failure_reason = "not_available"
            return None
        result = self.client.chat(
            self._messages(request),
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        self.last_usage = result.usage
        if not result.success:
            category = (
                result.error_category.value
                if result.error_category is not None
                else "unknown"
            )
            self.last_failure_reason = f"request_failed:{category}"
            return None
        return self._parse(
            result.content,
            request.recently_shown_asins,
            request.user_message,
        )

    def _messages(self, request: RecognitionRequest) -> list[dict[str, str]]:
        constraints = [
            {
                "attribute": item.attribute,
                "value": item.value,
                "polarity": item.polarity.value,
                "strength": item.strength.value,
            }
            for item in request.state.active_constraints
        ]
        context = {
            "turn": request.turn,
            "category": request.state.category,
            "constraints": constraints,
            "recently_shown_asins": list(request.recently_shown_asins),
            "user_message": request.user_message,
        }
        response_schema = deepcopy(INTENT_RESPONSE_SCHEMA)
        operation_properties = response_schema["properties"]["constraint_operations"][
            "items"
        ]["properties"]
        operation_properties["evidence"]["maxLength"] = self.max_evidence_length
        replace_example = {
            "dialogue_act": "replace_constraint",
            "category": None,
            "constraint_operations": [
                {
                    "operation": "replace",
                    "attribute": "material",
                    "value": "cotton",
                    "polarity": "include",
                    "strength": "hard",
                    "evidence": "cotton",
                    "confidence": 0.95,
                }
            ],
            "explicit_rejected_asins": [],
            "confidence": 0.95,
            "ambiguities": [],
        }
        return [
            {
                "role": "system",
                "content": "\n".join(
                    (
                        (
                            "Extract the user's shopping intent from the supplied conversation "
                            "context."
                        ),
                        "Return exactly one JSON object matching this JSON Schema:",
                        json.dumps(
                            response_schema,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        (
                            "Include every required field, use no additional fields, and output "
                            "no markdown or commentary."
                        ),
                        (
                            "Use category=null when no category is stated. Use empty arrays when "
                            "there are no operations, rejected ASINs, or ambiguities."
                        ),
                        (
                            "For each constraint operation, copy evidence from user_message and "
                            f"keep it at most {self.max_evidence_length} characters long."
                        ),
                        (
                            "Every constraint operation must include all seven required fields. "
                            "Use the shortest exact span from user_message as evidence; never copy "
                            "the whole user_message."
                        ),
                        "Valid replace-constraint example:",
                        json.dumps(replace_example, ensure_ascii=False, separators=(",", ":")),
                        "Only include explicit_rejected_asins that appear in recently_shown_asins.",
                    )
                ),
            },
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ]

    def _parse(
        self,
        content: str,
        recently_shown_asins: tuple[str, ...],
        user_message: str,
    ) -> RecognitionResult | None:
        self.last_failure_reason = None
        try:
            payload = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            self.last_failure_reason = "invalid_json"
            return None
        if not isinstance(payload, dict) or set(payload) != TOP_LEVEL_FIELDS:
            self.last_failure_reason = "invalid_top_level_schema"
            return None
        try:
            dialogue_act = DialogueAct(payload["dialogue_act"])
            category = payload["category"]
            confidence = self._confidence(payload["confidence"])
            operations = self._operations(payload["constraint_operations"], user_message)
            rejected = self._strings(payload["explicit_rejected_asins"])
            ambiguities = self._strings(payload["ambiguities"])
        except (KeyError, TypeError, ValueError):
            if self.last_failure_reason is None:
                self.last_failure_reason = "invalid_field_value"
            return None
        if category is not None and (not isinstance(category, str) or not category.strip()):
            self.last_failure_reason = "invalid_category"
            return None
        shown = set(recently_shown_asins)
        if any(asin not in shown for asin in rejected):
            self.last_failure_reason = "rejected_asin_out_of_scope"
            return None
        self.last_failure_reason = None
        return RecognitionResult(
            dialogue_act=dialogue_act,
            category=category.strip() if isinstance(category, str) else None,
            constraint_operations=operations,
            explicit_rejected_asins=rejected,
            confidence=confidence,
            source=RecognitionSource.LLM,
            ambiguities=ambiguities,
            explicit_no_more_preferences=bool(
                RE_EXPLICIT_NO_MORE_PREFERENCES.search(user_message)
            ),
        )

    def _operations(
        self,
        value: object,
        user_message: str,
    ) -> tuple[ConstraintOperation, ...]:
        if not isinstance(value, list):
            raise TypeError
        operations: list[ConstraintOperation] = []
        for item in value:
            if not isinstance(item, dict) or set(item) != OPERATION_FIELDS:
                raise TypeError
            attribute = item["attribute"]
            raw_value = item["value"]
            evidence = item["evidence"]
            if attribute not in ALLOWED_ATTRIBUTES:
                raise ValueError
            if not isinstance(raw_value, str) or not raw_value.strip():
                raise TypeError
            if not isinstance(evidence, str) or not evidence.strip():
                self.last_failure_reason = "invalid_evidence"
                raise TypeError
            if len(evidence) > self.max_evidence_length:
                self.last_failure_reason = "evidence_too_long"
                raise TypeError
            if evidence not in user_message:
                self.last_failure_reason = "evidence_not_grounded"
                raise ValueError
            operations.append(
                ConstraintOperation(
                    operation=OperationKind(item["operation"]),
                    attribute=attribute,
                    value=raw_value.strip(),
                    polarity=Polarity(item["polarity"]),
                    strength=ConstraintStrength(item["strength"]),
                    evidence=evidence,
                    confidence=self._confidence(item["confidence"]),
                )
            )
        return tuple(operations)

    @staticmethod
    def _confidence(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError
        result = float(value)
        if not math.isfinite(result) or not 0 <= result <= 1:
            raise ValueError
        return result

    @staticmethod
    def _strings(value: object) -> tuple[str, ...]:
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise TypeError
        return tuple(dict.fromkeys(item.strip() for item in value if item.strip()))
