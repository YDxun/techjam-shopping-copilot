from __future__ import annotations

import json
import math

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
        self.last_usage = LLMUsage()

    @property
    def available(self) -> bool:
        return self.client.status.state == LLMState.AVAILABLE

    def recognize(self, request: RecognitionRequest) -> RecognitionResult | None:
        self.last_usage = LLMUsage()
        if not self.available:
            return None
        result = self.client.chat(
            self._messages(request),
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        self.last_usage = result.usage
        if not result.success:
            return None
        return self._parse(result.content, request.recently_shown_asins)

    @staticmethod
    def _messages(request: RecognitionRequest) -> list[dict[str, str]]:
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
        return [
            {
                "role": "system",
                "content": (
                    "Extract shopping intent as one strict JSON object. Use only the requested "
                    "fields and enum values; do not add markdown or commentary."
                ),
            },
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ]

    def _parse(
        self,
        content: str,
        recently_shown_asins: tuple[str, ...],
    ) -> RecognitionResult | None:
        try:
            payload = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or set(payload) != TOP_LEVEL_FIELDS:
            return None
        try:
            dialogue_act = DialogueAct(payload["dialogue_act"])
            category = payload["category"]
            confidence = self._confidence(payload["confidence"])
            operations = self._operations(payload["constraint_operations"])
            rejected = self._strings(payload["explicit_rejected_asins"])
            ambiguities = self._strings(payload["ambiguities"])
        except (KeyError, TypeError, ValueError):
            return None
        if category is not None and (not isinstance(category, str) or not category.strip()):
            return None
        shown = set(recently_shown_asins)
        if any(asin not in shown for asin in rejected):
            return None
        return RecognitionResult(
            dialogue_act=dialogue_act,
            category=category.strip() if isinstance(category, str) else None,
            constraint_operations=operations,
            explicit_rejected_asins=rejected,
            confidence=confidence,
            source=RecognitionSource.LLM,
            ambiguities=ambiguities,
        )

    def _operations(self, value: object) -> tuple[ConstraintOperation, ...]:
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
            if not isinstance(evidence, str) or len(evidence) > self.max_evidence_length:
                raise TypeError
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
