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
from llm.base import LLMClient, LLMRequestOptions, LLMState, LLMUsage
from utils.data_assets import NormalizationVocabulary

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
    r"(?:\b(?:no more preferences|no additional preferences)\b|没有其他要求了|沒有其他要求了)",
    re.I,
)
RE_CJK_IDEOGRAPH = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f]"
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
        request_options: LLMRequestOptions | None = None,
        normalization_vocabulary: NormalizationVocabulary | None = None,
    ) -> None:
        self.client = client
        self.max_evidence_length = max_evidence_length
        self.max_tokens = max_tokens
        self.request_options = request_options or LLMRequestOptions()
        self.normalization_vocabulary = normalization_vocabulary
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
            request_options=self.request_options,
        )
        self.last_usage = result.usage
        if not result.success:
            category = (
                result.error_category.value if result.error_category is not None else "unknown"
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
        operation_properties = response_schema["properties"]["constraint_operations"]["items"][
            "properties"
        ]
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
        instructions = [
            "Extract the user's shopping intent from the supplied conversation context.",
            (
                "Only extract requirements stated in this turn: never infer product "
                "attributes, repeat prior constraints, invent ASINs, or recommend products."
            ),
            (
                "Use hard strength for necessary language (must, need, require, important, "
                "key requirement); use replace only for a new requirement that supersedes an "
                "earlier preference; use remove for explicit no-preference statements."
            ),
            (
                "For reject_products, explicit_rejected_asins may contain only IDs from "
                "recently_shown_asins. Evidence must be an exact substring of user_message."
            ),
        ]
        if RE_CJK_IDEOGRAPH.search(request.user_message):
            instructions.extend(self._chinese_language_guidance())
            if self.normalization_vocabulary is not None:
                instructions.extend(self._normalization_vocabulary_guidance())
        instructions.extend(
            (
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
        )
        return [
            {
                "role": "system",
                "content": "\n".join(instructions),
            },
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ]

    def _normalization_vocabulary_guidance(self) -> tuple[str, ...]:
        assert self.normalization_vocabulary is not None
        allowed_values = {
            attribute: list(values)
            for attribute, values in self.normalization_vocabulary.allowed_values.items()
        }
        return (
            (
                "For category, material, color, size, style, and use_case, use the exact "
                "canonical value from allowed_values when it faithfully matches the user's "
                "meaning. Do not force a wrong match; keep a concise English value when no "
                "canonical value is suitable. feature, brand, budget, and other remain open."
            ),
            "allowed_values="
            + json.dumps(allowed_values, ensure_ascii=False, separators=(",", ":")),
        )

    @staticmethod
    def _chinese_language_guidance() -> tuple[str, ...]:
        example = {
            "dialogue_act": "new_search",
            "category": "jacket",
            "constraint_operations": [
                {
                    "operation": "add",
                    "attribute": "use_case",
                    "value": "hiking",
                    "polarity": "include",
                    "strength": "hard",
                    "evidence": "徒步",
                    "confidence": 0.97,
                },
                {
                    "operation": "add",
                    "attribute": "feature",
                    "value": "waterproof",
                    "polarity": "include",
                    "strength": "hard",
                    "evidence": "必须防水",
                    "confidence": 0.97,
                }
            ],
            "explicit_rejected_asins": [],
            "confidence": 0.97,
            "ambiguities": [],
        }
        return (
            (
                "Chinese-language input mode: user_message contains Simplified Chinese, "
                "Traditional Chinese, or mixed Chinese and English."
            ),
            (
                "Keep all JSON keys and protocol enum values in English. Normalize category, "
                "constraint values, and ambiguity descriptions in English; do not return Chinese "
                "semantic values."
            ),
            (
                "Evidence is the only language exception: copy the shortest supporting span "
                "verbatim from user_message, preserving Chinese characters and punctuation."
            ),
            (
                "Treat 必须/必須, 一定, 务必/務必, 最重要, and 不能接受 as hard necessity; "
                "treat 希望, 偏好, 最好, and 倾向于/傾向於 as soft preference."
            ),
            (
                "Treat 不要, 排除, and 不能是 as exclude constraints. Distinguish them from "
                "single-attribute no preference such as 品牌都可以 or 没有颜色偏好/沒有顏色偏好."
            ),
            (
                "Use replace_constraint and a replace operation for 改成, 换成/換成, "
                "不要之前的, or 忽略前面的 when a new value supersedes an earlier one, "
                "for example 改成深蓝色."
            ),
            (
                "Distinguish 没有其他要求了/沒有其他要求了 (no_more_preferences) from "
                "这些都不合适/這些都不合適 (reject_products)."
            ),
            (
                "For mixed input such as 我想要 waterproof 的外套，预算不超过 100 USD, "
                "normalize the values to English while grounding each evidence span in the input."
            ),
            "Valid Chinese normalization example input: 我想买一件徒步外套，必须防水。",
            (
                "Valid Chinese normalization example output: "
                + json.dumps(example, ensure_ascii=False, separators=(",", ":"))
            ),
        )

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
            normalize_values = bool(
                self.normalization_vocabulary is not None
                and RE_CJK_IDEOGRAPH.search(user_message)
            )
            operations = self._operations(
                payload["constraint_operations"],
                user_message,
                normalize_values=normalize_values,
            )
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
        normalized_category = category.strip() if isinstance(category, str) else None
        if normalize_values and normalized_category is not None:
            assert self.normalization_vocabulary is not None
            normalized_category = self.normalization_vocabulary.canonicalize(
                "category", normalized_category
            )
        return RecognitionResult(
            dialogue_act=dialogue_act,
            category=normalized_category,
            constraint_operations=operations,
            explicit_rejected_asins=rejected,
            confidence=confidence,
            source=RecognitionSource.LLM,
            ambiguities=ambiguities,
            explicit_no_more_preferences=bool(RE_EXPLICIT_NO_MORE_PREFERENCES.search(user_message)),
        )

    def _operations(
        self,
        value: object,
        user_message: str,
        *,
        normalize_values: bool = False,
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
            normalized_value = raw_value.strip()
            if normalize_values:
                assert self.normalization_vocabulary is not None
                normalized_value = self.normalization_vocabulary.canonicalize(
                    attribute,
                    normalized_value,
                )
            operations.append(
                ConstraintOperation(
                    operation=OperationKind(item["operation"]),
                    attribute=attribute,
                    value=normalized_value,
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
