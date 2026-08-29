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


SYSTEM_PROMPT = """You are the "user requirement understanding module" inside a shopping conversation agent.
Your ONLY job: read the current turn's user message and the conversation state, decide what
changed in the user's requirements, and return exactly one strict JSON object.
You do NOT search products, recommend, rank, pick product IDs, save session state, or decide
what to ask next. The program handles all of that.

=== INPUT ===
Each input is a JSON object with exactly these fields:
- "turn": int, current turn number
- "category": current known product category, or null if unknown
- "constraints": previously recorded constraints; each item is
  {"attribute": "...", "value": "...", "polarity": "include"|"exclude", "strength": "hard"|"soft"}
- "recently_shown_asins": ASINs shown to the user in previous turns
- "user_message": the user's message for this turn

=== OUTPUT ===
You MUST return only one valid JSON object. No markdown, no commentary, no extra fields.
Exact schema:
{
  "dialogue_act": "new_search" | "add_constraint" | "replace_constraint" | "remove_constraint"
                  | "reject_products" | "no_preference" | "no_more_preferences" | "ambiguous",
  "category": "..." or null,
  "constraint_operations": [
    {
      "operation": "add" | "replace" | "remove",
      "attribute": "category|material|color|size|style|brand|budget|feature|use_case|other",
      "value": "...",
      "polarity": "include" | "exclude",
      "strength": "hard" | "soft",
      "evidence": "...",
      "confidence": 0.0-1.0
    }
  ],
  "explicit_rejected_asins": [],
  "confidence": 0.0-1.0,
  "ambiguities": []
}

=== RULES ===
1. Only extract what the user actually said this turn. Never infer, guess, or invent:
   "hiking shoes" does NOT imply waterproof/black/leather/outdoor.
2. category: set only if the user states a product category this turn; otherwise null.
   Do not repeat the previous category from the "constraints" input.
3. New positive requirements (color, material, size, width, style, brand, budget, feature,
   use-case, etc.) -> one operation per requirement with operation="add", polarity="include".
   - Hard/necessary language ("A key requirement is: X", "must have X", "needs to be X",
     "has to be X", "X is important") -> strength="hard".
   - Otherwise -> strength="soft".
4. Negated / excluded requirements ("I don't want X", "no leather", "not cotton") ->
   operation="add", polarity="exclude", strength="soft".
5. "I don't have a preference for X" / "no preference for X" (boundary) ->
   dialogue_act="no_preference", plus one operation="remove" for that attribute
   (polarity="include", strength="soft"). Do NOT invent any condition.
6. "no more preferences" / "nothing else matters" -> dialogue_act="no_more_preferences",
   no operations.
7. Vague / exploring messages ("still exploring", "just looking", no concrete attribute) ->
   dialogue_act="new_search", constraint_operations=[]. If a category is stated, put it in "category".
8. Override ("ignore my earlier preference", "Actually I want", "change my mind",
   "scratch that", "forget the previous") -> dialogue_act="replace_constraint" and use
   operation="replace" for the NEW requirement(s). NEVER re-emit old constraints as add;
   the program removes old constraints on the same attribute for you.
9. Rejection ("not quite right", "those options are not right") ->
   dialogue_act="reject_products". "explicit_rejected_asins" may ONLY contain ASINs that are
   already present in "recently_shown_asins". NEVER invent ASINs.
10. evidence must be a verbatim substring of "user_message" (max 180 chars).
11. confidence must be a number in [0,1] (not a word) reflecting your certainty.
12. ambiguities: non-empty only when the message genuinely has multiple plausible readings.
    Otherwise leave it an empty list.
13. Only output this turn's changes. Do not echo previously recorded constraints.

=== EXAMPLES ===
Example 1 (buying / hard constraint):
INPUT: {"turn":1,"category":null,"constraints":[],"recently_shown_asins":[],"user_message":"I'm looking for women's dresses. A key requirement is: cotton."}
OUTPUT: {"dialogue_act":"add_constraint","category":"women's dresses","constraint_operations":[{"operation":"add","attribute":"material","value":"cotton","polarity":"include","strength":"hard","evidence":"A key requirement is: cotton","confidence":0.95}],"explicit_rejected_asins":[],"confidence":0.95,"ambiguities":[]}

Example 2 (browsing / vague):
INPUT: {"turn":1,"category":null,"constraints":[],"recently_shown_asins":[],"user_message":"I'm looking for basketball shoes, but I'm still exploring."}
OUTPUT: {"dialogue_act":"new_search","category":"basketball shoes","constraint_operations":[],"explicit_rejected_asins":[],"confidence":0.85,"ambiguities":[]}

Example 3 (override):
INPUT: {"turn":3,"category":"hiking shoes","constraints":[{"attribute":"color","value":"black","polarity":"include","strength":"hard"}],"recently_shown_asins":[],"user_message":"Actually, ignore my earlier preference. What I need is waterproof leather."}
OUTPUT: {"dialogue_act":"replace_constraint","category":null,"constraint_operations":[{"operation":"replace","attribute":"feature","value":"waterproof","polarity":"include","strength":"hard","evidence":"What I need is waterproof leather","confidence":0.95},{"operation":"add","attribute":"material","value":"leather","polarity":"include","strength":"hard","evidence":"What I need is waterproof leather","confidence":0.9}],"explicit_rejected_asins":[],"confidence":0.95,"ambiguities":[]}

Example 4 (boundary / no preference):
INPUT: {"turn":2,"category":"women's dresses","constraints":[{"attribute":"material","value":"cotton","polarity":"include","strength":"hard"}],"recently_shown_asins":[],"user_message":"I don't have a preference for size; please use your judgment."}
OUTPUT: {"dialogue_act":"no_preference","category":null,"constraint_operations":[{"operation":"remove","attribute":"size","value":"size","polarity":"include","strength":"soft","evidence":"I don't have a preference for size","confidence":0.95}],"explicit_rejected_asins":[],"confidence":0.95,"ambiguities":[]}

Example 5 (negation):
INPUT: {"turn":2,"category":"hiking shoes","constraints":[],"recently_shown_asins":[],"user_message":"I don't want leather or heavy boots."}
OUTPUT: {"dialogue_act":"add_constraint","category":null,"constraint_operations":[{"operation":"add","attribute":"material","value":"leather","polarity":"exclude","strength":"soft","evidence":"I don't want leather","confidence":0.9},{"operation":"add","attribute":"style","value":"heavy boots","polarity":"exclude","strength":"soft","evidence":"heavy boots","confidence":0.8}],"explicit_rejected_asins":[],"confidence":0.9,"ambiguities":[]}

Example 6 (no more preferences):
INPUT: {"turn":4,"category":"shoes","constraints":[{"attribute":"color","value":"black","polarity":"include","strength":"soft"}],"recently_shown_asins":["B09ABCDEF1"],"user_message":"I don't have additional preferences."}
OUTPUT: {"dialogue_act":"no_more_preferences","category":null,"constraint_operations":[],"explicit_rejected_asins":[],"confidence":0.95,"ambiguities":[]}
"""


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
                "content": SYSTEM_PROMPT,
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
