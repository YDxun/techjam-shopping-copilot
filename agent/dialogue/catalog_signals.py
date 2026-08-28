from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from config import constants
from utils import session_utils as su

ATTRIBUTE_ORDER = (
    "material",
    "feature",
    "color",
    "size",
    "style",
    "use_case",
    "budget",
    "brand",
    "category",
    "other",
)
COLORS = (
    "black",
    "white",
    "blue",
    "red",
    "pink",
    "green",
    "brown",
    "gray",
    "grey",
    "purple",
    "yellow",
    "orange",
)
ANSWERABILITY_PRIORS = {
    "material": 0.90,
    "feature": 0.95,
    "color": 0.80,
    "size": 0.65,
    "style": 0.75,
    "use_case": 0.70,
    "budget": 0.75,
    "brand": 0.10,
    "category": 0.30,
    "other": 0.98,
}


@dataclass(frozen=True)
class AttributeSignal:
    coverage: float
    entropy: float
    answer_probability: float

    @property
    def information_gain(self) -> float:
        return self.coverage * self.entropy


@dataclass(frozen=True)
class CatalogQuestionSignals:
    by_category: Mapping[str, Mapping[str, AttributeSignal]]
    constraint_gap_overrides: Mapping[str, float] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> CatalogQuestionSignals:
        return cls(
            by_category={
                "__all__": {
                    attribute: AttributeSignal(0.0, 0.0, 0.15) for attribute in ATTRIBUTE_ORDER
                }
            }
        )

    @classmethod
    def from_products(cls, products: Iterable[dict]) -> CatalogQuestionSignals:
        grouped: dict[str, list[dict]] = defaultdict(list)
        all_products: list[dict] = []
        for product in products:
            if not isinstance(product, dict):
                continue
            all_products.append(product)
            grouped[cls._category(product)].append(product)
        grouped["__all__"] = all_products
        return cls(
            by_category={
                category: cls._signals_for_products(rows) for category, rows in grouped.items()
            }
        )

    def for_category(self, category: str) -> Mapping[str, AttributeSignal]:
        normalized = su.normalize(category)
        if normalized in self.by_category:
            return self.by_category[normalized]
        matching = [
            key
            for key in self.by_category
            if key != "__all__" and (key in normalized or normalized in key)
        ]
        if matching:
            return self.by_category[sorted(matching, key=lambda item: (-len(item), item))[0]]
        return self.by_category.get("__all__", {})

    @staticmethod
    def _category(product: dict) -> str:
        categories = product.get("categories") or []
        if isinstance(categories, list) and categories:
            return su.normalize(str(categories[-1])) or "__all__"
        return "__all__"

    @classmethod
    def _signals_for_products(cls, products: list[dict]) -> dict[str, AttributeSignal]:
        total = len(products)
        counters: dict[str, Counter[str]] = {attribute: Counter() for attribute in ATTRIBUTE_ORDER}
        covered: Counter[str] = Counter()
        for product in products:
            values = cls._attribute_values(product)
            for attribute, extracted in values.items():
                unique = tuple(dict.fromkeys(value for value in extracted if value))
                if not unique:
                    continue
                covered[attribute] += 1
                counters[attribute].update(unique)

        result: dict[str, AttributeSignal] = {}
        for attribute in ATTRIBUTE_ORDER:
            coverage = covered[attribute] / total if total else 0.0
            entropy = cls._normalized_entropy(counters[attribute])
            answer_probability = min(
                1.0,
                0.7 * ANSWERABILITY_PRIORS[attribute] + 0.3 * coverage,
            )
            result[attribute] = AttributeSignal(coverage, entropy, answer_probability)
        result["other"] = AttributeSignal(
            1.0 if total else 0.0,
            1.0 if total else 0.0,
            ANSWERABILITY_PRIORS["other"] if total else 0.15,
        )
        return result

    @classmethod
    def _attribute_values(cls, product: dict) -> dict[str, list[str]]:
        text = cls._text(product).lower()
        details = product.get("details") if isinstance(product.get("details"), dict) else {}
        features = product.get("features") if isinstance(product.get("features"), list) else []
        categories = (
            product.get("categories") if isinstance(product.get("categories"), list) else []
        )
        materials = [
            value for value in constants.MATERIALS if re.search(rf"\b{re.escape(value)}\b", text)
        ]
        colors = [value for value in COLORS if re.search(rf"\b{value}\b", text)]
        sizes = [
            str(value).lower()
            for key, value in details.items()
            if any(token in str(key).lower() for token in ("size", "fit", "width"))
        ]
        styles = [
            str(value).lower()
            for key, value in details.items()
            if any(token in str(key).lower() for token in ("style", "department", "sleeve", "neck"))
        ]
        use_cases = [
            token
            for token in ("running", "hiking", "gym", "winter", "outdoor", "work", "travel")
            if re.search(rf"\b{token}\b", text)
        ]
        return {
            "material": materials[:1],
            "feature": [su.normalize(str(features[0]))] if features else [],
            "color": colors[:1],
            "size": sizes[:1],
            "style": styles[:1],
            "use_case": use_cases[:1],
            "budget": [str(product["price"])] if product.get("price") not in (None, "") else [],
            "brand": [su.normalize(str(product["store"]))] if product.get("store") else [],
            "category": [su.normalize(str(categories[-1]))] if categories else [],
        }

    @staticmethod
    def _text(product: dict) -> str:
        parts: list[str] = []
        for field_name in ("title", "features", "description", "details", "categories", "store"):
            value = product.get(field_name)
            if isinstance(value, dict):
                parts.extend(f"{key} {item}" for key, item in value.items())
            elif isinstance(value, list):
                parts.extend(str(item) for item in value)
            elif value is not None:
                parts.append(str(value))
        return " ".join(parts)

    @staticmethod
    def _normalized_entropy(counter: Counter[str]) -> float:
        total = sum(counter.values())
        if total <= 0 or len(counter) <= 1:
            return 0.0
        entropy = -sum((count / total) * math.log(count / total) for count in counter.values())
        return entropy / math.log(len(counter))
