from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from utils import session_utils as su

ATTRIBUTE_NAMES = (
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "category",
)
MATCHED_ATTRIBUTES = ("material", "color", "style", "use_case")
DETAIL_KEYWORDS = {
    "material": ("material", "fabric", "textile", "made of", "made from"),
    "color": ("color", "colour", "shade", "tone"),
    "size": ("size", "sizing", "fit", "width", "measurement", "dimension"),
    "style": ("style", "design", "look", "cut", "department", "sleeve", "neck"),
    "brand": ("brand", "manufacturer", "label"),
    "feature": ("feature", "features", "benefit", "function"),
    "use_case": ("use", "purpose", "occasion", "activity", "event"),
}
SHOE_PATTERN = re.compile(
    r"\b(?:shoe|shoes|sneaker|sneakers|boot|boots|sandal|sandals|heel|heels|flat|flats|"
    r"loafer|loafers|slipper|slippers)\b"
)
RAW_SIZE_NUMBER_PATTERN = re.compile(r"(?<![a-z0-9])(\d{1,2}(?:\.5)?)(?![a-z0-9])")
STRUCTURED_COMPACT_REGION_SIZE_PATTERN = re.compile(r"\b(?:us|uk|eu)(\d{1,2}(?:\.5)?)\b")
SIZE_NUMBER_CONTEXT_PATTERN = re.compile(
    r"\b(?:size|sizing|us|uk|eu)(?:\s+size)?\b\s*[:#-]?\s*"
    r"(?P<after>\d{1,2}(?:\.5)?)\b"
    r"|(?<![a-z0-9])(?P<before>\d{1,2}(?:\.5)?)\s*"
    r"(?:size|sizing|us|uk|eu)(?:\s+size)?\b"
)
GENERIC_BRANDS = frozenset(
    {
        "",
        "amazon",
        "amazon com",
        "brand",
        "generic",
        "n a",
        "na",
        "no brand",
        "none",
        "not specified",
        "store",
        "unbranded",
        "unknown",
    }
)
CONTROLLED_FEATURES = {
    "breathable": ("breathable",),
    "lightweight": ("lightweight", "ultra light"),
    "quick_dry": ("quick dry", "quick-dry"),
    "waterproof": ("waterproof", "water resistant", "water-resistant"),
    "insulated": ("insulated",),
    "non_slip": ("non slip", "non-slip", "anti slip", "anti-slip"),
    "adjustable": ("adjustable",),
}


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


@lru_cache(maxsize=1)
def load_vocabulary() -> Mapping[str, object]:
    """Load the checked-in normalized vocabulary once without exposing mutable state."""
    path = Path(__file__).resolve().parents[2] / "data" / "analysis" / "vocab.json"
    return _freeze(json.loads(path.read_text(encoding="utf-8-sig")))  # type: ignore[return-value]


def _vocabulary_version(vocabulary: Mapping[str, object]) -> str:
    plain = _plain(vocabulary)
    digest = sha256(
        json.dumps(plain, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
    ).hexdigest()
    meta = vocabulary.get("meta")
    version = meta.get("version", "unknown") if isinstance(meta, Mapping) else "unknown"
    return f"{version}:{digest}"


@dataclass(frozen=True)
class AttributeProfile:
    parent_asin: str
    values: Mapping[str, frozenset[str]]
    confidence: Mapping[str, float]
    sources: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        frozen_values = {
            str(attribute): frozenset(str(value) for value in values)
            for attribute, values in self.values.items()
        }
        frozen_confidence = {
            str(attribute): float(value) for attribute, value in self.confidence.items()
        }
        frozen_sources = {
            str(attribute): tuple(str(value) for value in values)
            for attribute, values in self.sources.items()
        }
        object.__setattr__(self, "values", MappingProxyType(frozen_values))
        object.__setattr__(self, "confidence", MappingProxyType(frozen_confidence))
        object.__setattr__(self, "sources", MappingProxyType(frozen_sources))


class CatalogAttributeExtractor(Protocol):
    @property
    def vocabulary_version(self) -> str: ...

    def prepare(self, products: Iterable[dict[str, object]]) -> "CatalogAttributeExtractor": ...

    def extract(self, product: dict[str, object]) -> AttributeProfile: ...


@dataclass(frozen=True)
class RuleVocabularyExtractor:
    vocabulary: Mapping[str, object] = field(default_factory=load_vocabulary)
    price_boundaries: Mapping[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def vocabulary_version(self) -> str:
        return _vocabulary_version(self.vocabulary)

    def prepare(self, products: Iterable[dict[str, object]]) -> "RuleVocabularyExtractor":
        prices: dict[str, list[float]] = {}
        for product in products:
            price = _price(product.get("price"))
            if price is None:
                continue
            prices.setdefault(_category_key(product), []).append(price)
        boundaries = {
            category: (_quantile(values, 0.25), _quantile(values, 0.75))
            for category, values in prices.items()
        }
        return replace(self, price_boundaries=MappingProxyType(boundaries))

    def extract(self, product: dict[str, object]) -> AttributeProfile:
        values = {attribute: frozenset() for attribute in ATTRIBUTE_NAMES}
        confidence = {attribute: 0.0 for attribute in ATTRIBUTE_NAMES}
        sources = {attribute: () for attribute in ATTRIBUTE_NAMES}
        details = product.get("details")
        detail_map = details if isinstance(details, Mapping) else {}

        for attribute in MATCHED_ATTRIBUTES:
            extracted, source, score = self._extract_attribute(attribute, product, detail_map)
            values[attribute], sources[attribute], confidence[attribute] = extracted, source, score

        size, size_source, size_score = self._extract_size(product, detail_map)
        values["size"], sources["size"], confidence["size"] = size, size_source, size_score

        feature, feature_source, feature_score = self._extract_feature(product, detail_map)
        values["feature"], sources["feature"], confidence["feature"] = (
            feature,
            feature_source,
            feature_score,
        )

        brand, brand_source, brand_score = self._extract_brand(product, detail_map)
        values["brand"], sources["brand"], confidence["brand"] = brand, brand_source, brand_score

        category = _category_key(product)
        if category != "__all__":
            values["category"] = frozenset({category})
            confidence["category"] = 0.9
            sources["category"] = ("categories",)

        price = _price(product.get("price"))
        boundary = self.price_boundaries.get(category)
        if price is not None and boundary is not None:
            low, high = boundary
            values["budget"] = frozenset({_price_bucket(price, low, high)})
            confidence["budget"] = 1.0
            sources["budget"] = ("price",)

        return AttributeProfile(
            parent_asin=str(product.get("parent_asin", "")),
            values=values,
            confidence=confidence,
            sources=sources,
        )

    def _extract_attribute(
        self,
        attribute: str,
        product: Mapping[str, object],
        details: Mapping[object, object],
    ) -> tuple[frozenset[str], tuple[str, ...], float]:
        detail_text = _details_for(attribute, details)
        structured = self._matches(attribute, detail_text, include_short=True)
        if structured:
            return structured, ("details",), 1.0
        for field_name, score in (("title", 0.8), ("features", 0.75)):
            matched = self._matches(attribute, _as_text(product.get(field_name)))
            if matched:
                return matched, (field_name,), score
        description = self._matches(attribute, _as_text(product.get("description")))
        if description:
            return description, ("description",), 0.35
        return frozenset(), (), 0.0

    def _extract_size(
        self,
        product: Mapping[str, object],
        details: Mapping[object, object],
    ) -> tuple[frozenset[str], tuple[str, ...], float]:
        detail_text = _details_for("size", details)
        for text, source, score, include_short in (
            (detail_text, "details", 1.0, True),
            (_as_text(product.get("title")), "title", 0.8, False),
            (_as_text(product.get("features")), "features", 0.75, False),
            (_as_text(product.get("description")), "description", 0.35, False),
        ):
            size = self._size_value(text, _is_shoe_product(product), include_short)
            if size:
                return frozenset({size}), (source,), score
        return frozenset(), (), 0.0

    def _size_value(self, text: str, is_shoe: bool, include_short: bool) -> str | None:
        normalized = su.normalize(text)
        if not normalized:
            return None
        number = RAW_SIZE_NUMBER_PATTERN.search(normalized)
        compact_structured_number = (
            _compact_structured_size_number(normalized) if include_short else None
        )
        contextual_number = _size_number_with_context(normalized)
        if is_shoe and include_short and (compact_structured_number or number):
            return f"shoe_size:{compact_structured_number or number.group(1)}"
        if is_shoe and contextual_number:
            return f"shoe_size:{contextual_number}"
        if number and not (include_short or contextual_number):
            return None
        matches = self._matches(
            "size", normalized, include_short=include_short or bool(contextual_number)
        )
        apparel = sorted(value for value in matches if value != "shoe_size")
        if apparel:
            return f"apparel_size:{apparel[0]}"
        return None

    def _extract_feature(
        self,
        product: Mapping[str, object],
        details: Mapping[object, object],
    ) -> tuple[frozenset[str], tuple[str, ...], float]:
        detail_text = _details_for("feature", details)
        for text, source, score in (
            (detail_text, "details", 1.0),
            (_as_text(product.get("title")), "title", 0.8),
            (_as_text(product.get("features")), "features", 0.75),
            (_as_text(product.get("description")), "description", 0.35),
        ):
            matched = _controlled_matches(text)
            if matched:
                return matched, (source,), score
        return frozenset(), (), 0.0

    def _extract_brand(
        self,
        product: Mapping[str, object],
        details: Mapping[object, object],
    ) -> tuple[frozenset[str], tuple[str, ...], float]:
        detailed = _normalize_brand(_details_for("brand", details))
        if detailed:
            return frozenset({detailed}), ("details",), 1.0
        store = _normalize_brand(_as_text(product.get("store")))
        if store:
            return frozenset({store}), ("store",), 0.75
        return frozenset(), (), 0.0

    def _matches(self, attribute: str, text: str, *, include_short: bool = False) -> frozenset[str]:
        dictionaries = self.vocabulary.get("dictionaries")
        entries = dictionaries.get(attribute) if isinstance(dictionaries, Mapping) else None
        if not isinstance(entries, Mapping):
            return frozenset()
        normalized = su.normalize(text)
        matches: set[str] = set()
        for entry in entries.values():
            if not isinstance(entry, Mapping) or "seed" not in entry.get("sources", ()):
                continue
            canonical = entry.get("canonical")
            synonyms = entry.get("synonyms")
            if not isinstance(canonical, str) or not isinstance(synonyms, (list, tuple)):
                continue
            if any(_contains(normalized, str(term), include_short) for term in synonyms):
                matches.add(canonical)
        return frozenset(matches)


@dataclass(frozen=True)
class CatalogAttributeCache:
    profiles: Mapping[str, AttributeProfile]
    vocabulary_version: str
    catalog_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "profiles", MappingProxyType(dict(self.profiles)))

    @classmethod
    def from_products(
        cls,
        products: Iterable[dict[str, object]],
        extractor: CatalogAttributeExtractor,
    ) -> "CatalogAttributeCache":
        materialized = _normalized_unique_products(products)
        prepared = extractor.prepare(materialized)
        profiles = {
            profile.parent_asin: profile
            for profile in (prepared.extract(product) for product in materialized)
            if profile.parent_asin
        }
        return cls(
            profiles=profiles,
            vocabulary_version=prepared.vocabulary_version,
            catalog_fingerprint=catalog_fingerprint(profiles),
        )

    def for_asin(self, asin: str) -> AttributeProfile | None:
        return self.profiles.get(asin)


def catalog_fingerprint(profiles: Mapping[str, AttributeProfile]) -> str:
    rows = []
    for asin in sorted(profiles):
        profile = profiles[asin]
        rows.append(
            json.dumps(
                {
                    "asin": asin,
                    "confidence": dict(sorted(profile.confidence.items())),
                    "values": {
                        attribute: sorted(values)
                        for attribute, values in sorted(profile.values.items())
                    },
                },
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    return sha256("\n".join(rows).encode()).hexdigest()


def _as_text(value: object) -> str:
    if isinstance(value, Mapping):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, (list, tuple, set, frozenset)):
        return " ".join(str(item) for item in value)
    return str(value) if value is not None else ""


def _details_for(attribute: str, details: Mapping[object, object]) -> str:
    keywords = DETAIL_KEYWORDS[attribute]
    return " ".join(
        _as_text(value)
        for key, value in details.items()
        if any(keyword in su.normalize(str(key)) for keyword in keywords)
    )


def _contains(text: str, term: str, include_short: bool) -> bool:
    normalized = su.normalize(term)
    if not normalized or (len(normalized) == 1 and not include_short):
        return False
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])", text))


def _controlled_matches(text: str) -> frozenset[str]:
    normalized = su.normalize(text)
    return frozenset(
        canonical
        for canonical, terms in CONTROLLED_FEATURES.items()
        if any(_contains(normalized, term, include_short=True) for term in terms)
    )


def _size_number_with_context(text: str) -> str | None:
    match = SIZE_NUMBER_CONTEXT_PATTERN.search(text)
    if match is None:
        return None
    return match.group("after") or match.group("before")


def _compact_structured_size_number(text: str) -> str | None:
    match = STRUCTURED_COMPACT_REGION_SIZE_PATTERN.search(text)
    return match.group(1) if match else None


def _normalize_brand(value: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", " ", su.normalize(value)).strip()
    if normalized in GENERIC_BRANDS or len(normalized) < 2:
        return None
    return normalized


def _category_key(product: Mapping[str, object]) -> str:
    categories = product.get("categories")
    if isinstance(categories, (list, tuple)) and categories:
        return su.normalize(str(categories[-1])) or "__all__"
    return "__all__"


def _is_shoe_product(product: Mapping[str, object]) -> bool:
    text = " ".join(
        (
            _as_text(product.get("categories")),
            _as_text(product.get("title")),
        )
    ).lower()
    return bool(SHOE_PATTERN.search(text))


def _normalized_unique_products(
    products: Iterable[dict[str, object]],
) -> tuple[dict[str, object], ...]:
    selected: dict[str, tuple[str, dict[str, object]]] = {}
    for product in products:
        if not isinstance(product, dict):
            continue
        asin = _normalized_asin(product.get("parent_asin"))
        if asin is None:
            continue
        normalized = dict(product)
        normalized["parent_asin"] = asin
        canonical = json.dumps(
            normalized,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        current = selected.get(asin)
        if current is None or canonical < current[0]:
            selected[asin] = (canonical, normalized)
    return tuple(selected[asin][1] for asin in sorted(selected))


def _normalized_asin(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _price(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _price_bucket(price: float, low: float, high: float) -> str:
    if price <= low:
        return "budget_low"
    if price >= high:
        return "budget_high"
    return "budget_mid"
