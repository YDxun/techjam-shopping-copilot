"""Consumer of the field_mapping static table (Pillar I structured-filter precision).

- load():              reads data/analysis/field_mapping.json (lazy cache; falls back to defaults
when missing)
- field_texts(product): extracts per-field lowercase text from a product dict
(title/features/details.<Key>/store/.../price)
- constraint_hit():   field-aware constraint hit score [0,1] (authoritative high confidence; missing
policy pass/unmet/soft)
- expand_with_vocab(): expands constraint values with data/analysis/vocab.json synonyms
("say-it-differently" robustness)

Used with scripts/build_field_mapping.py (generator) and data/analysis/field_mapping.json (finalized
table).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_FIELD_MAPPING_PATH = _ROOT / "data" / "analysis" / "field_mapping.json"
_VOCAB_PATH = _ROOT / "data" / "analysis" / "vocab.json"

# constraint attribute -> vocab dict name (category is category_product_type in the vocab)
_VOCAB_ATTR_MAP = {
    "material": "material",
    "color": "color",
    "size": "size",
    "style": "style",
    "use_case": "use_case",
    "category": "category_product_type",
}

_SINGLE_TOKEN_RE = re.compile(r"[a-z0-9%]+")
_mapping_cache: dict[str, Any] | None = None
_vocab_cache: dict[str, Any] | None = None


def _contains(text: str, term: str) -> bool:
    """Word-boundary matching for single words; substring for phrases (consistent with
        build_field_mapping)."""
    if not term:
        return False
    if re.fullmatch(r"[a-z0-9%]+", term):
        return re.search(rf"(?<![a-z0-9%]){re.escape(term)}(?![a-z0-9%])", text) is not None
    return term in text


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load(path: str | Path | None = None) -> dict[str, Any]:
    """Read field_mapping.json (lazy cache); returns a minimal default table when the file is
        missing."""
    global _mapping_cache
    if _mapping_cache is not None:
        return _mapping_cache
    p = Path(path) if path else _FIELD_MAPPING_PATH
    if p.exists():
        try:
            _mapping_cache = json.loads(p.read_text(encoding="utf-8"))
            return _mapping_cache
        except Exception as exc:
            logger.warning("[field_mapping] load failed (%s), fall back to the default table", exc)
    _mapping_cache = {
        "attributes": {
            "material": {
                "lookup_fields": [{"field": "title", "weight": 1.0}],
                "tolerance": "strict",
                "missing_policy": "unmet",
            },
        },
        "default": {"tolerance": "lenient", "missing_policy": "soft_unmet"},
    }
    return _mapping_cache


def attribute_entry(attribute: str) -> dict[str, Any]:
    mapping = load()
    entry = mapping.get("attributes", {}).get(attribute)
    if entry is None:
        entry = dict(mapping.get("default", {}))
        entry["lookup_fields"] = [
            {"field": "title", "weight": 1.0},
            {"field": "features", "weight": 0.9},
            {"field": "description", "weight": 0.5},
        ]
        entry["primary_field"] = "title"
        entry["tolerance"] = entry.get("tolerance", "lenient")
        entry["missing_policy"] = entry.get("missing_policy", "soft_unmet")
    return entry


# ---------------------------------------------------------------------------
# Field-text extraction
# ---------------------------------------------------------------------------
def _join_lower(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{k} {v}" for k, v in value.items() if v not in (None, "")).lower()
    if isinstance(value, list):
        return " ".join(str(x) for x in value if x not in (None, "")).lower()
    return str(value).lower()


def field_texts(product: dict[str, Any]) -> dict[str, str]:
    """Extract per-field lowercase product text; details.<Key> is normalized to forms like
        details.Material."""
    out: dict[str, str] = {}
    out["title"] = _join_lower(product.get("title"))
    out["features"] = _join_lower(product.get("features"))
    out["store"] = _join_lower(product.get("store"))
    out["categories"] = _join_lower(product.get("categories"))
    out["description"] = _join_lower(product.get("description"))
    details = product.get("details")
    if isinstance(details, dict):
        out["details"] = _join_lower(details)
        for key, val in details.items():
            if val in (None, ""):
                continue
            kl = str(key).lower()
            out[f"details.{kl.capitalize()}"] = str(val).lower()
    price = product.get("price")
    out["price"] = "" if price is None else str(price).lower()
    return out


# ---------------------------------------------------------------------------
# vocab synonym expansion ("say-it-differently" robustness)
# ---------------------------------------------------------------------------
def expand_with_vocab(attribute: str, value: str) -> list[str]:
    """Return [value] + synonyms (expands when value matches a vocab canonical)."""
    terms = [value]
    vocab = _load_vocab()
    dict_name = _VOCAB_ATTR_MAP.get(attribute)
    if dict_name is None or not vocab:
        return terms
    entries = vocab.get("dictionaries", {}).get(dict_name, {})
    v = value.strip().lower()
    if v in entries:
        syns = [s for s in entries[v].get("synonyms", []) if isinstance(s, str) and s]
        terms.extend(syns)
    return terms


def _load_vocab() -> dict[str, Any] | None:
    """With ASSET_VOCAB_EXPAND=1, prefer data/assets/vocab_v2_clean.json (denoised/refined
        version)."""
    global _vocab_cache
    if _vocab_cache is not None:
        return _vocab_cache
    import os

    path = _VOCAB_PATH
    if os.environ.get("ASSET_VOCAB_EXPAND", "0").strip().lower() in {"1", "true", "yes", "on"}:
        alt = _ROOT / "data" / "assets" / "vocab_v2_clean.json"
        if alt.exists():
            path = alt
    if path.exists():
        try:
            _vocab_cache = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            _vocab_cache = None
    else:
        _vocab_cache = None
    return _vocab_cache


# ---------------------------------------------------------------------------
# Field-aware constraint hit
# ---------------------------------------------------------------------------
def _budget_hit(value: str, product: dict[str, Any]) -> float:
    """Numeric price check: missing price passes through (79% missing); 'under $50'-><=,
        'over'->>=."""
    price = product.get("price")
    if not isinstance(price, (int, float)) or isinstance(price, bool):
        return 1.0  # missing price passes through (missing_policy=pass)
    m = re.search(r"(\d+(?:\.\d+)?)", str(value))
    if not m:
        return 0.5
    limit = float(m.group(1))
    v = value.lower()
    if any(w in v for w in ("over", "more than", "above", "min", ">=", "at least")):
        return 1.0 if price >= limit else 0.0
    return 1.0 if price <= limit else 0.0


def constraint_hit(
    attribute: str,
    value: str,
    tokens: tuple[str, ...] | list[str] | None = None,
    product: dict[str, Any] | None = None,
    text: str = "",
    *,
    extra_terms: tuple[str, ...] | list[str] = (),
) -> float:
    """Field-aware constraint hit score [0,1].

    - attribute=='budget': numeric price check (missing passes through)
    - hit on an authoritative field -> 1.0; hit on other lookup fields -> weight;
    - no hit: unmet->0 / pass->1 / soft_unmet->partial credit by token coverage.
    - without a product, fall back to full-text matching (legacy callers).
    """
    if not value and not tokens:
        return 0.5
    entry = attribute_entry(attribute)
    terms = [t for t in expand_with_vocab(attribute, value) if t] + list(extra_terms)
    if tokens:
        terms.extend(t for t in tokens if t not in terms)

    if attribute == "budget" and product is not None:
        return _budget_hit(value, product)

    if product is not None:
        ft = field_texts(product)
    else:
        ft = {}  # no product -> use full-text matching on text

    best = 0.0
    for lf in entry.get("lookup_fields", []):
        field = lf.get("field", "title")
        weight = float(lf.get("weight", 0.5))
        txt = ft.get(field, "")
        if txt and any(_contains(txt, t) for t in terms):
            if lf.get("authoritative"):
                return 1.0
            best = max(best, weight)
    if best > 0:
        return best

    # without a product, fall back to full-text matching (legacy callers: the text argument)
    if product is None and text:
        if any(_contains(text, t) for t in terms):
            return 1.0
        if tokens:
            hit = sum(1 for t in tokens if t in text)
            return hit / len(tokens)

    missing_policy = entry.get("missing_policy", "soft_unmet")
    if missing_policy == "pass":
        return 1.0
    if missing_policy == "unmet":
        return 0.0
    # soft_unmet: partial credit by token coverage
    if tokens:
        if text:
            hit = sum(1 for t in tokens if t in text)
            return hit / len(tokens) * 0.5
        if product is not None:
            ft_all = ft.get("details", "") + ft.get("title", "") + ft.get("features", "")
            hit = sum(1 for t in tokens if t in ft_all)
            return hit / len(tokens) * 0.5
    return 0.0
