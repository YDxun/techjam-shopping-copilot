"""Module 1: query building (task step 4): constraint parsing, synonym expansion, optional LLM
    rewrite, query variants.

Output QueryBundle:
    main_query         the main query text
    variant_queries    variant queries (2-3 when RECOVER/enabled)
    structured_filters parsed structured filter conditions (price text -> numeric constraints)
    synonym_expanded   whether synonym expansion was applied
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from retrieval_pipeline import config
from retrieval_pipeline.models import QueryBundle, SessionState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# built-in apparel/e-commerce synonym dictionary (jumper<->sweater etc.)
# ---------------------------------------------------------------------------
SYNONYM_MAP: dict[str, list[str]] = {
    "jumper": ["sweater", "pullover"],
    "sweater": ["jumper", "pullover"],
    "pullover": ["sweater", "jumper"],
    "trainers": ["sneakers", "running shoes"],
    "sneakers": ["trainers", "athletic shoes"],
    "sneaker": ["trainer", "running shoe"],
    "trousers": ["pants", "slacks"],
    "pants": ["trousers", "slacks"],
    "t-shirt": ["tee", "tees", "t shirts"],
    "tee": ["t-shirt", "t shirts"],
    "hoodie": ["sweatshirt", "hooded sweatshirt"],
    "sweatshirt": ["hoodie"],
    "joggers": ["sweatpants", "track pants"],
    "sweatpants": ["joggers", "track pants"],
    "handbag": ["purse", "bag"],
    "purse": ["handbag"],
    "backpack": ["rucksack", "knapsack"],
    "jewelry": ["jewellery"],
    "gym shoes": ["training shoes", "workout shoes"],
    "workout": ["gym", "training"],
    "gift": ["present"],
    "dress": ["gown"],
}

# ---------------------------------------------------------------------------
# data-analysis synonym resource: data/analysis/vocab.json (72 materials / 45 colors / 34 sizes / 44
# styles ...)
# built-in SYNONYM_MAP is the fallback; silently falls back when vocab.json is missing/corrupt.
# ---------------------------------------------------------------------------
_VOCAB_PATH = Path(__file__).resolve().parent.parent / "data" / "analysis" / "vocab.json"
_vocab_cache: dict[str, list[str]] | None = None


def _load_vocab_synonyms() -> dict[str, list[str]]:
    """canonical -> [canonical, *synonyms], sourced from data/analysis/vocab.json."""
    global _vocab_cache
    if _vocab_cache is not None:
        return _vocab_cache
    out: dict[str, list[str]] = {}
    try:
        data = json.loads(_VOCAB_PATH.read_text(encoding="utf-8"))
        for entries in (data.get("dictionaries") or {}).values():
            if not isinstance(entries, dict):
                continue
            for canonical, ent in entries.items():
                if not isinstance(ent, dict):
                    continue
                syns = ent.get("synonyms") or []
                if isinstance(syns, list) and syns:
                    out[str(canonical).lower()] = [
                        str(s).lower() for s in syns if isinstance(s, str)
                    ]
    except Exception as exc:
        logger.warning("[query_builder] vocab.json load failed (fallback to the built-in synonym table): %s", exc)  # noqa: E501
    _vocab_cache = out
    return out


# price text -> numeric constraints
_PRICE_RE = [
    # under $50
    re.compile(r"(?:under|below|less than|max|<=|≤|at most)\s*\$?\s*(\d+(?:\.\d+)?)", re.I),
    # budget around $50
    re.compile(r"budget\s*(?:around|of|:)?\s*\$?\s*(\d+(?:\.\d+)?)", re.I),
    # $30-$50
    re.compile(r"\$?\s*(\d+(?:\.\d+)?)\s*[-–]\s*\$?\s*(\d+(?:\.\d+)?)", re.I),
]
_BUDGET_KEYS = {"budget_max", "budget_min", "price_max", "price_min", "budget"}

# constraint key -> structured-filter field name (used by channel 1)
FILTER_KEY_ALIAS = {
    "material": "material",
    "color": "color",
    "size": "size",
    "style": "style",
    "brand": "brand",
    "budget": "budget",
    "budget_max": "budget_max",
    "budget_min": "budget_min",
    "price_max": "budget_max",
    "price_min": "budget_min",
    "feature": "feature",
    "use_case": "use_case",
    "category": "category",
}


class QueryBuilder:
    """Step 4: compile session_state into a QueryBundle."""

    def __init__(self, enable_llm_rewrite: bool | None = None) -> None:
        self.enable_llm_rewrite = (config.QUERY_REWRITE_ENABLE
                                   if enable_llm_rewrite is None else enable_llm_rewrite)

    # ------------------------------------------------------------------
    def build(self, state: SessionState) -> QueryBundle:
        bundle = QueryBundle()
        raw = (state.user_raw_query or "").strip()

        # 1) constraint parsing (price text -> numeric)
        bundle.structured_filters = self._parse_constraints(state.constraints)
        # 1b) mine price constraints from the raw query text (under $50 -> budget_max=50) when not
        # explicitly given
        if (
            "budget_max" not in bundle.structured_filters
            and "budget_min" not in bundle.structured_filters
        ):
            price = self._extract_price(raw)
            if price is not None:
                bundle.structured_filters["budget_max"] = price

        # 2) synonym expansion: forced on in recovery_mode
        enable_syn = state.strategy_config.enable_synonym or state.recovery_mode
        expanded_query = self._expand_synonyms(raw) if enable_syn else raw
        bundle.synonym_expanded = enable_syn and (expanded_query != raw)

        # 3) main query: optional LLM rewrite / template concatenation
        if state.constraints:
            constraint_text = self._constraints_to_text(bundle.structured_filters)
        else:
            constraint_text = ""   # override cleared the constraints -> use the raw query directly
        if self.enable_llm_rewrite:
            bundle.main_query = self._llm_rewrite(constraint_text, raw) or self._template_join(
                constraint_text, expanded_query
            )
        else:
            bundle.main_query = self._template_join(constraint_text, expanded_query)

        # 4) query variants (RECOVER or enable_query_variant)
        if state.strategy_config.enable_query_variant or state.recovery_mode:
            bundle.variant_queries = self._build_variants(bundle, state)

        logger.debug("[query_builder] main=%r variants=%d filters=%s",
                     bundle.main_query[:80], len(bundle.variant_queries), bundle.structured_filters)
        return bundle

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_constraints(constraints: dict) -> dict:
        """Constraint -> structured filter field; converts natural-language prices to numeric
            (under $50 -> budget_max=50)."""
        filters: dict = {}
        for key, value in (constraints or {}).items():
            k = FILTER_KEY_ALIAS.get(str(key).lower(), str(key).lower())
            if k in _BUDGET_KEYS and not isinstance(value, (int, float)):
                num = QueryBuilder._extract_price(str(value))
                if num is not None:
                    filters["budget_max" if "min" not in k else "budget_min"] = num
                continue
            if value in (None, "", []):
                continue
            if k == "budget" and isinstance(value, str):
                num = QueryBuilder._extract_price(value)
                if num is not None:
                    filters["budget_max"] = num
                continue
            filters[k] = value
        return filters

    @staticmethod
    def _extract_price(text: str) -> float | None:
        """'under $50' / 'less than 30' / '30-50' / 'budget around 25' → float。"""
        for i, pattern in enumerate(_PRICE_RE):
            m = pattern.search(text)
            if not m:
                continue
            if i == 2:  # range -> take the upper bound as budget_max
                return float(m.group(2))
            return float(m.group(1))
        return None

    # ------------------------------------------------------------------
    def _expand_synonyms(self, query: str) -> str:
        out = query
        # built-in e-commerce synonyms (jumper<->sweater etc.)
        for term, synonyms in SYNONYM_MAP.items():
            if re.search(rf"\b{re.escape(term)}\b", query, re.I):
                extra = " ".join(s for s in synonyms if s.lower() not in query.lower())
                if extra:
                    out = f"{out} {extra}"
        # data-analysis vocab.json: on a canonical hit, append all synonyms ("100% cotton" <-> "pure
        # cotton" ...)
        for canonical, synonyms in _load_vocab_synonyms().items():
            if re.search(rf"\b{re.escape(canonical)}\b", query, re.I):
                extra = " ".join(s for s in synonyms
                                 if s != canonical and s.lower() not in out.lower())
                if extra:
                    out = f"{out} {extra}"
        return re.sub(r"\s+", " ", out).strip()

    # ------------------------------------------------------------------
    @staticmethod
    def _constraints_to_text(filters: dict) -> str:
        """Structured filter conditions -> query text (folded into the BM25/dense query)."""
        parts: list[str] = []
        for k, v in filters.items():
            if k in ("budget_max",):
                parts.append(f"under ${v}")
            elif k in ("budget_min",):
                parts.append(f"over ${v}")
            elif isinstance(v, (list, tuple)):
                parts.append(" ".join(str(x) for x in v))
            else:
                parts.append(str(v))
        return " ".join(parts)

    @staticmethod
    def _template_join(constraint_text: str, query: str) -> str:
        """Template concatenation without an LLM: constraints + raw query."""
        if constraint_text:
            return f"{constraint_text} {query}".strip()
        return query.strip()

    # ------------------------------------------------------------------
    def _llm_rewrite(self, constraint_text: str, raw_query: str) -> str | None:
        """Optional LLM query rewrite (step 4); returns None on failure/no-key -> template
            concatenation."""
        try:
            import os

            import openai
            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not api_key:
                return None
            client = openai.OpenAI(
                api_key=api_key, base_url=os.environ.get("OPENAI_BASE_URL") or None
            )
            prompt = (
                "Rewrite this shopping search query into a concise e-commerce retrieval query "
                "(keep material/color/size/budget, drop filler words).\n"
                f"constraints: {constraint_text or 'none'}\n"
                f"query: {raw_query}\nOnly output the rewritten query."
            )
            resp = client.chat.completions.create(
                model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=64,
            )
            text = (resp.choices[0].message.content or "").strip()
            return text or None
        except Exception as exc:
            logger.warning("[query_builder] LLM rewrite failed, fallback to template: %s", exc)
            return None

    # ------------------------------------------------------------------
    def _build_variants(self, bundle: QueryBundle, state: SessionState) -> list[str]:
        """Generate 2-3 query variants: drop secondary constraints, swap synonyms, reorder words."""
        variants: list[str] = []
        base = bundle.main_query
        filters = bundle.structured_filters

        # variant 1: drop color
        if "color" in filters:
            variants.append(self._drop_term(base, str(filters["color"])))

        # variant 2: drop size / budget (secondary constraints)
        for k in ("size", "budget_max"):
            if k in filters:
                drop = str(filters[k])
                if k == "budget_max":
                    drop = f"under ${filters[k]}"
                variants.append(self._drop_term(base, drop))
                break

        # variant 3: the synonym-substituted main query (if not already included)
        if bundle.synonym_expanded and variants:
            variants.append(base)

        # variant 4 (backfill): if all above are empty, use the raw query without constraint text
        if not variants and state.user_raw_query:
            variants.append(state.user_raw_query.strip())

        seen: list[str] = []
        for v in variants:
            v = re.sub(r"\s+", " ", v).strip()
            if v and v != base and v not in seen:
                seen.append(v)
        return seen[: config.MAX_VARIANTS]

    @staticmethod
    def _drop_term(query: str, term: str) -> str:
        return re.sub(rf"\b{re.escape(term)}\b", "", query, flags=re.I)
