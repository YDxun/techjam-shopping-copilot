"""模块1｜查询构建（赛题第4步）：约束解析、同义词扩展、可选LLM改写、查询变体。

输出 QueryBundle：
    main_query         主查询文本
    variant_queries    变体查询（RECOVER/开启时 2-3 条）
    structured_filters 解析完成的结构化过滤条件（价格文本→数值约束）
    synonym_expanded   是否做同义词扩展
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
# 内置服饰电商同义词词典（jumper↔sweater 等）
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
# 数据分析产物同义词库：data/analysis/vocab.json（72 材质 / 45 颜色 / 34 尺码 / 44 风格 …）
# 内置 SYNONYM_MAP 作为兜底；vocab.json 缺失/损坏时静默回退内置表。
# ---------------------------------------------------------------------------
_VOCAB_PATH = Path(__file__).resolve().parent.parent / "data" / "analysis" / "vocab.json"
_vocab_cache: dict[str, list[str]] | None = None


def _load_vocab_synonyms() -> dict[str, list[str]]:
    """canonical → [canonical, *synonyms]，来自 data/analysis/vocab.json。"""
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
        logger.warning("[query_builder] vocab.json 加载失败（回退内置同义词表）: %s", exc)
    _vocab_cache = out
    return out


# 价格文本 → 数值约束
_PRICE_RE = [
    # under $50
    re.compile(r"(?:under|below|less than|max|<=|≤|at most)\s*\$?\s*(\d+(?:\.\d+)?)", re.I),
    # budget around $50
    re.compile(r"budget\s*(?:around|of|:)?\s*\$?\s*(\d+(?:\.\d+)?)", re.I),
    # $30-$50
    re.compile(r"\$?\s*(\d+(?:\.\d+)?)\s*[-–]\s*\$?\s*(\d+(?:\.\d+)?)", re.I),
]
_BUDGET_KEYS = {"budget_max", "budget_min", "price_max", "price_min", "budget"}

# 约束键 → 结构化过滤字段名（通道1使用）
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
    """第4步：把 session_state 编译成 QueryBundle。"""

    def __init__(self, enable_llm_rewrite: bool | None = None) -> None:
        self.enable_llm_rewrite = (config.QUERY_REWRITE_ENABLE
                                   if enable_llm_rewrite is None else enable_llm_rewrite)

    # ------------------------------------------------------------------
    def build(self, state: SessionState) -> QueryBundle:
        bundle = QueryBundle()
        raw = (state.user_raw_query or "").strip()

        # 1) 约束解析（价格文本 → 数值）
        bundle.structured_filters = self._parse_constraints(state.constraints)
        # 1b) 从原始 query 文本挖掘价格约束（under $50 → budget_max=50），未显式给定时补全
        if (
            "budget_max" not in bundle.structured_filters
            and "budget_min" not in bundle.structured_filters
        ):
            price = self._extract_price(raw)
            if price is not None:
                bundle.structured_filters["budget_max"] = price

        # 2) 同义词扩展：recovery_mode 强制开启
        enable_syn = state.strategy_config.enable_synonym or state.recovery_mode
        expanded_query = self._expand_synonyms(raw) if enable_syn else raw
        bundle.synonym_expanded = enable_syn and (expanded_query != raw)

        # 3) 主查询：LLM 可选改写 / 模板拼接
        if state.constraints:
            constraint_text = self._constraints_to_text(bundle.structured_filters)
        else:
            constraint_text = ""   # override 已清空约束 → 直接用原始 query
        if self.enable_llm_rewrite:
            bundle.main_query = self._llm_rewrite(constraint_text, raw) or self._template_join(
                constraint_text, expanded_query
            )
        else:
            bundle.main_query = self._template_join(constraint_text, expanded_query)

        # 4) 查询变体（RECOVER 或 enable_query_variant）
        if state.strategy_config.enable_query_variant or state.recovery_mode:
            bundle.variant_queries = self._build_variants(bundle, state)

        logger.debug("[query_builder] main=%r variants=%d filters=%s",
                     bundle.main_query[:80], len(bundle.variant_queries), bundle.structured_filters)
        return bundle

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_constraints(constraints: dict) -> dict:
        """约束 → 结构化过滤字段；把自然语言价格转为数值（under $50 → budget_max=50）。"""
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
            if i == 2:  # range → 取上限作为 budget_max
                return float(m.group(2))
            return float(m.group(1))
        return None

    # ------------------------------------------------------------------
    def _expand_synonyms(self, query: str) -> str:
        out = query
        # 内置电商同义词（jumper↔sweater 等）
        for term, synonyms in SYNONYM_MAP.items():
            if re.search(rf"\b{re.escape(term)}\b", query, re.I):
                extra = " ".join(s for s in synonyms if s.lower() not in query.lower())
                if extra:
                    out = f"{out} {extra}"
        # 数据分析产物 vocab.json：canonical 命中 → 追加全部同义词（"100% cotton"↔"纯棉"…）
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
        """结构化过滤条件 → 查询文本（拼进 BM25/稠密查询）。"""
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
        """无 LLM 时的模板拼接：约束 + 原始 query。"""
        if constraint_text:
            return f"{constraint_text} {query}".strip()
        return query.strip()

    # ------------------------------------------------------------------
    def _llm_rewrite(self, constraint_text: str, raw_query: str) -> str | None:
        """可选 LLM 查询改写（第4步）；失败/无 key 返回 None → 走模板拼接。"""
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
        """生成 2-3 条查询变体：删减次要约束、替换同义词、调整语序。"""
        variants: list[str] = []
        base = bundle.main_query
        filters = bundle.structured_filters

        # 变体1：删掉 color
        if "color" in filters:
            variants.append(self._drop_term(base, str(filters["color"])))

        # 变体2：删掉 size / budget（次要约束）
        for k in ("size", "budget_max"):
            if k in filters:
                drop = str(filters[k])
                if k == "budget_max":
                    drop = f"under ${filters[k]}"
                variants.append(self._drop_term(base, drop))
                break

        # 变体3：同义词替换后的主查询（若尚未包含）
        if bundle.synonym_expanded and variants:
            variants.append(base)

        # 变体4（补足）：若上面都为空，用不含约束文本的原始 query
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
