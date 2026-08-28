"""通用会话/文本工具（Pillar II/III：槽位解析、约束类型归类、检索词构建）。"""
from __future__ import annotations

import re
from typing import Iterable

from config import constants

TOKEN_RE = re.compile(r"[a-z0-9%]+", re.IGNORECASE)
_PREFIX_RE = re.compile(
    r"^\s*(material|color|size|style|feature|use[_-]?case|budget|brand|department)\s*[:：]\s*",
    re.I,
)
_COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b",
    re.I,
)
_SIZE_RE = re.compile(
    r"\b(size|sizing|width|wide|narrow|small|medium|large|x[sl]|extra\s*(small|large))\b",
    re.I,
)
_STYLE_RE = re.compile(
    r"\b(department|style|fit|sleeve|neck|crew|v-?neck|round|regular|slim|loose|classic|casual|formal)\b",
    re.I,
)
_USE_CASE_RE = re.compile(
    r"\b(hiking|running|gym|winter|outdoor|work|sports|travel|gift|party|wedding|athletic|walking)\b",
    re.I,
)
_BUDGET_RE = re.compile(r"(?:budget|under|<=|\\$)\s*\d", re.I)


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    """小写 token 化，默认去除停用词（用于检索与覆盖度匹配）。"""
    tokens = [t.lower() for t in TOKEN_RE.findall(text or "")]
    if keep_stopwords:
        return tokens
    return [t for t in tokens if len(t) > 1 and t not in constants.STOPWORDS]


def normalize(text: str) -> str:
    """约束/短语规范化：去首尾标点、压缩空白、转小写。"""
    return re.sub(r"\s+", " ", (text or "").strip(" -;,.\t\n")).strip().lower()


def strip_constraint_prefix(value: str) -> str:
    """剥掉 'Material:' / 'color:' 等前缀，保留真实取值词。"""
    return _PREFIX_RE.sub("", value or "").strip(" -;,.\t\n")


def classify_attribute(value: str) -> str:
    """把约束取值归类为官方 ask_attribute 集合（与评估器逻辑镜像，但自包含）。"""
    lowered = (value or "").lower()
    if _BUDGET_RE.search(lowered) or "budget" in lowered:
        return "budget"
    if any(m in lowered for m in constants.MATERIALS):
        return "material"
    if _COLOR_RE.search(lowered) or any(
        c in lowered for c in ("color", "black", "white", "blue", "red", "pink", "green")
    ):
        return "color"
    if _SIZE_RE.search(lowered):
        return "size"
    if _STYLE_RE.search(lowered):
        return "style"
    if _USE_CASE_RE.search(lowered):
        return "use_case"
    return "feature"


def split_values(text: str) -> list[str]:
    """把 'X; Y; Z' 消息拆成多条约束取值。"""
    return [v.strip() for v in (text or "").split(";") if v.strip()]


def constraint_key(value: str) -> str:
    """约束去重键：规范化 + 去前缀。"""
    return normalize(strip_constraint_prefix(value))


def group_tokens(value: str, max_tokens: int = 6) -> tuple[str, ...]:
    """约束值的检索词元组：保留 '%'（如 100% cotton），去停用词，截断防噪声。"""
    toks = tokenize(strip_constraint_prefix(value))
    return tuple(dict.fromkeys(toks[:max_tokens]))


def phrase_exists(lower_text: str, value: str) -> bool:
    """约束原文（去前缀、截断）是否作为子串出现在商品文本中。"""
    key = constraint_key(value)[:180]
    return len(key) >= 3 and key in lower_text


def all_tokens_in(lower_text: str, tokens: Iterable[str]) -> bool:
    """一组 token 是否全部命中商品文本（组内 AND）。"""
    toks = list(tokens)
    return bool(toks) and all(t in lower_text for t in toks)


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
