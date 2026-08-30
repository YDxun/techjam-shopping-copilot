"""General session/text utilities (Pillar II/III: slot parsing, constraint-type classification,
    query-term construction)."""
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
    """Lowercase tokenization with default stop-word removal (for retrieval/coverage)."""
    tokens = [t.lower() for t in TOKEN_RE.findall(text or "")]
    if keep_stopwords:
        return tokens
    return [t for t in tokens if len(t) > 1 and t not in constants.STOPWORDS]


def normalize(text: str) -> str:
    """Normalize a constraint/phrase: strip edge punctuation, collapse whitespace, lowercase."""
    return re.sub(r"\s+", " ", (text or "").strip(" -;,.\t\n")).strip().lower()


def strip_constraint_prefix(value: str) -> str:
    """Strip prefixes like 'Material:' / 'color:', keeping the actual value words."""
    return _PREFIX_RE.sub("", value or "").strip(" -;,.\t\n")


def classify_attribute(value: str) -> str:
    """Classify a constraint value into the official ask_attribute set (mirrors the evaluator logic
        but self-contained)."""
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
    """Split a 'X; Y; Z' message into multiple constraint values."""
    return [v.strip() for v in (text or "").split(";") if v.strip()]


def constraint_key(value: str) -> str:
    """Constraint dedup key: normalized + prefix-stripped."""
    return normalize(strip_constraint_prefix(value))


def group_tokens(value: str, max_tokens: int = 6) -> tuple[str, ...]:
    """Query-term tuple for a constraint value: keeps '%' (e.g. 100% cotton), removes stop words,
        truncates to avoid noise."""
    toks = tokenize(strip_constraint_prefix(value))
    return tuple(dict.fromkeys(toks[:max_tokens]))


def phrase_exists(lower_text: str, value: str) -> bool:
    """Whether the constraint text (prefix-stripped, truncated) appears as a substring in the
        product text."""
    key = constraint_key(value)[:180]
    return len(key) >= 3 and key in lower_text


def all_tokens_in(lower_text: str, tokens: Iterable[str]) -> bool:
    """Whether all tokens in a group hit the product text (AND within the group)."""
    toks = list(tokens)
    return bool(toks) and all(t in lower_text for t in toks)


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
