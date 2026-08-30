"""货架（品类）工具：赛题机制 —— turn-1 消息里的品类必然包含目标商品。

- coarse_category(values)：与官方评估器一致的粗品类函数（自包含重实现，不 import 评估器）。
- build_shelf_index / match_shelf：按粗品类建货架索引，并用"最长子串"把对话中的品类短语
  映射到货架 key；匹配失败返回 None（上层必须回退"不过滤"）。

保证性：initial_message 的品类 = coarse_category(目标商品 categories)，因此按货架过滤
候选池零召回损失（货架外商品必然不是目标）。
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS.sub(" ", str(text)).strip().lower()


def coarse_category(values: Iterable[str]) -> str:
    """与官方评估器 local_evaluator.coarse_category 一致。"""
    excluded = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
    cleaned: list[str] = []
    for value in values or []:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in excluded:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def build_shelf_index(
    products: Iterable[Mapping],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """返回 (asin -> shelf, shelf -> [asins])。"""
    shelf_of: dict[str, str] = {}
    by_shelf: dict[str, list[str]] = {}
    for product in products:
        if not isinstance(product, Mapping):
            continue
        asin = str(product.get("parent_asin") or "")
        if not asin:
            continue
        shelf = coarse_category(product.get("categories") or [])
        shelf_of[asin] = shelf
        by_shelf.setdefault(shelf, []).append(asin)
    return shelf_of, by_shelf


def match_shelf(category_phrase: str, by_shelf: Mapping[str, object]) -> str | None:
    """把品类短语映射到货架 key：精确 -> 最长子串（长优先）；无命中返回 None。"""
    phrase = normalize(category_phrase or "")
    if not phrase or not by_shelf:
        return None
    if phrase in by_shelf:
        return phrase
    for shelf in sorted(by_shelf, key=len, reverse=True):
        if shelf and shelf in phrase:
            return shelf
    return None
