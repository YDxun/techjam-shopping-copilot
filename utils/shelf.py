"""Shelf (category) utilities: task mechanic -- the category in the turn-1 message always contains
    the target product.

- coarse_category(values): coarse-category function matching the official evaluator (self-contained
reimplementation; does not import the evaluator).
- build_shelf_index / match_shelf: build a shelf index by coarse category and map in-dialog category
phrases
   to shelf keys via "longest substring"; a failed match returns None (the caller must fall back to
   "no filter").

Guarantee: initial_message's category = coarse_category(target's categories), so filtering by shelf
costs zero recall on the candidate pool (anything outside the shelf cannot be the target).
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS.sub(" ", str(text)).strip().lower()


def coarse_category(values: Iterable[str]) -> str:
    """Matches the official evaluator's local_evaluator.coarse_category."""
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
    """Return (asin -> shelf, shelf -> [asins])."""
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
    """Map a category phrase to a shelf key: exact -> longest substring (longest wins); None when
        nothing matches."""
    phrase = normalize(category_phrase or "")
    if not phrase or not by_shelf:
        return None
    if phrase in by_shelf:
        return phrase
    for shelf in sorted(by_shelf, key=len, reverse=True):
        if shelf and shelf in phrase:
            return shelf
    return None
