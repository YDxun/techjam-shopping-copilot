"""Static data-asset loader (bundled offline optimization assets).

Offline-first; missing/corrupt assets auto-degrade to "no expansion":
- vocab_v2_clean.json     vocabulary: canonical + synonyms (material/color/size/style/...)
- category_mapping.json   category routing: audience/family aliases -> canonical product types
- review_paraphrases.json review paraphrases: size_fit / material_language / color_language
- field_mapping.json      field mapping: attribute -> lookup fields / weights / match policy
(reserved)

Usage:
    assets = load_assets()
    assets.vocab_expand("material", "grey")      -> ["grey", "gray", "heather grey", ...]
    assets.category_expand("Tops & Tees Tanks & Camis") -> ["tank", "tops", "tshirts", ...]
    assets.paraphrase_operations("I need something made of cotton")
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).resolve().parent.parent / "data" / "assets"

_VOCAB_FILE = "vocab_v2_clean.json"
_CATEGORY_FILE = "category_mapping.json"
_PARAPHRASE_FILE = "review_paraphrases.json"
_FIELD_FILE = "field_mapping.json"


@dataclass(frozen=True)
class DataAssets:
    vocab: dict = field(default_factory=dict)
    category_map: dict = field(default_factory=dict)
    paraphrases: dict = field(default_factory=dict)
    field_map: dict = field(default_factory=dict)
    loaded: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return bool(self.loaded)

    # ------------------------------------------------------------------
    # vocab: constraint value -> synonym phrases (for BM25 query expansion)
    # ------------------------------------------------------------------
    def vocab_expand(self, value: str, max_extra: int = 6) -> list[str]:
        """Map a constraint value to canonical + synonym phrases; returns empty when nothing
            matches."""
        if not self.vocab:
            return []
        lowered = (value or "").strip().lower()
        if not lowered:
            return []
        dictionaries = self.vocab.get("dictionaries") or {}
        for _attr_type, entries in dictionaries.items():
            if not isinstance(entries, dict):
                continue
            for canonical, entry in entries.items():
                if not isinstance(entry, dict):
                    continue
                synonyms = entry.get("synonyms") or []
                if isinstance(synonyms, str):
                    synonyms = [synonyms]
                if lowered == str(canonical).lower() or lowered in {
                    str(s).lower() for s in synonyms
                }:
                    extra = [
                        str(s)
                        for s in synonyms
                        if str(s).lower() not in (lowered, str(canonical).lower())
                    ]
                    return (extra[:max_extra] + [str(canonical)])[: max_extra + 1]
        return []

    # ------------------------------------------------------------------
    # category: category phrase -> product-type tokens (family-alias matching)
    # ------------------------------------------------------------------
    def category_expand(self, category_phrase: str, max_extra: int = 6) -> list[str]:
        if not self.category_map:
            return []
        phrase = (category_phrase or "").lower()
        if not phrase:
            return []
        routing = self.category_map.get("routing") or {}
        family_aliases = routing.get("family_aliases") or {}
        result: list[str] = []
        for alias, canonical in family_aliases.items():
            alias_l = str(alias).lower()
            if alias_l and alias_l in phrase and canonical not in result:
                # canonical e.g. tank_tops -> tokens: tank, tops
                for part in str(canonical).split("_"):
                    if part and part not in result and part not in phrase:
                        result.append(part)
            if len(result) >= max_extra:
                break
        return result[:max_extra]

    # ------------------------------------------------------------------
    # paraphrase: review-paraphrase patterns -> (regex, attribute) list
    # ------------------------------------------------------------------
    def paraphrase_patterns(self) -> list[tuple[re.Pattern, str]]:
        if not self.paraphrases:
            return []
        patterns: list[tuple[re.Pattern, str]] = []
        ml = self.paraphrases.get("material_language") or {}
        for pattern in ml.get("context_patterns") or []:
            # "made of {material}" -> capture the value
            p = str(pattern).replace("{material}", r"([a-z][a-z0-9 %\-]{2,40})")
            patterns.append((re.compile(p, re.I), "material"))
        # size/fit language (intent signal -> soft size constraint)
        sf = self.paraphrases.get("size_fit") or {}
        for _key, entry in sf.items():
            for phrase in (entry or {}).get("phrases") or []:
                if len(phrase) < 3:
                    continue
                patterns.append((re.compile(re.escape(str(phrase)), re.I), "size"))
        # color aliases (grey -> gray etc.)
        cl = self.paraphrases.get("color_language") or {}
        for _base, aliases in (cl.get("literal_aliases") or {}).items():
            for alias in aliases:
                patterns.append((re.compile(re.escape(str(alias)), re.I), "color"))
        return patterns[:120]


_instances: dict[str, DataAssets] = {}


def load_assets(path: str | Path | None = None) -> DataAssets:
    """Load data assets; each directory is loaded once (in-process cache)."""
    directory = Path(path) if path is not None else ASSETS_DIR
    key = str(directory)
    if key in _instances:
        return _instances[key]
    loaded: list[str] = []
    vocab: dict = {}
    category_map: dict = {}
    paraphrases: dict = {}
    field_map: dict = {}
    for name, target in (
        (_VOCAB_FILE, "vocab"),
        (_CATEGORY_FILE, "category_map"),
        (_PARAPHRASE_FILE, "paraphrases"),
        (_FIELD_FILE, "field_map"),
    ):
        file_path = directory / name
        try:
            with file_path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                continue
            if target == "vocab":
                vocab = data
            elif target == "category_map":
                category_map = data
            elif target == "paraphrases":
                paraphrases = data
            else:
                field_map = data
            loaded.append(name)
        except (OSError, json.JSONDecodeError):
            logger.warning("[assets] missing or corrupt %s (skipped)", name)
    assets = DataAssets(
        vocab=vocab,
        category_map=category_map,
        paraphrases=paraphrases,
        field_map=field_map,
        loaded=tuple(loaded),
    )
    _instances[key] = assets
    return assets
