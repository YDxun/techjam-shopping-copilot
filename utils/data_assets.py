"""静态数据资产加载器（队友打包：数据优化资产）。

离线优先、缺失/损坏自动降级为"无扩展"：
- vocab_v2_clean.json     词汇表：canonical + synonyms（material/color/size/style/...）
- category_mapping.json   品类路由：audience/family 别名 -> canonical 商品类型
- review_paraphrases.json 评论改写：size_fit / material_language / color_language
- field_mapping.json      字段映射：属性 -> 检索字段/权重/匹配策略（预留）

用法：
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
    # vocab: 约束值 -> 同义词短语（用于 BM25 查询扩展）
    # ------------------------------------------------------------------
    def vocab_expand(self, value: str, max_extra: int = 6) -> list[str]:
        """把约束取值映射到 canonical + 同义词短语；无命中返回空。"""
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
    # category: 品类短语 -> 商品类型 token（family alias 匹配）
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
                # canonical 如 tank_tops -> token：tank, tops
                for part in str(canonical).split("_"):
                    if part and part not in result and part not in phrase:
                        result.append(part)
            if len(result) >= max_extra:
                break
        return result[:max_extra]

    # ------------------------------------------------------------------
    # paraphrase: 评论改写模式 -> (正则, 属性) 列表
    # ------------------------------------------------------------------
    def paraphrase_patterns(self) -> list[tuple[re.Pattern, str]]:
        if not self.paraphrases:
            return []
        patterns: list[tuple[re.Pattern, str]] = []
        ml = self.paraphrases.get("material_language") or {}
        for pattern in ml.get("context_patterns") or []:
            # "made of {material}" -> 捕获取值
            p = str(pattern).replace("{material}", r"([a-z][a-z0-9 %\-]{2,40})")
            patterns.append((re.compile(p, re.I), "material"))
        # 尺寸贴合语言（intent signal -> size 软约束）
        sf = self.paraphrases.get("size_fit") or {}
        for _key, entry in sf.items():
            for phrase in (entry or {}).get("phrases") or []:
                if len(phrase) < 3:
                    continue
                patterns.append((re.compile(re.escape(str(phrase)), re.I), "size"))
        # 颜色别名（grey -> gray 等）
        cl = self.paraphrases.get("color_language") or {}
        for _base, aliases in (cl.get("literal_aliases") or {}).items():
            for alias in aliases:
                patterns.append((re.compile(re.escape(str(alias)), re.I), "color"))
        return patterns[:120]


_instances: dict[str, DataAssets] = {}


def load_assets(path: str | Path | None = None) -> DataAssets:
    """加载数据资产；同目录只加载一次（进程内缓存）。"""
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
            logger.warning("[assets] 缺失或损坏 %s（跳过）", name)
    assets = DataAssets(
        vocab=vocab,
        category_map=category_map,
        paraphrases=paraphrases,
        field_map=field_map,
        loaded=tuple(loaded),
    )
    _instances[key] = assets
    return assets
