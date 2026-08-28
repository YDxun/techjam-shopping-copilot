"""构建 field_mapping.json：属性 → 去哪找（字段）+ 权重 + 过滤严格度（静态表）。

回答的问题："用户约束 {material: cotton} 时，去目录的哪些字段找 'cotton'？找不到算不算不满足？"

数据来源：
- data/analysis/vocab.json  各属性 canonical 词 + 同义词（已建，本脚本复用）
- data/analysis/stats.json  字段缺失率（price 缺 78.9%、description 空 47.8%…）
- data/catalog.jsonl        50k 商品（按字段抽取文本做覆盖统计）

方法（与建 vocab 的反推统计同源）：
对每个属性（material/color/size/style/use_case/category）取 vocab 的 canonical+同义词，
统计"约束词出现在 title / features / details.<key> / store / categories / description 的占比"。
用倒排索引（字段->token->商品集合）加速；短语走子串扫描。聚合后按
"约束确认真实存在（title/features 命中）后各字段命中率"给出 lookup_fields 权重，
details 权威键（Material/Color/Size…）命中即高置信。brand/budget/feature/other 无词表人工定稿。

产物：data/analysis/field_mapping.json（定稿）+ field_mapping_raw.json（中间统计）
用法：python scripts/build_field_mapping.py [--quick N]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:6.1f}s] {msg}", flush=True)


# 各属性在 details 字典里的权威键（命中即高置信）
DETAILS_KEY_HINTS = {
    "material": ("material", "fabric"),
    "color": ("color",),
    "size": ("size",),
    "style": ("style",),
    "brand": ("brand", "manufacturer"),
    "use_case": ("occasion", "sport", "activity", "recommended use"),
    "category": ("category", "department"),
}

FIELDS = ("title", "features", "store", "categories", "description")
TOKEN_RE = re.compile(r"[a-z0-9%]+")

# 非属性词黑名单：从 top 类目 token 反推（商品/类目名词，绝不可能作为属性值）。
# vocab 反向统计会混入这类词（如 material 出现 "shoes"/"women"、style 出现 "jewelry"），
# 统计 field_mapping 时剔除，避免污染"约束词在哪个字段出现"的占比。
NON_ATTRIBUTE_WORDS = frozenset({
    "shoes", "women", "men", "girls", "boys", "baby", "kids", "unisex",
    "jewelry", "jewellery", "earrings", "necklaces", "bracelets", "rings", "watches",
    "wallets", "clothing", "apparel", "accessories", "socks", "dresses", "shirts",
    "blouses", "tees", "tops", "pants", "jeans", "shorts", "jackets", "coats",
    "sandals", "boots", "sneakers", "costumes", "lingerie", "lounge", "sleep",
    "activewear", "novelty", "sets", "more", "t", "scarves", "gloves", "hats",
    "belts", "bags", "handbags", "purses", "umbrellas", "sunglasses", "suits",
    "skirts", "leggings", "swimwear", "underwear", "pajamas", "robes", "jumpsuits",
})


def _tokens(text: str) -> frozenset[str]:
    return frozenset(TOKEN_RE.findall(text))


def _single_token(term: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9%]+", term))


def _contains(text: str, term: str) -> bool:
    if not term:
        return False
    if _single_token(term):
        return re.search(rf"(?<![a-z0-9%]){re.escape(term)}(?![a-z0-9%])", text) is not None
    return term in text


def _join_lower(value) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{k} {v}" for k, v in value.items() if v not in (None, "")).lower()
    if isinstance(value, list):
        return " ".join(str(x) for x in value if x not in (None, "")).lower()
    return str(value).lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=str(ROOT / "data" / "catalog.jsonl"))
    ap.add_argument("--quick", type=int, default=0, help=">0 时只用前 N 个商品（调试）")
    args = ap.parse_args()

    vocab = json.loads((ROOT / "data" / "analysis" / "vocab.json").read_text(encoding="utf-8"))
    stats = json.loads((ROOT / "data" / "analysis" / "stats.json").read_text(encoding="utf-8"))

    # ---- 1) 构建倒排索引：field -> token -> set(asin_idx) ----
    idx: dict[str, dict[str, set[int]]] = {f: defaultdict(set) for f in FIELDS}
    idx["details"] = defaultdict(set)
    idx_tf = defaultdict(set)          # title+features 合并索引（约束检测）
    texts: dict[str, list[str]] = {f: [] for f in FIELDS}
    texts["details"] = []
    details_auth: list[dict[str, str]] = []   # 每商品权威键文本（小写）
    asins: list[str] = []

    with Path(args.catalog).open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            if args.quick and i >= args.quick:
                break
            p = json.loads(line)
            asins.append(str(p["parent_asin"]))
            ft = {}
            for f in FIELDS:
                t = _join_lower(p.get(f))
                ft[f] = t
                texts[f].append(t)
                for tok in _tokens(t):
                    idx[f][tok].add(i)
            det = _join_lower(p.get("details"))
            texts["details"].append(det)
            for tok in _tokens(det):
                idx["details"][tok].add(i)
            tft = ft["title"] + " " + ft["features"]
            for tok in _tokens(tft):
                idx_tf[tok].add(i)
            # 权威键文本
            auth: dict[str, str] = {}
            details = p.get("details") or {}
            if isinstance(details, dict):
                for k in details:
                    kl = k.lower()
                    if any(
                        h in kl for h in ("material", "fabric", "color", "size", "style", "brand")
                    ):
                        v = details[k]
                        if v is not None:
                            auth[kl] = str(v).lower()
            details_auth.append(auth)
    n = len(asins)
    log(f"catalog={n}")

    # ---- 2) 每属性每字段聚合命中率 ----
    attr_terms: dict[str, dict[str, list[str]]] = {}
    for attr in ("material", "color", "size", "style", "use_case", "category_product_type"):
        entries = vocab["dictionaries"].get(attr, {})
        terms = {}
        for canonical, ent in entries.items():
            if not isinstance(ent, dict):
                continue
            if canonical.lower() in NON_ATTRIBUTE_WORDS:
                continue  # canonical 本身就是商品词（如 material 的 "shoes"）
            syns = [
                s for s in ent.get("synonyms", [])
                if isinstance(s, str) and s and s not in NON_ATTRIBUTE_WORDS
            ]
            if syns:
                terms[str(canonical).lower()] = syns
        if terms:
            attr_terms[attr] = terms

    attr_agg: dict[str, dict] = {}
    for attr, terms in attr_terms.items():
        # 每字段: tf 命中商品集（title/features 任一同义词）
        tf_sets: dict[str, set[int]] = {}
        for canonical, syns in terms.items():
            single = [s for s in syns if _single_token(s)]
            phrases = [s for s in syns if not _single_token(s)]
            sset: set[int] = set()
            for s in single:
                sset |= idx_tf.get(s, set())
            if phrases:
                for i in range(n):
                    if any(
                        _contains(texts["title"][i] + " " + texts["features"][i], ph)
                        for ph in phrases
                    ):
                        sset.add(i)
            tf_sets[canonical] = sset
        # 聚合 tf 并集（所有 canonical）
        all_tf: set[int] = set()
        for sset in tf_sets.values():
            all_tf |= sset
        if not all_tf:
            attr_agg[attr] = {"note": "no data"}
            continue
        # 每字段命中率 = 该字段命中的 tf 商品数 / tf 商品数（加权按 canonical 计数）
        # 简化：按商品级并集算（一个商品多 canonical 只计一次）
        field_stats: dict[str, dict] = {}
        for f in FIELDS:
            fhit: set[int] = set()
            for syns in terms.values():
                single = [s for s in syns if _single_token(s)]
                phrases = [s for s in syns if not _single_token(s)]
                for s in single:
                    fhit |= idx[f].get(s, set())
                if phrases:
                    ftxt = texts[f]
                    for i in range(n):
                        if any(_contains(ftxt[i], ph) for ph in phrases):
                            fhit.add(i)
            inter = fhit & all_tf
            field_stats[f] = {
                "hit_rate": round(len(inter) / len(all_tf), 3) if all_tf else 0.0,
                "overall": round(len(fhit) / n, 4),
            }
        # details 权威键命中率
        dhit: set[int] = set()
        for i in range(n):
            auth = details_auth[i]
            for syns in terms.values():
                if any(_contains(auth.get(k, ""), s) for k in auth for s in syns):
                    dhit.add(i)
                    break
        attr_agg[attr] = {
            "tf_products": len(all_tf),
            "field_stats": field_stats,
            "details_authoritative_hit_rate": (
                round(len(dhit & all_tf) / len(all_tf), 3) if all_tf else 0.0
            ),
        }

    out = {
        "version": "1.0.0",
        "generated_by": "scripts/build_field_mapping.py",
        "generated_at": time.strftime("%Y-%m-%d"),
        "n_products": n,
        "price_missing_ratio": stats["missing_dirty"]["price_missing"]["ratio"],
        "description_empty_ratio": stats["missing_dirty"]["description_empty"]["ratio"],
        "attr_coverage": attr_agg,
    }
    out_path = ROOT / "data" / "analysis" / "field_mapping_raw.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"raw stats written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
