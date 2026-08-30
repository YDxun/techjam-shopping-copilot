"""Build field_mapping.json: attribute -> where to look (fields) + weights + filter strictness
    (static table).

Question answered: "for a user constraint {material: cotton}, which catalog fields should be
searched for 'cotton'? If not found, is the constraint unmet?"

Data sources:
- data/analysis/vocab.json  per-attribute canonical words + synonyms (already built; reused here)
- data/analysis/stats.json  field missing rates (price missing 78.9%, description empty 47.8%...)
- data/catalog.jsonl        50k products (per-field text extraction for coverage stats)

Method (same source as the reverse stats used to build the vocab):
For each attribute (material/color/size/style/use_case/category), take the vocab's canonical +
synonyms,
and measure "the share of constraint words appearing in title / features / details.<key> / store /
categories / description".
Uses an inverted index (field -> token -> product set) for speed; phrases use substring scans. After
aggregation,
lookup-field weights come from "per-field hit rates after a constraint is confirmed to exist
(title/features hit)",
and an authoritative details key (Material/Color/Size...) hit is high-confidence.
brand/budget/feature/other have no vocab and are finalized manually.

Output: data/analysis/field_mapping.json (final) + field_mapping_raw.json (intermediate stats)
Usage: python scripts/build_field_mapping.py [--quick N]
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


# authoritative keys per attribute in the details dict (a hit is high-confidence)
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

# non-attribute word blacklist: derived from top category tokens (product/category nouns that can
# never be attribute values).
# reverse vocab stats can mix in such words (e.g. "shoes"/"women" under material, "jewelry" under
# style),
# so they are removed when building field_mapping to keep the per-field shares clean.
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
    ap.add_argument("--quick", type=int, default=0, help="when >0, use only the first N products (debug)")  # noqa: E501
    args = ap.parse_args()

    vocab = json.loads((ROOT / "data" / "analysis" / "vocab.json").read_text(encoding="utf-8"))
    stats = json.loads((ROOT / "data" / "analysis" / "stats.json").read_text(encoding="utf-8"))

    # ---- 1) build an inverted index: field -> token -> set(asin_idx) ----
    idx: dict[str, dict[str, set[int]]] = {f: defaultdict(set) for f in FIELDS}
    idx["details"] = defaultdict(set)
    idx_tf = defaultdict(set)          # merged title+features index (constraint detection)
    texts: dict[str, list[str]] = {f: [] for f in FIELDS}
    texts["details"] = []
    details_auth: list[dict[str, str]] = []   # per-product authoritative-key text (lowercase)
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
            # authoritative-key text
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

    # ---- 2) aggregate per-attribute per-field hit rates ----
    attr_terms: dict[str, dict[str, list[str]]] = {}
    for attr in ("material", "color", "size", "style", "use_case", "category_product_type"):
        entries = vocab["dictionaries"].get(attr, {})
        terms = {}
        for canonical, ent in entries.items():
            if not isinstance(ent, dict):
                continue
            if canonical.lower() in NON_ATTRIBUTE_WORDS:
                continue  # canonical is itself a product word (e.g. "shoes" under material)
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
        # per field: tf-hit product set (any synonym in title/features)
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
        # aggregate the tf union (all canonicals)
        all_tf: set[int] = set()
        for sset in tf_sets.values():
            all_tf |= sset
        if not all_tf:
            attr_agg[attr] = {"note": "no data"}
            continue
        # per-field hit rate = tf products hit in this field / tf products (weighted by canonical
        # count)
        # simplified: product-level union (a product with multiple canonicals counts once)
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
        # details authoritative-key hit rate
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
