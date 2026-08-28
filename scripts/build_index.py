"""TechJam2026 数据分析与字典构建（对齐 data_1.md 任务书第1阶段）。

只读官方 data/catalog.jsonl + data/public_set.jsonl，产出到 data/analysis/：
  stats.json                    数据速览（字段覆盖/缺失/脏数据/品类/评分/价格）
  vocab.json                    商品字典（种子 + 目录反推计数 + 公开集校准）
  public_set_constraints.json   公开集约束分布 + 问属性收益统计
  question_value_analysis.json  提问价值分析（每个问题能把候选缩小到多少件）
  report.md                     人类可读报告（含"先问什么"建议表）

用法：python scripts/build_index.py [--quick]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    classify_constraint,
    coarse_category,
    customer_reply,
    initial_message,
    intent_card,
    materialize_hidden_fields,
)

DATA_DIR = ROOT / "data"
CATALOG = DATA_DIR / "catalog.jsonl"
PUBLIC = DATA_DIR / "public_set.jsonl"
OUT = DATA_DIR / "analysis"
SEEDS = DATA_DIR / "seeds" / "seed_vocab.json"

_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time()-_T0:6.1f}s] {msg}")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_empty(v) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return not v.strip()
    if isinstance(v, (list, dict)):
        return len(v) == 0
    return False


def searchable_text(p: dict) -> str:
    parts = []
    for key in ("title", "features", "description", "categories", "details", "store"):
        v = p.get(key)
        if isinstance(v, dict):
            parts.append(" ".join(f"{k} {x}" for k, x in v.items() if x not in (None, "")))
        elif isinstance(v, list):
            parts.append(" ".join(str(x) for x in v if x not in (None, "")))
        elif v is not None:
            parts.append(str(v))
    return " ".join(parts).strip()


def tf_text(p: dict) -> str:
    parts = [str(p.get("title") or "")]
    feats = p.get("features") or []
    parts.append(" ".join(str(x) for x in feats if x not in (None, "")))
    return " ".join(parts).lower()


def norm_phrase(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


# ===========================================================================
# 词频计数器：单词按 token 集合（词边界），短语按子串（带缓存）
# ===========================================================================
class TermCounter:
    def __init__(self, tf_texts: list[str]) -> None:
        self._texts = tf_texts
        self._token_counts: dict[str, int] = collections.Counter()
        for t in tf_texts:
            self._token_counts.update(set(re.findall(r"[a-z0-9]+", t)))
        self._phrase_cache: dict[str, int] = {}

    def count(self, term: str) -> int:
        if re.fullmatch(r"[a-z0-9]+", term):
            return self._token_counts.get(term, 0)
        if term not in self._phrase_cache:
            self._phrase_cache[term] = sum(1 for t in self._texts if term in t)
        return self._phrase_cache[term]


# ===========================================================================
# Stage A｜数据速览
# ===========================================================================
def stage_stats(rows: list[dict]) -> dict:
    log("Stage A: 数据速览统计...")
    n = len(rows)
    keys = collections.Counter()
    for r in rows:
        for k in r:
            keys[k] += 1
    coverage = {k: round(v / n, 4) for k, v in keys.items()}

    missing = collections.Counter()
    price_vals: list[float] = []
    rating_avgs: list[float] = []
    rating_counts: list[int] = []
    store_counter = collections.Counter()
    dirty_store_examples = collections.Counter()
    dirty_store_set = {
        "null", "generic", "unknown", "(unknown)", "n/a", "none", "-", "na", "unknown-"
    }
    for p in rows:
        title = str(p.get("title") or "")
        if not title.strip():
            missing["title_empty"] += 1
        if title.strip().lower() in ("generic", "unknown", "n/a") or len(title.strip()) < 3:
            missing["title_short_or_generic"] += 1
        if is_empty(p.get("description")):
            missing["description_empty"] += 1
        if is_empty(p.get("features")):
            missing["features_empty"] += 1
        price = p.get("price")
        if is_empty(price):
            missing["price_missing"] += 1
        elif not isinstance(price, (int, float)) or isinstance(price, bool):
            missing["price_non_numeric"] += 1
        else:
            price_vals.append(float(price))
            if float(price) <= 0:
                missing["price_zero_or_neg"] += 1
        store = str(p.get("store") or "")
        store_l = store.strip().lower()
        if store_l in dirty_store_set:
            missing["store_generic"] += 1
            dirty_store_examples[store_l] += 1
        if not store_l or store_l in dirty_store_set or len(store_l) > 80:
            missing["store_dirty"] += 1
        store_counter[store.strip() or "(empty)"] += 1
        if is_empty(p.get("categories")):
            missing["categories_empty"] += 1
        if is_empty(p.get("details")):
            missing["details_empty"] += 1
        ra, rn = p.get("average_rating"), p.get("rating_number")
        if ra is None:
            missing["rating_missing"] += 1
        elif isinstance(ra, (int, float)) and not isinstance(ra, bool):
            rating_avgs.append(float(ra))
            if not (0 <= float(ra) <= 5):
                missing["rating_out_of_range"] += 1
        if isinstance(rn, (int, float)) and not isinstance(rn, bool):
            rating_counts.append(int(rn))
            if int(rn) == 0:
                missing["rating_number_zero"] += 1
        if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", tf_text(p)):
            missing["control_chars"] += 1

    top_level, second_level, full_path = (
        collections.Counter(),
        collections.Counter(),
        collections.Counter(),
    )
    for p in rows:
        cats = [str(c) for c in (p.get("categories") or [])]
        if cats:
            top_level[cats[0]] += 1
            if len(cats) > 1:
                second_level[cats[1]] += 1
            full_path[" > ".join(cats)] += 1

    rating_hist = collections.Counter()
    for v in rating_avgs:
        for bucket, lo, hi in (("<3.0", 0, 3.0), ("3.0-3.5", 3.0, 3.5), ("3.5-4.0", 3.5, 4.0),
                               ("4.0-4.5", 4.0, 4.5), ("4.5-5.0", 4.5, 5.0)):
            if lo <= v < hi:
                rating_hist[bucket] += 1
                break
        else:
            rating_hist["5.0"] += 1
    rating_num_buckets = {
        "0": sum(1 for v in rating_counts if v == 0),
        "1-10": sum(1 for v in rating_counts if 1 <= v <= 10),
        "11-100": sum(1 for v in rating_counts if 11 <= v <= 100),
        "101-1000": sum(1 for v in rating_counts if 101 <= v <= 1000),
        "1000+": sum(1 for v in rating_counts if v > 1000),
    }

    def q(vals: list, pct: float):
        if not vals:
            return None
        s = sorted(vals)
        return s[min(len(s) - 1, int(len(s) * pct))]

    return {
        "total_products": n,
        "field_coverage": coverage,
        "missing_dirty": {k: {"count": v, "ratio": round(v / n, 4)} for k, v in missing.items()},
        "price_stats": {
            "has_price_ratio": round(len(price_vals) / n, 4),
            "median": q(price_vals, 0.5), "p25": q(price_vals, 0.25), "p75": q(price_vals, 0.75),
            "min": q(price_vals, 0.0), "max": q(price_vals, 1.0),
        },
        "category_top_level": dict(top_level.most_common(15)),
        "category_second_level_top": dict(second_level.most_common(20)),
        "category_full_path_top": dict(full_path.most_common(25)),
        "rating_avg_hist": dict(rating_hist),
        "rating_number_buckets": rating_num_buckets,
        "rating_avg_median": q(rating_avgs, 0.5),
        "store_top_20": dict(store_counter.most_common(20)),
        "dirty_store_examples": dict(dirty_store_examples.most_common(10)),
        "description_empty_ratio": round(missing["description_empty"] / n, 4),
    }


# ===========================================================================
# Stage B｜商品字典 vocab.json
# ===========================================================================
_TOKEN_RE = re.compile(r"[a-z0-9%]+", re.IGNORECASE)
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "this", "that", "your", "our", "are",
    "was", "were", "has", "have", "not", "but", "you", "all", "any", "can",
    "get", "its", "made", "use", "used", "new", "one", "two", "size", "fits",
    "item", "also", "just", "into", "over", "under", "very", "well", "will",
    "more", "most", "some", "than", "then", "they", "them", "these", "those",
    "like", "look", "great", "good", "best", "high", "low", "set", "each",
    "day", "days", "wear", "wearing", "fit", "feature", "features",
})


def keep_public_term(term: str, ctype: str, freq: int) -> bool:
    if ctype == "feature":          # feature 约束是长短语，token 不入词典
        return False
    if len(term) < 3 or term in _STOPWORDS:
        return False
    if re.fullmatch(r"\d+(\.\d+)?%?", term):
        return False
    if freq < 2:
        return False
    return True


def stage_vocab(rows: list[dict], tf_texts: list[str], samples: list[dict],
                products: dict[str, dict]) -> dict:
    log("Stage B: 商品字典构建（种子 + 目录反推 + 公开集校准）...")
    seeds = json.loads(SEEDS.read_text(encoding="utf-8-sig"))
    tc = TermCounter(tf_texts)
    vocab: dict = {
        "meta": {
            "version": "1.0.0",
            "built_by": "scripts/build_index.py",
            "method": (
                "种子词表 + 目录 title/features 反推计数 + "
                "公开集约束校准（data_1.md 三步法）"
            ),
            "date": time.strftime("%Y-%m-%d"),
        },
        "dictionaries": {},
    }

    # --- 1) 种子同义词 → 目录计数（单词按词边界，短语按子串）---
    for attr, entries in seeds.items():
        if attr in ("meta",):
            continue
        attr_dict: dict = {}
        for canonical, synonyms in entries.items():
            hits = []
            for syn in synonyms:
                s = norm_phrase(syn)
                cnt = tc.count(s)
                if cnt:
                    hits.append({"term": s, "product_count": cnt})
            hits.sort(key=lambda x: x["product_count"], reverse=True)
            attr_dict[canonical] = {
                "canonical": canonical,
                "synonyms": [h["term"] for h in hits],
                "synonym_counts": hits[:10],
                "product_count": max((h["product_count"] for h in hits), default=0),
                "sources": ["seed", "catalog"],
            }
        vocab["dictionaries"][attr] = attr_dict

    # --- 2) 百分比材质组合反推（"67% Polyester, 33% Cotton" 类写法）---
    mat_tokens = set()
    for ent in seeds["material"].values():
        for syn in ent:
            mat_tokens.update(norm_phrase(syn).split())
    comp = collections.Counter()
    comp_re = re.compile(r"(\d{1,3})\s*%\s*([a-z][a-z ]{2,20})", re.I)
    for t in tf_texts:
        for m in comp_re.finditer(t):
            pct, name = int(m.group(1)), norm_phrase(m.group(2))
            if name in mat_tokens and pct >= 30:
                comp[(pct, name)] += 1
    vocab["composition_patterns_top"] = [
        {"pattern": f"{pct}% {name}", "product_count": cnt}
        for (pct, name), cnt in sorted(comp.items(), key=lambda x: -x[1])[:30]
    ]

    # --- 3) 公开集校准：收集模拟器实际披露约束，补漏进字典 ---
    missing_terms: dict[str, dict] = collections.defaultdict(dict)
    for s in samples:
        tgt = str(s["ground_truth"]["parent_asin"])
        card = intent_card(products[tgt])
        for v in [*card["hard_constraints"], *card["soft_preferences"]]:
            ctype = classify_constraint(v)
            for term in {x.lower() for x in _TOKEN_RE.findall(v) if len(x) > 1}:
                if not any(
                    term in (syn or "")
                    for syns in seeds.get(ctype, {}).values()
                    for syn in syns
                ):
                    info = missing_terms[ctype].setdefault(term, {"count": 0, "examples": []})
                    info["count"] += 1
                    if len(info["examples"]) < 3:
                        info["examples"].append(v[:80])
    for ctype, terms in missing_terms.items():
        for term, info in terms.items():
            if not keep_public_term(term, ctype, info["count"]):
                continue
            entry = vocab["dictionaries"].setdefault(ctype, {}).setdefault(term, {
                "canonical": term, "synonyms": [], "product_count": 0, "sources": []})
            entry["synonyms"] = list(dict.fromkeys([*entry["synonyms"], term]))
            entry["synonym_counts"] = [{"term": term, "count": info["count"]}]
            entry["sources"] = list(dict.fromkeys([*entry["sources"], "public_set"]))
            entry["public_set_count"] = info["count"]
            entry["public_set_examples"] = info["examples"]
    vocab["public_set_added_terms"] = {
        ctype: sum(1 for t, i in terms.items() if keep_public_term(t, ctype, i["count"]))
        for ctype, terms in missing_terms.items()
    }
    return vocab


# ===========================================================================
# Stage C｜公开集约束分布
# ===========================================================================
def stage_public_set(samples: list[dict], products: dict[str, dict]) -> dict:
    log("Stage C: 公开集约束分布...")
    by_scenario = collections.defaultdict(
        lambda: {"hard": collections.defaultdict(collections.Counter),
                 "soft": collections.defaultdict(collections.Counter)})
    for s in samples:
        sc = s["scenario_type"]
        card = intent_card(products[str(s["ground_truth"]["parent_asin"])])
        for pos, val in enumerate(card["hard_constraints"]):
            by_scenario[sc]["hard"][pos].update([classify_constraint(val)])
        for pos, val in enumerate(card["soft_preferences"]):
            by_scenario[sc]["soft"][pos].update([classify_constraint(val)])
    result = {}
    for sc in sorted(by_scenario):
        result[sc] = {
            "n": sum(sum(c.values()) for c in by_scenario[sc]["hard"].values()),
            "hard_pos_class": {str(k): dict(v) for k, v in by_scenario[sc]["hard"].items()},
            "soft_pos_class": {str(k): dict(v) for k, v in by_scenario[sc]["soft"].items()},
        }
    return result


# ===========================================================================
# Stage D｜提问价值分析
# ===========================================================================
_ALLOWED = (
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
)
_PREFIX_STRIP = re.compile(
    r"^\s*(material|color|size|style|feature|use[_-]?case|budget|brand|department)\s*[:：]\s*",
    re.I,
)


def norm_key(value: str) -> str:
    v = _PREFIX_STRIP.sub("", value or "").lower()
    v = re.sub(r"\s+", " ", v).strip(" -;,.\t\n")
    return v[:180]


def _count_match(
    texts: list[str], cat_texts: list[str], keys: list[str], cat_tokens: list[str]
) -> int:
    cnt = 0
    for i, t in enumerate(texts):
        if cat_tokens:
            frac = sum(1 for c in cat_tokens if c in cat_texts[i])
            if frac / len(cat_tokens) < 0.5:
                continue
        if all(k in t for k in keys):
            cnt += 1
    return cnt


def stage_question_value(samples: list[dict], products: dict[str, dict],
                         texts: list[str], cat_texts: list[str]) -> dict:
    log("Stage D: 提问价值分析（模拟每个问题能把候选缩小到多少件）...")
    asin_index = {a: i for i, a in enumerate(products)}
    rows = []
    for s in samples:
        sc = s["scenario_type"]
        tgt = str(s["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(s, products)
        eff = {**s, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        cat_tokens = [
            t.lower()
            for t in coarse_category(products[tgt].get("categories") or []).split()
            if len(t) > 1
        ]
        initial_message(eff, coarse_category(products[tgt].get("categories") or []), disclosed)
        base_keys = sorted({norm_key(v) for v in disclosed}) if disclosed else []
        tgt_idx = asin_index.get(tgt, -1)
        tgt_text = texts[tgt_idx] if tgt_idx >= 0 else ""

        baseline_pool = _count_match(texts, cat_texts, base_keys, cat_tokens)
        row = {
            "sample_id": s["sample_id"], "scenario": sc, "target": tgt,
            "base_pool": baseline_pool,
            "base_hit": (not base_keys) or all(k in tgt_text for k in base_keys),
            "asks": {},
        }
        for attr in _ALLOWED:
            d2 = set(disclosed)
            reply, _ = customer_reply(eff, attr, d2, False)
            new_keys = sorted({norm_key(v) for v in d2})
            if not new_keys or new_keys == base_keys:
                row["asks"][attr] = {"revealed": 0, "pool": baseline_pool,
                                     "hit": row["base_hit"],
                                     "boundary_reply": "don't have a preference" in reply.lower()}
                continue
            pool = _count_match(texts, cat_texts, new_keys, cat_tokens)
            hit = all(k in tgt_text for k in new_keys)
            row["asks"][attr] = {
                "revealed": len(new_keys) - len(base_keys),
                "pool": pool,
                "hit": hit,
            }
        rows.append(row)

    def agg(rs):
        n = len(rs)
        out = {"n": n}
        for attr in _ALLOWED:
            pools = [r["asks"][attr]["pool"] for r in rs]
            revealed = [r["asks"][attr]["revealed"] for r in rs]
            hits = [r["asks"][attr]["hit"] for r in rs]
            pool_sorted = sorted(pools)
            out[attr] = {
                "avg_revealed": round(sum(revealed) / n, 3),
                "avg_pool": round(sum(pools) / n, 1),
                "median_pool": pool_sorted[n // 2],
                "hit_potential": round(sum(hits) / n, 3),
                "shrink_vs_base": round((sum(r["base_pool"] for r in rs) - sum(pools)) / n, 1),
            }
        out["base_avg_pool"] = round(sum(r["base_pool"] for r in rs) / n, 1)
        return out

    overall = agg(rows)
    by_scenario = {sc: agg([r for r in rows if r["scenario"] == sc])
                   for sc in ("buying", "browsing", "intent_override", "boundary")}
    ranked = sorted(
        _ALLOWED,
        key=lambda a: (overall[a]["shrink_vs_base"], overall[a]["hit_potential"]),
        reverse=True,
    )
    overall["recommended_ask_order"] = ranked
    overall["recommendation"] = (
        f"优先问 {ranked[0]}（平均每轮把候选从 {overall['base_avg_pool']} 缩小到 "
        f"{overall[ranked[0]]['avg_pool']}，"
        f"命中保持 {overall[ranked[0]]['hit_potential']}）；其次 {ranked[1]} / {ranked[2]}。"
        "注：'other' 一次最多披露 2 条任意约束，信息量最大，通常最划算。"
    )
    return {"overall": overall, "by_scenario": by_scenario, "per_session": rows}


# ===========================================================================
# 报告
# ===========================================================================
def write_report(stats: dict, vocab: dict, pub: dict, qv: dict) -> str:
    m = stats["missing_dirty"]
    line = []
    line.append("# TechJam2026 数据盘点 + 商品字典 + 提问价值分析报告\n")
    line.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    line.append(
        f"- 数据源：`data/catalog.jsonl`（{stats['total_products']} 商品）"
        f"+ `data/public_set.jsonl`（200 会话）\n"
    )

    line.append("## 一、数据速览\n")
    line.append("### 字段覆盖率")
    line.append("| 字段 | 覆盖率 |")
    line.append("|---|---|")
    for k, v in stats["field_coverage"].items():
        line.append(f"| `{k}` | {v:.1%} |")
    line.append("\n### 缺失与脏数据")
    line.append("| 问题 | 数量 | 占比 |")
    line.append("|---|---|---|")
    for k, v in m.items():
        line.append(f"| {k} | {v['count']} | {v['ratio']:.2%} |")
    line.append("\n### 价格")
    ps = stats["price_stats"]
    line.append(
        f"- 有价格商品占比 **{ps['has_price_ratio']:.1%}**；中位数 ${ps['median']}，"
        f"P25 ${ps['p25']}，P75 ${ps['p75']}，区间 ${ps['min']}–${ps['max']}"
    )
    line.append("  → **budget 约束必须 lenient**：79% 商品无价格，不能用 budget 硬过滤\n")
    line.append("### 品类分布（Top 二级品类）")
    line.append("| 二级品类 | 数量 |")
    line.append("|---|---|")
    for k, v in list(stats["category_second_level_top"].items())[:20]:
        line.append(f"| {k} | {v} |")
    line.append("\n### 评分分布")
    line.append("| average_rating 区间 | 数量 |")
    line.append("|---|---|")
    for k, v in stats["rating_avg_hist"].items():
        line.append(f"| {k} | {v} |")
    line.append("\n| rating_number 区间 | 数量 |")
    line.append("|---|---|")
    for k, v in stats["rating_number_buckets"].items():
        line.append(f"| {k} | {v} |")

    line.append("\n## 二、商品字典 vocab.json\n")
    for attr, entries in vocab["dictionaries"].items():
        if attr in ("meta",):
            continue
        n_terms = len(entries)
        top_terms = sorted(entries.items(), key=lambda x: -x[1].get("product_count", 0))[:8]
        line.append(f"### {attr}（{n_terms} 个标准词）")
        line.append("| 标准词 | 同义词（带目录商品数） |")
        line.append("|---|---|")
        for canonical, ent in top_terms:
            syns = "、".join(
                f"{s['term']}({s.get('count', s.get('product_count', 0))})"
                for s in ent.get("synonym_counts", [])[:3]
            )
            line.append(f"| {canonical} | {syns} |")
        line.append("")
    if vocab.get("composition_patterns_top"):
        line.append("### 常见成分写法（百分比组合）")
        line.append("| 写法 | 商品数 |")
        line.append("|---|---|")
        for c in vocab["composition_patterns_top"][:15]:
            line.append(f"| {c['pattern']} | {c['product_count']} |")
        line.append("")

    line.append("## 三、提问价值分析\n")
    ov = qv["overall"]
    line.append("### 总体（200 会话）")
    line.append(f"- 不提问基线候选池均值：**{ov['base_avg_pool']}** 件")
    line.append(
        "\n| 问法 ask_attribute | 平均披露约束数 | 缩小后候选池(均值) | "
        "中位池 | 命中保持率 | 相对基线缩小 |"
    )
    line.append("|---|---|---|---|---|---|")
    for attr in ov["recommended_ask_order"]:
        a = ov[attr]
        line.append(
            f"| {attr} | {a['avg_revealed']} | {a['avg_pool']} | {a['median_pool']} "
            f"| {a['hit_potential']:.1%} | {a['shrink_vs_base']} |"
        )
    line.append(f"\n### 先问什么建议\n{ov['recommendation']}\n")
    line.append("### 分场景")
    for sc, d in qv["by_scenario"].items():
        best = sorted(
            [a for a in d if isinstance(d[a], dict)],
            key=lambda a: -d[a]["shrink_vs_base"],
        )[:3]
        line.append(
            f"- **{sc}**（{d['n']} 会话）：先问 `{best[0]}`，"
            f"候选池均值 {d['base_avg_pool']} → {d[best[0]]['avg_pool']}；"
            f"次选 `{best[1]}` / `{best[2]}`"
        )
    return "\n".join(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="只用前 50 会话做提问价值分析（调试）")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(CATALOG)
    samples = load_jsonl(PUBLIC)
    if args.quick:
        samples = samples[:50]
    log(f"catalog={len(rows)} public={len(samples)}")
    products = {str(p["parent_asin"]): p for p in rows}
    texts = [searchable_text(p).lower() for p in rows]
    cat_texts = [" ".join(str(c) for c in (p.get("categories") or [])).lower() for p in rows]
    tf_texts = [tf_text(p).lower() for p in rows]

    stats = stage_stats(rows)
    (OUT / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log("stats.json 写入完成")

    vocab = stage_vocab(rows, tf_texts, samples, products)
    (OUT / "vocab.json").write_text(
        json.dumps(vocab, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log("vocab.json 写入完成")

    pub = stage_public_set(samples, products)
    (OUT / "public_set_constraints.json").write_text(
        json.dumps(pub, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log("public_set_constraints.json 写入完成")

    qv = stage_question_value(samples, products, texts, cat_texts)
    (OUT / "question_value_analysis.json").write_text(
        json.dumps(qv, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log("question_value_analysis 完成")

    report = write_report(stats, vocab, pub, qv)
    (OUT / "report.md").write_text(report, encoding="utf-8")
    log(f"全部产出写入 {OUT}")


if __name__ == "__main__":
    main()
