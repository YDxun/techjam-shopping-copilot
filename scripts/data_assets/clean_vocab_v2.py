import copy
import json
from pathlib import Path

# ============================================================
# 路径
# ============================================================

INPUT_PATH = Path("vocab_v2.json")

OUTPUT_PATH = Path("vocab_v2_clean.json")

REPORT_PATH = Path("vocab_cleanup_report.json")


# ============================================================
# 明确确认过的噪声
#
# 注意：
# 不是只要来自 public_set 就删除。
#
# 必须同时满足：
# 1. canonical 在对应黑名单
# 2. 来源仅 public_set
# 3. product_count == 0
# 4. 没有 metadata_structured 支持
# ============================================================

NOISE = {
    "material": {
        "colors",
        "quality",
        "convenient",
        "stretchy",
        "heathers",
        "solids",
        "lining",
        "weight",
        "5%polyester",
        "5%spandex",
        "keep",
        "cool",
        "material",
        "comfortable",
        "dry",
        "smooth",
        "light",
        "ultra",
        "solid",
        "other",
        "heather",
        "grey",
        "comfy",
        "breathable",
        "loafers",
        "women",
        "durable",
        "fabrics",
        "mens",
        "slip",
        "shoes",
        "waist",
        "design",
        "super",
        "socks",
        "thigh",
        "lightweight",
        "spring",
        "featuring",
        "band",
        "watch",
        "quick",
        "take",
    },
    "color": {
        "eva",
        "covered",
        "fashion",
        "nice",
        "colors",
        "different",
        "bracelet",
        "classic",
        "mardi",
        "tassel",
        "gras",
        "feet",
        "design",
        "suitable",
        "midsole",
        "available",
        "match",
        "women",
        "comfort",
        "quality",
        "stainless",
        "steel",
        "casual",
        "strong",
        "registered",
    },
    # Size 不做大规模黑名单清洗。
    #
    # 前面已经做了 context-aware matcher，
    # 所以这里保持保守。
    "size": set(),
    # Style 暂时不动。
    "style": set(),
    # Brand 不动。
    "brand": set(),
}


# ============================================================
# 工具
# ============================================================


def normalize(value):

    if value is None:
        return ""

    return " ".join(str(value).strip().lower().split())


def load_json(path):

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

        f.write("\n")


# ============================================================
# 判断来源
# ============================================================


def get_sources(info):

    if not isinstance(info, dict):
        return []

    sources = info.get("sources")

    # metadata 新增项使用 source
    if sources is None:
        sources = info.get("source")

    if sources is None:
        return []

    if isinstance(sources, str):
        return [sources]

    if isinstance(sources, list):
        return sources

    return []


# ============================================================
# 判断是否有 metadata support
# ============================================================


def has_metadata_support(info):

    if not isinstance(info, dict):
        return False

    sources = {normalize(x) for x in get_sources(info)}

    if "metadata_structured" in sources:
        return True

    support = info.get("metadata_support")

    if isinstance(support, dict):
        count = support.get("count")

        try:
            if int(count or 0) > 0:
                return True

        except (TypeError, ValueError):
            pass

    return False


# ============================================================
# product_count
# ============================================================


def get_product_count(info):

    if not isinstance(info, dict):
        return 0

    value = info.get("product_count", 0)

    try:
        return int(value or 0)

    except (TypeError, ValueError):
        return 0


# ============================================================
# 是否只有 public_set 来源
# ============================================================


def public_set_only(info):

    sources = {normalize(x) for x in get_sources(info) if normalize(x)}

    return sources == {"public_set"}


# ============================================================
# 是否允许删除
# ============================================================


def should_remove(attr, canonical, info):

    canonical_norm = normalize(canonical)

    # 1. 必须在明确黑名单
    if canonical_norm not in NOISE.get(attr, set()):
        return (False, "not_in_noise_list")

    # 2. metadata 支持则绝不删
    if has_metadata_support(info):
        return (False, "has_metadata_support")

    # 3. catalog 中真实出现过则不删
    if get_product_count(info) > 0:
        return (False, "has_catalog_product_support")

    # 4. 必须只有 public_set 来源
    if not public_set_only(info):
        return (False, "not_public_set_only")

    return (True, "confirmed_public_set_noise")


# ============================================================
# 读取
# ============================================================

print("读取 vocab_v2.json...")

original = load_json(INPUT_PATH)


cleaned = copy.deepcopy(original)


dictionaries = cleaned.get("dictionaries", {})


# ============================================================
# Report
# ============================================================

report = {
    "meta": {
        "source": INPUT_PATH.name,
        "output": OUTPUT_PATH.name,
        "policy": "explicit_noise_list_plus_provenance_guard",
        "rules": [
            ("Never delete merely because a term came from public_set."),
            ("Deletion requires explicit attribute-specific noise classification."),
            ("Metadata-supported terms are protected."),
            ("Catalog-supported terms are protected."),
            ("Only public_set-only unsupported terms may be removed."),
        ],
    },
    "removed": [],
    "protected": [],
    "not_found": [],
    "summary": {},
}


# ============================================================
# 清洗
# ============================================================

for attr, noise_terms in NOISE.items():
    dictionary = dictionaries.get(attr)

    if not isinstance(dictionary, dict):
        continue

    # canonical normalize -> 实际 key
    normalized_keys = {normalize(key): key for key in dictionary.keys()}

    for noise_term in sorted(noise_terms):
        actual_key = normalized_keys.get(normalize(noise_term))

        if actual_key is None:
            report["not_found"].append(
                {
                    "attribute": attr,
                    "canonical": noise_term,
                }
            )

            continue

        info = dictionary[actual_key]

        remove, reason = should_remove(attr, actual_key, info)

        if remove:
            report["removed"].append(
                {
                    "attribute": attr,
                    "canonical": actual_key,
                    "reason": reason,
                    "sources": get_sources(info),
                    "product_count": get_product_count(info),
                    "public_set_count": info.get("public_set_count", 0),
                }
            )

            del dictionary[actual_key]

        else:
            report["protected"].append(
                {
                    "attribute": attr,
                    "canonical": actual_key,
                    "reason": reason,
                    "sources": get_sources(info),
                    "product_count": get_product_count(info),
                    "metadata_support": info.get("metadata_support"),
                }
            )


# ============================================================
# 清洗后 synonym 冲突检查
#
# 同一个 attribute 内：
# synonym / canonical term 不应该映射多个 canonical。
# ============================================================

term_owners = {}

duplicate_terms = []


for attr, dictionary in dictionaries.items():
    if not isinstance(dictionary, dict):
        continue

    attr_index = {}

    for canonical, info in dictionary.items():
        terms = [canonical]

        if isinstance(info, dict):
            terms.extend(info.get("synonyms", []))

        for term in terms:
            term_norm = normalize(term)

            if not term_norm:
                continue

            previous = attr_index.get(term_norm)

            if previous is not None and previous != canonical:
                duplicate_terms.append(
                    {
                        "attribute": attr,
                        "term": term,
                        "normalized": term_norm,
                        "owner_1": previous,
                        "owner_2": canonical,
                    }
                )

            else:
                attr_index[term_norm] = canonical

    term_owners[attr] = attr_index


# ============================================================
# 更新 provenance
# ============================================================

cleaned.setdefault("meta", {})


cleaned["meta"]["cleanup"] = {
    "source": INPUT_PATH.name,
    "policy": "explicit_noise_list_plus_provenance_guard",
    "removed_count": len(report["removed"]),
    "duplicate_term_count": len(duplicate_terms),
}


# ============================================================
# 最终 summary
# ============================================================

removed_by_attr = {}


for row in report["removed"]:
    attr = row["attribute"]

    removed_by_attr[attr] = removed_by_attr.get(attr, 0) + 1


report["duplicate_terms_after_cleanup"] = duplicate_terms


report["summary"] = {
    "removed": len(report["removed"]),
    "protected": len(report["protected"]),
    "not_found": len(report["not_found"]),
    "removed_by_attribute": removed_by_attr,
    "duplicate_terms_after_cleanup": len(duplicate_terms),
}


# ============================================================
# 保存
# ============================================================

save_json(OUTPUT_PATH, cleaned)


save_json(REPORT_PATH, report)


# ============================================================
# Terminal
# ============================================================

print()
print("=" * 75)
print("VOCAB V2 CLEANUP COMPLETE")
print("=" * 75)


print(f"输入：{INPUT_PATH}")

print(f"输出：{OUTPUT_PATH}")

print(f"报告：{REPORT_PATH}")


print("\n清洗摘要：")

print(f"removed:   {report['summary']['removed']}")

print(f"protected: {report['summary']['protected']}")

print(f"not found: {report['summary']['not_found']}")

print(f"duplicate terms: {report['summary']['duplicate_terms_after_cleanup']}")


print("\n按属性删除：")

for attr, count in removed_by_attr.items():
    print(f"{attr:<15}{count:>5}")


# ============================================================
# 删除内容预览
# ============================================================

print("\nRemoved canonicals:")


for row in report["removed"]:
    print(f"{row['attribute']:<12}{row['canonical']}")


# ============================================================
# 冲突警告
# ============================================================

if duplicate_terms:
    print()
    print("WARNING: 仍存在 synonym/canonical ownership 冲突。")

    print("这些冲突没有被自动修改，请查看 vocab_cleanup_report.json。")


print()
print("原 vocab_v2.json 未修改。")
