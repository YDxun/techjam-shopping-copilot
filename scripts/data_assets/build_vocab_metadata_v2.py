import copy
import json
from pathlib import Path

# ============================================================
# 配置
# ============================================================

VOCAB_PATH = Path("vocab(1).json")
REVIEW_PATH = Path("vocab_candidate_review.json")

OUTPUT_VOCAB = Path("vocab_metadata_v2.json")
OUTPUT_REPORT = Path("vocab_metadata_v2_report.json")


# ============================================================
# 这一版采用“保守白名单”
#
# 原则：
# 1. 已有 canonical 不删除
# 2. 只加入明确可靠的新 canonical
# 3. 只加入明确可靠的 synonym
# 4. REVIEW / REJECT 不自动写入
# 5. brand/style 本轮不扩
# ============================================================


# ============================================================
# MATERIAL
#
# 只选：
# - Clothing / Shoes / Jewelry 中有明确检索价值
# - metadata 有结构化证据
# - 语义明确
#
# 不因为频率高就全部加入。
# ============================================================

NEW_MATERIAL_CANONICALS = {
    "stainless steel",
    "synthetic",
    "plastic",
    "wood",
    "silicone",
    "sterling silver",
    "ceramic",
    "glass",
    "aluminum",
    "canvas",
    "brass",
    "copper",
    "neoprene",
    "resin",
    "polyvinyl chloride",
    "felt",
    "carbon fiber",
    "steel",
    "microfiber",
    "polycarbonate",
    "polypropylene",
    "latex",
    "titanium",
    "hemp",
    "cork",
    "cowhide",
    "flannel",
    "acetate",
    "polyethylene",
    "tungsten",
    "tungsten carbide",
    "polyurethane",
    "thermoplastic polyurethane",
    "ethylene vinyl acetate",
    "synthetic leather",
    "faux leather",
    "faux suede",
    "synthetic fiber",
    "oxford cloth",
    "ripstop",
    "burlap",
}


# ============================================================
# Material alias
#
# key   = canonical
# value = 要补进去的 synonyms
# ============================================================

MATERIAL_SYNONYMS = {
    "stainless steel": [
        "stainless-steel",
    ],
    "aluminum": [
        "aluminium",
    ],
    "polyvinyl chloride": [
        "pvc",
        "polyvinyl chloride (pvc)",
    ],
    "polyurethane": [
        "pu",
        "polyurethane (pu)",
    ],
    "thermoplastic polyurethane": [
        "tpu",
        "thermoplastic polyurethane (tpu)",
        "thermoplastic urethane",
    ],
    "ethylene vinyl acetate": [
        "eva",
        "ethylene vinyl acetate (eva)",
    ],
    "faux leather": [
        "pleather",
    ],
    "synthetic fiber": [
        "synthetic fibers",
        "synthetic-fiber",
    ],
    "polyester": [
        "polyster",
    ],
}


# ============================================================
# COLOR
#
# 新增的必须是明确颜色。
# ============================================================

NEW_COLOR_CANONICALS = {
    "rose gold",
    "clear",
    "coffee",
    "dark brown",
    "blonde",
    "copper",
    "bronze",
}


COLOR_SYNONYMS = {
    "multicolor": [
        "multicolored",
        "multicoloured",
        "multi-colored",
        "multi coloured",
    ],
    "gray": [
        "grey",
    ],
    "rose gold": [
        "rose-gold",
    ],
}


# ============================================================
# SIZE
#
# 不新增数字 size canonical。
# 这里只补可靠的字母尺码和 alias。
#
# review 文件已经证明：
# xx-large -> xxl
# 3x-large -> 3xl
# 4x-large -> 4xl
# 等表达是可靠的。
# ============================================================

NEW_SIZE_CANONICALS = {
    "xxs",
    "5x",
}


SIZE_SYNONYMS = {
    "xxs": [
        "xx-small",
    ],
    "xs": [
        "x-small",
        "extra small",
    ],
    "s": [
        "small",
    ],
    "m": [
        "medium",
    ],
    "l": [
        "large",
    ],
    "xl": [
        "x-large",
        "extra large",
    ],
    "xxl": [
        "xx-large",
        "2x-large",
    ],
    "3xl": [
        "xxx-large",
        "3x-large",
    ],
    "4xl": [
        "4x-large",
    ],
}


# ============================================================
# 工具函数
# ============================================================


def normalize(value):

    if value is None:
        return ""

    return " ".join(str(value).strip().lower().split())


def ensure_dictionary(vocab, attr):

    vocab.setdefault("dictionaries", {})

    vocab["dictionaries"].setdefault(attr, {})

    return vocab["dictionaries"][attr]


def build_term_index(dictionary):
    """
    term -> canonical

    canonical 自己以及所有 synonyms
    都进入 index。
    """

    index = {}

    for canonical, info in dictionary.items():
        index[normalize(canonical)] = canonical

        if not isinstance(info, dict):
            continue

        for synonym in info.get("synonyms", []):
            synonym_norm = normalize(synonym)

            if synonym_norm:
                index[synonym_norm] = canonical

    return index


def candidate_evidence(review_data, attr, canonical):
    """
    从 vocab_candidate_review.json 中
    找到某个 canonical 的 metadata 证据。

    只查 AUTO_ACCEPT。
    """

    rows = review_data.get("attributes", {}).get(attr, {}).get("AUTO_ACCEPT", [])

    canonical_norm = normalize(canonical)

    matches = []

    for row in rows:
        row_canonical = normalize(row.get("canonical"))

        row_normalized = normalize(row.get("normalized"))

        if row_canonical == canonical_norm or row_normalized == canonical_norm:
            matches.append(row)

    return matches


# ============================================================
# 添加 canonical
# ============================================================


def add_canonical(vocab, review_data, attr, canonical, report):

    dictionary = ensure_dictionary(vocab, attr)

    term_index = build_term_index(dictionary)

    canonical_norm = normalize(canonical)

    # --------------------------------------------------------
    # 已经存在
    # --------------------------------------------------------

    if canonical_norm in term_index:
        existing_canonical = term_index[canonical_norm]

        report["already_present"].append(
            {
                "attribute": attr,
                "requested": canonical,
                "existing_canonical": existing_canonical,
            }
        )

        return existing_canonical

    # --------------------------------------------------------
    # 新建
    # --------------------------------------------------------

    evidence = candidate_evidence(review_data, attr, canonical)

    # 新 canonical 必须在 AUTO_ACCEPT 有证据
    if not evidence:
        report["skipped_no_auto_accept_evidence"].append(
            {
                "attribute": attr,
                "canonical": canonical,
            }
        )

        return None

    total_count = sum(int(row.get("count") or 0) for row in evidence)

    dictionary[canonical] = {
        "synonyms": [],
        "source": ["metadata_structured"],
        "metadata_support": {
            "count": total_count,
            "evidence_rows": len(evidence),
        },
    }

    report["new_canonicals"].append(
        {
            "attribute": attr,
            "canonical": canonical,
            "metadata_count": total_count,
        }
    )

    return canonical


# ============================================================
# 添加 synonym
# ============================================================


def add_synonym(vocab, attr, canonical, synonym, report):

    dictionary = ensure_dictionary(vocab, attr)

    term_index = build_term_index(dictionary)

    canonical_norm = normalize(canonical)

    synonym_norm = normalize(synonym)

    if not synonym_norm:
        return

    # --------------------------------------------------------
    # 找 canonical
    # --------------------------------------------------------

    target = term_index.get(canonical_norm)

    if target is None:
        report["skipped_missing_canonical"].append(
            {
                "attribute": attr,
                "canonical": canonical,
                "synonym": synonym,
            }
        )

        return

    # --------------------------------------------------------
    # synonym 已经属于某个 canonical
    # --------------------------------------------------------

    existing_owner = term_index.get(synonym_norm)

    if existing_owner is not None:
        # 已经在正确 canonical 下
        if existing_owner == target:
            report["synonym_already_present"].append(
                {
                    "attribute": attr,
                    "canonical": target,
                    "synonym": synonym,
                }
            )

            return

        # 冲突：
        # synonym 已经属于另一个 canonical
        report["synonym_conflicts"].append(
            {
                "attribute": attr,
                "requested_canonical": target,
                "synonym": synonym,
                "existing_owner": existing_owner,
            }
        )

        return

    # --------------------------------------------------------
    # 确保结构
    # --------------------------------------------------------

    info = dictionary[target]

    if not isinstance(info, dict):
        info = {"synonyms": []}

        dictionary[target] = info

    info.setdefault("synonyms", [])

    # --------------------------------------------------------
    # 添加
    # --------------------------------------------------------

    info["synonyms"].append(synonym)

    report["new_synonyms"].append(
        {
            "attribute": attr,
            "canonical": target,
            "synonym": synonym,
        }
    )


# ============================================================
# 读取
# ============================================================

print("读取原 vocab...")

with VOCAB_PATH.open("r", encoding="utf-8") as f:
    original_vocab = json.load(f)


print("读取 metadata review...")

with REVIEW_PATH.open("r", encoding="utf-8") as f:
    review_data = json.load(f)


# ============================================================
# 深拷贝
#
# 永远不直接修改原始 vocab 对象。
# ============================================================

vocab_v2 = copy.deepcopy(original_vocab)


# ============================================================
# Report
# ============================================================

report = {
    "meta": {
        "source_vocab": VOCAB_PATH.name,
        "source_review": REVIEW_PATH.name,
        "output_vocab": OUTPUT_VOCAB.name,
        "policy": "conservative_metadata_whitelist",
        "brand_expansion": False,
        "style_expansion": False,
        "review_candidates_auto_added": False,
        "reject_candidates_added": False,
        "notes": [
            ("Original vocab is preserved."),
            ("Only whitelisted metadata-backed material/color/size changes are applied."),
            ("Brand is handled primarily through field_mapping store routing."),
            ("Style expansion is deferred because details.Style contains product-type noise."),
            ("Numeric size candidates are not automatically added."),
        ],
    },
    "new_canonicals": [],
    "new_synonyms": [],
    "already_present": [],
    "synonym_already_present": [],
    "synonym_conflicts": [],
    "skipped_no_auto_accept_evidence": [],
    "skipped_missing_canonical": [],
}


# ============================================================
# MATERIAL
# ============================================================

print("处理 material...")


for canonical in sorted(NEW_MATERIAL_CANONICALS):
    add_canonical(vocab_v2, review_data, "material", canonical, report)


for canonical, synonyms in MATERIAL_SYNONYMS.items():
    # 如果 canonical 不存在，
    # 先尝试按 metadata 证据建立
    dictionary = ensure_dictionary(vocab_v2, "material")

    index = build_term_index(dictionary)

    if normalize(canonical) not in index:
        add_canonical(vocab_v2, review_data, "material", canonical, report)

    for synonym in synonyms:
        add_synonym(vocab_v2, "material", canonical, synonym, report)


# ============================================================
# COLOR
# ============================================================

print("处理 color...")


for canonical in sorted(NEW_COLOR_CANONICALS):
    add_canonical(vocab_v2, review_data, "color", canonical, report)


for canonical, synonyms in COLOR_SYNONYMS.items():
    dictionary = ensure_dictionary(vocab_v2, "color")

    index = build_term_index(dictionary)

    if normalize(canonical) not in index:
        add_canonical(vocab_v2, review_data, "color", canonical, report)

    for synonym in synonyms:
        add_synonym(vocab_v2, "color", canonical, synonym, report)


# ============================================================
# SIZE
# ============================================================

print("处理 size...")


for canonical in sorted(NEW_SIZE_CANONICALS):
    add_canonical(vocab_v2, review_data, "size", canonical, report)


for canonical, synonyms in SIZE_SYNONYMS.items():
    dictionary = ensure_dictionary(vocab_v2, "size")

    index = build_term_index(dictionary)

    if normalize(canonical) not in index:
        add_canonical(vocab_v2, review_data, "size", canonical, report)

    for synonym in synonyms:
        add_synonym(vocab_v2, "size", canonical, synonym, report)


# ============================================================
# 给 vocab v2 添加 provenance
#
# 不覆盖已有 meta。
# ============================================================

vocab_v2.setdefault("meta", {})


vocab_v2["meta"]["metadata_v2"] = {
    "built_from": VOCAB_PATH.name,
    "metadata_review": REVIEW_PATH.name,
    "policy": "conservative_metadata_whitelist",
    "brand_expansion": False,
    "style_expansion": False,
    "numeric_size_auto_expansion": False,
}


# ============================================================
# 最终统计
# ============================================================

report["summary"] = {
    "new_canonicals": len(report["new_canonicals"]),
    "new_synonyms": len(report["new_synonyms"]),
    "already_present": len(report["already_present"]),
    "synonym_already_present": len(report["synonym_already_present"]),
    "synonym_conflicts": len(report["synonym_conflicts"]),
    "skipped_no_auto_accept_evidence": len(report["skipped_no_auto_accept_evidence"]),
    "skipped_missing_canonical": len(report["skipped_missing_canonical"]),
}


# ============================================================
# 保存 vocab
# ============================================================

with OUTPUT_VOCAB.open("w", encoding="utf-8") as f:
    json.dump(vocab_v2, f, ensure_ascii=False, indent=2)

    f.write("\n")


# ============================================================
# 保存 report
# ============================================================

with OUTPUT_REPORT.open("w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

    f.write("\n")


# ============================================================
# Terminal 输出
# ============================================================

print()
print("=" * 75)
print("VOCAB METADATA V2 COMPLETE")
print("=" * 75)

print(f"原始 vocab：{VOCAB_PATH}")

print(f"新 vocab：  {OUTPUT_VOCAB}")

print(f"变更报告：  {OUTPUT_REPORT}")


print("\n变更摘要：")

for key, value in report["summary"].items():
    print(f"{key:<38}{value:>6}")


print()
print("原 vocab 未修改。")
