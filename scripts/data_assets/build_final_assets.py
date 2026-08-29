import copy
import json
from pathlib import Path

# ============================================================
# 路径
# ============================================================

VOCAB_PATH = Path("vocab_metadata_v2.json")

REVIEW_STATS_PATH = Path("review_paraphrase_stats_full.json")

FIELD_MAPPING_PATH = Path("field_mapping.json")


OUTPUT_VOCAB = Path("vocab_v2.json")

OUTPUT_PARAPHRASES = Path("review_paraphrases.json")

OUTPUT_REPORT = Path("final_assets_report.json")


# ============================================================
# 基础工具
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
# 读取资产
# ============================================================

print("读取 vocab...")

vocab_source = load_json(VOCAB_PATH)


print("读取 full review stats...")

review_stats = load_json(REVIEW_STATS_PATH)


print("读取 field mapping...")

field_mapping = load_json(FIELD_MAPPING_PATH)


# ============================================================
# vocab_v2
#
# 原则：
#
# 1. vocab_metadata_v2 是基础
# 2. reviews 不自动创造大量 canonical
# 3. reviews 只加入“严格安全”的 literal synonym
# 4. fit intent 不写进 size synonym
# ============================================================

vocab_v2 = copy.deepcopy(vocab_source)


# ============================================================
# 安全 review synonym
#
# 这些属于真正的 literal lexical equivalence。
#
# 注意：
# runs small / size up 等绝对不放这里。
# ============================================================

SAFE_REVIEW_SYNONYMS = {
    "material": {
        "leather": [
            "real leather",
            "genuine leather",
        ],
        "faux leather": [
            "fake leather",
            "pleather",
        ],
        "synthetic leather": [
            "pu leather",
        ],
        "spandex": [
            "lycra",
        ],
        "polyvinyl chloride": [
            "pvc",
        ],
        "thermoplastic polyurethane": [
            "tpu",
        ],
        "ethylene vinyl acetate": [
            "eva",
        ],
        "microfiber": [
            "microfibre",
        ],
        "aluminum": [
            "aluminium",
        ],
    },
    "color": {
        "gray": [
            "grey",
        ],
        "multicolor": [
            "multi color",
            "multi-color",
            "multicolored",
            "multicoloured",
        ],
        "navy": [
            "navy blue",
        ],
        "rose gold": [
            "rose-gold",
        ],
    },
}


# ============================================================
# 建立 dictionary term index
# ============================================================


def build_term_index(dictionary):

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


# ============================================================
# synonym report
# ============================================================

synonym_added = []
synonym_existing = []
synonym_conflicts = []
synonym_missing_canonical = []


# ============================================================
# 安全加入 synonym
# ============================================================

for attr, mappings in SAFE_REVIEW_SYNONYMS.items():
    dictionary = vocab_v2.get("dictionaries", {}).get(attr, {})

    for requested_canonical, synonyms in mappings.items():
        # 每轮重建，避免刚添加的 synonym
        # 后续又被其他 canonical 抢走
        term_index = build_term_index(dictionary)

        canonical_norm = normalize(requested_canonical)

        target = term_index.get(canonical_norm)

        if target is None:
            synonym_missing_canonical.append(
                {
                    "attribute": attr,
                    "canonical": requested_canonical,
                    "synonyms": synonyms,
                }
            )

            continue

        info = dictionary[target]

        if not isinstance(info, dict):
            info = {"synonyms": []}

            dictionary[target] = info

        info.setdefault("synonyms", [])

        for synonym in synonyms:
            synonym_norm = normalize(synonym)

            term_index = build_term_index(dictionary)

            existing_owner = term_index.get(synonym_norm)

            # --------------------------------
            # 已经存在
            # --------------------------------

            if existing_owner == target:
                synonym_existing.append(
                    {
                        "attribute": attr,
                        "canonical": target,
                        "synonym": synonym,
                    }
                )

                continue

            # --------------------------------
            # 被其他 canonical 占用
            #
            # 不自动重写！
            # --------------------------------

            if existing_owner is not None and existing_owner != target:
                synonym_conflicts.append(
                    {
                        "attribute": attr,
                        "requested_canonical": target,
                        "synonym": synonym,
                        "existing_owner": existing_owner,
                    }
                )

                continue

            # --------------------------------
            # 添加
            # --------------------------------

            info["synonyms"].append(synonym)

            synonym_added.append(
                {
                    "attribute": attr,
                    "canonical": target,
                    "synonym": synonym,
                }
            )


# ============================================================
# vocab provenance
# ============================================================

vocab_v2.setdefault("meta", {})


vocab_v2["meta"]["final_v2"] = {
    "base": VOCAB_PATH.name,
    "review_source": REVIEW_STATS_PATH.name,
    "review_policy": (
        "Only conservative literal lexical "
        "equivalences are merged into vocab. "
        "Fit intents and fuzzy color expressions "
        "remain in review_paraphrases.json."
    ),
}


# ============================================================
# review_paraphrases.json
# ============================================================

review_paraphrases = {
    "meta": {
        "version": "1.0.0",
        "source": REVIEW_STATS_PATH.name,
        "source_vocab": VOCAB_PATH.name,
        "purpose": (
            "Natural-language query interpretation "
            "and paraphrase expansion derived from "
            "full Amazon review corpus."
        ),
        "important": (
            "These phrases are NOT all literal "
            "attribute synonyms. Intent expressions "
            "must be interpreted semantically."
        ),
        "total_reviews": review_stats["meta"]["total_reviews"],
        "verified_reviews": review_stats["meta"]["verified_reviews"],
        "verified_rate": review_stats["meta"]["verified_rate"],
    },
    "size_fit": {},
    "material_language": {},
    "color_language": {},
}


# ============================================================
# SIZE FIT
#
# 这是 reviews 最重要的产物。
# ============================================================

SIZE_FIT_QUERY_EXPANSIONS = {
    "true_to_size": [
        "true to size",
        "true-to-size",
        "tts",
        "fits true to size",
        "fit true to size",
        "fits as expected",
        "fit as expected",
        "sizing is accurate",
        "size is accurate",
    ],
    "runs_small": [
        "runs small",
        "run small",
        "runs a little small",
        "runs a bit small",
        "runs slightly small",
        "fits small",
        "on the small side",
        "on the smaller side",
        "smaller than expected",
        "smaller than usual",
    ],
    "runs_large": [
        "runs large",
        "run large",
        "runs big",
        "run big",
        "runs a little large",
        "runs a bit large",
        "fits large",
        "fits big",
        "on the large side",
        "on the larger side",
        "larger than expected",
        "bigger than expected",
    ],
    "size_up": [
        "size up",
        "sized up",
        "sizing up",
        "go a size up",
        "go one size up",
        "order a size up",
        "order one size up",
        "one size larger",
    ],
    "size_down": [
        "size down",
        "sized down",
        "sizing down",
        "go a size down",
        "go one size down",
        "order a size down",
        "order one size down",
        "one size smaller",
    ],
    "tight_or_snug": [
        "too tight",
        "a little tight",
        "slightly tight",
        "too snug",
        "a little snug",
        "slightly snug",
    ],
    "loose": [
        "too loose",
        "a little loose",
        "slightly loose",
        "loose fit",
    ],
}


size_stats = review_stats.get("size_fit", {}).get("summary", {})


for intent, phrases in SIZE_FIT_QUERY_EXPANSIONS.items():
    stats = size_stats.get(intent, {})

    review_paraphrases["size_fit"][intent] = {
        "phrases": phrases,
        "review_count": stats.get("count", 0),
        "negated_nearby": stats.get("negated_nearby", 0),
        "negated_rate": stats.get("negated_rate", 0),
        "filter_semantics": "intent_only",
        "note": ("Do not convert this directly to literal size=S/M/L."),
    }


# ============================================================
# MATERIAL LANGUAGE
#
# 这里只保留 review 里真正有解释价值的表达。
# ============================================================

review_paraphrases["material_language"] = {
    "literal_aliases": {
        "leather": [
            "real leather",
            "genuine leather",
        ],
        "faux leather": [
            "fake leather",
            "pleather",
        ],
        "synthetic leather": [
            "pu leather",
        ],
        "spandex": [
            "lycra",
        ],
        "polyvinyl chloride": [
            "pvc",
        ],
        "thermoplastic polyurethane": [
            "tpu",
        ],
        "ethylene vinyl acetate": [
            "eva",
        ],
    },
    "context_patterns": [
        "made of {material}",
        "made out of {material}",
        "{material} material",
        "{material} fabric",
        "{material} blend",
    ],
    "soft_descriptors": {
        "synthetic_material": [
            "synthetic material",
            "synthetic fabric",
        ],
        "stretch_material": [
            "stretchy fabric",
            "stretchy material",
            "stretch fabric",
        ],
    },
    "rules": {
        "require_known_material_target": True,
        "respect_negation": True,
        "examples_of_negation": [
            "not leather",
            "not real leather",
            "doesn't feel like leather",
        ],
        "do_not_treat_as_material": [
            "comfortable",
            "quality",
            "good quality",
            "cheap material",
            "feels good",
            "felt",
        ],
    },
}


# ============================================================
# COLOR LANGUAGE
#
# fuzzy 修饰词只用于 soft expansion。
# ============================================================

COLOR_MODIFIERS = [
    "light",
    "dark",
    "deep",
    "bright",
    "pale",
    "dusty",
    "muted",
    "soft",
    "hot",
    "neon",
    "pastel",
]


COLOR_ISH_MODIFIERS = [
    "bluish",
    "greenish",
    "reddish",
    "pinkish",
    "brownish",
    "grayish",
    "greyish",
    "purplish",
    "yellowish",
]


review_paraphrases["color_language"] = {
    "literal_aliases": {
        "gray": [
            "grey",
        ],
        "navy": [
            "navy blue",
        ],
        "multicolor": [
            "multi color",
            "multi-color",
            "multicolored",
            "multicoloured",
        ],
        "rose gold": [
            "rose-gold",
        ],
    },
    "modifier_words": COLOR_MODIFIERS,
    "ish_modifier_words": COLOR_ISH_MODIFIERS,
    "normalization_examples": {
        "dark navy": "navy",
        "deep navy": "navy",
        "very deep navy": "navy",
        "light blue": "blue",
        "muted purple": "purple",
        "dusty pink": "pink",
    },
    "soft_only_examples": [
        "bluish black",
        "grayish pink",
        "brownish purple",
        "greenish blue",
    ],
    "rules": {
        "exact_modifier_plus_known_color": "normalize_to_base_color",
        "ish_color": "soft_expansion_only",
        "respect_negation": True,
        "do_not_hard_filter_fuzzy_color": True,
    },
}


# ============================================================
# final report
# ============================================================

report = {
    "meta": {
        "builder": "build_final_assets.py",
        "base_vocab": VOCAB_PATH.name,
        "review_stats": REVIEW_STATS_PATH.name,
        "field_mapping": FIELD_MAPPING_PATH.name,
        "outputs": [
            OUTPUT_VOCAB.name,
            OUTPUT_PARAPHRASES.name,
            OUTPUT_REPORT.name,
        ],
    },
    "review_corpus": {
        "total_reviews": review_stats["meta"]["total_reviews"],
        "bad_lines": review_stats["meta"]["bad_lines"],
        "verified_reviews": review_stats["meta"]["verified_reviews"],
        "verified_rate": review_stats["meta"]["verified_rate"],
    },
    "vocab_changes": {
        "new_review_synonyms": synonym_added,
        "already_present": synonym_existing,
        "conflicts_not_overwritten": synonym_conflicts,
        "missing_canonical": synonym_missing_canonical,
        "summary": {
            "added": len(synonym_added),
            "already_present": len(synonym_existing),
            "conflicts": len(synonym_conflicts),
            "missing_canonical": len(synonym_missing_canonical),
        },
    },
    "size_fit": {
        intent: {
            "review_count": review_paraphrases["size_fit"][intent]["review_count"],
            "negated_rate": review_paraphrases["size_fit"][intent]["negated_rate"],
        }
        for intent in (review_paraphrases["size_fit"])
    },
    "design_decisions": [
        ("Review-derived fit expressions are kept separate from literal size values."),
        (
            "Review material anchor frequency is "
            "not used as ground truth because words "
            "such as 'felt' are polysemous."
        ),
        (
            "Fuzzy color expressions are soft "
            "query-expansion signals rather than "
            "hard-filter equivalences."
        ),
        ("Existing synonym conflicts are reported but never silently overwritten."),
        ("field_mapping.json remains a separate retrieval routing asset."),
    ],
}


# ============================================================
# 保存
# ============================================================

save_json(OUTPUT_VOCAB, vocab_v2)

save_json(OUTPUT_PARAPHRASES, review_paraphrases)

save_json(OUTPUT_REPORT, report)


# ============================================================
# Terminal summary
# ============================================================

print()
print("=" * 75)
print("FINAL ASSETS COMPLETE")
print("=" * 75)

print(f"生成：{OUTPUT_VOCAB}")

print(f"生成：{OUTPUT_PARAPHRASES}")

print(f"生成：{OUTPUT_REPORT}")


print("\nReview corpus：")

print(f"reviews = {report['review_corpus']['total_reviews']:,}")

print(f"verified = {report['review_corpus']['verified_reviews']:,}")

print(f"bad lines = {report['review_corpus']['bad_lines']:,}")


print("\nVocab review synonyms：")

summary = report["vocab_changes"]["summary"]

print(f"added = {summary['added']}")

print(f"already present = {summary['already_present']}")

print(f"conflicts = {summary['conflicts']}")

print(f"missing canonical = {summary['missing_canonical']}")


print("\nSize/Fit：")

for intent, data in report["size_fit"].items():
    print(f"{intent:<20}{data['review_count']:>12,}")


print("\n完成。")
print("原 vocab_metadata_v2.json、field_mapping.json、review stats 均未修改。")
