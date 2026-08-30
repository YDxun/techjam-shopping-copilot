import copy
import json
from pathlib import Path

# ============================================================
# config
# ============================================================

VOCAB_PATH = Path("vocab(1).json")
REVIEW_PATH = Path("vocab_candidate_review.json")

OUTPUT_VOCAB = Path("vocab_metadata_v2.json")
OUTPUT_REPORT = Path("vocab_metadata_v2_report.json")


# ============================================================
# this version uses a conservative whitelist
#
# principles:
# 1. never delete existing canonicals
# 2. add only clearly reliable new canonicals
# 3. add only clearly reliable synonyms
# 4. REVIEW / REJECT are never auto-written
# 5. brand/style are not expanded this round
# ============================================================


# ============================================================
# MATERIAL
#
# only select:
# - clear retrieval value within Clothing / Shoes / Jewelry
# - structured evidence in the metadata
# - unambiguous semantics
#
# high frequency alone is not sufficient.
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
# value = synonyms to add
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
# additions must be unambiguous colors.
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
# never add numeric-size canonicals.
# only reliable letter sizes and aliases are added here.
#
# the review file has already proven that:
# xx-large -> xxl
# 3x-large -> 3xl
# 4x-large -> 4xl
# such expressions are reliable.
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
# helper functions
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

    the canonical itself and all its synonyms
    all enter the index.
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
    from vocab_candidate_review.json,
    find a canonical's metadata evidence.

    only AUTO_ACCEPT entries are consulted.
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
# add a canonical
# ============================================================


def add_canonical(vocab, review_data, attr, canonical, report):

    dictionary = ensure_dictionary(vocab, attr)

    term_index = build_term_index(dictionary)

    canonical_norm = normalize(canonical)

    # --------------------------------------------------------
    # already exists
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
    # new
    # --------------------------------------------------------

    evidence = candidate_evidence(review_data, attr, canonical)

    # a new canonical must have AUTO_ACCEPT evidence
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
# add a synonym
# ============================================================


def add_synonym(vocab, attr, canonical, synonym, report):

    dictionary = ensure_dictionary(vocab, attr)

    term_index = build_term_index(dictionary)

    canonical_norm = normalize(canonical)

    synonym_norm = normalize(synonym)

    if not synonym_norm:
        return

    # --------------------------------------------------------
    # find the canonical
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
    # the synonym already belongs to a canonical
    # --------------------------------------------------------

    existing_owner = term_index.get(synonym_norm)

    if existing_owner is not None:
        # already under the right canonical
        if existing_owner == target:
            report["synonym_already_present"].append(
                {
                    "attribute": attr,
                    "canonical": target,
                    "synonym": synonym,
                }
            )

            return

        # conflict:
        # the synonym already belongs to another canonical
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
    # ensure the structure
    # --------------------------------------------------------

    info = dictionary[target]

    if not isinstance(info, dict):
        info = {"synonyms": []}

        dictionary[target] = info

    info.setdefault("synonyms", [])

    # --------------------------------------------------------
    # add
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
# load
# ============================================================

print("loading the original vocab...")

with VOCAB_PATH.open("r", encoding="utf-8") as f:
    original_vocab = json.load(f)


print("loading the metadata review...")

with REVIEW_PATH.open("r", encoding="utf-8") as f:
    review_data = json.load(f)


# ============================================================
# deep copy
#
# never modify the original vocab object directly.
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

print("processing material...")


for canonical in sorted(NEW_MATERIAL_CANONICALS):
    add_canonical(vocab_v2, review_data, "material", canonical, report)


for canonical, synonyms in MATERIAL_SYNONYMS.items():
    # if the canonical does not exist,
    # first try to create it from metadata evidence
    dictionary = ensure_dictionary(vocab_v2, "material")

    index = build_term_index(dictionary)

    if normalize(canonical) not in index:
        add_canonical(vocab_v2, review_data, "material", canonical, report)

    for synonym in synonyms:
        add_synonym(vocab_v2, "material", canonical, synonym, report)


# ============================================================
# COLOR
# ============================================================

print("processing color...")


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

print("processing size...")


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
# add provenance to vocab v2
#
# do not overwrite existing meta.
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
# final statistics
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
# save the vocab
# ============================================================

with OUTPUT_VOCAB.open("w", encoding="utf-8") as f:
    json.dump(vocab_v2, f, ensure_ascii=False, indent=2)

    f.write("\n")


# ============================================================
# save the report
# ============================================================

with OUTPUT_REPORT.open("w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

    f.write("\n")


# ============================================================
# terminal output
# ============================================================

print()
print("=" * 75)
print("VOCAB METADATA V2 COMPLETE")
print("=" * 75)

print(f"original vocab: {VOCAB_PATH}")

print(f"new vocab:   {OUTPUT_VOCAB}")

print(f"change report: {OUTPUT_REPORT}")


print("\nchange summary:")

for key, value in report["summary"].items():
    print(f"{key:<38}{value:>6}")


print()
print("the original vocab was not modified.")
