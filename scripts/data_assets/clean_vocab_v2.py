import copy
import json
from pathlib import Path

# ============================================================
# paths
# ============================================================

INPUT_PATH = Path("vocab_v2.json")

OUTPUT_PATH = Path("vocab_v2_clean.json")

REPORT_PATH = Path("vocab_cleanup_report.json")


# ============================================================
# explicitly confirmed noise
#
# note:
# originating from the public set alone is not a reason to delete.
#
# all of the following must hold:
# 1. the canonical is on the matching blacklist
# 2. its only source is the public set
# 3. product_count == 0
# 4. no metadata_structured support
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
    # Size is not bulk-cleaned by blacklist.
    #
    # a context-aware matcher already ran,
    # so stay conservative here.
    "size": set(),
    # Style is left untouched for now.
    "style": set(),
    # Brand is left untouched.
    "brand": set(),
}


# ============================================================
# helpers
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
# determine the source
# ============================================================


def get_sources(info):

    if not isinstance(info, dict):
        return []

    sources = info.get("sources")

    # metadata-added terms carry a source
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
# whether metadata support exists
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
# whether the only source is the public set
# ============================================================


def public_set_only(info):

    sources = {normalize(x) for x in get_sources(info) if normalize(x)}

    return sources == {"public_set"}


# ============================================================
# whether deletion is allowed
# ============================================================


def should_remove(attr, canonical, info):

    canonical_norm = normalize(canonical)

    # 1. must be on an explicit blacklist
    if canonical_norm not in NOISE.get(attr, set()):
        return (False, "not_in_noise_list")

    # 2. never delete with metadata support
    if has_metadata_support(info):
        return (False, "has_metadata_support")

    # 3. never delete if it really appears in the catalog
    if get_product_count(info) > 0:
        return (False, "has_catalog_product_support")

    # 4. the only source must be the public set
    if not public_set_only(info):
        return (False, "not_public_set_only")

    return (True, "confirmed_public_set_noise")


# ============================================================
# load
# ============================================================

print("loading vocab_v2.json...")

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
# clean
# ============================================================

for attr, noise_terms in NOISE.items():
    dictionary = dictionaries.get(attr)

    if not isinstance(dictionary, dict):
        continue

    # canonical normalize -> actual key
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
# synonym-conflict check after cleaning
#
# within the same attribute:
# a synonym/canonical term must not map to multiple canonicals.
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
# update provenance
# ============================================================

cleaned.setdefault("meta", {})


cleaned["meta"]["cleanup"] = {
    "source": INPUT_PATH.name,
    "policy": "explicit_noise_list_plus_provenance_guard",
    "removed_count": len(report["removed"]),
    "duplicate_term_count": len(duplicate_terms),
}


# ============================================================
# final summary
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
# save
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


print(f"input: {INPUT_PATH}")

print(f"output: {OUTPUT_PATH}")

print(f"report: {REPORT_PATH}")


print("\ncleanup summary:")

print(f"removed:   {report['summary']['removed']}")

print(f"protected: {report['summary']['protected']}")

print(f"not found: {report['summary']['not_found']}")

print(f"duplicate terms: {report['summary']['duplicate_terms_after_cleanup']}")


print("\ndeleting by attribute:")

for attr, count in removed_by_attr.items():
    print(f"{attr:<15}{count:>5}")


# ============================================================
# deletion preview
# ============================================================

print("\nRemoved canonicals:")


for row in report["removed"]:
    print(f"{row['attribute']:<12}{row['canonical']}")


# ============================================================
# conflict warnings
# ============================================================

if duplicate_terms:
    print()
    print("WARNING: synonym/canonical ownership conflicts remain.")

    print("these conflicts were not auto-fixed; see vocab_cleanup_report.json.")


print()
print("the original vocab_v2.json was not modified.")
