import gzip
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

# ============================================================
# config
# ============================================================

REVIEWS_PATH = Path("Clothing_Shoes_and_Jewelry.jsonl.gz")

VOCAB_PATH = Path("vocab_metadata_v2.json")

OUTPUT_PATH = Path("review_context_candidates_sample.json")

# round 1 processes only 1M rows
MAX_REVIEWS = 1_000_000

# how many chars to keep on each side of an anchor
CONTEXT_WINDOW = 100

# max real examples saved per anchor
MAX_EXAMPLES_PER_ANCHOR = 20

# anchors that are too short over-match easily
MIN_TERM_LENGTH = 3


# ============================================================
# only these attributes are processed
# ============================================================

TARGET_ATTRIBUTES = [
    "material",
    "color",
    "size",
]


# ============================================================
# Size-specific allowed words
#
# s / m / l are single-letter tokens,
# but they are allowed only with size context.
# ============================================================

SIZE_SHORT_TERMS = {
    "xs",
    "xxs",
    "xl",
    "xxl",
    "2xl",
    "3xl",
    "4xl",
    "5x",
}


# ============================================================
# helpers
# ============================================================


def normalize(value):

    if value is None:
        return ""

    value = str(value).lower()

    value = value.replace("’", "'")

    value = re.sub(r"\s+", " ", value)

    return value.strip()


# ============================================================
# load the vocab
# ============================================================

print("loading the vocab...")

with VOCAB_PATH.open("r", encoding="utf-8") as f:
    vocab = json.load(f)


dictionaries = vocab.get("dictionaries", {})


# ============================================================
# build anchor -> canonical
# ============================================================

anchor_map = {}

attribute_anchor_counts = Counter()


for attr in TARGET_ATTRIBUTES:
    dictionary = dictionaries.get(attr, {})

    for canonical, info in dictionary.items():
        terms = {normalize(canonical)}

        if isinstance(info, dict):
            for synonym in info.get("synonyms", []):
                terms.add(normalize(synonym))

        for term in terms:
            if not term:
                continue

            # --------------------------------
            # normal words need at least 3 chars
            # --------------------------------

            if len(term) < MIN_TERM_LENGTH:
                continue

            # --------------------------------
            # short size words are allowed separately
            # --------------------------------

            if attr == "size" and term in SIZE_SHORT_TERMS:
                pass

            anchor_map.setdefault(term, []).append(
                {
                    "attribute": attr,
                    "canonical": canonical,
                }
            )

            attribute_anchor_counts[attr] += 1


print("\nanchor count:")

for attr in TARGET_ATTRIBUTES:
    print(f"{attr:<12}{attribute_anchor_counts[attr]:>8,}")


print(f"\ntotal anchors: {len(anchor_map):,}")


# ============================================================
# build the regex
#
# longer terms first, avoiding:
#
# stainless steel
# steel
#
# when multiple match, the longer expression wins.
# ============================================================

anchors = sorted(anchor_map.keys(), key=len, reverse=True)


escaped = [re.escape(x) for x in anchors]


ANCHOR_PATTERN = re.compile(r"(?<![a-z0-9])(" + "|".join(escaped) + r")(?![a-z0-9])", re.IGNORECASE)


# ============================================================
# negation / context words
#
# phase 1 does not delete anything;
# it only tags.
# ============================================================

NEGATION_PATTERN = re.compile(
    r"\b(?:"
    r"not|no|never|isn't|isnt|"
    r"wasn't|wasnt|without|"
    r"doesn't|doesnt|didn't|didnt"
    r")\b",
    re.IGNORECASE,
)


SIZE_CONTEXT_PATTERN = re.compile(
    r"\b(?:"
    r"size|sized|sizing|fit|fits|"
    r"fitting|runs|true to size|"
    r"too small|too large|"
    r"size up|size down|"
    r"sized up|sized down"
    r")\b",
    re.IGNORECASE,
)


# ============================================================
# counters
# ============================================================

total_reviews = 0
bad_lines = 0

reviews_with_anchor = 0
total_anchor_hits = 0

attribute_hits = Counter()

canonical_hits = {attr: Counter() for attr in TARGET_ATTRIBUTES}

anchor_hits = Counter()

negated_hits = Counter()
size_context_hits = Counter()


# ============================================================
# save context samples
# ============================================================

examples = defaultdict(list)


def add_example(key, record):

    bucket = examples[key]

    if len(bucket) < MAX_EXAMPLES_PER_ANCHOR:
        bucket.append(record)


# ============================================================
# start the scan
# ============================================================

print("\nscanning reviews...")
print(f"this-round limit: {MAX_REVIEWS:,}")

start = time.time()


with gzip.open(REVIEWS_PATH, "rt", encoding="utf-8") as f:
    for line in f:
        if MAX_REVIEWS is not None and total_reviews >= MAX_REVIEWS:
            break

        try:
            review = json.loads(line)

        except json.JSONDecodeError:
            bad_lines += 1
            continue

        total_reviews += 1

        # ====================================================
        # title + text
        # ====================================================

        title = normalize(review.get("title"))

        text = normalize(review.get("text"))

        if not title and not text:
            continue

        full_text = (title + ". " + text).strip()

        # ====================================================
        # find the anchor
        # ====================================================

        matches = list(ANCHOR_PATTERN.finditer(full_text))

        if not matches:
            continue

        reviews_with_anchor += 1

        # avoid fully duplicate counts within the same review
        seen_in_review = set()

        for match in matches:
            anchor = normalize(match.group(1))

            mappings = anchor_map.get(anchor, [])

            if not mappings:
                continue

            # --------------------------------
            # context
            # --------------------------------

            left = max(0, match.start() - CONTEXT_WINDOW)

            right = min(len(full_text), match.end() + CONTEXT_WINDOW)

            context = full_text[left:right]

            is_negated = bool(NEGATION_PATTERN.search(context))

            has_size_context = bool(SIZE_CONTEXT_PATTERN.search(context))

            for mapping in mappings:
                attr = mapping["attribute"]

                canonical = mapping["canonical"]

                unique_key = (attr, canonical, anchor)

                if unique_key in seen_in_review:
                    continue

                seen_in_review.add(unique_key)

                total_anchor_hits += 1

                attribute_hits[attr] += 1

                canonical_hits[attr][canonical] += 1

                anchor_hits[(attr, canonical, anchor)] += 1

                if is_negated:
                    negated_hits[(attr, canonical, anchor)] += 1

                if attr == "size" and has_size_context:
                    size_context_hits[(canonical, anchor)] += 1

                # --------------------------------
                # save real examples
                # --------------------------------

                example_key = (attr, canonical, anchor)

                add_example(
                    example_key,
                    {
                        "context": context,
                        "rating": review.get("rating"),
                        "helpful_vote": review.get("helpful_vote"),
                        "verified_purchase": review.get("verified_purchase"),
                        "asin": review.get("asin"),
                        "parent_asin": review.get("parent_asin"),
                        "negation_nearby": is_negated,
                        "size_context": has_size_context,
                    },
                )

        # ====================================================
        # progress
        # ====================================================

        if total_reviews % 100000 == 0:
            elapsed = time.time() - start

            speed = total_reviews / elapsed if elapsed > 0 else 0

            print(
                f"\rscanned "
                f"{total_reviews:,} reviews | "
                f"{speed:,.0f}/s | "
                f"with anchor "
                f"{reviews_with_anchor:,} | "
                f"errors {bad_lines:,}",
                end="",
                flush=True,
            )


# ============================================================
# done
# ============================================================

elapsed = time.time() - start


# ============================================================
# build the output
# ============================================================

output = {
    "meta": {
        "source": REVIEWS_PATH.name,
        "source_vocab": VOCAB_PATH.name,
        "sample_reviews": total_reviews,
        "bad_lines": bad_lines,
        "elapsed_seconds": round(elapsed, 2),
        "method": "review_anchor_context_extraction_v1",
        "context_window_chars": CONTEXT_WINDOW,
        "note": (
            "This file contains review contexts "
            "around existing vocab anchors. "
            "It does not automatically add synonyms."
        ),
    },
    "summary": {
        "reviews_with_anchor": reviews_with_anchor,
        "review_anchor_rate": round(reviews_with_anchor / total_reviews if total_reviews else 0, 6),
        "total_anchor_hits": total_anchor_hits,
        "attribute_hits": dict(attribute_hits),
    },
    "attributes": {},
}


# ============================================================
# organize per attribute
# ============================================================

for attr in TARGET_ATTRIBUTES:
    canonical_rows = []

    for canonical, count in canonical_hits[attr].most_common():
        anchor_rows = []

        # find this canonical's anchors
        relevant = []

        for (a, c, anchor), anchor_count in anchor_hits.items():
            if a == attr and c == canonical:
                relevant.append((anchor, anchor_count))

        relevant.sort(key=lambda x: -x[1])

        for anchor, anchor_count in relevant:
            key = (attr, canonical, anchor)

            row = {
                "anchor": anchor,
                "count": anchor_count,
                "negated_count": negated_hits[key],
                "negated_rate": round(negated_hits[key] / anchor_count if anchor_count else 0, 6),
                "examples": examples[key],
            }

            if attr == "size":
                size_key = (canonical, anchor)

                row["size_context_count"] = size_context_hits[size_key]

                row["size_context_rate"] = round(
                    size_context_hits[size_key] / anchor_count if anchor_count else 0, 6
                )

            anchor_rows.append(row)

        canonical_rows.append(
            {
                "canonical": canonical,
                "count": count,
                "anchors": anchor_rows,
            }
        )

    output["attributes"][attr] = canonical_rows


# ============================================================
# save
# ============================================================

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

    f.write("\n")


# ============================================================
# terminal summary
# ============================================================

print("\n\n")
print("=" * 75)
print("REVIEW CONTEXT SAMPLE COMPLETE")
print("=" * 75)

print(f"scanned reviews: {total_reviews:,}")

print(f"errors: {bad_lines:,}")

print(f"reviews with vocab anchors: {reviews_with_anchor:,}")

print(f"anchor review rate：{reviews_with_anchor / total_reviews:.2%}")

print(f"total anchor hits: {total_anchor_hits:,}")

print(f"elapsed: {elapsed / 60:.2f} minutes")


print("\nattribute hits:")

for attr in TARGET_ATTRIBUTES:
    print(f"{attr:<12}{attribute_hits[attr]:>12,}")


print(f"\noutput: {OUTPUT_PATH}")

print("\nnote: no synonyms were auto-added to the vocab.")
