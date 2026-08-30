import json
import re
import time
from collections import Counter
from pathlib import Path

# ============================================================
# config
# ============================================================

META_PATH = Path("meta_Clothing_Shoes_and_Jewelry.jsonl")
OUTPUT_PATH = Path("size_field_stats.json")

FIELDS = [
    "title",
    "features",
    "description",
    "details.Size",
]


# ============================================================
# counter
# ============================================================

field_nonempty = Counter()
field_hits = Counter()
pattern_hits = Counter()

total = 0
bad_lines = 0


# ============================================================
# text normalization
# ============================================================


def normalize(value):

    if value is None:
        return ""

    if isinstance(value, list):
        value = " ".join(str(x) for x in value if x is not None)

    value = str(value).lower()

    value = re.sub(r"\s+", " ", value)

    return value.strip()


# ============================================================
# 1. letter sizes
#
# e.g.:
# XS
# S
# M
# L
# XL
# XXL
# 2XL
# 3XL
# ============================================================

LETTER_SIZE = (
    r"(?:"
    r"xxxs|xxs|xs|s|m|l|xl|xxl|xxxl|"
    r"2xl|3xl|4xl|5xl|"
    r"2x|3x|4x|5x"
    r")"
)


# ------------------------------------------------------------
# with Size context
#
# Size: M
# Size XL
# Sizes: XXL
# ------------------------------------------------------------

LETTER_WITH_CONTEXT = re.compile(
    rf"\b(?:size|sizes)\s*[:=\-]?\s*"
    rf"({LETTER_SIZE})\b",
    re.IGNORECASE,
)


# ------------------------------------------------------------
# letter-size ranges
#
# S/M
# S/M/L
# S-3XL
# XS-4XL
# ------------------------------------------------------------

LETTER_RANGE = re.compile(
    rf"\b({LETTER_SIZE})"
    rf"\s*[/\-–—]\s*"
    rf"({LETTER_SIZE})\b",
    re.IGNORECASE,
)


# ============================================================
# 2. US sizes
#
# US 8
# US 8.5
# US Size 6
# ============================================================

US_SIZE = re.compile(
    r"\bUS\s*(?:SIZE\s*)?"
    r"[:=\-]?\s*"
    r"\d{1,2}(?:\.5)?\b",
    re.IGNORECASE,
)


# ============================================================
# 3. numeric sizes
#
# size 6
# size: 9
# shoe size 8.5
# waist 32
# inseam 30
#
# note:
# physical dimensions are filtered out later.
# ============================================================

NUMERIC_WITH_CONTEXT = re.compile(
    r"\b(?:"
    r"size|shoe size|waist|inseam"
    r")"
    r"\s*[:=\-]?\s*"
    r"\d{1,2}(?:\.5)?\b",
    re.IGNORECASE,
)


# ============================================================
# 4. physical-dimension exclusion
#
# used to exclude:
#
# Size: 12.5"
# Size: 17 inches
# Size: 9 x 6
# Size: 20 cm
#
# these are not wearable sizes.
# ============================================================

DIMENSION_AFTER = re.compile(
    r"""
    ^\s*
    (?:
        ["'″”]
        |
        inch(?:es)?
        |
        in\b
        |
        cm\b
        |
        mm\b
        |
        feet\b
        |
        ft\b
        |
        x\s*\d
        |
        ×\s*\d
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


DIMENSION_TEXT = re.compile(
    r"""
    (?:
        \d+(?:\.\d+)?\s*
        (?:inch(?:es)?|cm|mm|["'″”])
        |
        \d+(?:\.\d+)?\s*
        [x×]\s*
        \d+(?:\.\d+)?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def looks_like_dimension_after(text, match):

    # only inspect a short window after the matched word
    after = text[match.end() : match.end() + 25]

    return bool(DIMENSION_AFTER.search(after))


# ============================================================
# 5. general text Size matcher
#
# used for:
# title
# features
# description
# ============================================================


def match_text_size(text):

    # --------------------------------
    # letter size + size context
    # --------------------------------

    match = LETTER_WITH_CONTEXT.search(text)

    if match:
        return "letter_with_context"

    # --------------------------------
    # S/M/L or S-3XL
    # --------------------------------

    match = LETTER_RANGE.search(text)

    if match:
        return "letter_range"

    # --------------------------------
    # US 8 / US 8.5
    # --------------------------------

    match = US_SIZE.search(text)

    if match:
        return "us_size"

    # --------------------------------
    # size 6 / waist 32 etc.
    # --------------------------------

    match = NUMERIC_WITH_CONTEXT.search(text)

    if match:
        # exclude:
        #
        # Size: 12.5"
        # Size: 17 inch
        # Size: 9 x 6

        if looks_like_dimension_after(text, match):
            return None

        return "numeric_with_context"

    return None


# ============================================================
# 6. details.Size-specific matcher
#
# although details.Size is a structured field,
# manual inspection already found:
#
# 10 Watch Box
# 8.0 inches
# 9 x 6 Inches
#
# dirty values.
#
# so it cannot be trusted unconditionally.
# ============================================================


# ------------------------------------------------------------
# letter-type Size
# ------------------------------------------------------------

STRUCTURED_LETTER = re.compile(
    rf"\b{LETTER_SIZE}\b"
    rf"|"
    rf"\b(?:"
    rf"small|medium|large|"
    rf"x-small|x-large|"
    rf"extra small|extra large"
    rf")\b",
    re.IGNORECASE,
)


# ------------------------------------------------------------
# plain numeric
#
# 7
# 8
# 38
# ------------------------------------------------------------

STRUCTURED_NUMERIC_ONLY = re.compile(r"^\s*\d{1,2}(?:\.5)?\s*$", re.IGNORECASE)


# ------------------------------------------------------------
# numeric + single-letter width
#
# 8 M
# 9 W
# ------------------------------------------------------------

STRUCTURED_NUMERIC_WIDTH = re.compile(r"^\s*\d{1,2}(?:\.5)?\s*[a-z]?\s*$", re.IGNORECASE)


def match_structured_size(text):

    # --------------------------------
    # first exclude obvious dimensions
    # --------------------------------

    if DIMENSION_TEXT.search(text):
        return None

    lower = text.lower()

    # --------------------------------
    # exclude clearly non-wearable sizes
    # --------------------------------

    bad_words = [
        "inch",
        "inches",
        "cm",
        "mm",
        "watch box",
        "package",
        "dimension",
        "dimensions",
        "length",
        "height",
    ]

    if any(word in lower for word in bad_words):
        return None

    # --------------------------------
    # letter sizes
    # --------------------------------

    if STRUCTURED_LETTER.search(text):
        return "structured_letter_size"

    # --------------------------------
    # plain numeric sizes
    # --------------------------------

    if STRUCTURED_NUMERIC_ONLY.fullmatch(text):
        return "structured_numeric_size"

    # --------------------------------
    # numeric + width
    # --------------------------------

    if STRUCTURED_NUMERIC_WIDTH.fullmatch(text):
        return "structured_numeric_width"

    return None


# ============================================================
# 7. start the full scan
# ============================================================

print("=" * 75)
print("SIZE FULL CATALOG SCAN V2")
print("=" * 75)

print(f"input: {META_PATH}")

print(f"output: {OUTPUT_PATH}")

print()


start_time = time.time()


with META_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        # --------------------------------
        # JSON parsing
        # --------------------------------

        try:
            item = json.loads(line)

        except json.JSONDecodeError:
            bad_lines += 1
            continue

        total += 1

        # --------------------------------
        # details
        # --------------------------------

        details = item.get("details")

        if not isinstance(details, dict):
            details = {}

        # --------------------------------
        # fields to inspect for the current product
        # --------------------------------

        values = {
            "title": item.get("title"),
            "features": item.get("features"),
            "description": item.get("description"),
            "details.Size": details.get("Size"),
        }

        # --------------------------------
        # check each field once
        # --------------------------------

        for field_name in FIELDS:
            text = normalize(values[field_name])

            if not text:
                continue

            # field non-empty
            field_nonempty[field_name] += 1

            # --------------------------------
            # details.Size uses dedicated rules
            # --------------------------------

            if field_name == "details.Size":
                match_type = match_structured_size(text)

            # --------------------------------
            # title/features/description
            # --------------------------------

            else:
                match_type = match_text_size(text)

            # --------------------------------
            # matched
            # --------------------------------

            if match_type:
                field_hits[field_name] += 1

                pattern_hits[match_type] += 1

        # ====================================================
        # progress
        # ====================================================

        if total % 100000 == 0:
            elapsed = time.time() - start_time

            speed = total / elapsed if elapsed > 0 else 0

            print(
                f"\rscanned {total:,} products | {speed:,.0f} items/s | errors {bad_lines:,}",
                end="",
                flush=True,
            )


# ============================================================
# 8. scan complete
# ============================================================

elapsed = time.time() - start_time


# ============================================================
# 9. build the JSON output
# ============================================================

output = {
    "meta": {
        "source": META_PATH.name,
        "total_products": total,
        "bad_lines": bad_lines,
        "matcher": "size_context_matcher_v2",
        "elapsed_seconds": round(elapsed, 2),
        "notes": [
            ("Size uses a dedicated context-aware matcher."),
            ("Generic single-letter substring matching is not used."),
            ("Physical dimensions such as inches/cm/mm and LxW patterns are filtered."),
            (
                "details.Size is also cleaned "
                "because the source field "
                "contains non-wearable dimensions."
            ),
        ],
    },
    "fields": {},
    "match_types": dict(pattern_hits),
}


# ============================================================
# 10. field statistics
# ============================================================

for field_name in FIELDS:
    nonempty = field_nonempty[field_name]

    hits = field_hits[field_name]

    nonempty_coverage = nonempty / total if total else 0

    global_match_coverage = hits / total if total else 0

    match_rate_when_nonempty = hits / nonempty if nonempty else 0

    output["fields"][field_name] = {
        "nonempty_count": nonempty,
        "nonempty_coverage": round(nonempty_coverage, 6),
        "matched_products": hits,
        "global_match_coverage": round(global_match_coverage, 6),
        "match_rate_when_nonempty": round(match_rate_when_nonempty, 6),
    }


# ============================================================
# 11. save the JSON
# ============================================================

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)


# ============================================================
# 12. terminal final results
# ============================================================

print("\n\n")
print("=" * 75)
print("SIZE FULL STATS COMPLETE")
print("=" * 75)

print(f"total products: {total:,}")

print(f"bad lines: {bad_lines:,}")

print(f"elapsed: {elapsed / 60:.2f} minutes")

print(f"output: {OUTPUT_PATH}")


print("\nfield results:")


for field_name in FIELDS:
    data = output["fields"][field_name]

    print(
        f"{field_name:<25}"
        f"nonempty="
        f"{data['nonempty_count']:>10,} "
        f"({data['nonempty_coverage']:>7.2%})   "
        f"hit="
        f"{data['matched_products']:>10,} "
        f"({data['global_match_coverage']:>7.2%})   "
        f"when_nonempty="
        f"{data['match_rate_when_nonempty']:>7.2%}"
    )


print("\nmatch types:")


for name, count in pattern_hits.most_common():
    print(f"{name:<32}{count:>12,}")


print("\ndone.")
