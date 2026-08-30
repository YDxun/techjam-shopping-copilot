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

OUTPUT_PATH = Path("review_paraphrase_stats_full.json")

# None = full
MAX_REVIEWS = None

# max real examples kept per expression
MAX_EXAMPLES = 20

# progress print interval
PROGRESS_EVERY = 500_000


# ============================================================
# helpers
# ============================================================


def normalize(value):

    if value is None:
        return ""

    value = str(value).lower()

    value = (
        value.replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
    )

    value = re.sub(r"<br\s*/?>", " ", value)

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
# trusted Material
#
# deliberately not using the whole old material vocab.
#
# excluded:
# comfortable
# quality
# shoes
# super
# other
# material
# words already proven to be noise by the 1M sample.
# ============================================================

TRUSTED_MATERIALS = {
    # textile
    "cotton",
    "polyester",
    "nylon",
    "wool",
    "silk",
    "linen",
    "spandex",
    "elastane",
    "rayon",
    "viscose",
    "acrylic",
    "cashmere",
    "fleece",
    "denim",
    "canvas",
    "velvet",
    "satin",
    "lace",
    "mesh",
    "jersey",
    "flannel",
    "felt",
    "neoprene",
    "hemp",
    # leather
    "leather",
    "synthetic leather",
    "faux leather",
    "faux suede",
    "suede",
    "cowhide",
    # synthetic / footwear
    "rubber",
    "plastic",
    "silicone",
    "polyurethane",
    "polyvinyl chloride",
    "thermoplastic polyurethane",
    "ethylene vinyl acetate",
    "synthetic",
    "synthetic fiber",
    "microfiber",
    # jewelry / accessories
    "stainless steel",
    "sterling silver",
    "silver",
    "gold",
    "brass",
    "copper",
    "titanium",
    "tungsten",
    "tungsten carbide",
    "aluminum",
    "steel",
    "ceramic",
    "glass",
    "resin",
    "wood",
    "carbon fiber",
}


# ============================================================
# Material alias
# ============================================================

MATERIAL_ALIASES = {
    "poly": "polyester",
    "spandex": "spandex",
    "lycra": "spandex",
    "elastane": "elastane",
    "viscose": "viscose",
    "fake leather": "faux leather",
    "faux leather": "faux leather",
    "vegan leather": "faux leather",
    "pleather": "faux leather",
    "synthetic leather": "synthetic leather",
    "genuine leather": "leather",
    "real leather": "leather",
    "pu leather": "synthetic leather",
    "pvc": "polyvinyl chloride",
    "tpu": "thermoplastic polyurethane",
    "eva": "ethylene vinyl acetate",
    "microfibre": "microfiber",
    "aluminium": "aluminum",
}


# ============================================================
# trusted Color
# ============================================================

TRUSTED_COLORS = {
    "black",
    "white",
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "gray",
    "grey",
    "navy",
    "navy blue",
    "teal",
    "turquoise",
    "beige",
    "tan",
    "khaki",
    "cream",
    "ivory",
    "burgundy",
    "maroon",
    "olive",
    "mint",
    "lavender",
    "violet",
    "coral",
    "peach",
    "gold",
    "silver",
    "rose gold",
    "bronze",
    "copper",
    "clear",
    "multicolor",
}


# ============================================================
# Color alias
# ============================================================

COLOR_ALIASES = {
    "grey": "gray",
    "navy blue": "navy",
    "multi color": "multicolor",
    "multi-color": "multicolor",
    "multicolored": "multicolor",
    "multicoloured": "multicolor",
    "rose-gold": "rose gold",
}


# ============================================================
# Size/Fit phrase patterns
#
# these are not synonyms of size=S/M/L.
#
# they belong to natural-language fit intent.
# ============================================================

SIZE_FIT_PATTERNS = {
    # --------------------------------------------------------
    # Runs small
    # --------------------------------------------------------
    "runs_small": [
        r"\bruns?\s+(?:a\s+)?(?:little\s+|bit\s+|slightly\s+)?small\b",
        r"\bruns?\s+(?:at\s+least\s+)?one\s+size\s+small\b",
        r"\bfits?\s+(?:a\s+)?(?:little\s+|bit\s+|slightly\s+)?small\b",
        r"\bon\s+the\s+small(?:er)?\s+side\b",
        r"\bsmaller\s+than\s+expected\b",
        r"\bsmaller\s+than\s+usual\b",
        r"\bfit\s+more\s+like\s+(?:a\s+)?smaller\s+size\b",
    ],
    # --------------------------------------------------------
    # Runs large
    # --------------------------------------------------------
    "runs_large": [
        r"\bruns?\s+(?:a\s+)?(?:little\s+|bit\s+|slightly\s+)?large\b",
        r"\bruns?\s+(?:a\s+)?(?:little\s+|bit\s+|slightly\s+)?big\b",
        r"\bfits?\s+(?:a\s+)?(?:little\s+|bit\s+|slightly\s+)?large\b",
        r"\bfits?\s+(?:a\s+)?(?:little\s+|bit\s+|slightly\s+)?big\b",
        r"\bon\s+the\s+large(?:r)?\s+side\b",
        r"\bbigger\s+than\s+expected\b",
        r"\blarger\s+than\s+expected\b",
        r"\blarger\s+than\s+usual\b",
    ],
    # --------------------------------------------------------
    # True to size
    # --------------------------------------------------------
    "true_to_size": [
        r"\btrue\s+to\s+size\b",
        r"\btrue-to-size\b",
        r"\btts\b",
        r"\bfits?\s+as\s+expected\b",
        r"\bfit\s+as\s+expected\b",
        r"\bsizing\s+is\s+accurate\b",
        r"\bsize\s+is\s+accurate\b",
    ],
    # --------------------------------------------------------
    # Size up
    # --------------------------------------------------------
    "size_up": [
        r"\bsize\s+up\b",
        r"\bsized\s+up\b",
        r"\bsizing\s+up\b",
        r"\border(?:ed)?\s+(?:a\s+)?size\s+up\b",
        r"\border(?:ed)?\s+one\s+size\s+up\b",
        r"\bgo\s+(?:a\s+)?size\s+up\b",
        r"\bgo\s+one\s+size\s+up\b",
        r"\bone\s+size\s+larger\b",
    ],
    # --------------------------------------------------------
    # Size down
    # --------------------------------------------------------
    "size_down": [
        r"\bsize\s+down\b",
        r"\bsized\s+down\b",
        r"\bsizing\s+down\b",
        r"\border(?:ed)?\s+(?:a\s+)?size\s+down\b",
        r"\border(?:ed)?\s+one\s+size\s+down\b",
        r"\bgo\s+(?:a\s+)?size\s+down\b",
        r"\bgo\s+one\s+size\s+down\b",
        r"\bone\s+size\s+smaller\b",
    ],
    # --------------------------------------------------------
    # Tight / snug
    # --------------------------------------------------------
    "tight_or_snug": [
        r"\btoo\s+tight\b",
        r"\ba\s+little\s+tight\b",
        r"\bslightly\s+tight\b",
        r"\btoo\s+snug\b",
        r"\ba\s+little\s+snug\b",
        r"\bslightly\s+snug\b",
    ],
    # --------------------------------------------------------
    # Loose
    # --------------------------------------------------------
    "loose": [
        r"\btoo\s+loose\b",
        r"\ba\s+little\s+loose\b",
        r"\bslightly\s+loose\b",
        r"\bloose\s+fit\b",
    ],
}


# ============================================================
# compile Size/Fit patterns
# ============================================================

COMPILED_SIZE_FIT = {}

for label, patterns in SIZE_FIT_PATTERNS.items():
    COMPILED_SIZE_FIT[label] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


# ============================================================
# Material regex
# ============================================================

material_terms = set(TRUSTED_MATERIALS) | set(MATERIAL_ALIASES.keys())


MATERIAL_PATTERN = re.compile(
    r"(?<![a-z0-9])("
    + "|".join(re.escape(x) for x in sorted(material_terms, key=len, reverse=True))
    + r")(?![a-z0-9])",
    re.IGNORECASE,
)


# ============================================================
# Color regex
# ============================================================

color_terms = set(TRUSTED_COLORS) | set(COLOR_ALIASES.keys())


COLOR_PATTERN = re.compile(
    r"(?<![a-z0-9])("
    + "|".join(re.escape(x) for x in sorted(color_terms, key=len, reverse=True))
    + r")(?![a-z0-9])",
    re.IGNORECASE,
)


# ============================================================
# Color modifier patterns
#
# used to discover:
#
# dark navy
# light blue
# muted purple
# dusty pink
# bluish green
# brownish purple
# ============================================================

COLOR_MODIFIER_PATTERN = re.compile(
    r"\b("
    r"(?:very\s+)?"
    r"(?:"
    r"light|dark|deep|bright|pale|"
    r"dusty|muted|soft|hot|"
    r"neon|pastel"
    r")"
    r"\s+"
    r"(?:"
    r"black|white|red|blue|green|"
    r"yellow|orange|purple|pink|"
    r"brown|gray|grey|navy|teal|"
    r"beige|tan|khaki|cream|ivory|"
    r"burgundy|maroon|olive|mint|"
    r"lavender|violet|coral|peach"
    r")"
    r")\b",
    re.IGNORECASE,
)


ISH_COLOR_PATTERN = re.compile(
    r"\b("
    r"(?:"
    r"bluish|greenish|reddish|"
    r"pinkish|brownish|grayish|"
    r"greyish|purplish|yellowish"
    r")"
    r"\s+"
    r"(?:"
    r"black|white|red|blue|green|"
    r"yellow|orange|purple|pink|"
    r"brown|gray|grey|navy|teal|"
    r"beige|tan|cream|ivory|"
    r"burgundy|maroon"
    r")"
    r")\b",
    re.IGNORECASE,
)


# ============================================================
# Material phrase patterns
#
# here we look for natural customer phrasing.
# ============================================================

MATERIAL_PHRASE_PATTERNS = {
    "made_of": re.compile(
        r"\bmade\s+(?:out\s+)?of\s+"
        r"([a-z][a-z -]{1,40})",
        re.IGNORECASE,
    ),
    "feels_like": re.compile(
        r"\bfeels?\s+like\s+"
        r"([a-z][a-z -]{1,40})",
        re.IGNORECASE,
    ),
    "looks_like": re.compile(
        r"\blooks?\s+like\s+"
        r"([a-z][a-z -]{1,40})",
        re.IGNORECASE,
    ),
}


# ============================================================
# negation detection
#
# note:
# only statistical tagging happens here,
# no reviews are deleted directly.
# ============================================================

NEGATION_PATTERN = re.compile(
    r"\b(?:"
    r"not|no|never|without|"
    r"isn't|isnt|"
    r"wasn't|wasnt|"
    r"aren't|arent|"
    r"weren't|werent|"
    r"doesn't|doesnt|"
    r"didn't|didnt"
    r")\b",
    re.IGNORECASE,
)


# ============================================================
# counters
# ============================================================

total_reviews = 0
bad_lines = 0

verified_reviews = 0

size_fit_counts = Counter()

material_counts = Counter()
material_alias_counts = Counter()

color_counts = Counter()
color_modifier_counts = Counter()

material_phrase_counts = {name: Counter() for name in MATERIAL_PHRASE_PATTERNS}

negated_counts = Counter()


# ============================================================
# Examples
# ============================================================

examples = defaultdict(list)


def add_example(key, context, review):

    bucket = examples[key]

    if len(bucket) >= MAX_EXAMPLES:
        return

    bucket.append(
        {
            "context": context[:500],
            "rating": review.get("rating"),
            "verified_purchase": review.get("verified_purchase"),
            "helpful_vote": review.get("helpful_vote"),
            "asin": review.get("asin"),
            "parent_asin": review.get("parent_asin"),
        }
    )


# ============================================================
# Context
# ============================================================


def get_context(text, start, end, window=100):

    left = max(0, start - window)

    right = min(len(text), end + window)

    return text[left:right]


# ============================================================
# start
# ============================================================

print("=" * 75)
print("FULL REVIEW PARAPHRASE MINING")
print("=" * 75)

print(f"input: {REVIEWS_PATH}")

print(f"output: {OUTPUT_PATH}")

print("mode: full" if MAX_REVIEWS is None else f"limit: {MAX_REVIEWS:,}")

print()


start_time = time.time()


# ============================================================
# scan
# ============================================================

with gzip.open(REVIEWS_PATH, "rt", encoding="utf-8") as f:
    for line in f:
        if MAX_REVIEWS is not None and total_reviews >= MAX_REVIEWS:
            break

        # ----------------------------------------------------
        # JSON
        # ----------------------------------------------------

        try:
            review = json.loads(line)

        except json.JSONDecodeError:
            bad_lines += 1
            continue

        total_reviews += 1

        if review.get("verified_purchase"):
            verified_reviews += 1

        # ----------------------------------------------------
        # title + text
        # ----------------------------------------------------

        title = normalize(review.get("title"))

        text = normalize(review.get("text"))

        if not title and not text:
            continue

        full_text = (title + ". " + text).strip()

        # ====================================================
        # 1. SIZE / FIT
        # ====================================================

        for label, patterns in COMPILED_SIZE_FIT.items():
            matched_this_review = False

            for pattern in patterns:
                match = pattern.search(full_text)

                if not match:
                    continue

                matched_this_review = True

                size_fit_counts[label] += 1

                context = get_context(full_text, match.start(), match.end())

                if NEGATION_PATTERN.search(context):
                    negated_counts[f"size_fit::{label}"] += 1

                add_example(f"size_fit::{label}", context, review)

                # same review, same label
                # counted only once
                break

        # ====================================================
        # 2. MATERIAL anchors
        # ====================================================

        seen_material = set()

        for match in MATERIAL_PATTERN.finditer(full_text):
            raw = normalize(match.group(1))

            canonical = MATERIAL_ALIASES.get(raw, raw)

            if canonical not in TRUSTED_MATERIALS:
                continue

            if canonical in seen_material:
                continue

            seen_material.add(canonical)

            material_counts[canonical] += 1

            if raw != canonical:
                material_alias_counts[(canonical, raw)] += 1

            context = get_context(full_text, match.start(), match.end())

            if NEGATION_PATTERN.search(context):
                negated_counts[f"material::{canonical}"] += 1

            add_example(f"material::{canonical}", context, review)

        # ====================================================
        # 3. MATERIAL natural phrases
        # ====================================================

        for phrase_type, pattern in MATERIAL_PHRASE_PATTERNS.items():
            for match in pattern.finditer(full_text):
                phrase = normalize(match.group(1))

                # skip too-long/too-short spans
                if len(phrase) < 2 or len(phrase) > 45:
                    continue

                # truncate common connectors
                phrase = re.split(
                    r"\b(?:"
                    r"and|but|because|"
                    r"which|that|with|"
                    r"when|while"
                    r")\b",
                    phrase,
                )[0].strip()

                if not phrase:
                    continue

                material_phrase_counts[phrase_type][phrase] += 1

                context = get_context(full_text, match.start(), match.end())

                add_example(f"material_phrase::{phrase_type}::{phrase}", context, review)

        # ====================================================
        # 4. COLOR anchors
        # ====================================================

        seen_color = set()

        for match in COLOR_PATTERN.finditer(full_text):
            raw = normalize(match.group(1))

            canonical = COLOR_ALIASES.get(raw, raw)

            if canonical == "grey":
                canonical = "gray"

            if canonical in seen_color:
                continue

            seen_color.add(canonical)

            color_counts[canonical] += 1

            context = get_context(full_text, match.start(), match.end())

            if NEGATION_PATTERN.search(context):
                negated_counts[f"color::{canonical}"] += 1

            add_example(f"color::{canonical}", context, review)

        # ====================================================
        # 5. COLOR modifiers
        # ====================================================

        for pattern_name, pattern in [  # noqa: B007
            ("modifier", COLOR_MODIFIER_PATTERN),
            ("ish", ISH_COLOR_PATTERN),
        ]:
            seen_phrase = set()

            for match in pattern.finditer(full_text):
                phrase = normalize(match.group(1))

                if phrase in seen_phrase:
                    continue

                seen_phrase.add(phrase)

                color_modifier_counts[phrase] += 1

                context = get_context(full_text, match.start(), match.end())

                add_example(f"color_phrase::{phrase}", context, review)

        # ====================================================
# progress
        # ====================================================

        if total_reviews % PROGRESS_EVERY == 0:
            elapsed = time.time() - start_time

            speed = total_reviews / elapsed if elapsed > 0 else 0

            print(
                f"\rscanned {total_reviews:,} reviews | {speed:,.0f}/s | errors {bad_lines:,}",
                end="",
                flush=True,
            )


# ============================================================
# scan complete
# ============================================================

elapsed = time.time() - start_time


# ============================================================
# output helpers
# ============================================================


def counter_rows(counter, prefix, top_k=None):

    rows = []

    items = counter.most_common(top_k) if top_k else counter.most_common()

    for key, count in items:
        example_key = f"{prefix}{key}"

        rows.append(
            {
                "value": key,
                "count": count,
                "examples": examples.get(example_key, []),
            }
        )

    return rows


# ============================================================
# Size/Fit
# ============================================================

size_fit_output = {}


for label, count in size_fit_counts.most_common():
    negated = negated_counts[f"size_fit::{label}"]

    size_fit_output[label] = {
        "count": count,
        "negated_nearby": negated,
        "negated_rate": round(negated / count if count else 0, 6),
        "patterns": SIZE_FIT_PATTERNS[label],
        "examples": examples.get(f"size_fit::{label}", []),
    }


# ============================================================
# Material
# ============================================================

material_output = []


for canonical, count in material_counts.most_common():
    negated = negated_counts[f"material::{canonical}"]

    aliases = []

    for (alias_canonical, raw), alias_count in material_alias_counts.items():
        if alias_canonical == canonical:
            aliases.append(
                {
                    "alias": raw,
                    "count": alias_count,
                }
            )

    aliases.sort(key=lambda x: -x["count"])

    material_output.append(
        {
            "canonical": canonical,
            "count": count,
            "negated_nearby": negated,
            "negated_rate": round(negated / count if count else 0, 6),
            "observed_aliases": aliases,
            "examples": examples.get(f"material::{canonical}", []),
        }
    )


# ============================================================
# Material phrase output
# ============================================================

material_phrase_output = {}


for phrase_type, counter in material_phrase_counts.items():
    rows = []

    for phrase, count in counter.most_common(500):
        rows.append(
            {
                "phrase": phrase,
                "count": count,
                "examples": examples.get(f"material_phrase::{phrase_type}::{phrase}", []),
            }
        )

    material_phrase_output[phrase_type] = rows


# ============================================================
# Color
# ============================================================

color_output = []


for canonical, count in color_counts.most_common():
    negated = negated_counts[f"color::{canonical}"]

    color_output.append(
        {
            "canonical": canonical,
            "count": count,
            "negated_nearby": negated,
            "negated_rate": round(negated / count if count else 0, 6),
            "examples": examples.get(f"color::{canonical}", []),
        }
    )


# ============================================================
# Color phrases
# ============================================================

color_phrase_output = []


for phrase, count in color_modifier_counts.most_common(1000):
    color_phrase_output.append(
        {
            "phrase": phrase,
            "count": count,
            "examples": examples.get(f"color_phrase::{phrase}", []),
        }
    )


# ============================================================
# Final JSON
# ============================================================

output = {
    "meta": {
        "source": REVIEWS_PATH.name,
        "source_vocab": VOCAB_PATH.name,
        "total_reviews": total_reviews,
        "bad_lines": bad_lines,
        "verified_reviews": verified_reviews,
        "verified_rate": round(verified_reviews / total_reviews if total_reviews else 0, 6),
        "elapsed_seconds": round(elapsed, 2),
        "method": "full_review_paraphrase_mining_v1",
        "notes": [
            ("No vocab file is modified."),
            ("Known noisy legacy material anchors are excluded."),
            ("Size fit phrases are stored separately from literal size values."),
            (
                "Material and color phrase "
                "candidates require review before "
                "being added to retrieval assets."
            ),
        ],
    },
    "size_fit": {"summary": size_fit_output},
    "material": {
        "trusted_anchor_stats": material_output,
        "natural_phrase_candidates": material_phrase_output,
    },
    "color": {
        "trusted_anchor_stats": color_output,
        "modifier_phrase_candidates": color_phrase_output,
    },
}


# ============================================================
# save
# ============================================================

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

    f.write("\n")


# ============================================================
# terminal final summary
# ============================================================

print("\n\n")
print("=" * 75)
print("FULL REVIEW MINING COMPLETE")
print("=" * 75)

print(f"Reviews：{total_reviews:,}")

print(f"errors: {bad_lines:,}")

print(f"Verified：{verified_reviews:,} ({verified_reviews / total_reviews:.2%})")

print(f"elapsed: {elapsed / 60:.2f} minutes")


print("\nSize/Fit：")

for label, count in size_fit_counts.most_common():
    print(f"{label:<20}{count:>12,}")


print("\nMaterial Top 15：")

for canonical, count in material_counts.most_common(15):
    print(f"{canonical:<30}{count:>12,}")


print("\nColor Top 15：")

for canonical, count in color_counts.most_common(15):
    print(f"{canonical:<30}{count:>12,}")


print("\nColor phrase Top 15：")

for phrase, count in color_modifier_counts.most_common(15):
    print(f"{phrase:<30}{count:>12,}")


print(f"\noutput: {OUTPUT_PATH}")

print("\nv vocab_metadata_v2.json was not modified.")
