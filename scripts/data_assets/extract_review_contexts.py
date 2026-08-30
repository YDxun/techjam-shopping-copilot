import gzip
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

# ============================================================
# 配置
# ============================================================

REVIEWS_PATH = Path("Clothing_Shoes_and_Jewelry.jsonl.gz")

VOCAB_PATH = Path("vocab_metadata_v2.json")

OUTPUT_PATH = Path("review_context_candidates_sample.json")

# 第一轮只跑 100 万条
MAX_REVIEWS = 1_000_000

# anchor 左右保留多少字符
CONTEXT_WINDOW = 100

# 每个 anchor 最多保存多少真实例子
MAX_EXAMPLES_PER_ANCHOR = 20

# 太短的 anchor 很容易误匹配
MIN_TERM_LENGTH = 3


# ============================================================
# 只处理这些属性
# ============================================================

TARGET_ATTRIBUTES = [
    "material",
    "color",
    "size",
]


# ============================================================
# Size 特殊允许词
#
# s / m / l 虽然只有一个字符，
# 但必须带 size context 才允许。
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
# 通用工具
# ============================================================


def normalize(value):

    if value is None:
        return ""

    value = str(value).lower()

    value = value.replace("’", "'")

    value = re.sub(r"\s+", " ", value)

    return value.strip()


# ============================================================
# 读取 vocab
# ============================================================

print("读取 vocab...")

with VOCAB_PATH.open("r", encoding="utf-8") as f:
    vocab = json.load(f)


dictionaries = vocab.get("dictionaries", {})


# ============================================================
# 建立 anchor → canonical
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
            # 普通词至少 3 字符
            # --------------------------------

            if len(term) < MIN_TERM_LENGTH:
                continue

            # --------------------------------
            # size 的短词单独允许
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


print("\nAnchor 数量：")

for attr in TARGET_ATTRIBUTES:
    print(f"{attr:<12}{attribute_anchor_counts[attr]:>8,}")


print(f"\n总 anchor：{len(anchor_map):,}")


# ============================================================
# 构建 regex
#
# 长词优先，避免：
#
# stainless steel
# steel
#
# 同时命中时优先长表达。
# ============================================================

anchors = sorted(anchor_map.keys(), key=len, reverse=True)


escaped = [re.escape(x) for x in anchors]


ANCHOR_PATTERN = re.compile(r"(?<![a-z0-9])(" + "|".join(escaped) + r")(?![a-z0-9])", re.IGNORECASE)


# ============================================================
# 否定/语境词
#
# 第一阶段不删除，
# 只做标记。
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
# 统计器
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
# 保存上下文样本
# ============================================================

examples = defaultdict(list)


def add_example(key, record):

    bucket = examples[key]

    if len(bucket) < MAX_EXAMPLES_PER_ANCHOR:
        bucket.append(record)


# ============================================================
# 开始扫描
# ============================================================

print("\n开始扫描 reviews...")
print(f"本轮上限：{MAX_REVIEWS:,}")

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
        # 找 anchor
        # ====================================================

        matches = list(ANCHOR_PATTERN.finditer(full_text))

        if not matches:
            continue

        reviews_with_anchor += 1

        # 同一 review 内避免完全重复统计
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
                # 保存真实例子
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
        # 进度
        # ====================================================

        if total_reviews % 100000 == 0:
            elapsed = time.time() - start

            speed = total_reviews / elapsed if elapsed > 0 else 0

            print(
                f"\r已扫描 "
                f"{total_reviews:,} reviews | "
                f"{speed:,.0f}/s | "
                f"有 anchor "
                f"{reviews_with_anchor:,} | "
                f"错误 {bad_lines:,}",
                end="",
                flush=True,
            )


# ============================================================
# 完成
# ============================================================

elapsed = time.time() - start


# ============================================================
# 构造输出
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
# 每个属性整理
# ============================================================

for attr in TARGET_ATTRIBUTES:
    canonical_rows = []

    for canonical, count in canonical_hits[attr].most_common():
        anchor_rows = []

        # 找这个 canonical 的 anchors
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
# 保存
# ============================================================

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

    f.write("\n")


# ============================================================
# Terminal 摘要
# ============================================================

print("\n\n")
print("=" * 75)
print("REVIEW CONTEXT SAMPLE COMPLETE")
print("=" * 75)

print(f"扫描 reviews：{total_reviews:,}")

print(f"错误：{bad_lines:,}")

print(f"含 vocab anchor：{reviews_with_anchor:,}")

print(f"anchor review rate：{reviews_with_anchor / total_reviews:.2%}")

print(f"总 anchor hits：{total_anchor_hits:,}")

print(f"耗时：{elapsed / 60:.2f} 分钟")


print("\n属性命中：")

for attr in TARGET_ATTRIBUTES:
    print(f"{attr:<12}{attribute_hits[attr]:>12,}")


print(f"\n输出：{OUTPUT_PATH}")

print("\n注意：当前没有向 vocab 自动添加任何 synonym。")
