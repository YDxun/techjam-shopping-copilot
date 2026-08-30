import json
import re
import time
from collections import Counter
from pathlib import Path

# ============================================================
# 配置
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
# 统计器
# ============================================================

field_nonempty = Counter()
field_hits = Counter()
pattern_hits = Counter()

total = 0
bad_lines = 0


# ============================================================
# 文本标准化
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
# 1. 字母尺码
#
# 例如：
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
# 带 Size 上下文
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
# 字母尺码范围
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
# 2. US 尺码
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
# 3. 数字尺码
#
# size 6
# size: 9
# shoe size 8.5
# waist 32
# inseam 30
#
# 注意：
# 后面还会过滤物理尺寸。
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
# 4. 物理尺寸排除
#
# 用于排除：
#
# Size: 12.5"
# Size: 17 inches
# Size: 9 x 6
# Size: 20 cm
#
# 这些不是 wearable size。
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

    # 只观察匹配词后面一小段文本
    after = text[match.end() : match.end() + 25]

    return bool(DIMENSION_AFTER.search(after))


# ============================================================
# 5. 普通文本 Size Matcher
#
# 用于：
# title
# features
# description
# ============================================================


def match_text_size(text):

    # --------------------------------
    # 字母尺码 + size 上下文
    # --------------------------------

    match = LETTER_WITH_CONTEXT.search(text)

    if match:
        return "letter_with_context"

    # --------------------------------
    # S/M/L 或 S-3XL
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
    # size 6 / waist 32 等
    # --------------------------------

    match = NUMERIC_WITH_CONTEXT.search(text)

    if match:
        # 排除：
        #
        # Size: 12.5"
        # Size: 17 inch
        # Size: 9 x 6

        if looks_like_dimension_after(text, match):
            return None

        return "numeric_with_context"

    return None


# ============================================================
# 6. details.Size 专用 Matcher
#
# details.Size 虽然是结构化字段，
# 但之前人工检查已经发现：
#
# 10 Watch Box
# 8.0 inches
# 9 x 6 Inches
#
# 等脏值。
#
# 所以不能无条件相信。
# ============================================================


# ------------------------------------------------------------
# 字母型 Size
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
# 纯数字
#
# 7
# 8
# 38
# ------------------------------------------------------------

STRUCTURED_NUMERIC_ONLY = re.compile(r"^\s*\d{1,2}(?:\.5)?\s*$", re.IGNORECASE)


# ------------------------------------------------------------
# 数字 + 单字母宽度
#
# 8 M
# 9 W
# ------------------------------------------------------------

STRUCTURED_NUMERIC_WIDTH = re.compile(r"^\s*\d{1,2}(?:\.5)?\s*[a-z]?\s*$", re.IGNORECASE)


def match_structured_size(text):

    # --------------------------------
    # 先排除明显 dimensions
    # --------------------------------

    if DIMENSION_TEXT.search(text):
        return None

    lower = text.lower()

    # --------------------------------
    # 排除明显非 wearable size
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
    # 字母尺码
    # --------------------------------

    if STRUCTURED_LETTER.search(text):
        return "structured_letter_size"

    # --------------------------------
    # 纯数字尺码
    # --------------------------------

    if STRUCTURED_NUMERIC_ONLY.fullmatch(text):
        return "structured_numeric_size"

    # --------------------------------
    # 数字 + width
    # --------------------------------

    if STRUCTURED_NUMERIC_WIDTH.fullmatch(text):
        return "structured_numeric_width"

    return None


# ============================================================
# 7. 开始全量扫描
# ============================================================

print("=" * 75)
print("SIZE FULL CATALOG SCAN V2")
print("=" * 75)

print(f"输入：{META_PATH}")

print(f"输出：{OUTPUT_PATH}")

print()


start_time = time.time()


with META_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        # --------------------------------
        # JSON 解析
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
        # 当前商品需要检查的字段
        # --------------------------------

        values = {
            "title": item.get("title"),
            "features": item.get("features"),
            "description": item.get("description"),
            "details.Size": details.get("Size"),
        }

        # --------------------------------
        # 每个字段检查一次
        # --------------------------------

        for field_name in FIELDS:
            text = normalize(values[field_name])

            if not text:
                continue

            # 字段非空
            field_nonempty[field_name] += 1

            # --------------------------------
            # details.Size 使用专用规则
            # --------------------------------

            if field_name == "details.Size":
                match_type = match_structured_size(text)

            # --------------------------------
            # title/features/description
            # --------------------------------

            else:
                match_type = match_text_size(text)

            # --------------------------------
            # 命中
            # --------------------------------

            if match_type:
                field_hits[field_name] += 1

                pattern_hits[match_type] += 1

        # ====================================================
        # 进度
        # ====================================================

        if total % 100000 == 0:
            elapsed = time.time() - start_time

            speed = total / elapsed if elapsed > 0 else 0

            print(
                f"\r已扫描 {total:,} 个商品 | {speed:,.0f} items/s | 错误 {bad_lines:,}",
                end="",
                flush=True,
            )


# ============================================================
# 8. 扫描完成
# ============================================================

elapsed = time.time() - start_time


# ============================================================
# 9. 构造 JSON 输出
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
# 10. 字段统计
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
# 11. 保存 JSON
# ============================================================

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)


# ============================================================
# 12. Terminal 最终结果
# ============================================================

print("\n\n")
print("=" * 75)
print("SIZE FULL STATS COMPLETE")
print("=" * 75)

print(f"商品总数：{total:,}")

print(f"错误行数：{bad_lines:,}")

print(f"耗时：{elapsed / 60:.2f} 分钟")

print(f"输出：{OUTPUT_PATH}")


print("\n字段结果：")


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


print("\n匹配类型：")


for name, count in pattern_hits.most_common():
    print(f"{name:<32}{count:>12,}")


print("\n完成。")
