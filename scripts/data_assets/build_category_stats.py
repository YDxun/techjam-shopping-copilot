import json
import time
from collections import Counter, defaultdict
from pathlib import Path

# ============================================================
# 配置
# ============================================================

META_PATH = Path("meta_Clothing_Shoes_and_Jewelry.jsonl")

OUTPUT_PATH = Path("category_stats.json")

# 保存多少个高频 category / path
TOP_K = 5000

# 每种结构最多保留多少样本
MAX_EXAMPLES = 20


# ============================================================
# 工具
# ============================================================


def normalize(value):

    if value is None:
        return ""

    value = str(value).strip()

    value = " ".join(value.split())

    return value


def normalize_lower(value):

    return normalize(value).lower()


def add_example(examples, key, value):

    bucket = examples[key]

    if len(bucket) < MAX_EXAMPLES:
        if value not in bucket:
            bucket.append(value)


# ============================================================
# 统计器
# ============================================================

total = 0
bad_lines = 0


# ------------------------------------------------------------
# main_category
# ------------------------------------------------------------

main_category_counts = Counter()

main_category_missing = 0


# ------------------------------------------------------------
# categories 原始类型
#
# 我们先不假设它一定是 list[str]
# ------------------------------------------------------------

categories_type_counts = Counter()

categories_missing = 0
categories_empty = 0


# ------------------------------------------------------------
# category value
# ------------------------------------------------------------

category_value_counts = Counter()

category_depth_counts = Counter()

category_path_counts = Counter()


# ------------------------------------------------------------
# main_category -> categories
# ------------------------------------------------------------

main_to_category = defaultdict(Counter)


# ------------------------------------------------------------
# 脏值
# ------------------------------------------------------------

SUSPICIOUS_VALUES = {
    "",
    "generic",
    "none",
    "null",
    "unknown",
    "n/a",
    "na",
    "other",
    "others",
}


suspicious_counts = Counter()


# ------------------------------------------------------------
# 示例
# ------------------------------------------------------------

examples = defaultdict(list)


# ============================================================
# category 解析
# ============================================================


def extract_category_paths(categories):
    """
    输出统一格式：

    [
        ["Women", "Clothing", "Dresses"],
        ...
    ]

    兼容：
    - list[str]
    - list[list[str]]
    - str
    - 其他脏结构
    """

    if categories is None:
        return []

    # --------------------------------------------------------
    # 单字符串
    # --------------------------------------------------------

    if isinstance(categories, str):
        value = normalize(categories)

        if not value:
            return []

        return [[value]]

    # --------------------------------------------------------
    # list
    # --------------------------------------------------------

    if isinstance(categories, list):
        if not categories:
            return []

        # list[str]
        if all(isinstance(x, str) for x in categories):
            path = [normalize(x) for x in categories if normalize(x)]

            return [path] if path else []

        # list[list[str]]
        paths = []

        for item in categories:
            if isinstance(item, list):
                path = [normalize(x) for x in item if normalize(x)]

                if path:
                    paths.append(path)

            elif isinstance(item, str):
                value = normalize(item)

                if value:
                    paths.append([value])

        return paths

    return []


# ============================================================
# 开始扫描
# ============================================================

print("=" * 75)
print("FULL CATEGORY STATISTICS")
print("=" * 75)

print(f"输入：{META_PATH}")

start = time.time()


with META_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        try:
            item = json.loads(line)

        except json.JSONDecodeError:
            bad_lines += 1
            continue

        total += 1

        # ====================================================
        # main_category
        # ====================================================

        main_category = normalize(item.get("main_category"))

        if main_category:
            main_category_counts[main_category] += 1

        else:
            main_category_missing += 1

        # ====================================================
        # categories
        # ====================================================

        categories = item.get("categories")

        if categories is None:
            categories_missing += 1

            categories_type_counts["null"] += 1

            continue

        categories_type_counts[type(categories).__name__] += 1

        paths = extract_category_paths(categories)

        if not paths:
            categories_empty += 1
            continue

        # ====================================================
        # 每条 path
        # ====================================================

        for path in paths:
            category_depth_counts[len(path)] += 1

            path_norm = [normalize(x) for x in path if normalize(x)]

            if not path_norm:
                continue

            path_string = " > ".join(path_norm)

            category_path_counts[path_string] += 1

            # -----------------------------------------------
            # 每个节点
            # -----------------------------------------------

            for value in path_norm:
                category_value_counts[value] += 1

                value_lower = normalize_lower(value)

                if value_lower in SUSPICIOUS_VALUES:
                    suspicious_counts[value] += 1

                    add_example(
                        examples,
                        f"suspicious::{value_lower}",
                        {
                            "main_category": main_category,
                            "path": path_norm,
                            "title": normalize(item.get("title"))[:300],
                        },
                    )

            # -----------------------------------------------
            # main_category -> leaf
            # -----------------------------------------------

            leaf = path_norm[-1]

            if main_category:
                main_to_category[main_category][leaf] += 1

        # ====================================================
        # 进度
        # ====================================================

        if total % 100000 == 0:
            elapsed = time.time() - start

            speed = total / elapsed if elapsed > 0 else 0

            print(
                f"\r已扫描 {total:,} products | {speed:,.0f}/s | errors {bad_lines:,}",
                end="",
                flush=True,
            )


# ============================================================
# 完成
# ============================================================

elapsed = time.time() - start


# ============================================================
# main -> leaf
# ============================================================

main_to_category_output = {}


for main, counter in sorted(main_to_category.items(), key=lambda x: -sum(x[1].values())):
    main_to_category_output[main] = [
        {
            "category": category,
            "count": count,
        }
        for category, count in counter.most_common(200)
    ]


# ============================================================
# 输出
# ============================================================

output = {
    "meta": {
        "source": META_PATH.name,
        "total_products": total,
        "bad_lines": bad_lines,
        "elapsed_seconds": round(elapsed, 2),
        "method": "full_metadata_category_statistics_v1",
    },
    "coverage": {
        "main_category_missing": main_category_missing,
        "main_category_missing_rate": round(main_category_missing / total if total else 0, 6),
        "categories_missing": categories_missing,
        "categories_missing_rate": round(categories_missing / total if total else 0, 6),
        "categories_empty": categories_empty,
        "categories_empty_rate": round(categories_empty / total if total else 0, 6),
    },
    "categories_type_counts": dict(categories_type_counts),
    "main_categories": [
        {
            "value": value,
            "count": count,
            "coverage": round(count / total, 6),
        }
        for value, count in main_category_counts.most_common()
    ],
    "category_depth_counts": [
        {
            "depth": depth,
            "count": count,
        }
        for depth, count in sorted(category_depth_counts.items())
    ],
    "top_category_values": [
        {
            "value": value,
            "count": count,
        }
        for value, count in category_value_counts.most_common(TOP_K)
    ],
    "top_category_paths": [
        {
            "path": path,
            "count": count,
        }
        for path, count in category_path_counts.most_common(TOP_K)
    ],
    "main_category_to_leaf": main_to_category_output,
    "suspicious_values": [
        {
            "value": value,
            "count": count,
            "examples": examples.get(f"suspicious::{normalize_lower(value)}", []),
        }
        for value, count in suspicious_counts.most_common()
    ],
}


# ============================================================
# 保存
# ============================================================

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

    f.write("\n")


# ============================================================
# Terminal summary
# ============================================================

print("\n\n")
print("=" * 75)
print("CATEGORY STATISTICS COMPLETE")
print("=" * 75)

print(f"商品：{total:,}")

print(f"错误：{bad_lines:,}")

print(f"耗时：{elapsed / 60:.2f} 分钟")


print("\n字段覆盖：")

print(f"main_category missing: {main_category_missing:,} ({main_category_missing / total:.2%})")

print(f"categories missing:    {categories_missing:,} ({categories_missing / total:.2%})")

print(f"categories empty:      {categories_empty:,} ({categories_empty / total:.2%})")


print("\ncategories 类型：")

for key, count in categories_type_counts.most_common():
    print(f"{key:<15}{count:>12,}")


print("\nTop main_category：")

for value, count in main_category_counts.most_common(20):
    print(f"{value:<45}{count:>12,}")


print("\nTop category paths：")

for path, count in category_path_counts.most_common(30):
    print(f"{path[:80]:<82}{count:>10,}")


print("\nSuspicious category values：")

for value, count in suspicious_counts.most_common(30):
    print(f"{value:<40}{count:>10,}")


print(f"\n输出：{OUTPUT_PATH}")
