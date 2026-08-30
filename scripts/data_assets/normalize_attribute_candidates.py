import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# ============================================================
# 配置
# ============================================================

CANDIDATE_PATH = Path("attribute_candidates.json")
VOCAB_PATH = Path("vocab(1).json")
OUTPUT_PATH = Path("vocab_candidate_review.json")

# 如果你的文件实际叫 attribute_candidates(1).json，
# 就把上面的 CANDIDATE_PATH 改成：
# CANDIDATE_PATH = Path("attribute_candidates(1).json")


# ============================================================
# 基础工具
# ============================================================


def normalize_text(value):

    if value is None:
        return ""

    value = str(value).strip().lower()

    value = value.replace("–", "-")
    value = value.replace("—", "-")

    value = re.sub(r"\s+", " ", value)

    return value.strip()


def clean_hyphen(value):

    value = normalize_text(value)

    # stainless-steel -> stainless steel
    value = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", value)

    value = re.sub(r"\s+", " ", value)

    return value.strip()


# ============================================================
# 读取文件
# ============================================================

print("读取文件...")

with CANDIDATE_PATH.open("r", encoding="utf-8") as f:
    candidate_data = json.load(f)

with VOCAB_PATH.open("r", encoding="utf-8") as f:
    vocab = json.load(f)


dictionaries = vocab.get("dictionaries", {})


# ============================================================
# 建立现有 vocab term → canonical 映射
# ============================================================

existing_map = defaultdict(dict)

for attr in [
    "material",
    "color",
    "size",
]:
    for canonical, info in dictionaries.get(attr, {}).items():
        canonical_norm = normalize_text(canonical)

        existing_map[attr][canonical_norm] = canonical

        for synonym in info.get("synonyms", []):
            synonym_norm = normalize_text(synonym)

            if synonym_norm:
                existing_map[attr][synonym_norm] = canonical


# ============================================================
# 通用垃圾值
# ============================================================

COMMON_REJECT = {
    "",
    "n/a",
    "na",
    "none",
    "unknown",
    "not applicable",
    "not-applicable",
    "not specified",
    "unspecified",
    "see description",
    "as shown",
    "other",
    "others",
    "various",
    "various colors",
    "assorted",
    "random",
    "default",
}


# ============================================================
# Material
# ============================================================

MATERIAL_REJECT = COMMON_REJECT | {
    "quality",
    "comfortable",
    "comfort",
    "women",
    "woman",
    "womens",
    "men",
    "mens",
    "shoes",
    "shoe",
    "brand new",
    "high quality",
    "base",
}


# 明确的材料基础词。
# 这些不是凭空造的，主要对应我们 metadata
# 高频结构化候选中的基础材料类型。
KNOWN_MATERIALS = {
    "metal",
    "stainless steel",
    "synthetic",
    "plastic",
    "wood",
    "silicone",
    "crystal",
    "sterling silver",
    "stone",
    "ceramic",
    "glass",
    "aluminum",
    "aluminium",
    "canvas",
    "alloy",
    "vinyl",
    "brass",
    "gemstone",
    "zinc",
    "copper",
    "neoprene",
    "resin",
    "polyvinyl chloride",
    "pvc",
    "rhinestone",
    "pearl",
    "felt",
    "carbon fiber",
    "steel",
    "microfiber",
    "iron",
    "polycarbonate",
    "polypropylene",
    "latex",
    "foam",
    "titanium",
    "bronze",
    "hemp",
    "cork",
    "cowhide",
    "flannel",
    "acetate",
    "polyethylene",
    "nickel",
    "tungsten",
    "tungsten carbide",
    "fiberglass",
    "polyurethane",
    "thermoplastic polyurethane",
    "thermoplastic urethane",
    "eva",
    "ethylene vinyl acetate",
    "synthetic leather",
    "faux leather",
    "faux suede",
    "pleather",
    "microfiber leather",
    "polycotton",
    "synthetic fiber",
    "synthetic fibers",
    "polyester fiber",
    "human hair",
    "synthetic hair",
    "oxford cloth",
    "ripstop",
    "burlap",
    "straw",
    "feather",
}


# 已知 alias → canonical
MATERIAL_ALIAS = {
    "aluminium": "aluminum",
    "pvc": "polyvinyl chloride",
    "polyvinyl chloride (pvc)": "polyvinyl chloride",
    "eva": "ethylene vinyl acetate",
    "ethylene vinyl acetate (eva)": "ethylene vinyl acetate",
    "tpu": "thermoplastic polyurethane",
    "thermoplastic polyurethane (tpu)": "thermoplastic polyurethane",
    "thermoplastic urethane": "thermoplastic polyurethane",
    "pu": "polyurethane",
    "polyurethane (pu)": "polyurethane",
    "synthetic-fiber": "synthetic fiber",
    "synthetic fibers": "synthetic fiber",
    "stainless-steel": "stainless steel",
    "polyster": "polyester",
    "pleather": "faux leather",
}


# ============================================================
# Material 配方解析
# ============================================================

PERCENTAGE = re.compile(r"\b\d{1,3}(?:\.\d+)?\s*%\s*")

MATERIAL_SEPARATORS = re.compile(r"\s*(?:,|/|&|\+|;)\s*")


def normalize_material_piece(piece):

    piece = normalize_text(piece)

    # 95% polyester -> polyester
    piece = PERCENTAGE.sub("", piece)

    piece = piece.strip(" ,;/+-")

    piece = clean_hyphen(piece)

    # 一些常见无意义修饰
    piece = re.sub(r"^100 percent\s+", "", piece)

    piece = re.sub(r"^100-percent-", "", piece)

    piece = re.sub(r"\s+blend$", "", piece)

    piece = piece.strip()

    if piece in MATERIAL_ALIAS:
        piece = MATERIAL_ALIAS[piece]

    return piece


def split_material(value):

    raw = normalize_text(value)

    # 先去百分比
    cleaned = PERCENTAGE.sub("", raw)

    # 常见逗号、斜线、& 拆分
    parts = MATERIAL_SEPARATORS.split(cleaned)

    output = []

    for part in parts:
        part = normalize_material_piece(part)

        if not part:
            continue

        output.append(part)

    return output


# ============================================================
# Color
# ============================================================

COLOR_REJECT = COMMON_REJECT | {
    "colors",
    "color",
    "different",
    "design",
    "fashion",
    "pattern",
    "strong",
    "women",
    "printed",
}


COLOR_ALIAS = {
    "multicolored": "multicolor",
    "multicoloured": "multicolor",
    "multi-colored": "multicolor",
    "multi coloured": "multicolor",
    "grey": "gray",
    "rose-gold": "rose gold",
    "black / white": "black/white",
    "white / black": "white/black",
}


# 基础颜色 + metadata 中明显有价值的颜色
KNOWN_COLORS = {
    "black",
    "white",
    "blue",
    "red",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "gray",
    "silver",
    "gold",
    "beige",
    "navy",
    "teal",
    "maroon",
    "burgundy",
    "ivory",
    "cream",
    "khaki",
    "tan",
    "turquoise",
    "coral",
    "lavender",
    "violet",
    "magenta",
    "cyan",
    "olive",
    "mint",
    "peach",
    "champagne",
    "multicolor",
    "rose gold",
    "clear",
    "coffee",
    "dark brown",
    "blonde",
    "copper",
    "bronze",
}


def normalize_color(value):

    value = normalize_text(value)

    value = COLOR_ALIAS.get(value, value)

    return value


# ============================================================
# Size
# ============================================================

SIZE_REJECT = COMMON_REJECT | {
    "standard",
    "one size",
    "one-size",
    "1 count (pack of 1)",
    "pack of 1",
}


SIZE_ALIAS = {
    "xx-small": "xxs",
    "x-small": "xs",
    "small": "s",
    "medium": "m",
    "large": "l",
    "x-large": "xl",
    "xx-large": "xxl",
    "xxx-large": "3xl",
    "2x-large": "2xl",
    "3x-large": "3xl",
    "4x-large": "4xl",
}


VALID_LETTER_SIZE = re.compile(
    r"^(?:"
    r"xxxs|xxs|xs|s|m|l|xl|xxl|"
    r"2xl|3xl|4xl|5xl|"
    r"2x|3x|4x|5x"
    r")$",
    re.IGNORECASE,
)


# 明显物理尺寸
PHYSICAL_SIZE = re.compile(
    r"""
    (?:
        \d+(?:\.\d+)?\s*
        (?:inch|inches|in\b|cm\b|mm\b|ft\b|feet\b|["'″”])
        |
        \d+(?:\.\d+)?\s*[x×]\s*
        \d+(?:\.\d+)?
        |
        pack\s+of
        |
        count\s*\(
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def normalize_size(value):

    value = normalize_text(value)

    if value in SIZE_ALIAS:
        return SIZE_ALIAS[value]

    return value


# ============================================================
# 输出结构
# ============================================================

result = {
    "meta": {
        "source_candidates": CANDIDATE_PATH.name,
        "source_vocab": VOCAB_PATH.name,
        "purpose": (
            "Normalize metadata-derived "
            "material/color/size candidates "
            "into AUTO_ACCEPT, REVIEW and REJECT."
        ),
        "important": (
            "AUTO_ACCEPT is a candidate decision "
            "for vocab_v2 construction, not an "
            "instruction to overwrite the original vocab."
        ),
    },
    "attributes": {},
}


# ============================================================
# 辅助添加
# ============================================================


def add_row(buckets, status, raw_value, normalized, count, fields, reason, canonical=None):

    buckets[status].append(
        {
            "raw_value": raw_value,
            "normalized": normalized,
            "canonical": canonical,
            "count": count,
            "fields": fields,
            "reason": reason,
        }
    )


# ============================================================
# MATERIAL
# ============================================================

print("\n处理 MATERIAL...")

material_buckets = {
    "AUTO_ACCEPT": [],
    "REVIEW": [],
    "REJECT": [],
}

material_candidates = candidate_data["attributes"]["material"]["top_new_candidates"]


# 聚合基础 material
material_aggregate = defaultdict(
    lambda: {
        "count": 0,
        "raw_values": Counter(),
        "fields": Counter(),
    }
)


for row in material_candidates:
    raw = normalize_text(row["value"])

    count = row["count"]
    fields = row.get("fields", {})

    if raw in MATERIAL_REJECT:
        add_row(material_buckets, "REJECT", raw, raw, count, fields, "known noise or placeholder")

        continue

    pieces = split_material(raw)

    if not pieces:
        add_row(material_buckets, "REJECT", raw, raw, count, fields, "empty after normalization")

        continue

    # 将组合值拆成基础材料
    for piece in pieces:
        if piece in MATERIAL_REJECT:
            continue

        canonical = MATERIAL_ALIAS.get(piece, piece)

        material_aggregate[canonical]["count"] += count

        material_aggregate[canonical]["raw_values"][raw] += count

        for field, field_count in fields.items():
            material_aggregate[canonical]["fields"][field] += field_count


for canonical, data in material_aggregate.items():
    count = data["count"]

    fields = dict(data["fields"])

    raw_examples = [x for x, _ in data["raw_values"].most_common(10)]

    # 已存在 vocab
    if canonical in existing_map["material"]:
        status = "AUTO_ACCEPT"

        reason = "normalizes to existing material vocab"

        target = existing_map["material"][canonical]

    # 高可信基础材料
    elif canonical in KNOWN_MATERIALS and count >= 20:
        status = "AUTO_ACCEPT"

        reason = "recognized base material with structured metadata support"

        target = canonical

    # 高频但未知
    elif count >= 100:
        status = "REVIEW"

        reason = "frequent structured material value but not safely recognized as a base material"

        target = canonical

    # 中频
    elif count >= 20:
        status = "REVIEW"

        reason = "structured material candidate requires semantic review"

        target = canonical

    else:
        status = "REJECT"

        reason = "low-frequency unverified material candidate"

        target = canonical

    material_buckets[status].append(
        {
            "canonical": target,
            "normalized": canonical,
            "count": count,
            "fields": fields,
            "raw_examples": raw_examples,
            "reason": reason,
        }
    )


# ============================================================
# COLOR
# ============================================================

print("处理 COLOR...")

color_buckets = {
    "AUTO_ACCEPT": [],
    "REVIEW": [],
    "REJECT": [],
}

color_candidates = candidate_data["attributes"]["color"]["top_new_candidates"]


for row in color_candidates:
    raw = normalize_text(row["value"])

    value = normalize_color(raw)

    count = row["count"]

    fields = row.get("fields", {})

    if value in COLOR_REJECT:
        add_row(
            color_buckets, "REJECT", raw, value, count, fields, "known color noise or placeholder"
        )

        continue

    # 已存在 vocab
    if value in existing_map["color"]:
        add_row(
            color_buckets,
            "AUTO_ACCEPT",
            raw,
            value,
            count,
            fields,
            "normalizes to existing color vocab",
            existing_map["color"][value],
        )

        continue

    # 明确颜色
    if value in KNOWN_COLORS and count >= 20:
        add_row(
            color_buckets,
            "AUTO_ACCEPT",
            raw,
            value,
            count,
            fields,
            "recognized color with structured metadata support",
            value,
        )

        continue

    # 两色组合：
    # black/white
    # black/red
    if re.fullmatch(r"[a-z -]+/[a-z -]+", value):
        add_row(
            color_buckets,
            "REVIEW",
            raw,
            value,
            count,
            fields,
            "multi-color combination; review mapping strategy",
            value,
        )

        continue

    if count >= 100:
        add_row(
            color_buckets,
            "REVIEW",
            raw,
            value,
            count,
            fields,
            "frequent structured color value requiring review",
            value,
        )

    else:
        add_row(
            color_buckets,
            "REJECT",
            raw,
            value,
            count,
            fields,
            "low-frequency unverified color candidate",
            value,
        )


# ============================================================
# SIZE
# ============================================================

print("处理 SIZE...")

size_buckets = {
    "AUTO_ACCEPT": [],
    "REVIEW": [],
    "REJECT": [],
}

size_candidates = candidate_data["attributes"]["size"]["top_new_candidates"]


for row in size_candidates:
    raw = normalize_text(row["value"])

    value = normalize_size(raw)

    count = row["count"]

    fields = row.get("fields", {})

    # 明显物理尺寸
    if PHYSICAL_SIZE.search(raw):
        add_row(
            size_buckets,
            "REJECT",
            raw,
            value,
            count,
            fields,
            "physical dimension rather than wearable size",
        )

        continue

    if raw in SIZE_REJECT:
        add_row(size_buckets, "REJECT", raw, value, count, fields, "known non-wearable size value")

        continue

    # 归一化后已有
    if value in existing_map["size"]:
        add_row(
            size_buckets,
            "AUTO_ACCEPT",
            raw,
            value,
            count,
            fields,
            "normalizes to existing size vocab",
            existing_map["size"][value],
        )

        continue

    # 标准字母尺码
    if VALID_LETTER_SIZE.fullmatch(value):
        add_row(
            size_buckets,
            "AUTO_ACCEPT",
            raw,
            value,
            count,
            fields,
            "recognized wearable letter size",
            value,
        )

        continue

    # 纯数字可能是服装/鞋码，
    # 但不能仅凭 details.Size 自动接受
    if re.fullmatch(r"\d{1,2}(?:\.5)?", value):
        add_row(
            size_buckets,
            "REVIEW",
            raw,
            value,
            count,
            fields,
            "numeric size; requires wearable-size context",
            value,
        )

        continue

    if count >= 100:
        add_row(
            size_buckets,
            "REVIEW",
            raw,
            value,
            count,
            fields,
            "frequent structured Size value requiring review",
            value,
        )

    else:
        add_row(
            size_buckets,
            "REJECT",
            raw,
            value,
            count,
            fields,
            "low-frequency unverified Size value",
            value,
        )


# ============================================================
# 排序
# ============================================================


def sort_buckets(buckets):

    for status in buckets:
        buckets[status].sort(
            key=lambda x: (
                -int(x.get("count") or 0),
                str(x.get("canonical") or x.get("normalized") or x.get("raw_value") or ""),
            )
        )

    return buckets


material_buckets = sort_buckets(material_buckets)

color_buckets = sort_buckets(color_buckets)

size_buckets = sort_buckets(size_buckets)


# ============================================================
# 保存
# ============================================================

result["attributes"]["material"] = material_buckets

result["attributes"]["color"] = color_buckets

result["attributes"]["size"] = size_buckets


with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)


# ============================================================
# Terminal 摘要
# ============================================================

print("\n")
print("=" * 75)
print("VOCAB CANDIDATE NORMALIZATION COMPLETE")
print("=" * 75)

for attr in [
    "material",
    "color",
    "size",
]:
    buckets = result["attributes"][attr]

    print(f"\n### {attr.upper()} ###")

    print(f"AUTO_ACCEPT: {len(buckets['AUTO_ACCEPT']):,}")

    print(f"REVIEW:      {len(buckets['REVIEW']):,}")

    print(f"REJECT:      {len(buckets['REJECT']):,}")

    print("\nAUTO_ACCEPT Top 15:")

    for row in buckets["AUTO_ACCEPT"][:15]:
        name = row.get("canonical", row.get("normalized", ""))

        print(f"{name:<30}{row['count']:>10,}")

    print("\nREVIEW Top 15:")

    for row in buckets["REVIEW"][:15]:
        name = row.get("canonical", row.get("normalized", ""))

        print(f"{name:<30}{row['count']:>10,}")


print("\n")
print(f"输出：{OUTPUT_PATH}")
print("原 vocab 未修改。")
