import json
import re
from collections import defaultdict
from pathlib import Path

# ============================================================
# Paths
# ============================================================

STATS_PATH = Path("category_stats.json")

OUTPUT_PATH = Path("category_mapping.json")

REPORT_PATH = Path("category_mapping_report.json")


# ============================================================
# Basic normalization
# ============================================================


def normalize(value):

    if value is None:
        return ""

    return " ".join(str(value).strip().lower().split())


def slug(value):

    value = normalize(value)

    value = value.replace("&", " and ")

    value = re.sub(r"[^a-z0-9]+", "_", value)

    value = re.sub(r"_+", "_", value)

    return value.strip("_")


# ============================================================
# Load statistics
# ============================================================

print("loading category_stats.json...")

with STATS_PATH.open("r", encoding="utf-8") as f:
    stats = json.load(f)


# ============================================================
# Root nodes
# ============================================================

ROOT_NODES = {
    "clothing, shoes & jewelry",
}


# ============================================================
# Audience aliases
# ============================================================

AUDIENCE_ALIASES = {
    # Women
    "women": "women",
    "woman": "women",
    "women's": "women",
    "womens": "women",
    # Men
    "men": "men",
    "man's": "men",
    "men's": "men",
    "mens": "men",
    # Girls
    "girls": "girls",
    "girl": "girls",
    "girls'": "girls",
    # Boys
    "boys": "boys",
    "boy": "boys",
    "boys'": "boys",
    # Baby girls
    "baby girls": "baby_girls",
    "baby girls'": "baby_girls",
    # Baby boys
    "baby boys": "baby_boys",
    "baby boys'": "baby_boys",
    # Baby
    "baby": "baby",
    "unisex baby clothing": "baby",
    # Kids
    "kids": "kids",
    "kids & baby": "kids",
    "kids' clothing": "kids",
    "kids apparel": "kids",
    "kids_apparel": "kids",
}


# ============================================================
# Product-family aliases
#
# IMPORTANT
#
# This is intentionally a COARSE taxonomy.
#
# We are NOT trying to reproduce every Amazon browse node.
# The purpose is stable retrieval routing.
# ============================================================

FAMILY_ALIASES = {
    # ========================================================
    # Dresses
    # ========================================================
    "dresses": "dresses",
    "dress": "dresses",
    "wedding dresses": "dresses",
    "active dresses": "dresses",
    "dresses & jumpers": "dresses",
    # ========================================================
    # Tops / Shirts
    # ========================================================
    "t-shirts": "tshirts",
    "tees": "tshirts",
    "t-shirts & tanks": "tshirts",
    "tops": "tops",
    "tops & tees": "tops",
    "tops, tees & blouses": "tops",
    "tops, tees & shirts": "tops",
    "blouses": "blouses",
    "blouses & button-down shirts": "blouses",
    "shirts": "shirts",
    "button-down shirts": "shirts",
    "casual button-down shirts": "shirts",
    "dress shirts": "shirts",
    "henleys": "shirts",
    "tank tops": "tank_tops",
    "tanks & camis": "tank_tops",
    "camisoles & tanks": "tank_tops",
    "tanks tops": "tank_tops",
    "tunics": "tunics",
    "polos": "polos",
    # ========================================================
    # Sweaters / Hoodies
    # ========================================================
    "sweaters": "sweaters",
    "pullovers": "sweaters",
    "cardigans": "sweaters",
    "cardigan": "sweaters",
    "fashion hoodies & sweatshirts": "hoodies_sweatshirts",
    "hoodies": "hoodies_sweatshirts",
    "sweatshirts": "hoodies_sweatshirts",
    "active hoodies": "hoodies_sweatshirts",
    "active sweatshirts": "hoodies_sweatshirts",
    # ========================================================
    # Jackets / Outerwear
    # ========================================================
    "coats, jackets & vests": "outerwear",
    "jackets & coats": "outerwear",
    "jackets and coats": "outerwear",
    "outerwear": "outerwear",
    "jackets": "jackets",
    "casual jackets": "jackets",
    "denim jackets": "jackets",
    "lightweight jackets": "jackets",
    "rain jackets": "jackets",
    "softshell jackets": "jackets",
    "windbreakers": "jackets",
    "vests": "vests",
    "blazers": "blazers",
    "suiting & blazers": "blazers",
    "raincoats": "rainwear",
    "trench, rain & anoraks": "rainwear",
    "trench & rain": "rainwear",
    "down jackets & parkas": "winter_outerwear",
    "snow & rainwear": "winter_outerwear",
    # ========================================================
    # Bottoms
    # ========================================================
    "pants": "pants",
    "pants & capris": "pants",
    "active pants": "pants",
    "sweatpants": "pants",
    "track pants": "pants",
    "jeans": "jeans",
    "leggings": "leggings",
    "shorts": "shorts",
    "active shorts": "shorts",
    "board shorts": "shorts",
    "skirts": "skirts",
    "skirts & skorts": "skirts",
    "active skirts": "skirts",
    "active skorts": "skirts",
    "skorts": "skirts",
    # ========================================================
    # Jumpsuits / Rompers
    # ========================================================
    "jumpsuits, rompers & overalls": "jumpsuits_rompers",
    "jumpsuits & rompers": "jumpsuits_rompers",
    "jumpsuits": "jumpsuits_rompers",
    "rompers": "jumpsuits_rompers",
    "overalls": "jumpsuits_rompers",
    # ========================================================
    # Activewear
    # ========================================================
    "active": "activewear",
    "active & performance": "activewear",
    "active shirts & tees": "activewear",
    "active base layers": "activewear",
    "base layers & compression": "activewear",
    "tracksuits": "activewear",
    "active tracksuits": "activewear",
    "workout top & bottom sets": "activewear",
    # ========================================================
    # Swimwear
    # ========================================================
    "swimsuits & cover ups": "swimwear",
    "swim": "swimwear",
    "swimwear": "swimwear",
    "women's swimwear": "swimwear",
    "men's swimwear": "swimwear",
    "girls' swimwear": "swimwear",
    "boys' swimwear": "swimwear",
    "bikinis": "swimwear",
    "one-pieces": "swimwear",
    "one pieces": "swimwear",
    "tankinis": "swimwear",
    "competitive swimwear": "swimwear",
    "cover-ups": "swimwear",
    # ========================================================
    # Underwear / Sleepwear
    # ========================================================
    "underwear": "underwear",
    "briefs": "underwear",
    "boxer briefs": "underwear",
    "boxers": "underwear",
    "panties": "underwear",
    "lingerie": "lingerie",
    "lingerie sets": "lingerie",
    "lingerie, sleep & lounge": "lingerie_sleepwear",
    "sleep & lounge": "sleepwear",
    "sleepwear & robes": "sleepwear",
    "pajama sets": "sleepwear",
    "nightgowns & sleepshirts": "sleepwear",
    "nightgowns": "sleepwear",
    "robes": "sleepwear",
    "sports bras": "bras",
    "bras": "bras",
    "everyday bras": "bras",
    "nursing & maternity bras": "bras",
    # ========================================================
    # Shoes
    # ========================================================
    "shoes": "shoes",
    "fashion sneakers": "sneakers",
    "sneakers": "sneakers",
    "girls sneakers": "sneakers",
    "boys sneakers": "sneakers",
    "running": "running_shoes",
    "road running": "running_shoes",
    "trail running": "running_shoes",
    "walking": "walking_shoes",
    "flats": "flats",
    "pumps": "pumps",
    "sandals": "sandals",
    "heeled sandals": "sandals",
    "sport sandals": "sandals",
    "sport sandals & slides": "sandals",
    "flip-flops": "sandals",
    "slides": "sandals",
    "platforms & wedges": "wedges",
    "boots": "boots",
    "hiking boots": "hiking_boots",
    "snow boots": "winter_boots",
    "industrial & construction boots": "work_boots",
    "ankle & bootie": "boots",
    "chelsea": "boots",
    "combat": "boots",
    "chukka": "boots",
    "slippers": "slippers",
    "loafers & slip-ons": "loafers",
    "loafers": "loafers",
    "oxfords": "oxfords",
    "oxford & derby": "oxfords",
    "mules & clogs": "mules_clogs",
    "clogs & mules": "mules_clogs",
    "water shoes": "water_shoes",
    "hiking shoes": "hiking_shoes",
    # ========================================================
    # Boot Shop special nodes
    # ========================================================
    "riding": "boots",
    "western": "boots",
    "shearling": "boots",
    "motorcycle": "boots",
    "snow & cold weather": "winter_boots",
    "rain": "rain_boots",
    "work & safety": "work_boots",
    "hiking & trekking": "hiking_boots",
    # ========================================================
    # Jewelry
    # ========================================================
    "earrings": "earrings",
    "drop & dangle": "earrings",
    "stud": "earrings",
    "studs": "earrings",
    "hoop": "earrings",
    "necklaces": "necklaces",
    "pendant necklaces": "necklaces",
    "necklaces & pendants": "necklaces",
    "chains": "necklaces",
    "chokers": "necklaces",
    "bracelets": "bracelets",
    "bangle": "bracelets",
    "charm bracelets": "bracelets",
    "charms & charm bracelets": "bracelets",
    "rings": "rings",
    "wedding rings": "rings",
    "engagement rings": "rings",
    "wedding bands": "rings",
    "promise rings": "rings",
    "body jewelry": "body_jewelry",
    "piercing jewelry": "body_jewelry",
    "brooches & pins": "brooches_pins",
    "jewelry sets": "jewelry_sets",
    "bridal sets": "jewelry_sets",
    "pendants & coins": "pendants",
    "pendants only": "pendants",
    "pendants": "pendants",
    "anklets": "anklets",
    "cuff links": "cufflinks",
    "tie clips": "tie_accessories",
    "loose gemstones": "gemstones",
    "gemstones": "gemstones",
    # ========================================================
    # Watches
    # ========================================================
    "watches": "watches",
    "wrist watches": "watches",
    "pocket watches": "watches",
    "sport watches": "watches",
    "fashion watches": "watches",
    "smartwatches": "smartwatches",
    "smart watches": "smartwatches",
    # ========================================================
    # Bags
    # ========================================================
    "handbags & wallets": "bags_wallets",
    "handbags & shoulder bags": "handbags",
    "crossbody bags": "handbags",
    "shoulder bags": "handbags",
    "top-handle bags": "handbags",
    "hobo bags": "handbags",
    "clutches": "handbags",
    "clutches & evening bags": "handbags",
    "evening bags": "handbags",
    "satchels": "handbags",
    "totes": "handbags",
    "travel totes": "handbags",
    "wallets": "wallets",
    "wallets, card cases & money organizers": "wallets",
    "wallets & money organizers": "wallets",
    "card & id cases": "wallets",
    "card cases": "wallets",
    "passport wallets": "wallets",
    "travel wallets": "wallets",
    "backpacks": "backpacks",
    "fashion backpacks": "backpacks",
    "casual daypacks": "backpacks",
    "kids' backpacks": "backpacks",
    "gym bags": "duffel_bags",
    "travel duffels": "duffel_bags",
    "sports duffels": "duffel_bags",
    "messenger bags": "messenger_bags",
    "briefcases": "briefcases",
    "waist packs": "waist_packs",
    "luggage": "luggage",
    "suitcases": "luggage",
    "carry-ons": "luggage",
    "luggage sets": "luggage",
    # ========================================================
    # Travel accessories
    # ========================================================
    "travel accessories": "travel_accessories",
    "packing organizers": "travel_accessories",
    "luggage tags & handle wraps": "travel_accessories",
    "luggage tags": "travel_accessories",
    "luggage straps": "travel_accessories",
    "passport covers": "travel_accessories",
    # ========================================================
    # Jewelry storage
    # ========================================================
    "jewelry boxes & organizers": "jewelry_storage",
    "jewelry boxes": "jewelry_storage",
    "jewelry trays": "jewelry_storage",
    "jewelry towers": "jewelry_storage",
    "jewelry armoires": "jewelry_storage",
    # ========================================================
    # Accessories
    # ========================================================
    "hats & caps": "hats",
    "baseball caps": "hats",
    "skullies & beanies": "hats",
    "beanies & knit hats": "hats",
    "sun hats": "hats",
    "fedoras": "hats",
    "headwear": "hats",
    "fascinators": "hats",
    "scarves & wraps": "scarves",
    "fashion scarves": "scarves",
    "neck gaiters": "scarves",
    "gloves & mittens": "gloves",
    "gloves": "gloves",
    "cold weather gloves": "gloves",
    "belts": "belts",
    "sunglasses": "sunglasses",
    "sunglasses & eyewear accessories": "eyewear",
    "eyewear frames": "eyewear",
    "umbrellas": "umbrellas",
    "ties, cummerbunds & pocket squares": "ties",
    "neckties": "ties",
    "bow ties": "ties",
    "keyrings, keychains & charms": "keychains",
    "keyrings & keychains": "keychains",
    "keychains": "keychains",
    "buttons & pins": "pins_patches",
    "applique patches": "pins_patches",
    "bandanas": "bandanas",
    "suspenders": "suspenders",
    "handkerchiefs": "handkerchiefs",
    "earmuffs": "cold_weather_accessories",
    "sport headbands": "headbands",
    # ========================================================
    # Costumes
    # ========================================================
    "costumes": "costumes",
    "costumes & cosplay apparel": "costumes",
    "women's halloween costumes": "costumes",
    "men's halloween costumes": "costumes",
    "wigs": "costume_accessories",
    "masks": "costume_accessories",
    "accessory sets": "costume_accessories",
    # ========================================================
    # Socks / Hosiery
    # ========================================================
    "socks": "socks",
    "athletic socks": "socks",
    "slipper socks": "socks",
    "calf socks": "socks",
    "no show & liner socks": "socks",
    "casual & dress socks": "socks",
    "socks & hosiery": "socks",
    "socks & tights": "socks",
    "tights": "hosiery",
    "hosiery": "hosiery",
    "sheers": "hosiery",
    "leg warmers": "hosiery",
    # ========================================================
    # Baby / Kids clothing
    # ========================================================
    "clothing sets": "clothing_sets",
    "pant sets": "clothing_sets",
    "short sets": "clothing_sets",
    "skirt sets": "clothing_sets",
    "bodysuits": "bodysuits",
    "footies & rompers": "baby_clothing",
    "footies": "baby_clothing",
    # ========================================================
    # Work / Medical / Suits
    # ========================================================
    "scrub tops": "scrubs",
    "scrub bottoms": "scrubs",
    "scrub sets": "scrubs",
    "suits": "suits",
    "suits & sport coats": "suits",
    "sport coats & blazers": "suits",
    "overalls & coveralls": "workwear",
    # ========================================================
    # Sport-specific clothing
    # ========================================================
    "jerseys": "sportswear",
    "breeches": "sportswear",
}


# ============================================================
# Promotion / merchandising / test detection
# ============================================================

NOISE_PATTERNS = [
    r"\btest\b",
    r"\btesting\b",
    r"\bprime day\b",
    r"\bblack friday\b",
    r"\bgreen monday\b",
    r"\bdeal\b",
    r"\bdeals\b",
    r"\bsale\b",
    r"\bsavings\b",
    r"\bmarkdown",
    r"\bclearance\b",
    r"\boutlet\b",
    r"\bunder \$?\d+",
    r"\bover \$?\d+",
    r"\b\d+%\s+off\b",
    r"\bup to \d+%",
    r"\bshop by designer\b",
    r"\bfeatured brands\b",
    r"\bnew arrivals\b",
    r"\bgift guide\b",
    r"\beditors?['’]?\s+picks\b",
    r"\bfavorites\b",
    r"\bcohort\b",
    r"\bgreen lit\b",
    r"\bmfn\b",
    r"\bdo not use\b",
    r"\bamazon fashion \d+\b",
    r"\btop \d+\b",
    r"\bmost[- ]loved\b",
    r"\bprime wardrobe\b",
    r"\bshopbop\b",
    r"\bwestlake\b",
    # additional obvious infrastructure/promo nodes
    r"\bbusiness pricing\b",
    r"\bbulk buying\b",
    r"\bcase packs\b",
    r"\bno title match\b",
]


NOISE_REGEX = [re.compile(pattern, re.IGNORECASE) for pattern in NOISE_PATTERNS]


def is_noise_node(value):

    value = normalize(value)

    if not value:
        return True

    for pattern in NOISE_REGEX:
        if pattern.search(value):
            return True

    return False


# ============================================================
# Path parser
# ============================================================


def split_path(path_string):

    return [x.strip() for x in path_string.split(">") if x.strip()]


# ============================================================
# Audience detection
#
# IMPORTANT:
#
# Specific audience beats generic parent nodes.
#
# Baby > Baby Girls
#     -> baby_girls
#
# Kids & Baby > Girls
#     -> girls
#
# This avoids the V1 first-match problem.
# ============================================================


def detect_audience(nodes):

    normalized_nodes = [normalize(node) for node in nodes]

    # --------------------------------------------------------
    # Exact-node detection by specificity
    # --------------------------------------------------------

    priority_groups = [
        (
            {
                "baby girls",
                "baby girls'",
            },
            "baby_girls",
        ),
        (
            {
                "baby boys",
                "baby boys'",
            },
            "baby_boys",
        ),
        (
            {
                "girls",
                "girl",
                "girls'",
            },
            "girls",
        ),
        (
            {
                "boys",
                "boy",
                "boys'",
            },
            "boys",
        ),
        (
            {
                "women",
                "woman",
                "women's",
                "womens",
            },
            "women",
        ),
        (
            {
                "men",
                "man's",
                "men's",
                "mens",
            },
            "men",
        ),
        (
            {
                "baby",
                "unisex baby clothing",
            },
            "baby",
        ),
        (
            {
                "kids",
                "kids & baby",
                "kids' clothing",
                "kids apparel",
                "kids_apparel",
            },
            "kids",
        ),
    ]

    for aliases, audience in priority_groups:
        for i, node_norm in enumerate(normalized_nodes):
            if node_norm in aliases:
                return (audience, nodes[i])

    # --------------------------------------------------------
    # Phrase fallback
    # --------------------------------------------------------

    joined = " ".join(normalized_nodes)

    phrase_patterns = [
        (r"\bbaby girls?\b", "baby_girls"),
        (r"\bbaby boys?\b", "baby_boys"),
        (r"\bwomen'?s?\b", "women"),
        (r"\bmen'?s?\b", "men"),
        (r"\bgirls?\b", "girls"),
        (r"\bboys?\b", "boys"),
        (r"\bbaby\b", "baby"),
        (r"\bkids?\b", "kids"),
    ]

    for pattern, audience in phrase_patterns:
        if re.search(pattern, joined):
            return (audience, None)

    return ("unisex", None)


# ============================================================
# Family detection
#
# Right-to-left scanning is intentional.
#
# Example:
#
# Women > Clothing > Dresses > Formal
#
# "Formal" is not the coarse product family.
# Moving left finds "Dresses".
# ============================================================


def detect_family(nodes):

    clean_nodes = [node for node in nodes if not is_noise_node(node)]

    for node in reversed(clean_nodes):
        node_norm = normalize(node)

        if node_norm in FAMILY_ALIASES:
            return (FAMILY_ALIASES[node_norm], node)

    return (None, None)


# ============================================================
# Department sets
# ============================================================

SHOE_FAMILIES = {
    "shoes",
    "sneakers",
    "running_shoes",
    "walking_shoes",
    "flats",
    "pumps",
    "sandals",
    "wedges",
    "boots",
    "hiking_boots",
    "winter_boots",
    "rain_boots",
    "work_boots",
    "slippers",
    "loafers",
    "oxfords",
    "mules_clogs",
    "water_shoes",
    "hiking_shoes",
}


JEWELRY_FAMILIES = {
    "earrings",
    "necklaces",
    "bracelets",
    "rings",
    "body_jewelry",
    "brooches_pins",
    "jewelry_sets",
    "pendants",
    "anklets",
    "cufflinks",
    "tie_accessories",
    "gemstones",
}


WATCH_FAMILIES = {
    "watches",
    "smartwatches",
}


BAG_FAMILIES = {
    "bags_wallets",
    "handbags",
    "wallets",
    "backpacks",
    "duffel_bags",
    "messenger_bags",
    "briefcases",
    "waist_packs",
    "luggage",
}


ACCESSORY_FAMILIES = {
    "hats",
    "scarves",
    "gloves",
    "belts",
    "sunglasses",
    "eyewear",
    "umbrellas",
    "ties",
    "keychains",
    "pins_patches",
    "bandanas",
    "suspenders",
    "handkerchiefs",
    "cold_weather_accessories",
    "headbands",
    "travel_accessories",
    "jewelry_storage",
    "costume_accessories",
}


def detect_department(family):

    if family in SHOE_FAMILIES:
        return "shoes"

    if family in JEWELRY_FAMILIES:
        return "jewelry"

    if family in WATCH_FAMILIES:
        return "watches"

    if family in BAG_FAMILIES:
        return "bags"

    if family in ACCESSORY_FAMILIES:
        return "accessories"

    if family == "costumes":
        return "costumes"

    return "clothing"


# ============================================================
# Canonical
# ============================================================


def build_canonical(audience, family):

    if not family:
        return None

    if audience and audience != "unisex":
        return f"{audience}_{family}"

    return family


# ============================================================
# Query aliases
#
# These are user-query aliases, not raw Amazon browse nodes.
# ============================================================

QUERY_ALIASES = {
    # Dresses
    "dress": "dresses",
    "dresses": "dresses",
    "women's dress": "women_dresses",
    "women's dresses": "women_dresses",
    # Shoes
    "sneaker": "sneakers",
    "sneakers": "sneakers",
    "mens sneakers": "men_sneakers",
    "men's sneakers": "men_sneakers",
    "women's sneakers": "women_sneakers",
    "girls sneakers": "girls_sneakers",
    "boys sneakers": "boys_sneakers",
    "running shoe": "running_shoes",
    "running shoes": "running_shoes",
    "boots": "boots",
    "sandals": "sandals",
    "heels": "pumps",
    # Tops
    "t shirt": "tshirts",
    "t-shirt": "tshirts",
    "tshirts": "tshirts",
    "shirt": "shirts",
    "shirts": "shirts",
    "hoodie": "hoodies_sweatshirts",
    "hoodies": "hoodies_sweatshirts",
    "sweatshirt": "hoodies_sweatshirts",
    # Bottoms
    "jeans": "jeans",
    "leggings": "leggings",
    "pants": "pants",
    "shorts": "shorts",
    "skirt": "skirts",
    "skirts": "skirts",
    # Active
    "activewear": "activewear",
    "workout clothes": "activewear",
    "sportswear": "sportswear",
    # Swim
    "swimsuit": "swimwear",
    "swimwear": "swimwear",
    # Underwear
    "bra": "bras",
    "underwear": "underwear",
    "socks": "socks",
    "tights": "hosiery",
    # Costumes
    "costume": "costumes",
    "costumes": "costumes",
    # Jewelry
    "earrings": "earrings",
    "necklace": "necklaces",
    "necklaces": "necklaces",
    "bracelet": "bracelets",
    "bracelets": "bracelets",
    "ring": "rings",
    "rings": "rings",
    "pendant": "pendants",
    "pendants": "pendants",
    "anklet": "anklets",
    "anklets": "anklets",
    # Watches
    "watch": "watches",
    "watches": "watches",
    # Bags
    "handbag": "handbags",
    "purse": "handbags",
    "wallet": "wallets",
    "backpack": "backpacks",
    "messenger bag": "messenger_bags",
    "briefcase": "briefcases",
    "luggage": "luggage",
    # Accessories
    "hat": "hats",
    "sunglasses": "sunglasses",
    "scarf": "scarves",
    "gloves": "gloves",
    "keychain": "keychains",
    "keychains": "keychains",
    # Work / medical
    "scrubs": "scrubs",
    "scrub top": "scrubs",
    "scrub bottoms": "scrubs",
}


# ============================================================
# Input paths
#
# category_stats.json contains the top category paths.
# Runtime can additionally use routing rules dynamically.
# ============================================================

path_rows = stats.get("top_category_paths", [])


# ============================================================
# Aggregation
# ============================================================

canonical_stats = defaultdict(
    lambda: {
        "count": 0,
        "paths": [],
        "raw_family_nodes": set(),
    }
)


path_mappings = []

unmapped_paths = []

noise_only_paths = []


# ============================================================
# Mapping
# ============================================================

for row in path_rows:
    path_string = row.get("path", "")

    count = int(row.get("count", 0))

    nodes = split_path(path_string)

    # --------------------------------------------------------
    # Remove root
    # --------------------------------------------------------

    working_nodes = [node for node in nodes if normalize(node) not in ROOT_NODES]

    if not working_nodes:
        continue

    # --------------------------------------------------------
    # Remove noise nodes
    # --------------------------------------------------------

    meaningful_nodes = [node for node in working_nodes if not is_noise_node(node)]

    if not meaningful_nodes:
        noise_only_paths.append(
            {
                "path": path_string,
                "count": count,
            }
        )

        continue

    # --------------------------------------------------------
    # Audience
    # --------------------------------------------------------

    audience, audience_source = detect_audience(meaningful_nodes)

    # --------------------------------------------------------
    # Family
    # --------------------------------------------------------

    family, family_source = detect_family(meaningful_nodes)

    # --------------------------------------------------------
    # No reliable family
    #
    # Keep unmapped instead of inventing a broad category.
    # --------------------------------------------------------

    if family is None:
        unmapped_paths.append(
            {
                "path": path_string,
                "count": count,
                "audience": audience,
                "meaningful_nodes": meaningful_nodes,
            }
        )

        continue

    # --------------------------------------------------------
    # Department
    # --------------------------------------------------------

    department = detect_department(family)

    # --------------------------------------------------------
    # Canonical
    # --------------------------------------------------------

    canonical = build_canonical(audience, family)

    # --------------------------------------------------------
    # Path mapping
    # --------------------------------------------------------

    mapping = {
        "raw_path": path_string,
        "count": count,
        "canonical": canonical,
        "audience": audience,
        "department": department,
        "family": family,
        "audience_source": audience_source,
        "family_source": family_source,
    }

    path_mappings.append(mapping)

    # --------------------------------------------------------
    # Aggregate canonical
    # --------------------------------------------------------

    bucket = canonical_stats[canonical]

    bucket["count"] += count

    if family_source:
        bucket["raw_family_nodes"].add(family_source)

    if len(bucket["paths"]) < 20:
        bucket["paths"].append(path_string)


# ============================================================
# Canonical output
# ============================================================

canonical_output = {}


for canonical, data in sorted(canonical_stats.items(), key=lambda x: -x[1]["count"]):
    canonical_output[canonical] = {
        "count": data["count"],
        "raw_family_nodes": sorted(data["raw_family_nodes"]),
        "example_paths": data["paths"],
    }


# ============================================================
# Coverage
# ============================================================

mapped_count = sum(x["count"] for x in path_mappings)

unmapped_count = sum(x["count"] for x in unmapped_paths)

noise_count = sum(x["count"] for x in noise_only_paths)

input_count = mapped_count + unmapped_count + noise_count


# ============================================================
# Output
# ============================================================

output = {
    "meta": {
        "version": "2.0.0",
        "source": STATS_PATH.name,
        "purpose": (
            "Normalize noisy Amazon category "
            "hierarchies into stable coarse "
            "shopping categories for retrieval routing."
        ),
        "strategy": "path_aware_audience_plus_product_family",
        "important": (
            "Do not map by leaf alone. "
            "Audience and product family are "
            "derived from the full category path."
        ),
        "runtime_policy": (
            "Unmapped broad or ambiguous paths "
            "must fall back to lexical/semantic "
            "retrieval rather than hard filtering."
        ),
    },
    "routing": {
        "root_nodes": sorted(ROOT_NODES),
        "audience_aliases": AUDIENCE_ALIASES,
        "family_aliases": FAMILY_ALIASES,
        "query_aliases": QUERY_ALIASES,
    },
    "canonicals": canonical_output,
    "known_path_mappings": path_mappings,
}


# ============================================================
# Report
# ============================================================

report = {
    "meta": {
        "source": STATS_PATH.name,
        "output": OUTPUT_PATH.name,
        "version": "2.0.0",
    },
    "summary": {
        "input_path_frequency": input_count,
        "mapped_path_frequency": mapped_count,
        "unmapped_path_frequency": unmapped_count,
        "noise_only_path_frequency": noise_count,
        "mapped_rate": round(mapped_count / input_count if input_count else 0, 6),
        "unmapped_rate": round(unmapped_count / input_count if input_count else 0, 6),
        "noise_only_rate": round(noise_count / input_count if input_count else 0, 6),
        "canonical_count": len(canonical_output),
        "known_path_mapping_count": len(path_mappings),
    },
    "top_unmapped_paths": sorted(unmapped_paths, key=lambda x: -x["count"])[:500],
    "noise_only_paths": sorted(noise_only_paths, key=lambda x: -x["count"])[:500],
}


# ============================================================
# Save
# ============================================================

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

    f.write("\n")


with REPORT_PATH.open("w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

    f.write("\n")


# ============================================================
# Terminal summary
# ============================================================

print()
print("=" * 75)
print("CATEGORY MAPPING V2 COMPLETE")
print("=" * 75)

print(f"generated: {OUTPUT_PATH}")

print(f"report: {REPORT_PATH}")


print(f"\ncanonical count: {len(canonical_output):,}")

print(f"Known path mappings：{len(path_mappings):,}")

print(f"Mapped frequency：{mapped_count:,}")

print(f"Unmapped frequency：{unmapped_count:,}")

print(f"Noise-only frequency：{noise_count:,}")


if input_count:
    print(f"Mapped rate：{mapped_count / input_count:.2%}")

    print(f"Unmapped rate：{unmapped_count / input_count:.2%}")

    print(f"Noise-only rate：{noise_count / input_count:.2%}")


# ============================================================
# Top canonicals
# ============================================================

print("\nTop 30 canonicals:")

for canonical, data in list(canonical_output.items())[:30]:
    print(f"{canonical:<40}{data['count']:>12,}")


# ============================================================
# Top unmapped
# ============================================================

print("\nTop 40 unmapped paths:")

for row in report["top_unmapped_paths"][:40]:
    print(f"{row['count']:>10,}  {row['path'][:110]}")


# ============================================================
# Top noise
# ============================================================

print("\nTop 20 noise-only paths:")

for row in report["noise_only_paths"][:20]:
    print(f"{row['count']:>10,}  {row['path'][:110]}")


print()
print("=" * 75)
print("done.")
print("=" * 75)
