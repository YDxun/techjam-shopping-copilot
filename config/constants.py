"""Competition constants and dataset-integrity verification info (Pillar IV: aligned with the official evaluation metrics).

Notes:
- Only the frozen toolkit's data/catalog.jsonl and data/public_set.jsonl are used.
- SHA256 values come from the official released toolkit's local files; overridable via env vars or skipped with SKIP_DATA_VERIFY=1.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CATALOG_PATH = DATA_DIR / "catalog.jsonl"
PUBLIC_SET_PATH = DATA_DIR / "public_set.jsonl"

# ---------------------------------------------------------------------------
# Dataset integrity (frozen toolkit matching the officially released SHA256SUMS)
# ---------------------------------------------------------------------------
EXPECTED_SHA256_CATALOG = "DA979B05A68AF864CB0DCF9EE6A81C010C7E66A57978AD286C7A2E005FC69A67"
EXPECTED_SHA256_PUBLIC_SET = "571359A8A69014C43FC30D39C996C4A28E875DCCC249DFFC707358757BEB16C0"
CATALOG_EXPECTED_ROWS = 50_000
PUBLIC_SET_EXPECTED_ROWS = 200

# ---------------------------------------------------------------------------
# Evaluation / interface alignment (Pillar IV)
# ---------------------------------------------------------------------------
DEFAULT_TOP_K = 10
MAX_TURNS = 10
MISS_TURN_VALUE = 11

# ask_attribute set allowed by the official API contract (docs/agent_api_contract.json)
ALLOWED_ASK_ATTRIBUTES = (
    "category", "material", "color", "size", "style",
    "brand", "budget", "feature", "use_case", "other",
)

# ---------------------------------------------------------------------------
# Domain vocab (used for slot extraction and constraint-type classification)
# ---------------------------------------------------------------------------
MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
COLORS = ("black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange")

STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "what", "matters", "key", "requirement", "around", "about", "your", "our",
    "not", "no", "preference", "additional", "more", "really", "need", "needs",
    "one", "specific", "attribute", "for", "im", "i'm", "still", "exploring",
})

# Clarify-question template rotation (message must be natural language; ask_attribute is the simulator's info channel)
CLARIFY_OPEN_MESSAGES = (
    "Got it — tell me more about what matters to you (material, style, features, color)?",
    "To narrow it down: any preference on material, color, or specific features?",
    "What else is important — fabric, fit, style, or a particular use case?",
)

# Retrieval field weights (Pillar I: weighted multi-field keyword retrieval)
SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")
BM25_FIELD_WEIGHTS = {"title": 6.0, "features": 4.0, "details": 2.5, "categories": 2.5, "store": 1.5, "description": 1.0}
