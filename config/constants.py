"""竞赛常量与数据集完整性校验信息（Pillar IV：与官方评估矩阵对齐）。

说明：
- 只使用竞赛冻结工具包内的 data/catalog.jsonl 与 data/public_set.jsonl。
- SHA256 取自官方发布工具的本地文件，可通过环境变量覆盖或 SKIP_DATA_VERIFY=1 跳过。
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CATALOG_PATH = DATA_DIR / "catalog.jsonl"
PUBLIC_SET_PATH = DATA_DIR / "public_set.jsonl"

# ---------------------------------------------------------------------------
# 数据集完整性（官方发布 SHA256SUMS 对应的冻结工具包）
# ---------------------------------------------------------------------------
EXPECTED_SHA256_CATALOG = "DA979B05A68AF864CB0DCF9EE6A81C010C7E66A57978AD286C7A2E005FC69A67"
EXPECTED_SHA256_PUBLIC_SET = "571359A8A69014C43FC30D39C996C4A28E875DCCC249DFFC707358757BEB16C0"
CATALOG_EXPECTED_ROWS = 50_000
PUBLIC_SET_EXPECTED_ROWS = 200

# ---------------------------------------------------------------------------
# 评估/接口对齐（Pillar IV）
# ---------------------------------------------------------------------------
DEFAULT_TOP_K = 10
MAX_TURNS = 10
MISS_TURN_VALUE = 11

# 官方 API 契约允许的 ask_attribute 集合（docs/agent_api_contract.json）
ALLOWED_ASK_ATTRIBUTES = (
    "category", "material", "color", "size", "style",
    "brand", "budget", "feature", "use_case", "other",
)

# ---------------------------------------------------------------------------
# 领域词表（用于槽位提取与约束类型归类）
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

# 澄清问题模板轮换（message 需为自然语言；ask_attribute 才是模拟器信息通道）
CLARIFY_OPEN_MESSAGES = (
    "Got it — tell me more about what matters to you (material, style, features, color)?",
    "To narrow it down: any preference on material, color, or specific features?",
    "What else is important — fabric, fit, style, or a particular use case?",
)

# 检索字段权重（Pillar I：关键词检索多字段加权）
SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")
BM25_FIELD_WEIGHTS = {"title": 6.0, "features": 4.0, "details": 2.5, "categories": 2.5, "store": 1.5, "description": 1.0}
