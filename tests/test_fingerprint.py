"""约束组合指纹（全目录精确计数）测试。

- 全目录精确计数"同时满足全部活跃约束的商品数"（count），count 越小组合越稀有；
- count==1 置顶 / ≤10 / ≤50 分级加成；>50（约束过泛）不加成（置信度门控）；
- 默认关（FP_ENABLE=False），COMBO_FINGERPRINT_ENABLE=1 开启。
"""

from __future__ import annotations

from agent.dialogue.models import Constraint, ConstraintStrength, Polarity
from agent.reranker import Reranker
from config.env_config import EnvConfig


class FakeRetriever:
    def __init__(self, products: dict[str, dict]) -> None:
        self._products = products

    def iter_products(self):
        return tuple(self._products.values())

    def product(self, asin: str):
        return self._products.get(asin)

    def text_lower(self, asin: str) -> str:
        return self._products[asin].get("_text", "")


def _c(attr: str, value: str, hardness: int) -> Constraint:
    return Constraint(
        attr,
        value,
        Polarity.INCLUDE,
        ConstraintStrength.HARD if hardness == 2 else ConstraintStrength.SOFT,
        "",
        1,
        tuple(value.split()),
    )


def test_fp_bonus_tiers():
    r = Reranker(env=EnvConfig.from_env())
    fp = r._fp
    assert r._fp_bonus(0) == 0.0
    assert r._fp_bonus(1) == fp.bonus_unique
    assert r._fp_bonus(5) == fp.bonus_ten
    assert r._fp_bonus(30) == fp.bonus_fifty
    assert r._fp_bonus(200) == 0.0  # 约束过泛 → 置信度门控不加成


def test_fingerprint_counts_all_constraint_satisfiers():
    env = EnvConfig.from_env(overrides={"fingerprint": {"enable": True}})
    products = {
        "p1": {"parent_asin": "p1", "title": "cotton black shirt", "_text": "cotton black shirt"},
        "p2": {"parent_asin": "p2", "title": "cotton black dress", "_text": "cotton black dress"},
        "p3": {"parent_asin": "p3", "title": "cotton white shirt", "_text": "cotton white shirt"},
    }
    reranker = Reranker(env=env)
    retriever = FakeRetriever(products)

    # active=[cotton, black] -> 同时满足 = {p1,p2}，count=2
    count, sset = reranker._fingerprint(
        retriever, [_c("material", "cotton", 2), _c("color", "black", 1)]
    )
    assert count == 2
    assert sset == {"p1", "p2"}

    # active=[cotton, white] -> 同时满足 = {p3}，count=1（唯一匹配 → 置顶）
    count2, sset2 = reranker._fingerprint(
        retriever, [_c("material", "cotton", 2), _c("color", "white", 1)]
    )
    assert count2 == 1
    assert sset2 == {"p3"}

    # 无活跃约束 → (None, None)（不触发）
    count3, sset3 = reranker._fingerprint(retriever, [])
    assert count3 is None and sset3 is None


def test_fingerprint_enabled_by_default():
    assert EnvConfig.from_env().fingerprint.enable is True  # 默认开（组合指纹经 A/B 提升 MRR）
