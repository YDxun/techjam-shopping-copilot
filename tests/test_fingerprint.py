"""Tests for the constraint-combination fingerprint (exact catalog count).

- exact catalog count of "products satisfying all active constraints" (count); the smaller the
count, the rarer the combination;
- tiered bonuses count==1 top / <=10 / <=50; no bonus above 50 (over-general constraints; confidence
gating);
- off by default (FP_ENABLE=False); enabled with COMBO_FINGERPRINT_ENABLE=1.
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
    assert r._fp_bonus(200) == 0.0  # over-general constraints -> confidence gating gives no bonus


def test_fingerprint_counts_all_constraint_satisfiers():
    env = EnvConfig.from_env(overrides={"fingerprint": {"enable": True}})
    products = {
        "p1": {"parent_asin": "p1", "title": "cotton black shirt", "_text": "cotton black shirt"},
        "p2": {"parent_asin": "p2", "title": "cotton black dress", "_text": "cotton black dress"},
        "p3": {"parent_asin": "p3", "title": "cotton white shirt", "_text": "cotton white shirt"},
    }
    reranker = Reranker(env=env)
    retriever = FakeRetriever(products)

    # active=[cotton, black] -> both satisfied = {p1,p2}, count=2
    count, sset = reranker._fingerprint(
        retriever, [_c("material", "cotton", 2), _c("color", "black", 1)]
    )
    assert count == 2
    assert sset == {"p1", "p2"}

    # active=[cotton, white] -> both satisfied = {p3}, count=1 (unique match -> top)
    count2, sset2 = reranker._fingerprint(
        retriever, [_c("material", "cotton", 2), _c("color", "white", 1)]
    )
    assert count2 == 1
    assert sset2 == {"p3"}

    # no active constraints -> (None, None) (not triggered)
    count3, sset3 = reranker._fingerprint(retriever, [])
    assert count3 is None and sset3 is None


def test_fingerprint_enabled_by_default():
    assert EnvConfig.from_env().fingerprint.enable is True  # on by default (fingerprint lifted MRR in A/B)  # noqa: E501
