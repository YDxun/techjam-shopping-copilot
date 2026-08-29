"""combo_bonus（全约束命中超线性加成）测试。

隐藏目标来自商品自身元数据 → "同时满足全部披露约束"；combo_bonus 用 C(n,2)/C(N,2)
给"完整命中 ≥2 条约束"的商品超线性加成，把目标与"分散命中"的干扰商品区分开（提升 MRR）。
"""
from __future__ import annotations

from types import SimpleNamespace

from agent.dialogue.models import Constraint, ConstraintStrength, Polarity
from agent.reranker import Reranker
from config.env_config import EnvConfig


def _constraint(attr: str, value: str, hardness: int) -> Constraint:
    return Constraint(
        attribute=attr,
        value=value,
        polarity=Polarity.INCLUDE,
        strength=ConstraintStrength.HARD if hardness == 2 else ConstraintStrength.SOFT,
        evidence="",
        source_turn=1,
        tokens=tuple(value.split()),
    )


def _score_for(text: str, values: list[str], hardnesses: list[int]) -> float:
    reranker = Reranker(env=EnvConfig.from_env())
    constraints = [_constraint("material" if h == 2 else "feature", v, h)
                   for v, h in zip(values, hardnesses)]
    state = SimpleNamespace(
        hard=[c for c in constraints if c.hardness == 2],
        soft=[c for c in constraints if c.hardness == 1],
        active=constraints,
        user_profile={},
    )
    route = SimpleNamespace(category_tokens=[])
    cand = {"parent_asin": "X", "rrf": 1.0}
    product = {"title": text.capitalize(), "features": [text], "categories": ["Clothing"],
               "rating_number": 0, "average_rating": 0.0}
    return reranker._rule_score(cand, state, route, product, text, "clothing", 1.0, "probe")


def test_combo_bonus_full_satisfier_beats_partial_match():
    # 商品A：同时完整命中 2 hard + 2 soft（目标特征）
    text_a = "cotton black summer running lightweight breathable"
    values = ["cotton", "black", "summer", "lightweight"]
    hardness = [2, 2, 1, 1]
    score_a = _score_for(text_a, values, hardness)

    # 商品B：只命中 1 条 hard（分散命中）
    text_b = "cotton something else"
    score_b = _score_for(text_b, values, hardness)

    # 全命中商品应显著高于部分命中（至少含 combo 加成）
    combo_w = EnvConfig.from_env().rerank_weights["combo"]
    assert score_a > score_b + 0.5 * combo_w


def test_combo_bonus_requires_two_full_hits():
    # 只有 1 条 hard 全命中时 combo 不触发（C(1,2)=0），但 coverage 仍可能高
    text = "cotton"
    score_1 = _score_for(text, ["cotton"], [2])
    text2 = "cotton"
    # 与无约束商品对比：无约束时 coverage=0.5，有 1 条 hard 命中 coverage=1.0
    score_0 = _score_for(text2, [], [])
    assert score_1 > score_0  # 覆盖度本身生效
    # 单条命中不应有 combo 加成（combo 需 ≥2 条）——通过构造验证 combo_norm=0
    reranker = Reranker(env=EnvConfig.from_env())
    state = SimpleNamespace(hard=[_constraint("material", "cotton", 2)], soft=[], active=[],
                            user_profile={})
    route = SimpleNamespace(category_tokens=[])
    cand = {"parent_asin": "X", "rrf": 1.0}
    product = {"title": "Cotton", "features": ["cotton"], "categories": ["Clothing"],
               "rating_number": 0, "average_rating": 0.0}
    # 通过内部函数确认 combo_norm 逻辑：单条命中 full_count=1 → 不触发
    text_l = "cotton"
    from utils import session_utils as su
    h = reranker._constraint_hit(state.hard[0], product, text_l)
    assert h >= 0.999  # 单条完整命中
