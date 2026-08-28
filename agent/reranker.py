"""Pillar I：重排序模块（LLM 语义排序 / 规则融合打分，双保险）。

- 规则打分（永远可用，无 LLM 也可运行）：
    final = 0.50*约束覆盖度 + 0.25*品类匹配 + 0.15*RRF融合分 + 0.05*热度 + 0.05*画像弱先验
- 约束覆盖度：hard 槽位权重 1.0、soft 槽位 0.4；
  短语子串命中给满分，否则按 token 覆盖率给分。
- LLM 重排：注入的统一 LLM 客户端可用且 LLM_RERANK 启用时执行；
  任何异常都回退规则排序，保证离线可用。
- 目标：Pillar IV —— 把目标商品尽量推到 Top-K 靠前（提升 MRR / HitRate@K）。
"""
from __future__ import annotations

import json
import logging
import math

from agent.dialogue.models import RecommendationContext
from agent.intent_router import IntentRoute
from agent.retriever import HybridRetriever
from config.env_config import EnvConfig
from llm.base import DisabledLLMClient, LLMClient, LLMState
from utils import session_utils as su

logger = logging.getLogger(__name__)

# 规则打分权重（可用环境变量微调，默认经验值）
W_COVERAGE = 0.50
W_CATEGORY = 0.25
W_RRF = 0.15
W_POPULARITY = 0.05
W_PROFILE = 0.05


class Reranker:
    """候选池精排：约束覆盖 + 品类匹配 + 融合分 + 可选 LLM 语义重排。"""

    def __init__(self, env: EnvConfig | None = None,
                 llm_client: LLMClient | None = None) -> None:
        self.env = env or EnvConfig.from_env()
        self.llm_client = llm_client if llm_client is not None else DisabledLLMClient()
        self.last_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0}

    # ------------------------------------------------------------------
    def rerank(self, retriever: HybridRetriever, candidates: list[dict],
               state: RecommendationContext, route: IntentRoute, top_k: int, mode: str) -> list[str]:
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        if not candidates:
            return []
        max_rrf = max((c.get("rrf", 0.0) for c in candidates), default=1.0) or 1.0
        scored: list[tuple[float, str, dict]] = []
        for cand in candidates:
            asin = cand["parent_asin"]
            product = retriever.product(asin)
            if product is None:
                continue
            text = retriever.text_lower(asin)
            cat = self._category_text(product)
            score = self._rule_score(cand, state, route, product, text, cat, max_rrf, mode)
            scored.append((score, asin, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        order = [asin for _, asin, _ in scored]

        # 可选 LLM 语义重排（Pillar I 管道末端；失败自动回退）
        if (self.env.llm.rerank_enabled
                and self.llm_client.status.state == LLMState.AVAILABLE
                and len(order) >= 2):
            llm_order = self._llm_rerank(order, retriever, state)
            if llm_order:
                order = llm_order
        return order[:top_k]

    # ------------------------------------------------------------------
    def _rule_score(self, cand: dict, state: RecommendationContext, route: IntentRoute,
                    product: dict, text: str, cat: str, max_rrf: float, mode: str) -> float:
        # 1) 约束覆盖度（核心强信号，Pillar I 硬约束过滤 + Pillar II 槽位）
        hard = state.hard
        soft = state.soft
        cov_numer = 0.0
        cov_denom = 0.0
        for c in hard:
            w = 1.0
            cov_numer += w * self._constraint_hit(c, text)
            cov_denom += w
        for c in soft:
            w = 0.4
            cov_numer += w * self._constraint_hit(c, text)
            cov_denom += w
        coverage = (cov_numer / cov_denom) if cov_denom > 0 else 0.5

        # 2) 品类匹配
        cat_frac = self._category_match(route.category_tokens, cat, text)

        # 3) RRF 融合分归一
        rrf_norm = min(1.0, cand.get("rrf", 0.0) / max_rrf)

        # 4) 热度 + 评分（微调 tie-break，Pillar IV 排序稳定性）
        rating_n = su.safe_float(product.get("rating_number"), 0.0)
        rating_avg = su.safe_float(product.get("average_rating"), 0.0)
        popularity = math.log1p(rating_n) / math.log1p(10000.0) * 0.5 + (rating_avg / 5.0) * 0.5

        # 5) 画像弱先验（Pillar III 长期画像，仅微小加成）
        profile = self._profile_match(state, text)

        score = (W_COVERAGE * coverage + W_CATEGORY * cat_frac + W_RRF * rrf_norm
                 + W_POPULARITY * popularity + W_PROFILE * profile)

        # EXPLOIT 模式：只有"全部活跃约束全覆盖"的商品才给强加成（提升 MRR：把唯一必中项推前）。
        # 不再只看 hard 组——那会让成百上千个仅匹配高频词（如 water resistant）的商品同分。
        if mode == "exploit" and hard and coverage >= 0.999:
            score += 1.0
        return score

    # ------------------------------------------------------------------
    @staticmethod
    def _constraint_hit(c, text: str) -> float:
        """单条约束命中度：短语子串满分；否则 token 覆盖率。"""
        if su.phrase_exists(text, c.value):
            return 1.0
        if c.tokens:
            hit = sum(1 for t in c.tokens if t in text)
            return hit / len(c.tokens)
        return 0.0

    @staticmethod
    def _tokens_hit(tokens, text: str) -> bool:
        return bool(tokens) and all(t in text for t in tokens)

    @staticmethod
    def _category_text(product: dict) -> str:
        cats = product.get("categories") or []
        return " ".join(str(c) for c in cats).lower()

    @staticmethod
    def _category_match(cat_tokens, cat_text: str, full_text: str) -> float:
        if not cat_tokens:
            return 0.5
        frac = sum(1 for t in cat_tokens if t in cat_text) / len(cat_tokens)
        title_bonus = 0.3 if any(t in full_text for t in cat_tokens) else 0.0
        return min(1.0, frac + title_bonus)

    @staticmethod
    def _profile_match(state: RecommendationContext, text: str) -> float:
        tags = [t for t in (state.user_profile or {}).get("preference_tags", []) if isinstance(t, str)]
        if not tags:
            return 0.0
        hits = sum(1 for t in tags if t.lower() in text)
        return min(1.0, hits * 0.25)

    # ------------------------------------------------------------------
    # 可选 LLM 语义重排（Pillar I：共享客户端；无网络/无 key 时自动回退）
    # ------------------------------------------------------------------
    def _llm_rerank(self, order: list[str], retriever: HybridRetriever,
                    state: RecommendationContext) -> list[str]:
        submitted = order[:self.env.llm.rerank_candidates]
        compact_candidates = [self._compact_candidate(retriever.product(asin) or {}, asin)
                              for asin in submitted]
        constraints = "; ".join(str(c.value) for c in state.active) or "no constraints yet"
        payload = json.dumps(
            {"constraints": constraints[:800], "candidates": compact_candidates},
            ensure_ascii=False,
        )
        try:
            result = self.llm_client.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "Rank the submitted shopping candidates. Respond only with JSON: "
                            "{\"ranked_parent_asins\": [\"...\"]}."
                        ),
                    },
                    {"role": "user", "content": payload},
                ],
                temperature=0.0,
                max_tokens=200,
            )
        except Exception as exc:
            logger.warning("[reranker] LLM rerank failed, fallback to rule order: %s", exc)
            return []

        self.last_usage = {
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
        }
        if not result.success:
            return []

        ranked = self._parse_ranked_asins(result.content)
        if not ranked:
            return []
        submitted_set = set(submitted)
        valid: list[str] = []
        for asin in ranked:
            if isinstance(asin, str) and asin in submitted_set and asin not in valid:
                valid.append(asin)
        if not valid:
            return []
        return valid + [asin for asin in submitted if asin not in valid] + order[len(submitted):]

    @staticmethod
    def _compact_candidate(product: dict, asin: str) -> dict[str, str]:
        categories = product.get("categories") or []
        if not isinstance(categories, (list, tuple)):
            categories = [categories]
        features = product.get("features") or []
        if not isinstance(features, (list, tuple)):
            features = [features]
        normalized_features = " ".join(" ".join(str(value) for value in features).split())
        return {
            "parent_asin": asin,
            "title": str(product.get("title") or "")[:240],
            "categories": " ".join(str(value) for value in categories)[:240],
            "features": normalized_features[:800],
        }

    @staticmethod
    def _parse_ranked_asins(content: str) -> list[object] | None:
        text = content.strip()
        if text.startswith("```") and text.endswith("```"):
            newline = text.find("\n")
            if newline == -1:
                return None
            text = text[newline + 1:-3].strip()
        try:
            parsed = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return None
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and isinstance(parsed.get("ranked_parent_asins"), list):
            return parsed["ranked_parent_asins"]
        return None
