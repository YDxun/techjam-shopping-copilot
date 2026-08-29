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
from llm.rerank import RerankClient
from utils import field_mapping as fm_utils
from utils import session_utils as su
from utils.rex_reranker import RexRerankerScorer, is_generation_reranker

logger = logging.getLogger(__name__)

# 规则打分权重（可用环境变量微调，默认经验值）
W_COVERAGE = 0.50
W_CATEGORY = 0.25
W_RRF = 0.15
W_POPULARITY = 0.05
W_PROFILE = 0.05
BGE_RERANK_CANDIDATES = 50  # bge 交叉编码重排候选规模（与检索管线 RERANK_CANDIDATES_NORMAL 一致）


class Reranker:
    """候选池精排：约束覆盖 + 品类匹配 + 融合分 + 可选 LLM 语义重排。"""

    def __init__(self, env: EnvConfig | None = None, llm_client: LLMClient | None = None) -> None:
        self.env = env or EnvConfig.from_env()
        self.llm_client = llm_client if llm_client is not None else DisabledLLMClient()
        self.last_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0}
        self._bge = None  # 惰性加载的 bge-reranker 实例（None=未加载/加载失败）
        self._rerank_client = None  # 惰性加载的 qwen3-rerank MaaS 客户端

    # ------------------------------------------------------------------
    def rerank(
        self,
        retriever: HybridRetriever,
        candidates: list[dict],
        state: RecommendationContext,
        route: IntentRoute,
        top_k: int,
        mode: str,
        use_reranker_model: bool = False,
        use_llm_rerank: bool = False,
    ) -> list[str]:
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

        # 可选文本重排（Pillar I 管道末端；runtime_controller 决定是否启用；失败自动回退）
        # 后端：text=qwen3-rerank MaaS（默认，替换原 chat JSON 打分）/ chat=旧 LLM /
        #       auto=text 可用优先，text 失败回退 chat。
        if use_llm_rerank and self.env.llm.rerank_enabled and len(order) >= 2:
            backend = getattr(self.env.llm, "rerank_backend", "text")
            if backend == "chat":
                if self.llm_client.status.state == LLMState.AVAILABLE:
                    llm_order = self._llm_rerank(order, retriever, state)
                    if llm_order:
                        order = llm_order
            else:  # text / auto -> qwen3-rerank 文本重排
                text_order = self._text_rerank(order, retriever, state, route)
                if text_order:
                    order = text_order
                elif backend == "auto" and self.llm_client.status.state == LLMState.AVAILABLE:
                    llm_order = self._llm_rerank(order, retriever, state)
                    if llm_order:
                        order = llm_order
        # 可选重排模型（RexReranker/bge，本地；环境自感知；失败自动回退）。
        # 仅 recover 模式启用：全量启用会把语义排序强加于已对齐的规则排序（A/B 掉 MRR），
        # recover（连 miss 需扩召回）时作"第二意见"精排 Top-50 最安全。
        if use_reranker_model and mode == "recover" and len(order) >= 2:
            bge_order = self._bge_rerank(order, retriever, state, route)
            if bge_order:
                order = bge_order

        # 状态机反馈闭环：已展示过（会话继续 => 非目标）、soft_demoted / hard_rejected
        # 的商品确认不是目标，直接从本轮输出剔除，强制探索新候选（低轮次命中）。
        # 注意：若目标是目标，会话早已在命中回合结束，因此排除这些 asin 是安全的。
        excluded = set(getattr(state, "evaluation_excluded_asins", None) or ())
        excluded.update(getattr(state, "soft_demoted_asins", None) or ())
        excluded.update(getattr(state, "hard_rejected_asins", None) or ())
        if excluded:
            order = [asin for asin in order if asin not in excluded]
        return order[:top_k]

    # ------------------------------------------------------------------
    def _rule_score(
        self,
        cand: dict,
        state: RecommendationContext,
        route: IntentRoute,
        product: dict,
        text: str,
        cat: str,
        max_rrf: float,
        mode: str,
    ) -> float:
        # 1) 约束覆盖度（核心强信号，Pillar I 硬约束过滤 + Pillar II 槽位）
        hard = state.hard
        soft = state.soft
        cov_numer = 0.0
        cov_denom = 0.0
        for c in hard:
            w = 1.0
            cov_numer += w * self._constraint_hit(c, product, text)
            cov_denom += w
        for c in soft:
            w = 0.4
            cov_numer += w * self._constraint_hit(c, product, text)
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

        score = (
            W_COVERAGE * coverage
            + W_CATEGORY * cat_frac
            + W_RRF * rrf_norm
            + W_POPULARITY * popularity
            + W_PROFILE * profile
        )

        # EXPLOIT 模式：只有"全部活跃约束全覆盖"的商品才给强加成（提升 MRR：把唯一必中项推前）。
        # 不再只看 hard 组——那会让成百上千个仅匹配高频词（如 water resistant）的商品同分。
        if mode == "exploit" and hard and coverage >= 0.999:
            score += 1.0
        return score

    # ------------------------------------------------------------------
    def _constraint_hit(self, c, product: dict, text: str) -> float:
        """约束命中度（Pillar I field_mapping 字段感知）。

        - budget：数值价格检查（price 缺失放行，79% 缺失不做硬过滤）；
        - 其它属性：先按旧全文本逻辑（短语满分/token 覆盖，保召回），
          再叠加字段感知分（authoritative details.<Key> 高置信、缺失策略 pass/soft），
          取两者较大值——字段感知只加分不掉分，避免覆盖已对齐的规则信号。
        """
        if getattr(c, "attribute", "") == "budget":
            # budget→price 数值检查（field_mapping；price 缺失放行，不做硬过滤）
            return fm_utils.constraint_hit("budget", c.value, c.tokens, product=product)
        # 其余属性保持全文本短语/token 逻辑：field_mapping A/B 显示纯字段/叠加打分
        # 都会扰动已对齐的规则排序（MRR 0.619→0.597/0.610），故不替换。
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
        tags = [
            t for t in (state.user_profile or {}).get("preference_tags", []) if isinstance(t, str)
        ]
        if not tags:
            return 0.0
        hits = sum(1 for t in tags if t.lower() in text)
        return min(1.0, hits * 0.25)

    # ------------------------------------------------------------------
    # 可选 bge-reranker-v2-m3 交叉编码重排（本地模型；失败自动回退规则排序）
    # ------------------------------------------------------------------
    def _ensure_reranker_model(self):
        """惰性加载重排模型（按模型名分发，device 自动 cuda/cpu）。

        - RexReranker-0.6B / Qwen3-Reranker（电商生成式重排）：transformers yes/no 打分；
        - BAAI/bge-reranker-v2-m3 等：FlagEmbedding 交叉编码。
        任一加载失败 → None，由调用方回退规则排序（环境自感知）。
        """
        if self._bge is not None:
            return self._bge
        model_name = self.env.reranker_model
        try:
            if is_generation_reranker(model_name):
                self._bge = RexRerankerScorer(model_name)
                logger.info("[reranker] RexReranker loaded: %s", model_name)
            else:
                import torch
                from FlagEmbedding import FlagReranker

                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._bge = FlagReranker(model_name, use_fp16=(device == "cuda"), device=device)
                logger.info("[reranker] cross-encoder loaded: %s on %s", model_name, device)
        except Exception as exc:
            logger.warning("[reranker] 重排模型不可用（%s）→ 不使用模型重排", exc)
            self._bge = None
        return self._bge

    def _bge_rerank(
        self,
        order: list[str],
        retriever: HybridRetriever,
        state: RecommendationContext,
        route: IntentRoute,
    ) -> list[str]:
        """用重排模型（RexReranker-0.6B / bge-reranker-v2-m3）对 Top-BGE_RERANK_CANDIDATES 精排。"""
        model = self._ensure_reranker_model()
        if model is None:
            return []
        submitted = order[:BGE_RERANK_CANDIDATES]
        query = self._bge_query_text(state, route)
        pairs: list[tuple[str, str]] = []
        for asin in submitted:
            product = retriever.product(asin) or {}
            pairs.append((query, self._bge_product_text(product)))
        if not any(p[1] for p in pairs):
            return []
        try:
            if hasattr(model, "score_pairs"):  # RexReranker/Qwen3 生成式
                scores = model.score_pairs(pairs)
            else:  # FlagEmbedding 交叉编码
                scores = model.compute_score(pairs, normalize=True)
                if isinstance(scores, float):
                    scores = [scores]
            ranked = [
                a
                for _, a in sorted(
                    zip(scores, submitted, strict=False), key=lambda x: x[0], reverse=True
                )
            ]
            return ranked + [a for a in order if a not in ranked]
        except Exception as exc:  # OOM / 其它异常 → 降级
            logger.warning("[reranker] bge rerank failed, fallback to rule order: %s", exc)
            return []

    @staticmethod
    def _bge_query_text(state: RecommendationContext, route: IntentRoute) -> str:
        parts = [c.value for c in state.active]
        if route.query_terms:
            parts.append(" ".join(route.query_terms))
        return " ".join(parts).strip() or "clothing"

    @staticmethod
    def _bge_product_text(product: dict) -> str:
        parts = [str(product.get("title") or "")]
        features = product.get("features") or []
        if isinstance(features, list):
            parts.extend(str(f) for f in features[:5])
        else:
            parts.append(str(features))
        return " | ".join(p for p in parts if p)

    # ------------------------------------------------------------------
    # 可选 qwen3-rerank 文本重排（阿里云 MaaS /reranks；环境自感知，失败回退规则）
    # ------------------------------------------------------------------
    def _ensure_text_rerank(self):
        """惰性创建 qwen3-rerank MaaS 客户端（key/base_url 缺失→disable，不发网络）。"""
        if self._rerank_client is not None:
            return self._rerank_client
        try:
            client = RerankClient(
                model=self.env.llm.qwen_rerank_model,
                workspace_id=self.env.llm.dashscope_workspace_id,
                base_url=self.env.llm.qwen_rerank_base_url,
            )
            client.initialize()
            self._rerank_client = client
        except Exception as exc:
            logger.warning("[reranker] qwen3-rerank 客户端初始化失败（%s）→ 回退规则排序", exc)
            self._rerank_client = None
        return self._rerank_client

    def _text_rerank(
        self, order: list[str], retriever: HybridRetriever, state, route
    ) -> list[str]:
        """用 qwen3-rerank 对 Top-rerank_candidates 按 query 相关性重排；异常回退原顺序。"""
        client = self._ensure_text_rerank()
        if client is None or not client.available:
            return []
        submitted = order[: self.env.llm.rerank_candidates]
        query = self._bge_query_text(state, route)
        docs: list[str] = []
        for asin in submitted:
            product = retriever.product(asin) or {}
            docs.append(self._bge_product_text(product))
        if not any(docs):
            return []
        results = client.rerank(query, docs, top_n=len(submitted))
        if not results:
            return []
        ranked = [submitted[r.index] for r in results if 0 <= r.index < len(submitted)]
        return ranked + [a for a in order if a not in ranked]

    # ------------------------------------------------------------------
    # 可选 LLM 语义重排（Pillar I：共享客户端；无网络/无 key 时自动回退）
    # ------------------------------------------------------------------
    def _llm_rerank(
        self, order: list[str], retriever: HybridRetriever, state: RecommendationContext
    ) -> list[str]:
        submitted = order[: self.env.llm.rerank_candidates]
        compact_candidates = [
            self._compact_candidate(retriever.product(asin) or {}, asin) for asin in submitted
        ]
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
                            '{"ranked_parent_asins": ["..."]}.'
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
        return valid + [asin for asin in submitted if asin not in valid] + order[len(submitted) :]

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
            text = text[newline + 1 : -3].strip()
        try:
            parsed = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return None
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and isinstance(parsed.get("ranked_parent_asins"), list):
            return parsed["ranked_parent_asins"]
        return None
