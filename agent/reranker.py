"""Pillar I: Reranking module (LLM semantic ranking / rule-based fused scoring, dual safety).

- Rule scoring (always available, runs without an LLM):
    final = 0.50*constraint coverage + 0.25*category match + 0.15*RRF fusion + 0.05*popularity +
    0.05*weak profile prior
- Constraint coverage: hard slots weigh 1.0, soft slots 0.4;
  exact phrase/substring hits get full credit, otherwise credit is proportional to token coverage.
- LLM reranking: runs when the injected unified LLM client is available and LLM_RERANK is enabled;
  any exception falls back to rule ordering, keeping the agent fully offline-capable.
- Goal (Pillar IV): push the target item toward the top of Top-K to raise MRR / HitRate@K.
"""

from __future__ import annotations

import json
import logging
import math
import re

from agent.dialogue.models import RecommendationContext
from agent.intent_router import IntentRoute
from agent.retriever import HybridRetriever
from config.env_config import EnvConfig
from llm.base import DisabledLLMClient, LLMClient, LLMState
from llm.rerank import RerankClient
from utils import field_mapping as fm_utils
from utils import session_utils as su
from utils.circuit_breaker import PhaseCircuitBreaker
from utils.rex_reranker import RexRerankerScorer, is_generation_reranker

logger = logging.getLogger(__name__)

# Rule scoring weights / combo / fingerprint thresholds now live in config (env.rerank_weights /
# env.fingerprint,
# Step 1 exposure: defaults equal current values so behavior is unchanged; the tune harness uses
# overrides).
# Fallback defaults (SimpleNamespace test env / legacy callers): strictly identical to config
# defaults.
_DEFAULT_WEIGHTS = {
    "coverage": 0.50,
    "combo": 0.10,
    "category": 0.25,
    "rrf": 0.15,
    "popularity": 0.05,
    "profile": 0.05,
}
_DEFAULT_FP = type(
    "_FP",
    (),
    {
        "enable": False,
        "bonus_unique": 1.0,
        "bonus_ten": 0.5,
        "bonus_fifty": 0.2,
        "max_count": 50,
    },
)()

BGE_RERANK_CANDIDATES = 50  # bge cross-encoder rerank candidate pool size (matches RERANK_CANDIDATES_NORMAL)  # noqa: E501


class Reranker:
    """Fine-ranking of the candidate pool: constraint coverage + category match + fusion score +
        optional LLM semantic rerank."""

    def __init__(self, env: EnvConfig | None = None, llm_client: LLMClient | None = None) -> None:
        self.env = env or EnvConfig.from_env()
        self.llm_client = llm_client if llm_client is not None else DisabledLLMClient()
        self.last_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0}
        self.last_usage_sources: list[dict[str, object]] = []
        self._bge = None  # lazily loaded bge-reranker instance (None = not loaded / failed)
        self._rerank_client = None  # lazily loaded qwen3-rerank MaaS client
        self._rerank_breaker = PhaseCircuitBreaker(
            "reranker", failure_threshold=2
        )  # P1: model rerank trips after consecutive failures -> fall back to rule ordering
        self._fp_texts: dict[str, str] | None = None  # catalog-wide asin -> text_lower (constraint fingerprint index)  # noqa: E501
        self._fp_satisfy_cache: dict[str, set[str]] = {}  # constraint key -> products satisfying that constraint  # noqa: E501
        self._weights = dict(getattr(env, "rerank_weights", None) or _DEFAULT_WEIGHTS)
        self._fp = getattr(env, "fingerprint", None) or _DEFAULT_FP
        self.last_fp_count: int | None = None  # fingerprint: #products exactly satisfying all active constraints (confidence signal)  # noqa: E501
        self.last_margin: float = 0.0  # rule-score margin between top-1 and top-2 (confidence signal)  # noqa: E501

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
        self.last_usage_sources = []
        if not candidates:
            return []
        max_rrf = max((c.get("rrf", 0.0) for c in candidates), default=1.0) or 1.0

        # Constraint-combination fingerprint (default off): exact catalog count of products
        # satisfying all active constraints,
        # the smaller the count, the rarer the combination -> the match is more likely the target;
        # tiered bonus (count==1 top / <=10 / <=50).
        fp_count: int | None = None
        fp_set: set[str] | None = None
        self.last_fp_count = None
        if self._fp.enable:
            fp_count, fp_set = self._fingerprint(retriever, getattr(state, "active", ()))
            self.last_fp_count = fp_count

        scored: list[tuple[float, str, dict]] = []
        for cand in candidates:
            asin = cand["parent_asin"]
            product = retriever.product(asin)
            if product is None:
                continue
            text = retriever.text_lower(asin)
            cat = self._category_text(product)
            score = self._rule_score(cand, state, route, product, text, cat, max_rrf, mode)
            # Fingerprint bonus: candidate in the exact-satisfier set -> add bonus by global rarity
            if fp_set is not None and asin in fp_set:
                score += self._fp_bonus(fp_count or 0)
            scored.append((score, asin, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        order = [asin for _, asin, _ in scored]

        # Optional text rerank (end of the Pillar I pipeline; enabled by runtime_controller;
        # auto-fallback on failure)
        # Backend: text=qwen3-rerank MaaS (default, replaces the old chat JSON scoring) /
        # chat=legacy LLM /
        #       auto=prefer text, fall back to chat when text is unavailable.
        if (
            use_llm_rerank
            and self.env.llm.rerank_enabled
            and len(order) >= 2
            and not self._rerank_breaker.open  # P1: already tripped -> fall back to rule ordering on the spot  # noqa: E501
        ):
            backend = getattr(self.env.llm, "rerank_backend", "text")
            if backend == "chat":
                if self.llm_client.status.state == LLMState.AVAILABLE:
                    llm_order = self._llm_rerank(order, retriever, state)
                    if llm_order:
                        order = llm_order
                        self._rerank_breaker.record_success()
                    else:
                        self._rerank_breaker.record_failure("chat rerank produced no output")
            else:  # text / auto -> qwen3-rerank text rerank
                text_order = self._text_rerank(order, retriever, state, route)
                if text_order:
                    order = text_order
                    self._rerank_breaker.record_success()
                else:
                    self._rerank_breaker.record_failure("text_rerank produced no output (no key/failure)")  # noqa: E501
                    if backend == "auto" and self.llm_client.status.state == LLMState.AVAILABLE:
                        llm_order = self._llm_rerank(order, retriever, state)
                        if llm_order:
                            order = llm_order
                            self._rerank_breaker.record_success()
        # Optional reranker model (RexReranker/bge, local; environment-aware; auto-fallback on
        # failure).
        # Enabled only in recover mode: full enablement forces semantic ordering onto the aligned
        # rule order (A/B: MRR drops),
        # in recover (miss streak needs broader recall), a "second opinion" rerank of Top-50 is
        # safest.
        if (
            use_reranker_model
            and mode == "recover"
            and len(order) >= 2
            and not self._rerank_breaker.open  # P1: already tripped -> fall back to rule ordering on the spot  # noqa: E501
        ):
            bge_order = self._bge_rerank(order, retriever, state, route)
            if bge_order:
                order = bge_order
                self._rerank_breaker.record_success()
            else:
                self._rerank_breaker.record_failure("reranker_model produced no output")

        # State-machine feedback loop: already-shown (session continued => not the target),
        # soft_demoted / hard_rejected
        # products are confirmed non-targets, so drop them from this turn's output to force
        # exploring new candidates (early-turn hits).
        # Note: had a candidate been the target, the session would already have ended at the hit
        # turn, so excluding these ASINs is safe.
        excluded = set(getattr(state, "evaluation_excluded_asins", None) or ())
        excluded.update(getattr(state, "soft_demoted_asins", None) or ())
        excluded.update(getattr(state, "hard_rejected_asins", None) or ())
        if excluded:
            order = [asin for asin in order if asin not in excluded]
        # Confidence signal: rule-score margin between top-1 and top-2 (lets output gating release
        # full capacity early on high confidence)
        if len(scored) >= 2:
            self.last_margin = scored[0][0] - scored[1][0]
        else:
            self.last_margin = 1.0
        return order[:top_k]

    # ------------------------------------------------------------------
    # Constraint-combination fingerprint (exact catalog count, default off): helper methods
    # ------------------------------------------------------------------
    def _ensure_fp_index(self, retriever) -> None:
        """Lazily cache catalog-wide asin -> text_lower (built once; reused)."""
        if self._fp_texts is not None:
            return
        texts: dict[str, str] = {}
        for product in retriever.iter_products():
            asin = str(product.get("parent_asin"))
            texts[asin] = retriever.text_lower(asin)
        self._fp_texts = texts

    def _fp_satisfiers(self, retriever, value: str) -> set[str]:
        """Products satisfying a single constraint (exact full hit, phrase_exists rule), cached by
            constraint key."""
        key = su.constraint_key(value)
        cached = self._fp_satisfy_cache.get(key)
        if cached is not None:
            return cached
        self._ensure_fp_index(retriever)
        sset = {asin for asin, text in self._fp_texts.items() if su.phrase_exists(text, value)}
        self._fp_satisfy_cache[key] = sset
        return sset

    def _fingerprint(self, retriever, active) -> tuple[int | None, set[str] | None]:
        """Exact catalog count: products satisfying ALL active constraints + the satisfying set.

        Returns (count, satisfied_set); no active constraints or feature off -> (None, None).
        """
        if not active or not self._fp.enable:
            return None, None
        self._ensure_fp_index(retriever)
        sset: set[str] | None = None
        for c in active:
            s = self._fp_satisfiers(retriever, getattr(c, "value", ""))
            sset = s if sset is None else (sset & s)
            if not sset:
                break
        count = len(sset) if sset is not None else 0
        return count, (sset or set())

    def _fp_bonus(self, count: int) -> float:
        """Confidence-gated bonus by global rarity: count==1 top / <=10 / <=50; >max_count no
            bonus."""
        if count <= 0:
            return 0.0
        if count == 1:
            return self._fp.bonus_unique
        if count <= 10:
            return self._fp.bonus_ten
        if count <= self._fp.max_count:
            return self._fp.bonus_fifty
        return 0.0

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
        # 1) Constraint coverage (core strong signal; Pillar I hard filtering + Pillar II slots)
        # + combo_bonus: the hidden target is generated from its own metadata (intent card),
        # "satisfies all disclosed constraints";
        # per-item weighted averaging is a linear signal; here we add a super-linear bonus for
        # "fully hitting >=2 constraints" (C(n,2) normalized),
        #    separating the full-hit target from distractors with scattered hits, which lifts MRR.
        hard = state.hard
        soft = state.soft
        cov_numer = 0.0
        cov_denom = 0.0
        full_count = 0.0  # weighted count of fully-hit constraints (hard=1.0, soft=0.5)
        full_denom = 0.0  # weighted total of all constraints (for normalization)
        for c in hard:
            w = 1.0
            h = self._constraint_hit(c, product, text)
            cov_numer += w * h
            cov_denom += w
            if h >= 0.999:
                full_count += 1.0
            full_denom += 1.0
        for c in soft:
            w = 0.4
            h = self._constraint_hit(c, product, text)
            cov_numer += w * h
            cov_denom += w
            if h >= 0.999:
                full_count += 0.5
            full_denom += 0.5
        coverage = (cov_numer / cov_denom) if cov_denom > 0 else 0.5

        # combo_bonus: triggers only when >=2 constraints are fully hit; C(n,2)/C(N,2) normalization
        # (share of "satisfied constraint pairs" among all pairs; naturally super-linear, full hit =
        # 1.0)
        combo_norm = 0.0
        if full_denom >= 2.0 and full_count >= 2.0:
            combo_norm = (full_count * (full_count - 1.0)) / (full_denom * (full_denom - 1.0))

        # 2) Category match
        cat_frac = self._category_match(route.category_tokens, cat, text)

        # 3) RRF fusion score (normalized)
        rrf_norm = min(1.0, cand.get("rrf", 0.0) / max_rrf)

        # 4) Popularity + rating (fine tie-break; Pillar IV ranking stability)
        rating_n = su.safe_float(product.get("rating_number"), 0.0)
        rating_avg = su.safe_float(product.get("average_rating"), 0.0)
        popularity = math.log1p(rating_n) / math.log1p(10000.0) * 0.5 + (rating_avg / 5.0) * 0.5

        # 5) Weak user-profile prior (Pillar III long-term profile; small bonus only)
        profile = self._profile_match(state, text)

        score = (
            self._weights.get("coverage", 0.5) * coverage
            + self._weights.get("combo", 0.1) * combo_norm
            + self._weights.get("category", 0.25) * cat_frac
            + self._weights.get("rrf", 0.15) * rrf_norm
            + self._weights.get("popularity", 0.05) * popularity
            + self._weights.get("profile", 0.05) * profile
        )

        # EXPLOIT mode: only products covering ALL active constraints get a strong bonus (MRR: push
        # the unique must-hit item up).
        # Not just the hard group -- that would tie hundreds of products matching only
        # high-frequency words (e.g. water resistant).
        if mode == "exploit" and hard and coverage >= 0.999:
            score += 1.0
        return score

    # ------------------------------------------------------------------
    def _constraint_hit(self, c, product: dict, text: str) -> float:
        """Constraint hit score (Pillar I field-aware matching via field_mapping).

        - budget: numeric price check (missing price passes through; 79% missing -> no hard filter);
        - other attributes: keep the legacy full-text logic first (phrase full credit / token
        coverage to preserve recall),
          then layer a field-aware score (authoritative details.<Key> high confidence, missing
          policy pass/soft),
          taking the max -- field awareness only adds, never subtracts, so aligned rule signals are
          preserved.
        """
        if getattr(c, "attribute", "") == "budget":
            # budget -> numeric price check (field_mapping; missing price passes through, no hard
            # filter)
            return fm_utils.constraint_hit("budget", c.value, c.tokens, product=product)
        # Other attributes keep full-text phrase/token logic: field_mapping A/B shows pure-field or
        # layered scoring
        # disturbs the aligned rule order (MRR 0.619->0.597/0.610), so they are not used.
        if su.phrase_exists(text, c.value):
            return 1.0
        # Punctuation-insensitive matching: handles mechanical mismatches like rendered "key value"
        # vs constraint "key: item"
        loose_text = re.sub(r"[^a-z0-9]+", " ", text)
        loose_key = re.sub(r"[^a-z0-9]+", " ", su.constraint_key(c.value))
        if len(loose_key) >= 3 and loose_key in loose_text:
            return 0.85
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
    # Optional bge-reranker-v2-m3 cross-encoder rerank (local model; auto-fallback to rule ordering
    # on failure)
    # ------------------------------------------------------------------
    def _ensure_reranker_model(self):
        """Lazily load the reranker model (dispatched by model name; device auto cuda/cpu).

        - RexReranker-0.6B / Qwen3-Reranker (e-commerce generative rerank): transformers yes/no
        scoring;
        - BAAI/bge-reranker-v2-m3 etc.: FlagEmbedding cross-encoder.
        Any load failure -> None, caller falls back to rule ordering (environment-aware).
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
            logger.warning("[reranker] reranker model unavailable (%s) -> skip model rerank", exc)
            self._bge = None
        return self._bge

    def _bge_rerank(
        self,
        order: list[str],
        retriever: HybridRetriever,
        state: RecommendationContext,
        route: IntentRoute,
    ) -> list[str]:
        """Rerank the top BGE_RERANK_CANDIDATES with the reranker model (RexReranker-0.6B /
            bge-reranker-v2-m3)."""
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
            if hasattr(model, "score_pairs"):  # RexReranker/Qwen3 generative
                scores = model.score_pairs(pairs)
            else:  # FlagEmbedding cross-encoder
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
        except Exception as exc:  # OOM / other exceptions -> degrade to rule ordering
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
    # Optional qwen3-rerank text rerank (Alibaba Cloud MaaS /reranks; environment-aware, fallback to
    # rule ordering on failure)
    # ------------------------------------------------------------------
    def _ensure_text_rerank(self):
        """Lazily create the qwen3-rerank MaaS client (missing key/base_url -> disabled, no network
            calls)."""
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
            logger.warning("[reranker] qwen3-rerank client init failed (%s) -> fall back to rule ordering", exc)  # noqa: E501
            self._rerank_client = None
        return self._rerank_client

    def _text_rerank(self, order: list[str], retriever: HybridRetriever, state, route) -> list[str]:
        """Rerank the top rerank_candidates by query relevance with qwen3-rerank; on failure keep
            the original order."""
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
        self.last_usage = dict(client.last_usage)
        self.last_usage_sources.append(
            {
                "provider": "dashscope",
                "model": client.status.model,
                **self.last_usage,
                "online": True,
            }
        )
        if not results:
            return []
        ranked = [submitted[r.index] for r in results if 0 <= r.index < len(submitted)]
        return ranked + [a for a in order if a not in ranked]

    # ------------------------------------------------------------------
    # Optional LLM semantic rerank (Pillar I: shared client; auto-fallback without network/key)
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

        call_usage = {
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
        }
        self.last_usage = {
            "prompt_tokens": self.last_usage["prompt_tokens"] + call_usage["prompt_tokens"],
            "completion_tokens": (
                self.last_usage["completion_tokens"] + call_usage["completion_tokens"]
            ),
        }
        self.last_usage_sources.append(
            {
                "provider": result.provider,
                "model": result.model,
                **call_usage,
                "online": True,
            }
        )
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
