"""Pillar I: dual-track intent routing (high-intent buying track / open browsing track).

- Buying track: hard constraints present ("key requirement"/"what matters") -> high-precision
hard-constraint filtering.
- Browsing track: no hard constraints, still exploring -> diverse dense/generalized recall.
- Output IntentRoute: query terms, category domain, hard-constraint token groups, soft terms;
   downstream retrieval/reranking pick route weights from it (Pillar III adaptive orchestration may
   rewrite weights).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.dialogue.models import RecommendationContext
from config.env_config import EnvConfig
from utils import data_assets as assets_util
from utils import field_mapping as fm


@dataclass
class IntentRoute:
    track: str = "browsing"  # buying / browsing
    confidence: float = 0.5
    category_tokens: list[str] = field(default_factory=list)
    hard_groups: list[tuple[str, ...]] = field(default_factory=list)  # AND within each group
    soft_terms: list[str] = field(default_factory=list)
    query_terms: list[str] = field(default_factory=list)

    @property
    def buying(self) -> bool:
        return self.track == "buying"


class IntentRouter:
    """Lightweight intent detection from state signals (runs without an LLM, satisfying offline
        constraints)."""

    def __init__(self, env: EnvConfig | None = None) -> None:
        self.env = env or EnvConfig.from_env()
        self._assets = assets_util.load_assets()

    # ------------------------------------------------------------------
    def route(self, state: RecommendationContext, mode: str) -> IntentRoute:
        hard = state.hard
        soft = state.soft
        route = IntentRoute()

        route.category_tokens = list(state.category_tokens)
        # Category-mapping expansion (ASSET_CATEGORY_EXPAND): family alias -> product-type tokens,
        # first-turn routing recall
        if self.env.asset_category_expand:
            for token in self._assets.category_expand(state.category_phrase or ""):
                if token not in route.category_tokens:
                    route.category_tokens.append(token)

        # Hard-constraint token groups: each hard constraint is an AND group (strong coverage
        # signal)
        for c in hard:
            if c.tokens:
                route.hard_groups.append(c.tokens)
        # soft constraint terms feed loose retrieval
        for c in soft:
            route.soft_terms.extend(c.tokens)

        # Dual-track decision (Pillar I)
        if len(hard) >= 1:
            route.track = "buying"
            route.confidence = min(0.95, 0.55 + 0.2 * len(hard))
        elif state.buying_or_browsing == "browsing" or state.total_constraints() == 0:
            route.track = "browsing"
            route.confidence = 0.5
        else:
            route.track = "browsing"
            route.confidence = 0.6

        # Query terms = category terms + constraint terms + vocab synonyms (Pillar I multi-route
        # query construction;
        # "say it differently" recall: jumper->sweater, 100% cotton->cotton, expanded via
        # vocab.json)
        route.query_terms = list(dict.fromkeys([*route.category_tokens, *route.soft_terms]))
        for group in route.hard_groups:
            route.query_terms.extend(group)
        # vocab synonym expansion ("say it differently" recall): only when intent_version==1 (no
        # override),
        # both hard and soft are expanded (biggest gains in browsing/buying); after an override
        # version>=2, old prefs are soft,
        # so synonyms are no longer expanded, avoiding old-preference terms polluting the new query
        # (A/B: override MRR 0.769->0.744; version gating optimal).
        if getattr(state, "intent_version", 1) == 1:
            for c in [*hard, *soft]:
                route.query_terms.extend(
                    fm.expand_with_vocab(
                        c.attribute,
                        c.value,
                        use_asset_vocab=self.env.asset_vocab_expand,
                    )
                )
        route.query_terms = list(dict.fromkeys(route.query_terms))[:40]

        # Pillar III adaptation: in RECOVER mode, downgrade hard groups to soft (relax filtering)
        if mode == "recover" and route.hard_groups:
            route.soft_terms.extend(t for g in route.hard_groups for t in g)
            route.hard_groups = []
            route.track = "browsing"
        return route
