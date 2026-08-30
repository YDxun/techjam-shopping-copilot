"""Autonomous decision controller (the agent's "decision layer"): decides how each stage runs from
    the capability probe + config.

Principles:
- All capability switches are off by default (LLM intent/clarify not enabled by default);
- config on + probe available -> truly enabled (environment-adaptive);
- config on but environment unavailable -> automatic degradation (fallback to rules / BM25) with the
reason recorded;
- retrieval_backend supports auto: dense available -> hybrid, otherwise bm25.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from agent.capability_probe import CapabilityProfile
from config.env_config import EnvConfig
from config.profiles import profile_ids, requires_met
from utils import lut as lut_utils

logger = logging.getLogger(__name__)


@dataclass
class RuntimeDecisions:
    """Decisions about how each stage executes (global per run)."""

    retrieval_backend: str = "bm25"  # effective retrieval backend (auto already resolved)
    use_dense: bool = False  # whether the dense channel is enabled
    use_llm_intent: bool = False  # whether intent recognition uses the LLM
    use_llm_clarify: bool = False  # whether clarify decisions use the LLM
    use_llm_rerank: bool = False  # whether reranking is enabled (qwen3-rerank text / chat LLM)
    text_rerank_active: bool = False  # whether qwen3-rerank text rerank is active
    use_reranker_model: bool = False  # whether the bge cross-encoder rerank is usable (FlagEmbedding)  # noqa: E501
    strategy: str = "bm25_rule"  # chosen strategy label (environment-adaptive, see decide())
    strategy_lut: str | None = None  # LUT-recommended optimal config (data-driven; missing -> None falls back to default)  # noqa: E501
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        rerank_label = (
            "qwen3" if self.text_rerank_active else "llm" if self.use_llm_rerank else "rule"
        )
        lut_txt = f" lut={self.strategy_lut}" if self.strategy_lut else ""
        return (
            f"strategy={self.strategy}{lut_txt} "
            f"retrieval={self.retrieval_backend} "
            f"intent={'llm' if self.use_llm_intent else 'rule'} "
            f"clarify={'llm' if self.use_llm_clarify else 'rule'} "
            f"rerank={rerank_label} "
            f"reranker_model={'yes' if self.use_reranker_model else 'no'}"
        )


class RuntimeController:
    """Compile a CapabilityProfile into RuntimeDecisions (called once per session/startup)."""

    def __init__(self, env: EnvConfig, profile: CapabilityProfile) -> None:
        self.env = env
        self.profile = profile

    def decide(self) -> RuntimeDecisions:
        d = RuntimeDecisions()

        # ---- Retrieval backend: auto / explicit config + environment fallback ----
        backend = self.env.retrieval_backend
        if backend == "auto":
            d.retrieval_backend = "hybrid" if self.profile.dense_available else "bm25"
            d.use_dense = self.profile.dense_available
            d.reasons.append(
                "retrieval_backend=auto -> "
                f"{d.retrieval_backend} (dense={'yes' if d.use_dense else 'no'})"
            )
        elif backend in ("hybrid", "dense"):
            d.retrieval_backend = backend if self.profile.dense_available else "bm25"
            d.use_dense = self.profile.dense_available
            if not self.profile.dense_available:
                d.reasons.append(f"retrieval_backend={backend} but dense unavailable -> fallback to bm25")  # noqa: E501
        else:
            d.retrieval_backend = backend
            d.use_dense = False

        # ---- LLM decisions: enabled only when config on && probe available (off by default,
        # environment-adaptive) ----
        llm_ok = self.profile.llm_available
        d.use_llm_intent = self.env.llm_intent_enabled and llm_ok
        d.use_llm_clarify = self.env.llm_clarify_enabled and llm_ok
        # Rerank-backend decision: text=qwen3-rerank MaaS (replaces the old chat JSON scoring) /
        # chat=legacy LLM /
        # auto=prefer text, otherwise fall back to chat; if all fail, fall back to rule ordering.
        rr_backend = self.env.llm.rerank_backend
        if self.env.llm.rerank_enabled:
            if rr_backend == "text":
                d.use_llm_rerank = self.profile.text_rerank_available
                d.text_rerank_active = self.profile.text_rerank_available
                if not self.profile.text_rerank_available:
                    d.reasons.append(
                        "LLM_RERANK=1 but qwen3-rerank unavailable"
                        f" ({self.profile.text_rerank_error or 'not configured'}) -> fallback to rule ordering"  # noqa: E501
                    )
            elif rr_backend == "chat":
                d.use_llm_rerank = llm_ok
                d.text_rerank_active = False
            else:  # auto
                d.use_llm_rerank = self.profile.text_rerank_available or llm_ok
                d.text_rerank_active = self.profile.text_rerank_available
        if (
            self.env.llm_intent_enabled
            or self.env.llm_clarify_enabled
        ) and not llm_ok:
            d.reasons.append(f"LLM configured but unavailable (state={self.profile.llm_state}) -> fallback to rules")  # noqa: E501

        # ---- Cross-encoder reranker model (bge-reranker-v2-m3 / RexReranker): enabled only when
        # config on && probe available ----
        d.use_reranker_model = self.env.reranker_model_enabled and self.profile.reranker_available
        if self.env.reranker_model_enabled and not self.profile.reranker_available:
            d.reasons.append(
                f"RERANKER_MODEL_ENABLE=1 but {self.env.reranker_model} unavailable -> fallback to rule ordering"  # noqa: E501
            )

        # ---- Strategy label: environment-adaptive selection of the "current-best" config (never
        # permanently pure rules) ----
        # Public-set A/B: with BLaIR, hybrid+dense(recover) 0.879 > pure rules 0.876;
        # without BLaIR, bm25 rules (0.8757) are environment-optimal. When the LLM is available and
        # enabled, cascaded intent is a safe net.
        parts = [d.retrieval_backend]
        if d.use_llm_intent:
            parts.append("llm_intent")
        if d.use_reranker_model:
            parts.append("rerank_model")
        if len(parts) == 1:
            parts.append("rule")  # pure rules (no enhancements): bm25_rule / hybrid_rule
        d.strategy = "_".join(parts) if parts else "rule"
        d.reasons.append(
            f"strategy={d.strategy}（dense={'yes' if d.use_dense else 'no'} "
            f"llm={'yes' if d.use_llm_intent else 'no'}）"
        )

        # Step 3: config-environment-performance LUT -- recommend the best config_id by environment
        # fingerprint (data-driven startup default;
        # LUT missing / environment not in table -> None, fall back to the default strategy computed
        # above; safe baseline)
        try:
            fp = lut_utils.env_fingerprint(
                device=self.profile.device,
                dense=self.profile.dense_available,
                llm=self.profile.llm_available,
                network=self.profile.network_available,
            )
            rec = lut_utils.recommend(fp)
            # P3: the LUT recommendation must be a CONFIG_PROFILES profile whose capability
            # requirements the current environment meets
            # (otherwise fall back to default -- avoid profiles that need an LLM but can only fall
            # back in an LLM-less env)
            cid = rec["config_id"] if rec else None
            ok = cid in profile_ids() and requires_met(
                cid,
                dense=self.profile.dense_available,
                llm=self.profile.llm_available,
                network=self.profile.network_available,
                model=self.profile.reranker_available,
            )
            d.strategy_lut = cid if ok else None
            if rec and ok:
                d.reasons.append(
                    f"LUT[{fp}] -> {cid} (ts={rec['technical_score']:.4f})"
                )
            elif rec and not ok:
                d.reasons.append(f"LUT[{fp}] -> {cid} capability not met, fallback to default strategy")  # noqa: E501
        except Exception as exc:  # any exception never blocks startup
            logger.warning("[runtime] LUT recommendation failed (%s) -> fallback to default strategy", exc)  # noqa: E501
            d.strategy_lut = None

        logger.info("[runtime] %s", d.summary())
        return d
