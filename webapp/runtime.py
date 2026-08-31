"""Per-config web runtime: build/cache/switch the shopping agent from user-selected options.

Product goal (environment-adaptive): the frontend lets a non-technical user pick a
combination (LLM provider/model, rerank backend, retrieval, output strategy, toggles)
or "Auto (LUT)" which resolves to the best config for the current environment.

Design notes:
- Building an Agent re-indexes the 50k catalog (~30s), so agents are cached by a
  config fingerprint and only rebuilt on switch.
- API keys are held in-memory in the built EnvConfig only; never logged, never
  written to disk, and never included in /api/runtime responses.
- Any provider/model that is unavailable (no key / no network / model not installed)
  degrades to the rule fallback at runtime (RuntimeController + capability probe).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.main_agent import Agent
from config.env_config import EnvConfig
from llm.factory import create_llm_client
from utils import lut as lut_utils
from webapp.catalog import CatalogPresenter
from webapp.metrics import UsageRecorder
from webapp.service import SessionManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Options metadata (frontend renders selectors from this; values must map to EnvConfig)
# ---------------------------------------------------------------------------
DEFAULT_PROVIDERS = {
    "deepseek": {
        "label": "DeepSeek",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "key_env": "DEEPSEEK_API_KEY",
        "requires_key": True,
        "online": True,
    },
    "openai": {
        "label": "OpenAI",
        "models": ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"],
        "key_env": "OPENAI_API_KEY",
        "requires_key": True,
        "online": True,
    },
}
RERANK_BACKENDS = {
    "none": {"label": "Off"},
    "auto": {"label": "Auto (qwen3 → chat → rule)"},
    "text": {"label": "qwen3-rerank (MaaS)"},
    "chat": {"label": "Chat LLM"},
}
RETRIEVAL_BACKENDS = {
    "auto": {"label": "Auto (BLaIR if available)"},
    "bm25": {"label": "BM25 (offline)"},
    "dense": {"label": "Dense (BLaIR)"},
    "hybrid": {"label": "Hybrid (BM25 + dense)"},
}
OUTPUT_STRATEGIES = {
    "holdback": {"label": "Hold-back (default)", "emit_gate": True},
    "full": {"label": "Full Top-10", "emit_gate": False},
    "confident": {
        "label": "Hold-back + confidence",
        "emit_gate": True,
        "emit_fp_confident": 3,
        "emit_margin_confident": 0.10,
    },
}
TOGGLES = {
    "llm_intent": {"label": "LLM intent recognition (needs key)", "field": "llm_intent_enabled"},
    "fingerprint": {"label": "Constraint-combination fingerprint", "field": "fingerprint"},
    "category_expand": {"label": "Category-mapping expansion", "field": "asset_category_expand"},
    "paraphrase": {"label": "Review-paraphrase robustness", "field": "asset_paraphrase"},
}


# LUT config profile id -> frontend-applicable engine fields. The LUT records only
# config_id + measured scores, so this map translates the recommended profile into the
# controls the /api/runtime/config endpoint accepts. Unknown profiles are skipped.
LUT_FRONTEND_MAP: dict[str, dict[str, Any]] = {
    "rule_bm25": {
        "retrieval_backend": "bm25",
        "rerank_backend": "none",
        "output_strategy": "holdback",
        "llm_intent_enabled": False,
    },
    "hybrid_dense": {
        "retrieval_backend": "hybrid",
        "rerank_backend": "none",
        "output_strategy": "holdback",
        "llm_intent_enabled": False,
    },
    "fingerprint_combo": {
        "retrieval_backend": "auto",
        "rerank_backend": "none",
        "output_strategy": "holdback",
        "llm_intent_enabled": False,
    },
    "text_rerank": {
        "retrieval_backend": "auto",
        "rerank_backend": "text",
        "output_strategy": "holdback",
        "llm_intent_enabled": False,
    },
    "reranker_model": {
        "retrieval_backend": "auto",
        "rerank_backend": "auto",
        "output_strategy": "holdback",
        "llm_intent_enabled": False,
    },
}


def config_fingerprint(cfg: dict[str, Any]) -> str:
    """Deterministic key for the agent cache (excluding secrets from the hash input)."""
    safe = {k: v for k, v in cfg.items() if k != "api_key"}
    raw = repr(sorted(safe.items()))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def apply_config(base_env: EnvConfig, cfg: dict[str, Any]) -> EnvConfig:
    """Map a frontend config dict into an EnvConfig (overrides + in-memory env keys)."""
    overrides: dict[str, Any] = {"skip_data_verify": True}
    environ: dict[str, str] = {}

    provider = (cfg.get("llm_provider") or "none").strip().lower()
    model = (cfg.get("llm_model") or "").strip()
    api_key = (cfg.get("api_key") or "").strip()
    rerank_backend = (cfg.get("rerank_backend") or "none").strip().lower()
    retrieval = (cfg.get("retrieval_backend") or "auto").strip().lower()
    output = cfg.get("output_strategy") or "holdback"

    # retrieval
    if retrieval in RETRIEVAL_BACKENDS:
        overrides["retrieval_backend"] = retrieval
    # output strategy / emit gate
    strat = OUTPUT_STRATEGIES.get(output, OUTPUT_STRATEGIES["holdback"])
    overrides["emit_gate"] = bool(strat.get("emit_gate"))
    if "emit_fp_confident" in strat:
        overrides["emit_fp_confident"] = strat["emit_fp_confident"]
        overrides["emit_margin_confident"] = strat["emit_margin_confident"]
    # LLM intent
    llm_intent = bool(cfg.get("llm_intent") or cfg.get("llm_intent_enabled"))
    overrides["llm_intent_enabled"] = llm_intent
    # fingerprint / asset toggles
    overrides["fingerprint"] = {"enable": bool(cfg.get("fingerprint", True))}
    overrides["asset_category_expand"] = bool(cfg.get("category_expand", True))
    overrides["asset_paraphrase"] = bool(cfg.get("paraphrase", True))
    # LLM provider + rerank
    llm: dict[str, Any] = {
        "provider": provider,
        "rerank_enabled": rerank_backend != "none",
        "rerank_backend": rerank_backend if rerank_backend != "none" else "auto",
    }
    if provider in DEFAULT_PROVIDERS:
        profile = DEFAULT_PROVIDERS[provider]
        llm["model"] = model or profile["models"][0]
        if api_key:
            environ[profile["key_env"]] = api_key
    else:
        llm["provider"] = "none"
        llm["rerank_enabled"] = False
    overrides["llm"] = llm

    env = EnvConfig.from_env(overrides=overrides, environ=environ)
    return env


def _usage_context(cfg: dict[str, Any]) -> dict[str, str]:
    """Non-secret config labels attached to each recorded usage event."""
    provider = str(cfg.get("llm_provider") or "none").strip().lower()
    return {
        "provider": provider,
        "model": str(cfg.get("llm_model") or "").strip(),
        "retrieval_backend": str(cfg.get("retrieval_backend") or "auto").strip().lower(),
        "rerank_backend": str(cfg.get("rerank_backend") or "none").strip().lower(),
        "output_strategy": str(cfg.get("output_strategy") or "holdback").strip().lower(),
    }


@dataclass
class RuntimeManager:
    """Build, cache and switch Agent runtimes per config; exposes runtime info."""

    catalog_path: Path
    base_env: EnvConfig
    catalog: CatalogPresenter
    recorder: UsageRecorder = field(default_factory=UsageRecorder.from_env)
    cache: dict[str, object] = field(default_factory=dict)  # fingerprint -> WebRuntime
    active_key: str | None = None
    active: object | None = None
    active_config: dict[str, Any] = field(default_factory=dict)
    _lock: Any = field(default_factory=lambda: __import__("threading").Lock())

    @classmethod
    def create(cls, catalog_path: Path, env_loader=EnvConfig.from_env) -> "RuntimeManager":
        env = env_loader()
        catalog = CatalogPresenter.build(catalog_path)
        return cls(catalog_path=catalog_path, base_env=env, catalog=catalog)

    def _build_runtime(self, env: EnvConfig, cfg: dict[str, Any]) -> object:
        llm_client = create_llm_client(env.llm)
        llm_client.initialize()
        agent = Agent(catalog_path=self.catalog_path, env=env, llm_client=llm_client)
        from webapp.app import WebRuntime

        return WebRuntime(
            sessions=SessionManager(
                agent,
                self.catalog,
                top_k=env.top_k,
                usage_recorder=self.recorder,
                usage_context=_usage_context(cfg),
            ),
            catalog=self.catalog,
        )

    def switch(self, cfg: dict[str, Any]) -> tuple[object, str]:
        """Build (or reuse cached) runtime for cfg; returns (runtime, config_key)."""
        env = apply_config(self.base_env, cfg)
        key = config_fingerprint(cfg)
        with self._lock:
            runtime = self.cache.get(key)
            if runtime is None:
                runtime = self._build_runtime(env, cfg)
                if len(self.cache) > 3:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[key] = runtime
            self.active = runtime
            self.active_key = key
            self.active_config = dict(cfg)
        return runtime, key

    def runtime_info(self) -> dict[str, object]:
        """Environment fingerprint, LUT recommendation, active config summary, options."""
        fp = lut_utils.env_fingerprint(
            device="cpu",
            dense=_dense_available(),
            llm=_llm_configured(self.active_config),
            network=_llm_configured(self.active_config),
        )
        rec = lut_utils.recommend(fp)
        lut_config: dict[str, Any] | None = None
        if rec is not None:
            lut_config = LUT_FRONTEND_MAP.get(str(rec.get("config_id") or ""))
        provider = self.active_config.get("llm_provider") or "none"
        api_key = bool((self.active_config.get("api_key") or "").strip())
        return {
            "fingerprint": fp,
            "lut_recommendation": rec["config_id"] if rec else None,
            "lut_ts": rec.get("technical_score") if rec else None,
            "lut_config": lut_config,
            "active": {
                "config_key": self.active_key,
                "provider": provider,
                "model": self.active_config.get("llm_model")
                or ("deepseek-chat" if provider == "deepseek" else "gpt-4o-mini"),
                "rerank_backend": self.active_config.get("rerank_backend") or "none",
                "retrieval_backend": self.active_config.get("retrieval_backend") or "auto",
                "output_strategy": self.active_config.get("output_strategy") or "holdback",
                "llm_intent_enabled": bool(self.active_config.get("llm_intent_enabled")),
                "fingerprint_enabled": bool(self.active_config.get("fingerprint", True)),
                "category_expand_enabled": bool(self.active_config.get("category_expand", True)),
                "paraphrase_enabled": bool(self.active_config.get("paraphrase", True)),
                "api_key_set": api_key,
                "offline": not api_key,
            },
            "providers": DEFAULT_PROVIDERS,
            "rerank_backends": RERANK_BACKENDS,
            "retrieval_backends": RETRIEVAL_BACKENDS,
            "output_strategies": {k: {"label": v["label"]} for k, v in OUTPUT_STRATEGIES.items()},
            "toggles": {k: {"label": v["label"]} for k, v in TOGGLES.items()},
        }


def _dense_available() -> bool:
    try:
        from utils.blair import BlairEmbeddingStore

        return BlairEmbeddingStore.load(Path("data/offline_blair_embeds.npy")) is not None
    except Exception:
        return False


def _llm_configured(cfg: dict[str, Any]) -> bool:
    return bool((cfg.get("api_key") or "").strip()) and cfg.get("llm_provider") in DEFAULT_PROVIDERS
