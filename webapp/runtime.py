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
import hmac
import logging
import os
import secrets
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from agent.main_agent import Agent
from config import constants
from config.env_config import EnvConfig
from config.models import SecretValue
from llm.factory import create_llm_client
from utils import lut as lut_utils
from utils.data_verify import verify_file
from webapp.catalog import CatalogPresenter
from webapp.metrics import UsageRecorder
from webapp.service import SessionManager

logger = logging.getLogger(__name__)
_CACHE_KEY_SALT = secrets.token_bytes(32)

# ---------------------------------------------------------------------------
# Options metadata (frontend renders selectors from this; values must map to EnvConfig)
# ---------------------------------------------------------------------------
DEFAULT_PROVIDERS = {
    "deepseek": {
        "label": "DeepSeek",
        "models": ["deepseek-v4-flash", "deepseek-chat", "deepseek-reasoner"],
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
    "text": {"label": "qwen3-rerank (requires DASHSCOPE_API_KEY)"},
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
    "fingerprint": {"label": "Constraint fingerprint", "field": "fingerprint"},
    "category_expand": {"label": "Category expansion", "field": "asset_category_expand"},
    "paraphrase": {"label": "Paraphrase robustness", "field": "asset_paraphrase"},
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
    """Per-process cache key that changes with secrets without exposing them."""
    safe = {k: v for k, v in cfg.items() if k != "api_key"}
    api_key = str(cfg.get("api_key") or "")
    if api_key:
        safe["api_key_digest"] = hmac.new(
            _CACHE_KEY_SALT,
            api_key.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
    raw = repr(sorted(safe.items()))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def apply_config(base_env: EnvConfig, cfg: dict[str, Any]) -> EnvConfig:
    """Apply web controls without resetting unrelated canonical configuration."""
    provider = (cfg.get("llm_provider") or "none").strip().lower()
    model = (cfg.get("llm_model") or "").strip()
    api_key = (cfg.get("api_key") or "").strip()
    rerank_backend = (cfg.get("rerank_backend") or "none").strip().lower()
    retrieval = (cfg.get("retrieval_backend") or "auto").strip().lower()
    output = cfg.get("output_strategy") or "holdback"
    app = base_env.app_config
    if retrieval not in RETRIEVAL_BACKENDS:
        retrieval = app.retrieval_backend
    if rerank_backend not in RERANK_BACKENDS:
        rerank_backend = "none"
    strat = OUTPUT_STRATEGIES.get(output, OUTPUT_STRATEGIES["holdback"])
    llm_intent = bool(cfg.get("llm_intent") or cfg.get("llm_intent_enabled"))

    profiles = app.llm.providers
    if provider in DEFAULT_PROVIDERS:
        base_profile = getattr(profiles, provider)
        selected_profile = replace(
            base_profile,
            model=model or base_profile.model,
            api_key=SecretValue(api_key) if api_key else base_profile.api_key,
        )
        profiles = replace(profiles, **{provider: selected_profile})
    else:
        provider = "none"
        if rerank_backend == "chat":
            rerank_backend = "none"

    llm = replace(
        app.llm,
        provider=provider,
        rerank_enabled=rerank_backend != "none",
        rerank_backend=rerank_backend if rerank_backend != "none" else "auto",
        providers=profiles,
    )
    app_changes: dict[str, Any] = {
        "skip_data_verify": True,
        "retrieval_backend": retrieval,
        "emit_gate": bool(strat.get("emit_gate")),
        "llm_intent_enabled": llm_intent,
        "fingerprint": replace(
            app.fingerprint,
            enable=bool(cfg.get("fingerprint", True)),
        ),
        "asset_category_expand": bool(cfg.get("category_expand", True)),
        "asset_paraphrase": bool(cfg.get("paraphrase", True)),
        "llm": llm,
    }
    if "emit_fp_confident" in strat:
        app_changes["emit_fp_confident"] = strat["emit_fp_confident"]
        app_changes["emit_margin_confident"] = strat["emit_margin_confident"]
    return EnvConfig(replace(app, **app_changes))


def _usage_context(cfg: dict[str, Any]) -> dict[str, str]:
    """Non-secret config labels attached to each recorded usage event."""
    provider = str(cfg.get("llm_provider") or "none").strip().lower()
    model = str(cfg.get("llm_model") or "").strip()
    rerank_backend = str(cfg.get("rerank_backend") or "none").strip().lower()
    if provider == "none" and rerank_backend in {"text", "auto"}:
        provider = "dashscope"
        model = "qwen3-rerank"
    return {
        "provider": provider,
        "model": model,
        "retrieval_backend": str(cfg.get("retrieval_backend") or "auto").strip().lower(),
        "rerank_backend": rerank_backend,
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
    active_api_key: SecretValue = field(default_factory=SecretValue, repr=False)
    _lock: Any = field(default_factory=lambda: __import__("threading").Lock())

    @classmethod
    def create(
        cls,
        catalog_path: Path,
        env_loader=EnvConfig.from_env,
        verifier=verify_file,
    ) -> "RuntimeManager":
        env = env_loader()
        if not env.skip_data_verify:
            verifier(
                catalog_path,
                constants.EXPECTED_SHA256_CATALOG,
                "catalog.jsonl",
                skip=False,
            )
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
        effective = dict(cfg)
        provider = str(effective.get("llm_provider") or "none").strip().lower()
        active_provider = str(
            self.active_config.get("llm_provider") or "none"
        ).strip().lower()
        if not str(effective.get("api_key") or "").strip() and provider == active_provider:
            effective["api_key"] = self.active_api_key.reveal()
        if (
            not str(effective.get("api_key") or "").strip()
            and provider == self.base_env.llm.provider
        ):
            effective["api_key"] = self.base_env.llm.api_key.reveal()
        env = apply_config(self.base_env, effective)
        key = config_fingerprint(effective)
        with self._lock:
            template = self.cache.get(key)
            if template is None:
                template = self._build_runtime(env, effective)
                if len(self.cache) >= 3:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[key] = template
            runtime = _fresh_runtime(template)
            self.active = runtime
            self.active_key = key
            self.active_config = {k: v for k, v in effective.items() if k != "api_key"}
            self.active_api_key = SecretValue(str(effective.get("api_key") or ""))
        return runtime, key

    def runtime_info(self) -> dict[str, object]:
        """Environment fingerprint, LUT recommendation, active config summary, options."""
        chat_llm_configured = _llm_configured(self.active_config, self.active_api_key)
        qwen_api_key_set = bool(os.environ.get("DASHSCOPE_API_KEY", "").strip())
        qwen_backend_selected = str(
            self.active_config.get("rerank_backend") or "none"
        ).lower() in {"text", "auto"}
        capability_profile = _runtime_capability_profile(self.active)
        if capability_profile is None:
            chat_llm_available = chat_llm_configured
            qwen_available = qwen_api_key_set
        else:
            chat_llm_available = chat_llm_configured and bool(
                getattr(capability_profile, "llm_available", False)
            )
            qwen_available = qwen_api_key_set and bool(
                getattr(capability_profile, "text_rerank_available", False)
            )
        environment_online_available = chat_llm_available or qwen_available
        active_online = chat_llm_available or (qwen_backend_selected and qwen_available)
        fp = lut_utils.env_fingerprint(
            device="cpu",
            dense=_dense_available(),
            llm=environment_online_available,
            network=environment_online_available,
        )
        rec = lut_utils.recommend(fp)
        lut_config: dict[str, Any] | None = None
        if rec is not None:
            lut_config = LUT_FRONTEND_MAP.get(str(rec.get("config_id") or ""))
        provider = self.active_config.get("llm_provider") or "none"
        api_key = bool(self.active_api_key)
        provider_options = _provider_options(self.base_env)
        rerank_options = {name: dict(metadata) for name, metadata in RERANK_BACKENDS.items()}
        rerank_options["text"].update(
            {
                "configured": qwen_api_key_set,
                "available": qwen_available,
                "requires_env": "DASHSCOPE_API_KEY",
            }
        )
        default_model = (
            getattr(self.base_env.llm.providers, provider).model
            if provider in DEFAULT_PROVIDERS
            else ""
        )
        return {
            "fingerprint": fp,
            "lut_recommendation": rec["config_id"] if rec else None,
            "lut_ts": rec.get("technical_score") if rec else None,
            "lut_config": lut_config,
            "active": {
                "config_key": self.active_key,
                "provider": provider,
                "model": self.active_config.get("llm_model") or default_model,
                "rerank_backend": self.active_config.get("rerank_backend") or "none",
                "retrieval_backend": self.active_config.get("retrieval_backend") or "auto",
                "output_strategy": self.active_config.get("output_strategy") or "holdback",
                "llm_intent_enabled": bool(self.active_config.get("llm_intent_enabled")),
                "fingerprint_enabled": bool(self.active_config.get("fingerprint", True)),
                "category_expand_enabled": bool(self.active_config.get("category_expand", True)),
                "paraphrase_enabled": bool(self.active_config.get("paraphrase", True)),
                "api_key_set": api_key,
                "qwen_api_key_set": qwen_api_key_set,
                "qwen_available": qwen_available,
                "chat_llm_available": chat_llm_available,
                "online": active_online,
                "offline": not active_online,
            },
            "providers": provider_options,
            "rerank_backends": rerank_options,
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


def _provider_options(base_env: EnvConfig) -> dict[str, dict[str, Any]]:
    options: dict[str, dict[str, Any]] = {}
    for name, metadata in DEFAULT_PROVIDERS.items():
        configured_model = getattr(base_env.llm.providers, name).model
        models = list(dict.fromkeys([configured_model, *metadata["models"]]))
        options[name] = {**metadata, "models": models}
    return options


def _fresh_runtime(template: object) -> object:
    from webapp.app import WebRuntime

    if not isinstance(template, WebRuntime):
        raise TypeError("cached web runtime template has an invalid type")
    return WebRuntime(template.sessions.fresh(), template.catalog)


def _runtime_capability_profile(runtime: object | None) -> object | None:
    sessions = getattr(runtime, "sessions", None)
    return getattr(sessions, "capability_profile", None)


def _llm_configured(cfg: dict[str, Any], api_key: SecretValue) -> bool:
    return bool(api_key) and cfg.get("llm_provider") in DEFAULT_PROVIDERS
