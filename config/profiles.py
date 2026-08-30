"""Config-as-data (P3): CONFIG_PROFILES = single source of truth (config_id -> AppConfig overrides).

- Both build_lut.py and RuntimeController read profiles from this table, avoiding drift across
tuning / LUT / runtime;
- `requires` declares each profile's capability dependencies (dense/llm/network/model) for
environment filtering and disclosure;
- `validate()`: checks every LUT config_id exists in this table and its fields are
schema-compatible.
"""
from __future__ import annotations

from typing import Any

CONFIG_PROFILES: dict[str, dict[str, Any]] = {
    "rule_bm25": {
        "overrides": {
            "retrieval_backend": "bm25",
            "fingerprint": {"enable": False},
            "llm": {"rerank_enabled": False},
            "reranker_model_enabled": False,
        },
        "label": "Pure rule offline (baseline; no BLaIR/LLM/model dependency)",
        "requires": {"dense": False, "llm": False, "network": False, "model": False},
    },
    "hybrid_dense": {
        "overrides": {
            "retrieval_backend": "auto",
            "fingerprint": {"enable": False},
            "llm": {"rerank_enabled": False},
            "reranker_model_enabled": False,
        },
        "label": "+BLaIR dense (dense-recover; requires offline npy + transformers)",
        "requires": {"dense": True, "llm": False, "network": False, "model": False},
    },
    "fingerprint_combo": {
        "overrides": {
            "retrieval_backend": "auto",
            "fingerprint": {"enable": True},
            "reranker_model_enabled": False,
        },
        "label": "Best rules + constraint-combination fingerprint + combo",
        "requires": {"dense": True, "llm": False, "network": False, "model": False},
    },
    "text_rerank": {
        "overrides": {
            "retrieval_backend": "auto",
            "llm": {"rerank_enabled": True, "rerank_backend": "text"},
            "reranker_model_enabled": False,
        },
        "label": "qwen3-rerank text rerank (requires DASHSCOPE key + network; auto-fallback without key)",  # noqa: E501
        "requires": {"dense": True, "llm": True, "network": True, "model": False},
    },
    "reranker_model": {
        "overrides": {
            "retrieval_backend": "auto",
            "reranker_model_enabled": True,
            "reranker_model": "thebajajra/RexReranker-0.6B",
        },
        "label": "RexReranker-0.6B cross-encoder (requires local model cache; second opinion in recover mode)",  # noqa: E501
        "requires": {"dense": True, "llm": False, "network": False, "model": True},
    },
}


def profile_ids() -> list[str]:
    return list(CONFIG_PROFILES.keys())


def profile_overrides(config_id: str) -> dict[str, Any]:
    """Get a profile's overrides; unknown config_id returns {} (caller falls back to defaults)."""
    return CONFIG_PROFILES.get(config_id, {}).get("overrides", {})


def requires_met(config_id: str, *, dense: bool, llm: bool, network: bool, model: bool) -> bool:
    """Whether a profile meets the current environment capabilities (for runtime filtering /
        documentation)."""
    profile = CONFIG_PROFILES.get(config_id)
    if profile is None:
        return False
    need = profile.get("requires", {})
    return (
        (not need.get("dense") or dense)
        and (not need.get("llm") or llm)
        and (not need.get("network") or network)
        and (not need.get("model") or model)
    )


def validate_lut(lut: dict[str, Any]) -> list[str]:
    """Validate a LUT: every config_id must exist in CONFIG_PROFILES; returns a list of problems
        (empty = OK)."""
    problems: list[str] = []
    known = profile_ids()
    for fp, env_entry in (lut.get("environments") or {}).items():
        for cfg in env_entry.get("configs", []):
            cid = cfg.get("config_id")
            if cid not in known:
                problems.append(f"{fp}: config_id={cid!r} not in CONFIG_PROFILES")
    return problems
