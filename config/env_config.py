"""Backward-compatible environment facade over the canonical configuration loader."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from config.loader import load_config
from config.models import AppConfig, LLMConfig


@dataclass(frozen=True, repr=False)
class EnvConfig:
    """Read-only compatibility facade for existing environment consumers."""

    _app_config: AppConfig
    _openai_api_key: str = field(default="", repr=False)
    _openai_base_url: str = field(default="", repr=False)

    @classmethod
    def from_env(
        cls,
        path: str | Path | None = None,
        overrides: Mapping[str, Any] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "EnvConfig":
        source = os.environ if environ is None else environ
        return cls(
            _app_config=load_config(path=path, overrides=overrides, environ=source),
            _openai_api_key=source.get("OPENAI_API_KEY", "").strip(),
            _openai_base_url=source.get("OPENAI_BASE_URL", "").strip(),
        )

    @property
    def app_config(self) -> AppConfig:
        return self._app_config

    @property
    def llm(self) -> LLMConfig:
        return self._app_config.llm

    @property
    def env_mode(self) -> str:
        return self._app_config.env_mode

    @property
    def llm_backend(self) -> str:
        return self._app_config.llm_backend

    @property
    def retrieval_backend(self) -> str:
        return self._app_config.retrieval_backend

    @property
    def top_k(self) -> int:
        return self._app_config.top_k

    @property
    def llm_model(self) -> str:
        return self.llm.model

    @property
    def openai_api_key(self) -> str:
        return self._openai_api_key

    @property
    def openai_base_url(self) -> str:
        return self._openai_base_url

    @property
    def embedding_model(self) -> str:
        return self._app_config.embedding_model

    @property
    def reranker_model(self) -> str:
        return self._app_config.reranker_model

    @property
    def clarify_strategy(self) -> str:
        return self._app_config.clarify_strategy

    @property
    def llm_rerank(self) -> bool:
        return self._app_config.llm_rerank

    @property
    def override_erase(self) -> bool:
        return self._app_config.override_erase

    @property
    def skip_data_verify(self) -> bool:
        return self._app_config.skip_data_verify

    @property
    def sample_limit(self) -> int | None:
        return self._app_config.sample_limit

    @property
    def output_path(self) -> str:
        return self._app_config.output_path

    @property
    def rerank_candidates(self) -> int:
        return self._app_config.rerank_candidates

    @property
    def max_constraint_asks(self) -> int:
        return self._app_config.max_constraint_asks

    @property
    def offline(self) -> bool:
        """Whether neither legacy nor DeepSeek configuration enables remote LLM use."""
        legacy_offline = self.llm_backend in {"none", "local"}
        deepseek_offline = self.llm.provider == "none" or not self.llm.api_key
        return legacy_offline and deepseek_offline

    def __repr__(self) -> str:
        return (
            "EnvConfig("
            f"env_mode={self.env_mode!r}, llm_backend={self.llm_backend!r}, "
            f"retrieval_backend={self.retrieval_backend!r}, top_k={self.top_k!r}, "
            f"llm_provider={self.llm.provider!r}, llm_configured={bool(self.llm.api_key)!r}, "
            f"legacy_openai_configured={bool(self._openai_api_key)!r})"
        )

    def summary(self) -> str:
        return (
            f"ENV_MODE={self.env_mode} LLM_BACKEND={self.llm_backend} "
            f"RETRIEVAL_BACKEND={self.retrieval_backend} TOP_K={self.top_k} "
            f"llm_configured={bool(self.llm.api_key)} offline={self.offline}"
        )
