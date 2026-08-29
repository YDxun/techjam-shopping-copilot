"""Backward-compatible environment facade over the canonical configuration loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from config.loader import load_config
from config.models import (
    AppConfig,
    DecisionConfig,
    DialogueUnderstandingConfig,
    LLMConfig,
)


@dataclass(frozen=True, repr=False)
class EnvConfig:
    """Read-only compatibility facade for existing environment consumers."""

    _app_config: AppConfig

    @classmethod
    def from_env(
        cls,
        path: str | Path | None = None,
        overrides: Mapping[str, Any] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "EnvConfig":
        return cls(_app_config=load_config(path=path, overrides=overrides, environ=environ))

    @property
    def app_config(self) -> AppConfig:
        return self._app_config

    @property
    def llm(self) -> LLMConfig:
        return self._app_config.llm

    @property
    def dialogue_understanding(self) -> DialogueUnderstandingConfig:
        return self._app_config.dialogue_understanding

    @property
    def decision(self) -> DecisionConfig:
        return self._app_config.decision

    @property
    def retrieval_mode(self):
        return self._app_config.retrieval_mode

    @property
    def hard_cue_enabled(self) -> bool:
        return self._app_config.hard_cue_enabled

    @property
    def env_mode(self) -> str:
        return self._app_config.env_mode

    @property
    def llm_backend(self) -> str:
        return self.llm.provider

    @property
    def retrieval_backend(self) -> str:
        return self._app_config.retrieval_backend

    @property
    def top_k(self) -> int:
        return self._app_config.top_k

    @property
    def llm_model(self) -> str:
        profile = self.llm.selected_profile
        return profile.model if profile else ""

    @property
    def openai_api_key(self) -> str:
        if self.llm.provider != "openai":
            return ""
        return self.llm.providers.openai.api_key.reveal()

    @property
    def openai_base_url(self) -> str:
        return self.llm.providers.openai.base_url

    @property
    def embedding_model(self) -> str:
        return self._app_config.embedding_model

    @property
    def reranker_model(self) -> str:
        return self._app_config.reranker_model

    @property
    def blair_offline_embedding_path(self) -> str:
        return self._app_config.blair_offline_embedding_path

    @property
    def blair_query_encoder_model(self) -> str:
        return self._app_config.blair_query_encoder_model

    @property
    def clarify_strategy(self) -> str:
        return self._app_config.clarify_strategy

    @property
    def llm_rerank(self) -> bool:
        return self.llm.rerank_enabled

    @property
    def reranker_model_enabled(self) -> bool:
        return self._app_config.reranker_model_enabled

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
        return self.llm.rerank_candidates

    @property
    def max_constraint_asks(self) -> int:
        return self.decision.max_questions

    @property
    def llm_intent_enabled(self) -> bool:
        return self._app_config.llm_intent_enabled

    @property
    def asset_vocab_expand(self) -> bool:
        return self._app_config.asset_vocab_expand

    @property
    def asset_category_expand(self) -> bool:
        return self._app_config.asset_category_expand

    @property
    def asset_paraphrase(self) -> bool:
        return self._app_config.asset_paraphrase

    @property
    def asset_field_map(self) -> bool:
        return self._app_config.asset_field_map

    @property
    def llm_clarify_enabled(self) -> bool:
        return self._app_config.llm_clarify_enabled

    @property
    def offline(self) -> bool:
        profile = self.llm.selected_profile
        return self.llm.provider == "none" or profile is None or not profile.api_key

    def __repr__(self) -> str:
        return (
            "EnvConfig("
            f"provider={self.llm.provider!r}, model={self.llm_model!r}, "
            f"configured={bool(self.llm.api_key)!r})"
        )

    def summary(self) -> str:
        return (
            f"provider={self.llm.provider} model={self.llm_model} "
            f"configured={bool(self.llm.api_key)} offline={self.offline}"
        )
