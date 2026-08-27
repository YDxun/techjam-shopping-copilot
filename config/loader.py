from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping

from config.models import (
    AppConfig,
    CircuitBreakerConfig,
    LLMConfig,
    ProviderConfig,
    ProviderConfigs,
    RetryConfig,
    SecretValue,
)


DEFAULT_CONFIG_PATH = Path(__file__).with_name("default.json")


class ConfigError(ValueError):
    pass


def load_config(
    path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> AppConfig:
    env = os.environ if environ is None else environ
    selected_path = Path(path) if path is not None else Path(
        env.get("APP_CONFIG_PATH", DEFAULT_CONFIG_PATH)
    )
    data = _load_json_object(selected_path)
    _reject_secret_fields(data, source=str(selected_path))
    merged = _deep_merge(_dataclass_defaults(), data)
    json_provider = _mapping(merged.get("llm"), "llm").get("provider", "deepseek")
    merged = _deep_merge(merged, _environment_overrides(env, json_provider))
    if overrides:
        _reject_secret_fields(overrides, source="explicit overrides")
        merged = _deep_merge(merged, overrides)

    llm_data = _mapping(merged.get("llm"), "llm")
    provider = _string_value(llm_data.get("provider"), "llm.provider")
    api_key = SecretValue(_selected_api_key(env, provider))
    return _build_and_validate(merged, api_key)


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        contents = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ConfigError(f"configuration file does not exist: {path}") from error
    except OSError as error:
        raise ConfigError(f"cannot read configuration file: {path}") from error
    try:
        data = json.loads(contents)
    except json.JSONDecodeError as error:
        raise ConfigError(f"invalid JSON in configuration file: {path}") from error
    if not isinstance(data, dict):
        raise ConfigError(f"configuration file must contain a JSON object: {path}")
    return data


def _reject_secret_fields(data: Mapping[str, Any], source: str) -> None:
    llm = data.get("llm")
    if not isinstance(llm, Mapping):
        return
    if "api_key" in llm:
        raise ConfigError(f"{source} may not set llm.api_key; use DEEPSEEK_API_KEY or OPENAI_API_KEY")
    providers = llm.get("providers")
    if not isinstance(providers, Mapping):
        return
    for provider, profile in providers.items():
        if isinstance(profile, Mapping) and "api_key" in profile:
            key_name = "OPENAI_API_KEY" if provider == "openai" else "DEEPSEEK_API_KEY"
            raise ConfigError(
                f"{source} may not set llm.providers.{provider}.api_key; use {key_name}"
            )


def _dataclass_defaults() -> dict[str, Any]:
    return asdict(AppConfig())


def _deep_merge(base: Mapping[str, Any], changes: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in changes.items():
        existing = result.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(existing, value)
        else:
            result[key] = value
    return result


def _environment_overrides(env: Mapping[str, str], json_provider: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    flat_fields: dict[str, tuple[str, Callable[[str, str], Any]]] = {
        "ENV_MODE": ("env_mode", _parse_text),
        "RETRIEVAL_BACKEND": ("retrieval_backend", _parse_text),
        "TOP_K": ("top_k", _parse_int),
        "EMBEDDING_MODEL": ("embedding_model", _parse_text),
        "RERANKER_MODEL": ("reranker_model", _parse_text),
        "CLARIFY_STRATEGY": ("clarify_strategy", _parse_text),
        "OVERRIDE_ERASE": ("override_erase", _parse_bool),
        "SKIP_DATA_VERIFY": ("skip_data_verify", _parse_bool),
        "SAMPLE_LIMIT": ("sample_limit", _parse_int),
        "OUTPUT_PATH": ("output_path", _parse_text),
        "MAX_CONSTRAINT_ASKS": ("max_constraint_asks", _parse_int),
    }
    for name, (field_name, parser) in flat_fields.items():
        value = _environment_value(env, name)
        if value is not None:
            result[field_name] = parser(value, name)

    provider = _selected_provider_from_environment(env, json_provider)
    if _environment_value(env, "LLM_PROVIDER") is not None or _environment_value(env, "LLM_BACKEND") is not None:
        _set_nested(result, ("llm", "provider"), provider)

    provider_fields: dict[str, tuple[tuple[str, ...], Callable[[str, str], Any]]] = {
        "DEEPSEEK_MODEL": (("llm", "providers", "deepseek", "model"), _parse_text),
        "DEEPSEEK_BASE_URL": (("llm", "providers", "deepseek", "base_url"), _parse_text),
        "OPENAI_MODEL": (("llm", "providers", "openai", "model"), _parse_text),
        "OPENAI_BASE_URL": (("llm", "providers", "openai", "base_url"), _parse_text),
    }
    for name, (field_path, parser) in provider_fields.items():
        value = _environment_value(env, name)
        if value is not None:
            _set_nested(result, field_path, parser(value, name))

    if provider in {"deepseek", "openai"}:
        for name, field in (("LLM_MODEL", "model"), ("LLM_BASE_URL", "base_url")):
            value = _environment_value(env, name)
            if value is not None:
                _set_nested(result, ("llm", "providers", provider, field), _parse_text(value, name))

    nested_fields: dict[str, tuple[tuple[str, ...], Callable[[str, str], Any]]] = {
        "LLM_RERANK": (("llm", "rerank_enabled"), _parse_bool),
        "LLM_RERANK_CANDIDATES": (("llm", "rerank_candidates"), _parse_int),
        "LLM_HEALTH_CHECK_ENABLED": (("llm", "health_check_enabled"), _parse_bool),
        "LLM_CONNECT_TIMEOUT_SECONDS": (("llm", "connect_timeout_seconds"), _parse_float),
        "LLM_TIMEOUT_SECONDS": (("llm", "timeout_seconds"), _parse_float),
        "LLM_MAX_RETRIES": (("llm", "retry", "max_retries"), _parse_int),
        "LLM_RETRY_BASE_DELAY_SECONDS": (("llm", "retry", "base_delay_seconds"), _parse_float),
        "LLM_RETRY_MAX_DELAY_SECONDS": (("llm", "retry", "max_delay_seconds"), _parse_float),
        "LLM_CIRCUIT_BREAKER_FAILURE_THRESHOLD": (("llm", "circuit_breaker", "failure_threshold"), _parse_int),
    }
    for name, (field_path, parser) in nested_fields.items():
        value = _environment_value(env, name)
        if value is not None:
            _set_nested(result, field_path, parser(value, name))
    return result


def _selected_provider_from_environment(env: Mapping[str, str], json_provider: Any) -> str:
    provider = _environment_value(env, "LLM_PROVIDER")
    if provider is not None:
        return provider
    backend = _environment_value(env, "LLM_BACKEND")
    if backend is not None:
        return {"openai": "openai", "none": "none", "local": "none"}.get(backend, backend)
    return json_provider if isinstance(json_provider, str) else "deepseek"


def _selected_api_key(env: Mapping[str, str], provider: str) -> str:
    if provider == "deepseek":
        return _environment_value(env, "DEEPSEEK_API_KEY") or ""
    if provider == "openai":
        return _environment_value(env, "OPENAI_API_KEY") or ""
    return ""


def _environment_value(env: Mapping[str, str], name: str) -> str | None:
    raw = env.get(name, "")
    if not isinstance(raw, str):
        raise ConfigError(f"{name} must be text")
    value = raw.strip()
    return value or None


def _set_nested(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = data
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def _parse_text(value: str, name: str) -> str:
    return value


def _parse_int(value: str, name: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise ConfigError(f"{name} must be an integer") from error


def _parse_float(value: str, name: str) -> float:
    try:
        return float(value)
    except ValueError as error:
        raise ConfigError(f"{name} must be a number") from error


def _parse_bool(value: str, name: str) -> bool:
    values = {"1": True, "true": True, "yes": True, "on": True,
              "0": False, "false": False, "no": False, "off": False}
    try:
        return values[value.lower()]
    except KeyError as error:
        raise ConfigError(
            f"{name} must be one of 1/0, true/false, yes/no, or on/off"
        ) from error


def _build_and_validate(data: Mapping[str, Any], selected_key: SecretValue) -> AppConfig:
    llm_data = _mapping(data.get("llm"), "llm")
    retry_data = _mapping(llm_data.get("retry"), "llm.retry")
    circuit_data = _mapping(llm_data.get("circuit_breaker"), "llm.circuit_breaker")
    providers_data = _mapping(llm_data.get("providers"), "llm.providers")
    provider = _string_value(llm_data.get("provider"), "llm.provider")
    providers = ProviderConfigs(
        deepseek=_build_provider(
            _mapping(providers_data.get("deepseek"), "llm.providers.deepseek"),
            "llm.providers.deepseek", selected_key if provider == "deepseek" else SecretValue(),
        ),
        openai=_build_provider(
            _mapping(providers_data.get("openai"), "llm.providers.openai"),
            "llm.providers.openai", selected_key if provider == "openai" else SecretValue(),
        ),
    )
    llm = LLMConfig(
        provider=provider,
        rerank_enabled=_bool_value(llm_data.get("rerank_enabled"), "llm.rerank_enabled"),
        rerank_candidates=_int_value(llm_data.get("rerank_candidates"), "llm.rerank_candidates"),
        health_check_enabled=_bool_value(llm_data.get("health_check_enabled"), "llm.health_check_enabled"),
        connect_timeout_seconds=_number_value(llm_data.get("connect_timeout_seconds"), "llm.connect_timeout_seconds"),
        timeout_seconds=_number_value(llm_data.get("timeout_seconds"), "llm.timeout_seconds"),
        temperature=_number_value(llm_data.get("temperature"), "llm.temperature"),
        max_tokens=_int_value(llm_data.get("max_tokens"), "llm.max_tokens"),
        retry=RetryConfig(
            max_retries=_int_value(retry_data.get("max_retries"), "llm.retry.max_retries"),
            base_delay_seconds=_number_value(retry_data.get("base_delay_seconds"), "llm.retry.base_delay_seconds"),
            max_delay_seconds=_number_value(retry_data.get("max_delay_seconds"), "llm.retry.max_delay_seconds"),
        ),
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=_int_value(circuit_data.get("failure_threshold"), "llm.circuit_breaker.failure_threshold")
        ),
        providers=providers,
    )
    config = AppConfig(
        env_mode=_string_value(data.get("env_mode"), "env_mode"),
        retrieval_backend=_string_value(data.get("retrieval_backend"), "retrieval_backend"),
        top_k=_int_value(data.get("top_k"), "top_k"),
        embedding_model=_string_value(data.get("embedding_model"), "embedding_model"),
        reranker_model=_string_value(data.get("reranker_model"), "reranker_model"),
        clarify_strategy=_string_value(data.get("clarify_strategy"), "clarify_strategy"),
        override_erase=_bool_value(data.get("override_erase"), "override_erase"),
        skip_data_verify=_bool_value(data.get("skip_data_verify"), "skip_data_verify"),
        sample_limit=_optional_int_value(data.get("sample_limit"), "sample_limit"),
        output_path=_string_value(data.get("output_path"), "output_path"),
        max_constraint_asks=_int_value(data.get("max_constraint_asks"), "max_constraint_asks"),
        llm=llm,
    )
    _validate(config)
    return config


def _build_provider(data: Mapping[str, Any], field: str, api_key: SecretValue) -> ProviderConfig:
    return ProviderConfig(
        model=_non_empty_string(data.get("model"), f"{field}.model"),
        base_url=_non_empty_string(data.get("base_url"), f"{field}.base_url"),
        token_limit_parameter=_string_value(data.get("token_limit_parameter"), f"{field}.token_limit_parameter"),
        supports_temperature=_bool_value(data.get("supports_temperature"), f"{field}.supports_temperature"),
        api_key=api_key,
    )


def _validate(config: AppConfig) -> None:
    _in(config.env_mode, "env_mode", {"dev", "submit"})
    _in(config.retrieval_backend, "retrieval_backend", {"bm25", "dense", "hybrid"})
    _in(config.clarify_strategy, "clarify_strategy", {"other", "attribute"})
    _in(config.llm.provider, "llm.provider", {"none", "deepseek", "openai"})
    _positive(config.top_k, "top_k")
    _positive(config.max_constraint_asks, "max_constraint_asks")
    _positive(config.llm.rerank_candidates, "llm.rerank_candidates")
    _positive(config.llm.connect_timeout_seconds, "llm.connect_timeout_seconds")
    _positive(config.llm.timeout_seconds, "llm.timeout_seconds")
    _positive(config.llm.max_tokens, "llm.max_tokens")
    _non_negative(config.llm.retry.max_retries, "llm.retry.max_retries")
    _non_negative(config.llm.retry.base_delay_seconds, "llm.retry.base_delay_seconds")
    _non_negative(config.llm.retry.max_delay_seconds, "llm.retry.max_delay_seconds")
    _positive(config.llm.circuit_breaker.failure_threshold, "llm.circuit_breaker.failure_threshold")
    if config.llm.retry.max_delay_seconds < config.llm.retry.base_delay_seconds:
        raise ConfigError("llm.retry.max_delay_seconds must be >= llm.retry.base_delay_seconds")
    for name, profile in (("deepseek", config.llm.providers.deepseek), ("openai", config.llm.providers.openai)):
        _in(profile.token_limit_parameter, f"llm.providers.{name}.token_limit_parameter", {"max_tokens", "max_completion_tokens"})


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field} must be an object")
    return value


def _string_value(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{field} must be text")
    return value


def _non_empty_string(value: Any, field: str) -> str:
    value = _string_value(value, field)
    if not value.strip():
        raise ConfigError(f"{field} must not be empty")
    return value


def _bool_value(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field} must be boolean")
    return value


def _int_value(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{field} must be an integer")
    return value


def _optional_int_value(value: Any, field: str) -> int | None:
    if value is None:
        return None
    return _int_value(value, field)


def _number_value(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be a number")
    if not math.isfinite(value):
        raise ConfigError(f"{field} must be a finite number")
    return float(value)


def _in(value: str, field: str, allowed: set[str]) -> None:
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ConfigError(f"{field} must be one of: {choices}")


def _positive(value: int | float, field: str) -> None:
    if value <= 0:
        raise ConfigError(f"{field} must be > 0")


def _non_negative(value: int | float, field: str) -> None:
    if value < 0:
        raise ConfigError(f"{field} must be >= 0")
