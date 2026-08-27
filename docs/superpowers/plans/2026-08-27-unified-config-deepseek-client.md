# Unified Configuration and DeepSeek Client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one validated configuration-loading path and a provider-neutral DeepSeek SDK client that probes availability at startup, retries transient failures, and degrades without changing recommendation behavior.

**Architecture:** A typed `AppConfig` is built by layering checked-in JSON, targeted environment variables, and explicit overrides. The existing `EnvConfig` becomes a compatibility facade over that object. A separate `llm` package owns provider-neutral result/status types, DeepSeek SDK interaction, retries, and a process-local circuit breaker; `run_local_eval.py` initializes it for diagnostics but does not inject it into the Agent.

**Tech Stack:** Python 3.10+, standard-library `dataclasses`, `json`, `unittest`, `unittest.mock`; `openai>=1.0,<2.0` SDK and its `httpx` dependency.

**Spec:** `docs/superpowers/specs/2026-08-27-unified-config-deepseek-client-design.md`

## Global Constraints

- DeepSeek is not used for reranking, intent detection, slot extraction, clarification, response generation, or any business decision in this change.
- Do not modify `evaluator/local_evaluator.py`, public data, the Agent API, or recommendation output.
- The only accepted DeepSeek credential source is `DEEPSEEK_API_KEY`; never log or serialize it.
- Configuration precedence is dataclass defaults < JSON < environment < explicit overrides.
- Importing modules and constructing configuration must not access the network.
- A missing DeepSeek key is a normal offline state.
- Startup health checking performs at most three total attempts: one initial request and two retries.
- Only connection, timeout, rate-limit, and server failures retry; deterministic client errors do not.
- Two consecutive runtime failures open a process-local circuit breaker.
- All automated tests use mocks and make no real API calls.
- Use the repository's `unittest` test style and keep Python 3.10 compatibility.

---

## File Structure

### Create

- `config/default.json` — checked-in non-secret defaults for existing runtime settings and the DeepSeek client.
- `config/models.py` — immutable typed configuration models and redacted LLM representation.
- `config/loader.py` — layered loading, targeted environment conversion, deep merge, and validation.
- `llm/__init__.py` — stable public exports.
- `llm/base.py` — provider-neutral protocol, state, status, error categories, and response types.
- `llm/deepseek.py` — OpenAI-SDK-backed DeepSeek client, retry policy, sanitization, and circuit breaker.
- `llm/factory.py` — provider selection and client construction.
- `tests/test_config_loader.py` — configuration precedence, validation, secret, and compatibility tests.
- `tests/test_deepseek_client.py` — mocked SDK initialization, retry, response, and circuit-breaker tests.
- `tests/test_llm_startup.py` — runner-level startup degradation tests.

### Modify

- `config/env_config.py` — replace direct environment parsing with a compatibility facade over `load_config()`.
- `config/__init__.py` — export the canonical loader and models while retaining `EnvConfig`.
- `run_local_eval.py` — explicitly create and initialize the LLM client before data loading.
- `requirements.txt` — add the actual `openai>=1.0,<2.0` dependency.
- `README.md` — document JSON configuration, environment precedence, DeepSeek status, and the current non-business-integration boundary.

---

### Task 1: Typed Configuration Models and Layered Loader

**Files:**
- Create: `config/default.json`
- Create: `config/models.py`
- Create: `config/loader.py`
- Create: `tests/test_config_loader.py`

**Interfaces:**
- Produces: `load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None, environ: Mapping[str, str] | None = None) -> AppConfig`
- Produces: `AppConfig.llm: LLMConfig`
- Produces: `ConfigError(ValueError)` for invalid files, values, and forbidden secret overrides.
- Produces: immutable `RetryConfig`, `CircuitBreakerConfig`, `LLMConfig`, and `AppConfig` dataclasses.

- [ ] **Step 1: Add failing tests for defaults, precedence, and secret handling**

Create `tests/test_config_loader.py` with temporary JSON files and an injected `environ` mapping so tests never mutate the developer's real environment:

```python
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from config.loader import ConfigError, load_config


class ConfigLoaderTest(unittest.TestCase):
    def write_config(self, root: Path, payload: dict) -> Path:
        path = root / "settings.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_layers_json_environment_and_explicit_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(Path(directory), {
                "top_k": 7,
                "llm": {"model": "json-model", "retry": {"max_retries": 1}},
            })
            config = load_config(
                path=path,
                environ={"LLM_MODEL": "env-model", "DEEPSEEK_API_KEY": "secret-value"},
                overrides={"top_k": 9, "llm": {"retry": {"max_retries": 2}}},
            )
            self.assertEqual(config.top_k, 9)
            self.assertEqual(config.llm.model, "env-model")
            self.assertEqual(config.llm.retry.max_retries, 2)
            self.assertEqual(config.llm.api_key, "secret-value")
            self.assertNotIn("secret-value", repr(config.llm))

    def test_rejects_secret_in_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(Path(directory), {"llm": {"api_key": "forbidden"}})
            with self.assertRaisesRegex(ConfigError, "DEEPSEEK_API_KEY"):
                load_config(path=path, environ={})

    def test_rejects_secret_in_explicit_overrides(self) -> None:
        with self.assertRaisesRegex(ConfigError, "DEEPSEEK_API_KEY"):
            load_config(environ={}, overrides={"llm": {"api_key": "forbidden"}})

    def test_missing_key_is_valid(self) -> None:
        config = load_config(environ={})
        self.assertEqual(config.llm.api_key, "")
        self.assertEqual(config.llm.provider, "deepseek")
```

Add focused cases in the same file for `APP_CONFIG_PATH`, direct path precedence, invalid JSON, missing explicit config path, invalid boolean text, unsupported provider, non-positive timeouts, negative retries, and a non-positive circuit threshold.

- [ ] **Step 2: Run the new tests and verify the loader does not exist**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_config_loader -v
```

Expected: FAIL with an import error for `config.loader`.

- [ ] **Step 3: Add checked-in defaults and immutable models**

Create `config/default.json` with all existing tunable runtime fields and this LLM section:

```json
{
  "env_mode": "dev",
  "llm_backend": "none",
  "retrieval_backend": "bm25",
  "top_k": 10,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "reranker_model": "BAAI/bge-reranker-v2-m3",
  "clarify_strategy": "other",
  "llm_rerank": true,
  "override_erase": false,
  "skip_data_verify": false,
  "sample_limit": null,
  "output_path": "results.json",
  "rerank_candidates": 300,
  "max_constraint_asks": 3,
  "llm": {
    "provider": "deepseek",
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com",
    "health_check_enabled": true,
    "connect_timeout_seconds": 3.0,
    "timeout_seconds": 8.0,
    "temperature": 0.0,
    "max_tokens": 256,
    "retry": {
      "max_retries": 2,
      "base_delay_seconds": 0.5,
      "max_delay_seconds": 1.5
    },
    "circuit_breaker": {
      "failure_threshold": 2
    }
  }
}
```

Create `config/models.py` with these exact public types and field names:

```python
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RetryConfig:
    max_retries: int = 2
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 1.5


@dataclass(frozen=True)
class CircuitBreakerConfig:
    failure_threshold: int = 2


@dataclass(frozen=True, repr=False)
class LLMConfig:
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    health_check_enabled: bool = True
    connect_timeout_seconds: float = 3.0
    timeout_seconds: float = 8.0
    temperature: float = 0.0
    max_tokens: int = 256
    retry: RetryConfig = field(default_factory=RetryConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    api_key: str = field(default="", repr=False)

    def __repr__(self) -> str:
        key_state = "<set>" if self.api_key else "<unset>"
        return (
            f"LLMConfig(provider={self.provider!r}, model={self.model!r}, "
            f"base_url={self.base_url!r}, api_key={key_state})"
        )


@dataclass(frozen=True)
class AppConfig:
    env_mode: str = "dev"
    llm_backend: str = "none"
    retrieval_backend: str = "bm25"
    top_k: int = 10
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    clarify_strategy: str = "other"
    llm_rerank: bool = True
    override_erase: bool = False
    skip_data_verify: bool = False
    sample_limit: int | None = None
    output_path: str = "results.json"
    rerank_candidates: int = 300
    max_constraint_asks: int = 3
    llm: LLMConfig = field(default_factory=LLMConfig)
```

- [ ] **Step 4: Implement layered loading and field-specific validation**

In `config/loader.py`, implement:

```python
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
    merged = _deep_merge(merged, _environment_overrides(env))
    if overrides:
        _reject_secret_fields(overrides, source="explicit overrides")
        merged = _deep_merge(merged, overrides)
    merged.setdefault("llm", {})["api_key"] = env.get("DEEPSEEK_API_KEY", "").strip()
    return _build_and_validate(merged)
```

Use an explicit environment mapping table rather than scanning all variables. Parse booleans only from `1/0`, `true/false`, `yes/no`, and `on/off`; reject other non-empty values. Deep-merge nested `retry` and `circuit_breaker` objects so a single override does not erase sibling defaults.

The flat environment mapping must include the existing variables `ENV_MODE`, `LLM_BACKEND`, `RETRIEVAL_BACKEND`, `TOP_K`, `EMBEDDING_MODEL`, `RERANKER_MODEL`, `CLARIFY_STRATEGY`, `LLM_RERANK`, `OVERRIDE_ERASE`, `SKIP_DATA_VERIFY`, `SAMPLE_LIMIT`, `OUTPUT_PATH`, `RERANK_CANDIDATES`, and `MAX_CONSTRAINT_ASKS`. The nested LLM mapping must include `LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`, `LLM_HEALTH_CHECK_ENABLED`, `LLM_CONNECT_TIMEOUT_SECONDS`, `LLM_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES`, `LLM_RETRY_BASE_DELAY_SECONDS`, `LLM_RETRY_MAX_DELAY_SECONDS`, and `LLM_CIRCUIT_BREAKER_FAILURE_THRESHOLD`. `DEEPSEEK_API_KEY` is handled separately after all non-secret layers.

Validation must accept only:

- `env_mode`: `dev` or `submit`;
- `llm_backend`: `none`, `local`, or `openai` for legacy behavior;
- `retrieval_backend`: `bm25`, `dense`, or `hybrid`;
- `clarify_strategy`: `other` or `attribute`;
- `llm.provider`: `none` or `deepseek`;
- positive `top_k`, `rerank_candidates`, `max_constraint_asks`, timeouts, and `max_tokens`;
- non-negative retry count and delays;
- positive circuit-breaker threshold;
- `retry.max_delay_seconds >= retry.base_delay_seconds`.

Raise messages such as `llm.timeout_seconds must be > 0` so tests and operators can locate the bad field.

- [ ] **Step 5: Run configuration tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_config_loader -v
```

Expected: all configuration model and loader tests PASS.

- [ ] **Step 6: Commit the configuration core**

```bash
git add config/default.json config/models.py config/loader.py tests/test_config_loader.py
git commit -m "feat: add unified configuration loader"
```

---

### Task 2: Backward-Compatible EnvConfig Facade

**Files:**
- Modify: `config/env_config.py`
- Modify: `config/__init__.py`
- Modify: `tests/test_config_loader.py`

**Interfaces:**
- Consumes: `load_config(...) -> AppConfig` from Task 1.
- Produces: `EnvConfig.from_env(path=None, overrides=None, environ=None) -> EnvConfig`.
- Produces: `EnvConfig.app_config: AppConfig` and `EnvConfig.llm: LLMConfig`.
- Preserves: flat attributes currently consumed by `agent/*` and `run_local_eval.py`.

- [ ] **Step 1: Add failing compatibility tests**

Append tests that exercise both existing access and new nested access:

```python
from config.env_config import EnvConfig


class EnvConfigCompatibilityTest(unittest.TestCase):
    def test_exposes_existing_flat_fields_and_nested_llm(self) -> None:
        env = EnvConfig.from_env(environ={
            "TOP_K": "8",
            "SAMPLE_LIMIT": "4",
            "LLM_MODEL": "deepseek-reasoner",
            "DEEPSEEK_API_KEY": "secret-value",
        })
        self.assertEqual(env.top_k, 8)
        self.assertEqual(env.sample_limit, 4)
        self.assertEqual(env.llm.model, "deepseek-reasoner")
        self.assertEqual(env.llm_model, "deepseek-reasoner")
        self.assertFalse(env.offline)
        self.assertFalse(hasattr(env, "env_overrides"))
        self.assertNotIn("secret-value", repr(env))

    def test_legacy_openai_fields_remain_targeted_environment_reads(self) -> None:
        env = EnvConfig.from_env(environ={
            "OPENAI_API_KEY": "legacy-secret",
            "OPENAI_BASE_URL": "https://legacy.invalid",
        })
        self.assertEqual(env.openai_api_key, "legacy-secret")
        self.assertEqual(env.openai_base_url, "https://legacy.invalid")
        self.assertNotIn("legacy-secret", repr(env))
```

Also retain a test for submit mode: `offline` is false when a DeepSeek key would enable an external provider, allowing the existing submit-mode assertion to reject online operation.

- [ ] **Step 2: Run compatibility tests and verify they fail**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_config_loader.EnvConfigCompatibilityTest -v
```

Expected: FAIL because the old dataclass has no `llm` object and retains `env_overrides`.

- [ ] **Step 3: Replace EnvConfig parsing with a facade**

Implement `EnvConfig` as a frozen facade holding `_app_config`, `_openai_api_key`, and `_openai_base_url`. `from_env()` calls `load_config()` exactly once. Add explicit read-only properties for every current flat field:

```python
@dataclass(frozen=True, repr=False)
class EnvConfig:
    _app_config: AppConfig
    _openai_api_key: str = field(default="", repr=False)
    _openai_base_url: str = ""

    @classmethod
    def from_env(cls, path=None, overrides=None, environ=None) -> "EnvConfig":
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
    def offline(self) -> bool:
        legacy_offline = self.llm_backend in {"none", "local"}
        deepseek_offline = self.llm.provider == "none" or not self.llm.api_key
        return legacy_offline and deepseek_offline
```

`llm_model` maps to `self.llm.model`; existing OpenAI fields remain targeted compatibility values. Do not restore `env_overrides` or retain the full environment mapping. Make `__repr__` and `summary()` report only non-secret fields and whether LLM state is configured, never the key value.

Update `config/__init__.py` to export `AppConfig`, `LLMConfig`, `ConfigError`, and `load_config` in addition to `EnvConfig` and `constants`.

- [ ] **Step 4: Run compatibility and existing tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_config_loader -v
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover tests -v
```

Expected: all configuration and original evaluator tests PASS; importing every current Agent module succeeds.

- [ ] **Step 5: Commit the compatibility facade**

```bash
git add config/env_config.py config/__init__.py tests/test_config_loader.py
git commit -m "refactor: route EnvConfig through unified loader"
```

---

### Task 3: Provider-Neutral Interface and DeepSeek SDK Client

**Files:**
- Create: `llm/__init__.py`
- Create: `llm/base.py`
- Create: `llm/deepseek.py`
- Create: `llm/factory.py`
- Create: `tests/test_deepseek_client.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: `LLMConfig` from Task 1.
- Produces: `create_llm_client(config: LLMConfig) -> LLMClient`.
- Produces: `DeepSeekClient.initialize() -> LLMStatus`.
- Produces: `DeepSeekClient.chat(messages, *, temperature=None, max_tokens=None) -> LLMResult`.
- Produces: `LLMState`, `LLMErrorCategory`, `LLMStatus`, `LLMUsage`, and `LLMResult`.
- Produces testable constructor: `DeepSeekClient(config, *, sdk_factory=None, sleep=time.sleep, jitter=None, clock=time.monotonic, failure_classifier=None)`; production callers pass only `config`.

- [ ] **Step 1: Add the real SDK dependency**

Replace the commented OpenAI dependency line in `requirements.txt` with:

```text
openai>=1.0,<2.0
```

Keep dense retrieval and local model dependencies documented as optional comments.

Install the declared dependency before running this task's tests:

```bash
python -m pip install -r requirements.txt
```

- [ ] **Step 2: Add failing tests for side-effect-free construction and startup probing**

Create `tests/test_deepseek_client.py`. Inject an SDK factory, sleep function, jitter function, and monotonic clock into `DeepSeekClient` so tests are deterministic:

```python
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from config.models import CircuitBreakerConfig, LLMConfig, RetryConfig
from llm.base import LLMErrorCategory, LLMState
from llm.deepseek import DeepSeekClient, FailureDisposition


def completion(content: str = "OK", prompt_tokens: int = 2, completion_tokens: int = 1):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


class DeepSeekClientTest(unittest.TestCase):
    def make_config(self, api_key: str = "test-key", **changes) -> LLMConfig:
        values = {
            "api_key": api_key,
            "retry": RetryConfig(max_retries=2, base_delay_seconds=0.0, max_delay_seconds=0.0),
            "circuit_breaker": CircuitBreakerConfig(failure_threshold=2),
        }
        values.update(changes)
        return LLMConfig(**values)

    def test_construction_has_no_sdk_or_network_side_effect(self) -> None:
        factory = Mock()
        client = DeepSeekClient(self.make_config(), sdk_factory=factory)
        factory.assert_not_called()
        self.assertEqual(client.status.state, LLMState.DISABLED)

    def test_missing_key_initializes_disabled_without_sdk(self) -> None:
        factory = Mock()
        client = DeepSeekClient(self.make_config(api_key=""), sdk_factory=factory)
        status = client.initialize()
        self.assertEqual(status.state, LLMState.DISABLED)
        factory.assert_not_called()

    def test_successful_probe_marks_available(self) -> None:
        sdk = Mock()
        sdk.chat.completions.create.return_value = completion()
        client = DeepSeekClient(self.make_config(), sdk_factory=Mock(return_value=sdk))
        status = client.initialize()
        self.assertEqual(status.state, LLMState.AVAILABLE)
        self.assertEqual(status.attempts, 1)
        sdk.chat.completions.create.assert_called_once()
```

- [ ] **Step 3: Add failing retry, sanitization, chat, and circuit tests**

In the same test class, add deterministic cases that:

- inject a retryable classified failure twice and a successful third result, asserting three calls and two sleeps;
- inject a non-retryable authentication failure, asserting one call and `AUTHENTICATION`;
- exhaust three transient attempts, asserting `UNAVAILABLE` and exactly three calls;
- place the literal test key inside an exception message and assert it is absent from `status.error_message` and `repr(client)`;
- initialize successfully, call `chat()`, and assert content, token usage, provider, model, and measured latency;
- fail two consecutive runtime chats and assert the third returns `CIRCUIT_OPEN` without an SDK call;
- succeed between two failed chats and assert the consecutive-failure count resets.

Use injected `failure_classifier` values of `FailureDisposition(category, retryable)` in unit tests. Add a separate table-driven test for the default OpenAI SDK exception classifier using mocked exception instances or status-bearing exceptions, without network access.

- [ ] **Step 4: Run the DeepSeek tests and verify the package is absent**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_deepseek_client -v
```

Expected: FAIL with an import error for `llm.base`.

- [ ] **Step 5: Implement provider-neutral public types**

Create `llm/base.py` with these signatures:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence


class LLMState(str, Enum):
    DISABLED = "disabled"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class LLMErrorCategory(str, Enum):
    DISABLED = "disabled"
    AUTHENTICATION = "authentication"
    BAD_REQUEST = "bad_request"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    RATE_LIMIT = "rate_limit"
    SERVER = "server"
    SDK_MISSING = "sdk_missing"
    CIRCUIT_OPEN = "circuit_open"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True)
class LLMStatus:
    state: LLMState
    provider: str
    model: str
    attempts: int = 0
    error_category: LLMErrorCategory | None = None
    error_message: str = ""


@dataclass(frozen=True)
class LLMResult:
    success: bool
    provider: str
    model: str
    content: str = ""
    usage: LLMUsage = LLMUsage()
    latency_ms: float = 0.0
    error_category: LLMErrorCategory | None = None
    error_message: str = ""


class LLMClient(Protocol):
    @property
    def status(self) -> LLMStatus: ...

    def initialize(self) -> LLMStatus: ...

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult: ...
```

Create a small `DisabledLLMClient` in `llm/base.py` for provider `none`; it always reports `disabled`, never imports the SDK, and returns an unsuccessful result with category `DISABLED` and a sanitized `disabled` message from `chat()`.

- [ ] **Step 6: Implement DeepSeek initialization, retry policy, and sanitization**

In `llm/deepseek.py`:

- Construct the OpenAI SDK only inside `initialize()` or the first explicit chat when health checking is disabled.
- Pass `api_key`, `base_url`, an `httpx.Timeout` with separate connect and overall values, and `max_retries=0` so the project policy is the sole retry mechanism.
- Probe through `chat.completions.create()` with model `config.model`, a fixed system-free user message, `temperature=0.0`, and `max_tokens=1`.
- Classify OpenAI SDK exceptions into the exact `LLMErrorCategory` values from `llm/base.py`.
- Retry only retryable dispositions and use `min(max_delay, base_delay * 2**retry_index) + jitter`.
- Sanitize error messages by removing the configured key, Bearer token patterns, and authorization header values; cap the stored message length.
- Return structured failures rather than raising for provider/network failures.
- Increment the consecutive runtime-failure count only for `chat()` failures after initialization; reset it on success and open the circuit at the configured threshold.

Implement `FailureDisposition` as:

```python
@dataclass(frozen=True)
class FailureDisposition:
    category: LLMErrorCategory
    retryable: bool
```

The production SDK factory imports `OpenAI` and `httpx.Timeout` lazily. If the package cannot be imported, initialization returns `SDK_MISSING` and `unavailable` without retrying.

- [ ] **Step 7: Implement the provider factory and exports**

Create `llm/factory.py`:

```python
def create_llm_client(config: LLMConfig) -> LLMClient:
    if config.provider == "none":
        return DisabledLLMClient(provider="none", model=config.model)
    if config.provider == "deepseek":
        return DeepSeekClient(config)
    raise ValueError(f"Unsupported LLM provider: {config.provider}")
```

Export the factory and public result/status types from `llm/__init__.py`. Provider classes may also be exported for direct testing, but application code should construct clients through the factory.

- [ ] **Step 8: Run DeepSeek and configuration tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_deepseek_client -v
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_config_loader -v
```

Expected: all tests PASS with zero real network calls and zero retry sleeps.

- [ ] **Step 9: Commit the LLM client layer**

```bash
git add llm requirements.txt tests/test_deepseek_client.py
git commit -m "feat: add resilient DeepSeek SDK client"
```

---

### Task 4: Explicit Startup Integration, Documentation, and Full Verification

**Files:**
- Create: `tests/test_llm_startup.py`
- Modify: `run_local_eval.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `EnvConfig.llm` from Task 2.
- Consumes: `create_llm_client(config: LLMConfig) -> LLMClient` from Task 3.
- Produces: `initialize_llm(env: EnvConfig) -> LLMClient` in `run_local_eval.py` for explicit, testable startup.

- [ ] **Step 1: Add failing startup-degradation tests**

Create `tests/test_llm_startup.py` and mock the factory at the `run_local_eval` import boundary:

```python
from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from config.env_config import EnvConfig
from llm.base import LLMErrorCategory, LLMState, LLMStatus
from run_local_eval import initialize_llm


class LLMStartupTest(unittest.TestCase):
    def test_disabled_client_is_reported_without_error(self) -> None:
        env = EnvConfig.from_env(environ={})
        client = Mock()
        client.initialize.return_value = LLMStatus(
            state=LLMState.DISABLED,
            provider="deepseek",
            model="deepseek-chat",
        )
        with patch("run_local_eval.create_llm_client", return_value=client):
            returned = initialize_llm(env)
        self.assertIs(returned, client)

    def test_unavailable_client_does_not_raise(self) -> None:
        env = EnvConfig.from_env(environ={"DEEPSEEK_API_KEY": "test-key"})
        client = Mock()
        client.initialize.return_value = LLMStatus(
            state=LLMState.UNAVAILABLE,
            provider="deepseek",
            model="deepseek-chat",
            attempts=3,
            error_category=LLMErrorCategory.TIMEOUT,
            error_message="request timed out",
        )
        with patch("run_local_eval.create_llm_client", return_value=client):
            returned = initialize_llm(env)
        self.assertIs(returned, client)
```

Capture stdout in additional assertions and verify that the status line contains provider, model, state, attempts, and error category but not the test key.

- [ ] **Step 2: Run startup tests and verify the helper is absent**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_llm_startup -v
```

Expected: FAIL because `initialize_llm` is not defined.

- [ ] **Step 3: Add explicit initialization to the runner**

Import `create_llm_client` and define:

```python
def initialize_llm(env: EnvConfig):
    client = create_llm_client(env.llm)
    status = client.initialize()
    details = (
        f"provider={status.provider} model={status.model} "
        f"state={status.state.value} attempts={status.attempts}"
    )
    if status.error_category is not None:
        details += f" error={status.error_category.value}"
    print(f"    LLM: {details}")
    return client
```

Call `initialize_llm(env)` after the existing submit-mode offline assertion and before dataset verification. Do not pass the returned client to `Agent`, `Reranker`, or any other business module.

- [ ] **Step 4: Update user-facing configuration documentation**

In `README.md`:

- replace the claim that all configuration is direct environment parsing with the four-level precedence;
- document `config/default.json` and `APP_CONFIG_PATH`;
- document `DEEPSEEK_API_KEY`, DeepSeek defaults, timeout/retry/circuit variables, and sanitized startup states;
- state explicitly that this version probes the API but does not use it for recommendations or questions;
- retain offline usage instructions and explain that no key means no network request;
- update dependency instructions so `pip install -r requirements.txt` installs the OpenAI SDK.

- [ ] **Step 5: Run focused startup and full unit tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_llm_startup -v
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover tests -v
```

Expected: all new and existing tests PASS without network access.

- [ ] **Step 6: Verify imports, offline initialization, and repository boundaries**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 env -u DEEPSEEK_API_KEY python -c 'from config import EnvConfig; from llm import create_llm_client; c = EnvConfig.from_env(); client = create_llm_client(c.llm); print(client.initialize().state.value)'
git diff --name-only HEAD~4..HEAD
git diff --exit-code HEAD~4..HEAD -- evaluator/local_evaluator.py starter/agent.py data/public_set.jsonl
```

Expected:

- the Python command prints `disabled` and does not access the network;
- the changed-file list contains only the implementation files named in this plan;
- the protected-file diff command exits successfully with no output.

Do not run the full catalog evaluation unless `data/catalog.jsonl` is present. If it is present, run `SAMPLE_LIMIT=1 SKIP_DATA_VERIFY=1 python run_local_eval.py` without a DeepSeek key and confirm evaluation starts after reporting `disabled`.

- [ ] **Step 7: Commit startup integration and documentation**

```bash
git add run_local_eval.py README.md tests/test_llm_startup.py
git commit -m "feat: initialize DeepSeek availability at startup"
```

- [ ] **Step 8: Record final verification evidence**

Run:

```bash
git status --short
git log --oneline -5
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover tests -v
```

Expected: empty worktree status, the task commits visible in history, and the complete suite PASS. Report the exact test count and note whether catalog evaluation was skipped because the catalog file was absent.
