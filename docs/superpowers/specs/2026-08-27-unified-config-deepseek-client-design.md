# Unified Configuration and DeepSeek Client Design

## 1. Objective

Add a single configuration-loading path and a provider-neutral LLM client layer to the existing Shopping Copilot project. The first provider implementation is DeepSeek through the OpenAI Python SDK.

This change prepares the project for online LLM capabilities without changing the current retrieval, reranking, intent-routing, clarification, evaluator, or response behavior. DeepSeek availability is detected explicitly during application startup; an unavailable API must never prevent the existing rule-based path from running.

## 2. Scope

### Included

- Load non-secret defaults from `config/default.json`.
- Overlay configuration from environment variables and explicit runtime overrides.
- Validate configuration types, ranges, and supported enum values.
- Preserve the existing `EnvConfig.from_env()` entry point and flat attributes used by current modules.
- Define a provider-neutral LLM client contract.
- Implement a DeepSeek client with the OpenAI Python SDK.
- Read the DeepSeek credential only from `DEEPSEEK_API_KEY`.
- Perform an explicit startup health check with retry classification.
- Track runtime failures and open a process-local circuit breaker.
- Surface sanitized status, usage, latency, and error information.
- Initialize and report LLM availability from `run_local_eval.py`.
- Add deterministic unit tests using a mocked SDK client.

### Excluded

- Using DeepSeek for reranking, intent detection, slot extraction, clarification, response generation, or any other business decision.
- Changing the official evaluator, public dataset, Agent API, recommendation output, or scoring behavior.
- Implementing OpenAI, local-model, or additional provider clients.
- Persisting circuit-breaker state across processes.
- Sending real network requests from tests.

## 3. Design Choice

Use a gradual compatibility design. New code consumes a structured root configuration, while `EnvConfig` remains as a compatibility facade for existing modules. This avoids a broad migration of the currently working retrieval pipeline and provides a stable path for future provider integrations.

The alternatives rejected for this change are:

- A one-shot migration of every module to nested configuration, because it expands the regression surface without helping the initial DeepSeek loader.
- A standalone DeepSeek utility with its own environment parsing, because it would create a second configuration system and defeat the unification goal.

## 4. Configuration Architecture

### 4.1 Files and responsibilities

- `config/default.json`: checked-in, non-secret defaults.
- `config/models.py`: immutable typed configuration objects.
- `config/loader.py`: JSON loading, layered merge, environment mapping, explicit overrides, and validation.
- `config/env_config.py`: compatibility facade that delegates loading to `config.loader` and exposes the flat attributes expected by existing code.

The root object is `AppConfig`. It contains the existing application, retrieval, and dialogue values plus a nested `LLMConfig`. The LLM section contains provider, model, base URL, timeout, health-check, retry, and circuit-breaker settings.

### 4.2 Precedence

Configuration values are resolved in this order, from lowest to highest priority:

1. Dataclass safety defaults.
2. `config/default.json`.
3. Environment variables.
4. Explicit overrides passed to `load_config()`.

The default file may be replaced with `APP_CONFIG_PATH`. A direct `load_config(path=...)` argument overrides `APP_CONFIG_PATH`.

### 4.3 Secret handling

`DEEPSEEK_API_KEY` is read directly from the environment and is never accepted from JSON or ordinary explicit overrides. The configuration object stores only the required credential value with a redacted representation. It must not retain a copy of the complete process environment.

Logs, summaries, exceptions, and serialized status objects must never include the API key, Authorization header, or raw SDK request object.

### 4.4 Initial LLM defaults

The checked-in defaults are:

- Provider: `deepseek`.
- Model: `deepseek-chat`.
- Base URL: `https://api.deepseek.com`.
- Health check: enabled.
- Maximum retries: `2`, meaning at most three total attempts.
- Retry base delay: `0.5` seconds.
- Retry maximum delay: `1.5` seconds.
- Runtime circuit-breaker threshold: `2` consecutive failures.

Timeout values remain configurable in `default.json`; the initial implementation will use a short startup timeout suitable for early degradation and a separate normal chat timeout.

### 4.5 Environment mapping

At minimum, the loader supports:

- `APP_CONFIG_PATH`
- `LLM_PROVIDER`
- `LLM_MODEL`
- `LLM_BASE_URL`
- `LLM_HEALTH_CHECK_ENABLED`
- `LLM_CONNECT_TIMEOUT_SECONDS`
- `LLM_TIMEOUT_SECONDS`
- `LLM_MAX_RETRIES`
- `LLM_RETRY_BASE_DELAY_SECONDS`
- `LLM_RETRY_MAX_DELAY_SECONDS`
- `LLM_CIRCUIT_BREAKER_FAILURE_THRESHOLD`
- `DEEPSEEK_API_KEY`

Existing environment variables remain readable through the compatibility facade. DeepSeek never treats `OPENAI_API_KEY` as its credential.

### 4.6 Validation and failure behavior

Missing default configuration, invalid JSON, unknown provider values, invalid booleans, non-numeric numeric fields, non-positive timeouts, negative retry counts, and non-positive circuit thresholds are configuration errors and fail fast with a concise field-specific message.

A valid configuration without `DEEPSEEK_API_KEY` is not an error. It produces an LLM client in `disabled` state and does not access the network.

## 5. LLM Client Architecture

### 5.1 Files and responsibilities

- `llm/base.py`: client protocol, status enum, response/result types, normalized error categories, and client exceptions used only for programmer/configuration errors.
- `llm/deepseek.py`: DeepSeek implementation using `openai>=1.0,<2.0` and the Chat Completions API.
- `llm/factory.py`: provider selection and client construction.
- `llm/__init__.py`: stable public exports.

### 5.2 Public interface

The provider-neutral interface exposes:

```python
initialize() -> LLMStatus
chat(messages, *, temperature=None, max_tokens=None) -> LLMResult
status -> LLMStatus
```

`LLMStatus` distinguishes `disabled`, `available`, and `unavailable`. `LLMResult` contains:

- success flag;
- response text when successful;
- provider and model;
- prompt and completion token counts when reported;
- request latency in milliseconds;
- normalized error category and sanitized error message when unsuccessful.

Expected provider and network failures are represented as unsuccessful results so future business callers can fall back without broad exception handling. Configuration and programming errors remain exceptions.

### 5.3 Explicit initialization

Constructing or importing the client never contacts the network. `run_local_eval.py` explicitly calls `initialize()` after configuration loading and before dataset evaluation.

Initialization behaves as follows:

1. If no credential is present, set status to `disabled` and return without creating a request.
2. If health checking is disabled, create the SDK client and mark it available without a probe.
3. Otherwise, send one minimal Chat Completions request with a fixed short message and `max_tokens=1`.
4. On success, set status to `available` and reset the failure counter.
5. On final failure, set status to `unavailable`; the evaluator continues normally.

The startup status line reports provider, model, state, attempt count, and sanitized error category only.

## 6. Retry and Circuit-Breaker Policy

The initial request plus two retries produces at most three startup attempts.

Retryable failures are:

- connection failures;
- timeouts;
- HTTP 429 / SDK rate-limit errors;
- HTTP 5xx / SDK server errors.

Non-retryable failures are:

- authentication and permission failures;
- invalid request or parameter failures;
- unknown model or endpoint failures;
- other deterministic client-side errors.

Retry delay uses bounded exponential backoff with small random jitter. Tests inject or mock the delay and randomness so they remain fast and deterministic.

After successful initialization, each successful `chat()` resets the consecutive runtime-failure count. A failed runtime request returns an unsuccessful result for that call. Two consecutive runtime failures set the client to `unavailable`; later calls return immediately without contacting the provider. Recovery requires construction and initialization of a new client in a new process or explicit future recovery functionality, which is outside this change.

## 7. Application Integration

`run_local_eval.py` will:

1. Load the unified configuration once.
2. Create the configured LLM client through the factory.
3. Call `initialize()` explicitly.
4. Print a sanitized one-line status.
5. Continue into the unchanged evaluator and current rule-based Agent regardless of `disabled` or `unavailable` status.

The initialized client is not injected into the Agent in this change. This intentionally prevents accidental LLM participation in rankings or dialogue behavior.

The official `evaluator/local_evaluator.py` remains untouched.

## 8. Dependency Policy

`requirements.txt` will contain the real dependency:

```text
openai>=1.0,<2.0
```

The application still supports an offline runtime when no key is configured. The SDK being installed does not trigger network access by itself.

## 9. Testing Strategy

Configuration tests cover:

- default JSON loading;
- `APP_CONFIG_PATH` and explicit path precedence;
- environment and explicit override precedence;
- secret acceptance only from `DEEPSEEK_API_KEY`;
- compatibility attributes exposed by `EnvConfig`;
- invalid JSON, invalid types, invalid ranges, and unsupported providers;
- absence of a key producing a valid offline configuration;
- redacted configuration representation.

DeepSeek client tests mock the OpenAI SDK and cover:

- imports and construction causing no request;
- no key producing `disabled` without constructing an SDK client;
- successful startup probe;
- transient startup failure followed by success;
- exhaustion after three attempts;
- authentication failure without retry;
- sanitized error/status output;
- successful chat usage and latency mapping;
- per-call fallback result after failure;
- circuit opening after two consecutive failures;
- success resetting the consecutive failure counter.

Runner tests or a focused integration test verify that LLM initialization status cannot prevent the rule-based evaluation path from starting. All tests use mocks and make no real API calls.

Existing official evaluator tests must continue to pass unchanged.

## 10. Acceptance Criteria

The change is complete when:

- every runtime configuration value has one canonical loading path;
- existing modules can continue calling `EnvConfig.from_env()`;
- secrets are not accepted from JSON, retained as a full environment snapshot, or exposed in logs/repr;
- DeepSeek initialization is explicit and performs no request without a key;
- transient initialization failures receive at most two retries;
- deterministic provider errors do not retry;
- two consecutive runtime failures open the process-local circuit;
- LLM failure never prevents the current rule-based evaluator path from running;
- DeepSeek does not influence recommendations or questions in this change;
- new tests and all existing tests pass without network access.
