# Unified Multi-Provider LLM Client and Reranking Design

## 1. Status and Authority

This design supersedes `2026-08-27-unified-config-deepseek-client-design.md` for all LLM configuration, client, startup, and reranking behavior. The earlier configuration-loader work remains valid where it does not conflict with this document.

The scope changed after the first implementation review: the existing OpenAI reranker must move into the new provider-neutral client path, and the unified client must support both DeepSeek and OpenAI models.

## 2. Objective

Provide one validated configuration system and one OpenAI-compatible client implementation for DeepSeek and OpenAI. The selected client is initialized explicitly at startup and injected into the enhanced Agent and Reranker. When enabled and available, it reranks a small rule-ranked candidate set; every disabled, invalid, unavailable, malformed, or circuit-open path preserves the existing rule order.

## 3. Scope

### Included

- Preserve the layered JSON/environment/explicit-override configuration loader.
- Add provider profiles for `deepseek` and `openai` plus a `none` provider.
- Select exactly one provider per process.
- Read credentials only from the provider-specific environment variable.
- Express model request capabilities in configuration instead of inferring them from model names.
- Replace the provider-specific DeepSeek implementation with a shared `OpenAICompatibleClient`.
- Retain a compatibility `DeepSeekClient` entry point that delegates to the shared client.
- Move legacy OpenAI reranking out of `agent/reranker.py` and into the shared client path.
- Inject the initialized client through `run_local_eval.py` into `Agent` and `Reranker`.
- Send compact product metadata for the first 12 rule-ranked candidates.
- Parse, validate, deduplicate, and complete model-produced rankings locally.
- Report per-turn reranking token usage through the existing Agent response contract.
- Preserve startup retry, error normalization, sanitization, and circuit-breaking behavior.
- Make API credentials opaque to ordinary dataclass serialization.
- Add no-network tests for both providers, model capability differences, injection, reranking, and fallback.

### Excluded

- Local-model loading or inference.
- Automatic cross-provider failover.
- Using an LLM for intent detection, slot extraction, clarification, or natural-language response generation.
- Provider-specific Responses API, function calling, tools, JSON Schema, or streaming.
- Changing the official evaluator, starter Agent, public data, Agent response schema, or scoring behavior.

## 4. Configuration Architecture

### 4.1 Provider profiles

`config/default.json` contains one selected provider and persistent profiles for both supported online providers:

```json
{
  "llm": {
    "provider": "deepseek",
    "rerank_enabled": true,
    "rerank_candidates": 12,
    "health_check_enabled": true,
    "connect_timeout_seconds": 3.0,
    "timeout_seconds": 8.0,
    "retry": {
      "max_retries": 2,
      "base_delay_seconds": 0.5,
      "max_delay_seconds": 1.5
    },
    "circuit_breaker": {
      "failure_threshold": 2
    },
    "providers": {
      "deepseek": {
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "token_limit_parameter": "max_tokens",
        "supports_temperature": true
      },
      "openai": {
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "token_limit_parameter": "max_completion_tokens",
        "supports_temperature": true
      }
    }
  }
}
```

The root configuration exposes the selected profile as a resolved immutable object so clients and business code do not perform provider lookups independently.

### 4.2 Capability configuration

Every provider profile defines:

- `model`: exact model identifier sent to the API;
- `base_url`: OpenAI-compatible API root;
- `token_limit_parameter`: exactly `max_tokens` or `max_completion_tokens`;
- `supports_temperature`: whether the client may send `temperature`.

The client sends only the configured token-limit parameter. It omits `temperature` when `supports_temperature` is false. Unsupported capability values fail during configuration loading, before client construction.

This makes additional compatible models configurable without hard-coded model-name prefix rules.

### 4.3 Precedence and environment variables

Non-secret precedence remains:

1. Dataclass safety defaults.
2. `config/default.json` or `APP_CONFIG_PATH`.
3. Environment variables.
4. Explicit overrides passed to `load_config()`.

Provider selection and model overrides use:

- `LLM_PROVIDER=none|deepseek|openai`;
- `DEEPSEEK_MODEL` and `DEEPSEEK_BASE_URL` for the DeepSeek profile;
- `OPENAI_MODEL` and `OPENAI_BASE_URL` for the OpenAI profile;
- `LLM_MODEL` and `LLM_BASE_URL` as highest-priority environment overrides for the currently selected profile;
- `LLM_RERANK`, `LLM_RERANK_CANDIDATES`, health, timeout, retry, and circuit variables for shared behavior.

`LLM_PROVIDER` wins over the legacy `LLM_BACKEND`. If `LLM_PROVIDER` is absent but `LLM_BACKEND` is explicitly set, compatibility mapping is:

- `openai` -> `openai`;
- `none` -> `none`;
- `local` -> `none` because local loading is outside this change.

If neither variable is set, the JSON provider applies.

### 4.4 Credentials and serialization

- DeepSeek reads only `DEEPSEEK_API_KEY`.
- OpenAI reads only `OPENAI_API_KEY`.
- Provider keys are forbidden in JSON and ordinary explicit overrides.
- Only the selected profile receives its matching credential.
- Credentials use an opaque `SecretValue` object with an explicit `reveal()` method.
- `repr`, `str`, `dataclasses.asdict`, JSON serialization with `default=str`, logs, status, errors, and summaries must not expose the raw value.
- The raw value is revealed only at the narrow SDK-construction boundary.
- The complete process environment is never retained.

Tests recursively inspect every string leaf produced by `dataclasses.asdict(AppConfig)` and verify the raw credential is absent.

## 5. Client Architecture

```text
LLMClient protocol
  ├── DisabledLLMClient
  └── OpenAICompatibleClient
        ├── selected DeepSeek profile
        └── selected OpenAI profile
```

`DeepSeekClient` remains as a thin compatibility constructor or alias around `OpenAICompatibleClient`; it must not duplicate request, retry, response, or circuit logic.

The public protocol remains:

```python
initialize() -> LLMStatus
chat(messages, *, temperature=None, max_tokens=None) -> LLMResult
status -> LLMStatus
```

`LLMResult` contains success, provider, model, content, prompt/completion tokens, latency, and a normalized sanitized error.

### 5.1 Explicit initialization

- Importing and constructing configuration or clients sends no request.
- `provider=none` or a missing selected-provider key yields `disabled` without SDK construction.
- With health checking enabled, `initialize()` sends one minimal Chat Completions request.
- With health checking disabled, `initialize()` constructs the SDK client and marks it available without a probe.
- SDK internal retries remain `0`; project retry logic is authoritative.
- Startup performs at most three total attempts regardless of a larger configured retry value.
- Probe usage contributes to client cumulative usage, not a conversation turn.

### 5.2 Retry and circuit behavior

Connection, timeout, HTTP 429, and HTTP 5xx failures retry. Authentication, permission, invalid request, unsupported model, and other deterministic client failures do not.

Backoff is exponential, capped by configuration, and includes small bounded production jitter. Sleep and jitter remain injectable so tests use no real delays.

Each successful runtime chat resets the consecutive-failure count. When a runtime failure reaches the configured threshold, public status immediately becomes `unavailable` with error category `circuit_open`. Later calls return immediately without SDK access.

Malformed completion structures never escape as `IndexError`, `AttributeError`, or `TypeError`; they become sanitized structured failures.

### 5.3 Single-provider policy

Only the selected provider is initialized. Failure does not automatically invoke the other provider. This avoids unexpected cost and preserves reproducible evaluation. Cross-provider fallback may be added later only through explicit configuration.

## 6. Application Dependency Injection

`run_local_eval.py` owns startup construction and initialization:

```text
EnvConfig.from_env()
  -> create_llm_client(env.llm)
  -> client.initialize()
  -> Agent(catalog_path, env=env, llm_client=client)
  -> Reranker(env=env, llm_client=client)
```

Constructors become:

```python
Agent(catalog_path="data/catalog.jsonl", env=None, llm_client=None)
Reranker(env=None, llm_client=None)
```

If no client is passed, each path uses `DisabledLLMClient`; neither constructor creates or initializes an online client. This keeps direct Agent construction deterministic and offline.

The official evaluator continues importing `starter.agent.Agent` and remains unchanged.

## 7. Unified Reranking Protocol

### 7.1 Invocation

The rule scorer remains the primary deterministic ranking. LLM reranking runs only when:

- `LLM_RERANK` / `llm.rerank_enabled` is true;
- client status is `available`;
- at least two candidates exist.

Otherwise the rule order is returned unchanged.

### 7.2 Candidate payload

The first `llm.rerank_candidates` candidates are sent, defaulting to 12. Every candidate includes:

- `parent_asin`;
- title truncated to 240 characters;
- categories truncated to 240 characters;
- normalized features truncated to 800 characters.

The prompt also includes active user constraints, truncated to a combined 800 characters. The entire catalog record, details dictionary, user profile, and unrelated conversation history are not sent.

### 7.3 Output contract

The preferred response is:

```json
{"ranked_parent_asins": ["B001...", "B002..."]}
```

For compatibility, the parser also accepts a bare JSON array and JSON wrapped in a Markdown code fence. Local validation:

- accepts only string ASINs from the submitted candidate set;
- preserves first occurrence and removes duplicates;
- appends omitted candidates in their original rule order;
- treats empty, malformed, or candidate-free output as failure and keeps the complete rule order.

No provider-specific structured-output feature is required.

### 7.4 Usage and fallback

The Reranker reads usage only from the `LLMResult` for the current rerank call. Successful calls populate the current Agent response `usage`. Disabled, skipped, circuit-open, or failed calls report zero unless the provider supplied reliable usage in the returned result.

Startup probe usage is retained only in client cumulative metrics and is not assigned to a session turn.

The legacy `Reranker._llm_rerank_openai()` method, direct SDK import, direct credential read, and client construction are removed.

## 8. Compatibility and Migration

- DeepSeek and OpenAI use the same Chat Completions request and local ranking protocol.
- Existing `LLM_BACKEND=openai` selects the OpenAI profile when `LLM_PROVIDER` is absent.
- `LLM_MODEL` now overrides the selected provider; `OPENAI_MODEL` and `DEEPSEEK_MODEL` persist provider-specific values.
- Existing custom `OPENAI_BASE_URL` remains supported through the OpenAI profile.
- Submit mode with an external provider but no matching key is offline and must not abort.
- Submit mode with a usable selected-provider key remains online and is rejected by the existing submit-mode offline assertion.
- `EnvConfig` retains flat compatibility properties, but Reranker no longer uses them to load an SDK.

## 9. Testing Strategy

All tests use `unittest` and mocks; none make real provider calls or sleep.

Configuration tests cover:

- provider-profile defaults and nested merge;
- provider-specific and selected-provider override precedence;
- legacy `LLM_BACKEND` mapping;
- credential selection and cross-provider isolation;
- capability validation;
- opaque secret serialization;
- submit-mode missing-key and configured-key behavior.

Client tests cover both profiles:

- identical shared retry, error, response, and breaker behavior;
- correct base URL, model, and selected key;
- `max_tokens` versus `max_completion_tokens` request construction;
- omission of temperature when unsupported;
- actual mocked OpenAI SDK exceptions for authentication/permission, 408 or timeout, connection, 429, 5xx, and unknown errors;
- startup maximum of three attempts;
- bounded jitter injection;
- circuit status transition to unavailable;
- malformed completion structures;
- token and cumulative-usage mapping;
- compatibility `DeepSeekClient` delegation.

Reranker and integration tests cover:

- explicit client injection through runner -> Agent -> Reranker;
- no implicit client construction;
- disabled/unavailable/failure fallback preserving exact rule order;
- compact candidate payload fields and truncation;
- object, array, and fenced JSON responses;
- removal of unknown/duplicate IDs and completion of missing IDs;
- per-turn usage mapping;
- changing an inactive provider profile does not affect the selected provider;
- protected official files remain unchanged.

## 10. Acceptance Criteria

The revised feature is complete when:

- one canonical configuration path resolves both provider profiles and the selected profile;
- one shared OpenAI-compatible client owns SDK construction, retries, error handling, parsing, usage, and circuit state;
- selected provider, model, Base URL, key, and capability flags produce the exact intended SDK request;
- ordinary dataclass serialization cannot expose a raw API key;
- client status becomes unavailable as soon as the circuit opens;
- no provider is contacted implicitly or as an automatic fallback;
- the old Reranker SDK-loading path is removed;
- the initialized selected client is explicitly injected into Agent and Reranker;
- DeepSeek and OpenAI can both rerank through the same validated JSON protocol;
- every unavailable or invalid LLM path preserves deterministic rule ranking;
- per-turn usage excludes startup probes;
- official evaluator, starter Agent, data, API schema, and scoring remain untouched;
- all new and existing tests pass without network access.
