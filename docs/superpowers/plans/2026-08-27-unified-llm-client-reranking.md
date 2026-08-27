# Unified Multi-Provider LLM Client and Reranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify DeepSeek and legacy OpenAI model loading behind one configurable OpenAI-compatible client and inject it into the enhanced Agent for safe, validated candidate reranking.

**Architecture:** The existing layered loader is extended with immutable DeepSeek/OpenAI provider profiles, explicit capability flags, and opaque secrets. One `OpenAICompatibleClient` handles both providers; the selected initialized client is injected through runner → Agent → Reranker. Rule ranking remains authoritative and every unavailable, malformed, disabled, or circuit-open path preserves it.

**Tech Stack:** Python 3.10+, `dataclasses`, `json`, `unittest`, `unittest.mock`; `openai>=1.0,<2.0` and its `httpx` dependency; OpenAI-compatible Chat Completions only.

**Spec:** `docs/superpowers/specs/2026-08-27-unified-llm-client-reranking-design.md`

## Global Constraints

- Support exactly one selected provider per process: `none`, `deepseek`, or `openai`.
- Do not implement local-model loading or automatic cross-provider fallback.
- `DEEPSEEK_API_KEY` and `OPENAI_API_KEY` are the only credential sources; never log or ordinarily serialize raw values.
- Configuration precedence is dataclass defaults < JSON < environment < explicit overrides.
- `LLM_PROVIDER` wins over legacy `LLM_BACKEND`; map legacy `openai` to `openai` and `none`/`local` to `none` when `LLM_PROVIDER` is absent.
- `LLM_MODEL` and `LLM_BASE_URL` override only the selected provider after provider-specific overrides.
- Express token parameter and temperature support through profile configuration, not model-name inference.
- Importing or constructing configuration, Agent, Reranker, or clients must not send a request.
- Startup performs at most three total attempts and SDK internal retries remain zero.
- Circuit threshold changes public client status immediately to `unavailable` / `circuit_open`.
- Reranking is the only LLM business use in this plan; do not add LLM intent, slots, clarification, or message generation.
- LLM reranking receives at most 12 compact rule-ranked candidates by default and validates every returned ASIN locally.
- Any LLM failure or invalid output preserves the complete deterministic rule order.
- Startup probe usage is cumulative-only; Agent per-turn usage includes only that turn's rerank result.
- Do not modify `evaluator/local_evaluator.py`, `starter/agent.py`, public data, Agent response schema, or scoring behavior.
- All tests use mocks, make no real API call, do not sleep, and remain Python 3.10 compatible.
- The worktree begins with interrupted, unstaged test changes in `tests/test_config_loader.py` and `tests/test_deepseek_client.py`. Preserve useful secret, submit, circuit, jitter, and real-exception cases; replace obsolete assertions that separate DeepSeek from legacy OpenAI loading. Do not discard either file wholesale.

---

## File Structure

### Create

- `llm/openai_compatible.py` — the only OpenAI-SDK request/retry/response/circuit implementation.
- `tests/test_openai_compatible_client.py` — cross-provider and capability behavior.
- `tests/test_reranker_llm.py` — provider-neutral reranking, parsing, fallback, payload, and usage tests.

### Modify

- `config/default.json` — shared LLM settings plus DeepSeek/OpenAI profiles.
- `config/models.py` — `SecretValue`, `ProviderConfig`, provider collection, and revised `LLMConfig`.
- `config/loader.py` — profile merge, selected-provider overrides, credential resolution, capability validation, and legacy mapping.
- `config/env_config.py` — compatibility properties over the selected unified profile and submit/offline behavior.
- `config/__init__.py` — export new model types.
- `llm/base.py` — cumulative usage and disabled-client behavior if required by the shared client.
- `llm/deepseek.py` — thin compatibility wrapper only.
- `llm/factory.py` — selected-provider construction.
- `llm/__init__.py` — shared client exports.
- `agent/reranker.py` — remove direct SDK loading and use injected `LLMClient`.
- `agent/main_agent.py` — accept and pass an optional client.
- `run_local_eval.py` — inject the initialized client into Agent.
- `README.md` — provider profiles, capability flags, reranking protocol, migration, and fallback.
- `requirements.txt` — correct wording so SDK support is not described as provider-specific clarification.
- `tests/test_config_loader.py` — adapt interrupted tests and add provider-profile coverage.
- `tests/test_deepseek_client.py` — retain only compatibility and shared-behavior cases that belong here.
- `tests/test_llm_startup.py` — assert runner-to-Agent client injection and safe degradation.

---

### Task 1: Multi-Provider Configuration Profiles and Opaque Secrets

**Files:**
- Modify: `config/default.json`
- Modify: `config/models.py`
- Modify: `config/loader.py`
- Modify: `config/env_config.py`
- Modify: `config/__init__.py`
- Modify: `llm/deepseek.py` (narrow compatibility bridge only; Task 2 replaces the implementation)
- Modify: `tests/test_config_loader.py`
- Modify: `tests/test_deepseek_client.py` only to adapt committed construction helpers to provider profiles; interrupted circuit/jitter/SDK-exception hunks are removed with a patch before Task 1 full-suite verification and explicitly reintroduced by Task 2.

**Interfaces:**
- Produces: `SecretValue.reveal() -> str`, with redacted `str`, `repr`, and deepcopy/serialization behavior.
- Produces: immutable `ProviderConfig(model, base_url, token_limit_parameter, supports_temperature, api_key)`.
- Produces: immutable `ProviderConfigs(deepseek, openai)`.
- Produces: `LLMConfig.selected_profile: ProviderConfig | None`.
- Preserves read-only `LLMConfig.model`, `LLMConfig.base_url`, and `LLMConfig.api_key` properties over the selected profile until callers migrate; `api_key` returns `SecretValue`, never `str`.
- Preserves: `load_config(path=None, overrides=None, environ=None) -> AppConfig` and current flat `EnvConfig` properties.

- [ ] **Step 1: Adapt interrupted configuration tests into the new RED suite**

Keep the interrupted `asdict` secret test, but construct the client only in Task 2. In Task 1 the test must recursively inspect `asdict(config)` and `json.dumps(asdict(config), default=str)`:

```python
def contains_string(value: object, expected: str) -> bool:
    if isinstance(value, str):
        return expected in value
    if isinstance(value, dict):
        return any(contains_string(item, expected) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_string(item, expected) for item in value)
    return False


def test_dataclass_serialization_never_contains_selected_provider_key(self) -> None:
    secret = "selected-provider-secret"
    config = load_config(environ={
        "LLM_PROVIDER": "deepseek",
        "DEEPSEEK_API_KEY": secret,
        "OPENAI_API_KEY": "inactive-secret",
    })
    serialized = asdict(config)
    self.assertFalse(contains_string(serialized, secret))
    self.assertFalse(contains_string(serialized, "inactive-secret"))
    self.assertNotIn(secret, json.dumps(serialized, default=str))
    self.assertEqual(config.llm.selected_profile.api_key.reveal(), secret)
```

Replace interrupted legacy-model-separation assertions with selected-profile behavior:

```python
def test_provider_specific_values_persist_and_selected_override_wins(self) -> None:
    config = load_config(environ={
        "LLM_PROVIDER": "openai",
        "DEEPSEEK_MODEL": "deepseek-reasoner",
        "OPENAI_MODEL": "gpt-profile-model",
        "LLM_MODEL": "selected-openai-model",
    })
    self.assertEqual(config.llm.providers.deepseek.model, "deepseek-reasoner")
    self.assertEqual(config.llm.providers.openai.model, "selected-openai-model")
    self.assertEqual(config.llm.selected_profile.model, "selected-openai-model")
```

Add tests for:

- default DeepSeek and OpenAI profile values;
- `LLM_PROVIDER` precedence over `LLM_BACKEND`;
- legacy backend mapping when `LLM_PROVIDER` is absent;
- `LLM_BASE_URL` affecting only the selected profile;
- only the selected provider receiving its matching key;
- JSON and explicit overrides rejecting `api_key` both at legacy `llm.api_key` and nested `llm.providers.<provider>.api_key` paths;
- `max_tokens` / `max_completion_tokens` validation;
- strict boolean parsing for `supports_temperature`;
- `LLM_RERANK_CANDIDATES > 0`;
- submit mode: external provider without its selected key is offline, with its key is online;
- inactive-provider key never makes the selected provider online.

- [ ] **Step 2: Run focused tests and capture RED evidence**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_config_loader -v
```

Expected: failures for missing profile types, selected profile, secret wrapper, and provider precedence.

- [ ] **Step 3: Implement immutable profile and secret types**

Add these public shapes in `config/models.py`:

```python
class SecretValue:
    __slots__ = ("__value",)

    def __init__(self, value: str = "") -> None:
        self.__value = value

    def reveal(self) -> str:
        return self.__value

    def __bool__(self) -> bool:
        return bool(self.__value)

    def __str__(self) -> str:
        return "<set>" if self else "<unset>"

    __repr__ = __str__


@dataclass(frozen=True)
class ProviderConfig:
    model: str
    base_url: str
    token_limit_parameter: str
    supports_temperature: bool
    api_key: SecretValue = field(default_factory=SecretValue, repr=False)


@dataclass(frozen=True)
class ProviderConfigs:
    deepseek: ProviderConfig
    openai: ProviderConfig
```

Give `SecretValue` an explicit `__deepcopy__` that returns another opaque `SecretValue`, never a raw string. Revise `LLMConfig` to include shared settings, `rerank_enabled`, `rerank_candidates`, and `providers`; add a `selected_profile` property returning `None` for provider `none`.

Add read-only compatibility properties so committed callers can migrate without a broken intermediate commit:

```python
@property
def model(self) -> str:
    return self.selected_profile.model if self.selected_profile else ""

@property
def base_url(self) -> str:
    return self.selected_profile.base_url if self.selected_profile else ""

@property
def api_key(self) -> SecretValue:
    profile = self.selected_profile
    return profile.api_key if profile else SecretValue()
```

- [ ] **Step 4: Implement profile-aware loading and validation**

Update `config/default.json` exactly as the spec's provider-profile example while retaining unrelated existing settings.

In `config/loader.py`:

- keep JSON and explicit secret rejection;
- merge nested provider profiles without erasing siblings;
- map provider-specific environment variables first;
- select provider using `LLM_PROVIDER`, then explicit legacy `LLM_BACKEND`, then JSON;
- apply `LLM_MODEL` / `LLM_BASE_URL` only to the selected profile after provider-specific variables;
- wrap only the selected matching environment key in `SecretValue`; both profiles receive empty secrets when provider is `none`;
- validate provider, model/base URL non-empty strings, exact token parameter enum, booleans, positive rerank count, existing finite timeout/retry constraints.

Do not scan or retain arbitrary environment variables.

- [ ] **Step 5: Update EnvConfig compatibility without reintroducing direct model loading**

Map:

```python
llm_backend -> app_config.llm.provider
llm_model -> selected_profile.model if selected_profile else ""
openai_api_key -> selected OpenAI SecretValue.reveal() only when provider == "openai", else ""
openai_base_url -> providers.openai.base_url
llm_rerank -> app_config.llm.rerank_enabled
offline -> provider == "none" or selected_profile is None or not selected_profile.api_key
```

Ensure custom `repr` and `summary` contain only selected provider, model, and key-presence boolean. They must never call `reveal()` for formatting.

As a narrow Task 1 bridge, update the existing `DeepSeekClient` to reveal `config.api_key` only at the SDK and sanitization boundaries. Do not otherwise refactor its request, retry, or circuit logic yet:

```python
raw_key = self._config.api_key.reveal()
self._sdk_factory(
    api_key=raw_key,
    base_url=self._config.base_url,
    timeout_seconds=self._config.timeout_seconds,
    connect_timeout_seconds=self._config.connect_timeout_seconds,
)
message = str(error).replace(raw_key, "[redacted]") if raw_key else str(error)
```

Adapt the committed `tests/test_deepseek_client.py::make_config` helper to build `ProviderConfigs` and put the requested key/model/base URL in the selected DeepSeek `ProviderConfig`. Preserve all committed behavioral assertions.

- [ ] **Step 6: Run focused and full tests**

Before the full suite, patch only the interrupted, uncommitted circuit/jitter/actual-SDK-exception additions out of `tests/test_deepseek_client.py`. Do not use `git checkout`; preserve committed behavioral tests while adapting only their configuration helper to the new constructor. The removed cases are exact Task 2 requirements and will be reintroduced against `OpenAICompatibleClient` there.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_config_loader -v
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover tests -v
```

Expected: all configuration and existing committed tests pass with no unexplained working-tree test failure.

- [ ] **Step 7: Commit Task 1**

```bash
git add config/default.json config/models.py config/loader.py config/env_config.py config/__init__.py llm/deepseek.py tests/test_config_loader.py tests/test_deepseek_client.py
git commit -m "feat: add multi-provider LLM configuration"
```

---

### Task 2: Shared OpenAI-Compatible Client

**Files:**
- Create: `llm/openai_compatible.py`
- Create: `tests/test_openai_compatible_client.py`
- Modify: `llm/base.py`
- Modify: `llm/deepseek.py`
- Modify: `llm/factory.py`
- Modify: `llm/__init__.py`
- Modify: `tests/test_deepseek_client.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: `LLMConfig.selected_profile` and `SecretValue.reveal()` from Task 1.
- Produces: `OpenAICompatibleClient(config: LLMConfig, *, sdk_factory=None, sleep=time.sleep, jitter=None, clock=time.monotonic, failure_classifier=None)`.
- Produces: `create_llm_client(config: LLMConfig) -> LLMClient` for all three providers.
- Preserves: `DeepSeekClient` as a thin compatibility wrapper with no duplicated logic.
- Produces: `LLMClient.cumulative_usage -> LLMUsage`.

- [ ] **Step 1: Create cross-provider RED tests**

Create `tests/test_openai_compatible_client.py` with a shared fake SDK and verify exact kwargs:

```python
def test_deepseek_uses_max_tokens_and_temperature(self) -> None:
    client, sdk = make_client(provider="deepseek")
    client.initialize()
    client.chat([{"role": "user", "content": "rank"}], temperature=0.2, max_tokens=77)
    kwargs = sdk.chat.completions.create.call_args.kwargs
    self.assertEqual(kwargs["max_tokens"], 77)
    self.assertEqual(kwargs["temperature"], 0.2)
    self.assertNotIn("max_completion_tokens", kwargs)

def test_openai_capability_uses_max_completion_tokens_and_omits_temperature(self) -> None:
    config = config_with_selected_profile(
        "openai", token_limit_parameter="max_completion_tokens", supports_temperature=False
    )
    client, sdk = make_client(config=config)
    client.initialize()
    client.chat([{"role": "user", "content": "rank"}], temperature=0.2, max_tokens=77)
    kwargs = sdk.chat.completions.create.call_args.kwargs
    self.assertEqual(kwargs["max_completion_tokens"], 77)
    self.assertNotIn("max_tokens", kwargs)
    self.assertNotIn("temperature", kwargs)
```

Add tests for correct selected key/base URL/model, no inactive-provider construction, `provider=none`, missing selected key, and compatibility `DeepSeekClient` delegation.

- [ ] **Step 2: Adapt interrupted client tests and add final-review regressions**

Keep and update the interrupted tests for:

- status immediately becoming `UNAVAILABLE` / `CIRCUIT_OPEN` at threshold;
- bounded default `random.uniform` jitter with injected sleep;
- actual OpenAI SDK exceptions: `AuthenticationError`, `PermissionDeniedError`, HTTP 408 `APIStatusError`, `RateLimitError`, `InternalServerError`, unknown-status `APIStatusError`, `APITimeoutError`, and `APIConnectionError`.

Move provider-neutral cases to `tests/test_openai_compatible_client.py`. Leave `tests/test_deepseek_client.py` focused on compatibility construction/delegation so it does not duplicate the full behavior suite.

Add cumulative usage assertions:

```python
self.assertEqual(client.cumulative_usage.prompt_tokens, probe_prompt + chat_prompt)
self.assertEqual(client.cumulative_usage.completion_tokens, probe_completion + chat_completion)
```

- [ ] **Step 3: Run client tests and capture RED evidence**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_openai_compatible_client tests.test_deepseek_client -v
```

Expected: import or behavior failures because the shared client and profile-aware request builder do not exist.

- [ ] **Step 4: Move all SDK behavior into OpenAICompatibleClient**

Move the implementation from `llm/deepseek.py` to `llm/openai_compatible.py` and adapt it to `config.selected_profile`.

SDK construction must pass:

```python
OpenAI(
    api_key=profile.api_key.reveal(),
    base_url=profile.base_url,
    timeout=httpx.Timeout(config.timeout_seconds, connect=config.connect_timeout_seconds),
    max_retries=0,
)
```

Build Chat Completions kwargs through one helper:

```python
kwargs = {"model": profile.model, "messages": list(messages)}
kwargs[profile.token_limit_parameter] = requested_max_tokens
if profile.supports_temperature and temperature is not None:
    kwargs["temperature"] = temperature
```

Preserve safe completion decoding, error sanitization, retry classification, three-attempt startup cap, health-check-disabled initialization ruling, and structured runtime failures.

Use a production jitter callable based on `random.uniform(0.0, 0.1)` when no jitter is injected. When the circuit threshold is reached, replace public status immediately with `LLMStatus(state=LLMState.UNAVAILABLE, provider=config.provider, model=profile.model, attempts=0, error_category=LLMErrorCategory.CIRCUIT_OPEN, error_message="circuit open")`.

Accumulate reliable probe and chat usage in a client-owned immutable `LLMUsage` snapshot. Per-call `LLMResult.usage` remains unchanged.

- [ ] **Step 5: Reduce DeepSeekClient to compatibility delegation and update factory**

`llm/deepseek.py` may contain classification re-exports needed by existing callers, but `DeepSeekClient` must subclass or construct `OpenAICompatibleClient` without overriding request/retry/circuit methods.

`create_llm_client()` returns:

- `DisabledLLMClient` for provider `none`;
- `OpenAICompatibleClient` for `deepseek` or `openai`;
- an error for any impossible unvalidated provider.

Update `llm/__init__.py` exports and correct `requirements.txt` wording to “OpenAI-compatible DeepSeek/OpenAI client and optional reranking.”

- [ ] **Step 6: Run focused and full tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_openai_compatible_client tests.test_deepseek_client -v
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover tests -v
```

Expected: all tests pass with zero real requests and zero real sleeps.

- [ ] **Step 7: Commit Task 2**

```bash
git add llm requirements.txt tests/test_openai_compatible_client.py tests/test_deepseek_client.py
git commit -m "refactor: unify OpenAI-compatible LLM clients"
```

---

### Task 3: Provider-Neutral Reranking and Dependency Injection

**Files:**
- Create: `tests/test_reranker_llm.py`
- Modify: `agent/reranker.py`
- Modify: `agent/main_agent.py`

**Interfaces:**
- Consumes: initialized `LLMClient` and `LLMResult` from Task 2.
- Produces: `Reranker(env: EnvConfig | None = None, llm_client: LLMClient | None = None)`.
- Produces: `Agent(catalog_path="data/catalog.jsonl", env=None, llm_client=None)`.
- Preserves: `Reranker.rerank(retriever, candidates, state, route, top_k, mode) -> list[str]` and Agent response schema.

- [ ] **Step 1: Write RED tests for disabled and failed fallback**

Use these focused helpers in `tests/test_reranker_llm.py` so the test never loads the catalog or SDK:

```python
class FakeRetriever:
    def __init__(self, products: dict[str, dict]) -> None:
        self.products = products

    def product(self, asin: str) -> dict | None:
        return self.products.get(asin)

    def text_lower(self, asin: str) -> str:
        return " ".join(str(value) for value in self.products[asin].values()).lower()


class FakeAvailableClient:
    def __init__(self, result: LLMResult, provider: str = "deepseek") -> None:
        self._status = LLMStatus(LLMState.AVAILABLE, provider, result.model)
        self.chat = Mock(return_value=result)

    @property
    def status(self) -> LLMStatus:
        return self._status


def env_with_rerank(enabled: bool = True, candidates: int = 12) -> object:
    return SimpleNamespace(
        llm=SimpleNamespace(rerank_enabled=enabled, rerank_candidates=candidates)
    )


def make_rule_case() -> tuple[FakeRetriever, list[dict], object, object, list[str]]:
    products = {
        "A": {"parent_asin": "A", "title": "Alpha", "categories": ["Shoes"],
              "features": ["light"], "rating_number": 0, "average_rating": 0},
        "B": {"parent_asin": "B", "title": "Beta", "categories": ["Shoes"],
              "features": ["wide"], "rating_number": 0, "average_rating": 0},
    }
    retriever = FakeRetriever(products)
    candidates = [{"parent_asin": "A", "rrf": 2.0}, {"parent_asin": "B", "rrf": 1.0}]
    state = SimpleNamespace(hard=[], soft=[], active=[], user_profile={})
    route = SimpleNamespace(category_tokens=[])
    return retriever, candidates, state, route, ["A", "B"]
```

Verify:

```python
def test_disabled_client_preserves_rule_order(self) -> None:
    retriever, candidates, state, route, rule_order = make_rule_case()
    reranker = Reranker(env=env_with_rerank(True), llm_client=DisabledLLMClient())
    actual = reranker.rerank(retriever, candidates, state, route, top_k=10, mode="probe")
    self.assertEqual(actual, rule_order)

def test_failed_client_preserves_rule_order_and_reports_returned_usage(self) -> None:
    retriever, candidates, state, route, rule_order = make_rule_case()
    client = FakeAvailableClient(result=LLMResult(
        success=False,
        provider="deepseek",
        model="deepseek-chat",
        usage=LLMUsage(prompt_tokens=4, completion_tokens=0),
        error_category=LLMErrorCategory.TIMEOUT,
    ))
    reranker = Reranker(env=env_with_rerank(True), llm_client=client)
    actual = reranker.rerank(
        retriever, candidates, state, route, top_k=10, mode="probe"
    )
    self.assertEqual(actual, rule_order)
    self.assertEqual(reranker.last_usage, {"prompt_tokens": 4, "completion_tokens": 0})
```

Also verify `LLM_RERANK=false`, unavailable state, and fewer than two candidates skip `chat()` completely and reset `last_usage` to zero.

- [ ] **Step 2: Write RED tests for payload and response compatibility**

Require the user message content to be JSON with exactly two top-level keys: `constraints` and `candidates`. Each candidate has exactly `parent_asin`, `title`, `categories`, and `features`. Assert title and joined categories are truncated to 240 characters, normalized features to 800 characters, and the joined active constraints to 800 characters. Assert it excludes details, profile, conversation history, and arbitrary full product data.

Test all accepted forms: the object string `{"ranked_parent_asins": ["B", "A"]}`, the bare array string `["B", "A"]`, and the same object wrapped in a Markdown `json` code fence.

For each, verify unknown IDs and duplicates are removed and omitted candidates append in original rule order. Empty or malformed output must return the exact rule order.

Run the same successful test with fake provider names `deepseek` and `openai` to prove provider-neutral behavior. Assert current-call usage reaches `last_usage`.

- [ ] **Step 3: Write RED constructor-injection tests**

Patch `Reranker` construction in `agent.main_agent` and verify an explicitly passed client is the identical object received by Reranker. With a temporary catalog path, verify direct `Agent(catalog_path=temp_catalog, llm_client=None)` and direct `Reranker(env=test_env, llm_client=None)` use disabled clients and do not call the factory or SDK.

- [ ] **Step 4: Run focused tests and capture RED evidence**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_reranker_llm -v
```

Expected: failures because constructors do not accept clients and Reranker still owns the old OpenAI SDK path.

- [ ] **Step 5: Implement compact prompt construction and strict local parsing**

In `agent/reranker.py`:

- accept or create a `DisabledLLMClient` without online factory use;
- reset `last_usage` at the start of every rerank;
- produce the deterministic rule order first;
- call the client only under the three invocation conditions in the spec;
- build at most `env.llm.rerank_candidates` compact candidates;
- construct the user message as `json.dumps({"constraints": constraints[:800], "candidates": compact_candidates}, ensure_ascii=False)`; compact candidates use `str(title)[:240]`, joined categories `[:240]`, and normalized/joined features `[:800]`;
- call `llm_client.chat()` once;
- copy the returned current-call usage whether success or structured failure;
- parse object, bare-array, and fenced JSON;
- validate against the submitted candidate set, deduplicate, append missing submitted candidates in rule order, then append every unsent tail candidate in rule order;
- preserve rule order for every failure or empty valid ranking.

Delete `_llm_rerank_openai()`, its SDK import, direct API-key reads, and provider-specific condition.

- [ ] **Step 6: Inject the client through Agent**

Update `agent/main_agent.py` constructor and pass the exact client to `Reranker`. No other Agent module receives it. Keep response `usage` sourced from `reranker.last_usage`.

- [ ] **Step 7: Run focused and full tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_reranker_llm -v
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover tests -v
```

Expected: all tests pass; no SDK construction occurs from direct Agent/Reranker construction.

- [ ] **Step 8: Commit Task 3**

```bash
git add agent/main_agent.py agent/reranker.py tests/test_reranker_llm.py
git commit -m "feat: rerank through unified LLM client"
```

---

### Task 4: Runner Injection, Documentation, and End-to-End Verification

**Files:**
- Modify: `run_local_eval.py`
- Modify: `tests/test_llm_startup.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `initialize_llm(env) -> LLMClient` and Agent's new `llm_client` parameter.
- Preserves: CLI arguments, results format, evaluator calls, and submit-mode assertion.

- [ ] **Step 1: Write RED runner injection tests**

Extend `tests/test_llm_startup.py` so a patched Agent constructor captures the exact client returned by `initialize_llm`. Avoid loading the missing catalog by patching data-loading/evaluation boundaries or by testing an extracted `build_agent(env, catalog_path, client)` helper.

Required assertions:

- disabled and unavailable clients are still injected safely;
- selected available DeepSeek and OpenAI clients use the same path;
- status text never includes either test secret;
- the client is not reconstructed after initialization;
- submit mode without the selected key remains offline; with a selected key the existing assertion prevents online submit before initialization.

- [ ] **Step 2: Run startup tests and capture RED evidence**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_llm_startup -v
```

Expected: injection assertions fail because `run_local_eval.py` currently discards the initialized client.

- [ ] **Step 3: Pass the initialized client into Agent**

Retain the existing explicit startup ordering:

```text
load EnvConfig
check submit offline policy
initialize selected client
verify/load data
construct Agent with the same client
evaluate
```

Modify only enhanced `run_local_eval.py`; official `evaluator/local_evaluator.py` stays unchanged.

- [ ] **Step 4: Update README for unified providers and reranking**

Document:

- JSON provider profile structure;
- `LLM_PROVIDER`, legacy `LLM_BACKEND` mapping, provider-specific keys/models/base URLs, and selected-provider overrides;
- capability flags and examples for models that use different token parameters or omit temperature;
- single-provider/no-automatic-fallback policy;
- startup states and retry/circuit behavior;
- compact 12-candidate reranking payload and strict local response validation;
- `LLM_RERANK=0` deterministic opt-out;
- startup versus per-turn token accounting;
- migration from the old Reranker-owned OpenAI loader;
- offline run with no key and provider-specific online examples;
- no local-model implementation in this change.

Remove claims that DeepSeek is probe-only and correct any statement implying LLM clarification is implemented.

- [ ] **Step 5: Run focused and full verification**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_llm_startup -v
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover tests -v
PYTHONDONTWRITEBYTECODE=1 env -u DEEPSEEK_API_KEY -u OPENAI_API_KEY python -c 'from config import EnvConfig; from llm import create_llm_client; c=EnvConfig.from_env(); x=create_llm_client(c.llm); print(x.initialize().state.value)'
git diff --exit-code 15f79b5..HEAD -- evaluator/local_evaluator.py starter/agent.py data/public_set.jsonl
```

Expected: all tests pass; offline command prints `disabled` with no network; protected-file diff is empty.

If `data/catalog.jsonl` remains absent, record catalog evaluation as skipped. If present, run one offline sample with `SAMPLE_LIMIT=1 SKIP_DATA_VERIFY=1` and verify startup state appears before evaluation.

- [ ] **Step 6: Review uncommitted and tracked boundaries before commit**

```bash
git status --short
git diff --check
git diff --name-only 15f79b5..HEAD
```

Confirm the interrupted test hunks have either been adapted and committed by Tasks 1/2 or intentionally removed through reviewed patches; no unexplained unstaged changes may remain.

- [ ] **Step 7: Commit Task 4**

```bash
git add run_local_eval.py tests/test_llm_startup.py README.md
git commit -m "feat: inject selected LLM client into evaluation"
```

- [ ] **Step 8: Record final evidence**

```bash
git status --short
git log --oneline -8
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover tests -v
```

Expected: clean worktree, the four revised-scope task commits visible, and the complete suite passing with exact count reported.
