# Legacy-Dynamic Hybrid Question Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a default-off Hybrid policy that preserves Legacy ask/stop behavior and may replace one repeated `other` question per session with a high-value concrete attribute, then compare Legacy with three Hybrid gates on one deterministic 20-session sample.

**Architecture:** `QuestionPolicy` computes the Legacy decision first and delegates only repeated-`other` decisions to a focused `HybridQuestionPolicy`. A frozen catalog-resource bundle allows four isolated Agents to share one retriever index, one global-signal object, and one per-ASIN attribute cache during the experiment; no dialogue state, Reranker state, or counters are shared. The experiment uses the official evaluator, rule-only mode, depth one, atomic reporting, and a 1,200-second process deadline.

**Tech Stack:** Python 3.10+, frozen dataclasses, standard-library JSON/time/signal/statistics, existing SQLite FTS5 retriever, existing official evaluator, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-30-legacy-dynamic-hybrid-design.md`

## Global Constraints

- Preserve the official `reset(...)` and `respond(...)` interface and the exact response keys `message`, `ask_attribute`, `recommendations`, and `usage`.
- Do not modify `evaluator/`, retrieval scoring, reranking formulas, Top10 ordering, intent recognition, reducer semantics unrelated to the replacement counter, or Transition Guard behavior.
- Hybrid is deterministic, rule-only, default-off, depth-one, and cannot change a Legacy stop decision or a Guard clarification.
- Preserve the first `other`; replace only a later Legacy-selected `other`; commit at most one replacement per complete session, including across intent overrides.
- Full dynamic mode and Hybrid mode are mutually exclusive.
- Catalog/cache failure and candidate-signal failure fall back to the already-computed Legacy decision.
- Runtime never reads scenario labels, ground truth, hidden intent cards, or simulator state.
- The experiment uses exactly 20 deterministic stratified public samples: 8 Buying, 8 Browsing, 3 Intent Override, and 1 Boundary.
- Build the shared product snapshot, retriever index, global catalog signals, and attribute cache exactly once for the comparison process.
- Each compared version owns independent dialogue sessions, policies, Reranker, diagnostics, and usage counters.
- Disable LLM, two-step lookahead, and decision-trace export for the experiment.
- Stop the process after 1,200 seconds. A partial report must have `status="time_budget_exceeded"` and cannot name a winner.
- Do not promote a new default from this 20-session screen.
- Use the existing project virtual environment at `/Users/zhengce/projects/techjam_shopping_copilot/techjam-shopping-copilot/.conda/bin/python`.

---

## File Structure

- `config/models.py`: frozen Hybrid configuration types.
- `config/loader.py`: JSON/environment construction, validation, and mutual-exclusion checks.
- `config/default.json`: default-off Hybrid configuration.
- `agent/dialogue/models.py`: per-session replacement count.
- `agent/dialogue/hybrid_question_policy.py`: pure replacement eligibility, scoring, and reason codes.
- `agent/dialogue/question_policy.py`: Legacy-first orchestration and signal-need probe.
- `agent/dialogue/reducer.py`: atomically record a committed Hybrid replacement.
- `agent/dialogue/catalog_resources.py`: frozen shared catalog-derived resources.
- `agent/dialogue/candidate_signals.py`: concrete-only depth-one calculation option.
- `agent/dialogue/pipeline.py`: resource injection, calculator wiring, and committed counter update.
- `agent/main_agent.py`: optional shared-resource injection and conditional signal calculation.
- `experiments/hybrid_question_comparison.py`: deterministic 20-session comparison and time budget.
- `tests/test_hybrid_question_policy.py`: pure policy behavior.
- `tests/test_dialogue_config.py`, `tests/test_config_loader.py`: configuration behavior.
- `tests/test_state_reducer.py`: session counter semantics.
- `tests/test_dialogue_flow.py`: Agent integration, fallback, cache sharing, and response contract.
- `tests/test_hybrid_question_comparison.py`: sample selection, isolation, timeout, and report behavior.
- `README.md`: Hybrid configuration and reproducible comparison command.

### Task 1: Add frozen Hybrid configuration and session accounting

**Files:**
- Modify: `config/models.py`
- Modify: `config/loader.py`
- Modify: `config/default.json`
- Modify: `agent/dialogue/models.py`
- Modify: `agent/dialogue/reducer.py`
- Modify: `tests/test_dialogue_config.py`
- Modify: `tests/test_config_loader.py`
- Modify: `tests/test_state_reducer.py`

**Interfaces:**
- Produces: `HybridQuestionWeights`, `HybridQuestionPolicyConfig`, `DecisionConfig.hybrid_question_policy`.
- Produces: `DialogueState.hybrid_replacements_used: int` and `StateReducer.record_question(..., hybrid_replacement: bool = False)`.
- Consumes: existing `load_config`, environment-overlay naming, immutable `DialogueState`, and allowed attributes.

- [ ] **Step 1: Write failing configuration tests**

Add assertions equivalent to:

```python
decision = load_config(environ={}).decision
assert decision.hybrid_question_policy.enabled is False
assert decision.hybrid_question_policy.pool_size == 300
assert decision.hybrid_question_policy.max_replacements_per_session == 1
assert decision.hybrid_question_policy.only_after_other_asked is True

configured = load_config(
    environ={"SHOPPING_DECISION__HYBRID_QUESTION_POLICY__ENABLED": "1"},
).decision.hybrid_question_policy
assert configured.enabled is True
```

Cover every numeric threshold and weight through explicit JSON overrides. Add rejection cases for non-finite/negative weights, thresholds outside `[0, 1]`, nonpositive pool/temperature, `max_replacements_per_session` outside `{0, 1}`, `only_after_other_asked=false`, and simultaneous `candidate_question_value.enabled=true` plus Hybrid enabled.

- [ ] **Step 2: Write failing reducer tests**

```python
state = DialogueState(session_id="s", user_profile={})
state = StateReducer.record_question(state, "material", hybrid_replacement=True)
assert state.asked_attributes == ("material",)
assert state.hybrid_replacements_used == 1

override = reducer.reduce(
    state,
    recognition(
        DialogueAct.REPLACE_CONSTRAINT,
        operation(OperationKind.REPLACE, "material", "cotton"),
    ),
    turn=3,
).state
assert override.hybrid_replacements_used == 1
```

Also assert an ordinary question does not increment the counter and a rejected/stale turn cannot call this mutation path.

- [ ] **Step 3: Run focused tests and verify RED**

```bash
/Users/zhengce/projects/techjam_shopping_copilot/techjam-shopping-copilot/.conda/bin/python \
  -m pytest -q -p no:cacheprovider tests/test_dialogue_config.py \
  tests/test_config_loader.py tests/test_state_reducer.py
```

Expected: missing Hybrid configuration fields and replacement accounting.

- [ ] **Step 4: Implement minimal frozen configuration and validation**

Use these exact defaults:

```python
@dataclass(frozen=True)
class HybridQuestionWeights:
    expected_shrink: float = 0.40
    resolve_at_10: float = 0.25
    coverage: float = 0.15
    answer_probability: float = 0.10
    extraction_confidence: float = 0.10
    missing_penalty: float = 0.25
    turn_cost: float = 0.10

@dataclass(frozen=True)
class HybridQuestionPolicyConfig:
    enabled: bool = False
    max_replacements_per_session: int = 1
    only_after_other_asked: bool = True
    pool_size: int = 300
    prior_alpha: float = 0.25
    prior_temperature: float = 1.0
    minimum_coverage: float = 0.60
    maximum_missing_rate: float = 0.40
    minimum_expected_shrink: float = 0.25
    minimum_resolve_at_10: float = 0.05
    minimum_gain: float = 0.25
    weights: HybridQuestionWeights = field(default_factory=HybridQuestionWeights)
```

Add environment overrides under `SHOPPING_DECISION__HYBRID_QUESTION_POLICY__...` and `...__WEIGHTS__...`. Validate all probability thresholds in `[0, 1]`, weights nonnegative, pool/temperature positive, replacement count in `{0, 1}`, `only_after_other_asked is True`, and mutual exclusion with full dynamic mode.

- [ ] **Step 5: Implement atomic session accounting**

Add `hybrid_replacements_used: int = 0` to `DialogueState`. Extend `record_question` with a keyword-only `hybrid_replacement=False`; increment only when a legal question is actually recorded as the Hybrid replacement. Preserve the count through `StateReducer.reduce`, including intent override.

- [ ] **Step 6: Run focused tests and Ruff**

Run the Step 3 command, then:

```bash
/Users/zhengce/projects/techjam_shopping_copilot/techjam-shopping-copilot/.conda/bin/python \
  -m ruff check --no-cache config agent/dialogue/models.py agent/dialogue/reducer.py \
  tests/test_dialogue_config.py tests/test_config_loader.py tests/test_state_reducer.py
```

Expected: all pass.

- [ ] **Step 7: Commit Task 1**

```bash
git add config/models.py config/loader.py config/default.json agent/dialogue/models.py \
  agent/dialogue/reducer.py tests/test_dialogue_config.py tests/test_config_loader.py \
  tests/test_state_reducer.py
git commit -m "feat: configure hybrid question replacement"
```

### Task 2: Implement the pure Legacy-first Hybrid policy

**Files:**
- Create: `agent/dialogue/hybrid_question_policy.py`
- Create: `tests/test_hybrid_question_policy.py`
- Modify: `agent/dialogue/question_policy.py`
- Modify: `agent/dialogue/pipeline.py`
- Modify: `tests/test_question_policy.py`

**Interfaces:**
- Consumes: `HybridQuestionPolicyConfig`, Legacy `QuestionDecision`, `DialogueState`, `CatalogQuestionSignals`, and `CandidateQuestionSignals`.
- Produces: `HybridQuestionPolicy.consider(state, legacy_decision, catalog_signals, candidate_signals) -> QuestionDecision`.
- Produces: `HybridQuestionPolicy.statistics() -> dict[str, object]` with bounded aggregate reason, selected-attribute, replacement, and latency data.
- Produces: `QuestionPolicy.needs_candidate_signals(state, recognition) -> bool`.
- Produces: `DialogueUnderstandingPipeline.needs_candidate_signals(pending) -> bool`, which rejects Guard-owned turns before delegating to `QuestionPolicy`.

- [ ] **Step 1: Write failing pure-policy tests**

Build hand-calculated candidate signals and assert:

```python
legacy_stop = QuestionDecision(False, None, "maximum_questions_reached", 0.0, {})
assert hybrid.consider(state, legacy_stop, static, candidates) == legacy_stop

first = hybrid.consider(
    replace(state, asked_attributes=()),
    QuestionDecision(True, "other", "ask_other_first", 1.0, {}),
    static,
    candidates,
)
assert first.ask_attribute == "other"
assert first.reason_code == "hybrid_first_other_preserved"

replacement = hybrid.consider(
    replace(state, asked_attributes=("other",)),
    QuestionDecision(True, "other", "ask_other_first", 1.0, {}),
    static,
    candidates,
)
assert replacement.ask_attribute == "material"
assert replacement.reason_code == "hybrid_specific_replacement"
```

Cover an existing constraint, previously asked attribute, no-preference attribute, known category, non-finite signal, failed threshold, absent signals, counter already used, and deterministic tie order. Assert no case converts a Legacy stop or concrete question into another action.

Add one statistics test that makes a preserved-first-other call and a replacement call, then asserts reason counts, selected-attribute counts, replacement count, and finite nonnegative p50/p95 decision latency without retaining user text or product IDs.

- [ ] **Step 2: Run pure-policy tests and verify RED**

```bash
/Users/zhengce/projects/techjam_shopping_copilot/techjam-shopping-copilot/.conda/bin/python \
  -m pytest -q -p no:cacheprovider tests/test_hybrid_question_policy.py
```

Expected: module missing.

- [ ] **Step 3: Implement the focused policy component**

`HybridQuestionPolicy.consider` must:

1. Return a Legacy stop or concrete question unchanged.
2. Preserve the first `other`.
3. Require unused replacement budget and candidate signals.
4. Filter legal concrete attributes exactly as the spec states.
5. Read static `answer_probability` from `CatalogQuestionSignals.for_category(state.category)`.
6. Clamp finite inputs and reject non-finite attribute rows rather than coercing them into eligibility.
7. Calculate the exact fixed-weight `HybridGain` from the spec.
8. Apply every threshold before selecting by score and `ATTRIBUTE_ORDER`.
9. Return one of the documented Hybrid reason codes and privacy-safe numeric components.

Maintain a lock-protected, bounded in-memory latency sample plus aggregate reason and selected-attribute counters only while Hybrid is enabled. `statistics()` must return a JSON-compatible snapshot and must never retain session IDs, user text, ASINs, titles, or raw candidate values.

- [ ] **Step 4: Integrate Legacy-first orchestration**

Refactor `QuestionPolicy.decide` so the behavior is:

```python
if full_dynamic_is_active:
    return self._decide_dynamic(...)
legacy = self._decide_legacy(...)
if self.config.hybrid_question_policy.enabled:
    return self.hybrid_policy.consider(
        state, legacy, signals, candidate_signals
    )
return legacy
```

`QuestionPolicy.needs_candidate_signals` returns true unconditionally for active full-dynamic mode, and for Hybrid only when the Legacy preview selects a repeated `other` and replacement budget remains. It must return false for first `other`, stops, and concrete Legacy questions. `DialogueUnderstandingPipeline.needs_candidate_signals(pending)` must additionally return false when `pending.guard_decision.action` is `CLARIFY` or `REJECT`.

In `DialogueUnderstandingPipeline.decide_question`, pass `hybrid_replacement=True` to `StateReducer.record_question` only for `hybrid_specific_replacement`.

- [ ] **Step 5: Verify behavior-compatible disabled path**

Run:

```bash
/Users/zhengce/projects/techjam_shopping_copilot/techjam-shopping-copilot/.conda/bin/python \
  -m pytest -q -p no:cacheprovider tests/test_hybrid_question_policy.py \
  tests/test_question_policy.py tests/test_dialogue_flow.py
```

Expected: all existing Legacy and full-dynamic tests plus new Hybrid tests pass.

- [ ] **Step 6: Run Ruff and commit Task 2**

```bash
/Users/zhengce/projects/techjam_shopping_copilot/techjam-shopping-copilot/.conda/bin/python \
  -m ruff check --no-cache agent/dialogue/hybrid_question_policy.py \
  agent/dialogue/question_policy.py agent/dialogue/pipeline.py \
  tests/test_hybrid_question_policy.py tests/test_question_policy.py
git add agent/dialogue/hybrid_question_policy.py agent/dialogue/question_policy.py \
  agent/dialogue/pipeline.py tests/test_hybrid_question_policy.py tests/test_question_policy.py \
  tests/test_dialogue_flow.py
git commit -m "feat: add legacy-first hybrid question policy"
```

### Task 3: Share immutable catalog resources and avoid unnecessary dynamic work

**Files:**
- Create: `agent/dialogue/catalog_resources.py`
- Modify: `agent/dialogue/candidate_signals.py`
- Modify: `agent/dialogue/pipeline.py`
- Modify: `agent/main_agent.py`
- Modify: `tests/test_candidate_signals.py`
- Modify: `tests/test_dialogue_flow.py`

**Interfaces:**
- Produces: `DialogueCatalogResources(catalog_signals, attribute_cache)`.
- Produces: `DialogueCatalogResources.from_products(products, include_attribute_cache)`.
- Extends: `DialogueUnderstandingPipeline(..., catalog_resources: DialogueCatalogResources | None = None)`.
- Extends: `Agent(..., dialogue_catalog_resources: DialogueCatalogResources | None = None)`.
- Extends: `CandidateSignalCalculator.calculate(..., include_other: bool = True)`.
- Produces: `Agent.hybrid_question_statistics() -> dict[str, object]`, forwarding the current Agent's independent Hybrid-policy snapshot.

- [ ] **Step 1: Write failing resource-sharing and concrete-only tests**

Assert one resource factory call can feed multiple Pipelines without rebuilding either derived cache:

```python
resources = DialogueCatalogResources.from_products(PRODUCTS, include_attribute_cache=True)
first = DialogueUnderstandingPipeline(..., products=(), catalog_resources=resources)
second = DialogueUnderstandingPipeline(..., products=(), catalog_resources=resources)
first.reset("a", {})
second.reset("b", {})
assert first.catalog_signals is second.catalog_signals
assert first.candidate_signal_calculator._cache is second.candidate_signal_calculator._cache
assert first.session("a") is not second.session("b")
```

Use mocks/counters to prove `CatalogQuestionSignals.from_products` and `CatalogAttributeCache.from_products` are not called by either injected Pipeline. Assert `calculate(..., include_other=False)` returns `best_other_pair is None`, `other_signal is None`, `lookahead_depth_used == 1`, while concrete signals match the existing depth-one results.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
/Users/zhengce/projects/techjam_shopping_copilot/techjam-shopping-copilot/.conda/bin/python \
  -m pytest -q -p no:cacheprovider tests/test_candidate_signals.py tests/test_dialogue_flow.py
```

Expected: resource type/injection and `include_other` are missing.

- [ ] **Step 3: Implement frozen shared resources**

Create:

```python
@dataclass(frozen=True)
class DialogueCatalogResources:
    catalog_signals: CatalogQuestionSignals
    attribute_cache: CatalogAttributeCache | None

    @classmethod
    def from_products(cls, products, *, include_attribute_cache: bool): ...
```

Materialize products once inside the factory. Build global signals once and the per-ASIN cache only when requested. Existing production construction remains compatible: a Pipeline without injected resources builds exactly one local resource bundle and catches attribute-cache failures by preserving static signals plus a `None` attribute cache.

- [ ] **Step 4: Wire calculators for full dynamic and Hybrid**

When full dynamic is enabled, preserve the current calculator configuration and behavior. When Hybrid is enabled, construct a depth-one calculator from the Hybrid pool/prior configuration and the shared attribute cache. A missing cache leaves the calculator `None` and therefore falls back to Legacy.

Change `Agent._respond_impl` to compute candidate signals only when `dialogue.needs_candidate_signals(pending)` is true. Hybrid calls `calculate(..., include_other=False)`; full dynamic retains `include_other=True`. Select retrieval pool size from the active mode without enabling full dynamic implicitly.

- [ ] **Step 5: Verify state and response isolation with shared resources**

Create two Agents using one injected retriever and one injected `DialogueCatalogResources`, reset different session IDs, and advance only one Agent. Assert the second Agent retains turn-zero state, independent Hybrid counter, independent decision statistics, and an official response with unchanged Top10 ordering.

- [ ] **Step 6: Run focused tests, Ruff, and commit Task 3**

```bash
/Users/zhengce/projects/techjam_shopping_copilot/techjam-shopping-copilot/.conda/bin/python \
  -m pytest -q -p no:cacheprovider tests/test_candidate_signals.py \
  tests/test_dialogue_flow.py tests/test_hybrid_question_policy.py
/Users/zhengce/projects/techjam_shopping_copilot/techjam-shopping-copilot/.conda/bin/python \
  -m ruff check --no-cache agent/dialogue/catalog_resources.py \
  agent/dialogue/candidate_signals.py agent/dialogue/pipeline.py agent/main_agent.py \
  tests/test_candidate_signals.py tests/test_dialogue_flow.py
git add agent/dialogue/catalog_resources.py agent/dialogue/candidate_signals.py \
  agent/dialogue/pipeline.py agent/main_agent.py tests/test_candidate_signals.py \
  tests/test_dialogue_flow.py
git commit -m "perf: share hybrid catalog resources"
```

### Task 4: Add and run the bounded Legacy-versus-Hybrid comparison

**Files:**
- Create: `experiments/hybrid_question_comparison.py`
- Create: `tests/test_hybrid_question_comparison.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `select_stratified_public_samples(samples, seed) -> list[dict]`.
- Produces: `comparison_configurations() -> tuple[dict, ...]` containing Legacy, conservative, balanced, and permissive variants.
- Produces: `run_comparison(catalog_path, dataset_path, seed, time_budget_seconds) -> dict`.
- CLI: `python -m experiments.hybrid_question_comparison --catalog PATH --dataset PATH --output PATH --seed 20260830 --time-budget-seconds 1200`.

- [ ] **Step 1: Write failing deterministic sampling tests**

Build synthetic rows and assert the same seed returns the same 20 sample IDs with exact counts:

```python
selected = select_stratified_public_samples(samples, seed=20260830)
assert Counter(row["scenario_type"] for row in selected) == {
    "buying": 8,
    "browsing": 8,
    "intent_override": 3,
    "boundary": 1,
}
```

Reject missing scenarios, duplicate sample IDs, and source sets smaller than the required strata.

- [ ] **Step 2: Write failing comparison-isolation and timeout tests**

Inject fake retriever/resource/Agent/evaluator factories. Assert:

- the catalog/retriever/resource factories are each called once;
- four separate Agents are created;
- all receive the same retriever/resource identities;
- all evaluate the identical ordered sample IDs;
- every overlay forces `llm.provider=none`, `dialogue_understanding.mode=rule_only`, `finish_strategy.enabled=false`, `lookahead_depth=1`, and decision trace disabled;
- a simulated deadline writes only completed configurations, sets `status="time_budget_exceeded"`, and omits `winner`/`recommendation`.

- [ ] **Step 3: Run tests and verify RED**

```bash
/Users/zhengce/projects/techjam_shopping_copilot/techjam-shopping-copilot/.conda/bin/python \
  -m pytest -q -p no:cacheprovider tests/test_hybrid_question_comparison.py
```

Expected: experiment module missing.

- [ ] **Step 4: Implement the four fixed configurations**

Use pool size 300 and the fixed weights from the spec. Only these gates vary:

```python
GATES = {
    "hybrid_conservative": (0.70, 0.30, 0.35, 0.10, 0.35),
    "hybrid_balanced":     (0.60, 0.40, 0.25, 0.05, 0.25),
    "hybrid_permissive":   (0.50, 0.50, 0.20, 0.00, 0.15),
}
```

The tuple order is minimum coverage, maximum missing rate, minimum expected shrink, minimum Resolve@10, and minimum gain. Legacy uses an empty decision overlay except for forced offline/test controls.

- [ ] **Step 5: Implement one-build resources and hard deadline**

Read the public dataset once. Build one `HybridRetriever`, derive evaluator `catalog_ids/categories/products` from its read-only product snapshot, and build one `DialogueCatalogResources(..., include_attribute_cache=True)`. Evaluate four independently constructed Agents sequentially through the unmodified official `evaluate()`.

Use a process-level `SIGALRM`/`setitimer` budget on supported Unix platforms, restore the previous signal handler in `finally`, and also check `time.monotonic()` between configurations. On timeout, discard the in-progress configuration, atomically write completed results and timing with `status="time_budget_exceeded"`, and return a nonzero CLI exit code. On unsupported platforms, require the caller to provide an external 1,200-second watchdog and record that enforcement mode in the report.

- [ ] **Step 6: Implement aggregate comparison reporting**

Persist:

- schema/version, seed, selected sample IDs as hashes rather than raw user text;
- exact stratum counts and configuration overlays;
- official overall/scenario metrics and token totals;
- initialization, per-configuration, and total elapsed seconds;
- Hybrid reason counts, replacement count/rate, selected-attribute counts, and decision latency p50/p95;
- `status="complete"` only when all four configurations finish.

Do not name a promotion winner. Report only pairwise metric deltas versus Legacy and whether the predeclared screening condition holds: HR@10 non-regression plus an improvement in TechnicalScore, MRR, or MTTC.

- [ ] **Step 7: Document and verify the runner**

Add the exact command, offline controls, 20-session limitation, no-promotion warning, report fields, and timeout behavior to `README.md`. Run:

```bash
/Users/zhengce/projects/techjam_shopping_copilot/techjam-shopping-copilot/.conda/bin/python \
  -m pytest -q -p no:cacheprovider tests/test_hybrid_question_comparison.py
/Users/zhengce/projects/techjam_shopping_copilot/techjam-shopping-copilot/.conda/bin/python \
  -m ruff check --no-cache experiments/hybrid_question_comparison.py \
  tests/test_hybrid_question_comparison.py
```

Expected: all pass.

- [ ] **Step 8: Commit Task 4**

```bash
git add experiments/hybrid_question_comparison.py tests/test_hybrid_question_comparison.py README.md
git commit -m "test: compare legacy and hybrid question policies"
```

- [ ] **Step 9: Run the real bounded comparison**

```bash
PYTHONDONTWRITEBYTECODE=1 \
/Users/zhengce/projects/techjam_shopping_copilot/techjam-shopping-copilot/.conda/bin/python \
  -m experiments.hybrid_question_comparison \
  --catalog /Users/zhengce/projects/participate_kit/catalog.jsonl \
  --dataset /Users/zhengce/projects/participate_kit/public_set.jsonl \
  --output /private/tmp/legacy-hybrid-comparison.json \
  --seed 20260830 \
  --time-budget-seconds 1200
```

Expected: exit `0` and `status="complete"` within 1,200 seconds, or a nonzero timeout exit with a valid partial report and no winner claim.

### Task 5: Final compatibility verification

**Files:**
- Modify only if a test exposes an in-scope defect.

**Interfaces:**
- Consumes: all prior task outputs.
- Produces: verified default-off Legacy compatibility and a concise experiment result.

- [ ] **Step 1: Run the full lightweight suite**

```bash
PYTHONDONTWRITEBYTECODE=1 \
/Users/zhengce/projects/techjam_shopping_copilot/techjam-shopping-copilot/.conda/bin/python \
  -m pytest -q -p no:cacheprovider
```

Expected: all non-live tests pass; existing explicitly skipped live tests remain skipped.

- [ ] **Step 2: Run changed-file Ruff and whitespace validation**

```bash
/Users/zhengce/projects/techjam_shopping_copilot/techjam-shopping-copilot/.conda/bin/python \
  -m ruff check --no-cache config agent experiments tests
git diff --check
```

Expected: both clean.

- [ ] **Step 3: Verify default-off behavior against Legacy**

Run the existing deterministic disabled-policy comparison or add a temporary non-committed probe that evaluates the same selected 20 sessions with Hybrid absent versus `enabled=false`. Require identical per-session hit/rank/turn values and identical `ask_attribute` sequences where traces are available.

- [ ] **Step 4: Record the outcome without promoting defaults**

Summarize elapsed time, completed rollout count, shared-cache build count, Legacy metrics, each Hybrid delta, trigger rate, and timeout status. Leave `config/default.json` with Hybrid disabled regardless of the 20-session result.
