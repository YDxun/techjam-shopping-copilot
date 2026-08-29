# Catalog-Aware Question Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select clarification questions from the live retrieval candidate distribution, including composite `other`, phase-aware finish value, and explicit-only termination, without changing retrieval or final Top10 ranking behavior.

**Architecture:** Precompute deterministic per-product attribute profiles, compute candidate-conditioned value-of-information from the single existing retrieval pool, and route `QuestionPolicy` between unchanged legacy behavior and the new catalog-dynamic behavior. Refactor the dialogue pipeline into interpretation and post-retrieval decision phases so the same candidate list feeds question analysis and reranking.

**Tech Stack:** Python 3.11+, frozen dataclasses, SQLite-backed existing retriever, JSON configuration, `unittest`/pytest, no new runtime dependency.

**Spec:** `docs/superpowers/specs/2026-08-29-catalog-aware-question-policy-design.md`

## Global Constraints

- Complete `docs/superpowers/plans/2026-08-29-transition-guard.md` first; this plan consumes `GuardDecision` in pending turns.
- Use TDD and commit each independently reviewable task.
- Retrieve once per turn; the same candidate list must feed dynamic question signals and the existing reranker.
- Do not alter HybridRetriever scoring, Reranker scoring, the evaluator or the four-field response schema.
- Defaults preserve legacy behavior until the catalog and public-set calibration plan promotes a configuration.
- Attribute extraction is deterministic and local; no LLM or local ML model is added.
- Missing metadata never means that a product lacks an attribute.
- Every dynamic failure falls back to existing static catalog signals and legacy-safe response behavior.

---

### Task 1: Add dynamic-policy configuration and immutable signal models

**Files:**
- Modify: `config/models.py`
- Modify: `config/default.json`
- Modify: `config/loader.py`
- Modify: `agent/dialogue/models.py`
- Test: `tests/test_dialogue_config.py`
- Test: `tests/test_config_loader.py`

**Interfaces:**
- Consumes: current `DecisionConfig` and `QuestionDecision`.
- Produces: `CandidateQuestionValueConfig`, `CandidateQuestionWeights`, `FinishStrategyConfig`, `FinishWeights`, `question_termination_mode`; signal dataclasses used by later tasks.

- [ ] **Step 1: Write failing configuration tests**

Assert legacy-preserving defaults and supported values:

```python
def test_dynamic_question_defaults_preserve_legacy(self) -> None:
    decision = EnvConfig.from_env(environ={}).decision
    self.assertFalse(decision.candidate_question_value.enabled)
    self.assertEqual(decision.question_termination_mode, "legacy")
    self.assertFalse(decision.finish_strategy.enabled)
    self.assertEqual(decision.candidate_question_value.pool_size, 300)

def test_explicit_only_and_pool_size_environment_overrides(self) -> None:
    decision = EnvConfig.from_env(
        environ={
            "SHOPPING_DECISION__QUESTION_TERMINATION_MODE": "explicit_only",
            "SHOPPING_DECISION__CANDIDATE_QUESTION_VALUE__ENABLED": "1",
            "SHOPPING_DECISION__CANDIDATE_QUESTION_VALUE__POOL_SIZE": "500",
        }
    ).decision
    self.assertEqual(decision.question_termination_mode, "explicit_only")
    self.assertTrue(decision.candidate_question_value.enabled)
    self.assertEqual(decision.candidate_question_value.pool_size, 500)

def test_utility_termination_mode_is_rejected(self) -> None:
    with self.assertRaisesRegex(ConfigError, "question_termination_mode"):
        load_config(overrides={"decision": {"question_termination_mode": "utility"}}, environ={})
```

- [ ] **Step 2: Run configuration tests and verify RED**

```bash
.conda/bin/python -m pytest -q -p no:cacheprovider tests/test_dialogue_config.py tests/test_config_loader.py
```

Expected: missing dynamic configuration fields and environment mappings.

- [ ] **Step 3: Add configuration dataclasses and initial search centers**

Define immutable models:

```python
@dataclass(frozen=True)
class CandidateQuestionWeights:
    expected_shrink: float = 0.30
    coverage: float = 0.15
    complementarity: float = 0.15
    answer_probability: float = 0.15
    missing_penalty: float = 0.20
    redundancy_penalty: float = 0.20
    repeat_penalty: float = 0.40
    no_preference_penalty: float = 0.60
    turn_cost: float = 0.15

@dataclass(frozen=True)
class CandidateQuestionValueConfig:
    enabled: bool = False
    pool_size: int = 300
    prior_alpha: float = 0.25
    prior_temperature: float = 1.0
    other_answer_probability: float = 0.75
    other_vagueness_penalty: float = 0.10
    weights: CandidateQuestionWeights = field(default_factory=CandidateQuestionWeights)

@dataclass(frozen=True)
class FinishWeights:
    resolve_at_10: float = 0.50
    resolve_at_3: float = 0.20
    resolve_at_1: float = 0.10
    terminal_progress: float = 0.30
    p90_remaining_penalty: float = 0.20

@dataclass(frozen=True)
class FinishStrategyConfig:
    enabled: bool = False
    candidate_threshold: int = 100
    remaining_question_threshold: int = 2
    lookahead_depth: int = 1
    minimum_finish_gain: float = 0.0
    weights: FinishWeights = field(default_factory=FinishWeights)
```

Add `question_termination_mode: str = "legacy"` to DecisionConfig. Treat these numbers as reproducible search centers, not promoted competition values. Validate positive counts and temperature, unit-interval probabilities and non-negative weights; restrict lookahead to 1 or 2 and termination mode to `legacy|explicit_only`.

Add immutable signal contracts:

```python
@dataclass(frozen=True)
class CandidateAttributeSignal:
    attribute: str
    coverage: float
    expected_remaining: float
    expected_shrink: float
    resolve_at_10: float
    resolve_at_3: float
    resolve_at_1: float
    p90_remaining: float
    worst_case_remaining: int
    missing_rate: float
    extraction_confidence: float
    two_step_finish_gain: float = 0.0

@dataclass(frozen=True)
class CandidateQuestionSignals:
    candidate_count: int
    by_attribute: Mapping[str, CandidateAttributeSignal]
    target_probabilities: Mapping[str, float]
    best_other_pair: tuple[str, str] | None = None
    other_signal: CandidateAttributeSignal | None = None
    previous_candidate_count: int | None = None
    source: str = "dynamic"
```

- [ ] **Step 4: Run configuration tests and verify GREEN**

Run the Task 1 test command. Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add config/models.py config/default.json config/loader.py agent/dialogue/models.py tests/test_dialogue_config.py tests/test_config_loader.py
git commit -m "feat: configure catalog-aware question decisions"
```

### Task 2: Build deterministic product attribute profiles

**Files:**
- Create: `agent/dialogue/catalog_attributes.py`
- Create: `tests/test_catalog_attributes.py`
- Reuse: `data/analysis/vocab.json`

**Interfaces:**
- Consumes: iterable catalog product dictionaries and existing normalized vocabulary.
- Produces: `AttributeProfile`, `CatalogAttributeCache.from_products(products)`, and `CatalogAttributeCache.for_asin(asin)`.

- [ ] **Step 1: Write failing extraction tests**

Use hand-authored products and expected canonical sets:

```python
def test_structured_fields_beat_free_text_and_synonyms_collapse(self) -> None:
    product = {
        "parent_asin": "A",
        "title": "Soft cotton blend running top",
        "features": ["95% cotton with 5% spandex"],
        "details": {"Material": "100% Cotton", "Color": "Jet Black"},
        "description": ["polyester-like appearance"],
        "categories": ["Women", "Tops"],
        "store": "Example Brand",
        "price": 29.99,
    }
    profile = RuleVocabularyExtractor(vocabulary()).extract(product)
    self.assertEqual(profile.values["material"], frozenset({"cotton"}))
    self.assertEqual(profile.values["color"], frozenset({"black"}))
    self.assertGreater(profile.confidence["material"], profile.confidence["use_case"])

def test_missing_attribute_stays_missing(self) -> None:
    profile = RuleVocabularyExtractor(vocabulary()).extract(
        {"parent_asin": "B", "title": "Generic item", "features": [], "details": {}}
    )
    self.assertEqual(profile.values["material"], frozenset())
```

Also cover apparel size versus shoe size, category-relative price quartiles, generic brand filtering, controlled feature vocabulary and cache lookup.

- [ ] **Step 2: Run the focused test and verify RED**

```bash
.conda/bin/python -m pytest -q -p no:cacheprovider tests/test_catalog_attributes.py
```

Expected: import failure because the extractor and cache do not exist.

- [ ] **Step 3: Implement focused immutable profiles**

Define:

```python
class CatalogAttributeExtractor(Protocol):
    @property
    def vocabulary_version(self) -> str:
        raise NotImplementedError

    def extract(self, product: dict[str, object]) -> AttributeProfile:
        raise NotImplementedError

@dataclass(frozen=True)
class AttributeProfile:
    parent_asin: str
    values: Mapping[str, frozenset[str]]
    confidence: Mapping[str, float]
    sources: Mapping[str, tuple[str, ...]]

class CatalogAttributeCache:
    @classmethod
    def from_products(
        cls,
        products: Iterable[dict],
        extractor: CatalogAttributeExtractor,
    ) -> "CatalogAttributeCache":
        profiles: dict[str, AttributeProfile] = {}
        for product in products:
            profile = extractor.extract(product)
            profiles[profile.parent_asin] = profile
        return cls(
            profiles=profiles,
            vocabulary_version=extractor.vocabulary_version,
            catalog_fingerprint=catalog_fingerprint(profiles),
        )

    def for_asin(self, asin: str) -> AttributeProfile | None:
        return self._profiles.get(asin)
```

The `Protocol` is the stable extension point for a future local extractor; this task implements only `RuleVocabularyExtractor`. Implement `catalog_fingerprint(profiles)` as SHA-256 over newline-joined, ASIN-sorted canonical JSON containing only normalized values and confidence fields (`sort_keys=True`, compact separators). Load vocabulary once. Normalize explicit details first, then controlled title/features matches, then low-confidence description matches. Build category price quartiles in a first pass and assign `budget_low|budget_mid|budget_high` only when numeric price is available. Store the extractor's deterministic vocabulary version and the catalog content fingerprint with the cache.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 2 command. Expected: all extraction and cache tests pass.

- [ ] **Step 5: Run the existing catalog-signal regression**

```bash
.conda/bin/python -m pytest -q -p no:cacheprovider tests/test_question_policy.py
```

Expected: existing static signal tests remain green.

- [ ] **Step 6: Commit Task 2**

```bash
git add agent/dialogue/catalog_attributes.py tests/test_catalog_attributes.py
git commit -m "feat: cache normalized catalog attributes"
```

### Task 3: Compute dynamic candidate value-of-information

**Files:**
- Create: `agent/dialogue/candidate_signals.py`
- Create: `tests/test_candidate_signals.py`

**Interfaces:**
- Consumes: `CandidateQuestionValueConfig`, `FinishStrategyConfig`, candidates shaped as `{parent_asin, rrf}`, and `CatalogAttributeCache`.
- Produces: `CandidateSignalCalculator.calculate(candidates) -> CandidateQuestionSignals`, including `best_other_pair`, `other_signal`, and bounded per-attribute `two_step_finish_gain`.

- [ ] **Step 1: Write failing hand-calculated signal tests**

Construct four products whose materials split 2/2 and whose color is constant. Assert exact values under uniform prior:

```python
signals = calculator(alpha=0.0).calculate(candidates)
material = signals.by_attribute["material"]
self.assertEqual(material.expected_remaining, 2.0)
self.assertEqual(material.expected_shrink, 0.5)
self.assertEqual(material.resolve_at_10, 1.0)
self.assertEqual(signals.by_attribute["color"].expected_shrink, 0.0)
```

Add tests for missing values remaining in every compatible set, RRF/uniform mixing, all-equal score fallback, temperature validation, weighted P90, multivalue overlap, joint `other` gain, and two-step gain being zero at depth 1.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
.conda/bin/python -m pytest -q -p no:cacheprovider tests/test_candidate_signals.py
```

Expected: import failure because `CandidateSignalCalculator` does not exist.

- [ ] **Step 3: Implement target probabilities and match sets**

Normalize RRF safely:

```python
uniform = 1.0 / len(candidates)
softmax = stable_softmax([score / config.prior_temperature for score in scores])
probabilities = {
    asin: (1.0 - config.prior_alpha) * uniform + config.prior_alpha * softmax[index]
    for index, asin in enumerate(asins)
}
```

If scores are absent, non-finite or all equal, use uniform probabilities. For each target profile and attribute, the remaining set contains candidates sharing at least one canonical value plus every candidate missing that attribute. A missing target attribute leaves the whole pool unresolved.

Compute weighted expected remaining, shrink, Resolve@K, weighted P90 and worst case. Clamp probabilities and ratios only at the public result boundary; keep internal floats unrounded.

- [ ] **Step 4: Implement joint signals for `other` and bounded lookahead**

For every pair of unresolved concrete attributes, calculate the remaining set using both answers. Store the best pair and its signal on `CandidateQuestionSignals`; do not include `category` or `other` in the pair. Use deterministic attribute order for ties.

When `lookahead_depth == 2`, enumerate each first attribute's value branches, select the best legal second concrete attribute in each branch, weight its finish gain by that branch's target probability, and subtract one configured turn cost. Store the resulting non-negative value in `CandidateAttributeSignal.two_step_finish_gain`. Do not recurse and do not run this calculation at depth 1.

- [ ] **Step 5: Run focused and static regression tests**

```bash
.conda/bin/python -m pytest -q -p no:cacheprovider tests/test_candidate_signals.py tests/test_question_policy.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add agent/dialogue/candidate_signals.py tests/test_candidate_signals.py
git commit -m "feat: calculate candidate-conditioned question value"
```

### Task 4: Add catalog-dynamic selection, finish value and explicit-only termination

**Files:**
- Modify: `agent/dialogue/question_policy.py`
- Modify: `tests/test_question_policy.py`
- Create: `tests/test_finish_strategy.py`

**Interfaces:**
- Consumes: static `CatalogQuestionSignals`, optional `CandidateQuestionSignals`, current state and recognition.
- Produces: `QuestionPolicy.decide(state, recognition, static_signals, candidate_signals=None)`; legacy output unchanged when dynamic is disabled.

- [ ] **Step 1: Write failing legacy-equivalence and explicit-only tests**

```python
def test_dynamic_disabled_matches_legacy_decision(self) -> None:
    policy = QuestionPolicy(config(candidate_enabled=False, termination="legacy"))
    self.assertEqual(
        policy.decide(state, parsed(), static_signals),
        legacy_expected_decision,
    )

def test_explicit_only_turn_nine_asks_and_turn_ten_does_not(self) -> None:
    policy = QuestionPolicy(config(candidate_enabled=True, termination="explicit_only"))
    self.assertTrue(policy.decide(state_at(9), parsed(), static, dynamic).should_ask)
    self.assertEqual(
        policy.decide(state_at(10), parsed(), static, dynamic).reason_code,
        "final_turn_no_followup",
    )

def test_no_preference_other_moves_to_specific_attribute(self) -> None:
    decision = policy.decide(
        state_with_no_preference("other"), parsed_no_preference_other(), static, dynamic
    )
    self.assertTrue(decision.should_ask)
    self.assertEqual(decision.ask_attribute, "material")
```

Also test `no_more_preferences`, max_questions as a soft cost, all-nonpositive fallback, all-attributes-exhausted, and exact legacy snapshots.

- [ ] **Step 2: Write failing exploration/finish and `other` tests**

Use explicit CandidateAttributeSignal fixtures. Assert exploration chooses expected shrink, finish chooses Resolve@10, a concrete attribute beats vague `other` late, and precomputed two-step gain contributes only under the finish gate.

- [ ] **Step 3: Run policy tests and verify RED**

```bash
.conda/bin/python -m pytest -q -p no:cacheprovider tests/test_question_policy.py tests/test_finish_strategy.py
```

Expected: signature and behavior failures because dynamic policy does not exist.

- [ ] **Step 4: Preserve legacy code behind an explicit branch**

Extract the current method body into
`_decide_legacy(self, state: DialogueState, recognition: RecognitionResult, signals: CatalogQuestionSignals) -> QuestionDecision`
without semantic edits. Route to it when dynamic signals are missing, candidate value is disabled, or termination mode is `legacy`.

- [ ] **Step 5: Implement catalog-dynamic utility**

Compute:

```python
utility = (
    (1.0 - finish_pressure) * exploration_gain
    + finish_pressure * finish_gain
    + state_gain
    - repeat_penalty
    - no_preference_penalty
    - turn_cost
)
```

Calculate finish pressure from candidate-to-Top10 distance, candidate shrink progress stored in state/turn context, remaining question budget and turn. Keep the calculation pure and expose component dictionaries for diagnostics. Calculate `other` from `best_other_pair` and `other_signal`, applying the configured answer-probability and vagueness terms.

For explicit-only mode, only `no_more_preferences`, turn 10 or exhausted attributes suppress a question. On turns 1-9, select the highest positive legal action; if none is positive, select the highest-scoring concrete attribute not marked no-preference, then `other`, then return `all_attributes_exhausted`.

- [ ] **Step 6: Gate the precomputed two-step finish value**

When lookahead depth is 2 and the finish gate is active, include each action's `two_step_finish_gain`; otherwise ignore it. The policy must not inspect catalog rows or recompute branches. Use deterministic attribute order for ties.

- [ ] **Step 7: Run policy tests and verify GREEN**

Run the Task 4 test command. Expected: all selected tests pass.

- [ ] **Step 8: Commit Task 4**

```bash
git add agent/dialogue/question_policy.py tests/test_question_policy.py tests/test_finish_strategy.py
git commit -m "feat: select questions from candidate distributions"
```

### Task 5: Refactor the turn pipeline around one retrieval

**Files:**
- Modify: `agent/dialogue/pipeline.py`
- Modify: `agent/main_agent.py`
- Modify: `tests/test_dialogue_flow.py`
- Modify: `tests/test_product_history.py`

**Interfaces:**
- Consumes: guard from the prerequisite plan, `CandidateSignalCalculator`, existing retriever `search()` and `product()`.
- Produces: `PendingDialogueTurn`, `DialogueUnderstandingPipeline.interpret_turn()`, `DialogueUnderstandingPipeline.decide_question()`.

- [ ] **Step 1: Write failing two-phase flow tests**

Assert retrieval is called exactly once, the same candidate object reaches the calculator and reranker, state is not double-applied, and the official response remains unchanged:

```python
response = agent.respond("s", "I'm looking for shoes.", 1, 3)
self.assertEqual(retriever.search_calls, 1)
self.assertIs(signal_calculator.last_candidates, reranker.last_candidates)
self.assertEqual(set(response), {"message", "ask_attribute", "recommendations", "usage"})
self.assertEqual(agent.dialogue.session("s").dialogue.turn, 1)
```

Add failure-path tests: candidate calculator exception falls back to static policy; empty candidates still return a valid response; dynamic disabled exactly matches prior behavior.

- [ ] **Step 2: Run flow tests and verify RED**

```bash
.conda/bin/python -m pytest -q -p no:cacheprovider tests/test_dialogue_flow.py tests/test_product_history.py
```

Expected: missing two-phase methods and candidate signal integration.

- [ ] **Step 3: Add the pending-turn contract**

Define this dataclass in `agent/dialogue/pipeline.py`, where both DialogueState and ProductHistory are already available without a circular import:

```python
@dataclass(frozen=True)
class PendingDialogueTurn:
    session_id: str
    turn: int
    state: DialogueState
    recognition: RecognitionResult
    guard_decision: GuardDecision
    recommendation_context: RecommendationContext
    products: ProductHistory
    prompt_tokens: int
    completion_tokens: int
```

Extend `SessionState` with `candidate_counts: tuple[int, ...] = ()`. `interpret_turn` performs recognition, guard, feedback and state reduction exactly once but does not record a question. `decide_question` copies the most recent count into `CandidateQuestionSignals.previous_candidate_count`, records the chosen attribute, appends the current count, commits the final SessionState and returns DialogueTurnResult.

- [ ] **Step 4: Reorder Agent orchestration**

Implement the exact sequence:

```python
pending = self.dialogue.interpret_turn(session_id, user_message, turn)
context = pending.recommendation_context
route = self.router.route(context, mode=context.retrieval_mode)
candidates = self.retriever.search(
    route,
    top_k=self.env.decision.candidate_question_value.pool_size,
    mode=context.retrieval_mode,
)
candidate_signals = self.dialogue.candidate_signal_calculator.calculate(candidates)
turn_result = self.dialogue.decide_question(pending, candidate_signals)
ranked = self.reranker.rerank(
    self.retriever,
    candidates,
    turn_result.recommendation_context,
    route,
    top_k=top_k,
    mode=context.retrieval_mode,
    use_reranker_model=self.decisions.use_reranker_model,
    use_llm_rerank=self.decisions.use_llm_rerank,
)
```

If dynamic is disabled, continue requesting `RETRIEVAL_POOL_SIZE=300`. If configured pool size exceeds 300, use that one larger retrieval for both analysis and reranking; never perform a second search.

- [ ] **Step 5: Make test doubles satisfy the real retriever contract**

Add `product(asin)` to StaticRetriever and capture object identity in StaticReranker. Avoid production-only `hasattr` branches.

- [ ] **Step 6: Run flow and policy tests and verify GREEN**

```bash
.conda/bin/python -m pytest -q -p no:cacheprovider tests/test_dialogue_flow.py tests/test_product_history.py tests/test_question_policy.py tests/test_candidate_signals.py
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit Task 5**

```bash
git add agent/dialogue/pipeline.py agent/main_agent.py tests/test_dialogue_flow.py tests/test_product_history.py
git commit -m "refactor: decide questions after candidate retrieval"
```

### Task 6: Verify compatibility and produce catalog-scale performance evidence

**Files:**
- Modify: `README.md`
- Review only: all files changed in Tasks 1-5.

**Interfaces:**
- Consumes: complete catalog-aware policy behind disabled defaults.
- Produces: independently mergeable code with reproducible commands and measured candidate-signal latency.

- [ ] **Step 1: Run the full fast suite**

```bash
.conda/bin/python -m pytest -q -p no:cacheprovider
.conda/bin/python -m ruff check --no-cache agent/dialogue agent/main_agent.py config tests/test_catalog_attributes.py tests/test_candidate_signals.py tests/test_finish_strategy.py tests/test_dialogue_flow.py
git diff --check
```

Expected: tests and targeted Ruff pass; diff check is clean.

- [ ] **Step 2: Run disabled legacy evaluation**

```bash
LLM_PROVIDER=none SKIP_DATA_VERIFY=1 .conda/bin/python run_local_eval.py --catalog /Users/zhengce/projects/participate_kit/catalog.jsonl --dataset data/public_set.jsonl --output /private/tmp/catalog-policy-legacy.json
```

Expected: official response contract remains valid and legacy aggregate behavior matches the prior baseline.

- [ ] **Step 3: Run a non-scoring catalog timing probe**

Build the attribute cache once, sample candidate pools of 300, 500 and 1000 from the real retriever, and measure calculator p50/p95 time using `time.perf_counter`. Save only aggregate timing and memory estimates; do not select competition weights in this task.

- [ ] **Step 4: Document feature switches and fallback semantics**

Add README examples for dynamic disabled, explicit-only experimental mode, pool size overrides and legacy rollback. State that numeric defaults are search centers pending the evaluation-tuning plan.

- [ ] **Step 5: Request code review and address blocking findings**

Review one-search orchestration, state commit order, disabled-path equivalence, multivalue math, missing-value safety, deterministic tie-breaking and response compatibility.

- [ ] **Step 6: Commit documentation or review fixes**

```bash
git add README.md agent/dialogue agent/main_agent.py config tests
git commit -m "docs: explain catalog-aware question policy rollout"
```
