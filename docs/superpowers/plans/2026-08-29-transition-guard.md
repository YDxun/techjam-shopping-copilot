# Transition Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configurable semantic gate between intent recognition and state mutation so destructive operations require stronger evidence while the disabled path remains behaviorally identical.

**Architecture:** Introduce immutable guard configuration and result models, implement a pure `TransitionGuard.evaluate()` decision, then integrate it before product feedback and `StateReducer`. The guard is disabled by default; its result and aggregate statistics remain local diagnostics and never alter the official response schema.

**Tech Stack:** Python 3.11+, frozen dataclasses, enums, `unittest`/pytest, existing unified JSON/environment configuration.

**Spec:** `docs/superpowers/specs/2026-08-29-catalog-aware-question-policy-design.md`

## Global Constraints

- Use TDD for every behavior change: failing test, observed failure, minimal implementation, passing test.
- `StateReducer` remains the only component that creates a changed `DialogueState`.
- `transition_guard.enabled=false` must preserve the current recognition, feedback, state and response behavior.
- LLM JSON/evidence validation remains active regardless of the guard switch.
- Do not change retrieval, reranking, the official evaluator or the four-field turn response.
- All configuration is immutable, validated centrally, and key switches support environment overrides.
- No API keys, raw LLM responses or unhashed session identifiers enter diagnostics.

---

### Task 1: Add immutable guard configuration and result contracts

**Files:**
- Modify: `config/models.py`
- Modify: `config/default.json`
- Modify: `config/loader.py`
- Modify: `agent/dialogue/models.py`
- Test: `tests/test_dialogue_config.py`
- Test: `tests/test_config_loader.py`

**Interfaces:**
- Consumes: existing `DialogueUnderstandingConfig`, `RecognitionResult` and `DialogueState`.
- Produces: `TransitionGuardConfig`, `GuardAction`, `GuardDecision`; environment override `SHOPPING_DIALOGUE__TRANSITION_GUARD__ENABLED`.

- [ ] **Step 1: Write failing configuration and model tests**

Add tests asserting the default switch is off, the environment switch works, invalid confidence thresholds fail, and unsupported actions fail:

```python
def test_transition_guard_defaults_to_disabled(self) -> None:
    config = EnvConfig.from_env(environ={})
    guard = config.dialogue_understanding.transition_guard
    self.assertFalse(guard.enabled)
    self.assertEqual(guard.low_confidence_add_action, "soften")
    self.assertEqual(guard.destructive_failure_action, "clarify")

def test_transition_guard_environment_switch(self) -> None:
    config = EnvConfig.from_env(
        environ={"SHOPPING_DIALOGUE__TRANSITION_GUARD__ENABLED": "1"}
    )
    self.assertTrue(config.dialogue_understanding.transition_guard.enabled)

def test_transition_guard_rejects_invalid_values(self) -> None:
    with self.assertRaisesRegex(ConfigError, "replace_min_confidence"):
        load_config(
            overrides={
                "dialogue_understanding": {
                    "transition_guard": {"replace_min_confidence": 1.1}
                }
            },
            environ={},
        )
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
.conda/bin/python -m pytest -q -p no:cacheprovider tests/test_dialogue_config.py tests/test_config_loader.py
```

Expected: failures because `transition_guard`, its dataclass and environment override do not exist.

- [ ] **Step 3: Add the contracts and defaults**

Add frozen configuration and guard models:

```python
@dataclass(frozen=True)
class TransitionGuardConfig:
    enabled: bool = False
    add_min_confidence: float = 0.65
    replace_min_confidence: float = 0.90
    remove_min_confidence: float = 0.90
    reject_products_min_confidence: float = 0.90
    no_preference_min_confidence: float = 0.85
    no_more_preferences_min_confidence: float = 0.95
    low_confidence_add_action: str = "soften"
    destructive_failure_action: str = "clarify"

class GuardAction(str, Enum):
    APPLY = "apply"
    SOFTEN = "soften"
    CLARIFY = "clarify"
    REJECT = "reject"

@dataclass(frozen=True)
class GuardDecision:
    action: GuardAction
    recognition: RecognitionResult
    reason_code: str
    clarify_attribute: str | None = None
```

Nest `TransitionGuardConfig` inside `DialogueUnderstandingConfig`, add matching JSON defaults, parse it in `_build_and_validate`, validate every confidence with `_unit_interval`, and restrict actions to `soften` and `clarify` respectively. Add the boolean environment mapping through `_set_nested`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 1 command. Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add config/models.py config/default.json config/loader.py agent/dialogue/models.py tests/test_dialogue_config.py tests/test_config_loader.py
git commit -m "feat: add transition guard configuration"
```

### Task 2: Implement the pure transition guard

**Files:**
- Create: `agent/dialogue/transition_guard.py`
- Create: `tests/test_transition_guard.py`

**Interfaces:**
- Consumes: `TransitionGuardConfig`, current `DialogueState`, and `RecognitionResult`.
- Produces: `TransitionGuard.evaluate(state, recognition) -> GuardDecision` and `TransitionGuard.statistics() -> dict[str, object]`.

- [ ] **Step 1: Write the failing guard behavior tests**

Cover disabled passthrough, low-confidence add softening, destructive clarification, absent remove target, grounded explicit rejection, and no-more threshold:

```python
def test_disabled_guard_is_exact_passthrough(self) -> None:
    result = recognition(DialogueAct.REPLACE_CONSTRAINT, confidence=0.2)
    decision = TransitionGuard(TransitionGuardConfig(enabled=False)).evaluate(
        DialogueState(session_id="s", user_profile={}), result
    )
    self.assertEqual(decision.action, GuardAction.APPLY)
    self.assertIs(decision.recognition, result)
    self.assertEqual(decision.reason_code, "guard_disabled")

def test_low_confidence_add_is_softened(self) -> None:
    result = recognition(
        DialogueAct.ADD_CONSTRAINT,
        operation(OperationKind.ADD, "material", "cotton", confidence=0.60),
        confidence=0.60,
    )
    decision = TransitionGuard(TransitionGuardConfig(enabled=True)).evaluate(empty_state(), result)
    self.assertEqual(decision.action, GuardAction.SOFTEN)
    self.assertEqual(
        decision.recognition.constraint_operations[0].strength,
        ConstraintStrength.SOFT,
    )

def test_low_confidence_replace_requests_attribute_clarification(self) -> None:
    result = recognition(
        DialogueAct.REPLACE_CONSTRAINT,
        operation(OperationKind.REPLACE, "material", "cotton", confidence=0.70),
        confidence=0.70,
    )
    decision = TransitionGuard(TransitionGuardConfig(enabled=True)).evaluate(state_with_style(), result)
    self.assertEqual(decision.action, GuardAction.CLARIFY)
    self.assertEqual(decision.clarify_attribute, "material")
    self.assertEqual(decision.reason_code, "replace_confidence_below_threshold")
```

- [ ] **Step 2: Run the focused test and verify RED**

```bash
.conda/bin/python -m pytest -q -p no:cacheprovider tests/test_transition_guard.py
```

Expected: import failure because `TransitionGuard` does not exist.

- [ ] **Step 3: Implement the pure evaluator**

Implement a side-effect-free class. Use `dataclasses.replace` to soften add operations; do not mutate the supplied recognition or state. Check operation-level confidence as well as top-level confidence by using their minimum. A remove is valid only if its normalized key exists; a replace requires explicit evidence and a non-empty replacement; explicit rejected ASINs remain bounded by the recognizer.

```python
class TransitionGuard:
    def __init__(self, config: TransitionGuardConfig) -> None:
        self.config = config
        self._actions: Counter[str] = Counter()
        self._reasons: Counter[str] = Counter()
        self._dialogue_acts: Counter[str] = Counter()
        self._sources: Counter[str] = Counter()

    def evaluate(
        self,
        state: DialogueState,
        recognition: RecognitionResult,
    ) -> GuardDecision:
        if not self.config.enabled:
            return self._decision(GuardAction.APPLY, recognition, "guard_disabled")

        act = recognition.dialogue_act
        operation_confidence = min(
            (item.confidence for item in recognition.constraint_operations),
            default=recognition.confidence,
        )
        confidence = min(recognition.confidence, operation_confidence)

        if act == DialogueAct.ADD_CONSTRAINT and confidence < self.config.add_min_confidence:
            softened = replace(
                recognition,
                constraint_operations=tuple(
                    replace(item, strength=ConstraintStrength.SOFT)
                    for item in recognition.constraint_operations
                ),
            )
            return self._decision(
                GuardAction.SOFTEN,
                softened,
                "add_confidence_below_threshold",
            )

        destructive = {
            DialogueAct.REPLACE_CONSTRAINT: self.config.replace_min_confidence,
            DialogueAct.REMOVE_CONSTRAINT: self.config.remove_min_confidence,
            DialogueAct.REJECT_PRODUCTS: self.config.reject_products_min_confidence,
        }
        if act in destructive and confidence < destructive[act]:
            return self._destructive_failure(recognition, f"{act.value}_confidence_below_threshold")

        if act == DialogueAct.REPLACE_CONSTRAINT:
            valid = bool(recognition.constraint_operations) and all(
                item.operation == OperationKind.REPLACE
                and item.value.strip()
                and item.evidence.strip()
                for item in recognition.constraint_operations
            )
            if not valid:
                return self._destructive_failure(recognition, "replace_missing_explicit_evidence")

        if act == DialogueAct.REMOVE_CONSTRAINT:
            active_keys = {
                (item.attribute, su.constraint_key(item.value))
                for item in state.active_constraints
            }
            targets = {
                (item.attribute, su.constraint_key(item.value))
                for item in recognition.constraint_operations
                if item.operation == OperationKind.REMOVE
            }
            if not targets or not targets.issubset(active_keys):
                return self._destructive_failure(recognition, "remove_target_absent")

        if act == DialogueAct.REJECT_PRODUCTS and not recognition.explicit_rejected_asins:
            return self._decision(GuardAction.REJECT, recognition, "reject_products_not_grounded")

        if act == DialogueAct.NO_PREFERENCE:
            explicit_attributes = {
                item.attribute
                for item in recognition.constraint_operations
                if item.operation == OperationKind.REMOVE
            }
            if (
                recognition.confidence < self.config.no_preference_min_confidence
                or not explicit_attributes
            ):
                return self._decision(
                    GuardAction.CLARIFY,
                    recognition,
                    "no_preference_attribute_unclear",
                    min(explicit_attributes) if explicit_attributes else "other",
                )

        if (
            act == DialogueAct.NO_MORE_PREFERENCES
            and recognition.confidence < self.config.no_more_preferences_min_confidence
        ):
            return self._decision(
                GuardAction.CLARIFY,
                recognition,
                "no_more_preferences_confidence_below_threshold",
                "other",
            )

        return self._decision(GuardAction.APPLY, recognition, "guard_passed")

    def _destructive_failure(
        self,
        recognition: RecognitionResult,
        reason: str,
    ) -> GuardDecision:
        action = GuardAction(self.config.destructive_failure_action)
        attribute = (
            recognition.constraint_operations[0].attribute
            if recognition.constraint_operations
            else "other"
        )
        return self._decision(action, recognition, reason, attribute)

    def _decision(
        self,
        action: GuardAction,
        recognition: RecognitionResult,
        reason: str,
        clarify_attribute: str | None = None,
    ) -> GuardDecision:
        self._actions[action.value] += 1
        self._reasons[reason] += 1
        self._dialogue_acts[recognition.dialogue_act.value] += 1
        self._sources[recognition.source.value] += 1
        return GuardDecision(action, recognition, reason, clarify_attribute)

    def statistics(self) -> dict[str, object]:
        return {
            "enabled": self.config.enabled,
            "total": sum(self._actions.values()),
            "actions": dict(sorted(self._actions.items())),
            "reasons": dict(sorted(self._reasons.items())),
            "dialogue_acts": dict(sorted(self._dialogue_acts.items())),
            "recognition_sources": dict(sorted(self._sources.items())),
        }
```

Do not store raw evidence or user messages in statistics.

- [ ] **Step 4: Run the guard tests and verify GREEN**

Run the Task 2 command. Expected: all guard tests pass.

- [ ] **Step 5: Run type-adjacent regression tests**

```bash
.conda/bin/python -m pytest -q -p no:cacheprovider tests/test_recognizers.py tests/test_state_reducer.py tests/test_product_history.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add agent/dialogue/transition_guard.py tests/test_transition_guard.py
git commit -m "feat: gate risky dialogue state transitions"
```

### Task 3: Integrate guard decisions before feedback and state reduction

**Files:**
- Modify: `agent/dialogue/pipeline.py`
- Modify: `agent/dialogue/models.py`
- Modify: `agent/main_agent.py`
- Modify: `run_local_eval.py`
- Modify: `tests/test_dialogue_flow.py`
- Modify: `tests/test_llm_startup.py`

**Interfaces:**
- Consumes: `TransitionGuard.evaluate()` and `GuardDecision` from Tasks 1-2.
- Produces: `DialogueTurnResult.guard_decision`, `Agent.transition_guard_statistics()`, local result key `transition_guard_statistics`.

- [ ] **Step 1: Write failing integration tests**

Add tests proving guarded destructive feedback does not mutate state or product history, disabled mode preserves current behavior, and local evaluation serializes only aggregate statistics:

```python
def test_guarded_rejection_does_not_mutate_products_or_dialogue(self) -> None:
    pipeline = build_pipeline(guard_enabled=True, llm_response=low_confidence_rejection("A"))
    pipeline.reset("s", {})
    pipeline.record_shown("s", ["A"], turn=1)
    result = pipeline.process_turn("s", "Reject A", turn=2)
    session = pipeline.session("s")
    self.assertEqual(result.guard_decision.action, GuardAction.CLARIFY)
    self.assertEqual(session.dialogue.active_constraints, ())
    self.assertEqual(session.products.context_lists(1).hard_rejected_asins, ())

def test_disabled_guard_preserves_existing_flow(self) -> None:
    before = legacy_pipeline_result()
    after = guarded_pipeline_result(enabled=False)
    self.assertEqual(after.state, before.state)
    self.assertEqual(after.question_decision, before.question_decision)
```

- [ ] **Step 2: Run focused integration tests and verify RED**

```bash
.conda/bin/python -m pytest -q -p no:cacheprovider tests/test_dialogue_flow.py tests/test_llm_startup.py
```

Expected: failures because the pipeline does not instantiate or expose the guard.

- [ ] **Step 3: Wire the guard into the pipeline**

Instantiate `TransitionGuard(env.dialogue_understanding.transition_guard)`. Evaluate immediately after recognition and before `ProductHistory.apply_feedback` or `StateReducer.reduce`.

For `APPLY` and `SOFTEN`, use `guard_decision.recognition` for feedback and reduction. For `CLARIFY` or `REJECT`, preserve the original state and product history and create a deterministic question decision:

```python
QuestionDecision(
    should_ask=True,
    ask_attribute=guard_decision.clarify_attribute or "other",
    reason_code=guard_decision.reason_code,
    utility_score=0.0,
    alternative_scores={},
)
```

Add `guard_decision` to `DialogueTurnResult`. Expose aggregate statistics from Agent and append them only to local evaluation output.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 3 command. Expected: all selected tests pass.

- [ ] **Step 5: Run the full fast suite and lint changed files**

```bash
.conda/bin/python -m pytest -q -p no:cacheprovider
.conda/bin/python -m ruff check --no-cache agent/dialogue config run_local_eval.py tests/test_transition_guard.py tests/test_dialogue_flow.py tests/test_llm_startup.py
git diff --check
```

Expected: tests pass, targeted Ruff passes, and diff check is clean.

- [ ] **Step 6: Commit Task 3**

```bash
git add agent/dialogue/pipeline.py agent/dialogue/models.py agent/main_agent.py run_local_eval.py tests/test_dialogue_flow.py tests/test_llm_startup.py
git commit -m "feat: integrate transition guard into dialogue flow"
```

### Task 4: Add generalization fixtures and transition-sequence regression gates

**Files:**
- Create: `tests/fixtures/intent/generalization.jsonl`
- Create: `tests/test_intent_generalization.py`
- Create: `tests/test_transition_sequences.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: existing recognizers, `TransitionGuard`, and `StateReducer`.
- Produces: deterministic fixture schema `{id, message, state, expected}` and a documented opt-in live-LLM command.

- [ ] **Step 1: Add a hand-reviewed fixture sample and failing loader assertions**

Use explicit labels rather than assertions derived from production code:

```json
{"id":"replace_material_01","message":"Keep the style, but switch the material to cotton.","state":{"category":"shirts","constraints":[{"attribute":"style","value":"casual","strength":"hard"}]},"expected":{"dialogue_act":"replace_constraint","operation":"replace","attribute":"material","value":"cotton"}}
{"id":"negated_reject_01","message":"I don't mean reject all of them; I only dislike the black one.","state":{"category":"shoes","recently_shown_asins":[]},"expected":{"dialogue_act":"add_constraint","attribute":"color","polarity":"exclude","value":"black"}}
```

The initial checked-in corpus must contain at least one reviewed example for every dialogue act, every operation kind, negation, correction, context reference and noisy spelling. The test reads each row, builds the declared state, and checks the explicitly stored expected fields.

- [ ] **Step 2: Add deterministic state-sequence tests**

Define full expected final states for sequences such as add → replace → remove, reject → intent override, and no-more → new intent. Assert:

```python
self.assertEqual(final.intent_version, 2)
self.assertEqual([(c.attribute, c.value, c.strength.value) for c in final.active_constraints], expected)
self.assertEqual(final.no_preference_attributes, frozenset())
self.assertFalse(final.no_more_preferences)
```

- [ ] **Step 3: Run tests and classify expected RED cases**

```bash
.conda/bin/python -m pytest -q -p no:cacheprovider tests/test_intent_generalization.py tests/test_transition_sequences.py
```

Expected: fixture cases unsupported by the current rule path fail. Record them as intentional generalization gaps; do not weaken expected labels to make tests green.

- [ ] **Step 4: Make only bounded recognizer corrections needed by reviewed fixtures**

Add high-precision rule patterns or LLM prompt examples only when a fixture identifies a generalizable language form. Keep complex cases routed to LLM; do not add sample IDs, product-specific strings or evaluator-only branches. Mark the live-only test class with `@unittest.skipUnless(os.environ.get("RUN_LIVE_LLM") == "1", "live LLM disabled")`; the ordinary offline command must never call an external API.

- [ ] **Step 5: Run offline and opt-in live checks**

```bash
.conda/bin/python -m pytest -q -p no:cacheprovider tests/test_intent_generalization.py tests/test_transition_sequences.py
RUN_LIVE_LLM=1 LLM_PROVIDER=deepseek LLM_INTENT_ENABLE=1 .conda/bin/python -m pytest -q -p no:cacheprovider tests/test_intent_generalization.py
```

Expected: offline tests pass; live tests report schema-valid rate, destructive precision and fallback rate without exact-output assertions.

- [ ] **Step 6: Document the fixture schema and commit**

```bash
git add tests/fixtures/intent/generalization.jsonl tests/test_intent_generalization.py tests/test_transition_sequences.py README.md agent/dialogue/recognizers
git commit -m "test: add intent generalization and transition gates"
```

### Task 5: Final verification and review checkpoint

**Files:**
- Review only: all files changed in Tasks 1-4.

**Interfaces:**
- Consumes: complete transition-guard increment.
- Produces: a verified, independently mergeable feature with the guard disabled by default.

- [ ] **Step 1: Run fresh verification**

```bash
.conda/bin/python -m pytest -q -p no:cacheprovider
.conda/bin/python -m ruff check --no-cache agent/dialogue config run_local_eval.py tests
git diff --check
```

Expected: all project tests pass; if full-repo Ruff exposes pre-existing findings, record them separately and require all changed files to pass.

- [ ] **Step 2: Run one offline official-evaluator regression**

```bash
LLM_PROVIDER=none SKIP_DATA_VERIFY=1 .conda/bin/python run_local_eval.py --catalog /Users/zhengce/projects/participate_kit/catalog.jsonl --dataset data/public_set.jsonl --output /private/tmp/transition-guard-disabled.json
```

Expected: the disabled result matches the pre-change legacy session behavior and contains JSON-safe guard statistics.

- [ ] **Step 3: Request code review and fix blocking findings**

Review configuration safety, disabled-path equivalence, destructive-operation semantics, state immutability and diagnostic privacy.

- [ ] **Step 4: Commit review fixes if needed**

```bash
git add agent/dialogue config tests run_local_eval.py README.md
git commit -m "fix: address transition guard review"
```
