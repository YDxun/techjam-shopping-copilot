# Decision Diagnostics and Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add privacy-safe decision traces, catalog-scale experiments, and nested cross-validation that select stable question-policy parameters from the 200 public sessions while regularizing against the 50,000-product catalog.

**Architecture:** Record structured per-turn decision facts outside the official response, expose deterministic experiment entry points, then run a bounded search with grouped stratified nested folds and paired uncertainty estimates. The output is a reviewed configuration recommendation; default promotion remains a separate explicit configuration change.

**Tech Stack:** Python 3.11+ standard library, JSON/JSONL, existing evaluator, `unittest`/pytest, no sklearn/pandas requirement.

**Spec:** `docs/superpowers/specs/2026-08-29-catalog-aware-question-policy-design.md`

## Global Constraints

- Complete the transition-guard and catalog-aware question-policy plans before this plan.
- Traces are disabled by default and never enter the official turn response.
- Store only hashed session identifiers; never store API keys, raw LLM responses or default full user messages.
- Decision sweeps use a fixed intent-recognition mode and version so LLM nondeterminism cannot choose policy weights.
- The public 200 sessions may tune global parameters; runtime policy may not read scenario labels or ground truth.
- Search at most 50 coarse configurations before local refinement.
- Do not promote experimental values into `config/default.json` without a separately reviewed result report.

---

### Task 1: Add local decision-trace configuration and recorder

**Files:**
- Modify: `config/models.py`
- Modify: `config/default.json`
- Modify: `config/loader.py`
- Modify: `config/env_config.py`
- Create: `agent/dialogue/diagnostics.py`
- Create: `tests/test_decision_diagnostics.py`

**Interfaces:**
- Consumes: recognition, GuardDecision, before/after state, CandidateQuestionSignals, QuestionDecision and token usage.
- Produces: `DecisionTraceConfig`, `DialogueDecisionTrace`, `DecisionTraceRecorder.record()`, `.summary()` and `.export_jsonl(path)`.

- [ ] **Step 1: Write failing configuration and privacy tests**

```python
def test_decision_trace_defaults_to_disabled(self) -> None:
    config = EnvConfig.from_env(environ={}).diagnostics.decision_trace
    self.assertFalse(config.enabled)
    self.assertEqual(config.max_traces, 5000)

def test_trace_hashes_session_and_omits_sensitive_fields(self) -> None:
    recorder = DecisionTraceRecorder(DecisionTraceConfig(enabled=True, max_traces=2))
    recorder.record(trace_input(session_id="secret-session", user_message="private text"))
    payload = recorder.records()[0].to_dict()
    encoded = json.dumps(payload)
    self.assertNotIn("secret-session", encoded)
    self.assertNotIn("private text", encoded)
    self.assertNotIn("raw_llm_response", payload)
```

Also assert max-trace truncation, deterministic hashes, JSON-safe finite numbers and disabled no-op behavior.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
.conda/bin/python -m pytest -q -p no:cacheprovider tests/test_decision_diagnostics.py tests/test_config_loader.py
```

Expected: missing diagnostics configuration and recorder.

- [ ] **Step 3: Add immutable configuration**

Define:

```python
@dataclass(frozen=True)
class DecisionTraceConfig:
    enabled: bool = False
    include_attribute_scores: bool = True
    include_state_diff: bool = True
    max_traces: int = 5000
    output_path: str = "decision_traces.jsonl"

@dataclass(frozen=True)
class DiagnosticsConfig:
    decision_trace: DecisionTraceConfig = field(default_factory=DecisionTraceConfig)
```

Add `diagnostics` to AppConfig, JSON defaults, parser validation and key environment switches `SHOPPING_DIAGNOSTICS__DECISION_TRACE__ENABLED` and `SHOPPING_DIAGNOSTICS__DECISION_TRACE__OUTPUT_PATH`.

- [ ] **Step 4: Implement focused trace records**

Define `DialogueDecisionTrace` with hashed session ID; turn; recognition source, act, confidence, ambiguities and fallback reason; guard action and reason; intent version; added/removed normalized constraints; candidate count, score summary and missing rates; per-attribute ExpectedShrink, Resolve@10/3/1, P90, exploration/finish/final scores; selected attribute, reason, finish pressure and lookahead depth; recommendation count; prompt/completion tokens. Build state differences as tuples of `(attribute, normalized_value, strength)` and round values only in `to_dict()`. Freeze nested mappings when constructing a record. `DecisionTraceRecorder.record()` increments aggregate counters even after the detailed-record cap is reached.

```python
class DecisionTraceRecorder:
    def summary(self) -> dict[str, object]:
        return {
            "enabled": self.config.enabled,
            "recorded": len(self._records),
            "total_seen": self._total_seen,
            "decision_reasons": dict(sorted(self._decision_reasons.items())),
            "guard_actions": dict(sorted(self._guard_actions.items())),
        }
```

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the Task 1 test command. Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add config/models.py config/default.json config/loader.py config/env_config.py agent/dialogue/diagnostics.py tests/test_decision_diagnostics.py
git commit -m "feat: record privacy-safe decision traces"
```

### Task 2: Integrate traces with the two-stage pipeline and local runner

**Files:**
- Modify: `agent/dialogue/pipeline.py`
- Modify: `agent/main_agent.py`
- Modify: `run_local_eval.py`
- Modify: `tests/test_dialogue_flow.py`
- Modify: `tests/test_llm_startup.py`

**Interfaces:**
- Consumes: `DecisionTraceRecorder` and finalized DialogueTurnResult.
- Produces: `Agent.dialogue_decision_statistics()` and optional JSONL trace export after local evaluation.

- [ ] **Step 1: Write failing integration tests**

```python
def test_trace_summary_is_local_only(self) -> None:
    agent = build_agent(trace_enabled=True)
    response = agent.respond("s", "I'm looking for shoes.", 1, 3)
    self.assertEqual(set(response), {"message", "ask_attribute", "recommendations", "usage"})
    summary = agent.dialogue_decision_statistics()
    self.assertEqual(summary["total_seen"], 1)

def test_runner_exports_trace_without_changing_sessions(self) -> None:
    result = run_with_mock_agent(trace_enabled=True)
    self.assertIn("dialogue_decision_statistics", result)
    self.assertNotIn("decision_trace", result["sessions"][0])
```

- [ ] **Step 2: Run focused tests and verify RED**

```bash
.conda/bin/python -m pytest -q -p no:cacheprovider tests/test_dialogue_flow.py tests/test_llm_startup.py
```

Expected: missing recorder integration and Agent statistics method.

- [ ] **Step 3: Record after the decision and candidate signals are final**

Call the recorder once in `decide_question`, after recording the selected attribute and before returning DialogueTurnResult. Pass only normalized state diffs, aggregate candidate metrics, component scores, reason codes and usage.

Expose summary through Agent. In `run_local_eval.py`, append summary to the local result and export JSONL only when enabled. Resolve a relative trace path against the repository working directory; refuse a path that resolves to the catalog or public dataset file.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 2 test command. Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add agent/dialogue/pipeline.py agent/main_agent.py run_local_eval.py tests/test_dialogue_flow.py tests/test_llm_startup.py
git commit -m "feat: export local dialogue decision diagnostics"
```

### Task 3: Add the catalog-scale question-value experiment

**Files:**
- Create: `experiments/__init__.py`
- Create: `experiments/catalog_question_value.py`
- Create: `tests/test_catalog_question_experiment.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: catalog path, AttributeCache, CandidateSignalCalculator, sample sizes and deterministic seed.
- Produces: JSON report with extraction coverage, pool-size stability, value metrics and timing; CLI returns nonzero on invalid input.

- [ ] **Step 1: Write failing experiment tests against a temporary catalog**

```python
report = run_catalog_experiment(
    catalog_path=catalog_path,
    pool_sizes=(3, 4),
    sample_count=4,
    seed=17,
)
self.assertEqual(report["catalog_count"], 4)
self.assertEqual(set(report["pool_sizes"]), {"3", "4"})
self.assertIn("attribute_coverage", report)
self.assertIn("latency_ms", report["pool_sizes"]["3"])
```

Also run twice and assert identical non-timing fields, validate missing catalog handling, and ensure reports contain no product titles or descriptions.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
.conda/bin/python -m pytest -q -p no:cacheprovider tests/test_catalog_question_experiment.py
```

Expected: missing experiment module.

- [ ] **Step 3: Implement deterministic sampling and aggregate reporting**

Use `random.Random(seed)`, category-stratified product sampling, and `time.perf_counter()` around signal calculation only. For every pool size report p50/p95 latency, chosen-attribute agreement against the largest pool, mean ExpectedShrink, mean Resolve@10, one-step versus two-step finish gain and per-attribute missing rates. Repeat each sample after deterministic 10% candidate deletion and report attribute-choice agreement plus metric deltas as the catalog perturbation stability measure.

Provide CLI arguments:

```text
--catalog PATH
--output PATH
--pool-sizes 300,500,1000
--sample-count 1000
--seed 20260829
```

Write JSON atomically through a temporary file in the output directory followed by `Path.replace()`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 3 test command. Expected: all selected tests pass.

- [ ] **Step 5: Run the real catalog experiment**

```bash
.conda/bin/python -m experiments.catalog_question_value --catalog /Users/zhengce/projects/participate_kit/catalog.jsonl --output /private/tmp/catalog-question-value.json --pool-sizes 300,500,1000 --sample-count 1000 --seed 20260829
```

Expected: a reproducible aggregate JSON report with no raw product text.

- [ ] **Step 6: Document and commit Task 3**

```bash
git add experiments/__init__.py experiments/catalog_question_value.py tests/test_catalog_question_experiment.py README.md
git commit -m "feat: analyze question value across the catalog"
```

### Task 4: Implement bounded nested cross-validation and paired selection

**Files:**
- Create: `experiments/decision_cross_validation.py`
- Create: `experiments/decision_search_space.json`
- Create: `tests/test_decision_cross_validation.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: public dataset, catalog, a maximum-50 coarse search space, fixed recognition mode/version and catalog experiment baseline.
- Produces: deterministic fold manifest, per-config fold metrics, paired bootstrap intervals, one-standard-error selection and recommended config overlay.

- [ ] **Step 1: Write failing split and selection tests**

Create 20 artificial samples with known scenario counts and repeated targets. Assert target groups never cross folds, each fold preserves scenario counts as closely as possible, and the same seed produces identical manifests:

```python
folds = grouped_stratified_folds(samples, fold_count=5, seed=20260829)
target_to_fold = {}
for fold_index, fold in enumerate(folds):
    for sample in fold:
        target = sample["ground_truth"]["parent_asin"]
        assigned = target_to_fold.setdefault(target, fold_index)
        self.assertEqual(assigned, fold_index)
self.assertEqual(folds, grouped_stratified_folds(samples, 5, 20260829))
```

Add tests for maximum search size, paired bootstrap determinism, catalog-regression rejection, 4/5 non-regressing-fold requirement and one-standard-error preference for the simpler configuration.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
.conda/bin/python -m pytest -q -p no:cacheprovider tests/test_decision_cross_validation.py
```

Expected: missing cross-validation module.

- [ ] **Step 3: Implement grouped stratified nested folds without sklearn**

Group rows by target ASIN, annotate each group with scenario, coarse category and initial candidate-count bin, sort largest/rarest groups first, then greedily assign each group to the fold with the lowest weighted imbalance. Persist the exact fold manifest in the output report.

Use five outer folds. Within each outer-training set, use four inner folds for parameter selection. Evaluate every config with the existing official `evaluate()` function; do not modify evaluator code.

- [ ] **Step 4: Define the bounded coarse search space**

Commit explicit global profiles and scalar arrays whose Cartesian product is sampled deterministically down to at most 50 configurations. Each profile expands to ordinary unified-config fields before evaluation:

```json
{
  "transition_guard_profile": [
    {"enabled": false},
    {"enabled": true, "add": 0.60, "replace": 0.85, "remove": 0.85, "reject_products": 0.85, "no_preference": 0.80, "no_more_preferences": 0.90},
    {"enabled": true, "add": 0.65, "replace": 0.90, "remove": 0.90, "reject_products": 0.90, "no_preference": 0.85, "no_more_preferences": 0.95}
  ],
  "candidate_weight_profile": [
    {"expected_shrink": 0.30, "coverage": 0.15, "complementarity": 0.15, "answer_probability": 0.15, "missing_penalty": 0.20, "redundancy_penalty": 0.20, "repeat_penalty": 0.40, "no_preference_penalty": 0.60, "turn_cost": 0.15},
    {"expected_shrink": 0.45, "coverage": 0.10, "complementarity": 0.10, "answer_probability": 0.10, "missing_penalty": 0.25, "redundancy_penalty": 0.20, "repeat_penalty": 0.40, "no_preference_penalty": 0.60, "turn_cost": 0.10},
    {"expected_shrink": 0.25, "coverage": 0.20, "complementarity": 0.10, "answer_probability": 0.25, "missing_penalty": 0.25, "redundancy_penalty": 0.20, "repeat_penalty": 0.50, "no_preference_penalty": 0.70, "turn_cost": 0.15}
  ],
  "finish_weight_profile": [
    {"resolve_at_10": 0.50, "resolve_at_3": 0.20, "resolve_at_1": 0.10, "terminal_progress": 0.30, "p90_remaining_penalty": 0.20},
    {"resolve_at_10": 0.65, "resolve_at_3": 0.15, "resolve_at_1": 0.05, "terminal_progress": 0.25, "p90_remaining_penalty": 0.20},
    {"resolve_at_10": 0.40, "resolve_at_3": 0.20, "resolve_at_1": 0.10, "terminal_progress": 0.45, "p90_remaining_penalty": 0.30}
  ],
  "pool_size": [300, 500, 1000],
  "prior_alpha": [0.0, 0.25, 0.5],
  "prior_temperature": [0.75, 1.0, 1.5],
  "other_answer_probability": [0.65, 0.75, 0.85],
  "other_vagueness_penalty": [0.05, 0.10, 0.20],
  "finish_candidate_threshold": [50, 100, 200],
  "remaining_question_threshold": [1, 2],
  "lookahead_depth": [1, 2],
  "termination_mode": ["explicit_only"]
}
```

Record the sampling seed and every evaluated expanded configuration. Include the legacy baseline even if deterministic sampling would omit it. After coarse selection, locally refine only the three most stable profiles by moving one scalar to an adjacent midpoint per trial, while keeping the combined coarse-plus-refinement evaluations bounded and recorded. Do not add per-category, per-scenario or per-sample parameters.

- [ ] **Step 5: Implement stable selection**

For each configuration calculate mean official technical score, fold standard deviation, scenario metrics, catalog simulation regression and complexity count. Reject configurations that lose HR@10 beyond the configured tolerance, regress more than one outer fold, materially collapse any scenario, or exceed measured latency budget.

Use paired bootstrap over session-level score contributions with a fixed seed. Among configurations within one standard error of the best eligible mean, select the lowest complexity, then lowest latency, then lexicographically smallest canonical JSON for deterministic ties.

- [ ] **Step 6: Freeze intent recognition during sweeps**

Set `LLM_PROVIDER=none` and record the recognizer version as the current commit SHA in the report. The official simulator language remains processed by the deterministic rule path during broad decision search. After selecting a shortlist, validate those configurations with the separately controlled live-LLM command; live results cannot change folds or expand the search space.

- [ ] **Step 7: Run focused tests and verify GREEN**

Run the Task 4 test command. Expected: all split, search and selection tests pass.

- [ ] **Step 8: Run the real nested evaluation**

```bash
LLM_PROVIDER=none SKIP_DATA_VERIFY=1 .conda/bin/python -m experiments.decision_cross_validation --catalog /Users/zhengce/projects/participate_kit/catalog.jsonl --dataset data/public_set.jsonl --search-space experiments/decision_search_space.json --catalog-report /private/tmp/catalog-question-value.json --output /private/tmp/decision-cross-validation.json --recommended-config-output /private/tmp/recommended-decision-config.json --seed 20260829
```

Expected: report contains the fold manifest, all tested configs, exclusions with reason codes, paired intervals and exactly one recommended overlay. The second output is a complete config document formed by applying that overlay to the loaded base config, so it can be passed directly through `APP_CONFIG_PATH`.

- [ ] **Step 9: Document and commit Task 4**

```bash
git add experiments/decision_cross_validation.py experiments/decision_search_space.json tests/test_decision_cross_validation.py README.md
git commit -m "feat: cross-validate dialogue decision parameters"
```

### Task 5: Produce the promotion report and final verification

**Files:**
- Create: `docs/decision-policy-calibration.md`
- Review only: complete implementation from all three plans.

**Interfaces:**
- Consumes: catalog report, nested-CV report, offline baseline and optional controlled live result.
- Produces: a human-reviewable recommendation; no automatic default promotion.

- [ ] **Step 1: Run fresh verification**

```bash
.conda/bin/python -m pytest -q -p no:cacheprovider
.conda/bin/python -m ruff check --no-cache agent config experiments run_local_eval.py tests
git diff --check
```

Expected: tests and targeted Ruff pass; diff check is clean.

- [ ] **Step 2: Generate the calibration document from exact report fields**

Document:

- baseline and recommended canonical JSON
- outer-fold mean and standard deviation
- per-scenario deltas
- paired bootstrap interval
- catalog simulation deltas
- p50/p95 latency and cache memory
- enabled/disabled TransitionGuard comparison
- explicit rollback configuration
- every exclusion reason applied to the winning neighborhood

Copy numeric values from the saved reports; do not manually recompute or round beyond six decimals.

- [ ] **Step 3: Run one controlled live-LLM comparison for the selected overlay**

```bash
LLM_PROVIDER=deepseek LLM_INTENT_ENABLE=1 SKIP_DATA_VERIFY=1 APP_CONFIG_PATH=/private/tmp/recommended-decision-config.json .conda/bin/python run_local_eval.py --catalog /Users/zhengce/projects/participate_kit/catalog.jsonl --dataset data/public_set.jsonl --output /private/tmp/recommended-live-result.json
```

Expected: report valid LLM acceptance/fallback statistics and decision metrics. Treat response variance as validation evidence, not as a new tuning search.

- [ ] **Step 4: Request code and methodology review**

Review privacy, evaluator isolation, fold leakage, target grouping, bounded search, one-standard-error selection, report reproducibility, disabled defaults and official response compatibility.

- [ ] **Step 5: Commit the reviewed calibration report**

```bash
git add docs/decision-policy-calibration.md
git commit -m "docs: report decision policy calibration"
```

- [ ] **Step 6: Stop before default promotion**

Present the recommended overlay and rollback configuration to the user. Change `config/default.json` only in a separately approved configuration commit after the user reviews `docs/decision-policy-calibration.md`.
