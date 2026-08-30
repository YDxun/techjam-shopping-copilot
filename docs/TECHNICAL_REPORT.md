# TechJam2026 Shopping Copilot · Independent Technical Report

> Numbers reflect `results.json` (full 200-session run with the current default config):
> **HitRate@10 = 1.0 / MRR = 0.9364 / MTTC = 2.33 / TechnicalScore = 0.9543**
> (`rrf_k=100` + constraint-combination fingerprint + confidence-gated output; fully reproducible offline with zero API cost).

## 1. Architecture (four pillars + dialogue pipeline + automation control)

```
Official Agent interface (reset/respond) -- strictly compatible; evaluator untouched
  |- DialogueUnderstandingPipeline (agent/dialogue/)
  |    |- recognizers: cascaded intent recognition (rules first + hard-cue upgrade + strict-JSON LLM fallback)
  |    |- reducer: atomic state reduction (intent_version versioning, override semantics)
  |    |- question_policy + catalog_signals: catalog-aware ask utility and stop policy
  |    |- product_history: versioned product display/feedback loop
  |- IntentRouter (dual-track buying/browsing intent routing)
  |- HybridRetriever (FTS5 weighted BM25 + category + hard-constraint AND + BLaIR dense-recover, RRF fusion)
  |- Reranker (rule coverage + combo_bonus + constraint fingerprint + optional qwen3-rerank / RexReranker)
  |- RuntimeController (capability probe -> environment-adaptive strategy selection; LUT-driven + phase circuit breakers + rewrite guard)
```

- **Pillar I Core architecture**: dual-track intent routing (high-precision hard filtering for buying / diverse recall for browsing);
  multi-route hybrid retrieval (FTS5 weighted BM25, category domain, hard-constraint AND, BLaIR dense only in recover mode);
  dual-safety reranking with rules + optional models.
- **Pillar II Multi-turn strategy**: dynamic state machine with incremental slot accumulation and sudden-intent override
  (override: old preferences stay as weak soft signals, versioned by intent_version); proactive clarification on candidate overflow,
  stop-asking on preference exhaustion; 4-5 phrasing templates per attribute (random mode, seeded, no consecutive repeats)
  that never affect ask_attribute or scoring.
- **Pillar III Self-evolution**: each turn distills the dialogue history into a RecommendationContext and dynamically switches
  probe/exploit/recover modes; capability probing + the config-environment-performance LUT pick the best strategy per environment
  -- no model training required.
- **Pillar IV Evaluation alignment**: hybrid retrieval protects HitRate; combo_bonus + constraint fingerprint + output gating push
  the target up for MRR; clarification/stop policies reduce MTTC.

## 1.1 Automation-control maturity (team highlight, P1-P4)

| Loop | Implementation | Deliverable |
|---|---|---|
| P1 startup selection | CapabilityProbe detects device/dense/LLM/network/reranker -> RuntimeController.decide() picks the best config_id via the LUT | agent/runtime_controller.py + utils/lut.py + data/assets/env_config_lut.json |
| P1 runtime degradation | phase circuit breakers (dense / reranker / LLM): consecutive failures degrade on the spot (hybrid->bm25, rerank->rule, llm->rule) | utils/circuit_breaker.py (wired into retriever/reranker) |
| P1 rewrite detection | signals like consecutive 'disclosure but zero new constraints' -> upgrade to LLM intent on the spot, else refined rules | agent/rewrite_guard.py + pipeline.set_recognition_mode() |
| P2 observability | per-session structured logs (strategy/latency/tokens/phase_timings/degradation/reasons) aggregated into results.json | agent/main_agent.py::_record_session_log + run_local_eval.py |
| P3 config-as-data | config/profiles.py CONFIG_PROFILES as the single source of truth, shared and validated by build_lut and the controller | config/profiles.py + tests/test_config_profiles.py |
| P4 cost/latency disclosure | LUT includes measured latency (3-run median) and cost_usd_per_session (tokens x unit price) | data/assets/env_config_lut.json + docs/cost_disclosure.md |

## 2. Models

| Model | Purpose | Dependency | Default |
|---|---|---|---|
| SQLite FTS5 (weighted BM25) | lexical recall | standard library | on |
| BLaIR hyp1231/blair-roberta-large (offline npy) | dense semantic recall (recover) | transformers + offline npy | auto |
| Rule rerank (coverage + combo + fingerprint + gating) | fine ranking | standard library | on |
| qwen3-rerank (Alibaba Cloud MaaS) | text rerank (optional) | DASHSCOPE key + network | off by default |
| RexReranker-0.6B / bge-reranker-v2-m3 | cross-encoder rerank (optional) | local model cache | off by default |

A/B evidence: semantic rerankers (qwen3/bge/Rex) as the final fallback reranker hurt MRR on this deterministic evaluator, so they stay off by default;
BLaIR is enabled only in recover mode (zero public-set loss, private-set safety net); `rrf_k=100` + constraint fingerprint + confidence gating
are the robust public-set winners (train160 / holdout40 both ~0.955, not pure overfitting).

## 3. Cost & latency

See `docs/cost_disclosure.md`: the default is zero-API, fully offline; online modes are billed per token (feasibility metric only,
not part of TechnicalScore). Latency baselines and per-session token estimates come from the LUT (`data/assets/env_config_lut.json`).

## 4. Limitations

1. The agent leverages the deterministic simulator wording; if the private set introduces paraphrases, hard-cue upgrades + cascaded LLM +
   the review_paraphrase asset + the rewrite guard back it up (tools/paraphrase_eval.py provides L1/L2 stress tests).
2. The user profile carries little information on the public set (weak prior only); a more discriminative private profile could be re-weighted.
3. Semantic reranking mismatches the deterministic evaluator mechanics (A/B proven), so it stays an optional enhancement.
4. The public 200 sessions and private 800 may differ in difficulty distribution; the LUT is measured on public data and needs recalibration for private.
5. LUT latency is measured on this machine (RTX 3050 Laptop); absolute CPU/cloud latency differs (relative ranking holds).

## 5. Team contributions (5 members)

| Member | Role | Main outputs |
|---|---|---|
| A - Data | data inventory / dictionary / question value | scripts/build_index.py, data/analysis/* (vocab/field_mapping/question_value), data/assets/* |
| B - Dialogue | dialogue understanding pipeline | agent/dialogue/ (recognizers/reducer/question_policy/product_history/pipeline) |
| C - Retrieval | retrieval/rerank pipeline | agent/retriever.py, agent/reranker.py (rules + combo + fingerprint), retrieval_pipeline/, scripts/encode_catalog_blair.py |
| D - Evaluation | evaluation alignment / tuning / LUT / automation | run_local_eval.py, scripts/tune_*.py, scripts/build_lut.py, data/assets/env_config_lut.json, agent/runtime_controller.py, circuit-breaker/rewrite-guard A/B |
| E - Coordination | integration / docs / delivery | four-pillar engineering integration, README.md, docs/*, unified dependencies & environment awareness |
