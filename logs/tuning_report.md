# TechJam2026 Offline Parameter-Tuning Report

## 0. Baseline (step 0)
- Git working-tree baseline (before knob exposure): **TS=0.880188 / HR=1.0 / MRR=0.648627 / MTTC=1.72**
- `python run_local_eval.py` (default config) -> results.json

## 1. Knob exposure (step 1)
- `agent/retriever.py` -> `config.retrieval`:
  - `bm25_field_weights` (FTS5 weights; both SQL sites unified through `_bm25_weights_sql`)
  - `rrf_k` (60), `rrf_constraint_k` (10), `dense_weight` (0.5)
  - `bm25_limit_mult` (2), `recall_limit_mult` (3)
  - new env vars: `RRF_K / RRF_CONSTRAINT_K / DENSE_WEIGHT / BM25_FIELD_WEIGHTS`
- `agent/reranker.py` -> `config.rerank_weights` (coverage/combo/category/rrf/popularity/profile)
  + `config.fingerprint` (enable/bonus_unique/ten/fifty/max_count)
- `agent/main_agent.py` -> `config.retrieval_pool_size` (300)
- Verification: the full score after exposure is **exactly 0.8802** (behavior unchanged)

## 2. Tuning (step 2; 160 tuning + 40 validation, `scripts/tune_knobs.py`)
- Four grid groups: rerank / retrieval / strategy / joint (logs in `logs/tune_*.json`)

### rerank group (160 tuning, top)
| Knob | Value | tune160 TS | tune160 MRR |
|---|---|---|---|
| baseline | rrf=0.15 | 0.8783 | 0.6375 |
| **rrf** | **0.05** | **0.8824** | **0.6539** |
| fingerprint | on | 0.8795 | 0.6417 |
| popularity | 0.10 | 0.8710 | 0.6093 (drops on 160) |

### retrieval group (160 tuning, top)
| Knob | Value | tune160 TS | tune160 MRR |
|---|---|---|---|
| baseline | rrf_k=60 | 0.8783 | 0.6375 |
| **rrf_k** | **100** | **0.8814** | **0.6508** |
| bm25_feat | 5.0 | 0.8785 | 0.6389 |
| dense_weight | 0.3 | 0.8783 | 0.6375 |

### strategy group (100 tuning)
- exploit_min_hard / exploit_min_constraints / max_questions / ask_ig / rule_conf / hard_cue: no change at all
  (flat 0.8658) -- **these knobs are inert under the public set's official templates** (they only affect private-set/paraphrase robustness); recorded honestly.

### Joint comparison (`logs/tune_final.json`: 160 tuning + 40 validation, reproducible via `scripts/tune_knobs.py`)
| Config | tune160 TS | tune160 MRR | valid40 TS | valid40 MRR |
|---|---|---|---|---|
| baseline | 0.8783 | 0.6375 | 0.8879 | 0.6931 |
| rrf=0.05 | 0.8824 | 0.6539 | 0.8762 | 0.6572 (regresses on 40) |
| **rrf_k=100** | 0.8814 | 0.6508 | **0.8943** | **0.7193** (OK) |
| rrf0.05+rrf_k100 | 0.8867 | 0.6699 | 0.8538 | 0.5859 (overfit) |
| rrf0.05+fp | 0.8843 | 0.6601 | 0.8762 | 0.6572 (regresses on 40) |

**Conclusion: `retrieval.rrf_k = 100` is the robust optimum** (up on 160, up on 40, up on the full set; lowering RRF fusion discriminability lets
coverage/combo dominate and reduces retrieval-fusion noise). Adopted as the default (default.json rrf_k 60->100).

## 3. LUT (step 3; `scripts/build_lut.py` -> `data/assets/env_config_lut.json`)
- Profiles: rule_bm25 / hybrid_dense / fingerprint_combo / text_rerank / reranker_model
- Environment axes: device x dense x llm x network (dense=no is simulated; llm/network without keys record the post-fallback behavior)
- 40-session smoke relative ordering + full-200 confirmation of key profiles; wired to `utils/lut.py` + `RuntimeController.decide()`
  (startup prints `lut=fingerprint_combo`); a missing/absent LUT falls back to the default.
- Full confirmation: **fingerprint_combo (rrf_k=100 + fingerprint) TS=0.8857 / MRR=0.6703** is the recommended optimum for this environment at that time.

## 4. Final defaults (post-tuning)
| Metric | Baseline (step 0) | Tuned default (rrf_k=100) | +fingerprint (LUT-recommended) |
|---|---|---|---|
| TS | 0.8802 | **0.8839** | **0.8857** |
| MRR | 0.6486 | **0.6645** | **0.6703** |
| HR@10 | 1.0 | 1.0 | 1.0 |
| MTTC | 1.72 | 1.77 | 1.77 |

Per scenario (fingerprint_combo): boundary MRR 0.6629->0.6935, browsing 0.5578->0.6224, buying ~0.690,
override 0.7731->0.7375 (slight drop; net positive).

## 5. Acceptance
1. After step 1 the full score equals 0.8802 (OK); 2. 160 improves (0.8783->0.8814) and 40 does not regress (0.8879->0.8943) (OK; see tables above);
3. LUT generated and the RuntimeController per-env selection unit test passes (OK); 4. evaluator untouched, no model training, no external data downloaded,
the 40 sessions are validation-only; 5. submission: code + logs + LUT + this report.
