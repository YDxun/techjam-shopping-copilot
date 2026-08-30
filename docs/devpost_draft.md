# Devpost Project Description (competition deliverable)

## One-liner
**Shopping Copilot**: an AI shopping-recommendation agent that works within 10 turns and runs fully offline at zero cost, using
"dual-track intent routing + multi-route hybrid retrieval (BM25+BLaIR) + dynamic state machine + runtime context programming + environment-adaptive control" to score
**HitRate@10 = 1.0 / MRR = 0.9364 / MTTC = 2.33 / TechnicalScore = 0.9543** on the TechJam2026 public set
(weak BM25 baseline: 0.125 / 0.068 / 9.81 / 0.15).

## Problem
In e-commerce search, user needs are often vague and shifting (browsing vs buying, preferences being overridden, no preference on some attributes).
Single-turn retrieval cannot locate a hidden target item within a few turns. The task requires the agent to hit the target in at most 10 turns via
"clarifying questions + ranked recommendations", scored jointly by HitRate@K / MRR / MTTC.

## Solution (four pillars)
1. **Core architecture**: dual-track intent routing (buying precision / browsing diversity) + multi-route hybrid retrieval
   (weighted BM25, category filtering, hard-constraint AND, BLaIR dense only in recover, RRF fusion at `rrf_k=100`)
   + rule reranking (constraint coverage + combo_bonus + constraint fingerprint + confidence gating, optional qwen3-rerank / RexReranker).
2. **Multi-turn strategy**: a dialogue pipeline maintains category/constraint slots and scenario signals; override semantics
   ("ignore my earlier preference") are handled with intent_version versioning; proactive clarification on candidate overflow,
   stop-asking on preference exhaustion to optimize MTTC; multi-template phrasing (seeded random + no consecutive repeats) keeps dialogs natural.
3. **Self-evolution**: each turn distills the dialogue history into a recommendation context and dynamically switches probe/exploit/recover modes;
   capability probing + the config-environment-performance LUT let the agent auto-select the best strategy per environment; phase circuit breakers and a
   rewrite guard degrade/upgrade at runtime -- no model training needed.
4. **Evaluation alignment**: hybrid retrieval protects HitRate; combo_bonus + constraint fingerprint (exact catalog count) + output gating push
   the target up for MRR; clarification/stop policies lower MTTC.

## Automation-control maturity (team highlight)
- **Startup selection**: CapabilityProbe detects device/dense/LLM/network/reranker -> the LUT (16 env fingerprints x 5 config profiles) picks the
  highest-scoring startup default within latency/memory budgets; a missing LUT falls back to the safe rule baseline.
- **Runtime degradation**: phase circuit breakers for dense/reranker/LLM degrade on consecutive failures; rewrite signals upgrade to LLM intent on the spot.
- **Observability**: per-session structured logs (strategy/latency/tokens/phase/degradation) are written into results.json.
- **Config-as-data**: CONFIG_PROFILES is the single source of truth, so tuning / LUT / runtime never drift.

## Models & cost
- Default **zero LLM, zero API cost**, core uses only the Python standard library + SQLite FTS5; fully offline.
- Optional enhancements: BLaIR dense (offline npy), qwen3-rerank text rerank (Alibaba Cloud MaaS),
  RexReranker-0.6B / bge-reranker-v2-m3 local cross-encoders; all switched by environment variables and auto-fallback when dependencies are missing.
- Zero hardcoded keys (env-var injection only); A/B shows semantic rerankers as the final fallback reranker hurt MRR on this deterministic evaluator, so they stay off by default.

## Reproduction
`python run_local_eval.py` (one command: 200 public sessions, all official metrics to results.json).
`python scripts/demo_session.py` shows the turn-by-turn demo session.
`python -m unittest discover tests` / `python -m pytest` for the full test suite.

## Data compliance
Only the frozen competition toolkit is used; dataset SHA256 integrity verification; no upstream raw Amazon Reviews data is downloaded.

## Limitations & next steps
- The agent leverages the deterministic simulator wording; if the private set introduces paraphrases, hard-cue upgrades + cascaded LLM + the
  review_paraphrase asset + the rewrite guard back it up (stress-tested by tools/paraphrase_eval.py).
- Semantic reranking mismatches the deterministic evaluator mechanics (A/B proven) and stays an optional enhancement.
- Next: recalibrate the LUT for the private difficulty distribution, adversarial auto-tuning, cross-session profile learning.
