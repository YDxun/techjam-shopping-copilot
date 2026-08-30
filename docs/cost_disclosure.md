# Cost / Latency Disclosure (competition deliverable)

> Note: **token usage is a feasibility metric only and does NOT count toward TechnicalScore** (the official evaluator only reports it).
> Latency is measured locally (RTX 3050 Laptop / 16GB RAM / Windows), for reference only.
> Full data: `data/assets/env_config_lut.json` -- 16 environment fingerprints x 5 config profiles, n_samples=200,
> latency is the median of 3 small-sample measurements (`latency_ms_per_turn`), plus `cost_usd_per_session`.

## 1. Per-mode latency & token baselines (measured locally)

### 1a. Small-sample warm-cache measurement (from the LUT, median)

| Mode | Latency per turn (ms, median) | Avg tokens/session | Note |
|---|---|---|---|
| rule_bm25 (pure rules offline, baseline) | ~113 | **0** | pure Python stdlib + SQLite FTS5, zero external calls |
| hybrid_dense (+BLaIR dense, recover-gated) | ~117 | **0** | dense triggers only in recover; almost never on public |
| fingerprint_combo (+fingerprint + gating, default) | ~119 | **0** | fingerprint exactly counts catalog products satisfying all constraints |
| text_rerank (qwen3-rerank, needs key + network) | ~111 (fallback w/o key) / 350-600 (with network) | ~3,000-4,000/session (est.) | reranks top-12 per turn; auto-fallback without key |
| reranker_model (RexReranker-0.6B / bge) | ~111 (fallback w/o model) / 1-10s in recover | 0 (local inference) | second opinion only in recover; models ~1.2-2.3GB |

### 1b. Full cold-start measurement (complete 200-session evaluation)

`python run_local_eval.py` over the full 200 sessions takes about **2-3 minutes** (including evaluator/data loading and cold caches),
which is roughly **350-450 ms/turn** at MTTC=2.33. The LUT small-sample latency is a warm-cache baseline; both are disclosed here.

## 2. Online-mode cost estimates

Unit prices (public reference prices, per token; actual prices depend on the vendor):

| Service | Input $/1M | Output $/1M |
|---|---|---|
| DeepSeek `deepseek-chat` | ~0.27 | ~1.10 |
| OpenAI `gpt-4o-mini` | ~0.15 | ~0.60 |
| qwen3-rerank (Alibaba Cloud MaaS) | per call | -- |

Per-session estimates (extrapolated from the 200-session measurements):
- **LLM intent recognition (cascaded)**: the public set's rules are high-confidence, so the LLM rarely triggers (10-session test: 777 prompt / 150 completion);
  full run ~**0.08-0.3K tokens/session** -> ~**$0.00002-0.0001/session**.
- **qwen3-rerank text rerank**: ~2K tokens/turn (query + 12 candidates), ~1.8 turns -> **~3.5K tokens/session**,
  cost depends on the MaaS unit price (order of $0.001-0.01/session).
- Default mode: **$0** (fully offline, zero API).
- Every LUT profile records `cost_usd_per_session` (estimated with DeepSeek prices; 0 for offline modes).

## 3. Disclosure points
- Models: BLaIR `hyp1231/blair-roberta-large` (offline local), rule rerank, optional qwen3-rerank /
  RexReranker-0.6B / bge-reranker-v2-m3 (local).
- Keys: all injected via environment variables; no key exists in code or the repository
  (`DASHSCOPE_API_KEY` / `DEEPSEEK_API_KEY` / `OPENAI_API_KEY`).
- Fallback: any LLM/model failure/timeout/offline -> automatic rule fallback (phase circuit breakers + LUT baseline); fully offline-capable.
- Tokens are a feasibility metric only: the official evaluator reports `usage` but it is **not part of TechnicalScore**.
- Latency definition: `latency_ms_per_turn = median of (small-sample eval duration / sessions / MTTC)` over 3 reps (warm cache);
  full cold start ~350-450 ms/turn; absolute CPU latency is higher (relative ordering holds).
