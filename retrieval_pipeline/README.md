# Retrieval Pipeline Module (task steps 4-6)

An agent-independent retrieval pipeline: **query building -> three-channel retrieval -> reranking**.
This module does **not** implement a state machine / intent parsing / Agent.respond / reset, and never modifies the evaluator.
The upper layer (intent recognition + dialogue state machine) passes a `SessionState`; this module returns `PipelineOutput.reranked_top10`.

## Layout
```
retrieval_pipeline/
|- config.py                 # constants (RRF k, BM25 field weights, penalty coefficients) + env vars
|- models.py                 # pydantic data classes (SessionState / QueryBundle / PipelineOutput)
|- data_access.py            # loads the offline npy product vectors + product catalog (never generates vectors)
|- query_builder.py          # step 4: constraint parsing / price-to-numeric / synonyms / variants / optional LLM rewrite
|- retriever_pipeline.py     # step 5: three-channel retrieval (structured/BM25/BLaIR) + RRF fusion
|- reranker_module.py        # step 6: BAAI/bge-reranker-v2-m3 rerank (degrades on no-GPU/OOM)
|- pipeline.py               # steps 4-6 orchestration entry: RetrievalPipeline.run()
|- test_pipeline.py          # demo: normal / RECOVER / override scenarios
```

## Data-class contract (boundary with the upper layer)
```python
SessionState = {
  "constraints": dict,          # {material:"cotton", color:"black", budget_max:50, ...}
  "recovery_mode": bool,        # miss streak >= 2
  "strategy_config": {"rrf_alpha":0.8, "retrieval_pool_size":50, "enable_query_variant":False, "enable_synonym":False},
  "user_raw_query": str,
}
PipelineOutput = {"raw_fused_candidates": [(asin, fused_score)], "reranked_top10": [asin x10]}
```

## Three channels (step 5)
1. **Structured-constraint matching**: hard filter in normal mode; score penalty in RECOVER
   (relaxation priority budget > size > material; penalty coefficients in config).
2. **Weighted BM25** (rank-bm25): title has the highest weight, description the lowest; supports query variants.
3. **BLaIR dense**: product vectors come from the **offline pre-computed npy** (this module only loads them); at inference
   **only the user query text is encoded**; dot-product recall.
4. RRF fusion: `score = sum 1/(k+rank)` with a dense-channel alpha weight; dedup -> candidate-pool truncation.

## Offline product-vector npy format (produced by scripts/encode_catalog_blair.py)
- `offline_blair_embeds.npy`: float32 `[N, dim]` (blair-roberta-large = 1024 dims)
- `offline_blair_embeds_asins.npy`: `[N]` parent_asins (row order matches the matrix)
- Missing/corrupt file -> the dense channel auto-disables without affecting the main flow.

### Pre-encoding with BLaIR (one-time offline preprocessing)
```bash
# smoke (validate dims/format first): 50 rows
python scripts/encode_catalog_blair.py --limit 50

# full 50k (CPU ~6h; --resume supports checkpoint continuation)
python scripts/encode_catalog_blair.py --output data/offline_blair_embeds.npy
```
The encoding convention matches the official `hyp1231/AmazonReviews2023 generate_emb.py`: **CLS pooling**
(`last_hidden_state[:, 0]`) + L2 normalization; retrieval uses dot product. Text construction follows the data-analysis findings
(`data/analysis/stats.json`): title + features(<=4) + categories; **description is dropped** (empty 47.8%) and so are details
(manufacturer-identifier noise) -- this both denoises and sharply cuts CPU encoding time.

## Environment variables
| Variable | Default | Description |
|---|---|---|
| `DEVICE` | `auto` | `auto` / `cpu` / `cuda` (rerank/encode device) |
| `QUERY_REWRITE_ENABLE` | `false` | enable LLM query rewrite (off -> template concatenation, fully offline) |
| `RERANKER_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | cross-encoder rerank model |
| `BLAIR_OFFLINE_EMBEDDING_PATH` | `data/offline_blair_embeds.npy` | offline product-vector path |
| `PRODUCT_CATALOG_PATH` | `data/catalog.jsonl` | frozen competition catalog |
| `BLAIR_QUERY_ENCODER_MODEL` | `hyp1231/blair-roberta-large` | BLaIR query-encoder model (matches offline encoding) |

## Running the demo
```bash
# core deps (pydantic numpy rank-bm25) are optional and listed in requirements.txt; everything degrades gracefully when absent
python retrieval_pipeline/test_pipeline.py
```
The demo covers: normal hard filter / RECOVER (penalty + synonyms + variants + pool 100) / override-cleared constraints.
Without FlagEmbedding, reranking degrades to fused-order ranking; without the npy, the dense channel auto-disables -- the whole chain runs with no paid API.
