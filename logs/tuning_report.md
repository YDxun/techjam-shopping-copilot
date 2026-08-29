# TechJam2026 离线参数优化报告

## 0. 基线（步骤 0）
- git 工作区基线（暴露旋钮前）：**TS=0.880188 / HR=1.0 / MRR=0.648627 / MTTC=1.72**
- `python run_local_eval.py`（默认配置）→ results.json

## 1. 旋钮暴露（步骤 1，队友 995b8ac/601becd 完成 + 本任务补齐 env）
- `agent/retriever.py` → `config.retrieval`：
  - `bm25_field_weights`（FTS5 权重，两处 SQL 统一走 `_bm25_weights_sql`）
  - `rrf_k`（60）、`rrf_constraint_k`（10）、`dense_weight`（0.5）
  - `bm25_limit_mult`（2）、`recall_limit_mult`（3）
  - 新增 env：`RRF_K / RRF_CONSTRAINT_K / DENSE_WEIGHT / BM25_FIELD_WEIGHTS`
- `agent/reranker.py` → `config.rerank_weights`（coverage/combo/category/rrf/popularity/profile）
  + `config.fingerprint`（enable/bonus_unique/ten/fifty/max_count）
- `agent/main_agent.py` → `config.retrieval_pool_size`（300）
- 验证：暴露后全量分数与 0.8802 **完全一致**（行为不变）

## 2. 调参（步骤 2，160 调参 + 40 验证，`scripts/tune_knobs.py` + `scripts/tune_final.py`）
- 分 4 组网格：rerank / retrieval / strategy / joint（日志 `logs/tune_*.json`）

### rerank 组（160 调参，Top）
| 旋钮 | 值 | tune160 TS | tune160 MRR |
|---|---|---|---|
| 基线 | rrf=0.15 | 0.8783 | 0.6375 |
| **rrf** | **0.05** | **0.8824** | **0.6539** |
| fingerprint | on | 0.8795 | 0.6417 |
| popularity | 0.10 | 0.8710 | 0.6093（160 上反而降） |

### retrieval 组（160 调参，Top）
| 旋钮 | 值 | tune160 TS | tune160 MRR |
|---|---|---|---|
| 基线 | rrf_k=60 | 0.8783 | 0.6375 |
| **rrf_k** | **100** | **0.8814** | **0.6508** |
| bm25_feat | 5.0 | 0.8785 | 0.6389 |
| dense_weight | 0.3 | 0.8783 | 0.6375 |

### strategy 组（100 调参）
- exploit_min_hard / exploit_min_constraints / max_questions / ask_ig / rule_conf / hard_cue 全部无变化
  （0.8658 一致）——**这些旋钮在公开集官方模板下是惰性的**（只影响私有集/改写鲁棒性），诚实记录。

### 联合对比（`logs/tune_final.json`：160 调参 + 40 验证）
| 配置 | tune160 TS | tune160 MRR | valid40 TS | valid40 MRR |
|---|---|---|---|---|
| 基线 | 0.8783 | 0.6375 | 0.8879 | 0.6931 |
| rrf=0.05 | 0.8824 | 0.6539 | 0.8762 | 0.6572 ❌40倒退 |
| **rrf_k=100** | 0.8814 | 0.6508 | **0.8943** | **0.7193** ✅ |
| rrf0.05+rrf_k100 | 0.8867 | 0.6699 | 0.8538 | 0.5859 ❌过拟合 |
| rrf0.05+fp | 0.8843 | 0.6601 | 0.8762 | 0.6572 ❌40倒退 |

**结论：`retrieval.rrf_k = 100` 是稳健最优**（160 升、40 升、全量升；降低 RRF 融合排序的区分度，
让覆盖度/combo 主导，减少检索融合噪音）。已采纳为默认（default.json rrf_k 60→100）。

## 3. LUT（步骤 3，`scripts/build_lut.py` → `data/assets/env_config_lut.json`）
- 配置档案：rule_bm25 / hybrid_dense / fingerprint_combo / text_rerank / reranker_model
- 环境维度：device × dense × llm × network（可模拟 dense=no；llm/network 无 key 记录回退）
- 40 条冒烟相对排序 + 关键档案全量 200 确认；`utils/lut.py` + `RuntimeController.decide()` 接线
  （启动打印 `lut=fingerprint_combo`），LUT 缺失/不在表内回退默认。
- 全量确认：**fingerprint_combo（rrf_k=100 + 指纹）TS=0.8857 / MRR=0.6703** 为该环境最优推荐。

## 4. 最终默认（调参后）
| 指标 | 基线（步骤 0） | 调参后默认（rrf_k=100） | +指纹（LUT 推荐） |
|---|---|---|---|
| TS | 0.8802 | **0.8839** | **0.8857** |
| MRR | 0.6486 | **0.6645** | **0.6703** |
| HR@10 | 1.0 | 1.0 | 1.0 |
| MTTC | 1.72 | 1.77 | 1.77 |

分场景（fingerprint_combo）：boundary MRR 0.6629→0.6935、browsing 0.5578→0.6224、buying ~0.690、
override 0.7731→0.7375（略降，净收益为正）。

## 5. 验收
1. 步骤 1 后全量 = 0.8802 ✅；2. 160 提升（0.8783→0.8814）+ 40 不倒退（0.8879→0.8943）✅ 对比表见上；
3. LUT 生成 + RuntimeController 按 env 选配置单测通过 ✅；4. 未改 evaluator、未训练模型、未下载外部数据、
40 条仅验证；5. 提交：代码 + logs + LUT + 本报告。
