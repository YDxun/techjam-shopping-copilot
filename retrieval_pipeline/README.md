# 检索管线模块（赛题第4-6步）

独立于上层 Agent 的检索管线：**查询构建 → 三通道检索 → 重排序**。
本模块**不实现**状态机 / 意图解析 / Agent.respond / reset，也不修改评测器。
上层（意图识别+对话状态机）传入 `SessionState`，本模块返回 `PipelineOutput.reranked_top10`。

## 目录
```
retrieval_pipeline/
├── config.py                 # 常量（RRF k、BM25 字段权重、惩罚系数）+ 环境变量
├── models.py                 # pydantic 数据类（SessionState / QueryBundle / PipelineOutput）
├── data_access.py            # 加载离线 npy 商品向量 + 产品目录（不生成向量）
├── query_builder.py          # 第4步：约束解析/价格转数值/同义词/变体/可选LLM改写
├── retriever_pipeline.py     # 第5步：三通道检索（结构化/BM25/BLaIR）+ RRF 融合
├── reranker_module.py        # 第6步：BAAI/bge-reranker-v2-m3 重排（无GPU/OOM降级）
├── pipeline.py               # 第4-6步编排入口 RetrievalPipeline.run()
├── test_pipeline.py          # 演示：普通/RECOVER/override 三场景
└── requirements-pipeline.txt
```

## 数据类契约（与上层边界）
```python
SessionState = {
  "constraints": dict,          # {material:"cotton", color:"black", budget_max:50, ...}
  "recovery_mode": bool,        # 连续 miss>=2
  "strategy_config": {"rrf_alpha":0.8, "retrieval_pool_size":50, "enable_query_variant":False, "enable_synonym":False},
  "user_raw_query": str,
}
PipelineOutput = {"raw_fused_candidates": [(asin, fused_score)], "reranked_top10": [asin x10]}
```

## 三通道（第5步）
1. **结构化约束匹配**：普通模式硬过滤；RECOVER 改为打分惩罚
   （放宽优先级 budget > size > material，惩罚系数见 config）。
2. **加权 BM25**（rank-bm25）：title 权重最高、description 最低；支持查询变体。
3. **BLaIR 稠密**：商品向量来自**离线预计算 npy**（本模块只加载），推理阶段**只编码用户查询文本**，点积召回。
4. RRF 融合：`score = Σ 1/(k+rank)`，稠密通道带 α 权重；去重 → 候选池截断。

## 离线商品向量 npy 格式（由 scripts/encode_catalog_blair.py 生成）
- `offline_blair_embeds.npy`：float32 `[N, dim]`（blair-roberta-large = 1024 维）
- `offline_blair_embeds_asins.npy`：`[N]` parent_asin（与矩阵行序一致）
- 文件缺失/损坏 → 稠密通道自动禁用，不影响主流程。

### 预先 BLaIR 编码（一次性离线预处理）
```bash
# 冒烟（先验证维度/格式）：50 条
python scripts/encode_catalog_blair.py --limit 50

# 全量 50k（CPU 约 6h；--resume 支持断点续跑）
python scripts/encode_catalog_blair.py --output data/offline_blair_embeds.npy
```
编码规范与官方 `hyp1231/AmazonReviews2023 generate_emb.py` 一致：**CLS pooling**
（`last_hidden_state[:, 0]`）+ L2 归一化，检索用点积。文本构造依据数据分析结论
（`data/analysis/stats.json`）：title + features(≤4) + categories；**剔除 description**
（空 47.8%）与 details（制造商标识噪声）——既降噪又显著缩短 CPU 编码耗时。

## 环境变量
| 变量 | 默认 | 说明 |
|---|---|---|
| `DEVICE` | `auto` | `auto` / `cpu` / `cuda`（重排/编码设备） |
| `QUERY_REWRITE_ENABLE` | `false` | 是否开启 LLM 查询改写（关闭时模板拼接，离线可跑） |
| `RERANKER_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | 交叉编码重排模型 |
| `BLAIR_OFFLINE_EMBEDDING_PATH` | `data/offline_blair_embeds.npy` | 离线商品向量路径 |
| `PRODUCT_CATALOG_PATH` | `data/catalog.jsonl` | 竞赛冻结产品目录 |
| `BLAIR_QUERY_ENCODER_MODEL` | `hyp1231/blair-roberta-large` | BLaIR 查询编码模型（与离线编码一致） |

## 运行演示
```bash
pip install -r retrieval_pipeline/requirements-pipeline.txt   # pydantic numpy rank-bm25（核心）
python retrieval_pipeline/test_pipeline.py
```
演示覆盖：普通硬过滤 / RECOVER（惩罚+同义词+变体+池100）/ override 清空约束。
FlagEmbedding 未安装时重排自动降级 fused 排序；npy 不存在时稠密通道自动禁用——全链路无付费 API 可跑。
