# TechJam2026 购物副驾 · 独立技术报告

> 数字以 `results.json` 为准：**HR@10 = 1.0 / MRR = 0.6703 / MTTC = 1.77 / TechnicalScore = 0.8857**
> （默认配置 `rrf_k=100` + 约束组合指纹；纯离线规则零 API）。

## 1. 架构（四支柱 + 对话理解管线）

```
官方 Agent 接口（reset/respond）
  └─ DialogueUnderstandingPipeline（agent/dialogue/）
       ├─ recognizers：级联意图识别（规则先行 + LLM 严格 JSON 兜底）
       ├─ reducer：原子状态归约（intent_version 版本化、override 处理）
       ├─ question_policy + catalog_signals：目录感知提问效用与停止策略
       └─ product_history：版本化商品展示/反馈闭环
  └─ IntentRouter（双轨 buying/browsing）
  └─ HybridRetriever（BM25 加权 + 类别 + 硬约束 AND + BLaIR dense-recover，RRF 融合）
  └─ Reranker（规则覆盖 + combo_bonus + 约束组合指纹 + 可选 qwen3-rerank / RexReranker）
  └─ RuntimeController（能力探测 → 环境自适应策略选择，LUT 驱动）
```

- **支柱 I 核心架构**：双轨意图路由（购买高精度硬过滤 / 浏览多样化召回）；多路由混合检索
  （FTS5 加权 BM25、类别域、硬约束 AND、BLaIR 稠密仅在 recover 模式启用）；规则/可选模型双保险重排。
- **支柱 II 多轮策略**：动态状态机增量槽位 + 突发意图覆盖（override 语义："ignore my earlier
  preference" 旧偏好保守保留为弱信号，`intent_version` 版本化）；候选过载主动澄清、无偏好停止提问。
- **支柱 III 自我进化**：每轮把对话历史蒸馏为 `RecommendationContext`，动态切换
  probe/exploit/recover 模式；能力探测 + LUT 按环境选最优策略——无需模型训练。
- **支柱 IV 评估对齐**：混合检索保 HitRate；combo_bonus + 约束组合指纹把目标推前保 MRR；
  澄清/停止策略降 MTTC。

## 2. 模型

| 模型 | 用途 | 依赖 | 默认 |
|---|---|---|---|
| SQLite FTS5（加权 BM25） | 词法召回 | 标准库 | ✅ |
| BLaIR `hyp1231/blair-roberta-large`（离线 npy） | 稠密语义召回（recover） | transformers + 离线 204MB npy | auto 启用 |
| 规则重排（覆盖 + combo + 指纹） | 精排 | 标准库 | ✅ |
| qwen3-rerank（阿里云 MaaS） | 文本重排（可选） | DASHSCOPE key + 网络 | ❌ 默认关 |
| RexReranker-0.6B / bge-reranker-v2-m3 | 交叉编码重排（可选） | 本地模型缓存 | ❌ 默认关 |

A/B 结论：语义重排（qwen3/bge/Rex）作为**兜底最终重排器**在该确定性评估器上会掉 MRR
（0.88→0.75~0.84），故默认关；BLaIR 仅在 recover 模式启用（公开集零损失、私有集安全网）；
`rrf_k=100` + 约束组合指纹是公开集稳健提升项。

## 3. 成本

见 `docs/cost_disclosure.md`：默认零 API 成本、纯离线；在线模式按 token 计费（可行性指标，
不计技术分）。延迟基准与每会话 token 估算均已披露。

## 4. 局限

1. 深度利用确定性模拟器话术；私有集若引入 paraphrase，靠 hard-cue 升级 + 级联 LLM + 队友
   review_paraphrase 资产兜底，但未做大规模对抗改写测试。
2. 画像在公开集信息量低（仅 5% 弱先验）；私有集画像若更有区分度可调权。
3. 语义重排与确定性评估器机制不匹配（A/B 实证），只能作可选增强。
4. 公开集 200 会话与私有 800 难度分布可能不同；LUT 基于公开集测量，私有集需重标定。

## 5. 团队贡献（5 人分工）

| 成员 | 分工 | 主要产出 |
|---|---|---|
| A · 数据 | 数据盘点/字典/提问价值 | `scripts/build_index.py`、`data/analysis/*`（vocab/field_mapping/question_value）、`data/assets/*`（category/review_paraphrase/refined vocab） |
| B · 对话 | 对话理解管线 | `agent/dialogue/`（recognizers/reducer/question_policy/product_history/pipeline） |
| C · 检索 | 检索/重排管线 | `agent/retriever.py`（BM25/类别/硬约束/BLaIR dense）、`agent/reranker.py`（规则+combo+指纹）、`retrieval_pipeline/`、`scripts/encode_catalog_blair.py` |
| D · 评测 | 评估对齐/调参/LUT | `run_local_eval.py`、`scripts/tune_*.py`、`data/assets/env_config_lut.json`、combo_bonus/指纹/override 设计的 A/B |
| E · 协调 | 集成/文档/交付 | 四支柱工程整合、`README.md`、`docs/*`、依赖/环境自感知统一 |
