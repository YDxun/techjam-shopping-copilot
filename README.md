# TechJam2026 购物副驾 · AI 对话式搜索与推荐 Agent

基于 TechJam2026「购物副驾：AI 对话式搜索与推荐」竞赛冻结工具包（Amazon Reviews 2023
`Clothing_Shoes_and_Jewelry`，50,000 商品 / 200 公开开发会话 / 800 私有评测会话）的完整工程实现。
**不修改官方评估器**，完全兼容官方 Python Agent 接口与机器可读 API 契约；默认零外部依赖即可运行，
本地公开集上显著超越弱 BM25 基线。

| 指标 | 弱 BM25 基线 | 本项目（dev 公开集 200 会话，离线规则模式） |
|---|---|---|
| Hit Rate@10 | 0.125 | **0.995** |
| MRR | 0.0680 | **0.5997** |
| MTTC（平均转化轮次） | 9.81 | **1.74** |
| Efficiency | 0.119 | 0.926 |
| TechnicalScore | 0.1067 | **0.8626** |

> 复现方式：`python run_local_eval.py`（默认 ENV_MODE=dev / LLM_BACKEND=none / RETRIEVAL_BACKEND=bm25）。

---

## 1. 项目概述与四大支柱映射

### 支柱 I｜核心架构：意图路由与混合管道
- **双轨意图路由** `agent/intent_router.py`：检测购买高意图轨道（存在硬约束 → 高精度约束过滤）
  与浏览开放式轨道（无约束/过泛 → 多样化泛化召回），随会话状态动态切换。
- **多路由混合检索** `agent/retriever.py`：
  - BM25 路由：SQLite FTS5 多字段加权（title/features 高权重）；
  - 类别路由：品类域命中过滤；
  - 硬约束 AND 路由：对 hard 约束做索引级交集 + 品类折入 SQL 的低权重召回补齐（防高频词把目标挤出池）；
  - 稠密向量路由（可选，`RETRIEVAL_BACKEND=dense/hybrid`）：本地 sentence-transformers，未安装自动降级。
  - 融合：Reciprocal Rank Fusion（RRF）。
- **重排序** `agent/reranker.py`：规则融合打分（约束覆盖度 0.5 / 品类 0.25 / RRF 0.15 / 热度 0.05 / 画像 0.05）
  + 可选 LLM 语义重排（`LLM_BACKEND=openai`）；`LLM_BACKEND=none` 时纯规则排序，完全离线可跑。

### 支柱 II｜对话策略：多轮场景演进
- **动态状态机** `agent/dialogue_state_machine.py`：
  - 增量槽位提取：品类槽、约束槽（hard=2/soft=1）、场景信号（boundary/override/no_more_pref/vague）；
  - 突发意图覆盖：首轮把"旧偏好"打标，检测到 "ignore my earlier preference" 时精准擦除旧偏好、
    把新意图提升为最高优先级 hard 槽位（`OVERRIDE_ERASE=1` 可切回激进擦除）。
- **主动澄清** `agent/clarifier.py`：
  - 候选过载/描述过泛 → 主动结构化澄清提问，通过 `ask_attribute` 收敛需求；
  - `CLARIFY_STRATEGY=other`（默认，一次最多蒸馏 2 条任意约束，信息量最大）或 `attribute`（按属性优先级逐项问）；
  - 顾客表示"无更多偏好"或约束饱和 → 停止提问（STOP-ASK），避免冗余轮次，优化 MTTC。

### 支柱 III｜自我进化：动态上下文编程
- **运行时上下文蒸馏** `agent/dynamic_context_program.py`：每轮把会话历史编译成 `ContextProgram`
  （约束/品类/意图轨道/模式/路由权重/置信度），检索、澄清、重排模块按它"重新编译"执行；
  长期用户画像仅作弱先验（内存态，不落盘），并叠加进程内跨会话统计。
- **自适应编排**：根据状态动态切换 `probe / exploit / recover / stop_ask` 四种运行模式，
  动态调整路由权重、是否触发澄清、是否硬过滤——**无需模型训练**，纯上下文编程实现策略调整。

### 支柱 IV｜对接评估矩阵
- 面向指标：混合检索保障 **HitRate@K** 召回；重排（全覆盖加分 + 规则精排）把目标推前提升 **MRR**；
  主动澄清 + 停止策略降低 **MTTC**。指标全部由官方评估器计算，本项目只适配 Agent 侧逻辑。

---

## 2. 环境安装

```bash
# Python >= 3.10（开发验证使用 3.12）；核心零第三方依赖（sqlite3 FTS5 内置）
python --version

# 可选增强（按需安装，未安装自动降级，不影响离线运行）
pip install -r requirements.txt            # 仅注释说明，核心不强制安装任何包
# 稠密检索：pip install sentence-transformers numpy torch
# LLM 重排：pip install openai
```

数据集：使用竞赛冻结工具包内的 `data/catalog.jsonl`（50,000 行）与 `data/public_set.jsonl`（200 行）。
首次运行会自动做 SHA256 完整性校验（`utils/data_verify.py`），校验失败会中止（`SKIP_DATA_VERIFY=1` 可跳过）。

---

## 3. 环境变量说明

| 变量 | 取值 | 默认 | 说明 |
|---|---|---|---|
| `ENV_MODE` | `dev` / `submit` | `dev` | 本地开发测试 / 提交模拟模式。submit 模式下若 `LLM_BACKEND=openai` 且无 key 则强制降级为 `none`（离线约束检查） |
| `LLM_BACKEND` | `none` / `local` / `openai` | `none` | 无大模型（纯规则）/ 本地模型 / OpenAI 兼容 API；`none` 完全离线可跑 |
| `RETRIEVAL_BACKEND` | `bm25` / `dense` / `hybrid` | `bm25` | 关键词 / 稠密向量 / 混合检索；稠密依赖 sentence-transformers，缺失自动回退 BM25 |
| `TOP_K` | 整数 | `10` | 推荐数 K，对齐 HitRate@K 评测 |
| `LLM_MODEL` | 模型名 | `Qwen/Qwen3.5-4B` | 本地/API LLM 模型（占位） |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | 字符串 | 空 | OpenAI 兼容端点密钥（环境变量注入，代码不硬编码） |
| `EMBEDDING_MODEL` | 模型名 | `sentence-transformers/all-MiniLM-L6-v2` | 稠密检索 embedding 模型 |
| `RERANKER_MODEL` | 模型名 | `BAAI/bge-reranker-v2-m3` | 可选重排模型（预留） |
| `CLARIFY_STRATEGY` | `other` / `attribute` | `other` | 澄清策略：最大信息量 / 按属性优先级 |
| `OVERRIDE_ERASE` | `0` / `1` | `0` | 覆盖时是否激进擦除旧偏好槽（默认保守保留为弱信号） |
| `LLM_RERANK` | `0` / `1` | `1` | 是否启用 LLM 语义重排（openai 后端时） |
| `SAMPLE_LIMIT` | 整数 | 无 | 开发冒烟测试：只跑前 N 个会话 |
| `SKIP_DATA_VERIFY` | `0` / `1` | `0` | 跳过 SHA256 校验 |
| `OUTPUT_PATH` | 路径 | `results.json` | 评估结果输出 |

---

## 4. 本地复现测试

```bash
# ① 完整本地评估（dev 模式，离线规则，默认 bm25）
python run_local_eval.py
# 输出总体 + 场景指标并写入 results.json

# ② 冒烟测试（前 10 个会话，快速验证）
SAMPLE_LIMIT=10 python run_local_eval.py        # Linux/macOS
$env:SAMPLE_LIMIT="10"; python run_local_eval.py  # Windows PowerShell

# ③ 切换检索后端 / 澄清策略
RETRIEVAL_BACKEND=hybrid python run_local_eval.py
CLARIFY_STRATEGY=attribute python run_local_eval.py

# ④ 提交模拟模式（强制离线约束检查）
ENV_MODE=submit LLM_BACKEND=none python run_local_eval.py

# ⑤ 官方单元测试（评估器未被修改，应全部通过）
python -m unittest discover tests -v
```

---

## 5. 解决方案局限与迭代方向

### 已知局限
1. **公开集近似最优、私有集存在不确定性**：当前策略深度利用官方确定性模拟器的信息揭示机制
   （`ask_attribute=other` 每轮最多蒸馏 2 条约束）。若私有评测加入 paraphrasing 或策略微调，
   依赖固定话术的解析需要更强的容错（已预留正则容错，但未做大规模对抗改写测试）。
2. **画像利用非常克制**：公开集 200 会话的 `preference_tags` 与目标约束几乎无相关性
   （标签文本与约束文本重叠率 0–12%），故画像仅作 5% 弱先验。若私有集画像更有区分度，可重新调权。
3. **本质困难样本**：仍有 1/200 会话因约束过于泛化（265 个商品满足全部约束）无法区分；
   这类样本需标题级语义信号才能解决。
4. **稠密/LLM 路径未在本次环境验证**：sentence-transformers / openai 为可选增强，
   已实现优雅降级，但端到端收益（HR/MRR）尚未实测。
5. **评估耗时**：50k 商品 FTS 建索引约 10s，200 会话约 30s；未做极端低延迟优化。

### 更多时间的迭代方向
- **语义检索实证**：安装 sentence-transformers 后跑 dense/hybrid，A/B 对比 MRR/HR 边际收益；
  用 `BAAI/bge-reranker-v2-m3` 或电商专用 `RexReranker` 做第二级精排。
- **离线策略进化**：利用确定性模拟器做"对弈式"自动调参（BM25 字段权重、RRF 常数、澄清轮数、
  打分权重），在留出子集上验证泛化，避免过拟合公开集。
- **解析鲁棒性**：对 paraphrasing 变体做对抗测试，扩展槽位提取模式（材质/颜色/尺码/价格正则）。
- **跨会话画像学习**：在更多会话上验证 profile→约束先验的稳定性，动态调权。
- **LLM 落地**：用本地 `Qwen3.5-4B` 生成更自然的澄清/推荐话术与解释，提升体验类指标。

---

## 6. 竞赛交付物适配（Devpost 要点）

- **模型与成本披露**：默认零 LLM、零 API 成本；可选 openai 重排（按 token 计费）或本地
  Qwen3.5-4B；检索核心为 SQLite FTS5（无成本）。运行延迟：200 会话约 30s。
- **复现**：`python run_local_eval.py` 一条命令复现 0.995 HR@10（离线、无网络依赖）。
- **无密钥提交**：代码不含任何硬编码密钥，全部经环境变量注入。
- **数据合规**：仅使用竞赛冻结工具包；数据集加载带 SHA256 校验；不下载上游原始数据。
