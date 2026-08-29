# TechJam2026 购物副驾 · AI 对话式搜索与推荐 Agent

基于 TechJam2026「购物副驾：AI 对话式搜索与推荐」竞赛冻结工具包（Amazon Reviews 2023
`Clothing_Shoes_and_Jewelry`，50,000 商品 / 200 公开开发会话 / 800 私有评测会话）的完整工程实现。
**不修改官方评估器**，完全兼容官方 Python Agent 接口与机器可读 API 契约；默认零外部依赖即可运行，
本地公开集上显著超越弱 BM25 基线。

| 指标 | 弱 BM25 基线 | 本项目（dev 公开集 200 会话，离线规则模式） |
|---|---|---|
| Hit Rate@10 | 0.125 | **1.0** |
| MRR | 0.0680 | **0.6192** |
| MTTC（平均转化轮次） | 9.81 | **1.725** |
| Efficiency | 0.119 | 0.9275 |
| TechnicalScore | 0.1067 | **0.8713** |

> 复现方式：`python run_local_eval.py`（默认 ENV_MODE=dev / LLM_PROVIDER=deepseek；未配置 selected key 时离线 / RETRIEVAL_BACKEND=bm25）。

---

## 1. 项目概述与四大支柱映射

### 支柱 I｜核心架构：意图路由与混合管道
- **双轨意图路由** `agent/intent_router.py`：检测购买高意图轨道（存在硬约束 → 高精度约束过滤）
  与浏览开放式轨道（无约束/过泛 → 多样化泛化召回），随会话状态动态切换。
- **多路由混合检索** `agent/retriever.py`：
  - BM25 路由：SQLite FTS5 多字段加权（title/features 高权重）；
  - 类别路由：品类域命中过滤；
  - 硬约束 AND 路由：对 hard 约束做索引级交集 + 品类折入 SQL 的低权重召回补齐（防高频词把目标挤出池）；
  - **BLaIR 稠密向量路由**（`RETRIEVAL_BACKEND=dense/hybrid`，推荐）：商品向量由
    `scripts/encode_catalog_blair.py` **离线预计算**（`hyp1231/blair-roberta-large`，CLS pooling + L2，
    产物 `data/offline_blair_embeds.npy`，维度 1024）；推理阶段只编码用户查询文本
    （`utils/blair.py`），与全目录向量做点积召回。**模式自适应**：仅在 `recover`（连 miss 需扩召回）
    启用 + 硬约束覆盖回验 + 0.5 权重——probe/exploit 下语义候选会扰动已对齐的规则排序（A/B 验证）；
    编码器（transformers）或离线 npy 任一缺失 → 自动回退 BM25。
  - 融合：Reciprocal Rank Fusion（RRF）。
- **结构化过滤字段感知** `data/analysis/field_mapping.json`（`scripts/build_field_mapping.py` 生成）：
  属性→去哪找（lookup_fields+权重）+ 过滤严格度（tolerance/missing_policy）。material 查
  details.Material/features/title，budget 只查 price（79% 缺失→放行），brand 查 store（缺失放行）。
  `retrieval_pipeline` 通道1 已接入；reranker 经 A/B 证明保持全文本打分最优（纯字段/叠加会掉 MRR，
  详见"override 设计"一节）。
- **重排序** `agent/reranker.py`：规则融合打分（约束覆盖度 0.5 / **combo_bonus 0.10** /
  品类 0.25 / RRF 0.15 / 热度 0.05 / 画像 0.05）——combo_bonus 给"同时完整命中 ≥2 条披露约束"
  的商品 C(n,2)/C(N,2) 超线性加成（隐藏目标来自商品自身元数据，天然全命中，用它推高 MRR）
  + 可选 **qwen3-rerank 文本重排**（`LLM_RERANK=1` + `LLM_RERANK_BACKEND=text`，阿里云 MaaS
  `/reranks`，`DASHSCOPE_API_KEY` 注入；失败自动回退规则）；+ 可选本地**重排模型**
  （`RERANKER_MODEL_ENABLE=1`，按 `RERANKER_MODEL` 自动分发：`thebajajra/RexReranker-0.6B` /
  `Qwen/Qwen3-Reranker-*` 走生成式 yes/no 打分，`BAAI/bge-reranker-v2-m3` 走 FlagEmbedding
  交叉编码；失败自动回退规则排序）；默认 `LLM_RERANK=0` / `RERANKER_MODEL_ENABLE=0` 纯规则排序，完全离线可跑。

### 支柱 II｜对话策略：多轮场景演进（对话理解管线 `agent/dialogue/`）
- **识别层** `agent/dialogue/recognizers/`：级联意图识别（规则先行 + LLM 严格 JSON 兜底），
  产出 `DialogueAct`（new_search / add_constraint / replace_constraint / remove_constraint /
  reject_products / no_preference / no_more_preferences / ambiguous）与 `ConstraintOperation`
  （极性 include/exclude、强度 hard/soft、证据、置信度）。
- **状态归约** `agent/dialogue/reducer.py`：唯一允许产出新 `DialogueState` 的组件——增量槽位累积 +
  突发意图覆盖（REPLACE / NEW_SEARCH 清空旧约束并升级 `intent_version`；默认保守保留旧偏好为 soft
  弱信号，`OVERRIDE_ERASE=1` 切回激进擦除）。
- **主动澄清** `agent/dialogue/question_policy.py` + `catalog_signals.py`：目录感知提问效用打分与
  停止策略——`ask_other_first` 默认（"other" 一轮平均把候选池 4930→307、命中保持 0.99，
  数据验证最优）；候选过载/信息不足 → `ask_attribute` 收敛需求；效用饱和/顾客"无更多偏好" →
  停止提问（STOP-ASK），避免冗余轮次，优化 MTTC。
- **商品反馈闭环** `agent/dialogue/product_history.py`：版本化商品展示/反馈追踪
  （hard_rejected / soft_demoted → 下一轮检索排除/降权）。

### 支柱 III｜自我进化：动态上下文编程
- **运行时上下文蒸馏** `agent/dialogue/pipeline.py::_build_context`：每轮把 `DialogueState` +
  识别结果 + `ProductHistory` 编译成 `RecommendationContext`（约束 / 品类 / 意图轨道 /
  检索模式 probe-exploit-recover / 已问属性 / 排除与降权商品），下游
  `intent_router -> retriever -> reranker` 按它"重新编译"执行；长期用户画像仅作弱先验
  （内存态，不落盘）。
- **自适应编排**：根据状态动态计算 `probe / exploit / recover` 检索模式；`IntentRouter` 在
  RECOVER 下把 hard 组降级为 soft 放宽过滤；`RuntimeController` 依据能力探测自主决定
  LLM/稠密/重排是否启用——**无需模型训练**，纯上下文编程实现策略调整。

### 支柱 IV｜对接评估矩阵
- 面向指标：混合检索保障 **HitRate@K** 召回；重排（全覆盖加分 + 规则精排）把目标推前提升 **MRR**；
  主动澄清 + 停止策略降低 **MTTC**。指标全部由官方评估器计算，本项目只适配 Agent 侧逻辑。

---

## 2. 环境安装

```bash
# Python >= 3.10（开发验证使用 3.12）；核心离线模式使用 sqlite3 FTS5
python --version

# 安装项目声明的依赖，其中包含 DeepSeek 所需的 OpenAI Python SDK
pip install -r requirements.txt

# 可选的本地检索增强（未安装会自动降级）
# pip install sentence-transformers numpy torch
# BLaIR 稠密检索（推荐；离线编码 + 查询编码，CPU 可跑）：
#   pip install "transformers>=4.40" torch
#   python scripts/encode_catalog_blair.py          # 一次性预计算 50k 商品向量（CPU 约 6h）
# bge-reranker-v2-m3 本地安装配置（可选，交叉编码精排）：
#   pip install "FlagEmbedding>=1.3" "huggingface-hub>=0.20"
#   python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-reranker-v2-m3')"
```

数据集使用竞赛冻结工具包内的 `data/catalog.jsonl`（50,000 行）与 `data/public_set.jsonl`（200 行）。首次运行会自动做 SHA256 完整性校验（`utils/data_verify.py`）；`SKIP_DATA_VERIFY=1` 可跳过该校验。

## 3. 统一 LLM provider 配置与可用性

配置不再是直接解析环境变量。通用设置按以下四层由低到高合并：

1. 内置 `AppConfig` 默认值；
2. 受版本控制的非机密文件 `config/default.json`；
3. 环境变量；
4. 调用 `load_config(..., overrides=...)` 时传入的显式运行时覆盖。

默认 JSON 文件可由 `APP_CONFIG_PATH` 指向其他非机密配置文件；直接传给 `load_config(path=...)` 的路径优先于 `APP_CONFIG_PATH`。凭据只能来自 selected provider 的 `DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY`，不能写入 JSON 或显式覆盖，且不会出现在配置摘要、启动输出或错误文本中。

| 变量 | 默认 | 说明 |
|---|---:|---|
| `ENV_MODE` | `dev` | 本地开发或 `submit` 提交模拟；submit 模式强制离线约束 |
| `LLM_BACKEND` | 未设置 | 兼容映射：`none/local → none`，`openai → openai`；若设置 `LLM_PROVIDER` 则后者优先 |
| `RETRIEVAL_BACKEND` | `bm25` | `bm25` / `dense` / `hybrid` / `auto`（auto：稠密可用→hybrid，否则→bm25） |
| `TOP_K` | `10` | 推荐数量 K |
| `APP_CONFIG_PATH` | `config/default.json` | 非机密 JSON 配置文件的位置 |
| `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` | 空 | 仅 selected provider 的匹配 key 生效；未设置时不会发起网络请求 |
| `LLM_PROVIDER` | `deepseek` | 统一 provider：`none` / `deepseek` / `openai` |
| `DEEPSEEK_MODEL` / `DEEPSEEK_BASE_URL` | 见 JSON | DeepSeek profile 覆盖 |
| `OPENAI_MODEL` / `OPENAI_BASE_URL` | 见 JSON | OpenAI profile 覆盖 |
| `LLM_MODEL` / `LLM_BASE_URL` | 见 selected profile | 仅覆盖 selected provider 的 model / base URL |
| `LLM_HEALTH_CHECK_ENABLED` | `true` | 是否在启动时发送轻量可用性探测 |
| `LLM_CONNECT_TIMEOUT_SECONDS` / `LLM_TIMEOUT_SECONDS` | `3` / `8` | 连接超时 / 请求超时（秒） |
| `LLM_MAX_RETRIES` | `2` | 可重试探测的额外重试次数（启动探测最多 3 次请求） |
| `LLM_RETRY_BASE_DELAY_SECONDS` / `LLM_RETRY_MAX_DELAY_SECONDS` | `0.5` / `1.5` | 重试退避范围（秒） |
| `LLM_CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `2` | 运行期失败后打开断路器的阈值 |
| `LLM_INTENT_ENABLE` | `0` | 意图识别使用 LLM（默认关；探测到 LLM 可用才真正启用，失败回退规则） |
| `LLM_CLARIFY_ENABLE` | `0` | 澄清决策使用 LLM（默认关；同上） |
| `CAPABILITY_NETWORK_PROBE` | `0` | LLM 不可用时是否额外探测外网连通性（1=启用，httpx 2s） |
| `EMBEDDING_MODEL` / `RERANKER_MODEL` | 见 `config/default.json` | 可选检索与重排模型 |
| `CLARIFY_STRATEGY` / `OVERRIDE_ERASE` / `LLM_RERANK` | `other` / `0` / `1` | 对话策略；`LLM_RERANK=0` 强制确定性规则重排 |
| `SAMPLE_LIMIT` / `SKIP_DATA_VERIFY` / `OUTPUT_PATH` | 空 / `0` / `results.json` | 冒烟范围、数据校验和结果路径 |

每次 `run_local_eval.py` 启动都会在数据校验前打印经脱敏的 LLM 状态，包含 provider、model、state、attempts，以及可用时的错误类别：

- `disabled`：没有密钥或提供者为 `none`；不会构造 SDK 或访问网络。
- `available`：SDK 已准备就绪，且健康检查成功；若关闭健康检查，则不发送探测请求也可进入此状态。
- `unavailable`：探测失败但评估不会因它抛出异常；输出仅显示分类（如 `timeout`），不显示凭据。

统一客户端已在启动后注入 Agent，并仅由 Reranker 的可选语义重排使用；澄清仍是本地规则逻辑，并未实现 LLM 澄清。离线用法保持不变：不设置 selected-provider key，或设 `LLM_PROVIDER=none`，即可无网络运行。

### Provider profiles, capabilities, and selection

The JSON configuration contains independent DeepSeek and OpenAI-compatible profiles. Profiles are non-secret; keys are accepted only from the selected provider's environment variable.

~~~json
{
  "llm": {
    "provider": "deepseek",
    "rerank_enabled": true,
    "rerank_candidates": 12,
    "providers": {
      "deepseek": {
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "token_limit_parameter": "max_tokens",
        "supports_temperature": true
      },
      "openai": {
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "token_limit_parameter": "max_completion_tokens",
        "supports_temperature": true
      }
    }
  }
}
~~~

Set LLM_PROVIDER to "none", "deepseek", or "openai". It takes precedence over the legacy LLM_BACKEND mapping; legacy "none" and "local" map to "none", and legacy "openai" maps to "openai". Selected-provider credentials are DEEPSEEK_API_KEY and OPENAI_API_KEY respectively. DeepSeek and OpenAI profile defaults can be overridden independently with DEEPSEEK_MODEL / DEEPSEEK_BASE_URL and OPENAI_MODEL / OPENAI_BASE_URL; LLM_MODEL and LLM_BASE_URL override only the selected profile.

A profile's token_limit_parameter is either "max_tokens" or "max_completion_tokens", and supports_temperature controls whether a temperature field is sent. For example, a profile that requires max_completion_tokens and sets supports_temperature to false receives the former token parameter and no temperature field. This supports provider/model API differences without per-call provider branching.

One process selects exactly one provider and constructs one shared client. There is no automatic DeepSeek-to-OpenAI fallback, no OpenAI-to-DeepSeek fallback, and no local-model implementation in this change. Missing selected credentials leave the client offline.

### Startup, retry, and reranking

The runner loads EnvConfig, enforces submit-mode offline policy, initializes the selected client, verifies/loads data, then constructs Agent with that exact client before evaluation. It prints only sanitized provider, model, state, attempts, and error category. States are disabled (no selected key or provider "none", no SDK/network), available, and unavailable. Health-check failures use bounded retry/backoff (at most three startup attempts); runtime failures are retried according to the profile and open the configured circuit breaker after its threshold, preserving rule ordering.

When an available client is selected, Reranker sends at most 12 candidates by default. Its compact payload contains only active constraints plus each candidate's parent_asin, title, categories, and normalized features; it excludes the conversation history and user profile. A parseable mixed response is normalized locally: retain only submitted string ASINs, filter unknown IDs, keep the first occurrence of each duplicate, then append every omitted submitted candidate in deterministic rule order. Empty, malformed, or candidate-free output falls back to the complete deterministic rule order. Set LLM_RERANK=0 for a deterministic opt-out even with an available key.

Usage is split deliberately: startup health-check tokens accumulate on the shared client, while each Agent response reports only that turn's reranking prompt/completion tokens. The former Reranker-owned OpenAI loader has been removed: create and initialize the client once at runner startup, then inject it into Agent (and therefore Reranker).


### 环境自感知与自主决策（团队特色）

Agent 启动时执行一次**能力探测**（`agent/capability_probe.py`），随后由**自主决策控制器**
（`agent/runtime_controller.py`）根据探测结果 + 配置决定各环节执行方式，并在启动时打印：

```
[capability] device=cpu llm=deepseek:disabled dense=yes reranker=yes network=no
[decisions ] retrieval=bm25 intent=rule clarify=rule rerank=rule reranker_model=no
```

- **探测项**：设备（cuda/cpu）、LLM 可用性（真实健康检查，无 key 不发网络请求）、
  稠密检索（BLaIR 查询编码器 transformers/sentence-transformers 可导入 **且** 离线商品向量 npy 存在）、
  交叉编码重排（FlagEmbedding 可导入 + bge-reranker-v2-m3 已缓存/可下载）、
  **文本重排 qwen3-rerank**（`DASHSCOPE_API_KEY` + 国际版端点真实 /reranks 探测）、可选外网探测。
- **自主决策原则**：所有 LLM/重排能力开关**默认关**；配置开启 + 探测可用 → 启用；
  配置开启但环境不可用 → **自动回退规则**（意图识别/澄清/重排/稠密检索全部可回退）；
  `RETRIEVAL_BACKEND=auto` → 稠密可用用 hybrid，否则 bm25。
- **BLaIR 稠密通道的鲁棒性（环境自感知）**：
  - 离线 npy 缺失 / 维度不符 → `BlairEmbeddingStore.load` 返回 None → 稠密通道禁用；
  - 查询编码器加载失败 → 自动尝试 sentence-transformers 兜底，仍失败则禁用；
  - 任意异常都被 `_route_dense` 捕获，只影响稠密路由，不阻塞 BM25/类别/约束路由主流程。
- **LLM 意图识别**（`agent/dialogue/recognizers/llm.py`）：级联识别中规则低置信/歧义/替换约束时
  咨询 LLM（严格 JSON，失败/非法输出一律回退规则）；**LLM 抽取的约束只作 soft 检索词**（防幻觉污染 hard 过滤）。
- **澄清决策固定走规则策略**（`ask_other_first`，数据验证最优），不使用 LLM。
- 超时（connect 3s / total 8s）与熔断（2 次失败）由统一 LLM 客户端保证，LLM 失效不阻塞主流程。

## 4. 本地复现测试

```bash
# ① 完整本地评估（dev 模式，离线规则，默认 bm25）
python run_local_eval.py
# 输出总体 + 场景指标并写入 results.json

# ② 冒烟测试（前 10 个会话，快速验证）
SAMPLE_LIMIT=10 python run_local_eval.py        # Linux/macOS
$env:SAMPLE_LIMIT="10"; python run_local_eval.py  # Windows PowerShell

# ③ 切换检索后端 / 澄清策略
# 先离线预计算 BLaIR 商品向量（一次即可）：
#   python scripts/encode_catalog_blair.py --limit 2000   # 冒烟（验证维度/格式）
#   python scripts/encode_catalog_blair.py                # 全量 50k（CPU 约 6h，支持 --resume 断点续跑）
RETRIEVAL_BACKEND=auto python run_local_eval.py   # 稠密可用→hybrid，否则 bm25（环境自感知）
RETRIEVAL_BACKEND=hybrid python run_local_eval.py
CLARIFY_STRATEGY=attribute python run_local_eval.py

# ③b 启用 bge-reranker-v2-m3 交叉编码精排（实验性：A/B 显示纯语义重排会掉分，默认关闭）
RERANKER_MODEL_ENABLE=1 python run_local_eval.py
# ③c 独立检索管线演示（第4-6步完整链路：BLaIR 稠密 + 加权 BM25 + RRF + bge 重排）：
python retrieval_pipeline/test_pipeline.py

# ④ 不含 key 的离线模式，以及 provider-specific 在线重排
LLM_PROVIDER=none python run_local_eval.py
LLM_PROVIDER=deepseek DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" python run_local_eval.py
LLM_PROVIDER=openai OPENAI_API_KEY="$OPENAI_API_KEY" python run_local_eval.py

# ⑤ 提交模拟模式（强制离线约束检查）
ENV_MODE=submit LLM_PROVIDER=none python run_local_eval.py

# ⑥ 官方单元测试（评估器未被修改，应全部通过）
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
4. **稠密/重排 A/B 结论（已实测，全量 50k BLaIR 向量 + GPU 编码）**：
   - `RETRIEVAL_BACKEND=hybrid`（BM25+BLaIR 稠密）：**0.99 HR / 0.6115 MRR / 0.8622 TS**，
     MRR 提升 +0.012，HR 微降 1 例——稠密通道把命中商品推得更靠前（MRR 收益），
     但极少数高频约束场景下 RRF 融合会把目标挤出 Top-10（HR 代价）；
   - `hybrid + RERANKER_MODEL_ENABLE=1`（bge-reranker-v2-m3 精排）：**0.92 HR / 0.5163 MRR / 0.7798 TS**，
     显著下降——bge 交叉编码按纯语义相关性重排，与官方确定性评估器的信息揭示机制不匹配，
     会推掉规则排序选中的目标商品，故**默认关闭**，仅作实验性开关保留。
5. **评估耗时**：50k 商品 FTS 建索引约 10s，200 会话约 30s；未做极端低延迟优化。

### 更多时间的迭代方向
- **bge 重排改进**：当前 bge 直接覆盖规则排序会掉分（纯语义与评估器机制不匹配）。
  可改为「bge 仅在规则分接近的候选内部做 tie-break / 或仅对浏览轨道生效」，
  或对 bge 打分与规则分做加权融合（如 0.3×bge + 0.7×rule），避免覆盖已对齐的规则信号。
- **稠密路由调优**：hybrid 的 HR 微降源于 RRF 融合把目标挤出 Top-10，可降低稠密通道 α 权重
  或对稠密候选做约束覆盖回验后再融合。
- **BLaIR 文本模板调优**：当前剔除 description/details（数据分析结论），可 A/B 加入带权描述字段
  观察语义召回变化；也可尝试 `hyp1231/blair-roberta-base`（4 倍速，768 维）做速度/质量权衡。
- **离线策略进化**：利用确定性模拟器做"对弈式"自动调参（BM25 字段权重、RRF 常数、澄清轮数、
  打分权重），在留出子集上验证泛化，避免过拟合公开集。
- **解析鲁棒性**：对 paraphrasing 变体做对抗测试，扩展槽位提取模式（材质/颜色/尺码/价格正则）。
- **跨会话画像学习**：在更多会话上验证 profile→约束先验的稳定性，动态调权。
- **未来工作：本地 LLM**：评估本地 Qwen 的加载与推理，用于生成更自然的澄清/推荐话术与解释；当前尚未实现。

---

## 6. 竞赛交付物适配（Devpost 要点）

- **模型与成本披露**：默认零 LLM、零 API 成本；已实现的可选在线重排 provider 为 DeepSeek / OpenAI（按 token 计费）；本地 Qwen 仅为未来方向。检索核心为 SQLite FTS5（无成本）。运行延迟：200 会话约 30s。
- **复现**：`python run_local_eval.py` 一条命令复现 0.995 HR@10（离线、无网络依赖）。
- **无密钥提交**：代码不含任何硬编码密钥，全部经环境变量注入。
- **数据合规**：仅使用竞赛冻结工具包；数据集加载带 SHA256 校验；不下载上游原始数据。

---

## LLM 接入与自动化控制（DeepSeek 实测 A/B）

LLM 全部通过**环境变量 + 能力探测 + 运行时控制器**启用，默认**全关（离线规则为主）**，
不可用时自动回退规则，保证评分环境断网/CPU 也能完整跑完：

- `LLM_PROVIDER=deepseek` + `DEEPSEEK_API_KEY`：客户端健康检查通过后 `state=available`；
- `LLM_INTENT_ENABLE=1`：对话管线级联识别（规则先行，仅低置信/歧义/替换约束时咨询 LLM，失败回退规则）；
- `LLM_RERANK=1`：Reranker 用统一客户端对 Top-12 候选做语义重排（失败回退规则排序）；
- 澄清决策始终用规则策略（`ask_other_first`，"other" 一轮平均把候选池 4930→307、命中保持 0.99，数据验证最优）。

### 同 10 会话实测（deepseek-chat，bm25）

| 配置 | HR@10 | MRR | MTTC | TS | token(p/c) | 说明 |
|---|---|---|---|---|---|---|
| 纯规则（默认） | 1.0 | 0.6875 | 2.1 | 0.8843 | 0/0 | 离线、零成本 |
| + LLM 意图 | 1.0 | 0.6875 | 2.1 | 0.8843 | 777/150 | 指标不变（规则已高置信，LLM 极少触发） |
| + LLM 语义排序 | 1.0 | **0.5017** | 2.0 | 0.8305 | 50687/2242 | **显著掉 MRR**（与 bge 重排 A/B 一致） |

结论：LLM 语义排序与确定性评估器的信息揭示机制不匹配，**默认关闭**（`LLM_RERANK=0`、
`reranker_model_enabled=false`）；LLM 意图识别作为安全增强保留（默认关，可开）。

### 披露（赛题要求）

- 模型：DeepSeek `deepseek-chat`（可选 OpenAI 兼容端点）；本地 BLaIR + bge-reranker 纯离线。
- 成本：默认 0；开启 LLM 后按 token 计费（上面 10 会话全开约 5.1 万 prompt token）。
- 延迟：DeepSeek 单次 chat 约 0.6s（健康检查 ~3s）；全开 10 会话约 148s，纯规则约 60s。
- 回退：任何 LLM 环节失败/超时/断网 → 自动回退规则，离线可用。
- 密钥：仅环境变量注入，代码/仓库不含任何 key（`DEEPSEEK_API_KEY`/`OPENAI_API_KEY`）。

## 数据优化资产融合（data/assets/，队友打包）

融合队友"数据处理与检索资产优化"的 4 个运行时资产（离线静态，`utils/data_assets.py` 惰性加载、缺失容错）：

- `data/assets/vocab_v2_clean.json`：精修词表（canonical + synonyms，去除 68 个噪声词）
- `data/assets/category_mapping.json`：品类路由（audience/family 别名 -> 商品类型 token）
- `data/assets/review_paraphrases.json`：评论改写语言（size_fit / material / color，含否定规则）
- `data/assets/field_mapping.json`：属性 -> 检索字段/权重/匹配策略（预留，当前主打分未用）

生成脚本收入 `scripts/data_assets/`（可复现）；原始打包目录 `new_data_porcess/` 不入库。

### 环境开关（默认值已按 public 200 A/B 设定）

| 变量 | 默认 | 说明 | A/B 结论（public 200, bm25, 离线） |
|---|---|---|---|
| `ASSET_CATEGORY_EXPAND` | `1` | 品类映射 token 扩展（首轮路由） | 单独开无变化；与 paraphrase 协同 MRR +0.010 |
| `ASSET_PARAPHRASE` | `1` | 评论改写软约束抽取（私有集鲁棒，含否定保护） | 公开集无变化（模板消息），私有集改写鲁棒 |
| `ASSET_VOCAB_EXPAND` | `0` | 用 vocab_v2_clean 替换 data/analysis/vocab.json 做同义词扩展 | MRR 0.6335→0.6298（略降），默认关 |
| `ASSET_FIELD_MAP` | `0` | field_mapping 字段感知匹配（预留） | 未启用 |

实测（默认配置，HR@10=1.0 / MRR 0.6438 / MTTC 1.72 / TS 0.8787）相比接入前（MRR 0.6335 / TS 0.8757）：MRR +0.010，TS +0.003。

## 对话理解管线（队友融合模块，agent/dialogue/）

合入队友的对话理解子系统，与 BLaIR 检索/重排管线共存：

- `agent/dialogue/models.py`：不可变 `DialogueState`（intent_version / 极性 / 强度 / 无偏好属性）
- `agent/dialogue/recognizers/`：级联意图识别（规则先行 + LLM 严格 JSON + 回退）
- `agent/dialogue/reducer.py`：原子状态归约（REPLACE/NEW_SEARCH 清空约束并升级 intent_version）
- `agent/dialogue/question_policy.py` + `catalog_signals.py`：目录感知问题效用打分与停止策略
- `agent/dialogue/product_history.py`：版本化商品展示/反馈追踪（hard_rejected / soft_demoted）
- `agent/dialogue/pipeline.py`：`DialogueUnderstandingPipeline` 产出 `RecommendationContext`，
  下游 `intent_router -> retriever(BLaIR) -> reranker(规则+bge/LLM)` 负责 Top10。

决策配置见 `config/default.json` 的 `dialogue_understanding` 与 `decision` 段。


---

## field_mapping 静态表 + override 设计（数据驱动 A/B）

### field_mapping.json（属性 → 去哪找 + 多严）
`scripts/build_field_mapping.py` 用全量 50k 商品按字段覆盖统计 + vocab 同义词生成
`data/analysis/field_mapping.json`（含中间统计 `field_mapping_raw.json`）：

| 属性 | 权威字段 | 主要 lookup | tolerance / missing |
|---|---|---|---|
| material | details.Material | features(0.9) / title(0.45) / description(0.3) | strict / unmet |
| color | details.Color | title / features(0.65) | strict / unmet |
| size | details.Size | title(0.8) / features(0.7) | lenient / soft_unmet |
| style | details.Style | features(0.85) / title(0.5) / categories(0.45) | lenient / soft_unmet |
| use_case | details.UseCase | features(0.75) / title(0.55) / categories(0.4) | lenient / soft_unmet |
| category | categories | title(0.85) | strict / unmet |
| brand | store | details.Brand(0.8) / title(0.5) | lenient / **pass**（store 脏数据放行） |
| budget | price | — | lenient / **pass**（79% 无价放行，数值检查） |
| feature | features | title(0.6) / description(0.5) | lenient / soft_unmet |

统计要点：material 最强在 features（0.906 命中率）、size 在 title（0.808）、style/use_case 在 features+title；
vocab 反向统计会混入商品词（material 的 "shoes"/"women"、style 的 "jewelry"），生成时用非属性词黑名单剔除。

### override（意图覆盖）设计 —— 按赛题语义 + 数据验证
官方评估器 override：`old_value = soft_preferences[-1]`（目标商品自己的软偏好文本）、
`new_value = hard_constraints[0]`（目标商品的硬约束）；用户第 3/4 轮说
"Actually, ignore my earlier preference. What I need is: {new_value}."。

**数据**（30 个 override 会话）：28/30 目标商品**同时含旧值和新值文本**（旧值就是从目标自身文本抽的）；
旧值多为噪音短语（"Buckle closure"、"Date First Available:…"）。

**设计结论（全部 A/B 验证）**：
1. **旧约束保守保留为 soft 全权重**：目标同时含新旧值 → 旧约束原始 token 是"复合信号"，
   从检索/覆盖打分里剔除会掉 MRR（全 demoted 版 override MRR 0.769→0.733，已回退）。
2. **同义词扩展按 intent_version 门控**：仅 version==1（未 override）时做 vocab 同义词扩展
   （browsing/buying 收益 MRR +0.01~0.03）；override 后（version>=2）停止扩展，
   避免旧噪音短语的同义词污染新查询（override MRR 保 0.769）。
3. **override 后强制 exploit 模式**：version>=2 且有 hard 时切 exploit，让"新旧全覆盖"的目标
   拿加成推前（override MRR 0.769→0.772，MTTC 1.725→1.715）。
4. reranker 覆盖打分**保持全文本短语/token 逻辑**：field_mapping 纯字段/叠加打分均被 A/B 否决
   （MRR 0.619→0.597/0.610），故仅保留 budget→price 数值分支（防未来 budget 提取）。

**最终离线评估**：1.0 HR / 0.6335 MRR / 1.715 MTTC / 0.8757 TS（基线 0.995/0.619/1.74/0.8626，全提升）。


---

## qwen3-rerank 文本重排接入（替换 LLM 语义重排分支）

`LLM_RERANK_BACKEND=text`（默认）时，重排分支走阿里云 MaaS **qwen3-rerank**（`/reranks`，
国际版端点 `https://dashscope-intl.aliyuncs.com/compatible-api/v1`，无需 workspace ID；
`DASHSCOPE_API_KEY` 环境变量注入，代码不含 key）。旧 chat JSON 打分保留为 `LLM_RERANK_BACKEND=chat`。
自动化控制：capability_probe 真实探测可用性（`text_rerank=yes/no`）→ runtime_controller 决策
（`rerank=qwen3/llm/rule`）→ 失败自动回退规则排序。

### 真实 API 全量 200 会话 A/B
| 配置 | HR@10 | MRR | MTTC | TS |
|---|---|---|---|---|
| 规则（默认） | 1.0 | **0.6335** | **1.715** | **0.8757** |
| **qwen3-rerank 重排** | 1.0 | 0.5011 | 1.76 | 0.8351 |

结论：作为**兜底最终重排器**，qwen3-rerank 与 bge（0.5163）、LLM chat（0.5017）三方一致的
结论——纯语义重排与官方确定性评估器的信息揭示机制不匹配，会把目标商品从靠前位置挤下去
（HR 不降但 MRR 掉）。因此**默认 `LLM_RERANK=0`**，由自动化控制按环境决定是否启用；
若要启用：`LLM_RERANK=1`（可选 `LLM_RERANK_BACKEND=text|chat|auto`）+
`DASHSCOPE_API_KEY` 注入。


---

## RexReranker-0.6B 电商重排模型部署（与 bge-reranker-v2-m3 A/B）

`RERANKER_MODEL=thebajajra/RexReranker-0.6B` 时，重排分支走**电商领域增强的生成式重排器**
（Qwen3-Reranker-0.6B 微调，633 万条电商 query-product 数据训练，Apache-2.0，本地 ~1.2GB）：
`utils/rex_reranker.py` 按模型卡实现——chat 模板 + assistant 后缀，取最后 token 的
"yes"/"no" logit，`score = exp(yes)/(exp(yes)+exp(no))`（GPU 批处理 ~22 条/秒）。
`agent/reranker.py` 按模型名自动分发（Rex/Qwen3-Reranker→transformers，bge→FlagEmbedding），
capability_probe 按模型类型探测可用性，失败一律回退规则排序。

### 真实全量 200 会话 A/B（RERANKER_MODEL_ENABLE=1）
| 重排模型 | HR@10 | MRR | MTTC | TS | 耗时 |
|---|---|---|---|---|---|
| 规则（默认不启用） | 1.0 | **0.6335** | **1.715** | **0.8757** | ~30s |
| **bge-reranker-v2-m3** | 0.92 | **0.5163** | **2.755** | **0.7798** | ~30min |
| **RexReranker-0.6B** | **0.995** | 0.3154 | 3.105 | 0.7500 | ~56min |

结论：RexReranker HR 更高（0.995 vs 0.92）但 **MRR/TS 明显不如 bge**（0.315/0.750 vs 0.516/0.780），
且生成式模型更慢（56min vs 30min）——**保留 bge-reranker-v2-m3 为默认重排模型，不删除**；
两者均可经 `RERANKER_MODEL` 环境变量切换。语义重排整体仍低于纯规则基线（同 bge/qwen3 三方一致），
默认关闭，由自动化控制按环境决定。


---

## 默认策略：环境自适应最优（非永远纯规则）

`run_local_eval.py` 默认不再写死纯规则，而是由 **capability_probe（环境探测）+ runtime_controller（策略决策）**
自动选择当前环境最优配置（启动打印 `strategy=...`）：

| 环境特征 | 选中的默认策略 | 公开集 200 会话 |
|---|---|---|
| 任意（含队友数据资产默认开） | `strategy=bm25` 或 `hybrid`（auto 按 BLaIR 可用性选） | **1.0 HR / 0.6438 MRR / 1.72 MTTC / 0.8787 TS** |
| 无队友资产（纯规则） | `strategy=bm25` | 1.0 HR / 0.6335 MRR / 1.715 MTTC / 0.8757 TS |
| LLM 可用（key）且 `llm_intent_enabled=true`（默认） | 级联意图识别兜底（规则高置信不改变结果） | 同左（安全） |
| 重排模型（bge/Rex）开启 | 仅 `recover` 模式作第二意见精排 | 略降（默认关，可选） |

关键设计（全部公开集 A/B 验证）：
1. **0.8787 的提升主要来自队友数据资产**（category 扩展 / review paraphrase / refined vocab，
   `ASSET_*` 默认开）：bm25+资产 = hybrid+资产 = 0.8787（六位小数一致）。
2. **BLaIR dense 只在 recover 模式启用**（`retrieval_backend=auto` 时）：公开集上 recover 几乎不触发
   → dense 休眠、零损失（与 bm25 完全同分）；hard 约束回验 + 0.5 权重使其在私有集连 miss 时
   可作语义召回安全网，而不扰动公开集已对齐排序（全量启用 dense 会掉到 0.870）。
3. **重排模型（bge-reranker-v2-m3 / RexReranker）默认关**：全量/ recover 用法均略降
   （A/B：bge-recover 0.8759 < 0.8787），保留为 `RERANKER_MODEL_ENABLE=1` 可选，已门控 recover 变安全。
4. **LLM 意图识别默认开**（`llm_intent_enabled=true`）：级联（规则先行，低置信才咨询 LLM），
   无 key 环境自动回退规则——零成本兜底，公开集不变。


---

## combo_bonus：全约束命中超线性加成（提升 MRR）

**洞察**：隐藏目标商品来自真实购买记录，intent card 约束从该商品自身元数据生成 →
目标文本**同时满足全部披露约束**；而干扰商品往往"分散命中"（各满足一部分）。
当前覆盖度是逐条加权平均（线性信号），无法区分"1 个商品全命中"与"3 个商品各中一条"。

**设计**（`agent/reranker.py::_rule_score`）：
- 统计"完整命中"约束的加权数 `full_count`（hard=1.0、soft=0.5，`hit>=0.999` 才算完整）；
- `combo_norm = C(full_count,2) / C(full_denom,2)`——"同时满足的约束对"占比，≥2 条才触发，
  全命中=1.0，天然超线性；
- `score += W_COMBO × combo_norm`（默认 0.10，环境变量 `COMBO_BONUS_WEIGHT` 可调）。

**A/B（公开集 200 会话）**：
| 配置 | HR@10 | MRR | MTTC | TS |
|---|---|---|---|---|
| 无 combo（上一版默认） | 1.0 | 0.6438 | 1.72 | 0.8787 |
| **+ combo_bonus** | 1.0 | **0.6486** | 1.72 | **0.8802** |

- 增益集中在 **buying**（MRR +0.012，hard 约束强、目标全命中）；
- 权重 0.05~0.30 结果相同（饱和、稳健）；hard-only 计数无效（目标只有 1 条 hard 无法触发
  C(1,2)=0，**soft 计数是关键**——目标=1 hard + 多个 soft 全命中）。


---

## 约束组合指纹（全目录精确计数，默认关）

**洞察**：隐藏目标同时满足全部披露约束；"满足全部约束的商品数"（组合稀有度）是更强的置信信号——
count 越小，约束组合越能锁定目标。

**设计**（`agent/reranker.py`，不碰其它模块）：
- `_fingerprint(retriever, active)`：全目录**精确计数**同时满足全部活跃约束（hard+soft，
  `phrase_exists` 全命中标准）的商品数 count + 满足集合；按约束键缓存、惰性建索引（不重复扫全目录）；
- 分级加成（仅对"满足全部约束"的候选）：`count==1` 置顶（+1.0）/ `count≤10`（+0.5）/
  `count≤50`（+0.2）；`count>50`（约束过泛）不加成——**置信度门控**；
- 开关：`COMBO_FINGERPRINT_ENABLE=1` 开启，**默认关**；加成可调
  `COMBO_FINGERPRINT_BONUS_UNIQUE/TEN/FIFTY`。

**A/B（公开集 200 会话）**：
| 配置 | HR@10 | MRR | MTTC | TS |
|---|---|---|---|---|
| 指纹关（默认） | 1.0 | 0.6486 | 1.72 | 0.8802 |
| **指纹开** | 1.0 | **0.6520** | 1.72 | **0.8812** |

- 与 combo_bonus 叠加：MRR 0.6438 → 0.6520、TS 0.8787 → 0.8812（browsing 额外 +0.008）；
- 加成量级 0.5~1.5 结果相同（饱和、稳健——加成对交集成员等值，只抬升"全满足 vs 部分"边界）；
- 默认关（尊重"默认关"要求），开启后无公开集损失且提升。


---

## 对话/决策修复（P0 + P1）

### Part A（P0）hard 约束提取鲁棒性 + 级联触发放宽
- **必要性线索词表**（`agent/dialogue/recognizers/rule_based.py`，开关 `hard_cue_enabled` 默认 true）：
  must / need / needs / has to / have to / require / requires / important / crucial / essential / key /
  the most important thing。泛化 ADD 路径命中任一线索词 → 本次提取的约束升级为 **HARD**
  （含线索词后的取值捕获："I need waterproof" → feature·HARD；"The most important thing is cotton" → material·HARD）。
  分支优先级不变：override / no_preference / no_more / key_requirement / what_matters 等官方分支优先，
  线索词只作用于泛化 ADD 路径 → **官方模板行为不变**（公开集 0.8802 保持）。
- **级联触发放宽**（`cascade.py`）：除低置信/歧义/REPLACE 外，命中线索词或 turn≥2 出现新约束也咨询 LLM；
  LLM 失败/不可用仍回退规则（默认无 key 环境不触发，公开集零变化）。

### Part B（P1）模式切换阈值进配置
`pipeline.py::_build_context` 的硬编码 exploit 阈值提为配置：
`retrieval_mode.exploit_min_hard`（默认 2）、`retrieval_mode.exploit_min_constraints`（默认 4），
经 `config/models.py` + `loader.py` 读取（env：`RETRIEVAL_MODE__EXPLOIT_MIN_HARD` /
`RETRIEVAL_MODE__EXPLOIT_MIN_CONSTRAINTS`），行为默认不变。

验收：`python -m unittest discover tests` Ran 115 OK；pytest 135 passed；默认公开集 0.8802 保持。
