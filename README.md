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
    （`utils/blair.py`），与全目录向量做点积召回。编码器（transformers）或离线 npy 任一缺失 → 自动回退 BM25。
  - 融合：Reciprocal Rank Fusion（RRF）。
- **重排序** `agent/reranker.py`：规则融合打分（约束覆盖度 0.5 / 品类 0.25 / RRF 0.15 / 热度 0.05 / 画像 0.05）
  + 可选的统一 LLM 语义重排（`LLM_PROVIDER=deepseek/openai`）；+ 可选本地 **bge-reranker-v2-m3**
  交叉编码精排（`RERANKER_MODEL_ENABLE=1` 且 FlagEmbedding 可用，失败自动回退规则排序）；
  无 selected key、`LLM_PROVIDER=none` 或 `LLM_RERANK=0` 时纯规则排序，完全离线可跑。

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
  交叉编码重排（FlagEmbedding 可导入 + bge-reranker-v2-m3 已缓存/可下载）、可选外网探测。
- **自主决策原则**：所有 LLM/重排能力开关**默认关**；配置开启 + 探测可用 → 启用；
  配置开启但环境不可用 → **自动回退规则**（意图识别/澄清/重排/稠密检索全部可回退）；
  `RETRIEVAL_BACKEND=auto` → 稠密可用用 hybrid，否则 bm25。
- **BLaIR 稠密通道的鲁棒性（环境自感知）**：
  - 离线 npy 缺失 / 维度不符 → `BlairEmbeddingStore.load` 返回 None → 稠密通道禁用；
  - 查询编码器加载失败 → 自动尝试 sentence-transformers 兜底，仍失败则禁用；
  - 任意异常都被 `_route_dense` 捕获，只影响稠密路由，不阻塞 BM25/类别/约束路由主流程。
- **LLM 意图识别**（`agent/llm_intent.py`）：LLM 判定 buying/browsing + 结构化槽位补充，
  严格 JSON 解析，失败/非法输出一律回退规则；**LLM 抽取的约束只作 soft 检索词**（防幻觉污染 hard 过滤）。
- **LLM 澄清决策**：LLM 决定 `ask_attribute` + 自然语言问题，非法属性回退规则策略。
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

### Intent generalization regression corpus

`tests/fixtures/intent/generalization.jsonl` is a hand-reviewed JSONL corpus.
Each row has the deterministic schema `{id, tags, message, state, expected}`:
`state` may declare `category`, `constraints` (with `attribute`, `value`, and
`strength`), and `recently_shown_asins`; `expected` stores literal dialogue-act
and operation labels. The offline regression test runs the rule recognizer only
and never contacts an external API:

```bash
.conda/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_intent_generalization.py tests/test_transition_sequences.py
```

When DeepSeek is configured, the opt-in live class can additionally check the
strict response schema and print aggregate schema-valid, destructive-precision,
and fallback rates. It makes no exact-output assertions and never stores raw
model responses:

```bash
RUN_LIVE_LLM=1 LLM_PROVIDER=deepseek LLM_INTENT_ENABLE=1 \
  .conda/bin/python -m pytest -q -p no:cacheprovider tests/test_intent_generalization.py
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

### 目录感知提问策略：分阶段启用与回退

新策略会从**同一次**宽召回候选池计算属性问题的信息价值；它不修改检索打分、重排序或官方四字段
响应协议。默认保持既有策略：`candidate_question_value.enabled=false`、`finish_strategy.enabled=false`、
`question_termination_mode=legacy`，因此不需要为常规离线评估设置任何新变量。

```bash
# 明确锁定为默认的、已验证的 legacy 行为（也适合快速回退）
LLM_PROVIDER=none \
SHOPPING_DECISION__CANDIDATE_QUESTION_VALUE__ENABLED=0 \
SHOPPING_DECISION__FINISH_STRATEGY__ENABLED=0 \
SHOPPING_DECISION__QUESTION_TERMINATION_MODE=legacy \
python run_local_eval.py
```

目录动态决策仍是实验性开关；只有显式设置以下变量才会启用。`explicit_only` 仅在用户明确表示没有更多偏好、
第 10 轮，或全部合法属性耗尽时停止提问；它不会把 `max_questions` 当作硬停止条件。

```bash
# 实验：候选分布驱动的问题选择 + explicit-only 终止（可离线运行）
LLM_PROVIDER=none \
SHOPPING_DECISION__CANDIDATE_QUESTION_VALUE__ENABLED=1 \
SHOPPING_DECISION__QUESTION_TERMINATION_MODE=explicit_only \
python run_local_eval.py

# 可选：把收尾价值和两步前瞻也纳入实验
SHOPPING_DECISION__CANDIDATE_QUESTION_VALUE__ENABLED=1 \
SHOPPING_DECISION__QUESTION_TERMINATION_MODE=explicit_only \
SHOPPING_DECISION__FINISH_STRATEGY__ENABLED=1 \
SHOPPING_DECISION__FINISH_STRATEGY__LOOKAHEAD_DEPTH=2 \
python run_local_eval.py
```

候选分析池独立于最终 Top10。默认是 300；启用动态策略后可覆写为 500 或 1000。运行时会取
`max(300, 覆写值)`，并把**同一批**候选同时交给动态问题计算和既有 reranker，绝不为问题分析发起第二次检索：

```bash
SHOPPING_DECISION__CANDIDATE_QUESTION_VALUE__ENABLED=1 \
SHOPPING_DECISION__QUESTION_TERMINATION_MODE=explicit_only \
SHOPPING_DECISION__CANDIDATE_QUESTION_VALUE__POOL_SIZE=500 \
python run_local_eval.py
```

仅在显式启用动态策略时，系统才会构建商品属性缓存和候选信号计算器；缓存/提取初始化或候选信号
计算异常都会退回静态目录信号和原有安全响应路径。两步前瞻也只会在启用收尾策略、仍有合法后续
提问、进入候选数或剩余问题预算的收尾阶段，并且已有一步收尾收益达到门槛时执行。空候选池或没有可用候选
则仍会产生合法的动态决策：没有可问属性时返回 `all_attributes_exhausted`，不再追问，并保持官方响应契约有效。
要完整回退到 legacy，使用上面的三项 legacy 设置即可。`transition_guard` 也是独立的安全开关，默认关闭，可用
`SHOPPING_DIALOGUE__TRANSITION_GUARD__ENABLED=1` 单独实验，不会自动随动态问题策略开启。

常用变量包括 `SHOPPING_DECISION__CANDIDATE_QUESTION_VALUE__PRIOR_ALPHA`、
`SHOPPING_DECISION__CANDIDATE_QUESTION_VALUE__PRIOR_TEMPERATURE`、
`SHOPPING_DECISION__CANDIDATE_QUESTION_VALUE__OTHER_ANSWER_PROBABILITY`、
`SHOPPING_DECISION__FINISH_STRATEGY__CANDIDATE_THRESHOLD`、
`SHOPPING_DECISION__FINISH_STRATEGY__REMAINING_QUESTION_THRESHOLD` 和
`SHOPPING_DECISION__FINISH_STRATEGY__LOOKAHEAD_DEPTH`；各权重也都可以按
`SHOPPING_DECISION__...__WEIGHTS__<NAME>` 覆写。`config/default.json` 中的所有数值都是可复现的搜索中心，
不是已经推广的比赛参数；只有经过公开集交叉验证及目录规模稳定性检查后才应考虑改变默认值。

### Catalog question-value diagnostic

The following offline diagnostic reads a JSONL catalog once, builds one immutable
attribute cache, and writes only aggregate measurements. It never changes runtime
policy defaults or `config/default.json`:

```bash
python -m experiments.catalog_question_value \
  --catalog /Users/zhengce/projects/participate_kit/catalog.jsonl \
  --output /private/tmp/catalog-question-value.json \
  --pool-sizes 300,500,1000 \
  --sample-count 1000 \
  --seed 20260829
```

Pool sizes must be positive, unique, and sorted; the command exits nonzero for an
invalid or malformed catalog. Output is atomically replaced only after complete,
valid JSON is written. Reports contain coverage, latency percentiles, stability,
and aggregate value metrics only—never ASINs, titles, descriptions, or other
product free text. Depth-two measurements are explicitly diagnostic and
non-promotion data because of the known depth-two gate mismatch; do not promote
any policy setting from this report.
