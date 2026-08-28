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
  - 稠密向量路由（可选，`RETRIEVAL_BACKEND=dense/hybrid`）：本地 sentence-transformers，未安装自动降级。
  - 融合：Reciprocal Rank Fusion（RRF）。
- **重排序** `agent/reranker.py`：规则融合打分（约束覆盖度 0.5 / 品类 0.25 / RRF 0.15 / 热度 0.05 / 画像 0.05）
  + 可选的统一 LLM 语义重排（`LLM_PROVIDER=deepseek/openai`）；无 selected key、`LLM_PROVIDER=none` 或 `LLM_RERANK=0` 时纯规则排序，完全离线可跑。

### 支柱 II｜对话理解与主动提问
- **级联意图识别** `agent/dialogue/recognizers/`：规则路径始终可用；`cascaded` 模式只在
  规则低置信度或输入复杂时调用共享 LLM 客户端。合法 LLM JSON 整体优先，任何失败都完整回退规则结果。
- **原子状态与商品反馈** `agent/dialogue/reducer.py`、`product_history.py`：所有状态变更统一经过
  复制、应用和校验；意图覆盖使用版本隔离，泛化拒绝软降权，明确商品拒绝硬排除。
- **确定性主动提问** `agent/dialogue/question_policy.py`：使用目录覆盖率、熵、当前约束缺口、
  用户可回答概率、歧义收益及提问成本计算效用；全部权重可配置，提问决策不调用 LLM。

### 支柱 III｜统一推荐上下文
- **单轮编排** `agent/dialogue/pipeline.py`：按顺序完成上轮展示结算、意图识别、状态归约、
  提问决策，并输出不可变的 `RecommendationContext`。
- **职责边界**：新对话子系统不召回、不排序、不去重，也不生成 Top10。现有
  `IntentRouter → HybridRetriever → Reranker` 推荐链消费上下文并保持推荐结果顺序。
- **可降级运行**：无 Key、provider 为 `none`、探测失败、请求超时或 LLM 输出非法时，主流程均
  使用本地规则继续工作；本地小模型仅保留 `IntentRecognizer` 接口扩展点，本次未实现加载与推理。

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
| `RETRIEVAL_BACKEND` | `bm25` | `bm25` / `dense` / `hybrid` |
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
| `SHOPPING_DIALOGUE__MODE` | `cascaded` | `rule_only` / `cascaded` 意图识别模式 |
| `SHOPPING_DIALOGUE__RULE_CONFIDENCE_THRESHOLD` | `0.75` | 规则结果达到该置信度时不调用 LLM |
| `SHOPPING_DIALOGUE__MAX_EVIDENCE_LENGTH` | `180` | 单条证据文本最大长度 |
| `SHOPPING_DECISION__MAX_QUESTIONS` | `3` | 单个意图版本的最大提问次数 |
| `EMBEDDING_MODEL` / `RERANKER_MODEL` | 见 `config/default.json` | 可选检索与重排模型 |
| `LLM_RERANK` | `1` | 设为 `0` 时强制使用确定性规则重排 |
| `SAMPLE_LIMIT` / `SKIP_DATA_VERIFY` / `OUTPUT_PATH` | 空 / `0` / `results.json` | 冒烟范围、数据校验和结果路径 |

每次 `run_local_eval.py` 启动都会在数据校验前打印经脱敏的 LLM 状态，包含 provider、model、state、attempts，以及可用时的错误类别：

- `disabled`：没有密钥或提供者为 `none`；不会构造 SDK 或访问网络。
- `available`：SDK 已准备就绪，且健康检查成功；若关闭健康检查，则不发送探测请求也可进入此状态。
- `unavailable`：探测失败但评估不会因它抛出异常；输出仅显示分类（如 `timeout`），不显示凭据。

统一客户端在启动后注入 Agent，并由级联意图识别器和 Reranker 共享。LLM 只解析输入或辅助重排，
不能直接决定是否提问、提问属性或生成推荐商品。离线用法保持不变：不设置 selected-provider key，
或设 `LLM_PROVIDER=none`，即可无网络运行。

### Dialogue modes and configurable question utility

`rule_only` 完全跳过意图识别网络调用；`cascaded` 是默认模式，仅在规则低置信度或存在复杂指代时
尝试 LLM，并对其 JSON 结果进行严格 Schema 和 ASIN 范围校验：

```bash
# 确定性离线识别
SHOPPING_DIALOGUE__MODE=rule_only LLM_PROVIDER=none python run_local_eval.py

# 级联识别；API Key 仍只从所选 provider 的环境变量读取
SHOPPING_DIALOGUE__MODE=cascaded LLM_PROVIDER=deepseek \
  DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" python run_local_eval.py
```

嵌套配置使用双下划线分隔。提问公式的七项权重均可通过同一规则覆盖，例如：

```bash
SHOPPING_DECISION__ASK_UTILITY__WEIGHTS__INFORMATION_GAIN=0.45 \
SHOPPING_DECISION__ASK_UTILITY__WEIGHTS__TURN_COST=0.25 \
SHOPPING_DECISION__ASK_UTILITY__MINIMUM_ASK_UTILITY=0.30 \
python run_local_eval.py
```

`StopUtility` 同样支持 `SHOPPING_DECISION__STOP_UTILITY__WEIGHTS__...` 与
`SHOPPING_DECISION__STOP_UTILITY__MINIMUM_STOP_UTILITY`。所有配置在 Agent 启动时加载、校验并冻结，
不进行运行时热更新。

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

Usage is split deliberately: startup health-check tokens accumulate on the shared client, while each Agent response reports only that turn's intent-recognition and reranking prompt/completion tokens. The former Reranker-owned OpenAI loader has been removed: create and initialize the client once at runner startup, then inject it into Agent, the recognizer, and Reranker.


## 4. 本地复现测试

```bash
# ① 完整本地评估（dev 模式，离线规则，默认 bm25）
python run_local_eval.py
# 输出总体 + 场景指标并写入 results.json

# ② 冒烟测试（前 10 个会话，快速验证）
SAMPLE_LIMIT=10 python run_local_eval.py        # Linux/macOS
$env:SAMPLE_LIMIT="10"; python run_local_eval.py  # Windows PowerShell

# ③ 切换检索后端 / 对话识别模式
RETRIEVAL_BACKEND=hybrid python run_local_eval.py
SHOPPING_DIALOGUE__MODE=rule_only python run_local_eval.py

# ④ 不含 key 的离线模式，以及 provider-specific 在线重排
LLM_PROVIDER=none python run_local_eval.py
LLM_PROVIDER=deepseek DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" python run_local_eval.py
LLM_PROVIDER=openai OPENAI_API_KEY="$OPENAI_API_KEY" python run_local_eval.py

# ⑤ 提交模拟模式（强制离线约束检查）
ENV_MODE=submit LLM_PROVIDER=none python run_local_eval.py

# ⑥ 官方单元测试（评估器未被修改，应全部通过）
python -m unittest discover tests -v

# ⑦ 开发期格式与基础静态检查（不会增加运行时依赖）
pip install -r requirements-dev.txt
ruff format --check agent/dialogue config tests
ruff check agent/dialogue config tests
```

---

## 5. 解决方案局限与迭代方向

### 已知局限
1. **公开集近似最优、私有集存在不确定性**：当前策略利用官方确定性模拟器的信息揭示机制，
   并以可配置效用函数选择 `ask_attribute`。若私有评测加入 paraphrasing 或策略微调，
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
- **未来工作：本地 LLM**：评估本地 Qwen 的加载与推理，用于生成更自然的澄清/推荐话术与解释；当前尚未实现。

---

## 6. 竞赛交付物适配（Devpost 要点）

- **模型与成本披露**：默认零 LLM、零 API 成本；已实现的可选在线重排 provider 为 DeepSeek / OpenAI（按 token 计费）；本地 Qwen 仅为未来方向。检索核心为 SQLite FTS5（无成本）。运行延迟：200 会话约 30s。
- **复现**：`python run_local_eval.py` 一条命令复现 0.995 HR@10（离线、无网络依赖）。
- **无密钥提交**：代码不含任何硬编码密钥，全部经环境变量注入。
- **数据合规**：仅使用竞赛冻结工具包；数据集加载带 SHA256 校验；不下载上游原始数据。
