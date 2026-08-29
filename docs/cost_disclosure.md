# 成本 / 延迟披露（竞赛交付物）

> 说明：**token 用量仅为可行性指标，不计入 TechnicalScore**（官方评估器只统计，不参与评分）。
> 延迟为本地基准（RTX 3050 Laptop / 16GB RAM / Windows，公开集 200 会话实测），仅供参考。

## 1. 按模式延迟与 token 基准

| 模式 | 每轮延迟（ms/turn） | 每会话平均 token | 说明 |
|---|---|---|---|
| 离线规则（默认，bm25 + 规则重排） | ~170–330 | **0** | 纯 Python 标准库 + SQLite FTS5，零外部调用 |
| hybrid（+BLaIR 稠密，recover 门控） | ~180–330 | **0** | 稠密仅 recover 触发，公开集几乎不触发 |
| fingerprint_combo（+约束组合指纹） | ~300–330 | **0** | 指纹对满足全部约束的候选做全目录精确计数 |
| text_rerank（qwen3-rerank，需 key+网络） | ~350–600（含网络） | ~3,000–4,000/会话（估算） | 每轮对 Top-12 候选 rerank，无 key 自动回退 |
| reranker_model（RexReranker-0.6B / bge） | recover 时 ~1–10s | 0（本地推理） | 仅 recover 模式作第二意见；模型加载约 1.2–2.3GB |

延迟数据来源：`data/assets/env_config_lut.json`（40 会话冒烟）+ 全量 200 会话运行时间折算
（默认配置全量约 116s / 200 会话 / 1.77 轮 ≈ 330ms/轮）。

## 2. 在线模式成本估算

单位价格（公开参考价，按 token 计；实际以供应商为准）：

| 服务 | 输入 $/1M | 输出 $/1M |
|---|---|---|
| DeepSeek `deepseek-chat` | ~0.27 | ~1.10 |
| OpenAI `gpt-4o-mini` | ~0.15 | ~0.60 |
| qwen3-rerank（阿里云 MaaS） | 按调用量计 | — |

每会话估算（以 200 公开会话实测为基准外推）：
- **LLM 意图识别（级联）**：公开集规则高置信，LLM 极少触发（10 会话实测 777 prompt / 150 completion），
  全量约 **0.08–0.3 千 token/会话** → 成本约 **$0.00002–0.0001/会话**。
- **qwen3-rerank 文本重排**：每轮约 2 千 token（query + 12 候选文档），~1.8 轮 → **~3.5 千 token/会话**，
  成本取决于 MaaS 单价（量级 $0.001–0.01/会话）。
- 默认模式：**$0**（纯离线，零 API）。

## 3. 披露要点
- 模型：BLaIR `hyp1231/blair-roberta-large`（离线本地）、规则重排、可选 qwen3-rerank /
  RexReranker-0.6B / bge-reranker-v2-m3（本地）。
- 密钥：全部环境变量注入，代码/仓库不含任何 key（`DASHSCOPE_API_KEY` / `DEEPSEEK_API_KEY` /
  `OPENAI_API_KEY`）。
- 回退：任何 LLM/模型环节失败/超时/断网 → 自动回退规则，离线可跑。
- token 仅可行性指标：官方评估器将 `usage` 计入报告但**不计入 TechnicalScore**。
