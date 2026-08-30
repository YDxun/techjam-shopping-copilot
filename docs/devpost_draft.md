# Devpost 项目描述（竞赛交付物）

## 一句话
**购物副驾（Shopping Copilot）**：一个可在 10 轮对话内、离线零成本运行的 AI 购物推荐 Agent，
通过"双轨意图路由 + 多路由混合检索（BM25+BLaIR）+ 动态状态机 + 运行时上下文编程"在
TechJam2026 公开集上取得 **HitRate@10 = 1.0 / MRR = 0.6703 / MTTC = 1.77 / TechnicalScore = 0.8857**
（弱 BM25 基线 0.125 / 0.068 / 9.81 / 0.15）。

## 问题
电商搜索中，用户需求往往模糊且多变（浏览 vs 购买、偏好临时推翻、对某些属性无偏好）。
传统单轮检索无法在有限轮次内定位隐藏目标商品。本赛题要求 Agent 在至多 10 轮内通过
"澄清提问 + 推荐排序"命中目标商品，按 HitRate@K / MRR / MTTC 综合评分。

## 方案（四大支柱）
1. **核心架构**：意图双轨路由（购买高精度/浏览多样化）+ 多路由混合检索
   （加权 BM25、类别过滤、硬约束 AND、BLaIR 稠密仅在 recover 启用，RRF 融合，`rrf_k=100`）
   + 规则重排（约束覆盖 + combo_bonus + 约束组合指纹，可选 qwen3-rerank / RexReranker）。
2. **多轮策略**：对话理解管线维护品类/约束槽与场景信号；override 语义（"ignore my earlier
   preference"）按 intent_version 版本化处理；候选过载主动澄清、无偏好停止提问以优化 MTTC。
3. **自我进化**：运行时把对话历史蒸馏为推荐上下文，动态切换 probe/exploit/recover 模式；
   能力探测 + 配置-环境-性能 LUT 让 Agent 按当前环境自动选择最优策略——无需模型训练。
4. **评估对齐**：混合检索保 HitRate；combo_bonus 与约束组合指纹（全目录精确计数）把目标
   推高保 MRR；澄清/停止策略降 MTTC。

## 模型与成本
- 默认**零 LLM、零 API 成本**，核心仅用 Python 标准库 + SQLite FTS5，完全离线可跑。
- 可选增强：BLaIR 稠密（离线 npy）、qwen3-rerank 文本重排（阿里云 MaaS）、
  RexReranker-0.6B / bge-reranker-v2-m3 本地交叉编码，全部经环境变量切换，缺失依赖自动降级。
- 密钥零硬编码，环境变量注入；语义重排 A/B 显示在该确定性评估器上作兜底重排会掉 MRR，故默认关。

## 复现
`python run_local_eval.py`（一条命令，200 公开会话，输出全部官方指标到 results.json）。
`python scripts/demo_session.py` 查看逐轮 Demo 会话。

## 数据合规
仅使用竞赛冻结工具包；数据集 SHA256 完整性校验；不下载上游原始 Amazon Reviews 数据。

## 局限与展望
- 深度利用确定性模拟器话术；私有集若引入改写，靠 hard-cue 升级 + 级联 LLM + review_paraphrase
  资产兜底（已预留）。
- 语义重排与确定性评估器机制不匹配（A/B 实证），保留为可选增强。
- 下一步：私有集难度分布重标定 LUT、对弈式自动调参、跨会话画像学习。
