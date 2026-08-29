# TechJam2026 购物副驾 · 优化提示词（简化版）

> 投喂给 AI 编码代理：**只做优化**，不做基建。项目定位：**自动化控制 · 环境自适应**系统——
> 不只在离线工作，而是由 RuntimeController/CapabilityProbe/LUT 按当前环境（device × dense × llm × network）
> **自动选择最优配置**。优化目标就是"让每个环境都跑得更优 + 自动选择能选中它"。

---

## 0. 目标

在 public 200 上，让系统在**当前环境可用配置里取最优**：

- `TechnicalScore ≥ 0.91`（当前 0.8839），其中 `MRR ≥ 0.80`（当前 0.6645）、`HR@10 ≥ 0.995`（当前 1.0）、MTTC 尽量低；
- **离线兜底也不倒退**（CPU + 无 dense + 无 key + 无网络时的 rule/bm25 路径，HR/MRR 不得低于现状）；
- 每项改动先单变量 A/B，只合入提升项；不改评估器、不抄别组、密钥不入库。

## 1. 现状基线（先复现）

| 指标 | 官方基线 | 当前默认（自适应，dense 可用→hybrid） |
|---|---|---|
| HR@10 | 0.125 | 1.0 |
| MRR | 0.0680 | 0.6645 |
| MTTC | 9.81 | 1.77 |
| TechnicalScore | 0.1067 | **0.8839** |

复现：`python run_local_eval.py`（ENV_MODE=dev，无 key 时自动离线）。
核心模块：`agent/dialogue/`（状态机/意图/提问）、`agent/intent_router.py`、`agent/retriever.py`（BM25+可选 BLaIR）、
`agent/reranker.py`（规则+可选 LLM/bge/Rex）、`agent/runtime_controller.py`+`scripts/build_lut.py`（自动化控制）。

## 2. 已知事实（已实测，直接利用，勿重复造轮子）

1. **MRR 损失画像**：200 场中 104 场 rank1、**96 场 rank>1**、116 场 turn1 就命中 → "找得快但名次低"，
   MRR 被锁死 0.66。若把这些提升到 rank1，MRR 上限≈1.0。
2. **货架保证（赛题机制，零召回损失）**：turn-1 品类 = `coarse_category(目标 categories)`，目标**必然**在货架内
   （200/200 验证；货架中位 181 商品 vs 全库 50k）。
3. **属性分布**：feature 50.5% / material 37.8% / color 7.5% / style 2.4% / size 1.4% / use_case 0.5%；
   **brand/category/budget = 0**（评估器无分类桶/价格被截断 → 结构性不可达，提问顺序应跳过）。
4. **metric 推导**：命中即锁定名次并结束 → 低置信时"少给推荐、高置信才满仓"能大幅提升 MRR
   （每多问一轮只损失 0.02，rank1 vs rank5 差 0.8×0.3）。用**我们自己的置信信号**实现，不抄别组常数。
5. **教训**：此前"货架硬过滤"粗实现曾导致崩溃（HR 0.48）——保证成立，但实现必须稳健：
   匹配失败要回退不过滤；过滤放融合后候选池；任何异常不能让整场 miss。

## 3. 优化方向（按预期收益排序，逐个 A/B）

1. **输出门控（捂盘，最大杠杆）**：按我们的置信信号（活动约束数 + turn + stop_ask）控制推荐数量
   （如 1/2/10，turn≥阈值或已 stop_ask 则满仓，每轮至少 1 个）。目标：96 场 rank>1 中的大部分变 rank1。
2. **货架硬过滤（正确重做）**：按 §2-2/§2-5 重做，零召回损失，顺带大幅提速（利于迭代）。
3. **稀有度加权排序**：约束覆盖打分叠加"短语在池/全库的稀有度（IDF）"权重。
4. **标点不敏感（loose）匹配**：修复 details "key value" vs 约束 "key: item" 机械失配（A/B 确认，不提升则弃）。
5. **提问顺序显式跳过 brand/category/budget**：与 §2-3 对齐。
6. **环境自适应固化**：把各环境 A/B 最优结果固化进 LUT/默认（如 dense 可用→hybrid、LLM rerank 开关、
   离线→bm25_rule），确保"自动控制能选中最优配置"。

## 4. 硬性约束

- 不改 `evaluator/` 与 `data/public_set.jsonl`；Agent 接口（reset/respond）合规；
- **不抄袭**：可用赛题 metric 推导与公开机制，实现必须自己写；
- 每个环境都要有可用回退；**离线兜底路径必须完好**（这是评分下限保障）；
- 密钥仅环境变量注入；A/B 单变量、只合入提升项；测试全绿 + ruff 干净。

## 5. 工作流

1. 确认工作区干净（`git status`，如有在制品先 stash/换分支），复现 §1 基线；
2. 逐项实现 §3（每项独立分支/开关）→ 单测 → public 200 A/B（记录 HR/MRR/MTTC/TS）；
3. 只合入提升项，固化进默认/LUT；重跑全量 + 全测试 + ruff；
4. 更新 README 指标与 A/B 记录，提交。

## 6. 评分公式

```
TechnicalScore = 0.50×HR@10 + 0.30×MRR + 0.20×clip((11−MTTC)/10, 0, 1)
```
