# Legacy 主控与单次动态替换混合提问架构

日期：2026-08-30  
状态：用户已批准
适用项目：TechJam Shopping Copilot

## 1. 目标

在保留当前 Legacy 提问策略稳定性的前提下，引入候选商品分布信号，但只赋予动态策略一项有限权限：当 Legacy 准备重复询问 `other` 时，可将其替换为一个具体属性；每个会话最多替换一次。

本次工作比较两个版本：

1. 当前 Legacy 策略。
2. Legacy 主控、单次动态替换的 Hybrid 策略。

Hybrid 只做少量参数筛选，不恢复此前成本较高的完整动态参数搜索。

## 2. 官方协议与项目边界

### 2.1 必须保持

- Agent 接口保持 `reset(session_id, user_profile)` 与 `respond(session_id, user_message, turn, top_k) -> dict`。
- 每轮一次 `respond` 同时返回 `message`、`ask_attribute`、`recommendations` 和 `usage`；提问与 Top10 共占一轮。
- `ask_attribute` 只能是官方允许属性或 `None`。
- 推荐结果仍由现有检索和重排链生成；本设计不生成、不修改、不重排 Top10。
- 会话最多 10 轮；目录保持只读；不修改官方评测器和评分公式。
- 策略完全离线、确定性运行，不调用 LLM。
- 既有意图识别 LLM、统一客户端和规则回退路径保持不变。
- 运行时不得读取 `scenario_type`、ground truth、隐藏 intent card 或模拟器内部状态。
- 50,000 条目录可用于属性提取和候选分布计算；200 条公开会话只用于开发评测与参数筛选。

### 2.2 本次不做

- 不改变 BM25、dense、hybrid、RRF、候选召回和重排公式。
- 不改变意图识别、状态归约和 Transition Guard 的既有语义。
- 不启用完整动态策略的两步前瞻、finish pressure、动态停止或 `explicit_only` 终止模式。
- 不进行大规模参数搜索、模型训练或公开集标签驱动的运行时分流。
- 不根据这次小样本筛选自动修改默认生产配置。

## 3. 选择的架构

### 3.1 决策流

```text
Legacy 先产生完整决定
  ├─ Legacy 停止                 → 原样返回
  ├─ Legacy 询问具体属性         → 原样返回
  ├─ Legacy 首次询问 other       → 原样返回
  └─ Legacy 再次询问 other
       ├─ 已使用过动态替换        → 原样返回
       ├─ 候选信号不可用          → 原样返回
       └─ 具体属性通过全部门控
            → 用该属性替换 other
            → 标记本会话已使用替换
            → 下一轮继续由 Legacy 主控
```

先执行 Legacy 的原因是停止条件、最大提问数、轮次限制、无更多偏好和边界处理已在公开集上表现稳定。Hybrid 不拥有延长或缩短会话的权限，也不能覆盖 Guard 产生的澄清决定。

首次 `other` 保持不变，因为现有数据分析显示：它平均将候选池从约 4,931 缩至 308，命中保持率约 99%，并且官方模拟器最多可为 `other` 披露两条未披露约束。动态替换只处理后续重复 `other`，以限制对强基线的破坏面。

### 3.2 会话状态

新增一个内部、不可变的布尔状态，表示本会话是否已经使用 Hybrid 替换。它不进入官方响应，不影响推荐上下文，不在意图覆盖时重置；“每个 session 最多一次”按完整会话计算，而不是按 `intent_version` 计算。

只有实际提交了替换问题后才写入该状态。候选信号异常、门控失败、陈旧 pending turn 或响应失败均不得消耗替换机会。

### 3.3 合法候选属性

具体属性必须同时满足：

- 属于官方允许属性且不是 `other`。
- 尚未询问。
- 未被用户标记为无偏好。
- 当前状态尚未包含同属性约束，避免重复索取已知信息。
- 已知 category 时不再询问 category。
- 当前候选信号包含该属性，所有数值均为有限值。

同分时沿用项目现有属性顺序，保证可复现。

## 4. 动态收益与门控

每个合法具体属性使用同一固定收益公式：

```text
HybridGain(attribute) =
    w_shrink                * ExpectedShrink
  + w_resolve_at_10         * Resolve@10
  + w_coverage              * Coverage
  + w_answer_probability    * AnswerProbability
  + w_extraction_confidence * ExtractionConfidence
  - w_missing               * MissingRate
  - w_turn_cost             * TurnPressure
```

所有输入先限制到 `[0, 1]`。权重统一由配置管理，但本次轻量筛选固定权重，只调整门控阈值，避免小样本下同时搜索权重与阈值。

默认固定权重：

```json
{
  "expected_shrink": 0.40,
  "resolve_at_10": 0.25,
  "coverage": 0.15,
  "answer_probability": 0.10,
  "extraction_confidence": 0.10,
  "missing_penalty": 0.25,
  "turn_cost": 0.10
}
```

具体属性必须通过以下门控：

- `coverage >= minimum_coverage`
- `missing_rate <= maximum_missing_rate`
- `expected_shrink >= minimum_expected_shrink`
- `resolve_at_10 >= minimum_resolve_at_10`
- `HybridGain >= minimum_gain`

`other` 是模拟器中特殊的多约束问法，不能与单属性用完全相同的目录信号公平比较，因此本版不构造虚假的 `other` 分数差。门控直接判断一个具体问题是否具备足够的独立价值；未通过时保留 Legacy 的 `other`。

## 5. 配置设计

在统一 `decision` 配置下增加独立的 `hybrid_question_policy`：

```json
{
  "enabled": false,
  "max_replacements_per_session": 1,
  "only_after_other_asked": true,
  "pool_size": 300,
  "prior_alpha": 0.25,
  "prior_temperature": 1.0,
  "minimum_coverage": 0.60,
  "maximum_missing_rate": 0.40,
  "minimum_expected_shrink": 0.25,
  "minimum_resolve_at_10": 0.05,
  "minimum_gain": 0.25,
  "weights": {
    "expected_shrink": 0.40,
    "resolve_at_10": 0.25,
    "coverage": 0.15,
    "answer_probability": 0.10,
    "extraction_confidence": 0.10,
    "missing_penalty": 0.25,
    "turn_cost": 0.10
  }
}
```

约束：

- 默认 `enabled=false`，保持当前 Legacy 行为。
- `max_replacements_per_session` 本版只接受 `0` 或 `1`。
- `only_after_other_asked` 本版固定为 `true`；配置字段用于明确语义，不支持首问替换。
- `pool_size`、`prior_alpha` 和 `prior_temperature` 只控制 Hybrid 的候选信号生成，不隐式启用完整动态策略。
- 所有权重和阈值在启动时校验并冻结。
- Hybrid 启用时可以构建候选属性缓存；构建或计算失败必须回退 Legacy。
- 现有 `candidate_question_value.enabled` 的完整动态模式保持兼容，但 Hybrid 与完整动态模式不得同时启用。

生产 Agent 仍自行拥有并构建一份目录资源。轻量对比实验则必须显式构建一个只读资源包，包含商品快照、检索索引、全局品类信号和 `CatalogAttributeCache`，Legacy 与三组 Hybrid 顺序共享该资源包。共享范围只限不可变目录数据；每个版本必须新建自己的 `DialogueUnderstandingPipeline`、`QuestionPolicy`、状态容器、Reranker 和诊断计数，防止会话状态或上一配置的结果泄漏。

## 6. 诊断与错误处理

Hybrid 决策复用现有隐私安全 trace，只记录聚合数值和原因码，不记录商品标题、用户原文或目录文本。

新增的主要原因码：

- `hybrid_specific_replacement`
- `hybrid_first_other_preserved`
- `hybrid_replacement_already_used`
- `hybrid_no_eligible_attribute`
- `hybrid_threshold_not_met`
- `hybrid_signals_unavailable`

任何 Hybrid 配置、缓存或计算异常均不向 `respond()` 外抛出；系统返回已经计算好的 Legacy 决定。

## 7. 轻量对比实验

### 7.1 数据与规模

- 从 200 条公开会话中按固定种子和官方场景比例抽取 20 条：Buying 8、Browsing 8、Intent Override 3、Boundary 1。
- 所有版本使用完全相同的样本顺序、检索设置、规则意图模式和随机种子。
- 比较 Legacy 加三组 Hybrid 门控，共 80 个会话 rollout。
- 不调用外部 LLM，不运行大规模嵌套交叉验证。
- 商品目录、检索索引、全局目录信号和逐商品属性缓存只构建一次，三组 Hybrid 共享；各版本的会话状态完全独立。
- 关闭两步前瞻和无关诊断导出，只保留结果报告所需的聚合计数与耗时。
- 从实验进程启动开始设置 1,200 秒硬截止；截止时停止剩余 rollout，原子写出已完成配置、已完成样本数、超时位置和 `status="time_budget_exceeded"`，不得把不完整结果作为胜负结论。

### 7.2 三组门控

| 参数 | 保守 | 均衡 | 宽松 |
|---|---:|---:|---:|
| minimum_coverage | 0.70 | 0.60 | 0.50 |
| maximum_missing_rate | 0.30 | 0.40 | 0.50 |
| minimum_expected_shrink | 0.35 | 0.25 | 0.20 |
| minimum_resolve_at_10 | 0.10 | 0.05 | 0.00 |
| minimum_gain | 0.35 | 0.25 | 0.15 |

候选池固定为 300，其余权重固定为第 4 节默认值。

### 7.3 报告指标

- 官方 TechnicalScore、HitRate@10、MRR、MTTC、Efficiency。
- 各场景指标，但 Boundary 只有一条，仅作诊断。
- Hybrid 替换触发次数、触发率、所选属性和回退原因。
- 决策计算 p50/p95 延迟以及初始化耗时。

该实验是快速筛选，不提供可靠的统计显著性。只有在 HR@10 不下降且 TechnicalScore、MRR 或 MTTC 出现一致改善时，才建议进一步扩大验证；否则继续以 Legacy 为默认。

## 8. 测试边界

实施时只增加高价值测试：

1. Legacy 停止和非 `other` 决定不可被 Hybrid 覆盖。
2. 首次 `other` 必须保留。
3. 后续重复 `other` 通过门控时只替换一次。
4. 非法、重复、无偏好和已有约束属性不可选择。
5. 信号缺失或异常完整回退 Legacy。
6. 配置默认关闭、非法值拒绝、完整动态与 Hybrid 互斥。
7. 一条 Agent 主流程测试确认官方响应结构和 Top10 透传不变。

不扩展大规模话术矩阵，不修改评测器测试，不追求新的覆盖率目标。

## 9. 成功标准

- Hybrid 关闭时，Legacy 决策与现有基线逐项一致。
- Hybrid 不改变停止轮次，不改变首次 `other`，每个会话最多替换一次。
- 所有异常路径安全回退 Legacy。
- 官方响应协议、Top10 和 usage 结构保持兼容。
- 轻量实验可复现并清楚报告新架构是否值得扩大验证。
