# 基于商品分布的意图门控与动态提问决策设计

日期：2026-08-29
状态：已完成讨论，待用户审阅
适用项目：TechJam Shopping Copilot

## 1. 目标

本设计优化两个相互独立但串联工作的模块：

1. 意图识别及状态变更强调泛化能力和破坏性操作安全性。
2. 提问决策优先优化官方评测表现，并直接利用 50,000 条商品目录及当前候选集合的分布特征。

核心目标是让系统回答两个问题：

- 当前用户表达是否有足够证据修改对话状态？
- 在当前候选商品中，询问哪个属性最可能在后续一轮快速缩小范围并命中目标？

本设计不修改 Top10 排序公式、检索算法、官方评测器或官方逐轮响应协议。

## 2. 已确认的原则

### 2.1 意图识别

- 规则识别器继续处理高确定性的表达。
- LLM 继续处理复合表达、指代、修正和歧义；触发 LLM 后仍以通过严格校验的 LLM 结果优先。
- LLM 输出必须通过 JSON、字段、枚举、evidence 和 ASIN 范围校验。
- `replace_constraint`、`remove_constraint`、`reject_products` 和 `no_more_preferences` 采用高精度优先策略。
- `add_constraint` 与 category 提取兼顾精确率和召回率。
- 意图泛化测试不只复制官方固定话术，应覆盖自然改写、组合约束、上下文指代、否定、修正和噪声输入。

### 2.2 决策优化

- 50,000 条商品目录用于属性提取、分布统计、当前候选效能估计和泛化正则。
- 200 条公开会话允许参与参数调节。
- 公开集采用嵌套交叉验证、商品全集正则、有限搜索和稳定性约束，降低小样本过拟合。
- 800 条隐藏会话不可访问，作为最终外部泛化验证。
- 运行时策略不能读取 `scenario_type`、ground truth 或隐藏 intent card。

## 3. 当前架构及问题

当前每轮执行顺序为：

```text
识别意图
→ 更新状态
→ 决定是否提问及提问属性
→ 生成检索上下文
→ 召回候选
→ 重排并返回 Top10
```

当前架构存在四个与本设计直接相关的问题：

1. QuestionPolicy 在检索之前运行，看不到当前候选商品。
2. CatalogQuestionSignals 只在启动时计算品类级覆盖率和熵，不能反映当前约束下的候选分布。
3. 默认 `ask_other_first=true` 会绕过 AskUtility 的大部分动态比较。
4. StateReducer 只验证结构和数值范围，不按操作风险判断置信度是否足够。

当前的“停止提问”还混合了用户主动结束、系统认为信息充分、问题收益低和达到人工问题数上限。官方评测器并不会因为 `ask_attribute=None` 而结束会话，因此比赛路径不应主动用效用阈值结束提问。

## 4. 目标架构

对话管线拆分为状态理解和检索后决策两个阶段：

```text
用户消息
→ CascadedIntentRecognizer
→ TransitionGuard
→ StateReducer
→ 临时 RecommendationContext
→ IntentRouter
→ HybridRetriever（一次宽召回）
→ CandidateQuestionSignals
→ QuestionPolicy
→ 记录 asked_attributes
→ 现有 Reranker
→ Top10 + 问题
```

检索仍只执行一次。同一批候选既用于动态问题价值分析，也用于现有重排。

建议将当前 `DialogueUnderstandingPipeline.process_turn()` 拆成语义明确的两个阶段：

- `interpret_turn(...)`：识别、门控、状态归并和临时上下文生成。
- `decide_question(...)`：接收候选信号，选择问题并记录提问状态。

两个阶段之间传递不可变的 PendingTurn 数据，最终提交后的 SessionState 仍保持不可变更新模式。

## 5. TransitionGuard

### 5.1 职责

TransitionGuard 位于 RecognitionResult 与 StateReducer 之间，只判断识别结果是否有足够依据改变状态，不解析自然语言，也不直接修改 DialogueState。

输出动作：

- `apply`：原样应用。
- `soften`：将低风险新增约束降为 soft 后应用。
- `clarify`：不修改状态，要求澄清对应属性。
- `reject`：识别结果与证据或当前状态冲突，保留原状态。

### 5.2 风险规则

- add：允许中等置信度；低于硬约束门槛时可以 soften。
- replace：要求高置信度、明确修正证据和可识别的新约束。
- remove：要求高置信度，并确认被移除约束存在。
- reject products：明确且属于最近展示集合的 ASIN 可硬拒绝；泛化否定只软降权。
- no preference：必须明确对应属性。
- no more preferences：要求最高置信度和明确结束表达。

### 5.3 配置与上线

TransitionGuard 必须支持统一配置和环境变量开关。关闭时 RecognitionResult 直接进入 StateReducer，完整保持当前语义；LLM 的基础结构校验始终有效，不受此开关影响。

开发阶段默认关闭，先比较门控关闭与开启的结果。只有在意图泛化、状态不变量和公开集交叉验证均通过后，才通过独立配置变更切换为默认开启。

配置结构：

```json
{
  "dialogue_understanding": {
    "transition_guard": {
      "enabled": false,
      "add_min_confidence": 0.65,
      "replace_min_confidence": 0.90,
      "remove_min_confidence": 0.90,
      "reject_products_min_confidence": 0.90,
      "no_preference_min_confidence": 0.85,
      "no_more_preferences_min_confidence": 0.95,
      "low_confidence_add_action": "soften",
      "destructive_failure_action": "clarify"
    }
  }
}
```

阈值是第一轮搜索空间的中心值，不直接视为最终比赛参数；最终值由已定义的交叉验证流程选择。

## 6. 商品属性缓存

### 6.1 接口

新增可替换接口：

```text
CatalogAttributeExtractor
├─ RuleVocabularyExtractor       默认实现
└─ LocalModelAttributeExtractor  未来扩展，不在本次范围
```

默认实现完全本地、确定性、不依赖 LLM。每个 ASIN 对应一个 AttributeProfile：

```text
AttributeProfile
├─ material: canonical value set
├─ color
├─ size
├─ style
├─ brand
├─ budget bucket
├─ feature
├─ use_case
├─ extraction confidence
└─ source fields
```

### 6.2 提取优先级

1. details 中明确的结构化字段。
2. title 和 features 中的受控词表匹配。
3. description 中的低权重匹配。
4. 无法可靠归一化的自由文本不参与候选划分。

同义表达必须归一化，例如 `100% cotton`、`cotton blend` 和 `soft cotton fabric` 均映射到 canonical material 值。feature 必须使用受控词表，避免完整句子形成伪高基数标签。

缺失值不能解释为商品不具有该属性。缓存携带商品数据哈希和词表版本，数据或词表变化后必须重建。

## 7. 动态候选问题信号

### 7.1 候选集合

QuestionPolicy 使用本轮宽召回候选池。候选分析池大小与最终 Top10 解耦，通过 `question_signal_pool_size` 配置，并在 300、500、1000 上比较稳定性、收益和延迟。

候选为空或动态统计失败时，回退现有品类级静态信号；不能影响主流程返回。

### 7.2 多值属性匹配

商品可能同时具有多个材质、风格或用途。对于候选商品 i 和属性 a：

```text
MatchSet(i, a)
= 与 i 至少共享一个 a 属性值的候选商品
+ a 属性缺失、不能被安全排除的候选商品
```

由此计算：

```text
ExpectedRemaining(a)
= Σ P(target=i) × |MatchSet(i, a)|

ExpectedShrink(a)
= 1 - ExpectedRemaining(a) / N

Resolve@K(a)
= Σ P(target=i) × I(|MatchSet(i, a)| <= K)
```

同时计算：

- Coverage
- Resolve@10、Resolve@3、Resolve@1
- P90Remaining
- WorstCaseRemaining
- MissingPenalty
- 与现有约束的冗余
- 属性提取置信度
- Top10 区分能力

### 7.3 混合目标先验

候选目标概率使用均匀先验与 RRF 分数的混合：

```text
P(target=i)
= (1 - alpha) × 1/N
+ alpha × Softmax(RRF_i / temperature)
```

alpha 为 0 时完全均匀，为 1 时完全依赖检索分数。alpha 和 temperature 参加公开集交叉验证，并受商品全集模拟退化约束。候选分数缺失、异常或全部相同时自动回退均匀先验。

## 8. 探索与收尾收益

问题效用拆成探索和收尾两部分：

```text
AskUtility(a)
= (1 - finish_pressure) × ExplorationGain(a)
+ finish_pressure × FinishGain(a)
+ 会话状态收益
- 各类惩罚
```

探索收益关注：

- 信息熵
- ExpectedShrink
- Coverage
- 与已有约束的互补性

收尾收益关注：

```text
FinishGain(a)
= w10 × Resolve@10
+ w3 × Resolve@3
+ w1 × Resolve@1
+ wp × TerminalProgress
- wr × P90Remaining
```

TerminalProgress 使用候选数到 Top10 的对数距离变化，保证没有属性能一步进入 Top10 时仍有连续梯度。

finish_pressure 由候选数距 Top10 的比例、候选收缩进度、剩余提问数和当前轮次共同决定。是否使用硬切换、具体阈值和权重不预设为最终值，必须通过商品全集实验和公开集交叉验证选择。

收尾阶段允许两步前瞻：对第一问题的各回答分支模拟下一最佳问题，并扣除额外轮次成本。探索阶段只做一步计算；剩余一轮时不做两步前瞻。

## 9. other 复合动作

无条件 `ask_other_first` 降为 legacy 基线。动态策略中，other 作为复合候选动作参与统一比较。

从当前未解决属性中选择联合收益最高的两个属性 a、b：

```text
Utility(other)
= OtherAnswerProbability × max CombinedGain(a, b)
- VaguenessPenalty
- RepeatPenalty
```

CombinedGain 根据 a、b 的联合属性匹配计算。OtherAnswerProbability 与模糊惩罚参加公开集交叉验证；联合分布来自商品目录。用户回答过一次 other 后提高重复惩罚。收尾阶段如果具体属性的 Resolve@10 更高，应优先具体属性。

## 10. 提问终止模式

只实现两种模式：

- `explicit_only`：比赛动态策略默认。
- `legacy`：完整保留当前 StopUtility、max_questions 硬停止和 ask_other_first 行为。

不实现 `utility` 模式，配置为该值时直接报错。

### 10.1 explicit_only

- 用户明确 `no_more_preferences` 时不再提问。
- 第 10 轮不提问，因为回答无法用于下一轮。
- 第 1 至 9 轮只要存在合法问题就继续选择问题。
- `no_preference(attribute)` 只排除该属性，不结束会话。
- `no_preference(other)` 转向具体属性，不视为会话结束。
- max_questions 只形成问题成本和重复惩罚，不作为硬停止。
- 候选池变小会提高 finish_pressure，不触发停止。
- 命中与会话终止由官方评测器判断。

收益均不大于零时的兜底顺序：

1. 选择收益最高且用户尚未声明无偏好的具体属性。
2. 具体属性均被拒绝时考虑未被拒绝的 other。
3. 所有属性均不可询问时返回不提问，reason code 为 `all_attributes_exhausted`。

## 11. 状态与失败处理

- StateReducer 仍是唯一能产生新 DialogueState 的组件。
- TransitionGuard 拒绝或澄清时，原状态必须保持完全不变。
- 有效 replace 才能递增 intent_version。
- 新意图版本继续隔离旧版本商品反馈。
- 动态候选统计失败只影响问题选择，不撤销已经通过门控的用户状态更新。
- 检索或重排异常继续由 Agent 现有兜底处理。
- 新增功能关闭时必须与当前 legacy 行为等价。

## 12. 本地诊断

新增本地逐轮 DialogueDecisionTrace，不改变官方 `respond()` 的四字段结果。轨迹记录：

- 识别来源、行为、置信度、歧义和回退原因
- TransitionGuard 动作与原因
- intent version 和约束差异
- 候选规模、分数分布和字段缺失率
- 各属性的 ExpectedShrink、Resolve@K、P90、探索收益、收尾收益和最终效用
- 最终属性、reason code、finish pressure 和 lookahead depth
- recommendation count 与 token usage

下一轮可补充上一轮预测候选数与实际候选数的差值。

诊断默认关闭，只在本地评测启用；session ID 只保存哈希，不记录 API key、完整 LLM 原始响应或默认用户原文。汇总进入本地评测结果，完整轨迹单独写文件并设置最大条数。

## 13. 测试结构

### 13.1 快速确定性测试

不联网、不加载完整目录，覆盖：

- 意图 JSON 契约、失败回退和 evidence 约束
- TransitionGuard 开关、动作和风险阈值
- 状态转移矩阵及不变量
- 小型人工商品集上的 ExpectedShrink、Resolve@K、混合先验和 other 联合收益
- explicit_only 与 legacy 行为
- 第 9 轮仍提问、第 10 轮不提问
- 新功能关闭时的行为等价性

### 13.2 商品全集实验

读取 50,000 条商品，不调用外部 LLM，覆盖：

- 属性缓存质量和数据缺失
- 不同品类和候选规模的最佳问题
- 300、500、1000 分析池的稳定性和延迟
- 一步与两步前瞻的收益
- 商品抽样扰动后的参数稳定性

### 13.3 公开集参数调节

使用 200 条公开会话进行嵌套 5 折交叉验证。外层按 scenario、初始候选规模和粗品类分层；同一目标商品不能跨折。内层选择配置。

第一阶段限制为少量全局参数，不设置每品类或每样例规则。粗搜索最多约 50 组，再对少数稳定配置局部细化。

选择目标：

```text
SelectionScore
= PublicCVTechnicalScore
- lambda1 × FoldVariance
- lambda2 × CatalogSimulationRegression
- lambda3 × ParameterComplexity
```

同时要求：

- HR@10 不出现明显退化
- 至少 4/5 个外层折不退化
- 各 scenario 不发生明显崩塌
- 商品全集的候选缩减和 Resolve@10 不退化
- 延迟满足预算

采用配对 Bootstrap 比较每条会话的 MRR、首次命中轮次和 hit/miss 变化，并使用一标准误差规则选择更简单的配置。

调决策参数时固定意图识别版本和结构化识别结果，避免真实 LLM 波动污染比较。真实 DeepSeek 测试保持 opt-in，并与快速测试分离。

## 14. 配置分组

新增配置应按职责分组：

```text
dialogue_understanding.transition_guard
decision.candidate_question_value
decision.finish_strategy
decision.question_termination_mode
diagnostics.decision_trace
```

关键开关支持环境变量覆盖。所有数值必须通过统一配置加载器验证范围和组合合法性。实验选择出的参数先写入独立实验配置；达到验收条件后再通过单独的配置变更提升为默认值。

## 15. 实现边界

本次包含：

- 两阶段对话编排
- TransitionGuard 及开关
- 确定性商品属性缓存
- 动态候选问题信号
- 混合候选先验
- 探索/收尾收益与可配置两步前瞻
- other 复合动作
- explicit_only 与 legacy 模式
- 本地决策轨迹
- 三档测试和交叉验证工具

本次不包含：

- 修改最终 Top10 排序公式
- 更换 BM25/BLaIR 检索算法
- 新增本地机器学习模型
- 使用 LLM 决定提问属性
- 修改官方评测器或官方响应协议
- 使用隐藏测试数据

## 16. 验收条件

- 新功能关闭时，现有快速测试和 legacy 行为全部通过。
- TransitionGuard 的破坏性操作不变量全部通过。
- 动态问题公式在人工商品分布上得到可手算验证的结果。
- 全目录实验能够生成稳定、可复现的问题效能报告。
- 公开集交叉验证完整记录折划分、配置和指标。
- 最终配置符合跨折、分场景、商品全集和延迟约束。
- 官方逐轮响应仍只包含 `message`、`ask_attribute`、`recommendations` 和 `usage`。
- 所有关键参数来自统一配置，秘密仍只从环境变量读取。
