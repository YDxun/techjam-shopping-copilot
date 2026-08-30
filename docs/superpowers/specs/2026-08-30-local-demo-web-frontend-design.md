# 本地比赛演示前端设计

日期：2026-08-30
状态：设计已确认，执行计划已形成

## 1. 目标

为现有 Shopping Copilot 增加一个本地浏览器演示界面，使用类似主流大模型产品的聊天交互展示多轮问答和商品推荐。

前端必须是现有系统之外的适配层，不改变：

- `Agent.reset(session_id, user_profile)`；
- `Agent.respond(session_id, user_message, turn, top_k)`；
- Agent 的意图识别、状态维护、提问、召回和重排逻辑；
- 官方 evaluator、评分规则和评测响应结构；
- 现有离线降级及可选 LLM 能力。

## 2. 范围

### 2.1 首版包含

- 单机本地运行，通过 `localhost` 访问；
- 英文聊天界面；
- 新建会话、连续多轮对话和刷新恢复；
- 三个可点击但不自动发送的示例需求；
- 非流式完整响应；
- 无真实图片的商品摘要卡片；
- 商品完整信息抽屉；
- 加载、空推荐、会话失效和请求失败状态；
- 基本响应式布局及键盘可访问性。

### 2.2 首版不包含

- 用户画像输入或新的画像逻辑；
- 登录、权限、多租户、数据库或服务端持久化；
- 历史会话列表；
- 云端部署、外网监听或跨域访问；
- Token 流式输出；
- 真实商品图片、外部商品链接或联网补全；
- 深色主题、多语言切换；
- 意图、候选池、决策分数等调试面板；
- 前端轮次上限。

## 3. 技术方案

采用单进程 FastAPI + 原生 HTML/CSS/JavaScript：

```text
Browser
  └─ FastAPI
      ├─ SessionManager
      │   └─ 单个共享 Agent
      ├─ CatalogPresenter
      │   └─ 只读 catalog.jsonl
      └─ Static UI
```

选择理由：

- Agent、配置和 catalog 均在 Python 进程内，避免进程通信；
- 不引入 Node.js 和前端构建链；
- FastAPI 同时提供静态页面和同源 API，不需要 CORS；
- catalog、索引和可选模型只初始化一次；
- 适合单机比赛演示，依赖和故障面最小。

服务默认只监听 `127.0.0.1`，不作为生产多用户服务设计。

## 4. 代码边界与目录

新增独立目录：

```text
webapp/
  __init__.py
  __main__.py
  app.py
  service.py
  catalog.py
  schemas.py
  static/
    index.html
    styles.css
    app.js
tests/
  test_webapp_api.py
  test_webapp_catalog.py
requirements-web.txt
```

各模块职责：

- `app.py`：应用工厂、生命周期、API 路由、静态文件和统一错误转换；
- `service.py`：共享 Agent、会话注册、服务端轮次、请求幂等和调用串行化；
- `catalog.py`：建立 ASIN 字节位置索引，读取摘要与详情；
- `schemas.py`：只定义 Web API 模型，不复用或修改官方 evaluator 模型；
- `static/`：无第三方 CDN 的浏览器资源；
- `__main__.py`：解析本地启动参数并启动 Uvicorn。

不得修改 `evaluator/`。如实现所需的测试替身，应通过应用工厂注入，不能在生产 Agent 中加入测试分支。

## 5. Agent 生命周期与并发

FastAPI lifespan 启动后台初始化任务并立即开始提供静态页面和健康接口。初始化任务在线程池中执行：

1. 加载现有 `EnvConfig.from_env()`；
2. 解析 catalog 路径；
3. 校验选定 catalog 并初始化 `CatalogPresenter`；
4. 创建一个共享 Agent；
5. 将健康状态从 `loading` 切换为 `ready`。

初始化未完成时，聊天 API 返回 503 `service_initializing`，页面轮询健康接口并显示 loading；失败时切换为 `failed`。关闭服务时取消或等待后台任务，不能留下工作线程。

Agent 初始化仍使用现有能力探测。缺少 LLM、稠密模型或重排模型时，由现有运行控制器自动降级，不由 Web 层重写策略。Web 层使用现有 `verify_file()` 校验实际选定的 catalog；仅在校验完成后，通过不可变配置副本关闭 Agent 内部针对默认路径的重复数据校验。若现有 `SKIP_DATA_VERIFY=1` 已显式开启，则沿用该行为。Web 界面不使用 public set，因此不会为了启动界面额外要求该文件。

虽然 Agent 的对话状态按 `session_id` 隔离，但部分检索与重排组件持有最近一次调用的可变诊断字段。首版使用一个全局 Agent 调用锁串行执行 `reset()` 和 `respond()`，以保证展示结果、Token 使用量和会话提交不会跨请求串扰。本地演示不追求并行吞吐。

## 6. 会话模型

`SessionManager` 为每个会话维护：

- 服务端生成的 UUID `session_id`；
- 下一轮轮次，从 1 开始递增；
- 最近 128 条已完成的 `message_id -> response` 有界缓存；
- 会话创建时间和最后访问时间。

创建会话时调用：

```python
agent.reset(session_id, {})
```

前端不发送 `turn`，也不提供用户画像。服务端不限制十轮；官方评测的十轮限制不扩展到演示界面。

浏览器在 `localStorage` 中保存版本化的：

- `session_id`；
- 用户与助手消息；
- 商品摘要；
- 最后更新时间。

刷新后先确认服务端会话是否存在。服务进程未重启时恢复并继续；会话不存在时清理旧状态并创建新会话。服务端不把聊天内容写入数据库或文件。

## 7. 消息幂等与轮次提交

每条用户消息由浏览器生成 UUID `message_id`。处理顺序：

1. 校验会话、`message_id` 和用户文本；
2. 在会话锁内检查幂等缓存；
3. 已完成的 `message_id` 直接返回原结果；
4. 未完成的消息使用当前服务端轮次调用 Agent；
5. Agent 未返回合法字典时不缓存、不递增轮次；
6. Agent 已返回时，其内部状态可能已经提交，因此必须缓存该轮并递增轮次；
7. Agent 返回后的商品展示补全若失败，降级为空 `products`，仍返回未经改写的 `agent_response`，不能再次调用 Agent。

前端不会自动重试。请求失败时保留用户消息并显示 `Retry`；重试复用相同 `message_id`，避免响应丢失后重复推进 Agent 状态。

用户文本满足以下规则：

- 仅空白文本返回 400；
- 最长 4,000 个 Unicode 字符；
- 通过校验后原样传给 Agent，不修剪、翻译、补全或改写。

## 8. Catalog 展示层

官方 catalog 没有图片和外部链接。界面只使用现有字段：

- `parent_asin`；
- `title`；
- `price`；
- `average_rating`、`rating_number`；
- `store`；
- `categories`；
- `features`；
- `description`；
- `details`。

`CatalogPresenter` 以二进制方式扫描 JSONL，建立 `parent_asin -> 行起始字节位置` 索引。请求商品时打开只读文件并按位置读取单行，避免在 Agent 已加载 catalog 的基础上再长期保存一份完整商品字典。重复 ASIN、无效 JSON 或缺少 `parent_asin` 视为 catalog 初始化失败。

摘要最多返回两条特征和用于卡片的末级类别；详情接口返回完整特征、描述、类别路径及 details。缺失价格显示 `Price unavailable`，缺失字段使用空值，不推断或联网补全。

展示补全不得影响推荐顺序，也不得把未知 ASIN 加入推荐结果。

## 9. Web API

### 9.1 健康状态

`GET /api/health`

返回 `loading`、`ready` 或 `failed`。失败信息使用固定错误码和脱敏说明，不返回异常栈、API 密钥、本地模型路径或请求头。

### 9.2 创建会话

`POST /api/sessions`

返回：

```json
{
  "session_id": "uuid",
  "next_turn": 1
}
```

### 9.3 确认会话

`GET /api/sessions/{session_id}`

会话存在时返回 `session_id` 和 `next_turn`；不存在时返回 404 `session_not_found`。

### 9.4 发送消息

`POST /api/sessions/{session_id}/messages`

请求：

```json
{
  "message_id": "uuid",
  "message": "Find me comfortable black shoes under $80."
}
```

响应：

```json
{
  "session_id": "uuid",
  "message_id": "uuid",
  "turn": 1,
  "agent_response": {
    "message": "What else matters most for your choice?",
    "ask_attribute": "other",
    "recommendations": [
      {"parent_asin": "B07KCFS4VC"}
    ],
    "usage": {
      "prompt_tokens": 0,
      "completion_tokens": 0
    }
  },
  "products": {
    "B07KCFS4VC": {
      "title": "Columbia Men's Thistletown Park Crew",
      "price": 27.99,
      "average_rating": 4.7,
      "rating_number": 5531,
      "store": "Columbia",
      "categories": ["Men", "Clothing", "T-Shirts"],
      "features": ["...", "..."]
    }
  }
}
```

`agent_response` 必须是 `Agent.respond()` 返回字典的无改写副本。`products` 是独立展示映射，前端按原始 `recommendations` 顺序渲染。

### 9.5 商品详情

`GET /api/products/{parent_asin}`

只返回 catalog 中存在的商品；不存在时返回 404 `product_not_found`。

## 10. 页面设计

界面使用英文、浅色、中性、低饱和视觉风格，不复制具体产品品牌。

### 10.1 布局

- 左侧窄栏：项目名称、说明和 `New chat`；
- 主区域顶部：标题与 `Local · Ready` 状态；
- 中央对话流：最大宽度约 800px；
- 底部固定输入区；
- 桌面端商品卡片两列，窄屏单列；
- 右侧商品详情抽屉。

左侧栏不展示历史会话。欢迎态提供三个英文示例：

- `I need a lightweight jacket for hiking.`
- `Find me comfortable black shoes under $80.`
- `I'm looking for a cotton shirt, but I'm still exploring.`

示例点击后只填入输入框。

### 10.2 对话交互

- 用户消息使用浅灰圆角块；
- 助手消息使用开放式内容布局；
- Enter 发送，Shift+Enter 换行；
- 请求期间显示 `Understanding your request and searching products...`；
- 同一输入区在请求完成前禁用再次发送；
- 不显示 `ask_attribute`、Token、决策分数或候选池；
- 不显示轮次限制，超过十轮仍正常调用。

### 10.3 商品卡片

卡片展示推荐序号、标题、价格、评分、评价数、品牌、类别标签和前两条特征。没有真实图片时使用统一的类别色块或图标，不发起图片请求。

点击卡片打开详情抽屉。点击遮罩、关闭按钮或按 Esc 关闭。抽屉展示完整类别、特征、描述、details 和 ASIN，不触发 Agent 请求。

## 11. 浏览器安全

- 页面、API 与静态资源同源，不启用 CORS；
- 默认绑定回环地址，不接受局域网外访问；
- 所有用户文本和 catalog 文本通过 `textContent` 等安全 DOM API 渲染；
- 不使用 `innerHTML` 插入动态内容；
- 不加载 CDN、远程字体、图片或分析脚本；
- API 错误不返回 Python 堆栈；
- 前端不读取环境变量或 API 密钥；
- `session_id` 和 `message_id` 必须符合 UUID 格式；
- 商品详情只允许查询索引中存在的 ASIN，不把 ASIN 拼接为文件路径。

## 12. 错误处理

- Agent 初始化中：展示全页加载状态；
- catalog 缺失或哈希校验失败：进入失败页并给出本地修复建议；
- 可选模型或 LLM 不可用：沿用现有自动降级，界面保持可用；
- Agent 已返回但商品展示补全失败：保留该轮 Agent 响应和推荐 ASIN，商品摘要显示不可用；
- 会话失效：提示服务已重启，创建新会话；
- 消息请求失败：保留原输入并提供 Retry；
- 空推荐：显示助手文本和继续补充需求的提示；
- 商品详情不存在：卡片仍保留 ASIN，抽屉显示详情不可用；
- Agent 返回兜底字典：按普通合法 Agent 响应展示，不由 Web 层重新解释。

## 13. 依赖与启动

前端依赖单独放入 `requirements-web.txt`，不改现有核心依赖文件。建议范围：

```text
fastapi>=0.115,<1
uvicorn>=0.30,<1
httpx>=0.27,<1
```

安装与启动：

```bash
pip install -r requirements-web.txt
python -m webapp
```

默认使用：

- catalog：`data/catalog.jsonl`；
- host：`127.0.0.1`；
- port：`8000`。

可覆盖：

```bash
python -m webapp --catalog /path/to/catalog.jsonl --port 8080
```

启动命令不得自动下载 catalog、模型或 Python 依赖。

## 14. 测试

### 14.1 现有系统回归

- 运行完整现有测试集；
- 验证官方 evaluator 未修改；
- 对固定 Fake Agent 输入断言 `agent_response` 与原始返回字典完全相等。

### 14.2 SessionManager 与 API

使用依赖注入的 Fake Agent 覆盖：

- 创建会话只调用一次 `reset(session_id, {})`；
- 轮次从 1 连续递增且第 11 轮仍可调用；
- 重复 `message_id` 不重复调用 Agent；
- 不同会话状态隔离；
- 全局 Agent 调用串行化；
- 适配异常不递增轮次；
- 会话失效返回稳定错误码；
- 空白、过长文本和非法 UUID 被拒绝；
- 错误响应不泄露异常信息。

### 14.3 CatalogPresenter

- 正确建立和读取 JSONL 字节位置；
- 正确处理 Unicode、缺失价格、缺失可选字段和未知 ASIN；
- 摘要字段不会改变推荐顺序；
- 商品文本作为数据而非 HTML 返回。

### 14.4 浏览器验收

人工在本地浏览器验证：

- 欢迎态与示例提示；
- 连续多轮发送和非流式加载状态；
- 商品卡片顺序和详情抽屉；
- 刷新恢复、新会话和服务重启后的失效处理；
- Retry 使用原 `message_id`；
- Enter、Shift+Enter、Esc 和键盘焦点；
- 桌面两列和窄屏单列布局；
- 用户输入及 catalog 特殊字符不会执行为 HTML。

## 15. 验收标准

- `python -m webapp` 可一条命令启动本地界面；
- 在离线规则模式下可完成不受前端轮数限制的连续对话；
- 同一输入下，Web API 的 `agent_response` 与直接调用 Agent 的结果一致；
- 推荐商品顺序与 Agent 原始 ASIN 顺序一致；
- 页面刷新恢复会话，服务重启安全创建新会话；
- 不修改官方 evaluator、Agent 接口和比赛核心行为；
- 原有完整测试及新增 Web API 测试全部通过；
- 页面不依赖外部网络、Node.js 或远程静态资源。
