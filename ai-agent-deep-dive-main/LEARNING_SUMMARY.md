# 学习进度总结

> 记录你在 AI Agent 学习中已掌握的知识和完成的进度

---

## 📅 学习时间线

### 第一阶段：前置准备

| 步骤 | 内容 | 完成 |
|------|------|:----:|
| 1 | Python 环境检查（3.13.3） | ✅ |
| 2 | Poetry 安装（2.4.1） | ✅ |
| 3 | 项目依赖安装（`my_agent` 注册到 pyproject.toml） | ✅ |
| 4 | PyCharm 配置（Python Console、测试运行） | ✅ |

### 第二阶段：亲手搭建 Agent

| 步骤 | 文件 | 内容 | 完成 |
|------|------|------|:----:|
| 1 | `src/my_agent/__init__.py` | 空文件，标记为 Python 包 | ✅ |
| 2 | `src/my_agent/models.py` | `Message`、`ToolResult`、`Tool` 三个数据类 | ✅ |
| 3 | `src/my_agent/llm.py` | `LLMClient` 协议 + `FakeLLMClient` | ✅ |
| 4 | `src/my_agent/agent.py` | `Agent` 核心类（10 个方法） | ✅ |

---

## 🧩 已掌握的核心概念

### 概念 1：Message（消息模型）

```python
@dataclass
class Message:
    role: str       # 谁说的（user/assistant/tool_result）
    content: str    # 说了什么
    meta: dict      # 附加信息（时间、轮次等）
```

**关键理解**：
- Agent 用 `self.messages: list[Message]` 记录所有对话
- `add_message()` 方法创建并追加消息
- `role` 区分说话人，决定 Agent 如何理解和回应

---

### 概念 2：Tool（工具系统）

```python
class Tool:
    def __init__(self, name, description, handler):
        self.name = name          # 工具名字（也是查找时的"钥匙"）
        self.description = desc   # 工具说明
        self.handler = handler    # 实际执行的函数

    def call(self, payload):
        return self.handler(payload)
```

**关键理解**：
- 工具通过 `self.tools[tool.name] = tool` 注册到字典
- `[tool.name]` 是字典的 key，用来给工具贴标签方便查找
- 不加 `[tool.name]` 会直接把字典替换掉，无法注册多个工具
- `ToolResult` 包含 `ok`（成功/失败）和 `content`（结果内容）

---

### 概念 3：LLMClient（模型抽象层）

```python
class LLMClient(Protocol):
    def stream_text(self, messages: list[Message]) -> Iterator[str]:
        ...
```

**关键理解**：
- 协议（Protocol）定义接口标准，不写具体实现
- Agent 只依赖协议，不依赖具体模型
- 好处：换模型（FakeLLM → 真实 API）不需要改 Agent 代码

---

### 概念 4：FakeLLMClient（教学用假模型）

```python
class FakeLLMClient:
    def stream_text(self, messages):
        last = 找到最后一条用户消息的 content
        response = f"[fake-llm] 你刚才说的是：{last}"
        for chunk in self._chunk_text(response, size=8):
            yield chunk
```

**关键理解**：
- `@staticmethod`：不需要 `self`，直接用 `类名.方法()` 调用
- `yield`：一次返回一小段，函数暂停，下次继续
- `_chunk_text`：自己写的功能函数，把文本切成小块模拟流式输出
- `for chunk in ...` 中的 `chunk` 和 `for m in messages` 中的 `m` 一样，都是临时变量名，可以随便改

---

### 概念 5：Agent（核心大脑）

```python
class Agent:
    def __init__(self, llm=None):
        self.messages = []      # 对话历史
        self.tools = {}         # 工具字典
        self.memory = []        # 简单记忆
        self.max_turns = 20     # 最大轮次
        self.llm = llm or FakeLLMClient()  # AI 大脑
```

**10 个方法**：

| 方法 | 作用 | 关键代码 |
|------|------|----------|
| `__init__` | 初始化所有属性 | `self.tools = {}` |
| `register_tool` | 注册工具 | `self.tools[tool.name] = tool` |
| `add_message` | 添加消息 | `self.messages.append(Message(...))` |
| `remember` | 记录记忆 | `self.memory.append(text)` |
| `can_use_tool` | 检查工具是否存在 | `return tool_name in self.tools` |
| `load_skills` | 发现技能文件夹 | `rglob("SKILL.md")` |
| `call_llm_stream` | 调用 LLM | `self.llm.stream_text(self.messages)` |
| `model_step` | LLM 单步思考 | 返回 `{"type": "message", ...}` |
| `run` | 主循环 | 处理 message / tool_call / 超限 |

---

### 概念 6：run 主循环（核心流程）

```
用户输入 → add_message("user", ...)
    ↓
进入 for 循环（max_turns 次）
    ↓
model_step() 调用 LLM 思考
    ↓
判断返回类型：
  ├─ "message" → 记录回复 → return（结束）
  ├─ "tool_call" → 执行工具 → continue（继续）
  └─ 其他 → raise ValueError（报错）
    ↓
超出 max_turns → 返回超限提示
```

---

## ⚙️ 关键技术点

### 1. `self.tools[tool.name] = tool`

- `tool.name` 是 key（标签），`tool` 是 value（整个对象）
- 不是"只存名字"，而是**用名字当钥匙，存整个对象**

### 2. `str | Path` 参数类型

```python
def load_skills(self, skills_dir: str | Path) -> list[str]:
```
表示这个参数可以是字符串或 Path 对象，都行。

### 3. `-> None` / `-> bool` / `-> dict` 返回值注解

告诉看代码的人这个方法返回什么类型：
- `-> None`：什么都不返回
- `-> bool`：返回 True/False
- `-> dict`：返回字典

### 4. `"".join(chunks)`

把列表里的多个字符串粘成一个：
```python
chunks = ["a", "b", "c"]
"".join(chunks)   # "abc"
",".join(chunks)  # "a,b,c"
```

### 5. `continue`

跳过本轮剩余代码，进入下一轮循环。用在不满足条件时让 Agent 重新思考。

### 6. `raise ValueError(...)`

主动报错，程序停止。用于处理意料之外的情况，防止静默忽略错误。

---

## 📂 文件清单

```
src/my_agent/
├── __init__.py          ← 空文件，标记为 Python 包
├── models.py            ← Message、ToolResult、Tool 数据模型
├── llm.py               ← LLMClient 协议 + FakeLLMClient
└── agent.py             ← Agent 核心类（全部业务逻辑）
```

---

## 🛠️ 已验证的功能

| 测试内容 | 命令 | 结果 |
|----------|------|:----:|
| Agent 基本对话 | `agent.run("你好")` | ✅ `[fake-llm]你刚才说的是：你好` |
| Skills 发现 | `agent.load_skills(tmp)` | ✅ `['coding', 'writing']` |
| 消息历史记录 | 查看 `agent.messages` | ✅ 正确记录 role 和 content |

---

## 🚧 已发现但未解决的问题

1. `model_step()` 永远返回 `"message"`，永远不会返回 `"tool_call"`
   - 结果：Agent 注册了工具也不会用
   - 解决方案见进阶教程"阶段一"

2. 原项目测试时只找到 `['writing']`（少了一个 `coding`）
   - 原因：之前的 `loaded.append()` 缩进错误，不在 for 循环里
   - 已修复 ✅

---

## 📖 已创建的学习文档

| 文档 | 内容 | 位置 |
|------|------|------|
| 零基础新手教程 | 从安装到第一个 Agent | `ZERO_TO_AGENT.md` |
| 环境搭建指南 | Python/Poetry 安装步骤 | `SETUP_ENVIRONMENT.md` |
| 进阶教程 | 工具调用、CLI、实用工具、记忆系统、多 Agent | `ADVANCED_TUTORIAL.md` |
| 本总结 | 已学知识汇总 | 当前文件 |

---

**下一步建议**：打开 `ADVANCED_TUTORIAL.md`，从"阶段一：让 Agent 真的会用工具"开始继续学习。
