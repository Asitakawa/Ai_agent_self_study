# AI Agent 进阶教程

> 在你已经完成基础 Agent 搭建的基础上，继续深入

---

## 📋 学习前提

你已经完成了：
- ✅ `Message`、`ToolResult`、`Tool` 数据模型
- ✅ `LLMClient` 协议 + `FakeLLMClient`
- ✅ `Agent` 核心类（`run` 主循环）
- ✅ 基础测试验证

---

## 🎯 阶段一：让 Agent 真的会用工具（2 天）

### 现状问题

当前的 `model_step()` 永远返回 `"message"` 类型，永远不会返回 `"tool_call"`。所以 Agent **注册了工具也不会用**，因为 LLM 根本不会告诉它"用工具"。

要让 Agent 真的会用工具，有两个方向：

- **方案 A**：接入真实 LLM API（OpenAI 等），让真 AI 来决定是否用工具
- **方案 B**：修改 `model_step()` 的逻辑，让它根据用户输入模拟工具调用

我们先用**方案 B**理解工具调用的完整流程，再用**方案 A**接入真实 AI。

---

### 模块 1：让 model_step 能返回 tool_call

#### 1.1 分析当前代码

打开你的 `agent.py`，看 `model_step` 方法：

```python
def model_step(self) -> dict[str, Any]:
    chunks = list(self.call_llm_stream())
    return {
        "type": "message",                    # ← 永远返回 message
        "content": "".join(chunks),
        "chunks": chunks,
    }
```

它永远返回 `"type": "message"`，所以 `run` 方法里的 `if step["type"] == "tool_call"` 这段代码**永远不会被执行**。

#### 1.2 修改 model_step：让它能识别工具调用意图

修改 `model_step` 方法，加入简单的规则判断：

```python
def model_step(self) -> dict[str, Any]:
    # 先调用 LLM
    chunks = list(self.call_llm_stream())
    content = "".join(chunks)

    # 获取最后一条用户消息
    last_user_msg = ""
    for msg in reversed(self.messages):
        if msg.role == "user":
            last_user_msg = msg.content
            break

    # 检查用户是否提到了已注册的工具名
    for tool_name, tool in self.tools.items():
        if tool_name in last_user_msg:
            return {
                "type": "tool_call",
                "tool": tool_name,
                "input": {"text": last_user_msg},
            }

    # 没有匹配到工具，就正常回复
    return {
        "type": "message",
        "content": content,
        "chunks": chunks,
    }
```

**这段代码的逻辑：**

```
用户说"请用 echo 工具"
  ↓
检查所有已注册的工具
  ↓
发现用户提到了 "echo"
  ↓
返回 tool_call，告诉 Agent："用 echo 工具，参数是 {'text': '请用 echo 工具'}"
  ↓
Agent 的主循环收到 tool_call，执行 echo 工具
  ↓
工具返回结果，记录到消息历史
  ↓
继续下一轮循环
```

**注意**：当前 `model_step` 返回 `tool_call` 后没有再次调用 LLM 来生成最终回复。这个我们在后面的模块会完善。

#### 1.3 测试

创建一个测试文件 `test_tool_call.py`：

```python
from my_agent.agent import Agent
from my_agent.models import Tool, ToolResult


# 创建一个回显工具
def echo_handler(payload):
    text = payload.get("text", "")
    return ToolResult(ok=True, content=f"Echo: {text}")


# 创建一个计算器工具
def calculator_handler(payload):
    expression = payload.get("expression", "")
    try:
        result = eval(expression)
        return ToolResult(ok=True, content=f"{expression} = {result}")
    except Exception as e:
        return ToolResult(ok=False, content=f"计算错误: {e}")


# 创建 Agent 并注册工具
agent = Agent()
agent.register_tool(Tool("echo", "回显文本", echo_handler))
agent.register_tool(Tool("calculator", "计算表达式", calculator_handler))

# 测试 1：直接对话
print("=== 测试 1：直接对话 ===")
result = agent.run("你好")
print(f"Agent: {result}")

# 测试 2：使用 echo 工具
print("\n=== 测试 2：使用 echo 工具 ===")
result = agent.run("请用 echo 工具回复这段话")
print(f"Agent: {result}")

# 查看消息历史
print("\n=== 消息历史 ===")
for i, msg in enumerate(agent.messages):
    print(f"{i}. [{msg.role}] {msg.content[:60]}")
```

---

### 模块 2：接入真实 LLM API

#### 2.1 创建 OpenAI 客户端

在 `src/my_agent/` 下创建 `openai_client.py`：

```python
import os
from typing import Iterator

from openai import OpenAI

from my_agent.models import Message
from my_agent.llm import LLMClient


class OpenAIClient(LLMClient):
    """接入 OpenAI 兼容的 API"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-3.5-turbo",
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def stream_text(self, messages: list[Message]) -> Iterator[str]:
        # 把我们的 Message 格式转成 OpenAI 的格式
        openai_messages = []
        for msg in messages:
            role = msg.role
            if role == "tool_result":
                role = "tool"
            openai_messages.append({"role": role, "content": msg.content})

        # 调用 OpenAI 的流式接口
        response = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            stream=True,
        )

        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content
```

#### 2.2 安装依赖

```bash
pip install openai
```

#### 2.3 测试真实 LLM

创建 `test_real_llm.py`：

```python
from my_agent.agent import Agent
from my_agent.openai_client import OpenAIClient

# 创建真实 LLM 客户端
# 方式 1：设置环境变量 OPENAI_API_KEY
# 方式 2：直接传入 api_key
llm = OpenAIClient(
    api_key="你的API_KEY",  # 替换成你的 key
    model="gpt-3.5-turbo",  # 或其他模型
)

# 创建 Agent 并使用真实 LLM
agent = Agent(llm=llm)

# 测试对话
result = agent.run("介绍一下你自己")
print(result)
```

#### 2.4 支持工具调用（Function Calling）

真正的 AI Agent 应该让 LLM 自己决定是否用工具。OpenAI 支持 function calling：

```python
class OpenAIToolClient(LLMClient):
    """支持 function calling 的 OpenAI 客户端"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-3.5-turbo",
        tools: dict[str, Tool] | None = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url
        self.model = model
        self.tools = tools or {}
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def _build_tools_spec(self) -> list[dict]:
        """把我们的 Tool 转成 OpenAI 的 tools 格式"""
        specs = []
        for name, tool in self.tools.items():
            specs.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "输入文本"},
                        },
                    },
                },
            })
        return specs

    def stream_text(self, messages: list[Message]) -> Iterator[str]:
        openai_messages = []
        for msg in messages:
            role = msg.role
            if role == "tool_result":
                role = "tool"
            openai_messages.append({"role": role, "content": msg.content})

        tools_spec = self._build_tools_spec()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            tools=tools_spec if tools_spec else None,
            stream=False,  # function calling 用非流式
        )

        choice = response.choices[0]

        # 检查是否调用了工具
        if choice.finish_reason == "tool_calls":
            for call in choice.message.tool_calls:
                yield f"__TOOL_CALL__:{call.function.name}:{call.function.arguments}"
        else:
            yield choice.message.content or ""
```

---

## 🎯 阶段二：构建 CLI 界面（1 天）

### 模块 3：创建交互式命令行

#### 3.1 创建 CLI 文件

在 `src/my_agent/` 下创建 `cli.py`：

```python
import argparse
import sys
from pathlib import Path

from my_agent.agent import Agent
from my_agent.models import Tool, ToolResult


# 定义一个回显工具
def echo_tool(payload: dict) -> ToolResult:
    text = payload.get("text", "")
    return ToolResult(ok=True, content=f"echo: {text}")


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        prog="my-agent",
        description="我的 AI Agent 命令行工具",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="你好",
        help="用户输入的提示词",
    )
    parser.add_argument(
        "--skills-dir",
        default="skills",
        help="Skills 文件夹路径",
    )
    parser.add_argument(
        "--list-skills",
        action="store_true",
        help="列出所有可用的 Skills",
    )
    return parser


def main() -> int:
    """主函数"""
    parser = build_parser()
    args = parser.parse_args()

    # 创建 Agent
    agent = Agent()

    # 注册内置工具
    agent.register_tool(Tool("echo", "回显文本", echo_tool))

    # 加载 Skills
    skills = agent.load_skills(Path(args.skills_dir))
    if args.list_skills:
        print("已发现的 Skills:")
        for skill in skills:
            print(f"  - {skill}")
        return 0

    # 如果有 Skills，记录到记忆
    if skills:
        agent.remember(f"loaded_skills={','.join(skills)}")

    # 运行 Agent
    reply = agent.run(args.prompt)
    print(reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

#### 3.2 注册为命令行命令

一种简单的方式是在项目根目录创建 `run_agent.py`：

```python
from src.my_agent.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

然后运行：

```bash
python run_agent.py "你好"
python run_agent.py --list-skills
```

#### 3.3 交互式模式

创建一个支持连续对话的版本 `interactive_cli.py`：

```python
from my_agent.agent import Agent
from my_agent.models import Tool, ToolResult


def echo_tool(payload):
    text = payload.get("text", "")
    return ToolResult(ok=True, content=f"echo: {text}")


def main():
    agent = Agent()
    agent.register_tool(Tool("echo", "回显文本", echo_tool))

    print("AI Agent 交互模式（输入 'exit' 退出）")
    print("-" * 40)

    while True:
        try:
            user_input = input("\n你: ")
            if user_input.lower() in ("exit", "quit", "q"):
                print("再见！")
                break

            reply = agent.run(user_input)
            print(f"Agent: {reply}")

        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    main()
```

运行：

```bash
python interactive_cli.py
```

---

## 🎯 阶段三：创建实用工具集（2 天）

### 模块 4：文件系统工具

#### 4.1 创建文件工具

在 `src/my_agent/` 下创建 `file_tools.py`：

```python
from pathlib import Path
from my_agent.models import Tool, ToolResult


def read_file(payload: dict) -> ToolResult:
    """读取文件内容"""
    filepath = payload.get("path", "")
    if not filepath:
        return ToolResult(ok=False, content="错误：请提供文件路径")

    try:
        content = Path(filepath).read_text(encoding="utf-8")
        return ToolResult(ok=True, content=content)
    except Exception as e:
        return ToolResult(ok=False, content=f"读取失败: {e}")


def write_file(payload: dict) -> ToolResult:
    """写入文件"""
    filepath = payload.get("path", "")
    content = payload.get("content", "")

    if not filepath:
        return ToolResult(ok=False, content="错误：请提供文件路径")

    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        Path(filepath).write_text(content, encoding="utf-8")
        return ToolResult(ok=True, content=f"已写入: {filepath}")
    except Exception as e:
        return ToolResult(ok=False, content=f"写入失败: {e}")


def list_files(payload: dict) -> ToolResult:
    """列出目录内容"""
    directory = payload.get("path", ".")

    try:
        p = Path(directory)
        if not p.exists():
            return ToolResult(ok=False, content=f"目录不存在: {directory}")

        items = []
        for item in p.iterdir():
            prefix = "[DIR]" if item.is_dir() else "[FILE]"
            items.append(f"{prefix} {item.name}")

        return ToolResult(ok=True, content="\n".join(items))
    except Exception as e:
        return ToolResult(ok=False, content=f"列出失败: {e}")


# 创建工具对象
read_file_tool = Tool("read_file", "读取文件内容", read_file)
write_file_tool = Tool("write_file", "写入内容到文件", write_file)
list_files_tool = Tool("list_files", "列出目录中的文件", list_files)
```

#### 4.2 测试文件工具

```python
from my_agent.agent import Agent
from my_agent.file_tools import read_file_tool, write_file_tool, list_files_tool

agent = Agent()
agent.register_tool(read_file_tool)
agent.register_tool(write_file_tool)
agent.register_tool(list_files_tool)

# 测试写入
result = agent.run("请写入 test.txt，内容为 Hello World")
print(result)

# 测试读取
result = agent.run("请读取 test.txt")
print(result)
```

### 模块 5：网络工具

#### 5.1 创建网络请求工具

在 `src/my_agent/` 下创建 `web_tools.py`：

```python
import json
import urllib.request
import urllib.error
from my_agent.models import Tool, ToolResult


def fetch_url(payload: dict) -> ToolResult:
    """获取网页内容"""
    url = payload.get("url", "")
    if not url:
        return ToolResult(ok=False, content="错误：请提供 URL")

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read().decode("utf-8")
            # 只返回前 500 个字符，避免太长
            return ToolResult(
                ok=True,
                content=content[:500],
                meta={"status": response.status, "length": len(content)},
            )
    except Exception as e:
        return ToolResult(ok=False, content=f"请求失败: {e}")


fetch_tool = Tool("fetch", "获取网页内容", fetch_url)
```

#### 5.2 安装依赖

如果需要更强大的 HTTP 库：

```bash
pip install requests
```

使用 requests 的版本：

```python
import requests

def fetch_url_v2(payload: dict) -> ToolResult:
    url = payload.get("url", "")
    if not url:
        return ToolResult(ok=False, content="错误：请提供 URL")

    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        return ToolResult(
            ok=True,
            content=resp.text[:500],
            meta={"status": resp.status_code},
        )
    except Exception as e:
        return ToolResult(ok=False, content=f"请求失败: {e}")
```

---

## 🎯 阶段四：记忆与上下文管理（2 天）

### 模块 6：实现长期记忆

#### 6.1 问题分析

当前的 `self.memory` 只是简单列表，Agent 关闭后记忆就丢失了。我们需要：
1. 把记忆保存到文件里
2. Agent 启动时加载之前的记忆
3. 自动总结重要的信息

#### 6.2 创建记忆管理器

在 `src/my_agent/` 下创建 `memory.py`：

```python
import json
from pathlib import Path
from datetime import datetime


class MemoryManager:
    """管理 Agent 的长期记忆"""

    def __init__(self, memory_file: str = "agent_memory.json"):
        self.memory_file = Path(memory_file)
        self.memories: list[dict] = []
        self._load()

    def _load(self):
        """从文件加载记忆"""
        if self.memory_file.exists():
            try:
                data = self.memory_file.read_text(encoding="utf-8")
                self.memories = json.loads(data)
            except Exception:
                self.memories = []

    def _save(self):
        """保存记忆到文件"""
        self.memory_file.write_text(
            json.dumps(self.memories, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add(self, category: str, content: str):
        """添加一条记忆"""
        self.memories.append({
            "category": category,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        self._save()

    def get_by_category(self, category: str) -> list[str]:
        """按类别获取记忆"""
        return [
            m["content"]
            for m in self.memories
            if m["category"] == category
        ]

    def get_recent(self, count: int = 5) -> list[str]:
        """获取最近的记忆"""
        return [
            m["content"]
            for m in self.memories[-count:]
        ]

    def summarize(self) -> str:
        """生成记忆摘要"""
        if not self.memories:
            return "暂无记忆"

        lines = []
        for m in self.memories[-10:]:
            lines.append(f"[{m['category']}] {m['content']}")
        return "\n".join(lines)

    def clear(self):
        """清空记忆"""
        self.memories = []
        self._save()
```

#### 6.3 集成到 Agent

扩展你的 Agent，添加记忆管理功能。在 `agent.py` 的 `__init__` 里增加：

```python
from my_agent.memory import MemoryManager

class Agent:
    def __init__(self, llm=None, memory_file: str | None = None):
        # ... 原有代码 ...
        self.memory_manager = MemoryManager(memory_file) if memory_file else None

    def add_memory(self, category: str, content: str):
        """添加长期记忆"""
        self.remember(content)
        if self.memory_manager:
            self.memory_manager.add(category, content)
```

---

## 🎯 阶段五：多 Agent 协作（3 天）

### 模块 7：主从 Agent 模式

#### 7.1 创建管理者 Agent

在 `src/my_agent/` 下创建 `supervisor.py`：

```python
from my_agent.agent import Agent
from my_agent.models import Message


class SupervisorAgent(Agent):
    """
    管理者 Agent：负责任务分配和结果汇总

    它本身也是一个 Agent，但多了管理其他 Agent 的能力。
    """

    def __init__(self, llm=None):
        super().__init__(llm)
        self.workers: dict[str, Agent] = {}

    def add_worker(self, name: str, agent: Agent, description: str = ""):
        """注册一个工作者 Agent"""
        self.workers[name] = agent
        self.remember(f"worker_{name}={description}")

    def remove_worker(self, name: str):
        """移除一个工作者 Agent"""
        if name in self.workers:
            del self.workers[name]

    def delegate(self, worker_name: str, task: str) -> str:
        """把任务分配给指定的工作者"""
        if worker_name not in self.workers:
            return f"错误：找不到工作者 '{worker_name}'"

        worker = self.workers[worker_name]
        result = worker.run(task)

        # 记录分配日志
        self.add_message(
            "tool_result",
            f"[委托] {worker_name} 完成任务: {result[:50]}...",
            tool="delegate",
            worker=worker_name,
        )

        return result

    def run(self, user_input: str) -> str:
        """
        重写 run 方法：如果有工作者，尝试分配任务
        """
        # 检查用户是否指定了工作者
        for name in self.workers:
            if name in user_input:
                return self.delegate(name, user_input)

        # 没有指定工作者，用自己的 LLM 处理
        return super().run(user_input)
```

#### 7.2 测试多 Agent 协作

```python
from my_agent.agent import Agent
from my_agent.supervisor import SupervisorAgent
from my_agent.models import Tool, ToolResult


# 创建一个"写文章"的 Agent
def write_article(payload):
    topic = payload.get("topic", "无主题")
    return ToolResult(ok=True, content=f"关于《{topic}》的文章：\n这是关于{topic}的一篇好文章。")


writer = Agent()
writer.register_tool(Tool("write", "写文章", write_article))


# 创建一个"检查错误"的 Agent
def check_errors(payload):
    text = payload.get("text", "")
    return ToolResult(ok=True, content=f"检查完成：没有发现错误。原文：{text[:30]}...")


checker = Agent()
checker.register_tool(Tool("check", "检查错误", check_errors))


# 创建管理者
boss = SupervisorAgent()
boss.add_worker("writer", writer, "负责写文章")
boss.add_worker("checker", checker, "负责检查错误")

# 测试分配任务
result = boss.run("writer 帮我写一篇关于 Python 的文章")
print(result)

result = boss.run("checker 帮我检查一下这段代码")
print(result)
```

### 模块 8：管道模式

让多个 Agent 按顺序处理同一个任务，前一个的输出是后一个的输入。

```python
class Pipeline:
    """
    管道模式：多个 Agent 按顺序处理任务

    例如：写文章 → 检查错误 → 润色
    """

    def __init__(self):
        self.steps: list[tuple[str, Agent]] = []

    def add_step(self, name: str, agent: Agent):
        """添加一个处理步骤"""
        self.steps.append((name, agent))

    def execute(self, initial_input: str) -> list[tuple[str, str]]:
        """按顺序执行所有步骤"""
        results = []
        current_input = initial_input

        for name, agent in self.steps:
            output = agent.run(current_input)
            results.append((name, output))
            current_input = output  # 当前输出作为下一步的输入

        return results


# 使用示例
pipeline = Pipeline()
pipeline.add_step("writer", writer_agent)
pipeline.add_step("checker", checker_agent)
pipeline.add_step("polisher", polisher_agent)

results = pipeline.execute("写一篇关于 AI 的文章")
for step_name, output in results:
    print(f"[{step_name}] {output[:50]}...")
```

---

## 🎯 阶段六：验证与错误处理（2 天）

### 模块 9：验证 Agent

#### 9.1 创建验证器

```python
class VerificationAgent(Agent):
    """
    验证 Agent：检查另一个 Agent 的工作结果

    它不执行任务，而是评价任务的执行结果。
    """

    def verify(self, task: str, result: str) -> dict:
        """
        验证任务结果

        返回：
            {"passed": bool, "reason": str, "suggestions": list[str]}
        """
        # 简单的规则验证
        issues = []

        # 检查结果是否为空
        if not result or len(result) < 10:
            issues.append("结果太短或为空")

        # 检查是否包含错误信息
        if "error" in result.lower() or "错误" in result:
            issues.append("结果包含错误信息")

        # 检查结果是否包含原始任务
        keywords = task.split()
        found_keywords = sum(1 for kw in keywords if kw in result)
        if found_keywords < len(keywords) * 0.3:
            issues.append("结果与任务关联度较低")

        if issues:
            return {
                "passed": False,
                "reason": "; ".join(issues),
                "suggestions": ["请重新执行任务", "请提供更详细的输入"],
            }

        return {
            "passed": True,
            "reason": "验证通过",
            "suggestions": [],
        }
```

#### 9.2 集成验证到 Agent

```python
class SelfVerifyingAgent(Agent):
    """带自检能力的 Agent"""

    def __init__(self, llm=None):
        super().__init__(llm)
        self.verifier = VerificationAgent()
        self.verification_history: list[dict] = []

    def run(self, user_input: str) -> str:
        # 先执行任务
        result = super().run(user_input)

        # 自动验证结果
        verification = self.verifier.verify(user_input, result)
        self.verification_history.append({
            "task": user_input,
            "result": result,
            "verification": verification,
        })

        # 如果验证失败，记录问题
        if not verification["passed"]:
            self.remember(f"验证失败: {verification['reason']}")
            return f"{result}\n\n[警告] {verification['reason']}"

        return result
```

---

## 🎯 阶段七：实用工具链（3 天）

### 模块 10：构建代码助手

创建一个能帮你写代码、运行测试、检查错误的 Agent。

```python
# code_assistant.py
import subprocess
from my_agent.agent import Agent
from my_agent.models import Tool, ToolResult


def run_command(payload: dict) -> ToolResult:
    """运行 Shell 命令"""
    cmd = payload.get("command", "")
    if not cmd:
        return ToolResult(ok=False, content="错误：请提供命令")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout or result.stderr
        return ToolResult(
            ok=result.returncode == 0,
            content=output[:1000],
            meta={"returncode": result.returncode},
        )
    except subprocess.TimeoutExpired:
        return ToolResult(ok=False, content="命令执行超时")
    except Exception as e:
        return ToolResult(ok=False, content=f"执行失败: {e}")


def run_python(payload: dict) -> ToolResult:
    """运行 Python 代码"""
    code = payload.get("code", "")
    if not code:
        return ToolResult(ok=False, content="错误：请提供 Python 代码")

    try:
        # 把代码临时写入文件并执行
        with open("_temp_code.py", "w", encoding="utf-8") as f:
            f.write(code)

        result = subprocess.run(
            ["python", "_temp_code.py"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout or result.stderr
        return ToolResult(
            ok=result.returncode == 0,
            content=output[:1000],
        )
    except Exception as e:
        return ToolResult(ok=False, content=f"执行失败: {e}")


# 创建代码助手 Agent
code_agent = Agent()
code_agent.register_tool(Tool("run_command", "运行 Shell 命令", run_command))
code_agent.register_tool(Tool("run_python", "运行 Python 代码", run_python))
```

### 模块 11：构建笔记管理 Agent

```python
# notes_agent.py
import json
from datetime import datetime
from pathlib import Path
from my_agent.agent import Agent
from my_agent.models import Tool, ToolResult


NOTES_FILE = "notes.json"


def load_notes():
    if Path(NOTES_FILE).exists():
        return json.loads(Path(NOTES_FILE).read_text())
    return []


def save_notes(notes):
    Path(NOTES_FILE).write_text(json.dumps(notes, ensure_ascii=False, indent=2))


def add_note(payload: dict) -> ToolResult:
    """添加笔记"""
    content = payload.get("content", "")
    if not content:
        return ToolResult(ok=False, content="错误：请提供笔记内容")

    notes = load_notes()
    notes.append({
        "id": len(notes) + 1,
        "content": content,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    save_notes(notes)
    return ToolResult(ok=True, content=f"笔记 #{len(notes)} 已保存")


def list_notes(payload: dict) -> ToolResult:
    """列出笔记"""
    notes = load_notes()
    if not notes:
        return ToolResult(ok=True, content="暂无笔记")

    lines = []
    for n in notes[-10:]:
        lines.append(f"#{n['id']} [{n['time']}] {n['content'][:50]}")
    return ToolResult(ok=True, content="\n".join(lines))


def search_notes(payload: dict) -> ToolResult:
    """搜索笔记"""
    keyword = payload.get("keyword", "")
    if not keyword:
        return ToolResult(ok=False, content="错误：请提供关键词")

    notes = load_notes()
    found = [n for n in notes if keyword in n["content"]]

    if not found:
        return ToolResult(ok=True, content="未找到匹配的笔记")

    lines = [f"找到 {len(found)} 条结果:"]
    for n in found:
        lines.append(f"#{n['id']} {n['content'][:60]}")
    return ToolResult(ok=True, content="\n".join(lines))


# 创建笔记 Agent
notes_agent = Agent()
notes_agent.register_tool(Tool("add_note", "添加笔记", add_note))
notes_agent.register_tool(Tool("list_notes", "列出最近笔记", list_notes))
notes_agent.register_tool(Tool("search_notes", "搜索笔记", search_notes))
```

---

## 🎓 进阶学习路线图

```
阶段一：工具调用（2天）
  ├─ model_step 返回 tool_call
  ├─ 接入真实 LLM API
  └─ Function Calling

阶段二：CLI 界面（1天）
  ├─ 命令行参数解析
  ├─ 交互式模式
  └─ Skills 集成

阶段三：实用工具（2天）
  ├─ 文件系统工具
  ├─ 网络请求工具
  └─ 工具组合使用

阶段四：记忆系统（2天）
  ├─ 文件持久化
  ├─ 记忆分类检索
  └─ 上下文管理

阶段五：多 Agent（3天）
  ├─ 主从模式
  ├─ 管道模式
  └─ 任务分发

阶段六：验证系统（2天）
  ├─ 结果验证
  ├─ 错误处理
  └─ 自检机制

阶段七：实用链（3天）
  ├─ 代码助手
  ├─ 笔记管理
  └─ 综合应用
```

---

## ✅ 检查清单

### 阶段一
- [ ] model_step 能返回 tool_call
- [ ] 注册的工具真的被调用了
- [ ] 接入了真实 LLM API
- [ ] 理解 Function Calling 的原理

### 阶段二
- [ ] 能通过命令行传参
- [ ] 能进入交互式对话模式
- [ ] 支持 Skills 列表展示

### 阶段三
- [ ] 文件读写工具正常工作
- [ ] 网络请求工具正常工作
- [ ] 多个工具协作无冲突

### 阶段四
- [ ] 记忆能保存到文件
- [ ] 重启后能加载之前的记忆
- [ ] 记忆分类检索正常

### 阶段五
- [ ] SupervisorAgent 能分配任务
- [ ] Pipeline 按顺序执行
- [ ] 工作者之间的结果传递正常

### 阶段六
- [ ] VerificationAgent 能验证结果
- [ ] 验证失败有提示
- [ ] 自检 Agent 能记录问题

### 阶段七
- [ ] 代码助手能运行命令
- [ ] 笔记管理能增删查
- [ ] 综合应用场景跑通

---

**完成这些进阶内容后，你将具备：**
- ✅ 独立设计和实现 AI Agent 的能力
- ✅ 理解真实产品中 Agent 的工程挑战
- ✅ 构建实用工具的动手能力
- ✅ 对接真实 AI 模型的经验
