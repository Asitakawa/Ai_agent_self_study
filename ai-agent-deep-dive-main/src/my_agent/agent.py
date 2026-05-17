from pathlib import Path
from typing import Any, Iterator

from my_agent.models import Message, Tool, ToolResult
from my_agent.llm import LLMClient, FakeLLMClient


class Agent:
    def __init__(self, llm: LLMClient | None = None):
        self.messages: list[Message] = []
        """
self.messages: list[Message] = []
存放所有对话历史，是一个列表，里面每个元素都是 Message 对象
初始是空列表
        """
        self.tools: dict[str, Tool] = {}
        """
self.tools: dict[str, Tool] = {}
存放注册的工具，是一个字典
key 是工具名（字符串），value 是 Tool 对象
初始是空字典
        """
        self.memory: list[str] = []
        """
self.memory: list[str] = []
存放简单的记忆，是一个字符串列表
初始是空列表
        """
        self.max_turns: int = 20
        """
self.max_turns: int = 20
Agent 最多思考多少轮
防止无限循环——如果 Agent 一直在调用工具停不下来，到 20 轮就强制停止
        """
        self.llm: LLMClient = llm or FakeLLMClient()
        """
self.llm: LLMClient = llm or FakeLLMClient()
如果传了 llm 参数，就用传进来的
如果没传（llm is None），就用 FakeLLMClient()
A or B 的意思是：如果 A 是"真"就用 A，否则用 B
        """

    def register_tool(self, tool: Tool) -> None:
        """
register_tool
把工具对象存到 self.tools 字典里
key 是工具的名字，value 是工具本身
以后 Agent 想用工具时，通过名字就能找到
        """
        self.tools[tool.name] = tool

    def add_message(self, role: str, content: str, **meta: Any) -> None:
        """
add_message
创建一条 Message 对象，追加到消息列表末尾
**meta 的意思是：可以传任意数量的额外参数，都会被装进 meta 字典里
比如 add_message("user", "你好", turn=1)，turn=1 就会存到 meta 里
        """
        self.messages.append(Message(role=role, content=content, meta=meta))

    def remember(self, text: str) -> None:
        """
remember
往记忆列表里加一条文本
目前只是存着，还没有用到
        """
        self.memory.append(text)

    def can_use_tool(self, tool_name: str) -> bool:
        """
can_use_tool
检查某个工具是否已注册
tool_name in self.tools：如果工具名在字典的 key 里，返回 True，否则 False
        """
        return tool_name in self.tools

    def load_skills(self, skills_dir: str | Path) -> list[str]:
        skills_path = Path(skills_dir)
        """
Path(skills_dir)：把传进来的路径转成 Python 的 Path 对象
Path 对象比字符串更方便操作文件路径
比如你传 "skills"，它就变成 Path("skills")
        """
        if not skills_path.exists():
            return []

        loaded: list[str] = []
        """
创建一个空列表，用来装找到的技能名字
        """
        for path in sorted(skills_path.rglob("SKILL.md")):
            """
skills_path.rglob("SKILL.md")
rglob 是 Path 对象的方法，意思是"递归地搜索"
"SKILL.md" 是要找的文件名
整个意思：在 skills_path 文件夹里，一层一层地往下翻，找到所有叫 SKILL.md 的文件

sorted(...)
把找到的结果按字母顺序排序
这样每次运行顺序都一样，不会乱跳

for path in ...
遍历找到的每一个 SKILL.md 文件
每找到一个，就临时叫它 path
            """
            loaded.append(path.parent.name)
        return loaded

    def call_llm_stream(self) -> Iterator[str]:
        """
调用当前绑定的 LLM 客户端（默认是 FakeLLMClient）
把当前所有消息历史 self.messages 传给它
返回流式文本（一段一段的字符串）
        """
        return self.llm.stream_text(self.messages)

    # 模型单步执行
    def model_step(self) -> dict[str, Any]:
        chunks = list(self.call_llm_stream())
        return {
            "type": "message",
            "content": "".join(chunks),
            "chunks": chunks,
        }

    # 方法定义 + 记录用户输入
    def run(self, user_input: str) -> str:
        self.add_message("user", user_input)

        # 循环 + 调用模型
        for turn in range(self.max_turns):
            step = self.model_step()

            # 处理"直接回复"的情况
            if step["type"] == "message":
                """
step["type"] == "message"：判断 LLM 返回的类型是不是 "message"
"message" 表示"LLM 决定直接回复，不调用工具"

content = step["content"]
从 step 里取出 LLM 生成的回复内容

self.add_message("assistant", content, turn=turn, chunks=step.get("chunks", []))
把 LLM 的回复记录到消息历史
角色是 "assistant"（AI 助手）
额外记下这是第几轮（turn=turn）和原始文本片段（chunks）

return content
把回复内容返回给调用者
return 会直接结束整个 run 方法，不再继续循环
                """
                content = step["content"]
                self.add_message(
                    "assistant",
                    content,
                    turn=turn,
                    chunks=step.get("chunks", []),
                )
                return content

            # 处理工具调用
            if step["type"] == "tool_call":
                """
step["type"] == "tool_call"：判断 LLM 返回的类型是不是 "tool_call"
"tool_call" 的意思是"LLM 想用工具"

tool_name = step["tool"]
取出 LLM 想用的工具名字

tool_input = step.get("input", {})
取出传给工具的参数字典，如果没有参数就默认给空字典
                """
                tool_name = step["tool"]
                tool_input = step.get("input", {})

                if not self.can_use_tool(tool_name):
                    """
self.can_use_tool(tool_name) 返回 True（工具存在）或 False（不存在）
not 取反：如果工具不存在，就执行里面的代码

error = f"Tool not allowed or not found: {tool_name}"
拼接错误提示信息
f"..." 是 f-string，{tool_name} 会被替换成实际的工具名

self.add_message("tool_result", error, ok=False, tool=tool_name)
把错误信息记录到消息历史
角色是 "tool_result"（工具执行结果）
ok=False 表示工具执行失败

continue
跳过本轮剩余代码，直接进入下一轮循环
让 Agent 知道工具不可用，重新思考
                    """
                    error = f"Tool not allowed or not found: {tool_name}"
                    self.add_message(
                        "tool_result",
                        error,
                        ok=False,
                        tool=tool_name,
                    )
                    continue

                # 执行工具并记录结果
                result = self.tools[tool_name].call(tool_input)
                """
self.tools[tool_name]：从工具字典里，通过名字找到对应的 Tool 对象
.call(tool_input)：调用这个工具的 call 方法，传入参数
返回的 result 是一个 ToolResult 对象
                """
                self.add_message(
                    "tool_result",
                    result.content,
                    ok=result.ok,
                    tool=tool_name,
                    tool_input=tool_input,
                    tool_meta=result.meta,
                )
                """
把工具的执行结果记录到消息历史
角色是 "tool_result"
内容是工具返回的文本（result.content）
                """
                continue

            # 处理未知返回类型
            """
如果 LLM 返回的 type 既不是 "message"，也不是 "tool_call"
说明出现了意料之外的情况
raise 是"抛出异常"，会直接让程序报错停止
这样就不会静默地忽略错误，让开发者能立即发现问题
            """
            raise ValueError(f"Unknown step type: {step['type']}")

        # 超出最大轮次的处理
        """
如果 for 循环正常结束了（没有 return），说明 max_turns 轮都用完了，Agent 还没返回结果
        """
        final_text = "Agent stopped because it reached max_turns."
        self.add_message("assistant", final_text)
        return final_text
