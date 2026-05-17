from my_agent.agent import Agent
from my_agent.models import Tool, ToolResult


def echo_handler(payload):
    text = payload.get("text", "")
    return ToolResult(ok=True, content=f"echo: {text}")


def test_basic():
    """测试 Agent 基本对话"""
    agent = Agent()
    result = agent.run("你好")
    print(f"Agent 回复：{result}")
    assert "[fake-llm]" in result


def test_with_tool():
    """测试 Agent 注册工具"""
    agent = Agent()
    agent.register_tool(Tool("echo", "回显文本", echo_handler))

    # 验证工具已注册
    assert agent.can_use_tool("echo") is True
    assert agent.can_use_tool("不存在的工具") is False

    print("工具注册测试通过")


def test_message_history():
    """测试消息历史"""
    agent = Agent()
    agent.run("第一条消息")
    agent.run("第二条消息")

    print(f"消息总数：{len(agent.messages)}")
    for msg in agent.messages:
        print(f"[{msg.role}] {msg.content[:30]}")


if __name__ == "__main__":
    print("=== 测试 1：基本对话 ===")
    test_basic()

    print("\n=== 测试 2：工具注册 ===")
    test_with_tool()

    print("\n=== 测试 3：消息历史 ===")
    test_message_history()