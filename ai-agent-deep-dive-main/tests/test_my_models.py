from my_agent.models import Message, ToolResult

def test_create_message():
    msg = Message(role="user", content="你好")
    assert msg.role == "user"
    assert msg.content == "你好"
    assert msg.meta == {}


def test_create_tool_result():
    result = ToolResult(ok=True, content="成功")
    assert result.ok is True
    assert result.content == "成功"


def test_tool_result_failed():
    result = ToolResult(ok=False, content="出错了")
    assert result.ok is False