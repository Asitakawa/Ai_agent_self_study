from typing import Iterator, Protocol
from my_agent.models import Message


class LLMClient(Protocol):
    #Protocol：Python 的"协议"类型。它定义了一个接口规范，任何实现了协议中方法的类，都自动被视为该协议的实现
    """
    定义了一个协议类
任何类只要实现了 stream_text 方法，就自动被认为是 LLMClient 的实现
    """
    def stream_text(self, messages: list[Message]) -> Iterator[str]:
       """
       输入：一个 Message 列表（当前对话历史）
输出：Iterator[str]——一个字符串迭代器，每次 yield 一段文本
为什么用迭代器？因为 AI 模型的回复是"流式"的，一个字一个字或一段一段返回，而不是等全部生成完再一次性返回。这样用户体验更好，不用干等
       """

class FakeLLMClient:

    def stream_text(self,messages: list[Message])-> Iterator[str]:
        last_user_messages=next(
            (m.content for m in reversed(messages) if m.role=="user"),
            "",
        )
        response = f"[fake-llm]你刚才说的是：{last_user_messages}"
        for chunk in self._chunk_text(response,size=8):
            """
            在类的方法内部调用同一个类的其他方法时，通常都用 self.xxx()
            普通方法（有 self）→ 必须用 self.xxx()
静态方法（@staticmethod，没 self）→ 可以用 self.xxx() 或 类名.xxx()，都行
如FakeLLMClient._chunk_text(response,size=8):
            """
            yield chunk

    @staticmethod
    def _chunk_text(text: str, size: int) -> Iterator[str]:
        for i in range(0, len(text), size):
            yield text[i : i + size]

