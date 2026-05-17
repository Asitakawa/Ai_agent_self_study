from dataclasses import dataclass,field
from typing import Any,Callable

@dataclass
#@dataclass 是装饰器，放在类定义前面，告诉 Python："这是一个数据类，自动帮我生成初始化方法"。
class Message:
    role:str
    content:str
    meta:dict[str,Any]=field(default_factory=dict)
'''
    # meta 是"元数据"，用来存一些额外的信息
    # dict[str, Any] 表示这是一个字典，key 是字符串，value 可以是任何类型
    # field(default_factory=dict) 意思是：如果不传 meta 参数，就默认给一个空字典 {}
    # 为什么要 meta？ 比如工具执行时，可以记录"耗时多少毫秒"、"读取了多少个文件"等额外信息
'''
@dataclass
class ToolResult:
    """工具执行结果"""
    ok:bool
    content:str
    meta:dict[str,Any]=field(default_factory=dict)
    '''
ok: bool：bool 是布尔类型，只有两个值：True（成功）或 False（失败）。Agent 通过这个字段快速判断工具是否执行成功。
content: str：工具返回的具体内容。比如计算器返回 "1+1=2"。
meta: dict[str, Any]：同 Message，存放额外信息。
    '''

class Tool:
    """
    为什么用 class 而不是 @dataclass？
Message 和 ToolResult 只是"装数据的容器"，所以用 @dataclass 最方便
Tool 除了装数据（名字、说明），还需要有行为（call 方法），所以用普通 class 更灵活
    """
    def __init__(
            self,
            name: str,
            description: str,
            handler: Callable[[dict], ToolResult],
    ):
        """
三个参数的解释：
name: str
工具的名字，比如 "calculator"、"echo"
必须是字符串，而且在同一个 Agent 中不能重复
Agent 通过这个名字来找到对应的工具

description: str
工具的说明，告诉别人这个工具是干什么的
比如 "执行数学计算" 或 "读取文件内容"
在真实产品中，这个说明会被发给 AI 模型，让它知道什么时候该用这个工具

handler: Callable[[dict], ToolResult]
这是最难理解的部分，我拆开解释：
Callable = "可调用的"，也就是函数
[[dict], ToolResult] = "接收一个字典参数，返回一个 ToolResult"
合起来意思就是：handler 是一个函数，这个函数的输入是字典，输出是 ToolResult
        """
        self.name= name
        self.description= description
        self.handler= handler

    def call(self, payload: dict)-> ToolResult:
        return self.handler(payload)

