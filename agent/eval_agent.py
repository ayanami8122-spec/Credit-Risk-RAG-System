from langchain_community.llms import Tongyi
import dashscope
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
import os
from dotenv import load_dotenv


load_dotenv()
dashscope.api_key = os.getenv("QWEN_API_KEY")


# 1. 配置 Qwen-Plus 模型
# 请确保环境变量中已设置 DASHSCOPE_API_KEY
llm = Tongyi(model_name="qwen-plus", temperature=0,dashscope_api_key=os.getenv("QWEN_API_KEY"))

# 2. 定义 Agent 的“手”（工具集）
def mock_search_tool(query):
    return f"""
    查询关键词：{query}

    企业名称：XX地产

    事件：
    2024年6月发生债务违约

    金额：
    50亿元
    """

def mock_rag_tool(query):
    return f"""
    查询关键词：{query}

    风控手册第12条：

    出现以下情况暂停授信：

    1. 债务违约
    2. 流动性危机
    3. 重大监管处罚
    """

tools = [
    Tool(
        name="RealTimeSearch",
        func=mock_search_tool,
        description="""
        用于查询：

        - 企业新闻
        - 财务风险事件
        - 违约事件
        - 监管处罚
        - 舆情风险

        不能提供内部制度信息。
        """
    ),
    Tool(
        name="InternalRiskRAG",
        func=mock_rag_tool,
        description="""
        用于查询：

        - 风控手册
        - 授信制度
        - 内部审批规则
        - 风险处置流程

        不能提供实时新闻。
        """
        )
]

# 3. 构建 Agent 的“大脑”提示词（ReAct 模式）
template = """
你是一名银行信贷风控分析助手。

你的核心职责：

1. 识别企业风险事件
2. 查询公司内部风控制度
3. 根据风险事件和内部制度给出风险处置建议
4. 所有结论必须基于工具返回的信息，不允许编造

你拥有以下工具：

{tools}

可用工具名称：

[{tool_names}]

====================
工具使用规则
====================

1. 涉及企业新闻、市场动态、风险事件、债务违约、监管处罚、舆情风险等外部信息时，
必须优先使用 RealTimeSearch。

2. 涉及风控手册、授信制度、审批规则、风险处置流程等内部制度时，
必须使用 InternalRiskRAG。

3. 如果问题同时涉及：
   - 风险事件
   - 内部制度

则必须先调用 RealTimeSearch，
再调用 InternalRiskRAG，
获取完整信息后才能给出最终结论。

4. 不允许跳过工具直接回答。

5. 不允许使用自身知识补充不存在于工具结果中的事实。

6. 当工具已经提供足够信息时，应立即给出 Final Answer，
不要无限循环调用工具。

====================
推理格式要求
====================

严格按照以下格式输出：

Question: 用户问题

Thought: 你当前的思考

Action: [{tool_names}] 中的一个工具

Action Input: 传递给工具的输入

Observation: 工具返回结果

（以上步骤可以重复多次）

Thought: 我已经掌握足够信息

Final Answer: 最终回答

====================
重要要求
====================

- Action 必须严格使用工具名称。
- 每次只能调用一个工具。
- 必须先获得 Observation 后才能继续推理。
- 不允许输出格式之外的内容。
- 最终必须以 Final Answer 结束。
- 如果问题同时涉及风险事件和内部制度，
  必须依次调用两个工具后再给出 Final Answer。

====================
开始任务
====================

Question: {input}

Thought: {agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)

# 4. 初始化 Agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, # 关键：开启 verbose 才能在终端看到推理过程（即可视化）
    handle_parsing_errors=False
)

# 5. 测试用例
test_query = "最近某地产企业有风险传闻吗？如果真的有风险，按照公司规定我们该怎么办？"

print("--- 开始测试 Agent 推理链路 ---")
response = agent_executor.invoke({"input": test_query})
print("\n--- 最终输出 ---")
print(response["output"])