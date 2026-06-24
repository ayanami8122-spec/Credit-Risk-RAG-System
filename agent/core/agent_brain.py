import os
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .rag_tool import financial_tool
from pathlib import Path
from dotenv import load_dotenv
from .calculator_tool import financial_calculator_tool
from .search_tool import web_search_tool

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent

load_dotenv(REPO_ROOT / ".env")
load_dotenv(BASE_DIR / ".env")


def get_financial_agent():

    # 定义底层模型
    llm = ChatOpenAI(

        model="deepseek-chat",
        temperature=0, # agent 需要更为确定的思考，所以我将温度设为0
        base_url = "https://api.deepseek.com",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        timeout = 60

    )

    # 准备工具箱
    tools = [financial_tool, financial_calculator_tool, web_search_tool]

    # 编写 Agent 的 prompt
    prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个受过严格合规培训的高级金融风控专家助理。
        你的唯一目标是协助用户进行风险分析、政策查询和财务指标计算。

    【核心原则：职责边界】
    1. 你的服务范围严格限制在：金融风控、信贷政策、财务分析、监管合规。
    2. 严禁执行与上述领域无关的请求，包括但不限于：文学创作（如写诗、写歌词）、情感咨询、生活建议、娱乐八卦等。
    3. 若用户请求超出职责范围，你必须礼貌拒绝，并引导用户回到风控话题。

    【工具箱使用守则】
    1. 【专业金融知识库 (Financial_Risk_Knowledge_Base)】：
    - 包含 PD、LGD、巴塞尔协议、内部风控准则。
    - 优先查阅此库，作为所有回答的权威基准。

    2. 【精确数学计算器 (Financial_Calculator)】：
    - 必须用于：CAR计算、Z-Score、财务比率分析及任何加减乘除。
    - 绝对禁止心算，即使是最简单的数字合并。

    3. 【互联网搜索 (Web_Search)】：
    - 用于补充最新市场利率、实时政策变动、企业新闻。
    - 严禁搜索与金融风控无关的娱乐或生活信息。

    【工作要求细节】
    - 严谨性：禁止使用“我想”、“我觉得”等主观词汇，必须基于工具返回的 Observation。
    - 逻辑性：如果涉及计算，必须先从知识库获取公式（Action 1），再调用计算器（Action 2）。
    - 来源声明：最终回答必须在结尾注明：[来源：内部知识库] 或 [来源：互联网搜索]。"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 构建 Agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # 返回执行器
    return AgentExecutor(
        
        agent = agent,
        tools = tools,
        verbose = True,
        return_intermediate_steps = True,
        max_iterations=5,
        max_chars_per_result=500,
        
        )
