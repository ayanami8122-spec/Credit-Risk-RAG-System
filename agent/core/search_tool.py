from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool

# 初始化 DuckDuckGo 搜索组件
ddg_search = DuckDuckGoSearchRun()

# 封装为 Agent 可用的工具
web_search_tool = Tool(
    name="Web_Search",
    func=ddg_search.run,
    description="用于在互联网上搜索实时的金融新闻、最新政策或实时市场数据。如果本地知识库查不到，请使用此工具。"
)