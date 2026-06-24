# 金融风险控制 Agent 智能助手


> 本次更新备注（2026-06-24）：
> - 重新整理 `agent/app.py` 的页面结构，使其更像金融工具台
> - 保留会话管理、工具调用过程和聊天主流程
> - 降低视觉噪音，突出主入口与结果展示
> - 以 `agent` 作为主产品入口，`rag_chat_app` 作为独立 RAG 参考实现保留

https://github.com/user-attachments/assets/06528c89-55c9-49b3-96ad-75b53c357545




基于 **LangChain Agent + RAG** 的金融风控决策 Web 应用。在 `rag_chat_app` 的层级检索与重排能力之上，引入 **Tool Calling Agent**，让模型按需调用本地知识库、数学计算器与互联网搜索，完成「查制度 → 算指标 → 补实时信息」的复合任务。

## 核心特性

| 能力 | 说明 |
|------|------|
| **多工具 Agent** | DeepSeek-Chat + `create_tool_calling_agent`，自动选择工具并展示 `intermediate_steps` |
| **共享官方 RAG 知识库** | `ParentDocumentRetriever` + Flashrank 重排，知识来源为 `knowledge_base/` |
| **精确计算** | `numexpr` 驱动 `Financial_Calculator`，避免 LLM 心算幻觉 |
| **联网搜索** | DuckDuckGo（需安装 `ddgs`），用于时效性政策/新闻类问题 |
| **多会话持久化** | Streamlit 侧边栏 + SQLite（`chat_history.db`） |

## 技术栈

- **大模型**：DeepSeek-Chat（`langchain-openai`，兼容 OpenAI API）
- **Agent 框架**：`langchain-classic`（`AgentExecutor` / `create_tool_calling_agent`）
- **向量模型**：`BAAI/bge-small-zh-v1.5`（HuggingFace 本地缓存）
- **向量库**：Chroma DB
- **重排**：Flashrank（`ms-marco-MiniLM-L-12-v2`）
- **前端**：Streamlit
- **评测脚本**：`eval_agent.py`（ReAct + 通义千问 `qwen-plus`，Mock 工具）

## 项目结构

```text
agent/
├── app.py                    # Streamlit 主入口
├── database.py               # SQLite 会话与消息持久化
├── session_manager.py        # 多会话状态管理（与 DB 同步）
├── ../knowledge_base/        # 共享官方知识库（RAG 数据源）
├── eval_agent.py             # ReAct Agent 离线评测（Qwen + Mock 工具）
├── core/
│   ├── agent_brain.py        # Agent 组装：LLM + Prompt + Executor
│   ├── rag_tool.py           # RAG 流水线 + Financial_Risk_Knowledge_Base 工具
│   ├── calculator_tool.py    # Financial_Calculator 工具
│   └── search_tool.py        # Web_Search 工具（DuckDuckGo）
├── chroma_db/                # 向量库（首次启动自动生成，已 gitignore）
├── parent_store/             # 父文档 KV 存储（自动生成）
├── hf_cache/                 # Embedding 模型缓存（自动生成）
├── flashrank_models/         # 重排模型（自动生成）
└── chat_history.db           # 对话历史（自动生成）
```

环境变量文件建议放在仓库根目录 `.env`（`agent_brain.py` 会加载），也可以放在 `agent/.env` 作为子项目局部配置。

## 系统架构

```text
┌─────────────────────────────────────────────────────────────┐
│  Streamlit UI (app.py)                                       │
│  · 多会话侧边栏 · 工具调用过程展示 · SQLite 历史同步          │
└──────────────────────────┬──────────────────────────────────┘
                           │ invoke({ input, chat_history })
┌──────────────────────────▼──────────────────────────────────┐
│  AgentExecutor (core/agent_brain.py)                         │
│  DeepSeek-Chat · max_iterations=5 · return_intermediate_steps│
└───┬─────────────────┬─────────────────────┬─────────────────┘
    │                 │                     │
    ▼                 ▼                     ▼
 Financial_Risk    Financial_           Web_Search
 Knowledge_Base    Calculator           (DuckDuckGo)
 (rag_tool)        (numexpr)
    │
    ▼
 ParentDocumentRetriever → Flashrank Rerank → DeepSeek 生成答案
 (Chroma + bge-small-zh + knowledge.md)
```

## 快速开始

### 1. 环境准备

建议使用你自己的 Python 虚拟环境或 Conda 环境，然后安装仓库根目录依赖：

```bash
cd ..
pip install -r requirements.txt
```

### 2. 配置环境变量

在仓库根目录 **`.env`** 中配置：

```env
OPENAI_API_KEY=你的_DeepSeek_API_Key
HF_ENDPOINT=https://hf-mirror.com
```

可选（仅运行 `eval_agent.py` 时需要）：

```env
QWEN_API_KEY=你的_通义千问_API_Key
```

> 不要将真实密钥提交到 Git。`.env` 已在 `.gitignore` 中忽略。

### 3. 准备知识库

确保仓库根目录下的 `knowledge_base/` 存在且为 UTF-8 编码 Markdown。首次启动会自动：

1. 切片并写入 `chroma_db/`
2. 下载/缓存 Embedding 到 `hf_cache/`
3. 下载 Flashrank 模型到 `flashrank_models/`

首次冷启动可能较慢，属正常现象。

### 4. 启动 Web 应用

```bash
cd agent
streamlit run app.py
```

浏览器访问：`http://localhost:8501`

### 5. 运行 Agent 评测脚本

```bash
cd agent
python eval_agent.py
```

该脚本使用 **Mock 搜索/RAG 工具** + **ReAct 推理链**，用于验证 Agent 是否按规则依次调用工具，不依赖完整 RAG 索引。

## Agent 工具说明

| 工具名 | 文件 | 适用场景 |
|--------|------|----------|
| `Financial_Risk_Knowledge_Base` | `core/rag_tool.py` | PD、LGD、巴塞尔协议、内部制度等专业概念 |
| `Financial_Calculator` | `core/calculator_tool.py` | CAR、Z-Score 等需代入数字的公式计算 |
| `Web_Search` | `core/search_tool.py` | 实时政策、新闻、市场利率等时效信息 |

Agent 策略（见 `agent_brain.py` 系统提示）：**优先本地知识库 → 必要时联网 → 计算类问题必须走计算器**。

## LangChain 1.x 导入说明

本项目使用 **LangChain 1.3+**。旧版 API 已迁移，请勿从 `langchain.agents` 直接导入：

```python
# 正确
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import Tool

# 错误（会 ImportError）
from langchain.agents import AgentExecutor
from langchain.tools import Tool
```

## 常见问题

**Q: `ImportError: cannot import name 'AgentExecutor'`**  
A: 安装 `langchain-classic`，并按上文修改导入路径。

**Q: `Could not import ddgs`**  
A: 执行 `pip install -U ddgs`，或暂时从 `agent_brain.py` 的 `tools` 列表中移除 `web_search_tool`。

**Q: 后端初始化失败 / 找不到知识库**  
A: 确认仓库根目录下的 `knowledge_base/` 存在；检查 `OPENAI_API_KEY` 与网络（DeepSeek API、HuggingFace 镜像）。

**Q: 修改 `knowledge.md` 后检索结果未更新**  
A: 删除 `chroma_db/` 与 `parent_store/` 后重启，触发重新建库。

## 与 `rag_chat_app` 的关系

| 项目 | 定位 |
|------|------|
| `rag_chat_app/` | 纯 RAG 对话 + Ragas 评测 + 流式输出 |
| `agent/` | 在 RAG 之上封装 **多工具 Agent**，支持计算与联网 |

两者共享相似的 RAG 管线设计（父子切片、Flashrank、Chroma），`agent` 将 RAG 封装为 Agent 可调用的单一工具。

## 许可证

仅供学习与内部演示使用。部署生产环境前请自行审查 API 密钥管理、数据合规与模型许可。
