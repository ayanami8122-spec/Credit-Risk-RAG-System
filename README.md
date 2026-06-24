# Credit Risk RAG System

金融风控领域的轻量级 RAG / Agent 项目。项目包含一个纯 RAG 对话应用、一个多工具 Agent 应用，以及一份信用违约建模字段字典，适合作为金融风控知识库问答与 Agent 编排的演示基座。

## 项目定位

本项目当前分为三层：

| 模块 | 路径 | 作用 |
| --- | --- | --- |
| Agent 应用 | `agent/` | 推荐主入口。在 RAG 知识库之上加入计算器与联网搜索工具，由 Agent 自行选择工具。 |
| RAG 应用 | `rag_chat_app/` | 纯 RAG 对话应用，支持历史感知检索、流式输出、来源片段展示。 |
| 风险评估资料 | `risk_eval/` | 信用违约建模数据字典，当前只作为资料沉淀，暂未接入应用。 |

## 核心能力

- 共享官方金融风控知识库：基于 `knowledge_base/` 中的 BIS / IFRS 官方材料构建。
- 层级检索：父子切片 + Chroma 向量库。
- 二阶段重排：BGE 召回后使用 Flashrank rerank。
- 多轮上下文：RAG 应用支持 history-aware query rewrite。
- 多工具 Agent：Agent 应用包含知识库、精确计算器、DuckDuckGo 搜索。
- 会话持久化：Streamlit + SQLite 保存多轮会话。

## 推荐启动

### 1. 安装依赖

建议使用 Python 3.10 或 3.11，并在虚拟环境中安装依赖：

```powershell
cd D:\code\Credit-Risk-RAG-System-main
pip install -r requirements.txt
```

### 2. 配置环境变量

复制根目录的 `.env.example` 为 `.env`，填入真实密钥：

```powershell
Copy-Item .env.example .env
```

至少需要：

```env
OPENAI_API_KEY=你的_DeepSeek_API_Key
HF_ENDPOINT=https://hf-mirror.com
```

如果运行评测脚本，还需要：

```env
QWEN_API_KEY=你的_通义千问_API_Key
```

### 3. 启动主 Agent 应用

```powershell
.\start_agent.ps1
```

默认访问地址：`http://localhost:8501`

### 4. 启动纯 RAG 应用

```powershell
.\start_rag.ps1
```

默认访问地址：`http://localhost:8502`

## 手动启动命令

如果不使用启动脚本，也可以直接进入对应目录启动。

Agent 应用：

```powershell
cd D:\code\Credit-Risk-RAG-System-main\agent
streamlit run app.py
```

纯 RAG 应用：

```powershell
cd D:\code\Credit-Risk-RAG-System-main\rag_chat_app
streamlit run app.py --server.port 8502
```

## 首次运行说明

首次启动会自动下载或生成以下本地资源：

- `hf_cache/`：Embedding 模型缓存。
- `flashrank_models/`：Flashrank 重排模型缓存。
- `chroma_db/`：Chroma 向量库。
- `parent_store/`：父文档映射存储。
- `chat_history.db`：本地会话历史。

这些文件均属于本地产物，不建议提交到 Git。

## 目录结构

```text
Credit-Risk-RAG-System-main/
├── README.md
├── docs/
│   └── CURRENT_STATE.md
├── requirements.txt
├── .env.example
├── start_agent.ps1
├── start_rag.ps1
├── knowledge_base/
│   ├── 00_manifest.md
│   ├── 01_basel_iii_finalising_post_crisis_reforms.md
│   ├── 02_guidance_on_credit_risk_and_ecl.md
│   ├── 03_ifrs_9_financial_instruments.md
│   └── 90_official_sources_to_add_next.md
├── agent/
│   ├── app.py
│   ├── knowledge.md
│   ├── database.py
│   ├── session_manager.py
│   └── core/
│       ├── agent_brain.py
│       ├── rag_tool.py
│       ├── calculator_tool.py
│       └── search_tool.py
├── rag_chat_app/
│   ├── app.py
│   ├── rag_pipeline.py
│   ├── test.py
│   ├── knowledge.md
│   ├── eval_rag.py
│   └── verify_stream.py
└── risk_eval/
    └── Data Dictionary.xls
```

## 后续整理方向

- 将 `agent/` 与 `rag_chat_app/` 的重复数据库和会话代码抽成共享模块。
- 继续补充 `knowledge_base/`，优先加入 NFRA / PBC 的官方监管文件。
- 把 `risk_eval/` 接入为可解释的信用风险评分或轻量建模模块。
- 增加无 API Key 的本地 smoke test，降低新环境调试成本。
- 变更前先看 `docs/AI_BOUNDARY.md`，避免无边界扩功能。
