# 当前项目状态

这份文档记录当前已经确定下来的项目形态，方便后续继续改造。

## 当前目标

先把项目整理为一个直观、可运行、可继续扩展的金融风控 RAG / Agent 项目。当前阶段不重写核心算法，优先解决入口分散、配置不统一、文件命名不直观的问题。

## 已确定模块

### 1. Agent 主应用

路径：`agent/`

定位：面向最终演示的主入口。

能力：

- 使用 DeepSeek Chat 作为底层 LLM。
- 使用 Tool Calling Agent 编排工具。
- 工具包括本地 RAG 知识库、精确数学计算器、联网搜索。
- 使用 Streamlit 提供聊天界面。
- 使用 SQLite 保存多会话历史。

启动：

```powershell
.\start_agent.ps1
```

### 2. 纯 RAG 应用

路径：`rag_chat_app/`

定位：用于验证和展示 RAG 管线本身。

能力：

- 父子切片层级检索。
- Chroma 本地向量库。
- BGE 中文 embedding。
- Flashrank rerank。
- 多轮历史感知 query rewrite。
- 流式输出。
- 来源片段展示。

启动：

```powershell
.\start_rag.ps1
```

### 3. 风险评估资料

路径：`risk_eval/`

当前状态：仅包含 `Data Dictionary.xls`。它像是消费金融违约预测数据集的字段说明，尚未接入应用逻辑。

后续可能方向：

- 做成轻量信用评分/违约概率预测模块。
- 将字段解释接入 RAG 知识库。
- 做一个“上传客户指标 -> 输出风险解释”的工具，供 Agent 调用。

## 当前技术基座

- Web UI：Streamlit
- LLM：DeepSeek Chat（OpenAI compatible API）
- Agent：LangChain tool-calling agent
- RAG：LangChain retrieval chain
- Vector DB：Chroma
- Embedding：BAAI/bge-small-zh-v1.5
- Rerank：Flashrank ms-marco-MiniLM-L-12-v2
- Session：SQLite
- Evaluation：RAGAS + Qwen

## 已整理内容

- 根目录新增 `README.md`，作为统一入口。
- 根目录新增 `.env.example`，统一 API Key 配置方式。
- 根目录新增 `requirements.txt`，替换不可移植的冻结依赖。
- 根目录新增 `start_agent.ps1` 和 `start_rag.ps1`。
- 新增共享官方知识库目录 `knowledge_base/`，两套应用都开始读取同一份地基。
- `rag_chat_app/test.py` 重命名为 `rag_pipeline.py`。
- 保留 `rag_chat_app/test.py` 作为兼容入口。
- 统一 RAG 缓存、向量库、知识库路径，避免受运行目录影响。
- SQLite `journal_mode` 改为 `WAL`。

## 当前已知边界

- 当前环境中 `python` 指向 Windows Store alias，无法直接执行项目。
- 首次运行需要网络下载 HuggingFace/Flashrank 模型。
- 没有 API Key 时，RAG 初始化可以建库，但生成回答会失败。
- `agent/` 与 `rag_chat_app/` 仍存在重复代码，例如数据库和会话管理。
- 旧的两份 `knowledge.md` 仍然保留，但已不再是默认入口；后续可逐步归档。
- `risk_eval/` 还没有形成可运行模块。

## 下一步候选改造

1. 抽出共享模块：`common/database.py`、`common/session_manager.py`、`common/rag_base.py`。
2. 继续扩展 `knowledge_base/`，补足 NFRA / PBC 的官方监管文件。
3. 增加本地 smoke test：验证配置、文件路径、SQLite、RAG pipeline import。
4. 接入 `risk_eval`：做一个轻量违约风险评分工具，并挂到 Agent。
5. 优化 UI：让 Agent 页面更明确展示“本地知识库 / 计算 / 联网搜索”的调用轨迹。
6. 增加项目截图和架构图，提升 GitHub 可读性。
