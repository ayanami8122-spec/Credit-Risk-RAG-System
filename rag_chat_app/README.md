

https://github.com/user-attachments/assets/9a04ef61-6269-4855-af2f-7305e846e438



# 🛡️ 金融风险控制 RAG 智能助手 (rag_chat_app)

基于 LangChain 核心生态与 DeepSeek API 打造的工业级 RAG（检索增强生成）全栈 Web 应用。本项目专为金融风控等高壁垒知识库设计，目的是为了解决大模型在垂直领域对话中“找不准”、“讲不深”和“忘上文”的三大痛点。

## 1、核心特性与架构演进

本系统在底层逻辑上实现了“三位一体”的检索闭环：

* **Hierarchical 层级检索（找得深）**：摒弃单一尺寸文本切片，采用“父-子”双层绑定关系。系统将子切片（300字符）进行高维向量化用于精准匹配，命中后通过本地 KV 存储器自动回溯捞出完整的父切片（1200字符），保障了大模型能看懂完整的因果传导链条。
* **Flashrank 交叉重排（找得准）**：引入 Cross-Attention 机制作为第二道过滤器。在本地 bge-small 模型粗筛出前 15 个卡片后，利用 Rerank 机制对问题与 Chunk 进行全文本交互打分，精准剔除表面文字相似的噪声，保留前 5 块核心材料。
* **多轮上下文感知（听得懂）**：引入 `create_history_aware_retriever` 包装器。在查库前，系统会利用独立的大模型 Prompt，结合过去的对话历史对用户的新问题进行“指代消解”与重写，完美支持用户在多轮会话中的连续递进式追问。

## 2、技术栈

* **大语言模型**：DeepSeek-Chat API (流式输出，高性价比)。
* **业务框架**：LangChain (LCEL 表达式编排)。
* **向量模型**：`BAAI/bge-small-zh-v1.5` (本地 HuggingFace 缓存运行，快速响应)。
* **向量数据库**：Chroma DB (本地持久化)。
* **重排模型**：`ms-marco-MiniLM-L-12-v2` (基于 Flashrank 极速本地重排)。
* **前端 UI**：Streamlit (全局资源缓存隔离后端重型加载)。
* **状态与持久化**：SQLite3 (`chat_sessions` 和 `chat_messages` 双表关联，外键级联删除)。

## 3、项目结构

```text
rag_chat_app/
├── app.py                 # Streamlit 前端主入口与两阶段状态提交流程
├── rag_pipeline.py        # LangChain 后端核心流水线构建与 LLM 流式迭代器
├── test.py                # 兼容旧命令的轻量入口
├── database.py            # SQLite 数据库表初始化与 SQL 交互逻辑
├── session_manager.py     # 前端 UI 与数据库之间的状态管理、UUID 分发机制
├── ../knowledge_base/     # 共享官方知识库素材
├── chroma_db/             # Chroma 向量库本地持久化目录（自动生成）
├── parent_store/          # 父级文档映射存储目录（自动生成）
├── chat_history.db        # SQLite 对话历史本地数据库文件（自动生成）
└── .env                   # 可选本地覆盖配置；推荐使用仓库根目录 .env
```


## 4、rag_chat_app 全栈系统架构图

- **前端 UI 层 (Streamlit)**

  - **输入：**  接收用户多轮连续追问（支持指代消解上下文）。
  - **渲染：**  采用 Streamlit `write_stream` 实现流式打字机输出效果。
  - **状态：**  侧边栏多会话持久化切换，配合自动重置焦点逻辑。
- **会话持久层 (SQLite + Session Manager)**

  - **数据模型：**  基于 `chat_sessions` 与 `chat_messages` 的外键级联存储。
  - **逻辑：**  自动清洗系统欢迎语，动态构建格式化对话历史。
- **检索增强后端 (LangChain RAG Pipeline)**

  1. **历史感知重写 (History-Aware Retriever)：**  结合 Chat History 将模糊 Query 消解为语义完备的独立查询。
  2. **向量粗筛 (Chroma DB + BGE-Small)：**  针对 Child Chunks (300 tokens) 进行高频语义检索 (k\=15)。
  3. **层级回溯 (ParentDocumentRetriever)：**  通过 LocalFileStore 找回父级上下文 (1200 tokens)，解决碎片化信息丢失问题。
  4. **交叉重排 (Flashrank Rerank)：**  使用 MS-Marco 模型对候选文本进行精细化打分，精选 Top-5。
  5. **LLM 生成 (DeepSeek API)：**  注入系统风控 Prompt，基于共享官方知识库生成严谨回复。


## 5、RAG 评估结果分析 (Ragas 0.4.3 + Qwen-plus)

根据最新的评估报告，系统在金融风控专业领域的表现如下：[cite: uploaded:rag_eval_results.csv]

- **检索层 (Retriever) 表现优异**：
    - **Context Recall (0.96)**：检索召回率得分较高，说明 BGE-small 模型与分段策略能够精准覆盖 20 个金融风控核心知识点。 
    - **Context Precision (0.88)**：检索精确度高，核心信息在检索结果中排序靠前。 
- **生成层 (Generator) 表现稳健**：
    - **Answer Relevance (0.95)**：回答相关性优秀，DeepSeek 模型能够准确理解并针对性回答复杂的金融意图。 
    - **Faithfulness (0.97)**：忠实度良好。部分得分波动主要源于模型在回答时引入了少许外部预训练知识，而非完全局限于检索到的上下文。 

**优化方向**：
目前的系统呈现“检索极强、生成稳健”的特点。后续可通过在 System Prompt 中进一步强化“封闭域”约束（即严禁使用外部知识），来进一步提升 Faithfulness 得分。
