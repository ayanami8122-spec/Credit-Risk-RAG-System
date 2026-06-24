"""RAG pipeline assembly and streaming helpers for the Streamlit app."""

import os
import sys
from collections.abc import Iterator
from io import TextIOWrapper
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

CHROMA_DB_DIR = str(BASE_DIR / "chroma_db")
HF_CACHE_DIR = str(BASE_DIR / "hf_cache")
FLASHRANK_CACHE_DIR = str(BASE_DIR / "flashrank_models")
PARENT_STORE_DIR = str(BASE_DIR / "parent_store")
KNOWLEDGE_DIR = REPO_ROOT / "knowledge_base"

load_dotenv(REPO_ROOT / ".env")
load_dotenv(BASE_DIR / ".env")

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import (ContextualCompressionRetriever,ParentDocumentRetriever)
from langchain_classic.storage import  LocalFileStore, create_kv_docstore
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from flashrank import Ranker
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain,create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder




# 读取共享官方知识库逻辑
def load_knowledge() -> list[Any] | None:
    if not KNOWLEDGE_DIR.exists():
        print(f"错误：未找到知识库目录 {KNOWLEDGE_DIR}")
        return None

    markdown_files = sorted(KNOWLEDGE_DIR.rglob("*.md"))
    if not markdown_files:
        print(f"错误：知识库目录 {KNOWLEDGE_DIR} 中未找到 Markdown 文档")
        return None

    docs: list[Any] = []
    for file_path in markdown_files:
        try:
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs.extend(loader.load())
        except FileNotFoundError:
            print(f"错误：未找到 {file_path} 文件，请先创建它或检查文件名")
            return None
    return docs


def get_rag_chain():
    """
    此时我们将其完全改动为后端函数，供给前端调用，我们在本脚本内部完成：
    向量初筛 -> rerank重排 -> 语义回溯 -> 生成回答
    """
    raw_docs = load_knowledge()
    if not raw_docs:
        raise FileNotFoundError("未找到信用风险知识库文档。")

    # 定义父切片器（大块）和子切片器（小块）
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    
    # 初始化本地向量模型
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5", cache_folder=HF_CACHE_DIR)

    # 初始化/加载本地 Chroma 数据库
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    try:
        chroma_doc_count = vectorstore._collection.count()
    except Exception:
        chroma_doc_count = 0
    db_exists = chroma_doc_count > 0

    # 初始化用于存放父块完整文本的内存存储器
    fs = LocalFileStore(PARENT_STORE_DIR)
    docstore = create_kv_docstore(fs)

    # 构建父子块检索器
    base_retriever = ParentDocumentRetriever(
        vectorstore = vectorstore,
        docstore=docstore,
        child_splitter = child_splitter,
        parent_splitter = parent_splitter,
        search_kwargs={"k": 15},
        id_key="doc_id"
    )

    # 只在没有本地向量库文件夹时才进行切片与高维向量注入
    if not db_exists:
        # 将最原始的 raw_docs 喂进去，Retriever 会自动触发 parent_splitter & child_splitter
        # 将子块算成向量存入 Chroma，把父块存入 docstore，并自动绑定其ID映射
        base_retriever.add_documents(raw_docs, ids=None)

    # 初始化 flashrank_client 
    flashrank_client = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=FLASHRANK_CACHE_DIR)
    compressor = FlashrankRerank(client=flashrank_client, top_n=5)

    # 组装检索器
    retriever = ContextualCompressionRetriever(base_compressor=compressor,base_retriever=base_retriever)

    # 组装 RAG 问答链中的底层 LLM 驱动
    llm = ChatOpenAI(
        model="deepseek-chat",
        temperature=0.2,
        base_url="https://api.deepseek.com",
        streaming=True,
    ) # type: ignore

    # 书写问题重写的 Prompt
    # 作用是在有历史对话时，让LLM结合历史把用户的新问题重写为一个独立、适合向量检索的问题
    contextualize_q_system_prompt = (
        "给出一句话和对话历史，如果它引用了历史对话中的上下文，"
        "请将其重写为一个可以独立理解的、适合检索的、简明扼要的问题。"
        "切记：不要回答问题，只需重写或原样返回它。"
    )   

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"), # 预留给历史对话
        ("human", "{input}")
    ]
    )

    # 组装历史感知检索器
    # 作为一个包装器在收到请求后先走 LLM +  contextualize_q_system_prompt 得到新的问题
    # 在用新问题去调用前面的底层
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 书写最终回答prompt
    system_prompt =(
        "你是一个专业的金融风险控制助手。请严格根据以下提供的【已知知识库上下文】内容来回答用户的问题。\n"
        "【核心交互规则】\n"
        "1. 允许模糊匹配：请自动忽略'何为'、'什么是'等修饰词，重点围绕上下文检索核心金融概念。\n"
        "2. 跨语言知识：知识库是中文的，但如果用户用英文术语提问，请自动对齐并用中文清晰回答。\n"
        "3. 严谨底线：如果用户询问的内容与上下文中的金融风控概念完全无关（如天气、娱乐、写代码等），直接回答：‘抱歉，本地知识库中未包含此信息。’\n"
        "4. 如果上下文信息不足，请明确说明知识库未提供完整依据，禁止编造。\n\n"
        "【当前知识库】:\n"
        "{context}"
    )
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"), # 让最终回答的大模型也能看到过去的聊天记录
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


def _answer_piece_to_str(piece: Any) -> str:
    if piece is None:
        return ""
    if isinstance(piece, str):
        return piece
    content = getattr(piece, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(piece)


def _normalize_retrieved_docs(context: Any) -> list[Any]:
    if not context:
        return []
    if isinstance(context, list):
        return [doc for doc in context if getattr(doc, "page_content", None)]
    return []


def _notify_retrieval_done(
    on_retrieval_done: Any | None,
    context: Any,
    notified: bool,
) -> bool:
    if notified or on_retrieval_done is None:
        return notified
    docs = _normalize_retrieved_docs(context)
    on_retrieval_done(docs)
    return True


def _yield_answer_deltas_from_stream(
    rag_chain: Any,
    question: str,
    chat_history: list | None = None,
    on_retrieval_done: Any | None = None,
) -> Iterator[str]:
    prev = ""
    retrieval_notified = False

    input_data = {"input": question, "chat_history": chat_history or []}

    for chunk in rag_chain.stream(input_data):
        if "context" in chunk:
            retrieval_notified = _notify_retrieval_done(
                on_retrieval_done, chunk.get("context"), retrieval_notified
            )
        if "answer" not in chunk:
            continue
        text = _answer_piece_to_str(chunk["answer"])
        if not text:
            continue
        if text.startswith(prev):
            delta = text[len(prev) :]
            prev = text
        else:
            delta = text
            prev = prev + text
        if delta:
            yield delta


def iter_answer_deltas(
    rag_chain: Any,
    question: str,
    chat_history: list | None = None,
    on_retrieval_done: Any | None = None,
) -> Iterator[str]:
    """产出可交给 st.write_stream 的增量 answer 文本。

    说明：当前 LangChain 的 retrieval_chain (RunnableSequence) 不支持
    同步 stream_events(version='v2')，因此统一走 stream() 并在 context 分片
    回调检索文档。
    """
    yield from _yield_answer_deltas_from_stream(
        rag_chain, question, chat_history, on_retrieval_done
    )


def main():
    # windows控制中文乱码与字符集崩溃
    if sys.platform.startswith("win"):
        os.system('chcp 65001 > nul')
        cast(TextIOWrapper, sys.stdin).reconfigure(encoding='utf-8')
        cast(TextIOWrapper, sys.stdout).reconfigure(encoding='utf-8')

    print(" 正在读取信用风险知识库...")

    # 调用上方函数获取完整的 RAG 链条
    try:
        rag_chain = get_rag_chain()
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    print("==================================================")
    print("  RAG 知识库系统加载成功！（控制台调试模式）")
    print("==================================================\n")

    try:
        while True:
            user_input = input("你的问题是：").strip()

            if user_input.lower() in ['exit', 'quit']:
                print("\n 程序已安全退出。")
                sys.exit(0)
            
            if not user_input:
                continue

            try:
                print("\n [正在检索知识库并思考...]")
                print("AI 回复：\n", end="", flush=True)
                for token in iter_answer_deltas(rag_chain, user_input):
                    print(token, end="", flush=True)
                print("\n\n" + "=" * 40 + "\n")
            except Exception as api_error:
                print(f"\n[API 请求失败]: {api_error}\n")
        
    except (KeyboardInterrupt, EOFError):
        print("\n\n[检测到 Ctrl+C 强行终止信号]，程序已安全退出。")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

if __name__ == "__main__":
    main()
