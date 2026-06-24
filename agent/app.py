import sys
import traceback
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

import database as db
import session_manager as sm
from core.agent_brain import get_financial_agent
from core.rag_tool import warmup_rag


st.set_page_config(
    page_title="金融风险控制 Agent",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


if "rag_warmed" not in st.session_state:
    warmup_rag()
    st.session_state.rag_warmed = True

db.init_db()


@st.cache_resource(show_spinner="正在加载本地知识库与 Agent...")
def load_backend_instance():
    return get_financial_agent()


try:
    financial_agent = load_backend_instance()
except Exception as init_error:
    financial_agent = None
    st.error(f"后端初始化失败: {init_error}")


sm.init_sessions()
current_session_id = st.session_state.current_session_id
current_session = sm.get_current_session()


st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    [data-testid="stSidebar"] {background: #f7f8fa;}
    .app-title {font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem;}
    .app-subtitle {color: #5f6368; margin-bottom: 1rem;}
    .tool-card {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        background: white;
        margin-bottom: 0.75rem;
    }
    .tool-tag {
        display: inline-block;
        padding: 0.12rem 0.45rem;
        border-radius: 999px;
        background: #eef2ff;
        color: #3730a3;
        font-size: 0.8rem;
        margin-right: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("### 会话")
    if st.button("新建会话", use_container_width=True, type="primary"):
        sm.create_new_session()
        st.rerun()

    if st.button("清空全部对话", use_container_width=True):
        sm.clear_all_sessions()
        st.rerun()

    st.divider()
    st.markdown("### 历史窗口")
    sessions = list(st.session_state.all_sessions.items())[::-1]
    for s_id, s_data in sessions:
        col1, col2 = st.columns([4, 1])
        is_current = s_id == current_session_id
        btn_label = f"当前：{s_data['title']}" if is_current else s_data["title"]
        with col1:
            if st.button(btn_label, key=f"select_{s_id}", use_container_width=True):
                sm.switch_session(s_id)
                st.rerun()
        with col2:
            if st.button("删", key=f"delete_btn_{s_id}", use_container_width=True):
                sm.delete_session(s_id)
                st.rerun()

    st.divider()
    st.markdown("### 工具边界")
    st.caption("本 Agent 只覆盖金融风控、信贷政策、指标计算和必要联网查询。")


st.markdown('<div class="app-title">金融风险控制 Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">以本地知识库为主，按需调用计算器与联网检索，适合风控问答与演示。</div>',
    unsafe_allow_html=True,
)


top1, top2, top3 = st.columns(3)
top1.metric("当前会话", current_session["title"])
top2.metric("消息数", len(current_session["history"]))
top3.metric("主入口", "Agent")


for msg in current_session["history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


MAX_QUERY_LENGTH = 3000

if query_string := st.chat_input("请输入金融风控、信贷政策或指标计算问题"):
    query_string = query_string.strip()
    if len(query_string) > MAX_QUERY_LENGTH:
        st.error(f"输入内容过长，请控制在 {MAX_QUERY_LENGTH} 字符以内")
        st.stop()

    if current_session["title"] == "新建会话":
        new_title = query_string[:10] + "..." if len(query_string) > 10 else query_string
        sm.update_current_session_title(new_title)

    formatted_history = []
    for msg in current_session["history"]:
        if msg.get("system_welcome"):
            continue
        if msg["role"] == "user":
            formatted_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            formatted_history.append(AIMessage(content=msg["content"]))

    with st.chat_message("user"):
        st.markdown(query_string)
    sm.add_message_to_current("user", query_string)

    with st.chat_message("assistant"):
        if financial_agent is None:
            st.error("Agent 未能初始化。")
        else:
            try:
                with st.status("Agent 正在分析并选择工具...", expanded=True) as status:
                    response = financial_agent.invoke(
                        {
                            "input": query_string,
                            "chat_history": formatted_history,
                        }
                    )

                    steps = response.get("intermediate_steps", [])
                    if steps:
                        with st.expander("工具调用过程", expanded=True):
                            for action, observation in steps:
                                st.markdown(
                                    f'<div class="tool-card"><span class="tool-tag">{action.tool}</span>'
                                    f"<b>输入</b> {action.tool_input}<br/>"
                                    f"<b>结果</b> {str(observation)[:220]}</div>",
                                    unsafe_allow_html=True,
                                )

                    extracted_answer = response["output"]
                    if isinstance(extracted_answer, list):
                        extracted_answer = "".join(map(str, extracted_answer))

                    st.markdown(extracted_answer)
                    status.update(label="回答完成", state="complete", expanded=False)

                if extracted_answer:
                    sm.add_message_to_current("assistant", extracted_answer)

            except Exception as api_exception:
                st.error(f"运行出错: {api_exception}")
                traceback.print_exc()
