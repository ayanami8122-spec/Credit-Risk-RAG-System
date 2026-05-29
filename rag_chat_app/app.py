import sys
import traceback
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import HumanMessage, AIMessage
import session_manager as sm 
import database as db
from test import get_rag_chain, iter_answer_deltas

db.init_db()


_APP_DIR = Path(__file__).resolve().parent


if str(_APP_DIR) not in sys.path:

    sys.path.insert(0, str(_APP_DIR))


st.set_page_config(

    page_title="金融风险控制决策系统",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded", # 建议展开以便看时钟

)


@st.cache_resource(show_spinner="正在拉起本地信用风险知识库与后端检索工具，请稍等...")


def load_backend_instance():

    return get_rag_chain()


rag_pipeline = None


try:

    rag_pipeline = load_backend_instance()

except Exception as init_error:

    st.error(f"后端系统初始化失败，请检查知识库文件、依赖或网络配置。错误摘要: {init_error}")


# 1. 初始化会话管理 
sm.init_sessions()
current_session_id = st.session_state.current_session_id
current_session = sm.get_current_session()


# 2. 侧边栏 UI 渲染
with st.sidebar:

    st.header("控制台")
    
    # 注入动态时钟组件
    clock_html = """
    <div id="clock" style="font-size: 1.1em; font-weight: bold; color: #4CAF50; text-align: center; margin-bottom: 10px; font-family: 'Courier New', Courier, monospace;">
    </div>
    <script>
    function updateClock() {
        var now = new Date();
        document.getElementById('clock').innerText = now.toLocaleString('zh-CN', { hour12: false });
        setTimeout(updateClock, 1000);
    }
    updateClock();
    </script>
    """
    components.html(clock_html, height=40)
    
    st.write("---")

    if st.button("➕ 新建会话", use_container_width=True, type="primary"):

        sm.create_new_session()
        st.rerun()

    st.write("---")
    st.subheader("历史窗口")
    
    sessions = list(st.session_state.all_sessions.items())[::-1]

    for s_id, s_data in sessions:

        col1, col2 = st.columns([4, 1])
        is_current = (s_id == current_session_id)
        btn_label = f"📁 {s_data['title']}" if not is_current else f"🎯 {s_data['title']} (当前)"

        with col1:

            if st.button(btn_label, key=f"select_{s_id}", use_container_width=True):

                sm.switch_session(s_id)
                st.rerun()
        
        with col2:

            if st.button("🗑️", key=f"delete_btn_{s_id}", use_container_width=True):

                sm.delete_session(s_id)
                st.rerun()
        
    st.write("---")

    if st.button("清空历史对话", use_container_width=True):

        sm.clear_all_sessions()
        st.rerun()


# 3. 主界面 UI 渲染
st.title("🛡️ 金融风险控制 RAG 智能助手")
st.caption("后端 RAG 架构：本地双层检索 + 交叉注意力重排")


# 渲染历史消息
for msg in current_session["history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


MAX_QUERY_LENGTH = 3000


# 4. 对话逻辑处理
if query_string := st.chat_input("请输入具体问题："):

    query_string = query_string.strip()



    if len(query_string) > MAX_QUERY_LENGTH:

        st.error(f"❌ 输入内容过长，请控制在 {MAX_QUERY_LENGTH} 字符以内")
        st.stop()

    # 首次提问，自动重命名标题
    if current_session["title"] == "🆕 新建会话":

        new_title = query_string[:5] + "..." if len(query_string) > 5 else query_string 
        sm.update_current_session_title(new_title)

    # 提取格式化历史，屏蔽系统欢迎语
    formatted_history = []
    for msg in current_session["history"]:

        if msg.get("system_welcome"):
            continue

        if msg["role"] == "user":

            formatted_history.append(HumanMessage(content=msg["content"]))

        elif msg["role"] == "assistant":

            formatted_history.append(AIMessage(content=msg["content"]))

    # 显示并保存用户提问
    with st.chat_message("user"):

        st.markdown(query_string)

    sm.add_message_to_current("user", query_string)

    # 显示并生成 AI 响应
    with st.chat_message("assistant"):

        if rag_pipeline is None:

            st.error("❌ RAG 未能初始化成功")

        else:

            try:

                with st.spinner("正在检索知识库并获取答案…"):

                    extracted_answer = st.write_stream(
                        iter_answer_deltas(

                            rag_pipeline,
                            query_string,
                            chat_history=formatted_history,

                        )
                    )

                if isinstance(extracted_answer, list):

                    extracted_answer = "".join(map(str, extracted_answer))

                if extracted_answer:

                    sm.add_message_to_current("assistant", extracted_answer)

            except Exception as api_exception:

                st.error(str(api_exception))
                traceback.print_exc()