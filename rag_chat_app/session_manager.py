import uuid
import streamlit as st
import database as db

def init_sessions():
    """初始化会话状态，在 app.py 最开始调用"""
    if "all_sessions" not in st.session_state:
        st.session_state.all_sessions = db.load_all_sessions()
        
    if "current_session_id" not in st.session_state:
        if st.session_state.all_sessions:
            st.session_state.current_session_id = list(st.session_state.all_sessions.keys())[-1]
        else:
            st.session_state.current_session_id = None

    # 如果没有任何对话，初始化一个
    if not st.session_state.all_sessions or st.session_state.current_session_id is None:
        create_new_session()

def create_new_session():
    """新建会话"""
    new_id = str(uuid.uuid4())
    welcome_text = "您好！系统已成功挂载信用风险合规核心资产。请提出您需要分析的违约概率（PD）、违约损失率（LGD）或宏观审慎传导等风控领域专业问题。"

    st.session_state.all_sessions[new_id] = {
        "title": "🆕 新建会话",
        "history": [{"role": "assistant", "content": welcome_text, "system_welcome": True}]
    }
    st.session_state.current_session_id = new_id

    # 同步写入 SQLite
    db.create_session(new_id, "🆕 新建会话")
    db.add_message(new_id, "assistant", welcome_text, system_welcome=True)

def switch_session(session_id: str):
    """切换当前会话"""
    st.session_state.current_session_id = session_id

def delete_session(session_id: str):
    """删除指定会话"""
    db.delete_session(session_id)
    del st.session_state.all_sessions[session_id]

    if st.session_state.current_session_id == session_id:
        remaining_sessions = list(st.session_state.all_sessions.keys())
        st.session_state.current_session_id = remaining_sessions[-1] if remaining_sessions else None
        
        # 如果删光了，自动新建一个
        if not remaining_sessions:
            create_new_session()

def clear_all_sessions():
    """清空所有会话"""
    db.clear_all_sessions()
    st.session_state.all_sessions = {}
    st.session_state.current_session_id = None
    create_new_session()

def get_current_session() -> dict:
    """获取当前正在进行的会话数据对象"""
    return st.session_state.all_sessions[st.session_state.current_session_id]

def update_current_session_title(new_title: str):
    """更新当前会话标题"""
    st.session_state.all_sessions[st.session_state.current_session_id]["title"] = new_title
    db.update_session_title(st.session_state.current_session_id, new_title)

def add_message_to_current(role: str, content: str):
    """向当前会话追加消息"""
    st.session_state.all_sessions[st.session_state.current_session_id]["history"].append(
        {"role": role, "content": content}
    )
    db.add_message(st.session_state.current_session_id, role, content)