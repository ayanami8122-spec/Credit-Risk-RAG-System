import sqlite3
from pathlib import Path


DB_FILE = Path(__file__).resolve().parent / "chat_history.db"


def get_connection():

    # 启动外键支持
    conn = sqlite3.connect(DB_FILE, check_same_thread=False, timeout = 10)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db():
    
    # 初始化数据库表结构
    with get_connection() as conn:

        # 会话表
        conn.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        '''
        )

        # 消息表
        conn.execute('''
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                system_welcome BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
            )
        ''')


def load_all_sessions() -> dict:

    # 启动时从数据库全量拉取历史并拼装成 session_state 需要的字典格式
    session_dict = {}
    with get_connection() as conn:

        # 获取所有对话
        cur = conn.execute("SELECT session_id, title FROM chat_sessions ORDER BY created_at ASC")

        for row in cur.fetchall():

            s_id, title = row[0], row[1]
            session_dict[s_id] = {"title": title, "history": []}

        # 获取所有消息并拼装
        cur = conn.execute("""
            SELECT SESSION_ID, role, content, system_welcome 
            FROM chat_messages 
            ORDER BY session_id,id ASC
        """)

        for row in cur.fetchall():
            s_id, role, content, sys_welcome = row[0], row[1], row[2], bool(row[3])
            if s_id in session_dict:
                session_dict[s_id]["history"].append(
                    {
                        "role": role,
                        "content": content,
                        "system_welcome": sys_welcome,
                    }
                )
    
    return session_dict


def create_session(session_id: str, title: str):

    # 新建会话
    with get_connection() as conn:
        conn.execute("INSERT INTO chat_sessions (session_id, title) VALUES (?, ?)", (session_id, title))


def update_session_title(session_id: str, new_title: str):

    # 更新会话标题
    with get_connection() as conn:
        conn.execute("UPDATE chat_sessions SET title = ? WHERE session_id = ?", (new_title, session_id))


def add_message(session_id: str, role: str, content: str, system_welcome: bool = False):

    # 追加单条信息
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO chat_messages (session_id, role, content, system_welcome) VALUES (?,?,?,?)",
            (session_id, role, content, int(system_welcome))
        )


def delete_session(session_id: str):

    # 删除单一历史会话
    with get_connection() as conn:
        conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))


def clear_all_sessions():

    # 清空所有数据
    with get_connection() as conn:
        conn.execute("DELETE FROM chat_sessions") # 由于外键 CASCADE 联级删除， messages 也会被自动清空
