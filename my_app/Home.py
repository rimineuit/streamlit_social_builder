import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import psycopg2
from psycopg2.pool import SimpleConnectionPool
import os


@st.cache_resource
def get_pool():
    load_dotenv(dotenv_path="./.env")
    url = os.getenv("POSTGRESQL_DB_URL", "").strip()
    if not url:
        raise RuntimeError(
            "POSTGRESQL_DB_URL chưa được thiết lập. "
            "Hãy đặt env hoặc dùng st.secrets để cung cấp chuỗi kết nối."
        )
    # Nếu muốn, bạn có thể thêm keepalive vào URL:
    # url += ("&" if "?" in url else "?") + "keepalives=1&keepalives_idle=30&keepalives_interval=10&keepalives_count=5"
    return SimpleConnectionPool(minconn=1, maxconn=5, dsn=url)

pool = get_pool()

# ===== Helper: mượn connection có 'ping' =====
def _borrow_conn():
    """
    Lấy connection từ pool, kiểm tra còn sống (SELECT 1).
    Nếu chết/đứt, đóng và lấy conn mới.
    """
    conn = pool.getconn()
    try:
        if conn.closed:
            raise RuntimeError("Connection closed")
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        return conn
    except Exception:
        # đóng hẳn conn cũ và lấy conn mới
        try:
            pool.putconn(conn, close=True)
        except Exception:
            pass
        return pool.getconn()

# ===== API truy vấn tiện lợi =====
def query_df(sql: str, params: tuple | None = None) -> pd.DataFrame:
    conn = _borrow_conn()
    try:
        return pd.read_sql(sql, conn, params=params)
    finally:
        pool.putconn(conn)

def execute(sql: str, params: tuple | None = None) -> int:
    """
    Thực thi câu lệnh ghi. Dùng context manager của connection để auto-commit/rollback.
    Trả về rowcount.
    """
    conn = _borrow_conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.rowcount
    finally:
        pool.putconn(conn)

load_dotenv(dotenv_path="./.env")
st.set_page_config(page_title="Home", layout="wide", initial_sidebar_state="expanded")
st.title("Tool quản lý quy trình tạo video từ bài viết")

st.header("Tổng quan trạng thái các task")
summary_df = query_df("""
    SELECT post_state, COUNT(*) as count
    FROM new_social_posts
    GROUP BY post_state
    ORDER BY count DESC
""")

# Friendly labels and pastel colors for each state
state_info = {
    "CREATED_CONTENT": {"label": "Kiểm tra nội dung", "color": "#ffe082", "page": "my_app/pages/Check_content.py"},  # pastel yellow
    "IMAGE_PROMPT_CREATED": {"label": "Tạo ảnh", "color": "#b3e5fc", "page": "my_app/pages/Generate_image.py"},      # pastel blue
    "IMAGE_PROMPT_CHECKED": {"label": "Tạo âm thanh", "color": "#c8e6c9", "page": "my_app/pages/Generate_voice.py"},   # pastel green
    "AUDIO_CHECKED": {"label": "Tạo video", "color": "#e1bee7", "page": "my_app/pages/Gernerate_video.py"},         # pastel purple
    "WAIT_FOR_MAKE_VIDEO": {"label": "Chờ tạo video", "color": "#ffcdd2", "page": "my_app/pages/Gernerate_video.py"}, # pastel red
    # Thêm các state khác nếu có
}

# Tổng số bài viết
st.markdown("""
<h2 style="color:#1976d2; font-size:2.1rem; font-weight:800; margin-bottom:0.2em; letter-spacing:0.5px;">
    📊 Tổng quan hệ thống
</h2>
""", unsafe_allow_html=True)
post_count = query_df("SELECT COUNT(*) as total FROM new_social_posts")['total'][0]
st.markdown(f"""
<div style="font-size:1.25rem; color:#333; margin-bottom:1.2em;">
    <b>Tổng số bài viết:</b> <span style='color:#1976d2; font-size:1.5rem; font-weight:700'>{post_count}</span>
</div>
""", unsafe_allow_html=True)

# Biểu đồ số bài viết tạo theo ngày
created_chart_df = query_df("""
    SELECT DATE(created_at) as date, COUNT(*) as count
    FROM new_social_posts
    GROUP BY DATE(created_at)
    ORDER BY date DESC
""")
if not created_chart_df.empty:
    st.markdown("""
    <h3 style="color:#388e3c; font-size:1.3rem; font-weight:700; margin-top:1.5em; margin-bottom:0.5em;">
        🗓️ Thống kê số bài viết mới theo ngày
    </h3>
    """, unsafe_allow_html=True)
    st.bar_chart(created_chart_df.set_index('date')['count'])

st.markdown("""
<h3 style="color:#1976d2; font-size:1.3rem; font-weight:700; margin-top:2em; margin-bottom:0.5em;">
    🚦 Trạng thái các bước xử lý bài viết
</h3>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.state-grid {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 1.5rem;
    margin-top: 2.2rem;
}
.state-card {
    border-radius: 1.2rem;
    box-shadow: 0 2px 10px rgba(26,115,232,0.10);
    padding: 1.6rem 2.8rem 1.2rem 2.8rem;
    text-align: center;
    min-width: 240px;
    max-width: 340px;
    cursor: pointer;
    transition: box-shadow 0.2s, background 0.2s;
    font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
    border: 2px solid #e3eafc;
    margin-bottom: 1.2rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.state-card:hover {
    box-shadow: 0 6px 18px rgba(26,115,232,0.18);
    filter: brightness(1.06);
    border: 2px solid #90caf9;
}
.state-count {
    font-size: 2.1rem;
    font-weight: bold;
    margin-bottom: 0.5rem;
    color: #263238;
    text-shadow: 0 2px 8px #e3eafc;
}
.state-label {
    font-size: 1.35rem;
    font-weight: 700;
    color: #263238;
    letter-spacing: 0.5px;
    margin-top: 0.2rem;
    margin-bottom: 0.2rem;
    text-transform: none;
    line-height: 1.2;
}
.state-desc {
    font-size: 0.98rem;
    font-weight: 500;
    margin-top: 0.3rem;
    color: #666;
    letter-spacing: 0.2px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="state-grid">', unsafe_allow_html=True)
for _, row in summary_df.iterrows():
    state = row['post_state']
    count = row['count']
    info = state_info.get(state, None)
    if info:
        card_color = info['color']
        label = info['label']
        page = info['page']
        st.markdown(f'''
        <div class="state-card" style="background: {card_color}; border-color: {card_color};" onclick="window.location.href='/{page}'">
            <div class="state-label">{label}</div>
            <div class="state-count">{count} task</div>
            <div class="state-desc">Mã trạng thái: <b>{state}</b></div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="state-card">
            <div class="state-label">{state}</div>
            <div class="state-count">{count} task</div>
        </div>
        ''', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)