import requests
def execute(sql: str, params: tuple | None = None) -> int:
    conn = pool.getconn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.rowcount
    finally:
        pool.putconn(conn)
import os
import pandas as pd
from dotenv import load_dotenv
import psycopg2
from psycopg2.pool import SimpleConnectionPool
import streamlit as st

load_dotenv(dotenv_path="./.env")

@st.cache_resource
def get_pool():
    load_dotenv(dotenv_path="./.env")
    url = os.getenv("POSTGRESQL_DB_URL", "").strip()
    if not url:
        raise RuntimeError(
            "POSTGRESQL_DB_URL chưa được thiết lập. "
            "Hãy đặt env hoặc dùng st.secrets để cung cấp chuỗi kết nối."
        )
    return SimpleConnectionPool(minconn=1, maxconn=5, dsn=url)

pool = get_pool()

def query_df(sql: str, params: tuple | None = None) -> pd.DataFrame:
    conn = pool.getconn()
    try:
        return pd.read_sql(sql, conn, params=params)
    finally:
        pool.putconn(conn)

# ========= KHU VỰC 1: CHỌN VIDEO =========
st.header("📹 Danh sách các video ở công đoạn tạo video")

videos = query_df("""
    SELECT id, title, hash_tags, post_state
    FROM new_social_posts
    WHERE post_state = 'AUDIO_CHECKED'
    ORDER BY id DESC
    LIMIT %s
""", (200,))

if videos.empty:
    st.info("Không có bản ghi nào cần tạo video.")
    st.stop()

st.dataframe(videos, use_container_width=True)
st.subheader("🎬 Chọn video để xử lý tiếp")

options = [f"{row.id} — {str(row.title)}" for _, row in videos.iterrows()]
display_to_id = {opt: int(opt.split(" — ")[0]) for opt in options}

selected_display = st.selectbox("Chọn video theo ID", options)
selected_id = display_to_id[selected_display]

if st.button("Duyệt để tạo video", key="approve_make_video", type="primary"):
    try:
        rowcount = execute("""
            UPDATE new_social_posts SET post_state = 'WAIT_FOR_MAKE_VIDEO' WHERE id = %s
        """, (selected_id,))
        if rowcount > 0:
            # Gửi POST tới webhook
            try:
                resp = requests.post(
                    "https://n8n.thuong.cloud/webhook/433dc95d-0365-435e-b66b-2a10054b2504",
                    json={"id": str(selected_id)}
                )
                if resp.status_code == 200:
                    st.success("Đã chuyển trạng thái bài viết sang WAIT_FOR_MAKE_VIDEO và gửi webhook thành công!")
                else:
                    st.warning(f"Đã chuyển trạng thái, nhưng gửi webhook thất bại: {resp.status_code}")
            except Exception as ex:
                st.warning(f"Đã chuyển trạng thái, nhưng gửi webhook lỗi: {ex}")
            st.rerun()
        else:
            st.warning("Không cập nhật được trạng thái bài viết.")
    except Exception as e:
        st.error(f"Lỗi khi chuyển trạng thái bài viết: {e}")
