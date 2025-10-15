import os
import streamlit as st
import pandas as pd
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv
import requests


# ===== Pool kết nối: tạo 1 lần cho cả app =====
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
        
st.set_page_config(page_title="Trang để kiểm tra nội dung bài viết", layout="wide", initial_sidebar_state="expanded")
st.title("Trang để kiểm tra nội dung bài viết")


# ✅ Lấy dữ liệu từ new_social_posts (giới hạn 200 dòng cho nhẹ)
posts = query_df("SELECT id, title, hash_tags, post_state FROM new_social_posts WHERE post_state = 'CREATED_CONTENT' ORDER BY id DESC LIMIT %s", (200,))
st.header("📊 Danh sách các bài viết chưa duyệt nội dung")

if posts.empty:
    st.info("Không có bản ghi nào ở trạng thái CONTENT_CREATED.")
else:
    st.dataframe(posts, width='stretch')

    # ========== Chọn dòng cần sửa ========== 
    st.subheader("✍️ Chỉnh sửa 1 bản ghi")

    options = [
        f"{row.id} — {str(row.title)}" 
        for _, row in posts.iterrows()
    ]
    display_to_id = {opt: int(opt.split(" — ")[0]) for opt in options}

    selected_display = st.selectbox("Chọn bản ghi theo ID", options)
    selected_id = display_to_id[selected_display]
    row = posts.loc[posts["id"] == selected_id].iloc[0]

    # Lấy scripts (contents) cho post này
    scripts = query_df("SELECT id, transcript FROM scripts WHERE post_id = %s ORDER BY idx ASC, id ASC", (selected_id,))

    # ========== Form edit ========== 
    with st.form(key="edit_form", clear_on_submit=False):
        new_title = st.text_input("Title", value=row["title"] or "")

        # Hash tags: nhiều dòng
        hash_tags_display = "\n".join(row["hash_tags"]) if isinstance(row["hash_tags"], list) else (row["hash_tags"] or "")
        new_hash_tags_multiline = st.text_area("Hash tags (mỗi dòng 1 hashtag)", value=hash_tags_display, height=120)

        # Contents: nhiều dòng (từ scripts.transcript)

        # Hiển thị từng đoạn transcript thành từng cell riêng
        transcript_cells = []
        if not scripts.empty:
            st.markdown("#### Nội dung từng đoạn (mỗi đoạn 1 ô)")
            for idx, script_row in scripts.iterrows():
                val = script_row["transcript"] or ""
                cell = st.text_area(f"Đoạn {idx+1}", value=val, key=f"transcript_{script_row['id']}", height=80)
                transcript_cells.append(cell)
        else:
            st.info("Chưa có đoạn nội dung nào cho bài viết này.")
        # Cho phép thêm đoạn mới
        new_segment = st.text_area("Thêm đoạn mới (nếu có)", value="", key="add_new_segment", height=80)

        col1, col2 = st.columns(2)
        with col1:
            save = st.form_submit_button("💾 Save changes")
        with col2:
            cancel = st.form_submit_button("↩️ Reset form")

    if save:
        tags_list = [t.strip() for t in new_hash_tags_multiline.splitlines() if t.strip()]
        # Gom lại các đoạn transcript từ các cell
        contents_list = [c.strip() for c in transcript_cells if c.strip()]
        if new_segment.strip():
            contents_list.append(new_segment.strip())

        # Update new_social_posts
        affected_post = execute(
            """
            UPDATE new_social_posts
            SET title=%s,
                hash_tags=%s,
                post_state='CONTENT_CHECKED'
            WHERE id=%s
            """,
            (new_title, tags_list, int(selected_id))
        )

        # Update scripts.transcript (update từng dòng theo idx)
        affected_scripts = 0
        if not scripts.empty:
            for i, script_row in scripts.iterrows():
                if i < len(contents_list):
                    if script_row["transcript"] != contents_list[i]:
                        affected_scripts += execute(
                            "UPDATE scripts SET transcript=%s WHERE id=%s",
                            (contents_list[i], script_row["id"])
                        )
        # Nếu có thêm contents mới, insert thêm scripts mới
        if len(contents_list) > len(scripts):
            for idx_new, content in enumerate(contents_list[len(scripts):], start=len(scripts)):
                execute(
                    "INSERT INTO scripts (post_id, transcript, idx) VALUES (%s, %s, %s)",
                    (selected_id, content, idx_new)
                )

        if affected_post > 0 or affected_scripts > 0:
            st.success(f"Đã lưu thay đổi & đánh dấu CONTENT_CHECKED cho ID {selected_id}.")
            try:
                resp = requests.post(
                    "https://n8n.thuong.cloud/webhook/make-image-prompt",
                    json={"id": str(selected_id)}
                )
                if resp.status_code == 200:
                    st.info("Đã gửi webhook tạo image prompt thành công!")
                else:
                    st.warning(f"Gửi webhook thất bại: {resp.status_code}")
            except Exception as ex:
                st.warning(f"Gửi webhook lỗi: {ex}")
            st.rerun()
        else:
            st.warning("Không có hàng nào được cập nhật.")

    if cancel:
        st.experimental_rerun()
