import os
import json
import requests
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from psycopg2.pool import SimpleConnectionPool

# ========= ENV =========
load_dotenv(dotenv_path="./.env")
WEBHOOK_URL = os.getenv("POST_CONTENT_WEBHOOK", "https://n8n.thuong.cloud/webhook/post_content")
WEBHOOK_TOKEN = os.getenv("N8N_WEBHOOK_TOKEN", "").strip()  # nếu có auth thì set vào .env

# ========= DB POOL =========
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

# ========= PAGE =========
st.set_page_config(page_title="Tải + Đăng video đã tạo", layout="wide", initial_sidebar_state="expanded")
st.title("📥 Danh sách video đã tạo")

# Nền tảng có thể chọn
SOCIAL_OPTIONS = ["tiktok", "facebook", "instagram", "youtube", "zalo", "threads"]

video_df = query_df("""
    SELECT id, content_type, title as "Tiêu đề", video_url
    FROM new_social_posts
    WHERE video_url IS NOT NULL AND video_url != ''
    ORDER BY id DESC
""")

if video_df.empty:
    st.info("Chưa có video nào được tạo.")
else:
    st.dataframe(video_df, use_container_width=True)

    st.markdown("### Đăng từng bài")
    st.caption("Chọn các nền tảng và bấm **Đăng bài** để gửi đến webhook.")

    # function gửi webhook
    def post_to_webhook(post_id: int | str, platforms: list[str]) -> tuple[bool, str]:
        payload = {"id": post_id, "social": platforms}
        headers = {"Content-Type": "application/json"}
        if WEBHOOK_TOKEN:
            headers["Authorization"] = f"Bearer {WEBHOOK_TOKEN}"
        try:
            r = requests.post(WEBHOOK_URL, data=json.dumps(payload), headers=headers, timeout=20)
            if r.ok:
                return True, f"Đã gửi thành công: {payload} (status {r.status_code})"
            return False, f"Lỗi webhook {r.status_code}: {r.text}"
        except requests.RequestException as e:
            return False, f"Lỗi kết nối webhook: {e}"

    # Render từng dòng với form riêng để tránh rerun lẫn nhau
    for idx, row in video_df.iterrows():
        with st.container(border=True):
            left, mid, right = st.columns([3, 3, 2])

            with left:
                st.markdown(f"**ID:** {row['id']}  \n"
                            f"**Loại nội dung:** {row['content_type']}  \n"
                            f"**Tiêu đề:** {row['Tiêu đề']}")
                st.markdown(f"**Video URL:** [Tải về/Xem]({row['video_url']})", unsafe_allow_html=False)

            with mid:
                # Mặc định gợi ý chọn theo nội dung (ví dụ nếu tiêu đề có 'shorts' → youtube)
                default_select = []
                title_lower = str(row["Tiêu đề"]).lower()
                if "short" in title_lower or "youtube" in title_lower:
                    default_select.append("youtube")
                if "tiktok" in title_lower or "tt" in title_lower:
                    default_select.append("tiktok")

                selected = st.multiselect(
                    "Nền tảng đăng",
                    options=SOCIAL_OPTIONS,
                    default=default_select,
                    key=f"platforms_{row['id']}"
                )

            with right:
                with st.form(key=f"form_{row['id']}", clear_on_submit=False):
                    submit = st.form_submit_button("🚀 Đăng bài", use_container_width=True)
                    if submit:
                        if not selected:
                            st.warning("Hãy chọn ít nhất 1 nền tảng.", icon="⚠️")
                        else:
                            ok, msg = post_to_webhook(row["id"], selected)
                            if ok:
                                st.success(msg, icon="✅")
                                st.toast(f"Đã gửi đăng bài ID {row['id']} lên: {', '.join(selected)}")
                            else:
                                st.error(msg, icon="❌")

    st.divider()
    st.caption("Tip: đặt `POST_CONTENT_WEBHOOK` và `N8N_WEBHOOK_TOKEN` trong `.env` để cấu hình nhanh.")
