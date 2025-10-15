# page_generate_content.py
import os
import json
import time
import requests
import streamlit as st
from dotenv import load_dotenv

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Generate Content • Gửi bài viết đến webhook",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("📝 Generate Content (Gửi bài viết từ người dùng)")

load_dotenv(dotenv_path="./.env")  # nếu bạn muốn override webhook qua .env

# ================== SIDEBAR CONFIG ==================
st.sidebar.header("⚙️ Cấu hình")
DEFAULT_WEBHOOK = "https://n8n.thuong.cloud/webhook/make_content_from_user"
webhook_url = st.sidebar.text_input(
    "Webhook URL",
    value=os.getenv("MAKE_CONTENT_WEBHOOK_URL", DEFAULT_WEBHOOK),
    help="URL endpoint sẽ nhận JSON { content: <bài viết> }"
)
timeout_sec = st.sidebar.slider("Timeout (giây)", 5, 90, 30)
st.sidebar.caption("Nếu server chậm có thể tăng timeout.")

# ================== MAIN FORM ==================
with st.form(key="content_form", clear_on_submit=False):
    st.subheader("✍️ Nhập nội dung bài viết")
    content = st.text_area(
        "Bài viết",
        value=st.session_state.get("last_content", ""),
        height=350,
        placeholder="Dán nội dung bài viết của bạn vào đây..."
    )

    # thống kê nhanh
    words = len(content.split()) if content else 0
    chars = len(content) if content else 0
    st.caption(f"📏 {words} từ • {chars} ký tự")

    col1, col2, col3 = st.columns([1,1,2])
    with col1:
        submit = st.form_submit_button("🚀 Gửi tạo nội dung", use_container_width=True)
    with col2:
        reset = st.form_submit_button("♻️ Reset", use_container_width=True)

# ================== HANDLERS ==================
if reset:
    st.session_state["last_content"] = ""
    st.rerun()

if submit:
    payload = {"content": (content or "").strip()}

    # validate
    if not payload["content"]:
        st.error("⚠️ Vui lòng nhập nội dung bài viết trước khi gửi.")
    else:
        # lưu lại nội dung gần nhất
        st.session_state["last_content"] = payload["content"]

        # gọi webhook
        with st.spinner("⏳ Đang gửi bài viết đến webhook..."):
            try:
                resp = requests.post(
                    webhook_url,
                    json=payload,
                    timeout=timeout_sec,
                )
                # Hiển thị kết quả
                st.markdown("### ✅ Kết quả")
                st.write(f"Trạng thái: {resp.status_code}")
                # cố gắng parse JSON; nếu không được thì hiện text
                try:
                    st.json(resp.json())
                except Exception:
                    st.code(resp.text)

                if 200 <= resp.status_code < 300:
                    st.success("Đã gửi bài viết thành công!")
                else:
                    st.warning("Webhook trả về mã khác 2xx — kiểm tra lại nội dung/endpoint.")
            except requests.Timeout:
                st.error(f"⏱️ Quá thời gian chờ ({timeout_sec}s). Hãy thử tăng timeout trong sidebar.")
            except requests.RequestException as e:
                st.error(f"❌ Lỗi khi gọi webhook: {e}")

# ================== OPTIONAL: HƯỚNG DẪN NHANH ==================
with st.expander("ℹ️ Payload gửi đi (tham khảo)"):
    st.code(
        json.dumps(
            {"content": "Nội dung bài viết của bạn..."},
            ensure_ascii=False,
            indent=2
        ),
        language="json"
    )
