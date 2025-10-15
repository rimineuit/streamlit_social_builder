import os
import time
import glob
import streamlit as st
import pandas as pd
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv

# ========= KHAI BÁO GIAO DIỆN CHÍNH =========
st.set_page_config(page_title="Trang để tạo ảnh tương ứng với nội dung bài viết", layout="wide", initial_sidebar_state="expanded")
st.title("🎨 Trang để tạo ảnh tương ứng với nội dung bài viết")
load_dotenv(dotenv_path="./.env")

# ========= KẾT NỐI DB (CACHE) =========

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


# ========= KHU VỰC 1: CHỌN BÀI VIẾT =========
st.header("📊 Danh sách các bài viết chưa duyệt ảnh")

posts = query_df("""
    SELECT id, title, hash_tags, post_state
    FROM new_social_posts
    WHERE post_state = 'IMAGE_PROMPT_CREATED'
    ORDER BY id DESC
    LIMIT %s
""", (200,))

if posts.empty:
    st.info("Không có bản ghi nào ở trạng thái IMAGE_PROMPT_CREATED.")
    st.stop()

st.dataframe(posts, width='stretch')
st.subheader("✍️ Chọn bản ghi để tạo ảnh")

options = [f"{row.id} — {str(row.title)[:60]}" for _, row in posts.iterrows()]
display_to_id = {opt: int(opt.split(" — ")[0]) for opt in options}

selected_display = st.selectbox("Chọn bản ghi theo ID", options)
selected_id = display_to_id[selected_display]

# Lấy row bài viết đã chọn
post_row = query_df("SELECT * FROM new_social_posts WHERE id = %s", (selected_id,)).iloc[0]


# ========= KHU VỰC 3: CẤU HÌNH SIDEBAR =========
st.sidebar.header("⚙️ Config")
from google import genai
from google.genai import types as gtypes
from PIL import Image
from io import BytesIO
load_dotenv(dotenv_path="./.env")
API_KEY=os.getenv("GEMINI_API_KEY", "").strip()
api_key = st.sidebar.text_input("Gemini API Key", type="password", value=API_KEY, key="sidebar_gemini_api_key")

model_name = st.sidebar.selectbox(
    "Model",
    ["imagen-4.0-generate-001", "imagen-4.0-ultra-generate-001", "imagen-4.0-fast-generate-001"],
    index=2,
    key="sidebar_model_name"
)
aspect_ratio = st.sidebar.selectbox("Aspect ratio", ["1:1", "16:9", "9:16", "4:3", "3:4"], index=2, key="sidebar_aspect_ratio")
n_images = st.sidebar.slider("Số ảnh", 1, 4, 1, key="sidebar_n_images")
# NEW: cấu hình số luồng song song
max_workers = st.sidebar.slider("Số luồng song song", 1, 8, 4, key="sidebar_max_workers")

lang_choice = st.radio(
    "Ngôn ngữ dùng để sinh ảnh",
    ["English (context + image_prompt)", "Vietnamese (context_vi + image_prompt_vi)"],
    horizontal=True,
    key="sidebar_lang_choice"
)


# Hàm tiện ích chuyển ảnh bất kể dạng trả về
def to_pil(img_obj):
    # 1) Đã là PIL.Image thì trả về luôn
    if isinstance(img_obj, Image.Image):
        return img_obj

    # 2) Nếu là google.genai.types.Image → dùng as_pil_image() (nếu có)
    if isinstance(img_obj, gtypes.Image):
        if hasattr(img_obj, "as_pil_image") and callable(img_obj.as_pil_image):
            return img_obj.as_pil_image()
        blob = getattr(img_obj, "image_bytes", None) \
            or getattr(img_obj, "bytes", None) \
            or getattr(img_obj, "data", None)
        if blob:
            return Image.open(BytesIO(blob)).convert("RGB")
        inner = getattr(img_obj, "image", None)
        if inner is not None and inner is not img_obj:
            return to_pil(inner)

    # 3) Nếu là dict chứa bytes
    if isinstance(img_obj, dict):
        blob = img_obj.get("image") or img_obj.get("image_bytes") or img_obj.get("bytes") or img_obj.get("data")
        if isinstance(blob, (bytes, bytearray, memoryview)):
            return Image.open(BytesIO(blob)).convert("RGB")

    # 4) Nếu là bytes
    if isinstance(img_obj, (bytes, bytearray, memoryview)):
        return Image.open(BytesIO(img_obj)).convert("RGB")

    raise TypeError(f"Không nhận diện được định dạng ảnh: {type(img_obj)}")

_api_key = api_key
_model_name = model_name
_aspect_ratio = aspect_ratio
_lang_choice = lang_choice
_to_pil = to_pil


# ========= KHU VỰC 2: LOAD SCRIPTS THEO post_id =========
st.markdown("---")
st.header("🧩 Prompt theo bài viết đã chọn")

scripts_df = query_df("""
    SELECT 
        id,
        post_id,
        context_prompt,
        image_prompt,
        context_prompt_vietnamese,
        image_prompt_vietnamese,
        image_url,
        transcript
    FROM scripts
    WHERE post_id = %s
    ORDER BY idx ASC, id ASC
""", (selected_id,))

if scripts_df.empty:
    st.warning("Bài viết này chưa có bản ghi nào trong bảng scripts.")
    st.stop()


# ========== Hàm upload lên GCP =============
def upload_to_gcp(local_path, dest_blob_name):
    """
    Upload file lên Google Cloud Storage.
    Dùng storage.Client() trực tiếp, rely vào GOOGLE_APPLICATION_CREDENTIALS.
    Trả về: (ok: bool, public_url_or_none: str|None)
    """
    try:
        from google.cloud import storage

        bucket_name = os.getenv("GCP_BUCKET_NAME")
        if not bucket_name:
            st.error("⚠️ Chưa cấu hình GCP_BUCKET_NAME trong biến môi trường.")
            return False, None

        client = storage.Client()  # tự đọc GOOGLE_APPLICATION_CREDENTIALS
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(dest_blob_name)
        blob.upload_from_filename(local_path)

        # Nếu bucket public (hoặc dùng Uniform Access), public_url usable
        try:
            blob.make_public()
        except Exception:
            pass

        return True, blob.public_url

    except Exception as e:
        st.error(f"❌ Lỗi upload GCP: {e}")
        return False, None


# Hiển thị tất cả prompt và ảnh (nếu có) của bài viết
st.subheader("🖼️ Danh sách prompt và ảnh đã lưu (nếu có)")
prompt_states = []

# Đưa các biến cấu hình ra ngoài để dùng trong expander
_api_key = api_key
_model_name = model_name
_aspect_ratio = aspect_ratio
_lang_choice = lang_choice
_to_pil = to_pil

for idx, row in scripts_df.iterrows():
    with st.expander(f"Script ID: {row['id']}"):
        st.markdown(f"**Transcript:**\n{row['transcript']}")
        transcript_val = st.text_area(f"Nội dung transcript (Script {row['id']})", value=row["transcript"] or "", key=f"transcript_{row['id']}", height=120)
        col1, col2 = st.columns([2,2])
        with col1:
            st.markdown("**Context (EN):**")
            context_en = st.text_area(f"Context EN {row['id']}", value=row["context_prompt"] or "", key=f"context_en_{row['id']}", height=220)
            st.markdown("**Context (VI):**")
            context_vi = st.text_area(f"Context VI {row['id']}", value=row["context_prompt_vietnamese"] or "", key=f"context_vi_{row['id']}", height=220)
        with col2:
            st.markdown("**Image Prompt (EN):**")
            img_prompt_en = st.text_area(f"Image Prompt EN {row['id']}", value=row["image_prompt"] or "", key=f"img_prompt_en_{row['id']}", height=220)
            st.markdown("**Image Prompt (VI):**")
            img_prompt_vi = st.text_area(f"Image Prompt VI {row['id']}", value=row["image_prompt_vietnamese"] or "", key=f"img_prompt_vi_{row['id']}", height=220)
        # Nút generate nhiều ảnh cho từng script
        gen_btn = st.button(f"✨ Generate ảnh cho script này", key=f"gen_script_{row['id']}_{idx}")
        if gen_btn:
            if not _api_key:
                st.error("⚠️ Hãy nhập Gemini API Key trước.")
            else:
                try:
                    client = genai.Client(api_key=_api_key)
                    if _lang_choice.startswith("English"):
                        prompt_text = (context_en or "").strip() + "\n\n" + (img_prompt_en or "").strip()
                    else:
                        prompt_text = (context_vi or "").strip() + "\n\n" + (img_prompt_vi or "").strip()
                    if not prompt_text.strip():
                        st.warning("❗ Prompt đang trống.")
                    else:
                        with st.spinner(f"⏳ Đang tạo ảnh cho script {row['id']}..."):
                            resp = client.models.generate_images(
                                model=_model_name,
                                prompt=prompt_text,
                                config=gtypes.GenerateImagesConfig(
                                    number_of_images=n_images,
                                    aspect_ratio=_aspect_ratio
                                )
                            )
                            if not getattr(resp, "generated_images", None):
                                st.error(f"❌ Không có ảnh nào được tạo cho script {row['id']}")
                            else:
                                os.makedirs("./tmp", exist_ok=True)
                                ts = int(time.time()*1000)
                                for i, img_obj in enumerate(resp.generated_images):
                                    pil_img = _to_pil(img_obj.image)
                                    out_path = f"./tmp/generated_post_{row['post_id']}_script_{row['id']}_{ts}_{i}.png"
                                    pil_img.save(out_path)
                except Exception as e:
                    st.error(f"Lỗi khi gọi Gemini API: {e}")
        # Lưu lại prompt state để cập nhật hàng loạt nếu cần
        prompt_states.append({
            "script_id": row["id"],
            "context_en": context_en,
            "context_vi": context_vi,
            "img_prompt_en": img_prompt_en,
            "img_prompt_vi": img_prompt_vi
        })


# ========== GALLERY: CHỌN ẢNH ĐẠI DIỆN VÀ LƯU TẤT CẢ =============
st.markdown("---")
st.header("🖼️ Gallery: Chọn ảnh đại diện cho từng script và lưu tất cả lên GCP")

from PIL import Image as PILImage

gallery_selected = {}
scripts = list(scripts_df.iterrows())
per_row = 5
img_width = 200
def image_with_tooltip(img, caption, key=None):
    import base64
    from io import BytesIO
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    # Sử dụng thuộc tính title để tooltip luôn hiển thị khi hover, tương thích mọi trình duyệt/Streamlit
    # Custom tooltip overlay đẹp, chữ to, hiệu ứng mượt
    safe_caption = (caption or '').replace('"', '').replace("'", '').replace("<", "&lt;").replace(">", "&gt;")
    import uuid
    tooltip_id = f"tt_{uuid.uuid4().hex}"
    html = (
        '<style>\n'
        '.img-tooltip-wrap { position:relative; display:inline-block; z-index:100; }\n'
        '.img-tooltip-img { border-radius:8px; }\n'
        '.img-tooltip-text {\n'
        '  visibility:hidden; opacity:0; pointer-events:auto;\n'
        '  background:rgba(30,30,30,0.97); color:#fff; text-align:center; border-radius:10px;\n'
        '  padding:10px 34px; position:absolute; z-index:9999; top:calc(100% + 12px); left:50%; transform:translateX(-50%);\n'
        '  font-size:15px; font-weight:400; min-width:420px; max-width:450px; max-height:250px; overflow:auto; box-shadow:0 4px 16px rgba(0,0,0,0.25);\n'
        '  transition:opacity 0.25s; white-space:pre-line;\n'
        '}\n'
        '.img-tooltip-text.left { left:0; transform:translateX(0); }\n'
        '.img-tooltip-text.right { left:auto; right:0; transform:translateX(-100%); }\n'
        '.img-tooltip-text::before {\n'
        '  content:""; position:absolute; bottom:100%; left:50%; transform:translateX(-50%);\n'
        '  border-width:8px; border-style:solid; border-color:transparent transparent rgba(30,30,30,0.97) transparent;\n'
        '}\n'
        '.img-tooltip-wrap:hover .img-tooltip-text { visibility:visible; opacity:1; }\n'
        '</style>\n'
    )
    html += f'<div class="img-tooltip-wrap" id="{tooltip_id}">'\
            f'<img class="img-tooltip-img" src="data:image/png;base64,{img_b64}" width="{img_width}">'\
            f'<div class="img-tooltip-text">{safe_caption}</div>'\
            '</div>'
    html += f'''<script>
    (function() {{
        var wrap = document.getElementById('{tooltip_id}');
        if (!wrap) return;
        var img = wrap.querySelector('.img-tooltip-img');
        var tip = wrap.querySelector('.img-tooltip-text');
        img.addEventListener('mousemove', function() {{
            var rect = img.getBoundingClientRect();
            var ww = window.innerWidth;
            var tw = tip.offsetWidth;
            tip.classList.remove('left','right');
            if(rect.left+rect.width/2-tw/2<10){{tip.classList.add('left');}}
            else if(rect.left+rect.width/2+tw/2>ww-10){{tip.classList.add('right');}}
            else{{tip.classList.remove('left','right');}}
        }});
    }})();
    </script>'''
    st.markdown(html, unsafe_allow_html=True)

for row_start in range(0, len(scripts), per_row):
    cols = st.columns(per_row)
    for col_idx, (idx, row_script) in enumerate(scripts[row_start:row_start+per_row]):
        with cols[col_idx]:
            img_files = sorted(glob.glob(f"./tmp/generated_post_{row_script['post_id']}_script_{row_script['id']}_*.png"), reverse=True)
            options = img_files.copy()
            if pd.notna(row_script.get("image_url")) and str(row_script["image_url"]).strip():
                options.append(row_script["image_url"])
            st.markdown(f"**Script {row_script['id']}**")
            if options:
                def _img_label(x):
                    if x in img_files:
                        return os.path.basename(x)
                    return "[GCP] " + os.path.basename(str(x))
                selected = st.selectbox(f"Chọn ảnh đại diện", options, format_func=_img_label, key=f"gallery_select_{row_script['id']}")
                # Hiển thị ảnh preview lớn hơn, có tooltip
                pil_img = None
                if selected in img_files:
                    pil_img = PILImage.open(selected)
                else:
                    try:
                        from urllib.request import urlopen
                        with urlopen(selected) as resp:
                            pil_img = PILImage.open(BytesIO(resp.read()))
                    except Exception:
                        pil_img = None
                if pil_img:
                    pil_img = pil_img.resize((img_width, int(pil_img.height * img_width / pil_img.width)))
                    image_with_tooltip(pil_img, row_script["transcript"] or "", key=f"imgtt_{row_script['id']}")
                gallery_selected[row_script["id"]] = selected
            else:
                st.info("Chưa có ảnh nào cho script này.")


# ================== GENERATE ẢNH CHẠY SONG SONG (TOÀN BỘ) ==================
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

def _build_prompt(row, lang: str) -> str:
    if lang.startswith("English"):
        return (row.get("context_prompt") or "").strip() + "\n\n" + (row.get("image_prompt") or "").strip()
    else:
        return (row.get("context_prompt_vietnamese") or "").strip() + "\n\n" + (row.get("image_prompt_vietnamese") or "").strip()

def generate_images_for_script(row, client, model_name, n_imgs, aspect):
    """
    Trả về (ok: bool, saved_files: list[str], error: str|None)
    """
    try:
        prompt_text = _build_prompt(row, _lang_choice)
        if not prompt_text.strip():
            return False, [], "Prompt rỗng"

        resp = client.models.generate_images(
            model=model_name,
            prompt=prompt_text,
            config=gtypes.GenerateImagesConfig(
                number_of_images=n_imgs,
                aspect_ratio=aspect
            )
        )
        if not getattr(resp, "generated_images", None):
            return False, [], "API không trả ảnh"

        os.makedirs("./tmp", exist_ok=True)
        ts = int(time.time() * 1000)
        saved_files = []
        for i, img_obj in enumerate(resp.generated_images):
            pil_img = _to_pil(img_obj.image)
            out_path = f"./tmp/generated_post_{row['post_id']}_script_{row['id']}_{ts}_{i}.png"
            pil_img.save(out_path)
            saved_files.append(out_path)
        return True, saved_files, None
    except Exception as e:
        return False, [], str(e)

# Nút generate ảnh cho tất cả script (song song + progress)
st.markdown("---")
if st.button("✨ Generate ảnh cho tất cả script (song song + progress)"):
    if not _api_key:
        st.error("⚠️ Hãy nhập Gemini API Key trước.")
    else:
        client = genai.Client(api_key=_api_key)

        # Thanh progress + khu vực log
        progress = st.progress(0.0, text="Đang chuẩn bị...")
        log_area = st.empty()

        rows = [row for _, row in scripts_df.iterrows()]
        total = len(rows)
        done = 0

        # Tạo executor chạy song song
        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for row in rows:
                fut = executor.submit(
                    generate_images_for_script,
                    row=row,
                    client=client,
                    model_name=_model_name,
                    n_imgs=n_images,
                    aspect=_aspect_ratio
                )
                futures[fut] = row

            # Thu kết quả dần dần và cập nhật progress
            for fut in as_completed(futures):
                row = futures[fut]
                ok, saved_files, err = fut.result()
                done += 1
                pct = done / total
                progress.progress(pct, text=f"Đang tạo ảnh... {done}/{total} script")

                # Ghi log từng script
                if ok:
                    log_area.write(f"✅ Script {row['id']}: tạo {len(saved_files)} ảnh → {', '.join(os.path.basename(p) for p in saved_files)}")
                else:
                    log_area.write(f"❌ Script {row['id']}: {err or 'Lỗi không xác định'}")

        progress.empty()
        st.success("🎉 Đã tạo xong ảnh cho tất cả script!")
        st.rerun()


# Đặt nút duyệt xuống cuối cùng, đổi tên thành 'Duyệt'
if st.button("✅ Duyệt", key="approve_all_image_post"):
    # Debug: log gallery_selected
    st.write("[DEBUG] Ảnh đang chọn cho từng script:", gallery_selected)
    missing = []
    for idx, row_script in scripts_df.iterrows():
        script_id = row_script["id"]
        selected = gallery_selected.get(script_id)
        if not selected:
            missing.append(script_id)
            continue
        # Nếu là file local thì upload lên GCP
        if os.path.exists(selected):
            upload_success, gcp_url = upload_to_gcp(selected, f"generated_post_{row_script['post_id']}_script_{script_id}.png")
            if upload_success:
                execute("""
                    UPDATE scripts
                    SET image_url = %s
                    WHERE id = %s
                """, (gcp_url, script_id))
                # Xóa tất cả file local liên quan script này trong tmp
                for f in glob.glob(f"./tmp/generated_post_{row_script['post_id']}_script_{script_id}_*.png"):
                    try:
                        os.remove(f)
                    except Exception:
                        pass
                st.success(f"Script {script_id}: Đã upload lên GCP, cập nhật DB và xóa file local.")
            else:
                st.error(f"Script {script_id}: Upload lên GCP thất bại!")
        else:
            # Nếu là link GCP thì không cần upload, chỉ cập nhật DB nếu chưa có
            if selected == row_script.get("image_url"):
                st.info(f"Script {script_id}: Ảnh đã ở trên GCP.")
            else:
                execute("""
                    UPDATE scripts
                    SET image_url = %s
                    WHERE id = %s
                """, (selected, script_id))
                st.success(f"Script {script_id}: Đã cập nhật DB với link GCP.")
    if missing:
        st.warning(f"Chưa chọn ảnh cho các script: {', '.join(map(str, missing))}. Vui lòng chọn đủ ảnh trước khi duyệt.")
    else:
        # Cập nhật trạng thái bài viết
        execute("""
            UPDATE new_social_posts
            SET post_state = 'IMAGE_PROMPT_CHECKED'
            WHERE id = %s
        """, (selected_id,))
        st.success("Đã lưu tất cả ảnh đã chọn lên GCP và cập nhật trạng thái bài viết sang IMAGE_PROMPT_CHECKED!")
        st.rerun()