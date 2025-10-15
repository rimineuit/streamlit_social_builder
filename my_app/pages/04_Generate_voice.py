import os
import glob
import time
import hashlib
import wave
import pandas as pd
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv
import streamlit as st

from google import genai
from google.genai import types


# ===================== GCP UPLOAD =====================
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

        # storage.Client() tự đọc credentials từ GOOGLE_APPLICATION_CREDENTIALS
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(dest_blob_name)
        blob.upload_from_filename(local_path)

        # Nếu bucket private, public_url có thể không truy cập được
        # Có thể bật public nếu bạn muốn (yêu cầu quyền storage.objects.update):
        try:
            blob.make_public()
        except Exception:
            # nếu không thể public (do chính sách), vẫn trả về public_url (có thể không mở được)
            pass

        return True, blob.public_url

    except Exception as e:
        st.error(f"❌ Lỗi upload GCP: {e}")
        return False, None


# ===================== AUDIO UTILS =====================
def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)


VOICE_DESCRIPTIONS = {
    'Achernar': {'desc': 'Soft', 'gender': 'Female'},
    'Achird': {'desc': 'Friendly', 'gender': 'Male'},
    'Algenib': {'desc': 'Gravelly', 'gender': 'Male'},
    'Algieba': {'desc': 'Smooth', 'gender': 'Male'},
    'Alnilam': {'desc': 'Corporate', 'gender': 'Male'},
    'Aoede': {'desc': 'Breezy', 'gender': 'Female'},
    'Autonoe': {'desc': 'Bright', 'gender': 'Female'},
    'Callirrhoe': {'desc': 'Easygoing', 'gender': 'Female'},
    'Charon': {'desc': 'Informative', 'gender': 'Male'},
    'Despina': {'desc': 'Smooth', 'gender': 'Female'},
    'Enceladus': {'desc': 'Breathy', 'gender': 'Male'},
    'Erinome': {'desc': 'Clear', 'gender': 'Female'},
    'Fenrir': {'desc': 'Excitable', 'gender': 'Male'},
    'Gacrux': {'desc': 'Mature', 'gender': 'Female'},
    'Iapetus': {'desc': 'Clear', 'gender': 'Male'},
    'Kore': {'desc': 'Corporate', 'gender': 'Female'},
    'Laomedeia': {'desc': 'Cheerful', 'gender': 'Female'},
    'Leda': {'desc': 'Youthful', 'gender': 'Female'},
    'Orus': {'desc': 'Corporate', 'gender': 'Male'},
    'Pulcherrima': {'desc': 'Upbeat', 'gender': 'Female'},
    'Puck': {'desc': 'Cheerful', 'gender': 'Male'},
    'Rasalgethi': {'desc': 'Informative', 'gender': 'Male'},
    'Sadachbia': {'desc': 'Lively', 'gender': 'Male'},
    'Sadaltager': {'desc': 'Knowledgeable', 'gender': 'Male'},
    'Schedar': {'desc': 'Even', 'gender': 'Male'},
    'Sulafat': {'desc': 'Warm', 'gender': 'Female'},
    'Umbriel': {'desc': 'Easygoing', 'gender': 'Male'},
    'Vindemiatrix': {'desc': 'Gentle', 'gender': 'Female'},
    'Zephyr': {'desc': 'Bright', 'gender': 'Female'},
    'Zubenelgenubi': {'desc': 'Casual', 'gender': 'Male'},
}
VOICE_LIST = list(VOICE_DESCRIPTIONS.keys())


# ===================== STREAMLIT PAGE =====================
st.set_page_config(page_title="Trang tạo voice cho bài viết đã duyệt ảnh", layout="wide", initial_sidebar_state="expanded")
st.title("🔊 Trang tạo voice cho bài viết đã duyệt ảnh")
load_dotenv(dotenv_path="./.env")


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
st.header("📊 Danh sách các bài viết ở công đoạn tạo audio")
posts = query_df("""
    SELECT id, title, hash_tags, post_state
    FROM new_social_posts
    WHERE post_state = 'IMAGE_PROMPT_CHECKED'
    ORDER BY id DESC
    LIMIT %s
""", (200,))

st.dataframe(posts, use_container_width=True)
st.subheader("✍️ Chọn bản ghi để tạo voice")

options = [f"{row.id} — {str(row.title)}" for _, row in posts.iterrows()]
display_to_id = {opt: int(opt.split(" — ")[0]) for opt in options}

selected_display = st.selectbox("Chọn bản ghi theo ID", options) if options else None
selected_id = display_to_id[selected_display] if selected_display in display_to_id else None

if not options or selected_id is None:
    st.warning("Không có bài viết nào ở trạng thái IMAGE_PROMPT_CHECKED để tạo voice.")
    st.stop()


# ==== Chọn giọng điệu và nhân vật áp dụng mặc định cho toàn bộ transcript ====
st.markdown("---")
st.header(":loud_sound: Giọng điệu (tone, tuỳ chọn) mặc định cho toàn bộ đoạn audio", divider="rainbow")

voice_options = [f"{v} ({VOICE_DESCRIPTIONS[v]['gender']}) -- {VOICE_DESCRIPTIONS[v]['desc']}" for v in VOICE_LIST]
default_voice_label = st.selectbox("Chọn nhân vật (voice) mặc định", voice_options, key="default_voice")
default_voice = default_voice_label.split(' (')[0]

# Play example audio for selected voice if available
example_audio_path = os.path.join("example_audio", f"{default_voice}.wav")
if os.path.exists(example_audio_path):
    st.audio(example_audio_path, format="audio/wav", start_time=0)
    st.caption(f"Voice sample: {default_voice}")
else:
    st.caption(f"No sample available for {default_voice}")

default_tone = st.text_input(
    "Nhập giọng điệu (tone) mặc định",
    value="Giọng điệu đọc bài phong thủy, trầm ấm, rõ ràng: ",
    key="default_tone",
    help="Ví dụ: Vui vẻ, Sống động, Trưởng thành..."
)


# ===================== GEMINI CLIENT =====================
# Lưu ý: KHÔNG nên hardcode API key trong code production
load_dotenv(dotenv_path="./.env")
API_KEY=os.getenv("GEMINI_API_KEY", "").strip()
client = genai.Client(api_key=API_KEY)


# ===================== LOAD SCRIPTS =====================
scripts_df = query_df(
    '''SELECT id, character, transcript, audio_url FROM scripts WHERE post_id = %s ORDER BY idx ASC, id ASC''',
    (selected_id,)
)

st.header("📝 Danh sách transcript và audio từng đoạn")
if scripts_df.empty:
    st.info("Chưa có transcript nào cho bài viết này trong bảng scripts.")
else:
    # Gallery audio cho từng script trong expander
    for idx, row in scripts_df.iterrows():
        with st.expander(f"Đoạn {idx+1}"):
            st.markdown("**Transcript sẽ được sinh audio:**")
            st.text_area("", row['transcript'], height=120, key=f"transcript_{row['id']}", disabled=True)
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Tạo audio Flash (Đoạn {row['id']})", key=f"tts_flash_{row['id']}"):
                    prompt = (default_tone.strip() + ": " if default_tone.strip() else "") + (row['transcript'] or "")
                    response = client.models.generate_content(
                        model="gemini-2.5-flash-preview-tts",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_modalities=["AUDIO"],
                            speech_config=types.SpeechConfig(
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                        voice_name=default_voice,
                                    )
                                )
                            ),
                        )
                    )
                    data = response.candidates[0].content.parts[0].inline_data.data
                    os.makedirs("./tmp", exist_ok=True)
                    ts = int(time.time() * 1000)
                    audio_flash_file = f"./tmp/audio_post_{selected_id}_audio_{row['id']}_flash_{ts}.wav"

                    # chống trùng
                    def _hash(d): return hashlib.md5(d).hexdigest()
                    new_hash = _hash(data)
                    duplicate = False
                    for f in glob.glob(f"./tmp/audio_post_{selected_id}_audio_{row['id']}_flash_*.wav"):
                        try:
                            with open(f, 'rb') as existing:
                                if _hash(existing.read()) == new_hash:
                                    duplicate = True
                                    break
                        except Exception:
                            continue
                    if duplicate:
                        st.warning("Audio Flash này đã tồn tại, không tạo file trùng lặp.")
                    else:
                        wave_file(audio_flash_file, data)
            with col2:
                if st.button(f"Tạo audio Pro (Đoạn {row['id']})", key=f"tts_pro_{row['id']}"):
                    prompt = (default_tone.strip() + ": " if default_tone.strip() else "") + (row['transcript'] or "")
                    response = client.models.generate_content(
                        model="gemini-2.5-pro-preview-tts",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_modalities=["AUDIO"],
                            speech_config=types.SpeechConfig(
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                        voice_name=default_voice,
                                    )
                                )
                            ),
                        )
                    )
                    data = response.candidates[0].content.parts[0].inline_data.data
                    os.makedirs("./tmp", exist_ok=True)
                    ts = int(time.time() * 1000)
                    audio_pro_file = f"./tmp/audio_post_{selected_id}_audio_{row['id']}_pro_{ts}.wav"
                    wave_file(audio_pro_file, data)

            # Gộp tất cả file flash và pro
            audio_flash_pattern = f"./tmp/audio_post_{selected_id}_audio_{row['id']}_flash_*.wav"
            audio_pro_pattern = f"./tmp/audio_post_{selected_id}_audio_{row['id']}_pro_*.wav"
            all_files = sorted(glob.glob(audio_flash_pattern) + glob.glob(audio_pro_pattern), reverse=True)

            if all_files:
                def label_func(x):
                    if '_flash_' in x:
                        return 'Flash ' + x.split('_flash_')[-1].replace('.wav', '')
                    elif '_pro_' in x:
                        return 'Pro ' + x.split('_pro_')[-1].replace('.wav', '')
                    return x

                all_choice = st.selectbox(
                    f"Chọn 1 bản audio để duyệt",
                    all_files,
                    format_func=label_func,
                    key=f"radio_all_{row['id']}"
                )
                show_audio = False
                if all_choice and os.path.exists(all_choice) and os.path.getsize(all_choice) > 1000:
                    show_audio = True
                if show_audio:
                    st.audio(all_choice, format="audio/wav")

            # Auto-play nếu vừa tạo (tuỳ biến theo session_state)
            autoplay_key = f'autoplay_audio_{row["id"]}'
            if autoplay_key in st.session_state:
                st.markdown(f'''<audio src="{st.session_state[autoplay_key]}" controls autoplay></audio>''', unsafe_allow_html=True)
                del st.session_state[autoplay_key]

            # Trạng thái đã duyệt trước đó
            if row['audio_url'] and all_files:
                if os.path.basename(row['audio_url']) in [os.path.basename(p) for p in all_files]:
                    st.info("Đã duyệt và upload bản audio này trước đó.")


# ========== BULK GENERATE ==========
def generate_audio_flash(row):
    try:
        prompt = (default_tone.strip() + ": " if default_tone.strip() else "") + (row['transcript'] or "")
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=default_voice,
                        )
                    )
                ),
            )
        )
        data = response.candidates[0].content.parts[0].inline_data.data
        os.makedirs("./tmp", exist_ok=True)
        ts = int(time.time() * 1000)
        audio_flash_file = f"./tmp/audio_post_{selected_id}_audio_{row['id']}_flash_{ts}.wav"

        # chống trùng nội dung
        def file_hash(d): return hashlib.md5(d).hexdigest()
        new_hash = file_hash(data)
        duplicate = False
        for f in glob.glob(f"./tmp/audio_post_{selected_id}_audio_{row['id']}_flash_*.wav"):
            try:
                with open(f, 'rb') as existing:
                    if file_hash(existing.read()) == new_hash:
                        duplicate = True
                        break
            except Exception:
                continue

        if not duplicate:
            wave_file(audio_flash_file, data)
        return True
    except Exception:
        return False


def generate_audio_pro(row):
    try:
        prompt = (default_tone.strip() + ": " if default_tone.strip() else "") + (row['transcript'] or "")
        response = client.models.generate_content(
            model="gemini-2.5-pro-preview-tts",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=default_voice,
                        )
                    )
                ),
            )
        )
        data = response.candidates[0].content.parts[0].inline_data.data
        os.makedirs("./tmp", exist_ok=True)
        ts = int(time.time() * 1000)
        audio_pro_file = f"./tmp/audio_post_{selected_id}_audio_{row['id']}_pro_{ts}.wav"

        # chống trùng
        def file_hash(d): return hashlib.md5(d).hexdigest()
        new_hash = file_hash(data)
        duplicate = False
        for f in glob.glob(f"./tmp/audio_post_{selected_id}_audio_{row['id']}_pro_*.wav"):
            try:
                with open(f, 'rb') as existing:
                    if file_hash(existing.read()) == new_hash:
                        duplicate = True
                        break
            except Exception:
                continue

        if not duplicate:
            wave_file(audio_pro_file, data)
        return True
    except Exception:
        return False


import concurrent.futures

if not scripts_df.empty:
    # Bulk Flash
    if st.button("✨ Tạo audio Flash cho tất cả đoạn transcript của bài viết này"):
        progress = st.progress(0, text="Đang tạo audio (Flash)...")
        rows = [row for _, row in scripts_df.iterrows()]
        total = len(rows)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(generate_audio_flash, row): i for i, row in enumerate(rows)}
            for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                progress.progress((idx + 1) / total, text=f"Đã tạo {idx + 1}/{total} đoạn (Flash)...")
        progress.empty()
        st.success("Đã tạo xong audio Flash cho tất cả đoạn transcript!")
        st.rerun()

    # Bulk Pro (MỚI)
    if st.button("🚀 Tạo audio Pro cho tất cả đoạn transcript của bài viết này"):
        progress = st.progress(0, text="Đang tạo audio (Pro)...")
        rows = [row for _, row in scripts_df.iterrows()]
        total = len(rows)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(generate_audio_pro, row): i for i, row in enumerate(rows)}
            for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                progress.progress((idx + 1) / total, text=f"Đã tạo {idx + 1}/{total} đoạn (Pro)...")
        progress.empty()
        st.success("Đã tạo xong audio Pro cho tất cả đoạn transcript!")
        st.rerun()


# ========== GALLERY AUDIO TỔNG HỢP Ở DƯỚI CÙNG ==========
st.markdown("---")
st.header("🎵 Gallery: Chọn audio đại diện cho từng đoạn và duyệt lên GCP")

per_row = 5
audio_selected = {}
scripts = list(scripts_df.iterrows())
for row_start in range(0, len(scripts), per_row):
    cols = st.columns(per_row)
    for col_idx, (idx, row_script) in enumerate(scripts[row_start:row_start + per_row]):
        with cols[col_idx]:
            audio_flash_pattern = f"./tmp/audio_post_{selected_id}_audio_{row_script['id']}_flash_*.wav"
            audio_pro_pattern = f"./tmp/audio_post_{selected_id}_audio_{row_script['id']}_pro_*.wav"
            all_files = sorted(glob.glob(audio_flash_pattern) + glob.glob(audio_pro_pattern), reverse=True)
            options = all_files.copy()
            if pd.notna(row_script.get("audio_url")) and str(row_script["audio_url"]).strip():
                options.append(row_script["audio_url"])
            st.markdown(f"**Script {row_script['id']}**")
            if options:
                def _audio_label(x):
                    if x in all_files:
                        return os.path.basename(x)
                    return "[GCP] " + os.path.basename(str(x))

                selected = st.selectbox(
                    f"Chọn audio đại diện",
                    options,
                    format_func=_audio_label,
                    key=f"gallery_audio_select_{row_script['id']}"
                )
                if selected in all_files:
                    st.audio(selected, format="audio/wav")
                else:
                    st.audio(selected)
                audio_selected[row_script["id"]] = selected
            else:
                st.info("Chưa có audio nào cho script này.")


# ========== DUYỆT & UPLOAD ==========
if st.button("✅ Duyệt", key="approve_all_audio_post"):
    missing = []
    for idx, row_script in scripts_df.iterrows():
        script_id = row_script["id"]
        selected = audio_selected.get(script_id)
        if not selected:
            missing.append(script_id)
            continue

        # Nếu là file local thì upload lên GCP
        if os.path.exists(selected):
            dest_name = f"audio_post_{selected_id}_audio_{script_id}.wav"
            ok, gcp_url = upload_to_gcp(selected, dest_name)  # <-- GIẢI NÉN TUPLE
            if ok and gcp_url:
                execute(
                    "UPDATE scripts SET audio_url = %s WHERE id = %s",
                    (gcp_url, script_id)
                )

                # Xóa file local
                for f in glob.glob(f"./tmp/audio_post_{selected_id}_audio_{script_id}_*.wav"):
                    try:
                        os.remove(f)
                    except Exception:
                        pass
                st.success(f"Script {script_id}: Đã upload lên GCP và cập nhật URL.")
            else:
                st.error(f"Script {script_id}: Upload lên GCP thất bại!")

        else:
            # Nếu đã là link GCP (hoặc link HTTP khác), chỉ cập nhật URL nếu khác DB
            current_url = (row_script.get("audio_url") or "").strip()
            selected_url = str(selected).strip()
            if selected_url and selected_url != current_url:
                execute(
                    "UPDATE scripts SET audio_url = %s WHERE id = %s",
                    (selected_url, script_id)
                )
                st.success(f"Script {script_id}: Đã cập nhật URL trong DB.")
            else:
                st.info(f"Script {script_id}: URL không thay đổi.")

    if missing:
        st.warning("Chưa chọn audio cho các script: " + ", ".join(map(str, missing)))
    else:
        rowcount = execute(
            "UPDATE new_social_posts SET post_state = 'AUDIO_CHECKED' WHERE id = %s",
            (selected_id,)
        )
        if rowcount > 0:
            st.success("✅ Đã cập nhật URL và chuyển trạng thái bài viết sang AUDIO_CHECKED!")
        else:
            st.warning("Không cập nhật được trạng thái bài viết.")
        st.rerun()