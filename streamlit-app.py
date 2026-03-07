import streamlit as st
import numpy as np
from PIL import Image
import os
import time

st.set_page_config(
    page_title="EmotiSense · Emotion AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

EMOTIONS = {
    "Surprise":  {"emoji": "😲", "color": "#F59E0B", "desc": "Unexpected stimulus"},
    "Fear":      {"emoji": "😨", "color": "#8B5CF6", "desc": "Threat perceived"},
    "Disgust":   {"emoji": "🤢", "color": "#10B981", "desc": "Aversive response"},
    "Happiness": {"emoji": "😄", "color": "#F97316", "desc": "Positive affect"},
    "Sadness":   {"emoji": "😢", "color": "#3B82F6", "desc": "Negative affect"},
    "Anger":     {"emoji": "😠", "color": "#EF4444", "desc": "High arousal"},
    "Neutral":   {"emoji": "😐", "color": "#9CA3AF", "desc": "No strong affect"},
}
CLASS_NAMES = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]

THEMES = {
    "dark": {
        "bg":     "#0D0D14",
        "card":   "#13131E",
        "border": "rgba(255,255,255,0.09)",
        "text":   "#EEEEF5",
        "muted":  "#60607A",
        "accent": "#7C3AED",
        "glow":   "rgba(124,58,237,0.35)",
        "bar_bg": "rgba(255,255,255,0.07)",
        "meta_bg":"rgba(255,255,255,0.04)",
    },
    "light": {
        "bg":     "#F2F1F8",
        "card":   "#FFFFFF",
        "border": "rgba(0,0,0,0.09)",
        "text":   "#18182C",
        "muted":  "#707090",
        "accent": "#6D28D9",
        "glow":   "rgba(109,40,217,0.18)",
        "bar_bg": "rgba(0,0,0,0.07)",
        "meta_bg":"rgba(0,0,0,0.03)",
    },
}

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

T = THEMES[st.session_state.theme]
is_dark = st.session_state.theme == "dark"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main .block-container {{
    background-color: {T['bg']} !important;
    color: {T['text']} !important;
    font-family: 'DM Sans', sans-serif !important;
}}
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
    background-color: {T['card']} !important;
    border-right: 1px solid {T['border']} !important;
}}
[data-testid="stHeader"] {{ background: transparent !important; border-bottom: 1px solid {T['border']} !important; }}
#MainMenu, footer {{ visibility: hidden !important; }}
::-webkit-scrollbar {{ width: 4px; }}
::-webkit-scrollbar-thumb {{ background: {T['accent']}; border-radius: 2px; }}
[data-testid="stFileUploadDropzone"] {{
    background: {T['meta_bg']} !important;
    border: 2px dashed {T['border']} !important;
    border-radius: 14px !important;
}}
.stButton > button {{
    background: linear-gradient(135deg, {T['accent']}, #4F46E5) !important;
    color: #fff !important; border: none !important; border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 0.95rem !important; padding: 0.65rem 1.5rem !important;
    box-shadow: 0 4px 20px {T['glow']} !important;
    transition: all 0.2s ease !important; width: 100% !important;
}}
.stButton > button:hover {{ transform: translateY(-2px) !important; box-shadow: 0 8px 30px {T['glow']} !important; }}
p, label, .stMarkdown p {{ color: {T['text']} !important; font-family: 'DM Sans', sans-serif !important; }}
[data-testid="stRadio"] label {{ color: {T['text']} !important; }}
[data-testid="stImage"] img {{ border-radius: 12px !important; border: 1px solid {T['border']} !important; }}
hr {{ border-color: {T['border']} !important; opacity: 1 !important; }}
[data-testid="stCaptionContainer"] p {{ color: {T['muted']} !important; }}
@keyframes floatEmoji {{ 0%,100% {{ transform:translateY(0); }} 50% {{ transform:translateY(-8px); }} }}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    from ultralytics import YOLO
    base = os.path.dirname(os.path.abspath(__file__))
    pt_path   = os.path.join(base, "best.pt")
    onnx_path = os.path.join(base, "best.onnx")
    if os.path.exists(pt_path):
        return YOLO(pt_path), ".pt"
    elif os.path.exists(onnx_path):
        return YOLO(onnx_path), ".onnx"
    raise FileNotFoundError("No model file found (best.pt or best.onnx)")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:16px 0 20px;">
      <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;
                  background:linear-gradient(135deg,#7C3AED,#F59E0B);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;display:inline-block;">
        🧠 EmotiSense
      </div>
      <div style="font-size:0.78rem;color:{T['muted']};margin-top:3px;">Facial Emotion Recognition</div>
    </div>
    <hr style="border-color:{T['border']};margin:0 0 16px;">
    """, unsafe_allow_html=True)

    st.markdown(f"""<div style="font-family:'Syne',sans-serif;font-size:0.68rem;font-weight:700;
        letter-spacing:1.5px;text-transform:uppercase;color:{T['muted']};margin-bottom:8px;">🎨 Theme</div>""",
        unsafe_allow_html=True)

    choice = st.radio("theme", ["🌙 Dark", "☀️ Light"],
                      index=0 if is_dark else 1, label_visibility="collapsed", horizontal=True)
    new_theme = "dark" if "Dark" in choice else "light"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    st.markdown(f"<hr style='border-color:{T['border']};margin:16px 0;'>", unsafe_allow_html=True)

    st.markdown(f"""<div style="font-family:'Syne',sans-serif;font-size:0.68rem;font-weight:700;
        letter-spacing:1.5px;text-transform:uppercase;color:{T['muted']};margin-bottom:10px;">🤖 Model Status</div>""",
        unsafe_allow_html=True)

    model_ok = False
    try:
        model, fmt = load_model()
        model_ok = True
        st.markdown(f"""
        <div style="display:inline-flex;align-items:center;gap:8px;
                    background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);
                    border-radius:20px;padding:6px 14px;">
          <span style="width:8px;height:8px;border-radius:50%;background:#10B981;
                       display:inline-block;box-shadow:0 0 6px #10B981;"></span>
          <span style="font-size:0.8rem;color:#10B981;">YOLOv8n-cls {fmt} · Ready</span>
        </div>""", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ {e}")

    st.markdown(f"<hr style='border-color:{T['border']};margin:16px 0;'>", unsafe_allow_html=True)

    st.markdown(f"""<div style="font-family:'Syne',sans-serif;font-size:0.68rem;font-weight:700;
        letter-spacing:1.5px;text-transform:uppercase;color:{T['muted']};margin-bottom:10px;">📊 Emotion Classes</div>""",
        unsafe_allow_html=True)

    for name, meta in EMOTIONS.items():
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;padding:5px 2px;">
          <span style="font-size:1.3rem;">{meta['emoji']}</span>
          <div>
            <div style="font-size:0.83rem;font-weight:500;color:{T['text']};">{name}</div>
            <div style="font-size:0.68rem;color:{T['muted']};">{meta['desc']}</div>
          </div>
          <div style="margin-left:auto;width:8px;height:8px;border-radius:50%;background:{meta['color']};flex-shrink:0;"></div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"<hr style='border-color:{T['border']};margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.72rem;color:{T['muted']};line-height:1.7;">
      Trained on <b style="color:{T['text']}">RAF-DB</b> dataset<br>
      Architecture: <b style="color:{T['text']}">YOLOv8n</b><br>
      Input size: <b style="color:{T['text']}">64×64 px</b>
    </div>""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
badges = "".join(f'<span style="display:inline-block;background:rgba(255,255,255,0.15);border:1px solid rgba(255,255,255,0.25);color:#fff;font-size:0.7rem;font-weight:600;padding:4px 12px;border-radius:20px;margin-right:8px;letter-spacing:0.5px;text-transform:uppercase;">{b}</span>' for b in ["YOLOv8n-cls","RAF-DB","7 Emotions","64×64"])
st.markdown(f"""
<div style="background:linear-gradient(135deg,{T['accent']} 0%,#4F46E5 55%,#0EA5E9 100%);
            border-radius:20px;padding:36px 44px;margin-bottom:28px;
            position:relative;overflow:hidden;box-shadow:0 16px 50px {T['glow']};">
  <div style="position:absolute;inset:0;opacity:0.06;background:radial-gradient(circle at 80% 50%,#fff 0%,transparent 60%);"></div>
  <div style="font-family:'Syne',sans-serif;font-size:2.6rem;font-weight:800;color:#fff;letter-spacing:-1px;position:relative;">🧠 EmotiSense</div>
  <div style="font-size:1rem;color:rgba(255,255,255,0.78);margin-top:6px;font-weight:300;position:relative;">Real-time facial emotion analysis powered by YOLOv8</div>
  <div style="margin-top:16px;position:relative;">{badges}</div>
</div>
""", unsafe_allow_html=True)

if not model_ok:
    st.stop()

col_upload, col_result = st.columns([1, 1], gap="large")

# ── UPLOAD ────────────────────────────────────────────────────────────────────
with col_upload:
    st.markdown(f"""<div style="font-family:'Syne',sans-serif;font-size:0.68rem;font-weight:700;
                letter-spacing:2px;text-transform:uppercase;color:{T['muted']};margin-bottom:12px;">📸 Input Image</div>""",
                unsafe_allow_html=True)

    uploaded = st.file_uploader("img", type=["jpg","jpeg","png","webp","bmp"], label_visibility="collapsed")

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_container_width=True, caption="Uploaded image")
        st.markdown(f"""
        <div style="background:{T['meta_bg']};border:1px solid {T['border']};border-radius:12px;
                    padding:16px 20px;margin-top:12px;display:flex;gap:28px;flex-wrap:wrap;">
          <div>
            <div style="font-size:0.65rem;color:{T['muted']};text-transform:uppercase;letter-spacing:1px;">Size</div>
            <div style="font-size:0.95rem;font-weight:600;color:{T['text']};font-family:'Syne',sans-serif;">{img.width}×{img.height}</div>
          </div>
          <div>
            <div style="font-size:0.65rem;color:{T['muted']};text-transform:uppercase;letter-spacing:1px;">Format</div>
            <div style="font-size:0.95rem;font-weight:600;color:{T['text']};font-family:'Syne',sans-serif;">{uploaded.type.split("/")[1].upper()}</div>
          </div>
          <div>
            <div style="font-size:0.65rem;color:{T['muted']};text-transform:uppercase;letter-spacing:1px;">Mode</div>
            <div style="font-size:0.95rem;font-weight:600;color:{T['text']};font-family:'Syne',sans-serif;">{img.mode}</div>
          </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("⚡ Analyse Emotion", use_container_width=True)
    else:
        st.markdown(f"""
        <div style="border:2px dashed {T['border']};border-radius:16px;padding:48px 24px;text-align:center;">
          <div style="font-size:3rem;margin-bottom:12px;">🖼️</div>
          <div style="font-size:1rem;font-weight:600;color:{T['text']};margin-bottom:8px;font-family:'Syne',sans-serif;">Drop a face image here</div>
          <div style="font-size:0.85rem;color:{T['muted']};">JPG · PNG · WEBP · BMP</div>
          <div style="font-size:0.78rem;color:{T['muted']};margin-top:6px;">Best with front-facing, well-lit portraits</div>
        </div>""", unsafe_allow_html=True)
        run_btn = False

# ── RESULT ────────────────────────────────────────────────────────────────────
with col_result:
    st.markdown(f"""<div style="font-family:'Syne',sans-serif;font-size:0.68rem;font-weight:700;
                letter-spacing:2px;text-transform:uppercase;color:{T['muted']};margin-bottom:12px;">🎯 Analysis Result</div>""",
                unsafe_allow_html=True)

    if uploaded and run_btn:
        with st.spinner("Analysing…"):
            t0 = time.time()
            results = model.predict(img, imgsz=64, verbose=False)
            elapsed_ms = (time.time() - t0) * 1000

        probs    = results[0].probs
        top_idx  = int(probs.top1)
        top_conf = float(probs.top1conf)
        emo_name = CLASS_NAMES[top_idx]
        emo      = EMOTIONS[emo_name]
        ec       = emo["color"]

        st.markdown(f"""
        <div style="background:{T['card']};border:1px solid {T['border']};border-radius:20px;
                    padding:32px;text-align:center;position:relative;overflow:hidden;margin-bottom:16px;">
          <div style="position:absolute;top:0;left:0;right:0;height:4px;
                      background:linear-gradient(90deg,{ec},{ec}66);"></div>
          <div style="font-size:5rem;animation:floatEmoji 3s ease-in-out infinite;display:block;margin-bottom:8px;">{emo['emoji']}</div>
          <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:{ec};margin-bottom:4px;">{emo_name}</div>
          <div style="font-size:0.88rem;color:{T['muted']};">{emo['desc']}</div>
          <div style="display:flex;justify-content:center;gap:32px;margin-top:24px;
                      padding-top:20px;border-top:1px solid {T['border']};">
            <div style="text-align:center;">
              <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:{ec};">{top_conf*100:.1f}%</div>
              <div style="font-size:0.68rem;color:{T['muted']};text-transform:uppercase;letter-spacing:1px;">Confidence</div>
            </div>
            <div style="text-align:center;">
              <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:{T['text']};">{elapsed_ms:.0f}ms</div>
              <div style="font-size:0.68rem;color:{T['muted']};text-transform:uppercase;letter-spacing:1px;">Inference</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div style="font-family:'Syne',sans-serif;font-size:0.68rem;font-weight:700;
                    letter-spacing:2px;text-transform:uppercase;color:{T['muted']};margin-bottom:10px;">📊 All Probabilities</div>""",
                    unsafe_allow_html=True)

        all_probs  = probs.data.cpu().numpy()
        sorted_idx = np.argsort(all_probs)[::-1]

        bars = f'<div style="background:{T["meta_bg"]};border:1px solid {T["border"]};border-radius:14px;padding:20px 22px;">'
        for i in sorted_idx:
            name = CLASS_NAMES[i]
            prob = float(all_probs[i])
            m    = EMOTIONS[name]
            bold = "700" if i == top_idx else "400"
            bars += f"""
            <div style="margin-bottom:10px;">
              <div style="display:flex;justify-content:space-between;font-size:0.82rem;margin-bottom:4px;">
                <span style="font-weight:{bold};color:{T['text']};">{m['emoji']} {name}</span>
                <span style="font-weight:{bold};color:{m['color']};">{prob*100:.1f}%</span>
              </div>
              <div style="height:7px;border-radius:4px;background:{T['bar_bg']};">
                <div style="width:{prob*100:.1f}%;height:100%;border-radius:4px;
                            background:linear-gradient(90deg,{m['color']},{m['color']}88);"></div>
              </div>
            </div>"""
        bars += "</div>"
        st.markdown(bars, unsafe_allow_html=True)

    elif not uploaded:
        st.markdown(f"""
        <div style="background:{T['card']};border:1px solid {T['border']};border-radius:20px;
                    padding:60px 32px;text-align:center;">
          <div style="font-size:4rem;opacity:0.25;margin-bottom:16px;">🔮</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:{T['muted']};">Awaiting Input</div>
          <div style="font-size:0.83rem;color:{T['muted']};margin-top:6px;">Upload a face image to begin</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:{T['card']};border:1px solid {T['border']};border-radius:20px;
                    padding:60px 32px;text-align:center;">
          <div style="font-size:4rem;opacity:0.4;margin-bottom:16px;">👆</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:{T['muted']};">Ready to Analyse</div>
          <div style="font-size:0.83rem;color:{T['muted']};margin-top:6px;">Click the button to run inference</div>
        </div>""", unsafe_allow_html=True)

st.markdown(f"""
<div style="text-align:center;margin-top:40px;padding:20px 0;border-top:1px solid {T['border']};">
  <span style="font-size:0.75rem;color:{T['muted']};">EmotiSense · YOLOv8n-cls · RAF-DB Dataset · 7 Basic Emotions</span>
</div>""", unsafe_allow_html=True)
