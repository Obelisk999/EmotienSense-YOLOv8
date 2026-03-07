import streamlit as st
import numpy as np
from PIL import Image
import io
import base64
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EmotiSense · Emotion AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Emotion metadata ──────────────────────────────────────────────────────────
EMOTIONS = {
    "Surprise":  {"emoji": "😲", "color": "#F59E0B", "desc": "Unexpected stimulus detected"},
    "Fear":      {"emoji": "😨", "color": "#8B5CF6", "desc": "Threat or danger perceived"},
    "Disgust":   {"emoji": "🤢", "color": "#10B981", "desc": "Aversive stimulus response"},
    "Happiness": {"emoji": "😄", "color": "#F97316", "desc": "Positive affect expressed"},
    "Sadness":   {"emoji": "😢", "color": "#3B82F6", "desc": "Negative affect expressed"},
    "Anger":     {"emoji": "😠", "color": "#EF4444", "desc": "High arousal negative affect"},
    "Neutral":   {"emoji": "😐", "color": "#6B7280", "desc": "No strong affect detected"},
}
CLASS_NAMES = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]

# ── Theme CSS ─────────────────────────────────────────────────────────────────
DARK_CSS = """
:root {
  --bg-primary:   #0A0A0F;
  --bg-card:      #12121A;
  --bg-glass:     rgba(255,255,255,0.04);
  --border:       rgba(255,255,255,0.08);
  --text-primary: #F0F0F5;
  --text-muted:   #6B6B80;
  --accent:       #7C3AED;
  --accent-glow:  rgba(124,58,237,0.35);
  --accent2:      #F59E0B;
}
"""

LIGHT_CSS = """
:root {
  --bg-primary:   #F5F4F8;
  --bg-card:      #FFFFFF;
  --bg-glass:     rgba(255,255,255,0.75);
  --border:       rgba(0,0,0,0.08);
  --text-primary: #1A1A2E;
  --text-muted:   #6B6B80;
  --accent:       #7C3AED;
  --accent-glow:  rgba(124,58,237,0.2);
  --accent2:      #F59E0B;
}
"""

COMMON_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg-primary) !important;
  color: var(--text-primary) !important;
  font-family: 'DM Sans', sans-serif !important;
  transition: background 0.4s ease, color 0.4s ease;
}

[data-testid="stSidebar"] {
  background: var(--bg-card) !important;
  border-right: 1px solid var(--border) !important;
}

[data-testid="stHeader"] { background: transparent !important; }

/* Hide default Streamlit elements */
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

/* Headings */
h1,h2,h3,h4 { font-family: 'Syne', sans-serif !important; color: var(--text-primary) !important; }

/* Custom scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 2px; }

/* Hero banner */
.hero-banner {
  background: linear-gradient(135deg, var(--accent) 0%, #4F46E5 50%, #0EA5E9 100%);
  border-radius: 20px;
  padding: 40px 48px;
  margin-bottom: 32px;
  position: relative;
  overflow: hidden;
  box-shadow: 0 20px 60px var(--accent-glow);
}
.hero-banner::before {
  content: '';
  position: absolute; inset: 0;
  background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.04'%3E%3Ccircle cx='30' cy='30' r='20'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
}
.hero-title {
  font-family: 'Syne', sans-serif;
  font-size: 2.8rem; font-weight: 800;
  color: #ffffff; letter-spacing: -1px;
  position: relative; z-index: 1;
}
.hero-sub {
  font-size: 1.05rem; color: rgba(255,255,255,0.75);
  margin-top: 8px; font-weight: 300;
  position: relative; z-index: 1;
}
.badge {
  display: inline-block;
  background: rgba(255,255,255,0.15);
  border: 1px solid rgba(255,255,255,0.25);
  color: #fff; font-size: 0.72rem; font-weight: 600;
  padding: 4px 12px; border-radius: 20px;
  margin-right: 8px; margin-top: 16px;
  letter-spacing: 0.5px; text-transform: uppercase;
  position: relative; z-index: 1;
}

/* Glass card */
.glass-card {
  background: var(--bg-glass);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 24px 28px;
  backdrop-filter: blur(20px);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover { transform: translateY(-2px); box-shadow: 0 12px 40px var(--accent-glow); }

/* Result card */
.result-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 32px;
  text-align: center;
  position: relative;
  overflow: hidden;
}
.result-card::after {
  content: '';
  position: absolute; top: 0; left: 0; right: 0;
  height: 4px;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
}

/* Emotion emoji */
.emotion-emoji {
  font-size: 5rem;
  display: block;
  animation: floatEmoji 3s ease-in-out infinite;
}
@keyframes floatEmoji {
  0%,100% { transform: translateY(0); }
  50% { transform: translateY(-8px); }
}

/* Emotion label */
.emotion-label {
  font-family: 'Syne', sans-serif;
  font-size: 2.2rem; font-weight: 800;
  margin: 12px 0 4px;
}
.emotion-desc {
  color: var(--text-muted);
  font-size: 0.9rem;
}

/* Confidence bar */
.conf-bar-wrap { margin: 8px 0; }
.conf-label {
  display: flex; justify-content: space-between;
  font-size: 0.82rem; margin-bottom: 4px;
  color: var(--text-primary);
}
.conf-bar-bg {
  height: 8px; border-radius: 4px;
  background: var(--border);
  overflow: hidden;
}
.conf-bar-fill {
  height: 100%; border-radius: 4px;
  transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Upload zone */
.upload-zone {
  border: 2px dashed var(--border);
  border-radius: 16px;
  padding: 40px;
  text-align: center;
  transition: border-color 0.3s, background 0.3s;
  cursor: pointer;
}
.upload-zone:hover {
  border-color: var(--accent);
  background: var(--accent-glow);
}
.upload-icon { font-size: 3rem; display: block; margin-bottom: 12px; }
.upload-text { color: var(--text-muted); font-size: 0.9rem; }

/* Sidebar label */
.sidebar-label {
  font-family: 'Syne', sans-serif;
  font-size: 0.7rem; font-weight: 700;
  letter-spacing: 1.5px; text-transform: uppercase;
  color: var(--text-muted); margin-bottom: 8px; margin-top: 20px;
}

/* Model info */
.model-chip {
  display: inline-flex; align-items: center; gap: 6px;
  background: var(--bg-glass);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 6px 14px; font-size: 0.8rem;
  color: var(--text-muted);
}
.model-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: #10B981;
  box-shadow: 0 0 6px #10B981;
  animation: pulse 2s infinite;
}
@keyframes pulse {
  0%,100% { opacity:1; } 50% { opacity:0.4; }
}

/* Section title */
.section-title {
  font-family: 'Syne', sans-serif;
  font-size: 0.72rem; font-weight: 700;
  letter-spacing: 2px; text-transform: uppercase;
  color: var(--text-muted); margin-bottom: 16px;
}

/* Streamlit widget overrides */
[data-testid="stFileUploadDropzone"] {
  background: var(--bg-glass) !important;
  border: 2px dashed var(--border) !important;
  border-radius: 16px !important;
  transition: border-color 0.3s !important;
}
[data-testid="stFileUploadDropzone"]:hover {
  border-color: var(--accent) !important;
}
.stButton > button {
  background: linear-gradient(135deg, var(--accent), #4F46E5) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 12px 28px !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: 0.9rem !important;
  letter-spacing: 0.5px !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 4px 20px var(--accent-glow) !important;
  width: 100% !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 30px var(--accent-glow) !important;
}
[data-testid="stMetricValue"] {
  font-family: 'Syne', sans-serif !important;
  color: var(--text-primary) !important;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.75rem !important; }

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stRadio"] label,
.stMarkdown p { color: var(--text-primary) !important; }

/* Image display */
[data-testid="stImage"] img {
  border-radius: 12px !important;
  border: 1px solid var(--border) !important;
}
"""

# ── Session defaults ──────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "model" not in st.session_state:
    st.session_state.model = None
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

# ── Apply theme ───────────────────────────────────────────────────────────────
theme_vars = DARK_CSS if st.session_state.theme == "dark" else LIGHT_CSS
st.markdown(f"<style>{theme_vars}{COMMON_CSS}</style>", unsafe_allow_html=True)

# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    from ultralytics import YOLO
    import os
    # prefer .pt (native, no extra runtime needed)
    pt_path = "/mnt/user-data/uploads/best.pt"
    if os.path.exists(pt_path):
        model = YOLO(pt_path)
        return model, "YOLOv8n-cls  ·  .pt"
    onnx_path = "/mnt/user-data/uploads/best.onnx"
    model = YOLO(onnx_path)
    return model, "YOLOv8n-cls  ·  .onnx"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 24px;'>
      <div style='font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;
                  background:linear-gradient(135deg,#7C3AED,#F59E0B);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        EmotiSense
      </div>
      <div style='font-size:0.78rem;color:var(--text-muted);margin-top:2px;'>
        Facial Emotion Recognition
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Theme toggle
    st.markdown('<div class="sidebar-label">🎨 Interface</div>', unsafe_allow_html=True)
    theme_choice = st.radio(
        "Theme",
        ["🌙 Dark", "☀️ Light"],
        index=0 if st.session_state.theme == "dark" else 1,
        label_visibility="collapsed",
    )
    new_theme = "dark" if "Dark" in theme_choice else "light"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    st.markdown("---")

    # Model status
    st.markdown('<div class="sidebar-label">🤖 Model</div>', unsafe_allow_html=True)
    if st.session_state.model_loaded:
        st.markdown("""
        <div class="model-chip">
          <span class="model-dot"></span>
          YOLOv8n-cls · Loaded
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="model-chip" style="border-color:#EF4444;">
          <span style="width:8px;height:8px;border-radius:50%;background:#EF4444;display:inline-block;"></span>
          Not loaded
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Emotions legend
    st.markdown('<div class="sidebar-label">📊 Emotion Classes</div>', unsafe_allow_html=True)
    for name, meta in EMOTIONS.items():
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:10px;padding:5px 0;'>
          <span style='font-size:1.2rem;'>{meta['emoji']}</span>
          <div>
            <div style='font-size:0.82rem;font-weight:500;color:var(--text-primary);'>{name}</div>
            <div style='font-size:0.68rem;color:var(--text-muted);'>{meta['desc']}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem;color:var(--text-muted);line-height:1.6;'>
      Trained on <strong style='color:var(--text-primary)'>RAF-DB</strong> dataset.<br>
      Model: <strong style='color:var(--text-primary)'>YOLOv8n</strong> · Input: 64×64px<br>
      7 basic emotion categories.
    </div>
    """, unsafe_allow_html=True)

# ── Main content ──────────────────────────────────────────────────────────────

# Hero banner
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">🧠 EmotiSense</div>
  <div class="hero-sub">Real-time facial emotion analysis powered by YOLOv8</div>
  <span class="badge">YOLOv8n-cls</span>
  <span class="badge">RAF-DB</span>
  <span class="badge">7 Emotions</span>
  <span class="badge">64×64</span>
</div>
""", unsafe_allow_html=True)

# Load model
if not st.session_state.model_loaded:
    with st.spinner("Loading YOLOv8 model…"):
        try:
            model, model_name = load_model()
            st.session_state.model = model
            st.session_state.model_name = model_name
            st.session_state.model_loaded = True
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            st.stop()
else:
    model = st.session_state.model

# Layout
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown('<div class="section-title">📸 Input Image</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop a face image here",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed",
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_container_width=True, caption="Uploaded image")

        # Image metadata
        st.markdown(f"""
        <div class="glass-card" style="margin-top:16px;">
          <div style='display:flex;gap:24px;flex-wrap:wrap;'>
            <div>
              <div style='font-size:0.7rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:1px;'>Size</div>
              <div style='font-size:1rem;font-weight:600;color:var(--text-primary);'>{img.width}×{img.height}</div>
            </div>
            <div>
              <div style='font-size:0.7rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:1px;'>Format</div>
              <div style='font-size:1rem;font-weight:600;color:var(--text-primary);'>{uploaded.type.split("/")[1].upper()}</div>
            </div>
            <div>
              <div style='font-size:0.7rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:1px;'>Mode</div>
              <div style='font-size:1rem;font-weight:600;color:var(--text-primary);'>{img.mode}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("⚡ Analyse Emotion", use_container_width=True)
    else:
        st.markdown("""
        <div class="upload-zone">
          <span class="upload-icon">🖼️</span>
          <div style='font-size:1rem;font-weight:600;color:var(--text-primary);margin-bottom:8px;'>
            Drop your image here
          </div>
          <div class="upload-text">Supports JPG, PNG, WEBP, BMP</div>
          <div class="upload-text" style='margin-top:8px;'>Best results with front-facing, well-lit portraits</div>
        </div>
        """, unsafe_allow_html=True)
        run_btn = False

with col_result:
    st.markdown('<div class="section-title">🎯 Analysis Result</div>', unsafe_allow_html=True)

    if uploaded and run_btn:
        with st.spinner("Analysing…"):
            t0 = time.time()
            results = model.predict(img, imgsz=64, verbose=False)
            elapsed_ms = (time.time() - t0) * 1000

        probs = results[0].probs
        top_idx = int(probs.top1)
        top_conf = float(probs.top1conf)
        emotion_name = CLASS_NAMES[top_idx]
        emotion_meta = EMOTIONS[emotion_name]

        # Primary result card
        accent_color = emotion_meta["color"]
        st.markdown(f"""
        <div class="result-card" style="border-top: none;">
          <div style='position:absolute;top:0;left:0;right:0;height:4px;
                      background:linear-gradient(90deg,{accent_color},{accent_color}88);'></div>
          <span class="emotion-emoji">{emotion_meta['emoji']}</span>
          <div class="emotion-label" style="color:{accent_color};">{emotion_name}</div>
          <div class="emotion-desc">{emotion_meta['desc']}</div>
          <div style='display:flex;justify-content:center;gap:24px;margin-top:24px;
                      padding-top:20px;border-top:1px solid var(--border);'>
            <div style='text-align:center;'>
              <div style='font-family:Syne,sans-serif;font-size:1.8rem;font-weight:800;
                          color:{accent_color};'>{top_conf*100:.1f}%</div>
              <div style='font-size:0.72rem;color:var(--text-muted);
                          text-transform:uppercase;letter-spacing:1px;'>Confidence</div>
            </div>
            <div style='text-align:center;'>
              <div style='font-family:Syne,sans-serif;font-size:1.8rem;font-weight:800;
                          color:var(--text-primary);'>{elapsed_ms:.0f}ms</div>
              <div style='font-size:0.72rem;color:var(--text-muted);
                          text-transform:uppercase;letter-spacing:1px;'>Inference</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Full probability bars
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 All Probabilities</div>', unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        all_probs = probs.data.cpu().numpy()
        sorted_idx = np.argsort(all_probs)[::-1]

        bars_html = ""
        for i in sorted_idx:
            name = CLASS_NAMES[i]
            prob = float(all_probs[i])
            meta = EMOTIONS[name]
            is_top = (i == top_idx)
            weight = "700" if is_top else "400"
            bars_html += f"""
            <div class="conf-bar-wrap">
              <div class="conf-label">
                <span style='font-weight:{weight};'>{meta['emoji']} {name}</span>
                <span style='font-weight:{weight};color:{meta["color"]};'>{prob*100:.1f}%</span>
              </div>
              <div class="conf-bar-bg">
                <div class="conf-bar-fill"
                     style="width:{prob*100:.1f}%;background:linear-gradient(90deg,{meta['color']},{meta['color']}99);">
                </div>
              </div>
            </div>
            """
        st.markdown(bars_html + "</div>", unsafe_allow_html=True)

    elif not uploaded:
        st.markdown("""
        <div class="result-card" style='min-height:320px;display:flex;
             flex-direction:column;align-items:center;justify-content:center;'>
          <span style='font-size:4rem;opacity:0.3;'>🔮</span>
          <div style='font-family:Syne,sans-serif;font-size:1.2rem;font-weight:700;
                      color:var(--text-muted);margin-top:16px;'>Awaiting Input</div>
          <div style='font-size:0.85rem;color:var(--text-muted);margin-top:8px;'>
            Upload a face image to begin analysis
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-card" style='min-height:320px;display:flex;
             flex-direction:column;align-items:center;justify-content:center;'>
          <span style='font-size:4rem;opacity:0.5;'>👆</span>
          <div style='font-family:Syne,sans-serif;font-size:1.2rem;font-weight:700;
                      color:var(--text-muted);margin-top:16px;'>Ready to Analyse</div>
          <div style='font-size:0.85rem;color:var(--text-muted);margin-top:8px;'>
            Click the button to run inference
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;margin-top:48px;padding:24px 0;
            border-top:1px solid var(--border);'>
  <span style='font-size:0.78rem;color:var(--text-muted);'>
    EmotiSense · YOLOv8n-cls trained on RAF-DB · 7 basic emotions
  </span>
</div>
""", unsafe_allow_html=True)
