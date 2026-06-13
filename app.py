import os
import gdown
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model


# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="CIFAR-10 Vision Lab Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_FILE = "cifar10_resnet50.keras"
FILE_ID = "1z97EwgoxXl3JnbPrhQlrV0n0jbO4grwF"

CLASS_NAMES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck",
]

CLASS_ICONS = {
    "airplane":"✈️","automobile":"🚗","bird":"🐦","cat":"🐱",
    "deer":"🦌","dog":"🐶","frog":"🐸","horse":"🐴","ship":"🚢","truck":"🚚",
}


# ---------------- MODERN UI STYLE ----------------
st.markdown("""
<style>

/* ===== GLOBAL ===== */
.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(56,189,248,0.18), transparent 40%),
        radial-gradient(circle at 85% 20%, rgba(167,139,250,0.18), transparent 45%),
        linear-gradient(180deg, #050816, #0b1020);
    color: #e5e7eb;
    font-family: "Inter", sans-serif;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background: rgba(2,6,23,0.85);
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

/* ===== HERO ===== */
.hero {
    padding: 2.8rem;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 60px 140px rgba(0,0,0,0.6);
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}

.hero::before {
    content:"";
    position:absolute;
    width:500px;
    height:500px;
    background: radial-gradient(circle, rgba(56,189,248,0.25), transparent 60%);
    top:-150px;
    right:-150px;
}

.hero-title {
    font-size: clamp(2.6rem, 5vw, 4.6rem);
    font-weight: 900;
    letter-spacing: -0.04em;
    background: linear-gradient(90deg,#fff,#7dd3fc,#a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    color: #94a3b8;
    font-size: 1.1rem;
    max-width: 800px;
    margin-top: 0.8rem;
}

/* ===== CARDS ===== */
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 1.2rem;
    backdrop-filter: blur(12px);
    transition: all 0.25s ease;
}

.card:hover {
    transform: translateY(-3px);
    border-color: rgba(56,189,248,0.25);
}

/* ===== RESULT ===== */
.result-card {
    border-radius: 22px;
    padding: 2rem;
    background: linear-gradient(135deg, rgba(56,189,248,0.20), rgba(167,139,250,0.12));
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 50px 120px rgba(0,0,0,0.55);
}

.result-class {
    font-size: 3.4rem;
    font-weight: 950;
    background: linear-gradient(90deg,#fff,#7dd3fc,#a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ===== UPLOAD ===== */
div[data-testid="stFileUploader"] {
    border-radius: 18px;
    border: 1px dashed rgba(148,163,184,0.35);
    padding: 1.2rem;
    background: rgba(255,255,255,0.02);
    transition: 0.2s ease;
}

div[data-testid="stFileUploader"]:hover {
    border-color: rgba(56,189,248,0.5);
}

/* ===== TOP CARDS ===== */
.top-card {
    padding: 1rem;
    border-radius: 16px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 10px;
    transition: 0.2s;
}

.top-card:hover {
    transform: translateY(-3px);
    border-color: rgba(167,139,250,0.3);
}

/* ===== DIVIDER ===== */
.glow {
    height: 1px;
    background: linear-gradient(90deg, transparent, #38bdf8, #a78bfa, transparent);
    margin: 1.5rem 0;
}

</style>
""", unsafe_allow_html=True)


# ---------------- MODEL ----------------
@st.cache_resource
def load_model_cached():
    if not os.path.exists(MODEL_FILE):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_FILE, quiet=False)
    return load_model(MODEL_FILE)

model = load_model_cached()


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🧠 Vision Lab Pro")
    st.caption("CIFAR-10 AI Image Classifier")

    st.markdown("### Classes")
    st.write(" ".join(f"{CLASS_ICONS[c]} {c.title()}" for c in CLASS_NAMES))

    st.markdown("---")
    st.success("Model Loaded & Ready")


# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <div style="color:#38bdf8; font-weight:700; font-size:0.85rem;">
        AI IMAGE CLASSIFICATION SYSTEM
    </div>
    <div class="hero-title">CIFAR-10 Vision Lab</div>
    <div class="hero-sub">
        Upload an image and get instant AI predictions with confidence scores and top-3 analysis.
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("Drop your image here", type=["jpg","jpeg","png"])


# ---------------- INFERENCE ----------------
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    resized = image.resize((32, 32))

    x = np.array(resized) / 255.0
    x = np.expand_dims(x, 0)

    with st.spinner("AI analyzing image..."):
        preds = model.predict(x, verbose=0)[0]

    idx = np.argmax(preds)
    label = CLASS_NAMES[idx]
    conf = float(preds[idx])
    top3 = np.argsort(preds)[-3:][::-1]


    st.markdown('<div class="glow"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1,1])

    # IMAGE
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # RESULT
    with col2:
        st.markdown(f"""
        <div class="result-card">
            <div style="color:#94a3b8;">Prediction</div>
            <div class="result-class">{CLASS_ICONS[label]} {label}</div>
            <div style="color:#bae6fd; font-weight:700;">
                Confidence: {conf:.2%}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(conf)

        st.markdown("### Top Predictions")

        for i in top3:
            st.markdown(f"""
            <div class="top-card">
                <b>{CLASS_ICONS[CLASS_NAMES[i]]} {CLASS_NAMES[i].title()}</b><br>
                <span style="color:#38bdf8; font-weight:700;">
                    {preds[i]:.2%}
                </span>
            </div>
            """, unsafe_allow_html=True)


else:
    st.info("Upload an image to start AI prediction 🚀")