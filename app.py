


import os

import gdown
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model


st.set_page_config(
    page_title="CIFAR-10 Vision Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_FILE = "cifar10_resnet50.keras"
FILE_ID = "1z97EwgoxXl3JnbPrhQlrV0n0jbO4grwF"

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CLASS_ICONS = {
    "airplane": "✈️",
    "automobile": "🚗",
    "bird": "🐦",
    "cat": "🐱",
    "deer": "🦌",
    "dog": "🐶",
    "frog": "🐸",
    "horse": "🐴",
    "ship": "🚢",
    "truck": "🚚",
}


st.markdown(
    """
    <style>
    :root {
        --surface: rgba(15, 23, 42, 0.78);
        --surface-strong: rgba(15, 23, 42, 0.94);
        --border: rgba(148, 163, 184, 0.24);
        --text-muted: #94a3b8;
        --accent: #38bdf8;
        --accent-2: #a78bfa;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(56, 189, 248, 0.20), transparent 28rem),
            radial-gradient(circle at top right, rgba(167, 139, 250, 0.18), transparent 24rem),
            linear-gradient(135deg, #020617 0%, #111827 48%, #0f172a 100%);
        color: #e5e7eb;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: rgba(2, 6, 23, 0.78);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    .hero {
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(30, 41, 59, 0.66));
        box-shadow: 0 24px 80px rgba(2, 6, 23, 0.34);
        margin-bottom: 1.5rem;
    }

    .eyebrow {
        color: var(--accent);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }

    .hero-title {
        color: #f8fafc;
        font-size: clamp(2.2rem, 5vw, 4.1rem);
        font-weight: 800;
        line-height: 1;
        margin: 0;
    }

    .hero-copy {
        color: #cbd5e1;
        font-size: 1.06rem;
        line-height: 1.7;
        max-width: 760px;
        margin-top: 1rem;
        margin-bottom: 0;
    }

    .panel {
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.25rem;
        background: var(--surface);
        box-shadow: 0 18px 42px rgba(2, 6, 23, 0.22);
        height: 100%;
    }

    .panel-title {
        color: #f8fafc;
        font-size: 1.05rem;
        font-weight: 750;
        margin-bottom: 0.9rem;
    }

    .result-card {
        border: 1px solid rgba(56, 189, 248, 0.34);
        border-radius: 8px;
        padding: 1.35rem;
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.16), rgba(167, 139, 250, 0.11));
        margin-bottom: 1rem;
    }

    .result-label {
        color: var(--text-muted);
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }

    .result-class {
        color: #f8fafc;
        font-size: clamp(2rem, 4vw, 3.3rem);
        font-weight: 850;
        line-height: 1;
        margin-bottom: 0.45rem;
    }

    .confidence {
        color: #bae6fd;
        font-size: 1.05rem;
        font-weight: 700;
    }

    .top-row {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin-top: 1rem;
    }

    .top-card {
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.9rem;
        background: rgba(2, 6, 23, 0.35);
    }

    .top-name {
        color: #f8fafc;
        font-weight: 750;
        margin-bottom: 0.2rem;
        overflow-wrap: anywhere;
    }

    .top-score {
        color: var(--accent);
        font-size: 1.2rem;
        font-weight: 800;
    }

    div[data-testid="stFileUploader"] {
        border: 1px dashed rgba(148, 163, 184, 0.42);
        border-radius: 8px;
        padding: 1rem;
        background: rgba(15, 23, 42, 0.58);
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
    }

    @media (max-width: 760px) {
        .hero {
            padding: 1.35rem;
        }

        .top-row {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_model():
    if not os.path.exists(MODEL_FILE):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_FILE, quiet=False)

    return load_model(MODEL_FILE)


with st.sidebar:
    st.title("Vision Lab")
    st.caption("ResNet50 classifier trained for CIFAR-10 image categories.")

    st.divider()
    st.subheader("Model")
    st.write("Architecture: **ResNet50**")
    st.write("Input size: **32 x 32 RGB**")
    st.write("Dataset: **CIFAR-10**")

    st.divider()
    st.subheader("Classes")
    st.write(" ".join(f"{CLASS_ICONS[name]} {name.title()}" for name in CLASS_NAMES))

    st.divider()
    st.success("Model ready")


st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">CIFAR-10 Image Recognition</div>
        <h1 class="hero-title">Vision Lab</h1>
        <p class="hero-copy">
            Upload an image and get a fast ResNet50 prediction with confidence scores,
            top alternatives, and a full class probability breakdown.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.spinner("Loading ResNet50 model..."):
    model = get_model()

upload_col, info_col = st.columns([1.15, 0.85], gap="large")

with upload_col:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Upload Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a JPG, JPEG, or PNG image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded_file is None:
            st.info("Upload an image to run classification.")

with info_col:
    with st.container(border=True):
        st.markdown('<div class="panel-title">How It Works</div>', unsafe_allow_html=True)
        st.write("Images are resized to the CIFAR-10 input shape before prediction.")
        st.write("The confidence score is the model's highest class probability.")
        st.write("Use clear, centered images for the most reliable results.")


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    resized_image = image.resize((32, 32))
    img_array = np.array(resized_image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing image..."):
        prediction = model.predict(img_array, verbose=0)

    probabilities = prediction[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])
    top3 = np.argsort(probabilities)[-3:][::-1]

    st.markdown("<br>", unsafe_allow_html=True)
    image_col, result_col = st.columns([0.95, 1.05], gap="large")

    with image_col:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Uploaded Image</div>', unsafe_allow_html=True)
            st.image(image, caption="Input image", use_container_width=True)

    with result_col:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Predicted Class</div>
                    <div class="result-class">{CLASS_ICONS[predicted_class]} {predicted_class.title()}</div>
                    <div class="confidence">{confidence:.2%} confidence</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(confidence)

            st.markdown('<div class="top-row">', unsafe_allow_html=True)
            for rank, idx in enumerate(top3, start=1):
                name = CLASS_NAMES[int(idx)]
                score = float(probabilities[int(idx)])
                st.markdown(
                    f"""
                    <div class="top-card">
                        <div class="result-label">Top {rank}</div>
                        <div class="top-name">{CLASS_ICONS[name]} {name.title()}</div>
                        <div class="top-score">{score:.2%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

    chart_df = (
        pd.DataFrame({"Class": CLASS_NAMES, "Probability": probabilities})
        .sort_values("Probability", ascending=True)
        .set_index("Class")
    )

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="panel-title">Probability Distribution</div>', unsafe_allow_html=True)
        st.bar_chart(chart_df, use_container_width=True)


st.markdown(
    """
    <p style="text-align:center; color:#94a3b8; font-size:0.9rem; margin-top:2rem;">
        Built with Streamlit, TensorFlow, ResNet50, and CIFAR-10
    </p>
    """,
    unsafe_allow_html=True,
)
