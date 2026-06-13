


# import streamlit as st
# import numpy as np
# import pandas as pd
# from PIL import Image
# from tensorflow.keras.models import load_model
# import os
# import gdown

# # ---------------- PAGE CONFIG ----------------

# st.set_page_config(
#     page_title="CIFAR-10 Classifier",
#     page_icon="🧠",
#     layout="wide"
# )

# # ---------------- CUSTOM CSS ----------------

# st.markdown("""
# <style>

# .title {
#     font-size: 3rem;
#     font-weight: 700;
#     text-align: center;
#     background: linear-gradient(90deg,#00DBDE,#FC00FF);
#     -webkit-background-clip: text;
#     -webkit-text-fill-color: transparent;
# }

# .subtitle {
#     text-align: center;
#     color: gray;
#     font-size: 1.1rem;
#     margin-bottom: 30px;
# }

# .block-container {
#     padding-top: 2rem;
# }

# </style>
# """, unsafe_allow_html=True)

# # ---------------- HEADER ----------------

# # st.markdown(
# #     '<p class="title">🧠 CIFAR-10 Image Classifier</p>',
# #     unsafe_allow_html=True
# # )

# st.markdown(
#     """
#     <h1 style='text-align:center; color:#4F46E5;'>
#         🧠 CIFAR-10 Image Classifier
#     </h1>
#     """,
#     unsafe_allow_html=True
# )

# # st.markdown(
# #     '<p class="subtitle">Powered by ResNet50 and TensorFlow</p>',
# #     unsafe_allow_html=True
# # )

# st.markdown(
#     """
#     <p style='text-align:center; color:gray; font-size:1.1rem; margin-bottom:30px;'>
#         Powered by ResNet50 and TensorFlow
#     </p>
#     """,
#     unsafe_allow_html=True
# )

# # ---------------- SIDEBAR ----------------

# with st.sidebar:
#     st.header("📌 Project Information")

#     st.write("""
#     **Model:** ResNet50
    
#     **Dataset:** CIFAR-10
    
#     **Classes:**
#     - ✈️ Airplane
#     - 🚗 Automobile
#     - 🐦 Bird
#     - 🐱 Cat
#     - 🦌 Deer
#     - 🐶 Dog
#     - 🐸 Frog
#     - 🐴 Horse
#     - 🚢 Ship
#     - 🚚 Truck
#     """)

#     st.success("Model Ready")

# # ---------------- MODEL ----------------

# MODEL_FILE = "cifar10_resnet50.keras"
# FILE_ID = "1z97EwgoxXl3JnbPrhQlrV0n0jbO4grwF"

# @st.cache_resource
# def get_model():

#     if not os.path.exists(MODEL_FILE):
#         url = f"https://drive.google.com/uc?id={FILE_ID}"
#         gdown.download(url, MODEL_FILE, quiet=False)

#     return load_model(MODEL_FILE)

# with st.spinner("Loading ResNet50 Model..."):
#     model = get_model()

# # ---------------- CLASSES ----------------

# class_names = [
#     "airplane",
#     "automobile",
#     "bird",
#     "cat",
#     "deer",
#     "dog",
#     "frog",
#     "horse",
#     "ship",
#     "truck"
# ]

# # ---------------- FILE UPLOAD ----------------

# uploaded_file = st.file_uploader(
#     "📤 Upload an Image",
#     type=["jpg", "jpeg", "png"]
# )

# # ---------------- PREDICTION ----------------

# if uploaded_file is not None:

#     image = Image.open(uploaded_file).convert("RGB")

#     resized_image = image.resize((32, 32))

#     img_array = np.array(resized_image) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     with st.spinner("Predicting..."):
#         prediction = model.predict(img_array, verbose=0)

#     predicted_index = np.argmax(prediction)
#     predicted_class = class_names[predicted_index]
#     confidence = float(np.max(prediction))

#     col1, col2 = st.columns([1, 1])

#     # IMAGE COLUMN
#     with col1:
#         st.subheader("🖼 Uploaded Image")
#         st.image(
#             image,
#             caption="Input Image",
#             use_container_width=True
#         )

#     # RESULT COLUMN
#     with col2:
#         st.subheader("🎯 Prediction Result")

#         st.success(
#             f"Predicted Class: {predicted_class.upper()}"
#         )

#         st.metric(
#             label="Confidence",
#             value=f"{confidence:.2%}"
#         )

#         st.progress(confidence)

#         st.write("### 🏆 Top 3 Predictions")

#         top3 = np.argsort(prediction[0])[-3:][::-1]

#         for idx in top3:
#             st.write(
#                 f"**{class_names[idx].capitalize()}** : "
#                 f"{prediction[0][idx]*100:.2f}%"
#             )

#     # CHART

#     st.markdown("---")

#     st.subheader("📊 Probability Distribution")

#     df = pd.DataFrame({
#         "Class": class_names,
#         "Probability": prediction[0]
#     })

#     st.bar_chart(
#         df.set_index("Class")
#     )

# # ---------------- FOOTER ----------------

# st.markdown("---")

# # st.caption(
# #     "Built with Streamlit • TensorFlow • ResNet50 • CIFAR-10"
# # )
# st.markdown(
#     """
#     <p style='text-align:center; color:gray; font-size:0.9rem;'>
#         Built with Streamlit • TensorFlow • ResNet50 • CIFAR-10
#     </p>
#     """,
#     unsafe_allow_html=True
# )


























import os
import gdown
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
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


# ---------------- UI STYLE ----------------
st.markdown("""
<style>

/* ===== GLOBAL THEME ===== */
:root {
    --bg1:#050816;
    --bg2:#0b1020;
    --glass:rgba(255,255,255,0.06);
    --glass2:rgba(255,255,255,0.10);
    --border:rgba(255,255,255,0.12);
    --text:#e5e7eb;
    --muted:#94a3b8;
    --accent:#38bdf8;
    --accent2:#a78bfa;
}

/* Background */
.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(56,189,248,0.22), transparent 40%),
        radial-gradient(circle at 85% 20%, rgba(167,139,250,0.22), transparent 45%),
        linear-gradient(180deg, var(--bg1), var(--bg2));
    color: var(--text);
}

/* Container */
.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(2,6,23,0.75);
    border-right: 1px solid var(--border);
}

/* HERO */
.hero {
    position: relative;
    padding: 2.6rem;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
    border: 1px solid var(--border);
    box-shadow: 0 60px 140px rgba(0,0,0,0.6);
    overflow: hidden;
    margin-bottom: 1.5rem;
}

.hero::before {
    content:"";
    position:absolute;
    width:420px;
    height:420px;
    background: radial-gradient(circle, rgba(56,189,248,0.25), transparent 70%);
    top:-120px;
    right:-120px;
}

.hero-title {
    font-size: clamp(2.4rem, 5vw, 4.5rem);
    font-weight: 900;
    letter-spacing: -0.04em;
}

.hero-copy {
    color: var(--muted);
    font-size: 1.05rem;
    max-width: 800px;
}

/* CARD */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.2rem;
    backdrop-filter: blur(14px);
}

/* RESULT */
.result-card {
    border-radius: 20px;
    padding: 1.8rem;
    background: linear-gradient(135deg, rgba(56,189,248,0.18), rgba(167,139,250,0.12));
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 40px 100px rgba(0,0,0,0.55);
}

.result-class {
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(90deg,#fff,#7dd3fc,#a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* TOP CARDS */
.top-card {
    padding: 1rem;
    border-radius: 14px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    transition: 0.2s;
}
.top-card:hover {
    transform: translateY(-3px);
    border-color: rgba(56,189,248,0.3);
}

/* FILE UPLOAD */
div[data-testid="stFileUploader"] {
    border-radius: 14px;
    border: 1px dashed rgba(148,163,184,0.4);
    padding: 1rem;
    background: rgba(255,255,255,0.03);
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
    st.write("CIFAR-10 ResNet50 Classifier")

    st.divider()
    st.subheader("Classes")
    st.write(" ".join(f"{CLASS_ICONS[c]} {c.title()}" for c in CLASS_NAMES))

    st.divider()
    st.success("Model Loaded")


# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <div style="color:#38bdf8; font-size:0.8rem; font-weight:700;">AI IMAGE CLASSIFICATION</div>
    <h1 class="hero-title">CIFAR-10 Vision Lab</h1>
    <p class="hero-copy">
        Upload an image and get real-time predictions with confidence scores and visual analytics.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])


# ---------------- INFERENCE ----------------
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    resized = image.resize((32,32))

    x = np.array(resized)/255.0
    x = np.expand_dims(x,0)

    with st.spinner("Running AI inference..."):
        preds = model.predict(x, verbose=0)[0]

    pred_idx = np.argmax(preds)
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(preds[pred_idx])
    top3 = np.argsort(preds)[-3:][::-1]


    # ---------------- LAYOUT ----------------
    col1, col2 = st.columns([1,1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="result-card">
            <div style="color:#94a3b8;">Predicted Class</div>
            <div class="result-class">{CLASS_ICONS[pred_class]} {pred_class}</div>
            <div style="color:#bae6fd; font-weight:700;">
                Confidence: {confidence:.2%}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(confidence)

        st.subheader("Top Predictions")

        for i in top3:
            st.markdown(f"""
            <div class="top-card">
                <b>{CLASS_ICONS[CLASS_NAMES[i]]} {CLASS_NAMES[i].title()}</b><br>
                <span style="color:#38bdf8; font-weight:700;">
                    {preds[i]:.2%}
                </span>
            </div>
            """, unsafe_allow_html=True)


    # ---------------- PLOTLY ----------------
    df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Probability": preds
    })

    fig = px.bar(
        df.sort_values("Probability", ascending=True),
        x="Probability",
        y="Class",
        orientation="h",
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e5e7eb",
        margin=dict(l=10,r=10,t=20,b=10)
    )

    st.plotly_chart(fig, use_container_width=True)


else:
    st.info("Upload an image to start prediction 🚀")