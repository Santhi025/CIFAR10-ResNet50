
# import streamlit as st
# import numpy as np
# from PIL import Image
# from tensorflow.keras.models import load_model
# import os
# import gdown

# MODEL_FILE = "cifar10_resnet50.keras"
# FILE_ID = "1z97EwgoxXl3JnbPrhQlrV0n0jbO4grwF"

# @st.cache_resource
# def get_model():
#     if not os.path.exists(MODEL_FILE):
#         url = f"https://drive.google.com/uc?id={FILE_ID}"
#         gdown.download(url, MODEL_FILE, quiet=False)

#     return load_model(MODEL_FILE)


# with st.spinner("Loading model..."):
#     model = get_model()

# class_names = [
#     'airplane',
#     'automobile',
#     'bird',
#     'cat',
#     'deer',
#     'dog',
#     'frog',
#     'horse',
#     'ship',
#     'truck'
# ]

# st.title("CIFAR-10 Image Classifier")

# uploaded_file = st.file_uploader(
#     "Upload an Image",
#     type=["jpg", "jpeg", "png"]
# )

# if uploaded_file is not None:

#     image = Image.open(uploaded_file).convert("RGB")

#     st.image(image, caption="Uploaded Image")

#     image = image.resize((32, 32))

#     img_array = np.array(image) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     prediction = model.predict(img_array)

#     predicted_class = class_names[np.argmax(prediction)]
#     confidence = np.max(prediction)

#     st.success(f"Prediction: {predicted_class}")
#     st.write(f"Confidence: {confidence:.2%}")


import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
import os
import gdown

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="CIFAR-10 Classifier",
    page_icon="🧠",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.title {
    font-size: 3rem;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(90deg,#00DBDE,#FC00FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 1.1rem;
    margin-bottom: 30px;
}

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

# st.markdown(
#     '<p class="title">🧠 CIFAR-10 Image Classifier</p>',
#     unsafe_allow_html=True
# )

st.markdown(
    """
    <h1 style='text-align:center; color:#4F46E5;'>
        🧠 CIFAR-10 Image Classifier
    </h1>
    """,
    unsafe_allow_html=True
)

# st.markdown(
#     '<p class="subtitle">Powered by ResNet50 and TensorFlow</p>',
#     unsafe_allow_html=True
# )

st.markdown(
    """
    <p style='text-align:center; color:gray; font-size:1.1rem; margin-bottom:30px;'>
        Powered by ResNet50 and TensorFlow
    </p>
    """,
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.header("📌 Project Information")

    st.write("""
    **Model:** ResNet50
    
    **Dataset:** CIFAR-10
    
    **Classes:**
    - ✈️ Airplane
    - 🚗 Automobile
    - 🐦 Bird
    - 🐱 Cat
    - 🦌 Deer
    - 🐶 Dog
    - 🐸 Frog
    - 🐴 Horse
    - 🚢 Ship
    - 🚚 Truck
    """)

    st.success("Model Ready")

# ---------------- MODEL ----------------

MODEL_FILE = "cifar10_resnet50.keras"
FILE_ID = "1z97EwgoxXl3JnbPrhQlrV0n0jbO4grwF"

@st.cache_resource
def get_model():

    if not os.path.exists(MODEL_FILE):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_FILE, quiet=False)

    return load_model(MODEL_FILE)

with st.spinner("Loading ResNet50 Model..."):
    model = get_model()

# ---------------- CLASSES ----------------

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# ---------------- FILE UPLOAD ----------------

uploaded_file = st.file_uploader(
    "📤 Upload an Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PREDICTION ----------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    resized_image = image.resize((32, 32))

    img_array = np.array(resized_image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Predicting..."):
        prediction = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction))

    col1, col2 = st.columns([1, 1])

    # IMAGE COLUMN
    with col1:
        st.subheader("🖼 Uploaded Image")
        st.image(
            image,
            caption="Input Image",
            use_container_width=True
        )

    # RESULT COLUMN
    with col2:
        st.subheader("🎯 Prediction Result")

        st.success(
            f"Predicted Class: {predicted_class.upper()}"
        )

        st.metric(
            label="Confidence",
            value=f"{confidence:.2%}"
        )

        st.progress(confidence)

        st.write("### 🏆 Top 3 Predictions")

        top3 = np.argsort(prediction[0])[-3:][::-1]

        for idx in top3:
            st.write(
                f"**{class_names[idx].capitalize()}** : "
                f"{prediction[0][idx]*100:.2f}%"
            )

    # CHART

    st.markdown("---")

    st.subheader("📊 Probability Distribution")

    df = pd.DataFrame({
        "Class": class_names,
        "Probability": prediction[0]
    })

    st.bar_chart(
        df.set_index("Class")
    )

# ---------------- FOOTER ----------------

st.markdown("---")

# st.caption(
#     "Built with Streamlit • TensorFlow • ResNet50 • CIFAR-10"
# )
st.markdown(
    """
    <p style='text-align:center; color:gray; font-size:0.9rem;'>
        Built with Streamlit • TensorFlow • ResNet50 • CIFAR-10
    </p>
    """,
    unsafe_allow_html=True
)