# import streamlit as st
# import numpy as np
# from PIL import Image
# from tensorflow.keras.models import load_model

# model = load_model("cifar10_resnet50.keras")

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
#     type=["jpg","jpeg","png"]
# )

# if uploaded_file is not None:

#     image = Image.open(uploaded_file).convert("RGB")

#     st.image(image, caption="Uploaded Image")

#     image = image.resize((32,32))

#     img_array = np.array(image)/255.0

#     img_array = np.expand_dims(img_array, axis=0)

#     prediction = model.predict(img_array)

#     predicted_class = class_names[np.argmax(prediction)]

#     confidence = np.max(prediction)

#     st.success(
#         f"Prediction: {predicted_class}"
#     )

#     st.write(
#         f"Confidence: {confidence:.2%}"
#     )














import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os
import gdown

MODEL_FILE = "cifar10_resnet50.keras"
FILE_ID = "1z97EwgoxXl3JnbPrhQlrV0n0jbO4grwF"

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_FILE):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_FILE, quiet=False)

    return load_model(MODEL_FILE)


with st.spinner("Loading model..."):
    model = get_model()

class_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

st.title("CIFAR-10 Image Classifier")

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image")

    image = image.resize((32, 32))

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2%}")

