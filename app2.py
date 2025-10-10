import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown  # <-- 1. Import gdown

# ======================
# CONFIG
# ======================
WEIGHTS_FILE = "efficientnetb5.weights.h5"
# Extract the file ID from your Google Drive URL
FILE_ID = "1wYoROoBNIbhtMiMAcJoZw7axolEJshBU" 
CLASS_LABELS = ["all_benign", "all_early", "all_pre", "all_pro"]
IMAGE_SIZE = (456, 456)

# ======================
# DOWNLOAD MODEL IF NOT PRESENT
# ======================
# This block replaces the tf.keras.utils.get_file call
if not os.path.exists(WEIGHTS_FILE):
    with st.spinner(f"Downloading model weights: {WEIGHTS_FILE}... This may take a moment."):
        gdown.download(id=FILE_ID, output=WEIGHTS_FILE, quiet=False)

# The path is now the local filename
weights_path = WEIGHTS_FILE
# ======================


# ======================
# BUILD MODEL ARCHITECTURE
# ======================
@st.cache_resource(show_spinner="Loading classification model...")
def load_model():
    base_model = tf.keras.applications.EfficientNetB5(
        include_top=False,
        weights=None,  # We load weights from our local file, not pre-trained ones
        input_shape=IMAGE_SIZE + (3,)
    )
    base_model.trainable = False

    inputs = tf.keras.layers.Input(shape=IMAGE_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_LABELS), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    # Load weights from the now-guaranteed-to-be-correct local path
    model.load_weights(weights_path)
    return model

model = load_model()

# ======================
# PREPROCESS IMAGE
# ======================
def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img).astype("float32")
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ======================
# STREAMLIT UI
# ======================
st.title("Cancer Type Classifier (ALL)")

uploaded_file = st.file_uploader("Upload a test image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    img_array = preprocess_image(Image.open(uploaded_file))

    with st.spinner("Predicting..."):
        pred = model.predict(img_array)
        class_index = np.argmax(pred, axis=1)[0]
        confidence = np.max(pred) * 100

    st.success(f"**Predicted Class:** {CLASS_LABELS[class_index]}")
    st.info(f"**Confidence:** {confidence:.2f}%")
