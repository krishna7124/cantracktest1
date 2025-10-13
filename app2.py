import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown
# import cv2  # --- GRAD-CAM REMOVED ---

# ==============================================================================
# 1. MODELS CONFIGURATION
# ==============================================================================
# Central dictionary to hold all information about your models.
MODELS_CONFIG = {
    "ALL (EfficientNetB5)": {
        "model_builder": tf.keras.applications.EfficientNetB5,
        "weights_file": "efficientnetb5_all.weights.h5",
        "file_id": "1wYoROoBNIbhtMiMAcJoZw7axolEJshBU",
        "class_labels": ["all_benign", "all_early", "all_pre", "all_pro"],
        "image_size": (456, 456)
    },
    "Brain Cancer (EfficientNetB4)": {
        "model_builder": tf.keras.applications.EfficientNetB4,
        "weights_file": "efficientnetb4_brain.weights.h5",
        "file_id": "1dkuKptJnse_FSM9zsJrpTRVNYjMmKscV",
        "class_labels": ['brain_glioma', 'brain_menin', 'brain_tumor'],
        "image_size": (380, 380)
    },
    "Breast Cancer (EfficientNetB4)": {
        "model_builder": tf.keras.applications.EfficientNetB4,
        "weights_file": "efficientnetb4_breast.weights.h5",
        "file_id": "1yFYAotWScH7utmt-4U-KYY3uP6lmgfY3",
        "class_labels": ['breast_benign', 'breast_malignant'],
        "image_size": (380, 380)
    },
    "Cervical Cancer (EfficientNetB4)": {
        "model_builder": tf.keras.applications.EfficientNetB4,
        "weights_file": "efficientnetb4_cervical.weights.h5",
        "file_id": "1g9to0qEO1cVpZGcKsoKo5CbAJ-1sABxv",
        "class_labels": ['cervix_dyk', 'cervix_koc', 'cervix_mep', 'cervix_pab', 'cervix_sfi'],
        "image_size": (380, 380)
    },
    "Kidney Cancer (EfficientNetB4)": {
        "model_builder": tf.keras.applications.EfficientNetB4,
        "weights_file": "efficientnetb4_kidney.weights.h5",
        "file_id": "1uZw8-Y0YnP07gqJTU2o-ftH9rEXWEu6G",
        "class_labels": ['kidney_normal', 'kidney_tumor'],
        "image_size": (380, 380)
    },
    "Lung & Colon Cancer (EfficientNetB4)": {
        "model_builder": tf.keras.applications.EfficientNetB4,
        "weights_file": "efficientnetb4_lung_colon.weights.h5",
        "file_id": "1zG3mCvEYi84WdyBmU-Wn7HxsZkkqJqxG",
        "class_labels": ['colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc'],
        "image_size": (380, 380)
    },
    "Lymphoma (EfficientNetB4)": {
        "model_builder": tf.keras.applications.EfficientNetB4,
        "weights_file": "efficientnetb4_lymphoma.weights.h5",
        "file_id": "12YqVtuJrTqEPh_JCfaigbe3GYExlLoOO",
        "class_labels": ['cll', 'fl', 'mcl'],
        "image_size": (380, 380)
    },
    "Oral Cancer (EfficientNetB4)": {
        "model_builder": tf.keras.applications.EfficientNetB4,
        "weights_file": "efficientnetb4_oral.weights.h5",
        "file_id": "1S_GMcKpUrVTv-4V6VlYHPU9lxxCQ6deW",
        "class_labels": ['oral_cancer', 'oral_normal'],
        "image_size": (380, 380)
    },
    # --- 9th MODEL ADDED BELOW ---
    "Liver Cancer (CT Scan)": {
        "model_builder": tf.keras.applications.EfficientNetB4,
        "weights_file": "efficientnetb4_liver_ct.weights.h5",
        "file_id": "1EVSbuyPc4gfa4Hdk577BNFtMu8AzPgPP", # <-- IMPORTANT: ADD THE NEW FILE ID
        "class_labels": ['cancerous', 'non_cancerous'],   # Based on your training script
        "image_size": (380, 380)
    }
}
CONFIDENCE_THRESHOLD = 50.0

# ==============================================================================
# 2. INITIAL SETUP & UI
# ==============================================================================
st.set_page_config(layout="wide")
st.title("🔬 Multi-Cancer Type Image Classifier")

@st.cache_resource(show_spinner="Performing initial setup: Downloading all model weights...")
def download_all_models():
    """
    Iterates through the model config and downloads any missing weights files.
    """
    for model_name, config in MODELS_CONFIG.items():
        weights_file = config["weights_file"]
        if not os.path.exists(weights_file):
            print(f"Downloading weights for: {model_name}")
            try:
                gdown.download(id=config["file_id"], output=weights_file, quiet=False)
            except Exception as e:
                st.error(f"Could not download weights for {model_name}. File ID may be incorrect. Error: {e}")

download_all_models()

st.sidebar.title("⚙️ Controls")
selected_model_name = st.sidebar.radio(
    "Choose the classification model:",
    list(MODELS_CONFIG.keys())
)

config = MODELS_CONFIG[selected_model_name]
CLASS_LABELS = config["class_labels"]
IMAGE_SIZE = config["image_size"]

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================

@st.cache_resource(show_spinner="Loading classification model...")
def load_model(model_name):
    """Loads a model into memory."""
    cfg = MODELS_CONFIG[model_name]
    base_model = cfg["model_builder"](include_top=False, weights=None, input_shape=cfg["image_size"] + (3,))
    base_model.trainable = False
    inputs = tf.keras.layers.Input(shape=cfg["image_size"] + (3,))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(len(cfg["class_labels"]), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.load_weights(cfg["weights_file"])
    return model

def preprocess_image(img: Image.Image, image_size: tuple):
    img = img.convert("RGB").resize(image_size)
    img_array = np.array(img).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

# ==============================================================================
# 4. MAIN APPLICATION LOGIC
# ==============================================================================

model = load_model(selected_model_name)
tab1, tab2 = st.tabs(["Single Image Analysis", "Batch Image Analysis"])

# --- TAB 1: SINGLE IMAGE UPLOAD ---
with tab1:
    st.header("Analyze a Single Image")
    uploaded_file = st.file_uploader(
        "Upload an image for detailed analysis",
        type=["jpg", "jpeg", "png"],
        key=f"single_uploader_{selected_model_name}"
    )

    if uploaded_file is not None:
        original_image = Image.open(uploaded_file).convert("RGB")
        
        st.subheader("Uploaded Image")
        st.image(original_image, use_container_width=True)
        
        with st.spinner("Analyzing..."):
            img_array = preprocess_image(original_image, IMAGE_SIZE)
            pred = model.predict(img_array)
            class_index = np.argmax(pred[0])
            confidence = np.max(pred) * 100
            
            st.subheader("Analysis Result")
            st.success(f"**Predicted Class:** `{CLASS_LABELS[class_index]}`")
            st.info(f"**Confidence:** `{confidence:.2f}%`")
            
            if confidence < CONFIDENCE_THRESHOLD:
                st.warning("⚠️ **Low Confidence:** The result may be inaccurate.")

# --- TAB 2: BATCH IMAGE UPLOAD ---
with tab2:
    st.header("Analyze Multiple Images in a Batch")
    uploaded_files = st.file_uploader(
        "Upload multiple images for classification",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"batch_uploader_{selected_model_name}"
    )

    if uploaded_files:
        with st.spinner(f"Processing {len(uploaded_files)} images..."):
            num_columns = 4
            cols = st.columns(num_columns)
            for i, uploaded_file in enumerate(uploaded_files):
                with cols[i % num_columns]:
                    original_image = Image.open(uploaded_file).convert("RGB")
                    
                    img_array = preprocess_image(original_image, IMAGE_SIZE)
                    pred = model.predict(img_array)
                    class_index = np.argmax(pred[0])
                    confidence = np.max(pred) * 100
                    predicted_class = CLASS_LABELS[class_index]

                    caption_text = f"Prediction: {predicted_class} ({confidence:.1f}%)"
                    st.image(original_image, caption=caption_text, use_container_width=True)
