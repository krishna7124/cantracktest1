import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown
import cv2  # Used for Grad-CAM visualization

# ==============================================================================
# 1. MODELS CONFIGURATION
# ==============================================================================
# Central dictionary to hold all information about your models.
MODELS_CONFIG = {
    "ALL (EfficientNetB5)": {
        "model_builder": tf.keras.applications.EfficientNetB5,
        "weights_file": "efficientnetb5_all.weights.h5",
        "file_id": "1wYoROoBNIbht_MiMAcJoZw7axolEJshBU",
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
    }
}
CONFIDENCE_THRESHOLD = 50.0

# ==============================================================================
# 2. INITIAL SETUP & UI
# ==============================================================================
st.set_page_config(layout="wide")
st.title("🔬 Multi-Cancer Type Image Classifier")

# --- NEW: Auto-download all models on startup ---
@st.cache_resource(show_spinner="Performing initial setup: Downloading all model weights...")
def download_all_models():
    """
    Iterates through the model config and downloads any missing weights files.
    This function is cached and runs only once per session.
    """
    for model_name, config in MODELS_CONFIG.items():
        weights_file = config["weights_file"]
        if not os.path.exists(weights_file):
            print(f"Downloading weights for: {model_name}") # Log for debugging
            gdown.download(id=config["file_id"], output=weights_file, quiet=False)

# Run the download function on app start.
download_all_models()

# --- Sidebar for Model Selection ---
st.sidebar.title("⚙️ Controls")
selected_model_name = st.sidebar.radio(
    "Choose the classification model:",
    list(MODELS_CONFIG.keys())
)

# Get configuration for the selected model
config = MODELS_CONFIG[selected_model_name]
CLASS_LABELS = config["class_labels"]
IMAGE_SIZE = config["image_size"]

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================

@st.cache_resource(show_spinner="Loading classification model...")
def load_model(model_name):
    """Loads a model into memory. Cached to prevent reloading when switching tabs."""
    cfg = MODELS_CONFIG[model_name]
    base_model = cfg["model_builder"](include_top=False, weights=None, input_shape=cfg["image_size"] + (3,))
    base_model.trainable = False
    inputs = tf.keras.layers.Input(shape=cfg["image_size"] + (3,))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(len(cfg["class_labels"]), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    # Weights are loaded from the locally downloaded file
    model.load_weights(cfg["weights_file"])
    return model

def preprocess_image(img: Image.Image, image_size: tuple):
    img = img.convert("RGB").resize(image_size)
    img_array = np.array(img).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

def get_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def superimpose_gradcam(img, heatmap, alpha=0.6):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img

# ==============================================================================
# 4. MAIN APPLICATION LOGIC
# ==============================================================================

# Load the selected model
model = load_model(selected_model_name)

# Tabs for Single vs. Batch Upload
tab1, tab2 = st.tabs(["Single Image Analysis", "Batch Image Analysis"])

# --- TAB 1: SINGLE IMAGE UPLOAD ---
with tab1:
    st.header("Analyze a Single Image")
    uploaded_file = st.file_uploader("Upload an image for detailed analysis", type=["jpg", "jpeg", "png"], key="single_uploader")

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        original_image = Image.open(uploaded_file).convert("RGB")
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(original_image, use_column_width=True)
        
        with st.spinner("Analyzing..."):
            img_array = preprocess_image(original_image, IMAGE_SIZE)
            pred = model.predict(img_array)
            class_index = np.argmax(pred[0])
            confidence = np.max(pred) * 100
            
            with col2:
                st.subheader("Analysis Result")
                st.success(f"**Predicted Class:** `{CLASS_LABELS[class_index]}`")
                st.info(f"**Confidence:** `{confidence:.2f}%`")
                
                if confidence < CONFIDENCE_THRESHOLD:
                    st.warning("⚠️ **Low Confidence:** The result may be inaccurate.")

                st.subheader("Model Attention (Grad-CAM)")
                last_conv_layer_name = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)][-1]
                heatmap = get_gradcam_heatmap(img_array, model, last_conv_layer_name)
                img_for_gradcam = cv2.resize(np.array(original_image), IMAGE_SIZE)
                gradcam_image = superimpose_gradcam(img_for_gradcam, heatmap)
                gradcam_image_rgb = cv2.cvtColor(gradcam_image, cv2.COLOR_BGR2RGB)
                st.image(gradcam_image_rgb, caption="Heatmap shows where the model is 'looking'.", use_column_width=True)

# --- TAB 2: BATCH IMAGE UPLOAD ---
with tab2:
    st.header("Analyze Multiple Images in a Batch")
    uploaded_files = st.file_uploader("Upload multiple images for classification", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="batch_uploader")

    if uploaded_files:
        with st.spinner(f"Processing {len(uploaded_files)} images..."):
            for uploaded_file in uploaded_files:
                col1, col2 = st.columns([1, 2])
                original_image = Image.open(uploaded_file).convert("RGB")

                with col1:
                    st.image(original_image, use_column_width=True)

                img_array = preprocess_image(original_image, IMAGE_SIZE)
                pred = model.predict(img_array)
                class_index = np.argmax(pred[0])
                confidence = np.max(pred) * 100
                
                with col2:
                    st.write(f"**File:** `{uploaded_file.name}`")
                    st.success(f"**Predicted Class:** `{CLASS_LABELS[class_index]}`")
                    st.info(f"**Confidence:** `{confidence:.2f}%`")
                    if confidence < CONFIDENCE_THRESHOLD:
                        st.warning("⚠️ **Low Confidence**")
                
                st.divider()
