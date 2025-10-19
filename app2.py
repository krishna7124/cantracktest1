import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import os
import gdown

# ==============================================================================
# 1. HELPER FUNCTION TO BUILD THE AUTOENCODER ARCHITECTURE
# ==============================================================================
# This function is required to build the anomaly detector's structure before loading its weights.
def build_autoencoder(input_shape):
    """Builds the autoencoder model architecture."""
    inputs = layers.Input(shape=input_shape)
    # --- Encoder ---
    x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)
    # --- Bottleneck ---
    x = layers.Flatten()(x)
    latent = layers.Dense(256, name='latent_vector')(x)
    # --- Decoder ---
    img_h, img_w, _ = input_shape
    x = layers.Dense((img_h // 16) * (img_w // 16) * 256, activation='relu')(latent)
    x = layers.Reshape((img_h // 16, img_w // 16, 256))(x)
    x = layers.Conv2DTranspose(256, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    outputs = layers.Conv2D(3, 3, activation='sigmoid', padding='same')(x)
    
    model = models.Model(inputs, outputs)
    return model

# ==============================================================================
# 2. MODELS CONFIGURATION
# ==============================================================================
MODELS_CONFIG = {
    "ALL (EfficientNetB5)": {
        "model_type": "classifier",
        "model_builder": tf.keras.applications.EfficientNetB5,
        "weights_file": "efficientnetb5_all.weights.h5",
        "file_id": "1wYoROoBNIbhtMiMAcJoZw7axolEJshBU",
        "class_labels": ["all_benign", "all_early", "all_pre", "all_pro"],
        "image_size": (456, 456)
    },
    "Brain Cancer (EfficientNetB4)": {
        "model_type": "classifier",
        "model_builder": tf.keras.applications.EfficientNetB4,
        "weights_file": "efficientnetb4_brain.weights.h5",
        "file_id": "1dkuKptJnse_FSM9zsJrpTRVNYjMmKscV",
        "class_labels": ['brain_glioma', 'brain_menin', 'brain_tumor'],
        "image_size": (380, 380)
    },
    "Breast Cancer (EfficientNetB4)": {
        "model_type": "classifier",
        "model_builder": tf.keras.applications.EfficientNetB4,
        "weights_file": "efficientnetb4_breast.weights.h5",
        "file_id": "1yFYAotWScH7utmt-4U-KYY3uP6lmgfY3",
        "class_labels": ['breast_benign', 'breast_malignant'],
        "image_size": (380, 380)
    },
    "Cervical Cancer (EfficientNetB4)": {
        "model_type": "classifier",
        "model_builder": tf.keras.applications.EfficientNetB4,
        "weights_file": "efficientnetb4_cervical.weights.h5",
        "file_id": "1g9to0qEO1cVpZGcKsoKo5CbAJ-1sABxv",
        "class_labels": ['cervix_dyk', 'cervix_koc', 'cervix_mep', 'cervix_pab', 'cervix_sfi'],
        "image_size": (380, 380)
    },
    "Kidney Cancer (EfficientNetB4)": {
        "model_type": "classifier",
        "model_builder": tf.keras.applications.EfficientNetB4,
        "weights_file": "efficientnetb4_kidney.weights.h5",
        "file_id": "1uZw8-Y0YnP07gqJTU2o-ftH9rEXWEu6G",
        "class_labels": ['kidney_normal', 'kidney_tumor'],
        "image_size": (380, 380)
    },
    "Liver Cancer (CT Scan)": {
        "model_type": "classifier",
        "model_builder": tf.keras.applications.EfficientNetB4,
        "weights_file": "efficientnetb4_liver_ct.weights.h5",
        "file_id": "1EVSbuyPc4gfa4Hdk577BNFtMu8AzPgPP",
        "class_labels": ['cancerous', 'non_cancerous'],
        "image_size": (380, 380)
    },
    "Lung & Colon Cancer (EfficientNetB4)": {
        "model_type": "classifier",
        "model_builder": tf.keras.applications.EfficientNetB4,
        "weights_file": "efficientnetb4_lung_colon.weights.h5",
        "file_id": "1zG3mCvEYi84WdyBmU-Wn7HxsZkkqJqxG",
        "class_labels": ['colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc'],
        "image_size": (380, 380)
    },
    "Lymphoma (EfficientNetB4)": {
        "model_type": "classifier",
        "model_builder": tf.keras.applications.EfficientNetB4,
        "weights_file": "efficientnetb4_lymphoma.weights.h5",
        "file_id": "12YqVtuJrTqEPh_JCfaigbe3GYExlLoOO",
        "class_labels": ['cll', 'fl', 'mcl'],
        "image_size": (380, 380)
    },
    "Oral Cancer (EfficientNetB4)": {
        "model_type": "classifier",
        "model_builder": tf.keras.applications.EfficientNetB4,
        "weights_file": "efficientnetb4_oral.weights.h5",
        "file_id": "1S_GMcKpUrVTv-4V6VlYHPU9lxxCQ6deW",
        "class_labels": ['oral_cancer', 'oral_normal'],
        "image_size": (380, 380)
    },
    # --- 10th MODEL: ANOMALY DETECTOR ---
    "Bone Cancer (Anomaly Detector)": {
        "model_type": "anomaly",
        "model_builder": build_autoencoder,
        "weights_file": "blood_cancer.weights.h5",
        "file_id": "1nY9v7DTNEDG_-sr4mb2aqMjj2k3iestB",
        "image_size": (256, 256),
        "threshold": 0.007659
    }
}
CONFIDENCE_THRESHOLD = 50.0

# ==============================================================================
# 3. INITIAL SETUP & UI
# ==============================================================================
st.set_page_config(layout="wide")
st.title("🔬 Multi-Modal Cancer Analysis Platform")

@st.cache_resource(show_spinner="Performing initial setup: Downloading required models...")
def download_all_models():
    """Downloads any missing model files specified in the config."""
    for model_name, config in MODELS_CONFIG.items():
        if config.get("file_id"):
            weights_file = config.get("weights_file")
            # --- FIX 1: Message now only shows when a download is needed ---
            if weights_file and not os.path.exists(weights_file):
                st.info(f"Downloading weights for: {model_name}...")
                try:
                    gdown.download(id=config["file_id"], output=weights_file, quiet=False)
                except Exception as e:
                    st.error(f"Could not download weights for {model_name}. Error: {e}")

download_all_models()

st.sidebar.title("⚙️ Controls")
# --- FIX 2: Removed sorted() to maintain the dictionary order in the UI ---
model_keys = list(MODELS_CONFIG.keys())
selected_model_name = st.sidebar.radio("Choose the analysis model:", model_keys)

config = MODELS_CONFIG[selected_model_name]
IMAGE_SIZE = config["image_size"]

# ==============================================================================
# 4. HELPER FUNCTIONS
# ==============================================================================

@st.cache_resource(show_spinner="Loading selected model...")
def load_model(model_name):
    """Builds a model based on its config and loads its weights."""
    cfg = MODELS_CONFIG[model_name]
    
    weights_file = cfg.get("weights_file")
    if not weights_file or not os.path.exists(weights_file):
        st.error(f"Weights file not found: {weights_file}. Please ensure it's in the app's directory.")
        return None

    if cfg["model_type"] == "anomaly":
        model = cfg["model_builder"](cfg["image_size"] + (3,))
        model.load_weights(weights_file)
        return model
        
    elif cfg["model_type"] == "classifier":
        base_model = cfg["model_builder"](include_top=False, weights=None, input_shape=cfg["image_size"] + (3,))
        inputs = layers.Input(shape=cfg["image_size"] + (3,))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(len(cfg["class_labels"]), activation="softmax")(x)
        model = models.Model(inputs, outputs)
        model.load_weights(weights_file)
        return model

def preprocess_for_classifier(img: Image.Image, image_size: tuple):
    """Preprocesses an image for EfficientNet classifiers."""
    img = img.convert("RGB").resize(image_size)
    img_array = np.array(img).astype("float32")
    return tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(img_array, axis=0))

def preprocess_for_anomaly(img: Image.Image, image_size: tuple):
    """Preprocesses an image for the autoencoder anomaly detector."""
    img = img.convert("RGB").resize(image_size)
    img_array = np.array(img).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)

# ==============================================================================
# 5. MAIN APPLICATION LOGIC
# ==============================================================================
model = load_model(selected_model_name)
if model is None:
    st.stop()

st.header(f"Analysis using: **{selected_model_name}**")

# --- ANOMALY DETECTOR LOGIC ---
if config["model_type"] == "anomaly":
    st.info("This model was trained only on **cancerous** images. It flags images that look **different** from its training data as anomalies.")
    uploaded_file = st.file_uploader("Upload a bone scan image", type=["jpg", "jpeg", "png"], key="anomaly_uploader")

    if uploaded_file:
        original_image = Image.open(uploaded_file).convert("RGB")
        img_array = preprocess_for_anomaly(original_image, IMAGE_SIZE)

        with st.spinner("Analyzing for anomalies..."):
            reconstructed_array = model.predict(img_array)
            error = np.mean(np.square(img_array - reconstructed_array))
            threshold = config["threshold"]

            st.subheader("Analysis Results")
            col1, col2 = st.columns(2)
            col1.image(original_image, caption="Original Image", use_container_width=True)
            reconstructed_img = (reconstructed_array.squeeze() * 255).astype(np.uint8)
            col2.image(reconstructed_img, caption="Model's Reconstruction", use_container_width=True)
            
            st.markdown("---")
            st.metric(label="Reconstruction Error", value=f"{error:.6f}")
            
            if error > threshold:
                st.error(f"### Verdict: Non-Cancerous (Anomaly Detected)")
                st.write(f"The reconstruction error **({error:.4f})** is **above** the threshold of **{threshold:.4f}**. This indicates the image is significantly different from the cancerous examples the model was trained on.")
            else:
                st.success(f"### Verdict: Cancerous (Normal)")
                st.write(f"The reconstruction error **({error:.4f})** is **below** the threshold of **{threshold:.4f}**. This indicates the image's features are consistent with the cancerous examples the model was trained on.")

# --- CLASSIFIER LOGIC ---
else:
    CLASS_LABELS = config["class_labels"]
    tab1, tab2 = st.tabs(["Single Image Analysis", "Batch Image Analysis"])

    with tab1:
        st.header("Analyze a Single Image")
        uploaded_file = st.file_uploader("Upload an image for analysis", type=["jpg", "jpeg", "png"], key=f"single_{selected_model_name}")
        if uploaded_file:
            original_image = Image.open(uploaded_file).convert("RGB")
            st.image(original_image, caption="Uploaded Image", use_container_width=True)
            with st.spinner("Classifying..."):
                img_array = preprocess_for_classifier(original_image, IMAGE_SIZE)
                pred = model.predict(img_array)
                confidence = np.max(pred) * 100
                class_index = np.argmax(pred[0])
                st.success(f"**Predicted Class:** `{CLASS_LABELS[class_index]}`")
                st.info(f"**Confidence:** `{confidence:.2f}%`")
                if confidence < CONFIDENCE_THRESHOLD:
                    st.warning("⚠️ **Low Confidence:** The result may be inaccurate.")

    with tab2:
        st.header("Analyze Multiple Images")
        uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key=f"batch_{selected_model_name}")
        if uploaded_files:
            num_columns = st.slider("Number of columns for batch display:", 2, 6, 4)
            cols = st.columns(num_columns)
            for i, file in enumerate(uploaded_files):
                with cols[i % num_columns]:
                    original_image = Image.open(file).convert("RGB")
                    img_array = preprocess_for_classifier(original_image, IMAGE_SIZE)
                    pred = model.predict(img_array)
                    confidence = np.max(pred) * 100
                    class_index = np.argmax(pred[0])
                    caption = f"{CLASS_LABELS[class_index]} ({confidence:.1f}%)"
                    st.image(original_image, caption=caption, use_container_width=True)
