import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import os
import cv2

# ======================================================================
# 1. HELPER FUNCTION TO BUILD THE AUTOENCODER ARCHITECTURE
# ======================================================================
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

# ======================================================================
# 2. MODELS CONFIGURATION
# ======================================================================
MODELS_CONFIG = {
    "ALL (EfficientNetB5)": {
        "model_type": "classifier",
        "model_builder": tf.keras.applications.EfficientNetB5,
        "weights_file": "efficientnetb5_all.weights.h5",
        "class_labels": ["all_benign", "all_early", "all_pre", "all_pro"],
        "image_size": (456, 456)
    },
    "Brain Cancer (EfficientNetB4)": {
        "model_type": "classifier",
        "model_builder": tf.keras.applications.EfficientNetB4,
        "weights_file": "efficientnetb4_brain.weights.h5",
        "class_labels": ['brain_glioma', 'brain_menin', 'brain_tumor'],
        "image_size": (380, 380)
    },
    "Breast Cancer (EfficientNetB4)": {
        "model_type": "classifier",
        "model_builder": tf.keras.applications.EfficientNetB4,
        "weights_file": "efficientnetb4_breast.weights.h5",
        "class_labels": ['breast_benign', 'breast_malignant'],
        "image_size": (380, 380)
    },
    "Cervical Cancer (EfficientNetB4)": {
        "model_type": "classifier",
        "model_builder": tf.keras.applications.EfficientNetB4,
        "weights_file": "efficientnetb4_cervical.weights.h5",
        "class_labels": ['cervix_dyk', 'cervix_koc', 'cervix_mep', 'cervix_pab', 'cervix_sfi'],
        "image_size": (380, 380)
    },
    "Kidney Cancer (EfficientNetB4)": {
        "model_type": "classifier",
        "model_builder": tf.keras.applications.EfficientNetB4,
        "weights_file": "efficientnetb4_kidney.weights.h5",
        "class_labels": ['kidney_normal', 'kidney_tumor'],
        "image_size": (380, 380)
    },
    "Bone Cancer (Anomaly Detector)": {
        "model_type": "anomaly",
        "model_builder": build_autoencoder,
        "weights_file": "bone_cancer.weights.h5",
        "image_size": (256, 256),
        "threshold": 0.007659
    }
}
CONFIDENCE_THRESHOLD = 50.0

# ======================================================================
# 3. INITIAL SETUP & UI
# ======================================================================
st.set_page_config(layout="wide")
st.title("🔬 Multi-Modal Cancer Analysis Platform")

st.sidebar.title("⚙️ Controls")
model_keys = list(MODELS_CONFIG.keys())
selected_model_name = st.sidebar.radio("Choose the analysis model:", model_keys)

config = MODELS_CONFIG[selected_model_name]
IMAGE_SIZE = config["image_size"]

# ======================================================================
# 4. HELPER FUNCTIONS
# ======================================================================
@st.cache_resource(show_spinner="Loading selected model...")
def load_model(model_name):
    cfg = MODELS_CONFIG[model_name]
    weights_file = cfg.get("weights_file")
    if not weights_file or not os.path.exists(weights_file):
        st.error(f"Weights file not found: {weights_file}. Please add it to the app directory.")
        return None

    if cfg["model_type"] == "anomaly":
        model = cfg["model_builder"](cfg["image_size"] + (3,))
        model.load_weights(weights_file)
        return model

    elif cfg["model_type"] == "classifier":
        base_model = cfg["model_builder"](include_top=False, weights=None, input_shape=cfg["image_size"] + (3,), name="efficientnet_base")
        base_model.trainable = False
        inputs = layers.Input(shape=cfg["image_size"] + (3,))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(len(cfg["class_labels"]), activation="softmax")(x)
        model = models.Model(inputs, outputs)
        model.load_weights(weights_file)
        return model

def preprocess_for_classifier(img: Image.Image, image_size: tuple):
    img = img.convert("RGB").resize(image_size)
    img_array = np.array(img).astype("float32")
    return tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(img_array, axis=0))

def preprocess_for_anomaly(img: Image.Image, image_size: tuple):
    img = img.convert("RGB").resize(image_size)
    img_array = np.array(img).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)

# ======================================================================
# 4.1. GRAD-CAM++ FUNCTIONS (REPLACEMENT)
# ======================================================================
def _find_last_conv_layer(base_model):
    for layer in reversed(base_model.layers):
        if len(layer.output_shape) == 4 and "conv" in layer.name:
            return layer.name
    return None

def get_last_conv_layer_name_from_model(model):
    try:
        base_model = model.get_layer('efficientnet_base')
    except Exception:
        return None
    return _find_last_conv_layer(base_model)

def make_gradcam_plus_plus_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generates a Grad-CAM++ heatmap."""
    base_model = model.get_layer('efficientnet_base')
    grad_model = models.Model(
        [model.inputs], [base_model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Grad-CAM++ calculations
    first_derivative = tf.exp(class_channel)[0] * grads
    second_derivative = first_derivative * grads
    third_derivative = second_derivative * grads
    
    alpha_num = second_derivative
    alpha_denom = 2.0 * second_derivative + tf.reduce_sum(third_derivative, axis=tuple(range(len(third_derivative.shape)-1)), keepdims=True) * last_conv_layer_output
    alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, 1e-10)
    
    alphas = alpha_num / alpha_denom
    
    weights = tf.maximum(first_derivative, 0.0)
    alpha_normalization_constant = tf.reduce_sum(alphas, axis=tuple(range(len(alphas.shape)-1)), keepdims=True)
    alphas /= alpha_normalization_constant
    
    deep_linearization_weights = tf.reduce_sum(weights * alphas, axis=tuple(range(len(weights.shape)-1)), keepdims=True)
    
    # Create heatmap
    grad_cam_map = tf.reduce_sum(deep_linearization_weights * last_conv_layer_output, axis=-1)
    heatmap = tf.squeeze(grad_cam_map)

    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        max_val = 1e-10
    heatmap = heatmap / max_val
    
    return heatmap.numpy()


def overlay_heatmap(original_img_pil, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
    if heatmap is None:
        return original_img_pil
    original_img_np = np.array(original_img_pil.convert("RGB"))
    h_img, w_img = original_img_np.shape[:2]
    heatmap_resized = cv2.resize((heatmap * 255).astype(np.uint8), (w_img, h_img))
    heatmap_colored = cv2.applyColorMap(heatmap_resized, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = (original_img_np.astype(np.float32) * (1 - alpha) + heatmap_colored.astype(np.float32) * alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)

def make_reconstruction_error_overlay(original_img_pil, orig_array, recon_array, alpha=0.5):
    error_map = np.mean((orig_array - recon_array) ** 2, axis=-1)
    err_min, err_max = float(error_map.min()), float(error_map.max())
    norm_err = (error_map - err_min) / (err_max - err_min + 1e-8)
    return overlay_heatmap(original_img_pil, norm_err, alpha=alpha)

# ======================================================================
# 5. MAIN APPLICATION LOGIC
# ======================================================================
model = load_model(selected_model_name)
if model is None:
    st.stop()

st.header(f"Analysis using: **{selected_model_name}**")

if config["model_type"] == "anomaly":
    st.info("This model flags images that look different from cancerous training data.")
    uploaded_file = st.file_uploader("Upload a bone scan image", type=["jpg","jpeg","png"], key="anomaly_uploader")
    if uploaded_file:
        original_image = Image.open(uploaded_file).convert("RGB")
        img_array = preprocess_for_anomaly(original_image, IMAGE_SIZE)

        with st.spinner("Analyzing..."):
            reconstructed_array = model.predict(img_array)
            error = np.mean(np.square(img_array - reconstructed_array))
            threshold = config["threshold"]

            st.subheader("Results")
            col1, col2 = st.columns(2)
            col1.image(original_image, caption="Original Image", use_container_width=True)
            reconstructed_img = (reconstructed_array.squeeze() * 255).astype(np.uint8)
            col2.image(reconstructed_img, caption="Reconstruction", use_container_width=True)

            try:
                recon_overlay = make_reconstruction_error_overlay(original_image, img_array.squeeze(), reconstructed_array.squeeze(), alpha=0.5)
                st.image(recon_overlay, caption="Reconstruction Error Overlay", use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create overlay: {e}")

            st.metric(label="Reconstruction Error", value=f"{error:.6f}")
            if error > threshold:
                st.error(f"### Verdict: Non-Cancerous (Anomaly Detected)")
            else:
                st.success(f"### Verdict: Cancerous (Normal)")

else:  # Classifier
    CLASS_LABELS = config["class_labels"]
    tab1, tab2 = st.tabs(["Single Image Analysis", "Batch Image Analysis"])

    with tab1:
        st.header("Single Image")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"], key=f"single_{selected_model_name}")
        if uploaded_file:
            original_image = Image.open(uploaded_file).convert("RGB")
            img_array = preprocess_for_classifier(original_image, IMAGE_SIZE)
            pred = model.predict(img_array)
            confidence = np.max(pred) * 100.0
            class_index = int(np.argmax(pred[0]))

            st.markdown("---")
            st.success(f"**Predicted Class:** `{CLASS_LABELS[class_index]}`")
            st.info(f"**Confidence:** `{confidence:.2f}%`")
            if confidence < CONFIDENCE_THRESHOLD:
                st.warning("⚠️ Low Confidence: Result may be inaccurate.")

            # Grad-CAM++
            try:
                last_conv_layer_name = get_last_conv_layer_name_from_model(model)
                if last_conv_layer_name:
                    # UPDATED FUNCTION CALL
                    heatmap = make_gradcam_plus_plus_heatmap(img_array, model, last_conv_layer_name, pred_index=class_index)
                    grad_cam_image = overlay_heatmap(original_image, heatmap, alpha=0.45)
                    col1, col2 = st.columns(2)
                    col1.image(original_image, caption="Original Image", use_container_width=True)
                    # UPDATED CAPTION
                    col2.image(grad_cam_image, caption="Grad-CAM++ Heatmap", use_container_width=True)
                else:
                    st.warning("No suitable conv layer for Grad-CAM++ found. Skipping heatmap.")
                    st.image(original_image, caption="Uploaded Image", use_container_width=True)
            except Exception as e:
                st.error(f"Grad-CAM++ error: {e}")
                st.image(original_image, caption="Uploaded Image", use_container_width=True)

    with tab2:
        st.header("Batch Images")
        uploaded_files = st.file_uploader("Upload multiple images", type=["jpg","jpeg","png"], accept_multiple_files=True, key=f"batch_{selected_model_name}")
        if uploaded_files:
            num_columns = st.slider("Columns for display:", 2, 6, 4)
            cols = st.columns(num_columns)
            for i, file in enumerate(uploaded_files):
                with cols[i % num_columns]:
                    original_image = Image.open(file).convert("RGB")
                    img_array = preprocess_for_classifier(original_image, IMAGE_SIZE)
                    pred = model.predict(img_array)
                    confidence = np.max(pred) * 100.0
                    class_index = int(np.argmax(pred[0]))
                    caption = f"{CLASS_LABELS[class_index]} ({confidence:.1f}%)"
                    st.image(original_image, caption=caption, use_container_width=True)
