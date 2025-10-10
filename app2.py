import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown
import cv2 # Used for Grad-CAM visualization

# ==============================================================================
# 1. MODELS CONFIGURATION
# ==============================================================================
# Central dictionary to hold all information about your models.
# This makes it easy to add or change models in the future.

MODELS_CONFIG = {
    "ALL (EfficientNetB5)": {
        "model_builder": tf.keras.applications.EfficientNetB5,
        "weights_file": "efficientnetb5_all.weights.h5",
        "file_id": "1wYoROoBNIbhtMiMAcJoZw7axolEJshBU", # <-- Replace with your file ID
        "class_labels": ["all_benign", "all_early", "all_pre", "all_pro"],
        "image_size": (456, 456)
    },
    "Brain Cancer (EfficientNetB4)": {
        "model_builder": tf.keras.applications.EfficientNetB4,
        "weights_file": "efficientnetb4_brain.weights.h5",
        "file_id": "1dkuKptJnse_FSM9zsJrpTRVNYjMmKscV", # <-- Replace with your file ID
        "class_labels": ['brain_glioma', 'brain_menin', 'brain_tumor'],
        "image_size": (380, 380)
    },
    "Breast Cancer (EfficientNetB4)": {
        "model_builder": tf.keras.applications.EfficientNetB4,
        "weights_file": "efficientnetb4_breast.weights.h5",
        "file_id": "1yFYAotWScH7utmt-4U-KYY3uP6lmgfY3", # <-- Replace with your file ID
        "class_labels": ['breast_benign', 'breast_malignant'],
        "image_size": (380, 380)
    },
    "Cervical Cancer (EfficientNetB4)": {
        "model_builder": tf.keras.applications.EfficientNetB4,
        "weights_file": "efficientnetb4_cervical.weights.h5",
        "file_id": "1g9to0qEO1cVpZGcKsoKo5CbAJ-1sABxv", # <-- Replace with your file ID
        "class_labels": ['cervix_dyk', 'cervix_koc', 'cervix_mep', 'cervix_pab', 'cervix_sfi'],
        "image_size": (380, 380)
    }
}
CONFIDENCE_THRESHOLD = 50.0 # Confidence below this will trigger a warning

# ==============================================================================
# 2. STREAMLIT UI - MODEL SELECTION
# ==============================================================================
st.set_page_config(layout="wide")
st.title("🔬 Multi-Cancer Type Image Classifier")
st.write("Select a model and upload an image to get a prediction and a Grad-CAM visualization.")

# Create a dropdown menu for model selection
selected_model_name = st.selectbox(
    "Choose the classification model:",
    list(MODELS_CONFIG.keys())
)

# Get the configuration for the selected model
config = MODELS_CONFIG[selected_model_name]

# Dynamically assign variables based on selected model
WEIGHTS_FILE = config["weights_file"]
FILE_ID = config["file_id"]
CLASS_LABELS = config["class_labels"]
IMAGE_SIZE = config["image_size"]
model_builder = config["model_builder"]

# ==============================================================================
# 3. HELPER FUNCTIONS (MODEL LOADING, PREPROCESSING, GRAD-CAM)
# ==============================================================================

# Download model weights from Google Drive if they don't exist
if not os.path.exists(WEIGHTS_FILE):
    if FILE_ID == "ADD_YOUR_GOOGLE_DRIVE_ID_HERE":
        st.error(f"Please update the Google Drive FILE_ID for the '{selected_model_name}' model in the script.")
    else:
        with st.spinner(f"Downloading model weights: {WEIGHTS_FILE}... This may take a moment."):
            gdown.download(id=FILE_ID, output=WEIGHTS_FILE, quiet=False)

# Cache the model loading to prevent reloading on every action
@st.cache_resource(show_spinner="Loading classification model...")
def load_model(model_name): # Pass model_name to make cache unique per model
    """Builds and loads a model based on the selected configuration."""
    cfg = MODELS_CONFIG[model_name]
    
    base_model = cfg["model_builder"](
        include_top=False,
        weights=None,
        input_shape=cfg["image_size"] + (3,)
    )
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
    """Prepares the uploaded image for the model."""
    img = img.convert("RGB")
    img = img.resize(image_size)
    img_array = np.array(img).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

def get_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generates the Grad-CAM heatmap."""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def superimpose_gradcam(img, heatmap, alpha=0.6):
    """Superimposes the heatmap on the original image."""
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

# File uploader
uploaded_file = st.file_uploader("Upload a test image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display columns for side-by-side view
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        original_image = Image.open(uploaded_file).convert("RGB")
        st.image(original_image, use_column_width=True)
        img_array = preprocess_image(original_image, IMAGE_SIZE)

    with st.spinner("Analyzing image..."):
        # Make prediction
        pred = model.predict(img_array)
        class_index = np.argmax(pred, axis=1)[0]
        confidence = np.max(pred) * 100

        # --- Invalid Image Check ---
        if confidence < CONFIDENCE_THRESHOLD:
            st.warning(f"⚠️ **Low Confidence Warning:** The model is only {confidence:.2f}% confident. "
                       "The uploaded image may not be suitable for this model.")
        
        # Display prediction
        st.success(f"**Predicted Class:** `{CLASS_LABELS[class_index]}`")
        st.info(f"**Confidence:** `{confidence:.2f}%`")
        
        # --- Grad-CAM Visualization ---
        st.subheader("Grad-CAM Visualization")
        
        # Find the last convolutional layer name automatically
        last_conv_layer_name = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)][-1]

        # Generate and display heatmap
        heatmap = get_gradcam_heatmap(img_array, model, last_conv_layer_name)
        
        # Prepare original image for overlay
        img_for_gradcam = cv2.resize(np.array(original_image), IMAGE_SIZE)
        gradcam_image = superimpose_gradcam(img_for_gradcam, heatmap)
        gradcam_image_rgb = cv2.cvtColor(gradcam_image, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
        
        with col2:
            st.subheader("Model Attention Heatmap")
            st.image(gradcam_image_rgb, caption="Heatmap shows where the model is 'looking'.", use_column_width=True)
