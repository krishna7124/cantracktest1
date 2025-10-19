import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import os
import gdown
import cv2 # <-- ADDED IMPORT FOR GRAD-CAM

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
        "weights_file": "bone_cancer.weights.h5",
        "file_id": "17eWk6R8eDrLIEouwHiX-3_LZKNkPYoMb",
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
            if weights_file and not os.path.exists(weights_file):
                st.info(f"Downloading weights for: {model_name}...")
                try:
                    gdown.download(id=config["file_id"], output=weights_file, quiet=False)
                except Exception as e:
                    st.error(f"Could not download weights for {model_name}. Error: {e}")

download_all_models()

st.sidebar.title("⚙️ Controls")
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
# 4.1. GRAD-CAM HELPER FUNCTIONS (NEW SECTION)
# ==============================================================================
def find_last_conv_layer(model):
    """Finds the name of the last convolutional layer in the model."""
    for layer in reversed(model.layers):
        # Check for Conv2D layer in the base model of the classifier
        if isinstance(layer, models.Model) and layer.name.startswith("efficientnet"):
             for sub_layer in reversed(layer.layers):
                 if isinstance(sub_layer, layers.Conv2D):
                     return sub_layer.name
    raise ValueError("No Conv2D layer found in the model's EfficientNet base.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generates the Grad-CAM heatmap."""
    # Create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(original_img, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """Overlays a heatmap onto the original image."""
    original_img_np = np.array(original_img.convert("RGB"))
    heatmap_resized = cv2.resize(heatmap, (original_img_np.shape[1], original_img_np.shape[0]))
    
    heatmap_colored = (cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap), cv2.COLOR_BGR2RGB))
    
    superimposed_img = (heatmap_colored * (1 - alpha) + original_img_np * alpha).astype(np.uint8)
    return Image.fromarray(superimposed_img)

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
        
        # --- MODIFIED LOGIC FOR GRAD-CAM ---
        if uploaded_file:
            original_image = Image.open(uploaded_file).convert("RGB")
            
            with st.spinner("Classifying and generating heatmap..."):
                img_array = preprocess_for_classifier(original_image, IMAGE_SIZE)
                
                pred = model.predict(img_array)
                confidence = np.max(pred) * 100
                class_index = np.argmax(pred[0])
                
                try:
                    # Find the last conv layer in the base model
                    base_model = next(l for l in model.layers if l.name.startswith('efficientnet'))
                    last_conv_layer_name = next(l.name for l in reversed(base_model.layers) if isinstance(l, layers.Conv2D))

                    # Generate and overlay heatmap
                    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
                    grad_cam_image = overlay_heatmap(original_image, heatmap)

                    # Display side-by-side
                    col1, col2 = st.columns(2)
                    col1.image(original_image, caption="Original Uploaded Image", use_container_width=True)
                    col2.image(grad_cam_image, caption="Grad-CAM: AI Focus Heatmap", use_container_width=True)

                except Exception as e:
                    st.error(f"Could not generate Grad-CAM heatmap. Displaying original image only. Error: {e}")
                    st.image(original_image, caption="Uploaded Image", use_container_width=True)

                st.markdown("---")
                st.success(f"**Predicted Class:** `{CLASS_LABELS[class_index]}`")
                st.info(f"**Confidence:** `{confidence:.2f}%`")

                if confidence < CONFIDENCE_THRESHOLD:
                    st.warning("⚠️ **Low Confidence:** The result may be inaccurate.")
        # --- END OF MODIFIED LOGIC ---

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
