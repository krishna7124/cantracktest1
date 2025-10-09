import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ======================
# ✅ CONFIG (UPDATE FOR LOCAL PC)
# ======================
# Path to saved weights (update to your local path)
WEIGHTS_PATH = r"F:\Pojects\Test\CanTrack\Test\efficientnetb5.weights.h5"

# Path to test image (update to your local image)
IMAGE_PATH = r"F:\Pojects\_output_\processed-cancer-all-types\ALL\test\all_benign\all_benign_0113.jpg"

CLASS_LABELS = ["all_benign", "all_early", "all_pre", "all_pro"]
IMAGE_SIZE = (456, 456)

# ======================
# ✅ BUILD MODEL ARCHITECTURE (same as training)
# ======================
base_model = tf.keras.applications.EfficientNetB5(
    include_top=False,
    weights=None,  # Do NOT load ImageNet weights here
    input_shape=IMAGE_SIZE + (3,)
)
base_model.trainable = False

inputs = tf.keras.layers.Input(shape=IMAGE_SIZE + (3,))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(CLASS_LABELS), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

# Load the saved weights
model.load_weights(WEIGHTS_PATH)
print("Model weights loaded successfully ✅")

# ======================
# ✅ PREPROCESS IMAGE
# ======================
def load_and_preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img = np.array(img).astype("float32")
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

img_array = load_and_preprocess_image(IMAGE_PATH)
print("Image preprocessed ✅")

# ======================
# ✅ PREDICT
# ======================
pred = model.predict(img_array)
class_index = np.argmax(pred, axis=1)[0]
confidence = np.max(pred) * 100

print(f"\n🎯 Predicted Class: {CLASS_LABELS[class_index]}")
print(f"📊 Confidence: {confidence:.2f}%")
