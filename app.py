import os
import sys
import cv2
import numpy as np
from flask import Flask, render_template, request

# Force legacy Keras behavior
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
from tensorflow.keras import layers, models

# Import local utilities
from utils.preprocess import preprocess_clf, preprocess_seg
from utils.severity import calculate_severity

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- MODEL ARCHITECTURE REBUILDERS ---

def build_detection_model():
    # Use the specific configuration that matches Colab's default MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(160, 160, 3), 
        include_top=False, 
        weights=None
    )
    x = layers.GlobalAveragePooling2D()(base_model.output)
    output = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inputs=base_model.input, outputs=output)

def build_segmentation_model():
    inputs = layers.Input((128, 128, 3))
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)
    b = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    b = layers.Conv2D(128, 3, activation='relu', padding='same')(b)
    u1 = layers.UpSampling2D()(b)
    u1 = layers.Concatenate()([u1, c2])
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(u1)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(c3)
    u2 = layers.UpSampling2D()(c3)
    u2 = layers.Concatenate()([u2, c1])
    c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(u2)
    c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(c4)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c4)
    return models.Model(inputs, outputs)

def safe_set_weights(model, weights_list):
    """Injects weights up to the maximum possible count to avoid length mismatches"""
    model_weights = model.get_weights()
    new_weights = []
    # Only take as many weights as the local model can hold
    for i in range(len(model_weights)):
        if i < len(weights_list):
            # Check if shapes match before injecting
            if model_weights[i].shape == weights_list[i].shape:
                new_weights.append(weights_list[i])
            else:
                print(f"⚠️ Skipping weight {i} due to shape mismatch.")
                new_weights.append(model_weights[i])
        else:
            new_weights.append(model_weights[i])
    model.set_weights(new_weights)

# --- INITIALIZE AND LOAD ---
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

print("\n" + "="*40)
print("  SELECTIVE WEIGHT INJECTION (NUMPY)  ")
print("="*40)

try:
    # 1. Detection Model
    clf_model = build_detection_model()
    clf_npy_path = os.path.join(MODELS_DIR, "clf_weights_simple.npy")
    clf_weights_list = np.load(clf_npy_path, allow_pickle=True)
    safe_set_weights(clf_model, clf_weights_list)
    print("✅ Detection Model Injected (Selective).")

    # 2. Segmentation Model
    seg_model = build_segmentation_model()
    seg_npy_path = os.path.join(MODELS_DIR, "seg_weights_simple.npy")
    seg_weights_list = np.load(seg_npy_path, allow_pickle=True)
    safe_set_weights(seg_model, seg_weights_list)
    print("✅ Segmentation Model Injected (Selective).")

except Exception as e:
    print(f"\n❌ Error: {str(e)}")
    sys.exit(1)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No image uploaded"
    file = request.files["image"]
    if file.filename == "":
        return "No selected file"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)
    img = cv2.imread(filepath)
    if img is None: return "Invalid image"

    # Inference
    clf_input = preprocess_clf(img)
    pred = float(clf_model.predict(clf_input, verbose=0)[0][0])

    if pred > 0.5:
        seg_input = preprocess_seg(img)
        mask = seg_model.predict(seg_input, verbose=0)[0]
        severity, ratio = calculate_severity(mask)
        result = "Crack Detected"
    else:
        severity, ratio, result = "None", 0.0, "No Crack"

    return render_template(
        "index.html", result=result, confidence=round(pred * 100, 2),
        severity=severity, crack_ratio=round(float(ratio), 4), image_path=filepath
    )

if __name__ == "__main__":
    print("\n🚀 Starting server at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
