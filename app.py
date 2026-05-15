import os
import sys
import cv2
import numpy as np
from flask import Flask, render_template, request

# Force legacy Keras behavior
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
from tensorflow.keras.models import load_model

# Import local utilities
from utils.preprocess import preprocess_clf, preprocess_seg
from utils.severity import calculate_severity

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- INITIALIZE MODELS ---
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

print("\n" + "="*40)
print("  LOADING MODELS (.h5 ONLY)  ")
print("="*40)

try:
    # Detection model
    clf_model_path = os.path.join(MODELS_DIR, "crack_detection_model_legacy.h5")
    clf_model = load_model(clf_model_path)
    print("✅ Detection Model Loaded.")

    # Segmentation model
    seg_model_path = os.path.join(MODELS_DIR, "crack_segmentation_model_legacy.h5")
    seg_model = load_model(seg_model_path)
    print("✅ Segmentation Model Loaded.")

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

    if img is None:
        return "Invalid image"

    # Detection
    clf_input = preprocess_clf(img)
    pred = float(clf_model.predict(clf_input, verbose=0)[0][0])

    if pred > 0.5:

        seg_input = preprocess_seg(img)
        mask = seg_model.predict(seg_input, verbose=0)[0]

        severity, ratio = calculate_severity(mask)

        result = "Crack Detected"

    else:

        severity = "None"
        ratio = 0.0
        result = "No Crack"

    return render_template(
        "index.html",
        result=result,
        confidence=round(pred * 100, 2),
        severity=severity,
        crack_ratio=round(float(ratio), 4),
        image_path=filepath
    )

if __name__ == "__main__":
    print("\n🚀 Starting server at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)