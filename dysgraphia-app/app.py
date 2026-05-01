"""
Dysgraphia Detection System — Flask Backend
Routes:
  GET  /          → Homepage
  POST /predict   → Upload image and get prediction
  GET  /evaluate  → Model evaluation metrics page
  GET  /api/metrics → Raw metrics JSON
"""

import os
import json
import traceback
import numpy as np
import cv2
import joblib
from flask import (Flask, render_template, request,
                   jsonify, redirect, url_for)
from werkzeug.utils import secure_filename

# ──────────────────────────────────────────────
# App configuration
# ──────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"))

app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "uploads")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB
app.secret_key = os.environ.get("SESSION_SECRET", "dysgraphia-secret-2025")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif", "webp"}
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ──────────────────────────────────────────────
# Model loading / auto-training
# ──────────────────────────────────────────────

_model_bundle = None


def load_model():
    """Load model from disk, training it first if not present."""
    global _model_bundle
    if _model_bundle is not None:
        return _model_bundle

    if not os.path.exists(MODEL_PATH):
        print("  No trained model found — running training pipeline...")
        from train import train
        train()

    _model_bundle = joblib.load(MODEL_PATH)
    print(f"  Model loaded: {_model_bundle['best_name']}")
    return _model_bundle


# Load model at startup (non-blocking; errors surface on first predict)
try:
    load_model()
except Exception as exc:
    print(f"  Warning: model load at startup failed — {exc}")


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def allowed_file(filename):
    return ("." in filename and
            filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS)


def preprocess_and_extract(filepath):
    """Read image, preprocess, and return (gray_resized, features)."""
    from train import extract_features

    img = cv2.imread(filepath)
    if img is None:
        raise ValueError("Could not read the image file.")

    features = extract_features(img_array=img)
    if features is None:
        raise ValueError("Feature extraction failed.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gray_resized = cv2.resize(gray, (64, 64))
    return gray_resized, features


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Accept an uploaded image and return a prediction JSON."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use PNG, JPG, BMP."}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        bundle = load_model()
        model = bundle["model"]
        scaler = bundle["scaler"]
        model_name = bundle["best_name"]

        _, features = preprocess_and_extract(filepath)
        features_scaled = scaler.transform(features.reshape(1, -1))

        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        label = "Dysgraphia Detected" if prediction == 1 else "Normal Handwriting"
        confidence = float(max(probabilities)) * 100

        # Feature summary for UI display
        feat_names = [
            "Mean Intensity", "Std Intensity", "Dark Pixel Ratio",
            "Edge Density", "Contour Count", "Mean Contour Area",
            "Mean Contour Perimeter", "Horizontal Variance",
            "Vertical Variance", "Laplacian Variance",
            "Pixel Skewness", "Ink Irregularity",
        ]
        feature_summary = [
            {"name": n, "value": round(float(v), 4)}
            for n, v in zip(feat_names, features)
        ]

        return jsonify({
            "prediction": int(prediction),
            "label": label,
            "confidence": round(confidence, 2),
            "model_used": model_name,
            "probabilities": {
                "normal": round(float(probabilities[0]) * 100, 2),
                "dysgraphia": round(float(probabilities[1]) * 100, 2),
            },
            "features": feature_summary,
        })

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/evaluate")
def evaluate():
    """Render evaluation results page."""
    return render_template("index.html", page="evaluate")


@app.route("/api/metrics")
def api_metrics():
    """Return raw metrics JSON for the evaluation page."""
    if not os.path.exists(METRICS_PATH):
        # Trigger training if metrics not found
        try:
            from train import train
            train()
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    with open(METRICS_PATH) as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/api/retrain", methods=["POST"])
def retrain():
    """Force re-training of all models."""
    global _model_bundle
    _model_bundle = None
    try:
        # Remove existing model so train() rebuilds everything
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        from train import train
        metrics = train()
        _model_bundle = joblib.load(MODEL_PATH)
        return jsonify({"success": True, "best_model": metrics["best_model"]})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
