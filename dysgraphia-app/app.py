"""
=======================================================================
  app.py — Flask Web Application Backend
  Dysgraphia Detection Project | APM.02 | Master 1 STIC | 2025-2026
  Instructor: Dr. NECIBI Khaled | University of Constantine 2
=======================================================================

Routes:
  GET  /              → Homepage (Predict page)
  POST /predict       → Upload image + get prediction (CNN or ML)
  GET  /api/metrics   → JSON: ML model metrics (SVM, RF, KNN)
  GET  /api/cnn       → JSON: CNN metrics + training history
  GET  /api/dashboard → JSON: Combined comparison of all 4 models
  POST /api/retrain   → Retrain all ML models
  POST /api/retrain-cnn → Retrain CNN model
"""

import os
import json
import traceback
import numpy as np
import cv2
import joblib
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# ──────────────────────────────────────────────────────────────
# Flask app configuration
# ──────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"))

app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "uploads")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB limit
app.secret_key = os.environ.get("SESSION_SECRET", "dysgraphia-2025")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif", "webp"}
MODELS_DIR  = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Model cache (loaded once, reused for every request)
# ──────────────────────────────────────────────────────────────

_ml_bundle  = None   # {"model": ..., "scaler": ..., "best_name": ...}
_cnn_model  = None   # Keras model


def load_ml_models():
    """Load the best classical ML model + scaler. Train if not found."""
    global _ml_bundle
    if _ml_bundle is not None:
        return _ml_bundle

    model_path = os.path.join(MODELS_DIR, "best_ml_model.pkl")
    if not os.path.exists(model_path):
        print("  Modèles ML non trouvés → lancement de l'entraînement...")
        from train_ml import train
        train()

    _ml_bundle = joblib.load(model_path)
    print(f"  ML model chargé: {_ml_bundle['best_name']}")
    return _ml_bundle


def load_cnn_model():
    """Load the CNN model. Train if not found."""
    global _cnn_model
    if _cnn_model is not None:
        return _cnn_model

    model_path = os.path.join(MODELS_DIR, "cnn_model.h5")
    if not os.path.exists(model_path):
        print("  CNN non trouvé → lancement de l'entraînement CNN...")
        from train_cnn import train as train_cnn
        train_cnn()

    # Import TensorFlow only when needed (saves startup time)
    import tensorflow as tf
    _cnn_model = tf.keras.models.load_model(model_path)
    print("  CNN chargé avec succès.")
    return _cnn_model


# ── Pre-load ML models at startup (faster first prediction)
try:
    load_ml_models()
except Exception as e:
    print(f"  Avertissement startup ML: {e}")


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def allowed_file(filename):
    """Check that the uploaded file has an accepted extension."""
    return ("." in filename and
            filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS)


def preprocess_for_cnn(filepath, img_size=64):
    """
    Preprocess an image for CNN prediction.
    Returns: numpy array, shape (1, img_size, img_size, 1), float32 in [0,1]
    """
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError("Impossible de lire l'image.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (img_size, img_size))
    gray = gray.astype(np.float32) / 255.0
    return gray[np.newaxis, ..., np.newaxis]   # shape: (1, 64, 64, 1)


def preprocess_for_ml(filepath):
    """
    Extract the 12 handwriting features from an image for ML prediction.
    Returns: numpy array, shape (1, 12)
    """
    from train_ml import extract_features
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError("Impossible de lire l'image.")
    features = extract_features(img)
    if features is None:
        raise ValueError("Extraction de features échouée.")
    return features.reshape(1, -1)


FEATURE_NAMES = [
    "Mean Intensity", "Std Intensity", "Dark Pixel Ratio",
    "Edge Density", "Contour Count", "Mean Contour Area",
    "Mean Contour Perimeter", "Horizontal Variance",
    "Vertical Variance", "Laplacian Variance",
    "Pixel Skewness", "Ink Irregularity",
]

# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main SPA (single-page application)."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept an uploaded handwriting image.
    Run prediction with the selected model (CNN or classical ML).
    Returns JSON with label, confidence, probabilities, and features.
    """
    if "image" not in request.files:
        return jsonify({"error": "Aucune image fournie."}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "Aucun fichier sélectionné."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Type de fichier non supporté (PNG, JPG, BMP)."}), 400

    # Which model to use? Default = CNN if available, else ML
    model_choice = request.form.get("model", "cnn").lower()

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # ── CNN Prediction
        if model_choice == "cnn":
            cnn = load_cnn_model()
            X = preprocess_for_cnn(filepath)
            prob_dys = float(cnn.predict(X, verbose=0)[0][0])
            prob_normal = 1.0 - prob_dys
            prediction = 1 if prob_dys > 0.5 else 0
            model_used = "CNN"
            confidence = max(prob_dys, prob_normal) * 100

        # ── Classical ML Prediction (SVM / RF / KNN)
        else:
            bundle = load_ml_models()
            model  = bundle["model"]
            scaler = bundle["scaler"]
            model_used = bundle["best_name"]

            features = preprocess_for_ml(filepath)
            features_sc = scaler.transform(features)
            prediction  = int(model.predict(features_sc)[0])
            probs = model.predict_proba(features_sc)[0]
            prob_normal = float(probs[0])
            prob_dys    = float(probs[1])
            confidence  = float(max(probs)) * 100

        label = "Dysgraphia Détectée" if prediction == 1 else "Écriture Normale"

        # ── Extract features for display (always shown)
        raw_features = preprocess_for_ml(filepath).flatten()
        feature_summary = [
            {"name": n, "value": round(float(v), 4)}
            for n, v in zip(FEATURE_NAMES, raw_features)
        ]

        return jsonify({
            "prediction":   int(prediction),
            "label":        label,
            "confidence":   round(confidence, 2),
            "model_used":   model_used,
            "probabilities": {
                "normal":     round(prob_normal * 100, 2),
                "dysgraphia": round(prob_dys    * 100, 2),
            },
            "features": feature_summary,
        })

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/metrics")
def api_metrics():
    """Return ML model metrics (SVM, RF, KNN) as JSON."""
    path = os.path.join(REPORTS_DIR, "ml_metrics.json")
    if not os.path.exists(path):
        try:
            from train_ml import train
            train()
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/api/cnn")
def api_cnn():
    """Return CNN metrics + training history as JSON."""
    path = os.path.join(REPORTS_DIR, "cnn_metrics.json")
    if not os.path.exists(path):
        return jsonify({"error": "CNN non encore entraîné. "
                        "Lancez /api/retrain-cnn d'abord."}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/api/dashboard")
def api_dashboard():
    """
    Return a unified comparison of ALL models (CNN + SVM + RF + KNN).
    Used by the Dashboard page to render charts and tables.
    """
    all_models = []

    # Load ML metrics
    ml_path = os.path.join(REPORTS_DIR, "ml_metrics.json")
    if os.path.exists(ml_path):
        with open(ml_path) as f:
            ml_data = json.load(f)
        all_models.extend(ml_data.get("models", []))
        dataset_info = {
            "size":       ml_data.get("dataset_size", "—"),
            "train":      ml_data.get("train_size", "—"),
            "test":       ml_data.get("test_size", "—"),
            "features":   len(ml_data.get("feature_names", [])),
            "augmentation": ml_data.get("augmentation_factor", 1),
        }
    else:
        dataset_info = {}

    # Load CNN metrics
    cnn_path = os.path.join(REPORTS_DIR, "cnn_metrics.json")
    cnn_history = None
    if os.path.exists(cnn_path):
        with open(cnn_path) as f:
            cnn_data = json.load(f)
        cnn_history = cnn_data.pop("training_history", None)
        # Insert CNN at the front of the list
        all_models.insert(0, {
            "name":             cnn_data.get("name", "CNN"),
            "accuracy":         cnn_data.get("accuracy", 0),
            "precision":        cnn_data.get("precision", 0),
            "recall":           cnn_data.get("recall", 0),
            "f1_score":         cnn_data.get("f1_score", 0),
            "confusion_matrix": cnn_data.get("confusion_matrix", [[0,0],[0,0]]),
        })

    if not all_models:
        return jsonify({"error": "Aucun modèle entraîné. "
                        "Lancez train_ml.py et train_cnn.py d'abord."}), 404

    # Find best model overall
    best = max(all_models, key=lambda m: m["f1_score"])

    return jsonify({
        "models":        all_models,
        "best_model":    best["name"],
        "dataset":       dataset_info,
        "cnn_history":   cnn_history,
        "cnn_available": os.path.exists(cnn_path),
    })


@app.route("/api/retrain", methods=["POST"])
def retrain_ml():
    """Force re-training of classical ML models (SVM, RF, KNN)."""
    global _ml_bundle
    _ml_bundle = None
    # Remove old model to force full retrain
    old = os.path.join(MODELS_DIR, "best_ml_model.pkl")
    if os.path.exists(old):
        os.remove(old)
    try:
        from train_ml import train
        metrics = train()
        _ml_bundle = joblib.load(os.path.join(MODELS_DIR, "best_ml_model.pkl"))
        return jsonify({"success": True, "best_model": metrics["best_model"]})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/retrain-cnn", methods=["POST"])
def retrain_cnn():
    """Force re-training of the CNN model."""
    global _cnn_model
    _cnn_model = None
    old = os.path.join(MODELS_DIR, "cnn_model.h5")
    if os.path.exists(old):
        os.remove(old)
    try:
        from train_cnn import train as train_cnn
        metrics = train_cnn()
        return jsonify({"success": True,
                        "accuracy": metrics["accuracy"],
                        "f1_score": metrics["f1_score"]})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  DysDetect Flask App — port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
