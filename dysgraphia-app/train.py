"""
Dysgraphia Detection - Model Training Script
Trains SVM, Random Forest, and KNN classifiers on handwriting image features.
If no real images are found, generates synthetic training data automatically.
"""

import os
import cv2
import numpy as np
import joblib
import json
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")
IMG_SIZE = 64
RANDOM_STATE = 42


# ──────────────────────────────────────────────
# Synthetic data generation
# ──────────────────────────────────────────────

def _draw_horizontal_lines(img, n_lines, y_spread, intensity_noise, spacing_noise):
    """Draw horizontal strokes simulating handwritten text lines."""
    h, w = img.shape
    for i in range(n_lines):
        base_y = int((i + 1) * h / (n_lines + 1))
        y = base_y + int(np.random.uniform(-y_spread, y_spread))
        y = max(4, min(h - 4, y))
        x_start = np.random.randint(2, 8)
        x_end = np.random.randint(w - 8, w - 2)
        thickness = np.random.randint(1, 3)
        points = []
        for x in range(x_start, x_end, 2):
            jitter = int(np.random.uniform(-spacing_noise, spacing_noise))
            points.append((x, y + jitter))
        for j in range(len(points) - 1):
            color = int(np.random.uniform(0, intensity_noise))
            cv2.line(img, points[j], points[j + 1], color, thickness)


def generate_normal_image():
    """Generate a synthetic image resembling normal, consistent handwriting."""
    img = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 255
    n_lines = np.random.randint(4, 7)
    _draw_horizontal_lines(img, n_lines,
                           y_spread=1.5,
                           intensity_noise=30,
                           spacing_noise=1)
    # Add very light noise
    noise = np.random.randint(0, 10, img.shape, dtype=np.uint8)
    img = cv2.subtract(img, noise)
    return img


def generate_dysgraphic_image():
    """Generate a synthetic image with irregular handwriting features."""
    img = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 255
    n_lines = np.random.randint(3, 7)
    _draw_horizontal_lines(img, n_lines,
                           y_spread=6,
                           intensity_noise=80,
                           spacing_noise=5)
    # Add heavy noise and blobs
    noise = np.random.randint(0, 40, img.shape, dtype=np.uint8)
    img = cv2.subtract(img, noise)
    # Random ink blobs
    for _ in range(np.random.randint(2, 6)):
        cx = np.random.randint(5, IMG_SIZE - 5)
        cy = np.random.randint(5, IMG_SIZE - 5)
        r = np.random.randint(1, 4)
        cv2.circle(img, (cx, cy), r, np.random.randint(0, 80), -1)
    return img


def ensure_synthetic_data(n_per_class=150):
    """Create synthetic training images if real data folders are empty."""
    normal_dir = os.path.join(DATA_DIR, "normal")
    dys_dir = os.path.join(DATA_DIR, "dysgraphia")
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(dys_dir, exist_ok=True)

    def count_images(folder):
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return sum(1 for f in os.listdir(folder)
                   if os.path.splitext(f)[1].lower() in exts)

    if count_images(normal_dir) < 10:
        print(f"  Generating {n_per_class} synthetic NORMAL images...")
        for i in range(n_per_class):
            img = generate_normal_image()
            cv2.imwrite(os.path.join(normal_dir, f"normal_{i:04d}.png"), img)

    if count_images(dys_dir) < 10:
        print(f"  Generating {n_per_class} synthetic DYSGRAPHIA images...")
        for i in range(n_per_class):
            img = generate_dysgraphic_image()
            cv2.imwrite(os.path.join(dys_dir, f"dysgraphia_{i:04d}.png"), img)


# ──────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────

def extract_features(image_path=None, img_array=None):
    """
    Extract handwriting-relevant features from an image.
    Accepts either a file path or a numpy array.

    Features:
    1.  Mean pixel intensity
    2.  Std deviation of pixels
    3.  Percentage of dark pixels
    4.  Edge density (Canny)
    5.  Contour count
    6.  Mean contour area
    7.  Mean contour perimeter
    8.  Horizontal line variance (row-wise std)
    9.  Vertical line variance (col-wise std)
    10. Laplacian variance (blurriness measure)
    11. Skewness of pixel intensity distribution
    12. Ink irregularity (local std mean over 8x8 blocks)
    """
    if img_array is not None:
        img = img_array
    else:
        img = cv2.imread(image_path)
        if img is None:
            return None

    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    # Gaussian blur + thresholding (Otsu)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Feature 1-3: Intensity statistics
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    dark_ratio = np.sum(gray < 128) / gray.size

    # Feature 4: Edge density
    edges = cv2.Canny(blurred, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # Feature 5-7: Contour analysis
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)
    if contours:
        areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 1]
        perims = [cv2.arcLength(c, True) for c in contours if cv2.contourArea(c) > 1]
        mean_area = np.mean(areas) if areas else 0
        mean_perim = np.mean(perims) if perims else 0
    else:
        mean_area = 0
        mean_perim = 0

    # Feature 8-9: Line regularity
    row_means = np.mean(gray, axis=1)
    col_means = np.mean(gray, axis=0)
    horiz_variance = np.std(row_means)
    vert_variance = np.std(col_means)

    # Feature 10: Laplacian variance (sharpness)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Feature 11: Pixel skewness (scipy-free)
    flat = gray.flatten().astype(np.float64)
    n = len(flat)
    mu = np.mean(flat)
    sigma = np.std(flat)
    skewness = np.mean(((flat - mu) / (sigma + 1e-8)) ** 3)

    # Feature 12: Ink irregularity (local block std)
    block = 8
    local_stds = []
    for r in range(0, IMG_SIZE - block, block):
        for c in range(0, IMG_SIZE - block, block):
            patch = gray[r:r + block, c:c + block]
            local_stds.append(np.std(patch))
    ink_irregularity = np.mean(local_stds)

    return np.array([
        mean_intensity,
        std_intensity,
        dark_ratio,
        edge_density,
        contour_count,
        mean_area,
        mean_perim,
        horiz_variance,
        vert_variance,
        laplacian_var,
        skewness,
        ink_irregularity,
    ], dtype=np.float64)


# ──────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────

def load_dataset():
    """Load all images and extract features, returning X, y arrays."""
    X, y = [], []
    categories = {
        "normal": 0,
        "dysgraphia": 1,
    }
    for label_name, label_idx in categories.items():
        folder = os.path.join(DATA_DIR, label_name)
        if not os.path.exists(folder):
            continue
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        for fname in sorted(os.listdir(folder)):
            if os.path.splitext(fname)[1].lower() not in exts:
                continue
            path = os.path.join(folder, fname)
            features = extract_features(image_path=path)
            if features is not None:
                X.append(features)
                y.append(label_idx)
    return np.array(X), np.array(y)


# ──────────────────────────────────────────────
# Model training & evaluation
# ──────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, name):
    """Compute classification metrics for a fitted model."""
    y_pred = model.predict(X_test)
    return {
        "name": name,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def train():
    """Main training routine. Returns metrics dict."""
    print("\n=== Dysgraphia Detection — Training Pipeline ===\n")

    # 1. Ensure data exists
    print("[1/5] Checking dataset...")
    ensure_synthetic_data()

    # 2. Load features
    print("[2/5] Extracting features from images...")
    X, y = load_dataset()
    if len(X) == 0:
        raise RuntimeError("No images found in data/ directory.")
    print(f"      Loaded {len(X)} samples | Normal: {sum(y==0)} | Dysgraphia: {sum(y==1)}")

    # 3. Split
    print("[3/5] Splitting dataset (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 4. Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # 5. Train three models
    print("[4/5] Training SVM, Random Forest, and KNN models...")
    models = {
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale",
                   probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    results = []
    fitted = {}
    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        fitted[name] = model
        metrics = evaluate_model(model, X_test_sc, y_test, name)
        results.append(metrics)
        print(f"      {name:<16} Acc: {metrics['accuracy']:.4f}  "
              f"F1: {metrics['f1_score']:.4f}")

    # 6. Select best model (by F1)
    best = max(results, key=lambda m: m["f1_score"])
    best_model = fitted[best["name"]]
    print(f"\n      Best model: {best['name']} (F1 = {best['f1_score']:.4f})")

    # 7. Save model + scaler + metrics
    print("[5/5] Saving model and metrics...")
    joblib.dump({"model": best_model, "scaler": scaler, "best_name": best["name"]},
                MODEL_PATH)

    metrics_data = {
        "models": results,
        "best_model": best["name"],
        "dataset_size": int(len(X)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "feature_names": [
            "Mean Intensity", "Std Intensity", "Dark Pixel Ratio",
            "Edge Density", "Contour Count", "Mean Contour Area",
            "Mean Contour Perimeter", "Horizontal Variance",
            "Vertical Variance", "Laplacian Variance",
            "Pixel Skewness", "Ink Irregularity",
        ],
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_data, f, indent=2)

    print("\n=== Training complete ===\n")
    print(f"  Model saved to : {MODEL_PATH}")
    print(f"  Metrics saved to: {METRICS_PATH}\n")

    # Console comparison table
    print("Model Comparison:")
    print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} "
          f"{'Recall':>10} {'F1-Score':>10}")
    print("-" * 65)
    for r in results:
        marker = " ← BEST" if r["name"] == best["name"] else ""
        print(f"{r['name']:<20} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
              f"{r['recall']:>10.4f} {r['f1_score']:>10.4f}{marker}")

    return metrics_data


if __name__ == "__main__":
    train()
