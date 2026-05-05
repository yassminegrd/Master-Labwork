"""
=======================================================================
  train_ml.py — Classical Machine Learning Training Script
  Dysgraphia Detection Project | APM.02 | Master 1 STIC | 2025-2026
  Instructor: Dr. NECIBI Khaled | University of Constantine 2
=======================================================================

This script trains three classical ML classifiers:
  - SVM   (Support Vector Machine)
  - RF    (Random Forest)
  - KNN   (K-Nearest Neighbors)

It uses image augmentation to multiply the dataset size x8.
Features are extracted from each image using OpenCV.
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Image size for feature extraction
IMG_SIZE = 64

# Random seed for reproducibility
RANDOM_STATE = 42

# How many augmented copies per original image
AUGMENTATION_FACTOR = 8

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# STEP 0 — SYNTHETIC DATA GENERATION
# (Only runs if the data/ folder is empty)
# ──────────────────────────────────────────────────────────────

def _draw_lines(img, n_lines, y_spread, intensity_noise, spacing_noise):
    """Draw horizontal strokes that simulate handwritten text lines."""
    h, w = img.shape
    for i in range(n_lines):
        base_y = int((i + 1) * h / (n_lines + 1))
        y = base_y + int(np.random.uniform(-y_spread, y_spread))
        y = max(4, min(h - 4, y))
        x_start = np.random.randint(2, 8)
        x_end = np.random.randint(w - 8, w - 2)
        thickness = np.random.randint(1, 3)
        points = [(x, y + int(np.random.uniform(-spacing_noise, spacing_noise)))
                  for x in range(x_start, x_end, 2)]
        for j in range(len(points) - 1):
            color = int(np.random.uniform(0, intensity_noise))
            cv2.line(img, points[j], points[j + 1], color, thickness)


def ensure_data_exists(n_per_class=150):
    """Generate synthetic handwriting images when no real data is available."""
    normal_dir = os.path.join(DATA_DIR, "normal")
    dys_dir = os.path.join(DATA_DIR, "dysgraphia")
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(dys_dir, exist_ok=True)

    def count(folder):
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return sum(1 for f in os.listdir(folder)
                   if os.path.splitext(f)[1].lower() in exts)

    if count(normal_dir) < 10:
        print(f"  Génération de {n_per_class} images NORMALES synthétiques...")
        for i in range(n_per_class):
            img = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 255
            _draw_lines(img, n_lines=np.random.randint(4, 7),
                        y_spread=1.5, intensity_noise=30, spacing_noise=1)
            noise = np.random.randint(0, 10, img.shape, dtype=np.uint8)
            img = cv2.subtract(img, noise)
            cv2.imwrite(os.path.join(normal_dir, f"normal_{i:04d}.png"), img)

    if count(dys_dir) < 10:
        print(f"  Génération de {n_per_class} images DYSGRAPHIQUES synthétiques...")
        for i in range(n_per_class):
            img = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 255
            _draw_lines(img, n_lines=np.random.randint(3, 7),
                        y_spread=6, intensity_noise=80, spacing_noise=5)
            noise = np.random.randint(0, 40, img.shape, dtype=np.uint8)
            img = cv2.subtract(img, noise)
            for _ in range(np.random.randint(2, 6)):
                cx, cy = np.random.randint(5, IMG_SIZE - 5, 2)
                cv2.circle(img, (cx, cy), np.random.randint(1, 4),
                           np.random.randint(0, 80), -1)
            cv2.imwrite(os.path.join(dys_dir, f"dysgraphia_{i:04d}.png"), img)


# ──────────────────────────────────────────────────────────────
# STEP 1 — IMAGE AUGMENTATION
# Techniques: rotation, zoom, shift, brightness, noise, flip
# ──────────────────────────────────────────────────────────────

def augment_image(img):
    """
    Apply a random combination of augmentation techniques to one image.
    Returns the augmented image (grayscale, same size as input).

    Augmentation techniques used:
      - Rotation          : rotate ±15 degrees
      - Zoom              : zoom in/out by 10%
      - Horizontal shift  : shift left/right by up to 5 pixels
      - Vertical shift    : shift up/down by up to 5 pixels
      - Brightness change : add/subtract up to 30 intensity units
      - Gaussian noise    : add random pixel noise
      - Horizontal flip   : mirror the image
    """
    h, w = img.shape[:2]

    # 1. Rotation (±15°)
    angle = np.random.uniform(-15, 15)
    M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M_rot, (w, h),
                         borderMode=cv2.BORDER_REFLECT)

    # 2. Zoom (0.9 – 1.1x)
    scale = np.random.uniform(0.9, 1.1)
    M_zoom = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    img = cv2.warpAffine(img, M_zoom, (w, h),
                         borderMode=cv2.BORDER_REFLECT)

    # 3. Horizontal and vertical shift (up to ±5 px)
    tx = np.random.uniform(-5, 5)
    ty = np.random.uniform(-5, 5)
    M_shift = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M_shift, (w, h),
                         borderMode=cv2.BORDER_REFLECT)

    # 4. Brightness variation (±30 intensity)
    delta = np.random.randint(-30, 30)
    img = np.clip(img.astype(np.int32) + delta, 0, 255).astype(np.uint8)

    # 5. Gaussian noise
    if np.random.rand() > 0.5:
        noise = np.random.normal(0, 10, img.shape).astype(np.int32)
        img = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)

    # 6. Horizontal flip (50% chance)
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)

    return img


# ──────────────────────────────────────────────────────────────
# STEP 2 — FEATURE EXTRACTION
# We extract 12 handwriting-relevant features per image.
# ──────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "Mean Intensity",         # Average pixel brightness
    "Std Intensity",          # Pixel brightness variation
    "Dark Pixel Ratio",       # Fraction of dark (ink) pixels
    "Edge Density",           # Proportion of edge pixels (Canny)
    "Contour Count",          # Number of connected strokes
    "Mean Contour Area",      # Average area of each stroke
    "Mean Contour Perimeter", # Average perimeter of each stroke
    "Horizontal Variance",    # Row-to-row brightness variation (line regularity)
    "Vertical Variance",      # Column-to-column variation (spacing)
    "Laplacian Variance",     # Image sharpness / focus
    "Pixel Skewness",         # Skewness of the pixel distribution
    "Ink Irregularity",       # Local texture irregularity (8x8 patches)
]


def extract_features(img_input):
    """
    Extract 12 numerical features from a handwriting image.

    Parameters:
        img_input : numpy array (BGR or grayscale) or file path (str)

    Returns:
        numpy array of shape (12,) or None if the image cannot be read
    """
    # Load image from file or use array directly
    if isinstance(img_input, str):
        img = cv2.imread(img_input)
        if img is None:
            return None
    else:
        img = img_input

    # Convert to grayscale if needed
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Resize to standard size
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    # Gaussian blur to reduce noise, then Otsu thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Feature 1-3: Basic intensity statistics
    mean_intensity = float(np.mean(gray))
    std_intensity  = float(np.std(gray))
    dark_ratio     = float(np.sum(gray < 128)) / gray.size

    # Feature 4: Edge density using Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    edge_density = float(np.sum(edges > 0)) / edges.size

    # Feature 5-7: Contour (stroke) analysis
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)
    areas  = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 1]
    perims = [cv2.arcLength(c, True) for c in contours if cv2.contourArea(c) > 1]
    mean_area  = float(np.mean(areas))  if areas  else 0.0
    mean_perim = float(np.mean(perims)) if perims else 0.0

    # Feature 8-9: Line regularity (variance of row/column means)
    horiz_variance = float(np.std(np.mean(gray, axis=1)))
    vert_variance  = float(np.std(np.mean(gray, axis=0)))

    # Feature 10: Laplacian variance → measures image sharpness
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Feature 11: Pixel skewness (asymmetry of brightness distribution)
    flat  = gray.flatten().astype(np.float64)
    mu    = np.mean(flat)
    sigma = np.std(flat)
    skewness = float(np.mean(((flat - mu) / (sigma + 1e-8)) ** 3))

    # Feature 12: Ink irregularity (mean std of 8x8 local blocks)
    block, local_stds = 8, []
    for r in range(0, IMG_SIZE - block, block):
        for c in range(0, IMG_SIZE - block, block):
            local_stds.append(np.std(gray[r:r + block, c:c + block]))
    ink_irregularity = float(np.mean(local_stds))

    return np.array([
        mean_intensity, std_intensity, dark_ratio, edge_density,
        contour_count, mean_area, mean_perim, horiz_variance,
        vert_variance, laplacian_var, skewness, ink_irregularity,
    ], dtype=np.float64)


# ──────────────────────────────────────────────────────────────
# STEP 3 — LOAD DATASET WITH AUGMENTATION
# ──────────────────────────────────────────────────────────────

def load_dataset_with_augmentation():
    """
    Load images from data/normal/ and data/dysgraphia/, apply augmentation
    (AUGMENTATION_FACTOR copies per image), and extract features.

    Labels:
        0 = Normal handwriting
        1 = Dysgraphia

    Returns:
        X : numpy array, shape (n_samples, 12)
        y : numpy array, shape (n_samples,)
    """
    categories = {"normal": 0, "dysgraphia": 1}
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    X, y = [], []
    total_original = 0
    errors = 0

    print(f"\n  Augmentation (x{AUGMENTATION_FACTOR}) + Extraction features en streaming...")

    for label_name, label_idx in categories.items():
        folder = os.path.join(DATA_DIR, label_name)
        if not os.path.exists(folder):
            print(f"  Attention : dossier '{folder}' introuvable, ignoré.")
            continue

        files = sorted(f for f in os.listdir(folder)
                       if os.path.splitext(f)[1].lower() in exts)

        for fname in files:
            path = os.path.join(folder, fname)
            img = cv2.imread(path)
            if img is None:
                errors += 1
                continue

            total_original += 1

            # Convert to grayscale for augmentation
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

            # Extract features from the ORIGINAL image
            feat = extract_features(img)
            if feat is not None:
                X.append(feat)
                y.append(label_idx)

            # Extract features from AUGMENTED copies
            for _ in range(AUGMENTATION_FACTOR - 1):
                aug_img = augment_image(gray.copy())
                # Convert back to BGR for extract_features compatibility
                aug_bgr = cv2.cvtColor(aug_img, cv2.COLOR_GRAY2BGR)
                feat_aug = extract_features(aug_bgr)
                if feat_aug is not None:
                    X.append(feat_aug)
                    y.append(label_idx)

            # Progress indicator every 50 images
            if total_original % 50 == 0:
                print(f"  ... {total_original} images traitées | "
                      f"features: {len(X)}")

    print(f"  Features extraites : ({len(X)}, {len(X[0]) if X else 0}) | "
          f"Erreurs : {errors}")
    return np.array(X), np.array(y)


# ──────────────────────────────────────────────────────────────
# STEP 4 — MODEL TRAINING & EVALUATION
# ──────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, name):
    """Compute accuracy, precision, recall, F1, confusion matrix."""
    y_pred = model.predict(X_test)
    return {
        "name": name,
        "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score":  round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "report": classification_report(y_test, y_pred,
                                        target_names=["Normal", "Dysgraphia"],
                                        zero_division=0),
    }


def train():
    """
    Main training function.
    Returns metrics dict for all three models.
    """
    print("\n" + "=" * 60)
    print("  DYSGRAPHIA DETECTION — MODÈLES CLASSIQUES (ML)")
    print("=" * 60)

    # ── Étape 0 : Vérifier/créer les données
    print("\n[1/5] Vérification des données...")
    ensure_data_exists()

    # Count raw images
    for lbl in ["normal", "dysgraphia"]:
        folder = os.path.join(DATA_DIR, lbl)
        if os.path.exists(folder):
            n = len([f for f in os.listdir(folder)
                     if os.path.splitext(f)[1].lower() in
                     {".jpg", ".jpeg", ".png", ".bmp"}])
            print(f"  {lbl:<12} -> {n} images")

    # ── Étape 1 : Charger le dataset avec augmentation
    print("\n[2/5] Chargement et augmentation des images...")
    X, y = load_dataset_with_augmentation()
    if len(X) == 0:
        raise RuntimeError("Aucune image trouvée dans data/")
    print(f"  Total : {len(X)} samples | "
          f"Normal: {sum(y==0)} | Dysgraphie: {sum(y==1)}")

    # ── Étape 2 : Diviser en train / test (80% / 20%)
    print("\n[3/5] Division train/test (80% / 20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train : {len(X_train)} | Test : {len(X_test)}")

    # ── Étape 3 : Normalisation des features (StandardScaler)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── Étape 4 : Entraîner les 3 modèles
    print("\n[4/5] Entraînement SVM, Random Forest et KNN...")
    models = {
        "SVM":           SVC(kernel="rbf", C=1.0, gamma="scale",
                             probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100,
                                                random_state=RANDOM_STATE),
        "KNN":           KNeighborsClassifier(n_neighbors=5),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        m = evaluate_model(model, X_test_sc, y_test, name)
        results.append(m)

        # Sauvegarde de chaque modèle
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, os.path.join(MODELS_DIR, f"{safe_name}.pkl"))
        print(f"  {name:<16} Acc: {m['accuracy']:.4f}  F1: {m['f1_score']:.4f}")
        print(m["report"])

    # ── Étape 5 : Sélectionner le meilleur modèle
    best = max(results, key=lambda m: m["f1_score"])
    best_model = models[best["name"]]
    print(f"\n  Meilleur modèle : {best['name']} (F1 = {best['f1_score']:.4f})")

    # ── Sauvegarde du meilleur modèle + scaler
    print("\n[5/5] Sauvegarde des modèles et métriques...")
    joblib.dump({"model": best_model, "scaler": scaler,
                 "best_name": best["name"]},
                os.path.join(MODELS_DIR, "best_ml_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    # ── Sauvegarde des métriques JSON
    metrics_data = {
        "models": results,
        "best_model": best["name"],
        "dataset_size": int(len(X)),
        "train_size":   int(len(X_train)),
        "test_size":    int(len(X_test)),
        "feature_names": FEATURE_NAMES,
        "augmentation_factor": AUGMENTATION_FACTOR,
    }
    with open(os.path.join(REPORTS_DIR, "ml_metrics.json"), "w") as f:
        json.dump(metrics_data, f, indent=2)

    # ── Résumé final
    print("\n" + "=" * 60)
    print("  RÉSUMÉ")
    print("=" * 60)
    print(f"  {'Modèle':<20} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print("  " + "-" * 54)
    for r in results:
        flag = " ← BEST" if r["name"] == best["name"] else ""
        print(f"  {r['name']:<20} {r['accuracy']:>8.4f} "
              f"{r['precision']:>8.4f} {r['recall']:>8.4f} "
              f"{r['f1_score']:>8.4f}{flag}")
    print("\n  Modèles sauvegardés -> models/")
    print("  Métriques sauvegardées -> reports/ml_metrics.json")
    print("=" * 60 + "\n")

    return metrics_data


if __name__ == "__main__":
    train()
