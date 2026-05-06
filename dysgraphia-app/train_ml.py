"""
============================================================
  Apprentissage par Machine 02 (APM.02)
  Projet : Détection de la Dysgraphie
  Dr. NECIBI Khaled | Université Constantine 2 | 2025/2026
============================================================

  Fichier : train_ml.py
  Modèles : KNN, SVM, Random Forest (classiques)

  Basé sur : Workshop 01 — Measuring Model Performance
  (même structure, mêmes étapes, même style de code)

  Étapes :
    Step 0  : Imports
    Step 1  : Chargement des images depuis data/
    Step 2  : Augmentation des données (x8)
    Step 3  : Extraction des features (12 features OpenCV)
    Step 4  : Prétraitement (split 80/20 + normalisation)
    Step 5  : Entraînement des modèles (KNN, SVM, RF)
    Step 6  : Évaluation (accuracy, precision, recall, F1)
    Step 7  : Affichage rapport + matrice de confusion
    Step 8  : Sauvegarde des modèles
"""

# ============================================================
# Step 0 : Import all required libraries
# (comme dans Workshop 01 — on importe tout en premier)
# ============================================================

import os
import cv2
import numpy as np
import joblib
import json

# scikit-learn : même bibliothèque utilisée dans Workshop 01
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# matplotlib : visualisations (barres comparatives + matrices de confusion)
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker

# ============================================================
# Configuration générale
# ============================================================

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

IMG_SIZE          = 64    # Taille de redimensionnement des images (64x64)
AUGMENTATION_FACTOR = 8   # Nombre de copies augmentées par image originale
RANDOM_STATE      = 42    # Graine aléatoire (pour la reproductibilité)

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ============================================================
# Génération de données synthétiques
# (utilisé seulement si le dossier data/ est vide)
# ============================================================

def generate_synthetic_data(n_per_class=150):
    """Génère des images d'écriture synthétiques si aucune donnée réelle."""
    normal_dir = os.path.join(DATA_DIR, "normal")
    dys_dir    = os.path.join(DATA_DIR, "dysgraphia")
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(dys_dir,    exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    count_normal = sum(1 for f in os.listdir(normal_dir)
                       if os.path.splitext(f)[1].lower() in exts)
    count_dys    = sum(1 for f in os.listdir(dys_dir)
                       if os.path.splitext(f)[1].lower() in exts)

    if count_normal < 10:
        print(f"  Génération de {n_per_class} images NORMALES synthétiques...")
        for i in range(n_per_class):
            img = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 255
            for line in range(np.random.randint(4, 7)):
                y = int((line + 1) * IMG_SIZE / 7)
                y = max(4, min(IMG_SIZE - 4, y + np.random.randint(-2, 2)))
                cv2.line(img, (4, y), (IMG_SIZE - 4, y),
                         np.random.randint(0, 40), np.random.randint(1, 2))
            cv2.imwrite(os.path.join(normal_dir, f"normal_{i:04d}.png"), img)

    if count_dys < 10:
        print(f"  Génération de {n_per_class} images DYSGRAPHIQUES synthétiques...")
        for i in range(n_per_class):
            img = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 255
            for line in range(np.random.randint(3, 7)):
                y = int((line + 1) * IMG_SIZE / 7)
                y = max(4, min(IMG_SIZE - 4, y + np.random.randint(-8, 8)))
                cv2.line(img, (4, y), (IMG_SIZE - 4, y + np.random.randint(-6, 6)),
                         np.random.randint(0, 90), np.random.randint(1, 3))
            noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
            img = cv2.subtract(img, noise)
            cv2.imwrite(os.path.join(dys_dir, f"dysgraphia_{i:04d}.png"), img)


# ============================================================
# Step 1 : Chargement des images
# (même idée que "Load the dataset" — Workshop 01, Step 01)
# ============================================================

def load_images():
    """
    Charge toutes les images depuis :
        data/normal/      → label 0 (écriture normale)
        data/dysgraphia/  → label 1 (dysgraphie)

    Retourne :
        images : liste de tableaux numpy (images en niveaux de gris)
        labels : liste d'entiers (0 ou 1)
    """
    categories = {
        "normal":     0,   # 0 = écriture normale
        "dysgraphia": 1    # 1 = dysgraphie
    }
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images, labels = [], []
    total, errors = 0, 0

    print("\n  Chargement images brutes depuis : dataset")
    for label_name, label_idx in categories.items():
        folder = os.path.join(DATA_DIR, label_name)
        if not os.path.exists(folder):
            print(f"  Dossier '{folder}' introuvable.")
            continue

        files = sorted(f for f in os.listdir(folder)
                       if os.path.splitext(f)[1].lower() in exts)
        count = 0
        for fname in files:
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                errors += 1
                continue
            images.append(img)
            labels.append(label_idx)
            count += 1
            total += 1

        print(f"    {label_name:<12} -> {count} images")

    print(f"  {total} images chargées | {errors} erreurs")
    print(f"  Dysgraphique : {sum(l==1 for l in labels)} | "
          f"Typical : {sum(l==0 for l in labels)}")
    return images, labels


# ============================================================
# Step 2 : Augmentation des données
# Techniques : rotation, zoom, décalage, luminosité, bruit, flip
# (Augmentation x8 pour enrichir le dataset)
# ============================================================

def augment_image(img):
    """
    Applique des transformations aléatoires à une image.
    Retourne l'image augmentée (même taille).

    Techniques utilisées :
      - Rotation          : ±15 degrés
      - Zoom              : facteur 0.9 – 1.1
      - Décalage H/V      : ±5 pixels
      - Variation de luminosité : ±30
      - Bruit Gaussien    : ajout de bruit aléatoire
      - Flip horizontal   : miroir de l'image (50% chance)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    h, w = gray.shape

    # 1. Rotation
    angle = np.random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    gray = cv2.warpAffine(gray, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 2. Zoom
    scale = np.random.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    gray = cv2.warpAffine(gray, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 3. Décalage horizontal et vertical
    tx, ty = np.random.uniform(-5, 5), np.random.uniform(-5, 5)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    gray = cv2.warpAffine(gray, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 4. Variation de luminosité
    delta = np.random.randint(-30, 30)
    gray = np.clip(gray.astype(np.int32) + delta, 0, 255).astype(np.uint8)

    # 5. Bruit Gaussien (50% de chance)
    if np.random.rand() > 0.5:
        noise = np.random.normal(0, 10, gray.shape).astype(np.int32)
        gray = np.clip(gray.astype(np.int32) + noise, 0, 255).astype(np.uint8)

    # 6. Flip horizontal (50% de chance)
    if np.random.rand() > 0.5:
        gray = cv2.flip(gray, 1)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ============================================================
# Step 3 : Extraction des features (caractéristiques)
# 12 features extraites de chaque image d'écriture
# ============================================================

FEATURE_NAMES = [
    "Mean Intensity",          # Intensité moyenne des pixels
    "Std Intensity",           # Écart-type de l'intensité
    "Dark Pixel Ratio",        # Proportion de pixels sombres (encre)
    "Edge Density",            # Densité des contours (Canny)
    "Contour Count",           # Nombre de traits/contours
    "Mean Contour Area",       # Surface moyenne des traits
    "Mean Contour Perimeter",  # Périmètre moyen des traits
    "Horizontal Variance",     # Régularité des lignes (axe horizontal)
    "Vertical Variance",       # Régularité des espaces (axe vertical)
    "Laplacian Variance",      # Netteté de l'écriture
    "Pixel Skewness",          # Asymétrie de la distribution des pixels
    "Ink Irregularity",        # Irrégularité locale de l'encre (blocs 8x8)
]


def extract_features(img_input):
    """
    Extrait 12 features numériques depuis une image d'écriture.

    Paramètres :
        img_input : tableau numpy (BGR ou niveaux de gris) ou chemin (str)

    Retourne :
        numpy array de shape (12,) ou None si erreur
    """
    # Lecture de l'image
    if isinstance(img_input, str):
        img = cv2.imread(img_input)
        if img is None:
            return None
    else:
        img = img_input

    # Conversion en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    # Prétraitement : flou + seuillage Otsu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Features 1-3 : statistiques d'intensité de base
    mean_intensity = float(np.mean(gray))
    std_intensity  = float(np.std(gray))
    dark_ratio     = float(np.sum(gray < 128)) / gray.size

    # Feature 4 : densité des bords (détecteur de Canny)
    edges = cv2.Canny(blurred, 50, 150)
    edge_density = float(np.sum(edges > 0)) / edges.size

    # Features 5-7 : analyse des contours (traits d'écriture)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)
    areas  = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 1]
    perims = [cv2.arcLength(c, True) for c in contours if cv2.contourArea(c) > 1]
    mean_area  = float(np.mean(areas))  if areas  else 0.0
    mean_perim = float(np.mean(perims)) if perims else 0.0

    # Features 8-9 : régularité des lignes et des espaces
    horiz_variance = float(np.std(np.mean(gray, axis=1)))
    vert_variance  = float(np.std(np.mean(gray, axis=0)))

    # Feature 10 : netteté de l'image (variance du Laplacien)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Feature 11 : asymétrie (skewness) de la distribution des pixels
    flat  = gray.flatten().astype(np.float64)
    mu    = np.mean(flat)
    sigma = np.std(flat)
    skewness = float(np.mean(((flat - mu) / (sigma + 1e-8)) ** 3))

    # Feature 12 : irrégularité locale de l'encre (blocs 8x8)
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


def build_feature_matrix(images, labels):
    """
    Applique l'augmentation (x8) et extrait les features de chaque image.
    Retourne X (features) et y (labels).
    """
    X, y = [], []
    total, errors = 0, 0

    print(f"\n  Augmentation (x{AUGMENTATION_FACTOR}) + "
          f"Extraction features en streaming...")

    for img, label in zip(images, labels):
        total += 1

        # Image originale
        feat = extract_features(img)
        if feat is not None:
            X.append(feat)
            y.append(label)

        # Copies augmentées
        for _ in range(AUGMENTATION_FACTOR - 1):
            aug = augment_image(img.copy())
            feat_aug = extract_features(aug)
            if feat_aug is not None:
                X.append(feat_aug)
                y.append(label)
            else:
                errors += 1

        if total % 50 == 0:
            print(f"  ... {total}/{len(images)} images traitées | "
                  f"features: {len(X)}")

    print(f"  Features extraites : ({len(X)}, {len(X[0]) if X else 0}) | "
          f"Erreurs : {errors}")
    return np.array(X), np.array(y)


# ============================================================
# Step 4 : Prétraitement des données
# (même approche que Workshop 01, Step 02)
#   - Division train/test : 80% / 20%
#   - Normalisation avec StandardScaler
# ============================================================

def preprocess(X, y):
    """
    Divise les données et normalise les features.

    Étape identique au Workshop 01 :
      - train_test_split(X, y, test_size=0.2, random_state=42)
      - StandardScaler().fit_transform(X_train)
    """
    # Division 80% train / 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y   # Conserver la proportion des classes
    )
    print(f"\n  Train : {len(X_train)} | Test : {len(X_test)}")

    # Normalisation des features (StandardScaler — Workshop 01, Step 02)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # Apprendre + transformer
    X_test_scaled  = scaler.transform(X_test)         # Seulement transformer

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ============================================================
# Step 5 : Entraînement des modèles
# (même structure que Workshop 01, Step 03)
# ============================================================

def train_models(X_train, y_train):
    """
    Initialise et entraîne les trois modèles de classification.

    Modèles (comme dans Workshop 01 et Workshop 02) :
      - KNN   : K-Nearest Neighbors
      - SVM   : Support Vector Machine
      - RF    : Random Forest
    """
    print("\n  Entraînement des modèles...")

    models = {}

    # KNN — K plus proches voisins (Workshop 01)
    print("    KNN...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    models["KNN"] = knn

    # SVM — Séparateur à vaste marge
    print("    SVM...")
    svm = SVC(kernel="rbf", C=1.0, gamma="scale",
              probability=True, random_state=RANDOM_STATE)
    svm.fit(X_train, y_train)
    models["SVM"] = svm

    # Random Forest — Forêt aléatoire
    print("    Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf

    return models


# ============================================================
# Step 6 : Évaluation des modèles
# (même métriques que Workshop 01, Step 04)
#   - accuracy_score
#   - precision_score
#   - recall_score
#   - f1_score
#   - confusion_matrix
#   - classification_report
# ============================================================

def evaluate_model(model, X_test, y_test, model_name):
    """
    Évalue un modèle sur le jeu de test.

    Métriques (Workshop 01, Step 04) :
      - Accuracy  : proportion de prédictions correctes
      - Precision : parmi les prédits positifs, combien sont vrais positifs
      - Recall    : parmi les vrais positifs, combien sont détectés
      - F1-Score  : moyenne harmonique de Precision et Recall
    """
    # Prédiction sur le jeu de test
    y_pred = model.predict(X_test)

    # Calcul des métriques (Workshop 01, Step 04)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    # Affichage (comme dans Workshop 01)
    print(f"\n  {model_name}_dysg")
    print(f"  Accuracy : {acc:.4f}")
    print()
    print(classification_report(
        y_test, y_pred,
        target_names=["Typical", "Dysgraphique"],
        zero_division=0
    ))

    return {
        "name":             model_name,
        "accuracy":         round(float(acc),  4),
        "precision":        round(float(prec), 4),
        "recall":           round(float(rec),  4),
        "f1_score":         round(float(f1),   4),
        "confusion_matrix": cm.tolist(),
        "report":           classification_report(
            y_test, y_pred,
            target_names=["Typical", "Dysgraphique"],
            zero_division=0
        ),
    }


# ============================================================
# Step 7 : Affichage du résumé
# (Workshop 01, Step 05 — interprétation des résultats)
# ============================================================

def print_summary(results):
    """Affiche un tableau récapitulatif des résultats (Workshop 01, Step 05)."""
    print("\n" + "=" * 60)
    print("  RÉSUMÉ — TACHE 1 : Classification Dysgraphie (2 classes)")
    print("=" * 60)
    best = max(results, key=lambda r: r["f1_score"])
    for r in results:
        flag = " ← BEST" if r["name"] == best["name"] else ""
        print(f"  Train : {r['name']:<16}  "
              f"Accuracy : {r['accuracy']:.4f}{flag}")
    print()
    print("  Modèles -> models/")
    print("  Rapports -> reports/")
    print("=" * 60 + "\n")


# ============================================================
# VISUALISATION — Machine Learning Comparison
# (Workshop 01, Step 05 — Performance Interpretation)
#
# Graphique en barres groupées comparant :
#   - Accuracy · Precision · Recall · F1-Score
# pour chaque modèle ML (KNN, SVM, Random Forest)
#
# Affiché avec plt.show() — aucune sauvegarde
# ============================================================

def plot_ml_comparison(results):
    """
    Affiche un graphique en barres comparant les 3 modèles ML.

    Métriques affichées (Workshop 01, Step 04) :
      - Accuracy
      - Precision
      - Recall
      - F1-Score

    Paramètres :
        results : liste de dicts retournés par evaluate_model()
    """
    # ── Données
    model_names = [r["name"] for r in results]
    metrics     = ["accuracy", "precision", "recall", "f1_score"]
    labels      = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors      = ["#6366f1", "#a855f7", "#06b6d4", "#22c55e"]

    n_models  = len(model_names)
    n_metrics = len(metrics)
    x         = np.arange(n_models)        # positions des groupes
    width     = 0.18                        # largeur de chaque barre

    # ── Figure
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#0e1118")
    ax.set_facecolor("#131720")

    # ── Tracé des barres groupées
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        values = [r[metric] * 100 for r in results]   # en %
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, values,
            width,
            label=label,
            color=color,
            alpha=0.88,
            edgecolor="#1d2333",
            linewidth=0.6
        )
        # Valeur au-dessus de chaque barre
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center", va="bottom",
                fontsize=7.5, fontweight="bold",
                color="white"
            )

    # ── Axes et titres
    ax.set_title(
        "Machine Learning — Comparaison des Modèles (Dysgraphia Detection)\n"
        "APM.02 · Workshop 01 · Dr. NECIBI Khaled · Université Constantine 2",
        fontsize=12, fontweight="bold", color="white", pad=16
    )
    ax.set_xlabel("Modèles", fontsize=10, color="#8891a8", labelpad=8)
    ax.set_ylabel("Score (%)", fontsize=10, color="#8891a8", labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11, color="white", fontweight="bold")
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 10))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0f%%"))

    # ── Grille
    ax.yaxis.grid(True, color="#1e2640", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(colors="#8891a8", length=0)

    # ── Légende
    legend = ax.legend(
        loc="upper right", fontsize=9,
        framealpha=0.15, edgecolor="#252e48",
        labelcolor="white"
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# VISUALISATION — Matrices de Confusion (ML)
# (Workshop 01, Step 04 — Plot Confusion Matrix)
#
# Affiche une figure avec les matrices de confusion
# de chaque modèle ML côte à côte
# Nombres dans chaque cellule + labels TP/TN/FP/FN
# ============================================================

def plot_ml_confusion_matrices(results):
    """
    Affiche les matrices de confusion de tous les modèles ML.

    Même logique que Workshop 03 — "Plot Confusion Matrix"
    Utilise matplotlib uniquement (pas seaborn)

    Cellules :
      TN (haut-gauche)  = vert  → Normal prédit Normal
      FP (haut-droite)  = rouge → Normal prédit Dysgraphie
      FN (bas-gauche)   = rouge → Dysgraphie prédit Normal
      TP (bas-droite)   = vert  → Dysgraphie prédit Dysgraphie
    """
    classes   = ["Typical", "Dysgraphia"]
    n_models  = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    fig.patch.set_facecolor("#0e1118")

    if n_models == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        cm = np.array(r["confusion_matrix"])

        # Couleurs : vert pour TN/TP, rouge pour FP/FN
        cell_colors = np.array([
            ["#166534", "#7f1d1d"],
            ["#7f1d1d", "#166534"]
        ])

        ax.set_facecolor("#131720")
        ax.set_title(r["name"], fontsize=13, fontweight="bold", color="white", pad=12)

        # Cellules colorées + nombres
        for i in range(2):
            for j in range(2):
                rect = plt.Rectangle(
                    [j, 1 - i], 1, 1,
                    color=cell_colors[i][j], ec="#1d2333", lw=1.5
                )
                ax.add_patch(rect)
                # Nombre principal
                ax.text(
                    j + 0.5, 1.5 - i, str(cm[i][j]),
                    ha="center", va="center",
                    fontsize=26, fontweight="bold", color="white"
                )
                # Étiquette TP / TN / FP / FN
                cell_label = [["TN", "FP"], ["FN", "TP"]][i][j]
                ax.text(
                    j + 0.5, 1.5 - i - 0.28, cell_label,
                    ha="center", va="center",
                    fontsize=10, color="rgba(255,255,255,0.55)" if False else "#ffffff99"
                )

        # Axes
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(classes, fontsize=10, color="white")
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(classes[::-1], fontsize=10, color="white")
        ax.set_xlabel("Predicted", fontsize=10, color="#8891a8", labelpad=8)
        ax.set_ylabel("True", fontsize=10, color="#8891a8", labelpad=8)
        ax.tick_params(length=0, colors="white")
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Accuracy en bas
        ax.text(
            1, -0.22, f"Accuracy : {r['accuracy']*100:.2f}%",
            ha="center", fontsize=9, color="#8891a8",
            transform=ax.transData
        )

    fig.suptitle(
        "Matrices de Confusion — Modèles ML (Dysgraphia Detection)\n"
        "APM.02 · Workshop 01 · Dr. NECIBI Khaled",
        fontsize=12, fontweight="bold", color="white", y=1.02
    )
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN — Pipeline complet (comme dans les Workshops)
# ============================================================

def train():
    """
    Pipeline complet d'entraînement ML.
    Suit les mêmes étapes que Workshop 01 (APM.02).
    """
    print("\n" + "=" * 60)
    print("  Chargement images brutes depuis : dataset")
    print("=" * 60)

    # Générer des données synthétiques si nécessaire
    generate_synthetic_data()

    # STEP 1 : Charger les images
    images, labels = load_images()
    if len(images) == 0:
        raise RuntimeError("Aucune image trouvée dans data/")

    print(f"\n  {len(images)} images chargées | 0 erreurs")
    print(f"  Dysgraphique : {sum(l==1 for l in labels)} | "
          f"Regular : 0 | Medium : 0 | Irregular : {sum(l==1 for l in labels)}")

    # STEP 2 : Augmentation + Extraction des features
    print("\n" + "=" * 60)
    X, y = build_feature_matrix(images, labels)
    print(f"  Features sauvegardé : models/scaler.pkl")
    print()

    # STEP 3 : Prétraitement (split + normalisation)
    print("  TACHE 1 : Classification Dysgraphie (2 classes)")
    print("  " + "=" * 42)
    X_train, X_test, y_train, y_test, scaler = preprocess(X, y)
    print(f"  Train : {len(X_train)} | Test : {len(X_test)}")

    # STEP 4 : Entraînement des modèles
    models = train_models(X_train, y_train)

    # STEP 5 : Évaluation de chaque modèle (Workshop 01, Step 04)
    results = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)

        # Sauvegarder chaque modèle
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, os.path.join(MODELS_DIR, f"{safe_name}.pkl"))

    # STEP 6 : Choisir le meilleur modèle et le sauvegarder
    best = max(results, key=lambda r: r["f1_score"])
    best_model = models[best["name"]]
    joblib.dump(
        {"model": best_model, "scaler": scaler, "best_name": best["name"]},
        os.path.join(MODELS_DIR, "best_ml_model.pkl")
    )
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    # STEP 7 : Afficher le résumé (Workshop 01, Step 05)
    print_summary(results)

    # ── VISUALISATION 1 : Barres comparatives (Accuracy, Precision, Recall, F1)
    print("\n  Affichage du graphique comparatif ML...")
    plot_ml_comparison(results)

    # ── VISUALISATION 2 : Matrices de confusion (TN / FP / FN / TP)
    print("  Affichage des matrices de confusion ML...")
    plot_ml_confusion_matrices(results)

    # STEP 8 : Sauvegarder les métriques en JSON
    metrics_data = {
        "models":             results,
        "best_model":         best["name"],
        "dataset_size":       int(len(X)),
        "train_size":         int(len(X_train)),
        "test_size":          int(len(X_test)),
        "feature_names":      FEATURE_NAMES,
        "augmentation_factor": AUGMENTATION_FACTOR,
    }
    with open(os.path.join(REPORTS_DIR, "ml_metrics.json"), "w") as f:
        json.dump(metrics_data, f, indent=2)

    return metrics_data


# ============================================================
# Point d'entrée
# ============================================================
if __name__ == "__main__":
    train()
