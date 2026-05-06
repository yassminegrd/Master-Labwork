"""
============================================================
  Apprentissage par Machine 02 (APM.02)
  Projet : Détection de la Dysgraphie
  Dr. NECIBI Khaled | Université Constantine 2 | 2025/2026
============================================================

  Fichier : train_cnn.py
  Modèle  : CNN — Réseau de Neurones Convolutif

  Basé sur :
    Workshop 03 — Simple Neural Network Architecture (MLP)
    Workshop 04 — CNN application (VGG-like)

  Même structure de code que les workshops :
    Step 0 : Import all the required libraries
    Step 1 : Charger et prétraiter les images
    Step 2 : Construire le modèle CNN (Sequential — comme Workshop 03)
    Step 3 : Compiler le modèle
    Step 4 : Entraîner le modèle (model.fit)
    Step 5 : Afficher l'historique d'entraînement
    Step 6 : Évaluation du modèle (comme Workshop 03)
    Step 7 : Rapport de classification + matrice de confusion
    Step 8 : Sauvegarder le modèle

  Architecture CNN (inspirée de Workshop 04 — VGG-like blocks) :
    Input (64×64×1)
      → Conv2D(32) + ReLU + MaxPool + Dropout
      → Conv2D(64) + ReLU + MaxPool + Dropout
      → Conv2D(128)+ ReLU + MaxPool + Dropout
      → Flatten → Dense(256, ReLU) → Dropout
      → Dense(1, Sigmoid)   ← classification binaire
"""

# ============================================================
# Step 0 : Import all the required libraries
# (même style que Workshop 04, Step 0)
# ============================================================

import os
import json
import numpy as np
import cv2

# TensorFlow / Keras (utilisé dans Workshop 03 et Workshop 04)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dropout,
    Flatten, Dense
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Métriques scikit-learn (comme Workshop 01 et Workshop 03)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# matplotlib : visualisations (courbes + matrice de confusion)
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Patch

# ============================================================
# Configuration
# ============================================================

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

IMG_SIZE     = 64    # Taille de l'image d'entrée : 64×64 pixels
BATCH_SIZE   = 32    # Nombre d'images par batch d'entraînement
EPOCHS       = 50    # Nombre maximum d'époques
RANDOM_STATE = 42    # Reproductibilité

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Réduire les messages verbose de TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ============================================================
# Step 1 : Charger et prétraiter les images
# (Workshop 03 — "Preprocessing image data")
#
# Étapes de prétraitement (comme Workshop 03) :
#   1. Convertir en niveaux de gris (grayscale)
#   2. Redimensionner en 64×64
#   3. Normaliser les pixels : diviser par 255 → valeurs dans [0, 1]
#   4. Ajouter la dimension du canal : (64,64) → (64,64,1)
# ============================================================

def load_and_preprocess_images():
    """
    Charge toutes les images depuis data/normal/ et data/dysgraphia/.
    Applique le prétraitement standard (Workshop 03) :
      - Conversion en niveaux de gris
      - Redimensionnement en 64×64
      - Normalisation : pixels / 255.0  (valeurs dans [0, 1])

    Retourne :
        X : numpy array, shape (N, 64, 64, 1), float32
        y : numpy array, shape (N,), int (0=normal, 1=dysgraphie)
    """
    categories = {
        "normal":     0,   # Label 0 = écriture normale
        "dysgraphia": 1    # Label 1 = dysgraphie
    }
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images, labels = [], []
    total, errors = 0, 0

    print("\n  Chargement des images brutes depuis : dataset")

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

            # Conversion en niveaux de gris (Workshop 03 — preprocessing)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Redimensionnement à 64×64
            gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

            # Normalisation : pixel / 255.0  (Workshop 03 — "Normalizing")
            gray = gray.astype(np.float32) / 255.0

            images.append(gray)
            labels.append(label_idx)
            count += 1
            total += 1

        print(f"    {label_name:<12} -> {count} images")

    print(f"  {total} images chargées | {errors} erreurs")
    print(f"  Dysgraphique : {sum(l==1 for l in labels)} | "
          f"Typical : {sum(l==0 for l in labels)}")

    # Conversion en tableaux numpy
    X = np.array(images)

    # Ajout de la dimension du canal : (N, 64, 64) → (N, 64, 64, 1)
    # (Workshop 04 — les CNN nécessitent une dimension canal)
    X = X[..., np.newaxis]

    y = np.array(labels)
    return X, y


# ============================================================
# Step 2 : Construire le modèle CNN
# (Workshop 04 — CNN architecture, inspiré de VGG)
# (Workshop 03 — même style Sequential de Keras)
#
# Architecture (identique à Workshop 03 pour la structure Sequential) :
#   model = Sequential([...])
#
# Blocs convolutifs (Workshop 04 — VGG-like blocks) :
#   Conv2D → ReLU → MaxPooling → Dropout
# ============================================================

def build_cnn_model():
    """
    Construit le modèle CNN avec l'API Sequential de Keras.

    Même style de code que Workshop 03 (MLP) et Workshop 04 (CNN/VGG) :
        model = Sequential([
            Conv2D(32, (3,3), activation='relu', ...),
            MaxPooling2D((2,2)),
            Dropout(0.25),
            ...
            Dense(1, activation='sigmoid')
        ])

    Couches :
      - Conv2D   : extrait les features locales (bords, textures, traits)
      - MaxPooling2D : réduit la taille spatiale (sous-échantillonnage)
      - Dropout  : évite le surapprentissage (regularisation)
      - Flatten  : convertit les feature maps 2D en vecteur 1D
      - Dense    : couche fully-connected (comme dans Workshop 03 MLP)
      - Sigmoid  : activation de sortie pour classification binaire
    """

    model = Sequential([
        # ── Bloc 1 : 32 filtres 3×3
        # Apprend les features bas niveau : bords, coins, traits
        Conv2D(32, (3, 3), activation='relu', padding='same',
               input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D((2, 2)),          # 64×64 → 32×32
        Dropout(0.25),                 # Dropout 25% des neurones

        # ── Bloc 2 : 64 filtres 3×3
        # Apprend les features intermédiaires : courbes, motifs d'écriture
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),          # 32×32 → 16×16
        Dropout(0.25),

        # ── Bloc 3 : 128 filtres 3×3
        # Apprend les features haut niveau : style d'écriture global
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),          # 16×16 → 8×8
        Dropout(0.25),

        # ── Aplatissement (Flatten — Workshop 03)
        # Convertit (8, 8, 128) → vecteur de 8192 dimensions
        Flatten(),

        # ── Couche Dense (comme Workshop 03 — hidden layer)
        Dense(256, activation='relu'),
        Dropout(0.5),                  # Dropout fort avant la sortie

        # ── Couche de sortie
        # 1 neurone, activation Sigmoid → probabilité dans [0, 1]
        # Sortie > 0.5 → Dysgraphie | Sortie ≤ 0.5 → Normal
        Dense(1, activation='sigmoid'),
    ])

    return model


# ============================================================
# Step 3 : Compiler le modèle
# (Workshop 03 — "Loss function & optimization")
#
# Optimiseur : Adam (Workshop 03 — le plus populaire)
# Perte      : Binary Crossentropy (pour classification binaire)
# Métrique   : Accuracy
# ============================================================

def compile_model(model):
    """
    Compile le modèle CNN.

    (Workshop 03 — même appel model.compile que dans le cours)
    """
    model.compile(
        optimizer='adam',                 # Optimiseur Adam (Workshop 03)
        loss='binary_crossentropy',       # Perte pour classification binaire
        metrics=['accuracy']             # Métrique : accuracy
    )
    return model


# ============================================================
# Step 4 : Entraîner le modèle
# (Workshop 03 — "Training the MLP model")
#
# Même appel : model.fit(X_train, y_train, ...)
# Avec validation_data pour surveiller la généralisation
# ============================================================

def train_cnn(model, X_train, y_train, X_test, y_test):
    """
    Entraîne le modèle CNN.

    Callbacks utilisés :
      - EarlyStopping : arrête si val_loss ne s'améliore pas (patience=8)
        → évite le surapprentissage (même principe que Workshop 03)
      - ModelCheckpoint : sauvegarde automatiquement le meilleur modèle

    ImageDataGenerator : augmentation de données à la volée
      (rotation, zoom, flip, etc.)
    """

    # Augmentation des données à la volée (Keras ImageDataGenerator)
    # Appliqué uniquement sur les données d'entraînement
    datagen = ImageDataGenerator(
        rotation_range=15,           # Rotation ±15°
        zoom_range=0.1,              # Zoom ±10%
        width_shift_range=0.1,       # Décalage horizontal ±10%
        height_shift_range=0.1,      # Décalage vertical ±10%
        brightness_range=[0.8, 1.2], # Variation de luminosité
        horizontal_flip=True,        # Flip horizontal
        fill_mode='reflect'
    )
    datagen.fit(X_train)

    # Early Stopping (Workshop 03 — "When does training stop?")
    # Arrêt si val_loss ne s'améliore pas après 8 époques
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,   # Restaurer les meilleurs poids
        verbose=1
    )

    # Sauvegarde automatique du meilleur modèle
    model_path = os.path.join(MODELS_DIR, 'cnn_model.h5')
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    print(f"\n  Entraînement CNN "
          f"(max {EPOCHS} époques, early stopping patience=8)...")

    # ENTRAÎNEMENT (Workshop 03 — model.fit)
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    return history


# ============================================================
# Step 5 : Afficher l'historique d'entraînement
# (Workshop 03 — "Plot the loss function over Epochs")
# ============================================================

def print_training_history(history):
    """
    Affiche l'historique d'entraînement epoch par epoch.
    (Workshop 03 — "Plot the loss function over Epochs")

    Dans le workshop, on utilisait matplotlib pour tracer la courbe.
    Ici on affiche les valeurs — les graphiques sont dans l'interface web.
    """
    print("\n  Historique d'entraînement (Loss + Accuracy par époque) :")
    print(f"  {'Époque':>6} | {'Train Loss':>10} | {'Val Loss':>10} | "
          f"{'Train Acc':>10} | {'Val Acc':>10}")
    print("  " + "-" * 56)

    acc     = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss    = history.history['loss']
    val_loss= history.history['val_loss']

    for epoch in range(len(acc)):
        print(f"  {epoch+1:>6} | {loss[epoch]:>10.4f} | "
              f"{val_loss[epoch]:>10.4f} | "
              f"{acc[epoch]:>10.4f} | {val_acc[epoch]:>10.4f}")


# ============================================================
# VISUALISATION — Courbes d'entraînement CNN
# (Workshop 03 — "Plot the loss function over Epochs")
#
# Figure avec 2 sous-graphes :
#   1. Accuracy vs Epoch  (train + validation)
#   2. Loss    vs Epoch   (train + validation)
#
# plt.show() — aucune sauvegarde
# ============================================================

def plot_training_curves(history):
    """
    Affiche les courbes d'entraînement CNN.

    (Workshop 03 — "Plot the loss function over Epochs")

    Graphique 1 : Accuracy vs Epoch
    Graphique 2 : Loss    vs Epoch

    Paramètres :
        history : objet retourné par model.fit()
    """
    acc      = history.history["accuracy"]
    val_acc  = history.history["val_accuracy"]
    loss     = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs   = range(1, len(acc) + 1)

    # ── Figure avec 2 sous-graphes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0e1118")

    # ── Couleurs
    TRAIN_COLOR = "#6366f1"
    VAL_COLOR   = "#22c55e"
    LOSS_TRAIN  = "#ef4444"
    LOSS_VAL    = "#f59e0b"
    BG          = "#131720"
    GRID_COLOR  = "#1e2640"
    TEXT_COLOR  = "#8891a8"

    # ──────────────────────────────────────────
    # Sous-graphe 1 : Accuracy vs Epoch
    # ──────────────────────────────────────────
    ax1.set_facecolor(BG)
    ax1.plot(epochs, [v * 100 for v in acc],
             color=TRAIN_COLOR, linewidth=2.2, marker="o", markersize=3.5,
             label="Train Accuracy")
    ax1.plot(epochs, [v * 100 for v in val_acc],
             color=VAL_COLOR, linewidth=2.2, marker="s", markersize=3.5,
             linestyle="--", label="Validation Accuracy")

    ax1.set_title("Accuracy vs Epoch", fontsize=12, fontweight="bold",
                  color="white", pad=10)
    ax1.set_xlabel("Epoch", fontsize=10, color=TEXT_COLOR, labelpad=6)
    ax1.set_ylabel("Accuracy (%)", fontsize=10, color=TEXT_COLOR, labelpad=6)
    ax1.set_ylim(0, 105)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0f%%"))
    ax1.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, linestyle="--", alpha=0.7)
    ax1.set_axisbelow(True)
    ax1.tick_params(colors=TEXT_COLOR, length=0)
    ax1.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax1.legend(fontsize=9, framealpha=0.15, edgecolor="#252e48", labelcolor="white")

    # Annoter le max de val_accuracy
    best_val_epoch = int(np.argmax(val_acc)) + 1
    best_val       = max(val_acc) * 100
    ax1.annotate(
        f"Max Val: {best_val:.1f}%\n(Epoch {best_val_epoch})",
        xy=(best_val_epoch, best_val),
        xytext=(best_val_epoch + max(1, len(epochs) * 0.06), best_val - 8),
        fontsize=8, color=VAL_COLOR,
        arrowprops=dict(arrowstyle="->", color=VAL_COLOR, lw=1.2)
    )

    # ──────────────────────────────────────────
    # Sous-graphe 2 : Loss vs Epoch
    # ──────────────────────────────────────────
    ax2.set_facecolor(BG)
    ax2.plot(epochs, loss,
             color=LOSS_TRAIN, linewidth=2.2, marker="o", markersize=3.5,
             label="Train Loss")
    ax2.plot(epochs, val_loss,
             color=LOSS_VAL, linewidth=2.2, marker="s", markersize=3.5,
             linestyle="--", label="Validation Loss")

    ax2.set_title("Loss vs Epoch", fontsize=12, fontweight="bold",
                  color="white", pad=10)
    ax2.set_xlabel("Epoch", fontsize=10, color=TEXT_COLOR, labelpad=6)
    ax2.set_ylabel("Loss (Binary Crossentropy)", fontsize=10, color=TEXT_COLOR, labelpad=6)
    ax2.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, linestyle="--", alpha=0.7)
    ax2.set_axisbelow(True)
    ax2.tick_params(colors=TEXT_COLOR, length=0)
    ax2.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax2.legend(fontsize=9, framealpha=0.15, edgecolor="#252e48", labelcolor="white")

    # Annoter le min de val_loss
    best_loss_epoch = int(np.argmin(val_loss)) + 1
    best_loss_val   = min(val_loss)
    ax2.annotate(
        f"Min Val Loss: {best_loss_val:.4f}\n(Epoch {best_loss_epoch})",
        xy=(best_loss_epoch, best_loss_val),
        xytext=(best_loss_epoch + max(1, len(epochs) * 0.06), best_loss_val + 0.05),
        fontsize=8, color=LOSS_VAL,
        arrowprops=dict(arrowstyle="->", color=LOSS_VAL, lw=1.2)
    )

    fig.suptitle(
        "Courbes d'Entraînement CNN — Dysgraphia Detection\n"
        "APM.02 · Workshop 03 · Dr. NECIBI Khaled · Université Constantine 2",
        fontsize=12, fontweight="bold", color="white", y=1.02
    )
    plt.tight_layout()
    plt.show()


# ============================================================
# VISUALISATION — Matrice de Confusion (CNN)
# (Workshop 03 — "Plot Confusion Matrix")
#
# matplotlib uniquement (pas seaborn)
# Nombres dans chaque cellule
# X-axis : Predicted | Y-axis : True
# Classes : ["Typical", "Dysgraphia"]
# plt.show() — aucune sauvegarde
# ============================================================

def plot_confusion_matrix_cnn(y_true, y_pred):
    """
    Affiche la matrice de confusion du CNN.

    (Workshop 03 — "Plot Confusion Matrix")
    matplotlib uniquement — pas seaborn, pas de sauvegarde

    Structure :
      Lignes   = True labels  (Y-axis)
      Colonnes = Predicted    (X-axis)
      Classes  : ["Typical", "Dysgraphia"]

    Cellules :
      [0,0] TN vert  — Typical    prédit Typical
      [0,1] FP rouge — Typical    prédit Dysgraphia
      [1,0] FN rouge — Dysgraphia prédit Typical
      [1,1] TP vert  — Dysgraphia prédit Dysgraphia

    Paramètres :
        y_true : array de vraies étiquettes (0 ou 1)
        y_pred : array de prédictions       (0 ou 1)
    """
    from sklearn.metrics import confusion_matrix as sk_cm
    cm      = sk_cm(y_true, y_pred)
    classes = ["Typical", "Dysgraphia"]

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0e1118")
    ax.set_facecolor("#131720")

    # ── Couleurs des cellules
    #    vert  = bonne prédiction (TN, TP)
    #    rouge = mauvaise prédiction (FP, FN)
    cell_colors = [
        ["#166534", "#7f1d1d"],   # ligne 0 : TN, FP
        ["#7f1d1d", "#166534"],   # ligne 1 : FN, TP
    ]
    cell_labels = [
        ["TN", "FP"],
        ["FN", "TP"],
    ]

    n = len(classes)
    for i in range(n):
        for j in range(n):
            # Rectangle coloré
            rect = plt.Rectangle(
                [j, n - 1 - i], 1, 1,
                color=cell_colors[i][j],
                ec="#0e1118", lw=2
            )
            ax.add_patch(rect)

            # Nombre (grand, centré)
            ax.text(
                j + 0.5, n - 0.5 - i, str(cm[i][j]),
                ha="center", va="center",
                fontsize=32, fontweight="bold", color="white"
            )

            # Étiquette TP / TN / FP / FN (petit, en dessous du nombre)
            ax.text(
                j + 0.5, n - 0.72 - i, cell_labels[i][j],
                ha="center", va="center",
                fontsize=11, color="white", alpha=0.65
            )

    # ── Axes
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)

    # X : Predicted (en bas)
    ax.set_xticks([i + 0.5 for i in range(n)])
    ax.set_xticklabels(classes, fontsize=12, color="white", fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12, color="#8891a8", labelpad=10)

    # Y : True (à gauche)
    ax.set_yticks([i + 0.5 for i in range(n)])
    ax.set_yticklabels(classes[::-1], fontsize=12, color="white", fontweight="bold")
    ax.set_ylabel("True", fontsize=12, color="#8891a8", labelpad=10)

    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── Titre
    ax.set_title(
        "Confusion Matrix — CNN (Dysgraphia)\n"
        "APM.02 · Workshop 03 · Dr. NECIBI Khaled · Université Constantine 2",
        fontsize=12, fontweight="bold", color="white", pad=16
    )

    # ── Légende TP/TN/FP/FN
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="#166534", label="TP / TN — Bonne prédiction"),
        Patch(facecolor="#7f1d1d", label="FP / FN — Mauvaise prédiction"),
    ]
    ax.legend(
        handles=legend_items, loc="upper right",
        bbox_to_anchor=(1.0, -0.08), ncol=2,
        fontsize=9, framealpha=0.1, edgecolor="#252e48",
        labelcolor="white"
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# Step 6 : Évaluation du modèle
# (Workshop 03 — "Model Evaluation and Prediction")
# (Workshop 01 — mêmes métriques : accuracy, precision, recall, F1)
# ============================================================

def evaluate_cnn(model, X_test, y_test):
    """
    Évalue le CNN sur le jeu de test.

    Mêmes métriques que Workshop 01 et Workshop 03 :
      - accuracy_score
      - precision_score
      - recall_score
      - f1_score
      - confusion_matrix
      - classification_report (Workshop 03 — "Generate Classification Report")
    """
    print("\n  Évaluation du CNN sur le jeu de test...")

    # Prédiction (Workshop 04 — Step 4 : "run the model's prediction")
    y_pred_prob = model.predict(X_test, verbose=0).flatten()

    # Convertir les probabilités en classes binaires
    # > 0.5 → Dysgraphie (1), ≤ 0.5 → Normal (0)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Métriques (Workshop 01, Step 04 — "Model Evaluation")
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    # Affichage Accuracy (Workshop 03)
    print(f"\n  CNN_dysg")
    print(f"  Accuracy : {acc:.4f}")
    print()

    # Rapport de classification (Workshop 03 — "Generate Classification Report")
    report = classification_report(
        y_test, y_pred,
        target_names=["Typical", "Dysgraphique"],
        zero_division=0
    )
    print(report)

    # Matrice de confusion (Workshop 03 — "Plot Confusion Matrix")
    print("  Matrice de Confusion :")
    print(f"  {cm}")

    return {
        "name":             "CNN",
        "accuracy":         round(float(acc),  4),
        "precision":        round(float(prec), 4),
        "recall":           round(float(rec),  4),
        "f1_score":         round(float(f1),   4),
        "confusion_matrix": cm.tolist(),
        "report":           report,
    }


# ============================================================
# MAIN — Pipeline complet CNN
# (suit les mêmes étapes que Workshop 03 et Workshop 04)
# ============================================================

def train():
    """
    Pipeline complet CNN.
    Même structure que Workshop 03 (MLP → CNN) et Workshop 04 (CNN/VGG).
    """
    print("\n" + "=" * 60)
    print("  DYSGRAPHIA DETECTION — CNN (Deep Learning)")
    print("  Basé sur Workshop 03 + Workshop 04 — APM.02")
    print("=" * 60)

    # STEP 1 : Charger et prétraiter les images
    print("\n[Step 1] Chargement et prétraitement des images...")
    X, y = load_and_preprocess_images()
    if len(X) == 0:
        raise RuntimeError("Aucune image trouvée dans data/")

    # Division 80/20 (Workshop 01 + Workshop 03 — même split)
    print("\n[Step 1b] Division train/test (80% / 20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"  Train : {len(X_train)} | Test : {len(X_test)}")

    # STEP 2 : Construire le modèle CNN
    print("\n[Step 2] Construction du modèle CNN...")
    model = build_cnn_model()

    # Afficher l'architecture (Workshop 04 — avant d'entraîner)
    model.summary()

    # STEP 3 : Compiler le modèle
    print("\n[Step 3] Compilation du modèle (Adam + Binary Crossentropy)...")
    model = compile_model(model)

    # STEP 4 : Entraîner le modèle (Workshop 03 — model.fit)
    print("\n[Step 4] Entraînement du modèle...")
    history = train_cnn(model, X_train, y_train, X_test, y_test)

    # STEP 5 : Afficher l'historique (Workshop 03 — "Plot loss over Epochs")
    print("\n[Step 5] Historique d'entraînement :")
    print_training_history(history)

    # STEP 6 : Évaluation (Workshop 03 — "Model Evaluation and Prediction")
    print("\n[Step 6] Évaluation du modèle CNN...")
    metrics = evaluate_cnn(model, X_test, y_test)

    # ── VISUALISATION 1 : Courbes d'entraînement (Accuracy + Loss vs Epoch)
    print("\n  Affichage des courbes d'entraînement...")
    plot_training_curves(history)

    # ── VISUALISATION 2 : Matrice de confusion CNN
    print("  Affichage de la matrice de confusion CNN...")
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred_vis  = (y_pred_prob > 0.5).astype(int)
    plot_confusion_matrix_cnn(y_test, y_pred_vis)

    # STEP 7 : Sauvegarder l'historique + métriques en JSON
    print("\n[Step 7] Sauvegarde du modèle et des métriques...")
    history_data = {
        "accuracy":     [float(v) for v in history.history["accuracy"]],
        "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
        "loss":         [float(v) for v in history.history["loss"]],
        "val_loss":     [float(v) for v in history.history["val_loss"]],
    }

    cnn_metrics = {
        **metrics,
        "training_history": history_data,
        "epochs_trained":   len(history.history["accuracy"]),
        "dataset_size":     int(len(X)),
        "train_size":       int(len(X_train)),
        "test_size":        int(len(X_test)),
    }

    with open(os.path.join(REPORTS_DIR, "cnn_metrics.json"), "w") as f:
        json.dump(cnn_metrics, f, indent=2)

    print(f"  Modèle sauvegardé  -> models/cnn_model.h5")
    print(f"  Métriques sauvegardées -> reports/cnn_metrics.json")

    # STEP 8 : Résumé final
    print("\n" + "=" * 60)
    print("  RÉSUMÉ CNN")
    print("=" * 60)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1_score']:.4f}")
    print(f"  Époques   : {cnn_metrics['epochs_trained']}")
    print("=" * 60 + "\n")

    return cnn_metrics


# ============================================================
# Point d'entrée
# ============================================================
if __name__ == "__main__":
    train()
