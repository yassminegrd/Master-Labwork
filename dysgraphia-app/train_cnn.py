"""
=======================================================================
  train_cnn.py — Convolutional Neural Network (CNN) Training Script
  Dysgraphia Detection Project | APM.02 | Master 1 STIC | 2025-2026
  Instructor: Dr. NECIBI Khaled | University of Constantine 2
=======================================================================

This script builds and trains a CNN using TensorFlow/Keras.

Architecture:
  Input (64x64 grayscale)
    → Conv2D(32) + MaxPool + Dropout
    → Conv2D(64) + MaxPool + Dropout
    → Conv2D(128) + MaxPool + Dropout
    → Flatten → Dense(256) → Dropout → Dense(1, sigmoid)

Binary classification:
    0 = Normal handwriting
    1 = Dysgraphia

Workshop reference:
  Workshop 03 (MLP) and Workshop 04 (CNN/VGG) — Dr. NECIBI Khaled
"""

import os
import json
import numpy as np
import cv2

# TensorFlow / Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

IMG_SIZE    = 64          # Input image size (64×64 pixels)
BATCH_SIZE  = 32          # Number of images per training batch
EPOCHS      = 50          # Maximum number of training epochs
RANDOM_STATE = 42         # Seed for reproducibility

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Silence verbose TF logs (show only errors)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ──────────────────────────────────────────────────────────────
# STEP 1 — LOAD RAW IMAGES
# ──────────────────────────────────────────────────────────────

def load_images():
    """
    Load all images from data/normal/ and data/dysgraphia/.

    Returns:
        images : numpy array, shape (N, IMG_SIZE, IMG_SIZE, 1)
                 — grayscale, normalized to [0, 1]
        labels : numpy array, shape (N,)
                 — 0 = normal, 1 = dysgraphia
    """
    categories = {"normal": 0, "dysgraphia": 1}
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images, labels = [], []
    total, errors = 0, 0

    print("\n  Chargement des images brutes depuis : dataset")
    for label_name, label_idx in categories.items():
        folder = os.path.join(DATA_DIR, label_name)
        if not os.path.exists(folder):
            print(f"  Attention : dossier '{folder}' introuvable.")
            continue

        files = sorted(f for f in os.listdir(folder)
                       if os.path.splitext(f)[1].lower() in exts)
        count = 0
        for fname in files:
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                errors += 1
                continue

            # Convert to grayscale, resize to 64×64, normalize [0,1]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
            gray = gray.astype(np.float32) / 255.0   # Normalize pixels

            images.append(gray)
            labels.append(label_idx)
            count += 1
            total += 1

        print(f"  {label_name:<12} -> {count} images")

    print(f"  {total} images chargées | {errors} erreurs")

    # Add channel dimension: (N, 64, 64) → (N, 64, 64, 1)
    images = np.array(images)[..., np.newaxis]
    labels = np.array(labels)

    # Print class distribution
    print(f"\n  Distribution: "
          f"Regular: {sum(labels==0)} | "
          f"Dysgraphique: {sum(labels==1)}")

    return images, labels


# ──────────────────────────────────────────────────────────────
# STEP 2 — BUILD THE CNN MODEL
# ──────────────────────────────────────────────────────────────

def build_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1)):
    """
    Build a simple but effective CNN for binary image classification.

    Architecture (inspired by Workshop 04 — VGG-like blocks):
    ┌─────────────────────────────────────────────────────┐
    │ Input: (64, 64, 1)                                  │
    │                                                     │
    │ Block 1: Conv2D(32) → ReLU → MaxPool → Dropout(0.25)│
    │ Block 2: Conv2D(64) → ReLU → MaxPool → Dropout(0.25)│
    │ Block 3: Conv2D(128)→ ReLU → MaxPool → Dropout(0.25)│
    │                                                     │
    │ Flatten → Dense(256, ReLU) → Dropout(0.5)          │
    │ Output: Dense(1, Sigmoid)   ← binary classification│
    └─────────────────────────────────────────────────────┘

    Loss    : Binary Crossentropy (for 2-class classification)
    Optimizer: Adam (adaptive learning rate)
    Metric  : Accuracy
    """
    model = keras.Sequential([
        # ── Input layer
        keras.Input(shape=input_shape),

        # ── Block 1: 32 filters, 3×3 kernel
        # Learns low-level features: edges, corners
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),          # Reduce size: 64→32
        layers.Dropout(0.25),                # Prevent overfitting

        # ── Block 2: 64 filters
        # Learns mid-level features: curves, stroke patterns
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),          # Reduce size: 32→16
        layers.Dropout(0.25),

        # ── Block 3: 128 filters
        # Learns high-level features: writing style patterns
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),          # Reduce size: 16→8
        layers.Dropout(0.25),

        # ── Flatten: convert 3D feature maps to 1D vector
        layers.Flatten(),

        # ── Fully connected layer
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),                 # Strong regularization

        # ── Output: single neuron with sigmoid → probability in [0,1]
        # Output > 0.5 → Dysgraphia, Output <= 0.5 → Normal
        layers.Dense(1, activation="sigmoid"),
    ])

    # Compile the model
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ──────────────────────────────────────────────────────────────
# STEP 3 — TRAIN THE CNN
# ──────────────────────────────────────────────────────────────

def train():
    """
    Full CNN training pipeline.
    Returns metrics dict.
    """
    print("\n" + "=" * 60)
    print("  DYSGRAPHIA DETECTION — CNN (Deep Learning)")
    print("=" * 60)

    # ── Load images
    X, y = load_images()

    if len(X) == 0:
        raise RuntimeError("Aucune image trouvée dans data/")

    # ── Split: 80% training, 20% testing
    print("\n  Division train/test (80% / 20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train : {len(X_train)} | Test : {len(X_test)}")

    # ── Data augmentation using Keras ImageDataGenerator
    # This generates augmented batches ON-THE-FLY during training
    # (no need to save augmented images to disk)
    print("\n  Préparation de l'augmentation de données...")
    datagen = ImageDataGenerator(
        rotation_range=15,           # Rotate ±15 degrees
        zoom_range=0.1,              # Zoom in/out by 10%
        width_shift_range=0.1,       # Horizontal shift by 10%
        height_shift_range=0.1,      # Vertical shift by 10%
        brightness_range=[0.8, 1.2], # Brightness variation
        horizontal_flip=True,        # Mirror the image
        fill_mode="reflect",         # Fill empty pixels with reflection
    )
    datagen.fit(X_train)

    # ── Build CNN
    print("\n  Construction du modèle CNN...")
    model = build_cnn()
    model.summary()

    # ── Callbacks
    # EarlyStopping: stop training when validation loss stops improving
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=8,          # Wait 8 epochs before stopping
        restore_best_weights=True,  # Restore the best model weights
        verbose=1,
    )

    # ModelCheckpoint: save the best model during training
    checkpoint_path = os.path.join(MODELS_DIR, "cnn_model.h5")
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    # ── Train the model
    print(f"\n  Entraînement CNN (max {EPOCHS} époques, "
          f"early stopping patience=8)...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        callbacks=[early_stopping, checkpoint],
        verbose=1,
    )

    # ── Evaluate on test set
    print("\n  Évaluation du CNN sur le jeu de test...")
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    acc   = float(accuracy_score(y_test, y_pred))
    prec  = float(precision_score(y_test, y_pred, zero_division=0))
    rec   = float(recall_score(y_test, y_pred, zero_division=0))
    f1    = float(f1_score(y_test, y_pred, zero_division=0))
    cm    = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred,
                                   target_names=["Normal", "Dysgraphia"],
                                   zero_division=0)
    print(f"\n  CNN — Accuracy: {acc:.4f}  F1: {f1:.4f}")
    print(report)

    # ── Save training history (for loss/accuracy plots in the UI)
    history_data = {
        "accuracy":     [float(v) for v in history.history["accuracy"]],
        "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
        "loss":         [float(v) for v in history.history["loss"]],
        "val_loss":     [float(v) for v in history.history["val_loss"]],
    }

    # ── Save CNN metrics
    cnn_metrics = {
        "name":             "CNN",
        "accuracy":         round(acc,  4),
        "precision":        round(prec, 4),
        "recall":           round(rec,  4),
        "f1_score":         round(f1,   4),
        "confusion_matrix": cm,
        "report":           report,
        "training_history": history_data,
        "epochs_trained":   len(history.history["accuracy"]),
        "dataset_size":     int(len(X)),
        "train_size":       int(len(X_train)),
        "test_size":        int(len(X_test)),
    }

    with open(os.path.join(REPORTS_DIR, "cnn_metrics.json"), "w") as f:
        json.dump(cnn_metrics, f, indent=2)

    print(f"\n  Modèle sauvegardé -> {checkpoint_path}")
    print(f"  Métriques sauvegardées -> reports/cnn_metrics.json")
    print("=" * 60 + "\n")

    return cnn_metrics


if __name__ == "__main__":
    train()
