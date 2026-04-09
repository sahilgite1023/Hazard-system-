"""
train_model.py  –  Real-Time Hazard Sound Detection
=====================================================
Trains a CNN model on the UrbanSound8K dataset to classify:
  • dog_bark   (label 0)
  • gun_shot   (label 1)
  • siren      (label 2)
  • normal     (label 3)

Usage
-----
    python train_model.py

The best-performing model is saved to  model/model.h5
Training curves are saved to           model/training_history.png
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers
import matplotlib.pyplot as plt

# ── TensorFlow / Keras ───────────────────────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # suppress verbose TF logs
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# ── Project utilities ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.dataset_loader import DatasetLoader, LABEL_NAMES
from utils.feature_extraction import FeatureExtractor

# ── Hyper-parameters ─────────────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.15          # fraction of train set used for validation
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")
HISTORY_PLOT_PATH = os.path.join(MODEL_DIR, "training_history.png")


# ─────────────────────────────────────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────────────────────────────────────

def build_model(input_shape: tuple, num_classes: int = 4) -> keras.Model:
    """
    CNN architecture for audio classification.

    Input  : (n_mels, max_frames, 1)   e.g. (128, 173, 1)
    Output : softmax over *num_classes* categories
    """
    inputs = keras.Input(shape=input_shape, name="log_mel_input")

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Dropout(0.25, name="drop1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = layers.Dropout(0.25, name="drop3")(x)

    # Block 4
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Classifier head
    x = layers.Dense(256, name="fc1")(x)
    x = layers.BatchNormalization(name="bn5")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5, name="drop4")(x)

    x = layers.Dense(128, activation="relu", name="fc2")(x)
    x = layers.Dropout(0.3, name="drop5")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs, outputs, name="HazardSoundCNN")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_history(history: keras.callbacks.History, save_path: str) -> None:
    """Save accuracy and loss curves to *save_path*."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History – Hazard Sound Detection CNN", fontsize=14)

    # Accuracy
    ax = axes[0]
    ax.plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    ax.plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.4)

    # Loss
    ax = axes[1]
    ax.plot(history.history["loss"], label="Train Loss", linewidth=2)
    ax.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Categorical Cross-Entropy")
    ax.legend()
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"📊  Training curves saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("  Real-Time Hazard Sound Detection – Training Pipeline")
    print("=" * 65)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── 1. Load dataset ──────────────────────────────────────────────────────
    extractor = FeatureExtractor()
    loader = DatasetLoader(extractor=extractor)
    X_train, X_test, y_train, y_test, label_names = loader.load()

    num_classes = len(label_names)
    print(f"\n  Classes ({num_classes}): {label_names}")
    print(f"  Train samples : {len(X_train)}")
    print(f"  Test  samples : {len(X_test)}")
    print(f"  Feature shape : {X_train.shape[1:]}")

    # ── 2. One-hot encode labels ─────────────────────────────────────────────
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # ── 3. Build model ───────────────────────────────────────────────────────
    model = build_model(input_shape=extractor.input_shape, num_classes=num_classes)
    model.summary()

    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ── 4. Callbacks ─────────────────────────────────────────────────────────
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ── 5. Train ─────────────────────────────────────────────────────────────
    print("\n🚀  Starting training …\n")
    history = model.fit(
        X_train,
        y_train_cat,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        shuffle=True,
    )

    # ── 6. Evaluate on held-out test set ────────────────────────────────────
    print("\n🧪  Evaluating on test set …")
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"    Test Loss     : {test_loss:.4f}")
    print(f"    Test Accuracy : {test_acc * 100:.2f} %")

    # ── 7. Save plot ─────────────────────────────────────────────────────────
    plot_history(history, HISTORY_PLOT_PATH)

    print(f"\n✅  Best model saved → {MODEL_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    main()
