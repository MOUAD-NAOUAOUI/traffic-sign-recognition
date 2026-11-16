import os 
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_training_data, NUM_CLASSES
from src.model import build_traffic_sign_cnn

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001

def train_model():
    X_train,X_val,y_train,y_val=load_training_data(DATA_DIR)

    model = build_traffic_sign_cnn(num_classes=NUM_CLASSES)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    model_path = str(MODELS_DIR / "traffic_sign_cnn.keras")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        verbose=1
    )
    callbacks=[checkpoint,early_stop]

    print("\n[Step 3/4] Launching Training Engine...")
    history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )
    return history

def plot_training_history(history, save_dir=OUTPUTS_DIR):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history["accuracy"], label="Train Accuracy", color="blue", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Validation Accuracy", color="orange", linewidth=2)
    ax1.set_title("Training vs Validation Accuracy", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax2.plot(history.history["loss"], label="Train Loss", color="blue", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Validation Loss", color="orange", linewidth=2)
    ax2.set_title("Training vs Validation Loss", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot_path = Path(save_dir) / "training_history.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"\n[Done] Training complete! Charts saved to: {plot_path}")


if __name__ == "__main__":
    history = train_model()
    plot_training_history(history)