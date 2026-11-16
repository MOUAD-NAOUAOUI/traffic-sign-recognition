"""
Evaluation and Confusion Matrix Generator.
Job: The Final Exam Examiner.
Tests the trained model on 12,630 unseen photos and draws a report card and confusion matrix.
"""
import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import preprocess_image, NUM_CLASSES
from src.sign_names import SIGN_CLASSES

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def load_test_dataset(data_dir=DATA_DIR):
    """Loads all test images and true labels using Test.csv."""
    test_csv = Path(data_dir) / "Test.csv"
    if not test_csv.exists():
        test_csv = Path(data_dir) / "GT-final_test.csv"
    df = pd.read_csv(test_csv)

    images = []
    labels = df["ClassId"].values.astype(np.int32)
    base_dir = Path(data_dir)

    for relative_path in df["Path"].values:
        img_path = base_dir / relative_path
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        images.append(preprocess_image(img_bgr))

    X_test = np.array(images, dtype=np.float32)
    return X_test, labels[:len(images)]

def evaluate_model(model_path=MODELS_DIR / "traffic_sign_cnn.keras", data_dir=DATA_DIR):
    """Evaluates trained model on unseen test set and computes metrics."""
    print("\n[Evaluation] Loading test dataset...")
    X_test, y_test = load_test_dataset(data_dir)
    print(f"Loaded {len(X_test)} test images.")

    print(f"[Evaluation] Loading model: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    predictions = model.predict(X_test, batch_size=64, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    accuracy = np.mean(y_pred == y_test) * 100.0
    print(f"\n>>> Final Test Accuracy: {accuracy:.2f}% <<<")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(18, 15))
    sns.heatmap(cm, cmap="Blues", annot=False, fmt="d", cbar=True)
    plt.title(f"GTSRB Confusion Matrix (Test Accuracy: {accuracy:.2f}%)", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Class ID", fontsize=12)
    plt.ylabel("True Class ID", fontsize=12)

    cm_path = OUTPUTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Done] Confusion matrix saved to: {cm_path}")
    return accuracy, cm


if __name__ == "__main__":
    evaluate_model()