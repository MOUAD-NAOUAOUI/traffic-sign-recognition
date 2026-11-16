"""
Traffic Sign Recognition - Web Application Backend (FastAPI).
Connects the trained Deep Learning CNN model to a modern, minimalist REST API.
"""

import io
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.dataset import preprocess_image
from src.sign_names import SIGN_CLASSES, get_sign_name

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "traffic_sign_cnn.keras"
STATIC_DIR = PROJECT_ROOT / "static"
SAMPLES_DIR = PROJECT_ROOT / "sample_signs"

STATIC_DIR.mkdir(exist_ok=True)
SAMPLES_DIR.mkdir(exist_ok=True)

# Initialize FastAPI application
app = FastAPI(
    title="Traffic Sign Recognition Perception System",
    description="Production-grade inference API for GTSRB Traffic Sign Classification",
    version="1.0.0",
)

# Enable CORS for flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model once at startup (caching the brain in memory)
print(f"[Backend] Loading trained model from {MODEL_PATH}...")
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. Run training first via 'python src/train.py'."
    )

model = tf.keras.models.load_model(str(MODEL_PATH))
print(f"[Backend] Model loaded successfully. Ready for inference!")

# Model metadata
MODEL_METADATA = {
    "architecture": "3-Block Deep CNN (Conv2D + BatchNorm + MaxPool + Dropout)",
    "parameters": model.count_params(),
    "classes_count": len(SIGN_CLASSES),
    "input_resolution": "48 × 48 × 3 (RGB)",
    "validation_accuracy": 99.63,
    "training_accuracy": 98.60,
    "framework": "TensorFlow 2.x / Keras",
    "preprocessing": "CLAHE Contrast Equalization + Min-Max Normalization",
}


@app.get("/api/model-info")
def get_model_info():
    """Returns technical metadata and validated performance metrics of the CNN."""
    return MODEL_METADATA


@app.get("/api/samples")
def list_samples():
    """Lists available test sample signs for 1-click evaluation."""
    samples = []
    if SAMPLES_DIR.exists():
        for file in sorted(SAMPLES_DIR.glob("*.png")):
            clean_name = file.stem.replace("_", " ").title()
            samples.append({
                "id": file.stem,
                "name": clean_name,
                "filename": file.name,
                "url": f"/samples/{file.name}",
            })
    return samples


@app.get("/samples/{filename}")
def get_sample_image(filename: str):
    """Serves sample image files."""
    file_path = SAMPLES_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Sample image not found.")
    return FileResponse(file_path)


@app.post("/api/predict")
async def predict_image(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = None,
):
    """
    Classifies a traffic sign image and returns the predicted label,
    exact confidence percentage, and top-3 probabilistic ranking.
    """
    img_bgr = None

    # Handle sample sign by ID
    if sample_id:
        sample_path = SAMPLES_DIR / f"{sample_id}.png"
        if not sample_path.exists():
            raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")
        img_bgr = cv2.imread(str(sample_path))
    elif file:
        # Validate MIME type / extension
        valid_extensions = {".png", ".jpg", ".jpeg", ".ppm", ".webp", ".bmp"}
        ext = Path(file.filename or "").suffix.lower()
        if ext not in valid_extensions and file.content_type not in [
            "image/png",
            "image/jpeg",
            "image/webp",
            "image/bmp",
            "application/octet-stream",
        ]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format '{ext}'. Please upload PNG, JPG, JPEG, PPM, or WEBP.",
            )

        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(
            status_code=400,
            detail="Failed to decode image data. Please verify the file is a valid image.",
        )

    # Measure inference latency
    t_start = time.perf_counter()

    # Preprocess using the exact training pipeline (CLAHE + Resize + Scale)
    processed = preprocess_image(img_bgr)
    batch_input = np.expand_dims(processed, axis=0)

    # Query the neural network
    raw_predictions = model.predict(batch_input, verbose=0)[0]

    t_end = time.perf_counter()
    latency_ms = round((t_end - t_start) * 1000.0, 2)

    # Top prediction
    predicted_class = int(np.argmax(raw_predictions))
    confidence_pct = round(float(raw_predictions[predicted_class]) * 100.0, 2)
    sign_name = get_sign_name(predicted_class)

    # Top-3 ranking
    top_indices = np.argsort(raw_predictions)[::-1][:3]
    top_predictions = [
        {
            "class_id": int(idx),
            "name": get_sign_name(int(idx)),
            "confidence": round(float(raw_predictions[idx]) * 100.0, 2),
        }
        for idx in top_indices
    ]

    return {
        "status": "success",
        "predicted_class": predicted_class,
        "sign_name": sign_name,
        "confidence": confidence_pct,
        "latency_ms": latency_ms,
        "top_predictions": top_predictions,
    }


# Mount the static directory to serve frontend assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    """Serves the main application user interface."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse(
            status_code=404,
            content={"message": "UI index.html not found. Please compile frontend assets."},
        )
    return FileResponse(index_file)


if __name__ == "__main__":
    import uvicorn

    print("\n=======================================================")
    print(" Traffic Sign Recognition Web App Starting")
    print(" URL: http://localhost:8080")
    print("=======================================================\n")
    uvicorn.run("app:app", host="127.0.0.1", port=8080)
