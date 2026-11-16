"""
Vercel Serverless Function for Traffic Sign Recognition.
Uses lightweight TFLite runtime (<40MB) to stay well within Vercel's 250MB size limit.
"""

import io
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Attempt to load tflite_runtime (serverless), fallback to tensorflow.lite if local
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# Paths
API_DIR = Path(__file__).resolve().parent
BASE_DIR = API_DIR.parent
MODEL_PATH = API_DIR / "traffic_sign_cnn.tflite"
if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR / "models" / "traffic_sign_cnn.tflite"

# Class Names
SIGN_CLASSES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons",
}

app = FastAPI(title="Traffic Sign Recognition Serverless API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize TFLite Interpreter
interpreter = None
input_details = None
output_details = None

if MODEL_PATH.exists():
    interpreter = tflite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


def preprocess_pil(image: Image.Image) -> np.ndarray:
    """Preprocess PIL Image to normalized (1, 48, 48, 3) float32 tensor."""
    image = image.convert("RGB")
    image = image.resize((48, 48), Image.Resampling.BILINEAR)
    # Apply local contrast equalization
    image = ImageOps.autocontrast(image, cutoff=2)
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


@app.post("/api/predict")
@app.post("/predict")
async def predict_sign(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = None,
):
    if interpreter is None:
        raise HTTPException(status_code=500, detail="TFLite model not initialized.")

    t_start = time.perf_counter()

    if file:
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file format.")
    elif sample_id:
        sample_path = BASE_DIR / "sample_signs" / f"{sample_id}.png"
        if not sample_path.exists():
            sample_path = BASE_DIR / "public" / "samples" / f"{sample_id}.png"
        if not sample_path.exists():
            raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")
        image = Image.open(sample_path)
    else:
        raise HTTPException(status_code=400, detail="No image file or sample_id provided.")

    input_tensor = preprocess_pil(image)

    # Run inference via TFLite
    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()
    raw_predictions = interpreter.get_tensor(output_details[0]["index"])[0]

    latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

    predicted_class = int(np.argmax(raw_predictions))
    confidence_pct = round(float(raw_predictions[predicted_class]) * 100.0, 2)
    sign_name = SIGN_CLASSES.get(predicted_class, f"Unknown Sign ({predicted_class})")

    top_indices = np.argsort(raw_predictions)[::-1][:3]
    top_predictions = [
        {
            "class_id": int(idx),
            "name": SIGN_CLASSES.get(int(idx), f"Sign {idx}"),
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


@app.get("/api/samples")
@app.get("/samples")
def get_samples():
    samples_dir = BASE_DIR / "sample_signs"
    if not samples_dir.exists():
        samples_dir = BASE_DIR / "public" / "samples"
    samples = []
    if samples_dir.exists():
        for f in sorted(samples_dir.glob("*.png")):
            clean_name = f.stem.replace("_", " ").title()
            samples.append({
                "id": f.stem,
                "name": clean_name,
                "filename": f.name,
                "url": f"/samples/{f.name}",
            })
    return samples
