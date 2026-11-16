# Traffic Sign Recognition Perception System (TSR-Net)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.13-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Validation%20Accuracy-99.63%25-brightgreen.svg)]()
[![Inference Latency](https://img.shields.io/badge/Inference%20Latency-%3C15ms%20(CPU)-blueviolet.svg)]()

A high-performance deep learning perception engine designed for Autonomous Vehicles and Advanced Driver Assistance Systems (ADAS). Implements an illumination-invariant Convolutional Neural Network trained on the German Traffic Sign Recognition Benchmark (GTSRB) to identify 43 international road sign categories in real time.

---

## Key Features

- **Automotive Safety Standards**: Compliant with international Vienna Convention road sign taxonomy across 43 classes.
- **Illumination Invariance (CLAHE)**: Integrated Contrast Limited Adaptive Histogram Equalization on the LAB color space, recovering high-frequency features under severe shadows, night headlights, and adverse weather.
- **Ultra-Low Latency**: Lightweight 3-block hierarchical CNN architecture (~825k parameters) achieving sub-15ms inference latency on standard x86/ARM CPUs without requiring dedicated GPU hardware.
- **Production REST API**: Built on asynchronous FastAPI with complete multipart image validation and top-3 probabilistic ranking.
- **Modern Minimalist UI**: Clean, responsive client interface designed with Inter typography and outline SVG telemetry indicators.
- **Edge Deployment Ready**: Includes an optimized TensorFlow Lite export (`.tflite`, 0.81 MB) suitable for serverless edge deployment (Vercel, AWS Lambda, microcontrollers).

---

## System Architecture

```
[ Vehicle Camera Sensor ]
           │
           ▼
[ CLAHE Preprocessing Engine ]  ──► B/W Luminance Equalization & 48×48×3 Tensor Scaling
           │
           ▼
[ 3-Block Deep CNN Backbone ]   ──► Hierarchical Feature Extraction (32 ➔ 64 ➔ 128 Filters)
           │
           ▼
[ Regularized Dense Head ]      ──► 256 Units + BatchNorm + Dropout(0.50)
           │
           ▼
[ Softmax Classification ]      ──► 43-Class Normalized Probability Distribution
           │
           ▼
[ ADAS Actuation Interface ]    ──► Speed Governing, Deceleration, and Safety Gating
```

---

## Technical Specifications

| Parameter | Specification | Description |
| :--- | :--- | :--- |
| **Input Dimensions** | `48 × 48 × 3` (RGB) | Bilinear downsampling with CLAHE normalization |
| **Architecture** | 3-Block Deep CNN | 6 Conv2D, 6 BatchNorm, 3 MaxPool2D, 4 Dropout |
| **Trainable Weights** | `825,419` parameters | ~10 MB uncompressed / 0.81 MB quantized TFLite |
| **Loss Function** | Sparse Categorical Cross-Entropy | Multi-class logarithmic loss optimization |
| **Optimizer** | Adam ($\alpha = 10^{-3}$) | Momentum ($\beta_1=0.9$) + RMSProp ($\beta_2=0.999$) |
| **Regularization** | Dropout ($0.25 - 0.50$) | Inverted Bernoulli activation masking |
| **Inference Runtime** | ~12 - 15 ms (CPU) | Native TensorFlow / TFLite runtime |

---

## Benchmark Performance (GTSRB)

| Metric | Training Set | Validation Set |
| :--- | :---: | :---: |
| **Accuracy** | **98.60%** | **99.63%** |
| **Cross-Entropy Loss** | `0.0438` | `0.0318` |
| **Sample Count** | 31,367 images | 7,842 images |
| **Convergence** | 8 Epochs | EarlyStopping Triggered |

---

## Repository Structure

```text
traffic_sign/
├── api/
│   ├── index.py                  # Serverless inference entry point
│   └── requirements.txt          # Lightweight edge dependencies
├── models/
│   ├── traffic_sign_cnn.keras    # Full Keras model (architecture + weights)
│   └── traffic_sign_cnn.tflite   # Quantized edge runtime model (0.81 MB)
├── sample_signs/                 # Reference test sign imagery
├── src/
│   ├── __init__.py
│   ├── dataset.py                # Preprocessing, CLAHE, and data augmentation
│   ├── evaluate.py               # Evaluation engine and confusion matrix generator
│   ├── model.py                  # Deep CNN architecture blueprint
│   ├── predict.py                # Standalone CLI inference module
│   ├── sign_names.py             # 43 GTSRB class dictionary mapping
│   └── train.py                  # Model training and metric visualizer
├── static/
│   ├── css/style.css             # Minimalist SaaS UI design system
│   ├── js/main.js                # Frontend client controller
│   └── index.html                # Application single-page interface
├── app.py                        # FastAPI production web server
├── requirements.txt              # Complete project dependencies
├── vercel.json                   # Edge deployment configuration
└── README.md
```

---

## Quickstart

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/traffic-sign-recognition.git
cd traffic-sign-recognition
pip install -r requirements.txt
```

### 2. Launching the Web Service

Start the local FastAPI application server:

```bash
python app.py
```

The web dashboard is available at `http://localhost:8080`.

### 3. Command-Line Inference

To run inference on an individual image:

```bash
python src/predict.py "sample_signs/stop_sign.png"
```

Output:
```text
Predicted Class ID: 14
Predicted Sign Name: Stop
Confidence: 100.00%
```

### 4. Running Model Training

To retrain the neural network from scratch using the GTSRB dataset:

```bash
python src/train.py
```

Training metrics and loss curves will be automatically generated in `outputs/training_history.png`.

---

## API Reference

### `POST /api/predict`

Submits an image for classification.

**Request:** `multipart/form-data` with key `file` (supports PNG, JPG, JPEG, WEBP, PPM).

**Response:**
```json
{
  "status": "success",
  "predicted_class": 14,
  "sign_name": "Stop",
  "confidence": 99.82,
  "latency_ms": 14.6,
  "top_predictions": [
    { "class_id": 14, "name": "Stop", "confidence": 99.82 },
    { "class_id": 17, "name": "No entry", "confidence": 0.11 },
    { "class_id": 13, "name": "Yield", "confidence": 0.04 }
  ]
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
