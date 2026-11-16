"""
Standalone Inference Engine for Autonomous Driving.
Job: The Vehicle Co-Pilot.
Takes any single real-world image from a camera and announces the sign name and confidence percentage.
"""
import sys 
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

PROJECT_ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PROJECT_ROOT))

from src.dataset import preprocess_image
from src.sign_names import get_sign_name

DEFAULT_MODEL_PATH=PROJECT_ROOT/"models"/"traffic_sign_cnn.keras"

def predict_traffic_sign(image_path,model_path=DEFAULT_MODEL_PATH):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not open or find image: {image_path}")
    processed = preprocess_image(img_bgr)
    batch_input = np.expand_dims(processed,axis=0)
    model = tf.keras.models.load_model(str(model_path))
    probabilities = model.predict(batch_input,verbose=0)[0]
    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class])
    sign_name = get_sign_name(predicted_class)
    return predicted_class, sign_name, confidence

if __name__=="__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [<model_path>]")
        sys.exit(1)

    img_file=sys.argv[1]
    cls_id, sign_name, confidence = predict_traffic_sign(img_file)
    print(f"Predicted Class ID: {cls_id}")
    print(f"Predicted Sign Name: {sign_name}")
    print(f"Confidence: {confidence*100:.2f}%")